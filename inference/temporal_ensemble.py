"""
Temporal ensembling for ACT inference.

During inference, ACT predicts a chunk of future actions at each timestep.
Consecutive predictions overlap — timestep t appears in chunks predicted at
times t, t-1, t-2, ..., t-chunk_size+1. Temporal ensembling averages these
overlapping predictions with exponential weighting to produce smoother actions.

Without temporal ensembling:
  - The policy is queried every chunk_size steps
  - Within each chunk, actions are executed sequentially
  - At chunk boundaries, there can be discontinuities

With temporal ensembling:
  - The policy is queried at EVERY timestep
  - For each timestep, all overlapping chunk predictions are weighted and averaged
  - The weighting favors earlier predictions (lower index = higher weight)
  - This produces smooth, continuous action trajectories

The exponential weighting uses:
  weight[i] = exp(-k * i)
where i is the age of the prediction (0 = most recent chunk, 1 = one step ago, etc.)
and k controls the decay rate:
  - k=0.01 (default): very gentle decay, nearly uniform averaging
  - k=0.1:  moderate decay, recent predictions weighted ~2x vs oldest
  - k=1.0:  strong decay, heavily favors recent predictions

Data structure:
  all_time_actions[max_timesteps, max_timesteps + chunk_size, action_dim]
  - Row i: the chunk_size actions predicted at timestep i
  - Column j: predictions for what action to take at timestep j
  - To get the action for timestep t: look at column t across all rows
    that have a prediction for that timestep
"""

import torch
import numpy as np


class TemporalEnsemble:
    """
    Temporal ensembling for smooth ACT inference.

    Collects action chunk predictions over time and produces smoothed
    actions via exponentially-weighted averaging of overlapping predictions.

    Args:
        chunk_size: number of future steps per prediction (num_queries)
        action_dim: dimension of the action space
        max_timesteps: maximum episode length
        k: exponential decay constant (default 0.01, gentle decay)
        device: torch device for computation
    """

    def __init__(self, chunk_size, action_dim, max_timesteps, k=0.01,
                 device='cpu'):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.max_timesteps = max_timesteps
        self.k = k
        self.device = device

        # Prediction buffer: row t stores the chunk predicted at timestep t
        # Columns extend to max_timesteps + chunk_size to handle the last chunk
        self.all_time_actions = torch.zeros(
            max_timesteps, max_timesteps + chunk_size, action_dim,
            device=device)

    def reset(self):
        """Clear all stored predictions for a new episode."""
        self.all_time_actions.zero_()

    def add_prediction(self, t, actions):
        """
        Store a chunk of predicted actions.

        Args:
            t: current timestep (when the prediction was made)
            actions: (chunk_size, action_dim) predicted action chunk
                     or (1, chunk_size, action_dim) with batch dim
        """
        if actions.dim() == 3:
            actions = actions.squeeze(0)
        self.all_time_actions[t, t:t + self.chunk_size] = actions

    def get_action(self, t):
        """
        Get the ensembled action for timestep t.

        Collects all predictions that target timestep t (from chunks
        predicted at times t, t-1, ..., max(0, t-chunk_size+1)),
        then computes a weighted average using exponential decay.

        Args:
            t: timestep to get the action for

        Returns:
            action: (action_dim,) smoothed action
        """
        # Gather all predictions for timestep t (column t across all rows)
        actions_for_t = self.all_time_actions[:, t]  # (max_timesteps, action_dim)

        # Find which rows have non-zero predictions for this timestep
        # (a row is "populated" if any dimension is non-zero)
        actions_populated = torch.all(actions_for_t != 0, axis=1)
        actions_for_t = actions_for_t[actions_populated]

        if len(actions_for_t) == 0:
            return torch.zeros(self.action_dim, device=self.device)

        # Compute exponential weights: earlier predictions get higher weight
        # index 0 = earliest prediction for this timestep (most "aged")
        exp_weights = np.exp(-self.k * np.arange(len(actions_for_t)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).float().to(
            self.device).unsqueeze(1)

        # Weighted average
        action = (actions_for_t * exp_weights).sum(dim=0)
        return action


class InferencePipeline:
    """
    Complete inference pipeline for ACT with optional temporal ensembling.

    Provides two modes:
    1. No ensembling: query the policy every chunk_size steps,
       execute actions sequentially within each chunk
    2. With ensembling: query the policy at every step,
       smooth overlapping predictions via TemporalEnsemble

    Args:
        policy: ACTPolicy instance (already loaded with trained weights)
        normalizer: ACTNormalizer instance (for input normalization
                    and output denormalization)
        chunk_size: action chunk size (num_queries)
        temporal_agg: whether to use temporal ensembling
        max_timesteps: maximum episode length
        k: temporal ensembling decay constant
        device: torch device
    """

    def __init__(self, policy, normalizer, chunk_size, temporal_agg=False,
                 max_timesteps=400, k=0.01, device='cpu'):
        self.policy = policy
        self.normalizer = normalizer
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.device = device

        self.policy.to(device)
        self.policy.eval()

        # Determine action dimension from the model
        action_dim = policy.model.action_dim

        if temporal_agg:
            self.ensemble = TemporalEnsemble(
                chunk_size, action_dim, max_timesteps, k, device)
        else:
            self.ensemble = None

        # State for non-ensembled mode
        self._cached_actions = None
        self._query_counter = 0

    def reset(self):
        """Reset for a new episode."""
        if self.ensemble is not None:
            self.ensemble.reset()
        self._cached_actions = None
        self._query_counter = 0

    def get_action(self, t, qpos_raw, image):
        """
        Get the action to execute at timestep t.

        Args:
            t: current timestep
            qpos_raw: (state_dim,) raw (unnormalized) joint positions
            image: (num_cam, 3, H, W) camera images in [0, 1]

        Returns:
            action_raw: (action_dim,) raw (denormalized) action to execute
        """
        # Normalize input
        qpos = self.normalizer.normalize_qpos(qpos_raw)
        qpos = qpos.unsqueeze(0).to(self.device)
        image = image.unsqueeze(0).to(self.device)

        if self.temporal_agg:
            # Query policy at every timestep
            a_hat = self.policy.predict_action(qpos, image)  # (1, chunk, act)
            self.ensemble.add_prediction(t, a_hat)
            action_norm = self.ensemble.get_action(t)
        else:
            # Query policy every chunk_size steps
            if t % self.chunk_size == 0:
                self._cached_actions = self.policy.predict_action(
                    qpos, image)  # (1, chunk, act)
            action_norm = self._cached_actions[0, t % self.chunk_size]

        # Denormalize to raw action space
        action_raw = self.normalizer.denormalize_action(
            action_norm.unsqueeze(0)).squeeze(0)
        return action_raw.cpu()
