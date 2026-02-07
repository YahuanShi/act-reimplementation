"""
Data normalization for ACT training and inference.

ACT normalizes robot state (qpos) and actions using dataset-wide statistics:
  normalized = (raw - mean) / std

This is important because:
1. Different joints may have very different scales (e.g., shoulder rotation
   in radians vs gripper opening in meters)
2. The L1 loss treats all dimensions equally, so normalization ensures each
   joint contributes proportionally to the loss
3. At inference, predicted actions must be denormalized back to the original
   scale before sending to the robot

The std is clipped to a minimum of 0.01 to prevent division by near-zero
values for joints that barely move in the dataset.

The normalizer statistics are saved alongside the model checkpoint so that
inference uses the exact same normalization as training.
"""

import torch
import numpy as np


class ACTNormalizer:
    """
    Normalizer for ACT policy training and inference.

    Computes mean and std for action and qpos (joint position) data from
    the training dataset, and provides normalize/denormalize methods.

    Usage:
        normalizer = ACTNormalizer()
        normalizer.fit(dataset)             # compute stats from training data
        qpos_norm = normalizer.normalize_qpos(qpos)
        action_norm = normalizer.normalize_action(action)
        action_raw = normalizer.denormalize_action(action_norm)
    """

    def __init__(self):
        self.action_mean = None
        self.action_std = None
        self.qpos_mean = None
        self.qpos_std = None
        self._fitted = False

    def fit(self, dataset, action_key='action', state_key='observation.state'):
        """
        Compute normalization statistics from a LeRobot dataset.

        Iterates over the underlying HuggingFace dataset (not the temporal-
        windowed version) to collect all action and state values, then
        computes per-dimension mean and std.

        Args:
            dataset: a LeRobotDataset instance (with .hf_dataset attribute)
            action_key: key for actions in the dataset
            state_key: key for robot state in the dataset
        """
        hf = dataset.hf_dataset
        n = len(hf)

        # Collect all actions
        all_actions = []
        for i in range(n):
            all_actions.append(torch.tensor(hf[i][action_key]))
        all_actions = torch.stack(all_actions)  # (N, action_dim)

        self.action_mean = all_actions.mean(dim=0)
        self.action_std = all_actions.std(dim=0)
        # Floor clipping: prevent division by near-zero for joints that
        # barely move in the dataset
        self.action_std = torch.clamp(self.action_std, min=0.01)

        # Collect all states
        if state_key in hf.features:
            all_states = []
            for i in range(n):
                all_states.append(torch.tensor(hf[i][state_key]))
            all_states = torch.stack(all_states)

            self.qpos_mean = all_states.mean(dim=0)
            self.qpos_std = all_states.std(dim=0)
            self.qpos_std = torch.clamp(self.qpos_std, min=0.01)

        self._fitted = True

    def fit_from_tensors(self, actions, qpos=None):
        """
        Compute normalization statistics from pre-collected tensors.

        Useful when data is already loaded in memory or for testing.

        Args:
            actions: (N, action_dim) tensor of all actions
            qpos: (N, state_dim) tensor of all states (optional)
        """
        self.action_mean = actions.mean(dim=0)
        self.action_std = torch.clamp(actions.std(dim=0), min=0.01)

        if qpos is not None:
            self.qpos_mean = qpos.mean(dim=0)
            self.qpos_std = torch.clamp(qpos.std(dim=0), min=0.01)

        self._fitted = True

    def normalize_action(self, action):
        """Normalize actions: (action - mean) / std"""
        assert self._fitted, "Call fit() before normalizing"
        mean = self.action_mean.to(action.device)
        std = self.action_std.to(action.device)
        return (action - mean) / std

    def denormalize_action(self, action):
        """Denormalize actions back to original scale: action * std + mean"""
        assert self._fitted, "Call fit() before denormalizing"
        mean = self.action_mean.to(action.device)
        std = self.action_std.to(action.device)
        return action * std + mean

    def normalize_qpos(self, qpos):
        """Normalize joint positions: (qpos - mean) / std"""
        assert self._fitted, "Call fit() before normalizing"
        assert self.qpos_mean is not None, "No qpos stats — fit with state data"
        mean = self.qpos_mean.to(qpos.device)
        std = self.qpos_std.to(qpos.device)
        return (qpos - mean) / std

    def denormalize_qpos(self, qpos):
        """Denormalize joint positions back to original scale."""
        assert self._fitted, "Call fit() before denormalizing"
        assert self.qpos_mean is not None, "No qpos stats — fit with state data"
        mean = self.qpos_mean.to(qpos.device)
        std = self.qpos_std.to(qpos.device)
        return qpos * std + mean

    def state_dict(self):
        """Serialize normalizer statistics for saving alongside checkpoints."""
        return {
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'qpos_mean': self.qpos_mean,
            'qpos_std': self.qpos_std,
            'fitted': self._fitted,
        }

    def load_state_dict(self, state_dict):
        """Load normalizer statistics from a saved state dict."""
        self.action_mean = state_dict['action_mean']
        self.action_std = state_dict['action_std']
        self.qpos_mean = state_dict['qpos_mean']
        self.qpos_std = state_dict['qpos_std']
        self._fitted = state_dict['fitted']
