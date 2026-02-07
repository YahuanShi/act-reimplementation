"""
LeRobot dataset adapter for ACT.

This module bridges the LeRobot data format to the ACT policy interface.
LeRobot stores robot learning data in a standardized format:
  - Parquet files for low-dimensional data (state, action)
  - MP4 video files for camera images
  - delta_timestamps for temporal window sampling

ACT's data requirements differ from diffusion policy:
  1. Only the CURRENT observation is needed (no history window)
  2. Future action chunk of length chunk_size is needed (not just one step)
  3. An is_pad mask marks which future actions are padding (beyond episode end)

This adapter uses LeRobot's delta_timestamps mechanism:
  - action timestamps: [0, 1/fps, 2/fps, ..., (chunk_size-1)/fps]
  - image timestamps:  [0]  (current frame only)
  - state timestamps:  [0]  (current state only)

When a sample is near the end of an episode, some future action timestamps
will fall outside the episode. LeRobot handles this by clamping to the last
available frame. We detect these clamped positions and mark them in is_pad.

Output format per sample:
  {
      'qpos':   (state_dim,)                  — current joint positions
      'image':  (num_cam, 3, H, W)            — current camera images
      'action': (chunk_size, action_dim)       — future action sequence
      'is_pad': (chunk_size,)                  — True for padded positions
  }
"""

import torch
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotACTDataset(Dataset):
    """
    Wraps a LeRobot dataset for ACT training.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., "lerobot/aloha_sim_transfer_cube")
        root: local path to dataset (alternative to repo_id)
        episodes: list of episode indices to include (None = all)
        chunk_size: number of future action steps to predict (num_queries)
        image_keys: list of image/video feature keys (auto-detected if None)
        state_key: key for robot state in the dataset
        action_key: key for robot actions in the dataset
        video_backend: video decoding backend ("pyav" or "torchvision")
        image_transforms: optional image augmentation transforms
    """

    def __init__(self, repo_id, root=None, episodes=None,
                 chunk_size=100,
                 image_keys=None,
                 state_key='observation.state',
                 action_key='action',
                 video_backend='pyav',
                 image_transforms=None):

        # First pass: load metadata to discover features and fps
        meta_ds = LeRobotDataset(
            repo_id=repo_id, root=root, episodes=episodes,
            video_backend=video_backend)
        fps = meta_ds.fps
        features = meta_ds.features

        # Auto-detect image keys if not provided
        if image_keys is None:
            image_keys = sorted(
                k for k, v in features.items()
                if v.get('dtype') == 'video' or
                (isinstance(v.get('shape', ()), (list, tuple))
                 and len(v['shape']) == 3))

        # Build delta_timestamps for each feature
        # Action: sample chunk_size future steps starting from current frame
        action_ts = [i / fps for i in range(chunk_size)]
        dt = {action_key: action_ts}

        # Images: only current frame
        for k in image_keys:
            dt[k] = [0.0]

        # State: only current frame
        if state_key in features:
            dt[state_key] = [0.0]

        del meta_ds

        # Second pass: create the actual dataset with delta_timestamps
        self.ds = LeRobotDataset(
            repo_id=repo_id, root=root, episodes=episodes,
            video_backend=video_backend,
            image_transforms=image_transforms,
            delta_timestamps=dt)

        self.chunk_size = chunk_size
        self.fps = fps
        self.image_keys = image_keys
        self.state_key = state_key
        self.action_key = action_key
        self.has_state = state_key in features

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Returns a single training sample.

        Returns:
            dict with keys:
                'qpos':   (state_dim,) current joint positions
                'image':  (num_cam, 3, H, W) current camera images
                'action': (chunk_size, action_dim) future action chunk
                'is_pad': (chunk_size,) boolean, True for padding positions
        """
        item = self.ds[idx]
        out = {}

        # ── State (qpos) ──
        if self.has_state:
            # LeRobot returns (1, state_dim) because delta_timestamps=[0.0]
            state = item[self.state_key]
            if state.dim() == 2:
                state = state.squeeze(0)
            out['qpos'] = state

        # ── Images ──
        # Stack multiple cameras into (num_cam, 3, H, W)
        cam_images = []
        for key in self.image_keys:
            img = item[key]
            # LeRobot may return (1, 3, H, W) for single timestamp
            if img.dim() == 4:
                img = img.squeeze(0)
            cam_images.append(img)
        out['image'] = torch.stack(cam_images, dim=0)  # (num_cam, 3, H, W)

        # ── Actions + is_pad ──
        # action shape: (chunk_size, action_dim)
        action = item[self.action_key]
        out['action'] = action

        # Build is_pad mask by checking which timestamps were clamped.
        # LeRobot provides 'action_is_pad' or we can detect by checking
        # the timestamp_key. We use a simple heuristic: check the episode
        # index boundaries.
        is_pad = self._compute_is_pad(item, idx)
        out['is_pad'] = is_pad

        return out

    def _compute_is_pad(self, item, idx):
        """
        Compute the is_pad mask for the action chunk.

        Near the end of an episode, some future timesteps in the action chunk
        fall beyond the episode boundary. LeRobot clamps these to the last
        available frame, resulting in duplicated actions at the tail. We detect
        this by checking the episode index boundaries.

        Returns:
            is_pad: (chunk_size,) boolean tensor
        """
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)

        # Get the current sample's episode index and position within episode
        current_idx = idx
        ep_idx = self.ds.hf_dataset[current_idx]['episode_index']

        # Find the last sample index belonging to this episode
        # by scanning forward (at most chunk_size steps)
        hf = self.ds.hf_dataset
        total_len = len(hf)

        for offset in range(self.chunk_size):
            future_idx = current_idx + offset
            if future_idx >= total_len:
                # Beyond dataset end -> padding
                is_pad[offset:] = True
                break
            if hf[future_idx]['episode_index'] != ep_idx:
                # Crossed into next episode -> padding
                is_pad[offset:] = True
                break

        return is_pad

    @property
    def state_dim(self):
        """Infer state dimension from dataset features."""
        if self.has_state:
            feat = self.ds.features[self.state_key]
            return feat['shape'][0] if isinstance(feat['shape'], (list, tuple)) else feat['shape']
        return None

    @property
    def action_dim(self):
        """Infer action dimension from dataset features."""
        feat = self.ds.features[self.action_key]
        return feat['shape'][0] if isinstance(feat['shape'], (list, tuple)) else feat['shape']
