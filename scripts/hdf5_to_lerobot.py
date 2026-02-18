"""
Convert HDF5 robot episodes to LeRobot v3 dataset format for ACT training.

Supports dual-arm robot setups (e.g., ALOHA) and single-arm setups.
The HDF5 file structure expected:
    /action                              (T, action_dim)   e.g. 14 for dual-arm
    /observations/qpos                   (T, state_dim)    joint positions
    /observations/images/<camera_name>   (T, H, W, 3)      uint8 RGB

Usage:
    # Single-arm (7-DOF), one camera
    python scripts/hdf5_to_lerobot.py \\
        --input /path/to/episodes \\
        --repo_id local/my_robot \\
        --root ./data \\
        --fps 50 \\
        --image_keys wrist_cam

    # Dual-arm ALOHA-style (14-DOF), two cameras
    python scripts/hdf5_to_lerobot.py \\
        --input /path/to/episodes \\
        --repo_id local/aloha_pick_place \\
        --root ./data \\
        --fps 50 \\
        --image_keys cam_high cam_low cam_left_wrist cam_right_wrist
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def get_episode_files(input_dir: str) -> list[str]:
    """Sort HDF5 episode files by embedded number.

    Sorting by number (not lexicographically) matters: episode_10.hdf5 must
    come after episode_9.hdf5, not before episode_2.hdf5. A deterministic
    order ensures train/val splits are reproducible across runs.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    files.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return files


def inspect_hdf5(filepath: str) -> dict:
    """Recursively collect all dataset paths and their shapes/dtypes.

    We inspect the first episode file to derive the features dict that
    LeRobot needs. Using h5py.visititems walks the entire HDF5 tree,
    which handles arbitrary nesting (e.g., observations/images/cam_high).
    """
    datasets = {}
    with h5py.File(filepath, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {"shape": obj.shape, "dtype": str(obj.dtype)}

        f.visititems(visitor)
    return datasets


def build_features(hdf5_info: dict, state_key: str, image_keys: list[str]) -> dict:
    """Construct the LeRobot features dict from HDF5 metadata.

    LeRobot stores each modality under a standardized key with an associated
    features dict that describes dtype, shape, and dimension names. The dtype
    "video" tells LeRobot to encode image frames into an mp4 file instead of
    storing raw arrays — this saves 10–30× disk space for typical robot datasets.
    """
    features = {}

    # State (joint positions): stored as float32 array, one row per timestep
    state_path = f"observations/{state_key}"
    if state_path in hdf5_info:
        state_dim = hdf5_info[state_path]["shape"][1]
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"motors": [f"joint_{i}" for i in range(state_dim)]},
        }

    # Camera images: dtype="video" tells LeRobot to write mp4 via ffmpeg.
    # The shape is (H, W, C) per frame (not (T, H, W, C)) — T is implicit.
    for img_key in image_keys:
        hdf5_path = f"observations/images/{img_key}"
        if hdf5_path not in hdf5_info:
            print(f"  WARNING: image key '{hdf5_path}' not found in HDF5, skipping")
            continue
        H, W, C = hdf5_info[hdf5_path]["shape"][1:]  # drop the T dimension
        features[f"observation.images.{img_key}"] = {
            "dtype": "video",
            "shape": (H, W, C),
            "names": ["height", "width", "channel"],
        }

    # Action: the target sequence the policy must learn to predict
    action_dim = hdf5_info["action"]["shape"][1]
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": {"motors": [f"motor_{i}" for i in range(action_dim)]},
    }

    return features


def convert(
    input_dir: str,
    repo_id: str,
    root: str,
    fps: int,
    state_key: str = "qpos",
    image_keys: list[str] = None,
    task_description: str = "Pick and place",
    vcodec: str = "auto",
):
    if image_keys is None:
        image_keys = ["cam_high"]

    episode_files = get_episode_files(input_dir)
    if not episode_files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")
    print(f"Found {len(episode_files)} episodes")

    # Inspect structure from the first file; all episodes must share the same layout
    hdf5_info = inspect_hdf5(os.path.join(input_dir, episode_files[0]))
    print("HDF5 structure:")
    for k, v in hdf5_info.items():
        print(f"  {k}: shape={v['shape']} dtype={v['dtype']}")

    features = build_features(hdf5_info, state_key, image_keys)
    print(f"\nLeRobot features: {list(features.keys())}")

    output_path = Path(root)
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # LeRobot.create writes meta/info.json (episode metadata) and sets up the
    # HuggingFace dataset structure. use_videos=True enables mp4 encoding.
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_path,
        features=features,
        use_videos=True,
        vcodec=vcodec,
    )

    # Episode lifecycle: add_frame (per timestep) → save_episode → repeat → finalize.
    # save_episode flushes the accumulated frames of one episode to disk and encodes
    # the video. finalize writes dataset_statistics.json and closes the HF dataset.
    for ep_idx, ep_file in enumerate(tqdm(episode_files, desc="Converting")):
        filepath = os.path.join(input_dir, ep_file)
        with h5py.File(filepath, "r") as f:
            # Load the full episode into memory to avoid repeated HDF5 reads per frame
            actions = f["action"][:].astype(np.float32)
            state_path = f"observations/{state_key}"
            states = f[state_path][:].astype(np.float32) if state_path in f else None
            images = {}
            for img_key in image_keys:
                hdf5_path = f"observations/images/{img_key}"
                if hdf5_path in f:
                    images[img_key] = f[hdf5_path][:]
            T = actions.shape[0]

        for t in range(T):
            frame = {
                "action": actions[t],
                "task": task_description,
            }
            if states is not None:
                frame["observation.state"] = states[t]
            for img_key, img_data in images.items():
                frame[f"observation.images.{img_key}"] = img_data[t]

            ds.add_frame(frame)

        ds.save_episode()

        if (ep_idx + 1) % 20 == 0:
            print(f"  Converted {ep_idx + 1}/{len(episode_files)} episodes")

    ds.finalize()
    print(f"\nDone. Dataset saved to: {output_path}")
    print(f"  repo_id:       {repo_id}")
    print(f"  total episodes: {ds.num_episodes}")
    print(f"  total frames:   {ds.num_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 robot episodes to LeRobot format")
    parser.add_argument("--input", required=True, help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--repo_id", default="local/robot_dataset")
    parser.add_argument("--root", required=True, help="Output root directory")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--state_key", default="qpos", help="HDF5 key under observations/ for robot state"
    )
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=["cam_high"],
        help="Image keys under observations/images/",
    )
    parser.add_argument("--task", default="Pick and place")
    parser.add_argument("--vcodec", default="auto")
    args = parser.parse_args()

    convert(
        input_dir=args.input,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        state_key=args.state_key,
        image_keys=args.image_keys,
        task_description=args.task,
        vcodec=args.vcodec,
    )
