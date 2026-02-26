"""
Trial recorder — saves one real-robot rollout to disk for demos and success-rate reporting.

Layout of one trial (record_dir/trial_YYYYmmdd_HHMMSS/):
  <camera>.mp4             — one stream per camera (e.g. exterior.mp4, wrist.mp4) at the
                             control rate, one frame per executed action
  steps.npz                — per-step arrays: t, state, action, gripper_open, inference_ms, obs_ms
  meta.json                — checkpoint, settings, n_steps, duration, success (true/false/null), notes

Video encoding runs in a background thread so a slow disk never stalls the control loop.
scripts/summarize_trials.py aggregates meta.json files into a success-rate table.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np


class _VideoWriterThread:
    def __init__(self, path, fps):
        self._path = path
        self._fps = fps
        self._queue = queue.Queue(maxsize=256)
        self._writer = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if self._writer is not None:
            self._writer.release()

    def write(self, rgb_frame):
        try:
            self._queue.put_nowait(rgb_frame)
        except queue.Full:
            pass  # dropping a video frame is better than stalling the robot

    def close(self):
        self._queue.put(None)
        self._thread.join()


class TrialRecorder:
    def __init__(self, record_dir, fps, meta=None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(record_dir, exist_ok=True)
        # Never overwrite a previous trial, even if two start within the same second
        suffix = 0
        while True:
            name = f"trial_{stamp}" + (f"_{suffix}" if suffix else "")
            self.trial_dir = os.path.join(record_dir, name)
            try:
                os.makedirs(self.trial_dir)
                break
            except FileExistsError:
                suffix += 1
        self._fps = fps
        self._meta = dict(meta or {})
        self._videos = {}
        self._steps = {
            "t": [],
            "state": [],
            "action": [],
            "gripper_open": [],
            "inference_ms": [],
            "obs_ms": [],
        }
        self._t0 = time.monotonic()

    def add_step(self, obs, action, gripper_open, inference_ms, obs_ms):
        """obs: dict from robot.get_obs() — "state" plus one "<camera>_rgb" image each."""
        for key in (k for k in obs if k.endswith("_rgb")):
            if key not in self._videos:
                name = key.replace("_rgb", "") + ".mp4"
                self._videos[key] = _VideoWriterThread(
                    os.path.join(self.trial_dir, name), self._fps
                )
            self._videos[key].write(obs[key])
        self._steps["t"].append(time.monotonic() - self._t0)
        self._steps["state"].append(np.asarray(obs["state"], dtype=np.float32))
        self._steps["action"].append(np.asarray(action, dtype=np.float32))
        self._steps["gripper_open"].append(bool(gripper_open))
        self._steps["inference_ms"].append(float(inference_ms))
        self._steps["obs_ms"].append(float(obs_ms))

    @property
    def n_steps(self):
        return len(self._steps["t"])

    def close(self, success=None, notes=""):
        """Flush videos and write steps.npz + meta.json. success: True / False / None."""
        for v in self._videos.values():
            v.close()
        arrays = {k: np.asarray(v) for k, v in self._steps.items()}
        np.savez_compressed(os.path.join(self.trial_dir, "steps.npz"), **arrays)
        meta = {
            **self._meta,
            "n_steps": self.n_steps,
            "duration_s": float(arrays["t"][-1]) if self.n_steps else 0.0,
            "success": success,
            "notes": notes,
        }
        with open(os.path.join(self.trial_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return self.trial_dir


def ask_trial_outcome():
    """Ask the operator whether the trial succeeded. Returns (success, notes)."""
    try:
        answer = input("Trial successful? [y/n, Enter = don't record]: ").strip().lower()
        success = {"y": True, "yes": True, "n": False, "no": False}.get(answer)
        notes = input("Notes (optional): ").strip() if success is not None else ""
    except EOFError:  # non-interactive run
        success, notes = None, ""
    return success, notes
