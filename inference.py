"""
Real robot inference — deploy a trained ACT policy on a UR robot.

Hardware:
    - UR robot via RTDE (configurable IP, tested on UR3e/UR5)
    - Weiss CRG 30-050 gripper via USB serial (optional)
    - One or two Intel RealSense cameras (exterior + wrist)

Action format (7-DOF single arm):
    action[:6]  — joint angles in radians (absolute targets)
    action[6]   — gripper command  0=open, 1=close  (if use_gripper)

For dual-arm (ALOHA-style, action_dim=14), set --no_gripper and handle
arm routing in your own wrapper around this script.

Consistency with training (read from config.json written by train.py):
    - image resolution  : frames are centre-cropped/resized to config["image_hw"]
    - control frequency : defaults to the dataset fps, warns on mismatch
    - camera order      : follows config["camera_names"], fails if a camera is missing
    - qpos range        : warns if the robot state is far outside the training stats

Usage:
    # Dry run (robot is not moved at all — use for sanity check)
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 --dry_run

    # Real inference, temporal ensembling enabled
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 \\
        --temporal_agg --max_steps 400

    # Two cameras
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 \\
        --cam_exterior 105422061000 --cam_wrist 352122273671

    # Real-robot evaluation: 20 rollouts, operator labels success, videos saved,
    # summary written to <ckpt_dir>/eval_<timestamp>.json
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 \\
        --temporal_agg --num_episodes 20 --record_dir eval_videos
"""

import argparse
import json
import logging
import os
import threading
import time

import cv2
import numpy as np
import torch

from eval import load_policy

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Configuration constants  (override via CLI args or subclass)
# ══════════════════════════════════════════════════════════════════════

GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_OPEN_THRESH_MM = 5.0

CAM_SERIAL_EXTERIOR = "105422061000"
CAM_SERIAL_WRIST = "352122273671"
IMAGE_SIZE = 480  # square crop/resize target

HOME_DEG = [45, -20, -140, -40, -270, 0]
HOME_RAD = np.radians(HOME_DEG).astype(np.float32)

SERVO_J_TIME = 0.1  # RTDE servoJ lookahead time [s]
SERVO_J_LOOKAHEAD = 0.2
SERVO_J_GAIN = 200
MAX_JOINT_VEL = 0.8  # rad/s velocity safety clamp


# ══════════════════════════════════════════════════════════════════════
# Timing
# ══════════════════════════════════════════════════════════════════════


def precise_wait(t_end: float, slack: float = 0.001, time_func=time.monotonic) -> None:
    """Sleep until t_end with sub-millisecond accuracy (sleep + busy-wait)."""
    t_now = time_func()
    if t_now < t_end - slack:
        time.sleep(t_end - t_now - slack)
    while time_func() < t_end:
        pass


# ══════════════════════════════════════════════════════════════════════
# Gripper driver — Weiss CRG 30-050 via DC-IOLink USB adapter
# ══════════════════════════════════════════════════════════════════════


class WeissCRGGripper:
    """
    Minimal driver for the Weiss CRG 30-050 finger gripper.

    Protocol: PDOUT=[02,00] = open | PDOUT=[03,00] = close | PDOUT=[07,00] = home
    Feedback:  @PDIN=[B0,B1,B2,B3] where pos_mm = ((B0<<8)|B1) / 100

    Thread-safe via an internal lock (camera thread may call get_width
    concurrently with the control thread calling move_to_pos).
    """

    FLAG_OPEN = 1
    FLAG_CLOSED = 2

    def __init__(self, port: str = GRIPPER_PORT, baudrate: int = GRIPPER_BAUDRATE):
        import serial as _serial

        self._lock = threading.Lock()
        self._ser = _serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        log.info(f"[Gripper] opened {port} @ {baudrate} baud")
        self._initialise()

    # ── Low-level serial ──

    def _send(self, cmd: str, wait: float = 0.3) -> None:
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                log.warning(f"[Gripper] serial write '{cmd}': {e}")
        time.sleep(wait)

    def _parse_pdin(self, line: str) -> bool:
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._position_mm = ((parts[0] << 8) | parts[1]) / 100.0
            self._flags = parts[3] if len(parts) >= 4 else 0
            return True
        except Exception:
            return False

    def _read_pdin(self, timeout: float = 1.0) -> bool:
        t0 = time.monotonic()
        with self._lock:
            saved, self._ser.timeout = self._ser.timeout, 0.15
        try:
            while time.monotonic() - t0 < timeout:
                with self._lock:
                    line = self._ser.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("@PDIN=[") and self._parse_pdin(line):
                    return True
        finally:
            with self._lock:
                self._ser.timeout = saved
        return False

    def _wait_flag(self, flag_bit: int, timeout: float = 6.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._read_pdin(0.5) and (self._flags & (1 << flag_bit)):
                return True
        return False

    def _set_positions(self, open_mm: float = GRIPPER_MAX_MM, close_mm: float = 0.5) -> None:
        def enc(mm: float) -> str:
            v = int(mm * 100)
            return f"[{(v >> 8) & 0xFF:02x},{v & 0xFF:02x}]"

        self._send(f"SETPARAM(96, 2, {enc(open_mm)})", 0.3)
        self._send(f"SETPARAM(96, 1, {enc(close_mm)})", 0.3)
        self._send("SETPARAM(96, 3, [64])", 0.3)

    # ── Public API ──

    def _initialise(self) -> None:
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?", "RESTART()", "OPERATE()"]:
            self._send(cmd, 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        log.info("[Gripper] initialised")

    def home(self) -> None:
        log.info("[Gripper] homing...")
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        self._send("PDOUT=[07,00]", 0.2)
        self._wait_flag(self.FLAG_OPEN, timeout=10.0)
        log.info("[Gripper] home complete — open")

    def move_to_pos(self, width_mm: float) -> None:
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        cmd = "PDOUT=[02,00]" if width_mm > GRIPPER_OPEN_THRESH_MM else "PDOUT=[03,00]"
        self._send(cmd, 0.2)

    def get_width(self) -> float:
        self._read_pdin(timeout=0.08)
        return self._position_mm

    def close(self) -> None:
        self._send("PDOUT=[00,00]", 0.3)
        self._send("FALLBACK(1)", 0.3)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
        log.info("[Gripper] serial closed")


# ══════════════════════════════════════════════════════════════════════
# Camera utilities
# ══════════════════════════════════════════════════════════════════════


def _center_crop_resize(
    bgr: np.ndarray, out_hw: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> np.ndarray:
    """
    Centre-crop to the target aspect ratio, resize to out_hw, BGR→RGB.

    out_hw should be the image resolution the policy was trained on
    (config["image_hw"]); for a square target this is the usual square
    centre crop, for e.g. 480x640 it keeps the full frame.
    """
    h, w = bgr.shape[:2]
    out_h, out_w = out_hw
    target_ratio = out_w / out_h
    if w / h > target_ratio:  # too wide → crop width
        cw, ch = int(round(h * target_ratio)), h
    else:  # too tall → crop height
        cw, ch = w, int(round(w / target_ratio))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = bgr[y0 : y0 + ch, x0 : x0 + cw]
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _start_realsense(serial_num: str):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial_num)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
    log.info(f"[Camera] started — serial {serial_num}")
    return pipeline


def _grab_frame(pipeline, retries: int = 3) -> np.ndarray:
    for attempt in range(retries):
        frames = pipeline.wait_for_frames(timeout_ms=500)
        color = frames.get_color_frame()
        if color:
            return np.asanyarray(color.get_data())
        log.warning(f"[Camera] no frame, retry {attempt + 1}/{retries}")
    raise RuntimeError("Camera: failed to grab frame after retries")


# ══════════════════════════════════════════════════════════════════════
# Robot driver — UR + optional Weiss gripper + RealSense cameras
# ══════════════════════════════════════════════════════════════════════


class URRobot:
    """
    UR robot arm + optional Weiss gripper + RealSense camera(s).

    Wraps RTDE control/receive interfaces and camera pipelines into a
    single object with a simple get_obs() / apply_action() interface.

    Args:
        robot_ip: robot controller IP address
        frequency: control loop frequency [Hz]
        use_gripper: whether to connect the Weiss gripper
        cam_serial_exterior: RealSense serial for exterior camera (None to skip)
        cam_serial_wrist: RealSense serial for wrist camera (None to skip)
        image_hw: (height, width) of camera frames fed to the policy [px]
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: int = 50,
        use_gripper: bool = True,
        cam_serial_exterior: str = CAM_SERIAL_EXTERIOR,
        cam_serial_wrist: str = None,
        image_hw: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    ):
        import rtde_control
        import rtde_receive

        self._dt = 1.0 / frequency
        self._image_hw = tuple(image_hw)

        log.info(f"[URRobot] connecting to {robot_ip}...")
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        log.info("[URRobot] RTDE connected")

        self._gripper = None
        self._last_gripper_open = True
        if use_gripper:
            self._gripper = WeissCRGGripper()

        self._cameras = {}
        if cam_serial_exterior:
            self._cameras["exterior"] = _start_realsense(cam_serial_exterior)
        if cam_serial_wrist:
            self._cameras["wrist"] = _start_realsense(cam_serial_wrist)

        if not self._cameras:
            raise RuntimeError("At least one camera serial must be provided")

        self._last_cmd_rad = HOME_RAD.copy()

    def home(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self._rtde_c.servoStop()
        time.sleep(0.2)
        log.info("[URRobot] moving to home pose...")
        self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.5, acceleration=0.5)
        self._last_cmd_rad = HOME_RAD.copy()
        if self._gripper:
            self._gripper.home()
            self._last_gripper_open = True
        log.info("[URRobot] ready")

    def get_obs(self) -> dict:
        """
        Read current state and capture camera frames.

        Returns dict:
            qpos   : (state_dim,) float32  — joint positions [rad] + optional gripper
            images : dict[cam_name -> (H, W, 3) uint8 RGB]
        """
        qpos = np.array(self._rtde_r.getActualQ(), dtype=np.float32)

        if self._gripper:
            w = self._gripper.get_width()
            gripper_state = np.array([0.0 if w > GRIPPER_OPEN_THRESH_MM else 1.0], dtype=np.float32)
            qpos = np.concatenate([qpos, gripper_state])

        images = {}
        for name, pipe in self._cameras.items():
            bgr = _grab_frame(pipe)
            images[name] = _center_crop_resize(bgr, self._image_hw)

        return {"qpos": qpos, "images": images}

    def apply_action(self, action: np.ndarray) -> None:
        """
        Send action to the robot.

        action[:6]  — joint angles [rad] (absolute, velocity-clamped)
        action[6]   — gripper (0=open, 1=close), present only if use_gripper=True
        """
        target_rad = action[:6].astype(np.float64)
        max_step = MAX_JOINT_VEL * self._dt
        delta = target_rad - self._last_cmd_rad[:6]
        cmd_rad = self._last_cmd_rad[:6] + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad[:6] = cmd_rad

        try:
            self._rtde_c.servoJ(
                cmd_rad.tolist(),
                0,
                0,
                SERVO_J_TIME,
                SERVO_J_LOOKAHEAD,
                SERVO_J_GAIN,
            )
        except Exception as e:
            log.warning(f"[URRobot] servoJ error: {e}")
            self._last_cmd_rad[:6] = np.array(self._rtde_r.getActualQ(), dtype=np.float32)

        if self._gripper and len(action) > 6:
            want_open = float(action[6]) < 0.5
            if want_open != self._last_gripper_open:
                mm = GRIPPER_MAX_MM if want_open else 0.0
                self._gripper.move_to_pos(mm)
                self._last_gripper_open = want_open
                log.info(f"[URRobot] gripper {'opening' if want_open else 'closing'}")

    def hold(self) -> None:
        """Stop servoing and hold the current pose (between evaluation episodes)."""
        try:
            self._rtde_c.servoStop()
        except Exception as e:
            log.warning(f"[URRobot] servoStop error: {e}")

    def stop(self) -> None:
        try:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        except Exception:
            pass
        if self._gripper:
            self._gripper.close()
        for pipe in self._cameras.values():
            try:
                pipe.stop()
            except Exception:
                pass
        log.info("[URRobot] stopped")


# ══════════════════════════════════════════════════════════════════════
# Observation preprocessing
# ══════════════════════════════════════════════════════════════════════


def build_image_tensor(obs: dict, cam_order: list[str]) -> torch.Tensor:
    """
    Convert raw camera frames to ACT input tensor.

    Args:
        obs: dict from URRobot.get_obs()
        cam_order: list of camera names in the order the model expects

    Returns:
        image: (num_cam, 3, H, W) float32 in [0, 1]
    """
    frames = []
    for name in cam_order:
        rgb = obs["images"][name]  # (H, W, 3) uint8
        t = torch.from_numpy(rgb).float() / 255.0  # (H, W, 3) float [0,1]
        frames.append(t.permute(2, 0, 1))  # (3, H, W)
    return torch.stack(frames, dim=0)  # (num_cam, 3, H, W)


# ══════════════════════════════════════════════════════════════════════
# Live visualisation
# ══════════════════════════════════════════════════════════════════════


def _draw_action_bars(action: np.ndarray, height: int, width: int = 200) -> np.ndarray:
    """
    Sidebar panel with one horizontal bar per action dimension.

    Bar length is proportional to |value|, normalised by the max absolute value
    across all dims so the widest bar fills the panel. Green = positive, red = negative
    (BGR convention). Useful for quickly spotting which joints are saturated or reversed.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(action)
    row_h = max(10, (height - 24) // max(n, 1))
    bar_max = float(np.abs(action).max()) or 1.0  # avoid divide-by-zero for zero actions
    bar_area = width - 64  # pixels reserved for the value bars (label takes the rest)

    cv2.putText(
        panel, "Actions", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA
    )

    for i, val in enumerate(action):
        y0 = 20 + i * row_h
        bar_px = int(abs(val) / bar_max * bar_area)
        color = (60, 200, 60) if val >= 0 else (60, 60, 220)  # BGR: green / red
        if bar_px > 0:
            cv2.rectangle(panel, (52, y0 + 2), (52 + bar_px, y0 + row_h - 3), color, -1)
        label = f"a{i}:{val:+.2f}"
        cv2.putText(
            panel,
            label,
            (2, y0 + row_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    return panel


def make_vis_frame(
    obs: dict,
    action: np.ndarray,
    step: int,
    inference_ms: float,
    temporal_agg: bool,
    chunk_size: int = 1,
    chunk_step: int = 0,
    ensemble_depth: int = 0,
) -> np.ndarray:
    """
    Side-by-side camera view with status overlay + action bars sidebar.

    Camera panels are tiled horizontally at 320 px height (aspect preserved).
    The action bars sidebar (right) shows per-joint magnitudes at a glance —
    handy for spotting saturated or reversed joints without reading numbers.
    """
    disp = 320
    panels = []
    for name in sorted(obs["images"]):
        rgb = obs["images"][name]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        panels.append(cv2.resize(bgr, (int(round(disp * w / h)), disp)))
    frame = np.concatenate(panels, axis=1)

    gripper_str = ""
    if len(action) > 6:
        gripper_str = f"  Grip: {'open' if action[6] < 0.5 else 'close'}"

    if temporal_agg:
        mode_str = f"ensemble (depth {ensemble_depth})"
    else:
        mode_str = f"chunk {chunk_step + 1}/{chunk_size}"

    lines = [
        f"Step: {step}",
        f"Infer: {inference_ms:.0f} ms",
        f"Mode: {mode_str}" + gripper_str,
        "q / ESC - end episode",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (10, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Action bars panel appended to the right of camera tiles
    bars = _draw_action_bars(action, height=disp)
    return np.concatenate([frame, bars], axis=1)


# ══════════════════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════════════════


def resolve_camera_order(camera_names: list[str], available: list[str]) -> list[str]:
    """
    Map the model's camera names (dataset image keys) to robot camera names.

    The model expects cameras in the order of config["camera_names"], e.g.
    ["observation.images.exterior", "observation.images.wrist"]. The robot
    names its cameras by the last key segment ("exterior", "wrist"), so we
    match on that — silently stacking cameras in a different order than in
    training would feed the wrong view into each image slot.

    Raises:
        ValueError if a camera the model needs is not connected.
    """
    order = [name.rsplit(".", 1)[-1] for name in camera_names]
    missing = [c for c in order if c not in available]
    if missing:
        raise ValueError(
            f"Model expects cameras {order} (from {camera_names}) but only {available} "
            f"are connected. Missing: {missing}. Check --cam_exterior / --cam_wrist."
        )
    return order


def check_qpos_in_distribution(qpos: np.ndarray, normalizer, z_thresh: float = 5.0) -> None:
    """
    Warn if the robot's current state is far outside the training distribution.

    Catches unit / convention mismatches between data collection and
    deployment before the policy moves the arm — e.g. degrees vs radians,
    gripper width in mm vs a 0/1 flag, or a home pose never seen in the data.
    """
    if normalizer.qpos_mean is None:
        return
    mean = normalizer.qpos_mean.numpy()
    std = normalizer.qpos_std.numpy()
    q = qpos[: len(mean)]
    z = (q - mean) / std
    bad = np.where(np.abs(z) > z_thresh)[0]
    if len(bad) == 0:
        log.info(f"qpos check OK — max |z| = {np.abs(z).max():.2f}")
        return
    for i in bad:
        log.warning(
            f"qpos[{i}] = {q[i]:.3f} is {z[i]:+.1f} std from the training mean "
            f"({mean[i]:.3f} ± {std[i]:.3f}) — check units / gripper convention / home pose"
        )


def run_episode(
    pipeline,
    robot: URRobot,
    config: dict,
    cam_order: list[str],
    frequency: int = 50,
    max_steps: int = 400,
    temporal_agg: bool = False,
    dry_run: bool = False,
    no_vis: bool = False,
    record: bool = False,
) -> tuple[int, list[np.ndarray]]:
    """
    Run one ACT inference episode on the robot.

    ACT only needs the *current* observation (no history window).
    The InferencePipeline handles action-chunk re-querying internally:
      - temporal_agg=False: re-queries every chunk_size steps
      - temporal_agg=True:  queries every step, smooths with exp weighting

    The episode ends after max_steps or when the operator presses q / ESC in
    the visualisation window. Ctrl+C propagates to the caller (abort all).
    The robot is NOT stopped here so that main() can run several episodes.

    Args:
        pipeline: InferencePipeline from eval.load_policy()
        robot: URRobot instance
        config: model config dict (for state_dim, action_dim etc.)
        cam_order: camera names in model order
        frequency: control loop [Hz]
        max_steps: episode length cap
        temporal_agg: whether temporal ensembling is active (for display)
        dry_run: if True, skips robot.apply_action()
        no_vis: if True, disables cv2 display (for headless / SSH deployments)
        record: if True, collect visualisation frames for a video

    Returns:
        (steps executed, recorded frames)
    """
    dt = 1.0 / frequency
    state_dim = config["state_dim"]
    chunk_size = pipeline.chunk_size
    frames = []

    pipeline.reset()
    log.info(
        f"Episode start — freq={frequency} Hz  max_steps={max_steps}  "
        f"temporal_agg={temporal_agg}  dry_run={dry_run}"
    )

    step = 0
    while step < max_steps:
        t_step_start = time.monotonic()

        # ── Observe ──
        obs = robot.get_obs()
        qpos = torch.tensor(obs["qpos"][:state_dim], dtype=torch.float32)
        image = build_image_tensor(obs, cam_order)

        # ── Inference ──
        t_inf = time.monotonic()
        action_t = pipeline.get_action(step, qpos, image)
        inference_ms = (time.monotonic() - t_inf) * 1000
        action_np = action_t.cpu().numpy()

        log.debug(
            f"step {step:>4}:  infer={inference_ms:.0f} ms  "
            f"action=[{' '.join(f'{a:.3f}' for a in action_np[:4])} ...]"
        )
        if inference_ms > dt * 1000:
            log.warning(f"step {step}: inference {inference_ms:.0f} ms exceeds {dt * 1000:.0f} ms")

        # ── Execute ──
        if not dry_run:
            robot.apply_action(action_np)

        # ── Visualise / record ──
        if not no_vis or record:
            # ensemble_depth: how many overlapping predictions cover this step
            # (saturates at chunk_size once the buffer is full)
            ensemble_depth = min(step + 1, chunk_size) if temporal_agg else 0
            frame = make_vis_frame(
                obs,
                action_np,
                step,
                inference_ms,
                temporal_agg,
                chunk_size=chunk_size,
                chunk_step=step % chunk_size,
                ensemble_depth=ensemble_depth,
            )
            if record:
                frames.append(frame)
            if not no_vis:
                cv2.imshow("ACT Inference", frame)
                key = cv2.pollKey()
                if key in (ord("q"), 27):  # q or ESC — end this episode
                    log.info("Episode ended by operator.")
                    step += 1
                    break

        step += 1

        # ── Hold loop timing ──
        precise_wait(t_step_start + dt)

    log.info(f"Episode ended at step {step}.")
    return step, frames


def save_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Write BGR frames to an mp4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    log.info(f"Saved video: {path}")


def ask_operator(prompt: str, choices: str = "yn") -> str:
    """Block until the operator types one of the given single-letter choices."""
    while True:
        ans = input(f"{prompt} [{'/'.join(choices)}]: ").strip().lower()
        if ans in choices:
            return ans


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT real robot inference / evaluation")

    # Policy
    parser.add_argument(
        "--ckpt_dir", required=True, help="Directory with policy_best.ckpt and dataset_stats.pkl"
    )
    parser.add_argument(
        "--temporal_agg", action="store_true", help="Use temporal ensembling (query every step)"
    )
    parser.add_argument("--k", type=float, default=0.01, help="Temporal ensemble decay constant")
    parser.add_argument("--device", default="cuda")

    # Robot
    parser.add_argument("--robot_ip", required=True, help="UR robot controller IP address")
    parser.add_argument(
        "--frequency",
        type=int,
        default=None,
        help="Control loop frequency [Hz] (default: dataset fps from config.json, else 50)",
    )
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--no_gripper", action="store_true", help="Disable Weiss gripper")

    # Cameras
    parser.add_argument(
        "--cam_exterior",
        default=CAM_SERIAL_EXTERIOR,
        help="RealSense serial number for exterior camera",
    )
    parser.add_argument(
        "--cam_wrist", default=None, help="RealSense serial number for wrist camera (optional)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Square image size override (default: training resolution from config.json)",
    )

    # Evaluation
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of rollouts; with >1 the operator labels each one success/failure",
    )
    parser.add_argument(
        "--record_dir", default=None, help="Save an mp4 of every episode into this directory"
    )

    # Misc
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Full pipeline without moving the robot (no homing, no servo commands)",
    )
    parser.add_argument(
        "--no_vis", action="store_true", help="Disable cv2 display (headless / SSH deployments)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    # ── Load policy ──
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    log.info(f"Loading policy from {args.ckpt_dir} ...")
    pipeline, config = load_policy(
        ckpt_dir=args.ckpt_dir,
        device=device,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_steps,
        k=args.k,
    )
    log.info(
        f"  state_dim={config['state_dim']}  action_dim={config['action_dim']}  "
        f"chunk_size={config['num_queries']}  cameras={config['camera_names']}"
    )

    # ── Consistency with training data ──
    # Control rate: actions are a time series sampled at the dataset fps;
    # replaying them at another rate speeds up / slows down every motion.
    train_fps = config.get("fps")
    frequency = args.frequency or train_fps or 50
    if train_fps is not None and frequency != train_fps:
        log.warning(
            f"Control frequency {frequency} Hz differs from training data fps {train_fps} — "
            f"the policy's motions will play back {frequency / train_fps:.2f}x faster"
        )

    # Image resolution: must match what the policy saw in training
    if args.image_size is not None:
        image_hw = (args.image_size, args.image_size)
        if config.get("image_hw") and list(image_hw) != list(config["image_hw"]):
            log.warning(
                f"--image_size {image_hw} overrides training resolution {config['image_hw']}"
            )
    elif config.get("image_hw"):
        image_hw = tuple(config["image_hw"])
    else:
        log.warning(f"config.json has no image_hw, assuming {IMAGE_SIZE}x{IMAGE_SIZE}")
        image_hw = (IMAGE_SIZE, IMAGE_SIZE)

    # Camera order: follow config["camera_names"], not CLI order
    available = (["exterior"] if args.cam_exterior else []) + (["wrist"] if args.cam_wrist else [])
    cam_order = resolve_camera_order(config["camera_names"], available)
    log.info(f"  camera order: {cam_order}  image_hw: {image_hw}  frequency: {frequency} Hz")

    # ── Connect robot ──
    log.info(f"Connecting to UR robot at {args.robot_ip} ...")
    robot = URRobot(
        robot_ip=args.robot_ip,
        frequency=frequency,
        use_gripper=not args.no_gripper,
        cam_serial_exterior=args.cam_exterior if args.cam_exterior else None,
        cam_serial_wrist=args.cam_wrist,
        image_hw=image_hw,
    )

    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
    evaluating = args.num_episodes > 1
    results = []

    try:
        # ── Warmup: cameras + CUDA kernels before the timed loop ──
        log.info("Warming up cameras and inference...")
        for _ in range(5):
            robot.get_obs()
            time.sleep(0.05)
        pipeline.get_action(
            0,
            torch.zeros(config["state_dim"]),
            torch.zeros(len(cam_order), 3, *image_hw),
        )

        for ep in range(args.num_episodes):
            if ep > 0 or evaluating:
                input(f"\n[Episode {ep + 1}/{args.num_episodes}] Reset the scene, then press Enter")

            if args.dry_run:
                log.info("Dry run — skipping homing, robot stays where it is")
            else:
                robot.home()
            check_qpos_in_distribution(robot.get_obs()["qpos"], pipeline.normalizer)

            steps, frames = run_episode(
                pipeline=pipeline,
                robot=robot,
                config=config,
                cam_order=cam_order,
                frequency=frequency,
                max_steps=args.max_steps,
                temporal_agg=args.temporal_agg,
                dry_run=args.dry_run,
                no_vis=args.no_vis,
                record=args.record_dir is not None,
            )
            # Freeze the arm where it is before talking to the operator
            if not args.dry_run:
                robot.hold()

            result = {"episode": ep, "steps": steps}
            if evaluating:
                result["success"] = ask_operator("Was the episode successful?") == "y"
                n_ok = sum(r["success"] for r in results) + result["success"]
                log.info(f"Success so far: {n_ok}/{ep + 1}")
            if args.record_dir:
                tag = "" if not evaluating else ("_success" if result["success"] else "_fail")
                video = os.path.join(args.record_dir, f"episode_{ep:03d}{tag}.mp4")
                save_video(frames, video, fps=frequency)
                result["video"] = video
            results.append(result)

    except KeyboardInterrupt:
        log.info("Aborted by user.")
    finally:
        if not args.no_vis:
            cv2.destroyAllWindows()
        robot.stop()

    # ── Evaluation summary ──
    if evaluating and results:
        n_ok = sum(r["success"] for r in results)
        summary = {
            "ckpt_dir": args.ckpt_dir,
            "temporal_agg": args.temporal_agg,
            "k": args.k,
            "frequency": frequency,
            "max_steps": args.max_steps,
            "n_episodes": len(results),
            "n_success": n_ok,
            "success_rate": n_ok / len(results),
            "episodes": results,
        }
        out = os.path.join(args.ckpt_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Success rate: {n_ok}/{len(results)} = {n_ok / len(results):.0%}  → {out}")


if __name__ == "__main__":
    main()
