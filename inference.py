"""
Real robot inference — deploy a trained ACT policy on a UR robot.

Hardware:
    - UR robot via RTDE (configurable IP, tested on UR3e/UR5)
    - Weiss CRG 30-050 gripper via USB serial (optional)
    - One or two Intel RealSense cameras (exterior + wrist)

Conventions (same as the Diffusion Policy reimplementation and its training data):
    state  (7,) = [6 joint angles in DEGREES, gripper (1 = open, 0 = closed)]
    action (7,) = [6 absolute joint targets in DEGREES, gripper (1 = open, 0 = closed)]
    getActualQ() returns radians: converted to degrees for the state, and the action is
    converted back to radians before servoJ. check_qpos_in_distribution() warns before
    the first move if a new dataset was recorded with different units / conventions.

For dual-arm (ALOHA-style, action_dim=14), set --no_gripper and handle
arm routing in your own wrapper around this script.

Consistency with training (read from the checkpoint config written by train.py):
    - image resolution  : frames are centre-cropped/resized to config["image_hw"]
    - control frequency : defaults to the dataset fps, refuses a mismatch unless allowed
    - camera order      : follows config["camera_names"], fails if a camera is missing
    - state range       : warns if the robot state is far outside the training stats

Usage:
    # Dry run (robot is not moved at all — use for sanity check)
    python inference.py --checkpoint outputs/policy_best.pt --robot_ip 192.168.1.1 --dry_run

    # Real inference, temporal ensembling enabled
    python inference.py --checkpoint outputs/policy_best.pt --robot_ip 192.168.1.1 \\
        --temporal_agg

    # Real-robot evaluation: 20 rollouts, each saved to trials/ with the operator's
    # success label; summarise with scripts/summarize_trials.py --record_dir trials/
    python inference.py --checkpoint outputs/policy_best.pt --robot_ip 192.168.1.1 \\
        --temporal_agg --num_episodes 20 --record_dir trials/
"""

import argparse
import logging
import sys
import threading
import time

import cv2
import numpy as np
import torch

from act.control import GripperHysteresis, TimingStats, precise_wait
from act.policy.loader import load_policy, resolve_checkpoint

log = logging.getLogger(__name__)


# ══════════════════════════════ Configuration ══════════════════════════════

GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUDRATE = 9600
GRIPPER_MAX_MM = 50
GRIPPER_OPEN_THRESH_MM = 5.0

# Gripper convention of the training data: 1 = open, 0 = closed (every demo starts and
# ends open). Both the state fed to the policy and the action channel use it.
GRIPPER_STATE_OPEN = 1.0
GRIPPER_STATE_CLOSED = 0.0

# Hysteresis thresholds for gripper state switching (see act/control.py).
GRIPPER_CLOSE_THRESH = 0.35
GRIPPER_OPEN_THRESH = 0.65
# Minimum steps to hold gripper state before allowing another switch.
# At 20 Hz: 20 steps = 1 second minimum hold.
GRIPPER_MIN_DWELL_STEPS = 20

CAM_SERIAL_EXTERIOR = "105422061000"
CAM_SERIAL_WRIST = "352122273671"
IMAGE_SIZE = 480  # square crop/resize target when the checkpoint has no image_hw
# A cached camera frame / gripper reading older than this means the reader stalled
MAX_SENSOR_AGE_S = 0.5

HOME_DEG = [45, -20, -140, -40, -270, 0]
HOME_RAD = np.radians(HOME_DEG).astype(np.float32)

SERVO_J_LOOKAHEAD = 0.2
SERVO_J_GAIN = 200
# Default safety velocity clamp (override with --max_joint_vel). Normal demo speeds
# (≈ 30–60°/s ≈ 0.5–1.0 rad/s) pass through unclipped.
MAX_JOINT_VEL = 1.0  # rad/s

# Per-joint position limits (radians), applied before velocity limiting.
# Joint 4 operates in the -270° neighbourhood (3rd winding); generic ±175° would
# clip all predictions. Tune to the actual workspace / training data range.
JOINT_LIMIT_LOW = np.radians([-180, -175, -170, -175, -310, -360]).astype(np.float32)
JOINT_LIMIT_HIGH = np.radians([180, 5, 170, 175, -220, 360]).astype(np.float32)


# ══════════════════════════════ Gripper ══════════════════════════════


def gripper_state_from_width(width_mm):
    """Measured jaw width → state value in the training-data convention (1 = open)."""
    return GRIPPER_STATE_OPEN if width_mm > GRIPPER_OPEN_THRESH_MM else GRIPPER_STATE_CLOSED


class WeissCRGGripper:
    """
    Weiss CRG 30-050 gripper driver via DC-IOLink USB adapter.

    Protocol: PDOUT=[02,00] open | PDOUT=[03,00] close | PDOUT=[07,00] reference
    Feedback: @PDIN=[B0,B1,B2,B3] where pos_mm = ((B0<<8)|B1)/100

    A background thread drains the serial port continuously and caches the latest
    @PDIN reading. Reading inside the control loop instead would block for up to the
    serial timeout and, worse, return the *oldest* buffered line — a stale state.
    """

    FLAG_OPEN = 1
    FLAG_CLOSED = 2
    READ_TIMEOUT_S = 0.02  # max time the reader holds the lock per readline

    def __init__(self, port=GRIPPER_PORT, baudrate=GRIPPER_BAUDRATE):
        import serial as _serial

        self._lock = threading.Lock()
        self._ser = _serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
        self._position_mm = 0.0
        self._flags = 0
        self._last_update = 0.0
        log.info(f"[Gripper] Opened {port} @ {baudrate}")
        self._initialise()

        self._ser.timeout = self.READ_TIMEOUT_S
        self._stop_event = threading.Event()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _send(self, cmd, wait=0.3):
        with self._lock:
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode("ascii"))
            except Exception as e:
                log.warning(f"[Gripper] Serial error '{cmd}': {e}")
        time.sleep(wait)

    def _parse_pdin(self, line):
        try:
            inner = line[7:].split("]")[0]
            parts = [int(x, 16) for x in inner.split(",")]
            self._position_mm = ((parts[0] << 8) | parts[1]) / 100.0
            self._flags = parts[3] if len(parts) >= 4 else 0
            self._last_update = time.monotonic()
            return True
        except (ValueError, IndexError):
            return False

    def _reader_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                try:
                    raw = self._ser.readline() if self._ser.is_open else b""
                except Exception as e:
                    log.warning(f"[Gripper] Read error: {e}")
                    raw = b""
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("@PDIN=["):
                self._parse_pdin(line)
            elif not raw:
                time.sleep(0.001)  # let _send grab the lock between empty reads

    def _wait_flag(self, flag_bit, timeout=6.0):
        """Wait for a flag in a reading that arrived *after* this call started."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._last_update > t0 and (self._flags & (1 << flag_bit)):
                return True
            time.sleep(0.02)
        return False

    def _set_positions(self, open_mm=GRIPPER_MAX_MM, close_mm=0.5):
        def enc(mm):
            v = int(mm * 100)
            return f"[{(v >> 8) & 0xFF:02x},{v & 0xFF:02x}]"

        self._send(f"SETPARAM(96, 2, {enc(open_mm)})", 0.3)
        self._send(f"SETPARAM(96, 1, {enc(close_mm)})", 0.3)
        self._send("SETPARAM(96, 3, [64])", 0.3)

    def _initialise(self):
        for cmd in ["ID?", "ID?", "FALLBACK(1)", "MODE?", "RESTART()", "OPERATE()"]:
            self._send(cmd, 0.5)
        self._send("PDOUT=[00,00]", 0.5)
        # Configure open/close positions once — no need to repeat before each move
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        log.info("[Gripper] Initialised.")

    def home(self):
        log.info("[Gripper] Homing...")
        self._set_positions(GRIPPER_MAX_MM, 0.5)
        self._send("PDOUT=[07,00]", 0.2)
        if not self._wait_flag(self.FLAG_OPEN, timeout=10.0):
            log.warning("[Gripper] No 'open' flag after homing — check the gripper")
        log.info("[Gripper] Home complete — open.")

    def move_to_pos(self, width_mm):
        # No blocking wait — the gripper moves while the control loop continues
        if width_mm > GRIPPER_OPEN_THRESH_MM:
            self._send("PDOUT=[02,00]", 0.0)
        else:
            self._send("PDOUT=[03,00]", 0.0)

    def get_width(self):
        """Latest cached width in mm and its age in seconds (never blocks)."""
        return self._position_mm, time.monotonic() - self._last_update

    def close(self):
        self._stop_event.set()
        self._reader.join(timeout=1.0)
        # Zero all outputs to cut drive, then close serial immediately.
        # Do NOT send FALLBACK(1) here — it activates the gripper's internal
        # fallback program which causes it to cycle open/close autonomously.
        self._send("PDOUT=[00,00]", 0.1)
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
        log.info("[Gripper] Closed.")


# ══════════════════════════════ Camera ══════════════════════════════


def _center_crop_resize(bgr, out_hw=(IMAGE_SIZE, IMAGE_SIZE)):
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


class RealSenseCamera:
    """
    One RealSense color stream captured by a background thread.

    wait_for_frames() blocks until the next frame (up to 33 ms at 30 fps); doing that
    for two cameras inside get_obs() would eat most of a 50 ms control period. The
    thread keeps only the latest frame, so read() returns immediately.
    """

    def __init__(self, serial_num, width=640, height=480, fps=30):
        import pyrealsense2 as rs

        self.serial = serial_num
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial_num)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(cfg)
        self._frame = None
        self._t_frame = 0.0
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info(f"[Camera] Started — serial {serial_num}")

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError as e:
                log.warning(f"[Camera {self.serial}] {e}")
                continue
            color = frames.get_color_frame()
            if color:
                bgr = np.asanyarray(color.get_data()).copy()
                with self._frame_lock:
                    self._frame, self._t_frame = bgr, time.monotonic()

    def read(self, timeout=2.0):
        """Latest BGR frame and its age in seconds. Waits only for the very first frame."""
        t0 = time.monotonic()
        while True:
            with self._frame_lock:
                frame, t_frame = self._frame, self._t_frame
            if frame is not None:
                return frame, time.monotonic() - t_frame
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f"Camera {self.serial} produced no frame")
            time.sleep(0.005)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._pipeline.stop()


# ══════════════════════════════ Robot ══════════════════════════════


class URRobot:
    """
    UR robot arm + optional Weiss gripper + RealSense camera(s).

    get_obs() returns {"state", "<cam>_rgb" per camera, "sensor_age_s"} — the same
    layout as the DP reimplementation, which evaluation/trial_recorder.py consumes.

    Args:
        robot_ip: robot controller IP address
        frequency: control loop frequency [Hz]
        use_gripper: whether to connect the Weiss gripper
        cam_serial_exterior: RealSense serial for exterior camera (None to skip)
        cam_serial_wrist: RealSense serial for wrist camera (None to skip)
        image_hw: (height, width) of camera frames fed to the policy [px]
        max_joint_vel: per-joint velocity clamp [rad/s]
    """

    def __init__(
        self,
        robot_ip,
        frequency=20,
        use_gripper=True,
        cam_serial_exterior=CAM_SERIAL_EXTERIOR,
        cam_serial_wrist=CAM_SERIAL_WRIST,
        image_hw=(IMAGE_SIZE, IMAGE_SIZE),
        max_joint_vel=MAX_JOINT_VEL,
    ):
        import rtde_control
        import rtde_receive

        self._dt = 1.0 / frequency
        self._image_hw = tuple(image_hw)
        self._max_joint_vel = max_joint_vel

        log.info(f"[URRobot] Connecting RTDE to {robot_ip}...")
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        log.info("[URRobot] RTDE connected.")

        self._gripper = None
        self._last_gripper_open = None
        if use_gripper:
            log.info(f"[URRobot] Connecting gripper on {GRIPPER_PORT}...")
            self._gripper = WeissCRGGripper()
            self._last_gripper_open = True

        self._cameras = {}
        if cam_serial_exterior:
            self._cameras["exterior"] = RealSenseCamera(cam_serial_exterior)
        if cam_serial_wrist:
            self._cameras["wrist"] = RealSenseCamera(cam_serial_wrist)
        if not self._cameras:
            raise RuntimeError("At least one camera serial must be provided")

        self._last_cmd_rad = HOME_RAD.copy()

    @property
    def camera_names(self):
        return list(self._cameras)

    def home(self):
        import contextlib

        with contextlib.suppress(Exception):
            self._rtde_c.servoStop()
        time.sleep(0.2)
        log.info("[URRobot] Moving to home...")
        self._rtde_c.moveJ(HOME_RAD.tolist(), speed=0.25, acceleration=0.25)
        self._last_cmd_rad = HOME_RAD.copy()
        if self._gripper:
            self._gripper.home()
            self._last_gripper_open = True
        log.info("[URRobot] Ready.")

    def get_obs(self):
        """
        Assemble one observation from cached sensor readings (non-blocking).

        Returns the obs dict plus "sensor_age_s": the age of the oldest reading used.
        Raises if a sensor stalled, so the robot stops instead of acting on stale data.
        """
        qpos = np.array(self._rtde_r.getActualQ(), dtype=np.float32)

        # No gripper attached → report "open", the state every demonstration starts in
        gripper_state = np.array([GRIPPER_STATE_OPEN], dtype=np.float32)
        ages = []
        if self._gripper:
            width, gripper_age = self._gripper.get_width()
            ages.append(gripper_age)
            gripper_state[0] = gripper_state_from_width(width)

        # Training data stores joint angles in DEGREES — the normalizer was fitted on
        # degrees. getActualQ() returns radians, so convert before building the state.
        obs = {"state": np.concatenate([np.degrees(qpos), gripper_state])}
        for name, cam in self._cameras.items():
            bgr, age = cam.read()
            ages.append(age)
            obs[f"{name}_rgb"] = _center_crop_resize(bgr, self._image_hw)

        sensor_age = max(ages)
        if sensor_age > MAX_SENSOR_AGE_S:
            raise RuntimeError(
                f"Stale sensor data ({sensor_age * 1000:.0f} ms old) — "
                "camera or gripper reader stalled"
            )
        obs["sensor_age_s"] = sensor_age
        return obs

    def apply_action(self, action, gripper_open):
        """
        action:       (7,) array — [6 joint angles DEGREES (model output), 1 gripper raw]
        gripper_open: effective gripper command after hysteresis (see GripperHysteresis)
        Converts degrees → radians before servoJ. Joint-clamped + velocity-limited.
        """
        target_rad = np.radians(action[:6])
        # Clamp to joint limits before velocity-limiting to prevent self-collision
        target_rad = np.clip(target_rad, JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)
        max_step = self._max_joint_vel * self._dt
        delta = target_rad - self._last_cmd_rad
        cmd_rad = self._last_cmd_rad + np.clip(delta, -max_step, max_step)
        self._last_cmd_rad = cmd_rad.astype(np.float32)

        # servoJ time must equal the control period so RTDE's interpolation matches
        # the actual step rate
        self._rtde_c.servoJ(cmd_rad.tolist(), 0, 0, self._dt, SERVO_J_LOOKAHEAD, SERVO_J_GAIN)

        if self._gripper and gripper_open != self._last_gripper_open:
            self._gripper.move_to_pos(GRIPPER_MAX_MM if gripper_open else 0.0)
            self._last_gripper_open = gripper_open
            log.info(
                f"[URRobot] Gripper {'opening' if gripper_open else 'closing'} "
                f"(raw={action[6]:.2f})"
            )

    def hold(self):
        """Stop servoing and hold the current pose (between evaluation episodes)."""
        try:
            self._rtde_c.servoStop()
        except Exception as e:
            log.warning(f"[URRobot] servoStop error: {e}")

    def stop(self):
        try:
            self._rtde_c.servoStop()
            self._rtde_c.stopScript()
        except Exception:
            pass
        if self._gripper:
            self._gripper.close()
        for cam in self._cameras.values():
            try:
                cam.stop()
            except Exception as e:
                log.warning(f"[Camera {cam.serial}] stop failed: {e}")
        log.info("[URRobot] Stopped.")


# ══════════════════════════════ Visualisation ══════════════════════════════


def _draw_action_bars(action, height, width=200):
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
        cv2.putText(
            panel,
            f"a{i}:{val:+.1f}",
            (2, y0 + row_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return panel


def make_vis_frame(obs, action, step, inference_ms, mode_str, gripper_open=True):
    """Side-by-side camera view with status overlay + action bars sidebar."""
    disp = 320
    panels = []
    for key in sorted(k for k in obs if k.endswith("_rgb")):
        bgr = cv2.cvtColor(obs[key], cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        panels.append(cv2.resize(bgr, (int(round(disp * w / h)), disp)))
    frame = np.concatenate(panels, axis=1)

    gripper_cmd = "open" if gripper_open else "close"
    gripper_actual = "open" if obs["state"][-1] > 0.5 else "closed"
    lines = [
        f"Step: {step}",
        f"Infer: {inference_ms:.0f} ms  Mode: {mode_str}",
        f"G raw={action[-1]:.2f} cmd={gripper_cmd} actual={gripper_actual}",
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
    return np.concatenate([frame, _draw_action_bars(action, height=disp)], axis=1)


# ══════════════════════════════ Inference Loop ══════════════════════════════


def resolve_camera_order(camera_names, available):
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


def build_image_tensor(obs, cam_order):
    """Robot obs → ACT image input (num_cam, 3, H, W) float32 in [0, 1], model order."""
    frames = [torch.from_numpy(obs[f"{name}_rgb"]).float().permute(2, 0, 1) for name in cam_order]
    return torch.stack(frames, dim=0) / 255.0


def check_qpos_in_distribution(qpos, normalizer, z_thresh=5.0):
    """
    Warn if the robot's current state is far outside the training distribution.

    Catches unit / convention mismatches between data collection and
    deployment before the policy moves the arm — e.g. degrees vs radians,
    gripper width in mm vs a 0/1 flag, or a home pose never seen in the data.
    Returns the indices of the out-of-distribution dimensions.
    """
    if normalizer.qpos_mean is None:
        return []
    mean = normalizer.qpos_mean.cpu().numpy()
    std = normalizer.qpos_std.cpu().numpy()
    q = qpos[: len(mean)]
    z = (q - mean) / std
    bad = np.where(np.abs(z) > z_thresh)[0]
    if len(bad) == 0:
        log.info(f"State check OK — max |z| = {np.abs(z).max():.2f}")
    for i in bad:
        log.warning(
            f"state[{i}] = {q[i]:.3f} is {z[i]:+.1f} std from the training mean "
            f"({mean[i]:.3f} ± {std[i]:.3f}) — check units / gripper convention / home pose"
        )
    return bad.tolist()


def run_episode(
    pipeline,
    robot,
    cam_order,
    state_dim,
    frequency=20,
    max_steps=800,
    temporal_agg=False,
    dry_run=False,
    no_vis=False,
    recorder=None,
):
    """
    Run one ACT episode at a fixed control rate.

    ACT only needs the *current* observation (no history window). The
    InferencePipeline handles chunking internally:
      - temporal_agg=False: re-queries every chunk_size steps
      - temporal_agg=True:  queries every step, smooths with exp weighting

    Every step is scheduled on an absolute clock (t_start + step * dt), so a slow
    step delays only itself instead of shifting all later ones. The episode ends
    after max_steps or when the operator presses q / ESC. Ctrl+C propagates to the
    caller (abort all). The robot is NOT stopped here so main() can run several
    episodes.

    recorder: optional evaluation.trial_recorder.TrialRecorder — every executed step
    is appended to it. Returns (steps executed, TimingStats).
    """
    dt = 1.0 / frequency
    chunk_size = pipeline.chunk_size
    pipeline.reset()
    # Single source of truth for the gripper command (robot + visualisation + recording)
    gripper = GripperHysteresis(
        close_thresh=GRIPPER_CLOSE_THRESH,
        open_thresh=GRIPPER_OPEN_THRESH,
        min_dwell_steps=GRIPPER_MIN_DWELL_STEPS,
    )
    # obs: get_obs() duration | infer: pipeline.get_action() | late: how far past its
    # slot a step started | age: age of the oldest sensor reading
    timing = TimingStats()
    log.info(
        f"Episode start — freq={frequency} Hz  max_steps={max_steps}  "
        f"temporal_agg={temporal_agg}  dry_run={dry_run}"
    )

    step = 0
    t_start = time.monotonic()
    while step < max_steps:
        t_slot = t_start + step * dt
        precise_wait(t_slot)
        timing.add("late", (time.monotonic() - t_slot) * 1000)

        # ── Observe ──
        t_obs = time.monotonic()
        obs = robot.get_obs()
        obs_ms = (time.monotonic() - t_obs) * 1000
        timing.add("obs", obs_ms)
        timing.add("age", obs["sensor_age_s"] * 1000)

        # ── Inference ──
        qpos = torch.from_numpy(obs["state"][:state_dim])
        t_inf = time.monotonic()
        action = pipeline.get_action(step, qpos, build_image_tensor(obs, cam_order))
        inference_ms = (time.monotonic() - t_inf) * 1000
        timing.add("infer", inference_ms)
        action = action.cpu().numpy()
        if inference_ms + obs_ms > dt * 1000:
            log.warning(
                f"step {step}: obs {obs_ms:.0f} ms + inference {inference_ms:.0f} ms "
                f"exceed the {dt * 1000:.0f} ms period"
            )

        # ── Execute ──
        gripper_open = gripper.update(float(action[6])) if len(action) > 6 else True
        if not dry_run:
            robot.apply_action(action, gripper_open)
        if recorder is not None:
            recorder.add_step(obs, action, gripper_open, inference_ms, obs_ms)

        # ── Visualise ──
        if not no_vis:
            if temporal_agg:
                mode_str = f"ensemble {min(step + 1, chunk_size)}"
            else:
                mode_str = f"chunk {step % chunk_size + 1}/{chunk_size}"
            cv2.imshow(
                "ACT Inference",
                make_vis_frame(obs, action, step, inference_ms, mode_str, gripper_open),
            )
            if cv2.pollKey() in (ord("q"), 27):  # q or ESC — end this episode
                log.info("Episode ended by operator.")
                step += 1
                break

        step += 1

    elapsed = time.monotonic() - t_start
    rate = step / elapsed if elapsed > 0 else 0.0
    log.info(f"Episode ended at step {step}. Effective rate {rate:.1f} Hz (target {frequency} Hz)")
    log.info(f"Timing: {timing.format()}")
    return step, timing


# ══════════════════════════════ Main ══════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="ACT real robot inference / evaluation")

    # Policy
    parser.add_argument(
        "--checkpoint", required=True, help="policy_*.pt, or a training output directory"
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
        help="Control rate in Hz (default: the training dataset's fps). Must match the "
        "dataset fps — per-step actions were learned at that rate.",
    )
    parser.add_argument(
        "--allow_fps_mismatch",
        action="store_true",
        help="Run even if --frequency differs from the dataset fps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=800,
        help="Control steps per episode (default: 800 = 40s at 20Hz; longest demo is ~33s)",
    )
    parser.add_argument(
        "--max_joint_vel",
        type=float,
        default=MAX_JOINT_VEL,
        help=f"Per-joint velocity limit in rad/s (default: {MAX_JOINT_VEL})",
    )
    parser.add_argument("--no_gripper", action="store_true", help="Disable Weiss gripper")

    # Cameras
    parser.add_argument(
        "--cam_exterior",
        default=CAM_SERIAL_EXTERIOR,
        help="RealSense serial number for exterior camera ('' to disable)",
    )
    parser.add_argument(
        "--cam_wrist",
        default=CAM_SERIAL_WRIST,
        help="RealSense serial number for wrist camera ('' to disable)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Square image size override (default: training resolution from the checkpoint)",
    )

    # Evaluation
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Rollouts in one session; the scene is reset between them",
    )
    parser.add_argument(
        "--record_dir",
        default=None,
        help="Save each episode (videos, actions, timing) here and ask for the outcome",
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

    checkpoint = resolve_checkpoint(args.checkpoint)
    log.info(f"Loading policy from {checkpoint} ...")
    pipeline, config = load_policy(
        checkpoint,
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
    dataset_fps = config.get("fps")
    frequency = args.frequency
    if frequency is None:
        if dataset_fps is None:
            parser.error("checkpoint does not record the dataset fps; pass --frequency")
        frequency = dataset_fps
    elif dataset_fps is not None and frequency != dataset_fps:
        msg = (
            f"--frequency {frequency} differs from the dataset fps {dataset_fps}: "
            "the robot would move at the wrong speed."
        )
        if not args.allow_fps_mismatch:
            parser.error(msg + " Pass --allow_fps_mismatch to run anyway.")
        log.warning(msg)

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
        log.warning(f"Checkpoint has no image_hw, assuming {IMAGE_SIZE}x{IMAGE_SIZE}")
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
        cam_serial_exterior=args.cam_exterior or None,
        cam_serial_wrist=args.cam_wrist or None,
        image_hw=image_hw,
        max_joint_vel=args.max_joint_vel,
    )

    try:
        # ── Warmup: cameras + CUDA kernels before the timed loop ──
        log.info("Warming up cameras and inference...")
        obs = robot.get_obs()
        pipeline.get_action(
            0,
            torch.from_numpy(obs["state"][: config["state_dim"]]),
            build_image_tensor(obs, cam_order),
        )

        for ep in range(args.num_episodes):
            if args.num_episodes > 1:
                input(f"\n[Episode {ep + 1}/{args.num_episodes}] Reset the scene, then press Enter")

            # A dry run must never move the robot — not even to home
            if args.dry_run:
                log.info("[dry run] Skipping homing — robot and gripper will not move.")
            else:
                robot.home()
            check_qpos_in_distribution(robot.get_obs()["state"], pipeline.normalizer)

            recorder = None
            if args.record_dir:
                from evaluation.trial_recorder import TrialRecorder

                recorder = TrialRecorder(
                    args.record_dir,
                    fps=frequency,
                    meta={
                        "checkpoint": checkpoint,
                        "frequency": frequency,
                        "temporal_agg": args.temporal_agg,
                        "k": args.k,
                        "max_steps": args.max_steps,
                        "dry_run": args.dry_run,
                        "episode": ep,
                    },
                )
                log.info(f"Recording episode to {recorder.trial_dir}")

            try:
                run_episode(
                    pipeline=pipeline,
                    robot=robot,
                    cam_order=cam_order,
                    state_dim=config["state_dim"],
                    frequency=frequency,
                    max_steps=args.max_steps,
                    temporal_agg=args.temporal_agg,
                    dry_run=args.dry_run,
                    no_vis=args.no_vis,
                    recorder=recorder,
                )
            finally:
                # Freeze the arm where it is before talking to the operator
                if not args.dry_run:
                    robot.hold()
                if recorder is not None:
                    from evaluation.trial_recorder import ask_trial_outcome

                    success, notes = ask_trial_outcome()
                    log.info(f"Saved episode to {recorder.close(success, notes)}")

    except KeyboardInterrupt:
        log.info("Aborted by user.")
    except Exception as e:
        log.error(f"[Fatal] {e}")
        log.error("Stopping robot and exiting.")
        # sys.exit raises SystemExit, so finally still runs robot.stop()
        sys.exit(1)
    finally:
        if not args.no_vis:
            cv2.destroyAllWindows()
        robot.stop()


if __name__ == "__main__":
    main()
