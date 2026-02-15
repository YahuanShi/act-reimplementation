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

Usage:
    # Dry run (no robot commands sent — use for sanity check)
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 --dry_run

    # Real inference, temporal ensembling enabled
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 \\
        --temporal_agg --frequency 50 --max_steps 400

    # Two cameras
    python inference.py --ckpt_dir checkpoints --robot_ip 192.168.1.1 \\
        --cam_exterior 105422061000 --cam_wrist 352122273671
"""

import argparse
import logging
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

HOME_DEG = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]
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


def _center_crop_resize(bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Square centre-crop + resize, BGR→RGB."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = bgr[y0 : y0 + s, x0 : x0 + s]
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
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
        image_size: square output size for camera frames [px]
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: int = 50,
        use_gripper: bool = True,
        cam_serial_exterior: str = CAM_SERIAL_EXTERIOR,
        cam_serial_wrist: str = None,
        image_size: int = IMAGE_SIZE,
    ):
        import rtde_control
        import rtde_receive

        self._dt = 1.0 / frequency
        self._image_size = image_size

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
            images[name] = _center_crop_resize(bgr, self._image_size)

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


def make_vis_frame(
    obs: dict, action: np.ndarray, step: int, inference_ms: float, temporal_agg: bool
) -> np.ndarray:
    """
    Side-by-side camera view with status overlay.
    Shows all cameras tiled horizontally; displays step, latency, gripper state.
    """
    disp = 320
    panels = []
    for name in sorted(obs["images"]):
        rgb = obs["images"][name]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        panels.append(cv2.resize(bgr, (disp, disp)))
    frame = np.concatenate(panels, axis=1)

    gripper_str = ""
    if len(action) > 6:
        gripper_str = f"  Grip: {'open' if action[6] < 0.5 else 'close'}"

    lines = [
        f"Step: {step}",
        f"Infer: {inference_ms:.0f} ms",
        f"Mode: {'ensemble' if temporal_agg else 'chunk'}" + gripper_str,
        "q / ESC — stop",
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
    return frame


# ══════════════════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════════════════


def run_inference(
    pipeline,
    robot: URRobot,
    config: dict,
    cam_order: list[str],
    frequency: int = 50,
    max_steps: int = 400,
    temporal_agg: bool = False,
    dry_run: bool = False,
) -> None:
    """
    ACT inference loop.

    ACT only needs the *current* observation (no history window).
    The InferencePipeline handles action-chunk re-querying internally:
      - temporal_agg=False: re-queries every chunk_size steps
      - temporal_agg=True:  queries every step, smooths with exp weighting

    Args:
        pipeline: InferencePipeline from eval.load_policy()
        robot: URRobot instance
        config: model config dict (for state_dim, action_dim etc.)
        cam_order: camera names in model order
        frequency: control loop [Hz]
        max_steps: episode length cap
        temporal_agg: whether temporal ensembling is active (for display)
        dry_run: if True, skips robot.apply_action()
    """
    dt = 1.0 / frequency
    state_dim = config["state_dim"]

    # ── Camera warmup ──
    log.info("Warming up cameras...")
    for _ in range(5):
        robot.get_obs()
        time.sleep(0.05)

    # ── CUDA warmup — trigger JIT compilation before the timed loop ──
    log.info("Warming up inference...")
    dummy_qpos = torch.zeros(state_dim)
    dummy_image = torch.zeros(len(cam_order), 3, IMAGE_SIZE, IMAGE_SIZE)
    pipeline.reset()
    pipeline.get_action(0, dummy_qpos, dummy_image)
    pipeline.reset()

    log.info(
        f"Ready — freq={frequency} Hz  max_steps={max_steps}  "
        f"temporal_agg={temporal_agg}  dry_run={dry_run}"
    )

    step = 0
    time.monotonic()

    try:
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

            log.info(
                f"step {step:>4}:  infer={inference_ms:.0f} ms  "
                f"action=[{' '.join(f'{a:.3f}' for a in action_np[:4])} ...]"
            )

            # ── Execute ──
            if not dry_run:
                robot.apply_action(action_np)

            # ── Visualise ──
            cv2.imshow(
                "ACT Inference", make_vis_frame(obs, action_np, step, inference_ms, temporal_agg)
            )
            key = cv2.pollKey()
            if key in (ord("q"), 27):  # q or ESC
                raise KeyboardInterrupt

            step += 1

            # ── Hold loop timing ──
            precise_wait(t_step_start + dt)

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        cv2.destroyAllWindows()
        if not dry_run:
            robot.stop()
        log.info(f"Inference ended at step {step}.")


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="ACT real robot inference")

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
    parser.add_argument("--frequency", type=int, default=50, help="Control loop frequency [Hz]")
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
        "--image_size", type=int, default=IMAGE_SIZE, help="Square image size after crop/resize"
    )

    # Misc
    parser.add_argument(
        "--dry_run", action="store_true", help="Full pipeline without sending robot commands"
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

    # Determine camera order for image stacking
    cam_order = []
    if args.cam_exterior:
        cam_order.append("exterior")
    if args.cam_wrist:
        cam_order.append("wrist")
    if len(cam_order) != len(config["camera_names"]):
        log.warning(
            f"Model has {len(config['camera_names'])} cameras but "
            f"{len(cam_order)} provided. Check --cam_exterior / --cam_wrist."
        )

    # ── Connect robot ──
    log.info(f"Connecting to UR robot at {args.robot_ip} ...")
    robot = URRobot(
        robot_ip=args.robot_ip,
        frequency=args.frequency,
        use_gripper=not args.no_gripper,
        cam_serial_exterior=args.cam_exterior if args.cam_exterior else None,
        cam_serial_wrist=args.cam_wrist,
        image_size=args.image_size,
    )

    robot.home()

    run_inference(
        pipeline=pipeline,
        robot=robot,
        config=config,
        cam_order=cam_order,
        frequency=args.frequency,
        max_steps=args.max_steps,
        temporal_agg=args.temporal_agg,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
