"""
Hardware-independent control helpers for real-robot deployment (used by inference.py).

Kept free of robot / camera / cv2 imports so they can be unit-tested in CI. Same helpers
as diffusion_policy/control.py in the DP reimplementation; DP's plan_execution (latency
compensation for its receding-horizon cycle) has no counterpart here because ACT queries
the policy inside a fixed-rate loop and InferencePipeline handles chunk timing.
"""

import time


def precise_wait(t_end, slack=0.001, time_func=time.monotonic):
    """Sleep until t_end with sub-millisecond accuracy (sleep + busy-wait)."""
    t_now = time_func()
    if t_now < t_end - slack:
        time.sleep(t_end - t_now - slack)
    while time_func() < t_end:
        pass


class GripperHysteresis:
    """
    Turn the continuous gripper channel of the action into a stable binary command.

    Convention (from the training data): 1 = open, 0 = closed. Every demonstration
    starts and ends with the gripper open (1) and is closed (0) while carrying the object.

    - Hysteresis: close only below close_thresh, open only above open_thresh, so values
      hovering around 0.5 don't toggle the gripper.
    - Dwell: after a switch, hold the new state for at least min_dwell_steps. Consecutive
      action chunks can disagree on the exact switch moment; the dwell absorbs that.
    """

    def __init__(self, close_thresh=0.35, open_thresh=0.65, min_dwell_steps=20):
        assert close_thresh < open_thresh
        self.close_thresh = close_thresh
        self.open_thresh = open_thresh
        self.min_dwell_steps = min_dwell_steps
        self.reset()

    def reset(self, is_open=True):
        self.is_open = is_open
        self._dwell = 0

    def update(self, raw):
        """Feed one raw gripper value (1 = open, 0 = closed); returns True if open."""
        self._dwell += 1
        if self._dwell >= self.min_dwell_steps:
            if self.is_open and raw <= self.close_thresh:
                self.is_open = False
                self._dwell = 0
            elif not self.is_open and raw >= self.open_thresh:
                self.is_open = True
                self._dwell = 0
        return self.is_open


class TimingStats:
    """
    Collect per-step timings (ms) of the control loop and summarise them.

    Used to verify on the real robot that sensor reads, inference and command
    timing actually fit the control period — otherwise the effective control rate
    silently drops below the training fps.
    """

    def __init__(self):
        self._values = {}

    def add(self, name, value_ms):
        self._values.setdefault(name, []).append(float(value_ms))

    def summary(self):
        """{name: {"mean", "p95", "max", "n"}} over everything recorded so far."""
        out = {}
        for name, vals in self._values.items():
            s = sorted(vals)
            p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
            out[name] = {"mean": sum(s) / len(s), "p95": p95, "max": s[-1], "n": len(s)}
        return out

    def format(self):
        return "  ".join(
            f"{k}: mean={v['mean']:.1f} p95={v['p95']:.1f} max={v['max']:.1f}ms"
            for k, v in self.summary().items()
        )
