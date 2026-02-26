"""
Unit tests for the real-robot control helpers (act/control.py).

Validates:
  - GripperHysteresis follows the dataset convention (1 = open), ignores values between
    the thresholds and respects the dwell time
"""

from act.control import GripperHysteresis, TimingStats, precise_wait


def test_gripper_hysteresis_band_does_not_toggle():
    g = GripperHysteresis(close_thresh=0.35, open_thresh=0.65, min_dwell_steps=1)
    for v in [0.5, 0.4, 0.6, 0.36]:
        assert g.update(v) is True  # stays open inside the band


def test_gripper_hysteresis_uses_dataset_convention():
    # Training data: 1 = open, 0 = closed. A policy output of ~1 must keep it open.
    g = GripperHysteresis(min_dwell_steps=1)
    assert all(g.update(1.0) for _ in range(10))
    assert g.update(0.0) is False


def test_gripper_hysteresis_switch_and_dwell():
    g = GripperHysteresis(close_thresh=0.35, open_thresh=0.65, min_dwell_steps=3)
    states = [g.update(v) for v in [0.1, 0.1, 0.1]]
    assert states == [True, True, False]  # closes once dwell is satisfied
    # Immediately asking to open is ignored until the dwell has passed again
    states = [g.update(1.0) for _ in range(3)]
    assert states == [False, False, True]


def test_gripper_hysteresis_reset():
    g = GripperHysteresis(min_dwell_steps=1)
    g.update(0.0)
    assert g.is_open is False
    g.reset()
    assert g.is_open is True


def test_timing_stats_summary():
    stats = TimingStats()
    for v in range(1, 101):  # 1..100 ms
        stats.add("obs", v)
    s = stats.summary()["obs"]
    assert s["n"] == 100 and s["max"] == 100
    assert s["mean"] == 50.5
    assert 94 <= s["p95"] <= 96


def test_precise_wait_uses_injected_clock():
    t = [0.0]

    def clock():
        t[0] += 0.0004  # each poll advances the fake clock by 0.4 ms
        return t[0]

    precise_wait(0.002, slack=0.01, time_func=clock)  # never sleeps, busy-waits
    assert t[0] >= 0.002
