"""
Tests for trial recording and success-rate reporting
(evaluation/trial_recorder.py, scripts/summarize_trials.py).

Validates:
  - several trials started in the same second get distinct directories
  - summarize() skips dry runs and trials without an outcome, groups by checkpoint
  - the Wilson interval is sane at the extremes
  - a demo GIF can be built from a recorded trial
"""

import importlib.util
import os

import numpy as np

from evaluation.trial_recorder import TrialRecorder

_spec = importlib.util.spec_from_file_location(
    "summarize_trials",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize_trials.py"),
)
summarize_trials = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize_trials)


def _record(root, success, dry_run=False, checkpoint="a.pt", n=5):
    rec = TrialRecorder(str(root), fps=10, meta={"checkpoint": checkpoint, "dry_run": dry_run})
    for _ in range(n):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        obs = {"exterior_rgb": img, "wrist_rgb": img, "state": np.zeros(7)}
        rec.add_step(obs, np.zeros(7), True, inference_ms=10.0, obs_ms=1.0)
    return rec.close(success=success)


def test_summarize_groups_and_filters(tmp_path):
    for ok in (True, True, False):
        _record(tmp_path, ok)
    _record(tmp_path, True, dry_run=True)  # ignored
    _record(tmp_path, None)  # outcome not recorded → ignored
    _record(tmp_path, False, checkpoint="b.pt")

    trials = summarize_trials.load_trials(str(tmp_path))
    assert len(trials) == 6  # all directories distinct despite the same timestamp
    rows = {r["checkpoint"]: r for r in summarize_trials.summarize(trials)}
    assert (rows["a.pt"]["success"], rows["a.pt"]["n"]) == (2, 3)
    assert (rows["b.pt"]["success"], rows["b.pt"]["n"]) == (0, 1)
    md = summarize_trials.to_markdown(list(rows.values()), trials)
    assert "2/3" in md


def test_wilson_interval_bounds():
    lo, hi = summarize_trials.wilson_interval(10, 10)
    assert 0.6 < lo < 1.0 and hi == 1.0
    lo, hi = summarize_trials.wilson_interval(0, 10)
    assert lo == 0.0 and hi < 0.4
    assert summarize_trials.wilson_interval(0, 0) == (0.0, 0.0)


def test_make_gif(tmp_path):
    trial_dir = _record(tmp_path / "trials", True, n=12)
    out = tmp_path / "demo.gif"
    summarize_trials.make_gif(trial_dir, str(out), fps=5, width=64)
    assert out.exists() and out.stat().st_size > 0
