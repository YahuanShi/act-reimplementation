"""
Summarise real-robot trials recorded with `inference.py --record_dir` and make demo GIFs.

Success rate is reported with a 95% Wilson interval: with 10–20 trials the
uncertainty is large (e.g. 8/10 → 49%–94%), and the interval makes that explicit.

Usage:
    python scripts/summarize_trials.py --record_dir trials/
    python scripts/summarize_trials.py --record_dir trials/ --markdown trials/results.md
    python scripts/summarize_trials.py --gif trials/trial_20260212_101500 --out demo.gif
"""

import argparse
import glob
import json
import math
import os


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_trials(record_dir):
    trials = []
    for path in sorted(glob.glob(os.path.join(record_dir, "trial_*", "meta.json"))):
        with open(path) as f:
            meta = json.load(f)
        meta["trial"] = os.path.basename(os.path.dirname(path))
        trials.append(meta)
    return trials


def summarize(trials):
    """Group trials by checkpoint; dry runs and trials without an outcome are skipped."""
    groups = {}
    for t in trials:
        if t.get("dry_run") or t.get("success") is None:
            continue
        groups.setdefault(t.get("checkpoint", "?"), []).append(t)
    rows = []
    for ckpt, ts in groups.items():
        n = len(ts)
        k = sum(bool(t["success"]) for t in ts)
        lo, hi = wilson_interval(k, n)
        mean_s = sum(t.get("duration_s", 0.0) for t in ts) / n
        rows.append(
            {
                "checkpoint": ckpt,
                "n": n,
                "success": k,
                "rate": k / n,
                "ci": (lo, hi),
                "mean_duration_s": mean_s,
            }
        )
    return rows


def to_markdown(rows, trials):
    lines = [
        "| Checkpoint | Success | Rate | 95% CI | Mean duration |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['checkpoint']}` | {r['success']}/{r['n']} | {r['rate']:.0%} | "
            f"{r['ci'][0]:.0%}–{r['ci'][1]:.0%} | {r['mean_duration_s']:.1f}s |"
        )
    lines += [
        "",
        "| Trial | Checkpoint | Steps | Success | Notes |",
        "|---|---|---|---|---|",
    ]
    for t in trials:
        if t.get("dry_run"):
            continue
        ok = {True: "yes", False: "no", None: "—"}[t.get("success")]
        lines.append(
            f"| {t['trial']} | `{t.get('checkpoint', '?')}` | {t.get('n_steps', 0)} | "
            f"{ok} | {t.get('notes', '')} |"
        )
    return "\n".join(lines) + "\n"


def make_gif(trial_dir, out_path, fps=10, width=640, max_seconds=None):
    """Side-by-side GIF of all camera videos of a recorded trial (exterior | wrist)."""
    import cv2
    import imageio
    import numpy as np

    videos = sorted(glob.glob(os.path.join(trial_dir, "*.mp4")))
    if not videos:
        raise SystemExit(f"No videos in {trial_dir}")
    caps = [cv2.VideoCapture(v) for v in videos]
    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 20
    stride = max(1, round(src_fps / fps))
    frames, idx = [], 0
    while True:
        reads = [c.read() for c in caps]
        if not all(ok for ok, _ in reads):
            break
        if idx % stride == 0:
            tile = width // len(caps)
            tiles = [cv2.resize(f, (tile, tile)) for _, f in reads]
            frames.append(cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_BGR2RGB))
        idx += 1
        if max_seconds and idx >= max_seconds * src_fps:
            break
    for c in caps:
        c.release()
    if not frames:
        raise SystemExit(f"No frames read from {trial_dir}")
    imageio.mimsave(out_path, frames, duration=1 / fps, loop=0)
    print(f"saved {out_path} ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--record_dir", help="Directory passed to inference.py --record_dir")
    parser.add_argument("--markdown", help="Write the summary tables to this .md file")
    parser.add_argument("--gif", metavar="TRIAL_DIR", help="Make a demo GIF from one trial")
    parser.add_argument("--out", default="demo.gif", help="GIF output path")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate")
    parser.add_argument("--width", type=int, default=640, help="GIF width in pixels")
    parser.add_argument("--max_seconds", type=float, default=None)
    args = parser.parse_args()

    if args.gif:
        make_gif(args.gif, args.out, args.fps, args.width, args.max_seconds)
        return
    if not args.record_dir:
        parser.error("pass --record_dir (summary) or --gif (demo GIF)")

    trials = load_trials(args.record_dir)
    rows = summarize(trials)
    if not rows:
        print(f"No completed (non-dry-run, outcome recorded) trials in {args.record_dir}")
        return
    md = to_markdown(rows, trials)
    print(md)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(md)
        print(f"saved {args.markdown}")


if __name__ == "__main__":
    main()
