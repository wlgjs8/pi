#!/usr/bin/env python3
"""Idle-timestep / gripper-close distribution of the PIKA UMI training set, per arm.

Motivation: during collection the operator deliberately held the arm still while the gripper
closed, and closed it slowly. OpenPI's DROID recipe drops timesteps whose FUTURE action chunk is
mostly idle, because a policy trained with many such windows learns to sit still. This measures
how much of our data looks like that, and -- since the right arm picks visibly worse than the
left on hardware -- whether the two arms differ.

Layout (velocity_grip / gripabs, 14 dims):
    actions[0:3]  left  pos delta (m, ee_local)      actions[6]  left  gripper opening (absolute)
    actions[7:10] right pos delta (m, ee_local)      actions[13] right gripper opening (absolute)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

L_POS, R_POS = slice(0, 3), slice(7, 10)
L_GRIP, R_GRIP = 6, 13


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="lerobot dataset root (contains data/, meta/)")
    ap.add_argument("--horizon", type=int, default=24, help="action chunk horizon used in training")
    ap.add_argument("--idle-mm", type=float, default=0.3,
                    help="per-step linear displacement below this counts as idle (mm)")
    ap.add_argument("--idle-frac", type=float, default=0.75,
                    help="DROID-style: a timestep is dropped if >= this fraction of its future chunk is idle")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "data", "**", "*.parquet"), recursive=True))
    if args.limit:
        files = files[: args.limit]
    print(f"episodes: {len(files)}")

    idle_m = args.idle_mm / 1000.0
    per_arm = {"left": [], "right": []}          # per-step |dpos| for every frame
    grip_series = {"left": [], "right": []}
    chunk_idle_frac = {"left": [], "right": []}  # per timestep: fraction of its future H rows that are idle
    close_motion = {"left": [], "right": []}     # |dpos| inside the closing window
    open_motion = {"left": [], "right": []}      # |dpos| outside it
    close_dur = {"left": [], "right": []}        # frames spent closing
    n_frames = 0

    for f in files:
        a = np.stack(pd.read_parquet(f, columns=["actions"])["actions"].to_numpy())
        n_frames += len(a)
        for arm, sl, gi in (("left", L_POS, L_GRIP), ("right", R_POS, R_GRIP)):
            d = np.linalg.norm(a[:, sl], axis=1)
            g = a[:, gi]
            per_arm[arm].append(d)
            grip_series[arm].append(g)
            idle = d < idle_m
            # future-chunk idle fraction (DROID filter criterion)
            T = len(d)
            if T > args.horizon:
                cs = np.concatenate([[0.0], np.cumsum(idle.astype(np.float64))])  # len T+1
                fr = (cs[args.horizon:] - cs[: len(cs) - args.horizon]) / args.horizon
                chunk_idle_frac[arm].append(fr)
            # closing window: gripper monotonically dropping from its episode max toward its min
            gmax, gmin = float(g.max()), float(g.min())
            if gmax - gmin > 0.1 * max(abs(gmax), 1e-6):
                hi, lo = gmin + 0.8 * (gmax - gmin), gmin + 0.2 * (gmax - gmin)
                closing = np.zeros(T, bool)
                state = None
                start = None
                for t in range(T):
                    if state is None and g[t] >= hi:
                        state = "open"
                    elif state == "open" and g[t] < hi:
                        state, start = "closing", t
                    elif state == "closing":
                        if g[t] <= lo:
                            closing[start : t + 1] = True
                            close_dur[arm].append(t - start + 1)
                            state = "closed"
                        elif g[t] >= hi:
                            state = "open"
                close_motion[arm].append(d[closing])
                open_motion[arm].append(d[~closing])

    print(f"frames: {n_frames}\n")
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")

    print(f"{'':<8} {'idle%':>7} {'|d| p50 mm':>11} {'p90 mm':>8} "
          f"{'chunk>=%.0f%% idle' % (100 * args.idle_frac):>18} {'close frames p50':>17}")
    print("-" * 78)
    for arm in ("left", "right"):
        d = np.concatenate(per_arm[arm]) * 1000.0
        fr = np.concatenate(chunk_idle_frac[arm]) if chunk_idle_frac[arm] else np.array([])
        drop = 100.0 * float((fr >= args.idle_frac).mean()) if len(fr) else float("nan")
        cd = np.array(close_dur[arm], dtype=float)
        print(f"{arm:<8} {100.0 * float((d < args.idle_mm).mean()):>6.1f}% {pct(d, 50):>11.3f} "
              f"{pct(d, 90):>8.3f} {drop:>17.1f}% {pct(cd, 50):>17.0f}")

    print(f"\n닫는 구간 안/밖 per-step 이동량 (mm)")
    print(f"{'':<8} {'close p50':>10} {'close p90':>10} {'other p50':>10} {'other p90':>10} {'비율':>7}")
    print("-" * 62)
    for arm in ("left", "right"):
        cm = np.concatenate(close_motion[arm]) * 1000.0 if close_motion[arm] else np.array([0.0])
        om = np.concatenate(open_motion[arm]) * 1000.0 if open_motion[arm] else np.array([0.0])
        r = pct(cm, 50) / max(pct(om, 50), 1e-9)
        print(f"{arm:<8} {pct(cm, 50):>10.3f} {pct(cm, 90):>10.3f} {pct(om, 50):>10.3f} "
              f"{pct(om, 90):>10.3f} {r:>6.2f}x")

    print(f"\n닫는 구간이 전체에서 차지하는 비중")
    for arm in ("left", "right"):
        cm = np.concatenate(close_motion[arm]) if close_motion[arm] else np.array([])
        om = np.concatenate(open_motion[arm]) if open_motion[arm] else np.array([])
        tot = len(cm) + len(om)
        if tot:
            print(f"  {arm:<6} {len(cm)}/{tot} = {100.0 * len(cm) / tot:.1f}%  "
                  f"(그 안에서 idle 비율 {100.0 * float((cm * 1000 < args.idle_mm).mean()):.1f}%)")

    print(f"\n에피소드당 닫힘 이벤트 수: left={len(close_dur['left']) / max(len(files),1):.2f} "
          f"right={len(close_dur['right']) / max(len(files),1):.2f}")


if __name__ == "__main__":
    main()
