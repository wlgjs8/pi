#!/usr/bin/env python3
"""Gripper-close trajectory distribution per arm, for the PIKA UMI training set.

Follow-up to analyze_idle_distribution.py, which ruled out the idle-timestep hypothesis
(DROID's filter would drop 0% here) but found the right gripper takes 1.4x longer to close
(44 vs 31 frames) and that the model's gripper axis is 3.7x worse on the right, with
`right_pick` the worst of the four phases.

The question this asks: is "when/how to close" LEARNABLE from the data? A policy can only learn
a close that is consistent across episodes. So measure, per arm and per episode,
  * when the close starts (as a fraction of the episode, and relative to the pick phase)
  * how long it takes, and how much that varies
  * the value it settles at (a per-bolt clamp depth would make the ABSOLUTE gripabs target
    ambiguous from vision alone)
  * how many frames sit in the ambiguous mid-band (neither open nor closed) -- those are the
    frames where a small timing error produces a large action error.

Layout (gripabs, 14 dims): actions[6] = left opening, actions[13] = right opening.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

L_POS, R_POS = slice(0, 3), slice(7, 10)
ARMS = {"left": (L_POS, 6), "right": (R_POS, 13)}


def close_events(g: np.ndarray, hi: float, lo: float):
    """Yield (start, end) frame pairs for each high->low crossing of the gripper opening."""
    state, start = None, None
    for t, v in enumerate(g):
        if state is None and v >= hi:
            state = "open"
        elif state == "open" and v < hi:
            state, start = "closing", t
        elif state == "closing":
            if v <= lo:
                yield start, t
                state = "closed"
            elif v >= hi:
                state = "open"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "data", "**", "*.parquet"), recursive=True))
    if args.limit:
        files = files[: args.limit]

    rec = {a: {"onset_frac": [], "dur": [], "open_v": [], "closed_v": [], "settle_v": [],
               "mid_frames": [], "ep_len": [], "onset_speed": []} for a in ARMS}
    mid_all = {a: 0 for a in ARMS}
    tot_all = {a: 0 for a in ARMS}
    profiles = {a: [] for a in ARMS}

    for f in files:
        a_ = np.stack(pd.read_parquet(f, columns=["actions"])["actions"].to_numpy())
        T = len(a_)
        for arm, (psl, gi) in ARMS.items():
            g = a_[:, gi].astype(np.float64)
            d = np.linalg.norm(a_[:, psl], axis=1) * 1000.0
            gmax, gmin = float(g.max()), float(g.min())
            rng = gmax - gmin
            tot_all[arm] += T
            if rng <= 1e-6:
                continue
            hi, lo = gmin + 0.8 * rng, gmin + 0.2 * rng
            mid_all[arm] += int(((g > lo) & (g < hi)).sum())
            evs = list(close_events(g, hi, lo))
            if not evs:
                continue
            s, e = evs[0]  # first close = the pick
            rec[arm]["onset_frac"].append(s / max(T - 1, 1))
            rec[arm]["dur"].append(e - s + 1)
            rec[arm]["open_v"].append(gmax)
            rec[arm]["closed_v"].append(gmin)
            # value it actually settles at: median over the 15 frames after the close completes
            tail = g[e : min(e + 15, T)]
            rec[arm]["settle_v"].append(float(np.median(tail)) if len(tail) else float(g[e]))
            rec[arm]["mid_frames"].append(int(((g > lo) & (g < hi)).sum()))
            rec[arm]["ep_len"].append(T)
            rec[arm]["onset_speed"].append(float(np.median(d[max(0, s - 5): s + 1])))
            # normalized close profile (10 samples between onset and completion)
            if e > s:
                xs = np.linspace(s, e, 10)
                prof = np.interp(xs, np.arange(T), g)
                if rng > 0:
                    profiles[arm].append((prof - gmin) / rng)

    q = lambda v, p: float(np.percentile(v, p)) if len(v) else float("nan")
    print(f"episodes: {len(files)}\n")
    print(f"{'':<7} {'n':>5} {'close dur (frames)':>26} {'CV':>6} {'onset (ep frac)':>22} {'CV':>6}")
    print("-" * 76)
    for arm in ARMS:
        du = np.array(rec[arm]["dur"], float)
        on = np.array(rec[arm]["onset_frac"], float)
        print(f"{arm:<7} {len(du):>5} "
              f"p10={q(du,10):>5.0f} p50={q(du,50):>5.0f} p90={q(du,90):>5.0f} "
              f"{du.std()/max(du.mean(),1e-9):>6.2f} "
              f"p10={q(on,10):>5.2f} p50={q(on,50):>5.2f} p90={q(on,90):>5.2f} "
              f"{on.std()/max(on.mean(),1e-9):>6.2f}")

    print(f"\n그리퍼 개도 값 분포 (열림 / 완전닫힘 / 안착값)")
    print(f"{'':<7} {'open p50':>9} {'open CV':>8} {'settle p10':>11} {'settle p50':>11} {'settle p90':>11} {'settle CV':>10}")
    print("-" * 74)
    for arm in ARMS:
        ov = np.array(rec[arm]["open_v"], float); sv = np.array(rec[arm]["settle_v"], float)
        print(f"{arm:<7} {q(ov,50):>9.3f} {ov.std()/max(abs(ov.mean()),1e-9):>8.2f} "
              f"{q(sv,10):>11.3f} {q(sv,50):>11.3f} {q(sv,90):>11.3f} {sv.std()/max(abs(sv.mean()),1e-9):>10.2f}")

    print(f"\n애매 구간(열림·닫힘 어느 쪽도 아닌 중간 밴드) 비중")
    for arm in ARMS:
        print(f"  {arm:<6} {mid_all[arm]}/{tot_all[arm]} = {100.0*mid_all[arm]/max(tot_all[arm],1):.1f}%  "
              f"(에피소드당 p50 {q(np.array(rec[arm]['mid_frames'],float),50):.0f} 프레임)")

    print(f"\n닫힘 시작 직전 팔 속도 (mm/step)")
    for arm in ARMS:
        sp = np.array(rec[arm]["onset_speed"], float)
        print(f"  {arm:<6} p10={q(sp,10):.2f} p50={q(sp,50):.2f} p90={q(sp,90):.2f}")

    print(f"\n정규화 닫힘 프로파일 (0=시작, 1=완료 / 값은 개도 정규화, 에피소드 간 표준편차)")
    print(f"{'':<7} " + " ".join(f"{x:>5.2f}" for x in np.linspace(0, 1, 10)))
    for arm in ARMS:
        P = np.array(profiles[arm])
        if len(P) == 0:
            continue
        print(f"{arm+' 평균':<7} " + " ".join(f"{v:>5.2f}" for v in P.mean(axis=0)))
        print(f"{arm+' std':<7} " + " ".join(f"{v:>5.2f}" for v in P.std(axis=0)))


if __name__ == "__main__":
    main()
