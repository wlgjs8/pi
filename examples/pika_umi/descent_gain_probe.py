#!/usr/bin/env python3
"""Does the policy command as much descent as the demo did, at the same remaining distance?

A true closed-loop rollout cannot be replayed from recorded video: once the arm deviates, the
observation it would have seen does not exist. But the question behind "the arm stops 5-10 mm high"
is answerable open-loop, because it is a question about the policy's GAIN, not its trajectory.

At every frame we know (a) how far the demo still had to travel to reach its grasp, and (b) what
per-step motion the policy commands there. Bin by remaining distance and compare the policy's
commanded step against the demo's. A ratio below 1 near the target means the closed-loop fixed point
sits ABOVE the bolt: each step under-shoots slightly, the arm asymptotes short, and no single-step
metric shows it -- teacher-forced error stays sub-millimetre while the accumulated deficit is
exactly the observed shortfall.

Reported per arm:
  gain_row0     policy row-0 step / demo step, along the approach axis
  gain_exec     same over the 5 rows deployment actually executes before replanning
  resid_mm      demo step minus policy step, in mm -- the per-step deficit that integrates
Ratios are computed on BINNED SUMS (sum of policy steps / sum of demo steps in the bin), not as a
mean of per-frame ratios, which would be dominated by near-zero demo steps.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROBOTICS_LAB = pathlib.Path("/home/plaif/workspace/robotics_lab")
ARMS = (("left", 0, "b3"), ("right", 7, "b1"))


def _load_phase_segmentation():
    path = ROBOTICS_LAB / "policy_runner/policy_runner/phase_segmentation.py"
    spec = importlib.util.spec_from_file_location("phase_segmentation", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_segmentation"] = mod
    spec.loader.exec_module(mod)
    return mod


def _video_frames_rgb(mp4: pathlib.Path) -> np.ndarray:
    import av

    with av.open(str(mp4)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _integrate(deltas: np.ndarray, k: int) -> np.ndarray:
    """Compose k per-step ee_local deltas into a displacement in the frame at step 0."""
    cp = np.zeros(3)
    cr = Rotation.identity()
    for i in range(k):
        cp = cp + cr.apply(deltas[i, :3])
        cr = cr * Rotation.from_rotvec(deltas[i, 3:6])
    return cp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-repo-id", required=True)
    ap.add_argument("--lerobot-home", required=True)
    ap.add_argument("--lookback", type=int, default=40)
    ap.add_argument("--execute", type=int, default=5, help="rows deployment runs before replanning")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        cfg, str(pathlib.Path(args.ckpt_base) / str(args.checkpoint_step)))

    phase_mod = _load_phase_segmentation()
    root = pathlib.Path(args.lerobot_home) / args.val_repo_id
    ds = LeRobotDataset(args.val_repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    actions_all = np.stack(hf["actions"]).astype(np.float64)
    ep_from, ep_to = list(ds.episode_data_index["from"]), list(ds.episode_data_index["to"])
    tmap = {int(k): v for k, v in ds.meta.tasks.items()}
    tasks_all = [tmap[int(i)] for i in np.asarray(hf["task_index"])]
    n_ep = len(ep_from) if args.limit is None else min(len(ep_from), args.limit)

    recs = {a: [] for a, _, _ in ARMS}
    for ei in range(n_ep):
        a0, b0 = int(ep_from[ei]), int(ep_to[ei])
        gt = actions_all[a0 : b0 - 1]
        states = states_all[a0 : b0 - 1]
        length = gt.shape[0]
        if states.shape[1] <= 13:
            continue
        bounds = phase_mod.extract_phase_boundaries(states[:, 6] * 100.0, states[:, 13] * 100.0, length)
        if not bounds.clean:
            continue
        li = _video_frames_rgb(root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        ri = _video_frames_rgb(root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4")

        for arm, base, bkey in ARMS:
            bf = int(getattr(bounds, bkey, -1))
            if not (0 <= bf < length):
                continue
            d = gt[:, base : base + 6]
            for t in range(max(0, bf - args.lookback), bf):
                if t + args.execute > length or t >= len(li) or t >= len(ri):
                    continue
                obs = {
                    "observation/left_wrist_0_rgb": li[t],
                    "observation/right_wrist_0_rgb": ri[t],
                    "observation/state": states[t].astype(np.float32),
                    "prompt": tasks_all[a0 + t],
                }
                pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)[:, base : base + 6]
                remain = _integrate(d[t:bf], bf - t)            # demo's remaining displacement
                recs[arm].append({
                    "L": int(bf - t),
                    "remain_mm": (remain * 1000.0).tolist(),
                    "remain_norm_mm": float(np.linalg.norm(remain) * 1000.0),
                    "gt_row0_mm": (d[t, :3] * 1000.0).tolist(),
                    "pr_row0_mm": (pred[0, :3] * 1000.0).tolist(),
                    "gt_exec_mm": (_integrate(d[t:t + args.execute], args.execute) * 1000.0).tolist(),
                    "pr_exec_mm": (_integrate(pred[:args.execute], args.execute) * 1000.0).tolist(),
                })
        print(f"[{ei + 1}/{n_ep}] episode_{ei:06d} ({sum(len(v) for v in recs.values())} rows)", flush=True)

    out = {"config": args.config, "step": args.checkpoint_step, "execute": args.execute, "arms": {}}
    # Approach axis = index 2 of the ee_local step (the extraction showed the remaining z at the
    # grasp is large and positive, so +z points at the bolt).
    EDGES = [0, 5, 10, 20, 40, 80, 1e9]
    for arm in recs:
        R = recs[arm]
        if not R:
            continue
        rn = np.array([r["remain_norm_mm"] for r in R])
        g0 = np.array([r["gt_row0_mm"][2] for r in R])
        p0 = np.array([r["pr_row0_mm"][2] for r in R])
        ge = np.array([r["gt_exec_mm"][2] for r in R])
        pe = np.array([r["pr_exec_mm"][2] for r in R])
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (rn >= lo) & (rn < hi)
            if m.sum() < 10:
                continue
            bins.append({
                "remain_mm": f"{lo}-{hi if hi < 1e8 else 'inf'}",
                "n": int(m.sum()),
                # summed, not averaged per-frame: near the target the demo step approaches zero and a
                # mean of ratios would be dominated by division by ~0.
                "gain_row0": float(p0[m].sum() / g0[m].sum()) if abs(g0[m].sum()) > 1e-9 else None,
                "gain_exec": float(pe[m].sum() / ge[m].sum()) if abs(ge[m].sum()) > 1e-9 else None,
                "gt_row0_mm_mean": float(g0[m].mean()),
                "pr_row0_mm_mean": float(p0[m].mean()),
                "resid_row0_mm_mean": float((g0[m] - p0[m]).mean()),
                "resid_exec_mm_mean": float((ge[m] - pe[m]).mean()),
            })
        out["arms"][arm] = {"n": len(R), "bins": bins}
        print(f"\n=== {arm} (n={len(R)}) ===", flush=True)
        for b in bins:
            print(f"  remain {b['remain_mm']:>8} mm  n={b['n']:5d}  gain_row0={b['gain_row0']:.3f}  "
                  f"gain_exec={b['gain_exec']:.3f}  demo step {b['gt_row0_mm_mean']:+6.3f} vs "
                  f"policy {b['pr_row0_mm_mean']:+6.3f} mm  (deficit {b['resid_row0_mm_mean']:+6.3f})",
                  flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
