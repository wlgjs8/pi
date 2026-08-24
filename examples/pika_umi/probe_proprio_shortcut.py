"""Causal intervention probe: is the deployed pi0.5 policy reading VISION, or just echoing proprio?

Queries a running `serve_policy` (default ws://127.0.0.1:8000) with the depth_z50 val set and, per
frame, re-infers under a set of ablations. Two families:

  * proprio ablations (state zeroed / shuffled / rescaled) -- if predictions barely move, proprio is
    not driving the output.
  * vision ablations (rgb greyed / depth blanked to far / whole scene swapped) -- if predictions barely
    move, VISION is not driving the output.

Everything is scored against (a) the ground-truth demo chunk and (b) the zero-parameter
`a_hat(t) = state(t)` copy baseline, both normalized by the val-set pose action std, so the numbers are
directly comparable to examples/pika_umi/eval_pika_umi_val_tcp_lerobot.py's pose-only nMSE.

`base` is drawn twice so every sensitivity has a flow-head sampling noise floor to beat.
"""

import argparse
import collections
import json
import pathlib

import numpy as np

POSE_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP_DIMS = (6, 13)
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt "
    "with the left arm and put it in the left box"
)


def _video_frames_rgb(path, want):
    """Decode only frames in `want` from an mp4 -> {index: HWC uint8 RGB}."""
    import av

    out, wset, mx = {}, set(want), max(want)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in wset:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= mx:
                break
    return out


def _grasp_windows(actions, length, horizon):
    """Frames worth probing: the approach->grasp window of each arm's first two closes, plus uniform."""
    want = set()
    for gcol in GRIP_DIMS:
        opened = (actions[:, gcol] >= 0.25).astype(int)
        closes = np.where((opened[1:] == 0) & (opened[:-1] == 1))[0] + 1
        for c0 in closes[:2]:
            for off in (-12, -6, -2, 0, 4):
                if 0 <= c0 + off < length - horizon:
                    want.add(int(c0 + off))
    for f in np.linspace(5, max(5, length - horizon - 1), 4).astype(int):
        if 0 <= f < length - horizon:
            want.add(int(f))
    return sorted(want)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="local LeRobot dataset root (depth_z50 val)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--n-episodes", type=int, default=32)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from openpi_client import websocket_client_policy

    root = pathlib.Path(args.root)
    rng = np.random.default_rng(args.seed)
    eps = [json.loads(l) for l in open(root / "meta/episodes.jsonl")]
    tasks = {int(json.loads(l)["task_index"]): json.loads(l)["task"] for l in open(root / "meta/tasks.jsonl")}

    # Fixed normalizer: per-dim std over ALL val GT actions (checkpoint-independent), same as fair-eval.
    all_actions = np.concatenate(
        [np.stack(pq.read_table(p, columns=["actions"])["actions"].to_numpy()) for p in
         sorted((root / "data").rglob("*.parquet"))]
    ).astype(np.float64)
    action_std = all_actions.std(axis=0)
    pose_scale = float(np.mean(np.square(np.maximum(action_std[POSE_DIMS], 1e-12))))
    print(f"val frames={len(all_actions)}  pose_scale={pose_scale:.3e}")

    # ---- collect the probe frames -------------------------------------------------------------
    sel = np.linspace(0, len(eps) - 1, args.n_episodes).astype(int)
    samples = []
    for ei in sel:
        ep = eps[int(ei)]
        idx = int(ep["episode_index"])
        length = int(ep["length"])
        tbl = pq.read_table(root / f"data/chunk-000/episode_{idx:06d}.parquet")
        acts = np.stack(tbl["actions"].to_numpy()).astype(np.float64)
        states = np.stack(tbl["state"].to_numpy()).astype(np.float32)
        task = tasks[int(np.asarray(tbl["task_index"].to_numpy())[0])]
        want = _grasp_windows(acts, length, args.horizon)
        if not want:
            continue
        vids = {
            k: _video_frames_rgb(root / f"videos/chunk-000/{k}/episode_{idx:06d}.mp4", want)
            for k in ("left_wrist_0_rgb", "right_wrist_0_rgb", "left_wrist_0_depth", "right_wrist_0_depth")
        }
        for f in want:
            if any(f not in vids[k] for k in vids):
                continue
            samples.append({
                "ep": idx, "f": f, "task": task,
                "state": states[f],
                "gt": acts[f : f + args.horizon],
                **{k: vids[k][f] for k in vids},
            })
    rng.shuffle(samples)
    samples = samples[: args.max_frames]
    print(f"probe frames: {len(samples)} from {len(sel)} episodes")

    # ---- the ablations ------------------------------------------------------------------------
    GREY = None  # lazily sized

    def build(s, cond, other):
        obs = {
            "observation/left_wrist_0_rgb": s["left_wrist_0_rgb"],
            "observation/right_wrist_0_rgb": s["right_wrist_0_rgb"],
            "observation/left_wrist_0_depth": s["left_wrist_0_depth"],
            "observation/right_wrist_0_depth": s["right_wrist_0_depth"],
            "observation/state": s["state"].astype(np.float32),
            "prompt": s["task"],
        }
        if cond in ("base", "base2"):
            return obs
        if cond == "state_zero":
            obs["observation/state"] = np.zeros_like(obs["observation/state"])
        elif cond == "state_shuf":
            obs["observation/state"] = other["state"].astype(np.float32)
        elif cond == "state_half":
            obs["observation/state"] = (0.5 * s["state"]).astype(np.float32)
        elif cond == "state_double":
            obs["observation/state"] = (2.0 * s["state"]).astype(np.float32)
        elif cond == "depth_far":  # == policy_runner --blank-depth
            for k in ("left", "right"):
                obs[f"observation/{k}_wrist_0_depth"] = np.full_like(s[f"{k}_wrist_0_depth"], 255)
        elif cond == "rgb_grey":
            for k in ("left", "right"):
                obs[f"observation/{k}_wrist_0_rgb"] = np.full_like(s[f"{k}_wrist_0_rgb"], 128)
        elif cond == "scene_shuf":  # every camera from an unrelated frame, real state
            for k in ("left_wrist_0_rgb", "right_wrist_0_rgb", "left_wrist_0_depth", "right_wrist_0_depth"):
                obs[f"observation/{k}"] = other[k]
        elif cond == "vision_off":  # rgb grey + depth far: prediction can only come from proprio+prompt
            for k in ("left", "right"):
                obs[f"observation/{k}_wrist_0_rgb"] = np.full_like(s[f"{k}_wrist_0_rgb"], 128)
                obs[f"observation/{k}_wrist_0_depth"] = np.full_like(s[f"{k}_wrist_0_depth"], 255)
        else:
            raise ValueError(cond)
        return obs

    CONDS = ["base", "base2", "state_zero", "state_shuf", "state_half", "state_double",
             "depth_far", "rgb_grey", "scene_shuf", "vision_off"]

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("server metadata:", client.get_server_metadata())

    preds = collections.defaultdict(list)
    gts, states = [], []
    for i, s in enumerate(samples):
        other = samples[int(rng.integers(len(samples)))]
        for cond in CONDS:
            out = client.infer(build(s, cond, other))
            preds[cond].append(np.asarray(out["actions"], dtype=np.float64))
        gts.append(s["gt"])
        states.append(s["state"].astype(np.float64))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)}", flush=True)

    gts = np.stack(gts)                                    # (N,H,14)
    states = np.stack(states)                              # (N,12)
    P = {c: np.stack(v) for c, v in preds.items()}          # (N,H,>=14)
    H = min(args.horizon, gts.shape[1], P["base"].shape[1])

    def pose_nmse(pred, target):
        e = (pred[..., POSE_DIMS] - target[..., POSE_DIMS]) ** 2
        return float(e.mean(axis=-1).mean() / pose_scale)

    # ---- reference predictors (zero parameters) -----------------------------------------------
    # A 1-step metric structurally favours velocity extrapolation, so the honest comparison is the
    # whole-chunk one: hold the last velocity for all H steps and see whether the policy beats it.
    copy1 = np.zeros((len(states), 1, 14))
    copy1[:, 0, POSE_DIMS] = states
    copyH = np.repeat(copy1, H, axis=1)                                    # constant-velocity extrapolation
    zeroH = np.zeros((len(states), H, 14))                                 # freeze in place
    meanH = np.broadcast_to(all_actions.mean(axis=0), (len(states), H, 14))  # dataset mean action
    refs = {
        "copy-proprio (v_{t-1})": (copy1, copyH),
        "zero (freeze)": (zeroH[:, :1], zeroH),
        "dataset mean": (meanH[:, :1], meanH),
    }

    print("\n" + "=" * 92)
    print("FIRST-STEP + WHOLE-CHUNK pose-only nMSE vs GT  (lower=better; 1.0 = predicting the mean)")
    print(f"{'condition':14s} {'first-step':>11s} {'chunk-H':>9s} {'|d vs base| (norm rms)':>24s}")
    print("-" * 92)
    rows = {}
    base = P["base"]
    for c in CONDS:
        fs = pose_nmse(P[c][:, :1, :], gts[:, :1, :])
        ch = pose_nmse(P[c][:, :H, :], gts[:, :H, :])
        d = np.sqrt(float((((P[c][:, :H, POSE_DIMS] - base[:, :H, POSE_DIMS]) ** 2).mean())) / pose_scale)
        rows[c] = dict(first_step=fs, chunk=ch, delta_vs_base=d)
        print(f"{c:14s} {fs:11.4f} {ch:9.4f} {d:24.4f}")
    print("-" * 92)
    ref_rows = {}
    for nm, (r1, rH) in refs.items():
        f1 = pose_nmse(r1, gts[:, :1, :])
        fH = pose_nmse(rH, gts[:, :H, :])
        ref_rows[nm] = dict(first_step=f1, chunk=fH)
        print(f"{nm:14.14s} {f1:11.4f} {fH:9.4f} {'-':>24s}   <- zero-parameter reference")
    copy_nmse = ref_rows["copy-proprio (v_{t-1})"]["first_step"]
    print("=" * 92)

    noise = rows["base2"]["delta_vs_base"]
    print(f"\nflow-head sampling noise floor (base vs base2) = {noise:.4f}")
    print("sensitivity in units of that noise floor:")
    for c in CONDS:
        if c in ("base", "base2"):
            continue
        r = rows[c]["delta_vs_base"] / max(noise, 1e-9)
        print(f"  {c:14s} {r:7.2f}x noise   (first-step nMSE {rows[c]['first_step']:.4f})")

    # per-axis first-step, base vs copy baseline
    names = ["Lx", "Ly", "Lz", "Lrx", "Lry", "Lrz", "Rx", "Ry", "Rz", "Rrx", "Rry", "Rrz"]
    print("\nper-axis first-step nMSE (normalized per-dim by that dim's val std):")
    print(f"{'axis':6s} {'model':>9s} {'copy':>9s} {'vision_off':>11s} {'model/copy':>11s}")
    per_axis = {}
    for j, (nm, d) in enumerate(zip(names, POSE_DIMS)):
        var = max(action_std[d] ** 2, 1e-24)
        m = float(((P["base"][:, 0, d] - gts[:, 0, d]) ** 2).mean() / var)
        cp = float(((states[:, j] - gts[:, 0, d]) ** 2).mean() / var)
        vo = float(((P["vision_off"][:, 0, d] - gts[:, 0, d]) ** 2).mean() / var)
        per_axis[nm] = dict(model=m, copy=cp, vision_off=vo)
        print(f"{nm:6s} {m:9.4f} {cp:9.4f} {vo:11.4f} {m/max(cp,1e-9):11.2f}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "server": {"host": args.host, "port": args.port,
                       "metadata": {k: str(v) for k, v in (client.get_server_metadata() or {}).items()}},
            "root": str(root), "n_frames": len(samples), "horizon": H,
            "pose_scale": pose_scale, "copy_proprio_first_step_nmse": copy_nmse,
            "conditions": rows, "reference_predictors": ref_rows, "per_axis_first_step": per_axis,
            "noise_floor_delta": noise,
        }, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
