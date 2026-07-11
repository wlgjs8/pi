"""A1 probe: flow-head MULTIMODALITY at choice frames (t~0, neutral gaze).

Question (gates RL/steering feasibility, see llm-wiki vla-rollout-diagnosis 2026-07-11):
at the episode-start choice frame, does the stochastic flow head EVER sample action
chunks committing toward a different target than its single dominant one? If draws are
unimodal (endpoint scatter << inter-bolt scale ~14-28cm), latent-noise steering (DSRL)
has no modes to select and only an edit-actor (EXPO-style) or new data can change the
target; if bimodal at bolt scale, test-time mode selection is cheap.

Method: N stochastic policy.infer draws per frame; integrate each ee_local delta chunk
per arm (same math as the eval's _integrate_local); measure per-(episode,frame,arm):
  - endpoint scatter: RMS distance from mean endpoint + max pairwise distance
  - direction spread: max pairwise angle between endpoint directions (deg)
  - 2-means bimodality: inter-centroid distance / mean intra-cluster RMS (+ cluster sizes)
  - GT chunk endpoint (same integration) for scale + sanity
Units: lerobot action units (meters for pos dims).

Run inside the openpi venv:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/pika_umi/probe_choiceframe_multimodality.py \
    --ckpt-base .../checkpoints/pi05_.../depth_z50_nopad_real_h24 --checkpoint-step 79999 \
    --config pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_nopad_h24 \
    --val-repo-id plaif/pika_umi_video_val_tcp_gripabs_velproprio_depth_z50
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

LEROBOT_HOME = pathlib.Path("/home/plaif/storage/pika/lerobot")
OUT_DIR = pathlib.Path("/home/plaif/workspace/openpi_runs/analysis")
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt with the "
    "left arm and put it in the left box"
)


def _integrate_local(deltas: np.ndarray):
    """ee_local per-step deltas (H,6) -> displacement-from-anchor positions (H,3). Same math as
    eval_pika_umi_val_tcp_lerobot._integrate_local (positions only)."""
    H = deltas.shape[0]
    pos = np.zeros((H, 3))
    cp = np.zeros(3)
    cr = Rotation.identity()
    for k in range(H):
        pos[k] = cp
        cp = cp + cr.apply(deltas[k, :3])
        cr = cr * Rotation.from_rotvec(deltas[k, 3:6])
    return pos


def _video_frames_rgb(path: pathlib.Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _two_means(x: np.ndarray, iters: int = 32):
    """Tiny 2-means over endpoints (N,3). Returns inter-centroid dist, mean intra-RMS, sizes."""
    c = x[[0, np.argmax(np.linalg.norm(x - x[0], axis=1))]].copy()  # farthest-pair init
    for _ in range(iters):
        d = np.linalg.norm(x[:, None] - c[None], axis=2)
        lab = d.argmin(1)
        for j in (0, 1):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    intra = []
    for j in (0, 1):
        if (lab == j).any():
            intra.append(float(np.sqrt(((x[lab == j] - c[j]) ** 2).sum(1).mean())))
    return float(np.linalg.norm(c[0] - c[1])), float(np.mean(intra)), [int((lab == 0).sum()), int((lab == 1).sum())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-repo-id", required=True)
    ap.add_argument("--lerobot-home", default=str(LEROBOT_HOME))
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--frames", default="0,12", help="comma list of frame indices per episode")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ckpt_dir = pathlib.Path(args.ckpt_base) / str(args.checkpoint_step)
    if not ckpt_dir.exists():
        raise FileNotFoundError(ckpt_dir)
    root = pathlib.Path(args.lerobot_home) / args.val_repo_id
    ds = LeRobotDataset(args.val_repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    actions_all = np.stack(hf["actions"]).astype(np.float64)
    ep_from = list(ds.episode_data_index["from"])
    ep_to = list(ds.episode_data_index["to"])
    _tmap = {int(k): v for k, v in ds.meta.tasks.items()}
    tasks_all = [_tmap.get(int(i), PROMPT) for i in np.asarray(hf["task_index"])]
    n_ep = len(ep_from)
    if args.limit:
        n_ep = min(n_ep, args.limit)
    frames = [int(x) for x in args.frames.split(",")]
    has_depth = (root / "videos/chunk-000/left_wrist_0_depth").exists()

    cfg = _config.get_config(args.config)
    H = int(cfg.model.action_horizon)
    policy = _policy_config.create_trained_policy(cfg, str(ckpt_dir))
    ARMS = [("left", 0), ("right", 7)]

    rows = []
    for ei in range(n_ep):
        a, b = int(ep_from[ei]), int(ep_to[ei])
        gt = actions_all[a : b - 1]
        length = gt.shape[0]
        states = states_all[a:b]
        left_img = _video_frames_rgb(root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        right_img = _video_frames_rgb(root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        if has_depth:
            left_depth = _video_frames_rgb(root / "videos/chunk-000/left_wrist_0_depth" / f"episode_{ei:06d}.mp4")
            right_depth = _video_frames_rgb(root / "videos/chunk-000/right_wrist_0_depth" / f"episode_{ei:06d}.mp4")
        for t in frames:
            if t + H > length:
                continue
            obs = {
                "observation/left_wrist_0_rgb": left_img[t],
                "observation/right_wrist_0_rgb": right_img[t],
                "observation/state": states[t].astype(np.float32),
                "prompt": tasks_all[a + t],
            }
            if has_depth:
                obs["observation/left_wrist_0_depth"] = left_depth[t]
                obs["observation/right_wrist_0_depth"] = right_depth[t]
            chunks = [np.asarray(policy.infer(obs)["actions"], dtype=np.float64) for _ in range(args.draws)]
            gt_chunk = gt[t : t + H]
            for arm, base in ARMS:
                ends = np.stack([_integrate_local(ch[:H, base : base + 6])[-1] for ch in chunks])
                gt_end = _integrate_local(gt_chunk[:, base : base + 6])[-1]
                mean_end = ends.mean(0)
                rms = float(np.sqrt(((ends - mean_end) ** 2).sum(1).mean()))
                pair = float(max(np.linalg.norm(ends[i] - ends[j]) for i in range(len(ends)) for j in range(i)))
                dirs = ends / np.maximum(np.linalg.norm(ends, axis=1, keepdims=True), 1e-9)
                cosmax = min(float(dirs[i] @ dirs[j]) for i in range(len(dirs)) for j in range(i))
                inter, intra, sizes = _two_means(ends)
                rows.append({
                    "ep": ei, "t": t, "arm": arm,
                    "rms_scatter_m": rms, "max_pairwise_m": pair,
                    "max_angle_deg": float(np.degrees(np.arccos(np.clip(cosmax, -1, 1)))),
                    "kmeans_inter_m": inter, "kmeans_intra_m": intra, "kmeans_sizes": sizes,
                    "gt_end_norm_m": float(np.linalg.norm(gt_end)),
                    "mean_end_norm_m": float(np.linalg.norm(mean_end)),
                    "gt_dist_from_mean_m": float(np.linalg.norm(gt_end - mean_end)),
                })
        if (ei + 1) % 10 == 0:
            print(f"episode {ei + 1}/{n_ep} done", flush=True)

    out = OUT_DIR / f"choiceframe_multimodality_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "rows": rows}, open(out, "w"))
    print(f"saved {out} ({len(rows)} rows)")

    # summary
    arr = lambda k, t: np.array([r[k] for r in rows if r["t"] == t])
    for t in frames:
        if not len(arr("rms_scatter_m", t)):
            continue
        print(f"\n=== frame t={t} (both arms pooled, n={len(arr('rms_scatter_m', t))}) ===")
        for k in ["rms_scatter_m", "max_pairwise_m", "max_angle_deg", "kmeans_inter_m", "gt_end_norm_m", "gt_dist_from_mean_m"]:
            v = arr(k, t)
            print(f"  {k:22s} p50={np.percentile(v, 50):.4f} p90={np.percentile(v, 90):.4f} p99={np.percentile(v, 99):.4f}")
        ratio = arr("kmeans_inter_m", t) / np.maximum(arr("kmeans_intra_m", t), 1e-9)
        print(f"  bimodality inter/intra  p50={np.percentile(ratio, 50):.2f} p90={np.percentile(ratio, 90):.2f} p99={np.percentile(ratio, 99):.2f}")


if __name__ == "__main__":
    main()
