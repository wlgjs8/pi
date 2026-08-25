#!/usr/bin/env python3
"""Precompute pi05 teacher action chunks over the training set, for VA distillation.

Why distillation is the chosen lever (measured, 2026-08-06): the generalisation gap does NOT
separate the winner from the rest -- pi05 H=8 memorises train HARDER than the VA (train z RMSE
0.61/0.71 vs 1.08/1.23) yet still wins val (2.37/1.63 vs 2.47/2.00). Its advantage is a better
prior lowering both curves, not a smaller gap; the regularisation arm (drop_path+aug) confirmed the
converse by making both curves worse. So the way to move the VA is to transfer the teacher's
function, not to constrain the student. Ego-centric constraint holds: teacher and student consume
identical ego inputs (wrist RGB + body-frame velocity proprio), so no state shortcut is introduced.

Output: one .npz per episode shard with teacher_chunks[t] = teacher's H rows at frame t (unnormalised
14-D actions). ~178k frames x 8 x 14 fp32 ≈ 80 MB total. Shardable by episode for parallel GPUs.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def _frames(mp4: pathlib.Path) -> np.ndarray:
    import av

    with av.open(str(mp4)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--lerobot-home", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(cfg, str(pathlib.Path(args.ckpt_base) / str(args.checkpoint_step)))
    H = cfg.model.action_horizon

    root = pathlib.Path(args.lerobot_home) / args.repo_id
    ds = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    tmap = {int(k): v for k, v in ds.meta.tasks.items()}
    tasks_all = [tmap[int(i)] for i in np.asarray(hf["task_index"])]
    ep_from = list(ds.episode_data_index["from"])
    ep_to = list(ds.episode_data_index["to"])

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ei in range(len(ep_from)):
        if ei % args.num_shards != args.shard_index:
            continue
        dst = out / f"ep_{ei:06d}.npz"
        if dst.exists():
            continue
        a0, b0 = int(ep_from[ei]), int(ep_to[ei])
        T = b0 - a0 - 1
        li = _frames(root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        ri = _frames(root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        n = min(T, len(li), len(ri))
        chunks = np.zeros((n, H, 14), dtype=np.float32)
        for t in range(n):
            obs = {
                "observation/left_wrist_0_rgb": li[t],
                "observation/right_wrist_0_rgb": ri[t],
                "observation/state": states_all[a0 + t].astype(np.float32),
                "prompt": tasks_all[a0 + t],
            }
            chunks[t] = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)[:H, :14]
        np.savez_compressed(dst, chunks=chunks, ep=ei, frame0=a0)
        print(f"[shard {args.shard_index}] ep {ei} ({n} frames) done", flush=True)
    print(f"[shard {args.shard_index}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
