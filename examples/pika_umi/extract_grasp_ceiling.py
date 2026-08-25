#!/usr/bin/env python3
"""Build the CEILING dataset: wrist image at frame t -> displacement from t to the grasp frame.

Why this exists: six VLA trainings (three proprio variants, delta vs anchored, LR/is_pad/loss-dim
fixes, and a resize_pad A/B) all land on the same right-arm z scatter, ~18.5-21.1 mm at L23, and a
PAIRED test showed even the resize change moves nothing. Before spending another 17-hour run on a
model variant, answer the prior question: is that scatter reducible from these images AT ALL?

A supervised regressor on exactly this target has a strictly easier job than the policy -- direct
supervision on the measured quantity, no action chunk, no flow matching, no language, no proprio
feedback. If it cannot beat the VLA, the information is not in the frames and model work is over.

Target = displacement (metres, ee_local frame AT t) from t to the arm's grasp frame b, obtained by
integrating the stored per-step ee_local deltas -- the same _integrate_local the eval metric uses, so
the numbers land directly next to grasp_localization's.

L (steps-to-grasp) is NOT stored as an input feature. The policy does not know it either, and "how
far am I from the bolt" is precisely the visual quantity under test. It IS stored as metadata so the
evaluation can be restricted to L=12 / L=23 and compared to the VLA cell-for-cell.

Images are written at NATIVE 480x640 as JPEG. The 224 runs downsample at load time; the native-
resolution run reads them as-is. One extraction serves both.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROBOTICS_LAB = pathlib.Path("/home/plaif/workspace/robotics_lab")
# (name, base column) -- left occupies action dims 0:6, right 7:13; matches the eval script's ARMS.
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

    with av.open(str(mp4)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def _displacement_to(deltas: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compose per-step ee_local deltas 0..k-1 into the displacement at step k.

    deltas: (>=k, 6) = [pos_delta_local3, rot_delta3]. Returns (position3, rotvec3), both expressed
    in the ee_local frame at step 0 -- i.e. exactly what the policy's chunk is asked to produce.
    """
    cp = np.zeros(3)
    cr = Rotation.identity()
    for i in range(k):
        cp = cp + cr.apply(deltas[i, :3])
        cr = cr * Rotation.from_rotvec(deltas[i, 3:6])
    return cp, cr.as_rotvec()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--lerobot-home", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", default="/home/plaif/grasp_ceiling")
    ap.add_argument(
        "--lookback", type=int, default=40, help="sample t in [b-lookback, b-1]; L23 (the primary eval point) must fit"
    )
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    phase_mod = _load_phase_segmentation()
    root = pathlib.Path(args.lerobot_home) / args.repo_id
    ds = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    actions_all = np.stack(hf["actions"]).astype(np.float64)
    ep_from = list(ds.episode_data_index["from"])
    ep_to = list(ds.episode_data_index["to"])
    n_ep = len(ep_from) if args.limit is None else min(len(ep_from), args.limit)

    out = pathlib.Path(args.out)
    img_dir = out / "images" / args.split
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_clean = 0
    for ei in range(n_ep):
        a, b_ = int(ep_from[ei]), int(ep_to[ei])
        gt = actions_all[a : b_ - 1]  # (T-1, 14) per-step ee_local deltas
        states = states_all[a : b_ - 1]
        length = gt.shape[0]
        if states.shape[1] > 13:
            left_grip = states[:, 6] * 100.0
            right_grip = states[:, 13] * 100.0
        else:  # velocity-only proprio carries no gripper
            left_grip = np.zeros(len(states))
            right_grip = np.zeros(len(states))
        bounds = phase_mod.extract_phase_boundaries(left_grip, right_grip, length)
        if not bounds.clean:
            print(f"[{ei + 1}/{n_ep}] episode_{ei:06d} SKIP (phase bounds not clean)", flush=True)
            continue
        n_clean += 1

        vids = {
            "left": root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4",
            "right": root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4",
        }
        frames = {k: _video_frames_rgb(v) for k, v in vids.items()}

        for arm, base, bkey in ARMS:
            bf = int(getattr(bounds, bkey, -1))
            if not (0 <= bf < length):
                continue
            d = gt[:, base : base + 6]
            # SAME-SIDE wrist only: the tightest form of the question is whether the bolt's position
            # is recoverable from the camera on the arm that has to reach it.
            fr = frames[arm]
            for t in range(max(0, bf - args.lookback), bf):
                if t >= len(fr):
                    continue
                pos, rot = _displacement_to(d[t:bf], bf - t)
                name = f"{ei:06d}_{arm}_{t:04d}.jpg"
                Image.fromarray(fr[t]).save(img_dir / name, quality=args.jpeg_quality)
                rows.append((name, ei, 0 if arm == "left" else 1, t, bf, bf - t, *pos, *rot, *states[t]))
        print(f"[{ei + 1}/{n_ep}] episode_{ei:06d} done (rows {len(rows)})", flush=True)

    if not rows:
        raise SystemExit("no rows extracted")
    names = np.array([r[0] for r in rows])
    num = np.array([r[1:] for r in rows], dtype=np.float64)
    np.savez_compressed(
        out / f"meta_{args.split}.npz",
        name=names,
        ep=num[:, 0].astype(np.int32),
        arm=num[:, 1].astype(np.int8),  # 0 = left, 1 = right
        t=num[:, 2].astype(np.int32),
        b=num[:, 3].astype(np.int32),
        L=num[:, 4].astype(np.int32),  # steps-to-grasp; metadata only, never an input
        target_pos=num[:, 5:8],  # metres, ee_local at t
        target_rot=num[:, 8:11],  # rotvec, ee_local at t
        state=num[:, 11:],
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "episodes_seen": n_ep,
                "episodes_clean": n_clean,
                "rows": len(rows),
                "images": str(img_dir),
                "target_pos_std_mm": (num[:, 5:8].std(axis=0) * 1000).round(2).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
