"""Convert PiKA-UMI HDF5 episodes (NEW storage-server schema) to a VIDEO-backed LeRobot dataset.

Differences from convert_pika_umi_data_to_lerobot.py:
  * New raw HDF5 layout (per-arm subgroups): observations/{left,right}/{pose,gripper,images/realsense_color}.
    - pose:    (T,7) pos(3)+quat(4), stand-frame TCP (same convention as old tcp_stand_*).
    - gripper: (T,2) col0=measured/actual %, col1=commanded target %. We use col0 (matches the
      old single-stream gripper used by _state/_arm_actions).
    - images:  PNG-encoded bytes now (was JPEG); cv2.imdecode handles both transparently.
  * No split manifest -- globs episode_*.hdf5 under the session dirs.
  * VIDEO backend (dtype="video", use_videos=True): each camera stream is encoded to one MP4 per
    episode instead of PNG-in-parquet -> far smaller + NFS-friendly (sequential reads).

State/action math (reset-relative proprio + ee_local per-step delta) is IDENTICAL to the existing
converter so the relrel checkpoints / policy_runner inference contract are preserved.
"""

import pathlib

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

REPO_ID = "plaif/pika_umi_video_test"
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt with the "
    "left arm and put it in the left box"
)


def _decode_color(encoded: np.ndarray) -> np.ndarray:
    bgr = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode color frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_IDENT_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _bad_pose_mask(pose: np.ndarray) -> np.ndarray:
    """A frame is bad if any of pos/quat is non-finite (tracking dropout writes NaN) or the
    quaternion is zero-norm. Concurrent collection produces such frames."""
    finite = np.isfinite(pose).all(axis=1)
    qnorm = np.linalg.norm(pose[:, 3:7], axis=1)
    return ~(finite & (qnorm > 1e-6))


def _sanitize_pose(pose: np.ndarray) -> np.ndarray:
    """Forward-fill bad frames (then back-fill leading) with the nearest valid pose; identity if all bad."""
    pose = pose.copy()
    bad = _bad_pose_mask(pose)
    if not bad.any():
        return pose
    last = None
    for i in range(len(pose)):
        if not bad[i]:
            last = pose[i].copy()
        else:
            pose[i] = last if last is not None else _IDENT_POSE
    return pose


def _state(left_pose, right_pose, left_grip, right_grip) -> np.ndarray:
    def _rel(pose):
        r0 = Rotation.from_quat(pose[0, 3:7])
        pos_rel = r0.inv().apply(pose[:, :3] - pose[0, :3])
        rot_rel = (r0.inv() * Rotation.from_quat(pose[:, 3:7])).as_rotvec()
        return pos_rel, rot_rel

    left_pos_rel, left_rotvec = _rel(left_pose)
    right_pos_rel, right_rotvec = _rel(right_pose)
    return np.concatenate(
        [
            left_pos_rel,
            left_rotvec,
            (left_grip[:, None] / 100.0),
            right_pos_rel,
            right_rotvec,
            (right_grip[:, None] / 100.0),
        ],
        axis=1,
    ).astype(np.float32)


def _arm_actions(pose, grip) -> np.ndarray:
    cur = Rotation.from_quat(pose[:-1, 3:7])
    nxt = Rotation.from_quat(pose[1:, 3:7])
    pos_delta_world = pose[1:, :3] - pose[:-1, :3]
    pos_delta_local = cur.inv().apply(pos_delta_world)
    rot_delta = (cur.inv() * nxt).as_rotvec()
    grip_delta = ((grip[1:] - grip[:-1]) / 100.0)[:, None]
    return np.concatenate([pos_delta_local, rot_delta, grip_delta], axis=1).astype(np.float32)


def _actions(left_pose, right_pose, left_grip, right_grip) -> np.ndarray:
    return np.concatenate(
        [_arm_actions(left_pose, left_grip), _arm_actions(right_pose, right_grip)], axis=1
    ).astype(np.float32)


def _find_episodes(data_root: pathlib.Path) -> list[pathlib.Path]:
    eps = sorted(p for p in data_root.glob("data_*/episode_*.hdf5") if "_tmp" not in str(p))
    if not eps:
        raise FileNotFoundError(f"no episode_*.hdf5 under {data_root}/data_*")
    return eps


def _make_dataset(repo_id: str, root: pathlib.Path):
    import shutil

    if root.exists():
        shutil.rmtree(root)
    _img_feat = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]}
    features = {
        "left_wrist_0_rgb": _img_feat,
        "right_wrist_0_rgb": _img_feat,
        "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="pika_umi_dual_arm",
        fps=30,
        features=features,
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=16,
        image_writer_processes=4,
    )


def _episode_frames(ep: pathlib.Path):
    """Read one episode -> list of LeRobot frames. Raises on incomplete/bad data (caller skips)."""
    with h5py.File(ep, "r") as f:
        L, R = f["observations/left"], f["observations/right"]
        lp = np.asarray(L["pose"], dtype=np.float64)
        rp = np.asarray(R["pose"], dtype=np.float64)
        bad_ratio = max(_bad_pose_mask(lp).mean(), _bad_pose_mask(rp).mean())
        if bad_ratio > 0.10:
            raise ValueError(f"tracking dropout: {bad_ratio:.0%} bad pose frames")
        left_pose = _sanitize_pose(lp)
        right_pose = _sanitize_pose(rp)
        left_grip = np.nan_to_num(np.asarray(L["gripper"], dtype=np.float64)[:, 0])
        right_grip = np.nan_to_num(np.asarray(R["gripper"], dtype=np.float64)[:, 0])
        left_images = L["images/realsense_color"]
        right_images = R["images/realsense_color"]

        states = _state(left_pose, right_pose, left_grip, right_grip)
        actions = _actions(left_pose, right_pose, left_grip, right_grip)
        if actions.shape[0] != states.shape[0] - 1:
            raise ValueError("length mismatch")
        return [
            {
                "left_wrist_0_rgb": _decode_color(left_images[t]),
                "right_wrist_0_rgb": _decode_color(right_images[t]),
                "state": states[t],
                "actions": actions[t],
                "task": PROMPT,
            }
            for t in range(actions.shape[0])
        ]


def main(
    data_root: pathlib.Path = pathlib.Path("/mnt/pika/bolt/data"),
    lerobot_home: pathlib.Path = pathlib.Path("/mnt/pika/lerobot"),
    train_repo_id: str = "plaif/pika_umi_video_train",
    val_repo_id: str = "plaif/pika_umi_video_val",
    val_frac: float = 0.1,
    seed: int = 0,
    vcodec: str = "h264",
    limit: int | None = None,
    split_record: pathlib.Path = pathlib.Path("/mnt/pika/lerobot/pika_umi_video_split.json"),
):
    import functools
    import json

    # LeRobot 0.1.0's encode_episode_videos() calls encode_video_frames() WITHOUT a vcodec arg, so
    # it always uses the libsvtav1 default. Monkeypatch the name in the dataset module to pin our
    # codec (h264 = much faster encode + decode than AV1; the create(video_backend=...) arg only
    # controls the *decode* backend, not the encoder).
    import lerobot.common.datasets.lerobot_dataset as _lrd
    from lerobot.common.datasets.video_utils import encode_video_frames as _enc

    _lrd.encode_video_frames = functools.partial(_enc, vcodec=vcodec)

    episodes = _find_episodes(data_root)
    if limit is not None:
        episodes = episodes[:limit]

    # Deterministic episode-level split: whole episodes go to val (no within-episode frame leakage).
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, round(val_frac * len(episodes)))
    val_idx = set(order[:n_val].tolist())
    print(f"found {len(episodes)} episodes; split seed={seed} -> {len(episodes) - n_val} train / {n_val} val")

    train_ds = _make_dataset(train_repo_id, lerobot_home / train_repo_id)
    val_ds = _make_dataset(val_repo_id, lerobot_home / val_repo_id)

    counts = {"train": [0, 0], "val": [0, 0]}  # [episodes, frames]
    skipped = []
    split_log = {"train": [], "val": []}
    for i, ep in enumerate(episodes):
        which = "val" if i in val_idx else "train"
        try:
            frames = _episode_frames(ep)
        except Exception as e:  # incomplete/locked file (concurrent collection) or bad data
            skipped.append((str(ep), repr(e)))
            print(f"[{i + 1}/{len(episodes)}] SKIP({which}) {ep.parent.name}/{ep.name}: {e}")
            continue
        ds = val_ds if which == "val" else train_ds
        for fr in frames:
            ds.add_frame(fr)
        ds.save_episode()
        counts[which][0] += 1
        counts[which][1] += len(frames)
        split_log[which].append(f"{ep.parent.name}/{ep.name}")
        print(f"[{i + 1}/{len(episodes)}] {which} {ep.parent.name}/{ep.name}: {len(frames)} frames")

    split_record.parent.mkdir(parents=True, exist_ok=True)
    split_record.write_text(json.dumps({"seed": seed, "val_frac": val_frac, **split_log}, indent=2))
    print(
        f"DONE: train {counts['train'][0]}ep/{counts['train'][1]}fr, "
        f"val {counts['val'][0]}ep/{counts['val'][1]}fr; skipped {len(skipped)}; split -> {split_record}"
    )
    if skipped:
        for s, e in skipped:
            print("  SKIP", s, e)


if __name__ == "__main__":
    tyro.cli(main)
