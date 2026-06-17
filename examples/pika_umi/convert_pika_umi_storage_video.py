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


def main(
    data_root: pathlib.Path = pathlib.Path("/mnt/pika/bolt/data"),
    root: pathlib.Path = pathlib.Path("/mnt/pika/lerobot/plaif/pika_umi_video_test"),
    repo_id: str = REPO_ID,
    vcodec: str = "h264",
    limit: int | None = None,
):
    import functools
    import shutil

    # LeRobot 0.1.0's encode_episode_videos() calls encode_video_frames() WITHOUT a vcodec arg, so
    # it always uses the libsvtav1 default. Monkeypatch the name in the dataset module to pin our
    # codec (h264 = much faster encode + decode than AV1; the create(video_backend=...) arg only
    # controls the *decode* backend, not the encoder).
    import lerobot.common.datasets.lerobot_dataset as _lrd
    from lerobot.common.datasets.video_utils import encode_video_frames as _enc

    _lrd.encode_video_frames = functools.partial(_enc, vcodec=vcodec)

    if root.exists():
        shutil.rmtree(root)

    episodes = _find_episodes(data_root)
    if limit is not None:
        episodes = episodes[:limit]
    print(f"found {len(episodes)} episodes under {data_root}")

    _img_feat = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]}
    features = {
        "left_wrist_0_rgb": _img_feat,
        "right_wrist_0_rgb": _img_feat,
        "state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (14,), "names": ["actions"]},
    }

    dataset = LeRobotDataset.create(
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

    total_frames = 0
    converted = 0
    skipped = []
    for i, ep in enumerate(episodes):
        try:
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

                frames = []
                for t in range(actions.shape[0]):
                    frames.append(
                        {
                            "left_wrist_0_rgb": _decode_color(left_images[t]),
                            "right_wrist_0_rgb": _decode_color(right_images[t]),
                            "state": states[t],
                            "actions": actions[t],
                            "task": PROMPT,
                        }
                    )
        except Exception as e:  # incomplete/locked file (concurrent collection) or bad data
            skipped.append((str(ep), repr(e)))
            print(f"[{i + 1}/{len(episodes)}] SKIP {ep.parent.name}/{ep.name}: {e}")
            continue

        for fr in frames:
            dataset.add_frame(fr)
        dataset.save_episode()
        converted += 1
        total_frames += len(frames)
        print(f"[{i + 1}/{len(episodes)}] {ep.parent.name}/{ep.name}: {len(frames)} frames")

    print(f"DONE: converted {converted}/{len(episodes)} episodes, {total_frames} frames -> {root}")
    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for s, e in skipped:
            print("  ", s, e)


if __name__ == "__main__":
    tyro.cli(main)
