"""Convert PiKA-UMI HDF5 episodes (NEW storage-server schema) to a VIDEO-backed LeRobot dataset.

Differences from convert_pika_umi_data_to_lerobot.py:
  * New raw HDF5 layout (per-arm subgroups): observations/{left,right}/{pose,gripper,images/realsense_color}.
    - pose:    (T,7) pos(3)+quat(4) in `steamvr_world` -- the RAW Vive-tracker device frame
      (HDF5 root attr `pose_frame=steamvr_world`). This is the tracker point, NOT the robot TCP.
    - gripper: (T,2) col0=measured/actual %, col1=commanded target %. We use col0 (matches the
      old single-stream gripper used by _state/_arm_actions).
    - images:  PNG-encoded bytes now (was JPEG); cv2.imdecode handles both transparently.
  * No split manifest -- globs episode_*.hdf5 under the session dirs.
  * VIDEO backend (dtype="video", use_videos=True): each camera stream is encoded to one MP4 per
    episode instead of PNG-in-parquet -> far smaller + NFS-friendly (sequential reads).

Tool-frame retarget (so we can train directly from raw, no separate data_tcp materialization):
  Before computing state/actions we apply the tracker->robot-TCP-equivalent tip offset
      converted = pose_raw . inv(T_tcp_umi_gripper)
  with T_tcp_umi_gripper loaded per-arm from the robotics_lab umi_retarget yaml (default
  `--retarget-config`). This is the SAME math as policy_runner.umi_pipeline._retarget_poses
  (the data->data_tcp step), so raw-only training matches the data_tcp pipeline exactly. Tool
  offset ONLY -- there is no steamvr->stand (world) transform; it was never measured and the
  body-frame (ee_local) representation cancels it (wiki umi-tcp-delta-frame). Pass
  `--retarget-config None` to reproduce the OLD raw-tracker-frame datasets (e.g. pika_umi_video_*_8020).

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


def _decode_color(encoded: np.ndarray, crop_frac: float | None = None) -> np.ndarray:
    bgr = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode color frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if crop_frac is not None:
        rgb = _center_crop(rgb, crop_frac)
    return rgb


def _center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Centered crop keeping `frac` of each dimension (aspect preserved). Used for the wide-FOV
    fisheye: dropping the strongly barrel-distorted periphery + vignette corners pulls the image
    closer to a rectilinear/pi0.5-style view AND lets the downstream resize spend its 224 budget on
    the central working area (better small-bolt detail) instead of distorted edges."""
    h, w = img.shape[:2]
    ch, cw = int(round(h * frac)), int(round(w * frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


def _crop_shape(frac: float | None, base=(480, 640)) -> tuple[int, int, int]:
    h, w = base
    if frac is None:
        return (h, w, 3)
    return (int(round(h * frac)), int(round(w * frac)), 3)


def _make_hold_frame(last_frame: dict, gripper_action: str) -> dict:
    """A frozen 'hold' frame: last image + state held, action = stay-put (zero pose delta; gripper stays
    at its current opening). Appended to an episode's FINAL segment to extend the short post-release tail
    (see the tail-pad rationale at the write loop). State layout is 14-D
    [L pos3, L rot3, L grip(6), R pos3, R rot3, R grip(13)]; action shares the layout."""
    hold = np.zeros(14, dtype=np.float32)
    if gripper_action == "absolute":
        # absolute action gripper = next-step opening; holding means keep the current (open) opening
        hold[6] = last_frame["state"][6]
        hold[13] = last_frame["state"][13]
    # delta mode: zero gripper delta == hold (leave hold[6]/[13] at 0)
    return {
        "left_wrist_0_rgb": last_frame["left_wrist_0_rgb"],
        "right_wrist_0_rgb": last_frame["right_wrist_0_rgb"],
        "state": last_frame["state"],
        "actions": hold,
        "task": PROMPT,
    }


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


def _arm_actions(pose, grip, gripper_action: str, binary_th: float = 25.0) -> np.ndarray:
    cur = Rotation.from_quat(pose[:-1, 3:7])
    nxt = Rotation.from_quat(pose[1:, 3:7])
    pos_delta_world = pose[1:, :3] - pose[:-1, :3]
    pos_delta_local = cur.inv().apply(pos_delta_world)
    rot_delta = (cur.inv() * nxt).as_rotvec()
    # Pose stays an ee_local per-step delta; only the GRIPPER channel switches representation:
    #   delta    : (grip[t+1]-grip[t])/100  -- needs runtime integration at deploy (accumulates drift)
    #   absolute : grip[t+1]/100            -- the target opening; self-correcting, no integration
    #   binary   : 1{grip[t+1] >= th}       -- open(1)/closed(0); strips the task-irrelevant open-angle
    #              noise (~40-95 random) + the under-close. Deploy commands 0 -> firm close (bolt blocks
    #              at its width) / 1 -> open, via the same absolute path (no deploy change).
    # Proprio gripper stays the absolute current opening in every mode.
    if gripper_action == "binary":
        grip_col = (grip[1:] >= binary_th).astype(np.float64)[:, None]
    elif gripper_action == "absolute":
        grip_col = (grip[1:] / 100.0)[:, None]
    else:
        grip_col = ((grip[1:] - grip[:-1]) / 100.0)[:, None]
    return np.concatenate([pos_delta_local, rot_delta, grip_col], axis=1).astype(np.float32)


def _actions(left_pose, right_pose, left_grip, right_grip, gripper_action: str, binary_th: float = 25.0) -> np.ndarray:
    return np.concatenate(
        [_arm_actions(left_pose, left_grip, gripper_action, binary_th),
         _arm_actions(right_pose, right_grip, gripper_action, binary_th)],
        axis=1,
    ).astype(np.float32)


def _load_tool_offset(retarget_config: pathlib.Path) -> dict:
    """Per-arm inv(T_tcp_umi_gripper) from a robotics_lab umi_retarget yaml.

    Returns {side: (Rotation, translation)} where (R, t) == inv(T_tcp_umi_gripper), to be
    right-composed onto each raw tracker pose:  converted = pose_raw . inv(T_tcp_umi_gripper).
    Mirrors policy_runner.umi_pipeline._pose_inverse / _retarget_poses exactly. Tool offset
    only (no steamvr->stand world transform; ee_local cancels it -- wiki umi-tcp-delta-frame).
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(pathlib.Path(retarget_config).read_text())
    schema = str(cfg.get("schema", ""))
    if not schema.startswith("robotics_lab.umi_retarget"):
        raise ValueError(f"{retarget_config}: unexpected retarget schema {schema!r}")
    out = {}
    for side in ("left", "right"):
        pose = cfg[side]["T_tcp_umi_gripper"]  # [tx,ty,tz, qx,qy,qz,qw]
        t = np.asarray(pose[:3], dtype=np.float64)
        r_inv = Rotation.from_quat(pose[3:7]).inv()
        out[side] = (r_inv, r_inv.apply(-t))  # inv(T): R^-1, R^-1 . (-t)
    return out


def _apply_tool_offset(pose: np.ndarray, tcp_inv: tuple) -> np.ndarray:
    """converted = pose . inv(T_tcp_umi_gripper) per frame (SE(3) right-compose)."""
    r_inv, t_inv = tcp_inv
    r = Rotation.from_quat(pose[:, 3:7])
    pos = pose[:, :3] + r.apply(t_inv)  # t_a + R_a . t_b
    quat = (r * r_inv).as_quat()  # R_a . R_b
    return np.concatenate([pos, quat], axis=1)


def _find_episodes(data_root: pathlib.Path) -> list[pathlib.Path]:
    eps = sorted(p for p in data_root.glob("data_*/episode_*.hdf5") if "_tmp" not in str(p))
    if not eps:
        raise FileNotFoundError(f"no episode_*.hdf5 under {data_root}/data_*")
    return eps


def _make_dataset(repo_id: str, root: pathlib.Path, img_shape: tuple[int, int, int] = (480, 640, 3)):
    import shutil

    if root.exists():
        shutil.rmtree(root)
    _img_feat = {"dtype": "video", "shape": img_shape, "names": ["height", "width", "channel"]}
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
        # Lowered from 16/4: the encoder writes video to the same NFS we read raw episodes from; high
        # write concurrency was contending with reads and triggering transient errno-5 EIO (see the
        # read-retry note in main). Fewer writers = less NFS pressure, at a modest encode-speed cost.
        image_writer_threads=8,
        image_writer_processes=2,
    )


def _episode_frames(
    ep: pathlib.Path,
    tool_offset: dict | None,
    gap_threshold_s: float,
    min_seg_frames: int,
    camera: str,
    crop_frac: float | None,
    gripper_action: str,
    binary_th: float = 25.0,
):
    """Read one episode -> (list of SEGMENTS, n_writable_frames).

    Some episodes were collected before the 30 Hz pipeline was stabilized and carry multi-second
    frame-time gaps (see wiki pika-data-collection-gaps). An ee_local per-step delta computed across
    such a gap is a huge bogus jump (corrupts the action chunk AND inflates the global norm-stats
    scale). So we SPLIT each episode at gap transitions (`dt = ts[t+1]-ts[t] > gap_threshold_s`) into
    clean contiguous segments and drop only the gap-spanning transitions + sub-`min_seg_frames` stubs
    — salvaging ~all clean frames instead of dropping whole episodes. Proprio (reset-relative) stays
    referenced to the ORIGINAL episode reset (frame 0), so a segment's states are still
    deploy-distribution poses (displacement-from-reset), not displacement-from-mid-trajectory.
    Raises on incomplete/bad data (caller skips).
    """
    with h5py.File(ep, "r") as f:
        L, R = f["observations/left"], f["observations/right"]
        lp = np.asarray(L["pose"], dtype=np.float64)
        rp = np.asarray(R["pose"], dtype=np.float64)
        bad_ratio = max(_bad_pose_mask(lp).mean(), _bad_pose_mask(rp).mean())
        if bad_ratio > 0.10:
            raise ValueError(f"tracking dropout: {bad_ratio:.0%} bad pose frames")
        left_pose = _sanitize_pose(lp)
        right_pose = _sanitize_pose(rp)
        # Tracker -> robot-TCP-equivalent tip frame (data_tcp-equivalent), before ee_local math.
        if tool_offset is not None:
            left_pose = _apply_tool_offset(left_pose, tool_offset["left"])
            right_pose = _apply_tool_offset(right_pose, tool_offset["right"])
        left_grip = np.nan_to_num(np.asarray(L["gripper"], dtype=np.float64)[:, 0])
        right_grip = np.nan_to_num(np.asarray(R["gripper"], dtype=np.float64)[:, 0])
        # Camera source is swappable: realsense_color (default) or fisheye_color. Both decode to
        # (480,640,3) uint8, so the LeRobot feature shape + downstream config are identical -- only
        # the pixel content (wide-FOV fisheye vs RealSense) changes. The camera keys stay
        # left/right_wrist_0_rgb so an existing config trains on the swapped images unchanged.
        left_images = L[f"images/{camera}_color"]
        right_images = R[f"images/{camera}_color"]

        states = _state(left_pose, right_pose, left_grip, right_grip)
        actions = _actions(left_pose, right_pose, left_grip, right_grip, gripper_action, binary_th)
        if actions.shape[0] != states.shape[0] - 1:
            raise ValueError("length mismatch")
        n_act = actions.shape[0]  # writable frames 0..n_act-1; action[t] spans ts[t]->ts[t+1]

        # Gap transitions: drop frame t when its action delta spans a frame-time gap.
        try:
            ts = np.asarray(f["timestamp"], dtype=np.float64)
            gap = np.diff(ts)[:n_act] > gap_threshold_s
        except Exception:
            gap = np.zeros(n_act, dtype=bool)

        # Contiguous runs of clean writable frames, broken at gap transitions.
        segments, run = [], []
        for t in range(n_act):
            if gap[t]:
                if run:
                    segments.append(run)
                    run = []
                continue
            run.append(t)
        if run:
            segments.append(run)
        segments = [s for s in segments if len(s) >= min_seg_frames]

        out = [
            [
                {
                    "left_wrist_0_rgb": _decode_color(left_images[t], crop_frac),
                    "right_wrist_0_rgb": _decode_color(right_images[t], crop_frac),
                    "state": states[t],
                    "actions": actions[t],
                    "task": PROMPT,
                }
                for t in seg
            ]
            for seg in segments
        ]
        return out, n_act


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
    exclude_record: pathlib.Path | None = None,
    retarget_config: pathlib.Path | None = pathlib.Path(
        "/home/plaif/workspace/robotics_lab/calibration/umi_retarget_eelocal.yaml"
    ),
    gap_threshold_s: float = 0.1,
    min_seg_frames: int = 16,
    camera: str = "realsense",
    split_in: pathlib.Path | None = None,
    gripper_action: str = "delta",
    fisheye_crop_frac: float | None = None,
    tail_pad_frames: int = 0,
    gripper_binary_th: float = 25.0,  # gripper_action=binary: opening >= th -> 1 (open) else 0 (closed/grip)
    exclude_path_substr: str | None = None,  # drop episodes whose path contains this (e.g. "onrobot")
    # Pin val to a prior split's `val` list; ALL other found episodes (incl. newly collected) -> train.
    # (Unlike --split-in, which reproduces the EXACT split and DROPS episodes not in train|val.)
    val_from_record: pathlib.Path | None = None,
):
    import functools
    import json
    import time

    if camera not in ("realsense", "fisheye"):
        raise ValueError(f"--camera must be 'realsense' or 'fisheye', got {camera!r}")
    if gripper_action not in ("delta", "absolute", "binary"):
        raise ValueError(f"--gripper-action must be 'delta', 'absolute', or 'binary', got {gripper_action!r}")
    # Center-crop only makes sense for the wide-FOV fisheye; ignore it for realsense so an A/B keeps
    # the realsense arm at full frame.
    crop_frac = fisheye_crop_frac if (camera == "fisheye" and fisheye_crop_frac) else None
    if fisheye_crop_frac is not None and not (0.0 < fisheye_crop_frac <= 1.0):
        raise ValueError(f"--fisheye-crop-frac must be in (0,1], got {fisheye_crop_frac}")
    img_shape = _crop_shape(crop_frac)

    # LeRobot 0.1.0's encode_episode_videos() calls encode_video_frames() WITHOUT a vcodec arg, so
    # it always uses the libsvtav1 default. Monkeypatch the name in the dataset module to pin our
    # codec (h264 = much faster encode + decode than AV1; the create(video_backend=...) arg only
    # controls the *decode* backend, not the encoder).
    import lerobot.common.datasets.lerobot_dataset as _lrd
    from lerobot.common.datasets.video_utils import encode_video_frames as _enc

    _lrd.encode_video_frames = functools.partial(_enc, vcodec=vcodec)

    # Tool-frame retarget: tracker -> robot-TCP-equivalent tip (data_tcp-equivalent). None => raw tracker frame.
    tool_offset = _load_tool_offset(retarget_config) if retarget_config is not None else None
    if tool_offset is not None:
        print(f"pose frame: tcp_tip (tool offset inv(T_tcp_umi_gripper) applied) <- {retarget_config}")
    else:
        print("pose frame: raw tracker (steamvr_world, NO tool offset) -- legacy *_8020 behavior")
    print(f"gap-aware split: dt>{gap_threshold_s*1000:.0f}ms breaks an episode; drop segments < {min_seg_frames} frames")
    print(
        f"camera={camera} | gripper_action={gripper_action} | "
        f"crop={'none' if crop_frac is None else f'{crop_frac:g} -> {img_shape[:2]}'} | "
        f"tail_pad_frames={tail_pad_frames} (train only)"
    )

    episodes = _find_episodes(data_root)
    if exclude_path_substr:
        n0 = len(episodes)
        episodes = [e for e in episodes if exclude_path_substr not in str(e)]
        print(f"excluded {n0 - len(episodes)} episodes matching path substr {exclude_path_substr!r} -> {len(episodes)} remain")
    # Drop episodes already used in a prior split (build a fresh/unseen test set from new collection).
    if exclude_record is not None:
        ex = json.loads(pathlib.Path(exclude_record).read_text())
        excluded = set(ex.get("train", [])) | set(ex.get("val", []))
        episodes = [e for e in episodes if f"{e.parent.name}/{e.name}" not in excluded]
        print(f"excluded {len(excluded)} prior-split episodes -> {len(episodes)} remaining")
    if limit is not None:
        episodes = episodes[:limit]

    # Episode-level split: whole episodes go to val (no within-episode frame leakage).
    def _key(e: pathlib.Path) -> str:
        return f"{e.parent.name}/{e.name}"

    if val_from_record is not None:
        # Pin val to a prior split's `val` list; everything ELSE (incl. newly collected episodes) -> train.
        rec = json.loads(pathlib.Path(val_from_record).read_text())
        val_keys = set(rec.get("val", []))
        miss = val_keys - {_key(e) for e in episodes}
        if miss:
            raise SystemExit(f"--val-from-record: {len(miss)} pinned val episodes not found: {sorted(miss)[:5]}")
        val_idx = {i for i, e in enumerate(episodes) if _key(e) in val_keys}
        n_val = len(val_idx)
        print(f"val-from-record {val_from_record}: {n_val} val pinned, {len(episodes) - n_val} -> train (new episodes included)")
    elif split_in is not None:
        # Reproduce an EXACT prior split (e.g. the tcp_8020 realsense run) so a fisheye dataset
        # differs only in pixels -> apples-to-apples camera comparison AND reusable norm-stats
        # (state/actions identical). Assign by the recorded train/val episode lists; drop any
        # episode not in the recorded split (collection may have grown since).
        rec = json.loads(pathlib.Path(split_in).read_text())
        train_keys, val_keys = set(rec.get("train", [])), set(rec.get("val", []))
        known = train_keys | val_keys
        episodes = [e for e in episodes if _key(e) in known]
        val_idx = {i for i, e in enumerate(episodes) if _key(e) in val_keys}
        n_val = len(val_idx)
        missing = known - {_key(e) for e in episodes}
        print(
            f"reproducing split from {split_in}: {len(episodes) - n_val} train / {n_val} val "
            f"(camera={camera}); {len(missing)} recorded episodes not found on disk"
        )
    else:
        # Deterministic episode-level split. val_frac<=0 -> single dataset (everything to train_repo_id).
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(episodes))
        n_val = max(1, round(val_frac * len(episodes))) if val_frac > 0 else 0
        val_idx = set(order[:n_val].tolist())
        print(f"found {len(episodes)} episodes; split seed={seed} -> {len(episodes) - n_val} train / {n_val} val (camera={camera})")

    train_ds = _make_dataset(train_repo_id, lerobot_home / train_repo_id, img_shape)
    val_ds = _make_dataset(val_repo_id, lerobot_home / val_repo_id, img_shape) if n_val > 0 else None

    counts = {"train": [0, 0], "val": [0, 0]}  # [segments(=lerobot episodes), frames]
    skipped = []
    split_log = {"train": [], "val": []}
    dropped_frames = 0  # frames lost to gap transitions + sub-min-length stubs
    n_src_with_gaps = 0  # source episodes that produced >1 segment or lost frames
    for i, ep in enumerate(episodes):
        which = "val" if i in val_idx else "train"
        # Reads come off the same NFS the encoder writes to; under heavy concurrent load the server can
        # return transient errno-5 EIO (verified: the same files read fine once load drops). Retry those
        # with backoff so a transient blip doesn't silently drop a good episode. Real data errors
        # (tracking dropout / no clean segment / length mismatch -> ValueError) are NOT retried.
        segments = n_writable = None
        for attempt in range(4):
            try:
                segments, n_writable = _episode_frames(
                    ep, tool_offset, gap_threshold_s, min_seg_frames, camera, crop_frac, gripper_action, gripper_binary_th
                )
                break
            except Exception as e:
                msg = str(e)
                transient = isinstance(e, OSError) or any(
                    s in msg for s in ("Input/output error", "file read failed", "errno = 5")
                )
                if transient and attempt < 3:
                    print(f"[{i + 1}/{len(episodes)}] retry {attempt + 1}/3 {ep.parent.name}/{ep.name}: EIO")
                    time.sleep(2.0 * (attempt + 1))
                    continue
                skipped.append((str(ep), repr(e)))
                print(f"[{i + 1}/{len(episodes)}] SKIP({which}) {ep.parent.name}/{ep.name}: {e}")
                segments = None
                break
        if segments is None:
            continue
        if not segments:
            skipped.append((str(ep), "no clean segment >= min_seg_frames"))
            print(f"[{i + 1}/{len(episodes)}] SKIP({which}) {ep.parent.name}/{ep.name}: all-gap/too-short")
            continue
        ds = val_ds if which == "val" else train_ds
        # TRAIN-ONLY tail padding: the LEFT release is the terminal action, so episodes end ~16 frames
        # after it -> the "fully-open" gripper label is rare + the end-of-episode chunk is truncated, and
        # the model under-opens the left gripper (wiki robotics-lab-pickplace-eval 2026-06-22). Freeze the
        # last frame (image+state held) with a HOLD action (zero pose delta; gripper stays open) and append
        # `tail_pad_frames` copies to the FINAL segment so the open target is no longer rare. Val is left
        # UNPADDED (honest test). Only the last segment (true episode end), not gap-split boundaries.
        n_pad_added = 0
        if tail_pad_frames > 0 and which == "train" and segments:
            hf = _make_hold_frame(segments[-1][-1], gripper_action)
            segments[-1].extend(dict(hf) for _ in range(tail_pad_frames))
            n_pad_added = tail_pad_frames
        kept = 0
        for seg in segments:
            for fr in seg:
                ds.add_frame(fr)
            ds.save_episode()
            kept += len(seg)
        counts[which][0] += len(segments)
        counts[which][1] += kept
        drop = n_writable - (kept - n_pad_added)
        dropped_frames += drop
        if len(segments) > 1 or drop > 0:
            n_src_with_gaps += 1
        split_log[which].append(f"{ep.parent.name}/{ep.name}")
        tag = f"{len(segments)} seg" + (f", -{drop}fr gap/stub" if drop else "")
        print(f"[{i + 1}/{len(episodes)}] {which} {ep.parent.name}/{ep.name}: {kept} frames ({tag})")

    split_record.parent.mkdir(parents=True, exist_ok=True)
    split_record.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_frac": val_frac,
                "camera": camera,
                "split_in": str(split_in) if split_in is not None else None,
                "gripper_action": gripper_action,
                "gripper_binary_th": gripper_binary_th if gripper_action == "binary" else None,
                "val_from_record": str(val_from_record) if val_from_record is not None else None,
                "exclude_path_substr": exclude_path_substr,
                "fisheye_crop_frac": fisheye_crop_frac if crop_frac is not None else None,
                "tail_pad_frames": tail_pad_frames,
                "img_shape": list(img_shape),
                "pose_frame": "tcp_tip" if tool_offset is not None else "raw_tracker_steamvr_world",
                "retarget_config": str(retarget_config) if tool_offset is not None else None,
                "gap_threshold_s": gap_threshold_s,
                "min_seg_frames": min_seg_frames,
                "dropped_frames_gap_stub": dropped_frames,
                "source_episodes_with_gaps": n_src_with_gaps,
                **split_log,
            },
            indent=2,
        )
    )
    print(
        f"DONE: train {counts['train'][0]}seg/{counts['train'][1]}fr, "
        f"val {counts['val'][0]}seg/{counts['val'][1]}fr; skipped {len(skipped)}; "
        f"gap-split: {n_src_with_gaps} src episodes affected, {dropped_frames} frames dropped; split -> {split_record}"
    )
    if skipped:
        for s, e in skipped:
            print("  SKIP", s, e)


if __name__ == "__main__":
    tyro.cli(main)
