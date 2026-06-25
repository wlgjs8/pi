"""Evaluate an openpi pi05_pika_umi(_video_*) checkpoint on a VIDEO-backed LeRobot val set.

Adapter of eval_pika_umi_val_relrel_aug.py whose GT comes from a LeRobot dataset
(state[14] + actions[14] already stored by convert_pika_umi_storage_video.py) instead of
data_tcp_v2 HDF5. Built to score the stopped tracker-frame _8020 checkpoints against the
NEW tool/TCP-frame val (plaif/pika_umi_video_val_tcp_8020) on a common yardstick.

Metric space (lerobot units, NO in-house *100 gripper rescale): per-step action MSE over
all 14 dims; gripper delta is in /100 units on BOTH pred and GT (the model's native output
space for this config). Normalizer is the val-set action std (mean of squared per-dim std),
so the scale is checkpoint-independent and identical for the tracker-now and tcp-later evals.

Frame caveat: a tracker-frame checkpoint scored on a tcp-frame val carries the constant
tracker->tip offset (~5 cm / 180deg yaw) inside the MSE -- this is the intended "unfair"
baseline, dominated by the frame gap, not model quality.

Run inside the openpi venv:
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python eval_pika_umi_val_tcp_lerobot.py \
    --checkpoint-step 30000 \
    --val-repo-id plaif/pika_umi_video_val_tcp_8020 \
    --config pi05_pika_umi_video_8020 \
    --ckpt-base /home/plaif/workspace/openpi_runs/checkpoints/pi05_pika_umi_video_8020/video_noaug_8to2_h8_40k \
    --tag tracker_on_tcpval
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROBOTICS_LAB = pathlib.Path("/home/plaif/workspace/robotics_lab")
LEROBOT_HOME = pathlib.Path("/mnt/pika/lerobot")
OUT_DIR = pathlib.Path("/home/plaif/workspace/openpi_runs/eval_val_tcp_lerobot")
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt with the "
    "left arm and put it in the left box"
)
HORIZON = 8
GRIP_DIMS = (6, 13)


def _load_phase_segmentation():
    path = ROBOTICS_LAB / "policy_runner/policy_runner/phase_segmentation.py"
    spec = importlib.util.spec_from_file_location("phase_segmentation", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_segmentation"] = mod
    spec.loader.exec_module(mod)
    return mod


def _video_frames_rgb(mp4: pathlib.Path) -> np.ndarray:
    """Decode a whole episode mp4 -> (T,H,W,3) uint8 RGB (cv2 returns BGR)."""
    cap = cv2.VideoCapture(str(mp4))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise ValueError(f"no frames decoded from {mp4}")
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--val-repo-id", type=str, default="plaif/pika_umi_video_val_tcp_8020")
    parser.add_argument("--config", type=str, default="pi05_pika_umi_video_8020",
                        help="openpi config name to load the checkpoint with (defines transforms + norm stats)")
    parser.add_argument("--ckpt-base", type=str, required=True,
                        help="run checkpoint dir holding step subdirs, e.g. .../video_noaug_8to2_h8_40k")
    parser.add_argument("--lerobot-home", type=str, default=str(LEROBOT_HOME))
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit number of val episodes (debug)")
    parser.add_argument("--n-select", type=int, default=1,
                        help="if >1, wrap in MedoidPolicy: sample N chunks/frame and use the consensus "
                             "(medoid) chunk for all metrics — measures performance WITH best-of-N selection")
    parser.add_argument("--gripper-action", type=str, default="delta", choices=["delta", "absolute"],
                        help="gripper action representation: 'delta' (free-run integrates pred*stride) or "
                             "'absolute' (pred IS the opening /100; reconstruct directly, no integration)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="action horizon; default = the config model's action_horizon")
    args = parser.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ckpt_dir = pathlib.Path(args.ckpt_base) / str(args.checkpoint_step)
    if not ckpt_dir.exists():
        raise FileNotFoundError(ckpt_dir)

    phase_mod = _load_phase_segmentation()
    root = pathlib.Path(args.lerobot_home) / args.val_repo_id
    ds = LeRobotDataset(args.val_repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)      # (N,14) reset-relative proprio
    actions_all = np.stack(hf["actions"]).astype(np.float64)   # (N,14) ee_local deltas, grip /100
    ep_from = list(ds.episode_data_index["from"])
    ep_to = list(ds.episode_data_index["to"])
    n_ep = len(ep_from)
    if args.limit:
        n_ep = min(n_ep, args.limit)

    # Fixed normalizer from the val GT actions (checkpoint-independent -> comparable across runs).
    action_std = actions_all.std(axis=0)
    scale = float(np.mean(np.square(np.maximum(action_std, 1e-12))))
    # POSE-only (12-dim) scale: exclude the two gripper dims (6,13). Because the gripper representation
    # (delta vs absolute) changes the gripper-dim variance and thus the 14-dim normalizer, the full
    # normalized MSE is NOT comparable across gripper reps. Pose-only error normalized by pose-only std
    # isolates trajectory accuracy and IS comparable across gripper reps and cameras.
    POSE_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    pose_scale = float(np.mean(np.square(np.maximum(action_std[POSE_DIMS], 1e-12))))

    # RGB-D: feed the realsense depth streams when the dataset has them (matches include_depth configs).
    has_depth = (root / "videos/chunk-000/left_wrist_0_depth").exists()
    if has_depth:
        print("RGB-D val: feeding *_wrist_0_depth alongside RGB")

    cfg = _config.get_config(args.config)
    HORIZON = int(args.horizon) if args.horizon else int(cfg.model.action_horizon)
    print(f"horizon={HORIZON} | gripper_action={args.gripper_action} | config={args.config}")
    policy = _policy_config.create_trained_policy(cfg, str(ckpt_dir))
    if args.n_select > 1:
        from openpi.policies.policy import MedoidPolicy
        policy = MedoidPolicy(policy, num_samples=args.n_select)
        print(f"selection: medoid-of-{args.n_select} (consensus best-of-N) — all metrics use the selected chunk")

    sq_first, n_first = 0.0, 0.0
    pose_sq, pose_n = 0.0, 0.0  # first-step POSE-only (12-dim, gripper excluded) MSE
    perdim_sq = np.zeros(14); perdim_n = 0  # per-dimension first-step squared error (raw units^2)
    sq_chunk, n_chunk = 0.0, 0.0
    chunk_pos_sq = [0.0] * HORIZON  # normalized action MSE per chunk position (step 0..H-1)
    chunk_pos_n = [0] * HORIZON
    phase_sq = {p: 0.0 for p in phase_mod.PHASE_NAMES}
    phase_n = {p: 0.0 for p in phase_mod.PHASE_NAMES}
    GRASP = {"right": (7, 8, 9, "b1"), "left": (0, 1, 2, "b3")}
    grasp_err = {arm: {ax: [] for ax in "xyz"} for arm in ("right", "left")}
    intz = {"right": [], "left": []}
    # gripper open/close TIMING (open-loop free-run) + gripper-delta MSE
    grip_dmse_sq, grip_dmse_n = 0.0, 0.0
    grip_timing = {ev: [] for ev in phase_mod.EVENT_NAMES}     # |pred_event - gt_event| in ms
    grip_detect = {ev: [0, 0] for ev in phase_mod.EVENT_NAMES}  # [matched, total]
    # arm -> (action gripper dim, close-event name, open-event name); GRIP_DIMS=(left=6, right=13)
    GRIP_ARM = {"right": (GRIP_DIMS[1], "right_close", "right_open"),
                "left": (GRIP_DIMS[0], "left_close", "left_open")}
    infer_times = []
    frame_count = 0

    for ei in range(n_ep):
        a, b = int(ep_from[ei]), int(ep_to[ei])
        states = states_all[a:b]
        # full-episode per-frame actions: GT[t] is the ee_local delta from t->t+1 (last frame has none)
        gt = actions_all[a:b - 1]  # (T-1, 14)
        length = gt.shape[0]
        # Obs gripper opening for phase events lives at state dims 6/13 in the 14-D pose proprio.
        # The velocity-only proprio (state_mode=velocity) is 12-D and carries NO gripper -> zero it,
        # which makes phase boundaries "not clean" and skips the gripper-timing/grasp/phase-MSE blocks.
        # The headline action MSE (first-step / pose-only / per-chunk) is independent of obs gripper.
        if states.shape[1] > max(GRIP_DIMS):
            left_grip = states[:, GRIP_DIMS[0]] * 100.0   # absolute % for phase events
            right_grip = states[:, GRIP_DIMS[1]] * 100.0
        else:
            left_grip = np.zeros(states.shape[0]); right_grip = np.zeros(states.shape[0])
        bounds = phase_mod.extract_phase_boundaries(left_grip, right_grip, length)

        left_mp4 = root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4"
        right_mp4 = root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4"
        left_img = _video_frames_rgb(left_mp4)
        right_img = _video_frames_rgb(right_mp4)
        if has_depth:
            left_depth = _video_frames_rgb(root / "videos/chunk-000/left_wrist_0_depth" / f"episode_{ei:06d}.mp4")
            right_depth = _video_frames_rgb(root / "videos/chunk-000/right_wrist_0_depth" / f"episode_{ei:06d}.mp4")

        ep_dz = {"right": 0.0, "left": 0.0}
        sf = []  # stride frame indices visited this segment
        pdg = {"left": [], "right": []}  # predicted first-step gripper delta (/100 units) per arm
        for t in range(0, length - HORIZON + 1, args.stride):
            obs = {
                "observation/left_wrist_0_rgb": left_img[t],
                "observation/right_wrist_0_rgb": right_img[t],
                "observation/state": states[t].astype(np.float32),
                "prompt": PROMPT,
            }
            if has_depth:
                obs["observation/left_wrist_0_depth"] = left_depth[t]
                obs["observation/right_wrist_0_depth"] = right_depth[t]
            t0 = time.perf_counter()
            pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
            infer_times.append(time.perf_counter() - t0)
            h = min(HORIZON, pred.shape[0], length - t)
            target = gt[t : t + h]
            err2 = (pred[:h] - target) ** 2
            sq_chunk += float(err2.sum()); n_chunk += err2.size
            for k in range(h):
                chunk_pos_sq[k] += float(err2[k].sum()); chunk_pos_n[k] += err2[k].size
            first = err2[0]
            perdim_sq += err2[0]; perdim_n += 1   # per-dim first-step squared error
            sq_first += float(first.sum()); n_first += first.size
            pose_sq += float(first[POSE_DIMS].sum()); pose_n += len(POSE_DIMS)
            phase = bounds.phase_for_frame(t)
            phase_sq[phase] += float(first.sum()); phase_n[phase] += first.size
            if phase == "right_pick":
                ep_dz["right"] += float(pred[0][9] - gt[t][9])
            elif phase == "left_pick":
                ep_dz["left"] += float(pred[0][2] - gt[t][2])
            sf.append(t)
            pdg["left"].append(float(pred[0][GRIP_DIMS[0]]))
            pdg["right"].append(float(pred[0][GRIP_DIMS[1]]))
            grip_dmse_sq += float(first[GRIP_DIMS[0]] + first[GRIP_DIMS[1]]); grip_dmse_n += 2
            frame_count += 1

        if bounds.clean:
            intz["right"].append(abs(ep_dz["right"]) * args.stride * 1000.0)
            intz["left"].append(abs(ep_dz["left"]) * args.stride * 1000.0)
            for arm, (xi, yi, zi, bkey) in GRASP.items():
                ef = int(getattr(bounds, bkey))
                if not (0 <= ef < length):
                    continue
                gobs = {
                    "observation/left_wrist_0_rgb": left_img[ef],
                    "observation/right_wrist_0_rgb": right_img[ef],
                    "observation/state": states[ef].astype(np.float32),
                    "prompt": PROMPT,
                }
                if has_depth:
                    gobs["observation/left_wrist_0_depth"] = left_depth[ef]
                    gobs["observation/right_wrist_0_depth"] = right_depth[ef]
                gp = np.asarray(policy.infer(gobs)["actions"], dtype=np.float64)[0]
                for ax, idx in zip("xyz", (xi, yi, zi)):
                    grasp_err[arm][ax].append(abs(gp[idx] - gt[ef][idx]) * 1000.0)
            # gripper open/close TIMING: open-loop free-run the predicted gripper from the GT segment
            # start (integrate pred delta * stride), detect close/open with GT-range hysteresis,
            # compare the event frame to the GT event frame (ms).
            gt_grip = {"left": left_grip, "right": right_grip}
            evf = bounds.event_frames()
            for arm, (gdim, ev_close, ev_open) in GRIP_ARM.items():
                th = phase_mod.gripper_thresholds(gt_grip[arm])
                if th is None or len(sf) < 2:
                    continue
                if args.gripper_action == "absolute":
                    # predicted gripper IS the absolute opening (/100) -> reconstruct directly, no integration
                    fr = [min(100.0, max(0.0, pdg[arm][k] * 100.0)) for k in range(len(sf))]
                else:
                    fr = [float(gt_grip[arm][sf[0]])]
                    for k in range(1, len(sf)):
                        fr.append(min(100.0, max(0.0, fr[-1] + pdg[arm][k - 1] * 100.0 * args.stride)))
                ev = phase_mod.gripper_events(np.asarray(fr), thresholds=th)
                pred_frames = {"close": [sf[k] for k in ev["close"]], "open": [sf[k] for k in ev["open"]]}
                for ev_name, direction in ((ev_close, "close"), (ev_open, "open")):
                    gf = evf[ev_name]
                    if not (0 <= gf < length):
                        continue
                    grip_detect[ev_name][1] += 1
                    cands = pred_frames[direction]
                    if cands:
                        pf = min(cands, key=lambda x: abs(x - gf))
                        grip_timing[ev_name].append(abs(pf - gf) * 1000.0 / 30.0)
                        grip_detect[ev_name][0] += 1
        print(f"[{ei + 1}/{n_ep}] episode_{ei:06d} done (frames so far {frame_count})", flush=True)

    result = {
        "checkpoint_step": args.checkpoint_step,
        "config": args.config,
        "gripper_action": args.gripper_action,
        "ckpt_base": args.ckpt_base,
        "val_repo_id": args.val_repo_id,
        "n_select": args.n_select,
        "selection": ("single_draw" if args.n_select <= 1 else f"medoid_of_{args.n_select}"),
        "stride": args.stride,
        "horizon": HORIZON,
        "episodes": n_ep,
        "eval_frames": frame_count,
        "sampling": "single stochastic draw per frame (openpi default)",
        "normalization_scale_source": "val-set action std (mean of squared per-dim std)",
        "normalization_scale": scale,
        "units_note": "lerobot units throughout; gripper delta in /100 on pred AND gt (no in-house *100)",
        "first_step": {
            "action_mse": sq_first / max(n_first, 1.0),
            "normalized_action_mse": (sq_first / max(n_first, 1.0)) / scale,
        },
        "per_axis_first_step": {
            # Per-dimension first-step error. RMSE in physical units (translation mm, rotation deg);
            # normalized = per-dim MSE / per-dim GT variance (1.0 == mean-predictor baseline, the worst
            # meaningful value). gripper dims (6,13) reported in /100 RMSE.
            arm: {
                ax: {
                    "rmse": (
                        float(np.sqrt(perdim_sq[di] / max(perdim_n, 1)) * (1000.0 if ax in ("x", "y", "z")
                                else (180.0 / np.pi) if ax in ("rx", "ry", "rz") else 1.0)),
                        ("mm" if ax in ("x", "y", "z") else "deg" if ax in ("rx", "ry", "rz") else "/100")
                    )[0],
                    "unit": ("mm" if ax in ("x", "y", "z") else "deg" if ax in ("rx", "ry", "rz") else "frac"),
                    "normalized_mse": float((perdim_sq[di] / max(perdim_n, 1)) / max(action_std[di] ** 2, 1e-24)),
                }
                for ax, di in axmap.items()
            }
            for arm, axmap in {
                "left":  {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5, "grip": 6},
                "right": {"x": 7, "y": 8, "z": 9, "rx": 10, "ry": 11, "rz": 12, "grip": 13},
            }.items()
        },
        "per_axis_note": "first-step per-dim error; rmse is per 30Hz step (translation mm, rotation deg); "
                         "normalized_mse=MSE/var, 1.0==predicting the mean (worst meaningful)",
        "first_step_pose_only_12dim": {
            "action_mse": pose_sq / max(pose_n, 1.0),
            "normalized_action_mse": (pose_sq / max(pose_n, 1.0)) / pose_scale,
            "pose_scale": pose_scale,
            "note": "12 pose dims (gripper 6,13 excluded), normalized by pose-only val action std -> "
                    "comparable across gripper reps (delta/absolute) and cameras",
        },
        "chunk8": {
            "action_mse": sq_chunk / max(n_chunk, 1.0),
            "normalized_action_mse": (sq_chunk / max(n_chunk, 1.0)) / scale,
        },
        "chunk_by_position_normalized": [
            (chunk_pos_sq[k] / max(chunk_pos_n[k], 1.0)) / scale for k in range(HORIZON)
        ],
        "chunk_by_position_note": (
            "normalized action MSE at each predicted chunk step 0..H-1 (step 0 == first_step); "
            "evaluated over all stride frames of every val episode; rising across positions = "
            "the chunk degrades further into the horizon"
        ),
        "first_step_by_phase_normalized": {
            p: (phase_sq[p] / max(phase_n[p], 1.0)) / scale for p in phase_mod.PHASE_NAMES
        },
        "grasp_instant_error_mm": {
            arm: {
                ax: {
                    "mean": (float(np.mean(grasp_err[arm][ax])) if grasp_err[arm][ax] else None),
                    "median": (float(np.median(grasp_err[arm][ax])) if grasp_err[arm][ax] else None),
                    "n": len(grasp_err[arm][ax]),
                }
                for ax in "xyz"
            }
            for arm in ("right", "left")
        },
        "integrated_z_drift_mm": {
            arm: {
                "mean": (float(np.mean(intz[arm])) if intz[arm] else None),
                "median": (float(np.median(intz[arm])) if intz[arm] else None),
                "n": len(intz[arm]),
            }
            for arm in ("right", "left")
        },
        "gripper_event_timing_ms_freerun": {
            ev: {
                "median": (float(np.median(grip_timing[ev])) if grip_timing[ev] else None),
                "mean": (float(np.mean(grip_timing[ev])) if grip_timing[ev] else None),
                "detect_rate": (grip_detect[ev][0] / grip_detect[ev][1] if grip_detect[ev][1] else None),
                "n_matched": grip_detect[ev][0],
                "n_total": grip_detect[ev][1],
            }
            for ev in phase_mod.EVENT_NAMES
        },
        "gripper_timing_note": (
            "open-loop free-run: integrate predicted gripper delta from the GT segment start "
            "(delta*stride), detect close/open via GT-range hysteresis, |pred_frame - GT_event_frame| "
            "in ms over clean-segmented episodes; detect_rate = fraction of GT events the free-run reproduced"
        ),
        "gripper_delta_mse": grip_dmse_sq / max(grip_dmse_n, 1.0),
        "infer_latency_ms": {
            "median": float(np.median(infer_times) * 1000.0) if infer_times else None,
            "p95": float(np.percentile(infer_times, 95) * 1000.0) if infer_times else None,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = OUT_DIR / f"step{args.checkpoint_step}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
