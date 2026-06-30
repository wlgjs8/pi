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
from scipy.spatial.transform import Rotation


def _integrate_local(deltas: np.ndarray):
    """Integrate ee_local per-step deltas (H,6 = [pos_delta_local3, rot_delta3]) into the absolute
    displacement-FROM-ANCHOR at each horizon step. pose[k] = compose of deltas 0..k-1 (k=0 -> anchor=0).
    Returns (H,3) positions and an (H,) Rotation. This is what the DELTA deploy path accumulates -> its
    per-step errors compound here, exactly the drift we want to measure."""
    H = deltas.shape[0]
    pos = np.zeros((H, 3)); rotvecs = np.zeros((H, 3))
    cp = np.zeros(3); cr = Rotation.identity()
    for k in range(H):
        pos[k] = cp; rotvecs[k] = cr.as_rotvec()
        cp = cp + cr.apply(deltas[k, :3])
        cr = cr * Rotation.from_rotvec(deltas[k, 3:6])
    return pos, Rotation.from_rotvec(rotvecs)


def _displacement_from_anchor(chunk: np.ndarray, anchored: bool, arms=(("left", 0), ("right", 7))):
    """Per-arm absolute displacement-from-anchor at each horizon step k, comparable across delta/anchored.
    delta -> integrate the per-step deltas (errors compound); anchored -> the row IS T_t^-1 T_{t+k} already.
    Returns {arm: (positions (H,3), Rotation length H)}. `arms` = (name, base-col) per arm (single-arm right
    passes [("right", 0)])."""
    out = {}
    for arm, base in arms:
        if anchored:
            pos = chunk[:, base : base + 3]
            rot = Rotation.from_rotvec(chunk[:, base + 3 : base + 6])
        else:
            pos, rot = _integrate_local(chunk[:, base : base + 6])
        out[arm] = (pos, rot)
    return out


ROBOTICS_LAB = pathlib.Path("/home/plaif/workspace/robotics_lab")
LEROBOT_HOME = pathlib.Path("/mnt/pika/lerobot")
OUT_DIR = pathlib.Path("/home/plaif/workspace/openpi_runs/eval_val_tcp_lerobot")
PROMPT = (
    "pick up the black bolt with the right arm and put it in the right box, then pick up the gray bolt with the "
    "left arm and put it in the left box"
)
HORIZON = 8
GRIP_DIMS = (6, 13)


def _swap_prompt(s: str) -> str:
    """COLOR-SWAP a phase_color prompt to test language sensitivity (wiki finding B re-test): swap the
    bolt color gray<->black AND the coordinated box gray<->green. If the policy READS the color, its
    output changes under this swap; if it ignores language, the output is ~unchanged."""
    s = s.replace("gray bolt", "\x00").replace("black bolt", "gray bolt").replace("\x00", "black bolt")
    s = s.replace("gray box", "\x01").replace("green box", "gray box").replace("\x01", "green box")
    return s


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
    parser.add_argument("--action-mode", type=str, default="delta", choices=["delta", "anchored"],
                        help="MUST match the dataset's --action-mode. 'delta' = stored per-step ee_local "
                             "deltas (chunk compared as-is). 'anchored' = dataset stores per-frame ABSOLUTE "
                             "poses; GT chunk is re-anchored to its first frame (T_t^-1 T_{t+k}) to match the "
                             "model output. anchored row 0 is structurally identity -> first-step metrics use "
                             "row 1 (= T_t^-1 T_{t+1}, comparable to a delta run's first step).")
    parser.add_argument("--prompt-swap", action="store_true",
                        help="LANGUAGE-SENSITIVITY test (wiki finding B re-test): per frame, also infer with "
                             "a COLOR-SWAPPED prompt (gray<->black bolt + gray<->green box) and report the "
                             "first-step divergence |pred_correct - pred_swapped| (normalized). High = the "
                             "policy READS the prompt color; ~0 = it ignores language.")
    parser.add_argument("--swap-noise-floor", action="store_true",
                        help="CONTROL for --prompt-swap: re-infer with the SAME (unswapped) prompt, so the "
                             "'divergence' is pure draw-to-draw stochastic noise of the flow head. Compare "
                             "the real swap divergence against this floor; swap≈floor => the swap metric is "
                             "just sampling noise, not a language effect.")
    parser.add_argument("--grounding-probe", type=int, default=0, metavar="K",
                        help="PROPER color-grounding probe (beats the noisy swap metric). For each frame, draw "
                             "K action chunks under the CORRECT per-frame prompt and K under the COLOR-SWAPPED "
                             "prompt, and measure each draw's CHUNK nMSE vs the GROUND-TRUTH demo chunk. Report "
                             "delta = nMSE(swapped) - nMSE(correct), PAIRED per frame (scene difficulty cancels) "
                             "and averaged over K draws (kills flow-head sampling noise). delta>0 => the correct "
                             "color prompt fits the demo better than the swapped one => the policy USES the color "
                             "word. Compare colorprompt vs a single-prompt baseline. K~8 recommended.")
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

    # Per-frame prompt: phase_color writes the per-phase COLOR prompt into each frame's task; legacy
    # datasets have a single task (== the old fixed PROMPT). Use the dataset task so the eval prompt
    # MATCHES training (in-distribution) -- the hardcoded PROMPT would be OOD for a phase_color model.
    _tmap = {int(k): v for k, v in ds.meta.tasks.items()}
    tasks_all = [_tmap.get(int(i), PROMPT) for i in np.asarray(hf["task_index"])]

    # Fixed normalizer from the val GT actions (checkpoint-independent -> comparable across runs).
    action_std = actions_all.std(axis=0)
    scale = float(np.mean(np.square(np.maximum(action_std, 1e-12))))
    # POSE-only (12-dim) scale: exclude the two gripper dims (6,13). Because the gripper representation
    # (delta vs absolute) changes the gripper-dim variance and thus the 14-dim normalizer, the full
    # normalized MSE is NOT comparable across gripper reps. Pose-only error normalized by pose-only std
    # isolates trajectory accuracy and IS comparable across gripper reps and cameras.
    # SINGLE-ARM (right, 7-D action [pos3,rot3,grip]) vs DUAL (14-D). Adapt the dim-dependent constants.
    SINGLE = actions_all.shape[1] == 7
    if SINGLE:
        POSE_DIMS = [0, 1, 2, 3, 4, 5]
        GRIPS = (6,)
        ARMS = [("right", 0)]
    else:
        POSE_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        GRIPS = (6, 13)
        ARMS = [("left", 0), ("right", 7)]
    pose_scale = float(np.mean(np.square(np.maximum(action_std[POSE_DIMS], 1e-12))))

    # RGB-D: feed the realsense depth streams when the dataset has them (matches include_depth configs).
    has_depth = (root / "videos/chunk-000/left_wrist_0_depth").exists()
    if has_depth:
        print("RGB-D val: feeding *_wrist_0_depth alongside RGB")

    cfg = _config.get_config(args.config)
    HORIZON = int(args.horizon) if args.horizon else int(cfg.model.action_horizon)

    ANCHORED = args.action_mode == "anchored"
    FI = 1 if ANCHORED else 0  # first-step row index (anchored row 0 == identity, so use row 1)
    if ANCHORED:
        from openpi.policies.pika_umi_policy import _anchor_relative_chunk
        # Re-derive the normalizer over the ANCHORED targets (the stored `actions` are abs poses, whose
        # raw std is the world-frame magnitude -> meaningless). Per-dim std over every window's anchored
        # rows -> the typical anchored-displacement scale; checkpoint-independent, comparable across runs.
        _sm = np.zeros(14); _sq = np.zeros(14); _nn = 0
        for ei in range(n_ep):
            a, b = int(ep_from[ei]), int(ep_to[ei]); g = actions_all[a : b - 1]
            for t in range(0, g.shape[0] - HORIZON + 1, args.stride):
                ch = _anchor_relative_chunk(g[t : t + HORIZON]).astype(np.float64)
                _sm += ch.sum(0); _sq += (ch ** 2).sum(0); _nn += ch.shape[0]
        _mean = _sm / max(_nn, 1)
        action_std = np.sqrt(np.maximum(_sq / max(_nn, 1) - _mean ** 2, 1e-24))
        scale = float(np.mean(np.square(np.maximum(action_std, 1e-12))))
        pose_scale = float(np.mean(np.square(np.maximum(action_std[POSE_DIMS], 1e-12))))
    print(f"horizon={HORIZON} | gripper_action={args.gripper_action} | action_mode={args.action_mode} | config={args.config}")
    policy = _policy_config.create_trained_policy(cfg, str(ckpt_dir))
    if args.n_select > 1:
        from openpi.policies.policy import MedoidPolicy
        policy = MedoidPolicy(policy, num_samples=args.n_select)
        print(f"selection: medoid-of-{args.n_select} (consensus best-of-N) — all metrics use the selected chunk")

    sq_first, n_first = 0.0, 0.0
    pose_sq, pose_n = 0.0, 0.0  # first-step POSE-only (12-dim, gripper excluded) MSE
    perdim_sq = np.zeros(actions_all.shape[1]); perdim_n = 0  # per-dim first-step squared error (raw units^2)
    sq_chunk, n_chunk = 0.0, 0.0
    # CUMULATIVE displacement-from-anchor error at each horizon step (mm / deg), per arm. The fair
    # drift A/B: delta integrates per-step deltas (errors compound) vs anchored predicts T_t^-1 T_{t+k}
    # directly. Normalizer-free; comparable across action_mode at the overlapping horizon range.
    cum_terr = {arm: np.zeros(HORIZON) for arm, _ in ARMS}
    cum_rerr = {arm: np.zeros(HORIZON) for arm, _ in ARMS}
    cum_n = 0
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
    swap_div_sq = swap_div_pose_sq = 0.0; swap_div_n = 0  # --prompt-swap: |pred - pred_colorswapped| first-step
    gp_corr_sq = gp_swap_sq = gp_corr_pose_sq = gp_swap_pose_sq = 0.0  # --grounding-probe: GT-ref chunk SSE
    gp_h_sum = gp_n = 0
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
                "prompt": tasks_all[a + t],  # per-frame task = the phase_color prompt the model trained on
            }
            if has_depth:
                obs["observation/left_wrist_0_depth"] = left_depth[t]
                obs["observation/right_wrist_0_depth"] = right_depth[t]
            t0 = time.perf_counter()
            pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)
            infer_times.append(time.perf_counter() - t0)
            if args.prompt_swap:  # language-sensitivity: re-infer with the color-swapped prompt
                # noise-floor control: keep the SAME prompt -> measures draw-to-draw stochastic variance
                # (the flow head is stochastic) so we can tell how much of the swap divergence is real.
                sw = dict(obs); sw["prompt"] = obs["prompt"] if args.swap_noise_floor else _swap_prompt(obs["prompt"])
                pred_sw = np.asarray(policy.infer(sw)["actions"], dtype=np.float64)
                swap_div_sq += float(((pred[0] - pred_sw[0]) ** 2).sum()); swap_div_n += 1
                swap_div_pose_sq += float(((pred[0] - pred_sw[0])[POSE_DIMS] ** 2).sum())
            h = min(HORIZON, pred.shape[0], length - t)
            # anchored: build the GT chunk relative to the window's first frame (matches the model output);
            # delta: the stored per-step deltas are already the chunk.
            target = _anchor_relative_chunk(gt[t : t + HORIZON])[:h] if ANCHORED else gt[t : t + h]
            err2 = (pred[:h] - target) ** 2
            if args.grounding_probe:  # GT-referenced, multi-draw, PAIRED color-grounding probe
                sw_g = dict(obs)
                sw_g["prompt"] = obs["prompt"] if args.swap_noise_floor else _swap_prompt(obs["prompt"])
                cs = ss = csp = ssp = 0.0
                for _ in range(args.grounding_probe):
                    pc = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)[:h]
                    ps = np.asarray(policy.infer(sw_g)["actions"], dtype=np.float64)[:h]
                    cs += float(((pc - target) ** 2).sum()); ss += float(((ps - target) ** 2).sum())
                    csp += float(((pc - target)[:, POSE_DIMS] ** 2).sum())
                    ssp += float(((ps - target)[:, POSE_DIMS] ** 2).sum())
                K = args.grounding_probe
                gp_corr_sq += cs / K; gp_swap_sq += ss / K
                gp_corr_pose_sq += csp / K; gp_swap_pose_sq += ssp / K
                gp_h_sum += h; gp_n += 1
            # cumulative displacement-from-anchor error (mm/deg) at each horizon step k (only full chunks)
            if h == HORIZON:
                pe = _displacement_from_anchor(pred[:HORIZON], ANCHORED, ARMS)
                ge = _displacement_from_anchor(target[:HORIZON], ANCHORED, ARMS)
                for arm, _ in ARMS:
                    cum_terr[arm] += np.linalg.norm(pe[arm][0] - ge[arm][0], axis=1) * 1000.0
                    cum_rerr[arm] += (pe[arm][1] * ge[arm][1].inv()).magnitude() * (180.0 / np.pi)
                cum_n += 1
            sq_chunk += float(err2.sum()); n_chunk += err2.size
            for k in range(h):
                chunk_pos_sq[k] += float(err2[k].sum()); chunk_pos_n[k] += err2[k].size
            fi = min(FI, h - 1)  # first MEANINGFUL row (anchored row 0 is identity -> use row 1)
            first = err2[fi]
            perdim_sq += err2[fi]; perdim_n += 1   # per-dim first-step squared error
            sq_first += float(first.sum()); n_first += first.size
            pose_sq += float(first[POSE_DIMS].sum()); pose_n += len(POSE_DIMS)
            phase = bounds.phase_for_frame(t)
            phase_sq[phase] += float(first.sum()); phase_n[phase] += first.size
            if not SINGLE:  # phase-z-drift + dual-arm gripper bookkeeping (dual-arm dims only)
                if phase == "right_pick":
                    ep_dz["right"] += float(pred[fi][9] - target[fi][9])
                elif phase == "left_pick":
                    ep_dz["left"] += float(pred[fi][2] - target[fi][2])
                pdg["left"].append(float(pred[fi][GRIP_DIMS[0]]))
                pdg["right"].append(float(pred[fi][GRIP_DIMS[1]]))
                grip_dmse_sq += float(first[GRIP_DIMS[0]] + first[GRIP_DIMS[1]]); grip_dmse_n += 2
            sf.append(t)
            frame_count += 1

        if bounds.clean:
            intz["right"].append(abs(ep_dz["right"]) * args.stride * 1000.0)
            intz["left"].append(abs(ep_dz["left"]) * args.stride * 1000.0)
            # grasp-instant-error uses the row-0 action vs the per-frame GT; for anchored, row 0 is
            # identity and the stored GT is an absolute pose -> not comparable, so skip (N/A). The
            # headline per-axis / per-chunk / pose-only nMSE carry the A/B signal.
            for arm, (xi, yi, zi, bkey) in (GRASP.items() if not ANCHORED else ()):
                ef = int(getattr(bounds, bkey))
                if not (0 <= ef < length):
                    continue
                gobs = {
                    "observation/left_wrist_0_rgb": left_img[ef],
                    "observation/right_wrist_0_rgb": right_img[ef],
                    "observation/state": states[ef].astype(np.float32),
                    "prompt": tasks_all[a + ef],
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
        "action_mode": args.action_mode,
        "first_step_row": FI,  # anchored: 1 (row 0 is identity); delta: 0
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
        "prompt_swap_sensitivity": (
            {
                "normalized_full": (swap_div_sq / max(swap_div_n, 1)) / scale,
                "normalized_pose_only": (swap_div_pose_sq / max(swap_div_n, 1)) / pose_scale,
                "n": swap_div_n,
                "note": "first-step |pred - pred_colorswapped|^2 normalized by val action var. ~0 => the "
                        "policy IGNORES the prompt color (wiki finding B); >0 (esp. pose dims) => it READS it.",
            }
            if args.prompt_swap else None
        ),
        "grounding_probe": (
            {
                "K_draws": args.grounding_probe,
                "n_frames": gp_n,
                "swap_was_identity_noisefloor": bool(args.swap_noise_floor),
                "nmse_correct_pose": (gp_corr_pose_sq / max(gp_h_sum * len(POSE_DIMS), 1)) / pose_scale,
                "nmse_swapped_pose": (gp_swap_pose_sq / max(gp_h_sum * len(POSE_DIMS), 1)) / pose_scale,
                "delta_pose": ((gp_swap_pose_sq - gp_corr_pose_sq) / max(gp_h_sum * len(POSE_DIMS), 1)) / pose_scale,
                "nmse_correct_full": (gp_corr_sq / max(gp_h_sum * actions_all.shape[1], 1)) / scale,
                "nmse_swapped_full": (gp_swap_sq / max(gp_h_sum * actions_all.shape[1], 1)) / scale,
                "delta_full": ((gp_swap_sq - gp_corr_sq) / max(gp_h_sum * actions_all.shape[1], 1)) / scale,
                "note": "GT-referenced CHUNK nMSE; K draws averaged per prompt; PAIRED per frame (scene cancels). "
                        "delta = swapped - correct; >0 => the correct color prompt fits the demo better than the "
                        "color-swapped one => policy USES the color word. A single-prompt baseline that ignores "
                        "color should give delta~0. Compare colorprompt vs baseline delta_pose.",
            }
            if args.grounding_probe else None
        ),
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
            for arm, axmap in ({"right": {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5, "grip": 6}}
                               if SINGLE else {
                "left":  {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5, "grip": 6},
                "right": {"x": 7, "y": 8, "z": 9, "rx": 10, "ry": 11, "rz": 12, "grip": 13},
            }).items()
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
        "cumulative_position_error": {
            # THE FAIR DRIFT A/B (normalizer-free, comparable across action_mode at overlapping horizon).
            # Absolute displacement-from-anchor error at each horizon step k: delta integrates predicted
            # per-step deltas (errors compound) vs anchored predicts T_t^-1 T_{t+k} directly. Compare A's
            # and B's translation_mm[k] / rotation_deg[k] curves over k=0..min(H_A,H_B)-1.
            "note": "mean over windows; translation mm, rotation deg; index k = displacement to frame t+k "
                    "(k=0 is the anchor, ~0). delta integrates predicted deltas; anchored is direct.",
            "windows": cum_n,
            **{
                f"{arm}_translation_mm": [float(cum_terr[arm][k] / max(cum_n, 1)) for k in range(HORIZON)]
                for arm, _ in ARMS
            },
            **{
                f"{arm}_rotation_deg": [float(cum_rerr[arm][k] / max(cum_n, 1)) for k in range(HORIZON)]
                for arm, _ in ARMS
            },
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
