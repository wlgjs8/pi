#!/usr/bin/env python3
"""Score a VA checkpoint on the SAME grasp-localisation metric the pi05 runs were scored on.

The number that decides whether the VA line continues is val right-arm z RMSE at L=5 -- the
deployment operating point, since the runner executes ~4-6 rows then replans. pi05 sits at 2.68 mm
there. The pre-registered pass mark is 4.0 mm: a 1.5x allowance, justified because the VA backbone
runs both wrist images in 3.4 ms against pi05's ~50 ms, which buys a shorter execute window, which
buys freshness, and the measured precision-vs-lookahead curve (1.04 / 2.68 / 6.78 / 20.63 mm at
L = 2 / 5 / 12 / 23) says freshness is worth more than raw single-shot accuracy.

L=12 and L=23 are structurally unavailable at H=8 -- the chunk is only 8 rows. That is intentional:
those rows never execute and, per the same curve plus a kNN ceiling, cannot be predicted anyway.

Normalisation is replicated by hand rather than reusing openpi's transform stack, because the model
consumes a single frame at an arbitrary index rather than a dataset sample. Formulas are copied from
transforms.Normalize/_normalize_quantile so train and eval cannot drift:
    normalize  : (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
    unnormalize: (x + 1) / 2 * (q99 - q01 + 1e-6) + q01
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROBOTICS_LAB = pathlib.Path("/home/plaif/workspace/robotics_lab")
ARMS = (("left", 0, "b3"), ("right", 7, "b1"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REAL_DIM = 14


def _load_phase_segmentation():
    path = ROBOTICS_LAB / "policy_runner/policy_runner/phase_segmentation.py"
    spec = importlib.util.spec_from_file_location("phase_segmentation", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase_segmentation"] = mod
    spec.loader.exec_module(mod)
    return mod


def _frames(mp4: pathlib.Path) -> np.ndarray:
    import av

    with av.open(str(mp4)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _integrate(deltas: np.ndarray, k: int) -> np.ndarray:
    cp = np.zeros(3)
    cr = Rotation.identity()
    for i in range(k):
        cp = cp + cr.apply(deltas[i, :3])
        cr = cr * Rotation.from_rotvec(deltas[i, 3:6])
    return cp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_pika_umi_wrist_velgrip_k1_h8_40k")
    ap.add_argument("--weights-dir", default="/home/plaif/dinov3_weights")
    ap.add_argument("--val-repo-id", default="plaif/pika_umi_video_val_tcp_gripabs_velgrip_k1")
    ap.add_argument("--lerobot-home", required=True)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--grasp-frame", default="gripper", choices=["gripper", "zapex"],
                    help="how the grasp instant b is defined. 'gripper' = phase_segmentation's opening-"
                         "threshold crossing -- measured to move by 5-20 frames under a mere +-10%% "
                         "threshold change, dragging the 'L=5' target by 16-33 mm of path. 'zapex' = "
                         "the deepest point of the approach (argmax of integrated local z in a window "
                         "around the gripper event): motion-defined, threshold-free. Model RANKINGS "
                         "are unaffected by this choice (same b for all), but absolute RMSE is -- "
                         "zapex is the honest floor.")
    ap.add_argument("--ckpt2", action="append",
                    help="extra checkpoint(s) to ensemble with --ckpt (prediction averaging)")
    ap.add_argument("--branch", default="flow", choices=["flow", "l2"],
                    help="for dual-head checkpoints: which branch to score")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="flow sampling temperature; see VAPolicy.sample_actions")
    ap.add_argument("--draws", type=int, default=1,
                    help=">1 averages several flow draws, mirroring the deployed medoid sampling")
    ap.add_argument("--mode-probe", type=int, default=0, metavar="K",
                    help="Multimodality probe. The task is 'pick AN ARBITRARY bolt from several', so a "
                         "policy can be RIGHT while disagreeing with the demonstrated choice -- the "
                         "grasp-localisation metric cannot see that, because by L=5 the demonstrator "
                         "has already committed. This probe evaluates the APPROACH phase instead: at "
                         "t = b-23 (and b-12) draw K chunks and score the chunk's own 8-step "
                         "displacement against the demo's next 8 steps:\n"
                         "  best_of_K  - error of the closest draw: low = the demo's mode EXISTS among "
                         "the draws even if a single draw picks another bolt\n"
                         "  spread     - pairwise std across draws: multimodality is ALIVE (>0) vs "
                         "collapsed to one mode (~0)\n"
                         "  mag_ratio  - |mean-of-draws| / |GT|: mode-AVERAGING collapses magnitude "
                         "toward the gap between bolts (the classic mean-seeking failure); ~1 = clean "
                         "commitment. An L2 head is expected to fail exactly here.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "/home/plaif")
    from openpi_client import image_tools
    from train_va_dinov3 import VAPolicy
    import openpi.training.config as _config
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ck = torch.load(args.ckpt, map_location="cpu")
    head = ck.get("head", "l2")
    H = ck["horizon"]
    model = VAPolicy(ck["size"], pathlib.Path(args.weights_dir), H, head_mode=head,
                     layers=ck.get("layers", 4), split_head=ck.get("split_head", False))
    # Prefer the EMA weights when present -- that is what pi05 serves, so it is the fair comparison.
    sd = ck.get("ema") or ck["model"]
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    if ck.get("ema"): print("using EMA weights", flush=True)
    model = model.cuda().eval().to(torch.bfloat16)
    print(f"loaded {args.ckpt}  head={head} H={H} step={ck.get('step')}", flush=True)

    # --ckpt2: average the predictions of two independently trained runs. Free accuracy if the two
    # runs' errors are partly independent, and the cost is one extra backbone pass (3.4 -> ~7 ms),
    # still an order of magnitude under pi05. Both must share head/H so the outputs are commensurate.
    models = [model]
    for extra in (args.ckpt2 or []):
        ck2 = torch.load(extra, map_location="cpu")
        assert ck2.get("head", "l2") == head and ck2["horizon"] == H, "ensemble members must match"
        m2 = VAPolicy(ck2["size"], pathlib.Path(args.weights_dir), H, head_mode=head,
                      layers=ck2.get("layers", 4), split_head=ck2.get("split_head", False))
        m2.load_state_dict({k: v.float() for k, v in (ck2.get("ema") or ck2["model"]).items()})
        models.append(m2.cuda().eval().to(torch.bfloat16))
        print(f"ensemble += {extra} (step {ck2.get('step')})", flush=True)

    cfg = _config.get_config(args.config)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ns = dc.norm_stats
    a_q01, a_q99 = np.asarray(ns["actions"].q01)[:REAL_DIM], np.asarray(ns["actions"].q99)[:REAL_DIM]
    s_q01, s_q99 = np.asarray(ns["state"].q01)[:REAL_DIM], np.asarray(ns["state"].q99)[:REAL_DIM]

    phase_mod = _load_phase_segmentation()
    root = pathlib.Path(args.lerobot_home) / args.val_repo_id
    ds = LeRobotDataset(args.val_repo_id, root=root, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states_all = np.stack(hf["state"]).astype(np.float64)
    actions_all = np.stack(hf["actions"]).astype(np.float64)
    ep_from, ep_to = list(ds.episode_data_index["from"]), list(ds.episode_data_index["to"])
    n_ep = len(ep_from) if args.limit is None else min(len(ep_from), args.limit)

    LOOKAHEADS = tuple(L for L in (2, 5, 12, 23) if L <= H - 1)
    acc = {a: {L: {"p": [], "g": [], "ep": []} for L in LOOKAHEADS} for a, _, _ in ARMS}
    mode_acc = {a: {23: [], 12: []} for a, _, _ in ARMS}

    RES = int(ck.get("resolution", 224))
    def prep(img):
        x = image_tools.resize_with_pad(img, RES, RES)            # identical to training
        x = torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN)[:, None, None]
        s = torch.tensor(IMAGENET_STD)[:, None, None]
        return ((x - m) / s)[None].cuda().to(torch.bfloat16)

    for ei in range(n_ep):
        a0, b0 = int(ep_from[ei]), int(ep_to[ei])
        gt = actions_all[a0 : b0 - 1]
        st_raw = states_all[a0 : b0 - 1]
        length = gt.shape[0]
        if st_raw.shape[1] <= 13:
            continue
        bounds = phase_mod.extract_phase_boundaries(st_raw[:, 6] * 100.0, st_raw[:, 13] * 100.0, length)
        if not bounds.clean:
            continue
        li = _frames(root / "videos/chunk-000/left_wrist_0_rgb" / f"episode_{ei:06d}.mp4")
        ri = _frames(root / "videos/chunk-000/right_wrist_0_rgb" / f"episode_{ei:06d}.mp4")

        for arm, base, bkey in ARMS:
            bf = int(getattr(bounds, bkey, -1))
            if not (0 <= bf < length):
                continue
            d = gt[:, base : base + 6]
            if args.grasp_frame == "zapex":
                lo_w, hi_w = max(0, bf - 15), min(length, bf + 45)
                zcum = np.cumsum(d[:, 2])
                bf = lo_w + int(np.argmax(zcum[lo_w:hi_w]))
            for L in LOOKAHEADS:
                t = bf - L
                if t < 0 or t + H > length or t >= len(li) or t >= len(ri):
                    continue
                sn = (st_raw[t, :REAL_DIM] - s_q01) / (s_q99 - s_q01 + 1e-6) * 2.0 - 1.0
                if ck.get("no_state"):
                    sn = np.zeros_like(sn)      # must match training-time zeroing
                stt = torch.from_numpy(sn).float()[None].cuda().to(torch.bfloat16)
                outs = []
                with torch.no_grad():
                    for _ in range(args.draws):
                        for m in models:
                            o = m.sample_actions(prep(li[t]), prep(ri[t]), stt,
                                                 num_steps=args.flow_steps,
                                                 noise_scale=args.noise_scale,
                                                 branch=args.branch) if head in ("flow", "dual") \
                                else m(prep(li[t]), prep(ri[t]), stt)
                            outs.append(o.float().cpu().numpy()[0])
                pn = np.mean(outs, axis=0)
                pred = (pn + 1.0) / 2.0 * (a_q99 - a_q01 + 1e-6) + a_q01   # unnormalize
                acc[arm][L]["p"].append(_integrate(pred[:, base : base + 6], L))
                acc[arm][L]["g"].append(_integrate(d[t:bf], L))
                acc[arm][L]["ep"].append(int(ei))
        if args.mode_probe > 0:
            for arm, base, bkey in ARMS:
                bf = int(getattr(bounds, bkey, -1))
                if not (0 <= bf < length):
                    continue
                d = gt[:, base : base + 6]
                for Lm in (23, 12):
                    t = bf - Lm
                    if t < 0 or t + H > length or t >= len(li) or t >= len(ri):
                        continue
                    sn = (st_raw[t, :REAL_DIM] - s_q01) / (s_q99 - s_q01 + 1e-6) * 2.0 - 1.0
                    stt = torch.from_numpy(sn).float()[None].cuda().to(torch.bfloat16)
                    if ck.get("no_state"):
                        stt = torch.zeros_like(stt)
                    il_, ir_ = prep(li[t]), prep(ri[t])
                    disp = []
                    with torch.no_grad():
                        for _ in range(args.mode_probe):
                            o = model.sample_actions(il_, ir_, stt, num_steps=args.flow_steps,
                                                     branch=args.branch) \
                                if head in ("flow", "dual") else model(il_, ir_, stt)
                            pn = o.float().cpu().numpy()[0]
                            pr = (pn + 1.0) / 2.0 * (a_q99 - a_q01 + 1e-6) + a_q01
                            disp.append(_integrate(pr[:, base : base + 6], H - 1))
                    D = np.array(disp) * 1000.0                       # (K,3) chunk displacement mm
                    G = _integrate(d[t : t + H], H - 1) * 1000.0      # demo's own next-chunk motion
                    err = np.linalg.norm(D - G, axis=1)
                    mode_acc[arm][Lm].append({
                        "ep": int(ei),
                        "single_mm": float(err[0]),
                        "best_of_K_mm": float(err.min()),
                        "spread_mm": float(np.linalg.norm(D.std(axis=0))),
                        "mag_ratio": float(np.linalg.norm(D.mean(0)) / max(np.linalg.norm(G), 1e-9)),
                    })
        print(f"[{ei + 1}/{n_ep}] episode_{ei:06d}", flush=True)

    res = {"ckpt": args.ckpt, "head": head, "horizon": H, "step": ck.get("step"),
           "grasp_frame": args.grasp_frame, "draws": args.draws, "flow_steps": args.flow_steps, "noise_scale": args.noise_scale, "branch": args.branch, "cells": {}}
    print()
    for arm, _, _ in ARMS:
        for L in LOOKAHEADS:
            P = np.array(acc[arm][L]["p"]) * 1000.0
            G = np.array(acc[arm][L]["g"]) * 1000.0
            if len(P) < 5:
                continue
            e = P - G
            cell = {
                "n": int(len(P)),
                "bias_mm": {ax: float(e[:, i].mean()) for i, ax in enumerate("xyz")},
                "scatter_mm": {ax: float(e[:, i].std(ddof=1)) for i, ax in enumerate("xyz")},
                "rmse_mm": {ax: float(np.sqrt((e[:, i] ** 2).mean())) for i, ax in enumerate("xyz")},
                "rmse_norm_mm": float(np.sqrt((e ** 2).sum(1).mean())),
                "r2_vs_mean": {ax: float(1 - e[:, i].var() / max(G[:, i].var(), 1e-12))
                               for i, ax in enumerate("xyz")},
                "episodes": acc[arm][L]["ep"],
                "signed_error_mm": e.tolist(),
            }
            res["cells"][f"{arm}_L{L}"] = cell
            print(f"  {arm}_L{L}  n={cell['n']:3d}  z rmse {cell['rmse_mm']['z']:6.2f}  "
                  f"z bias {cell['bias_mm']['z']:+6.2f}  z scatter {cell['scatter_mm']['z']:6.2f}  "
                  f"|.| {cell['rmse_norm_mm']:6.2f}  z R2 {cell['r2_vs_mean']['z']:+.3f}", flush=True)
    if args.mode_probe > 0:
        res["mode_probe"] = {"K": args.mode_probe}
        for arm, _, _ in ARMS:
            for Lm in (23, 12):
                rows = mode_acc[arm][Lm]
                if len(rows) < 5:
                    continue
                q = lambda k: float(np.median([r[k] for r in rows]))
                cell = {"n": len(rows), "single_mm": q("single_mm"),
                        "best_of_K_mm": q("best_of_K_mm"), "spread_mm": q("spread_mm"),
                        "mag_ratio": q("mag_ratio"), "rows": rows}
                res["mode_probe"][f"{arm}_L{Lm}"] = cell
                print(f"  [mode {arm}_L{Lm}] n={cell['n']}  single {cell['single_mm']:.1f}  "
                      f"best/{args.mode_probe} {cell['best_of_K_mm']:.1f}  "
                      f"spread {cell['spread_mm']:.1f}  mag_ratio {cell['mag_ratio']:.2f}", flush=True)
    key = res["cells"].get("right_L5", {}).get("rmse_mm", {}).get("z")
    if key is not None:
        print(f"\n판정: right_L5 z RMSE = {key:.2f} mm  (pi05 2.68, 통과선 4.00) -> "
              f"{'PASS' if key <= 4.0 else 'FAIL'}")
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
