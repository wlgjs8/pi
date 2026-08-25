#!/usr/bin/env python3
"""Serve a VA (DINOv3 wrist encoder) checkpoint over openpi's websocket policy protocol.

Why this file exists. The VA line was trained and scored entirely offline (train_va_dinov3.py,
eval_va_grasp.py); nothing ever served it. policy_runner's `openpi://host:port` action source speaks
openpi's WebsocketPolicyServer protocol, so the cheapest way to put VA on the robot is to wear that
protocol rather than touch the runner. Everything downstream -- camera polling, reset anchoring,
TcpPoseTarget emission, clamps, gripper runtime, rollout-mode gates -- is then bit-identical to the
pi05 runs, which is the only way a VA-vs-pi05 comparison stays interpretable.

Why the preprocessing is duplicated here instead of reused. The pi05 server normalises inside
PikaUmiInputs (SigLIP image statistics, 32-dim padded actions). VA wants ImageNet statistics and
emits 14 dims directly. Both consumed the SAME openpi transform stack at training time, so the state
normalisation and resize_with_pad geometry are shared, but the image statistics are NOT. The
formulas below are copied from eval_va_grasp.py so serving and val cannot drift -- if they drift,
the val numbers that justified deploying this checkpoint stop describing what runs on the robot.

Why the deployment branch is l2. On this checkpoint the dual head's branches disagree by more than
the run-to-run noise: l2 gives right/left L=5 z RMSE 2.50/1.89 mm against the flow branch's
2.62/2.29. The flow branch exists to keep multimodality (several candidate bolts) representable, and
it is what medoid sampling would need, but single-draw deployment reads l2.

RTC. Only `zeros` is implemented, and it is EXACT rather than approximated: openpi's zeros schedule
is get_prefix_weights = (idx < inference_delay), i.e. a hard freeze of the frozen prefix with no
mixing, which a deterministic head reproduces by copying those rows. `exp`/`linear` shape the
guidance DURING flow sampling and have no faithful post-hoc equivalent for an x0-regression head, so
they are rejected instead of silently approximated by a blend. That is also the posture the hardware
runs: exp was measured to make the descent 7-14 mm shallower and the chunk boundary rougher.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REAL_DIM = 14


def _normalize(x, q01, q99):
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _unnormalize(x, q01, q99):
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


class VAWebsocketPolicy:
    """openpi_client.base_policy.BasePolicy surface: infer(obs) -> dict."""

    def __init__(
        self,
        models,
        res,
        horizon,
        branch,
        flow_steps,
        noise_scale,
        norm_stats,
        device,
        pm1: bool = False,
        pin_state_grip=None,
    ):
        self.models = models
        self.pm1 = pm1  # SigLIP backbone: [-1,1] input (openpi convention), not ImageNet
        self.pin_state_grip = pin_state_grip
        self.res = res
        self.horizon = horizon
        self.branch = branch
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.device = device
        self.a_q01 = np.asarray(norm_stats["actions"].q01)[:REAL_DIM]
        self.a_q99 = np.asarray(norm_stats["actions"].q99)[:REAL_DIM]
        self.s_q01 = np.asarray(norm_stats["state"].q01)[:REAL_DIM]
        self.s_q99 = np.asarray(norm_stats["state"].q99)[:REAL_DIM]
        self._n = 0
        self._t_sum = 0.0

    def reset(self) -> None:
        pass

    def _prep(self, img):
        from openpi_client import image_tools

        x = image_tools.resize_with_pad(np.asarray(img), self.res, self.res)
        x = torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1).float() / 255.0
        if self.pm1 == "raw01":  # sigdino: branches normalise inside encode()
            return x[None].to(self.device).to(torch.bfloat16)
        if self.pm1 in (True, "pm1"):
            return (x * 2.0 - 1.0)[None].to(self.device).to(torch.bfloat16)
        m = torch.tensor(IMAGENET_MEAN)[:, None, None]
        s = torch.tensor(IMAGENET_STD)[:, None, None]
        return ((x - m) / s)[None].to(self.device).to(torch.bfloat16)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        img_l = self._prep(obs["observation/left_wrist_0_rgb"])
        img_r = self._prep(obs["observation/right_wrist_0_rgb"])

        state = np.asarray(obs["observation/state"], dtype=np.float64)[:REAL_DIM]
        if state.shape[0] != REAL_DIM:
            raise ValueError(
                f"observation/state has {state.shape[0]} dims, VA expects {REAL_DIM} " "(--proprio-mode velocity_grip)"
            )
        if self.pin_state_grip is not None:
            state = state.copy()
            state[6], state[13] = self.pin_state_grip
        sn = _normalize(state, self.s_q01, self.s_q99)
        st = torch.from_numpy(sn).float()[None].to(self.device).to(torch.bfloat16)

        # Ensemble members average in NORMALISED space, matching eval_va_grasp.py's --ckpt2 path.
        chunks = [
            m.sample_actions(
                img_l,
                img_r,
                st,
                num_steps=self.flow_steps,
                noise_scale=self.noise_scale,
                branch=self.branch,
            )
            .float()
            .cpu()
            .numpy()[0]
            for m in self.models
        ]
        norm_chunk = np.mean(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        prev = obs.get("prev_action_chunk")
        if prev is not None:
            schedule = str(obs.get("prefix_attention_schedule", "zeros"))
            if schedule != "zeros":
                raise ValueError(
                    f"prefix_attention_schedule={schedule!r} is not implementable for a "
                    "deterministic head; VA serving supports 'zeros' only "
                    "(set FLOW_INFER_RTC_SCHEDULE=zeros)"
                )
            d = int(np.clip(int(obs.get("inference_delay", 0)), 0, self.horizon))
            if d > 0:
                prev = np.asarray(prev, dtype=np.float32)[:, :REAL_DIM]
                norm_chunk[:d] = prev[:d]

        actions = _unnormalize(norm_chunk.astype(np.float64), self.a_q01, self.a_q99)

        self._n += 1
        self._t_sum += (time.perf_counter() - t0) * 1000.0
        if self._n % 100 == 0:
            print(f"[serve_va] {self._n} infers, mean {self._t_sum / self._n:.2f} ms", flush=True)

        # rtc_raw_actions is the MODEL-SPACE (normalised) chunk the client hands back as
        # prev_action_chunk next call; returning the post-freeze chunk keeps the round trip closed.
        return {
            "actions": actions.astype(np.float32),
            "rtc_raw_actions": norm_chunk.astype(np.float32),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="VA checkpoint (.pt). Repeat to average an ensemble; all members must " "share head/horizon.",
    )
    ap.add_argument("--weights-dir", default="/home/plaif/dinov3_weights")
    ap.add_argument("--dinov3-src", default="/home/plaif/dinov3_src")
    ap.add_argument(
        "--va-src",
        default="/home/plaif/workspace/openpi/examples/pika_umi",
        help="directory containing train_va_dinov3.py (VAPolicy definition)",
    )
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the c2 expert paths (reduce-overhead/CUDA graphs). The "
        "18-layer expert is kernel-launch-bound in eager mode: measured 101.5 -> "
        "36.0 ms/infer on gpu6. First few infers after start are slow (compile "
        "warmup) -- the server warms itself before accepting the first client.",
    )
    ap.add_argument(
        "--pin-state-grip",
        type=float,
        nargs=2,
        default=None,
        metavar=("L", "R"),
        help="Override the observation's state grip dims (6/13) with constants (raw "
        "fraction units, e.g. 0.83 0.66 = demo approach medians). Diagnostic for "
        "the measured on-robot grip echo spiral (cmd~=meas-5%%/tick): with the echo "
        "input pinned, any close that still fires must come from VISION. If the "
        "gripper then closes at the bolt, the visual close gate transfers to the "
        "robot and this is a viable scaffold; if it never closes, grip vision is "
        "domain-blind and the training-side fix (grip-state randomisation) is "
        "required.",
    )
    ap.add_argument(
        "--branch",
        default="l2",
        choices=["l2", "flow"],
        help="dual-head branch to serve. l2 = deterministic, the val champion at L=5. "
        "flow = multimodal, needed only for multi-draw sampling.",
    )
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument(
        "--config",
        default=None,
        help="openpi config for norm stats. Default: the config recorded in the "
        "checkpoint, which is what it trained through.",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.dinov3_src)
    sys.path.insert(0, args.va_src)
    from train_va_dinov3 import VAPolicy

    from openpi.serving import websocket_policy_server
    import openpi.training.config as _config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # VA's entire reason to exist is a 3.4 ms backbone; on CPU it is ~100x slower and the
        # deployment argument evaporates. Fail loudly rather than serve a policy that misses budget.
        raise SystemExit("no CUDA device: VA serving is pointless on CPU (see module docstring)")

    models = []
    head = horizon = res = cfg_name = None
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu")
        if head is None:
            head, horizon = ck.get("head", "l2"), ck["horizon"]
            res, cfg_name = int(ck.get("resolution", 224)), ck["config"]
        elif (ck.get("head", "l2"), ck["horizon"], int(ck.get("resolution", 224))) != (head, horizon, res):
            raise SystemExit(f"ensemble member {path} disagrees on head/horizon/resolution")
        if ck.get("arch") == "c2":
            # C2: transplanted pi05_base 300M expert head (c2_policy). Consumes [-1,1] images.
            from c2_policy import C2Policy

            m = C2Policy(ck["size"], pathlib.Path(args.weights_dir), ck["horizon"])
        else:
            m = VAPolicy(
                ck["size"],
                pathlib.Path(args.weights_dir),
                ck["horizon"],
                backbone_type=ck.get("backbone", "dinov3"),
                wm_head=ck.get("wm", False),
                state_tokens=ck.get("state_tokens", 0),
                head_mode=ck.get("head", "l2"),
                layers=ck.get("layers", 4),
                split_head=ck.get("split_head", False),
            )
        sd = ck.get("ema") or ck["model"]
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        models.append(m.to(device).eval().to(torch.bfloat16))
        print(
            f"[serve_va] loaded {path}  head={ck.get('head')} H={ck['horizon']} "
            f"step={ck.get('step')} ema={'yes' if ck.get('ema') else 'NO'}",
            flush=True,
        )

    if args.branch == "l2" and head == "flow":
        raise SystemExit("--branch l2 requires a dual- or l2-head checkpoint; this one is flow-only")

    cfg = _config.get_config(args.config or cfg_name)
    norm_stats = cfg.data.create(cfg.assets_dirs, cfg.model).norm_stats
    print(f"[serve_va] norm stats from config={cfg.name}", flush=True)

    if args.compile and ck.get("arch") == "c2":
        for m in models:
            m.suffix_pass = torch.compile(m.suffix_pass, mode="reduce-overhead")
            m.precompute = torch.compile(m.precompute)
        # warm the compiled graphs so the first client tick is not a 1-2 min stall
        _z = torch.zeros(1, 3, res, res, device=device, dtype=torch.bfloat16)
        _s = torch.zeros(1, REAL_DIM, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(4):
                models[0].sample_actions(_z, _z, _s, num_steps=args.flow_steps)
        print("[serve_va] c2 compiled + warmed", flush=True)
    pm1 = (
        "pm1"
        if ck.get("arch") == "c2"
        else {"siglip": "pm1", "sigdino": "raw01"}.get(ck.get("backbone", "dinov3"), False)
    )
    policy = VAWebsocketPolicy(
        models,
        res,
        horizon,
        args.branch,
        args.flow_steps,
        args.noise_scale,
        norm_stats,
        device,
        pm1=pm1,
        pin_state_grip=args.pin_state_grip,
    )
    if args.pin_state_grip:
        print(
            f"[serve_va] state grip PINNED to L={args.pin_state_grip[0]} "
            f"R={args.pin_state_grip[1]} (echo-spiral diagnostic)",
            flush=True,
        )

    print(
        f"[serve_va] serving {len(models)} model(s) branch={args.branch} H={horizon} "
        f"res={res} on {args.host}:{args.port}",
        flush=True,
    )
    print(
        "[serve_va] client must use --proprio-mode velocity_grip, "
        f"FLOW_INFER_ACTION_HORIZON={horizon}, FLOW_INFER_RTC_SCHEDULE=zeros",
        flush=True,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={"va_ckpt": args.ckpt, "branch": args.branch, "horizon": horizon},
    ).serve_forever()


if __name__ == "__main__":
    main()
