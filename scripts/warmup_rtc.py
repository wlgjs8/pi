"""Pre-warm the RTC (prev_action_chunk guidance) torch.compile graph on a running
policy server, so the FIRST RTC-guided inference of a real rollout does not stall
on recompilation (observed: rb_servo chunk_follower_engage_timeout at 3 s while
the server compiled the new graph signature for tens of seconds).

Run AFTER serve_policy.py is up (its built-in warmup covers only the plain
no-RTC graph), BEFORE launching flow-infer with FLOW_INFER_RTC=1:

    .venv/bin/python scripts/warmup_rtc.py --port 8000 --inference-delay 3

Match --inference-delay to the deploy value (= CHUNK_EXECUTE_STEPS - PREFETCH_AT);
scalar kwargs participate in torch.compile guards, so warming with a different
value would leave the deploy-value graph cold.
"""

import argparse
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--inference-delay", type=int, default=3)
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=6,
        help="Must match deploy CHUNK_EXECUTE_STEPS (openpi_remote sends "
        "this as execute_horizon; it participates in compile guards).",
    )
    parser.add_argument("--schedule", default="exp")
    parser.add_argument("--max-guidance-weight", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    rng = np.random.default_rng(0)

    def img():
        return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    obs = {
        "observation/left_wrist_0_rgb": img(),
        "observation/right_wrist_0_rgb": img(),
        "observation/left_wrist_0_depth": img(),
        "observation/right_wrist_0_depth": img(),
        "observation/state": rng.normal(0, 0.003, size=12).astype(np.float32),
        "prompt": "warmup",
    }

    t0 = time.monotonic()
    result = client.infer(obs)
    print(f"plain infer: {time.monotonic() - t0:.2f}s")
    raw = result.get("rtc_raw_actions")
    if raw is None:
        print("ERROR: server returned no 'rtc_raw_actions' — not a PyTorch policy? RTC unavailable.")
        return 1

    # Mirror openpi_remote.py's RTC request EXACTLY (all five _RTC_OBS_KEYS) —
    # any missing/extra scalar changes the torch.compile guard signature and the
    # first real rollout call recompiles for tens of seconds anyway.
    rtc_obs = dict(obs)
    rtc_obs["prev_action_chunk"] = np.asarray(raw, dtype=np.float32)
    rtc_obs["inference_delay"] = int(args.inference_delay)
    rtc_obs["execute_horizon"] = int(args.execute_horizon)
    rtc_obs["prefix_attention_schedule"] = args.schedule
    rtc_obs["max_guidance_weight"] = float(args.max_guidance_weight)

    for i in range(args.repeats):
        t0 = time.monotonic()
        result = client.infer(rtc_obs)
        dt = time.monotonic() - t0
        label = "COMPILE" if dt > 1.0 else "steady"
        print(f"rtc infer #{i + 1}: {dt:.2f}s ({label})")
        raw = result.get("rtc_raw_actions")
        if raw is not None:
            rtc_obs["prev_action_chunk"] = np.asarray(raw, dtype=np.float32)
    print("RTC graph warm — safe to launch flow-infer with FLOW_INFER_RTC=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
