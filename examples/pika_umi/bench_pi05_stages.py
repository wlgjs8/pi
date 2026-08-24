#!/usr/bin/env python3
"""Split a pi05 checkpoint's inference latency into Vision / LLM-prefix / Action-expert.

The deployed number we measure end-to-end (`inference_latency_ms` in the rollout step log,
~93 ms for the wrist-only E1 checkpoint on this PC) is a single scalar, so there is no way to
tell whether shrinking the image-token count, the prompt, or the denoising step count would
actually buy anything. `Pi0.sample_actions` decomposes cleanly into exactly three stages:

    1. embed_prefix()  -> PaliGemma.img(...) per image  == SigLIP vision encoder (+ a cheap
                          embedding lookup for the prompt tokens)
    2. PaliGemma.llm([prefix_tokens, None], ...)        == LLM forward over the prefix, i.e.
                          the pass that builds the KV cache the action expert attends to
    3. num_steps x velocity()                           == action expert denoising loop

Timing them inside the jitted sample_actions is not possible, so each stage is jitted on its
own and timed with block_until_ready; the later stages are measured as cumulative prefixes and
differenced. Shapes come from `model.fake_obs()` -- latency does not depend on pixel values, and
this keeps the benchmark independent of a dataset or a live camera.

Usage:
  .venv/bin/python examples/pika_umi/bench_pi05_stages.py \
      --config pi05_pika_umi_wrist_velgrip_k1_h24_80k \
      --checkpoint-dir /home/plaif/workspace/pika_umi_models_v2/wrist_velgrip_k1_80k/79999
"""
from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import model as _model
from openpi.training import config as _config
import openpi.models.pi0 as _pi0


def _bench(fn, *args, repeat: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--num-steps", type=int, default=10, help="flow denoising steps (server default)")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    train_cfg = _config.get_config(args.config)
    model = train_cfg.model.load(_model.restore_params(f"{args.checkpoint_dir}/params", dtype=jnp.bfloat16))
    obs = train_cfg.model.fake_obs(batch_size=1)
    obs = _model.preprocess_observation(None, obs, train=False, image_keys=list(obs.images.keys()))

    n_img = len(obs.images)
    print(f"config={args.config}")
    print(f"images={n_img} keys={list(obs.images)}  num_steps={args.num_steps}")

    # Freeze the module state once (same trick nnx_utils.module_jit uses) so each stage can be a
    # plain jax.jit -- module_jit itself only accepts bound methods, and stage 2 is a composite.
    graphdef, state = nnx.split(model)
    num_steps = args.num_steps

    # --- stage 1: SigLIP over every image (no prompt embedding, no LLM) -------------------
    @jax.jit
    def vision_only(st, o):
        m = nnx.merge(graphdef, st)
        return [m.PaliGemma.img(o.images[k], train=False)[0] for k in o.images]

    # --- stage 1+2: full prefix embed, then the LLM pass that fills the KV cache ----------
    @jax.jit
    def prefix_and_cache(st, o):
        m = nnx.merge(graphdef, st)
        prefix_tokens, prefix_mask, prefix_ar_mask = m.embed_prefix(o)
        attn = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv = m.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        return kv

    # --- stage 1+2+3: what the server actually calls --------------------------------------
    @jax.jit
    def full(st, o, rng):
        m = nnx.merge(graphdef, st)
        return m.sample_actions(rng, o, num_steps=num_steps)

    rng = jax.random.key(0)
    v_med, v_min = _bench(lambda: vision_only(state, obs), repeat=args.repeat, warmup=args.warmup)
    p_med, p_min = _bench(lambda: prefix_and_cache(state, obs), repeat=args.repeat, warmup=args.warmup)
    f_med, f_min = _bench(lambda: full(state, obs, rng), repeat=args.repeat, warmup=args.warmup)

    llm_med = p_med - v_med
    act_med = f_med - p_med
    print()
    print(f"{'stage':<26} {'median ms':>10} {'share':>8}")
    print("-" * 46)
    print(f"{'1. Vision (SigLIP x%d)' % n_img:<26} {v_med:>10.1f} {100 * v_med / f_med:>7.1f}%")
    print(f"{'2. LLM prefix (KV cache)':<26} {llm_med:>10.1f} {100 * llm_med / f_med:>7.1f}%")
    print(f"{'3. Action expert (%d steps)' % args.num_steps:<26} {act_med:>10.1f} {100 * act_med / f_med:>7.1f}%")
    print("-" * 46)
    print(f"{'TOTAL sample_actions':<26} {f_med:>10.1f} {100.0:>7.1f}%")
    print()
    print(f"per-denoising-step: {act_med / max(args.num_steps, 1):.1f} ms")
    print(f"(best-case mins: vision {v_min:.1f}, prefix+cache {p_min:.1f}, full {f_min:.1f})")
    print("NOTE: stages 2/3 are differences of cumulative measurements, so a small negative or")
    print("      inflated value means the stages overlap in the compiled graph -- read the totals.")


if __name__ == "__main__":
    main()
