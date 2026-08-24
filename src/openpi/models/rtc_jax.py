"""Real-Time Chunking (RTC) guidance for the JAX pi0/pi05 flow sampler.

JAX port of ``openpi.models_pytorch.rtc`` (itself a port of
``Physical-Intelligence/real-time-chunking-kinetix``). RTC denoises the NEXT
action chunk while the current one is still executing: the first
``inference_delay`` actions are HARD-FROZEN to the committed plan (they will
execute during inference latency no matter what) and the remainder is
SOFT-GUIDED toward it via pseudoinverse (VJP) guidance, so consecutive chunks
stay continuous. Retraining-free.

Split of responsibilities (deliberately different from the torch version):

* ``get_prefix_weights_np`` runs on the HOST (numpy) and its result is passed
  into the jitted sampler as a plain ``(H,)`` array, together with a boolean
  ``(H,)`` freeze mask and a scalar ``max_guidance_weight`` array. Because the
  jitted function only ever sees ARRAYS of fixed shape, changing
  ``inference_delay`` / ``execute_horizon`` / the schedule string never
  retraces or recompiles — unlike the torch path, where those scalars are
  compile guards (a first call with a new value stalls tens of seconds).
* ``guided_step_velocity`` runs INSIDE the jitted Euler loop and mirrors
  ``rtc_guided_velocity_openpi`` exactly, in openpi's time convention
  (``time``: 1 = noise -> 0 = data, ``dt = -1/num_steps``):

      x1_hat   = x_t - time * v_t
      error    = weights * (prev - x1_hat)
      v_guided = v_t - guidance_weight(1 - time) * VJP(x1_hat, x_t)(error)

  The guidance is SUBTRACTED because the Euler step uses dt < 0 (see the
  derivation note in ``models_pytorch/rtc.py``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def get_prefix_weights_np(start: int, end: int, total: int, schedule: str = "exp") -> np.ndarray:
    """Host-side soft-mask weights ``W[total]`` over the chunk index.

    Numerically identical to ``models_pytorch.rtc.get_prefix_weights``:
    ``1`` for indices ``< start`` (frozen prefix), a ramp to ``0`` at ``end``,
    ``0`` beyond. ``exp`` = convex reference shaping, ``zeros`` = hard step
    only, ``linear`` = plain clipped ramp.
    """
    start = int(start)
    end = int(end)
    total = int(total)
    idx = np.arange(total, dtype=np.float32)
    denom = float(end - start + 1)
    if denom <= 0.0:
        return (idx < start).astype(np.float32)
    w = np.clip((start - 1 - idx) / denom + 1.0, 0.0, 1.0)
    if schedule == "exp":
        w = w * np.expm1(w) / (math.e - 1.0)
    elif schedule == "zeros":
        w = (idx < start).astype(np.float32)
    elif schedule == "linear":
        pass
    else:
        raise ValueError(f"unknown prefix_attention_schedule: {schedule!r}")
    return w.astype(np.float32)


def guidance_weight(t_ref: jnp.ndarray, max_guidance_weight: jnp.ndarray) -> jnp.ndarray:
    """Per-flow-time guidance scale in the REFERENCE convention (t_ref=1 is data).

    ``min(c * inv_r2, max)`` with ``inv_r2 = (t^2 + (1-t)^2) / (1-t)^2`` and
    ``c = (1-t)/t`` (t -> 0 divergence clamped to ``max``), mirroring
    ``models_pytorch.rtc.guidance_weight``. The sampling grid never reaches
    ``t_ref = 1`` exactly (openpi's last step is at ``time = 1/num_steps``), and
    the divisors are guarded for safety under jit.
    """
    t = jnp.asarray(t_ref, dtype=jnp.float32)
    one_minus = 1.0 - t
    inv_r2 = (t * t + one_minus * one_minus) / jnp.maximum(one_minus * one_minus, 1e-12)
    c = one_minus / jnp.maximum(t, 1e-12)
    inv_r2 = jnp.where(t <= 0.0, 1.0, inv_r2)
    c = jnp.where(t <= 0.0, max_guidance_weight, c)
    return jnp.minimum(c * inv_r2, max_guidance_weight)


def freeze_prefix(x_t: jnp.ndarray, prev_action_chunk: jnp.ndarray, freeze_mask: jnp.ndarray) -> jnp.ndarray:
    """Overwrite the frozen rows (``freeze_mask`` True) with the committed plan.

    ``x_t``/``prev_action_chunk``: ``(b, H, A)``; ``freeze_mask``: bool
    ``(H,)`` or explicitly batch-tiled ``(b, H)``.
    """
    if freeze_mask.ndim == 1:
        freeze_mask = freeze_mask[None, ...]
    return jnp.where(freeze_mask[..., None], prev_action_chunk, x_t)


def guided_step_velocity(
    velocity_fn,
    x_t: jnp.ndarray,
    time: jnp.ndarray,
    prev_action_chunk: jnp.ndarray,
    weights: jnp.ndarray,
    max_guidance_weight: jnp.ndarray,
) -> jnp.ndarray:
    """One RTC-guided velocity in openpi's convention (``time``: 1=noise, 0=data).

    ``velocity_fn(x) -> v`` must close over the current ``time`` (the caller
    binds it) and be differentiable w.r.t. ``x`` — it is the model's KV-cached
    suffix forward. Mirrors ``rtc_guided_velocity_openpi``.
    """

    def clean_estimate(x):
        v = velocity_fn(x)
        return x - time * v, v

    x1_hat, vjp_fn, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)
    if weights.ndim == 1:
        weights = weights[None, ...]
    error = weights[..., None] * (prev_action_chunk - x1_hat)
    (pinv_correction,) = vjp_fn(error)
    gw = guidance_weight(1.0 - time, max_guidance_weight)
    return v_t - gw[..., None, None] * pinv_correction
