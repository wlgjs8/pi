"""Real-Time Chunking (RTC) core — standalone, model-agnostic torch primitives.

Port of the inference-time inpainting / guidance algorithm from
``Physical-Intelligence/real-time-chunking-kinetix`` (``src/model.py``) to torch.

RTC runs the *next* action chunk's flow denoising while the *current* chunk is
still executing. It HARD-FREEZES the first ``inference_delay`` actions (the ones
guaranteed to run during inference latency) and SOFT-GUIDES the rest toward the
previously committed chunk via pseudoinverse (VJP) guidance, so the new plan
stays continuous with the old one. Retraining-free; works on any flow/diffusion
chunk policy. See ``robotics_lab/docs/rtc_design.md`` and the wiki
``research/vla/rtc.md``.

Flow-time convention here is the REFERENCE (Kinetix) one::

    t = 0  -> noise,   t = 1 -> clean data,   x1_hat = x_t + (1 - t) * v_t

openpi's ``pi0_pytorch.sample_actions`` integrates the OPPOSITE direction
(``time`` = 1 at noise -> 0 at data). The integration adapter (M1) MUST map
``t = 1 - time`` before calling these helpers. Keeping the core in the reference
convention lets the unit tests check directly against the paper's equations.

This module is import-only: it changes no existing behavior. RTC stays OFF until
an integration layer explicitly calls it.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Literal

import torch

PrefixAttentionSchedule = Literal["exp", "linear", "zeros"]

# Velocity field: maps (x_t [..., H, A], flow-time t in [0, 1]) -> v_t [..., H, A].
# Must be differentiable w.r.t. x_t (RTC takes a VJP through it).
DenoiseFn = Callable[[torch.Tensor, float], torch.Tensor]


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixAttentionSchedule = "exp",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Soft-mask weights ``W[total]`` over the chunk index.

    Mirrors ``get_prefix_weights(start, end, total, schedule)`` in the reference,
    called there with ``start = inference_delay``, ``end =
    prefix_attention_horizon``, ``total = action_chunk_size``.

    The base ramp is ``1`` for indices ``< start`` (the frozen / fully-committed
    prefix), then decays linearly to ``0`` at index ``end``, and stays ``0``
    beyond. ``exp`` shapes that ramp to be convex (the reference default);
    ``zeros`` keeps only the hard step (``1`` for ``i < start`` else ``0``);
    ``linear`` is the plain clipped ramp.
    """
    start = int(start)
    end = int(end)
    total = int(total)
    idx = torch.arange(total, device=device, dtype=dtype)
    denom = float(end - start + 1)
    if denom <= 0.0:
        # Degenerate ramp (end <= start - 1): no soft region, hard step at start.
        return (idx < start).to(dtype)
    w = torch.clamp((start - 1 - idx) / denom + 1.0, 0.0, 1.0)
    if schedule == "exp":
        w = w * torch.expm1(w) / (math.e - 1.0)
    elif schedule == "zeros":
        w = (idx < start).to(dtype)
    elif schedule == "linear":
        pass  # plain clipped ramp
    else:  # pragma: no cover - guarded by the Literal type at call sites
        raise ValueError(f"unknown prefix_attention_schedule: {schedule!r}")
    return w


def guidance_weight(t: float, max_guidance_weight: float) -> float:
    """Per-flow-time guidance scale (reference convention, ``t = 1`` is data).

    ``inv_r2 = (t^2 + (1-t)^2) / (1-t)^2`` and ``c = (1-t)/t`` (with ``c``'s
    ``t -> 0`` divergence clamped to ``max_guidance_weight``, as the reference's
    ``nan_to_num(..., posinf=max_guidance_weight)`` does), then clipped to
    ``max_guidance_weight``. Large near the noise end, minimal mid-flow. The
    denoising grid never samples ``t = 1`` exactly, so the ``(1-t)^2`` divisor is
    safe on the grid.
    """
    t = float(t)
    if t <= 0.0:
        c = float(max_guidance_weight)
        inv_r2 = 1.0  # (0 + 1) / 1
    else:
        one_minus = 1.0 - t
        inv_r2 = (t * t + one_minus * one_minus) / (one_minus * one_minus)
        c = one_minus / t
    return float(min(c * inv_r2, float(max_guidance_weight)))


def clean_estimate(x_t: torch.Tensor, v_t: torch.Tensor, t: float) -> torch.Tensor:
    """First-order estimate of the clean action chunk: ``x1_hat = x_t + (1-t) v_t``."""
    return x_t + (1.0 - float(t)) * v_t


def freeze_prefix(
    x_t: torch.Tensor, prev_action_chunk: torch.Tensor, inference_delay: int
) -> torch.Tensor:
    """Overwrite the first ``inference_delay`` actions with the committed plan.

    These are guaranteed to execute during the inference delay, so they are hard
    pinned (not just guided) every denoising step. Returns a new tensor.
    """
    d = int(inference_delay)
    if d <= 0:
        return x_t
    horizon = x_t.shape[-2]
    d = min(d, horizon)
    out = x_t.clone()
    out[..., :d, :] = prev_action_chunk[..., :d, :]
    return out


def rtc_guided_velocity(
    x_t: torch.Tensor,
    t: float,
    prev_action_chunk: torch.Tensor,
    weights: torch.Tensor,
    denoise_fn: DenoiseFn,
    max_guidance_weight: float,
) -> torch.Tensor:
    """One RTC-guided velocity: ``v_t + guidance_weight * VJP(x1_hat, x_t)(error)``.

    ``error = weights * (prev_action_chunk - x1_hat)`` and the correction is the
    vector-Jacobian product of the clean estimate w.r.t. ``x_t`` applied to that
    error (the reference's ``vjp_fun(error)[0]``), i.e. a pseudoinverse guidance
    that pulls the soft-masked region of the sample toward the committed plan.
    The returned tensor is detached (guidance graph is internal to this step).
    """
    x_in = x_t.detach().requires_grad_()
    v_t = denoise_fn(x_in, t)
    x1_hat = clean_estimate(x_in, v_t, t)
    error = weights[:, None] * (prev_action_chunk - x1_hat)
    (pinv_correction,) = torch.autograd.grad(x1_hat, x_in, grad_outputs=error)
    gw = guidance_weight(t, max_guidance_weight)
    return v_t.detach() + gw * pinv_correction.detach()


def rtc_sample(
    noise: torch.Tensor,
    denoise_fn: DenoiseFn,
    *,
    prev_action_chunk: torch.Tensor,
    inference_delay: int,
    execute_horizon: int,
    num_steps: int = 5,
    prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    max_guidance_weight: float = 5.0,
) -> torch.Tensor:
    """Reference-convention RTC flow sample (t: 0 -> 1).

    Integrates ``x_t += dt * v_guided`` for ``num_steps`` Euler steps, freezing
    the first ``inference_delay`` actions and guiding the rest toward
    ``prev_action_chunk``. ``execute_horizon`` (= the committed steps per replan,
    ``s``) sets the guided region width via ``prefix_attention_horizon = H - s``.

    This is the self-contained M0 reference. The openpi integration (M1) reuses
    the helpers above but maps openpi's ``time = 1 - t`` and re-uses the model's
    KV-cached ``denoise_step`` as ``denoise_fn``.
    """
    horizon = noise.shape[-2]
    prefix_attention_horizon = horizon - int(execute_horizon)
    weights = get_prefix_weights(
        inference_delay,
        prefix_attention_horizon,
        horizon,
        prefix_attention_schedule,
        device=noise.device,
        dtype=noise.dtype,
    )
    x_t = noise
    dt = 1.0 / int(num_steps)
    t = 0.0
    for _ in range(int(num_steps)):
        x_t = freeze_prefix(x_t, prev_action_chunk, inference_delay)
        v_t = rtc_guided_velocity(
            x_t, t, prev_action_chunk, weights, denoise_fn, max_guidance_weight
        )
        x_t = x_t + dt * v_t
        t += dt
    return freeze_prefix(x_t, prev_action_chunk, inference_delay)


# --------------------------------------------------------------------------- #
# openpi-convention helpers (M1).
#
# pi0_pytorch.sample_actions integrates the OPPOSITE flow direction from the
# reference: `time` runs 1 (noise) -> 0 (data) with dt = -1/num_steps, and the
# model's velocity is v = dx/d(time). The reference-convention data-ness is
# t = 1 - time, the clean estimate is x1_hat = x_t - time * v_t, and -- crucially
# -- because the Euler step uses dt < 0, the guidance must be SUBTRACTED so it
# still pulls the sample toward the committed plan (two derivations: keeping
# dt*correction equal to the reference's, and requiring d<x1_hat,error> to grow).
# These functions let M1 reuse the verified core under openpi's convention.
# --------------------------------------------------------------------------- #


def rtc_guided_velocity_openpi(
    x_t: torch.Tensor,
    time: torch.Tensor | float,
    prev_action_chunk: torch.Tensor,
    weights: torch.Tensor,
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    max_guidance_weight: float,
) -> torch.Tensor:
    """RTC-guided velocity in openpi's convention (``time``: 1=noise, 0=data).

    ``v_guided = v_t - guidance_weight(1 - time) * VJP(x1_hat, x_t)(error)`` with
    ``x1_hat = x_t - time * v_t`` and ``error = weights * (prev - x1_hat)``. The
    guidance is computed under ``torch.enable_grad`` so it works inside the
    ``@torch.no_grad()`` ``sample_actions`` (the model uses ``no_grad``, not
    ``inference_mode``, so a local ``enable_grad`` is permitted). ``denoise_fn``
    closes over the model's KV-cached ``denoise_step`` and must be differentiable
    w.r.t. its input.
    """
    time_f = float(time)
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_()
        v_t = denoise_fn(x_in)
        x1_hat = x_in - time_f * v_t
        error = weights[:, None] * (prev_action_chunk - x1_hat)
        (pinv_correction,) = torch.autograd.grad(x1_hat, x_in, grad_outputs=error)
    gw = guidance_weight(1.0 - time_f, max_guidance_weight)
    return v_t.detach() - gw * pinv_correction.detach()


def rtc_sample_openpi(
    noise: torch.Tensor,
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    prev_action_chunk: torch.Tensor,
    inference_delay: int,
    execute_horizon: int,
    num_steps: int = 5,
    prefix_attention_schedule: PrefixAttentionSchedule = "exp",
    max_guidance_weight: float = 5.0,
) -> torch.Tensor:
    """openpi-convention RTC flow sample (``time``: 1 -> 0, ``dt = -1/num_steps``).

    Mirrors ``pi0_pytorch.sample_actions``'s Euler loop exactly (same grid, same
    ``x += dt * v``) but freezes the first ``inference_delay`` actions and guides
    the rest toward ``prev_action_chunk``. With ``inference_delay = 0`` and
    ``execute_horizon = H`` (all weights zero) it reduces bit-for-bit to the
    vanilla loop. ``denoise_fn(x_t, time_0dim) -> v_t`` lets the caller pass the
    model's KV-cached ``denoise_step`` (handling the ``expand(bsize)``).
    """
    horizon = noise.shape[-2]
    prefix_attention_horizon = horizon - int(execute_horizon)
    weights = get_prefix_weights(
        inference_delay,
        prefix_attention_horizon,
        horizon,
        prefix_attention_schedule,
        device=noise.device,
        dtype=noise.dtype,
    )
    prev = prev_action_chunk
    while prev.dim() < noise.dim():
        prev = prev.unsqueeze(0)

    dt = torch.tensor(-1.0 / int(num_steps), dtype=noise.dtype, device=noise.device)
    x_t = noise
    time = torch.tensor(1.0, dtype=noise.dtype, device=noise.device)
    while time >= -dt / 2:
        x_t = freeze_prefix(x_t, prev, inference_delay)
        current_time = time
        # Bind current_time as a default so the closure captures this iteration's
        # value (it is called synchronously inside rtc_guided_velocity_openpi).
        v_t = rtc_guided_velocity_openpi(
            x_t,
            current_time,
            prev,
            weights,
            lambda xx, t=current_time: denoise_fn(xx, t),
            max_guidance_weight,
        )
        x_t = x_t + dt * v_t
        time = time + dt
    return freeze_prefix(x_t, prev, inference_delay)
