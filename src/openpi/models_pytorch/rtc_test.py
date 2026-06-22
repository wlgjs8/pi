"""Unit tests for the standalone RTC core (M0).

Checks each primitive against the reference equations
(``Physical-Intelligence/real-time-chunking-kinetix/src/model.py``) and the RTC
guidance against an analytic VJP, with no pi0/jax dependency. CPU-only.
"""

from __future__ import annotations

import math

import torch

from openpi.models_pytorch import rtc


def test_prefix_weights_zeros_is_hard_step() -> None:
    # zeros schedule: 1 for i < start (=inference_delay), 0 otherwise.
    w = rtc.get_prefix_weights(start=2, end=4, total=6, schedule="zeros")
    assert torch.equal(w, torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))


def test_prefix_weights_linear_ramps_one_to_zero() -> None:
    # Base ramp: 1 for i < start, linear down to 0 at i == end, 0 beyond.
    w = rtc.get_prefix_weights(start=2, end=4, total=6, schedule="linear")
    # denom = end - start + 1 = 3; w_i = clip((start-1-i)/3 + 1, 0, 1)
    expected = torch.tensor(
        [
            min(1.0, (2 - 1 - 0) / 3 + 1),  # 1.33 -> 1
            (2 - 1 - 1) / 3 + 1,            # 1.00
            (2 - 1 - 2) / 3 + 1,            # 0.667
            (2 - 1 - 3) / 3 + 1,            # 0.333
            max(0.0, (2 - 1 - 4) / 3 + 1),  # 0.0
            max(0.0, (2 - 1 - 5) / 3 + 1),  # -0.33 -> 0
        ]
    )
    assert torch.allclose(w, expected, atol=1e-6)


def test_prefix_weights_exp_bounds_and_monotone() -> None:
    w = rtc.get_prefix_weights(start=3, end=12, total=16, schedule="exp")
    assert w.shape == (16,)
    assert torch.all(w >= 0.0)
    assert torch.all(w <= 1.0 + 1e-6)
    # frozen prefix weight == 1 (ramp clipped to 1 -> exp(1) maps to 1).
    assert torch.allclose(w[:3], torch.ones(3), atol=1e-6)
    # zero at/after `end`.
    assert torch.allclose(w[12:], torch.zeros(4), atol=1e-6)
    # non-increasing across the chunk.
    assert torch.all(w[1:] <= w[:-1] + 1e-6)


def test_prefix_weights_exp_value_at_ramp_midpoint() -> None:
    # exp shaping: w_exp = w * expm1(w) / (e - 1). At base-ramp value w=0.5:
    w = rtc.get_prefix_weights(start=2, end=4, total=6, schedule="exp")
    base = (2 - 1 - 3) / 3.0 + 1.0  # index 3 base ramp = 1/3
    expected_i3 = base * math.expm1(base) / (math.e - 1.0)
    assert math.isclose(float(w[3]), expected_i3, rel_tol=1e-6)


def test_guidance_weight_midflow_and_clip() -> None:
    # t=0.5: inv_r2 = (0.25+0.25)/0.25 = 2, c = 1 -> gw = 2.
    assert math.isclose(rtc.guidance_weight(0.5, max_guidance_weight=5.0), 2.0, rel_tol=1e-6)
    # t->0 (noise end): clamped to max_guidance_weight.
    assert math.isclose(rtc.guidance_weight(0.0, 5.0), 5.0, rel_tol=1e-6)
    # small t gives a large value, clipped at the ceiling.
    assert rtc.guidance_weight(0.05, 5.0) == 5.0
    # the closed form (t^2+(1-t)^2)/(t(1-t)) below the ceiling.
    t = 0.3
    expected = (t * t + (1 - t) ** 2) / (t * (1 - t))
    assert math.isclose(rtc.guidance_weight(t, 100.0), expected, rel_tol=1e-6)


def test_clean_estimate_linear_flow() -> None:
    x_t = torch.randn(4, 3)
    v_t = torch.randn(4, 3)
    out = rtc.clean_estimate(x_t, v_t, t=0.25)
    assert torch.allclose(out, x_t + 0.75 * v_t)


def test_freeze_prefix_overwrites_only_first_d() -> None:
    x_t = torch.zeros(6, 2)
    prev = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    out = rtc.freeze_prefix(x_t, prev, inference_delay=3)
    assert torch.equal(out[:3], prev[:3])          # frozen
    assert torch.equal(out[3:], torch.zeros(3, 2))  # untouched
    # d=0 is a no-op.
    assert torch.equal(rtc.freeze_prefix(x_t, prev, 0), x_t)


def test_guided_velocity_matches_analytic_vjp_linear_denoiser() -> None:
    torch.manual_seed(0)
    horizon, adim = 5, 4
    amat = torch.randn(adim, adim)

    def denoise_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return x @ amat.T  # per-row linear velocity, differentiable in x

    x_t = torch.randn(horizon, adim)
    prev = torch.randn(horizon, adim)
    weights = torch.linspace(1.0, 0.0, horizon)
    t = 0.3
    out = rtc.rtc_guided_velocity(x_t, t, prev, weights, denoise_fn, max_guidance_weight=100.0)

    # Analytic: v = x @ A^T; x1_hat = x + (1-t) x @ A^T = (I + (1-t)A) x (per row).
    # error = W * (prev - x1_hat); VJP = error + (1-t) * (error @ A).
    v_t = x_t @ amat.T
    x1_hat = x_t + (1 - t) * v_t
    error = weights[:, None] * (prev - x1_hat)
    pinv = error + (1 - t) * (error @ amat)
    gw = rtc.guidance_weight(t, 100.0)
    expected = v_t + gw * pinv
    assert torch.allclose(out, expected, atol=1e-5)


def test_zero_weights_recovers_plain_velocity() -> None:
    horizon, adim = 4, 3
    amat = torch.randn(adim, adim)

    def denoise_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return x @ amat.T

    x_t = torch.randn(horizon, adim)
    prev = torch.randn(horizon, adim)
    weights = torch.zeros(horizon)
    out = rtc.rtc_guided_velocity(x_t, 0.4, prev, weights, denoise_fn, max_guidance_weight=5.0)
    assert torch.allclose(out, x_t @ amat.T, atol=1e-6)


def test_rtc_sample_freezes_prefix_and_inpaints_toward_prev() -> None:
    # A denoiser whose unguided fixed point is the origin, so an unguided sample
    # drifts toward 0 and away from `prev`; RTC should pin the prefix to prev and
    # pull the guided region closer to prev than the unguided sample.
    torch.manual_seed(1)
    horizon, adim = 8, 4
    d, s = 2, 4

    def denoise_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return -x  # velocity toward the origin

    noise = torch.randn(horizon, adim)
    prev = torch.full((horizon, adim), 3.0)

    guided = rtc.rtc_sample(
        noise, denoise_fn, prev_action_chunk=prev, inference_delay=d,
        execute_horizon=s, num_steps=5, prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
    )
    unguided = rtc.rtc_sample(
        noise, denoise_fn, prev_action_chunk=prev, inference_delay=0,
        execute_horizon=horizon,  # prefix_attention_horizon = 0 -> all weights 0
        num_steps=5, prefix_attention_schedule="exp", max_guidance_weight=5.0,
    )

    # Frozen prefix is pinned exactly to the committed plan.
    assert torch.allclose(guided[:d], prev[:d], atol=1e-5)
    # Guided region (between the frozen prefix and the free tail) is closer to
    # prev than the unguided sample.
    region = slice(d, horizon - s + d)
    err_guided = (guided[region] - prev[region]).abs().mean()
    err_unguided = (unguided[region] - prev[region]).abs().mean()
    assert err_guided < err_unguided


# --------------------------------------------------------------------------- #
# openpi-convention helpers (M1): time runs 1 (noise) -> 0 (data), dt = -1/n.
# --------------------------------------------------------------------------- #


def test_guided_velocity_openpi_matches_analytic_vjp() -> None:
    torch.manual_seed(2)
    horizon, adim = 5, 4
    amat = torch.randn(adim, adim)

    def denoise_fn(x: torch.Tensor) -> torch.Tensor:
        return x @ amat.T

    x_t = torch.randn(horizon, adim)
    prev = torch.randn(horizon, adim)
    weights = torch.linspace(1.0, 0.0, horizon)
    time = 0.7  # openpi time (data-ness t_ref = 1 - time = 0.3)
    out = rtc.rtc_guided_velocity_openpi(x_t, time, prev, weights, denoise_fn, max_guidance_weight=100.0)

    # x1_hat = x - time * v = (I - time*A) x; VJP(error) = error - time*(error @ A);
    # openpi SUBTRACTS the guidance (dt < 0).
    v_t = x_t @ amat.T
    x1_hat = x_t - time * v_t
    error = weights[:, None] * (prev - x1_hat)
    pinv = error - time * (error @ amat)
    gw = rtc.guidance_weight(1.0 - time, 100.0)
    expected = v_t - gw * pinv
    assert torch.allclose(out, expected, atol=1e-5)


def test_rtc_sample_openpi_reduces_to_vanilla_when_guidance_off() -> None:
    # Loop-level parity: no freeze (d=0) + all-zero weights (execute_horizon=H)
    # must reproduce the vanilla openpi Euler loop bit-for-bit.
    torch.manual_seed(3)
    horizon, adim = 6, 4
    amat = torch.randn(adim, adim)
    bias = torch.randn(adim)
    num_steps = 10

    def denoise_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x @ amat.T + float(t) * bias

    noise = torch.randn(horizon, adim)
    out = rtc.rtc_sample_openpi(
        noise, denoise_fn, prev_action_chunk=torch.zeros(horizon, adim),
        inference_delay=0, execute_horizon=horizon, num_steps=num_steps,
        prefix_attention_schedule="exp", max_guidance_weight=5.0,
    )

    # Vanilla openpi Euler (mirrors pi0_pytorch.sample_actions).
    dt = torch.tensor(-1.0 / num_steps)
    x = noise.clone()
    t = torch.tensor(1.0)
    while t >= -dt / 2:
        x = x + dt * denoise_fn(x, t)
        t = t + dt
    assert torch.allclose(out, x, atol=1e-6)


def test_rtc_sample_openpi_freezes_and_inpaints() -> None:
    torch.manual_seed(4)
    horizon, adim = 8, 4
    d, s = 2, 4

    def denoise_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x  # vanilla sample drifts toward the origin, away from prev

    noise = torch.randn(horizon, adim)
    prev = torch.full((horizon, adim), 3.0)

    guided = rtc.rtc_sample_openpi(
        noise, denoise_fn, prev_action_chunk=prev, inference_delay=d,
        execute_horizon=s, num_steps=5, prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
    )
    unguided = rtc.rtc_sample_openpi(
        noise, denoise_fn, prev_action_chunk=prev, inference_delay=0,
        execute_horizon=horizon, num_steps=5, prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
    )

    assert torch.allclose(guided[:d], prev[:d], atol=1e-5)
    region = slice(d, horizon - s + d)
    err_guided = (guided[region] - prev[region]).abs().mean()
    err_unguided = (unguided[region] - prev[region]).abs().mean()
    assert err_guided < err_unguided
