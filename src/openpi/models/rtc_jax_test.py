"""Cross-implementation parity tests: models.rtc_jax vs models_pytorch.rtc.

The torch module is the verified reference (its own tests check the paper's
equations); here we require the JAX port to reproduce it numerically, including
the full openpi-convention guided Euler loop that pi0.sample_actions runs.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import rtc_jax
from openpi.models_pytorch import rtc as rtc_torch


@pytest.mark.parametrize("schedule", ["exp", "linear", "zeros"])
@pytest.mark.parametrize(("start", "end", "total"), [(3, 18, 24), (0, 24, 24), (5, 5, 24), (24, 24, 24)])
def test_prefix_weights_match_torch(schedule, start, end, total):
    ours = rtc_jax.get_prefix_weights_np(start, end, total, schedule)
    ref = rtc_torch.get_prefix_weights(start, end, total, schedule).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_guidance_weight_matches_torch():
    for t in [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        ours = float(rtc_jax.guidance_weight(jnp.float32(t), jnp.float32(5.0)))
        ref = rtc_torch.guidance_weight(t, 5.0)
        np.testing.assert_allclose(ours, ref, rtol=1e-5)


def _toy_field(action_dim, horizon, seed=0):
    rng = np.random.default_rng(seed)
    a_mat = (0.1 * rng.normal(size=(action_dim, action_dim))).astype(np.float32)
    b = rng.normal(size=(horizon, action_dim)).astype(np.float32) * 0.5
    return a_mat, b


def _jax_rtc_loop(noise, velocity_fn, prev, weights, freeze_mask, max_gw, num_steps):
    """Mirror of the RTC branch in pi0.sample_actions (openpi convention)."""
    dt = -1.0 / num_steps
    x_t, time = jnp.asarray(noise), 1.0
    for _ in range(num_steps):
        x_t = rtc_jax.freeze_prefix(x_t, prev, freeze_mask)
        v_t = rtc_jax.guided_step_velocity(
            lambda x, t=time: velocity_fn(x, t), x_t, jnp.float32(time), prev, weights, max_gw
        )
        x_t = x_t + dt * v_t
        time += dt
    return rtc_jax.freeze_prefix(x_t, prev, freeze_mask)


@pytest.mark.parametrize("schedule", ["exp", "zeros"])
@pytest.mark.parametrize(("delay", "execute_horizon"), [(3, 6), (5, 6), (0, 24)])
def test_guided_loop_matches_torch_reference(schedule, delay, execute_horizon):
    horizon, action_dim, num_steps = 24, 8, 10
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    prev = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    a_mat, b = _toy_field(action_dim, horizon)

    def velocity_jax(x, _time):
        return jnp.einsum("bha,ca->bhc", x, jnp.asarray(a_mat)) + jnp.asarray(b)[None]

    def velocity_torch(x, _time):
        return torch.einsum("bha,ca->bhc", x, torch.from_numpy(a_mat)) + torch.from_numpy(b)[None]

    weights = rtc_jax.get_prefix_weights_np(delay, horizon - execute_horizon, horizon, schedule)
    freeze_mask = np.arange(horizon) < delay

    ours = np.asarray(
        _jax_rtc_loop(
            noise,
            velocity_jax,
            jnp.asarray(prev),
            jnp.asarray(weights),
            jnp.asarray(freeze_mask),
            jnp.float32(5.0),
            num_steps,
        )
    )
    ref = rtc_torch.rtc_sample_openpi(
        torch.from_numpy(noise),
        velocity_torch,
        prev_action_chunk=torch.from_numpy(prev),
        inference_delay=delay,
        execute_horizon=execute_horizon,
        num_steps=num_steps,
        prefix_attention_schedule=schedule,
        max_guidance_weight=5.0,
    ).numpy()
    np.testing.assert_allclose(ours, ref, atol=2e-4)


def test_zero_guidance_reduces_to_vanilla_euler():
    """delay=0 + execute_horizon=H -> all weights zero -> plain flow loop."""
    horizon, action_dim, num_steps = 24, 8, 10
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    prev = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    a_mat, b = _toy_field(action_dim, horizon, seed=1)

    def velocity_jax(x, _time):
        return jnp.einsum("bha,ca->bhc", x, jnp.asarray(a_mat)) + jnp.asarray(b)[None]

    weights = rtc_jax.get_prefix_weights_np(0, horizon - horizon, horizon, "exp")
    assert float(weights.max()) == 0.0
    guided = np.asarray(
        _jax_rtc_loop(
            noise,
            velocity_jax,
            jnp.asarray(prev),
            jnp.asarray(weights),
            jnp.asarray(np.zeros(horizon, dtype=bool)),
            jnp.float32(5.0),
            num_steps,
        )
    )

    dt = -1.0 / num_steps
    x_t, time = jnp.asarray(noise), 1.0
    for _ in range(num_steps):
        x_t = x_t + dt * velocity_jax(x_t, time)
        time += dt
    np.testing.assert_allclose(guided, np.asarray(x_t), atol=1e-5)


def test_frozen_prefix_is_pinned():
    horizon, action_dim, num_steps, delay = 24, 8, 5, 5
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    prev = rng.normal(size=(1, horizon, action_dim)).astype(np.float32)
    a_mat, b = _toy_field(action_dim, horizon, seed=2)

    def velocity_jax(x, _time):
        return jnp.einsum("bha,ca->bhc", x, jnp.asarray(a_mat)) + jnp.asarray(b)[None]

    weights = rtc_jax.get_prefix_weights_np(delay, horizon - 6, horizon, "zeros")
    out = np.asarray(
        _jax_rtc_loop(
            noise,
            velocity_jax,
            jnp.asarray(prev),
            jnp.asarray(weights),
            jnp.asarray(np.arange(horizon) < delay),
            jnp.float32(5.0),
            num_steps,
        )
    )
    np.testing.assert_allclose(out[:, :delay], prev[:, :delay], atol=0)
    assert not np.allclose(out[:, delay:], prev[:, delay:])
