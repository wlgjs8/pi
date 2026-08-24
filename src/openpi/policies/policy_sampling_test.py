from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models import rtc_jax
from openpi.policies import policy as _policy


class _StubJaxModel(_model.BaseModel):
    def __init__(self, action_horizon: int = 2, action_dim: int = 2):
        super().__init__(action_dim, action_horizon, max_token_len=1)

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        del rng, observation, train
        return jnp.zeros(actions.shape[:-1])

    def sample_actions(
        self,
        rng,
        observation,
        *,
        noise=None,
        rtc_prev_action_chunk=None,
        rtc_prefix_weights=None,
        rtc_freeze_mask=None,
        rtc_max_guidance_weight=None,
    ):
        del rtc_prev_action_chunk, rtc_prefix_weights, rtc_freeze_mask, rtc_max_guidance_weight
        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        return noise + observation.state[:, None, :]


class _CountTransform:
    def __init__(self):
        self.calls = 0

    def __call__(self, data):
        self.calls += 1
        return data


class _ScaleActions:
    def __init__(self, scale: float):
        self._scale = scale

    def __call__(self, data):
        return {**data, "actions": data["actions"] * self._scale}


def _observation(**extra):
    return {
        "image": {},
        "image_mask": {},
        "state": np.array([1.0, -2.0], dtype=np.float32),
        **extra,
    }


def test_select_medoid_known_chunk() -> None:
    chunks = [
        np.array([[0.0, 0.0]]),
        np.array([[2.0, 0.0]]),
        np.array([[10.0, 0.0]]),
    ]

    assert _policy.select_medoid(chunks) == 1


def test_batched_samples_match_explicit_noise_sequential_samples() -> None:
    sequential_transform = _CountTransform()
    batched_transform = _CountTransform()
    sequential_policy = _policy.Policy(_StubJaxModel(), rng=jax.random.key(0), transforms=[sequential_transform])
    batched_policy = _policy.Policy(_StubJaxModel(), rng=jax.random.key(0), transforms=[batched_transform])
    noise = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[-2.0, -1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    sequential = [sequential_policy.infer(_observation(), noise=sample) for sample in noise]
    batched = batched_policy.infer_samples(_observation(), len(noise), noise=noise)

    assert sequential_transform.calls == len(noise)
    assert batched_transform.calls == 1
    for expected, actual in zip(sequential, batched, strict=True):
        np.testing.assert_allclose(actual["state"], expected["state"])
        np.testing.assert_allclose(actual["actions"], expected["actions"])
    sequential_medoid = _policy.select_medoid([result["actions"] for result in sequential])
    batched_medoid = _policy.select_medoid([result["actions"] for result in batched])
    assert batched_medoid == sequential_medoid


def test_infer_samples_tiles_rtc_kwargs() -> None:
    num_samples = 3
    horizon = 2
    policy = _policy.Policy(_StubJaxModel(action_horizon=horizon), rng=jax.random.key(0))
    recorded = {}

    def record_sample_actions(rng, observation, **kwargs):
        del rng
        recorded["observation_state"] = np.asarray(observation.state)
        recorded.update({key: np.asarray(value) for key, value in kwargs.items()})
        return kwargs["noise"]

    policy._sample_actions = record_sample_actions  # noqa: SLF001
    prev = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    policy.infer_samples(
        _observation(
            prev_action_chunk=prev,
            inference_delay=1,
            execute_horizon=1,
            prefix_attention_schedule="linear",
            max_guidance_weight=4.0,
        ),
        num_samples,
        noise=np.zeros((num_samples, horizon, 2), dtype=np.float32),
    )

    assert recorded["observation_state"].shape == (num_samples, 2)
    assert recorded["rtc_prev_action_chunk"].shape == (num_samples, horizon, 2)
    assert recorded["rtc_prefix_weights"].shape == (num_samples, horizon)
    assert recorded["rtc_freeze_mask"].shape == (num_samples, horizon)
    assert recorded["rtc_max_guidance_weight"].shape == (num_samples,)
    for row in recorded["rtc_prev_action_chunk"]:
        np.testing.assert_array_equal(row, prev)
    for key in ("rtc_prefix_weights", "rtc_freeze_mask", "rtc_max_guidance_weight"):
        np.testing.assert_array_equal(recorded[key], np.repeat(recorded[key][:1], num_samples, axis=0))


def test_batched_rtc_helpers_match_individual_rows() -> None:
    batch_size, horizon, action_dim = 3, 2, 2
    x_t = jnp.arange(batch_size * horizon * action_dim, dtype=jnp.float32).reshape(batch_size, horizon, action_dim)
    prev = -x_t
    weights = jnp.repeat(jnp.array([[1.0, 0.25]], dtype=jnp.float32), batch_size, axis=0)
    freeze_mask = jnp.repeat(jnp.array([[True, False]]), batch_size, axis=0)
    max_guidance_weight = jnp.full((batch_size,), 5.0, dtype=jnp.float32)

    def velocity(value):
        return value * 0.25

    batched_frozen = rtc_jax.freeze_prefix(x_t, prev, freeze_mask)
    batched_velocity = rtc_jax.guided_step_velocity(velocity, x_t, jnp.float32(0.5), prev, weights, max_guidance_weight)
    for row in range(batch_size):
        single_frozen = rtc_jax.freeze_prefix(x_t[row : row + 1], prev[row : row + 1], freeze_mask[row])
        single_velocity = rtc_jax.guided_step_velocity(
            velocity,
            x_t[row : row + 1],
            jnp.float32(0.5),
            prev[row : row + 1],
            weights[row],
            max_guidance_weight[row],
        )
        np.testing.assert_allclose(batched_frozen[row], single_frozen[0])
        np.testing.assert_allclose(batched_velocity[row], single_velocity[0])


def test_rtc_raw_actions_are_per_sample_and_pre_output_transform() -> None:
    policy = _policy.Policy(
        _StubJaxModel(),
        rng=jax.random.key(0),
        output_transforms=[_ScaleActions(2.0)],
    )
    noise = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

    results = policy.infer_samples(_observation(), 3, noise=noise)

    for sample_index, result in enumerate(results):
        expected_raw = noise[sample_index] + np.array([1.0, -2.0], dtype=np.float32)
        np.testing.assert_allclose(result["rtc_raw_actions"], expected_raw)
        np.testing.assert_allclose(result["actions"], expected_raw * 2.0)


class _StubTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def sample_actions(self, device, observation, *, noise=None):
        del device, noise
        self.calls += 1
        return torch.full(
            (observation.state.shape[0], 2, 2),
            float(self.calls),
            dtype=observation.state.dtype,
            device=observation.state.device,
        )


def test_pytorch_infer_samples_uses_sequential_fallback() -> None:
    model = _StubTorchModel()
    policy = _policy.Policy(model, is_pytorch=True)

    results = policy.infer_samples(_observation(), 3)

    assert model.calls == 3
    assert len(results) == 3
    for call_index, result in enumerate(results, start=1):
        np.testing.assert_array_equal(result["actions"], np.full((2, 2), call_index))


def test_medoid_policy_prefers_batched_interface() -> None:
    class _BatchedPolicy:
        def __init__(self):
            self.batched_calls = 0

        def infer(self, obs, **kwargs):
            raise AssertionError("sequential infer should not be called")

        def infer_samples(self, obs, num_samples, **kwargs):
            del obs, kwargs
            self.batched_calls += 1
            assert num_samples == 3
            return [
                {"actions": np.array([[0.0]])},
                {"actions": np.array([[2.0]])},
                {"actions": np.array([[10.0]])},
            ]

    base_policy = _BatchedPolicy()
    result = _policy.MedoidPolicy(base_policy, num_samples=3).infer({})

    assert base_policy.batched_calls == 1
    np.testing.assert_array_equal(result["actions"], np.array([[2.0]]))


def test_medoid_policy_keeps_sequential_fallback() -> None:
    class _SequentialPolicy:
        def __init__(self):
            self.calls = 0

        def infer(self, obs, **kwargs):
            del obs, kwargs
            actions = [0.0, 2.0, 10.0][self.calls]
            self.calls += 1
            return {"actions": np.array([[actions]])}

    base_policy = _SequentialPolicy()
    result = _policy.MedoidPolicy(base_policy, num_samples=3).infer({})

    assert base_policy.calls == 3
    np.testing.assert_array_equal(result["actions"], np.array([[2.0]]))
