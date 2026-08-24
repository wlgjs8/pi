from __future__ import annotations

import types

import jax.numpy as jnp

from openpi.models import model as _model
from scripts import serve_policy


def test_jax_warmup_uses_requested_batch_size(caplog) -> None:
    seen = {}

    class _FakeModelConfig:
        def fake_obs(self, batch_size):
            seen["fake_obs_batch_size"] = batch_size
            return _model.Observation(
                images={},
                image_masks={},
                state=jnp.ones((batch_size, 2), dtype=jnp.float32),
            )

    def sample_actions(rng, observation, **kwargs):
        del rng, kwargs
        seen["sample_batch_size"] = observation.state.shape[0]
        return jnp.ones((observation.state.shape[0], 2, 2), dtype=jnp.float32)

    policy = types.SimpleNamespace(
        _sample_kwargs={},
        _is_pytorch_model=False,
        _sample_actions=sample_actions,
    )
    train_config = types.SimpleNamespace(model=_FakeModelConfig())

    with caplog.at_level("INFO"):
        serve_policy.warmup_policy(policy, train_config, batch_size=4)

    assert seen == {"fake_obs_batch_size": 4, "sample_batch_size": 4}
    assert "batch_size=4" in caplog.text
