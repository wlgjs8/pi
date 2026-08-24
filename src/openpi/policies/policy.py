from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Real-Time Chunking (RTC) guidance fields the client may attach to the obs dict.
# They are NOT model observation fields: they are popped before the input
# transforms / Observation.from_dict and forwarded straight to sample_actions.
# `prev_action_chunk` (H x action_dim, model action units) is the gate -- absent
# means RTC is OFF. See openpi.models_pytorch.rtc and robotics_lab/docs/rtc_design.md.
_RTC_OBS_KEYS = (
    "prev_action_chunk",
    "inference_delay",
    "execute_horizon",
    "prefix_attention_schedule",
    "max_guidance_weight",
)
_RTC_SCALAR_KEYS = _RTC_OBS_KEYS[1:]  # everything except prev_action_chunk


def _pop_rtc_kwargs(inputs: dict) -> dict:
    """Remove the RTC guidance fields from the obs dict and return them.

    Returns an empty dict (RTC OFF) unless ``prev_action_chunk`` is present.
    Mutates ``inputs`` (a per-call copy in ``infer``), keeping the model
    observation keys intact for the transforms.
    """
    if not isinstance(inputs, dict) or "prev_action_chunk" not in inputs:
        return {}
    return {key: inputs.pop(key) for key in _RTC_OBS_KEYS if key in inputs}


def select_medoid(chunks: list[np.ndarray]) -> int:
    """Return the index of the chunk with minimum summed L2 distance to the others."""
    if not chunks:
        raise ValueError("select_medoid requires at least one action chunk")
    flat = np.stack([np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in chunks])
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2).sum(axis=1)
    return int(distances.argmin())


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng if rng is not None else jax.random.key(0)
        # RTC guided sampling is available when sample_actions accepts the RTC
        # array kwargs (JAX pi0/pi05 flow; pi0-FAST autoregressive has none).
        self._jax_rtc_supported = (not is_pytorch) and (
            "rtc_prev_action_chunk" in inspect.signature(model.sample_actions).parameters
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # Pull optional RTC guidance fields out before the transforms (they are
        # not model observation fields). Empty unless the client sent a prev chunk.
        rtc_kwargs = _pop_rtc_kwargs(inputs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # RTC: forward the guidance fields to sample_actions. Supported by the
        # PyTorch models (models_pytorch.rtc) and the JAX pi0/pi05 flow sampler
        # (models.rtc_jax); ignored (RTC OFF) for models without an RTC path.
        if rtc_kwargs:
            if self._is_pytorch_model:
                prev = np.asarray(rtc_kwargs["prev_action_chunk"], dtype=np.float32)
                prev_t = torch.from_numpy(prev).to(self._pytorch_device)
                if prev_t.ndim == 2:  # (H, action_dim) -> add batch dim
                    prev_t = prev_t[None, ...]
                sample_kwargs["prev_action_chunk"] = prev_t
                for key in _RTC_SCALAR_KEYS:
                    if key in rtc_kwargs:
                        sample_kwargs[key] = rtc_kwargs[key]
            elif self._jax_rtc_supported:
                # JAX RTC: convert the guidance fields to fixed-shape arrays on
                # the host (see openpi.models.rtc_jax) — delay/schedule changes
                # therefore never retrace the jitted sampler.
                from openpi.models import rtc_jax as _rtc_jax

                prev = np.asarray(rtc_kwargs["prev_action_chunk"], dtype=np.float32)
                if prev.ndim == 2:
                    prev = prev[None, ...]
                horizon = int(prev.shape[-2])
                delay = int(rtc_kwargs.get("inference_delay", 0))
                execute_horizon = int(rtc_kwargs.get("execute_horizon", horizon))
                schedule = str(rtc_kwargs.get("prefix_attention_schedule", "exp"))
                weights = _rtc_jax.get_prefix_weights_np(delay, horizon - execute_horizon, horizon, schedule)
                sample_kwargs["rtc_prev_action_chunk"] = jnp.asarray(prev)
                sample_kwargs["rtc_prefix_weights"] = jnp.asarray(weights)
                sample_kwargs["rtc_freeze_mask"] = jnp.asarray(np.arange(horizon) < delay)
                sample_kwargs["rtc_max_guidance_weight"] = jnp.asarray(
                    float(rtc_kwargs.get("max_guidance_weight", 5.0)), dtype=jnp.float32
                )
            else:
                logging.warning("RTC fields ignored: this model's sample_actions has no RTC path")

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # RTC seeding: stash the model-space action chunk (BEFORE the output
        # transform, i.e. the normalized units sample_actions denoises in) so the
        # client can round-trip it back as `prev_action_chunk` next call. It is
        # opaque to the client -- NOT un-normalized and NOT gripper-rescaled, so
        # the freeze/guidance run in the same space the model sampled. This is
        # returned for PyTorch and for JAX models that expose the RTC sampler.
        rtc_raw_actions = (
            np.asarray(outputs["actions"]).copy() if (self._is_pytorch_model or self._jax_rtc_supported) else None
        )

        outputs = self._output_transform(outputs)
        if rtc_raw_actions is not None:
            outputs["rtc_raw_actions"] = rtc_raw_actions
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_samples(self, obs: dict, num_samples: int, *, noise: np.ndarray | None = None) -> list[dict]:
        """Sample multiple chunks in one JAX model call.

        Input transforms run once on the unbatched observation, then every model
        observation and RTC leaf is explicitly tiled along the model batch axis.
        PyTorch keeps the established sequential inference path.
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if num_samples == 1:
            return [self.infer(obs, noise=noise)]

        if self._is_pytorch_model:
            if noise is None or np.asarray(noise).ndim == 2:
                return [self.infer(obs, noise=noise) for _ in range(num_samples)]
            batched_noise = np.asarray(noise)
            if batched_noise.ndim != 3 or batched_noise.shape[0] != num_samples:
                raise ValueError(
                    f"batched noise must have shape ({num_samples}, horizon, action_dim), got {batched_noise.shape}"
                )
            return [self.infer(obs, noise=batched_noise[i]) for i in range(num_samples)]

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc_kwargs = _pop_rtc_kwargs(inputs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[None, ...], num_samples, axis=0), inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            batched_noise = jnp.asarray(noise)
            if batched_noise.ndim == 2:
                batched_noise = jnp.repeat(batched_noise[None, ...], num_samples, axis=0)
            elif batched_noise.ndim != 3 or batched_noise.shape[0] != num_samples:
                raise ValueError(
                    f"batched noise must have shape ({num_samples}, horizon, action_dim), got {batched_noise.shape}"
                )
            sample_kwargs["noise"] = batched_noise

        if rtc_kwargs:
            if self._jax_rtc_supported:
                from openpi.models import rtc_jax as _rtc_jax

                prev = np.asarray(rtc_kwargs["prev_action_chunk"], dtype=np.float32)
                if prev.ndim == 3 and prev.shape[0] == 1:
                    prev = prev[0]
                if prev.ndim != 2:
                    raise ValueError(
                        "prev_action_chunk must have shape (horizon, action_dim) or "
                        f"(1, horizon, action_dim), got {prev.shape}"
                    )
                horizon = int(prev.shape[-2])
                delay = int(rtc_kwargs.get("inference_delay", 0))
                execute_horizon = int(rtc_kwargs.get("execute_horizon", horizon))
                schedule = str(rtc_kwargs.get("prefix_attention_schedule", "exp"))
                weights = _rtc_jax.get_prefix_weights_np(delay, horizon - execute_horizon, horizon, schedule)
                freeze_mask = np.arange(horizon) < delay
                max_guidance_weight = float(rtc_kwargs.get("max_guidance_weight", 5.0))

                # RTC guidance is identical for every stochastic draw. Tile all
                # leaves explicitly so the sampler never relies on batch broadcasting.
                sample_kwargs["rtc_prev_action_chunk"] = jnp.repeat(jnp.asarray(prev)[None, ...], num_samples, axis=0)
                sample_kwargs["rtc_prefix_weights"] = jnp.repeat(jnp.asarray(weights)[None, ...], num_samples, axis=0)
                sample_kwargs["rtc_freeze_mask"] = jnp.repeat(jnp.asarray(freeze_mask)[None, ...], num_samples, axis=0)
                sample_kwargs["rtc_max_guidance_weight"] = jnp.full(
                    (num_samples,), max_guidance_weight, dtype=jnp.float32
                )
            else:
                logging.warning("RTC fields ignored: this model's sample_actions has no RTC path")

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng, observation, **sample_kwargs)
        model_time = time.monotonic() - start_time
        batched_outputs = jax.tree.map(
            np.asarray,
            {
                "state": inputs["state"],
                "actions": actions,
            },
        )

        results = []
        for sample_index in range(num_samples):
            outputs = jax.tree.map(lambda x, index=sample_index: np.asarray(x[index, ...]), batched_outputs)

            # Preserve each chosen draw in model action units before its own
            # output transform; RTC clients must round-trip the medoid's raw chunk.
            rtc_raw_actions = np.asarray(outputs["actions"]).copy() if self._jax_rtc_supported else None
            outputs = self._output_transform(outputs)
            if rtc_raw_actions is not None:
                outputs["rtc_raw_actions"] = rtc_raw_actions
            outputs["policy_timing"] = {"infer_ms": model_time * 1000}
            results.append(outputs)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


class MedoidPolicy(_base_policy.BasePolicy):
    """Sample N action chunks per `infer` and return the MEDOID — the draw most central among the
    N (min summed L2 over the flattened chunk). A deployable consensus selector (no GT, no value
    function) that commits to the dominant mode, reducing the per-step mode-switching / indecision
    a single stochastic draw causes in a multimodal flow policy (a different valid mode each step →
    the gripper oscillates between targets and never commits). Diagnostic 1 (robotics-lab-pickplace-eval)
    found the medoid recovers ~45-51% of the mean→oracle best-of-N gap. `num_samples=1` is a no-op
    (plain single-draw behavior).
    """

    def __init__(self, policy: _base_policy.BasePolicy, num_samples: int = 8):
        self._policy = policy
        self._n = max(1, int(num_samples))

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        if self._n == 1:
            return self._policy.infer(obs, **kwargs)
        infer_samples = getattr(self._policy, "infer_samples", None)
        if callable(infer_samples):
            results = infer_samples(obs, self._n, **kwargs)
        else:
            results = [self._policy.infer(obs, **kwargs) for _ in range(self._n)]
        chunks = [np.asarray(r["actions"], dtype=np.float64) for r in results]
        return results[select_medoid(chunks)]

    @property
    def metadata(self) -> dict[str, Any]:
        return getattr(self._policy, "metadata", {})
