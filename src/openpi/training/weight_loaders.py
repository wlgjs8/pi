import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = _interpolate_posemb(loaded_params, params)
        loaded_params = _slice_trunk_layers(loaded_params, params)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _interpolate_posemb(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Resize any learned ViT position embedding whose token count does not match the model.

    SigLIP stores `pos_embedding` as (1, h*w, width) for a fixed patch grid: 14x14=196 at 224,
    24x24=576 at 384. Training at a new resolution from a 224 checkpoint therefore needs the grid
    resampled, which is exactly what the released 224->384 SigLIP variants did before fine-tuning.
    Bicubic on the 2-D grid; the checkpoint is left untouched when the shapes already agree.

    Without this, `_merge_params` copies the 196-token embedding into a 576-token slot purely
    because the KEY matches -- it never compares shapes -- and the failure would surface far from
    its cause.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    changed = 0
    for k, v in list(flat_loaded.items()):
        if not k.endswith("pos_embedding") or k not in flat_ref:
            continue
        ref = flat_ref[k]
        if v.shape == ref.shape:
            continue
        n_old, n_new = v.shape[-2], ref.shape[-2]
        g_old, g_new = int(round(n_old ** 0.5)), int(round(n_new ** 0.5))
        if g_old * g_old != n_old or g_new * g_new != n_new or v.shape[-1] != ref.shape[-1]:
            raise ValueError(
                f"cannot interpolate {k}: {v.shape} -> {ref.shape} (non-square or width mismatch)")
        width = v.shape[-1]
        grid = np.asarray(v, dtype=np.float32).reshape(g_old, g_old, width)
        # jax.image.resize handles the (h, w, c) grid directly and matches the reference impls.
        out = np.asarray(jax.image.resize(grid, (g_new, g_new, width), method="bicubic"))
        flat_loaded[k] = out.reshape(1, n_new, width).astype(v.dtype)
        logging.info("pos_embedding %s interpolated %dx%d -> %dx%d", k, g_old, g_old, g_new, g_new)
        changed += 1
    if changed:
        return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
    return loaded_params


def _slice_trunk_layers(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Take a STRIDED subset of a scan-stacked transformer's layers when the model is shallower
    than the checkpoint.

    openpi stacks the depth axis into leading dimension 0 of every `llm/layers/...` parameter, so a
    depth-reduced trunk differs from the checkpoint only in that axis. Layers are selected with an
    even stride rather than truncated: keeping layers 0..5 of an 18-layer stack keeps only the
    early-feature end of the network, while stride-3 keeps the whole depth-wise progression, which
    is what layer-dropping work does and what a *dedicated but small* vision stack wants.

    Silent-failure guard: `_merge_params` copies a tensor whenever the KEY matches and never looks
    at the shape, so without this a 18-layer parameter would be written into a 9-layer slot and the
    error would surface far from its cause -- the same trap the pos-embedding interpolation exists
    to close.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    picked = None
    for k, v in list(flat_loaded.items()):
        if "/layers/" not in k or k not in flat_ref:
            continue
        ref = flat_ref[k]
        if np.ndim(v) == 0 or np.shape(v) == np.shape(ref):
            continue
        n_old, n_new = np.shape(v)[0], np.shape(ref)[0]
        if np.shape(v)[1:] != np.shape(ref)[1:] or n_new >= n_old:
            raise ValueError(f"cannot slice layers for {k}: {np.shape(v)} -> {np.shape(ref)}")
        idx = np.linspace(0, n_old - 1, n_new).round().astype(int)
        if picked is None:
            picked = idx
            logging.info("trunk depth %d -> %d, keeping layers %s", n_old, n_new, idx.tolist())
        flat_loaded[k] = np.asarray(v)[idx]
    if picked is None:
        return loaded_params
    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
