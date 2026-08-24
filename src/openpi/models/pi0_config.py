import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Real width of the action vector. `action_dim` is the pi05_base tensor width (32); our data is
    # 14 and transforms.pad_to_dim zero-fills the rest, so an unweighted mean over `action_dim`
    # spends 18/32 = 56% of the flow-matching loss denoising a constant zero. Behaviour is unharmed
    # (the output transform slices back to 14) but the signal is diluted and the loss scalar becomes
    # useless for comparing checkpoints -- measured: E1 train loss 0.0106 -> 0.0093 from 60k to 80k
    # while val chunk nMSE went 0.6453 -> 0.6462, i.e. no generalization change at all.
    # None keeps the historical behaviour (average over all `action_dim`).
    loss_action_dim: int | None = None

    # ---- gripper loss ------------------------------------------------------------------
    # WHY. The flow loss averages uniformly over the 24 horizon rows. The gripper channel is
    # constant (held open, or held closed) in the overwhelming majority of those rows, so the
    # gradient on it is dominated by frames whose answer is "keep doing what you are doing".
    # The frames that actually carry the decision -- the open->close at the grasp and the
    # close->open at the release -- are a few percent of the data and are drowned out.
    #
    # That is not a hypothetical. The measured consequence (llm-wiki grip-echo diagnosis,
    # 2026-08-16) is that the grip target is ~99% explained by "current state grip - epsilon":
    # a copy of proprio, with a copy-proprio floor of 0.336. On the robot the commanded grip
    # became a pure echo of the measured one (corr 0.993, cmd ~= meas - 4.7%/tick) and the
    # gripper closed unconditionally before it reached the bolt. In simulation the same policy
    # issues ~7 close commands per episode of which ~90% grasp nothing.
    #
    # HOW. Up-weight the gripper dimensions on rows near a ground-truth grip transition:
    #     w = 1 + (grip_loss_weight - 1) * dilate(|d grip|>0, grip_loss_window)
    # then renormalise the whole weight tensor to mean 1 per sample, so the loss SCALE is
    # unchanged and only its DISTRIBUTION over rows/dims moves. Keeping the scale fixed is what
    # makes this a controlled A/B: the LR schedule still means the same thing.
    # None = off (historical behaviour).
    grip_loss_weight: float | None = None
    # Rows either side of a transition that inherit the weight. The grasp decision is not a
    # single frame -- the approach rows leading into it carry the same information.
    grip_loss_window: int = 4
    # Indices of the gripper dims in the action vector. pika_umi = 14-D, [pos3 rot3 grip] per arm.
    grip_dims: tuple[int, ...] = (6, 13)
    # Same mechanism aimed at the AIM channel instead of the gripper. The grasp is the moment the
    # gripper closes, so the grip transition already marks it; weighting the TRANSLATION dims on
    # those same rows concentrates supervision on the approach that decides where the fingers land.
    # This is the lever the measurement points at: closed-loop lateral aim is ~18 mm p50 against an
    # 18.4 mm bolt head, grasps succeed at 9 mm and fail at 18 mm, and beyond 20 mm nothing is ever
    # picked up. The VA line's equivalent (grasp-weighted loss) was its best offline arm; pi05 has
    # never had it. None = off.
    pose_loss_weight: float | None = None
    # Translation dims per arm: [pos3 rot3 grip] x 2.
    pose_dims: tuple[int, ...] = (0, 1, 2, 7, 8, 9)

    # ---- input resolution --------------------------------------------------------------
    # SigLIP is a 16x16-patch ViT, so 224 -> 14x14 = 196 tokens per image and 384 -> 24x24 = 576.
    # Raising this is a RESIZE OF THE WHOLE FRAME, not a crop: the full wrist FOV is kept and the
    # bolt simply lands on ~2.9x more pixels. That matters here because the measured failure is a
    # ~20 mm lateral aim scatter while the M12 head is 18.4 mm wide -- the target is a handful of
    # pixels at 224.
    # COSTS, so this is not a free knob: prefix tokens go 392 -> 1152 for two wrist images, and
    # pi05_base's LEARNED pos_embedding is (1, 196, width) -- it must be interpolated to the new
    # grid at load time (see weight_loaders._interpolate_posemb), which is what SigLIP's own
    # 224->384 variants do before fine-tuning.
    image_resolution: tuple[int, int] = (224, 224)

    pytorch_compile_mode: str | None = "max-autotune"

    # Camera image keys the model consumes (in order). Default = base + two wrist RGB.
    # Extend to add extra image inputs (e.g. depth-as-image) WITHOUT breaking existing
    # RGB-only checkpoints, which keep the default.
    image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # AUXILIARY color head: if > 0, add a small MLP on the per-wrist SigLIP features that classifies
    # the target bolt's color (black/gray), with weight `aux_color_weight` on its cross-entropy loss.
    # 0.0 = off (default; existing checkpoints unaffected). Forces the vision features the action
    # expert cross-attends to encode color (the encoder already separates black/silver at ~94%).
    aux_color_weight: float = 0.0

    # Train-time PHOTOMETRIC augmentation (brightness/contrast/saturation jitter) in the model's image
    # preprocess. Default True (legacy). Set False for COLOR-critical tasks: the jitter randomizes exactly
    # the brightness cue that separates black vs (shiny) silver bolts, training the model to ignore it.
    photometric_aug: bool = True

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *self.image_resolution, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={k: image_spec for k in self.image_keys},
                image_masks={k: image_mask_spec for k in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
