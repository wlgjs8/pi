import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model


def _anchor_relative_chunk(actions: np.ndarray) -> np.ndarray:
    """UMI-style anchored relative trajectory (load-time, action_mode='anchored').

    `actions` is the (H, 14) chunk the loader stacked from the dataset, where each row is the per-frame
    tool-frame ABSOLUTE pose `[L p3, L rotvec3, L grip, R p3, R rotvec3, R grip]` (the converter stored
    abs poses for action_mode=anchored). Re-express every row RELATIVE to the chunk's first frame (the
    anchor T_t): `pos_k = R0^-1 (p_k - p0)`, `rot_k = rotvec(R0^-1 R_k)`; gripper passes through. Row 0 is
    thus ~identity. This is invariant to any global rigid transform (like ee_local), and at deploy each row
    composes INDEPENDENTLY onto the live anchor -> no per-step delta integration / drift."""
    a = np.asarray(actions, dtype=np.float64)
    out = a.copy()
    for base in (0, 7):  # left arm cols 0:7, right arm cols 7:14
        p = a[:, base : base + 3]
        R = Rotation.from_rotvec(a[:, base + 3 : base + 6])
        r0_inv = R[0].inv()
        out[:, base : base + 3] = r0_inv.apply(p - p[0])
        out[:, base + 3 : base + 6] = (r0_inv * R).as_rotvec()
        # gripper col (base+6) unchanged -- it is the per-frame target opening
    return out.astype(np.float32)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PikaUmiInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    # If true, also feed the RealSense depth (pre-encoded as a normalized 3-channel
    # image by the converter) as extra `*_wrist_0_depth` camera inputs (RGB-D).
    include_depth: bool = False
    # If true, NEUTRALIZE proprio: zero the state so its discrete State: tokens become a
    # constant (carries no pose info). UMI handheld init pose is arbitrarily rotated per
    # episode, so reset-relative proprio is anchored to a per-episode-rotated t0 frame and is
    # NOT ego-centric -> a frame-inconsistent channel. Dropping it forces a purely vision-driven
    # (genuinely ego-centric wrist-cam) policy. Matches the .8 baseline (zero_state=True).
    zero_state: bool = False
    # Action representation. "delta" = per-step ee_local delta (default; chunk = independent 1-step deltas,
    # integrated/accumulated at deploy). "anchored" = UMI relative trajectory: the dataset stores per-frame
    # ABSOLUTE poses; here we re-express the chunk relative to its first frame -> each row composes onto the
    # live anchor at deploy with NO integration. MUST match the converter's --action-mode.
    action_mode: str = "delta"

    def __call__(self, data: dict) -> dict:
        left_wrist = _parse_image(data["observation/left_wrist_0_rgb"])
        right_wrist = _parse_image(data["observation/right_wrist_0_rgb"])
        base_image = np.zeros_like(left_wrist)

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_mask = {
            "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        if self.include_depth:
            images["left_wrist_0_depth"] = _parse_image(data["observation/left_wrist_0_depth"])
            images["right_wrist_0_depth"] = _parse_image(data["observation/right_wrist_0_depth"])
            image_mask["left_wrist_0_depth"] = np.True_
            image_mask["right_wrist_0_depth"] = np.True_

        state = data["observation/state"]
        if self.zero_state:
            state = np.zeros_like(state)  # proprio neutralized -> constant State: tokens, no pose info

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_mask,
        }

        if "actions" in data:
            acts = data["actions"]
            if self.action_mode == "anchored":
                acts = _anchor_relative_chunk(acts)  # abs-pose chunk -> anchored relative trajectory
            inputs["actions"] = acts
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PikaUmiOutputs(transforms.DataTransformFn):
    # Real action dim sliced from the model's padded output: 14 (dual arm) or 7 (single right arm).
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
