import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


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
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PikaUmiOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
