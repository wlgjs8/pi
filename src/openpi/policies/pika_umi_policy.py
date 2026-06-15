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

        inputs = {
            "state": data["observation/state"],
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
