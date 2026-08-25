import dataclasses
import re

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

# phase_color prompt -> (color, arm), e.g. "...the black bolt with the right arm...".
_BOLT_COLOR_RE = re.compile(r"the (black|gray) bolt with the (right|left) arm")


def _bolt_color_labels(prompt) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-frame AUXILIARY color labels from the phase_color prompt (free signal, no extra
    dataset columns). Returns (right_label, left_label) each int32[1]; 0=black, 1=gray, -1=ignore.
    Only the arm named in the prompt (the one whose wrist is looking at its target bolt this phase)
    gets a label; the other arm is -1 (ignored by the aux loss). Non-phase_color prompts -> both -1."""
    right, left = -1, -1
    if isinstance(prompt, str | bytes):
        s = prompt.decode() if isinstance(prompt, bytes) else prompt
        m = _BOLT_COLOR_RE.search(s)
        if m:
            c = 0 if m.group(1) == "black" else 1
            if m.group(2) == "right":
                right = c
            else:
                left = c
    return np.array([right], dtype=np.int32), np.array([left], dtype=np.int32)


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
        rot = Rotation.from_rotvec(a[:, base + 3 : base + 6])
        r0_inv = rot[0].inv()
        out[:, base : base + 3] = r0_inv.apply(p - p[0])
        out[:, base + 3 : base + 6] = (r0_inv * rot).as_rotvec()
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
    # If true, parse the phase_color prompt into per-arm bolt-color labels (0=black,1=gray,-1=ignore)
    # and pass them through as `bolt_color_right`/`bolt_color_left` for the model's auxiliary color head.
    aux_color_labels: bool = False
    # If true, do not emit the `base_0_rgb` slot at all. There is NO third-person/head camera in the
    # PIKA UMI captures, so that slot has always been an all-zero placeholder with image_mask=False:
    # excluded from attention, yet SigLIP still encodes it every step (256 wasted tokens = 1/3 of a
    # wrist-only prefix). Dropping it is free compute/latency, and inference latency is what bounds the
    # chunk replan budget. Requires the model's `image_keys` to omit base_0_rgb too. Keep False for
    # PI0_FAST (which masks the placeholder True) and for every legacy checkpoint, whose prefix layout
    # would otherwise change under it.
    drop_base_image: bool = False
    # Zero ONLY the ee_local velocity dims of a `velocity_grip` (14-D) state, keeping the absolute
    # gripper opening at dims 6/13. Rationale: the velocity channel is fed from the MEASURED robot
    # motion at deploy, and the controller only delivers ~0.87 of the commanded per-step displacement
    # (measured p50 over the deploy step logs; 35% of moving steps land under 0.70), so the policy's
    # own tracking shortfall is fed back into its next observation -- a contractive loop that the
    # demonstrations never contained (a handheld UMI capture has no servo lag). Zeroing those dims
    # keeps the gripper feedback (which fixes the mid-air open/close chatter) while removing the
    # controller-coupled channel entirely.
    # Implementation note: this masks at TRANSFORM time, so it reuses the velocity_grip dataset AND its
    # norm stats unchanged, and at deploy the SERVER applies the mask -- the client can keep sending
    # `--proprio-mode velocity_grip` with no runtime change. The zeroed dims normalize to a constant
    # (quantile norm is asymmetric, so not exactly 0), which is still information-free.
    drop_velocity_proprio: bool = False
    # State dims that carry the absolute gripper opening in the 14-D velocity_grip layout
    # ([pos_vel3, rot_vel3, grip] per arm, left block then right block).
    grip_state_dims: tuple[int, ...] = (6, 13)
    # Blank the TASK SENTENCE while keeping everything else about the language branch intact.
    # This is the language ablation, and it has to be done here rather than by dropping
    # `tokenized_prompt`: pi05 sets discrete_state_input=True, so the tokenizer packs the state INTO
    # the prompt as "Task: {text}, State: {14 ints};\nAction: ". Dropping the whole tokenized prompt
    # (which Pi0.embed_prefix supports -- it skips the language branch when tokenized_prompt is None)
    # would therefore remove the STATE as well and stop being a single-variable ablation. Emitting an
    # empty task string keeps the state tokens, the scaffolding and the token budget identical and
    # removes only the sentence (measured: 98 -> 64 informative tokens).
    # Motivation: the dataset carries ONE prompt string across all 177,945 frames and it never varies,
    # so the text branch is predicted to carry zero discriminative signal. That prediction is the
    # premise the whole VA (no-LLM) line was built on and it has never been measured directly.
    drop_task_prompt: bool = False

    def __call__(self, data: dict) -> dict:
        left_wrist = _parse_image(data["observation/left_wrist_0_rgb"])
        right_wrist = _parse_image(data["observation/right_wrist_0_rgb"])

        images = {
            "left_wrist_0_rgb": left_wrist,
            "right_wrist_0_rgb": right_wrist,
        }
        image_mask = {
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        if not self.drop_base_image:
            # Zero placeholder, masked off for pi0/pi05 (only pi0-FAST consumes it).
            images = {"base_0_rgb": np.zeros_like(left_wrist), **images}
            image_mask = {
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                **image_mask,
            }
        if self.include_depth:
            images["left_wrist_0_depth"] = _parse_image(data["observation/left_wrist_0_depth"])
            images["right_wrist_0_depth"] = _parse_image(data["observation/right_wrist_0_depth"])
            image_mask["left_wrist_0_depth"] = np.True_
            image_mask["right_wrist_0_depth"] = np.True_

        state = data["observation/state"]
        if self.zero_state:
            state = np.zeros_like(state)  # proprio neutralized -> constant State: tokens, no pose info
        elif self.drop_velocity_proprio:
            # Keep ONLY the absolute gripper dims; zero the controller-coupled velocity dims.
            state = np.asarray(state)
            keep = [d for d in self.grip_state_dims if d < state.shape[-1]]
            if len(keep) != len(self.grip_state_dims):
                raise ValueError(
                    f"drop_velocity_proprio expects a velocity_grip state with dims "
                    f"{self.grip_state_dims}, got width {state.shape[-1]} -- the dataset's "
                    f"--state-mode must be velocity_grip"
                )
            masked = np.zeros_like(state)
            masked[..., keep] = state[..., keep]
            state = masked

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
        # Episode-boundary padding flag for the action chunk (training only; see
        # Observation.action_is_pad). Passed straight through so the loss can drop those rows.
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]
        if self.drop_task_prompt:
            # Unconditional: the tokenizer requires a prompt key, and "" yields
            # "Task: , State: ...;\nAction: " -- state and scaffolding intact, sentence gone.
            inputs["prompt"] = ""
        elif "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.aux_color_labels:
            # Per-frame pre-grasp color labels are emitted by the converter (timed, from arm_bolt_colors).
            # Read them from the dataset columns; absent at inference -> -1 (aux loss is training-only).
            right_lbl = data.get("bolt_color_right")
            left_lbl = data.get("bolt_color_left")
            inputs["bolt_color_right"] = np.asarray(
                right_lbl if right_lbl is not None else [-1], dtype=np.int32
            ).reshape(1)
            inputs["bolt_color_left"] = np.asarray(left_lbl if left_lbl is not None else [-1], dtype=np.int32).reshape(
                1
            )

        return inputs


@dataclasses.dataclass(frozen=True)
class PikaUmiOutputs(transforms.DataTransformFn):
    # Real action dim sliced from the model's padded output: 14 (dual arm) or 7 (single right arm).
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
