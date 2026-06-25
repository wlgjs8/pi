"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.pika_umi_policy as pika_umi_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If set, train-only image augmentation applied in the training data pipeline (NOT at
    # inference or norm-stat computation, which build their transform stacks separately).
    image_aug: _transforms.ImageTransformConfig | None = None
    # If set, train-only DART-style state/action recovery-noise augmentation (covariate-shift /
    # compounding-error fix). Applied in the training pipeline only, on raw state/actions before norm.
    dart_noise: _transforms.DartNoiseConfig | None = None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If False, images are resized with a direct (no-pad, aspect-distorting) resize instead of the
    # default aspect-preserving resize_with_pad. Keeps full FOV + spends the whole 224x224 on the scene.
    resize_pad: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224, pad=self.resize_pad),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224, pad=self.resize_pad),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224, pad=self.resize_pad),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPikaUmiDataConfig(DataConfigFactory):
    # If true, also load the RealSense depth (pre-encoded as 3-channel images by the
    # converter) as extra `*_wrist_0_depth` camera inputs (RGB-D). Requires a dataset
    # converted with depth and a model whose `image_keys` include the depth keys.
    include_depth: bool = False
    # If set, center-crop each image to (center_crop, center_crop) instead of the default
    # aspect-preserving resize_with_pad — higher effective resolution on the central object.
    center_crop: int | None = None
    # If true, neutralize proprio (zero the state -> constant State: tokens) so the policy is
    # purely vision-driven. UMI per-episode-rotated init pose makes reset-relative proprio
    # non-ego-centric; dropping it removes that frame-inconsistent channel. Matches .8 baseline.
    zero_state: bool = False
    # If False, use a direct no-pad (aspect-distorting) image resize instead of resize_with_pad —
    # keeps full FOV and spends the whole 224x224 on the scene (more pixels on small bolts). The served
    # config carries this, so train/serve resize stay matched. (center_crop, if set, takes precedence.)
    resize_pad: bool = True
    # Action representation; MUST match the dataset's converter --action-mode. "delta" = per-step ee_local
    # delta (default). "anchored" = UMI relative trajectory (dataset stores abs poses; PikaUmiInputs
    # re-anchors the chunk to its first frame -> no deploy-time integration drift).
    action_mode: str = "delta"
    # Arm scope; MUST match the dataset's converter --arm. "dual" = 14-D both arms (default). "right" =
    # 7-D right-arm-only (single-arm policy; the model output is sliced to 7).
    arm: str = "dual"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_map = {
            "observation/left_wrist_0_rgb": "left_wrist_0_rgb",
            "observation/right_wrist_0_rgb": "right_wrist_0_rgb",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        if self.include_depth:
            repack_map["observation/left_wrist_0_depth"] = "left_wrist_0_depth"
            repack_map["observation/right_wrist_0_depth"] = "right_wrist_0_depth"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_map)])
        data_input_transforms = [pika_umi_policy.PikaUmiInputs(model_type=model_config.model_type, include_depth=self.include_depth, zero_state=self.zero_state, action_mode=self.action_mode)]
        if self.center_crop is not None:
            data_input_transforms.append(_transforms.CenterCropImages(self.center_crop, self.center_crop))
        data_transforms = _transforms.Group(
            inputs=data_input_transforms,
            outputs=[pika_umi_policy.PikaUmiOutputs(action_dim=7 if self.arm == "right" else 14)],
        )
        model_transforms = ModelTransformFactory(resize_pad=self.resize_pad)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/plaif/workspace/openpi_runs/pytorch_checkpoints/pi05_base",
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_pika_umi",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_openpi_train",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Same as pi05_pika_umi (relrel, action_horizon=8) but with train-only image
    # augmentation enabled and a longer schedule. Reuses the pi05_pika_umi norm stats.
    TrainConfig(
        name="pi05_pika_umi_aug",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_openpi_train",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi"),
            base_config=DataConfig(
                prompt_from_task=True,
                image_aug=_transforms.ImageTransformConfig(),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # RGB-D variant: same as pi05_pika_umi_aug (h8, relrel, aug) but also feeds the
    # RealSense depth as two extra `*_wrist_0_depth` camera inputs (model.image_keys +
    # data.include_depth). Needs a depth-converted dataset (plaif/pika_umi_openpi_rgbd)
    # and its own norm stats (compute_norm_stats after convert). aug stays RGB-only.
    TrainConfig(
        name="pi05_pika_umi_rgbd",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
            image_keys=(
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
                "left_wrist_0_depth",
                "right_wrist_0_depth",
            ),
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_openpi_rgbd",
            include_depth=True,
            base_config=DataConfig(
                prompt_from_task=True,
                image_aug=_transforms.ImageTransformConfig(),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Same as pi05_pika_umi_aug (h8, relrel, RGB aug) but center-crops each image to
    # 384x384 (cuts only ~48px top/bottom + L/R edges, keeping the lower-centre grasp
    # region) and then resize_with_pad downsamples 384->224 -> higher effective resolution
    # on the central bolt than full resize. Reuses the pi05_pika_umi dataset + norm stats.
    TrainConfig(
        name="pi05_pika_umi_aug_ccrop",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_openpi_train",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi"),
            center_crop=384,
            base_config=DataConfig(
                prompt_from_task=True,
                image_aug=_transforms.ImageTransformConfig(),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Speed/IO test: same model+relrel as pi05_pika_umi, but reads a VIDEO-backed dataset
    # (MP4 per episode/camera) hosted on the NFS storage server (HF_LEROBOT_HOME=/mnt/pika/lerobot,
    # repo_id plaif/pika_umi_video_test). num_workers bumped 2->12 to hide NFS read latency behind
    # compute. Used to compare data-loading throughput vs the local PNG-in-parquet (image-backed)
    # dataset. Run compute_norm_stats with the same HF_LEROBOT_HOME first.
    TrainConfig(
        name="pi05_pika_umi_video_test",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_test",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_test"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # "Real" no-aug run on the storage-server data with a held-out val split (9:1, seeded). Train
    # split = plaif/pika_umi_video_train (video/h264, NFS); the 10% val (plaif/pika_umi_video_val)
    # is evaluated separately. Same model/relrel as pi05_pika_umi. HF_LEROBOT_HOME=/mnt/pika/lerobot.
    TrainConfig(
        name="pi05_pika_umi_video",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Baseline redo on the full 429-episode storage dataset, 8:2 episode-level random split,
    # pretrained pi05_base, 40k, no-aug. Train = plaif/pika_umi_video_train_8020 (video/h264, NFS),
    # val = plaif/pika_umi_video_val_8020 (eval separately). HF_LEROBOT_HOME=/mnt/pika/lerobot.
    TrainConfig(
        name="pi05_pika_umi_video_8020",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_8020",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_8020"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Same as pi05_pika_umi_video_8020 but on the TOOL/TCP-frame dataset: the converter now applies
    # the tracker->robot-TCP tip offset (inv(T_tcp_umi_gripper), umi_retarget_eelocal.yaml) before the
    # ee_local math, so actions/proprio are at the robot TCP, not the raw Vive tracker. SAME episode
    # split as _8020 (seed=0, val_frac=0.2 -> 343/86) for an apples-to-apples tracker-vs-tcp comparison.
    # Train = plaif/pika_umi_video_train_tcp_8020, val = plaif/pika_umi_video_val_tcp_8020 (eval separately).
    # NOTE: recompute norm stats for this repo_id before training. HF_LEROBOT_HOME=/mnt/pika/lerobot.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_8020",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=8,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_8020",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_8020"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # Longer action horizon (24) on the SAME tcp dataset (no re-conversion: action_horizon is a model
    # param; openpi chunks the per-frame actions at load time). Motivated by real rollout: executing a
    # longer chunk (chunk-execute-steps) is much more accurate because it commits to ONE sampled mode
    # for longer instead of re-sampling a different mode (a different bolt) each replan -> averaging to
    # the pile center. chunk-execute is a deploy knob (<= horizon), so train the longer horizon and pick
    # execute at deploy via the per-chunk-position MSE. Only 3 segs (0.1% frames) are < 25 frames.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_8020_h24",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_8020",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_8020_h24"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # action_horizon=24 + gripper ABSOLUTE representation (from 8.8). Re-converted dataset
    # (`--gripper-action absolute --camera realsense`): action gripper dim = next-step absolute opening
    # (grip/100, ~0.13 closed-on-bolt .. ~0.98 open) instead of the per-step delta. Motivated by real
    # rollouts: the delta under-closed (closing relative to a varying start-open is multimodal). Absolute
    # makes grasp/release targets unimodal. DEPLOY CAVEAT: command the gripper as an ABSOLUTE opening, not
    # integrate a delta. Datasets: plaif/pika_umi_video_{train,val}_tcp_gripabs_8020.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_h24",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_8020",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_h24"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # A/B camera comparison at h24, BOTH ABSOLUTE gripper (`--gripper-action absolute`), target grip[t+1]/100.
    # Same tcp retarget + SAME 343/86 split as _8020 (`--split-in pika_umi_video_split_tcp_8020.json`) -> the
    # two runs differ ONLY in the wrist image -> isolates fisheye matching-gain vs RGB detail-loss. Recompute
    # norm-stats per repo_id. A) RealSense color, full frame (detail-rich, narrower FOV).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_8020_h24_rs_absgrip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_8020_rs_absgrip",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_8020_h24_rs_absgrip"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # B) Fisheye, center-cropped 0.65 -> (312,416) (`--camera fisheye --fisheye-crop-frac 0.65`): drops the
    # barrel-distorted/vignette periphery, spends the 224 resize budget on the central working area (better
    # small-bolt detail), matches the deploy fisheye cam. Cropped at CONVERSION time (baked into video).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_8020_h24_fe65_absgrip",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_8020_fe65_absgrip",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_8020_h24_fe65_absgrip"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # TAIL-PAD experiment: identical to ..._h24_fe65_absgrip but the TRAIN set is tail-padded
    # (`--tail-pad-frames 30`): freeze each episode's final frame with a hold-open action for +30 frames so
    # the rare "fully-open" left-gripper label (16-frame tail, terminal left release) is no longer starved.
    # Val is UNPADDED (honest test) -> eval on the existing ..._fe65_absgrip val to compare left_open detect
    # vs the non-padded 10k/20k/30k. Trains to 30k. See [[robotics-lab-pickplace-eval]] left_open root cause.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_8020_h24_fe65_absgrip_tp30",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=24,
        ),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_8020_fe65_absgrip_tp30",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_8020_h24_fe65_absgrip_tp30"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # BINARY gripper (open/closed @th25) + tail-pad 50 + action_horizon 50 + MORE DATA (458 handheld train,
    # val pinned to the gripabs 86 via --val-from-record). Binary strips the task-irrelevant open-angle noise
    # (~40-95 random) AND the under-close: deploy commands 0 -> firm close (bolt blocks at its width) / 1 ->
    # open, through the EXISTING absolute gripper path (no deploy change). tail-pad fixes the rare/truncated
    # left_open terminal label. horizon 50 (openpi default) = more closed-loop commitment (validated: deploy
    # chunk-execute 24 >> 8). proprio gripper stays ABSOLUTE. Eval reuses the gripabs 86 val with
    # --gripper-mode absolute (pose-only comparable; binary preds 0/1 -> 0/100%). See [[robotics-lab-pickplace-eval]].
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h50",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h50"),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # DEPTH (Option A): IDENTICAL to ..._binary_h50 (same binary gripper @25, ee_local action,
    # action_horizon=50) but with RGB-D — the realsense depth is added as extra wrist images
    # (include_depth=True -> *_wrist_0_depth through the SAME SigLIP, PikaUmiInputs). Needs its OWN
    # depth-converted dataset (storage converter `--include-depth --camera realsense`, default
    # z_near/z_far 120/700 mm, depth_units 1e-4) + its own assets_dir (norm-stats differ: more image
    # streams). Deploy: policy_runner must send the matching live D405 *_wrist_0_depth (same
    # _depth_to_image). To pair with velocity-proprio, merge with `--state-mode velocity` + the
    # velproprio dataset (the two are orthogonal: state-rep vs image-streams). Targets the
    # nostate↔depth coupling (wiki nonllm-rgbd-flow-aug / vla-rollout-diagnosis).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h50_depth",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50_depth",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h50_depth"),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # PROPRIO-FREE h24: IDENTICAL to ..._binary_h50 baseline (same binary_h50 dataset, same
    # action_dim=32, zero_state=True) EXCEPT action_horizon 50->24. A/B isolates the chunk-commitment
    # horizon under the proprio-neutralized regime (.8 runs the h50 baseline; this is the .13 h24 arm).
    # zero_state matches .8: UMI per-episode-rotated init pose makes reset-relative proprio non-ego-
    # centric, so the State: channel is dropped -> purely vision-driven (genuinely ego-centric wrist cams).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h24_nostate",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h24_nostate"),
            zero_state=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # PROPRIO-FREE h50 baseline (the .8 arm): IDENTICAL to ..._binary_h50 (same binary_h50 dataset,
    # same assets_dir -> reuses its norm_stats, no recompute; zeroing is pre-Normalize) + zero_state=True.
    # action_horizon=50. Mirrors the 8.8-committed config so the deploy PC can serve binary_h50_nostate.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h50_nostate",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h50"),
            zero_state=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # DART experiment: IDENTICAL to ..._binary_h50 (same dataset, no re-conversion) + train-only
    # DART-style recovery-noise augmentation (covariate-shift / compounding-error). Targets the
    # 2026-06-22 diagnosis (teacher-forced OK, rollout 50/50 approach = closed-loop §5). Reuses the
    # binary_h50 LeRobot dataset; recompute norm-stats for this assets_dir (DART is applied AFTER
    # PikaUmiInputs / BEFORE norm, so norm-stats are computed clean and match binary_h50).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h50_dart",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50",
            assets=AssetsConfig(assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h50_dart"),
            base_config=DataConfig(
                prompt_from_task=True,
                dart_noise=_transforms.DartNoiseConfig(sigma_pos_m=0.01, recover_steps=5, prob=0.5),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # VELOCITY-PROPRIO ablation (this PC, 8x RTX PRO 6000): IDENTICAL to ..._binary_h50 (binary gripper @th25,
    # action ee_local delta, action_horizon=50, 40k) EXCEPT the proprio `observation/state` is fully replaced
    # by a 12-D ee_local VELOCITY (per arm [pos_vel_local(3), rot_vel(3)]; no pose, no gripper) -- the
    # converter's --state-mode velocity. Pairs with the other server's "no proprio at all" run as a clean
    # ablation on what proprioception buys (none vs velocity-only). Needs its OWN dataset (12-D state) +
    # norm-stats; PikaUmiInputs passes state through and the model pads 12->32, so no model/transform change.
    # Deploy NOTE: the runtime proprio (policy_runner OpenpiRemoteActionSource._proprio_state) must emit the
    # matching 12-D ee_local velocity, not reset-relative pose, to match this training distribution.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_binary_h50_velproprio",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_binary_h50_velproprio",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_binary_h50_velproprio"
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # UMI-STYLE ANCHORED action A/B (vs the per-step-delta velproprio above). action_mode=anchored: the
    # dataset stores per-frame ABSOLUTE tool poses and PikaUmiInputs re-expresses each H=24 chunk relative
    # to its first frame (T_t^-1 T_{t+k}) -> at deploy each row composes INDEPENDENTLY onto the live anchor
    # (no per-step delta integration / within-chunk drift, the suspected rollout-grasp failure). proprio =
    # velocity_grip (14-D ee_local velocity + ABSOLUTE gripper, init-pose-independent); action gripper =
    # absolute. Needs its OWN dataset (--action-mode anchored --state-mode velocity_grip --gripper-action
    # absolute) + norm-stats. action_horizon=24.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_anchored_velgrip_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_anchored_velgrip_h24",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_anchored_velgrip_h24"
            ),
            action_mode="anchored",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # SINGLE-ARM RIGHT, single-bolt pick&place (new data_right, 177 ep / 167 train / 20 val). Targets the
    # diagnosed right-arm generalization failure with MORE right-arm data on a UNIMODAL (one bolt) task.
    # 7-D right-arm action = per-step ee_local delta pose + ABSOLUTE gripper; proprio = velocity_grip-right
    # (6-D ee_local velocity + abs gripper = 7-D, ego-centric/init-pose-independent). Both wrist cams kept.
    # Converter: --arm right --state-mode velocity_grip --action-mode delta --gripper-action absolute.
    TrainConfig(
        name="pi05_pika_umi_right_velgrip_delta_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_right_velgrip_delta_h24_train",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_right_velgrip_delta_h24"
            ),
            arm="right",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # DEPTH + VELOCITY-PROPRIO + GRIPPER-ABSOLUTE, h24. RGB-D (realsense depth as extra wrist images
    # through the SHARED SigLIP, include_depth=True) + 12-D ee_local VELOCITY proprio (--state-mode
    # velocity) + ABSOLUTE gripper opening (--gripper-action absolute) + ee_local per-step delta pose
    # action, action_horizon=24. Dataset converted with
    # `--include-depth --camera realsense --state-mode velocity --gripper-action absolute` (tail-pad 50,
    # min-seg 48, val pinned to the fixed 86 via pika_umi_video_split_tcp_8020). Own assets_dir (4 image
    # streams + 12-D state -> norm-stats differ). Targets the diagnosed RIGHT-ARM grasp/generalization
    # failure (velproprio 20K: right-y stuck on val) via depth geometric grounding. Deploy: policy_runner
    # must send the matching live D405 *_wrist_0_depth (same _depth_to_image) + velocity proprio.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velproprio_depth_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velproprio_depth_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # ABSOLUTE-ORIENTATION ANCHOR on the depth baseline: IDENTICAL to ..._velproprio_depth_h24 (depth +
    # abs gripper + ee_local delta + h24) EXCEPT the proprio is 20-D `velocity_absrot6d` (per arm
    # [pos_vel(3), abs_rot_6d(6), grip(1)]) instead of 12-D velocity. Targets the 2026-06-25 rollout
    # failures: #1 pick RX over-tilt + #2 place yaw-180 drift -> both = velocity proprio has NO absolute
    # orientation anchor, so EE attitude is an uncorrected integral that drifts (and rotvec's +-pi jump
    # bites at 180). Position stays VELOCITY (ego-centric, OOD-safe), orientation becomes an ABSOLUTE 6D
    # anchor (continuous, gripper-points-down distribution matches UMI<->robot -> OOD-safe). Dataset:
    # `--include-depth --state-mode velocity_absrot6d --gripper-action absolute`. DEPLOY: runtime
    # _proprio_state must emit the matching 20-D (live TCP abs orientation -> R_align -> 6D + pos-vel + grip).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velabsrot6d_depth_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velabsrot6d_depth",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velabsrot6d_depth_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # GRAVITY-TILT ANCHOR on the depth baseline (the chosen #1 fix): IDENTICAL to ..._velproprio_depth_h24
    # EXCEPT the proprio is 20-D `velocity_grav` (per arm [pos_vel(3), rot_vel(3), gravity_tool(3), grip(1)]).
    # gravity_tool = world-down in the tool frame = an ABSOLUTE TILT anchor that is YAW-INVARIANT and
    # R_world-cancelling (fully ego-centric, OOD-safe even with the unmeasured steamvr->stand heading,
    # unlike a full absolute-orientation channel). Targets the 2026-06-25 rollout #1 (pick RX over-tilt):
    # makes attitude drift observable/correctable while keeping the velocity breakthrough (no absolute
    # position, no progress-clock). Does NOT address #2 (yaw drift) -- gravity can't see yaw. Dataset:
    # `--include-depth --state-mode velocity_grav --gripper-action absolute`. DEPLOY: runtime _proprio_state
    # emits the matching 20-D using the live TCP orientation + its own frame's down ([0,0,-1] for z-up stand).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velgrav_depth_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velgrav_depth",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velgrav_depth_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # DEPTH z_near=50 (gripper-visible): IDENTICAL to ..._depth_h24 but the depth dataset is re-converted
    # with `--depth-z-near-mm 50` (instead of 120) so the gripper fingers (~70-120mm, previously clipped to
    # black) become a depth gradient -> the policy sees its own fingers vs the bolt. Deploy MUST match with
    # `--depth-z-near-mm 50`. resize unchanged (resize_with_pad). Single-variable vs ..._depth_h24.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth_z50",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # RESOLUTION A/B (no-pad): IDENTICAL to ..._depth_h24 (SAME original 120/700 depth dataset) EXCEPT
    # `resize_pad=False` -> direct no-pad resize (full FOV, no black bars, whole 224x224 on the scene;
    # +~33% vertical pixels where the grasps are). Clean single-variable resize test vs ..._depth_h24 (the
    # resize_with_pad control). The served config carries resize_pad, so deploy resize stays matched.
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velproprio_depth_nopad_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velproprio_depth_nopad_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
            resize_pad=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    # z_near=50 + no-pad combined (if both levers want testing together on the z50 data).
    TrainConfig(
        name="pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_nopad_h24",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=24),
        data=LeRobotPikaUmiDataConfig(
            repo_id="plaif/pika_umi_video_train_tcp_gripabs_velproprio_depth_z50",
            assets=AssetsConfig(
                assets_dir="/home/plaif/workspace/openpi_runs/assets/pi05_pika_umi_video_tcp_gripabs_velproprio_depth_z50_nopad_h24"
            ),
            base_config=DataConfig(prompt_from_task=True),
            include_depth=True,
            resize_pad=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        batch_size=64,
        save_interval=5000,
        keep_period=10000,
        fsdp_devices=8,
        num_workers=12,
        checkpoint_base_dir="/home/plaif/workspace/openpi_runs/checkpoints",
        assets_base_dir="/home/plaif/workspace/openpi_runs/assets",
        wandb_enabled=False,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
