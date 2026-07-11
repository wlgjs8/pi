import os

# Cap JAX GPU memory BEFORE jax is imported (via the openpi imports below).
# JAX defaults to preallocating 75% of total VRAM, so a JAX checkpoint shows
# ~24GB on a 32GB card even though it only needs ~8-10GB. Pin the fraction so
# inference stays small by default. Both are setdefault() so an explicit env
# var on the command line still wins (e.g. XLA_PYTHON_CLIENT_PREALLOCATE=false).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # If > 1, wrap the policy so each inference samples N action chunks and returns the MEDOID
    # (consensus) chunk — a deployable best-of-N selector that commits to the dominant mode and
    # reduces per-step mode-switching for multimodal tasks (e.g. the bolt pile). Costs N× inference.
    num_medoid_samples: int = 1

    # torch.compile mode for SERVING (overrides the checkpoint config's model.pytorch_compile_mode).
    # Training configs default to 'max-autotune', which compiles extremely slowly (minutes-to-hours,
    # + recompiles on shape changes) → deploy stutter. For serving, 'default' (fast compile) or 'off'
    # (eager, no compile) is far better. Allowed: default | reduce-overhead | max-autotune |
    # max-autotune-no-cudagraphs | off (eager) | config (keep the checkpoint's setting).
    compile_mode: str = "default"


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _config_with_compile_mode(config_name: str, compile_mode: str):
    """Load a config, overriding model.pytorch_compile_mode for serving (training unaffected)."""
    cfg = _config.get_config(config_name)
    if compile_mode == "config":
        return cfg
    mode = None if compile_mode in ("off", "none", "eager", "disable") else compile_mode
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=mode))


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, compile_mode: str = "default"
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config_with_compile_mode(checkpoint.config, compile_mode), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config_with_compile_mode(args.policy.config, args.compile_mode),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(
                args.env, default_prompt=args.default_prompt, compile_mode=args.compile_mode
            )


def main(args: Args) -> None:
    logging.info("Serving with compile_mode=%s, num_medoid_samples=%d", args.compile_mode, args.num_medoid_samples)
    policy = create_policy(args)
    if args.num_medoid_samples > 1:
        logging.info("Wrapping policy in MedoidPolicy (num_samples=%d): consensus selection at inference",
                     args.num_medoid_samples)
        policy = _policy.MedoidPolicy(policy, num_samples=args.num_medoid_samples)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
