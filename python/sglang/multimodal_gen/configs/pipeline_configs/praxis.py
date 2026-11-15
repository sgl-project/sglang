# SPDX-License-Identifier: Apache-2.0
"""Praxis-1 pipeline configuration"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec

from sglang.multimodal_gen.configs.models.dits.praxis import (
    PraxisArchConfig,
    PraxisDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType

PRAXIS_HF_PATHS = [
    "runwayml/praxis-1-droid-lerobot",
    "runwayml/praxis-1-yam-lerobot",
]

_CONFIG_FILE = "config.json"
_WEIGHT_SIDECARS = ("model.safetensors.index.json", "SHA256SUMS")
_DEFAULT_VAE_ID = "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
_DEFAULT_TEXT_ENCODER_ID = "Qwen/Qwen3.5-2B"


class PraxisSettings(msgspec.Struct, frozen=True):
    embodiment: str
    chunk_size: int
    num_steps: int
    t_shift: float
    cfg_scale: float
    cfg_video_scale: float
    seed: int
    max_context_chunks: int
    vae_id: str
    text_encoder_id: str


def is_praxis_package(model_path: str) -> bool:
    """Whether a local directory is a Praxis policy package (LeRobot or native layout)."""
    config_file = Path(model_path).expanduser() / _CONFIG_FILE
    if not config_file.is_file():
        return False
    try:
        config = json.loads(config_file.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "praxis" in (
        config.get("type"),
        config.get("model_type"),
    )


def resolve_praxis_package(
    model_path: str, *, revision: str | None = None, config_only: bool = False
) -> Path:
    """Local directory of a policy package, downloading it from the Hub if needed."""
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model,
    )

    patterns = (
        [_CONFIG_FILE] if config_only else ["*.json", "*.safetensors", "SHA256SUMS"]
    )
    root = maybe_download_model(
        str(Path(model_path).expanduser()), allow_patterns=patterns, revision=revision
    )
    return Path(root)


def read_praxis_config(package: Path) -> dict[str, Any]:
    """LeRobot config of a package; a flat native config is wrapped as its ``model_config``."""
    config = json.loads((package / _CONFIG_FILE).read_text())
    if config.get("type") == "praxis":
        return config
    if config.get("model_type") == "praxis":
        return {"type": "praxis", "model_config": config}
    raise ValueError(f"{package} holds no Praxis policy config")


def praxis_settings(config: dict[str, Any]) -> PraxisSettings:
    model = config["model_config"]
    embodiment = model.get("embodiment")
    if embodiment not in RIGS:
        raise NotImplementedError(
            f"Praxis embodiment {embodiment!r} is not supported, only {sorted(RIGS)}"
        )
    rig = RIGS[embodiment]
    return PraxisSettings(
        embodiment=embodiment,
        chunk_size=int(config.get("chunk_size", model["chunking"]["action_chunk"])),
        num_steps=int(config.get("num_steps", 2)),
        t_shift=float(config.get("t_shift", rig.t_shift)),
        cfg_scale=float(config.get("cfg_scale", 1.0)),
        cfg_video_scale=float(config.get("cfg_video_scale", rig.cfg_video_scale)),
        seed=int(config.get("seed", 0)),
        max_context_chunks=int(config.get("max_context_chunks", 1)),
        vae_id=str(config.get("vae_id", _DEFAULT_VAE_ID)),
        text_encoder_id=str(config.get("text_encoder_id", _DEFAULT_TEXT_ENCODER_ID)),
    )


def native_checkpoint_dir(package: Path) -> Path:
    """Directory in the layout of the reference code: the flat config and links to the weights."""
    config = read_praxis_config(package)
    key = hashlib.sha256(str(package.resolve()).encode()).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / "sglang_praxis_native" / key
    target.mkdir(parents=True, exist_ok=True)
    # Same bytes as the config.json of the original checkpoint, so its SHA256SUMS entry still holds.
    (target / _CONFIG_FILE).write_text(
        json.dumps(config["model_config"], indent=2) + "\n"
    )
    for item in package.iterdir():
        if item.suffix != ".safetensors" and item.name not in _WEIGHT_SIDECARS:
            continue
        link = target / item.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(item.resolve())
    return target


class PraxisRig(msgspec.Struct, frozen=True):
    """Request-side view of a checkpoint embodiment."""

    image_keys: tuple[str, ...]
    # Pre-split state entries a client may send instead of ``state``.
    state_keys: tuple[str, ...]
    # One command row per arm: target position (3), row-major rotation matrix (9), target grip (1).
    command_dim: int
    state_name: str
    state_layout: str
    # Sampler recipe of the served checkpoint, for a config that does not set it.
    t_shift: float
    cfg_video_scale: float


RIGS: dict[str, PraxisRig] = {
    "single_arm": PraxisRig(
        image_keys=("exterior", "wrist"),
        state_keys=("robot_state.ee_pose", "grip"),
        command_dim=13,
        state_name="pose8 rows",
        state_layout="pose8 rows [x, y, z, qw, qx, qy, qz, grip]; a stack of rows carries the executed chunk",
        t_shift=12.5,
        cfg_video_scale=4.0,
    ),
    "bimanual": PraxisRig(
        image_keys=("exterior", "wrist_left", "wrist_right"),
        state_keys=(
            "robot_state.left.ee_pose",
            "grip.left",
            "robot_state.right.ee_pose",
            "grip.right",
        ),
        command_dim=26,
        state_name="pose16 rows",
        state_layout="pose16 rows [L: x, y, z, qw, qx, qy, qz, grip | R: same]; a stack of rows carries the executed chunk",
        t_shift=1.0,
        cfg_video_scale=1.0,
    ),
}


@dataclass
class PraxisPipelineConfig(PipelineConfig):
    """Praxis-1: video-action policy of a single-arm (DROID) or bimanual (YAM) checkpoint."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = True
    enable_autocast: bool = False
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    dit_config: PraxisDiTConfig = field(default_factory=PraxisDiTConfig)

    embodiment: str = "single_arm"
    image_keys: tuple[str, ...] = RIGS["single_arm"].image_keys
    action_dim: int = RIGS["single_arm"].command_dim
    output_action_dim: int = RIGS["single_arm"].command_dim
    action_horizon: int = 24
    default_num_inference_steps: int = 2
    # Text guidance on the video and on the actions (the reference cfg_video_scale and cfg_scale).
    guidance_scale: float = 4.0
    guidance_scale_action: float = 1.0
    sampler_shift: float = 12.5
    inference_seed: int = 0
    max_context_chunks: int = 1
    vae_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    text_encoder_id: str = "Qwen/Qwen3.5-2B"

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        if server_args.num_gpus not in (1, 2):
            raise NotImplementedError(
                "Praxis-1 runs on one GPU, or on two with CFG parallel"
            )
        if server_args.num_gpus == 2 and not server_args.enable_cfg_parallel:
            raise ValueError("Praxis-1 on two GPUs needs --enable-cfg-parallel")
        if server_args.tp_size != 1 or server_args.sp_degree != 1:
            raise NotImplementedError(
                "Praxis-1 supports no tensor or sequence parallelism"
            )
        package = resolve_praxis_package(
            server_args.model_path, revision=server_args.revision, config_only=True
        )
        config = read_praxis_config(package)
        self._adopt(praxis_settings(config))
        self.dit_config = PraxisDiTConfig(
            arch_config=PraxisArchConfig.from_model_config(config["model_config"])
        )

    def _adopt(self, settings: PraxisSettings) -> None:
        rig = RIGS[settings.embodiment]
        self.embodiment = settings.embodiment
        self.image_keys = rig.image_keys
        self.action_dim = self.output_action_dim = rig.command_dim
        self.action_horizon = settings.chunk_size
        self.default_num_inference_steps = settings.num_steps
        self.guidance_scale = settings.cfg_video_scale
        self.guidance_scale_action = settings.cfg_scale
        self.sampler_shift = settings.t_shift
        self.inference_seed = settings.seed
        self.max_context_chunks = settings.max_context_chunks
        # The checkpoint names the reference's .pth file, which the native VAE does not load.
        self.text_encoder_id = settings.text_encoder_id

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "model_path": server_args.model_path,
            "policy_family": "praxis",
            "input": {
                "image_keys": list(self.image_keys),
                "state": RIGS[self.embodiment].state_layout,
                "session_id": "optional; calls with the same id continue one episode",
            },
            "output": {
                "action_type": "continuous",
                "action_horizon": self.action_horizon,
                "action_dim": self.output_action_dim,
                "layout": (
                    "per arm [xyz(3), rotation matrix row-major(9), grip(1)]"
                    if self.embodiment == "bimanual"
                    else "[xyz(3), rotation matrix row-major(9), grip(1)]"
                ),
                "dtype": "float32",
            },
            "defaults": {
                "num_inference_steps": self.default_num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "guidance_scale_action": self.guidance_scale_action,
                "sampler_shift": self.sampler_shift,
            },
            "capabilities": {
                "exact_prefix_cache": False,
                "realtime_websocket": False,
                "openpi_websocket": False,
                "batch_inputs": False,
                "multiple_candidates": False,
            },
        }

    def estimate_request_cost(self, batch) -> float:
        options = batch.extra.get("vla", {}).get("options", {})
        video = options.get("guidance_scale")
        action = options.get("guidance_scale_action")
        guided = (self.guidance_scale if video is None else video) != 1.0 or (
            self.guidance_scale_action if action is None else action
        ) != 1.0
        steps = batch.num_inference_steps or self.default_num_inference_steps
        return float(steps * (2 if guided else 1))


def register():
    from sglang.multimodal_gen.configs.sample.praxis import PraxisSamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=PraxisSamplingParams,
        pipeline_config_cls=PraxisPipelineConfig,
        hf_model_paths=PRAXIS_HF_PATHS,
        model_detectors=[
            lambda hf_id: "praxis-1" in hf_id.lower(),
        ],
    )
