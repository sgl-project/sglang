# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 Action policy pipeline configuration.

A FLUX 3 Action policy package (e.g. ``black-forest-labs/flux-3-action-droid``)
holds ``config.native.json`` (or ``config.json``), ``manifest.json`` and
``model.safetensors`` (the DiT with the embodiment's action streams). The
frozen text encoder and video VAE live in the shared base repository and are
referenced from the config as ``repo_id[:filename][@revision]``.

Most fields below are filled from the package config when the server starts
(:meth:`Flux3ActionPipelineConfig.validate_server_args`), so the checkpoint
defines its camera layout, action space, sampler and guidance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.flux3 import (
    Flux3ArchConfig,
    Flux3DiTConfig,
)
from sglang.multimodal_gen.configs.models.vaes.flux3_video import Flux3VideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)

FLUX3_ACTION_POLICY_FILES = (
    "config.json",
    "config.native.json",
    "manifest.json",
    "model.safetensors",
)
FLUX3_ACTION_VARIANTS = ("base", "fp8r", "gd", "gd-fp8r", "sd", "sd-fp8r")
_CONFIG_ONLY_FILES = ("config.json", "config.native.json", "manifest.json")


def flux3_action_variant_subfolder(variant: str | None) -> str:
    """``--model-variant`` -> package subfolder (``base`` / ``None`` is the repository root)."""
    if variant in (None, "", "base"):
        return ""
    if variant not in FLUX3_ACTION_VARIANTS:
        raise ValueError(
            f"unknown FLUX 3 Action variant {variant!r}; choose from {FLUX3_ACTION_VARIANTS}"
        )
    return f"variants/{variant}"


def resolve_flux3_action_package(
    model_path: str,
    variant: str | None = None,
    *,
    config_only: bool = False,
    revision: str | None = None,
) -> Path:
    """Local directory of a policy package, downloading it from the Hub if needed."""
    subfolder = flux3_action_variant_subfolder(variant)
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model,
    )

    prefix = f"{subfolder}/" if subfolder else ""
    names = _CONFIG_ONLY_FILES if config_only else FLUX3_ACTION_POLICY_FILES
    root = maybe_download_model(
        str(Path(model_path).expanduser()),
        allow_patterns=[prefix + name for name in names],
        revision=revision,
    )
    package = Path(root) / subfolder
    if not package.is_dir():
        raise FileNotFoundError(f"FLUX 3 Action package {package} does not exist")
    return package


def _policy_config_path(package: Path) -> Path:
    """``config.native.json`` when present, else ``config.json`` (FP8r packages)."""
    for name in ("config.native.json", "config.json"):
        path = package / name
        if path.is_file():
            config = json.loads(path.read_text())
            if isinstance(config, dict) and "action_modality" in config:
                return path
    raise FileNotFoundError(f"{package} holds no FLUX 3 Action policy config")


def read_flux3_action_config(package: Path) -> dict[str, Any]:
    return json.loads(_policy_config_path(package).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_flux3_action_manifest(package: Path, *, include_weights: bool) -> None:
    """Check the policy config (and weights) against the package manifest hashes."""
    manifest = json.loads((package / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("kind") != "policy_export":
        raise ValueError(f"{package}/manifest.json is not a policy export manifest")
    hashes = manifest.get("sha256") or {}
    names = [_policy_config_path(package).name]
    if include_weights:
        names.append("model.safetensors")
    for name in names:
        if _sha256(package / name) != hashes.get(name):
            raise ValueError(f"{package}/{name} does not match its manifest checksum")


# Reference ``JointSingleSeqParams`` fields -> Flux3ArchConfig fields.
_DIT_CONFIG_RENAMES = {"num_heads": "num_attention_heads"}
_DIT_CONFIG_PASSTHROUGH = (
    "vec_in_dim",
    "context_in_dim",
    "hidden_size",
    "depth",
    "depth_single_blocks",
    "axes_dim",
    "theta",
    "mlp_ratio",
)
# Training-only knobs; the released trunk uses none of the non-default values.
_DIT_CONFIG_UNSUPPORTED = {
    "depth_late_blocks": 0,
    "qkv_bias": False,
    "gate_type": None,
}
_DIT_CONFIG_IGNORED = ("attn_mode",)


def flux3_arch_config(
    dit_config: dict[str, Any], streams: tuple[str, ...]
) -> Flux3ArchConfig:
    """A reference ``dit_config`` restricted to ``streams`` -> :class:`Flux3ArchConfig`."""
    kwargs: dict[str, Any] = {}
    for key, value in dit_config.items():
        if key in _DIT_CONFIG_UNSUPPORTED:
            if value != _DIT_CONFIG_UNSUPPORTED[key]:
                raise NotImplementedError(
                    f"dit_config.{key}={value!r} is not supported"
                )
        elif key in _DIT_CONFIG_RENAMES:
            kwargs[_DIT_CONFIG_RENAMES[key]] = value
        elif key in _DIT_CONFIG_PASSTHROUGH:
            kwargs[key] = tuple(value) if key == "axes_dim" else value
        elif key not in _DIT_CONFIG_IGNORED + ("in_channels", "sequence"):
            raise ValueError(f"unknown dit_config field {key!r}")
    arch = Flux3ArchConfig(**kwargs)
    in_channels = dit_config.get("in_channels", arch.in_channels)
    missing = [s for s in streams if s not in in_channels]
    if missing:
        raise ValueError(f"dit_config.in_channels lacks content streams {missing}")
    # Streams of the trunk the policy does not feed (e.g. image) are not built.
    arch.in_channels = {s: in_channels[s] for s in streams}
    arch.sequence = {f"x_{s}": s for s in streams}
    arch.__post_init__()
    return arch


def is_flux3_action_package(model_path: str) -> bool:
    """Whether a local directory is a FLUX 3 Action policy export."""
    package = Path(model_path).expanduser()
    manifest = package / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        manifest_data = json.loads(manifest.read_text())
        _policy_config_path(package)
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest_data, dict) and manifest_data.get("kind") == "policy_export"
    )


@dataclass
class Flux3ActionPipelineConfig(PipelineConfig):
    """FLUX 3 Action: joint video + action flow matching, returns action chunks."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = True
    enable_autocast: bool = False
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"

    dit_config: DiTConfig = field(default_factory=Flux3DiTConfig)
    # Only the encoder serves the policy (the predicted video is never decoded).
    vae_config: VAEConfig = field(
        default_factory=lambda: Flux3VideoVAEConfig(load_decoder=False)
    )

    # --- filled from the policy package (validate_server_args) ---
    policy_family: str = "flux3_action"
    policy_variant: str = "base"
    policy_revision: str | None = None
    action_modality: str = "action_prediction_droid"
    action_dim: int = 8
    state_dim: int = 8
    output_action_dim: int = 8
    action_horizon: int = 32
    camera_layout: str = "droid"
    image_keys: tuple[str, ...] = ("wrist", "left", "right")
    # Alternative request names of the cameras (LeRobot / RoboArena DROID keys).
    camera_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "wrist_image_left": "wrist",
            "exterior_image_1_left": "left",
            "exterior_image_2_left": "right",
        }
    )
    canvas_hw: tuple[int, int] = (544, 736)
    fps: float = 15.0
    action_scale: float = 2.0
    gripper_flip_dims: tuple[int, ...] = (-1,)
    action_parameterization: str = "absolute"
    absolute_action_dims: tuple[int, ...] = ()
    action_normalization: dict[str, list[float]] | None = None
    state_normalization: dict[str, list[float]] | None = None
    normalization_clip: float = 6.0
    inference_profile: str = "default"
    sampler: str = "cosmos_unipc"
    default_num_inference_steps: int = 4
    guidance_scale: float = 4.0
    # None: the action stream follows ``guidance_scale`` (also for request overrides).
    guidance_scale_action: float | None = 1.0
    sampler_shift: float = 5.0
    inference_seed: int = 0
    quantization: str | None = None
    video_vae_id: str | None = None
    text_encoder_id: str | None = None

    # --- runtime ---
    text_pad_multiple: int = 80
    text_max_length: int = 8192
    text_output_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
    # Text contexts cached per caption, bounded by their total token count.
    caption_cache_max_tokens: int = 1 << 16

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        if server_args.num_gpus > 1:
            raise NotImplementedError("FLUX 3 Action runs on a single GPU")
        variant = server_args.model_variant or "base"
        package = resolve_flux3_action_package(
            server_args.model_path,
            variant,
            config_only=True,
            revision=server_args.revision,
        )
        verify_flux3_action_manifest(package, include_weights=False)
        self.load_policy_config(read_flux3_action_config(package))
        self.policy_variant = variant
        self.policy_revision = server_args.revision

    def load_policy_config(self, config: dict[str, Any]) -> None:
        """Adopt the embodiment, camera layout and inference recipe of a policy config."""
        profile = config.get("inference_profile", "default")
        if profile != "default":
            raise NotImplementedError(
                f"FLUX 3 Action inference profile {profile!r} is not supported yet"
            )
        if config.get("action_parameterization", "absolute") not in (
            "absolute",
            "joint_delta",
        ):
            raise ValueError(
                f"unknown action parameterization {config['action_parameterization']!r}"
            )
        streams = tuple(config.get("content_streams") or ("video", "video_cond"))
        if streams != ("video", "video_cond"):
            raise NotImplementedError(
                f"content streams {streams} are not supported yet"
            )

        self.action_modality = config["action_modality"]
        self.action_dim = int(config["action_dim"])
        self.state_dim = self.action_dim
        self.output_action_dim = self.action_dim
        self.action_horizon = int(config["chunk_size"])
        self.camera_layout = config.get("camera_layout", "droid")
        self.image_keys = tuple(
            key.removeprefix("images.") for key in config.get("camera_keys", ())
        )
        self.canvas_hw = tuple(config.get("canvas_hw", (544, 736)))
        self.fps = float(config.get("fps", 15.0))
        self.action_scale = float(config.get("action_scale", 2.0))
        self.gripper_flip_dims = tuple(config.get("gripper_flip_dims", ()))
        self.action_parameterization = config.get("action_parameterization", "absolute")
        self.absolute_action_dims = tuple(config.get("absolute_action_dims", ()))
        self.action_normalization = config.get("action_normalization")
        self.state_normalization = config.get("state_normalization")
        self.normalization_clip = float(config.get("normalization_clip", 6.0))
        self.inference_profile = profile
        self.sampler = config.get("sampler") or "cosmos_unipc"
        self.default_num_inference_steps = int(config.get("num_inference_steps") or 4)
        self.guidance_scale = float(
            1.0 if config.get("guidance_scale") is None else config["guidance_scale"]
        )
        action_guidance = config.get("guidance_scale_action")
        self.guidance_scale_action = (
            None if action_guidance is None else float(action_guidance)
        )
        self.sampler_shift = float(config.get("sampler_shift") or 1.0)
        self.inference_seed = int(config.get("inference_seed", 0))
        self.quantization = config.get("quantization")
        self.video_vae_id = config.get("video_vae_id")
        self.text_encoder_id = config.get("text_encoder_id")

        conditioning_channels = config.get("conditioning_channels") or self.action_dim
        self.dit_config = Flux3DiTConfig(
            arch_config=flux3_arch_config(
                config.get("dit_config") or {}, streams
            ).with_streams(
                {
                    self.action_modality: self.action_dim,
                    f"{self.action_modality}_cond": int(conditioning_channels),
                }
            )
        )

    @property
    def latent_hw(self) -> tuple[int, int]:
        """Latent grid kept after the VAE: the content region of the canvas / 32 (ceil)."""
        if self.camera_layout == "droid":
            content = (540, 640)
        else:
            content = tuple(self.canvas_hw)
        return tuple(-(-size // 32) for size in content)

    def resolve_guidance(
        self, video: float | None, action: float | None
    ) -> dict[str, float]:
        """Per-stream guidance: request overrides, then the package recipe."""
        video_scale = self.guidance_scale if video is None else float(video)
        if action is None:
            action = self.guidance_scale_action
        return {
            "video": video_scale,
            self.action_modality: video_scale if action is None else float(action),
        }

    def supports_openpi_endpoint(self) -> bool:
        return True

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        return {
            "object": "action.metadata",
            "model": server_args.served_model_name,
            "model_path": server_args.model_path,
            "policy_family": self.policy_family,
            "input": {
                "image_keys": list(self.image_keys),
                "camera_aliases": dict(self.camera_aliases),
                "camera_layout": self.camera_layout,
                "state_dim": self.state_dim,
            },
            "output": {
                "action_type": "continuous",
                "action_horizon": self.action_horizon,
                "action_dim": self.output_action_dim,
                "padded_action_dim": self.action_dim,
                "dtype": "float32",
            },
            "defaults": {
                "num_inference_steps": self.default_num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "guidance_scale_action": self.guidance_scale_action,
                "sampler": self.sampler,
                "variant": self.policy_variant,
            },
            "capabilities": {
                "exact_prefix_cache": True,
                "realtime_websocket": True,
                "openpi_websocket": True,
                "batch_inputs": False,
                "multiple_candidates": False,
            },
        }

    def estimate_request_cost(self, batch) -> float:
        options = batch.extra.get("vla", {}).get("options", {})
        guidance = self.resolve_guidance(
            options.get("guidance_scale"), options.get("guidance_scale_action")
        )
        passes = 1 if all(g == 1.0 for g in guidance.values()) else 2
        steps = batch.num_inference_steps or self.default_num_inference_steps
        return float(steps * passes)
