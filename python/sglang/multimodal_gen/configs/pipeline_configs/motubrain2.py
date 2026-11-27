# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Motubrain2 action policy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.task_type import ModelTaskType


def _camera_contract(
    views: list[dict[str, Any]],
) -> tuple[tuple[str, ...], dict[str, str]]:
    # Multiple crops can use a single physical camera. The request carries one
    # clip per source; the native processor applies the crops in profile order.
    sources = tuple(dict.fromkeys(view.get("source", view["name"]) for view in views))
    targets: dict[str, set[str]] = {}
    for view in views:
        name = view["name"]
        if name not in sources:
            targets.setdefault(name, set()).add(view["source"])
    aliases = {
        name: next(iter(source)) for name, source in targets.items() if len(source) == 1
    }
    return sources, aliases


@dataclass
class Motubrain2PipelineConfig(PipelineConfig):
    """Configuration for the Motubrain2 action policy.

    The policy path is the "V2A" / action-only inference mode described in the
    Motubrain2 report: observations + language -> action chunk, no video decode.
    """

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    supported_task_types: ClassVar[tuple[ModelTaskType, ...]] = (
        ModelTaskType.VLA_ACTION,
    )

    model_impl: str = "real"
    materialize_dtype: str = "bf16"
    should_use_guidance: bool = False

    bundle: str | None = None
    profile: str | None = None
    wan: str | None = None
    wan_subfolder: str | None = None
    wan_config: str | None = None
    text_encoder: str | None = None
    da3: str | None = None
    da3_config: str | None = None
    action_stats: str | None = None
    encoder_device: str | None = None
    compile: bool = False
    enforce_eager: bool = False
    multistream: str = "off"
    video_mxfp8: bool = False
    video_mxfp8_overlay: str | None = None
    deterministic: bool = False
    # ``None`` keeps the sampling recipe stored in the bundle (the released
    # RoboTwin/Astribot bundles already record sampler/steps/shifts).
    sampler_override: str | None = None
    steps_override: int | None = None
    video_shift_override: float | None = None
    action_shift_override: float | None = None
    dit_cache_enabled: bool | None = None
    dit_cache_threshold: float | None = None
    dit_cache_threshold_2: float | None = None
    dit_cache_skip_steps: int = 4
    dit_cache_skip_steps_2: int = 2
    dit_cache_damping: float = 0.25

    # Camera geometry and sampling parameters are resolved before serving.
    raw_observations: bool = True
    require_state: bool = True
    require_camera_extrinsic: bool = False
    observed_frames: int = 1
    temporal_stride: int = 4
    sparse_history_frames: int | None = None

    # Realtime chunking and model-side session memory.
    rtc_enabled: bool = False
    rtc_execute_prefix: int = 1
    rtc_delay: int = 1
    max_sessions: int = 64
    text_cache_size: int = 64

    # Filled from the serving bundle before HTTP or worker requests are accepted.
    image_keys: tuple[str, ...] = ()
    camera_aliases: dict[str, str] = field(default_factory=dict)
    action_dim: int = 0
    output_action_dim: int = 0
    state_dim: int = 0
    action_horizon: int = 0
    num_frames: int = 0
    sampler: str = ""
    default_num_inference_steps: int = 0
    video_shift: float = 0.0
    action_shift: float = 0.0

    policy_family: str = "motubrain2"
    inference_seed: int = 0
    guidance_scale: float | None = None
    guidance_scale_action: float | None = None
    text_encoder_name: str = "T5Gemma"
    text_dim: int = 3584
    max_token_len: int = 109

    enable_global_prefix_cache: bool = True
    prefix_cache_max_entries: int = 8
    prefix_cache_layout_version: str = "motubrain2-policy-v0"
    empty_cache_after_prefix: bool = False
    enable_segmented_actions: bool = False
    prefix_parallel_strategy: str = "monolithic"
    action_parallel_strategy: str = "monolithic"
    parallel_layout_version: str = "motubrain2-policy-v0"

    @property
    def action_websocket_sessions(self) -> bool:
        return True

    def check_pipeline_config(self) -> None:
        super().check_pipeline_config()
        if self.model_impl != "real":
            raise ValueError("Motubrain2 requires a real policy bundle")
        if self.compile and self.enforce_eager:
            raise ValueError("compile and enforce_eager are mutually exclusive")
        if self.inference_seed < 0:
            raise ValueError("inference_seed must be nonnegative")
        if self.guidance_scale not in (None, 1.0) or self.guidance_scale_action not in (
            None,
            1.0,
        ):
            raise ValueError("Motubrain2 supports guidance scales of None or 1 only")
        if self.enable_segmented_actions or not self.raw_observations:
            raise ValueError(
                "Motubrain2 requires raw observations and a monolithic action request"
            )
        if (self.prefix_parallel_strategy, self.action_parallel_strategy) != (
            "monolithic",
            "monolithic",
        ):
            raise ValueError("Motubrain2 only supports monolithic execution")

    # ------------------------------------------------- real backend helpers
    def check_real_requirements(self) -> None:
        """Fail fast when the real backend lacks its frozen encoders."""

        missing = [name for name in ("wan", "text_encoder") if not getattr(self, name)]
        if missing:
            raise ValueError(
                "model_impl='real' requires the frozen encoder sources "
                f"{missing}; set them in the pipeline config JSON "
                "(wan/text_encoder, plus da3 for geometry checkpoints)"
            )

    def native_sampler_override(self) -> str | None:
        if self.sampler_override in ("cm", "cm4"):
            return "consistency"
        return self.sampler_override

    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        if (
            server_args.num_gpus != 1
            or server_args.tp_size != 1
            or server_args.sp_degree != 1
            or server_args.dp_size != 1
            or server_args.enable_cfg_parallel
        ):
            raise ValueError(
                "Motubrain2 requires a single rank (one device, TP=SP=DP=1)"
            )
        self.load_serving_profile(server_args.model_path)

    def bundle_path(self, model_path: str) -> Path:
        if self.bundle:
            return Path(self.bundle)
        root = Path(model_path)
        return root / "policy" if (root / "policy").is_dir() else root

    def load_serving_profile(self, model_path: str) -> None:
        """Resolve the HTTP and worker contract using only checkpoint configuration."""
        if getattr(self, "_loaded_model_path", None) == model_path:
            return
        self.check_real_requirements()

        try:
            from motubrain2.data.history import SparseHistory
            from motubrain2.data.robot import RobotActionProfile
            from motubrain2.inference.sampling import sampling_options
        except ModuleNotFoundError as exc:
            if exc.name != "motubrain2":
                raise
            raise ImportError(
                "Install the optional Motubrain2 inference package in the SGLang "
                "environment to serve Motubrain2Pipeline: "
                "https://github.com/shengshu-ai/Motubrain-2.0"
            ) from exc

        from diffusers import AutoencoderKLWan

        bundle = self.bundle_path(model_path)
        profile_path = Path(self.profile or Path(model_path) / "profile.json")
        architecture = json.loads((bundle / "config.json").read_text())
        profile = json.loads(profile_path.read_text())
        feature_layers = architecture.get("feature_layers", ())
        if bool(feature_layers) != bool(self.da3):
            raise ValueError(
                "Supply da3 exactly when the Motubrain2 checkpoint uses geometry"
            )
        if feature_layers:
            geometry_layers = profile.get("geometry_layers")
            if not isinstance(geometry_layers, list) or len(geometry_layers) != len(
                feature_layers
            ):
                raise ValueError(
                    "The profile needs one geometry encoder layer per WAM feature layer"
                )
        # Native field mapping defines wire width, including seven-value EEFs.
        # Normalization statistics do not affect the wire layout.
        action_spec = dict(profile["action"])
        action_spec.pop("normalization", None)
        self.action_dim = int(architecture["action_dim"])
        self.output_action_dim = RobotActionProfile(
            action_spec, self.action_dim
        ).wire_dim
        self.state_dim = self.output_action_dim
        fields = action_spec["fields"]
        self.require_state = any(
            not field.get("model_used", True)
            or field["kind"] in ("relative", "eef_camera", "eef_relative")
            for field in fields
        )
        self.require_camera_extrinsic = any(
            field.get("model_used", True) and field["kind"] == "eef_camera"
            for field in fields
        )
        self.image_keys, self.camera_aliases = _camera_contract(profile["views"])
        # Raw HTTP inputs have no fixed spatial size. Per-view resizing and
        # cropping happen inside the native processor, after reading the clip.
        self.action_horizon = profile["action_steps"]
        self.observed_frames = profile.get("observed_frames", 1)
        history = profile.get("sparse_history")
        self.sparse_history_frames = (
            SparseHistory(**history).rgb_frames if history is not None else None
        )

        source, subfolder = self.wan, self.wan_subfolder
        if Path(source).suffix == ".pth":
            source = self.wan_config or "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            subfolder = self.wan_subfolder if self.wan_config else "vae"
        vae = AutoencoderKLWan.load_config(source, subfolder=subfolder)
        self.temporal_stride = int(vae.get("scale_factor_temporal", 4))
        self.num_frames = 1 + (profile["latent_frames"] - 1) * self.temporal_stride

        recipe = sampling_options(
            bundle,
            self.native_sampler_override(),
            self.steps_override,
            self.video_shift_override,
            self.action_shift_override,
        )
        self.sampler = recipe["sampler"]
        self.default_num_inference_steps = recipe["steps"]
        self.video_shift = recipe["video_shift"]
        self.action_shift = recipe["action_shift"]
        self.check_pipeline_config()
        self._loaded_model_path = model_path

    def validate_loaded_profile(self, processor: Any, sampling: dict[str, Any]) -> None:
        """Catch changed assets or native loader drift before a worker serves requests."""
        sources, aliases = _camera_contract(processor.views)
        actual = {
            "image_keys": sources,
            "camera_aliases": aliases,
            "action_horizon": processor.action_steps,
            "action_dim": processor.config.action_dim,
            "output_action_dim": processor.actions.wire_dim,
            "observed_frames": processor.observed_frames,
            "temporal_stride": processor.video.temporal_stride,
            "sparse_history_frames": processor.sparse_history.rgb_frames
            if processor.sparse_history
            else None,
            "sampler": sampling["sampler"],
            "default_num_inference_steps": sampling["steps"],
            "video_shift": sampling["video_shift"],
            "action_shift": sampling["action_shift"],
        }
        for name, value in actual.items():
            if value != getattr(self, name):
                raise ValueError(
                    f"Motubrain2 worker {name}={value!r} differs from the serving profile {getattr(self, name)!r}"
                )

    def rtc_config(self) -> dict[str, Any] | None:
        """Return the native RTC config mapping, or ``None`` when disabled."""

        if not self.rtc_enabled:
            return None
        return {
            "enabled": True,
            "execute_prefix": int(self.rtc_execute_prefix),
            "delay": int(self.rtc_delay),
        }

    # ------------------------------------------------- SGLang server contract
    def supports_action_endpoint(self) -> bool:
        return True

    def supports_openpi_endpoint(self) -> bool:
        return True

    def supports_dynamic_batching(self) -> bool:
        return False

    def supports_native_grouped_requests(self) -> bool:
        return False

    def prefix_cuda_graph_available(self) -> bool:
        return False

    def action_cuda_graph_available(self) -> bool:
        return False

    def estimate_request_cost(self, batch: Any) -> float:
        return float(
            getattr(batch, "num_inference_steps", None)
            or self.default_num_inference_steps
        )

    def action_metadata(self, server_args: Any) -> dict[str, Any]:
        """Describe the loaded checkpoint and the observation and action wire formats."""

        return {
            "object": "action.metadata",
            "model": getattr(server_args, "served_model_name", None),
            "model_path": getattr(server_args, "model_path", self.model_path),
            "policy_family": self.policy_family,
            "model_impl": self.model_impl,
            "input": {
                "image_keys": list(self.image_keys),
                "camera_aliases": dict(self.camera_aliases),
                "camera_layout": "separate",
                "image_size": None,
                "observation_frames": 1
                + (self.observed_frames - 1) * self.temporal_stride,
                "history_frames": self.sparse_history_frames,
                "state_required": self.require_state,
                "camera_extrinsic_required": self.require_camera_extrinsic,
                "http_session_id_required": self.rtc_enabled,
                "state_dim": self.state_dim,
                "modalities": ["image"],
                "text_encoder": self.text_encoder_name,
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
                "variant": getattr(server_args, "model_variant", None),
                "action_horizon": self.action_horizon,
                "action_dim": self.action_dim,
                "num_frames": self.num_frames,
                "prefix_cache": ("auto" if self.enable_global_prefix_cache else False),
                "cuda_graph": False,
                "output_format": "list",
            },
            "runtime": {
                "materialize_dtype": self.materialize_dtype,
                "enable_autocast": self.enable_autocast,
                "parallelism": {
                    "num_gpus": getattr(server_args, "num_gpus", 1),
                    "tp_size": getattr(server_args, "tp_size", 1),
                    "sp_degree": getattr(server_args, "sp_degree", 1),
                    "prefix_strategy": self.prefix_parallel_strategy,
                    "action_strategy": self.action_parallel_strategy,
                    "layout_version": self.parallel_layout_version,
                },
            },
            "capabilities": {
                "action_modes": ["policy"],
                "exact_prefix_cache": True,
                "cuda_graph": False,
                "realtime_websocket": True,
                "openpi_websocket": True,
                "batch_inputs": False,
                "multiple_candidates": False,
                "native_outputs": [
                    "video",
                    "action",
                    "decoded_action",
                    "decoded_valid",
                ],
                "native_outputs_format": "base64-encoded safetensors",
            },
        }


def register():
    from sglang.multimodal_gen.configs.sample.motubrain2 import Motubrain2SamplingParams
    from sglang.multimodal_gen.registry import (
        KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS,
        register_configs,
    )

    paths = [
        "GensPI/Motubrain2.0-Base",
        "GensPI/Motubrain2.0-Distill",
        "GensPI/Motubrain2.0-Base-serving",
        "GensPI/Motubrain2.0-Distill-serving",
        "GensPI/motubrain2-serving",
    ]
    register_configs(
        sampling_param_cls=Motubrain2SamplingParams,
        pipeline_config_cls=Motubrain2PipelineConfig,
        hf_model_paths=paths,
        model_detectors=[lambda name: name.lower() == "motubrain2pipeline"],
    )
    for path in paths:
        KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS[path.lower()] = (
            "Motubrain2Pipeline"
        )
