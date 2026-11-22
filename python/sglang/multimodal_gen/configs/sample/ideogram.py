# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

IDEOGRAM4_PRESETS: dict[str, dict[str, object]] = {
    "V4_QUALITY_48": {
        "num_steps": 48,
        "guidance_schedule": (3.0,) * 3 + (7.0,) * 45,
        "mu": 0.0,
        "std": 1.5,
    },
    "V4_DEFAULT_20": {
        "num_steps": 20,
        "guidance_schedule": (3.0,) * 2 + (7.0,) * 18,
        "mu": 0.0,
        "std": 1.75,
    },
    "V4_TURBO_12": {
        "num_steps": 12,
        "guidance_schedule": (3.0,) * 1 + (7.0,) * 11,
        "mu": 0.5,
        "std": 1.75,
    },
    "V4_TURBOTIME_LORA_2": {
        "num_steps": 2,
        "guidance_schedule": (1.0,) * 2,
        "mu": 0.5,
        "std": 1.75,
        "skip_unconditional": True,
        "lora_scale": 1.0,
        "requires_lora": True,
    },
    "V4_TURBOTIME_LORA_4": {
        "num_steps": 4,
        "guidance_schedule": (1.0,) * 4,
        "mu": 0.5,
        "std": 1.75,
        "skip_unconditional": True,
        "lora_scale": 1.0,
        "requires_lora": True,
    },
    "V4_TURBOTIME_LORA_8": {
        "num_steps": 8,
        "guidance_schedule": (1.0,) * 8,
        "mu": 0.5,
        "std": 1.75,
        "skip_unconditional": True,
        "lora_scale": 1.0,
        "requires_lora": True,
    },
    "V4_FAST_20": {
        "num_steps": 20,
        "guidance_schedule": (1.0,) * 20,
        "mu": 0.0,
        "std": 1.75,
    },
    "V4_INSTANT_8": {
        "num_steps": 8,
        "guidance_schedule": (1.0,) * 8,
        "mu": 0.0,
        "std": 1.75,
    },
}


@dataclass
class Ideogram4SamplingParams(SamplingParams):
    data_type: DataType = DataType.IMAGE
    prompt: str = " "
    negative_prompt: str = " "
    height: int = 1024
    width: int = 1024
    num_frames: int = 1
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    preset: str = "V4_DEFAULT_20"

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"preset"})

    def __post_init__(self) -> None:
        if self.preset not in IDEOGRAM4_PRESETS:
            raise ValueError(
                f"Unknown Ideogram 4 preset {self.preset!r}; "
                f"expected one of {sorted(IDEOGRAM4_PRESETS)}"
            )
        preset_cfg = IDEOGRAM4_PRESETS[self.preset]
        preset_steps = int(preset_cfg["num_steps"])
        explicit_fields = getattr(self, "_explicit_fields", None)
        num_steps_is_explicit = (
            explicit_fields is None or "num_inference_steps" in explicit_fields
        )
        guidance_is_explicit = (
            explicit_fields is None or "guidance_scale" in explicit_fields
        )
        if (
            self.num_inference_steps is not None
            and self.num_inference_steps != preset_steps
            and num_steps_is_explicit
        ):
            raise ValueError(
                "Ideogram 4 derives num_inference_steps from preset "
                f"{self.preset!r}; got {self.num_inference_steps}, expected "
                f"{preset_steps}."
            )
        if self.guidance_scale is not None and guidance_is_explicit:
            preset_guidance = float(preset_cfg["guidance_schedule"][-1])
            if self.guidance_scale != preset_guidance:
                raise ValueError(
                    "Ideogram 4 derives guidance from the preset guidance_schedule; "
                    "guidance_scale cannot be set directly."
                )
        self.num_inference_steps = preset_steps
        self.guidance_scale = float(preset_cfg["guidance_schedule"][-1])
        super().__post_init__()

    def _adjust_visual_fields(self, server_args, pipeline_config) -> None:
        super()._adjust_visual_fields(server_args, pipeline_config)
        self._align_resolution(pipeline_config)

    def prepare_synthetic_warmup_request_for_queue(self, req, server_args) -> None:
        # Warmup overwrites the dimensions after _adjust has run.
        self._align_resolution(server_args.pipeline_config)

    def _align_resolution(self, pipeline_config) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs.ideogram import (
            Ideogram4PipelineConfig,
        )

        # Ideogram4TextEncodingStage._build_inputs rejects sizes off the patch grid.
        if not isinstance(pipeline_config, Ideogram4PipelineConfig):
            return
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            return
        if self.width <= 0 or self.height <= 0:
            return
        m = pipeline_config.patch_size * pipeline_config.ae_scale_factor
        requested_size = (self.width, self.height)
        self.width = max(m, self.width // m * m)
        self.height = max(m, self.height // m * m)
        if (self.width, self.height) != requested_size:
            logger.warning(
                "Ideogram 4 requires dimensions divisible by %s; adjusted requested "
                "resolution from %sx%s to %sx%s",
                m,
                *requested_size,
                self.width,
                self.height,
            )


@dataclass
class Ideogram4FastSamplingParams(Ideogram4SamplingParams):
    preset: str = "V4_FAST_20"


@dataclass
class Ideogram4InstantSamplingParams(Ideogram4SamplingParams):
    preset: str = "V4_INSTANT_8"
