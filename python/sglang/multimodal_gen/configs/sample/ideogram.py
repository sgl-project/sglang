# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)

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
