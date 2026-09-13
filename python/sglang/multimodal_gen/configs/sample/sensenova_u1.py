# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_CFG_NORM_CHOICES,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
    SENSENOVA_U1_RESOLUTION_ALIGNMENT,
)

_PUBLIC_OVERRIDE_FIELDS = {
    "prompt",
    "prompt_path",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "num_outputs_per_prompt",
    "seed",
    "save_output",
    "output_path",
    "output_file_name",
    "output_quality",
    "output_compression",
    "quality",
}


@dataclass
class SenseNovaU1SamplingParams(SamplingParams):
    data_type: DataType = field(default=DataType.IMAGE, init=False)
    height: int = 2048
    width: int = 2048
    num_frames: int = 1
    fps: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE
    negative_prompt: None = field(default=None, init=False)

    @classmethod
    def supported_override_fields(cls) -> set[str]:
        return set(_PUBLIC_OVERRIDE_FIELDS)

    @classmethod
    def get_cli_args(cls, args):
        cli_args = super().get_cli_args(args)
        return {
            key: value
            for key, value in cli_args.items()
            if key in _PUBLIC_OVERRIDE_FIELDS
        }

    def __post_init__(self) -> None:
        if isinstance(self.cfg_interval, list):
            self.cfg_interval = tuple(float(x) for x in self.cfg_interval)
        super().__post_init__()

    def _validate(self) -> None:
        super()._validate()
        if (
            self.width % SENSENOVA_U1_RESOLUTION_ALIGNMENT != 0
            or self.height % SENSENOVA_U1_RESOLUTION_ALIGNMENT != 0
        ):
            raise ValueError(
                "SenseNova-U1 requires width and height to be divisible by "
                f"{SENSENOVA_U1_RESOLUTION_ALIGNMENT}, got "
                f"{self.width}x{self.height}."
            )
        if self.num_frames != 1:
            raise ValueError(
                f"SenseNova-U1 is an image model and requires num_frames=1, got {self.num_frames}."
            )
        if self.cfg_norm not in SENSENOVA_U1_CFG_NORM_CHOICES:
            raise ValueError(
                f"cfg_norm must be one of {SENSENOVA_U1_CFG_NORM_CHOICES}, "
                f"got {self.cfg_norm!r}"
            )
        if len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        start, end = self.cfg_interval
        if not 0.0 <= float(start) <= float(end) <= 1.0:
            raise ValueError(
                f"cfg_interval must satisfy 0 <= start <= end <= 1, got {self.cfg_interval!r}"
            )

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        extra[SENSENOVA_U1_REQUEST_EXTRA_KEY] = {
            "cfg_norm": self.cfg_norm,
            "timestep_shift": self.timestep_shift,
            "enable_timestep_shift": self.enable_timestep_shift,
            "cfg_interval": tuple(self.cfg_interval),
            "t_eps": self.t_eps,
            "think_mode": self.think_mode,
        }
        return extra
