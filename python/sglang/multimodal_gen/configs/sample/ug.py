# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class UGSamplingParams(SamplingParams):
    data_type: DataType = DataType.IMAGE
    num_frames: int = 1
    height: int | None = 1024
    width: int | None = 1024
    num_inference_steps: int | None = 50

    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    cfg_interval: list[float] = field(default_factory=lambda: [0.4, 1.0])
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    timestep_shift: float = 3.0
    think: bool = False
    think_max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_ug_fields()

    def _validate_ug_fields(self) -> None:
        if len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain [start, end]")
        start, end = [float(x) for x in self.cfg_interval]
        if not (0.0 <= start <= end <= 1.0):
            raise ValueError("cfg_interval must satisfy 0 <= start <= end <= 1")
        self.cfg_interval = [start, end]

        if self.cfg_renorm_type not in {"global", "channel", "text_channel"}:
            raise ValueError(
                "cfg_renorm_type must be one of: global, channel, text_channel"
            )

        for name in (
            "cfg_text_scale",
            "cfg_img_scale",
            "cfg_renorm_min",
            "timestep_shift",
        ):
            value: Any = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")
        if float(self.timestep_shift) <= 0.0:
            raise ValueError(
                f"timestep_shift must be positive, got {self.timestep_shift!r}"
            )
        if not isinstance(self.think, bool):
            raise ValueError(f"think must be a bool, got {self.think!r}")
        if self.think_max_new_tokens is not None:
            self.think_max_new_tokens = int(self.think_max_new_tokens)
            if self.think_max_new_tokens <= 0:
                raise ValueError(
                    "think_max_new_tokens must be positive when set, got "
                    f"{self.think_max_new_tokens!r}"
                )
