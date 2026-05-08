# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class SenseNovaU1SamplingParams(SamplingParams):
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_sensenova_u1_fields()

    def resolve_pixel_flow_cfg(self) -> "SenseNovaU1PixelFlowCFG":
        return resolve_sensenova_u1_pixel_flow_cfg(self)

    def _validate_sensenova_u1_fields(self) -> None:
        if len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain [start, end]")
        start, end = [float(x) for x in self.cfg_interval]
        if not (0.0 <= start <= end <= 1.0):
            raise ValueError("cfg_interval must satisfy 0 <= start <= end <= 1")
        self.cfg_interval = [start, end]

        if self.cfg_renorm_type not in {
            "none",
            "global",
            "channel",
            "text_channel",
        }:
            raise ValueError(
                "cfg_renorm_type must be one of: none, global, channel, text_channel"
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


@dataclass(frozen=True)
class SenseNovaU1PixelFlowCFG:
    text_scale: float
    img_scale: float
    needs_cfg: bool
    needs_img_condition: bool
    needs_uncondition: bool
    start: float
    end: float
    renorm_min: float
    renorm_type: str


def resolve_sensenova_u1_pixel_flow_cfg(
    params: Any,
) -> SenseNovaU1PixelFlowCFG:
    text_scale = float(getattr(params, "cfg_text_scale", 1.0))
    img_scale = float(getattr(params, "cfg_img_scale", 1.0))
    needs_cfg = not (text_scale == 1.0 and img_scale == 1.0)
    cfg_interval = list(getattr(params, "cfg_interval", [0.0, 1.0]))
    if len(cfg_interval) != 2:
        raise ValueError("SenseNova U1 cfg_interval must contain [start, end]")
    return SenseNovaU1PixelFlowCFG(
        text_scale=text_scale,
        img_scale=img_scale,
        needs_cfg=needs_cfg,
        needs_img_condition=needs_cfg and (img_scale == 1.0 or text_scale != img_scale),
        needs_uncondition=needs_cfg and img_scale != 1.0,
        start=float(cfg_interval[0]),
        end=float(cfg_interval[1]),
        renorm_min=float(getattr(params, "cfg_renorm_min", 0.0)),
        renorm_type=str(getattr(params, "cfg_renorm_type", "none")),
    )


def get_sensenova_u1_explicit_sampling_fields(params: Any | None) -> set[str]:
    if params is None:
        return set()
    return set(getattr(params, "_explicit_fields", set()) or set())


def mark_sensenova_u1_explicit_sampling_fields(
    params: SenseNovaU1SamplingParams,
    explicit_fields: set[str],
) -> SenseNovaU1SamplingParams:
    params._explicit_fields = get_sensenova_u1_explicit_sampling_fields(params) | set(
        explicit_fields
    )
    return params


def build_sensenova_u1_sampling_params(
    values: dict[str, Any] | None = None,
) -> SenseNovaU1SamplingParams:
    values = dict(values or {})
    return mark_sensenova_u1_explicit_sampling_fields(
        SenseNovaU1SamplingParams(**values),
        set(values.keys()),
    )
