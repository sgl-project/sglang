# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


def resolve_uncond_skip_layers(params: "MOSSImageSamplingParams") -> tuple[int, ...]:
    layers = params.uncond_skip_layers
    if layers is None:
        # The recommended two-pass mode is the default. Selecting three-pass
        # SLG keeps the unconditional branch complete unless explicitly overridden.
        return () if params.slg_layers and params.slg_scale > 0 else (22,)
    if not isinstance(layers, (list, tuple)):
        raise ValueError("uncond_skip_layers must be a list of indices.")
    return tuple(layers)


@dataclass
class MOSSImageSamplingParams(SamplingParams):
    _default_height: ClassVar[int] = 1024
    _default_width: ClassVar[int] = 1024
    seed: int | list[int] = field(default=0, metadata={"batch_sig_exclude": True})
    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    negative_prompt: str = ""
    # None selects the mode default; [] disables unconditional skipping.
    uncond_skip_layers: list[int] | None = None
    slg_layers: list[int] | None = field(default_factory=list)
    slg_scale: float = 0.0
    slg_start: float = 0.01
    slg_stop: float = 1.0

    def should_use_classifier_free_guidance(self) -> bool:
        # The reference evaluates both branches even for guidance below one.
        if self.negative_prompt is None:
            self.negative_prompt = ""
        return self.guidance_scale != 1.0

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "uncond_skip_layers",
                "slg_layers",
                "slg_scale",
                "slg_start",
                "slg_stop",
            }
        )
