# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for BRIA Fibo 2.0.

Two checkpoints share the architecture but not the schedule: the distilled
``Fibo-2.0-edit-turbo`` runs four steps with guidance off, while ``Fibo-2.0-dev``
runs thirty steps with a guidance scale of 5.0, which the reference implements as
a second forward pass on a null caption.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
    DEFAULT_NEGATIVE_PROMPT,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class BriaFibo2SamplingParams(SamplingParams):
    """The non-distilled `briaai/Fibo-2.0-dev` checkpoint: 30 steps, guidance 5.0."""

    num_inference_steps: int = 30
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    # Fibo has no guidance embedding in the transformer; the scale drives a
    # second, uncond, forward pass instead.
    embedded_cfg_scale: float = 1.0

    def should_use_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1.0


@dataclass
class BriaFibo2TurboSamplingParams(BriaFibo2SamplingParams):
    """The distilled `briaai/Fibo-2.0-edit-turbo` checkpoint: 4 steps, no CFG."""

    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    negative_prompt: str = ""

    def should_use_classifier_free_guidance(self) -> bool:
        return False


__all__ = [
    "BriaFibo2EditSamplingParams",
    "BriaFibo2SamplingParams",
    "BriaFibo2TurboSamplingParams",
]


@dataclass
class BriaFibo2EditSamplingParams(BriaFibo2TurboSamplingParams):
    """Editing with the distilled checkpoint: 4 steps, guidance off."""

    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    # Editing defaults the canvas to the first reference image's aspect ratio at
    # about one megapixel, so height/width are usually left unset.
    height: int | None = None
    width: int | None = None

    def should_use_classifier_free_guidance(self) -> bool:
        return False
