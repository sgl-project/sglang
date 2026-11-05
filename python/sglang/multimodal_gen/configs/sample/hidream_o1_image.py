# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import ClassVar

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HiDreamO1ImageSamplingParams(SamplingParams):
    # The reference implementation conditions the unconditional branch on a
    # single space rather than the empty string.
    negative_prompt: str = " "

    num_frames: int = 1
    # The reference `full` recipe, which is the only one this pipeline implements:
    # 50 steps, guidance 5.0, shift 3.0, FlowUniPC (inference.py:44,79-84).
    num_inference_steps: int = 50
    guidance_scale: float = 5.0

    # (width, height) pairs the model was trained on. The pixel patches carry no
    # scale invariance, so anything else degrades quality even when it is a legal
    # multiple of the 32-pixel patch size.
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (2048, 2048),
            (2304, 1728),
            (1728, 2304),
            (2560, 1440),
            (1440, 2560),
            (2496, 1664),
            (1664, 2496),
            (3104, 1312),
            (1312, 3104),
            (2304, 1792),
            (1792, 2304),
        ]
    )

    _default_height: ClassVar[int] = 2048
    _default_width: ClassVar[int] = 2048
