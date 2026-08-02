# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.boogu_image import BooguImageSamplingParams


@dataclass
class BooguImageEditSamplingParams(BooguImageSamplingParams):
    """Sampling defaults for the Boogu-Image edit (reference-image) pipeline.

    Inherits every text-to-image default (50 steps, 1024x1024, text
    ``guidance_scale`` 4.0, empty negative prompt, ``max_sequence_length`` 1280)
    and adds the second, image-conditioning guidance scale. ``guidance_scale_2``
    defaults to 1.0 (image guidance off), matching upstream: the default edit
    request uses text-only guidance, and double guidance is only enabled when the
    caller raises ``guidance_scale_2`` above 1.0.

    For a single reference image the working resolution is derived from the
    reference dimensions (upstream ``align_res``), so the 1024x1024 height/width
    defaults are only used when no reference image is supplied.
    """

    guidance_scale_2: float = 1.0
