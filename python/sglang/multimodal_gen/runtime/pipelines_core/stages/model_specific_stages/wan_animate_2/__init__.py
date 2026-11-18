# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2-specific pipeline stages."""

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.before_denoising import (
    WanAnimate2BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.denoising import (
    WanAnimate2DenoisingStage,
    WanAnimate2OutputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
    WanAnimate2ImageEncoderAdapter,
    WanAnimate2VaeAdapter,
)

__all__ = [
    "WanAnimate2BeforeDenoisingStage",
    "WanAnimate2DenoisingStage",
    "WanAnimate2OutputStage",
    "WanAnimate2ImageEncoderAdapter",
    "WanAnimate2VaeAdapter",
]
