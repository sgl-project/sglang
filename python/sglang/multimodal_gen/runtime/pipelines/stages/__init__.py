# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.conditioning import (
    ConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.encoding import EncodingStage
from sglang.multimodal_gen.runtime.pipelines.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.stepvideo_encoding import (
    StepvideoPromptEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines.stages.timestep_preparation import (
    TimestepPreparationStage,
)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "LatentPreparationStage",
    "ConditioningStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "CausalDMDDenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "TextEncodingStage",
    "StepvideoPromptEncodingStage",
]
