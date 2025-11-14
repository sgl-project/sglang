# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from sglang.multimodal_gen.runtime.pipelines_general.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_general.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.conditioning import (
    ConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.decoding import (
    DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.encoding import (
    EncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.stepvideo_encoding import (
    StepvideoPromptEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_general.stages.timestep_preparation import (
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
