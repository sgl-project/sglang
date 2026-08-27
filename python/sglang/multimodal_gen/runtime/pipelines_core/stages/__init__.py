# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Pipeline stages for diffusion models.

This package contains the various stages that can be composed to create
complete diffusion pipelines.
"""

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
    ComfyUILatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.conditioning import (
    ConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.encoding import EncodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation_av import (
    LTX2AVLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_connector import (
    LTX2TextConnectorStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "TimestepPreparationStage",
    "LatentPreparationStage",
    "ComfyUILatentPreparationStage",
    "LTX2AVLatentPreparationStage",
    "ConditioningStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "LTX2AVDenoisingStage",
    "CausalDMDDenoisingStage",
    "EncodingStage",
    "DecodingStage",
    "LTX2AVDecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "TextEncodingStage",
    "LTX2TextConnectorStage",
]
