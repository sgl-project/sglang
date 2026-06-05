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
from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    AuxiliaryConditionEncodingStage,
    ConditionEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_dmd import (
    DmdDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.encoding import EncodingStage

# Hunyuan3D paint stages
from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_paint import (
    Hunyuan3DPaintPostprocessStage,
    Hunyuan3DPaintPreprocessStage,
    Hunyuan3DPaintTexGenStage,
)

# Hunyuan3D shape stages
from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape import (
    Hunyuan3DShapeBeforeDenoisingStage,
    Hunyuan3DShapeDenoisingStage,
    Hunyuan3DShapeExportStage,
    Hunyuan3DShapeSaveStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
    ImageVAEEncodingStage,
    LTX2ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
    RealtimeChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation_av import (
    LTX2AVLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_input_validation import (
    RealtimeInputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_text_encoding import (
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime_vae import (
    CausalVaeDecodingStage,
    RealtimeImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_connector import (
    LTX2TextConnectorStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    DMDTimestepPreparationStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.upsampling import (
    LTX2HalveResolutionStage,
    LTX2LoRASwitchStage,
    LTX2UpsampleStage,
)

__all__ = [
    "PipelineStage",
    "InputValidationStage",
    "RealtimeInputValidationStage",
    "TimestepPreparationStage",
    "DMDTimestepPreparationStage",
    "LatentPreparationStage",
    "RealtimeChunkLatentPreparationStage",
    "ComfyUILatentPreparationStage",
    "LTX2AVLatentPreparationStage",
    "DenoisingStage",
    "DmdDenoisingStage",
    "LTX2DenoisingStage",
    "LTX2AVDenoisingStage",
    "CausalDMDDenoisingStage",
    "EncodingStage",
    "ConditionEncodingStage",
    "AuxiliaryConditionEncodingStage",
    "DecodingStage",
    "CausalVaeDecodingStage",
    "LTX2AVDecodingStage",
    "ImageEncodingStage",
    "ImageVAEEncodingStage",
    "RealtimeImageVAEEncodingStage",
    "LTX2ImageEncodingStage",
    "TextEncodingStage",
    "RealtimeTextEncodingStage",
    "LTX2TextConnectorStage",
    # Hunyuan3D shape stages
    "Hunyuan3DShapeBeforeDenoisingStage",
    "Hunyuan3DShapeDenoisingStage",
    "Hunyuan3DShapeExportStage",
    "Hunyuan3DShapeSaveStage",
    # Hunyuan3D paint stages
    "Hunyuan3DPaintPreprocessStage",
    "Hunyuan3DPaintTexGenStage",
    "Hunyuan3DPaintPostprocessStage",
    # LTX-2 two-stage
    "LTX2RefinementStage",
    "LTX2HalveResolutionStage",
    "LTX2LoRASwitchStage",
    "LTX2UpsampleStage",
]
