"""DeepSeek-V3 family flavour of the CuTe DSL AR fusion.

Also covers GLM-5.x: GlmMoeDsaForCausalLM subclasses DeepseekV2ForCausalLM.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    CuteDSLFusionService,
    MoeFinalizeHandoff,
    prepare_cutedsl_fusion,
)

DeepseekMoeFinalizeHandoff = MoeFinalizeHandoff
DeepseekFlashInferFusionService = CuteDSLFusionService

__all__ = [
    "DeepseekFlashInferFusionService",
    "DeepseekFlashInferLayerCommunicator",
    "DeepseekMoeFinalizeHandoff",
    "prepare_deepseek_flashinfer_fusion",
]


class DeepseekFlashInferLayerCommunicator(CuteDSLFusionLayerCommunicator):
    @classmethod
    def _norm_gamma(cls, layernorm) -> Optional[torch.Tensor]:
        if not isinstance(layernorm, RMSNorm):
            return None
        return layernorm.weight


def prepare_deepseek_flashinfer_fusion(model, model_runner) -> None:
    prepare_cutedsl_fusion(model, model_runner, label="DeepSeek-V3/GLM")
