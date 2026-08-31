"""Qwen3.5 flavour of the CuTe DSL AR fusion.

The mechanism lives in :mod:`sglang.srt.layers.moe.cutedsl_ar_fusion`. Qwen3.5
normalizes with GemmaRMSNorm, whose gemma_weight is the checkpoint weight
already folded to w + 1.
"""

from __future__ import annotations

from sglang.srt.layers.moe.cutedsl_ar_fusion import (
    CuteDSLFusionLayerCommunicator,
    CuteDSLFusionService,
    MoeFinalizeHandoff,
    is_supported_forward_mode,
    prepare_cutedsl_fusion,
    resolve_max_m,
)

# Kept for qwen2_moe, qwen3_5 and the registered Qwen test.
Qwen35MoeFinalizeHandoff = MoeFinalizeHandoff
Qwen35FlashInferFusionService = CuteDSLFusionService

__all__ = [
    "Qwen35FlashInferFusionService",
    "Qwen35FlashInferLayerCommunicator",
    "Qwen35MoeFinalizeHandoff",
    "is_supported_forward_mode",
    "prepare_qwen35_flashinfer_fusion",
    "resolve_max_m",
]


class Qwen35FlashInferLayerCommunicator(CuteDSLFusionLayerCommunicator):
    NORM_WEIGHT_ATTR = "gemma_weight"


def prepare_qwen35_flashinfer_fusion(model, model_runner) -> None:
    prepare_cutedsl_fusion(model, model_runner, label="Qwen3.5")
