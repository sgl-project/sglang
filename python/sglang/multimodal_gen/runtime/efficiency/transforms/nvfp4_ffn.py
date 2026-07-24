# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# NVFP4FFN -- LOAD-time transform that quantizes the video FFN to TE NVFP4. The
# quantization is baked into the weights at load; there is no per-step decision,
# so this is a ModelTransform (not a Technique). set_env() emits the existing
# SGLANG_HQ_ENABLE_TE_NVFP4_FFN recipe consumed by the loader.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_transform
from sglang.multimodal_gen.runtime.efficiency.technique import Seam
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)


@register_transform("nvfp4_ffn")
class NVFP4FFN(ModelTransform):
    """Quantize the video FFN proj_in/proj_out to NVFP4 at load (full-opt recipe:
    RHT + stochastic rounding + 2D quantization disabled)."""

    name = "nvfp4_ffn"
    phase = TransformPhase.LOAD
    writes = frozenset({Seam.FFN_PRECISION})

    def __init__(
        self,
        disable_rht: bool = True,
        disable_stochastic_rounding: bool = True,
        disable_2d_quantization: bool = True,
    ):
        self.disable_rht = disable_rht
        self.disable_stochastic_rounding = disable_stochastic_rounding
        self.disable_2d_quantization = disable_2d_quantization

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        e["SGLANG_HQ_ENABLE_TE_NVFP4_FFN"] = "1"
        e["SGLANG_LTX2_TE_NVFP4_VIDEO_FFN"] = "1"
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_RHT"] = "1" if self.disable_rht else "0"
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = (
            "1" if self.disable_stochastic_rounding else "0"
        )
        e["SGLANG_LTX2_TE_NVFP4_DISABLE_2D_QUANTIZATION"] = (
            "1" if self.disable_2d_quantization else "0"
        )
