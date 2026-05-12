from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.amx_utils import amx_fused_experts_mxfp4
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput


class Mxfp4Fp8ConfigCpuMoEMethod(Mxfp4FlashinferTrtllmMoEMethod):
    # MxFP4 MoE method for the `Fp8Config + is_fp4_experts=True` dispatch on CPU AMX.

    def __init__(self):
        # Skip parent's __init__: it reads a GPU-only server arg and stores
        # fp8_method/prefix used only by the GPU path.
        pass

    def process_weights_after_loading(self, layer: Module) -> None:
        kernel = torch.ops.sgl_kernel

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data

        # Weights are int8-storage of packed FP4 (two values/byte). The CPU
        # kernel checks w scalar_type == uint8, so view as uint8 (bit-equivalent).
        w13 = w13.view(torch.uint8)
        w2 = w2.view(torch.uint8)

        if w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)
        w13_scale = w13_scale.view(torch.uint8)
        w2_scale = w2_scale.view(torch.uint8)

        w13 = kernel.convert_weight_packed(w13)
        w2 = kernel.convert_weight_packed(w2)
        w13_scale = kernel.convert_scale_packed(w13_scale)
        w2_scale = kernel.convert_scale_packed(w2_scale)

        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w2_weight = Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scale, requires_grad=False)

        layer.use_intel_amx_backend = True

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        return amx_fused_experts_mxfp4(
            layer,
            dispatch_output,
            self.moe_runner_config,
            scale_attrs=("w13_weight_scale_inv", "w2_weight_scale_inv"),
        )
