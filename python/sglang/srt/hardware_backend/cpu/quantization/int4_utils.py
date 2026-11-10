# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading

# AMX INT4 weight packing works on 32-wide output channel blocks.
_INT4_AMX_BLOCK_N = 32


def process_int4_linear_weights(layer: torch.nn.Module, quant_method: str) -> None:
    output_size = layer.scales.shape[-1]
    output_padding = -output_size % _INT4_AMX_BLOCK_N
    layer._amx_output_size = output_size
    layer._amx_output_padding = output_padding
    if output_padding:
        # GPTQ packs qweight along K, AWQ packs it along N.
        qweight_padding = (
            output_padding // 8 if quant_method == "awq" else output_padding
        )
        layer.qweight.data = F.pad(layer.qweight.data, (0, qweight_padding))
        layer.qzeros.data = F.pad(layer.qzeros.data, (0, output_padding // 8))
        layer.scales.data = F.pad(layer.scales.data, (0, output_padding))
    _amx_process_weight_after_loading(
        layer, ["qweight", "qzeros", "scales"], None, quant_method
    )
    layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
    layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
    layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
    # layer.bias stays unpadded since skip_bias_add hands it back to the caller.
    layer._amx_padded_bias = (
        F.pad(layer.bias.data, (0, output_padding))
        if output_padding and layer.bias is not None
        else None
    )


def apply_int4_linear(
    layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    output_size = layer._amx_output_size
    output_padding = layer._amx_output_padding
    if bias is not None and output_padding:
        bias = (
            layer._amx_padded_bias
            if bias is layer.bias
            else F.pad(bias, (0, output_padding))
        )
    output = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
        x.reshape(-1, x.shape[-1]), layer.qweight, layer.qzeros, layer.scales, bias
    )
    if output_padding:
        output = output[:, :output_size].contiguous()
    return output.view(*x.shape[:-1], output_size)
