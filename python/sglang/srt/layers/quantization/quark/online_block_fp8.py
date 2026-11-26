# SPDX-License-Identifier: Apache-2.0
"""128x128 block FP8 for the BF16 linears a Quark checkpoint excludes, quantized
after loading and run on aiter's gfx1250 e8m0 block-scale GEMM."""

import logging
from typing import Optional

import torch
from torch.nn import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import (
    aiter_mxfp8_gfx1250_block_fp8_linear,
    gfx1250_mxfp8_prepare_weight,
    per_block_cast_to_fp8,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import ceil_align

logger = logging.getLogger(__name__)

_BLOCK = 128


def online_block_fp8_applies(prefix: str) -> bool:
    """Whether the excluded linear at ``prefix`` is one SGLANG_QUARK_DENSE_FP8_LAYERS names."""
    if not envs.SGLANG_QUARK_DENSE_FP8.get():
        return False
    if not fp8_utils._use_aiter_mxfp8_gfx1250:
        logger.warning_once(
            "SGLANG_QUARK_DENSE_FP8 needs SGLANG_USE_AITER=1 on gfx1250 with aiter's "
            "block-scale GEMM; the excluded linears stay BF16."
        )
        return False
    targets = [
        name.strip()
        for name in envs.SGLANG_QUARK_DENSE_FP8_LAYERS.get().split(",")
        if name.strip()
    ]
    return any(prefix == name or prefix.endswith("." + name) for name in targets)


class QuarkOnlineBlockFp8LinearMethod(UnquantizedLinearMethod):
    """Loads the BF16 weight like UnquantizedLinearMethod, then keeps only a
    preshuffled 128x128 block FP8 copy with power-of-two scales. Activations are
    quantized per 128-wide K group on every call. A weight whose K is not a
    multiple of 128 stays BF16."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        if weight.dtype != torch.bfloat16 or k % _BLOCK:
            super().process_weights_after_loading(layer)
            return
        # aiter derives the scale block from the scale shape, so N must be whole blocks.
        n_padded = ceil_align(n, _BLOCK)
        if n_padded != n:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, n_padded - n))
        qweight, scale = per_block_cast_to_fp8(weight, (_BLOCK, _BLOCK))
        prepared = gfx1250_mxfp8_prepare_weight(qweight, scale, [_BLOCK, _BLOCK])
        if prepared is None:
            super().process_weights_after_loading(layer)
            return
        qweight, scale = prepared
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.weight_scale_inv = Parameter(scale, requires_grad=False)
        layer.dense_fp8_out_features = n
        logger.info_once(
            "Quark dense FP8: excluded BF16 linears matching SGLANG_QUARK_DENSE_FP8_LAYERS "
            "run as 128x128 block FP8."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = getattr(layer, "dense_fp8_out_features", None)
        if n is None:
            return super().apply(layer, x, bias)
        output = aiter_mxfp8_gfx1250_block_fp8_linear(
            x, layer.weight, [_BLOCK, _BLOCK], layer.weight_scale_inv
        )
        if output.shape[-1] != n:
            output = output[..., :n]
        if bias is not None:
            output = output + bias
        return output

    def apply_with_addend(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        addend: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(layer, "dense_fp8_out_features", None) is None:
            return super().apply_with_addend(layer, x, addend, bias)
        return self.apply(layer, x, bias).add_(addend)

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(layer, "dense_fp8_out_features", None) is None:
            return super().apply_into(layer, x, output, bias)
        output.copy_(self.apply(layer, x, bias))
        return output
