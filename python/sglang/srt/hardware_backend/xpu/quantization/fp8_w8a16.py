# SPDX-License-Identifier: Apache-2.0
"""Weight-only FP8 (W8A16) dense linear for Intel XPU through ``sgl_kernel.fp8_linear``.

The weights are FP8 and the activations stay in bf16, on devices without an FP8
matrix engine. sgl-kernel-xpu decides where that applies
(``fp8_linear_supported``); other devices, CRI included, keep ``Fp8LinearMethod``
unchanged. Two sources of FP8 weights:

- online quantization of a bf16 checkpoint (``--quantization fp8``): weights are
  quantized at load time with per-channel scales, as Fp8LinearMethod does;
- pre-quantized block-FP8 checkpoints (128x128 ``weight_block_size``): the stored
  weights and ``weight_scale_inv`` are used as they are.

Layers the kernel cannot take (``fp8_linear_supported``) keep the existing
``Fp8LinearMethod`` path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.fp8 import Fp8Config

_BLOCK = 128


def _fp8_linear_supported_on_device() -> bool:
    try:
        from sgl_kernel import fp8_linear_supported
    except ImportError:
        # An sgl-kernel-xpu release without fp8_linear.
        return False
    return fp8_linear_supported()


def use_xpu_fp8_w8a16(quant_config: Fp8Config) -> bool:
    """Whether FP8 dense layers on this XPU should run ``sgl_kernel.fp8_linear``."""
    online = not quant_config.is_checkpoint_fp8_serialized
    block_checkpoint = (
        quant_config.is_checkpoint_fp8_serialized
        and quant_config.weight_block_size == [_BLOCK, _BLOCK]
    )
    return (
        (online or block_checkpoint)
        and not quant_config.use_mxfp8
        and quant_config.activation_scheme == "dynamic"
        and _fp8_linear_supported_on_device()
    )


class XPUFp8W8A16LinearMethod(Fp8LinearMethod):
    """Fp8LinearMethod that runs FP8 dense layers as W8A16 on ``sgl_kernel.fp8_linear``.

    Weight preparation is Fp8LinearMethod's (per-channel scales for online
    quantization, the checkpoint's block scales for block-FP8); only the GEMM
    changes. ``create_weights`` is inherited.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        from sgl_kernel import fp8_linear_supported

        super().process_weights_after_loading(layer)
        # The kernel takes bf16 activations.
        bf16 = getattr(layer, "params_dtype", torch.bfloat16) == torch.bfloat16
        if self.block_quant:
            n, k = layer.weight.shape
            scale = layer.weight_scale_inv.data
            # The kernel reads one scale per 128x128 block of the [N, K] weight. A
            # fused layer (qkv, gate_up) stores its partitions' block scales one
            # after another, so every partition but the last must end on a block
            # boundary.
            partitions_aligned = all(
                width % _BLOCK == 0
                for width in getattr(layer, "logical_widths", [n])[:-1]
            )
            layer.use_fp8_w8a16 = (
                bf16
                and partitions_aligned
                and tuple(scale.shape) == (-(-n // _BLOCK), k // _BLOCK)
                and fp8_linear_supported(n, k, block_scale=True)
            )
            if layer.use_fp8_w8a16 and not scale.is_contiguous():
                # Fp8LinearMethod stores XPU block scales transposed for
                # scaled_mm; fp8_linear reads them row-major.
                with torch.no_grad():
                    layer.weight_scale_inv.set_(scale.contiguous())
        else:
            # Online quantization: weight is the [K, N] view of the per-channel
            # quantized [N, K] FP8 weight, weight_scale holds N scales.
            weight = layer.weight.t()
            n, k = weight.shape
            layer.use_fp8_w8a16 = (
                bf16
                and weight.dtype == torch.float8_e4m3fn
                and weight.is_contiguous()
                and layer.weight_scale.numel() == n
                and fp8_linear_supported(n, k, block_scale=False)
            )

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.use_fp8_w8a16:
            return super().apply(layer, x, bias)

        from sgl_kernel import fp8_linear

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if self.block_quant:
            weight, weight_scale = layer.weight, layer.weight_scale_inv
        else:
            weight, weight_scale = layer.weight.t(), layer.weight_scale.view(-1)
        output = fp8_linear(x_2d, weight, weight_scale, bias=bias)
        return output.view(*x.shape[:-1], output.shape[-1])
