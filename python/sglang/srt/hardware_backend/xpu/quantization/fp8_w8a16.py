# SPDX-License-Identifier: Apache-2.0
"""Weight-only FP8 (W8A16) execution of FP8 dense layers on Intel XPU.

An execution backend of ``Fp8LinearMethod``, like Marlin on CUDA: the weights stay
FP8, the activations stay in bf16, and ``sgl_kernel.fp8_linear`` runs the GEMM.
sgl-kernel-xpu decides on which devices that applies (``fp8_linear_supported``);
on the others, CRI included, ``Fp8LinearMethod`` keeps its W8A8 path.

``Fp8LinearMethod`` prepares the weights as usual (per-channel FP8 weights for
online quantization, the checkpoint's FP8 weights and 128x128 block scales for
block-FP8); these helpers only check each layer and run the GEMM. Layers the
kernel does not take keep the W8A8 path.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import Module

_BLOCK = 128


def _fp8_linear_supported_on_device() -> bool:
    try:
        from sgl_kernel import fp8_linear_supported
    except ImportError:
        # An sgl-kernel-xpu release without fp8_linear.
        return False
    return fp8_linear_supported()


def xpu_fp8_w8a16_enabled(quant_config) -> bool:
    """Whether ``Fp8LinearMethod`` runs W8A16 for this config on this XPU."""
    from sglang.srt.layers.quantization.fp8 import Fp8Config

    if not isinstance(quant_config, Fp8Config):
        return False
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


def prepare_xpu_fp8_w8a16_layer(layer: Module, block_quant: bool) -> None:
    """After ``Fp8LinearMethod`` prepared the weights: decide whether this layer
    runs W8A16 (``layer.xpu_fp8_w8a16``) and lay out its scales for the kernel."""
    from sgl_kernel import fp8_linear_supported

    # The kernel takes bf16 activations.
    bf16 = getattr(layer, "params_dtype", torch.bfloat16) == torch.bfloat16
    if block_quant:
        n, k = layer.weight.shape
        scale = layer.weight_scale_inv.data
        # The kernel reads one scale per 128x128 block of the [N, K] weight. A fused
        # layer (qkv, gate_up) stores its partitions' block scales one after
        # another, so every partition but the last must end on a block boundary.
        partitions_aligned = all(
            width % _BLOCK == 0 for width in getattr(layer, "logical_widths", [n])[:-1]
        )
        layer.xpu_fp8_w8a16 = (
            bf16
            and partitions_aligned
            and tuple(scale.shape) == (-(-n // _BLOCK), k // _BLOCK)
            and fp8_linear_supported(n, k, block_scale=True)
        )
        if layer.xpu_fp8_w8a16 and not scale.is_contiguous():
            # Fp8LinearMethod stores XPU block scales transposed for scaled_mm;
            # fp8_linear reads them row-major.
            with torch.no_grad():
                layer.weight_scale_inv.set_(scale.contiguous())
    else:
        # Online quantization: weight is the [K, N] view of the per-channel
        # quantized [N, K] FP8 weight, weight_scale holds N scales.
        weight = layer.weight.t()
        n, k = weight.shape
        layer.xpu_fp8_w8a16 = (
            bf16
            and weight.dtype == torch.float8_e4m3fn
            and weight.is_contiguous()
            and layer.weight_scale.numel() == n
            and fp8_linear_supported(n, k, block_scale=False)
        )


def apply_xpu_fp8_w8a16_linear(
    layer: Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    block_quant: bool,
) -> torch.Tensor:
    from sgl_kernel import fp8_linear

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    if block_quant:
        weight, weight_scale = layer.weight, layer.weight_scale_inv
    else:
        weight, weight_scale = layer.weight.t(), layer.weight_scale.view(-1)
    output = fp8_linear(x_2d, weight, weight_scale, bias=bias)
    return output.view(*x.shape[:-1], output.shape[-1])
