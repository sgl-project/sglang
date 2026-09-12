"""gfx950 dense routes of `Fp8LinearMethod` for 32x32-block fp8 checkpoints served as
MXFP8 (`block_fp8_as_mxfp8`): the Triton dot_scaled kernel, or the native scaled-MFMA
kernels on a lane-ordered weight. The fused gfx950 producers hand these routes their
operand as an `Fp8GridActivation` (bf16 already on the fp8 grid) or an `Mxfp8Activation`."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    bf16_dequant_blockscaled_linear,
    dequant_mxfp8_to_bf16,
)
from sglang.srt.layers.utils import copy_or_rebind_param


def process_dense_weights(method, layer: torch.nn.Module, scale_u8) -> None:
    """The gfx950 branch of `Fp8LinearMethod.process_weights_after_loading`."""
    backend = method.mxfp8_dense_backend
    if backend.is_gfx95_dot_scaled():
        # dot_scaled reads canonical [N, K // 32] e8m0 bytes; block scales stay for direct readers
        if scale_u8 is not None:
            copy_or_rebind_param(layer, "weight_scale_inv_mx", scale_u8.contiguous())
        return
    assert backend.is_gfx95_mxfp8_native()
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        dequant_block_fp8_weight_to_bf16,
    )
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        native_route_supports,
        prepare_mxfp8_native_weight,
    )

    n, k = layer.weight.shape
    layer.mxfp8_native_ready = False
    if native_route_supports(n, k):
        # same bytes in scaled-MFMA lane order; a bf16 copy only where hipBLASLt serves M > 32
        shuffled, scale_ue8m0, weight_bf16 = prepare_mxfp8_native_weight(
            layer.weight.data,
            layer.weight_scale_inv.data,
            method.weight_block_size,
        )
        copy_or_rebind_param(layer, "weight", shuffled.view(torch.float8_e4m3fn))
        copy_or_rebind_param(layer, "weight_scale_mx_e8m0", scale_ue8m0)
        if weight_bf16 is not None:
            copy_or_rebind_param(layer, "weight_bf16", weight_bf16)
        else:
            layer.weight_bf16 = None
        layer.mxfp8_native_ready = True
    else:
        # a shape the native kernels do not tile keeps the bf16-dequant route
        copy_or_rebind_param(
            layer,
            "weight_bf16",
            dequant_block_fp8_weight_to_bf16(
                layer.weight.data,
                layer.weight_scale_inv.data,
                method.weight_block_size,
            ),
        )


def unwrap_activation(x):
    """Off a gfx950 route the producer's wrapper is unwrapped; re-quantizing is
    idempotent per-32 rounding."""
    if isinstance(x, Mxfp8Activation):
        x = Fp8GridActivation(dequant_mxfp8_to_bf16(x.q, x.scale))
    if isinstance(x, Fp8GridActivation):
        x = x.x
    return x


def apply_dense(
    method, layer: torch.nn.Module, x, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """`Fp8LinearMethod.apply` on a gfx950 route. `x` is a bf16 tensor, an `(fp8, scale)`
    tuple from a fused quant kernel, or one of the wrappers the fused producers emit."""
    backend = method.mxfp8_dense_backend
    mxfp8_ready = layer.block_fp8_mxfp8_ready
    native_route = mxfp8_ready and backend.is_gfx95_mxfp8_native()
    if isinstance(x, Mxfp8Activation):
        # quantized by a fused producer for the native route; other routes dequantize it (exact)
        if native_route:
            return _apply_native(method, layer, x.q, bias, input_scale=x.scale)
        x = Fp8GridActivation(dequant_mxfp8_to_bf16(x.q, x.scale))
    if isinstance(x, Fp8GridActivation):
        # the dot_scaled route quantizes the plain tensor itself (per-32 rounding is idempotent)
        if native_route:
            return _apply_native(method, layer, x.x, bias, input_on_fp8_grid=True)
        x = x.x
    x, input_scale = x if isinstance(x, tuple) else (x, None)
    if native_route:
        return _apply_native(method, layer, x, bias, input_scale=input_scale)
    if mxfp8_ready and input_scale is None:
        return method.w8a8_mxfp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale_inv_mx,
            input_scale=None,
            bias=bias,
        )
    # a shape the gfx950 kernels do not tile, or a pre-quantized tuple off the native route
    return method.w8a8_block_fp8_linear(
        input=x,
        weight=layer.weight,
        block_size=method.weight_block_size,
        weight_scale=layer.weight_scale_inv,
        input_scale=input_scale,
        bias=bias,
    )


def _apply_native(
    method,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor] = None,
    input_on_fp8_grid: bool = False,
) -> torch.Tensor:
    """The native route (`mxfp8_native_amd_gfx95`); a layer whose shape the native
    kernels do not tile keeps the bf16-dequant route."""
    if layer.mxfp8_native_ready:
        return method.w8a8_mxfp8_linear(
            input=x,
            weight_shuffled=layer.weight.view(torch.uint8),
            weight_scale_ue8m0=layer.weight_scale_mx_e8m0,
            weight_bf16=layer.weight_bf16,
            input_scale=input_scale,
            bias=bias,
            input_on_fp8_grid=input_on_fp8_grid,
        )
    if input_scale is not None:
        x = dequant_mxfp8_to_bf16(x, input_scale)
        input_on_fp8_grid = True
    return bf16_dequant_blockscaled_linear(
        input=x,
        weight=layer.weight_bf16,
        weight_scale=None,
        bias=bias,
        input_on_fp8_grid=input_on_fp8_grid,
    )
