"""gfx950 dense routes of Fp8LinearMethod for 32x32-block fp8 checkpoints served as
MXFP8 (block_fp8_as_mxfp8): the Triton dot_scaled kernel, the native scaled-MFMA
kernels on a lane-ordered weight, or aiter's MXFP8 GEMM on the plain one. With
SGLANG_OPT_HIP_FLYDSL_MXFP8 the aiter route hands its large-M (prefill) calls to the FlyDSL
kernel of mxfp8_flydsl_gfx95 on the same weight. The fused gfx950 producers hand these
routes their operand as an Fp8GridActivation (bf16 already on the fp8 grid) or an
Mxfp8Activation."""

from __future__ import annotations

import functools
import logging
from typing import Optional

import torch

from sglang.kernels.ops.gemm.mxfp8_amd_gfx95 import bf16_dequant_blockscaled_linear
from sglang.kernels.ops.gemm.mxfp8_native_amd_gfx95 import (
    native_route_supports,
    prepare_mxfp8_native_weight,
    ue8m0_weight_scale,
)
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    dequant_block_fp8_weight_to_bf16,
    dequant_mxfp8_to_bf16,
    mxfp8_operand,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils import copy_or_rebind_param

logger = logging.getLogger(__name__)

# Fewer rows (decode, short prefill chunks) stay on aiter: otherwise run in-tree FlyDSL
# kernel.
_FLYDSL_MXFP8_MIN_M = 2048


def process_dense_weights(method, layer: torch.nn.Module, scale_u8) -> None:
    """The gfx950 branch of Fp8LinearMethod.process_weights_after_loading."""
    backend = method.mxfp8_dense_backend
    if backend.is_gfx95_dot_scaled():
        # dot_scaled reads canonical [N, K // 32] e8m0 bytes; block scales stay for direct readers
        if scale_u8 is not None:
            copy_or_rebind_param(layer, "weight_scale_inv_mx", scale_u8.contiguous())
        return
    if backend.is_gfx95_mxfp8_aiter():
        # the weight stays [N, K]; aiter reads the block scales compact, one row per 32 output rows
        copy_or_rebind_param(
            layer,
            "weight_scale_mx_e8m0",
            ue8m0_weight_scale(layer.weight_scale_inv.data),
        )
        _prepare_flydsl_mxfp8(layer)
        from aiter.ops.shuffle import shuffle_weight

        from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import (
            aiter_preshuffle_supported,
        )

        n, k = layer.weight.shape
        layer.mxfp8_aiter_preshuffled = aiter_preshuffle_supported(n, k)
        if layer.mxfp8_aiter_preshuffled:
            # same bytes in aiter's (16, 16) tile order; the scales stay compact row-major
            copy_or_rebind_param(
                layer, "weight", shuffle_weight(layer.weight.data, (16, 16))
            )
            # mxfp8_flydsl_gfx95 reads the plain [N, K] weight
            layer.weight_scale_mx_flydsl = None
        return
    assert backend.is_gfx95_mxfp8_native()
    n, k = layer.weight.shape
    layer.mxfp8_native_ready = False
    if native_route_supports(n, k):
        # same bytes in scaled-MFMA lane order
        shuffled, scale_ue8m0 = prepare_mxfp8_native_weight(
            layer.weight.data,
            layer.weight_scale_inv.data,
            method.weight_block_size,
        )
        copy_or_rebind_param(layer, "weight", shuffled.view(torch.float8_e4m3fn))
        copy_or_rebind_param(layer, "weight_scale_mx_e8m0", scale_ue8m0)
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


def apply_dense(
    method, layer: torch.nn.Module, x, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Fp8LinearMethod.apply on a gfx950 route. x is a bf16 tensor, an (fp8, scale)
    tuple from a fused quant kernel, or one of the wrappers the fused producers emit."""
    backend = method.mxfp8_dense_backend
    mxfp8_ready = layer.block_fp8_mxfp8_ready
    native_route = mxfp8_ready and backend.is_gfx95_mxfp8_native()
    aiter_route = mxfp8_ready and backend.is_gfx95_mxfp8_aiter()
    # Unwrap the producer's operand: the native and aiter routes take fp8 + scales directly,
    # the native one also the fp8-grid bf16; the other routes quantize the plain tensor
    # themselves (per-32 rounding is idempotent, so the wrapper's rounding is exact for them).
    input_scale, on_fp8_grid = None, False
    if isinstance(x, Mxfp8Activation):
        if native_route or aiter_route:
            x, input_scale = x.q, x.scale
        else:
            x = dequant_mxfp8_to_bf16(x.q, x.scale)
    elif isinstance(x, Fp8GridActivation):
        x, on_fp8_grid = x.x, native_route
    elif isinstance(x, tuple):
        x, input_scale = x
    if native_route:
        return _apply_native(method, layer, x, bias, input_scale, on_fp8_grid)
    if aiter_route:
        if (
            layer.weight_scale_mx_flydsl is not None
            and x.numel() // x.shape[-1] >= _FLYDSL_MXFP8_MIN_M
        ):
            return _apply_flydsl_mxfp8(layer, x, bias, input_scale)
        return method.w8a8_mxfp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale_ue8m0=layer.weight_scale_mx_e8m0,
            input_scale=input_scale,
            bias=bias,
            weight_preshuffled=getattr(layer, "mxfp8_aiter_preshuffled", False),
        )
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


@functools.cache
def _flydsl_mxfp8_module():
    """The mxfp8_flydsl_gfx95 module, or None when FlyDSL is missing."""
    try:
        from sglang.kernels.ops.gemm import mxfp8_flydsl_gfx95
    except ImportError as e:
        logger.warning("SGLANG_OPT_HIP_FLYDSL_MXFP8 ignored, FlyDSL unavailable: %s", e)
        return None
    return mxfp8_flydsl_gfx95


def _prepare_flydsl_mxfp8(layer: torch.nn.Module) -> None:
    """
    Weight scales in mxfp8_flydsl_gfx95's layout when SGLANG_OPT_HIP_FLYDSL_MXFP8 is set
    and the kernel takes this weight; weight_scale_mx_flydsl stays None otherwise.
    """
    n, k = layer.weight.shape
    if (
        not envs.SGLANG_OPT_HIP_FLYDSL_MXFP8.get()
        or k % 128
        or n % 8
        or not layer.weight.is_contiguous()
        or _flydsl_mxfp8_module() is None
    ):
        layer.weight_scale_mx_flydsl = None
        return
    # one row of block scales per output row
    rows = layer.weight_scale_mx_e8m0.repeat_interleave(32, dim=0)[:n]
    copy_or_rebind_param(
        layer, "weight_scale_mx_flydsl", _flydsl_mxfp8_module().mxfp8_scale_layout(rows)
    )


def _apply_flydsl_mxfp8(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """The aiter route's GEMM on mxfp8_flydsl_gfx95."""
    xq, xs = mxfp8_operand(x, input_scale)
    flydsl = _flydsl_mxfp8_module()
    out = flydsl.mxfp8_gemm(
        xq, flydsl.mxfp8_scale_layout(xs), layer.weight, layer.weight_scale_mx_flydsl
    )
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], out.shape[-1])


def _apply_native(
    method,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor] = None,
    input_on_fp8_grid: bool = False,
) -> torch.Tensor:
    """The native route (mxfp8_native_amd_gfx95); a layer whose shape the native
    kernels do not tile keeps the bf16-dequant route."""
    if layer.mxfp8_native_ready:
        # an fp8-grid bf16 input re-encodes exactly, so it needs no flag here
        return method.w8a8_mxfp8_linear(
            input=x,
            weight_shuffled=layer.weight.view(torch.uint8),
            weight_scale_ue8m0=layer.weight_scale_mx_e8m0,
            input_scale=input_scale,
            bias=bias,
        )
    if input_scale is not None:
        x = dequant_mxfp8_to_bf16(x, input_scale)
        input_on_fp8_grid = True
    return bf16_dequant_blockscaled_linear(
        x, layer.weight_bf16, bias=bias, input_on_fp8_grid=input_on_fp8_grid
    )
