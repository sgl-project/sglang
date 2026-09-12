"""ROCm activation route of ``DeepseekV2MLP``: aiter's fused clamp + silu-and-mul for 128-wide half
widths (fp8 out for a 128x128-block ``down_proj``), else the Triton silu-and-mul-clamp, whose gfx950
epilogue lands on ``down_proj``'s fp8 grid or emits native MXFP8. Bound by ``deepseek_v2`` under ``_is_hip``."""

from __future__ import annotations

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import (
    silu_and_mul_clamp_fp8_grid_supported,
    silu_and_mul_clamp_triton,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod


def resolve_fused_clamp_route(mlp, half_width: int) -> None:
    """Fix ``mlp``'s activation route from ``down_proj``'s loaded weight (once per layer)."""
    quant_method = getattr(mlp.down_proj, "quant_method", None)
    # the aiter kernel tiles and quantizes the half width per 128, the 128x128 block GEMM's layout
    mlp.use_fused_clamp_act_mul = half_width % 128 == 0
    mlp._fused_clamp_use_fp8 = (
        isinstance(quant_method, Fp8LinearMethod)
        and quant_method.block_quant
        and quant_method.weight_block_size == [128, 128]
    )
    # gfx950 32-block route: the activation lands on the fp8 grid, so down_proj skips its fake-quant
    mlp._hip_act_fp8_grid = bool(
        isinstance(quant_method, Fp8LinearMethod)
        and quant_method.block_fp8_as_mxfp8
        and mlp.down_proj.block_fp8_mxfp8_ready
        and quant_method.mxfp8_dense_backend.takes_fp8_grid_activation()
        and silu_and_mul_clamp_fp8_grid_supported(half_width)
    )
    # the native MXFP8 route takes fp8 + ue8m0 straight from the epilogue at decode token counts
    mlp._hip_act_native_consumer = bool(
        mlp._hip_act_fp8_grid
        and quant_method.mxfp8_dense_backend.is_gfx95_mxfp8_native()
        and mlp.down_proj.mxfp8_native_ready
    )
    mlp._fused_clamp_fp8_checked = True


def _emit_fp8(mlp, num_tokens: int) -> bool:
    """The silu fp8-grid epilogue hands down_proj fp8 + ue8m0 when its native kernel for
    this token count consumes it directly (skinny range, or a measured dot_scaled bucket)."""
    if not mlp._hip_act_native_consumer:
        return False
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        native_consumer_wants_fp8,
    )

    tiles, steps, _ = mlp.down_proj.weight.shape  # lane-order [N/16, K/128, 2048]
    return native_consumer_wants_fp8(num_tokens, tiles * 16, steps * 128)


def silu_and_mul_clamp(mlp, gate_up: torch.Tensor):
    """``silu(clamp(g)) * clamp(u)`` for any half width (unlike the aiter kernel's
    multiple of 128), emitted on ``down_proj``'s fp8 grid or as native fp8 + ue8m0
    when the resolved route takes it."""
    return silu_and_mul_clamp_triton(
        gate_up,
        float(mlp.swiglu_limit),
        fp8_grid=mlp._hip_act_fp8_grid,
        emit_fp8=_emit_fp8(mlp, gate_up.shape[0]),
    )
