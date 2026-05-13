# NOTE: this file will be separated into sglang/srt/layers/moe/moe_runner/triton_utils.py
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.debug_utils.deepseek_v4_debug_utils import deepseek_v4_moe_code_path_checker
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_xpu,
)
from sglang.srt.utils.custom_op import register_custom_op

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import (
    act_and_mul_triton,
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
    support_tensor_descriptor,
)
from .moe_align_block_size import moe_align_block_size

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel import gelu_and_mul, moe_sum_reduce, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _use_aiter:
        try:
            from aiter import moe_sum
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
    else:
        from vllm import _custom_ops as vllm_ops
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul

# Try to import vllm_ops for non-CUDA/HIP/XPU platforms
_has_vllm_ops = False
if not _is_cuda and not _is_hip and not _is_xpu:
    try:
        from vllm import _custom_ops as vllm_ops

        _has_vllm_ops = True
    except ImportError:
        # Fallback: vllm not available, will use native PyTorch implementations
        _has_vllm_ops = False

padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0

logger = logging.getLogger(__name__)


def _is_mxfp4_xpu_packed(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
) -> bool:
    """Detect MXFP4-packed routed-expert weights on XPU.

    The DSv4 fp8 checkpoint loader passes ``use_fp8_w8a8=True`` for routed
    experts that are actually MXFP4 (e.g. DeepSeek-V4-Flash), so we must
    NOT exclude on ``use_fp8_w8a8``. We discriminate MXFP4 from real FP8
    weights via the packed-last-dim invariant:
        - MXFP4 routed experts: w1.shape[-1] == hidden_size // 2
        - FP8  shared experts : w1.shape[-1] == hidden_size  (skip)
    """
    return (
        _is_xpu
        and not (use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
        and (w1.dtype == torch.uint8 or w1.dtype == torch.int8)
        and (w2.dtype == torch.uint8 or w2.dtype == torch.int8)
        and w1_scale is not None
        and w2_scale is not None
        and w1.shape[-1] * 2 == hidden_states.shape[-1]
    )


# E2M1 lookup table: nibble value 0x0–0xF → float
_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,        # 0b0xxx (positive)
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # 0b1xxx (negative)
    ],
    dtype=torch.float32,
)

# Per-(device, dtype) cache of the LUT to avoid the per-call host->device
# copy + sync that ``_E2M1_LUT.to(device=..., dtype=...)`` triggers on XPU.
_E2M1_LUT_CACHE: dict = {}


def _get_e2m1_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    cached = _E2M1_LUT_CACHE.get(key)
    if cached is None:
        cached = _E2M1_LUT.to(device=device, dtype=dtype)
        _E2M1_LUT_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Triton MXFP4 dequant kernel: replaces the PyTorch loop-based upcast.
# One kernel launch converts [E, N, half_K] packed uint8 -> [E, N, K] bf16
# with fused block-scale multiplication.  No int64 intermediates, no
# host syncs, no thousands of small kernel launches.
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_dequant_kernel(
    W_ptr,     # [E*N, half_K] uint8  - packed weights (flattened E*N)
    S_ptr,     # [E*N, half_K // 16] float32 - block scales
    LUT_ptr,   # [16] bfloat16 - E2M1 lookup table
    Out_ptr,   # [E*N, K] bfloat16 - output
    half_K,    # K // 2  (packed dimension)
    stride_wn,
    stride_sn,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dequantise MXFP4-packed weights to bf16 with block-scale fusion.

    Grid: (cdiv(E*N, BLOCK_N), cdiv(half_K, BLOCK_K))
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    w_off = rn[:, None] * stride_wn + rk[None, :]
    mask = rk[None, :] < half_K
    packed = tl.load(W_ptr + w_off, mask=mask, other=0)

    lo = (packed & 0xF).to(tl.int32)
    hi = ((packed >> 4) & 0xF).to(tl.int32)

    lo_val = tl.load(LUT_ptr + lo).to(tl.float32)
    hi_val = tl.load(LUT_ptr + hi).to(tl.float32)

    scale_col = rk[None, :] // 16
    s_off = rn[:, None] * stride_sn + scale_col
    scale = tl.load(S_ptr + s_off, mask=mask, other=1.0).to(tl.float32)

    lo_out = (lo_val * scale).to(tl.bfloat16)
    hi_out = (hi_val * scale).to(tl.bfloat16)

    out_off_lo = rn[:, None] * stride_on + rk[None, :] * 2
    out_off_hi = out_off_lo + 1
    tl.store(Out_ptr + out_off_lo, lo_out, mask=mask)
    tl.store(Out_ptr + out_off_hi, hi_out, mask=mask)


def _upcast_mxfp4_triton(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Triton-accelerated MXFP4 -> bf16 dequant with fused block-scale multiply.

    Replaces the PyTorch loop that generated thousands of small kernel
    launches with a single Triton kernel launch per weight tensor.

    w_packed : [E, N, K//2]  uint8   - two E2M1 values per byte
    w_scale  : [E, N, K//32] float32 - MX block scale (direct multiplier)
    Returns  : [E, N, K]     target_dtype - contiguous
    """
    w_u8 = w_packed.view(torch.uint8).contiguous()
    E, N, half_K = w_u8.shape
    K = half_K * 2

    lut = _get_e2m1_lut(w_u8.device, torch.bfloat16).contiguous()
    out = torch.empty(E, N, K, dtype=torch.bfloat16, device=w_u8.device)

    w_flat = w_u8.reshape(E * N, half_K)
    s_flat = w_scale.to(torch.float32).reshape(E * N, half_K // 16).contiguous()
    out_flat = out.reshape(E * N, K)

    stride_wn = half_K
    stride_sn = half_K // 16
    stride_on = K

    BLOCK_N = 4
    BLOCK_K = min(128, half_K)
    total_rows = E * N
    grid = (
        triton.cdiv(total_rows, BLOCK_N),
        triton.cdiv(half_K, BLOCK_K),
    )

    _mxfp4_dequant_kernel[grid](
        w_flat, s_flat, lut, out_flat,
        half_K,
        stride_wn, stride_sn, stride_on,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if target_dtype != torch.bfloat16:
        out = out.to(target_dtype)
    return out


def _upcast_mxfp4_one_xpu(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Upcast MXFP4-packed expert weights to *target_dtype*.

    Uses a fused Triton dequant kernel (single kernel launch).

    w_packed : [E, N, K//2]  uint8/int8   — two E2M1 values per byte
    w_scale  : [E, N, K//32] float32      — MX block scale (direct multiplier)
    Returns  : [E, N, K]     target_dtype — contiguous
    """
    return _upcast_mxfp4_triton(w_packed, w_scale, target_dtype)


def _log_mxfp4_xpu_budget(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    target_dtype: torch.dtype,
) -> None:
    """Diagnostic: free/total XPU memory and per-weight bf16 transient cost.

    Logged at INFO so it's visible without raising the global log level;
    fires once per MXFP4 MoE call.
    """
    elem_size = torch.tensor([], dtype=target_dtype).element_size()
    # Packed weights have last dim = K // 2; bf16 output has last dim = K.
    w1_bytes = w1.numel() * 2 * elem_size
    w2_bytes = w2.numel() * 2 * elem_size
    try:
        free_bytes, total_dev_bytes = torch.xpu.mem_get_info(hidden_states.device)
        mem_str = (
            f"xpu_free={free_bytes / 1024**3:.2f} GiB "
            f"xpu_total={total_dev_bytes / 1024**3:.2f} GiB"
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        mem_str = f"xpu_free=<unavailable: {exc!r}>"
    logger.info(
        "MXFP4 upcast (sequenced) on %s: %s; per-GEMM transient w1=%.2f GiB, "
        "w2=%.2f GiB (target_dtype=%s, w1.shape=%s, w2.shape=%s, hidden=%d)",
        hidden_states.device,
        mem_str,
        w1_bytes / 1024**3,
        w2_bytes / 1024**3,
        target_dtype,
        tuple(w1.shape),
        tuple(w2.shape),
        hidden_states.shape[-1],
    )


@register_custom_op(mutates_args=["hidden_states"])
def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
        filter_expert,
        swiglu_limit=swiglu_limit,
    )


@register_custom_op(out_shape="hidden_states")
def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        is_gated,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        swiglu_limit=swiglu_limit,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    # MXFP4-packed routed experts on XPU are kept in their packed form
    # here and dequantized to bf16 lazily inside ``fused_experts_impl``,
    # interleaved with each GEMM call so peak transient memory is one
    # weight (~8 GiB w1 / ~4 GiB w2 at TP=1) instead of both at once.
    if moe_runner_config.inplace:
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
            filter_expert,
            moe_runner_config.swiglu_limit,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.is_gated,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            swiglu_limit=moe_runner_config.swiglu_limit,
        )


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@torch.compile
def swiglu_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


@functools.lru_cache()
def _down_moe_use_tma():
    return support_tensor_descriptor()


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
    swiglu_limit: Optional[float] = None,
):
    # MXFP4-packed routed experts on XPU arrive here as uint8 with last
    # dim = K/2 plus uint8 E8M0 scales. We dequantize to bf16 lazily, one
    # weight at a time, freeing each bf16 buffer between the up- and
    # down-projection GEMMs to keep peak transient at ~8 GiB (TP=1)
    # instead of ~12 GiB.
    mxfp4_xpu = _is_mxfp4_xpu_packed(
        hidden_states, w1, w2, w1_scale, w2_scale,
        use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
    )
    if mxfp4_xpu:
        mxfp4_target_dtype = (
            hidden_states.dtype
            if hidden_states.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        # _log_mxfp4_xpu_budget(hidden_states, w1, w2, mxfp4_target_dtype)
        # The bf16 GEMM that follows must NOT see fp8/block-quant flags or
        # the packed scales: those are folded into the upcast result.
        gemm_use_fp8_w8a8 = False
        gemm_block_shape: Optional[List[int]] = None
        gemm_w1_scale: Optional[torch.Tensor] = None
        gemm_w2_scale: Optional[torch.Tensor] = None
        padded_size = 0
    else:
        gemm_use_fp8_w8a8 = use_fp8_w8a8
        gemm_block_shape = block_shape
        gemm_w1_scale = w1_scale
        gemm_w2_scale = w2_scale
        padded_size = padding_size
        if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
            padded_size = 0

    # Check constraints.
    if mxfp4_xpu:
        # Packed last dim is K/2; bf16 last dim after upcast is K.
        assert (
            hidden_states.shape[1] == w1.shape[2] * 2
        ), "MXFP4 packed hidden dim mismatch"
    elif use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=gemm_use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    # MXFP4 weights are packed with last dim = K/2; the GEMM operates on
    # the bf16 tensor with last dim = K, so feed the unpacked shape to
    # the tile-config selector.
    if mxfp4_xpu:
        w1_shape_for_cfg = (w1.shape[0], w1.shape[1], hidden_states.shape[1])
        w2_shape_for_cfg = (w2.shape[0], w2.shape[1], w2.shape[2] * 2)
    else:
        w1_shape_for_cfg = w1.shape
        w2_shape_for_cfg = (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1_shape_for_cfg,
        w2_shape_for_cfg,
        topk_ids.shape[1],
        config_dtype,
        block_shape=gemm_block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )

    config, (down_config, max_block_m) = get_config_func(M)
    down_moe_use_tma = (
        _down_moe_use_tma()
        and down_config is not None
        and down_config.pop("USE_TMA", False)
    )
    topk = topk_ids.shape[1]
    max_padded_tokens = (
        min(M * topk, E + 1) * (max_block_m - 1) if down_moe_use_tma else 0
    )
    total_tokens = M * topk + max_padded_tokens
    cache = torch.empty(
        total_tokens * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk * w2.shape[1]].view(
        (M, topk, w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            config, (down_config, _) = get_config_func(tokens_in_chunk)
            down_moe_use_tma = (
                _down_moe_use_tma()
                and down_config is not None
                and down_config.pop("USE_TMA", False)
            )
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        padded_tokens = (
            min(tokens_in_chunk * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1)
            if down_moe_use_tma
            else 0
        )
        total_tokens = tokens_in_chunk * topk + padded_tokens
        intermediate_cache1 = cache[: total_tokens * N].view(
            (total_tokens, N),
        )
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        # MXFP4 (XPU): upcast w1 just-in-time. Released right after GEMM1
        # so w2's upcast doesn't have to share the budget.
        if mxfp4_xpu:
            w1_eff = _upcast_mxfp4_one_xpu(w1, w1_scale, mxfp4_target_dtype)
        else:
            w1_eff = w1

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1_eff,
            b1,
            intermediate_cache1,
            a1_scale,
            gemm_w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=gemm_use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=gemm_block_shape,
            c_sorted=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        # Drop the bf16 w1 reference so the allocator can reuse its block
        # for the w2 upcast that follows GEMM1.
        if mxfp4_xpu:
            del w1_eff

        # Activation function with multiplication
        if activation == "silu" and is_gated:
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = swiglu_with_alpha_and_limit(
                    intermediate_cache1.view(-1, N),
                    gemm1_alpha,
                    gemm1_limit,
                )
            else:
                is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"
                assert is_2604b == (swiglu_limit is not None), (
                    f"swiglu_limit must be non-None iff submode=2604B "
                    f"(got submode={envs.SGLANG_DSV4_2604_SUBMODE.get()!r}, swiglu_limit={swiglu_limit!r})"
                )

                swiglu_limit_for_triton: Optional[float] = None
                swiglu_limit_for_silu_and_mul_clamp: Optional[float] = None
                if is_2604b:
                    assert swiglu_limit == 10
                    assert intermediate_cache1.shape == (total_tokens, N)
                    assert (
                        _is_cuda or _is_hip or _is_xpu
                    ), "DSV4 2604 submode 2604B only supports CUDA/HIP/XPU downstream"

                    if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
                        if filter_expert:
                            swiglu_limit_for_triton = swiglu_limit
                        else:
                            assert (
                                _is_cuda
                            ), "fused silu_and_mul_clamp kernel is CUDA-only; HIP must disable SWIGLU_CLAMP_FUSION"
                            swiglu_limit_for_silu_and_mul_clamp = swiglu_limit
                    else:
                        half = N // 2
                        intermediate_cache1[:, :half].clamp_(max=swiglu_limit)
                        intermediate_cache1[:, half:].clamp_(
                            min=-swiglu_limit, max=swiglu_limit
                        )
                        deepseek_v4_moe_code_path_checker.observed += 1

                if _is_cuda or _is_hip or _is_xpu:
                    if not filter_expert:
                        if swiglu_limit_for_silu_and_mul_clamp is not None:
                            from sglang.jit_kernel.deepseek_v4 import silu_and_mul_clamp

                            silu_and_mul_clamp(
                                intermediate_cache1.view(-1, N),
                                intermediate_cache2,
                                swiglu_limit_for_silu_and_mul_clamp,
                            )
                        else:
                            silu_and_mul(
                                intermediate_cache1.view(-1, N), intermediate_cache2
                            )
                    else:
                        act_and_mul_triton(
                            intermediate_cache1.view(-1, N),
                            intermediate_cache2,
                            config,
                            topk_ids,
                            expert_ids,
                            down_moe_use_tma,
                            activation,
                            swiglu_limit=swiglu_limit_for_triton,
                        )
                else:
                    vllm_ops.silu_and_mul(
                        intermediate_cache2, intermediate_cache1.view(-1, N)
                    )
        elif activation == "gelu" and is_gated:
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                if not filter_expert:
                    gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
                else:
                    act_and_mul_triton(
                        intermediate_cache1.view(-1, N),
                        intermediate_cache2,
                        config,
                        topk_ids,
                        expert_ids,
                        down_moe_use_tma,
                        activation,
                    )
            else:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        # Activation function without multiplication
        elif activation == "silu" and not is_gated:
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu" and not is_gated:
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))
        elif activation == "relu2" and not is_gated:
            intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
        else:
            raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

        # MXFP4 (XPU): upcast w2 just-in-time for GEMM2.
        if mxfp4_xpu:
            w2_eff = _upcast_mxfp4_one_xpu(w2, w2_scale, mxfp4_target_dtype)
        else:
            w2_eff = w2

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2_eff,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            a2_scale,
            gemm_w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            down_config or config,
            compute_type=compute_type,
            use_fp8_w8a8=gemm_use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=gemm_block_shape,
            a_use_tma=down_moe_use_tma,
            b_use_tma=down_moe_use_tma,
            filter_expert=filter_expert,
        )

        if mxfp4_xpu:
            del w2_eff

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )

        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        elif _is_xpu:
            _force_triton = envs.SGLANG_FORCE_TRITON_MOE_FP8.get()
            if not _force_triton:
                moe_sum_reduce(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        else:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )

    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=moe_runner_config,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )
