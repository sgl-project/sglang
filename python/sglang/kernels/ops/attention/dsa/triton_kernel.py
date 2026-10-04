from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _supports_fp8() -> bool:
    """Check if current GPU supports FP8 compute (requires SM89+ for CUDA, or HIP)."""
    if not torch.cuda.is_available():
        return False

    from sglang.srt.utils import get_device_sm

    cc = get_device_sm()
    return cc >= 89


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )


import functools as _functools


@_functools.lru_cache(maxsize=1)
def is_fp4_indexer_cache_enabled() -> bool:
    """Resolve the effective FP4 indexer-cache flag.

    Combines the user opt-in (``SGLANG_OPT_USE_FP4_INDEXER_CACHE``) with a
    hardware capability check. DeepGEMM's ``fp8_fp4_mqa_logits`` /
    ``fp8_fp4_paged_mqa_logits`` are sm_100-only (Blackwell datacenter such
    as B200/GB200). On sm_90 (Hopper), sm_120 (consumer Blackwell), AMD/HIP,
    or older devices we silently fall back to the FP8 path so misconfigured
    runs don't crash mid-forward.

    Cached so the warning fires at most once per process."""
    from sglang.srt.environ import envs
    from sglang.srt.utils.common import get_device_sm

    if not envs.SGLANG_OPT_USE_FP4_INDEXER_CACHE.get():
        return False
    if get_device_sm() < 89:
        import logging

        logging.getLogger(__name__).warning(
            "SGLANG_OPT_USE_FP4_INDEXER_CACHE=1 was requested but the current "
            "device is not sm_89. FP4 indexer cache "
        )
        return False
    return True


MXFP_BLOCK_SIZE: tl.constexpr = tl.constexpr(32)


def _get_cuda_capability():
    """Get CUDA compute capability as (major, minor) tuple"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def supports_e2m1_ptx():
    """Check if GPU supports e2m1 PTX instruction (SM 8.9+)"""
    major, minor = _get_cuda_capability()
    return major > 8 or (major == 8 and minor >= 9)


@triton.jit
def _downcast_to_mxfp4_opt(
    # Output: packed e2m1 tensor (uint8, two fp4 per byte)
    mx_tensor_ptr,
    stride_mxt_outer,
    stride_mxt_quant: tl.constexpr,
    # Output: raw E8M0 scales (uint8, row-major [N, S_groups])
    mx_scale_ptr,
    stride_mx_scale_outer,
    stride_mx_scale_k,
    # Input: source tensor (bf16/fp16/fp32)
    src_ptr,
    stride_src_outer,
    stride_src_quant,
    outer_dim,
    quant_dim,
    orig_quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    USE_PTX: tl.constexpr,
):
    tl.static_assert(
        stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1."
    )
    tl.static_assert(
        BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
        f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32",
    )

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(
        mx_tensor_ptr.dtype.element_ty == tl.uint8,
        "Output must be uint8 (packed e2m1).",
    )
    tl.static_assert(
        mx_scale_ptr.dtype.element_ty == tl.int32,
        "Scale must be int32 (4 packed E8M0 bytes).",
    )
    tl.static_assert(
        (src_dtype == tl.bfloat16)
        or (src_dtype == tl.float16)
        or (src_dtype == tl.float32),
        f"{src_dtype=} must be bfloat16, float16, or float32",
    )
    IS_SRC_FP32: tl.constexpr = src_dtype == tl.float32

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = (
        BLOCK_SIZE_QUANT_DIM // 2
    )  # 2 e2m1 per uint8
    tl.static_assert(
        BLOCK_SIZE_QUANT_MX_SCALE == 4,
        "DeepSeek V4 indexer head_dim=128 produces exactly 4 E8M0 scales "
        "per tile, packed into one int32.",
    )

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_k = quant_block
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += (
        start_mx_scale_k * stride_mx_scale_k + start_out * stride_mx_scale_outer
    )
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < orig_quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < orig_quant_dim // 2
    full_mask_mxt = mask_mxt_quant & mask_n

    ORIG_S_GROUPS = orig_quant_dim // MXFP_BLOCK_SIZE
    ORIG_PACKED_S_GROUPS = ORIG_S_GROUPS // 4
    scale_mask_k = start_mx_scale_k < ORIG_PACKED_S_GROUPS
    full_scale_mask = (scale_mask_k & mask_n).reshape([BLOCK_SIZE_OUT_DIM])

    src_tensor_offsets = (
        offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    )
    mx_scale_offsets = offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = (
        offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    )
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(full_mask_src, abs_tensor, -1.0)
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)

    dequant_scale = max_val / 6.0
    dequant_scale_exponent = (
        dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
    ) & 0x7F800000

    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE]
    )
    quant_tensor = f32_tensor * quant_scale
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set invalid portions to 0 to ensure padding values are 0 in mx format
    quant_tensor = tl.where(full_mask_src, quant_tensor, 0)

    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )
    # Zero out exponents for padding groups (beyond original quant_dim)
    scale_group_idx = (
        quant_block * BLOCK_SIZE_QUANT_MX_SCALE
        + tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :]
    )
    dequant_scale_exponent = tl.where(
        scale_group_idx < ORIG_S_GROUPS, dequant_scale_exponent, 0
    )
    scale_tensor = (dequant_scale_exponent >> 23).to(tl.int32)
    # pack 4 e8m0 to int32
    scale_shift = (tl.arange(0, 4)[None, :] * 8).to(tl.int32)
    packed_scale = tl.sum(scale_tensor << scale_shift, axis=1)

    # ======== FP32 -> e2m1 conversion ========
    if USE_PTX:
        # SM 8.9+: PTX hardware instruction for e2m1 conversion
        pairs = tl.reshape(
            quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        lo_f, hi_f = tl.split(pairs)
        lo_f32 = lo_f.to(tl.float32)
        hi_f32 = hi_f.to(tl.float32)

        # cvt.rn.satfinite.e2m1x2.f32: two f32 -> one packed e2m1x2 (uint8)
        out_tensor = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f32, lo_f32],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )
    else:
        # Software RTNE bit manipulation fallback
        quant_uint32 = quant_tensor.to(tl.uint32, bitcast=True)
        exponents = (quant_uint32 >> 23) & 0xFF
        mantissas_orig = quant_uint32 & 0x7FFFFF

        E8_BIAS = 127
        E2_BIAS = 1

        # Subnormal handling: move implicit bit into mantissa
        is_subnormal = exponents < E8_BIAS
        adjusted_exponents = tl.core.sub(
            E8_BIAS, exponents + 1, sanitize_overflow=False
        )
        mantissas_pre = 0x400000 | (mantissas_orig >> 1)
        mantissas = tl.where(
            is_subnormal, mantissas_pre >> adjusted_exponents, mantissas_orig
        )

        # Rebias exponents: E8 bias 127 -> E2 bias 1
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # RTNE rounding: guard / sticky / lsb
        m2bits = mantissas >> 21
        lsb_keep = (m2bits >> 1) & 0x1
        guard = m2bits & 0x1
        if IS_SRC_FP32:
            bit0_dropped = (mantissas_orig & 0x1) != 0
            mask = (1 << tl.minimum(adjusted_exponents, 31)) - 1
            dropped_post = (mantissas_pre & mask) != 0
            sticky = is_subnormal & (bit0_dropped | dropped_post)
            sticky |= ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        else:
            sticky = ((mantissas & 0x1FFFFF) != 0).to(tl.uint32)
        round_inc = guard & (sticky | lsb_keep)

        # Combine sign + magnitude in single step: (sign_bit >> 28) gives 0x8 for negative
        e2m1_tmp = tl.minimum((((exponents << 2) | m2bits) + round_inc) >> 1, 0x7)
        e2m1_value = (((quant_uint32 >> 28) & 0x8) | e2m1_tmp).to(tl.uint8)

        # Pack 2 e2m1 into 1 uint8: low nibble = even, high nibble = odd
        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    # Write scale (int32, row-major [N, S_groups]) and tensor
    tl.store(
        mx_scale_ptr + mx_scale_offsets.reshape([BLOCK_SIZE_OUT_DIM]),
        packed_scale,
        mask=full_scale_mask,
    )
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


def downcast_to_mxfp4_indexer(
    src_tensor: torch.Tensor,
    axis: int,
):
    """Convert src tensor to MXFP4 (packed e2m1 uint8) with raw E8M0 scales.

    Scale output is raw E8M0 exponent bytes in row-major layout (no preprocessing).

    Args:
        src_tensor: Input tensor (bf16/fp16/fp32).
        axis:       Quantization axis.
        round_up:   True for ROUND_UP scale mode, False for ROUND_DOWN.

    Returns:
        (out_quant_tensor, out_scale):
          - out_quant_tensor: uint8, two e2m1 packed per byte.
            Shape is same as src_tensor except axis dim is halved.
          - out_scale: int32, 4 raw E8M0 exponent bytes pack into 1 int32.
            Shape: [..., S_groups] (quant axis replaced by ceil(L / 32 / 4 )).
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    src_tensor = src_tensor.transpose(axis, ndim - 1)
    L = src_tensor.shape[-1]
    assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    assert L % 32 == 0, f"axis dim must be divisible by 32 for MXFP block. Got {L}"

    S_groups = L // MXFP_BLOCK_SIZE.value

    # Pad quant_dim to multiple of 32 (BLOCK_SIZE_QUANT_DIM alignment for kernel)
    # No actual data padding — kernel uses mask for OOB elements
    padded_L = ((L + 127) // 128) * 128  # align to BLOCK_QUANT=128

    out_shape = src_tensor.shape[:-1] + (L // 2,)
    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        N = kernel_src.shape[0]  # flattened outer dim

        # Scale output: [N, S_groups] uint8 contiguous (row-major)
        kernel_scale = torch.empty(
            (N, S_groups // 4), dtype=torch.int32, device=src_tensor.device
        )

        BLOCK_OUT = 32
        BLOCK_QUANT = 128
        NUM_WARPS = 2

        grid = (
            triton.cdiv(N, BLOCK_OUT),
            triton.cdiv(padded_L, BLOCK_QUANT),
        )

        _downcast_to_mxfp4_opt[grid](
            kernel_quant,
            *kernel_quant.stride(),
            kernel_scale,
            *kernel_scale.stride(),
            kernel_src,
            *kernel_src.stride(),
            N,
            padded_L,
            L,
            BLOCK_SIZE_OUT_DIM=BLOCK_OUT,
            BLOCK_SIZE_QUANT_DIM=BLOCK_QUANT,
            USE_PTX=supports_e2m1_ptx(),
            num_warps=NUM_WARPS,
        )

        # Reshape scale from [N, S_groups] to logical [..., S_groups]
        batch_shape = src_tensor.shape[:-1]
        out_scale = kernel_scale.reshape(*batch_shape, 1)
        # out_scale = pack_e8m0_scales_to_int32(out_scale)
    else:
        batch_shape = src_tensor.shape[:-1]
        out_scale = torch.empty(
            *batch_shape, S_groups, dtype=torch.uint8, device=src_tensor.device
        )

    out_quant_tensor = out_quant_tensor.transpose(axis, ndim - 1)
    out_scale = out_scale.transpose(axis, ndim - 1)
    return out_quant_tensor, out_scale.squeeze(-1)
