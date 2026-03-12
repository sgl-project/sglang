from typing import Optional, Tuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["inner_dim"],
)
@triton.jit
def _fused_scale_shift_4d_kernel(
    output_ptr,
    normalized_ptr,
    scale_ptr,
    shift_ptr,
    scale_constant: tl.constexpr,  # scale_constant is either 0 or 1.
    rows,
    inner_dim,
    seq_len,
    num_frames,
    frame_seqlen,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    col_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col_offsets < inner_dim

    # Pointers for normalized and output
    row_base = pid_row * inner_dim
    norm_ptrs = normalized_ptr + row_base + col_offsets
    out_ptrs = output_ptr + row_base + col_offsets

    # Pointers for scale (per-frame) and shift (per-token)
    b_idx = pid_row // seq_len
    t_idx = pid_row % seq_len
    frame_idx_in_batch = t_idx // frame_seqlen

    scale_row_idx = b_idx * num_frames + frame_idx_in_batch
    scale_ptrs = scale_ptr + scale_row_idx * inner_dim + col_offsets
    # shift is per-token [B*L, C], indexed by pid_row directly
    shift_ptrs = shift_ptr + pid_row * inner_dim + col_offsets

    normalized = tl.load(norm_ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0)

    scale_const_tensor = tl.full([BLOCK_N], scale_constant, dtype=scale.dtype)
    output = normalized * (scale_const_tensor + scale) + shift

    tl.store(out_ptrs, output, mask=mask)


@triton.jit
def fuse_scale_shift_kernel_blc_opt(
    x_ptr,
    shift_ptr,
    scale_ptr,
    scale_constant: tl.constexpr,  # scale_constant is either 0 or 1.,
    y_ptr,
    B,
    L,
    C,
    stride_x_b,
    stride_x_l,
    stride_x_c,
    stride_s_b,
    stride_s_l,
    stride_s_c,
    stride_sc_b,
    stride_sc_l,
    stride_sc_c,
    SCALE_IS_SCALAR: tl.constexpr,
    SHIFT_IS_SCALAR: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_l = l_offsets < L
    mask_c = c_offsets < C
    mask = mask_l[:, None] & mask_c[None, :]

    x_off = (
        pid_b * stride_x_b
        + l_offsets[:, None] * stride_x_l
        + c_offsets[None, :] * stride_x_c
    )
    x = tl.load(x_ptr + x_off, mask=mask, other=0)

    if SHIFT_IS_SCALAR:
        shift_val = tl.load(shift_ptr)
        shift = tl.full((BLOCK_L, BLOCK_C), shift_val, dtype=shift_val.dtype)
    else:
        s_off = (
            pid_b * stride_s_b
            + l_offsets[:, None] * stride_s_l
            + c_offsets[None, :] * stride_s_c
        )
        shift = tl.load(shift_ptr + s_off, mask=mask, other=0)

    if SCALE_IS_SCALAR:
        scale_val = tl.load(scale_ptr)
        scale = tl.full((BLOCK_L, BLOCK_C), scale_val, dtype=scale_val.dtype)
    else:
        sc_off = (
            pid_b * stride_sc_b
            + l_offsets[:, None] * stride_sc_l
            + c_offsets[None, :] * stride_sc_c
        )
        scale = tl.load(scale_ptr + sc_off, mask=mask, other=0)

    y = x * (scale_constant + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)


@triton.jit
def fuse_scale_shift_gate_select01_kernel_blc_opt(
    x_ptr,
    shift0_ptr,
    scale0_ptr,
    gate0_ptr,
    shift1_ptr,
    scale1_ptr,
    gate1_ptr,
    index_ptr,
    y_ptr,
    gate_out_ptr,
    B,
    L,
    C,
    stride_x_b,
    stride_x_l,
    stride_x_c,
    stride_s0_b,
    stride_s0_c,
    stride_sc0_b,
    stride_sc0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sc1_b,
    stride_sc1_c,
    stride_g1_b,
    stride_g1_c,
    stride_i_b,
    stride_i_l,
    stride_go_b,
    stride_go_l,
    stride_go_c,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_l = l_offsets < L
    mask_c = c_offsets < C
    mask = mask_l[:, None] & mask_c[None, :]

    x_off = (
        pid_b * stride_x_b
        + l_offsets[:, None] * stride_x_l
        + c_offsets[None, :] * stride_x_c
    )
    x = tl.load(x_ptr + x_off, mask=mask, other=0)

    idx_off = pid_b * stride_i_b + l_offsets * stride_i_l
    idx = tl.load(index_ptr + idx_off, mask=mask_l, other=0).to(tl.int1)[
        :, None
    ]  # [BLOCK_L, 1]

    s0_off = pid_b * stride_s0_b + c_offsets[None, :] * stride_s0_c  # [1, BLOCK_C]
    sc0_off = pid_b * stride_sc0_b + c_offsets[None, :] * stride_sc0_c
    g0_off = pid_b * stride_g0_b + c_offsets[None, :] * stride_g0_c
    s1_off = pid_b * stride_s1_b + c_offsets[None, :] * stride_s1_c
    sc1_off = pid_b * stride_sc1_b + c_offsets[None, :] * stride_sc1_c
    g1_off = pid_b * stride_g1_b + c_offsets[None, :] * stride_g1_c

    shift0 = tl.load(shift0_ptr + s0_off, mask=mask_c[None, :], other=0)
    scale0 = tl.load(scale0_ptr + sc0_off, mask=mask_c[None, :], other=0)
    gate0 = tl.load(gate0_ptr + g0_off, mask=mask_c[None, :], other=0)
    shift1 = tl.load(shift1_ptr + s1_off, mask=mask_c[None, :], other=0)
    scale1 = tl.load(scale1_ptr + sc1_off, mask=mask_c[None, :], other=0)
    gate1 = tl.load(gate1_ptr + g1_off, mask=mask_c[None, :], other=0)

    shift = tl.where(idx, shift1, shift0)  # [BLOCK_L, BLOCK_C]
    scale = tl.where(idx, scale1, scale0)
    gate = tl.where(idx, gate1, gate0)

    y = x * (1 + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)

    go_off = (
        pid_b * stride_go_b
        + l_offsets[:, None] * stride_go_l
        + c_offsets[None, :] * stride_go_c
    )
    tl.store(gate_out_ptr + go_off, gate, mask=mask)


def fuse_scale_shift_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
    block_l: int = 128,
    block_c: int = 128,
):
    assert x.is_cuda and scale.is_cuda
    assert x.is_contiguous()

    B, L, C = x.shape
    output = torch.empty_like(x)

    if scale.dim() == 4:
        # scale/shift: [B, F, 1, C]
        rows = B * L
        x_2d = x.view(rows, C)
        output_2d = output.view(rows, C)
        grid = lambda META: (rows, triton.cdiv(C, META["BLOCK_N"]))
        num_frames = scale.shape[1]
        assert (
            L % num_frames == 0
        ), "seq_len must be divisible by num_frames for 4D scale/shift"
        frame_seqlen = L // num_frames

        # Compact scale [B, F, 1, C] -> [B*F, C] (per-frame)
        scale_reshaped = scale.squeeze(2).reshape(-1, C).contiguous()
        # shift is per-token [B, L, C] -> [B*L, C]
        shift_reshaped = shift.reshape(rows, C).contiguous()

        _fused_scale_shift_4d_kernel[grid](
            output_2d,
            x_2d,
            scale_reshaped,
            shift_reshaped,
            scale_constant,
            rows,
            C,
            L,
            num_frames,
            frame_seqlen,
        )
    else:
        # 2D: [B, C] or [1, C]  -> treat as [B, 1, C] and broadcast over L
        # 3D: [B, L, C] (or broadcastable variants like [B, 1, C], [1, L, C], [1, 1, C])
        # Also support scalar (0D or 1-element)
        if scale.dim() == 0 or (scale.dim() == 1 and scale.numel() == 1):
            scale_blc = scale.reshape(1)
        elif scale.dim() == 2:
            scale_blc = scale[:, None, :]
        elif scale.dim() == 3:
            scale_blc = scale
        else:
            raise ValueError("scale must be 0D/1D(1)/2D/3D or 4D")

        if shift.dim() == 0 or (shift.dim() == 1 and shift.numel() == 1):
            shift_blc = shift.reshape(1)
        elif shift.dim() == 2:
            shift_blc = shift[:, None, :]
        elif shift.dim() == 3:
            shift_blc = shift
        else:
            # broadcast later via expand if possible
            shift_blc = shift

        need_scale_scalar = scale_blc.dim() == 1 and scale_blc.numel() == 1
        need_shift_scalar = shift_blc.dim() == 1 and shift_blc.numel() == 1

        if not need_scale_scalar:
            scale_exp = scale_blc.expand(B, L, C)
            s_sb, s_sl, s_sc = scale_exp.stride()
        else:
            s_sb = s_sl = s_sc = 0

        if not need_shift_scalar:
            shift_exp = shift_blc.expand(B, L, C)
            sh_sb, sh_sl, sh_sc = shift_exp.stride()
        else:
            sh_sb = sh_sl = sh_sc = 0

        # If both scalars and both zero, copy fast-path
        if need_scale_scalar and need_shift_scalar:
            if not (
                scale_blc.any().to("cpu", non_blocking=True)
                or shift_blc.any().to("cpu", non_blocking=True)
            ):
                output.copy_(x)
                return output

        grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
        fuse_scale_shift_kernel_blc_opt[grid](
            x,
            shift_blc if need_shift_scalar else shift_exp,
            scale_blc if need_scale_scalar else scale_exp,
            scale_constant,
            output,
            B,
            L,
            C,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sh_sb,
            sh_sl,
            sh_sc,
            s_sb,
            s_sl,
            s_sc,
            SCALE_IS_SCALAR=need_scale_scalar,
            SHIFT_IS_SCALAR=need_shift_scalar,
            BLOCK_L=block_l,
            BLOCK_C=block_c,
            num_warps=4,
            num_stages=2,
        )
    return output


def fuse_scale_shift_gate_select01_kernel(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    block_l: int = 128,
    block_c: int = 128,
):
    assert x.is_contiguous()
    B, L, C = x.shape
    output = torch.empty_like(x)
    gate_out = torch.empty_like(x)

    if (
        scale0.dim() != 2
        or shift0.dim() != 2
        or gate0.dim() != 2
        or scale1.dim() != 2
        or shift1.dim() != 2
        or gate1.dim() != 2
    ):
        raise ValueError("scale0/shift0/gate0/scale1/shift1/gate1 must be 2D [B, C]")
    if index.dim() != 2:
        raise ValueError("index must be 2D [B, L]")

    grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
    fuse_scale_shift_gate_select01_kernel_blc_opt[grid](
        x,
        shift0,
        scale0,
        gate0,
        shift1,
        scale1,
        gate1,
        index,
        output,
        gate_out,
        B,
        L,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        shift0.stride(0),
        shift0.stride(1),
        scale0.stride(0),
        scale0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        gate_out.stride(0),
        gate_out.stride(1),
        gate_out.stride(2),
        BLOCK_L=block_l,
        BLOCK_C=block_c,
        num_warps=4,
        num_stages=2,
    )
    return output, gate_out


@triton.jit
def _fused_modulate_kernel(
    # Input tensors
    x_ptr,
    residual_ptr,  # Optional: for norm2
    gate_x_ptr,  # Optional: for norm2
    # LayerNorm params
    ln_weight_ptr,
    ln_bias_ptr,
    # Modulation params (two sets for index selection)
    shift0_ptr,
    scale0_ptr,
    gate0_ptr,
    shift1_ptr,
    scale1_ptr,
    gate1_ptr,
    # Index for selection
    index_ptr,
    # Output tensors
    y_ptr,
    residual_out_ptr,  # Optional: for norm2
    gate_out_ptr,
    # Dimensions
    B,
    L,
    C,
    # Strides for x/residual/gate_x (all [B, L, C])
    stride_x_b,
    stride_x_l,
    stride_x_c,
    stride_r_b,
    stride_r_l,
    stride_r_c,
    stride_gx_b,
    stride_gx_l,
    stride_gx_c,
    # Strides for modulation params (all [B, C])
    stride_s0_b,
    stride_s0_c,
    stride_sc0_b,
    stride_sc0_c,
    stride_g0_b,
    stride_g0_c,
    stride_s1_b,
    stride_s1_c,
    stride_sc1_b,
    stride_sc1_c,
    stride_g1_b,
    stride_g1_c,
    # Strides for index [B, L]
    stride_idx_b,
    stride_idx_l,
    # Strides for output
    stride_y_b,
    stride_y_l,
    stride_y_c,
    stride_ro_b,
    stride_ro_l,
    stride_ro_c,
    stride_go_b,
    stride_go_l,
    stride_go_c,
    # Constants
    eps,
    HAS_RESIDUAL: tl.constexpr,  # Whether to compute residual (norm2 vs norm1)
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel for _modulate operation:
    1. residual_out = gate_x * x + residual (if HAS_RESIDUAL)
    2. normed = LayerNorm(residual_out or x)
    3. Select (shift, scale, gate) based on index
    4. y = normed * (1 + scale) + shift

    Each program instance processes one token (row).
    """
    # Get batch and sequence indices
    pid = tl.program_id(0)
    b_idx = pid // L
    l_idx = pid % L

    # Compute base pointers for this token
    x_base = x_ptr + b_idx * stride_x_b + l_idx * stride_x_l
    y_base = y_ptr + b_idx * stride_y_b + l_idx * stride_y_l

    # Load input x
    cols = tl.arange(0, BLOCK_C)
    mask_c = cols < C
    x = tl.load(x_base + cols, mask=mask_c, other=0.0)

    # Step 1: Compute residual if needed
    if HAS_RESIDUAL:
        residual_base = residual_ptr + b_idx * stride_r_b + l_idx * stride_r_l
        gate_x_base = gate_x_ptr + b_idx * stride_gx_b + l_idx * stride_gx_l

        residual = tl.load(residual_base + cols, mask=mask_c, other=0.0).to(tl.float32)
        gate_x = tl.load(gate_x_base + cols, mask=mask_c, other=0.0).to(tl.float32)

        residual_out = gate_x * x + residual

        # Store residual_out
        ro_base = residual_out_ptr + b_idx * stride_ro_b + l_idx * stride_ro_l
        tl.store(ro_base + cols, residual_out, mask=mask_c)

        # Convert to float32 for layernorm
        x_for_norm = residual_out.to(tl.float32)
    else:
        # Convert to fp32 for layernorm
        x_for_norm = x.to(tl.float32)

    # Step 2: LayerNorm (compute mean, variance, normalize)
    mean = tl.sum(x_for_norm, axis=0) / C
    xbar = tl.where(mask_c, x_for_norm - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / C
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (x_for_norm - mean) * rstd

    # Apply layernorm weight and bias
    if HAS_WEIGHT:
        ln_w = tl.load(ln_weight_ptr + cols, mask=mask_c, other=1.0).to(tl.float32)
        x_hat = x_hat * ln_w
    if HAS_BIAS:
        ln_b = tl.load(ln_bias_ptr + cols, mask=mask_c, other=0.0).to(tl.float32)
        x_hat = x_hat + ln_b

    # Step 3: Load index and select modulation params
    idx_ptr = index_ptr + b_idx * stride_idx_b + l_idx * stride_idx_l
    idx = tl.load(idx_ptr).to(tl.int1)  # 0 or 1

    # Load both sets of params
    shift0 = tl.load(
        shift0_ptr + b_idx * stride_s0_b + cols * stride_s0_c, mask=mask_c, other=0.0
    )
    scale0 = tl.load(
        scale0_ptr + b_idx * stride_sc0_b + cols * stride_sc0_c, mask=mask_c, other=0.0
    )
    gate0 = tl.load(
        gate0_ptr + b_idx * stride_g0_b + cols * stride_g0_c, mask=mask_c, other=0.0
    )

    shift1 = tl.load(
        shift1_ptr + b_idx * stride_s1_b + cols * stride_s1_c, mask=mask_c, other=0.0
    )
    scale1 = tl.load(
        scale1_ptr + b_idx * stride_sc1_b + cols * stride_sc1_c, mask=mask_c, other=0.0
    )
    gate1 = tl.load(
        gate1_ptr + b_idx * stride_g1_b + cols * stride_g1_c, mask=mask_c, other=0.0
    )

    # Select based on index
    shift = tl.where(idx, shift1, shift0)
    scale = tl.where(idx, scale1, scale0)
    gate = tl.where(idx, gate1, gate0)

    # Step 4: Apply modulation
    y = x_hat * (1.0 + scale) + shift

    # Store outputs
    tl.store(y_base + cols, y, mask=mask_c)

    # Store gate result
    go_base = gate_out_ptr + b_idx * stride_go_b + l_idx * stride_go_l
    tl.store(go_base + cols, gate, mask=mask_c)


def fused_modulate_kernel(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    gate_x: Optional[torch.Tensor],
    ln_weight: Optional[torch.Tensor],
    ln_bias: Optional[torch.Tensor],
    shift0: torch.Tensor,
    scale0: torch.Tensor,
    gate0: torch.Tensor,
    shift1: torch.Tensor,
    scale1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Fused modulation kernel combining:
    1. Residual computation (if residual/gate_x provided)
    2. LayerNorm
    3. Index-based scale/shift/gate selection

    Args:
        x: Input tensor [B, L, C]
        residual: Residual tensor [B, L, C] (None for norm1)
        gate_x: Gate tensor [B, L, C] (None for norm1)
        ln_weight: LayerNorm weight [C]
        ln_bias: LayerNorm bias [C]
        shift0, scale0, gate0: Modulation params for index==0 [B, C]
        shift1, scale1, gate1: Modulation params for index==1 [B, C]
        index: Token index [B, L] (0 or 1)
        eps: LayerNorm epsilon

    Returns:
        y: Modulated output [B, L, C]
        residual_out: Updated residual [B, L, C] (or None)
        gate_out: Gate result [B, L, C]
    """
    B, L, C = x.shape

    # Ensure contiguous
    x = x.contiguous()
    if residual is not None:
        residual = residual.contiguous()
        gate_x = gate_x.contiguous()
    index = index.contiguous()

    # Output tensors
    y = torch.empty_like(x)
    gate_out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else None

    # Determine block size (must cover entire C for correct layernorm)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_C = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    if C > BLOCK_C:
        raise RuntimeError(f"Hidden dim {C} exceeds max fused size {BLOCK_C}")

    has_residual = residual is not None
    has_weight = ln_weight is not None
    has_bias = ln_bias is not None

    # Launch kernel: one program per token
    grid = (B * L,)

    _fused_modulate_kernel[grid](
        x,
        residual if has_residual else x,  # dummy
        gate_x if has_residual else x,  # dummy
        ln_weight if has_weight else x,  # dummy
        ln_bias if has_bias else x,  # dummy
        shift0.contiguous(),
        scale0.contiguous(),
        gate0.contiguous(),
        shift1.contiguous(),
        scale1.contiguous(),
        gate1.contiguous(),
        index,
        y,
        residual_out if has_residual else y,  # dummy
        gate_out,
        B,
        L,
        C,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        residual.stride(0) if has_residual else 0,
        residual.stride(1) if has_residual else 0,
        residual.stride(2) if has_residual else 0,
        gate_x.stride(0) if has_residual else 0,
        gate_x.stride(1) if has_residual else 0,
        gate_x.stride(2) if has_residual else 0,
        shift0.stride(0),
        shift0.stride(1),
        scale0.stride(0),
        scale0.stride(1),
        gate0.stride(0),
        gate0.stride(1),
        shift1.stride(0),
        shift1.stride(1),
        scale1.stride(0),
        scale1.stride(1),
        gate1.stride(0),
        gate1.stride(1),
        index.stride(0),
        index.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        residual_out.stride(0) if has_residual else 0,
        residual_out.stride(1) if has_residual else 0,
        residual_out.stride(2) if has_residual else 0,
        gate_out.stride(0),
        gate_out.stride(1),
        gate_out.stride(2),
        eps,
        HAS_RESIDUAL=has_residual,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        BLOCK_C=BLOCK_C,
        num_warps=min(max(BLOCK_C // 256, 1), 8),
    )

    return y, residual_out, gate_out


if current_platform.is_npu():
    from .npu_fallback import fuse_scale_shift_native

    fuse_scale_shift_kernel = fuse_scale_shift_native

if current_platform.is_mps():
    from .mps_fallback import (
        fuse_scale_shift_gate_select01_kernel_native,
        fuse_scale_shift_kernel_native,
    )

    fuse_scale_shift_kernel = fuse_scale_shift_kernel_native
    fuse_scale_shift_gate_select01_kernel = fuse_scale_shift_gate_select01_kernel_native
