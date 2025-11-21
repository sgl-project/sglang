# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# TODO: for temporary usage, expecting a refactor
from typing import Optional

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch import Tensor


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

    # Pointers for scale and shift for 4D
    b_idx = pid_row // seq_len
    t_idx = pid_row % seq_len
    frame_idx_in_batch = t_idx // frame_seqlen

    scale_row_idx = b_idx * num_frames + frame_idx_in_batch
    scale_ptrs = scale_ptr + scale_row_idx * inner_dim + col_offsets
    shift_ptrs = shift_ptr + scale_row_idx * inner_dim + col_offsets

    normalized = tl.load(norm_ptrs, mask=mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    shift = tl.load(shift_ptrs, mask=mask, other=0.0)

    one = tl.full([BLOCK_N], 1.0, dtype=scale.dtype)
    output = normalized * (one + scale) + shift

    tl.store(out_ptrs, output, mask=mask)


@triton.jit
def fuse_scale_shift_kernel_blc_opt(
    x_ptr,
    shift_ptr,
    scale_ptr,
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

    y = x * (1 + scale) + shift
    tl.store(y_ptr + x_off, y, mask=mask)


def fuse_scale_shift_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
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

        # Compact [B, F, C] without the singleton dim into [B*F, C]
        scale_reshaped = scale.squeeze(2).reshape(-1, C).contiguous()
        shift_reshaped = shift.squeeze(2).reshape(-1, C).contiguous()

        _fused_scale_shift_4d_kernel[grid](
            output_2d,
            x_2d,
            scale_reshaped,
            shift_reshaped,
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
            if (scale_blc.abs().max() == 0) and (shift_blc.abs().max() == 0):
                output.copy_(x)
                return output

        grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
        fuse_scale_shift_kernel_blc_opt[grid](
            x,
            shift_blc if need_shift_scalar else shift_exp,
            scale_blc if need_scale_scalar else scale_exp,
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved"],
)
@triton.jit
def _rotary_embedding_kernel(
    output_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    interleaved: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,
):
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row
    output_row_ptr = output_ptr + row_idx * stride_x_row

    # half size for x1 and x2
    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(output_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(output_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    output = torch.empty_like(x)

    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)
    output_reshaped = output.view(-1, head_size)

    # num_tokens per head, 1 token per block
    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    _rotary_embedding_kernel[grid](
        output_reshaped,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0),
        sin.stride(0),
        interleaved,
    )

    return output


# RMSNorm-fp32
def maybe_contiguous_lastdim(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def maybe_contiguous(x):
    return x.contiguous() if x is not None else None


def triton_autotune_configs():
    # Return configs with a valid warp count for the current device
    configs = []
    # Maximum threads per block is architecture-dependent in theory, but in reality all are 1024
    max_threads_per_block = 1024
    # Default to warp size 32 if not defined by device
    warp_size = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()), "warp_size", 32
    )
    # Autotune for warp counts which are powers of 2 and do not exceed thread per block limit
    return [
        triton.Config({}, num_warps=warp_count)
        for warp_count in [1, 2, 4, 8, 16, 32]
        if warp_count * warp_size <= max_threads_per_block
    ]
    # return [triton.Config({}, num_warps=8)]


# Copied from flash-attn
@triton.autotune(
    configs=triton_autotune_configs(),
    key=[
        "N",
        "HAS_RESIDUAL",
        "STORE_RESIDUAL_OUT",
        "IS_RMS_NORM",
        "HAS_BIAS",
        "HAS_WEIGHT",
        "HAS_X1",
        "HAS_W1",
        "HAS_B1",
    ],
)
# torch compile doesn't like triton.heuristics, so we set these manually when calling the kernel
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
# @triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
# @triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
# @triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    X1,
    W1,
    B1,
    Y1,
    RESIDUAL_OUT,  # pointer to the residual
    ROWSCALE,
    SEEDS,  # Dropout seeds for each row
    DROPOUT_MASK,
    DROPOUT_MASK1,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    stride_x1_row,
    stride_y1_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    dropout_p,  # Dropout probability
    zero_centered_weight,  # If true, add 1.0 to the weight
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr,
    HAS_B1: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        # Compute dropout mask
        # 7 rounds is good enough, and reduces register pressure
        keep_mask = (
            tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        )
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            # Compute dropout mask
            # 7 rounds is good enough, and reduces register pressure
            keep_mask = (
                tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7)
                > dropout_p
            )
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK1 + row * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        if zero_centered_weight:
            w += 1.0
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    if HAS_WEIGHT:
        y = x_hat * w + b if HAS_BIAS else x_hat * w
    else:
        y = x_hat + b if HAS_BIAS else x_hat
    # Write output
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if zero_centered_weight:
            w1 += 1.0
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    residual: Optional[Tensor] = None,
    x1: Optional[Tensor] = None,
    weight1: Optional[Tensor] = None,
    bias1: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
    out: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
    # Need to wrap to handle the case where residual_out is a alias of x, which makes torch.library
    # and torch.compile unhappy. Also allocate memory for out and residual_out if they are None
    # so that _layer_norm_fwd_impl doesn't have to return them.
    if out is None:
        out = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None:
        residual_dtype = residual.dtype
    if residual_out is None and (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
        or dropout_p > 0.0
        or rowscale is not None
        or x1 is not None
    ):
        residual_out = torch.empty_like(
            x, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
    else:
        residual_out = None
    y1, mean, rstd, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd_impl(
        x,
        weight,
        bias,
        eps,
        out,
        residual=residual,
        x1=x1,
        weight1=weight1,
        bias1=bias1,
        dropout_p=dropout_p,
        rowscale=rowscale,
        zero_centered_weight=zero_centered_weight,
        is_rms_norm=is_rms_norm,
        return_dropout_mask=return_dropout_mask,
        residual_out=residual_out,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if residual_out is None:
        residual_out = x
    return out, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1


# [2025-04-28] torch.library.triton_op ignores the schema argument, but here we need the schema
# since we're returning a tuple of tensors
def _layer_norm_fwd_impl(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Tensor,
    eps: float,
    out: Tensor,
    residual: Optional[Tensor] = None,
    x1: Optional[Tensor] = None,
    weight1: Optional[Tensor] = None,
    bias1: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[Tensor] = None,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
    residual_out: Optional[Tensor] = None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    assert out.shape == x.shape
    assert out.stride(-1) == 1
    if residual_out is not None:
        assert residual_out.shape == x.shape
        assert residual_out.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(out)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    mean = (
        torch.empty((M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(
            2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64
        )
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M, N, device=x.device, dtype=torch.bool)
        if x1 is not None:
            dropout_mask1 = torch.empty(M, N, device=x.device, dtype=torch.bool)
        else:
            dropout_mask1 = None
    else:
        dropout_mask, dropout_mask1 = None, None
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_layer_norm_fwd_1pass_kernel)[(M,)](
            x,
            out,
            weight if weight is not None else x,  # unused when HAS_WEIGHT == False
            bias,
            residual,
            x1,
            weight1,
            bias1,
            y1,
            residual_out,
            rowscale,
            seeds,
            dropout_mask,
            dropout_mask1,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            # Passing bool make torch inductor very unhappy since it then tries to compare to int_max
            int(zero_centered_weight),
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            weight is not None,
            bias is not None,
            dropout_p > 0.0,
            dropout_mask is not None,
            rowscale is not None,
            HAS_X1=x1 is not None,
            HAS_W1=weight1 is not None,
            HAS_B1=bias1 is not None,
        )
    return y1, mean, rstd, seeds, dropout_mask, dropout_mask1


class LayerNormFn:

    @staticmethod
    def forward(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        zero_centered_weight=False,
        is_rms_norm=False,
        return_dropout_mask=False,
        out_dtype=None,
        out=None,
        residual_out=None,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = maybe_contiguous_lastdim(x.reshape(-1, x.shape[-1]))
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = maybe_contiguous_lastdim(
                residual.reshape(-1, residual.shape[-1])
            )
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
            x1 = maybe_contiguous_lastdim(x1.reshape(-1, x1.shape[-1]))
        # weight can be None when elementwise_affine=False for LayerNorm
        if weight is not None:
            weight = weight.contiguous()
        bias = maybe_contiguous(bias)
        weight1 = maybe_contiguous(weight1)
        bias1 = maybe_contiguous(bias1)
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        if out is not None:
            out = out.reshape(-1, out.shape[-1])
        if residual_out is not None:
            residual_out = residual_out.reshape(-1, residual_out.shape[-1])
        y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = (
            _layer_norm_fwd(
                x,
                weight,
                bias,
                eps,
                residual,
                x1,
                weight1,
                bias1,
                dropout_p=dropout_p,
                rowscale=rowscale,
                out_dtype=out_dtype,
                residual_dtype=residual_dtype,
                zero_centered_weight=zero_centered_weight,
                is_rms_norm=is_rms_norm,
                return_dropout_mask=return_dropout_mask,
                out=out,
                residual_out=residual_out,
            )
        )
        y = y.reshape(x_shape_og)
        return y


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    return LayerNormFn.forward(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        zero_centered_weight,
        is_rms_norm,
        return_dropout_mask,
        out_dtype,
        out,
        residual_out,
    )


@triton.jit
def _norm_infer_kernel(
    X,
    Y,
    W,
    B,
    stride_x_row,
    stride_y_row,
    M,
    N,
    eps,
    IS_RMS_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_WEIGHT:
        W += 0
    if HAS_BIAS:
        B += 0
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
        y = x_hat * w
    else:
        y = x_hat
    if HAS_BIAS:
        b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
        y += b
    tl.store(Y + cols, y, mask=cols < N)


def norm_infer(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    is_rms_norm: bool = False,
    out: Optional[Tensor] = None,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (N,)
        assert bias.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    _norm_infer_kernel[(M,)](
        x,
        out,
        weight if weight is not None else x,  # dummy when HAS_WEIGHT=False
        bias if bias is not None else x,  # dummy when HAS_BIAS=False
        x.stride(0),
        out.stride(0),
        M,
        N,
        eps,
        IS_RMS_NORM=is_rms_norm,
        HAS_WEIGHT=weight is not None,
        HAS_BIAS=bias is not None,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return out


def rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    return LayerNormFn.forward(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        zero_centered_weight,
        True,
        return_dropout_mask,
        out_dtype,
        out,
        residual_out,
    )
