from typing import Optional

import torch
import torch._C._distributed_c10d as c10d
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch import Tensor

from sglang.multimodal_gen.runtime.platforms import current_platform


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
        torch.get_device_module().get_device_properties(
            torch.get_device_module().current_device()
        ),
        "warp_size",
        32,
    )
    if warp_size is None:
        warp_size = 32
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
    with torch.get_device_module().device(x.device.index):
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
        if residual is not None:
            residual_out = residual_out.reshape(x_shape_og)
            return y, residual_out
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
    x = x.contiguous()
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


# Adapted from https://github.com/ModelTC/LightX2V/blob/main/lightx2v/common/ops/norm/triton_ops.py#L905-L956
@triton.jit
def _rms_norm_tiled_onepass(
    y_ptr,
    x_ptr,
    w_ptr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    seq_blk_id = tl.program_id(0)
    seq_id = seq_blk_id * BLOCK_SIZE_SEQ

    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    s_mask = seq_offset < SEQ
    d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    d_mask = d_offset < DIM
    y_blk = y_ptr + seq_offset * DIM + d_offset
    x_blk = x_ptr + seq_offset * DIM + d_offset
    mask = s_mask & d_mask

    x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + d_offset, mask=d_mask)
    tl.store(y_blk, x * rstd * w, mask=mask)


def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    shape = x.shape
    x = x.contiguous()
    y = torch.empty_like(x)
    x_view = x.reshape(-1, shape[-1])
    y_view = y.reshape(-1, shape[-1])
    S, D = x_view.shape

    BLOCK_SIZE_SEQ = min(16, triton.next_power_of_2(max(1, S // 512)))
    grid = (triton.cdiv(S, BLOCK_SIZE_SEQ),)

    with torch.cuda.device(x.device):
        torch.library.wrap_triton(_rms_norm_tiled_onepass)[grid](
            y_view,
            x_view,
            w,
            S,
            D,
            eps,
            BLOCK_SIZE_DIM=triton.next_power_of_2(D),
            BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        )
    return y


################################################################################
# Fused RMSNorm + Interleaved RoPE kernels for MOVA DiT
################################################################################


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _fused_rmsnorm_rope_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    M,  # total rows (B * S)
    D: tl.constexpr,  # hidden dimension
    seq_len,  # sequence length (for cos/sin indexing)
    head_dim_half: tl.constexpr,  # head_dim // 2
    eps,
    stride_x_row,
    stride_out_row,
    stride_cos_row,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm + interleaved RoPE kernel (TP=1).

    Two-pass algorithm:
      Pass 1: accumulate sum-of-squares over the row to compute rstd.
      Pass 2: normalize with weight, then apply interleaved RoPE in pairs.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * stride_x_row
    out_row_ptr = out_ptr + row * stride_out_row

    # Token index within the sequence (for cos/sin lookup)
    token_idx = row % seq_len

    # --- Pass 1: compute rstd = rsqrt(mean(x^2) + eps) ---
    sum_sq = tl.zeros([], dtype=tl.float32)
    for block_start in range(0, D, BLOCK_D):
        offsets = block_start + tl.arange(0, BLOCK_D)
        mask = offsets < D
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    rstd = tl.math.rsqrt(sum_sq / D + eps)

    # --- Pass 2: fused normalize + RoPE (process in pairs) ---
    # Total number of pairs = D // 2
    num_pairs: tl.constexpr = D // 2
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2

    for block_start in range(0, num_pairs, BLOCK_PAIRS):
        pair_offsets = block_start + tl.arange(0, BLOCK_PAIRS)
        pair_mask = pair_offsets < num_pairs

        # Element indices for even/odd of each pair
        even_offsets = 2 * pair_offsets
        odd_offsets = 2 * pair_offsets + 1

        # Load input pairs
        x_even = tl.load(x_row_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        x_odd = tl.load(x_row_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        # Load weights
        w_even = tl.load(weight_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        w_odd = tl.load(weight_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        # Normalize
        xn_even = x_even * rstd * w_even
        xn_odd = x_odd * rstd * w_odd

        # RoPE index: wraps every head_dim_half pairs
        rope_idx = pair_offsets % head_dim_half

        # Load cos/sin
        cos_row_ptr = cos_ptr + token_idx * stride_cos_row
        sin_row_ptr = sin_ptr + token_idx * stride_cos_row
        cos_vals = tl.load(cos_row_ptr + rope_idx, mask=pair_mask, other=1.0).to(
            tl.float32
        )
        sin_vals = tl.load(sin_row_ptr + rope_idx, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        # Interleaved RoPE: (even', odd') = R(theta) * (even, odd)
        o_even = xn_even * cos_vals - xn_odd * sin_vals
        o_odd = xn_even * sin_vals + xn_odd * cos_vals

        # Store
        tl.store(out_row_ptr + even_offsets, o_even, mask=pair_mask)
        tl.store(out_row_ptr + odd_offsets, o_odd, mask=pair_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _compute_local_mean_sq_kernel(
    mean_sq_ptr,
    x_ptr,
    M,
    D: tl.constexpr,
    stride_x_row,
    BLOCK_D: tl.constexpr,
):
    """Compute per-row mean of squares for TP>1 RMSNorm (phase 1)."""
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * stride_x_row

    sum_sq = tl.zeros([], dtype=tl.float32)
    for block_start in range(0, D, BLOCK_D):
        offsets = block_start + tl.arange(0, BLOCK_D)
        mask = offsets < D
        x_vals = tl.load(x_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    tl.store(mean_sq_ptr + row, sum_sq / D)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def _fused_apply_norm_rope_kernel(
    out_ptr,
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    rstd_ptr,
    M,
    D: tl.constexpr,
    seq_len,
    head_dim_half: tl.constexpr,
    stride_x_row,
    stride_out_row,
    stride_cos_row,
    BLOCK_D: tl.constexpr,
):
    """Apply normalization (with precomputed rstd) + interleaved RoPE (TP>1 phase 3)."""
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * stride_x_row
    out_row_ptr = out_ptr + row * stride_out_row
    token_idx = row % seq_len

    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    num_pairs: tl.constexpr = D // 2
    BLOCK_PAIRS: tl.constexpr = BLOCK_D // 2

    for block_start in range(0, num_pairs, BLOCK_PAIRS):
        pair_offsets = block_start + tl.arange(0, BLOCK_PAIRS)
        pair_mask = pair_offsets < num_pairs

        even_offsets = 2 * pair_offsets
        odd_offsets = 2 * pair_offsets + 1

        x_even = tl.load(x_row_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        x_odd = tl.load(x_row_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        w_even = tl.load(weight_ptr + even_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        w_odd = tl.load(weight_ptr + odd_offsets, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        xn_even = x_even * rstd * w_even
        xn_odd = x_odd * rstd * w_odd

        rope_idx = pair_offsets % head_dim_half

        cos_row_ptr = cos_ptr + token_idx * stride_cos_row
        sin_row_ptr = sin_ptr + token_idx * stride_cos_row
        cos_vals = tl.load(cos_row_ptr + rope_idx, mask=pair_mask, other=1.0).to(
            tl.float32
        )
        sin_vals = tl.load(sin_row_ptr + rope_idx, mask=pair_mask, other=0.0).to(
            tl.float32
        )

        o_even = xn_even * cos_vals - xn_odd * sin_vals
        o_odd = xn_even * sin_vals + xn_odd * cos_vals

        tl.store(out_row_ptr + even_offsets, o_even, mask=pair_mask)
        tl.store(out_row_ptr + odd_offsets, o_odd, mask=pair_mask)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + interleaved RoPE for TP=1.

    Args:
        x: Input tensor [B, S, D] or [*, D] (bf16/fp16).
        weight: RMSNorm weight [D] (any dtype, cast to fp32 internally).
        cos: Precomputed cosine [S, head_dim//2] float32.
        sin: Precomputed sine [S, head_dim//2] float32.
        head_dim: Per-head dimension (e.g. 128).
        eps: RMSNorm epsilon.

    Returns:
        Output tensor, same shape and dtype as x.
    """
    orig_shape = x.shape
    seq_len = cos.shape[0]
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, D = x_2d.shape
    assert D % 2 == 0, f"Hidden dim must be even, got {D}"

    out = torch.empty_like(x_2d)
    head_dim_half = head_dim // 2

    grid = (M,)
    _fused_rmsnorm_rope_kernel[grid](
        out,
        x_2d,
        weight,
        cos,
        sin,
        M,
        D,
        seq_len,
        head_dim_half,
        eps,
        x_2d.stride(0),
        out.stride(0),
        cos.stride(0),
    )
    return out.view(orig_shape)


def fused_rmsnorm_rope_tp(
    x: torch.Tensor,
    norm_module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    """Fused RMSNorm + interleaved RoPE for TP>1.

    Three-phase:
      1. Compute local mean-of-squares via Triton kernel.
      2. All-reduce mean_sq across TP group.
      3. Apply normalization (with global rstd) + RoPE via Triton kernel.

    Args:
        x: Local input tensor [B, S, D_local] (bf16/fp16).
        norm_module: RMSNorm module (has .weight [D_full] and .variance_epsilon).
        cos: Precomputed cosine [S, head_dim//2] float32.
        sin: Precomputed sine [S, head_dim//2] float32.
        head_dim: Per-head dimension (e.g. 128).
        tp_rank: Current TP rank.
        tp_size: TP world size.
        tp_group: TP process group (has .all_reduce method).

    Returns:
        Output tensor, same shape and dtype as x.
    """
    orig_shape = x.shape
    seq_len = cos.shape[0]
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, D_local = x_2d.shape
    assert D_local % 2 == 0, f"Local hidden dim must be even, got {D_local}"

    head_dim_half = head_dim // 2
    weight_local = norm_module.weight.tensor_split(tp_size)[tp_rank].float()
    eps = norm_module.variance_epsilon

    # Phase 1: compute local mean-of-squares
    mean_sq = torch.empty(M, dtype=torch.float32, device=x.device)
    grid = (M,)
    _compute_local_mean_sq_kernel[grid](
        mean_sq,
        x_2d,
        M,
        D_local,
        x_2d.stride(0),
    )

    # Phase 2: all-reduce mean_sq across TP ranks
    mean_sq = tp_group.all_reduce(mean_sq, op=c10d.ReduceOp.AVG)

    # Phase 3: rstd from global mean_sq, then fused norm + RoPE
    rstd = torch.rsqrt(mean_sq + eps)

    out = torch.empty_like(x_2d)
    _fused_apply_norm_rope_kernel[grid](
        out,
        x_2d,
        weight_local,
        cos,
        sin,
        rstd,
        M,
        D_local,
        seq_len,
        head_dim_half,
        x_2d.stride(0),
        out.stride(0),
        cos.stride(0),
    )
    return out.view(orig_shape)


if current_platform.is_npu():
    from .npu_fallback import fused_rmsnorm_rope_native, fused_rmsnorm_rope_tp_native

    fused_rmsnorm_rope = fused_rmsnorm_rope_native
    fused_rmsnorm_rope_tp = fused_rmsnorm_rope_tp_native
