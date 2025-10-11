# coding: utf-8

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N"],
)
@triton.jit
def layer_norm_fwd_kernel(
    X,  # pointer to the input  # (M, N)
    Y,  # pointer to the output  # (M, N)
    W,  # pointer to the weights  # (N)
    Rstd,  # pointer to the 1/std  # (M)
    N,  # number of columns in X
    stride_x0,
    stride_x1,
    stride_y0,
    stride_y1,
    stride_w0,
    stride_rstd0,
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x0
    Y += row * stride_y0
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols * stride_x1, mask=cols < N, other=0.0).to(
        tl.float32
    )  # (BLOCK_N, )
    xbar = tl.where(cols < N, x, 0.0)  # (BLOCK_N, )
    var = tl.sum(xbar * xbar, axis=0) / N  # (1, )
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row * stride_rstd0, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols * stride_w0, mask=mask).to(tl.float32)  # (BLOCK_N, )
    x_hat = x * rstd  # (BLOCK_N, )
    y = x_hat * w  # (BLOCK_N, )
    # Write output
    tl.store(Y + cols * stride_y1, y, mask=mask)


def rms_norm_fwd(
    x: torch.Tensor,  # (M, hidden_size)  attn_output, rmsnorm
    weight: torch.Tensor,  # (hidden_size)
    eps: float,
):
    M, N = x.shape
    # allocate output
    y = torch.empty_like(x)
    rstd = torch.empty((M,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    layer_norm_fwd_kernel[(M,)](
        x,
        y,
        weight,
        rstd,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        rstd.stride(0),
        eps,
        BLOCK_N,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, rstd


class RMSNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,  # (*, hidden_size)  attn_output, rmsnorm
        weight,  # (hidden_size)
        eps=1e-6,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])  # (M, hidden_size)
        y, rstd = rms_norm_fwd(x, weight, eps)
        y = y.reshape(x_shape_og)
        return y


def rms_norm_triton_fn(x, weight, eps=1e-6):  # (*, hidden_size)  # (hidden_size)
    return RMSNormTriton.apply(x, weight, eps)
