import math

import torch
import triton
import triton.language as tl


# From FlagGems
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


# Integrated from FlagGems
class _RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        rms_norm_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)
        return y


def rms_norm_triton(x, normalized_shape, weight, eps=1e-5):
    return _RmsNorm.apply(x, normalized_shape, weight, eps)


@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    input_ptr,  # [..., hidden_size]
    residual_ptr,  # [..., hidden_size]
    weight_ptr,  # [hidden_size]
    y_stride_r,
    y_stride_c,
    x_stride_r,  # stride for input rows
    x_stride_c,  # stride for input columns
    N,  # hidden_size
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    input_ptr += pid * y_stride_r
    residual_ptr += pid * y_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)

    # Load data from input and residual, then add them together
    x = tl.load(input_ptr + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + cols * x_stride_c, mask, other=0.0).to(tl.float32)
    z = x + r
    tl.store(residual_ptr + cols * y_stride_c, z, mask=mask)

    # Compute variance
    var = tl.sum(z * z, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    # Load weight and apply RMS normalization
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    normed_z = (z * rrms).to(input_ptr.dtype.element_ty) * weight

    # Store result back to input
    tl.store(input_ptr + cols * y_stride_c, normed_z, mask=mask)


class _FusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, normalized_shape, weight, eps=1e-5):
        dim = input.ndim - len(normalized_shape)
        M = math.prod(input.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = triton.next_power_of_2(N)
        input = input.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

        # Launch the Triton kernel
        fused_add_rms_norm_kernel[(M,)](
            input, residual, weight, N, 1, N, 1, N, eps, BLOCK_SIZE
        )
        return input


def fused_add_rms_norm_triton(input, residual, normalized_shape, weight, eps=1e-5):
    return _FusedAddRMSNorm.apply(input, residual, normalized_shape, weight, eps)
