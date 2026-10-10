import logging
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda:
    from sglang.kernels.ops.quantization.per_token_group_quant import (
        per_token_group_quant,
    )

logger = logging.getLogger(__name__)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    N,
    CAL_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    if IS_HIP:
        # ROCm Triton dropped the CUDA `tl.extra.cuda.libdevice.*` shim
        # (`__nv_roundf`); use the backend-agnostic libdevice instead.
        x_q = libdevice.round(x_q).to(tl.int8)
    else:
        x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=0)
        tl.store(x_sum_ptr + row_id, x_sum.to(x_sum_ptr.dtype.element_ty))

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def per_token_quant_int8(x, scale_dtype=torch.float32, cal_sum=False):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)
    if cal_sum:
        x_sum = torch.empty(x.shape[:-1], device=x.device, dtype=x.dtype)
    else:
        x_sum = None
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        x_sum,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        CAL_SUM=cal_sum,
        BLOCK=BLOCK,
        IS_HIP=_is_hip,
        num_warps=num_warps,
        num_stages=1,
    )
    if cal_sum:
        return x_q, scales, x_sum
    else:
        return x_q, scales


@triton.jit
def _per_token_group_quant_int8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Columns of input
    N,
    # Avoid division by zero
    eps,
    # Information for int8
    int8_min,
    int8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into int8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / int8_max
    y_q = tl.clamp(y / y_s, int8_min, int8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.int8` is supported for now.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max
    int8_min = iinfo.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1
    _per_token_group_quant_int8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        int8_min=int8_min,
        int8_max=int8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def sglang_per_token_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
):
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"
    assert dtype == torch.int8
    # per_token_group_quant bakes the int8 constants in ([-128, 127], eps 1e-10).
    assert eps == 1e-10, (
        f"per_token_group_quant bakes the absmax floor in at 1e-10, got {eps}"
    )

    return per_token_group_quant(x, group_size=group_size, out_dtype=dtype)
