import itertools
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import sgl_per_token_group_quant_fp8

from sglang.srt.utils import get_device_core_count, get_device_name, is_hip

is_hip_ = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    # Stride of input
    y_stride,
    # Collums of input
    N,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group quantization on a
    tensor.

    This function converts the tensor values into float8 values.
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
    y_s = _absmax / fp8_max
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def triton_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_type_,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.

    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tenosr with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn` is supported for now.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the scaling factor for quantization.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_max = finfo.max

    fp8_min = -fp8_max

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
    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        N,
        eps,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return x_q, x_s


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = fp8_type_,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_max = finfo.max

    fp8_min = -fp8_max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)

    return x_q, x_s


@pytest.mark.parametrize(
    "batch_size, seq_len, group_size",
    list(
        itertools.product(
            [1, 2, 4, 8, 16],  # batch_size
            [64, 128, 256, 512, 1024, 2048],  # seq_len
            [64, 128, 256],  # group_size
        )
    ),
)
def test_per_token_group_quant_compare_implementations(batch_size, seq_len, group_size):
    x = torch.randn(
        (batch_size, seq_len, group_size * 2), device="cuda", dtype=torch.float16
    )

    x_q_triton, x_s_triton = triton_per_token_group_quant_fp8(x, group_size)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_fp8(x, group_size)

    assert torch.allclose(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    )
    assert torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
