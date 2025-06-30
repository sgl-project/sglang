from typing import Optional

import pytest
import torch
from sgl_kernel import (
    sgl_per_token_group_quant_fp8,
    sgl_silu_and_mul_per_token_group_quant_fp8,
    silu_and_mul,
)

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max
fp8_min = -fp8_max


def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=fp8_dtype)
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (x.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                x.shape[:-2] + (x.shape[-1] // group_size, aligned_size),
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x.shape[-2], :]
        else:
            x_s = torch.empty(
                (x.shape[-1] // group_size,) + x.shape[:-1],
                device=x.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            x.shape[:-1] + (x.shape[-1] // group_size,),
            device=x.device,
            dtype=torch.float32,
        )
    if x.shape[0] > 0:
        sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)

    return x_q, x_s


def split_silu_and_mul_per_token_group_quant_fp8(gateup_output, down_input, group_size):
    down_input = silu_and_mul(gateup_output, down_input)
    return sglang_per_token_group_quant_fp8(down_input, group_size)


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def fuse_silu_and_mul_per_token_group_quant_fp8(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    group_size: int = 128,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
):
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    assert (
        out.shape[-1] % group_size == 0
    ), "the last dimension of `out` cannot be divisible by `group_size`"
    assert out.is_contiguous(), "`out` is not contiguous"

    x_q = torch.empty_like(out, device=out.device, dtype=torch.float8_e4m3fn)
    if column_major_scales:
        if scale_tma_aligned:
            # aligned to 4 * sizeof(float)
            aligned_size = (out.shape[-2] + 3) // 4 * 4
            x_s = torch.empty(
                out.shape[:-2] + (out.shape[-1] // group_size, aligned_size),
                device=out.device,
                dtype=torch.float32,
            ).permute(-1, -2)[: out.shape[-2], :]
        else:
            x_s = torch.empty(
                (out.shape[-1] // group_size,) + out.shape[:-1],
                device=out.device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        x_s = torch.empty(
            out.shape[:-1] + (out.shape[-1] // group_size,),
            device=out.device,
            dtype=torch.float32,
        )
    if out.shape[0] > 0:
        sgl_silu_and_mul_per_token_group_quant_fp8(
            input, x_q, x_s, group_size, eps, fp8_min, fp8_max
        )

    return x_q, x_s


def _test_accuracy_once(N, M, input_type, device):
    group_size = 128
    gateup_output = torch.randn(N, M * 2, device=device, dtype=input_type)
    down_input = torch.empty(N, M, device=device, dtype=input_type)
    o = fuse_silu_and_mul_per_token_group_quant_fp8(
        gateup_output.clone(), down_input.clone(), group_size
    )
    o1 = split_silu_and_mul_per_token_group_quant_fp8(
        gateup_output.clone(), down_input.clone(), group_size
    )
    torch.testing.assert_close(o[1], o1[1])
    torch.testing.assert_close(o[0], o1[0])


@pytest.mark.parametrize("N", [2**i for i in range(2, 15)])
@pytest.mark.parametrize("M", [2**i for i in range(7, 14)])
@pytest.mark.parametrize("input_type", [torch.float16, torch.bfloat16])
def test_accuracy(N, M, input_type):
    _test_accuracy_once(N, M, input_type, device="cuda")


if __name__ == "__main__":
    pytest.main([__file__])
