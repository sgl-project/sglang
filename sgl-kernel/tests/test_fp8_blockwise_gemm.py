import os
import random
from typing import Optional, Type

import pytest
import torch
from sgl_kernel import fp8_blockwise_scaled_mm


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)
    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def _test_accuracy_once(M, N, K, out_dtype, device):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()
    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)
    scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
    scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)
    scale_a = torch.randn(scale_a_shape, device=device, dtype=torch.float32) * 0.001
    scale_b = torch.randn(scale_b_shape, device=device, dtype=torch.float32) * 0.001
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    o = baseline_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    o1 = fp8_blockwise_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype)
    rtol = 0.02
    atol = 1
    torch.testing.assert_close(o, o1, rtol=rtol, atol=atol)
    print(f"M={M}, N={N}, K={K}, out_dtype={out_dtype}: OK")


@pytest.mark.parametrize("M", [1, 128, 512, 1024, 4096])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_accuracy(M, N, K, out_dtype):
    _test_accuracy_once(M, N, K, out_dtype, "cuda")


if __name__ == "__main__":
    pytest.main([__file__])
