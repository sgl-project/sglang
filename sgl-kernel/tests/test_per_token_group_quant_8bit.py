import itertools

import pytest
import torch
from sgl_kernel import sgl_per_token_group_quant_8bit

from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.layers.quantization.int8_kernel import per_token_group_quant_int8


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.int8,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    if dtype == torch.int8:
        iinfo = torch.iinfo(dtype)
        max_8bit = iinfo.max
        min_8bit = iinfo.min
    else:
        f8_info = torch.finfo(dtype)
        max_8bit = f8_info.max
        min_8bit = f8_info.min

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )

    sgl_per_token_group_quant_8bit(x, x_q, x_s, group_size, eps, min_8bit, max_8bit)

    return x_q, x_s


@pytest.mark.parametrize(
    "batch_size, seq_len, group_size, dtype",
    list(
        itertools.product(
            [1, 2, 4, 8, 16, 32, 64, 128],  # batch_size
            [64, 128, 256, 512, 1024, 2048],  # seq_len
            [16, 32, 64, 128, 256],  # group_size
            [torch.int8, torch.float8_e4m3fn],  # dtype
        )
    ),
)
def test_per_token_group_quant_compare_implementations(
    batch_size, seq_len, group_size, dtype
):
    x = torch.randn(
        (batch_size, seq_len, group_size * 2), device="cuda", dtype=torch.float16
    )

    if dtype == torch.int8:
        x_q_triton, x_s_triton = per_token_group_quant_int8(x, group_size)
    else:
        x_q_triton, x_s_triton = per_token_group_quant_fp8(x, group_size)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(
        x, group_size, dtype=dtype
    )

    assert torch.allclose(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    )
    assert torch.allclose(x_s_triton, x_s_sglang, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
