import itertools

import pytest
import torch
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn



@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, dst_dtype, column_major_scales, scale_tma_aligned",
    list(
        itertools.product(
            [127, 128, 512, 1024, 4096, 8192],  # num_tokens
            [256, 512, 1024, 2048, 4096],  # hidden_dim
            [8, 16, 32, 64, 128],  # group_size
            [torch.int8, fp8_type_],  # dtype
            [False, True],  # column_major_scales
            [False, True],  # scale_tma_aligned
        )
    ),
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    column_major_scales,
    scale_tma_aligned,
):
    if not column_major_scales and scale_tma_aligned:
        return

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(
        x,
        group_size,
        eps=1e-10,
        dtype=dst_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
    )

    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(
        x,
        group_size,
        eps=1e-10,
        dtype=dst_dtype,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
    )

    torch.testing.assert_close(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    )
    torch.testing.assert_close(
        x_s_triton.contiguous(), x_s_sglang.contiguous(), rtol=1e-3, atol=1e-5
    )


if __name__ == "__main__":
    pytest.main([__file__])
