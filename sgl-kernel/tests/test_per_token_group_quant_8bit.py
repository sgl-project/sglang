import itertools

import pytest
import torch

from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.utils import is_hip
from sglang.srt.layers.quantization.fp8_kernel import PER_TOKEN_GROUP_QUANT_8BIT_VALID_FLAGS

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, dst_dtype, flags",
    list(
        itertools.product(
            [127, 128, 512, 1024, 4096, 8192],  # num_tokens
            [256, 512, 1024, 2048, 4096],  # hidden_dim
            [8, 16, 32, 64, 128],  # group_size
            [torch.int8, fp8_type_],  # dtype
            PER_TOKEN_GROUP_QUANT_8BIT_VALID_FLAGS,
        )
    ),
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    dst_dtype,
    flags,
):
    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)

    execute_kwargs = dict(
        x=x,
        group_size=group_size,
        eps=1e-10,
        dtype=dst_dtype,
        **flags,
    )

    x_q_triton, x_s_triton = triton_per_token_group_quant_8bit(**execute_kwargs)
    x_q_sglang, x_s_sglang = sglang_per_token_group_quant_8bit(**execute_kwargs)

    torch.testing.assert_close(
        x_q_triton.to(torch.float32), x_q_sglang.to(torch.float32), rtol=1e-3, atol=1e-5
    )
    torch.testing.assert_close(
        x_s_triton.contiguous(), x_s_sglang.contiguous(), rtol=1e-3, atol=1e-5
    )


if __name__ == "__main__":
    pytest.main([__file__])
