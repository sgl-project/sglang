import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import sgl_per_token_quant_fp8
from vllm import _custom_ops as ops

from sglang.srt.utils import is_hip

is_hip_ = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn


def vllm_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input, use_per_token_if_dynamic=True)


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)

    sgl_per_token_quant_fp8(input, output, scale)
    scale = scale.reshape(-1, 1)

    return output, scale


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([128, 256, 512], [512, 2048, 4096])),
)
def test_per_token_quant_compare_implementations(
    num_tokens: int,
    hidden_dim: int,
):
    device = torch.device("cuda")
    x = torch.rand((num_tokens, hidden_dim), dtype=torch.float16, device=device)

    vllm_out, vllm_scale = vllm_per_token_quant_fp8(x)
    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)

    torch.testing.assert_close(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        vllm_out.float(), sglang_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
