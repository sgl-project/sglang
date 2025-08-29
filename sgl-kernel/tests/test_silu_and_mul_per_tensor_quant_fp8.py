import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import (
    sgl_per_tensor_quant_fp8,
    sgl_silu_and_mul_per_tensor_quant_fp8,
    silu_and_mul,
)

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


def sglang_silu_and_mul_scaled_fp8_quant(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)

    intermediate_cache = torch.empty(
        input.shape[0], input.shape[1] // 2, device=input.device, dtype=input.dtype
    )
    silu_and_mul(input, intermediate_cache)
    scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    sgl_per_tensor_quant_fp8(intermediate_cache, output, scale, is_static=False)

    return output, scale


def sglang_fused_silu_and_mul_scaled_fp8_quant(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    scale = torch.zeros(1, device=input.device, dtype=torch.float32)
    input_gate, input_up = input.chunk(2, dim=-1)
    input_gate = input_gate.contiguous()
    input_up = input_up.contiguous()
    sgl_silu_and_mul_per_tensor_quant_fp8(
        input_gate, input_up, output, scale, is_static=False
    )

    return output, scale


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([128, 256, 512], [512, 2048, 4096])),
)
def test_silu_and_mul_per_tensor_quant_compare_implementations(
    num_tokens: int,
    hidden_dim: int,
):
    device = torch.device("cuda")
    x = torch.rand((num_tokens, hidden_dim), dtype=torch.float16, device=device)

    sglang_out, sglang_scale = sglang_silu_and_mul_scaled_fp8_quant(x)
    fused_out, fused_scale = sglang_fused_silu_and_mul_scaled_fp8_quant(x)

    torch.testing.assert_close(
        sglang_out.float(), fused_out.float(), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        sglang_scale.float(), fused_scale.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
