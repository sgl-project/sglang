import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import sgl_per_tensor_quant_fp8
from vllm import _custom_ops as ops

from sglang.srt.utils import is_hip

is_hip_ = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn


def vllm_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input, scale)


def sglang_scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_type_: torch.dtype = torch.float8_e4m3fn
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        is_static = False
    sgl_per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([128, 256, 512], [512, 2048, 4096])),
)
def test_per_tensor_quant_compare_implementations(
    num_tokens: int,
    hidden_dim: int,
):
    device = torch.device("cuda")
    x = torch.rand((num_tokens, hidden_dim), dtype=torch.float16, device=device)

    vllm_out, vllm_scale = vllm_scaled_fp8_quant(x)
    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x)

    torch.testing.assert_close(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        vllm_out.float(), sglang_out.float(), rtol=1e-3, atol=1e-3
    )

    scale = torch.rand(1, dtype=torch.float32, device=device)
    vllm_out, vllm_scale = vllm_scaled_fp8_quant(x, scale)
    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x, scale)

    torch.testing.assert_close(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        vllm_out.float(), sglang_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
