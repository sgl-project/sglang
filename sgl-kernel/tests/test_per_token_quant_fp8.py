import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import sgl_per_token_quant_fp8
from vllm import _custom_ops as ops

from sglang.srt.utils import is_hip

is_hip_ = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn


# vLLM doesn't support static per token quant
# https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp8/common.cu
def vllm_dynamic_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input, use_per_token_if_dynamic=True)


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    is_static = True

    if scale is None:
        scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
        is_static = False

    sgl_per_token_quant_fp8(input, output, scale, is_static)

    # Reshape scale to match VLLM's output shape
    if not is_static:
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

    # we only test dynamic per token quant since
    # static per token quant is not supported in vllm
    vllm_out, vllm_scale = vllm_dynamic_per_token_quant_fp8(x)
    sglang_out, sglang_scale = sglang_per_token_quant_fp8(x)
    print(sglang_out)

    torch.testing.assert_close(vllm_scale, sglang_scale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        vllm_out.float(), sglang_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    # Run the specific test function directly
    pytest.main([__file__])
