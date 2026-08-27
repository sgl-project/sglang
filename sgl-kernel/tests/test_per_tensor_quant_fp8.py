import itertools
from typing import Optional, Tuple

import pytest
import torch
from sgl_kernel import sgl_per_tensor_quant_fp8

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


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


def torch_scaled_fp8_quant(tensor, inv_scale):
    # The reference implementation that fully aligns to
    # the kernel being tested.
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = inv_scale.reciprocal()
    qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight


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

    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x)
    torch_out = torch_scaled_fp8_quant(x, sglang_scale)

    torch.testing.assert_close(
        sglang_out.float(), torch_out.float(), rtol=1e-3, atol=1e-3
    )

    scale = torch.rand(1, dtype=torch.float32, device=device)
    sglang_out, sglang_scale = sglang_scaled_fp8_quant(x, scale)
    torch_out = torch_scaled_fp8_quant(x, scale)

    torch.testing.assert_close(
        sglang_out.float(), torch_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
