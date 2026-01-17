import itertools
from typing import Optional, Tuple

import pytest
import torch

from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8

try:
    from sglang.srt.utils import is_hip

    _is_hip = is_hip()
except ImportError:
    _is_hip = False

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
    per_tensor_quant_fp8(input, output, scale, is_static)

    return output, scale


def torch_scaled_fp8_quant(tensor, inv_scale):
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = inv_scale.reciprocal()
    qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    list(itertools.product([128, 256, 512], [512, 2048, 4096])),
)
def test_jit_per_tensor_quant_compare_implementations(
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


@pytest.mark.parametrize("shape", [(4, 8, 64), (2, 16, 128), (19260817, 1, 1)])
def test_jit_per_tensor_quant_supports_3d(shape):
    device = torch.device("cuda")
    x = torch.rand(shape, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x, device=x.device, dtype=fp8_type_)
    scale = torch.zeros(1, device=x.device, dtype=torch.float32)

    per_tensor_quant_fp8(x, out, scale, is_static=False)

    x_2d = x.flatten(0, -2)
    out_ref_2d = torch_scaled_fp8_quant(x_2d, scale)
    out_ref = out_ref_2d.reshape(shape)

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-3, atol=1e-3)

    scale = torch.rand(1, dtype=torch.float32, device=device)
    sglang_out, _ = sglang_scaled_fp8_quant(x, scale)
    torch_out = torch_scaled_fp8_quant(x, scale)

    torch.testing.assert_close(
        sglang_out.float(), torch_out.float(), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
