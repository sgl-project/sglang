import itertools

import pytest
import torch

from sglang.jit_kernel.per_token_quant_fp8 import per_token_quant_fp8
from sglang.jit_kernel.utils import get_ci_test_range


DEVICE = "cuda"
M_LIST = get_ci_test_range(
    [1, 2, 4, 8, 16, 32, 64, 128, 256],
    [1, 16, 256],
)
K_LIST = get_ci_test_range(
    [512, 1024, 2048, 4096, 7168],
    [512, 4096],
)
DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32]

configs = list(itertools.product(M_LIST, K_LIST, DTYPE_LIST))


def _reference_per_token_quant_fp8(input_tensor):
    FP8_E4M3_MAX = 448.0
    absmax = input_tensor.float().abs().max(dim=-1, keepdim=True).values
    scale = absmax / FP8_E4M3_MAX
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    quantized = (input_tensor.float() * scale_inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized, scale.squeeze(-1)


@pytest.mark.parametrize("m, k, dtype", configs)
def test_per_token_quant_fp8(m, k, dtype):
    input_tensor = torch.randn(m, k, dtype=dtype, device=DEVICE)
    output_q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEVICE)
    output_s = torch.empty(m, dtype=torch.float32, device=DEVICE)

    per_token_quant_fp8(input_tensor, output_q, output_s)

    ref_q, ref_s = _reference_per_token_quant_fp8(input_tensor)

    torch.testing.assert_close(output_s, ref_s, rtol=1e-5, atol=1e-6)

    output_q_float = output_q.float()
    ref_q_fp8 = ref_q.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(output_q_float, ref_q_fp8, rtol=0.15, atol=64.0)


@pytest.mark.parametrize("m", [1, 16, 64])
def test_per_token_quant_fp8_scale_shape(m):
    k = 1024
    dtype = torch.float16
    input_tensor = torch.randn(m, k, dtype=dtype, device=DEVICE)
    output_q = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=DEVICE)
    output_s = torch.empty(m, 1, dtype=torch.float32, device=DEVICE)

    per_token_quant_fp8(input_tensor, output_q, output_s)

    ref_q, ref_s = _reference_per_token_quant_fp8(input_tensor)
    torch.testing.assert_close(output_s.view(-1), ref_s, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
