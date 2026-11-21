"""Public-API UE8M0 group quantization (FP32 power-of-two scales) against a torch reference.

#48165 sent the bf16 group-32 case to the trait kernel, which is CUDA-only JIT; on ROCm that
case must stay on the v2 kernel. The multiplier is an exact power of two, so the scales and the
FP8 codes are compared bit-exact.
"""

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

EPS = 1e-10


def ref_fp32_pow2(x: torch.Tensor, group_size: int):
    xf = x.float().unflatten(-1, (-1, group_size))
    raw = (xf.abs().amax(-1).clamp_min(EPS) * (1.0 / fp8_max)).contiguous()
    bits = raw.view(torch.int32)
    # Round the exponent up, so the scale is the next power of two.
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    scale = (exp << 23).view(torch.float32)
    multiplier = ((254 - exp) << 23).view(torch.float32)  # exactly 1 / scale
    q = (xf * multiplier.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q.flatten(-2), scale


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize(
    "num_tokens,hidden", [(1, 7168), (7, 4096), (333, 7168), (4096, 1536)]
)
def test_ue8m0_fp32_scales_bitexact(num_tokens, hidden, group_size):
    torch.manual_seed(num_tokens * 10 + hidden)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)

    q, s = sglang_per_token_group_quant_fp8(x, group_size, scale_ue8m0=True)
    q_ref, s_ref = ref_fp32_pow2(x, group_size)

    assert s.dtype == torch.float32 and s.shape == s_ref.shape
    assert torch.equal(s, s_ref), "scales differ"
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), "codes differ"
