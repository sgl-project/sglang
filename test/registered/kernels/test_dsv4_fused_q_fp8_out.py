"""fused_q_norm_rope with an e4m3 output must be bit-identical to the bf16
output followed by .to(float8_e4m3fn) -- same kernel math, cast at store."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64


def _make_freqs(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize("num_tokens", [1, 3, 64, 511])
@pytest.mark.parametrize("num_heads", [16, 64])
def test_fused_q_norm_rope_fp8_out(num_tokens, num_heads):
    torch.manual_seed(num_tokens * 131 + num_heads)
    dev = torch.device("cuda")
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    freqs = _make_freqs(65536)
    positions = torch.randint(0, 65536, (num_tokens,), device=dev, dtype=torch.int64)
    eps = 1e-6

    ref_bf16 = torch.empty_like(q)
    fused_q_norm_rope(q, ref_bf16, eps, freqs, positions)
    ref_fp8 = ref_bf16.to(torch.float8_e4m3fn)

    out_fp8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=dev)
    fused_q_norm_rope(q, out_fp8, eps, freqs, positions)

    assert torch.equal(
        out_fp8.view(torch.uint8), ref_fp8.view(torch.uint8)
    ), "fp8-out kernel differs from bf16-out + cast"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
