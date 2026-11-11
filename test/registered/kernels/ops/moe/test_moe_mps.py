"""MoE expert compute on Apple MPS must match a per-token PyTorch reference."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.moe_mps import fused_experts_mps
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=20, suite="stage-a-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)

E, H, I, TOP_K = 16, 256, 128, 4


def _reference(x, w1, w2, weights, ids):
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            gate, up = (w1[e].float() @ x[t].float()).chunk(2)
            out[t] += weights[t, j] * (w2[e].float() @ (F.silu(gate) * up))
    return out


# Up to 32 tokens (128 pairs) run the Metal GEMV kernels; 40 and 300 sort by expert.
@pytest.mark.parametrize("num_tokens", [1, 8, 17, 32, 40, 300])
def test_matches_reference(num_tokens):
    gen = torch.Generator().manual_seed(num_tokens)
    x = torch.randn(num_tokens, H, generator=gen).to(torch.bfloat16)
    w1 = (torch.randn(E, 2 * I, H, generator=gen) * H**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=gen) * I**-0.5).to(torch.bfloat16)
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOP_K] for _ in range(num_tokens)]
    )
    # Every token hits expert 3 so one expert serves many rows.
    ids[:, 0] = 3
    ids = ids.to(torch.int32)
    weights = torch.rand(num_tokens, TOP_K, generator=gen).softmax(-1)
    out = fused_experts_mps(
        x.to("mps"), w1.to("mps"), w2.to("mps"), weights.to("mps"), ids.to("mps")
    )
    ref = _reference(x, w1, w2, weights, ids)
    torch.testing.assert_close(out.float().cpu(), ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
