"""``diffusion.head_gate_mul``: LTX-2.3's per-head attention gate, bit-exact with eager."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import can_use_head_gate_mul, head_gate_mul
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "batch,tokens,heads,head_dim",
    [(2, 1536, 32, 128), (2, 126, 32, 64), (1, 7, 4, 16), (3, 5, 8, 24)],
)
@pytest.mark.parametrize("flat", [False, True])
def test_head_gate_mul_matches_eager_bitwise(
    batch: int, tokens: int, heads: int, head_dim: int, flat: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    out = torch.randn(batch, tokens, heads, head_dim, device="cuda") * 3
    out = out.to(torch.bfloat16)
    logits = (torch.randn(batch, tokens, heads, device="cuda") * 4).to(torch.bfloat16)
    gate = torch.sigmoid(logits)
    expected = (out * (2.0 * gate.unsqueeze(-1))).view(batch, tokens, -1)
    x = out.view(batch, tokens, -1) if flat else out
    assert can_use_head_gate_mul(x, gate)
    actual = head_gate_mul(x, gate)
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_head_gate_mul_rejects_strided_or_mismatched_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    out = torch.randn(2, 6, 4, 8, device="cuda", dtype=torch.bfloat16)
    gate = torch.rand(2, 6, 4, device="cuda", dtype=torch.bfloat16)
    assert not can_use_head_gate_mul(out.transpose(1, 2), gate)
    assert not can_use_head_gate_mul(out, gate[:, :5])
    assert not can_use_head_gate_mul(out.float(), gate)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
