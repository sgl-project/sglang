import sys

import pytest
import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the many-block chain sampler serves the ROCm DSpark accept
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not is_hip(), reason="the ROCm chain sampler")

VOCAB = 129280


def _outputs(bs, slots):
    return (
        torch.full((bs * slots,), -1, dtype=torch.int32, device="cuda"),
        torch.full((bs, slots), -1, dtype=torch.int32, device="cuda"),
        torch.full((bs,), -1, dtype=torch.int32, device="cuda"),
    )


@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("seed", range(8))
def test_matches_the_classic_kernel(bs, seed):
    """The same accepts, accept indices and sampled tokens as the single-CTA kernel,
    from the same coins: drafts near and far from the target, rejected and bonus rows."""
    from sglang.kernels.ops.speculative.chain_rejection import (
        chain_rejection_sampling,
        chain_rejection_supported,
    )

    torch.manual_seed(seed)
    slots = 6
    logits = torch.randn(bs, slots, VOCAB, device="cuda") * 4
    target = torch.softmax(logits, -1)
    noise = torch.randn(bs, slots - 1, VOCAB, device="cuda") * (0.05 + 0.4 * seed)
    draft = torch.softmax(logits[:, : slots - 1] + noise, -1)
    picks = torch.multinomial(draft.view(-1, VOCAB), 1).view(bs, slots - 1)
    candidates = torch.cat(
        [torch.zeros(bs, 1, dtype=torch.int64, device="cuda"), picks], 1
    )
    # a global index a slot, reversed: the outputs must follow it, not the row
    indices = torch.arange(bs * slots, device="cuda").flip(0).view(bs, slots)
    coins = torch.rand(bs, slots - 1, device="cuda")
    final = torch.rand(bs, device="cuda")
    want, got = _outputs(bs, slots), _outputs(bs, slots)
    chain_speculative_sampling_triton(
        *want, candidates, indices, None, None, coins, final, target, draft,
        1.0, 1.0, True,
    )  # fmt: skip
    assert chain_rejection_supported(target, draft)
    chain_rejection_sampling(
        *got, candidates, indices, coins, final, target, draft
    )  # fmt: skip
    assert torch.equal(got[2], want[2])
    for r in range(bs):
        n = int(want[2][r])
        assert torch.equal(got[1][r, : n + 1], want[1][r, : n + 1])
        written = want[1][r, : n + 1].long()
        assert torch.equal(got[0][written], want[0][written])


def test_zero_residual_takes_the_last_token_twice():
    """p == q leaves no residual mass: the classic kernel's last token, and the
    barrier words are clean for the next launch."""
    from sglang.kernels.ops.speculative.chain_rejection import (
        chain_rejection_sampling,
    )

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(1, 2, VOCAB, device="cuda"), -1)
    # a draft token the target gives no mass: rejected at the first step
    probs[0, 0, 7] = 0.0
    candidates = torch.tensor([[0, 7]], device="cuda")
    indices = torch.arange(2, device="cuda").view(1, 2)
    for _ in range(2):
        got = _outputs(1, 2)
        chain_rejection_sampling(
            *got, candidates, indices, torch.rand(1, 1, device="cuda"),
            torch.rand(1, device="cuda"), probs, probs[:, :1].contiguous(),
        )  # fmt: skip
        assert int(got[2][0]) == 0
        assert int(got[0][0]) == VOCAB - 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
