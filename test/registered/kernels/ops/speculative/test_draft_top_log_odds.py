import sys

import pytest
import torch

from sglang.kernels.ops.speculative.acceptance_estimator import (
    draft_top_log_odds,
    draft_top_log_odds_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [248320, 1000])
def test_matches_torch_reference(dtype, vocab):
    torch.manual_seed(0)
    bs, gamma = 5, 7
    logits = (torch.randn(bs, gamma, vocab, device="cuda") * 4).to(dtype)
    # A row whose leading vocab blocks are fully masked, and a near point mass.
    logits[0, 0, : vocab // 2] = float("-inf")
    logits[1, 2, 7] = 80.0
    temperatures = torch.tensor([1.0, 0.6, 1.3, 1.0, 2.0], device="cuda")
    # The folded sampler hands a [:bs] slice of a larger buffer.
    buffer = torch.zeros(bs + 3, gamma, vocab, device="cuda", dtype=dtype)
    buffer[:bs] = logits
    flat = buffer[:bs].reshape(bs * gamma, vocab)
    out = draft_top_log_odds(
        logits=flat, temperatures=temperatures, rows_per_request=gamma
    )
    ref = draft_top_log_odds_torch(
        logits=flat, temperatures=temperatures, rows_per_request=gamma
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
