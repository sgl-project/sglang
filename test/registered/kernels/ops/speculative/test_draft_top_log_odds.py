import sys

import pytest
import torch

from sglang.kernels.ops.speculative.acceptance_estimator import (
    accumulate_bin_counts,
    accumulate_bin_counts_torch,
    draft_top_log_odds,
    draft_top_log_odds_torch,
    predict_acceptance,
    predict_acceptance_torch,
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


def test_predict_and_bin_counts_match_torch():
    torch.manual_seed(1)
    bs, gamma, vocab, num_bins = 37, 7, 151936, 128
    logits = (torch.randn(bs * gamma, vocab, device="cuda") * 3).to(torch.bfloat16)
    temperatures = torch.rand(bs, device="cuda") + 0.5
    slope = torch.tensor([0.55], device="cuda")
    intercepts = torch.linspace(0.5, -1.0, gamma, device="cuda")
    kwargs = dict(
        logits=logits,
        temperatures=temperatures,
        rows_per_request=gamma,
        slope=slope,
        intercepts=intercepts,
    )
    features, predictions = predict_acceptance(**kwargs)
    ref_features, ref_predictions = predict_acceptance_torch(**kwargs)
    torch.testing.assert_close(features, ref_features, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(predictions, ref_predictions, rtol=1e-4, atol=1e-4)

    # Spread features over every bin, including both clamped edges.
    features = torch.linspace(-12.0, 30.0, bs * gamma, device="cuda").view(bs, gamma)
    num_correct = torch.randint(0, gamma + 1, (bs,), device="cuda")
    num_graded = torch.randint(1, gamma + 1, (bs,), device="cuda")
    counts = torch.zeros(gamma, num_bins, 2, device="cuda")
    ref_counts = torch.zeros_like(counts)
    for out, fn in (
        (counts, accumulate_bin_counts),
        (ref_counts, accumulate_bin_counts_torch),
    ):
        for _ in range(3):
            fn(
                counts=out,
                features=features,
                num_correct_drafts=num_correct,
                num_graded_drafts=num_graded,
                bin_low=-8.0,
                bin_width=0.25,
            )
    assert torch.equal(counts, ref_counts)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
