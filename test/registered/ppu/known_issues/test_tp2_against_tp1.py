"""Tracked TP2/TP1 numerical regression; excluded from the blocking PR gate."""

import math
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import generate, prompt_ids, server

register_ppu_ci(est_time=240, suite="known-issues-2-ppu", nightly=True)


def logprobs(result):
    values = result["meta_info"]["input_token_logprobs"]
    pairs = [(item[1], item[0]) for item in values if item[0] is not None]
    assert len(pairs) >= 16
    assert all(math.isfinite(value) for _, value in pairs)
    return pairs


def test_tp2_against_tp1():
    ids = prompt_ids(128)
    with server(tp=1) as base:
        reference = logprobs(
            generate(base, ids, return_logprob=True, logprob_start_len=0)
        )
    # No overlapping model/server lifetimes: TP=2 gets both devices.
    with server(tp=2) as base:
        single = generate(base, ids, return_logprob=True, logprob_start_len=0)
        actual = logprobs(single)
        assert [token for token, _ in reference] == [token for token, _ in actual]
        # BF16 reductions may change low bits. Compare probabilities, not decoded text.
        torch.testing.assert_close(
            torch.tensor([value for _, value in actual]),
            torch.tensor([value for _, value in reference]),
            atol=0.15,
            rtol=0.02,
        )
        batch = generate(base, [ids] * 4)
        assert len(batch) == 4
        for item in [single, *batch]:
            assert isinstance(item["text"], str)
            assert 0 < item["meta_info"]["completion_tokens"] <= 16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
