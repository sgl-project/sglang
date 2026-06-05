"""Regression test: the next-token sampler path must not choke on per-position ids.

OPD per-position scoring folds a ``List[List[int]]`` (one id-list per input position)
into the per-request ``token_ids_logprob`` slot. That data is consumed by the
logits_processor INPUT-logprob path; the next-token sampler path
(``get_token_ids_logprobs_batch_optimized`` / ``Sampler.compute_logprobs_only``) has no
meaning for it and previously crashed on the nested structure. Callers now filter
per-position entries out; this guards against a future caller feeding them back in.

Imports the sampler module, so run in a real sglang environment (GPU not required).
"""

import pytest
import torch

from sglang.srt.layers.sampler import get_token_ids_logprobs_batch_optimized
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small")


def test_flat_batch_optimized_unchanged():
    torch.manual_seed(0)
    lp = torch.randn(3, 8)
    vals, idxs = get_token_ids_logprobs_batch_optimized(lp, [[1, 3], [2], None])
    assert idxs[0] == [1, 3] and idxs[1] == [2]
    assert torch.allclose(torch.as_tensor(vals[0]).flatten(), lp[0, [1, 3]])
    assert torch.allclose(torch.as_tensor(vals[1]).flatten(), lp[1, [2]])


def test_per_position_entry_is_rejected_loudly():
    lp = torch.randn(1, 8)
    with pytest.raises(AssertionError):
        get_token_ids_logprobs_batch_optimized(lp, [[[1, 2], [3, 4]]])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
