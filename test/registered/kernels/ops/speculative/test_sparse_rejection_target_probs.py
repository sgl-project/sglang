import sys
import types

import pytest
import torch

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.sampling.renorm_triton import (
    top_k_renorm_probs_triton,
    top_p_renorm_probs_triton,
)
from sglang.srt.speculative.eagle_utils import _sparse_rejection_target_probs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

ROWS_PER_REQUEST = 4


def _sampling_info(top_ks, top_ps, temperatures):
    return types.SimpleNamespace(
        temperatures=torch.tensor(temperatures, device="cuda").view(-1, 1),
        top_ks=torch.tensor(top_ks, device="cuda", dtype=torch.int32),
        top_ps=torch.tensor(top_ps, device="cuda"),
        need_top_k_sampling=True,
        need_top_p_sampling=any(p < 1.0 for p in top_ps),
    )


def _dense_target_probs(logits, info):
    probs = sampling_softmax(
        logits, temperatures=info.temperatures, rows_per_request=ROWS_PER_REQUEST
    )
    # top-p on the full distribution first, then top-k, as _apply_joint_top_k_top_p
    if info.need_top_p_sampling:
        probs = top_p_renorm_probs_triton(
            probs, info.top_ps.repeat_interleave(ROWS_PER_REQUEST)
        )
    return top_k_renorm_probs_triton(
        probs, info.top_ks.repeat_interleave(ROWS_PER_REQUEST)
    )


@pytest.mark.parametrize("vocab_size", [151936, 248320])
@pytest.mark.parametrize(
    "top_ks, top_ps, temperatures",
    [
        ([20] * 8, [0.95] * 8, [1.0] * 8),
        ([20, 64, 128, 2], [0.95, 0.8, 1.0, 0.5], [1.0, 0.7, 1.3, 0.5]),
        ([5, 40], [1.0, 1.0], [0.6, 1.0]),
    ],
)
def test_sparse_rejection_target_probs_match_dense(
    vocab_size, top_ks, top_ps, temperatures
):
    torch.manual_seed(0)
    info = _sampling_info(top_ks, top_ps, temperatures)
    rows = len(top_ks) * ROWS_PER_REQUEST
    # bf16-rounded logits tie at the k-th value and at the top-p pivot, as in serving.
    logits = (torch.randn(rows, vocab_size, device="cuda") * 4).bfloat16().float()
    logits[0, 7] += 30.0

    dense = _dense_target_probs(logits, info)
    sparse = _sparse_rejection_target_probs(logits, info, ROWS_PER_REQUEST, max(top_ks))

    assert torch.equal(dense > 0, sparse > 0)
    total_variation = 0.5 * (dense - sparse).abs().sum(-1)
    assert total_variation.max().item() < 1e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
