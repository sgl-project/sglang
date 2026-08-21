"""Regression tests for DFlash and DSpark sampling-filter semantics."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("use_sparse_topk", [False, True])
def test_filtered_support_matches_joint_top_k_top_p_support(use_sparse_topk):
    # The original top-p nucleus includes token 3 because the mass before it is
    # 0.8. Renormalizing top-k before applying top-p incorrectly drops it.
    original_probs = torch.tensor(
        [[0.35, 0.25, 0.20, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]],
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        top_ks=torch.tensor([4], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.82], device="cuda"),
    )

    filtered_probs = build_dflash_verify_target_probs(
        next_token_logits=original_probs.log(),
        sampling_info=sampling_info,
        draft_token_num=1,
        bs=1,
        max_top_k=4,
        uniform_top_k_value=4,
        use_sparse_topk=use_sparse_topk,
    ).view_as(original_probs)

    expected_probs = torch.zeros_like(original_probs)
    expected_probs[0, :4] = original_probs[0, :4] / original_probs[0, :4].sum()
    torch.testing.assert_close(filtered_probs, expected_probs)
