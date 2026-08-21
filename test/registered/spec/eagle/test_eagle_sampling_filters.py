"""Regression tests for EAGLE sampling-filter semantics."""

from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_utils import _apply_joint_top_k_top_p
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def test_filtered_support_matches_joint_top_k_top_p_support():
    # The original top-p nucleus includes token 3 because the mass before it is
    # 0.8. Applying and renormalizing top-k first incorrectly drops that token.
    original_probs = torch.tensor(
        [[0.35, 0.25, 0.20, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]],
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        top_ks=torch.tensor([4], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.82], device="cuda"),
    )

    filtered_probs = _apply_joint_top_k_top_p(
        original_probs, sampling_info, draft_token_num=1
    )

    expected_probs = torch.zeros_like(original_probs)
    expected_probs[0, :4] = original_probs[0, :4] / original_probs[0, :4].sum()
    torch.testing.assert_close(filtered_probs, expected_probs)
