"""Unit tests for the fused group-limited top-k used by the LingBot MoE router."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_group_limited_topk,
    group_limited_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def _ref_group_limited_topk(scores_for_choice, num_experts, n_group, topk_group, top_k):
    seq_len = scores_for_choice.shape[0]
    epg = num_experts // n_group
    grouped = scores_for_choice.view(seq_len, n_group, epg)
    group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1).expand(seq_len, n_group, epg).reshape(seq_len, -1)
    )
    masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,top_k",
    [
        (128, 4, 2, 8),  # LingBot Video production configuration
        (64, 8, 3, 8),
        (128, 8, 3, 8),
        (256, 8, 3, 8),
        (64, 8, 3, 4),
        (32, 4, 2, 4),
    ],
)
@pytest.mark.parametrize("seq_len", [1, 17, 128, 1024])
def test_group_limited_topk_matches_reference(
    num_experts, n_group, topk_group, top_k, seq_len
):
    torch.manual_seed(0)
    scores = torch.randn(seq_len, num_experts, device="cuda", dtype=torch.float32)
    ref = _ref_group_limited_topk(scores, num_experts, n_group, topk_group, top_k)
    fused = group_limited_topk(scores, n_group, topk_group, top_k)
    # Selection order is not guaranteed (sorted=False); compare per-row id sets.
    assert torch.equal(ref.sort(dim=-1)[0], fused.sort(dim=-1)[0])


def test_group_limited_topk_handles_ties():
    # Uniform scores => every group/expert ties. Selection must stay in-range and
    # match the reference (both use first-max tie-breaking).
    seq_len, num_experts, n_group, topk_group, top_k = 8, 64, 8, 3, 8
    scores = torch.ones(seq_len, num_experts, device="cuda", dtype=torch.float32)
    ref = _ref_group_limited_topk(scores, num_experts, n_group, topk_group, top_k)
    fused = group_limited_topk(scores, n_group, topk_group, top_k)
    assert fused.shape == (seq_len, top_k)
    assert (fused >= 0).all() and (fused < num_experts).all()
    assert torch.equal(ref.sort(dim=-1)[0], fused.sort(dim=-1)[0])


def test_group_limited_topk_handles_repeated_nonuniform_scores():
    # Router logits can contain repeated, nonuniform values after rounding or
    # saturation. This exercises duplicate maxima both within and across groups.
    torch.manual_seed(123)
    seq_len, num_experts, n_group, topk_group, top_k = 64, 64, 8, 3, 8
    scores = torch.randint(
        0,
        4,
        (seq_len, num_experts),
        device="cuda",
        dtype=torch.int32,
    ).float()
    ref = _ref_group_limited_topk(scores, num_experts, n_group, topk_group, top_k)
    fused = group_limited_topk(scores, n_group, topk_group, top_k)
    assert torch.equal(ref.sort(dim=-1)[0], fused.sort(dim=-1)[0])


def test_group_limited_topk_rejects_unsupported_group_layout():
    scores = torch.randn(1, 30, device="cuda", dtype=torch.float32)
    assert not can_use_group_limited_topk(scores, 3, 2, 4)
    with pytest.raises(ValueError, match="power-of-two experts per group"):
        group_limited_topk(scores, 3, 2, 4)


def test_group_limited_topk_torch_compile_fullgraph():
    scores = torch.randn(17, 128, device="cuda", dtype=torch.float32)
    ref = _ref_group_limited_topk(scores, 128, 4, 2, 8)
    compiled = torch.compile(lambda x: group_limited_topk(x, 4, 2, 8), fullgraph=True)
    fused = compiled(scores)
    assert torch.equal(ref.sort(dim=-1)[0], fused.sort(dim=-1)[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
