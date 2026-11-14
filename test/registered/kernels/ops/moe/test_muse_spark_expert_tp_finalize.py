"""Muse Spark expert-TP finalize fallback: per-rank partial expert outputs, each in
its own TRT-LLM permutation, gathered in token order, reduced and then normalized,
must match the finalize of the unsharded output."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.muse_spark import muse_spark_fp4_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

TOP_K = 8
H = 4096
EPS = 1e-8


def _reference(rows, topk_scores, topk_ids, norm_weight, weight_first):
    x = rows.float()
    x = x * norm_weight.float() if weight_first else x
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    x = x if weight_first else x * norm_weight.float()
    num_tokens = topk_ids.shape[0]
    x = x.view(num_tokens, TOP_K, H) * (topk_scores * (topk_ids >= 0)).unsqueeze(-1)
    return x.sum(1)


def _permute(rows, topk_ids):
    # Each rank gets its own random row order, like an independent routing launch.
    live = (topk_ids.reshape(-1) >= 0).nonzero().squeeze(1)
    perm = live[torch.randperm(live.numel(), device=rows.device)]
    permuted = torch.zeros(perm.numel() + 5, H, device=rows.device, dtype=rows.dtype)
    permuted[: perm.numel()] = rows[perm]
    idx = torch.full((rows.shape[0],), -1, dtype=torch.int32, device=rows.device)
    idx[perm] = torch.arange(perm.numel(), dtype=torch.int32, device=rows.device)
    return permuted, idx


@pytest.mark.parametrize("num_tokens", [1, 9, 72, 300])
@pytest.mark.parametrize("weight_first", [True, False])
def test_expert_tp_finalize_fallback_matches_unsharded(num_tokens, weight_first):
    torch.manual_seed(num_tokens)
    world = 8
    topk_ids = torch.randint(0, 256, (num_tokens, TOP_K), device="cuda")
    topk_ids[0, -1] = -1
    partials = [
        (torch.randn(num_tokens * TOP_K, H, device="cuda") / 4).bfloat16()
        for _ in range(world)
    ]
    for p in partials:
        p[topk_ids.reshape(-1) < 0] = 0
    reduced = torch.stack([p.float() for p in partials]).sum(0).bfloat16()
    topk_scores = torch.rand(num_tokens, TOP_K, device="cuda")
    norm_weight = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
    permuted, idx = _permute(partials[0], topk_ids)

    def fake_all_reduce(rows):
        live = topk_ids.reshape(-1) >= 0
        torch.testing.assert_close(rows[live], partials[0][live], rtol=0, atol=0)
        assert (rows[~live] == 0).all()
        return reduced

    with patch.object(
        muse_spark_fp4_moe, "moe_tensor_model_parallel_all_reduce", fake_all_reduce
    ):
        out = muse_spark_fp4_moe.finalize_expert_tp(
            permuted,
            idx,
            topk_scores,
            topk_ids,
            hidden_size=H,
            norm_weight=norm_weight,
            eps=EPS,
            weight_first=weight_first,
            apply_norm=True,
        )
    ref = _reference(reduced, topk_scores, topk_ids, norm_weight, weight_first)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=1e-1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
