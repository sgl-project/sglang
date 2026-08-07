"""Unit tests for DeepseekV2MoE._reduce_and_add_tp1_shared_output.

Covers the TP1 shared-expert add across the three post-experts reduction
regimes, with compatibility (unchanged behavior) as the primary concern:

1. flag OFF                        -> helper never touches shared_output
2. flag ON, all-reduce runs (TP)   -> full S added (pre-fix behavior)
3. flag ON, all-reduce skipped     -> S/moe_ep_size added (the fix): a
   downstream cross-rank SUM (dp-attention reduce-scatterv / fused
   all-reduce) sums the replicated S once per rank, so the pre-scaled add
   yields exactly R + S after that SUM.
4. tp_size == 1                    -> full S added, no reduction
"""

import unittest
from unittest import mock

import torch

from sglang.srt.models import deepseek_v2 as dsv2_mod
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-a", runner_config="1-gpu-small")


def _make_moe(tp_size: int, moe_ep_size: int, shared_expert_tp1: bool):
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    moe.tp_size = tp_size
    moe.moe_ep_size = moe_ep_size
    moe._shared_expert_tp1 = shared_expert_tp1
    return moe


class TestSharedExpertTp1Reduce(CustomTestCase):
    def setUp(self):
        self.routed = torch.randn(4, 8)
        self.shared = torch.randn(4, 8)

    def _run(self, moe, *, skip_all_reduce: bool, shared=...):
        shared = self.shared if shared is ... else shared
        with mock.patch.object(
            dsv2_mod,
            "should_skip_post_experts_all_reduce",
            return_value=skip_all_reduce,
        ), mock.patch.object(
            dsv2_mod,
            "tensor_model_parallel_all_reduce",
            side_effect=lambda t: t,
        ) as all_reduce:
            out = moe._reduce_and_add_tp1_shared_output(self.routed.clone(), shared)
        return out, all_reduce

    # ── Compatibility: pre-fix behavior must be unchanged ──────────────

    def test_flag_off_shared_untouched(self):
        moe = _make_moe(tp_size=8, moe_ep_size=8, shared_expert_tp1=False)
        for skip in (False, True):
            out, _ = self._run(moe, skip_all_reduce=skip)
            torch.testing.assert_close(out, self.routed)

    def test_flag_on_all_reduce_runs_adds_full_shared(self):
        moe = _make_moe(tp_size=8, moe_ep_size=8, shared_expert_tp1=True)
        out, all_reduce = self._run(moe, skip_all_reduce=False)
        all_reduce.assert_called_once()
        torch.testing.assert_close(out, self.routed + self.shared)

    def test_tp1_adds_full_shared_no_reduce(self):
        moe = _make_moe(tp_size=1, moe_ep_size=1, shared_expert_tp1=True)
        for skip in (False, True):
            out, all_reduce = self._run(moe, skip_all_reduce=skip)
            all_reduce.assert_not_called()
            torch.testing.assert_close(out, self.routed + self.shared)

    def test_shared_none_is_noop(self):
        moe = _make_moe(tp_size=8, moe_ep_size=8, shared_expert_tp1=True)
        for skip in (False, True):
            out, _ = self._run(moe, skip_all_reduce=skip, shared=None)
            torch.testing.assert_close(out, self.routed)

    # ── The fix: skipped all-reduce pre-scales the replicated shared ───

    def test_flag_on_all_reduce_skipped_prescales_shared(self):
        moe = _make_moe(tp_size=8, moe_ep_size=8, shared_expert_tp1=True)
        out, all_reduce = self._run(moe, skip_all_reduce=True)
        all_reduce.assert_not_called()
        torch.testing.assert_close(out, self.routed + self.shared / 8)

    def test_downstream_sum_restores_r_plus_s(self):
        # Per-rank routed partials sum to R across the EP group; the
        # replicated shared output is identical on every rank. The
        # downstream reduce-scatter SUMs the per-rank tensors, so with the
        # pre-scale the total is exactly R + S (pre-fix: R + ep_size*S).
        ep_size = 8
        moe = _make_moe(tp_size=ep_size, moe_ep_size=ep_size, shared_expert_tp1=True)
        rank_partials = [torch.randn(4, 8) for _ in range(ep_size)]
        routed_total = torch.stack(rank_partials).sum(dim=0)

        summed = torch.zeros_like(routed_total)
        for partial in rank_partials:
            self.routed = partial
            out, _ = self._run(moe, skip_all_reduce=True)
            summed += out

        torch.testing.assert_close(
            summed, routed_total + self.shared, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
