"""Kimi-K3 MegaMoE hands DeepGEMM the EP-wide token count under attention DP."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

import sglang.kernels.ops.moe.dsv4 as dsv4_ops
import sglang.srt.layers.moe.mega_moe as mega_moe
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EP_SIZE = 4
TOP_K = 2
HIDDEN = 8


class TestKimiK3MegaMoeGlobalTokens(CustomTestCase):
    def _run(self, num_tokens, global_num_tokens, capture=False, ep_size=EP_SIZE):
        calls = []
        fake_deep_gemm = ModuleType("deep_gemm")
        fake_deep_gemm.fp8_fp4_mega_moe = lambda y, *a, **kw: calls.append(kw)
        buf = SimpleNamespace(
            group=SimpleNamespace(size=lambda: ep_size),
            num_max_tokens_per_rank=8192,
            x=None,
            x_sf=None,
            topk_idx=None,
            topk_weights=None,
        )
        moe = SimpleNamespace(
            _mega_moe_buffer=lambda: buf,
            _mega_top_k=TOP_K,
            moe_hidden_size=HIDDEN,
            experts=SimpleNamespace(
                mega_l1_weights=None,
                mega_l2_weights=None,
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            routed_scaling_factor=None,
        )
        topk = SimpleNamespace(
            topk_ids=torch.zeros(num_tokens, TOP_K, dtype=torch.int64),
            topk_weights=torch.ones(num_tokens, TOP_K),
        )
        with (
            mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}),
            mock.patch.object(dsv4_ops, "mega_moe_pre_dispatch", lambda *a, **kw: None),
            mock.patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                lambda _: nullcontext(),
            ),
            mock.patch.object(
                mega_moe, "get_dp_global_num_tokens", lambda: global_num_tokens
            ),
            mock.patch.object(mega_moe, "get_is_capture_mode", lambda: capture),
            mock.patch.object(mega_moe, "is_dsa_enable_prefill_cp", lambda: False),
        ):
            y = KimiK3MoE._forward_mega_experts(
                moe, torch.zeros(num_tokens, HIDDEN), topk
            )
        self.assertEqual(tuple(y.shape), (num_tokens, HIDDEN))
        self.assertEqual(len(calls), 1)
        return calls[0]["global_num_tokens"]

    def test_unequal_dp_ranks_share_the_ep_wide_count(self):
        counts = [8192, 0, 64, 2048]
        for rank_rows in counts:
            self.assertEqual(self._run(rank_rows, counts), sum(counts))

    def test_attention_tp_slices_share_the_ep_wide_count(self):
        # DP4 x attention TP2 on EP8: the counts are padded to attention-TP
        # multiples and each rank runs half of its DP group's rows.
        counts = [8192, 0, 64, 2048]
        for dp_rows in counts:
            self.assertEqual(
                self._run(dp_rows // 2, counts, ep_size=2 * len(counts)), sum(counts)
            )

    def test_no_hint_when_ep_is_not_a_multiple_of_dp(self):
        self.assertIsNone(self._run(16, [16] * 3))

    def test_no_hint_without_dp_counts_or_during_capture(self):
        self.assertIsNone(self._run(16, None))
        self.assertIsNone(self._run(16, [16] * EP_SIZE, capture=True))


if __name__ == "__main__":
    unittest.main()
