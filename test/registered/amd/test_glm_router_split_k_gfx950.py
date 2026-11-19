"""GLM-5.3-Flash's ROCm decode router: the Triton router reads the split-K GEMV partials and
writes the padded rows the aiter MoE expects; the BF16 linear keeps tuned shapes on tgemm."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm.router_gemv_hip import (
    ROCM_GLM_ROUTER_MAX_TOKENS,
    rocm_router_gemv_split_k,
    rocm_router_reduce_partials,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd-mi35x")

# GLM-5.3-Flash: 288 routed experts, top-8 plus one fused shared expert, hidden 4096
NUM_EXPERTS, TOPK, HIDDEN, SCALE = 288, 8, 4096, 2.5


def _gate(scores, bias, **kwargs):
    return moe_fused_gate(
        scores,
        bias,
        TOPK + 1,
        scoring_func="sigmoid",
        num_fused_shared_experts=1,
        renormalize=True,
        routed_scaling_factor=SCALE,
        **kwargs,
    )


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "ROCm gfx95 only")
class TestGlmRouterSplitK(CustomTestCase):
    def setUp(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        self.weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        self.x = torch.randn(
            ROCM_GLM_ROUTER_MAX_TOKENS, HIDDEN, device="cuda", generator=gen
        ).to(torch.bfloat16)
        self.bias = torch.randn(NUM_EXPERTS, device="cuda", generator=gen) * 0.1

    def test_partials_match_reduced_logits(self):
        """Summing the partials inside the router gives the same bits as reducing them
        into fp32 logits first, at every MTP verify row count up to 96."""
        for m in (6, 18, 36, 60, ROCM_GLM_ROUTER_MAX_TOKENS):
            partials = rocm_router_gemv_split_k(self.x[:m], self.weight)
            logits = torch.empty(m, NUM_EXPERTS, device="cuda")
            rocm_router_reduce_partials(partials, logits)
            w_ref, ids_ref = _gate(logits, self.bias)
            w, ids = _gate(partials, self.bias)
            self.assertTrue(torch.equal(ids, ids_ref), m)
            self.assertTrue(torch.equal(w, w_ref), m)

    def test_padded_rows_take_routed_id_zero(self):
        """CUDA graph padding: rows past num_token_non_padded get routed id 0, the
        shared id and zero weights, and the live rows are unchanged."""
        m, live = 36, 25
        partials = rocm_router_gemv_split_k(self.x[:m], self.weight)
        w_ref, ids_ref = _gate(partials, self.bias)
        n = torch.tensor([live], dtype=torch.int32, device="cuda")
        w, ids = _gate(partials, self.bias, num_token_non_padded=n, padded_routed_id=0)
        self.assertTrue(torch.equal(ids[:live], ids_ref[:live]))
        self.assertTrue(torch.equal(w[:live], w_ref[:live]))
        self.assertTrue(torch.all(ids[live:, :TOPK] == 0))
        self.assertTrue(torch.all(ids[live:, TOPK] == NUM_EXPERTS))
        self.assertTrue(torch.all(w[live:] == 0))
        _, ids_neg = _gate(partials, self.bias, num_token_non_padded=n)
        self.assertTrue(torch.all(ids_neg[live:] == -1))


class TestAiterJitRouterGate(CustomTestCase):
    def test_single_group_shared_on_gfx95_only(self):
        """Only single-group routing with fused shared experts and a shared bias takes
        the Triton router, and only on gfx95: MI300 keeps aiter biased_grouped_topk."""
        from sglang.srt.layers.moe import topk

        with mock.patch.object(topk, "_use_aiter", True):
            with mock.patch.object(topk, "_is_gfx95", True):
                self.assertTrue(topk._aiter_uses_jit_router(1, 1, False, 1))
                self.assertTrue(topk._aiter_uses_jit_router(None, None, False, 1))
                self.assertFalse(topk._aiter_uses_jit_router(8, 4, False, 1))
                self.assertFalse(topk._aiter_uses_jit_router(1, 1, False, 0))
                self.assertFalse(topk._aiter_uses_jit_router(1, 1, True, 1))
            with mock.patch.object(topk, "_is_gfx95", False):
                self.assertFalse(topk._aiter_uses_jit_router(1, 1, False, 1))


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "ROCm gfx95 only")
class TestSharedExpertsFusionDisabled(CustomTestCase):
    """--disable-shared-experts-fusion leaves no shared slot for the Triton router."""

    def test_gate_needs_a_global_shared_slot(self):
        """The split-K GEMV serves the gate only when the Triton router will read its
        partials: one global fused shared slot, not zero and not per-rank slots."""
        from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe as hip_moe

        if not hip_moe._use_aiter:
            self.skipTest("needs SGLANG_USE_AITER=1")
        gate = SimpleNamespace(
            is_deepseek_v4=False,
            weight=torch.empty(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16),
        )
        config = SimpleNamespace(
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            n_group=1,
            topk_group=1,
            n_routed_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
        )

        def max_tokens(num_fused, per_rank=False):
            model = SimpleNamespace(
                shared_experts=SharedExpertsConfig(
                    fusion_disabled=num_fused == 0, num_fused_shared_experts=num_fused
                )
            )
            with mock.patch.object(hip_moe, "get_model", return_value=model):
                with mock.patch.object(
                    hip_moe, "has_per_rank_fused_shared_slots", return_value=per_rank
                ):
                    return hip_moe.router_max_tokens(gate, config, is_hash_moe=False)

        self.assertEqual(max_tokens(1), ROCM_GLM_ROUTER_MAX_TOKENS)
        self.assertEqual(max_tokens(0), -1)
        self.assertEqual(max_tokens(1, per_rank=True), -1)

    def test_partials_without_shared_slot_reduce_first(self):
        """Partials reaching select_experts with no shared slot are reduced into the
        logits before routing, matching a run on the reduced logits."""
        from sglang.srt.layers.moe.topk import (
            TopKConfig,
            TopKOutputFormat,
            select_experts,
        )

        gen = torch.Generator(device="cuda").manual_seed(0)
        weight = (
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        x = torch.randn(36, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
        config = TopKConfig(
            top_k=TOPK,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            renormalize=True,
            shared_experts=SharedExpertsConfig(fusion_disabled=True),
            correction_bias=torch.randn(NUM_EXPERTS, device="cuda", generator=gen),
            routed_scaling_factor=SCALE,
            output_format=TopKOutputFormat.STANDARD,
            scoring_func="sigmoid",
        )
        partials = rocm_router_gemv_split_k(x, weight)
        logits = torch.empty(x.shape[0], NUM_EXPERTS, device="cuda")
        rocm_router_reduce_partials(partials, logits)
        ref = select_experts(x, logits.clone(), config)
        out = select_experts(
            x,
            torch.empty_like(logits),
            config,
            router_logits_partials=partials,
        )
        self.assertEqual(out.topk_ids.shape[1], TOPK)
        self.assertTrue(torch.equal(out.topk_ids, ref.topk_ids))
        self.assertTrue(torch.equal(out.topk_weights, ref.topk_weights))


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "ROCm gfx95 only")
class TestPreferTritonA16W16(CustomTestCase):
    def test_tuned_row_keeps_tgemm(self):
        """A small-K decode shape goes to the Triton GEMM only when no tuned row
        covers it; a registered exact shape stays on Triton either way."""
        from sglang.srt.layers.quantization import unquant

        x = torch.empty(36, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.empty(4096, 2048, device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(unquant, "_has_tuned_a16w16", return_value=False):
            self.assertTrue(unquant._prefer_triton_a16w16(x, w))
        with mock.patch.object(unquant, "_has_tuned_a16w16", return_value=True):
            self.assertFalse(unquant._prefer_triton_a16w16(x, w))
            with mock.patch.object(
                unquant, "_gfx950_triton_a16w16_shapes", {(36, 4096, 2048)}
            ):
                self.assertTrue(unquant._prefer_triton_a16w16(x, w))


if __name__ == "__main__":
    unittest.main()
