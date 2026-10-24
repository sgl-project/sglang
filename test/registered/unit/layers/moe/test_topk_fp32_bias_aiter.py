"""Regression test for https://github.com/sgl-project/sglang/issues/44794.

``select_experts`` downcast every correction bias to the router logits' dtype
whenever the aiter grouped-topk path was taken. For GLM-5.x
(``GlmMoeDsa``), ``MoEGate`` keeps ``e_score_correction_bias`` in fp32 on
purpose (#37133) and the aiter branch of ``biased_grouped_topk_gpu`` upcasts
the logits to fp32 when it sees an fp32 bias. The downcast in
``select_experts`` silently disabled that path, so GLM-5.x on ROCm routed with
a bf16 bias (which collapses the offset bias and reorders top-k) while CUDA
kept fp32 routing.

The fix: skip the downcast when the bias is already fp32. Non-fp32 biases
still follow the old behavior.
"""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class SelectExpertsFp32BiasTestMixin:
    device: str
    num_experts = 256

    def _bias_reaching_grouped_topk(self, bias_dtype):
        """Run select_experts on the aiter grouped-topk path and capture the
        correction bias handed to the grouped topk kernel."""
        captured = {}

        def fake_biased_grouped_topk(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        ):
            captured["bias"] = correction_bias
            return torch.zeros(
                (gating_output.shape[0], topk), device=gating_output.device
            ), torch.zeros(
                (gating_output.shape[0], topk),
                dtype=torch.int32,
                device=gating_output.device,
            )

        bias = torch.randn(self.num_experts, dtype=bias_dtype, device=self.device)
        config = TopKConfig(
            top_k=8,
            use_grouped_topk=True,
            topk_group=8,
            num_expert_group=32,
            correction_bias=bias,
        )
        router_logits = torch.randn(
            4, self.num_experts, dtype=torch.bfloat16, device=self.device
        )
        hidden_states = torch.randn(4, 16, dtype=torch.bfloat16, device=self.device)

        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                topk_module, "biased_grouped_topk", fake_biased_grouped_topk
            ),
        ):
            select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=config,
            )
        return bias, captured["bias"]

    def test_fp32_bias_is_not_downcast_before_aiter_path(self):
        # #44794: the fp32 bias must reach the aiter path untouched so that
        # biased_grouped_topk_gpu's fp32 gate engages (it upcasts the logits
        # and routes in fp32, like CUDA).
        original, received = self._bias_reaching_grouped_topk(torch.float32)
        self.assertIsNotNone(received)
        self.assertEqual(received.dtype, torch.float32)
        self.assertEqual(received.data_ptr(), original.data_ptr())

    def test_bf16_bias_keeps_old_cast_to_logits_dtype(self):
        # Non-fp32 biases still go through the pre-existing downcast to the
        # gating dtype that aiter_biased_grouped_topk expects.
        _original, received = self._bias_reaching_grouped_topk(torch.bfloat16)
        self.assertEqual(received.dtype, torch.bfloat16)

    def test_fp16_bias_keeps_old_cast_to_logits_dtype(self):
        _original, received = self._bias_reaching_grouped_topk(torch.float16)
        self.assertEqual(received.dtype, torch.bfloat16)

    def test_bf16_cast_of_offset_bias_reorders_topk(self):
        # Mechanism behind the reporter's 67.6% divergence: an offset bias
        # whose spread is a few bf16 ULPs wide collapses in bf16 and reorders
        # top-k routing. Deterministic seeds; asserts the divergence exists at
        # all (a lower bound of zero would make this test vacuous).
        torch.manual_seed(20261016)
        bias_fp32 = 34.0 + 0.12 * torch.randn(
            self.num_experts, dtype=torch.float32, device=self.device
        )
        logits = torch.randn(256, self.num_experts, dtype=torch.bfloat16)
        ids_fp32 = torch.topk((logits.float() + bias_fp32.float()).float(), k=8).indices
        ids_bf16 = (
            torch.topk((logits.float() + bias_fp32.bfloat16().float()).float(), k=8)
            .indices
        )
        diff_frac = (ids_fp32 != ids_bf16).any(dim=1).float().mean().item()
        self.assertGreater(diff_frac, 0.0)


class TestSelectExpertsFp32BiasCPU(SelectExpertsFp32BiasTestMixin, CustomTestCase):
    device = "cpu"


if __name__ == "__main__":
    unittest.main()
