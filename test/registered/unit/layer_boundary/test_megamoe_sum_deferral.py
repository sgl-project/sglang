"""Local MegaMoE TP partials may carry their sum to the next boundary."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.layer_boundary import exit as policy
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    OutputContract,
    StageKind,
)
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.output import UnreducedOutput, complete_owed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMegaMoESumDeferral(unittest.TestCase):
    def facts(
        self,
        *,
        group=SumGroup.TP,
        megamoe=True,
        terminal=False,
        scattered=False,
        dp=False,
        quant=False,
        lora=False,
        shared_tp1=False,
    ):
        plan = SimpleNamespace(
            finishes_directly=False, terminal=terminal, _speculative_algo=None
        )
        path = SimpleNamespace(
            output=OutputContract(
                Layout(frozenset()), group=group, may_defer_to_next=True
            ),
            output_move_completes_sum=False,
            returns_over_dp=False,
        )
        with ExitStack() as stack:
            values = {
                "get_parallel": SimpleNamespace(
                    tp_size=4, moe_ep_size=4, moe_tp_size=1
                ),
                "get_moe_a2a_backend": (
                    policy.MoeA2ABackend.MEGAMOE
                    if megamoe
                    else policy.MoeA2ABackend.DEEPEP
                ),
                "is_dp_attention_enabled": dp,
                "is_enable_moe_cp_allgather": False,
                "get_exec": SimpleNamespace(
                    comm=SimpleNamespace(enable_quant_communications=quant)
                ),
                "get_lora": SimpleNamespace(enable_lora=lora),
                "post_experts_sum_is_one_all_reduce": False,
                "_fused_consumer_sum": False,
            }
            for name, value in values.items():
                stack.enter_context(patch.object(policy, name, return_value=value))
            stack.enter_context(
                patch.object(
                    policy.envs.SGLANG_SHARED_EXPERT_TP1,
                    "get",
                    return_value=shared_tp1,
                )
            )
            return policy.exit_facts(
                StageKind.FFN,
                plan,
                BatchVariant.INPUT_SCATTERED if scattered else BatchVariant.ORDINARY,
                path,
            )

    def test_tp_partial_can_defer_even_when_post_experts_helper_rejects_a2a(self):
        facts = self.facts()
        self.assertTrue(facts.single_sum)
        with patch.object(policy, "_ffn_has_tokens", return_value=True):
            self.assertTrue(policy._batch_allows_deferred_sum(object(), facts))
        with patch.object(policy, "_ffn_has_tokens", return_value=False):
            self.assertFalse(policy._batch_allows_deferred_sum(object(), facts))

    def test_other_outputs_and_existing_guards_stay_ineligible(self):
        for kwargs in (
            {"group": None},
            {"group": SumGroup.MOE_OUTPUT},
            {"group": SumGroup.ATTN_TP},
            {"megamoe": False},
            {"terminal": True},
            {"scattered": True},
            {"dp": True},
            {"quant": True},
            {"lora": True},
            {"shared_tp1": True},
        ):
            with self.subTest(**kwargs):
                self.assertFalse(self.facts(**kwargs).single_sum)

    def test_unfused_consumer_completes_tp_partial_once(self):
        value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        reduce = Mock(side_effect=lambda x: x * 4)
        group = SimpleNamespace(all_reduce=reduce)
        result = complete_owed(UnreducedOutput(value, group))
        torch.testing.assert_close(result, value * 4, rtol=0, atol=0)
        reduce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
