"""Numerical and admission regressions for the opt-in MoE/EP reference."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples.runtime.sid_gr.launch_moe_reference import validate_configuration
from examples.runtime.sid_gr.moe_model import (
    SIDQwen3MoeForCausalLM,
    combine_expert_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMoeReference(unittest.TestCase):
    def test_slot_ownership_prevents_bf16_partial_sum_loss(self):
        slots = torch.tensor([[[1.00390625] * 8, [-1.0] * 8]], dtype=torch.float32)
        expected = slots.sum(1).to(torch.bfloat16)
        prematurely_rounded = slots.to(torch.bfloat16).sum(1)
        self.assertFalse(torch.equal(expected, prematurely_rounded))
        for size in (1, 2, 4):
            assembled = []
            for rank in range(size):
                local = torch.zeros_like(slots)
                for k in range(2):
                    if k % size == rank:
                        local[:, k] = slots[:, k]

                def reduce(x, local=local):
                    self.assertTrue(torch.equal(x, local))
                    return slots.clone()

                group = SimpleNamespace(
                    world_size=size, rank_in_group=rank, all_reduce=reduce
                )
                assembled.append(combine_expert_slots(local, group, 8))
            self.assertTrue(torch.equal(sum(assembled), expected))

    def test_reject_unsupported_topology_and_model(self):
        config = SimpleNamespace(
            hidden_size=1024,
            moe_intermediate_size=1024,
            vocab_size=256,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
        )
        for tp, ep in ((2, 1), (4, 2), (8, 8)):
            with (
                self.subTest(tp=tp, ep=ep),
                patch(
                    "examples.runtime.sid_gr.moe_model.get_parallel",
                    return_value=SimpleNamespace(
                        tp_size=tp, moe_ep_size=ep, moe_tp_size=tp // ep
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "TP=EP"):
                    SIDQwen3MoeForCausalLM(config)
        config.num_experts = 16
        with patch(
            "examples.runtime.sid_gr.moe_model.get_parallel",
            return_value=SimpleNamespace(tp_size=2, moe_ep_size=2, moe_tp_size=1),
        ):
            with self.assertRaisesRegex(ValueError, "geometry"):
                SIDQwen3MoeForCausalLM(config)

    def test_reject_unsupported_launch_modes(self):
        config = dict(
            enable_gr=True,
            dtype="bfloat16",
            page_size=1,
            tp_size=2,
            ep_size=2,
            pp_size=1,
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attention_backend="sid_gr_reference",
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            cuda_graph_backend_decode="disabled",
            cuda_graph_backend_prefill="disabled",
            kv_cache_dtype="auto",
            moe_a2a_backend="none",
            moe_runner_backend="triton",
        )
        validate_configuration(config)
        for field, value in dict(
            ep_size=1,
            moe_a2a_backend="deepep",
            moe_runner_backend="cutlass",
            enable_eplb=True,
            ep_num_redundant_experts=1,
            disable_overlap_schedule=False,
            attn_dp_size=2,
            page_size=16,
            quantization="fp8",
        ).items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_configuration({**config, field: value})
