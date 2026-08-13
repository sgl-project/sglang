"""Shared-outer token-dedup gate/up-A: bitwise-equal to the control form.

The §41.1(3) dedup form computes each (token, adapter) A product once
(token-major bridge, B reads row ``pair // K``) instead of K times.  What
these cases pin:

* **Bitwise equality through B.** The control's K per-pair copies of a
  token's A row come from identical inputs and weights, so B consumes
  identical BF16 values in both forms and the materialized gate/up delta
  must be EXACTLY equal — including the zero-overwrite at sentinel
  destinations (both output buffers start as different garbage).  This is
  what turns the Step-3 dedup decision into a pure performance question;
  any masking/indexing slip in the ``intermediate_top_k`` path breaks
  equality loudly.
* **The identity ID pass** (§41.1(3a)): with ``lora_experts_per_adapter == 1``
  the T-domain plan's fused key IS the adapter slot — pinned so a future
  routing change cannot silently reintroduce a nontrivial key where the
  dedup form assumes identity.
* **Graph determinism**: the dedup chain (A over the token plan + B with
  ``intermediate_top_k=K``) replays bitwise under CUDA graph, same bar as
  every other Step-3 arm.

Triton only: no SM100 requirement, so this also runs on the H200 pod.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.lora_a_shared import (
    build_token_adapter_plan,
    shared_gate_up_a_token_dedup,
    shared_gate_up_delta_from_token_bridge,
)
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a, stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing

LORA_A = PROVISIONAL_LAUNCH_CONFIG.lora_a
LORA_B = PROVISIONAL_LAUNCH_CONFIG.lora_b


class TestSharedOuterTokenDedup(CustomTestCase):
    T = 24
    K = 4
    H = 128
    I_LOCAL = 64
    E_LOCAL = 8
    L_CAP = 4
    RANK = 16

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def setUp(self):
        torch.manual_seed(31)
        device = self.device
        self.hidden = torch.randn(self.T, self.H, dtype=torch.bfloat16, device=device)
        # One shared factor per adapter (E_f = 1), canonical [gate|up] packing.
        self.a_shared = (
            torch.randn(
                self.L_CAP * 1,
                2 * self.RANK,
                self.H,
                dtype=torch.float32,
                device=device,
            )
            * self.H**-0.5
        ).to(torch.bfloat16)
        self.b_gate_up = (
            torch.randn(
                self.L_CAP * self.E_LOCAL,
                2 * self.I_LOCAL,
                self.RANK,
                dtype=torch.float32,
                device=device,
            )
            * 0.5
            * self.RANK**-0.5
        ).to(torch.bfloat16)
        self.topk_ids = torch.randint(
            0, self.E_LOCAL, (self.T, self.K), dtype=torch.int32, device=device
        )
        self.topk_ids[1, 2] = -1  # literal sentinel
        slots = torch.tensor(
            [i % (self.L_CAP + 1) - 1 for i in range(self.T)],
            dtype=torch.int32,
            device=device,
        )  # includes -1 base tokens
        self.token_slots = slots
        self.per_expert = build_virtual_expert_routing(
            self.topk_ids,
            self.token_slots,
            lora_experts_per_adapter=self.E_LOCAL,
            max_loras=self.L_CAP,
            block_size=16,
        )
        # The section 60.5 form: the adapter's single LoRA expert (id 0) via constexpr, the
        # explicit expert-range bound replacing the degenerate zeros map.
        self.outer_pairs = build_virtual_expert_routing(
            self.topk_ids,
            self.token_slots,
            lora_experts_per_adapter=1,
            max_loras=self.L_CAP,
            block_size=16,
            shared_outer_local_expert_count=self.E_LOCAL,
        )
        self.token_plan = build_token_adapter_plan(
            self.token_slots, max_loras=self.L_CAP, block_size=16
        )

    def _control_delta(self) -> torch.Tensor:
        pairs = self.T * self.K
        rank_out = torch.empty(
            pairs, 2 * self.RANK, dtype=torch.bfloat16, device=self.device
        )
        grouped_lora_a(
            self.hidden, self.a_shared, rank_out, self.outer_pairs, config=LORA_A
        )
        delta = torch.full(
            (pairs, 2 * self.I_LOCAL), 71.0, dtype=torch.bfloat16, device=self.device
        )
        stock_grouped_lora_b(
            rank_out,
            self.b_gate_up,
            delta,
            self.per_expert,
            destination_offsets=(0, self.I_LOCAL),
            config=LORA_B,
        )
        return delta

    def _dedup_delta(self) -> torch.Tensor:
        rank_out_tokens = torch.empty(
            self.T, 2 * self.RANK, dtype=torch.bfloat16, device=self.device
        )
        shared_gate_up_a_token_dedup(
            self.hidden,
            self.a_shared,
            self.token_plan,
            rank_out_tokens,
            config=LORA_A,
        )
        delta = torch.full(
            (self.T * self.K, 2 * self.I_LOCAL),
            -3.0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        shared_gate_up_delta_from_token_bridge(
            rank_out_tokens,
            self.b_gate_up,
            delta,
            self.per_expert,
            intermediate_size=self.I_LOCAL,
            config=LORA_B,
        )
        return delta

    def test_dedup_is_bitwise_equal_to_the_control_form(self):
        control = self._control_delta()
        dedup = self._dedup_delta()
        self.assertTrue(torch.equal(control, dedup))
        # The comparison must not be vacuous: real signal and real sentinels.
        self.assertGreater(float(control.abs().max()), 0.0)
        pair_keys = self.per_expert.virtual_topk_ids.reshape(-1)
        self.assertTrue(bool((pair_keys == -1).any()))
        zero_rows = control[pair_keys == -1]
        self.assertTrue(torch.equal(zero_rows, torch.zeros_like(zero_rows)))

    def test_intermediate_top_k_contract_fails_closed(self):
        # Third S3 review: a pair-major bridge accidentally passed with
        # top_k=K used to SUCCEED while reading row pair//K — silently
        # wrong results. Only the two real shapes may pass.
        pairs = self.T * self.K
        pair_major = torch.zeros(
            pairs, 2 * self.RANK, dtype=torch.bfloat16, device=self.device
        )
        delta = torch.zeros(
            pairs, 2 * self.I_LOCAL, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "token-major"):
            stock_grouped_lora_b(
                pair_major,
                self.b_gate_up,
                delta,
                self.per_expert,
                destination_offsets=(0, self.I_LOCAL),
                config=LORA_B,
                intermediate_top_k=self.K,
            )
        token_major = torch.zeros(
            self.T, 2 * self.RANK, dtype=torch.bfloat16, device=self.device
        )
        with self.assertRaisesRegex(ValueError, "pair-major"):
            stock_grouped_lora_b(
                token_major,
                self.b_gate_up,
                delta,
                self.per_expert,
                destination_offsets=(0, self.I_LOCAL),
                config=LORA_B,
                intermediate_top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "intermediate_top_k"):
            stock_grouped_lora_b(
                token_major,
                self.b_gate_up,
                delta,
                self.per_expert,
                destination_offsets=(0, self.I_LOCAL),
                config=LORA_B,
                intermediate_top_k=2,
            )

    def test_token_plan_key_is_the_adapter_slot(self):
        keys = self.token_plan.virtual_topk_ids.reshape(-1)
        expected = torch.where(
            (self.token_slots >= 0) & (self.token_slots < self.L_CAP),
            self.token_slots,
            torch.full_like(self.token_slots, -1),
        )
        self.assertTrue(torch.equal(keys, expected.to(keys.dtype)))

    def test_dedup_chain_replays_bitwise_under_graph(self):
        first = self._dedup_delta()
        rank_out_tokens = torch.empty(
            self.T, 2 * self.RANK, dtype=torch.bfloat16, device=self.device
        )
        delta = torch.zeros(
            self.T * self.K, 2 * self.I_LOCAL, dtype=torch.bfloat16, device=self.device
        )

        def chain():
            shared_gate_up_a_token_dedup(
                self.hidden,
                self.a_shared,
                self.token_plan,
                rank_out_tokens,
                config=LORA_A,
            )
            shared_gate_up_delta_from_token_bridge(
                rank_out_tokens,
                self.b_gate_up,
                delta,
                self.per_expert,
                intermediate_size=self.I_LOCAL,
                config=LORA_B,
            )

        chain()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain()
        for _ in range(32):
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(delta, first))


if __name__ == "__main__":
    unittest.main()
