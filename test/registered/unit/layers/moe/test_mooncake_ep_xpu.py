"""
python3 -m unittest test_mooncake_ep_xpu.py
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.ep_moe.layer import (
    DeepEPMoE,
    _xpu_masked_activation,
    _xpu_masked_grouped_experts,
)
from sglang.srt.layers.moe.token_dispatcher.mooncake import _ActiveRanksRing
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

NUM_EXPERTS, CAPACITY, HIDDEN, INTER, TOPK = 8, 16, 256, 128, 2
EP_SIZE, TOKENS_PER_RANK = 2, 4
# Expert 0 gets every dispatched token,
# and the total is the most rows a step can carry: TOPK * EP_SIZE * TOKENS_PER_RANK.
MASKED_M = [8, 0, 3, 0, 5, 0, 0, 0]


@unittest.skipUnless(is_xpu(), "requires an Intel XPU")
class TestXpuMaskedMoe(CustomTestCase):
    def setUp(self):
        generator = torch.Generator(device="xpu").manual_seed(0)

        def randn(*shape, scale=1.0):
            x = torch.randn(shape, device="xpu", generator=generator) * scale
            return x.to(torch.bfloat16)

        self.hidden = randn(NUM_EXPERTS, CAPACITY, HIDDEN)
        self.w13 = randn(NUM_EXPERTS, 2 * INTER, HIDDEN, scale=HIDDEN**-0.5)
        self.w2 = randn(NUM_EXPERTS, HIDDEN, INTER, scale=INTER**-0.5)
        self.masked_m = torch.tensor(MASKED_M, dtype=torch.int32, device="xpu")

    def _assert_matches_reference(self, out):
        for expert, m in enumerate(MASKED_M):
            x = self.hidden[expert, :m].float()
            gate, up = (x @ self.w13[expert].float().T).chunk(2, dim=-1)
            ref = (torch.nn.functional.silu(gate) * up) @ self.w2[expert].float().T
            torch.testing.assert_close(
                out[expert, :m].float(), ref, rtol=3e-2, atol=3e-2
            )

    def test_grouped_runner_matches_per_expert_reference(self):
        # Fewer step tokens than were dispatched only change the GEMM's tile:
        # the rows still land in buffers sized for the worst case.
        for num_step_tokens in (EP_SIZE * TOKENS_PER_RANK, 1):
            with self.subTest(num_step_tokens=num_step_tokens):
                out = _xpu_masked_grouped_experts(
                    hidden_states=self.hidden.clone(),
                    masked_m=self.masked_m,
                    num_topk=TOPK,
                    num_step_tokens=num_step_tokens,
                    w13=self.w13,
                    w2=self.w2,
                    activation_type=0,
                )
                self._assert_matches_reference(out)

    def test_bmm_fallback_matches_per_expert_reference(self):
        layer = SimpleNamespace(
            xpu_grouped_activation_type=None,
            w13_weight=self.w13,
            w2_weight=self.w2,
            xpu_masked_activation=_xpu_masked_activation("silu", True),
            moe_ep_size=EP_SIZE,
        )
        topk_ids = torch.zeros((TOKENS_PER_RANK, TOPK), dtype=torch.int64, device="xpu")
        out = DeepEPMoE._forward_masked_xpu(
            layer, (self.hidden.clone(), None, topk_ids, None, self.masked_m, None)
        )
        self._assert_matches_reference(out)


class TestActiveRanksRing(CustomTestCase):
    def test_rows_keep_their_storage_and_reset_between_uses(self):
        # The receive hook reads a row after dispatch returns, and a timeout zeroes
        # an entry; the next call that gets the same row must see every rank active.
        ring = _ActiveRanksRing(num_ranks=4, device="cpu", num_rows=3)
        storage = []
        for cycle in range(3):
            for i in range(3):
                row = ring.take()
                if cycle == 0:
                    storage.append(row.data_ptr())
                self.assertEqual(row.data_ptr(), storage[i])
                self.assertTrue(bool((row == 1).all()), (cycle, i))
                row[i] = 0


if __name__ == "__main__":
    unittest.main()
