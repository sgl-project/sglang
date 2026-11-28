import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.dflash import candidate_topk
from sglang.srt.speculative.dflash_cohort import DFlashCohort, DraftHeadView
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CohortContractTests(CustomTestCase):
    def make(self, values):
        obj = object.__new__(DFlashCohort)
        obj.attn_tp_size = 2
        obj.sequence = 0
        obj.owner_slots = 2
        obj.group = SimpleNamespace(
            all_gather_object=lambda value: [
                dict(phase="PLAN", sequence=1, value=x) for x in values
            ]
        )
        return obj

    def test_attention_peers_are_one_owner_and_owner_slots_do_not_alias(self):
        a = {"slots": [1], "generations": [1], "ids": ["a"], "prefixes": [10]}
        b = {"slots": [1], "generations": [4], "ids": ["b"], "prefixes": [30]}
        obj = self.make([a, a, b, b])
        owners = obj._exchange("PLAN", a)
        self.assertEqual(owners, [a, b])
        self.assertEqual(obj._slots(0, a), [1])
        self.assertEqual(obj._slots(1, b), [3])

    def test_owner_peer_disagreement_is_rejected(self):
        a = {"slots": [1], "generations": [1]}
        b = {"slots": [1], "generations": [2]}
        with self.assertRaisesRegex(RuntimeError, "attention-TP peers"):
            self.make([a, b])._exchange("PLAN", a)

    def test_phase_disagreement_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "phase mismatch"):
            self.make([{}])._exchange("COMMIT", {})

    def test_target_only_tail_does_not_evaluate_a_draft_block(self):
        obj = object.__new__(DFlashCohort)
        obj.owner_rank = 0
        obj.owner_slots = 1
        obj.pending = None
        active = dict(
            slots=[1],
            generations=[4],
            ids=["a"],
            prefixes=[7],
            bonuses=[19],
            greedy=True,
        )
        idle = dict(
            slots=[], generations=[], ids=[], prefixes=[], bonuses=[], greedy=True
        )
        obj._descriptor = lambda batch, prefixes: dict(active)
        obj._exchange = lambda phase, value: [active, idle]
        obj.committed = {1: (4, "a", 7)}
        cache = SimpleNamespace(
            layout=SimpleNamespace(context_length=8),
            locations=lambda slots, positions, lengths: (
                positions + 64,
                positions + 128,
                None,
            ),
        )
        obj.worker = SimpleNamespace(
            device="cpu", block_size=12, _independent_cache=cache
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            spec_info=SimpleNamespace(bonus_tokens=torch.tensor([19])),
        )
        tokens = obj.propose(batch, prefix_lens=torch.tensor([7]), target_only=True)
        torch.testing.assert_close(tokens, torch.tensor([[19]]))
        self.assertEqual(obj.pending["block"], 1)
        self.assertEqual(obj.pending["positions"].tolist(), [7])
        active["prefixes"] = [8]
        obj.committed = {1: (4, "a", 8)}
        with self.assertRaisesRegex(RuntimeError, "context boundary"):
            obj.propose(batch, prefix_lens=torch.tensor([8]), target_only=True)

    def test_replicated_head_ignores_ambient_tp_and_padding(self):
        torch.manual_seed(7)
        weight = torch.randn(17, 8)
        weight[13:] = 10000
        head = DraftHeadView(weight, 13)
        hidden = torch.randn(3, 8)
        backend = SimpleNamespace(is_cutedsl=lambda: False, is_gemv=lambda: False)
        with (
            patch(
                "sglang.srt.models.dflash.get_parallel",
                side_effect=AssertionError("ambient group leaked"),
            ),
            patch(
                "sglang.srt.models.dflash.get_bf16_gemm_backend", return_value=backend
            ),
            patch(
                "sglang.srt.models.dflash.radix_topk",
                side_effect=lambda logits, k: torch.topk(logits, k, dim=-1),
            ),
        ):
            ids, values, _ = candidate_topk(hidden, head, 4)
        expected_values, expected_ids = torch.topk(hidden @ weight[:13].T, 4, dim=-1)
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(values, expected_values)


if __name__ == "__main__":
    unittest.main()
