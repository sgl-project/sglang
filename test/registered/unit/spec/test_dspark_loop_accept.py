"""DSpark greedy accept vs a repeating n-gram (no model, no Hopper).

γ=5 drafts sit in candidates[:, 1:]. Accept is consecutive equality
``candidates[:, 1:] == target_argmax[:, :-1]``. If the target's argmax is
the loop token, the whole block commits — that is the amplifier. If the
loop logit is penalised below the alternative, accept_len falls to 0.

This is one 6-token block. It is not 30 min of TG. Live soak with
per-turn spec_accept_length:
test/manual/dsv41_v100/test_tg_soak_cohesion.py
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GAMMA = 5
BLOCK = GAMMA + 1  # current token + 5 drafts (serve speculative_num_draft_tokens=6)
LOOP = 42
OTHER = 7
VOCAB = 128


def _logits_argmax_token(token: int, rows: int, vocab: int = VOCAB) -> torch.Tensor:
    """[rows, vocab] with a unique max at ``token``."""
    logits = torch.full((rows, vocab), -10.0)
    logits[:, token] = 5.0
    return logits


class TestDsparkLoopAccept(CustomTestCase):
    def test_full_block_commits_when_target_also_emits_the_loop(self):
        # current=LOOP, five LOOP drafts. Target argmax is LOOP at every slot.
        candidates = torch.full((1, BLOCK), LOOP, dtype=torch.int64)
        target = _logits_argmax_token(LOOP, BLOCK)
        correct_len, bonus, _trim = AcceptGreedy.torch(
            candidates=candidates,
            target_logits=target,
            verify_num_draft_tokens=BLOCK,
        )
        print(
            f"\nloop+target-loop: accept_drafts={int(correct_len[0])} "
            f"bonus={int(bonus[0])} gamma={GAMMA}"
        )
        self.assertEqual(int(correct_len[0]), GAMMA)
        self.assertEqual(int(bonus[0]), LOOP)

    def test_first_mismatch_stops_the_block(self):
        candidates = torch.full((1, BLOCK), LOOP, dtype=torch.int64)
        target = _logits_argmax_token(LOOP, BLOCK)
        target[0, LOOP] = -10.0
        target[0, OTHER] = 5.0  # first target step wants OTHER
        correct_len, bonus, _trim = AcceptGreedy.torch(
            candidates=candidates,
            target_logits=target,
            verify_num_draft_tokens=BLOCK,
        )
        print(
            f"\nloop+target-other@0: accept_drafts={int(correct_len[0])} "
            f"bonus={int(bonus[0])}"
        )
        self.assertEqual(int(correct_len[0]), 0)
        self.assertEqual(int(bonus[0]), OTHER)

    def test_simulated_repetition_penalty_cuts_accept_length(self):
        """repetition_penalty is applied to logits *before* accept. Lowering
        the loop token below OTHER is what 1.15 would do after many repeats.
        """
        candidates = torch.full((1, BLOCK), LOOP, dtype=torch.int64)
        target = _logits_argmax_token(LOOP, BLOCK)
        target[:, LOOP] = 0.0
        target[:, OTHER] = 1.0
        correct_len, bonus, _trim = AcceptGreedy.torch(
            candidates=candidates,
            target_logits=target,
            verify_num_draft_tokens=BLOCK,
        )
        print(
            f"\npenalised loop logit: accept_drafts={int(correct_len[0])} "
            f"bonus={int(bonus[0])}"
        )
        self.assertEqual(int(correct_len[0]), 0)
        self.assertEqual(int(bonus[0]), OTHER)


if __name__ == "__main__":
    unittest.main()
