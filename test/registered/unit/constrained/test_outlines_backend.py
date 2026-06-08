"""
Unit tests for sglang.srt.constrained.outlines_backend.

Test Coverage:
- OutlinesGrammar: rollback state history, dense bool vocab masks, copy
  semantics, and logit mask application.

Usage:
    python -m pytest test_outlines_backend.py -v
"""

import unittest
from dataclasses import dataclass

import torch

from sglang.srt.constrained.outlines_backend import OutlinesGrammar
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


@dataclass
class _Instruction:
    tokens: list[int]


class _Guide:
    def __init__(self):
        # State machine:
        #        10        11       12
        #   0 --------> 1 -----> 3 ----->
        #   |    20        21
        #   | --------> 2 ----->
        # Allowed next tokens:
        #   state 0: 10, 20
        #   state 1: 11
        #   state 2: 21
        #   state 3: 12
        self.allowed_tokens = {
            0: [10, 20],
            1: [11],
            2: [21],
            3: [12],
        }
        self.transitions = {
            (0, 10): 1,
            (0, 20): 2,
            (1, 11): 3,
        }

    def get_next_instruction(self, state):
        return _Instruction(self.allowed_tokens.get(state, []))

    def get_next_state(self, state, token):
        return self.transitions.get((state, token), -1)


class TestOutlinesGrammarRollback(unittest.TestCase):
    def test_rollback_and_restore(self):
        grammar = OutlinesGrammar(_Guide(), None)
        mask = grammar.allocate_vocab_mask(vocab_size=32, batch_size=3, device="cpu")

        grammar.fill_vocab_mask(mask, 0)
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[0], 10))
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[0], 20))
        self.assertFalse(grammar.is_vocab_mask_allowed_token(mask[0], 11))

        grammar.accept_token(10)
        self.assertEqual(grammar.state, 1)
        grammar.fill_vocab_mask(mask, 1)
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[1], 11))
        self.assertFalse(grammar.is_vocab_mask_allowed_token(mask[1], 21))

        grammar.rollback(1)
        self.assertEqual(grammar.state, 0)

        grammar.accept_token(20)
        self.assertEqual(grammar.state, 2)
        grammar.fill_vocab_mask(mask, 2)
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[2], 21))
        self.assertFalse(grammar.is_vocab_mask_allowed_token(mask[2], 11))

    def test_rollback_multiple_tokens_and_bounds(self):
        grammar = OutlinesGrammar(_Guide(), None)
        grammar.accept_token(10)
        grammar.accept_token(11)
        self.assertEqual(grammar.state, 3)

        grammar.rollback(0)
        self.assertEqual(grammar.state, 3)

        grammar.rollback(1)
        self.assertEqual(grammar.state, 1)

        grammar.rollback(1)
        self.assertEqual(grammar.state, 0)

        with self.assertRaisesRegex(ValueError, "Cannot rollback 1 tokens"):
            grammar.rollback(1)

    def test_rollback_multiple_tokens_at_once(self):
        grammar = OutlinesGrammar(_Guide(), None)
        grammar.accept_token(10)
        grammar.accept_token(11)

        grammar.rollback(2)

        self.assertEqual(grammar.state, 0)


class TestOutlinesGrammarApis(unittest.TestCase):
    def test_dense_bool_vocab_mask_and_apply_mask(self):
        grammar = OutlinesGrammar(_Guide(), None)
        mask = grammar.allocate_vocab_mask(vocab_size=32, batch_size=1, device="cpu")

        self.assertEqual(mask.shape, (1, 32))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask.any().item())
        self.assertIs(grammar.move_vocab_mask(mask, "cpu"), mask)

        grammar.fill_vocab_mask(mask, 0)
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[0], 10))
        self.assertTrue(grammar.is_vocab_mask_allowed_token(mask[0], 20))
        self.assertFalse(grammar.is_vocab_mask_allowed_token(mask[0], 11))
        self.assertFalse(grammar.is_vocab_mask_allowed_token(mask[0], 32))
        self.assertFalse(
            grammar.is_vocab_mask_allowed_token(mask[0], 20, vocab_size=20)
        )

        logits = torch.zeros(1, 32)
        grammar.apply_vocab_mask(logits, mask)
        self.assertEqual(logits[0, 10].item(), 0.0)
        self.assertEqual(logits[0, 20].item(), 0.0)
        self.assertTrue(torch.isneginf(logits[0, 11]).item())

    def test_copy_returns_fresh_grammar(self):
        guide = _Guide()
        grammar = OutlinesGrammar(guide, None)
        grammar.accept_token(10)
        self.assertEqual(grammar.state, 1)

        copied = grammar.copy()

        self.assertIs(copied.guide, guide)
        self.assertEqual(copied.state, 0)
        with self.assertRaisesRegex(ValueError, "Cannot rollback 1 tokens"):
            copied.rollback(1)


if __name__ == "__main__":
    unittest.main()
