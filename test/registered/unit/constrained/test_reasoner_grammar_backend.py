import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarBackend
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.srt.constrained.torch_ops.token_filter_torch_ops import (
    set_token_filter_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


class _DummyTokenizer:
    def __init__(self, token_map):
        self._token_map = token_map

    def encode(self, text, add_special_tokens=False):
        return list(self._token_map.get(text, []))


class _DummyGrammarBackend(BaseGrammarBackend):
    def __init__(self, support_token_filter=True):
        super().__init__()
        self._support_token_filter = support_token_filter
        self._dispatch_result = None

    @property
    def is_support_token_filter(self):
        return self._support_token_filter

    @staticmethod
    def allocate_vocab_mask(vocab_size, batch_size, device):
        return torch.zeros((batch_size, (vocab_size + 31) // 32), dtype=torch.int32)

    @staticmethod
    def move_vocab_mask(vocab_mask, device):
        return vocab_mask

    @staticmethod
    def apply_vocab_mask(logits, vocab_mask):
        return None

    @staticmethod
    def set_token_filter(
        vocab_mask, token_ids, batch_idx, is_allowed=True, reset_vocab_mask=True
    ):
        set_token_filter_torch(
            vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
        )

    def _init_value_dispatch(self, key, reasoning):
        return self._dispatch_result


def _allowed_token_ids(vocab_mask, token_ids):
    allowed = []
    for token_id in token_ids:
        elem = token_id // 32
        bit = token_id % 32
        if int(vocab_mask[0, elem].item()) & (1 << bit):
            allowed.append(token_id)
    return allowed


class TestReasonerGrammarObject(unittest.TestCase):
    def _make_strict_object(self):
        return ReasonerGrammarObject(
            grammar=None,
            think_end_id=7,
            think_excluded_token_ids=[3, 5],
            max_think_tokens=2,
            enable_token_filter=True,
            token_filter_fn=set_token_filter_torch,
            allocate_vocab_mask_fn=lambda vocab_size, batch_size, device: torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            ),
            move_vocab_mask_fn=lambda vocab_mask, device: vocab_mask,
            apply_vocab_mask_fn=lambda logits, vocab_mask: None,
        )

    def test_strict_thinking_phase_excludes_configured_tokens(self):
        obj = self._make_strict_object()
        obj.maybe_init_reasoning(True)
        mask = obj.allocate_vocab_mask(64, 1, "cpu")

        obj.fill_vocab_mask(mask, 0)

        allowed = _allowed_token_ids(mask, [0, 1, 3, 5, 7, 8])
        self.assertEqual(allowed, [0, 1, 7, 8])

    def test_budget_exhaustion_allows_only_think_end(self):
        obj = self._make_strict_object()
        obj.maybe_init_reasoning(True)
        obj.accept_token(10)
        obj.accept_token(11)
        mask = obj.allocate_vocab_mask(64, 1, "cpu")

        obj.fill_vocab_mask(mask, 0)

        allowed = _allowed_token_ids(mask, [0, 1, 3, 5, 7, 8, 10, 11])
        self.assertEqual(allowed, [7])

    def test_strict_only_wrapper_exposes_backend_mask_hooks(self):
        obj = self._make_strict_object()
        mask = obj.allocate_vocab_mask(64, 2, "cpu")

        self.assertEqual(mask.shape, (2, 2))
        self.assertIs(obj.move_vocab_mask(mask, "cpu"), mask)
        self.assertIsNotNone(obj.apply_vocab_mask)


class TestReasonerGrammarBackend(unittest.TestCase):
    def setUp(self):
        self._prev_budget = os.environ.get("SGLANG_MAX_THINK_TOKENS")

    def tearDown(self):
        if self._prev_budget is None:
            os.environ.pop("SGLANG_MAX_THINK_TOKENS", None)
        else:
            os.environ["SGLANG_MAX_THINK_TOKENS"] = self._prev_budget

    def _make_parser(self):
        detector = SimpleNamespace(
            think_start_token="<think>",
            think_end_token="</think>",
            think_excluded_tokens=["<tool_call>", "</tool_call>"],
        )
        return SimpleNamespace(detector=detector)

    def _make_tokenizer(self, start_ids=None, end_ids=None):
        return _DummyTokenizer(
            {
                "<think>": [1] if start_ids is None else start_ids,
                "</think>": [2] if end_ids is None else end_ids,
                "<tool_call>": [3],
                "</tool_call>": [4],
            }
        )

    def test_init_strict_reasoning_grammar_uses_token_filter_and_budget(self):
        os.environ["SGLANG_MAX_THINK_TOKENS"] = "2"
        backend = _DummyGrammarBackend(support_token_filter=True)
        reasoner = ReasonerGrammarBackend(
            backend,
            self._make_parser(),
            self._make_tokenizer(),
            enable_strict_thinking=True,
        )

        obj = reasoner.init_strict_reasoning_grammar(reasoning=True)

        self.assertIsInstance(obj, ReasonerGrammarObject)
        self.assertTrue(obj.enable_token_filter)
        self.assertEqual(obj.max_think_tokens, 2)
        self.assertEqual(obj.think_excluded_token_ids, [3, 4])

    def test_init_strict_reasoning_grammar_none_when_strict_disabled(self):
        backend = _DummyGrammarBackend(support_token_filter=True)
        reasoner = ReasonerGrammarBackend(
            backend,
            self._make_parser(),
            self._make_tokenizer(),
            enable_strict_thinking=False,
        )

        self.assertIsNone(reasoner.init_strict_reasoning_grammar(reasoning=True))

    def test_wraps_inner_grammar_with_reasoning_state_machine(self):
        os.environ["SGLANG_MAX_THINK_TOKENS"] = "1"
        backend = _DummyGrammarBackend(support_token_filter=True)
        inner_grammar = MagicMock()
        backend._dispatch_result = inner_grammar
        reasoner = ReasonerGrammarBackend(
            backend,
            self._make_parser(),
            self._make_tokenizer(),
            enable_strict_thinking=True,
        )

        wrapped = reasoner._init_value_dispatch(("json", "{}"), reasoning=True)
        self.assertIsInstance(wrapped, ReasonerGrammarObject)
        wrapped.accept_token(10)
        inner_grammar.accept_token.assert_not_called()
        wrapped.accept_token(2)
        wrapped.accept_token(42)
        inner_grammar.accept_token.assert_called_once_with(42)

    def test_accepts_multi_token_think_start_marker(self):
        """think_start_token can be multi-token (e.g., GPT-OSS) since it's not used."""
        backend = _DummyGrammarBackend(support_token_filter=True)
        reasoner = ReasonerGrammarBackend(
            backend,
            self._make_parser(),
            self._make_tokenizer(start_ids=[1, 2]),
            enable_strict_thinking=True,
        )
        self.assertIsNotNone(reasoner)

    def test_rejects_multi_token_think_end_marker(self):
        backend = _DummyGrammarBackend(support_token_filter=True)

        with self.assertRaisesRegex(ValueError, "must encode to exactly one token"):
            ReasonerGrammarBackend(
                backend,
                self._make_parser(),
                self._make_tokenizer(end_ids=[2, 3]),
                enable_strict_thinking=True,
            )

    def test_rejects_unencodable_excluded_token(self):
        backend = _DummyGrammarBackend(support_token_filter=True)
        parser = self._make_parser()
        parser.detector.think_excluded_tokens = ["<unknown>"]
        tokenizer = _DummyTokenizer(
            {
                "<think>": [1],
                "</think>": [2],
            }
        )

        with self.assertRaisesRegex(ValueError, "could not be encoded"):
            ReasonerGrammarBackend(
                backend,
                parser,
                tokenizer,
                enable_strict_thinking=True,
            )

    def test_strict_mode_fails_when_backend_lacks_token_filter(self):
        backend = _DummyGrammarBackend(support_token_filter=False)

        with self.assertRaisesRegex(ValueError, "does not support token filtering"):
            ReasonerGrammarBackend(
                backend,
                self._make_parser(),
                self._make_tokenizer(),
                enable_strict_thinking=True,
            )


class TestReasonerGrammarObjectRollback(unittest.TestCase):
    """Tests for rollback correctness at the THINKING→GENERATION boundary."""

    def _make_object_with_mock_grammar(self):
        inner_grammar = MagicMock()
        inner_grammar.is_terminated.return_value = False
        obj = ReasonerGrammarObject(
            grammar=inner_grammar,
            think_end_id=7,
            think_excluded_token_ids=[3, 5],
            max_think_tokens=-1,
            enable_token_filter=True,
            token_filter_fn=set_token_filter_torch,
            allocate_vocab_mask_fn=lambda vs, bs, d: torch.zeros(
                (bs, (vs + 31) // 32), dtype=torch.int32
            ),
            move_vocab_mask_fn=lambda vm, d: vm,
            apply_vocab_mask_fn=lambda l, vm: None,
        )
        return obj, inner_grammar

    def test_rollback_at_generation_boundary_returns_to_thinking(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        # Accept 3 thinking tokens then think_end_id
        obj.accept_token(10)
        obj.accept_token(11)
        obj.accept_token(12)
        obj.accept_token(7)  # think_end_id → tokens_after_end = 0

        self.assertTrue(obj._is_generation())
        self.assertEqual(obj.tokens_after_end, 0)

        # Rollback 1 step: should return to THINKING
        obj.rollback(1)
        self.assertTrue(obj._is_thinking())
        self.assertEqual(obj.tokens_in_think, 3)
        self.assertEqual(obj.tokens_after_end, -1)
        # Grammar should not have been rolled back (no generation tokens were accepted)
        inner_grammar.rollback.assert_not_called()

    def test_rollback_spanning_both_phases(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        # 2 thinking tokens + think_end + 3 generation tokens
        obj.accept_token(10)  # think
        obj.accept_token(11)  # think
        obj.accept_token(7)  # think_end_id
        obj.accept_token(20)  # gen 1
        obj.accept_token(21)  # gen 2
        obj.accept_token(22)  # gen 3

        self.assertEqual(obj.tokens_after_end, 3)

        # Rollback 5: should roll back 3 generation tokens + think_end + 1 thinking token
        obj.rollback(5)
        self.assertTrue(obj._is_thinking())
        self.assertEqual(obj.tokens_in_think, 1)
        # Grammar should be rolled back by 3 (only generation tokens)
        inner_grammar.rollback.assert_called_once_with(3)

    def test_rollback_generation_tokens_only(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        obj.accept_token(10)  # think
        obj.accept_token(7)  # think_end_id
        obj.accept_token(20)  # gen 1
        obj.accept_token(21)  # gen 2

        # Rollback 1: should only roll back 1 generation token
        obj.rollback(1)
        self.assertTrue(obj._is_generation())
        self.assertEqual(obj.tokens_after_end, 1)
        inner_grammar.rollback.assert_called_once_with(1)

    def test_rollback_thinking_tokens_does_not_touch_grammar(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        obj.accept_token(10)
        obj.accept_token(11)
        obj.accept_token(12)

        obj.rollback(2)
        self.assertTrue(obj._is_thinking())
        self.assertEqual(obj.tokens_in_think, 1)
        inner_grammar.rollback.assert_not_called()
        inner_grammar.accept_token.assert_not_called()

    def test_copy_preserves_state(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        obj.accept_token(10)
        obj.accept_token(7)  # think_end_id → GENERATION
        obj.accept_token(20)

        self.assertEqual(obj.tokens_in_think, 1)
        self.assertEqual(obj.tokens_after_end, 1)

        copy = obj.copy()
        # State counters must be preserved for speculative decoding
        self.assertEqual(copy.tokens_in_think, 1)
        self.assertEqual(copy.tokens_after_end, 1)
        self.assertTrue(copy._is_generation())
        self.assertIsNotNone(copy.grammar)
        inner_grammar.copy.assert_called_once()

    def test_copy_preserves_thinking_state(self):
        obj, inner_grammar = self._make_object_with_mock_grammar()
        obj.maybe_init_reasoning(True)

        obj.accept_token(10)
        obj.accept_token(11)

        copy = obj.copy()
        self.assertEqual(copy.tokens_in_think, 2)
        self.assertEqual(copy.tokens_after_end, -1)
        self.assertTrue(copy._is_thinking())


class TestReasonerGrammarObjectFillVocabMask(unittest.TestCase):
    """Tests for fill_vocab_mask behavior in different states."""

    def test_thinking_phase_does_not_consult_inner_grammar(self):
        inner_grammar = MagicMock()
        # Must return a real tensor for allocate_vocab_mask since fill_vocab_mask
        # delegates to allocate_vocab_mask via self.grammar when grammar is not None
        inner_grammar.allocate_vocab_mask.side_effect = lambda vs, bs, d: torch.zeros(
            (bs, (vs + 31) // 32), dtype=torch.int32
        )
        obj = ReasonerGrammarObject(
            grammar=inner_grammar,
            think_end_id=7,
            think_excluded_token_ids=[3, 5],
            max_think_tokens=-1,
            enable_token_filter=True,
            token_filter_fn=set_token_filter_torch,
            allocate_vocab_mask_fn=lambda vs, bs, d: torch.zeros(
                (bs, (vs + 31) // 32), dtype=torch.int32
            ),
            move_vocab_mask_fn=lambda vm, d: vm,
            apply_vocab_mask_fn=lambda l, vm: None,
        )
        obj.maybe_init_reasoning(True)
        mask = obj.allocate_vocab_mask(64, 1, "cpu")

        obj.fill_vocab_mask(mask, 0)

        inner_grammar.fill_vocab_mask.assert_not_called()
        # Excluded tokens (3, 5) should be blocked
        allowed = _allowed_token_ids(mask, [0, 1, 3, 5, 7, 8])
        self.assertEqual(allowed, [0, 1, 7, 8])

    def test_generation_phase_consults_inner_grammar(self):
        inner_grammar = MagicMock()
        inner_grammar.allocate_vocab_mask.side_effect = lambda vs, bs, d: torch.zeros(
            (bs, (vs + 31) // 32), dtype=torch.int32
        )
        obj = ReasonerGrammarObject(
            grammar=inner_grammar,
            think_end_id=7,
            think_excluded_token_ids=[3, 5],
            max_think_tokens=-1,
            enable_token_filter=True,
            token_filter_fn=set_token_filter_torch,
            allocate_vocab_mask_fn=lambda vs, bs, d: torch.zeros(
                (bs, (vs + 31) // 32), dtype=torch.int32
            ),
            move_vocab_mask_fn=lambda vm, d: vm,
            apply_vocab_mask_fn=lambda l, vm: None,
        )
        obj.maybe_init_reasoning(True)
        obj.accept_token(10)
        obj.accept_token(7)  # think_end_id → GENERATION

        mask = obj.allocate_vocab_mask(64, 1, "cpu")
        obj.fill_vocab_mask(mask, 0)

        inner_grammar.fill_vocab_mask.assert_called_once_with(mask, 0)

    def test_non_strict_thinking_is_noop(self):
        inner_grammar = MagicMock()
        obj = ReasonerGrammarObject(
            grammar=inner_grammar,
            think_end_id=7,
            think_excluded_token_ids=None,
            max_think_tokens=-1,
            enable_token_filter=False,
            token_filter_fn=None,
        )
        obj.maybe_init_reasoning(True)
        mask = torch.zeros((1, 2), dtype=torch.int32)

        obj.fill_vocab_mask(mask, 0)

        inner_grammar.fill_vocab_mask.assert_not_called()
        # Mask should remain all zeros (no filtering)
        self.assertTrue(torch.all(mask == 0))


if __name__ == "__main__":
    unittest.main()
