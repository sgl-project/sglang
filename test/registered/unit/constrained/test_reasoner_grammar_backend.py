"""
Unit tests for sglang.srt.constrained.reasoner_grammar_backend.

Test Coverage:
- ReasonerGrammarObject: state transitions, accept_token during thinking
  vs post-thinking, rollback across think boundary, fill_vocab_mask gating,
  copy semantics, finished delegation, delegation of jump methods
- ReasonerGrammarBackend: dispatch wrapping, invalid grammar passthrough,
  None grammar passthrough, reasoning init on wrapped object

Usage:
    python -m pytest test_reasoner_grammar_backend.py -v
"""

import unittest
from unittest.mock import MagicMock, call

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-cpu-only")

THINK_END_ID = 99


class TestReasonerGrammarObjectStateTransitions(unittest.TestCase):
    """Test thinking state machine in ReasonerGrammarObject."""

    def _make(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        return ReasonerGrammarObject(grammar, THINK_END_ID), grammar

    def test_initial_state_thinking(self):
        obj, _ = self._make()
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_transfer_state_during_thinking(self):
        """Regular tokens during thinking don't change state."""
        obj, _ = self._make()
        obj.transfer_state(10)
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_transfer_state_think_end_token(self):
        """Think end token transitions from -1 to 0."""
        obj, _ = self._make()
        obj.transfer_state(THINK_END_ID)
        self.assertEqual(obj.tokens_after_think_end, 0)

    def test_transfer_state_increments_after_thinking(self):
        """After thinking ends, each token increments counter."""
        obj, _ = self._make()
        obj.tokens_after_think_end = 0
        obj.transfer_state(10)
        self.assertEqual(obj.tokens_after_think_end, 1)
        obj.transfer_state(20)
        self.assertEqual(obj.tokens_after_think_end, 2)

    def test_think_end_after_thinking_already_ended(self):
        """Second think_end_id after thinking ended just increments."""
        obj, _ = self._make()
        obj.tokens_after_think_end = 3
        obj.transfer_state(THINK_END_ID)
        self.assertEqual(obj.tokens_after_think_end, 4)

    def test_rollback_state_from_post_thinking(self):
        obj, _ = self._make()
        obj.tokens_after_think_end = 3
        obj.rollback_state()
        self.assertEqual(obj.tokens_after_think_end, 2)

    def test_rollback_state_at_boundary(self):
        """Rollback from 0 goes back to -1 (thinking)."""
        obj, _ = self._make()
        obj.tokens_after_think_end = 0
        obj.rollback_state()
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_rollback_state_during_thinking(self):
        """Rollback during thinking stays at -1."""
        obj, _ = self._make()
        obj.rollback_state()
        self.assertEqual(obj.tokens_after_think_end, -1)


class TestReasonerGrammarObjectAcceptToken(unittest.TestCase):
    """Test accept_token behavior with thinking/post-thinking states."""

    def _make(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        return ReasonerGrammarObject(grammar, THINK_END_ID), grammar

    def test_accept_during_thinking_skips_grammar(self):
        """During thinking phase, inner grammar should NOT receive tokens."""
        obj, grammar = self._make()
        obj.accept_token(10)
        grammar.accept_token.assert_not_called()
        # State should still be -1
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_accept_think_end_token(self):
        """Think end token transitions state but doesn't call inner grammar (state was -1 before transfer)."""
        obj, grammar = self._make()
        # tokens_after_think_end is -1, so grammar.accept_token is not called
        # But wait: accept_token checks `>= 0` BEFORE transfer_state
        # At call time tokens_after_think_end == -1, so grammar.accept_token skipped
        obj.accept_token(THINK_END_ID)
        grammar.accept_token.assert_not_called()
        self.assertEqual(obj.tokens_after_think_end, 0)

    def test_accept_after_thinking_calls_grammar(self):
        """After thinking ends, tokens go to inner grammar."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 0
        obj.accept_token(42)
        grammar.accept_token.assert_called_once_with(42)
        self.assertEqual(obj.tokens_after_think_end, 1)

    def test_accept_sequence_through_thinking_and_generation(self):
        """Full sequence: think tokens -> think_end -> generation tokens."""
        obj, grammar = self._make()

        # Thinking phase
        obj.accept_token(1)
        obj.accept_token(2)
        self.assertEqual(grammar.accept_token.call_count, 0)

        # Think end
        obj.accept_token(THINK_END_ID)
        self.assertEqual(grammar.accept_token.call_count, 0)

        # Generation phase
        obj.accept_token(10)
        obj.accept_token(20)
        self.assertEqual(grammar.accept_token.call_count, 2)
        grammar.accept_token.assert_has_calls([call(10), call(20)])


class TestReasonerGrammarObjectRollback(unittest.TestCase):
    """Test rollback across thinking boundary."""

    def _make(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        return ReasonerGrammarObject(grammar, THINK_END_ID), grammar

    def test_rollback_within_generation(self):
        """Rollback entirely within generation phase."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 5
        obj.rollback(3)
        grammar.rollback.assert_called_once_with(3)
        self.assertEqual(obj.tokens_after_think_end, 2)

    def test_rollback_across_boundary(self):
        """Rollback that crosses from generation back into thinking."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 2
        obj.rollback(4)
        # Only 2 tokens were post-thinking, so inner grammar rolls back 2
        grammar.rollback.assert_called_once_with(2)
        # After 4 rollback_state calls from 2: 2->1->0->-1->-1
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_rollback_during_thinking(self):
        """Rollback during thinking phase doesn't touch inner grammar."""
        obj, grammar = self._make()
        obj.rollback(3)
        grammar.rollback.assert_not_called()
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_rollback_zero(self):
        obj, grammar = self._make()
        obj.tokens_after_think_end = 2
        obj.rollback(0)
        grammar.rollback.assert_not_called()
        self.assertEqual(obj.tokens_after_think_end, 2)

    def test_rollback_exactly_to_boundary(self):
        """Rollback exactly the number of post-thinking tokens."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 3
        obj.rollback(3)
        grammar.rollback.assert_called_once_with(3)
        self.assertEqual(obj.tokens_after_think_end, 0)

    def test_rollback_far_beyond_all_tokens(self):
        """Rollback k much larger than tokens_after_think_end clamps grammar rollback."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 2
        obj.rollback(100)
        # Inner grammar only rolls back the 2 post-thinking tokens
        grammar.rollback.assert_called_once_with(2)
        # State bottoms out at -1
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_accept_then_rollback_roundtrip(self):
        """Accept tokens then rollback should restore original state."""
        obj, grammar = self._make()
        obj.tokens_after_think_end = 0  # Just finished thinking

        # Accept 3 generation tokens
        obj.accept_token(10)
        obj.accept_token(20)
        obj.accept_token(30)
        self.assertEqual(obj.tokens_after_think_end, 3)
        self.assertEqual(grammar.accept_token.call_count, 3)

        # Rollback all 3
        obj.rollback(3)
        self.assertEqual(obj.tokens_after_think_end, 0)
        grammar.rollback.assert_called_once_with(3)


class TestReasonerGrammarObjectVocabMask(unittest.TestCase):
    """Test vocab mask gating based on thinking state."""

    def _make(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        return ReasonerGrammarObject(grammar, THINK_END_ID), grammar

    def test_fill_during_thinking_skips(self):
        obj, grammar = self._make()
        obj.fill_vocab_mask("mask", 0)
        grammar.fill_vocab_mask.assert_not_called()

    def test_fill_after_thinking_delegates(self):
        obj, grammar = self._make()
        obj.tokens_after_think_end = 0
        obj.fill_vocab_mask("mask", 0)
        grammar.fill_vocab_mask.assert_called_once_with("mask", 0)

    def test_fill_well_into_generation(self):
        obj, grammar = self._make()
        obj.tokens_after_think_end = 5
        obj.fill_vocab_mask("mask", 2)
        grammar.fill_vocab_mask.assert_called_once_with("mask", 2)

    def test_fill_at_think_end_boundary(self):
        """After accepting think_end token, fill_vocab_mask should delegate."""
        obj, grammar = self._make()
        # Simulate: accept think_end, state goes from -1 to 0
        obj.accept_token(THINK_END_ID)
        self.assertEqual(obj.tokens_after_think_end, 0)
        obj.fill_vocab_mask("mask", 0)
        grammar.fill_vocab_mask.assert_called_once_with("mask", 0)

    def test_allocate_delegates(self):
        obj, grammar = self._make()
        obj.allocate_vocab_mask(32000, 4, "cpu")
        grammar.allocate_vocab_mask.assert_called_once_with(32000, 4, "cpu")

    def test_move_delegates(self):
        obj, grammar = self._make()
        obj.move_vocab_mask("mask", "cuda")
        grammar.move_vocab_mask.assert_called_once_with("mask", "cuda")


class TestReasonerGrammarObjectDelegation(unittest.TestCase):
    """Test that non-state methods delegate to inner grammar."""

    def _make(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        return ReasonerGrammarObject(grammar, THINK_END_ID), grammar

    def test_is_terminated_delegates(self):
        obj, grammar = self._make()
        grammar.is_terminated.return_value = True
        self.assertTrue(obj.is_terminated())

    def test_finished_getter_delegates(self):
        obj, grammar = self._make()
        grammar.finished = True
        self.assertTrue(obj.finished)

    def test_finished_setter_delegates(self):
        obj, grammar = self._make()
        obj.finished = True
        self.assertTrue(grammar.finished)

    def test_try_jump_forward_delegates(self):
        obj, grammar = self._make()
        grammar.try_jump_forward.return_value = ([1, 2], "ab")
        result = obj.try_jump_forward("tokenizer")
        grammar.try_jump_forward.assert_called_once_with("tokenizer")
        self.assertEqual(result, ([1, 2], "ab"))

    def test_jump_forward_str_state_delegates(self):
        obj, grammar = self._make()
        grammar.jump_forward_str_state.return_value = ("str", 5)
        result = obj.jump_forward_str_state("helper")
        self.assertEqual(result, ("str", 5))

    def test_jump_and_retokenize_delegates(self):
        obj, grammar = self._make()
        obj.jump_and_retokenize([1], [2], 3)
        grammar.jump_and_retokenize.assert_called_once_with([1], [2], 3)

    def test_apply_vocab_mask_property(self):
        obj, grammar = self._make()
        grammar.apply_vocab_mask = "mask_fn"
        self.assertEqual(obj.apply_vocab_mask, "mask_fn")

    def test_copy_creates_new_wrapper(self):
        obj, grammar = self._make()
        grammar_copy = MagicMock(spec=BaseGrammarObject)
        grammar.copy.return_value = grammar_copy

        copied = obj.copy()
        self.assertIsInstance(copied, ReasonerGrammarObject)
        self.assertIsNot(copied, obj)
        self.assertIs(copied.grammar, grammar_copy)
        self.assertEqual(copied.think_end_id, THINK_END_ID)

    def test_copy_does_not_share_state(self):
        """Modifying copy's state should not affect the original."""
        obj, grammar = self._make()
        grammar_copy = MagicMock(spec=BaseGrammarObject)
        grammar.copy.return_value = grammar_copy

        copied = obj.copy()
        copied.tokens_after_think_end = 5
        self.assertEqual(obj.tokens_after_think_end, -1)


class TestReasonerGrammarObjectMaybeInitReasoning(unittest.TestCase):
    """Test maybe_init_reasoning state initialization."""

    def test_reasoning_true_sets_thinking(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        obj = ReasonerGrammarObject(grammar, THINK_END_ID)
        obj.maybe_init_reasoning(True)
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_reasoning_false_skips_thinking(self):
        grammar = MagicMock(spec=BaseGrammarObject)
        obj = ReasonerGrammarObject(grammar, THINK_END_ID)
        obj.maybe_init_reasoning(False)
        self.assertEqual(obj.tokens_after_think_end, 0)

    def test_reasoning_toggle(self):
        """Toggling reasoning resets state regardless of current position."""
        grammar = MagicMock(spec=BaseGrammarObject)
        obj = ReasonerGrammarObject(grammar, THINK_END_ID)
        obj.tokens_after_think_end = 5  # Deep into generation

        obj.maybe_init_reasoning(True)
        self.assertEqual(obj.tokens_after_think_end, -1)

        obj.maybe_init_reasoning(False)
        self.assertEqual(obj.tokens_after_think_end, 0)


class TestReasonerGrammarBackend(unittest.TestCase):
    """Test ReasonerGrammarBackend dispatch wrapping."""

    def _make(self):
        inner = MagicMock(spec=BaseGrammarBackend)
        backend = ReasonerGrammarBackend(inner, THINK_END_ID)
        return backend, inner

    def test_wraps_valid_grammar(self):
        backend, inner = self._make()
        mock_grammar = MagicMock(spec=BaseGrammarObject)
        inner._init_value_dispatch.return_value = mock_grammar

        result = backend._init_value_dispatch(("json", "schema"), True)
        self.assertIsInstance(result, ReasonerGrammarObject)
        self.assertIs(result.grammar, mock_grammar)
        self.assertEqual(result.think_end_id, THINK_END_ID)

    def test_passes_through_invalid_grammar(self):
        backend, inner = self._make()
        invalid = InvalidGrammarObject("bad grammar")
        inner._init_value_dispatch.return_value = invalid

        result = backend._init_value_dispatch(("json", "schema"), False)
        self.assertIs(result, invalid)
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_passes_through_none(self):
        backend, inner = self._make()
        inner._init_value_dispatch.return_value = None

        result = backend._init_value_dispatch(("json", "schema"), False)
        self.assertIsNone(result)

    def test_inits_reasoning_on_wrapped(self):
        backend, inner = self._make()
        mock_grammar = MagicMock(spec=BaseGrammarObject)
        inner._init_value_dispatch.return_value = mock_grammar

        result = backend._init_value_dispatch(("json", "schema"), True)
        # reasoning=True → tokens_after_think_end should be -1
        self.assertEqual(result.tokens_after_think_end, -1)

    def test_inits_no_reasoning_on_wrapped(self):
        backend, inner = self._make()
        mock_grammar = MagicMock(spec=BaseGrammarObject)
        inner._init_value_dispatch.return_value = mock_grammar

        result = backend._init_value_dispatch(("json", "schema"), False)
        # reasoning=False → tokens_after_think_end should be 0
        self.assertEqual(result.tokens_after_think_end, 0)


if __name__ == "__main__":
    unittest.main()
