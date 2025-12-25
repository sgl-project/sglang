# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for grammar termination handling in speculative decoding.

This test verifies that generate_token_bitmask correctly skips terminated
grammars, preventing the XGrammar RuntimeError:
"GrammarMatcher has terminated and no bitmask is required"

See: https://github.com/sgl-project/sglang/issues/15050
"""

import unittest

import torch
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject


class MockGrammarObject(BaseGrammarObject):
    """Mock grammar object for testing termination handling."""

    def __init__(self, is_terminated: bool = False):
        super().__init__()
        self._is_terminated = is_terminated
        self.fill_vocab_mask_called = False
        self.allocate_vocab_mask_called = False

    def is_terminated(self):
        return self._is_terminated

    def accept_token(self, token: int) -> None:
        pass

    def rollback(self, k: int):
        pass

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        self.allocate_vocab_mask_called = True
        # Return a properly shaped bitmask (vocab_size packed into 32-bit ints)
        num_int32_per_token = (vocab_size + 31) // 32
        return torch.zeros(batch_size, num_int32_per_token, dtype=torch.int32)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.fill_vocab_mask_called = True
        # This should NOT be called for terminated grammars
        if self._is_terminated:
            raise RuntimeError(
                "[XGrammar] The GrammarMatcher has terminated and "
                "no bitmask is required"
            )


class MockReq:
    """Mock request object for testing."""

    def __init__(self, grammar=None):
        self.grammar = grammar


class MockVerifyInput:
    """Mock verify input for testing."""

    def __init__(self):
        self.grammar = None


class TestSpeculativeGrammarTermination(unittest.TestCase):
    """Test suite for grammar termination handling in speculative decoding."""

    def test_generate_token_bitmask_skips_terminated_grammar(self):
        """Verify that terminated grammars are skipped in generate_token_bitmask.

        This is the core test for the fix in spec_utils.py:
        - Before fix: would call fill_vocab_mask on terminated grammar -> crash
        - After fix: checks is_terminated() and skips processing
        """
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        # Create a terminated grammar
        terminated_grammar = MockGrammarObject(is_terminated=True)
        req = MockReq(grammar=terminated_grammar)
        verify_input = MockVerifyInput()

        # Create test tensors (batch_size=1, num_draft_tokens=4)
        batch_size = 1
        num_draft_tokens = 4
        vocab_size = 32000

        retrieve_next_token_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        retrieve_next_sibling_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        draft_tokens_cpu = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int64
        )

        # This should NOT raise an exception with the fix
        result = generate_token_bitmask(
            reqs=[req],
            verify_input=verify_input,
            retrieve_next_token_cpu=retrieve_next_token_cpu,
            retrieve_next_sibling_cpu=retrieve_next_sibling_cpu,
            draft_tokens_cpu=draft_tokens_cpu,
            vocab_size=vocab_size,
        )

        # With terminated grammar, no bitmask should be allocated
        self.assertIsNone(result)
        # fill_vocab_mask should NOT have been called
        self.assertFalse(terminated_grammar.fill_vocab_mask_called)
        # allocate_vocab_mask should NOT have been called
        self.assertFalse(terminated_grammar.allocate_vocab_mask_called)

    def test_generate_token_bitmask_processes_active_grammar(self):
        """Verify that active (non-terminated) grammars are processed."""
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        # Create an active grammar
        active_grammar = MockGrammarObject(is_terminated=False)
        req = MockReq(grammar=active_grammar)
        verify_input = MockVerifyInput()

        # Create test tensors (batch_size=1, num_draft_tokens=4)
        batch_size = 1
        num_draft_tokens = 4
        vocab_size = 32000

        retrieve_next_token_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        retrieve_next_sibling_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        draft_tokens_cpu = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int64
        )

        # This should process the active grammar
        result = generate_token_bitmask(
            reqs=[req],
            verify_input=verify_input,
            retrieve_next_token_cpu=retrieve_next_token_cpu,
            retrieve_next_sibling_cpu=retrieve_next_sibling_cpu,
            draft_tokens_cpu=draft_tokens_cpu,
            vocab_size=vocab_size,
        )

        # Active grammar should have bitmask allocated
        self.assertIsNotNone(result)
        self.assertTrue(active_grammar.allocate_vocab_mask_called)
        # verify_input should have grammar set
        self.assertEqual(verify_input.grammar, active_grammar)

    def test_generate_token_bitmask_mixed_batch(self):
        """Test batch with both terminated and active grammars."""
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        # Create a batch with:
        # - req 0: terminated grammar (should be skipped)
        # - req 1: active grammar (should be processed)
        # - req 2: no grammar (should be skipped)
        terminated_grammar = MockGrammarObject(is_terminated=True)
        active_grammar = MockGrammarObject(is_terminated=False)

        reqs = [
            MockReq(grammar=terminated_grammar),
            MockReq(grammar=active_grammar),
            MockReq(grammar=None),
        ]
        verify_input = MockVerifyInput()

        batch_size = 3
        num_draft_tokens = 4
        vocab_size = 32000

        retrieve_next_token_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        retrieve_next_sibling_cpu = torch.full(
            (batch_size, num_draft_tokens), -1, dtype=torch.int64
        )
        draft_tokens_cpu = torch.zeros(
            (batch_size, num_draft_tokens), dtype=torch.int64
        )

        result = generate_token_bitmask(
            reqs=reqs,
            verify_input=verify_input,
            retrieve_next_token_cpu=retrieve_next_token_cpu,
            retrieve_next_sibling_cpu=retrieve_next_sibling_cpu,
            draft_tokens_cpu=draft_tokens_cpu,
            vocab_size=vocab_size,
        )

        # Bitmask should be allocated (for the active grammar)
        self.assertIsNotNone(result)
        # Terminated grammar should NOT have fill_vocab_mask called
        self.assertFalse(terminated_grammar.fill_vocab_mask_called)
        # Active grammar should have been processed
        self.assertTrue(active_grammar.allocate_vocab_mask_called)
        # verify_input.grammar should be the active grammar
        self.assertEqual(verify_input.grammar, active_grammar)

    def test_traverse_tree_respects_termination(self):
        """Test that traverse_tree stops generating masks after termination."""
        from sglang.srt.speculative.spec_utils import traverse_tree

        # Create a mock grammar that terminates after accepting token
        class TerminatingGrammar(MockGrammarObject):
            def __init__(self):
                super().__init__(is_terminated=False)
                self.accepted_tokens = []
                self.rollback_count = 0

            def accept_token(self, token: int):
                self.accepted_tokens.append(token)
                # Simulate termination after accepting any token
                self._is_terminated = True

            def rollback(self, k: int):
                self.rollback_count += 1
                for _ in range(k):
                    if self.accepted_tokens:
                        self.accepted_tokens.pop()
                self._is_terminated = False

        grammar = TerminatingGrammar()
        vocab_size = 100
        num_int32_per_token = (vocab_size + 31) // 32
        num_tokens = 4
        allocate_token_bitmask = torch.zeros(
            num_tokens, num_int32_per_token, dtype=torch.int32
        )

        # Simple tree: token 0 -> token 1 -> token 2
        retrieve_next_token = torch.tensor([1, 2, -1, -1])
        retrieve_next_sibling = torch.tensor([-1, -1, -1, -1])
        draft_tokens = torch.tensor([100, 101, 102, 103])

        # Should not raise even with termination
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            allocate_token_bitmask,
        )

        # Grammar should have accepted token 100 at position 0
        # After accepting, it terminates, so child nodes should not trigger
        # additional fill_vocab_mask calls


if __name__ == "__main__":
    unittest.main()
