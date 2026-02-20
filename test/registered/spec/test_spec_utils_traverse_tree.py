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
"""Unit tests for traverse_tree with grammar termination (speculative decoding crash repro).

These tests provide a 100% reproducible way to trigger the crash that occurs when
the grammar terminates mid-DFS during speculative decoding: a mock grammar
terminates at a chosen point and raises in fill_vocab_mask when terminated,
and a minimal draft tree forces the DFS to visit a sibling after termination.
With the fix (guards + try/except in spec_utils.traverse_tree), the test passes.
Without the fix, traverse_tree would call fill_vocab_mask on a terminated
grammar and the mock would raise RuntimeError.
"""

import unittest

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.speculative.spec_utils import traverse_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=2, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=2, suite="stage-b-test-small-1-gpu-amd")


# -----------------------------------------------------------------------------
# Mock grammar: terminates after N accepts, does not clear on rollback,
# raises RuntimeError in fill_vocab_mask when terminated.
# -----------------------------------------------------------------------------
class MockGrammarTerminatesAfterAccept(BaseGrammarObject):
    """Test-only grammar that terminates after a chosen number of accept_token calls.

    - After the Nth accept_token, _terminated is set and never cleared by rollback.
    - fill_vocab_mask raises RuntimeError when _terminated is True (simulates
      xgrammar matcher behavior when grammar is already terminated).
    """

    def __init__(self, terminate_after_accept_count: int = 1):
        super().__init__()
        self._accept_count = 0
        self._terminate_after = terminate_after_accept_count
        self._terminated = False

    def accept_token(self, token: int) -> None:
        if self._terminated:
            raise RuntimeError(
                "accept_token called on terminated grammar (speculative decoding crash)"
            )
        self._accept_count += 1
        if self._accept_count >= self._terminate_after:
            self._terminated = True

    def rollback(self, k: int) -> None:
        self._accept_count = max(0, self._accept_count - k)
        # Do NOT clear _terminated so that sibling/child visits see terminated state.

    def is_terminated(self) -> bool:
        return self._terminated

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        # Packed: 32 booleans per int; need (vocab_size + 31) // 32 ints per position.
        num_packed = (vocab_size + 31) // 32
        return torch.zeros(batch_size, num_packed, dtype=torch.int32, device=device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        if self._terminated:
            raise RuntimeError(
                "fill_vocab_mask called on terminated grammar (speculative decoding crash)"
            )
        # Otherwise no-op; parent bitmask was already set so DFS acceptance still works.

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device)

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        pass


# -----------------------------------------------------------------------------
# Minimal draft tree: root 0 -> child 1; node 1 has sibling 2.
# DFS order: 0 (fill_vocab_mask(0), dfs(1)) -> 1 (accept_token(1) -> terminate)
# -> rollback(1), then sibling dfs(2). Without fix, at 2 we'd call fill_vocab_mask(2)
# while terminated -> mock raises. With fix we skip fill_vocab_mask and sibling
# visit when terminated.
# -----------------------------------------------------------------------------
def make_minimal_tree_that_triggers_termination_then_sibling():
    """Build tensors for a 3-node tree: 0 -> 1, sibling 2.

    - retrieve_next_token: root 0 has child 1; nodes 1 and 2 have no child (-1).
    - retrieve_next_sibling: root 0 has no sibling (-1); node 1 has sibling 2; node 2 none (-1).
    - draft_tokens: [token0, token1, token2] with token1 accepted at node 1 (mock will terminate).
    - allocate_token_bitmask: position 0 must allow draft_tokens[1] so node 1 is "accepted".
    """
    num_nodes = 3
    vocab_size = 64
    num_packed = (vocab_size + 31) // 32  # 2

    retrieve_next_token = torch.tensor([1, -1, -1], dtype=torch.int64)
    retrieve_next_sibling = torch.tensor([-1, 2, -1], dtype=torch.int64)
    draft_tokens = torch.tensor([0, 1, 2], dtype=torch.int64)

    # Bitmask: 32 booleans per int. For node 1 to be accepted, parent (0) must have
    # draft_tokens[1]=1 set. So allocate_token_bitmask[0, 1//32] |= (1 << (1 % 32)).
    allocate_token_bitmask = torch.zeros(num_nodes, num_packed, dtype=torch.int32)
    allocate_token_bitmask[0, 0] = (1 << 1) | (
        1 << 2
    )  # allow tokens 1 and 2 so node 1 and sibling 2 are accepted

    return (
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
        allocate_token_bitmask,
    )


class TestTraverseTreeGrammarTermination(unittest.TestCase):
    """Regression tests for traverse_tree when grammar terminates mid-DFS."""

    def test_traverse_tree_with_grammar_termination_does_not_raise(self):
        """With the fix, traverse_tree must not raise when grammar terminates during DFS.

        The mock grammar terminates after the first accept_token (at node 1).
        DFS then rolls back and would visit sibling 2; with the fix we skip
        fill_vocab_mask and sibling visits when terminated, so no RuntimeError.
        Without the fix, fill_vocab_mask(2) would be called while terminated
        and the mock would raise.
        """
        (
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            allocate_token_bitmask,
        ) = make_minimal_tree_that_triggers_termination_then_sibling()

        grammar = MockGrammarTerminatesAfterAccept(terminate_after_accept_count=1)

        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            allocate_token_bitmask,
        )

        self.assertTrue(grammar.is_terminated())


if __name__ == "__main__":
    unittest.main()
