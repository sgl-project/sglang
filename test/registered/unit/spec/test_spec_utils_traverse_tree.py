"""Regression test for spec_utils.traverse_tree calling xgrammar with tensors.

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected. The dfs in traverse_tree recurses with `retrieve_next_token[curr]`
and reads `draft_tokens[curr]`, both of which return 0-d tensors and must be
explicitly cast before being handed to the grammar matcher.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.spec_utils import GrammarTree, traverse_tree
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTraverseTreePassesIntsToGrammar(unittest.TestCase):
    def _record_grammar(self):
        """A grammar mock that records every call argument and rejects torch tensors."""
        grammar = MagicMock()
        grammar.is_terminated.return_value = False
        accept_calls = []
        fill_calls = []

        def record_accept(token):
            if isinstance(token, torch.Tensor):
                raise TypeError(f"accept_token got torch.Tensor: {token!r}")
            accept_calls.append(token)

        def record_fill(bitmask, idx):
            if isinstance(idx, torch.Tensor):
                raise TypeError(f"fill_vocab_mask got torch.Tensor idx: {idx!r}")
            fill_calls.append(idx)

        grammar.accept_token.side_effect = record_accept
        grammar.fill_vocab_mask.side_effect = record_fill
        grammar.rollback.return_value = None
        return grammar, accept_calls, fill_calls

    def _chain(self, verify_ids_2d):
        """Row 0 of the links chain-verify algorithms actually feed traverse_tree."""
        links = GrammarTree.from_linear_chain(verify_ids_2d).resolve()
        return tuple(t[0] for t in links)

    def test_branching_tree_passes_ints(self):
        # Binary tree exercises both child recursion and sibling recursion:
        #   0 ─┬─ 1
        #      └─ 2 ─── 3
        retrieve_next_token = torch.tensor([1, -1, 3, -1], dtype=torch.int32)
        retrieve_next_sibling = torch.tensor([-1, 2, -1, -1], dtype=torch.int32)
        draft_tokens = torch.tensor([100, 11, 22, 33], dtype=torch.int64)
        # all bits set: every draft token passes the parent's bitmask check
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
            grammar,
            bitmask,
        )

        self.assertEqual(set(accept_calls), {11, 22, 33})
        self.assertEqual(set(fill_calls), {0, 1, 2, 3})
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_visits_all_positions_in_order(self):
        # Chain-verify algorithms (DFLASH/DSPARK) have no branching, so their tree
        # degenerates to 0 -- 1 -- 2 -- 3 with column 0 the already-committed token.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 11, 22, 33]]))
        self.assertEqual(rnt.tolist(), [1, 2, 3, -1])
        self.assertEqual(rns.tolist(), [-1, -1, -1, -1])
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Root (col 0) is never accepted; every draft token is, in chain order.
        self.assertEqual(accept_calls, [11, 22, 33])
        self.assertEqual(fill_calls, [0, 1, 2, 3])
        for token in accept_calls:
            self.assertIsInstance(token, int)
        for idx in fill_calls:
            self.assertIsInstance(idx, int)

    def test_linear_chain_stops_at_grammar_reject(self):
        # A draft token the grammar disallows must stop the descent: no accept/fill
        # for that node or anything after it, so the mask rows past it stay unfilled
        # and only the already-filled prefix can be committed.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 5, 7, 9]]))
        bitmask = torch.full((4, 4), -1, dtype=torch.int32)  # all allowed
        # Disallow token id 7 (draft_tokens[2]) in node 1's mask (its parent).
        bitmask[1, 7 // 32] &= ~(1 << (7 % 32))

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        # Node 1 accepted+filled; node 2 rejected -> node 2 and node 3 skipped.
        self.assertEqual(accept_calls, [5])
        self.assertEqual(fill_calls, [0, 1])


if __name__ == "__main__":
    unittest.main()
