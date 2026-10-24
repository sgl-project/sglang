"""Regression test for spec_utils.traverse_tree calling xgrammar with tensors.

xgrammar 0.2.0 tightened its FFI binding and rejects 0-d tensors where Python
ints are expected. The dfs in traverse_tree recurses with `retrieve_next_token[curr]`
and reads `draft_tokens[curr]`, both of which return 0-d tensors and must be
explicitly cast before being handed to the grammar matcher.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    generate_token_bitmask,
    traverse_tree,
)
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

    def test_sign_bit_token_passes_parent_check(self):
        # Token 31 of a word lives in the int32 sign bit; the host-side check must
        # still read it as allowed when the parent row permits it.
        rnt, rns, draft_tokens = self._chain(torch.tensor([[100, 31, 63]]))
        bitmask = torch.zeros((3, 4), dtype=torch.int32)
        bitmask[0, 0] = -(2**31)  # only token 31 allowed after the root
        bitmask[1, 1] = -(2**31)  # only token 63 allowed after node 1

        grammar, accept_calls, fill_calls = self._record_grammar()
        traverse_tree(rnt, rns, draft_tokens, grammar, bitmask)

        self.assertEqual(accept_calls, [31, 63])
        self.assertEqual(fill_calls, [0, 1, 2])


class TestGenerateTokenBitmaskSkipsUnconstrained(unittest.TestCase):
    """Grammars that cannot constrain the step keep their rows all-allowed
    without walking the draft tree."""

    def _req(self, needs_mask, vocab_size=64):
        grammar = MagicMock()
        grammar.needs_vocab_mask.return_value = needs_mask
        grammar.is_terminated.return_value = False
        grammar.allocate_vocab_mask.side_effect = (
            lambda vocab_size, batch_size, device: torch.full(
                (batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32
            )
        )
        # fill marks its row so a walk is visible in the mask; bit 0 cleared only,
        # so the chain's draft tokens (11, 22) still pass the parent check.
        grammar.fill_vocab_mask.side_effect = lambda mask, idx: mask[idx].fill_(-2)
        req = MagicMock()
        req.grammar = grammar
        return req

    def _chain(self, bs, n=3):
        ids = torch.tensor([[100, 11, 22]] * bs)
        return GrammarTree.from_linear_chain(ids).resolve()

    def test_all_unconstrained_returns_no_mask(self):
        reqs = [self._req(False), self._req(False)]
        rnt, rns, tokens = self._chain(2)
        mask, grammar = generate_token_bitmask(reqs, rnt, rns, tokens, 64)
        self.assertIsNone(mask)
        self.assertIsNone(grammar)
        for req in reqs:
            req.grammar.fill_vocab_mask.assert_not_called()
            req.grammar.accept_token.assert_not_called()

    def test_mixed_batch_walks_only_constrained_rows(self):
        reqs = [self._req(False), self._req(True), self._req(False)]
        rnt, rns, tokens = self._chain(3)
        mask, grammar = generate_token_bitmask(reqs, rnt, rns, tokens, 64)
        self.assertIs(grammar, reqs[1].grammar)
        # Rows of the walked request were filled; the others stay all-allowed.
        self.assertTrue(torch.all(mask[3:6] == -2))
        self.assertTrue(torch.all(mask[0:3] == -1))
        self.assertTrue(torch.all(mask[6:9] == -1))
        reqs[0].grammar.fill_vocab_mask.assert_not_called()
        reqs[2].grammar.fill_vocab_mask.assert_not_called()

    def test_build_clears_leftover_mask_when_nothing_constrains(self):
        reqs = [self._req(False)]
        sampling_info = MagicMock()
        sampling_info.vocab_size = 64
        sampling_info.grammar_mask = object()  # the extend stage's mask
        ids = torch.tensor([[100, 11, 22]])
        result = build_grammar_vocab_mask(
            reqs=reqs,
            tree=GrammarTree.from_linear_chain(ids),
            sampling_info=sampling_info,
            device="cpu",
            barrier=None,
        )
        self.assertIsNone(result)
        self.assertIsNone(sampling_info.grammar_mask)

    def test_build_without_grammars_leaves_sampling_info_alone(self):
        req = MagicMock()
        req.grammar = None
        sampling_info = MagicMock()
        sampling_info.vocab_size = 64
        leftover = object()
        sampling_info.grammar_mask = leftover
        ids = torch.tensor([[100, 11, 22]])
        result = build_grammar_vocab_mask(
            reqs=[req],
            tree=GrammarTree.from_linear_chain(ids),
            sampling_info=sampling_info,
            device="cpu",
            barrier=None,
        )
        self.assertIsNone(result)
        self.assertIs(sampling_info.grammar_mask, leftover)


if __name__ == "__main__":
    unittest.main()
