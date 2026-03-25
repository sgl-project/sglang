"""Unit tests for pure-Python logic in srt/speculative/spec_utils.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.spec_utils import (
    create_accept_length_filter,
    spec_need_hidden_states,
    traverse_tree,
)
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# spec_need_hidden_states
# ---------------------------------------------------------------------------


class TestSpecNeedHiddenStates(CustomTestCase):
    """Tests for spec_need_hidden_states()."""

    def test_multi_layer_eagle_enabled_returns_false(self):
        server_args = MagicMock()
        server_args.enable_multi_layer_eagle = True
        self.assertFalse(spec_need_hidden_states(server_args))

    def test_multi_layer_eagle_disabled_returns_true(self):
        server_args = MagicMock()
        server_args.enable_multi_layer_eagle = False
        self.assertTrue(spec_need_hidden_states(server_args))

    @patch("sglang.srt.speculative.spec_utils.get_global_server_args")
    def test_none_falls_back_to_global(self, mock_get_global):
        mock_get_global.return_value = MagicMock(enable_multi_layer_eagle=True)
        self.assertFalse(spec_need_hidden_states(None))
        mock_get_global.assert_called_once()


# ---------------------------------------------------------------------------
# create_accept_length_filter
# ---------------------------------------------------------------------------


class TestCreateAcceptLengthFilter(CustomTestCase):
    """Tests for create_accept_length_filter() on CPU tensors.

    The function:
    1. Creates a zero tensor shaped like accept_length
    2. For unfinished indices: sets filter[i] = accept_length[i] + 1
    3. Mutates seq_lens in-place: seq_lens += accept_length + 1
    """

    def _run(self, accept_length, unfinished_indices, seq_lens):
        """Helper that calls the compiled function with CPU tensors."""
        al = torch.tensor(accept_length, dtype=torch.int64)
        ui = torch.tensor(unfinished_indices, dtype=torch.int64)
        sl = torch.tensor(seq_lens, dtype=torch.int64)
        result = create_accept_length_filter(al, ui, sl)
        return result, sl

    def test_all_unfinished(self):
        """All requests are unfinished — filter should be accept_length + 1 for all."""
        result, seq_lens = self._run(
            accept_length=[2, 3, 1],
            unfinished_indices=[0, 1, 2],
            seq_lens=[10, 20, 30],
        )
        self.assertTrue(torch.equal(result, torch.tensor([3, 4, 2])))
        # seq_lens should be mutated: 10+3, 20+4, 30+2
        self.assertTrue(torch.equal(seq_lens, torch.tensor([13, 24, 32])))

    def test_none_unfinished(self):
        """No unfinished requests — filter should be all zeros."""
        result, seq_lens = self._run(
            accept_length=[2, 3],
            unfinished_indices=[],
            seq_lens=[10, 20],
        )
        self.assertTrue(torch.equal(result, torch.tensor([0, 0])))
        # seq_lens still mutated: 10+3, 20+4
        self.assertTrue(torch.equal(seq_lens, torch.tensor([13, 24])))

    def test_partial_unfinished(self):
        """Only some requests are unfinished."""
        result, seq_lens = self._run(
            accept_length=[2, 3, 4, 1],
            unfinished_indices=[1, 3],
            seq_lens=[10, 20, 30, 40],
        )
        expected_filter = torch.tensor([0, 4, 0, 2])
        self.assertTrue(torch.equal(result, expected_filter))

    def test_single_request(self):
        result, seq_lens = self._run(
            accept_length=[5],
            unfinished_indices=[0],
            seq_lens=[100],
        )
        self.assertTrue(torch.equal(result, torch.tensor([6])))
        self.assertTrue(torch.equal(seq_lens, torch.tensor([106])))

    def test_zero_accept_length(self):
        """accept_length=0 means only 1 token accepted (the verified token)."""
        result, seq_lens = self._run(
            accept_length=[0],
            unfinished_indices=[0],
            seq_lens=[50],
        )
        self.assertTrue(torch.equal(result, torch.tensor([1])))
        self.assertTrue(torch.equal(seq_lens, torch.tensor([51])))


# ---------------------------------------------------------------------------
# traverse_tree (DFS grammar-constrained tree traversal)
# ---------------------------------------------------------------------------


class MockGrammar:
    """Mock grammar object that tracks accept/rollback/fill calls."""

    def __init__(self, vocab_size=64):
        self.accepted_tokens = []
        self.rollback_counts = []
        self.fill_positions = []
        self._terminated = False
        self._vocab_size = vocab_size

    def accept_token(self, token_id):
        self.accepted_tokens.append(int(token_id))

    def is_terminated(self):
        return self._terminated

    def fill_vocab_mask(self, bitmask, pos):
        """Fill bitmask to accept all tokens."""
        self.fill_positions.append(pos)
        # -1 in two's complement int32 = 0xFFFFFFFF = all bits set
        bitmask[pos].fill_(-1)

    def rollback(self, n):
        self.rollback_counts.append(n)


def _make_bitmask(num_tokens, vocab_size=64):
    """Create a token bitmask tensor (packed 32 booleans per int32)."""
    num_ints = (vocab_size + 31) // 32
    return torch.zeros(num_tokens, num_ints, dtype=torch.int32)


class TestTraverseTree(CustomTestCase):
    """Tests for traverse_tree() DFS logic with mock grammar.

    Tree structure is encoded by:
    - retrieve_next_token[i]: index of the first child of node i (-1 = leaf)
    - retrieve_next_sibling[i]: index of the next sibling of node i (-1 = none)
    - draft_tokens[i]: the token ID at node i
    """

    def test_single_node_tree(self):
        """Tree with only the root (node 0). No children, no siblings.
        Node 0 is always accepted. Grammar should get fill_vocab_mask called once."""
        grammar = MockGrammar()
        bitmask = _make_bitmask(1)
        traverse_tree(
            retrieve_next_token=torch.tensor([-1]),
            retrieve_next_sibling=torch.tensor([-1]),
            draft_tokens=torch.tensor([42]),
            grammar=grammar,
            allocate_token_bitmask=bitmask,
        )
        # Node 0 is always accepted without accept_token call
        self.assertEqual(grammar.accepted_tokens, [])
        # fill_vocab_mask is called for node 0
        self.assertEqual(grammar.fill_positions, [0])
        # No rollback needed for root
        self.assertEqual(grammar.rollback_counts, [])

    def test_linear_chain(self):
        """Tree: 0 -> 1 -> 2 (linear chain).
        All tokens should be accepted if bitmask allows them."""
        grammar = MockGrammar()
        bitmask = _make_bitmask(3)
        # Node 0 generates bitmask that accepts token at node 1
        # Node 1 generates bitmask that accepts token at node 2
        draft_tokens = torch.tensor([10, 5, 3])

        traverse_tree(
            retrieve_next_token=torch.tensor([1, 2, -1]),
            retrieve_next_sibling=torch.tensor([-1, -1, -1]),
            draft_tokens=draft_tokens,
            grammar=grammar,
            allocate_token_bitmask=bitmask,
        )
        # Tokens 5 and 3 should be accepted (node 0 doesn't call accept_token)
        self.assertEqual(grammar.accepted_tokens, [5, 3])
        # fill_vocab_mask called for nodes 0, 1, 2
        self.assertEqual(grammar.fill_positions, [0, 1, 2])
        # Rollbacks: node 2 rollback(1), node 1 rollback(1)
        self.assertEqual(grammar.rollback_counts, [1, 1])

    def test_branching_tree(self):
        """Tree: 0 -> 1, 0 -> 2 (root with two children via sibling link).
        Structure: next_token[0]=1, next_sibling[1]=2."""
        grammar = MockGrammar()
        bitmask = _make_bitmask(3)
        draft_tokens = torch.tensor([10, 5, 7])

        traverse_tree(
            retrieve_next_token=torch.tensor([1, -1, -1]),
            retrieve_next_sibling=torch.tensor([-1, 2, -1]),
            draft_tokens=draft_tokens,
            grammar=grammar,
            allocate_token_bitmask=bitmask,
        )
        # Both children 5 and 7 should be accepted
        self.assertEqual(grammar.accepted_tokens, [5, 7])
        # fill_vocab_mask: node 0, node 1, node 2
        self.assertEqual(grammar.fill_positions, [0, 1, 2])
        # Rollbacks: node 1 rollback(1), node 2 rollback(1)
        self.assertEqual(grammar.rollback_counts, [1, 1])

    def test_rejected_token_by_vocab_size(self):
        """Token ID >= vocab_size should be rejected."""
        grammar = MockGrammar(vocab_size=10)
        bitmask = _make_bitmask(2, vocab_size=10)
        # Node 1 has token_id=15 which is >= vocab_size=10
        draft_tokens = torch.tensor([0, 15])

        traverse_tree(
            retrieve_next_token=torch.tensor([1, -1]),
            retrieve_next_sibling=torch.tensor([-1, -1]),
            draft_tokens=draft_tokens,
            grammar=grammar,
            allocate_token_bitmask=bitmask,
            vocab_size=10,
        )
        # Token 15 should be rejected — no accept_token call for it
        self.assertEqual(grammar.accepted_tokens, [])
        # fill_vocab_mask only for node 0 (node 1 is rejected)
        self.assertEqual(grammar.fill_positions, [0])
        self.assertEqual(grammar.rollback_counts, [])

    def test_rejected_token_by_bitmask(self):
        """Token rejected because parent bitmask doesn't allow it."""
        grammar = MockGrammar()
        bitmask = _make_bitmask(2)
        # After fill_vocab_mask for node 0, we manually clear the bit for token 5
        # We need to override the mock grammar's fill_vocab_mask to NOT accept token 5

        class SelectiveGrammar(MockGrammar):
            def fill_vocab_mask(self, bm, pos):
                self.fill_positions.append(pos)
                # Accept all tokens EXCEPT token 5
                bm[pos].fill_(-1)
                bm[pos][5 // 32] &= ~(1 << (5 % 32))  # clear bit for token 5

        grammar = SelectiveGrammar()
        draft_tokens = torch.tensor([10, 5])

        traverse_tree(
            retrieve_next_token=torch.tensor([1, -1]),
            retrieve_next_sibling=torch.tensor([-1, -1]),
            draft_tokens=draft_tokens,
            grammar=grammar,
            allocate_token_bitmask=bitmask,
        )
        # Token 5 should be rejected by bitmask
        self.assertEqual(grammar.accepted_tokens, [])
        # fill_vocab_mask only for node 0
        self.assertEqual(grammar.fill_positions, [0])

    def test_terminated_grammar_stops_expansion(self):
        """If grammar terminates, children should not be visited."""
        grammar = MockGrammar()
        grammar._terminated = True  # grammar is already terminated
        bitmask = _make_bitmask(2)
        draft_tokens = torch.tensor([10, 5])

        traverse_tree(
            retrieve_next_token=torch.tensor([1, -1]),
            retrieve_next_sibling=torch.tensor([-1, -1]),
            draft_tokens=draft_tokens,
            grammar=grammar,
            allocate_token_bitmask=bitmask,
        )
        # Node 0 accepted but grammar is terminated — no fill, no child visit
        self.assertEqual(grammar.accepted_tokens, [])
        self.assertEqual(grammar.fill_positions, [])


if __name__ == "__main__":
    unittest.main()
