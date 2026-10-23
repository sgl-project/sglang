"""Unit tests for tree-aware DSA sparse-index construction (EAGLE topk>1, Ascend).

On Ascend, DSA visibility is not expressed by a mask: each query row gets an
explicit list of KV positions (`sparse_indices[T, K]`) that
`torch_npu.npu_sparse_flash_attention` may read. For tree verify the list must be

    row_r = [ ancestors(r) | prefix top-(K-A) | -1 padding ]

and the ORDER matters:

* ancestors come first because `sparse_mode=3` only reads
  ``min(K, Lkv - Lq + i + 1)`` slots per row (short contexts truncate the tail);
* ``-1`` terminates the scan, so it may only appear in the tail;
* SFA does not require ascending order (its own tests feed randperm), and sorting
  would push ancestors (larger positions) into the truncated tail.

These tests run on CPU with plain torch.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

try:  # normal path (Linux CI, sglang installed / PYTHONPATH=python)
    from sglang.srt.hardware_backend.npu.attention.tree_sparse_indices import (
        ancestor_width_for,
        draft_ancestor_ids_batched,
        gather_tree_blocks,
        merge_tree_sparse_indices,
    )
except Exception:  # Windows dev boxes cannot import sglang (stdlib `resource`)
    _MODULE = (
        Path(__file__).resolve().parents[4]
        / "python"
        / "sglang"
        / "srt"
        / "hardware_backend"
        / "npu"
        / "attention"
        / "tree_sparse_indices.py"
    )
    _spec = importlib.util.spec_from_file_location("tree_sparse_indices", _MODULE)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ancestor_width_for = _mod.ancestor_width_for
    draft_ancestor_ids_batched = _mod.draft_ancestor_ids_batched
    gather_tree_blocks = _mod.gather_tree_blocks
    merge_tree_sparse_indices = _mod.merge_tree_sparse_indices

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
except Exception:
    pass


TREE_PARENTS = [-1, 0, 0, 1, 1, 2, 2]
CHAIN_PARENTS = [-1, 0, 1, 2, 3, 4, 5]
#          0
#        /   \
#       1     2
#      / \   / \
#     3   4 5   6


def _visibility_from_parents(parents, num_nodes):
    """Oracle: row r sees r and all its ancestors."""
    vis = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for r in range(num_nodes):
        cur = r
        while cur != -1:
            vis[r, cur] = True
            cur = parents[cur]
    return vis


def _make_full_mask(blocks, prefix_lens, num_nodes):
    """Reproduce build_tree_kernel_efficient's flat FULL_MASK layout."""
    rows = []
    for vis, p in zip(blocks, prefix_lens):
        block = torch.zeros((num_nodes, p + num_nodes), dtype=torch.bool)
        block[:, :p] = True
        block[:, p:] = vis
        rows.append(block.reshape(-1))
    return torch.cat(rows).contiguous()


def _scores_from_allowed(allowed):
    torch.manual_seed(0)
    return torch.rand(allowed.shape).masked_fill(~allowed, float("-inf"))


class TestAncestorWidth(unittest.TestCase):
    def test_steps_plus_one(self):
        self.assertEqual(ancestor_width_for(3), 4)

    def test_clamped_to_num_nodes(self):
        self.assertEqual(ancestor_width_for(9, num_verify_tokens=4), 4)


class TestGatherTreeBlocks(unittest.TestCase):
    def test_single_request_matches_oracle(self):
        vis = _visibility_from_parents(TREE_PARENTS, 7)
        flat = _make_full_mask([vis], [4], 7)

        got = gather_tree_blocks(flat, torch.tensor([4]), 7)

        self.assertEqual(tuple(got.shape), (1, 7, 7))
        self.assertTrue(torch.equal(got[0], vis))

    def test_batched_requests_with_varying_prefix(self):
        tree = _visibility_from_parents(TREE_PARENTS, 7)
        chain = _visibility_from_parents(CHAIN_PARENTS, 7)
        prefix_lens = [4, 0, 130]
        blocks = [tree, chain, tree]
        flat = _make_full_mask(blocks, prefix_lens, 7)

        got = gather_tree_blocks(flat, torch.tensor(prefix_lens), 7)

        self.assertEqual(tuple(got.shape), (3, 7, 7))
        for i in range(3):
            self.assertTrue(torch.equal(got[i], blocks[i]), f"request {i}")

    def test_buffer_too_small_raises(self):
        flat = torch.ones(10, dtype=torch.bool)
        with self.assertRaises(ValueError):
            gather_tree_blocks(flat, torch.tensor([4]), 7)


class TestDraftAncestorIds(unittest.TestCase):
    def test_only_allowed_and_self_present(self):
        num_nodes = 7
        allowed = _visibility_from_parents(TREE_PARENTS, num_nodes)
        scores = _scores_from_allowed(allowed)

        got = draft_ancestor_ids_batched(scores.unsqueeze(0), allowed.unsqueeze(0), 3)

        self.assertEqual(tuple(got.shape), (1, num_nodes, 3))
        self.assertEqual(got.dtype, torch.int32)
        for row in range(num_nodes):
            self.assertIn(row, got[0, row].tolist(), "self must be present")
            for col in got[0, row].tolist():
                self.assertTrue(bool(allowed[row, col]), f"row {row} picked {col}")

    def test_width_clamped(self):
        num_nodes = 3
        allowed = _visibility_from_parents([-1, 0, 1], num_nodes)
        scores = _scores_from_allowed(allowed)
        got = draft_ancestor_ids_batched(scores.unsqueeze(0), allowed.unsqueeze(0), 99)
        self.assertEqual(tuple(got.shape), (1, num_nodes, num_nodes))

    def test_root_gets_all_self_slots(self):
        allowed = _visibility_from_parents(TREE_PARENTS, 7)
        scores = _scores_from_allowed(allowed)
        got = draft_ancestor_ids_batched(scores.unsqueeze(0), allowed.unsqueeze(0), 3)
        self.assertEqual(got[0, 0].tolist(), [0, 0, 0])


class TestMergeTreeSparseIndices(unittest.TestCase):
    def _draft(self, num_nodes, seq_len, ancestor_width, parents=TREE_PARENTS):
        allowed = _visibility_from_parents(parents, num_nodes)
        scores = _scores_from_allowed(allowed)
        return draft_ancestor_ids_batched(
            scores.unsqueeze(0), allowed.unsqueeze(0), ancestor_width
        )[0]

    def test_ancestors_lead_and_prefix_follows(self):
        num_nodes, topk, a = 7, 16, 3
        seq_lens = torch.tensor([4], dtype=torch.int32)
        draft = self._draft(num_nodes, 4, a)
        prefix = (torch.arange(topk - a, dtype=torch.int32) % 4).repeat(num_nodes, 1)

        merged = merge_tree_sparse_indices(prefix, draft, seq_lens, num_nodes, topk)

        self.assertEqual(tuple(merged.shape), (num_nodes, topk))
        self.assertEqual(merged.dtype, torch.int32)
        # Ancestors occupy the leading slots, as absolute KV positions.
        for row in range(num_nodes):
            self.assertEqual(merged[row, :a].tolist(), (draft[row] + 4).tolist())
            # Prefix entries keep score order and are < P.
            self.assertTrue(torch.equal(merged[row, a:], prefix[row]))

    def test_no_sorting_of_prefix(self):
        """SFA does not require ascending order; sorting would reorder ancestors."""
        num_nodes, topk, a = 4, 8, 4
        seq_lens = torch.tensor([9], dtype=torch.int32)
        draft = self._draft(num_nodes, 9, a, parents=CHAIN_PARENTS[:4])
        # Deliberately descending prefix scores -> descending indices.
        prefix = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32).repeat(num_nodes, 1)

        merged = merge_tree_sparse_indices(prefix, draft, seq_lens, num_nodes, topk)

        self.assertTrue(torch.equal(merged[:, a:], prefix))
        self.assertNotEqual(merged[0].tolist(), sorted(merged[0].tolist()))

    def test_negative_prefix_padding_only_in_tail(self):
        num_nodes, topk, a = 4, 8, 4
        seq_lens = torch.tensor([2], dtype=torch.int32)
        draft = self._draft(num_nodes, 2, a, parents=CHAIN_PARENTS[:4])
        prefix = torch.full((num_nodes, topk - a), -1, dtype=torch.int32)
        prefix[:, :2] = torch.arange(2)

        merged = merge_tree_sparse_indices(prefix, draft, seq_lens, num_nodes, topk)

        for row in range(num_nodes):
            tail = merged[row, a:].tolist()
            first_neg = tail.index(-1) if -1 in tail else len(tail)
            self.assertTrue(all(v >= 0 for v in tail[:first_neg]), f"row {row}")
            self.assertTrue(all(v == -1 for v in tail[first_neg:]), f"row {row}")

    def test_ancestor_out_of_range_raises(self):
        num_nodes, topk, a = 4, 8, 4
        seq_lens = torch.tensor([5], dtype=torch.int32)
        draft = torch.zeros((num_nodes, a), dtype=torch.int32)
        draft[0, 0] = num_nodes + 3  # outside [0, N)
        prefix = torch.zeros((num_nodes, topk - a), dtype=torch.int32)
        with self.assertRaises(ValueError):
            merge_tree_sparse_indices(prefix, draft, seq_lens, num_nodes, topk)

    def test_width_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            merge_tree_sparse_indices(
                prefix_indices=torch.zeros((1, 4), dtype=torch.int32),
                draft_ids=torch.zeros((1, 4), dtype=torch.int32),
                seq_lens=torch.tensor([3], dtype=torch.int32),
                num_verify_tokens=7,
                topk=16,
            )


if __name__ == "__main__":
    unittest.main()
