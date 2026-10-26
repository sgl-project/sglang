"""Unit tests for tree-aware DSA sparse-index construction (EAGLE topk>1, Ascend).

On Ascend, DSA visibility is not expressed by a mask: each query row gets an
explicit list of KV positions (`sparse_indices[T, K]`) that
`torch_npu.npu_sparse_flash_attention` may read. For tree verify the list must be

    row_r = [ ancestors(r) | prefix top-* (score order) | -1 padding ]

and the ORDER matters:

* ancestors come first because `sparse_mode=3` bounds a row to
  `min(K, L_kv - L_q + i + 1)` entries (short contexts truncate the tail);
* ``-1`` terminates the scan, so it may only appear in the tail;
* SFA does not require ascending order (its own tests feed randperm), and sorting
  would push ancestors (larger positions) into the truncated tail;
* **no position may repeat**: the kernel has no uniqueness check
  (`sparse_flash_attention_kernel_mla.h` only stops at ``-1``) and
  `sparse_block_size=1`, so a duplicate is read twice and double-counts in the
  softmax.

These tests run on CPU with plain torch.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

try:  # normal path (Linux CI, sglang installed / PYTHONPATH=python)
    from sglang.srt.hardware_backend.npu.attention.tree_sparse_indices import (
        DRAFT_TREE_ANCESTORS_ATTR,
        DRAFT_TREE_PREFIX_ATTR,
        DraftTreeSparseState,
        ancestor_width_for,
        draft_ancestors,
        draft_step_ancestor_positions,
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
    draft_ancestors = _mod.draft_ancestors
    draft_step_ancestor_positions = _mod.draft_step_ancestor_positions
    DraftTreeSparseState = _mod.DraftTreeSparseState
    DRAFT_TREE_ANCESTORS_ATTR = _mod.DRAFT_TREE_ANCESTORS_ATTR
    DRAFT_TREE_PREFIX_ATTR = _mod.DRAFT_TREE_PREFIX_ATTR
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


def _ancestor_ids(allowed, ancestor_width):
    return draft_ancestors(allowed.unsqueeze(0), ancestor_width)[0]


def _prefix_rows(num_rows, width, prefix_len):
    """Distinct, descending positions so ordering is observable."""
    return (
        torch.arange(width, dtype=torch.int32).flip(0) % max(prefix_len, 1)
    ).repeat(num_rows, 1)


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


class TestDraftAncestors(unittest.TestCase):
    def test_ascending_and_padded_with_negative_one(self):
        num_nodes = 7
        allowed = _visibility_from_parents(TREE_PARENTS, num_nodes)

        got = _ancestor_ids(allowed, 4)

        self.assertEqual(tuple(got.shape), (num_nodes, 4))
        self.assertEqual(got.dtype, torch.int32)
        for row in range(num_nodes):
            ids = got[row].tolist()
            valid = [v for v in ids if v >= 0]
            self.assertEqual(valid, sorted(valid), f"row {row} must ascend")
            self.assertIn(row, valid, "self must be present")
            for v in valid:
                self.assertTrue(bool(allowed[row, v]), f"row {row} picked {v}")
            # Padding is -1 and strictly in the tail.
            self.assertEqual(ids[: len(valid)] + [-1] * (len(ids) - len(valid)), ids)

    def test_shallow_node_pads_with_negative_one_not_self(self):
        """A shallow node must not repeat itself: a repeat is a double-count."""
        allowed = _visibility_from_parents(TREE_PARENTS, 7)

        got = _ancestor_ids(allowed, 4)

        # Node 0 is the root: only itself is visible.
        self.assertEqual(got[0].tolist(), [0, -1, -1, -1])
        # Node 1 sits at depth 1: self + root.
        self.assertEqual(got[1].tolist(), [0, 1, -1, -1])
        # Node 3 is a leaf at depth 2: the whole reserved width fits.
        self.assertEqual(got[3].tolist(), [0, 1, 3, -1])

    def test_width_clamped(self):
        num_nodes = 3
        allowed = _visibility_from_parents([-1, 0, 1], num_nodes)
        got = _ancestor_ids(allowed, 99)
        self.assertEqual(tuple(got.shape), (num_nodes, num_nodes))

    def test_non_square_mask_raises(self):
        with self.assertRaises(ValueError):
            draft_ancestors(torch.zeros((1, 4, 5), dtype=torch.bool), 2)

    def test_more_visible_than_reserved_raises(self):
        """A 7-deep chain does not fit num_steps=3: fail loudly, never drop an
        ancestor silently."""
        allowed = _visibility_from_parents(CHAIN_PARENTS, 7)
        with self.assertRaises(ValueError):
            draft_ancestors(allowed.unsqueeze(0), 4)

    def test_missing_self_raises_at_merge(self):
        allowed = _visibility_from_parents(TREE_PARENTS, 7)
        ids = _ancestor_ids(allowed, 4)
        ids[2] = -1  # row without any visible draft node
        with self.assertRaises(ValueError):
            merge_tree_sparse_indices(
                torch.zeros((7, 12), dtype=torch.int32),
                ids,
                torch.tensor([16], dtype=torch.int32),
                7,
                16,
            )


class TestMergeTreeSparseIndices(unittest.TestCase):
    def _merged(
        self, num_nodes, prefix_len, topk, ancestor_width, parents=TREE_PARENTS
    ):
        allowed = _visibility_from_parents(parents, num_nodes)
        ids = _ancestor_ids(allowed, ancestor_width)
        prefix = _prefix_rows(num_nodes, topk - ancestor_width, prefix_len)
        return merge_tree_sparse_indices(
            prefix, ids, torch.tensor([prefix_len], dtype=torch.int32), num_nodes, topk
        )

    def test_ancestors_lead_and_prefix_follows(self):
        num_nodes, topk, a, prefix_len = 7, 16, 4, 12
        allowed = _visibility_from_parents(TREE_PARENTS, num_nodes)
        ids = _ancestor_ids(allowed, a)
        prefix = _prefix_rows(num_nodes, topk - a, prefix_len)

        merged = merge_tree_sparse_indices(
            prefix, ids, torch.tensor([prefix_len], dtype=torch.int32), num_nodes, topk
        )

        self.assertEqual(tuple(merged.shape), (num_nodes, topk))
        self.assertEqual(merged.dtype, torch.int32)
        for row in range(num_nodes):
            valid = [v for v in ids[row].tolist() if v >= 0]
            self.assertEqual(
                merged[row, : len(valid)].tolist(),
                [prefix_len + v for v in valid],
                f"row {row} ancestors",
            )
            # The prefix block follows intact, in score order, shifted per row.
            self.assertTrue(
                torch.equal(merged[row, len(valid) : len(valid) + prefix.shape[1]],
                            prefix[row]),
                f"row {row} prefix",
            )

    def test_no_duplicate_kv_positions(self):
        """Regression: SFA has no dedup, so a repeated position double-counts.

        The old builder padded the ancestor block with the node itself, which
        repeated every shallow node's position up to ``A`` times.
        """
        num_nodes, topk, a, prefix_len = 7, 16, 4, 12
        merged = self._merged(num_nodes, prefix_len, topk, a)

        for row in range(num_nodes):
            valid = [int(v) for v in merged[row] if int(v) >= 0]
            self.assertEqual(
                len(set(valid)), len(valid), f"row {row} repeats a position: {valid}"
            )

    def test_shallow_row_keeps_every_prefix_entry(self):
        """Shifting (not padding) is what keeps the full prefix block reachable."""
        num_nodes, topk, a, prefix_len = 7, 16, 4, 12
        allowed = _visibility_from_parents(TREE_PARENTS, num_nodes)
        ids = _ancestor_ids(allowed, a)
        prefix = _prefix_rows(num_nodes, topk - a, prefix_len)

        merged = merge_tree_sparse_indices(
            prefix, ids, torch.tensor([prefix_len], dtype=torch.int32), num_nodes, topk
        )

        for row in range(num_nodes):
            n_anc = int((ids[row] >= 0).sum())
            valid = [int(v) for v in merged[row] if int(v) >= 0]
            self.assertEqual(len(valid), n_anc + prefix.shape[1], f"row {row}")
            # And the slack is exactly the unused reserved width.
            self.assertEqual(list(merged[row, len(valid):]), [-1] * (topk - len(valid)))

    def test_negative_prefix_padding_only_in_tail(self):
        num_nodes, topk, a = 4, 8, 4
        ids = _ancestor_ids(_visibility_from_parents(CHAIN_PARENTS[:4], num_nodes), a)
        prefix = torch.full((num_nodes, topk - a), -1, dtype=torch.int32)
        prefix[:, :2] = torch.arange(2)

        merged = merge_tree_sparse_indices(
            prefix, ids, torch.tensor([2], dtype=torch.int32), num_nodes, topk
        )

        for row in range(num_nodes):
            valid = [int(v) for v in merged[row] if int(v) >= 0]
            first_neg = merged[row].tolist().index(-1)
            self.assertEqual(first_neg, len(valid), f"row {row} -1 before a value")
            self.assertTrue(all(int(v) == -1 for v in merged[row, first_neg:]))

    def test_no_sorting_of_prefix(self):
        """SFA does not require ascending order; sorting would reorder ancestors."""
        num_nodes, topk, a, prefix_len = 4, 8, 4, 9
        ids = _ancestor_ids(_visibility_from_parents(CHAIN_PARENTS[:4], num_nodes), a)
        # Deliberately descending prefix scores -> descending indices.
        prefix = torch.tensor([[3, 2, 1, 0]], dtype=torch.int32).repeat(num_nodes, 1)

        merged = merge_tree_sparse_indices(
            prefix, ids, torch.tensor([prefix_len], dtype=torch.int32), num_nodes, topk
        )

        for row in range(num_nodes):
            n_anc = int((ids[row] >= 0).sum())
            self.assertTrue(
                torch.equal(merged[row, n_anc : n_anc + prefix.shape[1]], prefix[row])
            )
            self.assertNotEqual(merged[row].tolist(), sorted(merged[row].tolist()))

    def test_ancestor_out_of_range_raises(self):
        num_nodes, topk, a = 4, 8, 4
        draft = torch.zeros((num_nodes, a), dtype=torch.int32)
        draft[:, 0] = torch.arange(num_nodes)  # self, so no all-(-1) row
        draft[0, 0] = num_nodes + 3  # outside [0, N)
        draft[0, 1:] = -1
        prefix = torch.zeros((num_nodes, topk - a), dtype=torch.int32)
        with self.assertRaises(ValueError):
            merge_tree_sparse_indices(
                prefix, draft, torch.tensor([5], dtype=torch.int32), num_nodes, topk
            )

    def test_width_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            merge_tree_sparse_indices(
                prefix_indices=torch.zeros((1, 4), dtype=torch.int32),
                draft_ids=torch.zeros((1, 4), dtype=torch.int32),
                seq_lens=torch.tensor([3], dtype=torch.int32),
                num_verify_tokens=7,
                topk=16,
            )

    def test_batched_varying_prefix_lens(self):
        num_nodes, topk, a = 7, 16, 4
        allowed = _visibility_from_parents(TREE_PARENTS, num_nodes)
        ids = draft_ancestors(allowed.unsqueeze(0).expand(3, -1, -1), a)
        prefix_lens = [4, 0, 130]
        # P == 0 leaves nothing to pick: the indexer pads the whole block with -1.
        blocks = []
        for p in prefix_lens:
            if p == 0:
                block = torch.full((topk - a,), -1, dtype=torch.int32)
            else:
                block = torch.arange(topk - a, dtype=torch.int32) % p
            blocks.extend([block] * num_nodes)
        prefix = torch.stack(blocks)

        merged = merge_tree_sparse_indices(
            prefix, ids.reshape(-1, a), torch.tensor(prefix_lens), num_nodes, topk
        )

        self.assertEqual(tuple(merged.shape), (3 * num_nodes, topk))
        for req, p in enumerate(prefix_lens):
            for row in range(num_nodes):
                flat_row = merged[req * num_nodes + row]
                valid = [int(v) for v in flat_row if int(v) >= 0]
                n_anc = int((ids[req, row] >= 0).sum())
                picked = sorted(int(x) for x in ids[req, row] if int(x) >= 0)
                self.assertEqual(
                    sorted(valid[:n_anc]),
                    [p + v for v in picked],
                    f"req {req} row {row} ancestors",
                )
                self.assertTrue(all(v < p for v in valid[n_anc:]), "prefix must be < P")
                # -1 only after the last valid entry.
                as_list = flat_row.tolist()
                first_neg = as_list.index(-1) if -1 in as_list else topk
                self.assertEqual(first_neg, len(valid), f"req {req} row {row}")


    def test_end_to_end_mask_to_row_matches_parent_oracle(self):
        """parents -> FULL_MASK buffer -> gather -> ancestors -> row.

        This is the exact offline pipeline: the NPU tree kernel writes the
        FULL_MASK layout, and the indexer only reads it back.
        """
        num_nodes, topk, a, prefix_len = 7, 16, 4, 12
        for parents, nodes in ((TREE_PARENTS, 7), (CHAIN_PARENTS[:4], 4)):
            vis = _visibility_from_parents(parents, nodes)
            flat = _make_full_mask([vis], [prefix_len], nodes)

            got = gather_tree_blocks(flat, torch.tensor([prefix_len]), nodes)
            ids = draft_ancestors(got, a)
            prefix = _prefix_rows(nodes, topk - a, prefix_len)
            merged = merge_tree_sparse_indices(
                prefix, ids[0], torch.tensor([prefix_len], dtype=torch.int32),
                nodes, topk,
            )

            for row in range(nodes):
                # Expected: {self} | ancestors, as absolute positions.
                expected = {
                    prefix_len + c
                    for c in range(nodes)
                    if vis[row, c]
                }
                valid = [int(v) for v in merged[row] if int(v) >= 0]
                anc = valid[: len(expected)]

                self.assertEqual(set(anc), expected, f"{parents} row {row}")
                self.assertEqual(len(anc), len(set(anc)), "no ancestor repeats")
                self.assertEqual(
                    len(valid), len(expected) + prefix.shape[1], "prefix intact"
                )
                self.assertTrue(all(v < prefix_len for v in valid[len(expected):]))


class TestDraftStepAncestorPositions(unittest.TestCase):
    """Draft-stage ancestor paths (D3). Slots are branch-major:

    branch t owns [prefix + t * num_steps, + num_steps), so the ancestor at
    level d of a chain that passed branch t_d sits at prefix + t_d * num_steps + d.
    Positions therefore do not ascend across levels; uniqueness is what matters.
    """

    TOPK, STEPS, PREFIX = 2, 3, 10

    def _call(self, paths, parent_slot, step):
        return draft_step_ancestor_positions(
            paths=paths,
            parent_slot=parent_slot,
            prefix_len=torch.tensor([self.PREFIX]),
            step=step,
            topk=self.TOPK,
            num_steps=self.STEPS,
        )

    def test_step0_has_only_self(self):
        paths, pos = self._call(None, None, 0)

        self.assertEqual(tuple(paths.shape), (1, self.TOPK, 0))
        self.assertEqual(tuple(pos.shape), (1, self.TOPK, 1))
        # Branch t starts at prefix + t * num_steps.
        self.assertEqual(pos.tolist(), [[[10], [13]]])

    def test_step1_links_to_level0(self):
        paths, pos = self._call(torch.zeros((1, 2, 0), dtype=torch.int32),
                                torch.tensor([[0, 1]]), 1)

        self.assertEqual(paths.tolist(), [[[0], [1]]])
        # ancestor(level0) then self(level1) per branch.
        self.assertEqual(pos.tolist(), [[[10, 11], [13, 14]]])

    def test_step2_walks_the_chain(self):
        paths, pos = self._call(torch.tensor([[[0], [1]]], dtype=torch.int32),
                                torch.tensor([[1, 0]]), 2)

        self.assertEqual(paths.tolist(), [[[1, 1], [0, 0]]])
        self.assertEqual(pos.tolist(), [[[13, 14, 12], [10, 11, 15]]])

    def test_per_request_prefix_len(self):
        _, pos = draft_step_ancestor_positions(
            paths=torch.zeros((2, 2, 0), dtype=torch.int32),
            parent_slot=torch.tensor([[0, 1], [1, 0]]),
            prefix_len=torch.tensor([10, 20]),
            step=1,
            topk=2,
            num_steps=3,
        )

        self.assertEqual(pos[0].tolist(), [[10, 11], [13, 14]])
        # Request 1: branch base 20, so ancestor of branch t sits at 20 + 3t.
        self.assertEqual(pos[1].tolist(), [[23, 21], [20, 24]])

    def test_shape_uniqueness_and_window(self):
        torch.manual_seed(0)
        topk, steps, prefix = 3, 4, 17
        paths, parent = None, None
        for step in range(steps):
            if step > 0:
                parent = torch.randint(0, topk, (1, topk), dtype=torch.int32)
            paths, pos = draft_step_ancestor_positions(
                paths=paths,
                parent_slot=parent,
                prefix_len=torch.tensor([prefix]),
                step=step,
                topk=topk,
                num_steps=steps,
            )
            self.assertEqual(tuple(pos.shape), (1, topk, step + 1))
            window = prefix + (topk - 1) * steps + step + 1
            for j in range(topk):
                row = pos[0, j].tolist()
                self.assertEqual(len(row), step + 1)
                self.assertEqual(len(set(row)), len(row), "no repeated slot")
                self.assertEqual(row[-1], prefix + j * steps + step)
                self.assertTrue(all(prefix <= v < window for v in row))

    def test_rejects_bad_parent_slot(self):
        with self.assertRaises(ValueError):
            self._call(torch.zeros((1, 2, 0), dtype=torch.int32),
                       torch.tensor([[0, 4]]), 1)

    def test_rejects_parent_at_step0(self):
        with self.assertRaises(ValueError):
            self._call(None, torch.tensor([[0, 1]]), 0)

    def test_rejects_negative_prefix(self):
        with self.assertRaises(ValueError):
            draft_step_ancestor_positions(
                paths=None, parent_slot=None, prefix_len=torch.tensor([-1]),
                step=0, topk=2, num_steps=3,
            )


class TestDraftTreeSparseState(unittest.TestCase):
    TOPK, STEPS, PREFIX = 2, 3, 10

    def _batch(self, prefix=None):
        import types

        return types.SimpleNamespace(
            seq_lens=torch.tensor([self.PREFIX if prefix is None else prefix])
        )

    def test_accumulates_across_steps_and_stashes_positions(self):
        state = DraftTreeSparseState(self.TOPK, self.STEPS)
        fb = self._batch()
        parents_list = [torch.arange(-1, self.TOPK).expand(1, self.TOPK + 1)]

        pos0 = state.begin_step(0, fb, parents_list)
        self.assertEqual(pos0.tolist(), [[[10], [13]]])
        self.assertIs(getattr(fb, DRAFT_TREE_ANCESTORS_ATTR), pos0)
        self.assertEqual(getattr(fb, DRAFT_TREE_PREFIX_ATTR).tolist(), [10])

        # step 1: flat [0, 3] -> parent slots [0, 1], offset 2
        parents_list.append(torch.tensor([[0, 3]]) + 2)
        pos1 = state.begin_step(1, fb, parents_list)
        self.assertEqual(pos1.tolist(), [[[10, 11], [13, 14]]])

        # step 2: flat [2, 1] -> parent slots [1, 0], offset 6
        parents_list.append(torch.tensor([[2, 1]]) + 6)
        pos2 = state.begin_step(2, fb, parents_list)
        self.assertEqual(pos2.tolist(), [[[13, 14, 12], [10, 11, 15]]])

    def test_rejects_a_parent_row_without_the_step_offset(self):
        state = DraftTreeSparseState(2, 2)
        parents = [torch.zeros((1, 3)), torch.tensor([[0, 1]])]
        with self.assertRaises(ValueError):
            state.begin_step(1, self._batch(), parents)

    def test_rejects_bad_step(self):
        state = DraftTreeSparseState(2, 2)
        with self.assertRaises(ValueError):
            state.begin_step(2, self._batch(), [torch.zeros((1, 3))])
        with self.assertRaises(ValueError):
            state.begin_step(0, self._batch(), [])


if __name__ == "__main__":
    unittest.main()
