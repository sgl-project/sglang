"""A sliding-window layer's read of each request, and the mask a verify tree
is planned with over it.

Every attention mode takes its window read from ``window_read``: an extend, a
target verify and a draft extend read the last ``window`` tokens before their
first query, through their last. A verify tree's rows sit at their tree depth,
not their row index, so its window is folded into the tree's mask; each row
then sees exactly the keys a query at its position sees.
"""

import unittest

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import window_read
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _full_mask(seq_lens, trees):
    """A verify mask over each request's whole sequence: ``width`` rows of
    ``seq_len + width`` columns, prefix all visible, then the tree."""
    rows = []
    for seq_len, tree in zip(seq_lens, trees):
        width = tree.shape[0]
        rows.append(
            torch.cat(
                [torch.ones(width, seq_len, dtype=torch.bool), tree], dim=1
            ).flatten()
        )
    return torch.cat(rows)


def _expected(seq_lens, trees, window):
    """Each row's visible keys over the window read, from positions alone: a
    row at depth d queries position seq_len + d and sees prefix keys from
    seq_len + d - window on, and its tree ancestors."""
    rows = []
    for seq_len, tree in zip(seq_lens, trees):
        width = tree.shape[0]
        read = min(seq_len, window)
        start = seq_len - read
        depth = tree.sum(-1) - 1
        prefix = torch.stack(
            [
                torch.tensor(
                    [start + j >= seq_len + int(d) - window for j in range(read)]
                )
                for d in depth
            ]
        ).reshape(width, read)
        rows.append(torch.cat([prefix, tree], dim=1).flatten())
    return torch.cat(rows)


def _chain(width):
    return torch.tril(torch.ones(width, width, dtype=torch.bool))


def _branched():
    # Root; two children of the root; a child of the second child.
    parents = [-1, 0, 0, 2]
    tree = torch.zeros(4, 4, dtype=torch.bool)
    for row, parent in enumerate(parents):
        tree[row, row] = True
        while parent >= 0:
            tree[row, parent] = True
            parent = parents[parent]
    return tree


class TestWindowRead(CustomTestCase):
    def test_every_mode_reads_the_window_before_its_first_query(self):
        window = 8
        seq_lens = torch.tensor([3, 20, 40])
        # Target verify: the whole sequence is cached; the queries follow.
        verify = window_read(
            seq_lens=seq_lens, prefix_lens=seq_lens, window=window, prefix_only=False
        )
        self.assertEqual(verify.lens.tolist(), [3, 8, 8])
        self.assertEqual(verify.start.tolist(), [0, 12, 32])
        self.assertEqual(verify.window_left, window)
        # Draft extend or extend: two queries are the sequence's last tokens.
        extend = window_read(
            seq_lens=seq_lens,
            prefix_lens=seq_lens - 2,
            window=window,
            prefix_only=False,
        )
        self.assertEqual(extend.lens.tolist(), [3, 10, 10])
        self.assertEqual((extend.start + extend.lens).tolist(), seq_lens.tolist())
        self.assertEqual(extend.window_left, window)
        # A ragged extend's paged pass reads the prefix alone, masked by its
        # caller.
        ragged = window_read(
            seq_lens=seq_lens,
            prefix_lens=seq_lens - 2,
            window=window,
            prefix_only=True,
        )
        self.assertEqual(ragged.lens.tolist(), [1, 8, 8])
        self.assertEqual((ragged.start + ragged.lens).tolist(), [1, 18, 38])
        self.assertEqual(ragged.window_left, -1)

    def test_a_tree_mask_masks_each_row_to_its_own_window(self):
        """BUG REGRESSION. A verify over a window layer read from column 0,
        the sequence's first tokens; and a kernel's row-indexed window
        misplaces a branched tree's rows, which sit at their depth."""
        window = 8
        for name, tree in (("chain", _chain(4)), ("branched", _branched())):
            with self.subTest(tree=name):
                seq_lens = torch.tensor([3, 20, 8, 40])
                trees = [tree] * len(seq_lens)
                read = window_read(
                    seq_lens=seq_lens,
                    prefix_lens=seq_lens,
                    window=window,
                    prefix_only=False,
                )
                mask = read.tree_mask(
                    _full_mask(seq_lens.tolist(), trees),
                    width=tree.shape[0],
                    read_lens_sum=int(read.lens.sum()),
                )
                torch.testing.assert_close(
                    mask, _expected(seq_lens.tolist(), trees, window)
                )

    def test_tree_ancestors_are_clipped_by_position(self):
        """Verify excludes ancestors before the window without admitting
        siblings or treating tree row indices as token positions."""
        for name, parents in (
            ("chain", [-1, 0, 1, 2, 3, 4, 5]),
            ("branched", [-1, 0, 0, 2, 1, 3, 5]),
        ):
            paths = []
            for node, parent in enumerate(parents):
                paths.append([node] if parent == -1 else paths[parent] + [node])
            width = len(paths)
            tree = torch.tensor(
                [[key in path for key in range(width)] for path in paths]
            )
            for window in (2, 8):
                with self.subTest(tree=name, window=window):
                    seq_lens = torch.tensor([0, 3, 20])
                    read = window_read(
                        seq_lens=seq_lens,
                        prefix_lens=seq_lens,
                        window=window,
                        prefix_only=False,
                    )
                    mask = read.tree_mask(
                        _full_mask(seq_lens.tolist(), [tree] * len(seq_lens)),
                        width=width,
                        read_lens_sum=int(read.lens.sum()),
                    )
                    expected = []
                    for seq_len in seq_lens.tolist():
                        prefix_positions = list(
                            range(max(0, seq_len - window), seq_len)
                        )
                        key_positions = prefix_positions + [
                            seq_len + len(path) - 1 for path in paths
                        ]
                        for path in paths:
                            query_position = seq_len + len(path) - 1
                            causal = [True] * len(prefix_positions) + [
                                key in path for key in range(width)
                            ]
                            expected.extend(
                                visible and query_position - window <= position
                                for visible, position in zip(causal, key_positions)
                            )
                    torch.testing.assert_close(mask, torch.tensor(expected))

    def test_rows_padded_past_the_mask_do_not_index_it(self):
        """A graph replay pads the batch past the live requests, whose mask
        holds no rows for them; the live rows are laid out unchanged."""
        window, tree = 8, _branched()
        live = [20, 40]
        seq_lens = torch.tensor(live + [1, 1])
        read = window_read(
            seq_lens=seq_lens, prefix_lens=seq_lens, window=window, prefix_only=False
        )
        mask = read.tree_mask(
            _full_mask(live, [tree] * 2),
            width=4,
            read_lens_sum=int(read.lens.sum()),
        )
        expected = _expected(live, [tree] * 2, window)
        torch.testing.assert_close(mask[: expected.numel()], expected)
        self.assertEqual(mask.numel(), expected.numel() + 2 * 4 * (1 + 4))


if __name__ == "__main__":
    unittest.main()
