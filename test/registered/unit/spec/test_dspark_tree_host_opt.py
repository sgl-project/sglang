"""Characterization tests for the DSpark tree host-planning path.

These pin the observable host outputs before the host-loop vectorization work.
They intentionally construct the expected values independently of the
production implementation so the same tests can guard the refactor.
"""

import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(
    0,
    str(
        _ROOT
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
        / "dspark_components"
    ),
)

from dspark_tree import (  # noqa: E402
    TreeLayout,
    build_tree_metadata,
    build_tree_ori_sparse_indices,
    finalize_tree_layout_torch,
    plan_tree_commit,
)


def _layout_buf(batch: int, budget: int) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((batch, budget), -1, dtype=torch.long),
        depths=torch.zeros((batch, budget), dtype=torch.long),
        parents=torch.full((batch, budget), -1, dtype=torch.long),
        num_nodes=torch.zeros(batch, dtype=torch.int32),
        visibility=torch.zeros((batch, budget, budget), dtype=torch.bool),
        first_child=torch.full((batch, budget + 1), -1, dtype=torch.long),
        next_sibling=torch.full((batch, budget + 1), -1, dtype=torch.long),
    )


def _two_request_layout(padded: bool = False) -> TreeLayout:
    if padded:
        return finalize_tree_layout_torch(
            _layout_buf(2, 4),
            tokens=torch.tensor([[5, 6, -1, -1], [9, 10, -1, -1]]),
            depths=torch.tensor([[1, 2, 0, 0], [1, 2, 0, 0]]),
            parent_ids=torch.tensor([[0, 1, -1, -1], [0, 1, -1, -1]]),
            num_nodes=2,
        )
    return finalize_tree_layout_torch(
        _layout_buf(2, 4),
        tokens=torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]),
        depths=torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]]),
        parent_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        num_nodes=4,
    )


def _reference_ancestor_row(layout, req: int, node_id: int, max_depth: int):
    if node_id <= 0:
        row = [0]
    else:
        row = []
        cur = node_id
        while cur > 0:
            row.append(int(cur))
            cur = int(layout.parents[req, cur - 1])
        row.append(0)
        row.reverse()
    row.extend([-1] * (max_depth + 1 - len(row)))
    return row


def _reference_sparse(
    layout,
    *,
    prefix_lens,
    block_table,
    block_size,
    window,
    budget,
    full_to_swa_mapping=None,
    draft_slots=None,
):
    batch, _ = layout.tokens.shape
    query_width = budget + 1
    width = ((window + budget + 127) // 128) * 128
    slots = torch.full(
        (batch * query_width, 1, width), -1, dtype=torch.int32
    )
    for req in range(batch):
        prefix = int(prefix_lens[req])
        for node_id in range(query_width):
            depth = 0 if node_id == 0 else int(layout.depths[req, node_id - 1])
            positions = list(range(max(0, prefix + depth - window), prefix))
            row = _reference_ancestor_row(
                layout, req, node_id, int(layout.depths.max())
            )
            positions.extend(prefix + ancestor for ancestor in row if ancestor >= 0)
            seen = set()
            col = 0
            for pos in positions:
                if pos in seen:
                    continue
                seen.add(pos)
                if col >= width:
                    raise ValueError("row overflow")
                if draft_slots is not None and pos >= prefix:
                    node = pos - prefix
                    full_slot = int(draft_slots[req, node])
                else:
                    page = int(block_table[req, pos // block_size])
                    full_slot = page * block_size + (pos % block_size)
                if full_to_swa_mapping is not None:
                    full_slot = int(full_to_swa_mapping[full_slot])
                slots[req * query_width + node_id, 0, col] = full_slot
                col += 1
    return slots


def test_metadata_ancestors_multi_request_and_padding():
    meta = build_tree_metadata(
        _two_request_layout(),
        prefix_lens=torch.tensor([10, 20]),
        anchor_tokens=torch.tensor([100, 200]),
    )
    assert meta.tree_ancestors.tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 2, -1],
        [0, 1, 3],
        [0, 1, 4],
        [0, -1, -1],
        [0, 1, -1],
        [0, 2, -1],
        [0, 1, 3],
        [0, 1, 4],
    ]

    padded = build_tree_metadata(
        _two_request_layout(padded=True),
        prefix_lens=torch.tensor([10, 20]),
        anchor_tokens=torch.tensor([100, 200]),
    )
    assert padded.tree_ancestors.tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [0, 3, -1],
        [0, 4, -1],
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [0, 3, -1],
        [0, 4, -1],
    ]


def test_sparse_indices_match_reference_with_draft_slots_and_mapping():
    layout = _two_request_layout()
    prefix_lens = torch.tensor([3, 5])
    block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]
    )
    draft_slots = torch.tensor(
        [
            [100, 101, 102, 103, 104],
            [200, 201, 202, 203, 204],
        ]
    )
    mapping = torch.arange(300, dtype=torch.long)

    expected = _reference_sparse(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=1,
        window=3,
        budget=4,
        full_to_swa_mapping=mapping,
        draft_slots=draft_slots,
    )
    actual = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=1,
        window=3,
        budget=4,
        full_to_swa_mapping=mapping,
        draft_slots=draft_slots,
    )
    assert torch.equal(actual, expected), (actual, expected)


def test_sparse_indices_match_reference_without_mapping():
    layout = _two_request_layout()
    prefix_lens = torch.tensor([3, 7])
    block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ]
    )

    expected = _reference_sparse(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=4,
        window=4,
        budget=4,
    )
    actual = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=prefix_lens,
        block_table=block_table,
        block_size=4,
        window=4,
        budget=4,
    )
    assert torch.equal(actual, expected), (actual, expected)


def test_sparse_indices_randomized_equivalence_and_optional_ancestors():
    torch.manual_seed(0)
    layout = _two_request_layout()
    budget = 4
    for _ in range(20):
        prefix_lens = torch.randint(0, 21, (2,))
        block_size = int(torch.randint(1, 5, (1,)).item())
        window = int(torch.randint(1, 8, (1,)).item())
        block_table = torch.randint(0, 64, (2, 64))
        draft_slots = torch.randint(0, 1000, (2, budget + 1))
        mapping = torch.arange(2000)

        expected = _reference_sparse(
            layout,
            prefix_lens=prefix_lens,
            block_table=block_table,
            block_size=block_size,
            window=window,
            budget=budget,
            full_to_swa_mapping=mapping,
            draft_slots=draft_slots,
        )
        actual = build_tree_ori_sparse_indices(
            layout,
            prefix_lens=prefix_lens,
            block_table=block_table,
            block_size=block_size,
            window=window,
            budget=budget,
            full_to_swa_mapping=mapping,
            draft_slots=draft_slots,
        )
        assert torch.equal(actual, expected), (prefix_lens, block_size, window)

        meta = build_tree_metadata(
            layout,
            prefix_lens=prefix_lens,
            anchor_tokens=torch.tensor([100, 200]),
        )
        with_ancestors = build_tree_ori_sparse_indices(
            layout,
            prefix_lens=prefix_lens,
            block_table=block_table,
            block_size=block_size,
            window=window,
            budget=budget,
            full_to_swa_mapping=mapping,
            draft_slots=draft_slots,
            ancestors=meta.tree_ancestors.reshape(2, budget + 1, -1),
        )
        assert torch.equal(with_ancestors, expected)


def test_commit_plan_multi_request():
    plan = plan_tree_commit(
        path_node_ids=torch.tensor(
            [
                [1, 3, -1, -1],
                [2, 4, 1, -1],
            ]
        ),
        verify_cache_loc=torch.tensor(
            [
                [100, 101, 102, 103, 104],
                [200, 201, 202, 203, 204],
            ]
        ),
        prefix_lens=torch.tensor([10, 20]),
        block_table=torch.tensor([[0], [1]]),
        block_size=4,
    )
    assert plan.source_rows.tolist() == [
        [100, 101, 103, -1],
        [200, 202, 204, 201],
    ]
    assert plan.dest_positions.tolist() == [
        [10, 11, 12, -1],
        [20, 21, 22, 23],
    ]
    assert plan.dest_slots.tolist() == [
        [100, 101, 102, -1],
        [200, 201, 202, 203],
    ]
    assert plan.commit_lens.tolist() == [3, 4]
    assert plan.new_seq_lens.tolist() == [13, 24]
