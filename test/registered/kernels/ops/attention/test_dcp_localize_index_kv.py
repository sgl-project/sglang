from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_local_capacity,
    dcp_localize_page_table,
    dcp_pack_local_to_global,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _reference(
    page_table_1: torch.Tensor, dcp_size: int, dcp_rank: int, page_size: int
):
    """Naive per-row Python loop with PAGE-level ownership: the ground truth
    this derivation must match. Page-level (not per-token) ownership is
    required for the compacted table to preserve physical page contiguity --
    see dcp_localize_index_kv's module docstring."""
    num_rows, max_len = page_table_1.shape
    local_causal_count = torch.zeros_like(page_table_1, dtype=torch.int32)
    local_page_table_rows = []
    local_to_global_rows = []
    for row in range(num_rows):
        count = 0
        compacted_local = []
        compacted_global = []
        for i in range(max_len):
            slot = int(page_table_1[row, i])
            owned = slot >= 0 and (slot // page_size) % dcp_size == dcp_rank
            if owned:
                count += 1
                page_id = slot // page_size
                local_page_id = page_id // dcp_size
                local_slot = local_page_id * page_size + slot % page_size
                compacted_local.append(local_slot)
                compacted_global.append(slot)
            local_causal_count[row, i] = count
        local_page_table_rows.append(compacted_local)
        local_to_global_rows.append(compacted_global)
    return local_causal_count, local_page_table_rows, local_to_global_rows


def _check(
    page_table_1: torch.Tensor, dcp_size: int, dcp_rank: int, page_size: int
) -> None:
    max_len = page_table_1.shape[1]
    capacity = dcp_local_capacity(max_len, dcp_size, page_size)
    local_page_table, local_to_global, local_causal_count = dcp_localize_page_table(
        page_table_1, dcp_size, dcp_rank, capacity, page_size
    )
    ref_causal_count, ref_local_rows, ref_global_rows = _reference(
        page_table_1, dcp_size, dcp_rank, page_size
    )

    torch.testing.assert_close(local_causal_count, ref_causal_count)
    for row in range(page_table_1.shape[0]):
        n = len(ref_local_rows[row])
        assert local_page_table[row, :n].tolist() == ref_local_rows[row]
        assert (local_page_table[row, n:] == -1).all()
        assert local_to_global[row, :n].tolist() == ref_global_rows[row]
        assert (local_to_global[row, n:] == -1).all()
        for t in range(page_table_1.shape[1]):
            local_len_t = int(local_causal_count[row, t])
            owned_upto_t = []
            for s in page_table_1[row, : t + 1].tolist():
                if s >= 0 and (s // page_size) % dcp_size == dcp_rank:
                    page_id = s // page_size
                    owned_upto_t.append(
                        (page_id // dcp_size) * page_size + s % page_size
                    )
            assert local_page_table[row, :local_len_t].tolist() == owned_upto_t


def _assert_windows_are_contiguous_pages(
    local_page_table: torch.Tensor, page_size: int
) -> None:
    """The property that actually broke in production: every page_size-row
    window of the compacted local table must be one physically contiguous
    page (block_tables extraction -- `local_page_table[:, ::page_size] //
    page_size` -- silently reads garbage from unrelated physical rows
    otherwise, since the paged-MQA-logits kernel treats each such window as
    a single page)."""
    num_windows = local_page_table.shape[1] // page_size
    for row in range(local_page_table.shape[0]):
        for k in range(num_windows):
            window = local_page_table[row, k * page_size : (k + 1) * page_size]
            if (window < 0).all():
                continue
            assert (window >= 0).all(), (
                f"row {row} window {k}: partially-filled page, expected "
                f"either fully -1 or fully valid -- {window.tolist()}"
            )
            expected = torch.arange(
                window[0].item(), window[0].item() + page_size, dtype=window.dtype
            )
            assert torch.equal(window, expected), (
                f"row {row} window {k} is NOT one contiguous physical page: "
                f"{window.tolist()} (expected {expected.tolist()})"
            )


def test_localize_matches_reference_for_interleaved_slots() -> None:
    # Slot values interleave ownership every page (dcp_size=3 round robin
    # over pages), the common case when the allocator hands out pages in
    # position order.
    page_size = 4
    page_table_1 = (
        torch.arange(10 * page_size, dtype=torch.int32).unsqueeze(0).repeat(2, 1)
    )
    page_table_1[1] += 100 * page_size
    for rank in range(3):
        _check(page_table_1, dcp_size=3, dcp_rank=rank, page_size=page_size)


def test_localize_matches_reference_for_non_contiguous_pages() -> None:
    # Pages are NOT physically adjacent to each other (models a paged
    # allocator: each page is internally contiguous, but pages can be
    # anywhere -- that's the whole point of block_tables indirection). This
    # is the exact scenario that broke under per-token ownership.
    torch.manual_seed(0)
    page_size = 64
    dcp_size = 4
    num_pages = 8
    page_ids = torch.randperm(1000)[:num_pages]
    page_table_1 = torch.cat(
        [
            torch.arange(
                pid.item() * page_size,
                pid.item() * page_size + page_size,
                dtype=torch.int32,
            )
            for pid in page_ids
        ]
    ).unsqueeze(0)
    page_table_1 = torch.cat([page_table_1, page_table_1.flip(dims=[1])], dim=0)
    for rank in range(dcp_size):
        _check(page_table_1, dcp_size=dcp_size, dcp_rank=rank, page_size=page_size)


def test_localize_compacted_windows_are_contiguous_pages_with_non_adjacent_input() -> (
    None
):
    # Regression test for the production accuracy bug: verifies the
    # contiguity property itself (not just the value/set correctness
    # _check already covers), against page-table input whose global pages
    # are deliberately scattered/non-adjacent.
    torch.manual_seed(1)
    page_size = 64
    for dcp_size in (2, 3, 4):
        num_pages = 12
        page_ids = torch.randperm(2000)[:num_pages]
        page_table_1 = torch.cat(
            [
                torch.arange(
                    pid.item() * page_size,
                    pid.item() * page_size + page_size,
                    dtype=torch.int32,
                )
                for pid in page_ids
            ]
        ).unsqueeze(0)
        for rank in range(dcp_size):
            capacity = dcp_local_capacity(page_table_1.shape[1], dcp_size, page_size)
            local_page_table, _, _ = dcp_localize_page_table(
                page_table_1, dcp_size, rank, capacity, page_size
            )
            _assert_windows_are_contiguous_pages(local_page_table, page_size)


def test_localize_handles_padding() -> None:
    page_size = 2
    page_table_1 = torch.full((2, 10), -1, dtype=torch.int32)
    page_table_1[0, :6] = torch.arange(6, dtype=torch.int32)
    page_table_1[1, :3] = torch.arange(3, dtype=torch.int32) * 2
    for rank in range(2):
        _check(page_table_1, dcp_size=2, dcp_rank=rank, page_size=page_size)


def test_localize_dcp_size_one_is_identity() -> None:
    page_table_1 = torch.arange(20, dtype=torch.int32).unsqueeze(0)
    local_page_table, local_to_global, local_causal_count = dcp_localize_page_table(
        page_table_1, dcp_size=1, dcp_rank=0, local_capacity=20, page_size=4
    )
    assert torch.equal(local_page_table, page_table_1)
    assert torch.equal(local_to_global, page_table_1)
    assert torch.equal(
        local_causal_count, torch.arange(1, 21, dtype=torch.int32).unsqueeze(0)
    )


def test_local_capacity_is_safe_under_arbitrary_page_distribution() -> None:
    # Regression test for a real bug: with physically scattered (arbitrary,
    # non-sequential) global pages -- the normal case, since pages need not
    # be mutually adjacent -- ownership does NOT divide evenly across ranks
    # by page count. A rank can by chance own far more than num_pages/dcp_size
    # pages (verified below with a concrete seed where one rank owns half of
    # all 8 pages), so capacity must not assume a balanced split.
    torch.manual_seed(0)
    page_size = 64
    dcp_size = 4
    num_pages = 8
    max_len = num_pages * page_size
    page_ids = torch.randperm(1000)[:num_pages]
    page_table_1 = torch.cat(
        [
            torch.arange(
                pid.item() * page_size,
                pid.item() * page_size + page_size,
                dtype=torch.int32,
            )
            for pid in page_ids
        ]
    ).unsqueeze(0)

    capacity = dcp_local_capacity(max_len, dcp_size, page_size)
    # Must be large enough for a rank owning EVERY page (the true worst case),
    # not just ceil(num_pages/dcp_size).
    assert capacity >= num_pages * page_size

    owned_pages_per_rank = [
        int(((page_ids % dcp_size) == rank).sum()) for rank in range(dcp_size)
    ]
    assert max(owned_pages_per_rank) > -(-num_pages // dcp_size), (
        "test fixture no longer demonstrates uneven ownership -- "
        f"{owned_pages_per_rank}"
    )

    for rank in range(dcp_size):
        local_page_table, _, local_causal_count = dcp_localize_page_table(
            page_table_1, dcp_size, rank, capacity, page_size
        )
        assert int(local_causal_count[0, -1]) == owned_pages_per_rank[rank] * page_size
        _assert_windows_are_contiguous_pages(local_page_table, page_size)


def _reference_packed(
    page_table_1: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    seq_lens: torch.Tensor,
    page_size: int,
) -> list[int]:
    expected = []
    for row in range(page_table_1.shape[0]):
        sl = int(seq_lens[row])
        for i in range(sl):
            slot = int(page_table_1[row, i])
            if slot >= 0 and (slot // page_size) % dcp_size == dcp_rank:
                expected.append(slot)
    return expected


def test_pack_local_to_global_matches_reference_ragged_layout() -> None:
    # Mirrors the extend/chunked-prefill ragged K layout: each row (request)
    # packs only its own causal-bounded owned slots into one flat buffer at
    # its own offset, unlike decode's fixed-width per-row layout.
    torch.manual_seed(3)
    page_size = 4
    num_rows, max_len, dcp_size = 4, 50, 3
    page_table_1 = torch.stack(
        [torch.randperm(max_len, dtype=torch.int32) for _ in range(num_rows)]
    )
    seq_lens = torch.tensor([10, 30, 50, 5], dtype=torch.int32)
    capacity = dcp_local_capacity(max_len, dcp_size, page_size)

    for rank in range(dcp_size):
        _, _, local_causal_count = dcp_localize_page_table(
            page_table_1, dcp_size, rank, capacity, page_size
        )
        row_totals = torch.gather(
            local_causal_count, 1, (seq_lens.long() - 1).clamp(min=0).unsqueeze(1)
        ).squeeze(1)
        row_totals = torch.where(seq_lens > 0, row_totals, torch.zeros_like(row_totals))
        row_offsets = (torch.cumsum(row_totals, dim=0) - row_totals).to(torch.int32)
        total_size = int(row_totals.sum().item())

        packed = dcp_pack_local_to_global(
            page_table_1, dcp_size, rank, seq_lens, row_offsets, total_size, page_size
        )
        expected = _reference_packed(page_table_1, dcp_size, rank, seq_lens, page_size)
        assert packed.tolist() == expected


def test_pack_local_to_global_respects_row_causal_bound() -> None:
    # A row's causal length can be shorter than the page table's padded
    # width; entries beyond it must never leak into the packed output even
    # if they'd otherwise be "owned".
    page_size = 2
    page_table_1 = torch.arange(10, dtype=torch.int32).unsqueeze(0)  # row of [0..9]
    seq_lens = torch.tensor([4], dtype=torch.int32)  # only positions 0..3 in-window
    dcp_size, rank = 2, 0
    row_offsets = torch.tensor([0], dtype=torch.int32)
    packed = dcp_pack_local_to_global(
        page_table_1,
        dcp_size,
        rank,
        seq_lens,
        row_offsets,
        total_size=2,
        page_size=page_size,
    )
    # page_size=2: pages are {0,1},{2,3},{4,5},...; page%dcp_size==0 -> pages
    # 0,2,4,...; within causal window [0,4): pages 0 ({0,1}) and 1 ({2,3}) --
    # page 0 owned (0%2==0), page 1 not (1%2==1). So only slots 0,1 qualify.
    assert packed.tolist() == [0, 1]
