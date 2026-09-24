"""The DeepSeek-V4.1 decode glue on HIP: single-launch replacements must be bitwise the torch chains they replace, and the two-level top-k must select the reference's blocks and positions without reading past a row's reach."""

from __future__ import annotations

import random
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_index_page_table,
    low_ratio_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import sort_selection_rows
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _low_ratio_compression_metadata,
)
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    expand_index_page_table as _expand_index_page_table,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    CandidateBlocks,
    _aot_topk_sorts_output,
    _extend_k_slots,
    topk_transform_paged_sorted,
    topk_within_candidate_blocks_hip,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-kernel-test-1-gpu-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="HIP decode glue for the AITER / FlyDSL DeepSeek-V4.1 path (gfx95x).",
)

DEVICE = "cuda"

INDEX_PAGE_SIZE = 64
TOPK = 512
# Released V4.1 config; the span 2048 x 8 = 16384 is where level one starts to bind.
TOPK_BLOCKS, BLOCK_SIZE = 2048, 8


def index_slots(page_table, pos):
    """Slot of compressed position pos through the expanded indexer page table."""
    return (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )


def reference_position_mask(logits, lens, topk_blocks, block_size):
    """The reference's level one on logits whose tail past the reach is -inf."""
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    col = torch.arange(logits.shape[1], device=logits.device)
    pre = logits.masked_fill(col >= lens[:, None], -torch.inf)
    return select_candidate_blocks(
        pre, lens[:, None], topk_blocks=topk_blocks, block_size=block_size
    )


def candidate_block_ids_to_mask(ids, num_blocks):
    """bool [rows, num_blocks] block mask of CandidateBlocks.ids."""
    rows = ids.shape[0]
    # column num_blocks is the sink for the -1 padding; the mask is the view before it
    keep = torch.zeros((rows, num_blocks + 1), dtype=torch.bool, device=ids.device)
    keep.scatter_(1, ids.masked_fill(ids < 0, num_blocks).to(torch.int64), True)
    return keep[:, :num_blocks]


def ids_to_position_mask(ids, block_size, width):
    num_blocks = (width + block_size - 1) // block_size
    keep = candidate_block_ids_to_mask(ids, num_blocks)
    return keep.repeat_interleave(block_size, dim=-1)[:, :width]


def reference_consumer_rows(logits, lens, pos_mask, topk):
    """Per row: the set the reference consumer selects among the reachable candidates
    and its valid count; tie-free logits make it exact."""
    col = torch.arange(logits.shape[1], device=logits.device)
    out = []
    for b in range(logits.shape[0]):
        n = int(lens[b])
        cand = pos_mask[b] & (col < n)
        n_cand = int(cand.sum())
        k = min(topk, n_cand)
        s = logits[b].masked_fill(~cand, -torch.inf)
        out.append((set(s.topk(k).indices.tolist()), k))
    return out


class TestTwoLevelDecodeHip(CustomTestCase):
    """Level one of the two-level top-k: the source keeps TOPK_BLOCKS x BLOCK_SIZE positions, later ratio-1 sources select inside them."""

    def _garbage_tail(self, logits, lens):
        """Kernel garbage past each row's reach (large positives on even rows, NaN on
        odd), so a helper reading the tail fails loudly."""
        col = torch.arange(logits.shape[1], device=logits.device)
        tail = col[None, :] >= lens[:, None]
        odd = (torch.arange(logits.shape[0], device=logits.device) % 2 == 1)[:, None]
        logits = logits.masked_fill(tail & ~odd, 1e4)
        return logits.masked_fill(tail & odd, torch.nan)

    def _assert_consumer_matches_reference(self, logits, seq, cands, page_table, msg):
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            topk_within_candidate_blocks_hip,
        )

        rows, width = logits.shape
        page_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        raw_indices = torch.full((rows, TOPK), 7, dtype=torch.int32, device="cuda")
        topk_within_candidate_blocks_hip(
            logits,
            seq,
            cands,
            page_table=page_table,
            page_size=INDEX_PAGE_SIZE,
            page_indices=page_indices,
            raw_indices=raw_indices,
        )
        pos_mask = ids_to_position_mask(cands.ids, cands.block_size, width)
        for b, (want, k) in enumerate(
            reference_consumer_rows(logits, seq, pos_mask, TOPK)
        ):
            got = raw_indices[b]
            self.assertTrue(bool((got[:k] >= 0).all()), f"{msg}: prefix row {b}")
            self.assertTrue(bool((got[k:] == -1).all()), f"{msg}: padding row {b}")
            self.assertEqual(set(got[:k].tolist()), want, f"{msg}: selection row {b}")
            sel = got[:k].to(torch.int64)
            expect = index_slots(page_table[b : b + 1], sel[None])[0]
            self.assertTrue(
                torch.equal(page_indices[b, :k].to(torch.int64), expect),
                f"{msg}: slots row {b}",
            )
            self.assertTrue(bool((page_indices[b, k:] == -1).all()))

    def test_level_one_matches_reference_under_garbage_tail(self):
        """The HIP block top-k (AOT row-split and torch fallback) must publish the
        reference's blocks and never read past a row's reach."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            select_candidate_blocks_hip,
        )

        torch.manual_seed(11)
        cases = (
            # Released blocks, a rectangle just wider than the longest row.
            (TOPK_BLOCKS, BLOCK_SIZE, 40000, [3, 8, 16384, 16385, 20000, 40000, 1]),
            # Released blocks on a 1M-wide rectangle, the page table's capacity on
            # a 1M-context server: the block top-k takes the AOT row-split path.
            (TOPK_BLOCKS, BLOCK_SIZE, 1 << 20, [16385, 131072, 7, 600]),
        )
        for topk_blocks, block_size, width, lens in cases:
            with self.subTest(topk_blocks=topk_blocks, width=width, lens=lens):
                seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
                raw = torch.randn(len(lens), width, device="cuda")
                raw = self._garbage_tail(raw, seq)
                expected = reference_position_mask(raw, seq, topk_blocks, block_size)

                cands = select_candidate_blocks_hip(
                    raw, seq, topk_blocks=topk_blocks, block_size=block_size
                )
                ids = cands.ids
                self.assertEqual(ids.shape, (len(lens), topk_blocks))
                self.assertTrue(
                    torch.equal(
                        cands.compact_lens.cpu(),
                        torch.tensor(
                            [
                                min((n + block_size - 1) // block_size, topk_blocks)
                                * block_size
                                for n in lens
                            ],
                            dtype=torch.int32,
                        ),
                    )
                )
                self.assertEqual(ids.dtype, torch.int32)
                got = ids_to_position_mask(ids, block_size, width)
                self.assertTrue(torch.equal(got, expected), "published blocks")
                for b, n in enumerate(lens):
                    row = ids[b]
                    n_ids = int((row >= 0).sum())
                    self.assertTrue(bool((row[:n_ids] >= 0).all()), "padding last")
                    self.assertLessEqual(
                        int(got[b, :n].sum()), topk_blocks * block_size
                    )

                n_pages = (width + INDEX_PAGE_SIZE - 1) // INDEX_PAGE_SIZE
                page_table = torch.stack(
                    [torch.randperm(n_pages, device="cuda") for _ in lens]
                ).to(torch.int32)
                self._assert_consumer_matches_reference(
                    raw, seq, cands, page_table, "consumer"
                )


def _seed(seed: int) -> random.Random:
    torch.manual_seed(seed)
    return random.Random(seed)


def _topk_inputs(bs, width, page_size, lens):
    # a row longer than its logits reads past the tensor (scores[0, len) and
    # page_table[0, len // page_size]), and what it selects there differs by launch
    assert max(lens, default=0) <= width, (lens, width)
    # distinct scores per row: the radix top-k breaks a tie at the threshold in
    # atomic-counter order, so two launches over ties can select different sets
    scores = (
        torch.stack([torch.randperm(width, device=DEVICE).float() for _ in range(bs)])
        * 0.37
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    n_pages = (width + page_size - 1) // page_size
    page_table = torch.randint(
        0, 1 << 20, (bs, n_pages), dtype=torch.int32, device=DEVICE
    )
    return scores, seq_lens, page_table


@pytest.mark.skipif(
    not _aot_topk_sorts_output(), reason="sgl_kernel predates sort_output"
)
@pytest.mark.parametrize("topk", [512, 100], ids=["512", "100"])
@pytest.mark.parametrize("with_raw", [True, False], ids=["True", "False"])
def test_sorted_topk_epilogue_matches_transform_then_sort(topk: int, with_raw: bool):
    rng = _seed(topk)
    for width in (1024, 70000):
        for bs in (1, 33):
            page_size = rng.choice([16, 64])
            # the topk + 1 edge (a radix row of exactly topk picks) only where the
            # logits are that wide; at width == topk it would run past the row
            edges = [
                n
                for n in (0, 1, topk - 1, topk, topk + 1, width // 2, width)
                if n <= width
            ]
            lens = [rng.choice(edges) for _ in range(bs)]
            scores, seq_lens, page_table = _topk_inputs(bs, width, page_size, lens)
            # the unsorted transform, then sort_selection_rows' order (by position with
            # raw indices, by slot without); the Triton sort for a power-of-two k
            ref = torch.empty(bs, topk, dtype=torch.int32, device=DEVICE)
            ref_raw = torch.empty_like(ref) if with_raw else None
            torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
                scores, seq_lens, page_table, ref, page_size, ref_raw
            )
            if topk & (topk - 1) == 0:
                sort_selection_rows(ref, ref_raw)
            else:
                by = ref_raw if with_raw else ref
                key = torch.where(by < 0, torch.iinfo(torch.int32).max, by)
                order = torch.sort(key, dim=1, stable=True).indices
                ref = torch.gather(ref, 1, order)
                if with_raw:
                    ref_raw = torch.gather(ref_raw, 1, order)
            out = torch.empty_like(ref)
            out_raw = torch.empty_like(ref) if with_raw else None
            topk_transform_paged_sorted(
                scores, seq_lens, page_table, out, page_size, out_raw
            )
            assert torch.equal(out, ref), (bs, width, page_size, lens)
            if with_raw:
                assert torch.equal(out_raw, ref_raw)
            # padding last, keys ascending
            for row in (ref_raw if with_raw else ref).tolist():
                n = sum(x >= 0 for x in row)
                assert all(x < 0 for x in row[n:])
                assert row[:n] == sorted(row[:n])


def _candidates(rng, rows, num_blocks, topk_blocks, block_size, seq_lens):
    ids = torch.full((rows, topk_blocks), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        reach = min(num_blocks, -(-int(seq_lens[r]) // block_size))
        picks = rng.sample(range(reach), min(reach, topk_blocks))
        rng.shuffle(picks)
        if picks:
            ids[r, : len(picks)] = torch.tensor(picks, dtype=torch.int32, device=DEVICE)
    block_lens = (seq_lens + block_size - 1) // block_size
    compact_lens = (torch.clamp(block_lens, max=topk_blocks) * block_size).to(
        torch.int32
    )
    width = topk_blocks * block_size
    return CandidateBlocks(
        ids=ids,
        compact_lens=compact_lens,
        compact_page_table=torch.zeros((rows, 1), dtype=torch.int32, device=DEVICE),
        compact_page_size=1 << (width - 1).bit_length(),
        block_size=block_size,
    )


@pytest.mark.parametrize("topk", [512], ids=["512"])
@pytest.mark.parametrize("with_raw", [True, False], ids=["True", "False"])
def test_sorted_candidate_mapping_matches_pack_then_sort(topk: int, with_raw: bool):
    rng = _seed(11 + topk)
    block_size, topk_blocks, page_size = 64, 16, 64
    for rows in (1, 7):
        width = 8192
        lens = [
            rng.choice([0, 1, 300, topk, topk + 5, 3000, width]) for _ in range(rows)
        ]
        scores, seq_lens, page_table = _topk_inputs(rows, width, page_size, lens)
        cands = _candidates(
            rng, rows, width // block_size, topk_blocks, block_size, seq_lens
        )
        outs = []
        for sort in (False, True):
            page = torch.empty(rows, topk, dtype=torch.int32, device=DEVICE)
            raw = torch.empty_like(page) if with_raw else None
            topk_within_candidate_blocks_hip(
                scores,
                seq_lens,
                cands,
                page_table=page_table,
                page_size=page_size,
                page_indices=page,
                raw_indices=raw,
                sort_output=sort,
            )
            if not sort:
                sort_selection_rows(page, raw)
            outs.append((page, raw))
        assert torch.equal(outs[0][0], outs[1][0])
        if with_raw:
            assert torch.equal(outs[0][1], outs[1][1])


@pytest.mark.parametrize("bpp", [4], ids=["4"])
def test_expand_index_page_table(bpp: int):
    _seed(5)
    for bs, n in ((1, 4608), (3, 17), (0, 10)):
        page_table = torch.randint(
            0, 1 << 20, (bs, n), dtype=torch.int32, device=DEVICE
        )
        ref = _expand_index_page_table(
            page_table, full_page_size=64 * bpp, compress_ratio=1, index_page_size=64
        )
        out = expand_index_page_table(page_table, bpp)
        assert out.dtype is torch.int32 and out.shape == ref.shape
        assert torch.equal(out, ref)
    # a strided (row-sliced) table is read through its strides
    page_table = torch.randint(0, 1 << 20, (8, 33), dtype=torch.int32, device=DEVICE)[
        ::2
    ]
    assert torch.equal(
        expand_index_page_table(page_table, 4),
        _expand_index_page_table(
            page_table, full_page_size=256, compress_ratio=1, index_page_size=64
        ),
    )


@pytest.mark.parametrize("loc_dtype", [torch.int64], ids=["torch.int64"])
@pytest.mark.parametrize("ratios", [(1, 2)], ids=["(1, 2)"])
def test_low_ratio_compression_metadata(loc_dtype, ratios):
    rng = _seed(7)
    for rows, nw in ((1, 1), (9, 9), (12, 5)):
        seq_lens = torch.tensor(
            [rng.choice([0, 1, 2, 3, 1000, 1001, 65535]) for _ in range(rows)],
            dtype=torch.int32,
            device=DEVICE,
        )
        raw_out_loc = torch.randint(0, 1 << 24, (nw,), dtype=loc_dtype, device=DEVICE)
        out = low_ratio_compression_metadata(seq_lens, raw_out_loc, ratios)
        assert set(out) == {
            f"c{r}_{k}" for r in ratios for k in ("out_loc", "topk_lengths_clamp1")
        }
        for r in ratios:
            ref_loc, ref_clamp1 = _low_ratio_compression_metadata(
                r, seq_lens, raw_out_loc
            )
            assert out[f"c{r}_out_loc"].dtype is ref_loc.dtype
            assert torch.equal(out[f"c{r}_out_loc"], ref_loc)
            assert out[f"c{r}_topk_lengths_clamp1"].dtype is ref_clamp1.dtype
            assert torch.equal(out[f"c{r}_topk_lengths_clamp1"], ref_clamp1)


@pytest.mark.parametrize("seq_len", [1024], ids=["1024"])
def test_selection_past_index_topk_is_repeatable(seq_len: int) -> None:
    """Rows longer than k: the AOT top-k emits its picks in atomic-counter order, so two launches
    on the same scores differ; ordered by position they are identical, -1 padding last."""
    torch.manual_seed(seq_len)
    k, rows = 512, seq_len
    scores = torch.randn(rows, seq_len, device=DEVICE)
    seq_lens = torch.arange(1, rows + 1, device=DEVICE, dtype=torch.int32)
    pages = -(-seq_len // INDEX_PAGE_SIZE)
    page_table = (
        torch.randperm(pages, device=DEVICE)
        .to(torch.int32)
        .expand(rows, -1)
        .contiguous()
    )

    def select():
        page = torch.empty((rows, k), dtype=torch.int32, device=DEVICE)
        raw = torch.empty_like(page)
        topk_transform_paged(scores, seq_lens, page_table, page, INDEX_PAGE_SIZE, raw)
        sort_selection_rows(page, raw)
        return page, raw

    page_a, raw_a = select()
    page_b, raw_b = select()
    assert torch.equal(raw_a, raw_b) and torch.equal(page_a, page_b)
    valid = raw_a >= 0
    assert torch.equal(valid.sum(1), seq_lens.clamp_max(k))
    # ascending positions inside the valid prefix, padding after it
    assert bool((raw_a[:, 1:][valid[:, 1:]] > raw_a[:, :-1][valid[:, 1:]]).all())
    assert bool((valid[:, :-1] | ~valid[:, 1:]).all())
    pos = raw_a.clamp_min(0)
    slots = (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )
    assert torch.equal(page_a, torch.where(valid, slots, -1))


def test_extend_k_slots_gathers_every_request_in_order():
    """The one-shot gather of the visible compressed slots equals the per-request
    walk, including a request with nothing visible and the start offsets."""
    _seed(29)
    ratio, lc_per_req = 2, [5, 0, 3, 1]
    req_to_token = torch.randint(0, 4096, (8, 64), device=DEVICE, dtype=torch.int32)
    req_pool_indices = torch.tensor([6, 1, 7, 0], device=DEVICE, dtype=torch.int32)
    slots, starts = _extend_k_slots(
        req_to_token,
        ratio=ratio,
        lc_per_req=lc_per_req,
        req_pool_indices=req_pool_indices,
        device=DEVICE,
    )
    expected = torch.cat(
        [
            req_to_token[int(r), torch.arange(lc, device=DEVICE) * ratio].to(
                torch.int64
            )
            // ratio
            for r, lc in zip(req_pool_indices, lc_per_req)
        ]
    )
    assert torch.equal(slots, expected)
    assert starts == [0, 5, 5, 8]


def test_page_table_from_req_to_token_matches_torch():
    from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
        page_table_from_req_to_token,
    )

    _seed(23)
    req_to_token = torch.randint(
        0, 2**20, (300, 8192), device=DEVICE, dtype=torch.int32
    )
    req_to_token[
        7, :64
    ] = -3  # torch floor-divides; the slot values are never negative in serving
    for bs, max_seq_len, page in (
        (1, 1000, 256),
        (64, 8191, 256),
    ):
        req = torch.randint(0, 300, (bs,), device=DEVICE, dtype=torch.int32)
        req[0] = 7
        ref = (req_to_token[req, :max_seq_len:page] // page).to(torch.int32)
        got = page_table_from_req_to_token(req_to_token, req, max_seq_len, page)
        assert got.shape == ref.shape and got.dtype == torch.int32
        assert torch.equal(got, ref)
    empty = page_table_from_req_to_token(
        req_to_token, torch.empty(0, device=DEVICE, dtype=torch.int32), 1000, 64
    )
    assert empty.shape == (0, 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
