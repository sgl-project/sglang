"""The V4.1 HIP decode glue is bitwise the torch chains it replaces and selects within each row's reach."""

from __future__ import annotations

import random
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_index_page_table,
    expand_same_length_rows,
    gather_i32,
    low_ratio_compression_metadata,
    sparse_buffers,
)
from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    CandidateBlocks,
    select_candidate_blocks_hip,
    topk_within_candidate_blocks_hip,
)
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
    BuildCausalSwaPageIndices,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _low_ratio_compression_metadata,
)
from sglang.srt.layers.attention.dsv4.metadata import (
    expand_index_page_table as _expand_index_page_table,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=25, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="HIP decode glue for the AITER / FlyDSL DeepSeek-V4.1 path (gfx95x).",
)

DEVICE = "cuda"

INDEX_PAGE_SIZE = 64
TOPK = 512
# Released V4.1 config; the span 2048 x 8 = 16384 is where level one starts to bind.
TOPK_BLOCKS, BLOCK_SIZE = 2048, 8
SWA_WINDOW = 128
PAGE_INDEX_ALIGNED_SIZE = 64


def index_slots(page_table, pos):
    """Slot of compressed position pos through the expanded indexer page table."""
    return (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )


def reference_position_mask(logits, lens, topk_blocks, block_size):
    """The reference's level one on logits whose tail past the reach is -inf."""
    col = torch.arange(logits.shape[1], device=logits.device)
    pre = logits.masked_fill(col >= lens[:, None], -torch.inf)
    ids = select_candidate_block_ids(pre, lens[:, None], topk_blocks, block_size)
    return ids_to_position_mask(ids, block_size, logits.shape[1])


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


def _sort_rows(page, raw):
    """Rows ascending by position (by slot without raw), -1 padding last."""
    key = (raw if raw is not None else page).to(torch.int64)
    order = key.masked_fill(key < 0, torch.iinfo(torch.int64).max).argsort(dim=1)
    return page.gather(1, order), (raw.gather(1, order) if raw is not None else None)


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
                page, raw = _sort_rows(page, raw)
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


def _expand_reference(seq_lens, req_pool_indices, qo_len, offset, extend_seq_lens):
    """The torch chain the verify / draft-block metadata built before the fused launch."""
    bs = len(seq_lens)
    seq = seq_lens + offset
    ext = (
        torch.full_like(seq, qo_len)
        if extend_seq_lens is None
        else extend_seq_lens.to(device=seq.device, dtype=seq.dtype)
    )
    ext = torch.minimum(ext, seq).clamp_min_(1)
    casual = seq[:, None] + torch.arange(
        -qo_len + 1, 1, dtype=torch.int32, device=seq.device
    )
    casual = casual.flatten().to(torch.int32)
    idx = torch.arange(bs, dtype=torch.int32, device=seq.device).repeat_interleave(
        qo_len
    )
    return seq, ext, casual, casual - 1, req_pool_indices[idx]


@pytest.mark.parametrize("seq_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("with_ext", [False, True])
def test_expand_same_length_rows_matches_torch(seq_dtype, with_ext):
    """A wrong seq_lens offset, extend clamp, row order or output dtype in the fused
    verify / draft-block metadata shifts every attention row's reach."""
    _seed(31)
    for bs, qo, offset in ((1, 5, 5), (7, 5, 0), (33, 3, 3), (0, 5, 5)):
        seq_lens = torch.randint(0, 5000, (bs,), device=DEVICE).to(seq_dtype)
        req = torch.randint(0, 64, (bs,), device=DEVICE, dtype=torch.int64)
        ext = (
            torch.randint(-2, 9, (bs,), device=DEVICE, dtype=seq_dtype)
            if with_ext
            else None
        )
        ref = _expand_reference(seq_lens, req, qo, offset, ext)
        got = expand_same_length_rows(
            seq_lens, req, qo_len=qo, seq_lens_offset=offset, extend_seq_lens=ext
        )
        for r, g in zip(ref, got):
            assert r.dtype == g.dtype and r.shape == g.shape, (r, g)
            assert torch.equal(r, g)


def _hip_swa_page_indices(req_to_token, mapping, req_rep, seq_lens_casual):
    """`get_swa_page_indices` + `_pad_last_dim` of the HIP radix backend."""
    offsets = (seq_lens_casual - 1).unsqueeze(1) - torch.arange(
        SWA_WINDOW, dtype=torch.int32, device=DEVICE
    ).unsqueeze(0)
    invalid = offsets < 0
    offsets.masked_fill_(invalid, 0)
    raw = req_to_token[req_rep[:, None], offsets]
    raw.masked_fill_(invalid, -1)
    swa = mapping[raw].to(torch.int32)
    width = -(-SWA_WINDOW // PAGE_INDEX_ALIGNED_SIZE) * PAGE_INDEX_ALIGNED_SIZE
    return F.pad(swa, (0, width - SWA_WINDOW), value=-1)


def test_full_to_swa_translation_wraps_unset_slots():
    """An unset -1 slot must read the mapping's trailing -1 sentinel like torch
    indexing, in both the SWA page-index kernel and the write-location gather;
    otherwise it reads out of bounds and attends a stale page."""
    _seed(32)
    pool, max_ctx, n_req = 20000, 4096, 16
    mapping = torch.randint(0, 5000, (pool + 1,), device=DEVICE, dtype=torch.int64)
    mapping[-1] = -1
    req_to_token = torch.randint(
        0, pool, (n_req, max_ctx), device=DEVICE, dtype=torch.int32
    )
    req_to_token[3, 100:110] = -1
    for rows in (1, 10, 160):
        req_rep = torch.randint(0, n_req, (rows,), device=DEVICE, dtype=torch.int64)
        req_rep[0] = 3
        seq = torch.randint(1, max_ctx, (rows,), device=DEVICE, dtype=torch.int32)
        seq[0] = 105
        got = BuildCausalSwaPageIndices.triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=mapping,
            req_pool_indices_repeated=req_rep,
            seq_lens_casual=seq,
            swa_window=SWA_WINDOW,
            page_index_aligned_size=PAGE_INDEX_ALIGNED_SIZE,
        )
        assert torch.equal(
            got, _hip_swa_page_indices(req_to_token, mapping, req_rep, seq)
        )
    for n in (0, 3000):
        loc = torch.randint(-1, pool, (n,), device=DEVICE, dtype=torch.int64)
        assert torch.equal(gather_i32(mapping, loc), mapping[loc].to(torch.int32))


@pytest.mark.parametrize("ratios", [(2,), (1, 2)], ids=["c2", "c1-c2"])
def test_sparse_buffers_fills_low_ratio_raw_indices(ratios):
    """Every live buffer packed into the fill slots, raw indices included, must come
    out -1; a mis-packed slot leaves uninitialized raw indices for the OPUS prefill."""
    _seed(33)
    for rows in (1, 513):
        lens = {
            r: torch.randint(1, 4000, (rows,), device=DEVICE, dtype=torch.int32)
            for r in (4, 0, 1, 2)
        }
        kw = dict(
            c4_topk_lengths_clamp1=lens[4],
            c4_topk_lengths_raw=lens[0],
            c1_topk_lengths_clamp1=lens[1] if 1 in ratios else None,
            c2_topk_lengths_clamp1=lens[2] if 2 in ratios else None,
            index_topk=512,
            page_index_align=PAGE_INDEX_ALIGNED_SIZE,
        )
        ref = sparse_buffers(**kw)
        for r in ratios:
            ref[f"c{r}_sparse_raw_indices"] = torch.full_like(
                ref[f"c{r}_sparse_page_indices"], -1
            )
        got = sparse_buffers(**kw, low_ratio_raw_indices=True)
        assert ref.keys() == got.keys()
        for k in ref:
            assert torch.equal(ref[k], got[k]), k


def test_init_compression_metadata_pads_c128_page_indices():
    """The in-launch alignment must equal F.pad(..., value=-1) of the unaligned c128
    page indices and leave every other output unchanged."""
    _seed(34)
    page_size = 256
    for rows, max_pages in ((5, 37), (10, 64), (1, 3)):
        seq = torch.randint(1, max_pages * page_size, (rows,), device=DEVICE).to(
            torch.int32
        )
        out_loc = torch.randint(0, 1 << 20, (rows,), device=DEVICE, dtype=torch.int64)
        page_table = torch.randint(
            0, 1000, (rows, max_pages), device=DEVICE, dtype=torch.int32
        )
        args = (seq, seq - 1, out_loc, page_table, page_size)
        *ref, ref_c128_page_indices = init_compression_metadata(*args)
        *got, got_c128_page_indices = init_compression_metadata(
            *args, page_indices_align=PAGE_INDEX_ALIGNED_SIZE
        )
        for r, g in zip(ref, got):
            assert torch.equal(r, g)
        width = ref_c128_page_indices.shape[-1]
        pad = -(-width // PAGE_INDEX_ALIGNED_SIZE) * PAGE_INDEX_ALIGNED_SIZE - width
        assert torch.equal(
            F.pad(ref_c128_page_indices, (0, pad), value=-1), got_c128_page_indices
        )


def test_engram_gather_reads_column_slice_ids():
    """A [T, C] column slice of the per-layer hash table must gather the same rows as
    its flattened copy; a wrong row stride reads another layer's ids."""
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    _seed(35)
    rows, dim, blk = 3000, 256, 128
    weight = (torch.randn(rows, dim, device=DEVICE) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(100, 140, (rows, dim // blk), device=DEVICE).to(torch.uint8)
    for t, layers, cols in ((1, 3, 8), (37, 4, 6)):
        ids = torch.randint(0, rows, (t, layers, cols), device=DEVICE)[:, 1]
        ref = torch.empty(t * cols, dim, dtype=torch.bfloat16, device=DEVICE)
        got = torch.empty_like(ref)
        for out, i in ((ref, ids.reshape(-1)), (got, ids)):
            engram_gather(
                weight.data_ptr(), scale.data_ptr(), i, out, dim, blk, row_lo=500, row_hi=2500
            )
        assert torch.equal(ref, got)


def test_rope_copy_into_matches_rope_then_copy():
    """The one-launch rope + copy into a head slice of the padded q must be bitwise the
    in-place rope followed by the copy, and must not write the source q."""
    from sglang.kernels.ops.attention import deepseek_v4_rope as rope

    _seed(36)
    prev = rope._USE_BATCHED_ROPE
    rope.set_batched_rope(True)
    try:
        head_dim, rd, max_pos, pad_heads = 512, 64, 8192, 64
        freqs = torch.polar(
            torch.ones(max_pos, rd // 2, device=DEVICE),
            torch.rand(max_pos, rd // 2, device=DEVICE) * 6.28,
        )
        for tokens, heads in ((1, 8), (10, 32)):
            q = torch.randn(tokens, heads, head_dim, device=DEVICE).to(torch.bfloat16)
            pos = torch.randint(0, max_pos, (tokens,), device=DEVICE)
            ref_pad = torch.zeros(
                tokens, pad_heads, head_dim, dtype=torch.bfloat16, device=DEVICE
            )
            got_pad = ref_pad.clone()
            q_ref = q.clone()
            rope.apply_rotary_emb_triton(q_ref[..., -rd:], freqs, positions=pos)
            ref_pad[:, :heads].copy_(q_ref)
            q_in = q.clone()
            assert (
                rope.apply_rotary_emb_copy_into(
                    q_in, got_pad[:, :heads], freqs, pos, rd
                )
                is not None
            )
            assert torch.equal(q_in.view(torch.int16), q.view(torch.int16))
            assert torch.equal(ref_pad.view(torch.int16), got_pad.view(torch.int16))
    finally:
        rope.set_batched_rope(prev)


def test_verify_epilogue_writes_persistent_buffers():
    """finalize and the out-token build given out= must write those buffers in place
    with the fresh-tensor values, including from int32 correct_len."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        FinalizeAcceptLensResult,
        finalize_accept_lens_triton,
    )
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        build_out_tokens_triton,
    )

    _seed(37)
    stride, gamma, max_bs = 5, 4, 16
    for bs in (1, 16):
        correct = torch.randint(0, stride, (bs,), device=DEVICE, dtype=torch.int32)
        trim = torch.randint(0, 3, (bs,), device=DEVICE, dtype=torch.int32)
        prefix = torch.randint(0, 9000, (bs,), device=DEVICE, dtype=torch.int64)
        bonus = torch.randint(0, 1 << 17, (bs,), device=DEVICE, dtype=torch.int64)
        draft = torch.randint(0, 1 << 17, (bs, gamma), device=DEVICE, dtype=torch.int64)
        ref = finalize_accept_lens_triton(
            correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix
        )
        bufs = FinalizeAcceptLensResult(
            *(
                torch.full((max_bs,), 7, dtype=dtype, device=DEVICE)[:bs]
                for dtype in (torch.int32, torch.int64, torch.int32)
            )
        )
        got = finalize_accept_lens_triton(
            correct_len=correct, cap_trim_lens=trim, prefix_lens=prefix, out=bufs
        )
        for name in ("commit_lens", "new_seq_lens", "cap_trim_lens"):
            assert torch.equal(getattr(ref, name), getattr(bufs, name)), name
            assert getattr(got, name).data_ptr() == getattr(bufs, name).data_ptr()
        kw = dict(
            draft_tokens=draft,
            bonus=bonus,
            verify_num_draft_tokens=stride,
            gamma=gamma,
        )
        ref_tokens = build_out_tokens_triton(**kw, correct_len=correct.to(torch.int64))
        out_buf = torch.full((max_bs, stride), 7, dtype=torch.int64, device=DEVICE)
        build_out_tokens_triton(**kw, correct_len=correct, out=out_buf[:bs])
        assert torch.equal(ref_tokens, out_buf[:bs])


def test_k_norm_rope_flashmla_reads_strided_kv():
    """The DSpark commit projection hands the ROCm FlashMLA writer column slices of
    the stacked KV; the paged cache must match the contiguous input byte for byte."""
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_k_norm_rope_flashmla,
    )

    _seed(38)
    head_dim, rd, page_size, stages, max_pos = 512, 64, 128, 3, 8192
    freqs = torch.polar(
        torch.ones(max_pos, rd // 2, device=DEVICE),
        torch.rand(max_pos, rd // 2, device=DEVICE) * 6.28,
    )
    weight = torch.randn(head_dim, device=DEVICE).to(torch.bfloat16)
    page_bytes = -(-584 * page_size // 576) * 576
    for tokens in (1, 10):
        kv_all = torch.randn(tokens, stages * head_dim, device=DEVICE).to(
            torch.bfloat16
        )
        pos = torch.randint(0, max_pos, (tokens,), device=DEVICE)
        loc = torch.randperm(4 * page_size, device=DEVICE)[:tokens].to(torch.int32)
        caches = []
        for strided in (False, True):
            cache = torch.zeros(4, page_bytes, dtype=torch.uint8, device=DEVICE)
            for kv in kv_all.split(head_dim, dim=-1):
                fused_k_norm_rope_flashmla(
                    kv=kv if strided else kv.contiguous(),
                    kv_weight=weight,
                    eps=1e-6,
                    freqs_cis=freqs,
                    positions=pos,
                    out_loc=loc,
                    kvcache=cache,
                    page_size=page_size,
                )
            caches.append(cache)
        assert torch.equal(caches[0], caches[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
