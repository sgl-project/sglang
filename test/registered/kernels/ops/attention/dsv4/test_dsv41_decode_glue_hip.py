"""The V4.1 HIP decode glue is bitwise the torch chains it replaces and selects within each row's reach."""

from __future__ import annotations

import random
import sys
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_index_page_table,
    low_ratio_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    CandidateBlocks,
    select_candidate_blocks_hip,
    topk_within_candidate_blocks_hip,
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

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

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


class _Rank0Decision:
    """A TP sync whose broadcast hands this rank rank 0's (different) accept decision."""

    def __init__(self, correct_len, bonus, enabled):
        self._correct_len, self._bonus = correct_len, bonus
        self._enabled = enabled
        self._calls = 0

    def enabled(self, site):
        return self._enabled

    def sync(self, site, values):
        if not self._enabled:
            return values
        bs = self._bonus.shape[0]
        if values.shape == (2 * bs,):
            # the fused epilogue's packed (correct_len, bonus)
            values[:bs].copy_(self._correct_len)
            values[bs:].copy_(self._bonus)
        elif self._calls % 3 < 2:
            # _accept broadcasts correct_len, bonus, then the cap trims (zero everywhere)
            values.copy_((self._correct_len, self._bonus)[self._calls % 3])
        self._calls += 1
        return values


@pytest.mark.parametrize("simulated", [False, True], ids=["served", "simulated"])
@pytest.mark.parametrize("synced", [False, True], ids=["local", "rank0"])
@pytest.mark.parametrize("stride", [2, 5, 8])
def test_static_verify_epilogue_matches_accept_chain(simulated, synced, stride):
    """The one-launch static epilogue must write every persistent accept buffer and the
    gated commit lengths bitwise as _accept + the gate did, rank 0's decision and a
    SGLANG_SIMULATE_ACC_LEN length included."""
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        DsparkVerifyEpilogue,
    )

    _seed(39)
    max_bs, vocab = 16, 1000
    # gate None: the epilogue folds no KV commit, so nothing is gated
    for bs, gate in ((1, None), (7, 0), (16, 1)):
        logits = torch.randn(bs * stride, vocab, device=DEVICE)
        predict = logits.argmax(-1).view(bs, stride)
        candidates = torch.randint(0, vocab, (bs, stride), device=DEVICE)
        # row b matches its first b % stride drafts, so every correct length occurs
        for b in range(bs):
            candidates[b, 1 : 1 + b % stride] = predict[b, : b % stride]
        seq_lens = torch.randint(1, 9000, (max_bs,), device=DEVICE)
        verify_lens = torch.randint(1, stride + 2, (bs,), device=DEVICE)
        rank0_correct_len = torch.randint(0, stride, (bs,), device=DEVICE).int()
        rank0_bonus = torch.randint(0, vocab, (bs,), device=DEVICE)
        # one length for every row, where the served lengths vary across rows
        simulated_correct_len = (
            torch.tensor([stride - 1], dtype=torch.int64, device=DEVICE)
            if simulated
            else None
        )
        outs = []
        for fused in (False, True):
            epilogue = DsparkVerifyEpilogue(
                max_bs=max_bs,
                verify_num_draft_tokens=stride,
                device=DEVICE,
                tp_sync=_Rank0Decision(rank0_correct_len, rank0_bonus, synced),
                simulated_correct_len=simulated_correct_len,
            )
            persistent = epilogue.read_accept(max_bs)
            for name in persistent.__struct_fields__:
                getattr(persistent, name).fill_(-7)
            epilogue.inject_gate_buf.fill_(gate or 0)
            if fused:
                gated = epilogue._static_accept_fused(
                    candidates=candidates,
                    logits=logits,
                    seq_lens=seq_lens,
                    verify_lens=None if gate is None else verify_lens,
                )
            else:
                commit_lens = epilogue._accept(
                    candidates=candidates,
                    logits=logits,
                    draft_tokens=candidates[:, 1:].contiguous(),
                    seq_lens=seq_lens,
                )
                gated = (
                    None
                    if gate is None
                    else torch.minimum(commit_lens, verify_lens.to(torch.int32))
                    * epilogue.inject_gate_buf
                )
            outs.append((epilogue.read_accept(bs), gated))
        (ref, ref_gated), (got, got_gated) = outs
        for name in ref.__struct_fields__:
            r, g = getattr(ref, name), getattr(got, name)
            assert r.dtype == g.dtype and torch.equal(r, g), (name, bs, r, g)
        if gate is None:
            assert got_gated is None
        else:
            assert ref_gated.dtype == got_gated.dtype
            assert torch.equal(ref_gated, got_gated), (bs, ref_gated, got_gated)


def test_candidate_blocks_write_the_zero_page_table():
    """The compact page table is an empty allocation the lengths launch zero-fills; a
    row it misses hands the consumer a stale page."""
    _seed(40)
    empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        return empty(*args, **kwargs).fill_(5)

    width = 40000
    logits = torch.randn(3, width, device=DEVICE)
    seq_lens = torch.tensor([1, 16385, width], dtype=torch.int32, device=DEVICE)
    with mock.patch.object(torch, "empty", poisoned_empty):
        cands = select_candidate_blocks_hip(
            logits, seq_lens, topk_blocks=TOPK_BLOCKS, block_size=BLOCK_SIZE
        )
    assert cands.compact_page_table.shape == (3, 1)
    assert not cands.compact_page_table.any()


@pytest.mark.parametrize("rows", [1, 3, 16])
def test_hc_combine_widens_bf16_rows_exactly(rows):
    """The V4 heads hand hc_combine the bf16 rows on HIP; its result must stay bitwise
    the fp32 copy's, so the kernel must widen on load rather than compute in bf16."""
    from sglang.kernels.ops.layernorm.mhc import hc_combine

    _seed(41)
    hc, hidden = 4, 4096
    x = (torch.randn(rows, hc * hidden, device=DEVICE) * 30).bfloat16()
    pre = torch.rand(rows, hc, device=DEVICE)
    ref = hc_combine(x.float(), pre, hc, torch.bfloat16)
    got = hc_combine(x, pre, hc, torch.bfloat16)
    assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


def test_packed_aux_means_match_mean_then_cat():
    """Each hc-mean written into its slice of the packed buffer must be bitwise the
    separate mean the logits processor used to concatenate, in capture order."""
    from sglang.srt.models.deepseek_v4 import _PackedAuxMeans

    _seed(42)
    hc, hidden, captures = 4, 4096, 3
    for tokens in (1, 5, 40):
        # wide magnitudes, so a different reduction order would round differently
        auxes = [
            (
                torch.randn(tokens, hc, hidden, device=DEVICE)
                * torch.exp(torch.randn(tokens, hc, hidden, device=DEVICE) * 4)
            ).bfloat16()
            for _ in range(captures)
        ]
        means = [a.mean(dim=1) for a in auxes]
        packed = _PackedAuxMeans(captures)
        for a in auxes[:-1]:
            packed.capture_mean(a)
        # a partial capture keeps the list form instead of unwritten slices
        incomplete = packed.packed()
        assert isinstance(incomplete, list) and len(incomplete) == captures - 1
        packed.capture_mean(auxes[-1])
        got = packed.packed()
        ref = torch.cat(means, dim=-1)
        assert isinstance(got, torch.Tensor) and got.shape == ref.shape
        assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
