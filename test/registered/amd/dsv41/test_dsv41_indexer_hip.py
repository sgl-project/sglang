"""The V4.1 low-ratio indexer kernels on ROCm: FlyDSL fp4 paged logits + AOT top-k against a torch golden, the two-level top-k against the reference, the split-K head weights bitwise."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=50, suite="stage-b-test-1-gpu-small-amd-mi35x")

FULL_PAGE_SIZE = 256
INDEX_PAGE_SIZE = 64
HEAD_DIM = 128
N_HEADS = 32
TOPK = 512
# Released V4.1 config; the span 2048 x 8 = 16384 is where level one starts to bind.
TOPK_BLOCKS, BLOCK_SIZE = 2048, 8

# the released hidden size; the scale is 2 ** -6 in fp32, so the aten multiply is exact
HIDDEN = 5120
SCALE = 128**-0.5 * 32**-0.5


def golden_scores(q, k, weights):
    """Reference scoring (einsum(q, k).relu() * weights).sum(heads) in fp32: q [b, H,
    d], weights [b, H], k [n, d] shared by every row or [b, n, d] per row -> [b, n]."""
    eq = "bhd,bnd->bhn" if k.dim() == 3 else "bhd,nd->bhn"
    s = torch.einsum(eq, q.float(), k.float())
    return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)


def compressed_slot(full_page_table, ratio, j):
    """Slot of compressed position j through the FULL page table at `ratio`."""
    slots_per_page = FULL_PAGE_SIZE // ratio
    phys_page = full_page_table.gather(1, (j // slots_per_page).to(torch.int64))
    return phys_page.to(torch.int64) * slots_per_page + j % slots_per_page


def index_slots(page_table, pos):
    """Slot of compressed position `pos` through the expanded indexer page table."""
    return (
        page_table.gather(1, pos // INDEX_PAGE_SIZE) * INDEX_PAGE_SIZE
        + pos % INDEX_PAGE_SIZE
    )


def reference_position_mask(logits, lens, topk_blocks, block_size):
    """The reference's level one on logits whose tail past the reach is -inf."""
    from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

    col = torch.arange(logits.shape[1], device=logits.device)
    pre = logits.masked_fill(col >= lens[:, None], -torch.inf)
    return select_candidate_blocks(
        pre, lens[:, None], topk_blocks=topk_blocks, block_size=block_size
    )


def candidate_block_ids_to_mask(ids, num_blocks):
    """bool [rows, num_blocks] block mask of `CandidateBlocks.ids`."""
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


# -- the kernels: FlyDSL paged logits and the AOT paged top-k transform ---------


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "FlyDSL fp4 indexer kernels are gfx950 only"
)
class TestFp4PagedLogitsKernels(CustomTestCase):
    @staticmethod
    def _queries_and_golden(rows, full_page_table, seq_lens, k_all, ratio, max_slots):
        """Random fp4-grid queries and head weights per row, packed for FlyDSL, with the
        fp32 golden scores over `max_slots` and each row's visible mask."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            pack_fp4_query_flydsl,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        q = fake_quant_fp4(
            torch.randn(rows, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.rand(rows, N_HEADS, device="cuda", dtype=torch.bfloat16)
        q_fp4, q_scale = pack_fp4_query_flydsl(q)
        j = torch.arange(max_slots, device="cuda")
        slots = compressed_slot(full_page_table, ratio, j.expand(rows, -1))
        return dict(
            q_fp4=q_fp4,
            q_scale=q_scale,
            weights=weights,
            ref=golden_scores(q, k_all[slots], weights),
            visible=j[None, :] < seq_lens[:, None],
        )

    def _setup(self, ratio):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            store_fp4_index_k_cache_split,
        )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _expand_index_page_table,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(ratio)
        bs, n_full_pages = 4, 8
        slots_per_page = FULL_PAGE_SIZE // ratio
        max_slots = n_full_pages * slots_per_page
        n_phys_pages = bs * n_full_pages + 3
        total_slots = n_phys_pages * slots_per_page
        n_index_pages = total_slots // INDEX_PAGE_SIZE

        full_page_table = (
            torch.randperm(n_phys_pages, device="cuda")[: bs * n_full_pages]
            .view(bs, n_full_pages)
            .to(torch.int32)
        )
        seq_lens = torch.tensor(
            [slots_per_page, slots_per_page + 5, 37, max_slots],
            dtype=torch.int32,
            device="cuda",
        )
        k_all = fake_quant_fp4(
            torch.randn(total_slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        payload = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, 16, dtype=torch.uint8, device="cuda"
        ).view(torch.float4_e2m1fn_x2)
        scale = torch.zeros(
            n_index_pages, 1, 4, INDEX_PAGE_SIZE, dtype=torch.uint8, device="cuda"
        )
        loc = torch.arange(total_slots, dtype=torch.int32, device="cuda")
        store_fp4_index_k_cache_split(
            k_all, payload, scale, loc, page_size=INDEX_PAGE_SIZE, rne=True
        )
        page_table = _expand_index_page_table(
            full_page_table,
            full_page_size=FULL_PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        return SimpleNamespace(
            bs=bs,
            max_slots=max_slots,
            k_all=k_all,
            full_page_table=full_page_table,
            seq_lens=seq_lens,
            payload=payload,
            scale=scale,
            page_table=page_table,
            **self._queries_and_golden(
                bs, full_page_table, seq_lens, k_all, ratio, max_slots
            ),
        )

    def _assert_logits_match_golden(self, case, logits):
        diff = (logits.float()[:, : case.max_slots] - case.ref).abs()
        rel = diff / case.ref.abs().clamp_min(1.0)
        self.assertLess(
            rel[case.visible].max().item(),
            2e-2,
            msg=f"kernel logits diverged: max rel diff {rel[case.visible].max()}",
        )

    def _assert_topk_matches_golden(self, case, ratio, page_indices, raw_indices):
        for b in range(case.bs):
            n_valid = min(TOPK, int(case.seq_lens[b]))
            sel_raw = raw_indices[b, :n_valid]
            sel_slot = page_indices[b, :n_valid]
            self.assertTrue(
                bool((sel_raw >= 0).all()) and bool((sel_raw < case.seq_lens[b]).all())
            )
            self.assertTrue(bool((raw_indices[b, n_valid:] == -1).all()))
            self.assertTrue(bool((page_indices[b, n_valid:] == -1).all()))
            expect = compressed_slot(
                case.full_page_table[b : b + 1], ratio, sel_raw[None].to(torch.int64)
            )[0]
            self.assertTrue(torch.equal(sel_slot.to(torch.int64), expect))
            ref_sel = (
                case.ref[b]
                .masked_fill(~case.visible[b], -torch.inf)
                .topk(n_valid)
                .indices
            )
            overlap = len(set(ref_sel.tolist()) & set(sel_raw.tolist())) / n_valid
            self.assertGreaterEqual(overlap, 0.9, msg=f"top-k overlap {overlap:.3f}")

    def _run_decode(self, ratio, verify_block=None):
        from sglang.kernels.ops.attention.dsv4 import topk_transform_paged
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_fp4_paged_mqa_logits,
            prepare_fp4_decode_workspace,
        )

        case = self._setup(ratio)
        if verify_block is not None:
            case = self._expand_to_verify_rows(case, ratio, verify_block)
        workspace = prepare_fp4_decode_workspace(case.page_table, case.seq_lens)
        logits = aiter_fp4_paged_mqa_logits(
            q_fp4=case.q_fp4,
            q_scale=case.q_scale,
            k_payload=case.payload,
            k_scale=case.scale,
            weights=case.weights,
            page_table=case.page_table,
            c4_seq_lens=case.seq_lens,
            weight_scale=1.0,
            is_decode=True,
            decode_workspace=workspace,
        )
        self._assert_logits_match_golden(case, logits)
        bs = case.bs
        page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        topk_transform_paged(
            logits,
            case.seq_lens,
            case.page_table,
            page_indices,
            INDEX_PAGE_SIZE,
            raw_indices,
        )
        self._assert_topk_matches_golden(case, ratio, page_indices, raw_indices)

    def _run_prefill(self, ratio):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            aiter_fp4_paged_mqa_logits,
            prepare_fp4_prefill_workspace,
        )

        case = self._setup(ratio)
        for workspace in (
            None,
            prepare_fp4_prefill_workspace(case.page_table, case.seq_lens),
        ):
            # One row per request stands in for one row per token: the prefill
            # kernel is row-wise, so the shapes are the same contract.
            logits = aiter_fp4_paged_mqa_logits(
                q_fp4=case.q_fp4,
                q_scale=case.q_scale,
                k_payload=case.payload,
                k_scale=case.scale,
                weights=case.weights,
                page_table=case.page_table,
                c4_seq_lens=case.seq_lens,
                weight_scale=1.0,
                is_decode=False,
                prefill_workspace=workspace,
            )
            self._assert_logits_match_golden(case, logits)

    def test_decode_matches_golden(self):
        """The decode logits and the paged transform must reproduce the reference scores
        and slots through a permuted FULL page table at both ratios."""
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self._run_decode(ratio=ratio)

    def test_prefill_matches_golden(self):
        """The row-wise prefill kernel must match the reference with and without a
        prepared workspace."""
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self._run_prefill(ratio=ratio)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "FlyDSL fp4 indexer kernels are gfx950 only"
)
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


# -- the head weights: split-K Triton GEMV route --------------------------------


def _served_chain(x, w, scale):
    from aiter.tuned_gemm import tgemm

    return (tgemm.mm(x, w, None, otype=x.dtype) * scale).contiguous()


def _ordered_bits(t):
    """bf16 -> integers ordered like the values, so differences count ulps."""
    i = t.contiguous().view(torch.int16).int()
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _ulp_distance(a, b):
    return (_ordered_bits(a) - _ordered_bits(b)).abs()


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "split-K MFMA GEMV pair is gfx95 only"
)
class TestIndexerHeadWeightsHip(CustomTestCase):
    """The ROCm head-weights route must be `bf16(bf16(fp32 split-K sum) * scale)` bit for bit, batch-invariant and within one bf16 ulp of the exact product."""

    def setUp(self):
        torch.manual_seed(20260909)
        self.w = (torch.randn(N_HEADS, HIDDEN, device="cuda") * 0.02).bfloat16()

    def _x(self, m):
        # Hidden states of varying magnitude per row.
        return (
            torch.randn(m, HIDDEN, device="cuda")
            * (0.5 + 3 * torch.rand(m, 1, device="cuda"))
        ).bfloat16()

    def test_bitwise_definition_and_batch_invariance(self):
        """The route must equal the fixed-order split-K definition bit for bit and give
        a row the same weights alone as inside the batch."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights,
        )
        from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_gemv_split_k

        for m in (1, 16):
            with self.subTest(m=m):
                x = self._x(m)
                got = rocm_indexer_head_weights(x, self.w, SCALE)
                self.assertEqual(got.shape, (m, N_HEADS))
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(got.is_contiguous())
                partials = rocm_router_gemv_split_k(x, self.w)
                acc = partials[0].clone()
                for s in range(1, partials.shape[0]):
                    acc += partials[s]
                expect = (acc.bfloat16().float() * SCALE).bfloat16()
                self.assertTrue(torch.equal(got, expect), "fixed-order definition")
                for r in range(m):
                    self.assertTrue(
                        torch.equal(
                            rocm_indexer_head_weights(x[r : r + 1], self.w, SCALE)[0],
                            got[r],
                        ),
                        f"row {r} depends on the batch",
                    )
                # Repeatable.
                self.assertTrue(
                    torch.equal(rocm_indexer_head_weights(x, self.w, SCALE), got)
                )

    def test_within_one_ulp_of_exact_and_against_served_chain(self):
        """Every element within one bf16 ulp of the exactly rounded product, and only a
        small fraction differing from the served chain."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
            rocm_indexer_head_weights,
        )

        total = differ = 0
        for trial in range(24):
            m = (1, 16)[trial % 2]
            x = self._x(m)
            exact = (x.double() @ self.w.double().T * SCALE).bfloat16()
            got = rocm_indexer_head_weights(x, self.w, SCALE)
            served = _served_chain(x, self.w, SCALE)
            self.assertLessEqual(
                int(_ulp_distance(got, exact).max()),
                1,
                f"{m=}: more than 1 ulp from exact",
            )
            total += got.numel()
            differ += int((got != served).sum())
        # the chains disagree only where their fp32 sums straddle a bf16 rounding boundary
        self.assertLess(
            differ / total,
            2e-3,
            f"{differ} of {total} elements differ from the served chain",
        )


if __name__ == "__main__":
    unittest.main()
