"""Decode two-level indexer on DeepGEMM's paged sparse MQA logits.

The block table layer 20 publishes (``amax_topk_blocks``) is checked against
the model code's ``select_candidate_blocks``; a consumer's sparse logits
are checked against DeepGEMM's dense bf16 paged logits gathered at the published
positions, which the kernel is documented to match bitwise; its selection is
checked against a torch top-k of those. The kernel path needs a DeepGEMM with
``fp8_fp4_paged_sparse_mqa_logits`` on an SM100 device; it skips elsewhere.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="stage-b-test-small-1-gpu", runner_config="1-gpu")

HEADS = 32
HEAD_DIM = 128
PAGE = 128  # index pool page on SM100: 128 * 68 bytes = 17 * 512
TOPK = 512
BLOCKS = 2048


def _sparse_indexer_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_paged_sparse_mqa_logits")


def _reference_blocks(logits, lens, block_size=8):
    """Block ids the model code keeps, per row, ascending."""
    from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

    width = logits.shape[1]
    reach = torch.arange(width, device=logits.device)[None, :] < lens[:, None]
    mask = select_candidate_blocks(
        logits.masked_fill(~reach, -torch.inf),
        lens[:, None],
        topk_blocks=BLOCKS,
        block_size=block_size,
    )
    return [m.view(-1, block_size).any(-1).nonzero().flatten() for m in mask]


class TestSparseIndexer(CustomTestCase):
    def test_amax_topk_blocks_matches_reference(self):
        # short rows: the block top-k skips its plan; 40 rows of 300K tokens:
        # 37500 keys per row on a batch above the persistent pool, plan needed
        self._check_amax_topk_blocks(
            torch.tensor(
                [1, 37, 16384, 16389, 40000, 131072], dtype=torch.int32, device="cuda"
            ),
            131072,
        )
        self._check_amax_topk_blocks(
            torch.randint(200000, 300001, (40,), dtype=torch.int32, device="cuda"),
            300000,
        )

    def _check_amax_topk_blocks(self, lens, width):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_row_lens,
        )
        from sglang.kernels.ops.attention.dsv4.topk import sort_candidate_blocks
        from sglang.srt.layers.attention.dsv4.candidate_deep_gemm import (
            amax_topk_blocks,
            valid_lens,
        )

        torch.manual_seed(0)
        bs = lens.numel()
        logits = torch.randn(bs, width, device="cuda")
        # the tail past the length is garbage in production: make it loud
        logits.masked_fill_(
            torch.arange(width, device="cuda")[None, :] >= lens[:, None], 1e4
        )
        pages = (width + PAGE - 1) // PAGE
        page_table = torch.stack(
            [torch.randperm(pages, device="cuda") for _ in range(bs)]
        ).to(torch.int32)
        nblocks, valid = candidate_row_lens(lens, BLOCKS)
        self.assertTrue(torch.equal(nblocks, (lens + 7) // 8))
        self.assertTrue(torch.equal(valid, valid_lens(lens, BLOCKS)))
        blocks = amax_topk_blocks(logits, lens, nblocks, BLOCKS)
        phys = sort_candidate_blocks(blocks, lens, page_table, PAGE)
        keys = logits.view(bs, -1, 8).amax(-1)
        bpp = PAGE // 8
        for b, ref in enumerate(_reference_blocks(logits, lens)):
            nb = (int(lens[b]) + 7) // 8
            n = min(nb, BLOCKS)
            got = blocks[b, :n].long()
            self.assertTrue(torch.equal(got, got.sort().values), "not ascending")
            self.assertEqual(got.unique().numel(), n)
            self.assertIn(nb - 1, got.tolist(), "newest block not kept")
            self.assertTrue(bool((got < nb).all()))
            # equal keys may swap blocks: compare the key multiset (the forced
            # block excluded, its key is arbitrary garbage)
            keep = got != nb - 1
            keep_ref = ref != nb - 1
            self.assertTrue(
                torch.equal(
                    keys[b][got[keep]].sort().values,
                    keys[b][ref[keep_ref]].sort().values,
                ),
                msg=f"row {b}",
            )
            # past the valid count nothing looks like a block DeepGEMM could read
            self.assertTrue(bool((blocks[b, n:] >= nb).all()))
            # the same blocks as pool slots / 8 through the row's page table
            ref_phys = page_table[b][got // bpp].long() * bpp + got % bpp
            self.assertTrue(torch.equal(phys[b, :n].long(), ref_phys))
            self.assertTrue(bool((phys[b, n:] == torch.iinfo(torch.int32).max).all()))
            expect_valid = 8 * (n - 1) + ((int(lens[b]) - 1) % 8 + 1)
            self.assertEqual(int(valid[b]), expect_valid)

    @unittest.skipUnless(
        _sparse_indexer_available(), "needs DeepGEMM's paged sparse MQA logits on SM100"
    )
    def test_sparse_chain_matches_dense_logits(self):
        """Layer 20 publishes from its dense fp32 logits; a consumer's sparse bf16
        logits equal the dense bf16 logits at the published positions bitwise."""
        import deep_gemm

        from sglang.srt.layers.attention.dsv4.candidate_deep_gemm import (
            DeepGemmCandidateIndexer,
            valid_lens,
        )
        from sglang.srt.layers.attention.dsv4.candidate_indexer import IndexerInputs

        torch.manual_seed(1)
        lens = torch.tensor(
            [300, 16384, 70000, 131072], dtype=torch.int32, device="cuda"
        )
        bs = lens.numel()
        max_pages = (int(lens.max()) + PAGE - 1) // PAGE
        num_pages = bs * max_pages
        # fp4 index-K pool, the store kernel's page layout: PAGE * 64 payload bytes
        # then PAGE * 4 packed-ue8m0 scale bytes. Random codes with small scales
        # (2^-9 .. 2^-5) keep the logits inside the fp16 range the top-k's coarse
        # keys resolve; production indexer logits are O(1-100).
        pool = torch.randint(
            0, 255, (num_pages, PAGE * 68), dtype=torch.uint8, device="cuda"
        )
        pool[:, PAGE * 64 :] = torch.randint(
            118, 123, (num_pages, PAGE * 4), dtype=torch.uint8, device="cuda"
        )
        k_cache = pool.view(num_pages, PAGE, 1, 68)
        self.assertEqual(k_cache.stride(0) % 512, 0)
        page_table = (
            torch.randperm(num_pages, device="cuda").view(bs, max_pages).to(torch.int32)
        )
        q_fp4 = torch.randint(
            0, 255, (bs, 1, HEADS, HEAD_DIM // 2), dtype=torch.uint8, device="cuda"
        ).view(torch.int8)
        q_sf = (
            torch.randint(118, 123, (bs, 1, HEADS, 4), dtype=torch.uint8, device="cuda")
            .view(torch.int32)
            .squeeze(-1)
        )
        weights = (torch.rand(bs, HEADS, device="cuda") * 0.05).to(torch.bfloat16)

        sched = deep_gemm.get_paged_mqa_logits_metadata(
            lens.view(-1, 1), PAGE, deep_gemm.get_num_sms()
        )
        dense = lambda dtype: deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            weights.float() if dtype == torch.float32 else weights,
            lens.view(-1, 1),
            page_table,
            sched,
            int(lens.max()),
            False,
            dtype,
        )
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        impl = DeepGemmCandidateIndexer(BLOCKS, 8)
        inputs = IndexerInputs(
            q_fp4,
            q_sf,
            k_cache,
            weights.float(),
            SimpleNamespace(
                c4_seq_lens=lens,
                page_table=page_table,
                c4_page_size=PAGE,
                deep_gemm_metadata=sched,
                max_c4_seq_len=int(lens.max()),
                use_topk_v2=True,
                topk_metadata=plan_topk_v2(lens),
            ),
        )
        source_slots = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        table = impl.publish_decode(inputs, source_slots)
        # the source's own selection: min(k, len) slots per row, the rest -1
        for b in range(bs):
            self.assertEqual(int((source_slots[b] >= 0).sum()), min(TOPK, int(lens[b])))
        sparse = impl.scores(table, inputs)
        self.assertEqual(sparse.shape, (bs, BLOCKS * 8))
        dense16 = dense(torch.bfloat16)
        reach = torch.arange(dense16.shape[1], device="cuda")[None, :] < lens[:, None]
        seen = dense16.float()[reach]
        self.assertTrue(
            bool(torch.isfinite(seen).all()) and float(seen.max()) < 65504.0
        )
        cols = torch.arange(BLOCKS * 8, device="cuda")
        pos = table.blocks.long().repeat_interleave(8, dim=1) * 8 + (cols % 8)[None, :]
        row_valid = valid_lens(lens, BLOCKS)
        valid = cols[None, :] < row_valid[:, None].long()
        ref = dense16.gather(1, pos.clamp(max=dense16.shape[1] - 1))
        self.assertTrue(
            torch.equal(sparse[valid], ref[valid]), "sparse logits differ from dense"
        )
        # the consumer's selection: per row the top-k of the valid columns as
        # slots (any order), -1 padded; bf16 ties may swap positions, so the picked
        # scores are compared as a multiset. Slots invert through the page table.
        page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device="cuda")
        impl.select_decode(table, inputs, page_indices)
        slot_to_pos = torch.empty(
            bs, num_pages * PAGE, dtype=torch.int64, device="cuda"
        )
        for b in range(bs):
            n_valid = min(TOPK, int(row_valid[b]))
            chosen = page_indices[b] >= 0
            self.assertEqual(int(chosen.sum()), n_valid)
            slots = page_indices[b][chosen].long()
            pages_b = page_table[b].long()
            slot_to_pos[b].fill_(-1)
            slot_to_pos[b][
                (
                    pages_b[:, None] * PAGE + torch.arange(PAGE, device="cuda")[None, :]
                ).flatten()
            ] = torch.arange(pages_b.numel() * PAGE, device="cuda")
            p = slot_to_pos[b][slots]
            self.assertTrue(bool((p >= 0).all()) and bool((p < lens[b]).all()))
            self.assertEqual(p.unique().numel(), n_valid)
            row = sparse[b].float().masked_fill(~valid[b], -torch.inf)
            ref_vals = row.topk(n_valid).values.sort().values
            got_vals = dense16[b].float()[p].sort().values
            self.assertTrue(torch.equal(got_vals, ref_vals), f"row {b}")

    @unittest.skipUnless(
        _sparse_indexer_available(), "needs DeepGEMM's paged sparse MQA logits on SM100"
    )
    def test_paired_verify_rows_match_unpaired(self):
        """Verify shape: each request has 6 consecutive rows (its draft tokens) with
        lengths L, L+1, ..., sharing one page-table row. With request ids DeepGEMM
        pairs the rows on one KV pass; the sparse logits must equal the unpaired
        (every row its own request) result bitwise, row layout unchanged."""
        import deep_gemm

        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_row_lens,
        )
        from sglang.kernels.ops.attention.dsv4.topk import (
            sort_candidate_blocks,
        )
        from sglang.srt.layers.attention.dsv4.candidate_deep_gemm import (
            SparseBlockTable,
            amax_topk_blocks,
            build_sparse_indexer_schedule,
            sparse_logits,
            valid_lens,
        )

        torch.manual_seed(3)
        draft = 6
        base = torch.tensor([20000, 16385, 70000], dtype=torch.int32, device="cuda")
        lens = (
            base[:, None] + torch.arange(draft, device="cuda", dtype=torch.int32)
        ).flatten()
        request_ids = torch.repeat_interleave(
            torch.tensor([7, 3, 11], dtype=torch.int64, device="cuda"), draft
        )
        rows = lens.numel()
        max_pages = (int(lens.max()) + PAGE - 1) // PAGE
        num_pages = base.numel() * max_pages
        pool = torch.randint(
            0, 255, (num_pages, PAGE * 68), dtype=torch.uint8, device="cuda"
        )
        pool[:, PAGE * 64 :] = torch.randint(
            118, 123, (num_pages, PAGE * 4), dtype=torch.uint8, device="cuda"
        )
        k_cache = pool.view(num_pages, PAGE, 1, 68)
        per_request = (
            torch.randperm(num_pages, device="cuda")
            .view(base.numel(), max_pages)
            .to(torch.int32)
        )
        page_table = per_request.repeat_interleave(draft, dim=0)  # rows share theirs
        q_fp4 = torch.randint(
            0, 255, (rows, 1, HEADS, HEAD_DIM // 2), dtype=torch.uint8, device="cuda"
        ).view(torch.int8)
        q_sf = (
            torch.randint(
                118, 123, (rows, 1, HEADS, 4), dtype=torch.uint8, device="cuda"
            )
            .view(torch.int32)
            .squeeze(-1)
        )
        weights = (torch.rand(rows, HEADS, device="cuda") * 0.05).to(torch.bfloat16)
        sched = deep_gemm.get_paged_mqa_logits_metadata(
            lens.view(-1, 1), PAGE, deep_gemm.get_num_sms()
        )
        dense = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            weights.float(),
            lens.view(-1, 1),
            page_table,
            sched,
            int(lens.max()),
            False,
            torch.float32,
        )
        nblocks, row_valid = candidate_row_lens(lens, BLOCKS)
        blocks = amax_topk_blocks(dense, lens, nblocks, BLOCKS)
        phys = sort_candidate_blocks(blocks, lens, page_table, PAGE)
        out = {}
        for name, ids in (("paired", request_ids), ("unpaired", None)):
            schedule = build_sparse_indexer_schedule(
                blocks, lens, page_table, PAGE, q_fp4.dtype, ids
            )
            table = SparseBlockTable(
                blocks=blocks, schedule=schedule, phys_blocks=phys, valid_lens=row_valid
            )
            out[name] = sparse_logits(q_fp4, q_sf, k_cache, weights, table)
        cols = torch.arange(BLOCKS * 8, device="cuda")
        valid = cols[None, :] < row_valid[:, None].long()
        self.assertTrue(torch.equal(row_valid, valid_lens(lens, BLOCKS)))
        self.assertTrue(
            torch.equal(out["paired"][valid], out["unpaired"][valid]),
            "pairing changed the sparse logits",
        )
        # and both equal the dense bf16 logits at the published positions
        dense16 = deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            weights,
            lens.view(-1, 1),
            page_table,
            sched,
            int(lens.max()),
            False,
            torch.bfloat16,
        )
        pos = blocks.long().repeat_interleave(8, dim=1) * 8 + (cols % 8)[None, :]
        ref = dense16.gather(1, pos.clamp(max=dense16.shape[1] - 1))
        self.assertTrue(torch.equal(out["paired"][valid], ref[valid]))


if __name__ == "__main__":
    unittest.main()
