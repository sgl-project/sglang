"""DeepSeek V4.1 low compress ratios (1 and 2): the metadata, pool mapping and
torch compressor / indexer contracts the FlashMLA path relies on. CPU only; the
kernel-level checks live in test_deepseek_v4.py (c1 / c2 compress cases).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

INT32 = dict(dtype=torch.int32)


def _extend_forward_batch(*, req_pool_indices=(0,), extend_seq_lens=None):
    """Not decode, and without the CPU length copies, so the extend dispatchers
    take the torch path (the oracle these tests exercise)."""
    req = torch.tensor(req_pool_indices, dtype=torch.int64)
    lens = torch.tensor(
        extend_seq_lens if extend_seq_lens is not None else [1] * len(req),
        dtype=torch.int32,
    )
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: False,
            is_extend=lambda: True,
            is_target_verify=lambda: False,
        ),
        req_pool_indices=req,
        extend_seq_lens=lens,
        seq_lens_cpu=None,
        extend_seq_lens_cpu=None,
    )


class TestLowRatioCompressionMetadata(CustomTestCase):
    def test_out_loc_and_lengths(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_compression_metadata,
        )

        # Two requests: causal lengths 1..4 and 7..8, arbitrary full-pool locs.
        seq_lens = torch.tensor([1, 2, 3, 4, 7, 8], **INT32)
        raw_out_loc = torch.tensor([256, 257, 258, 259, 518, 519], **INT32)

        out_loc, lens = _low_ratio_compression_metadata(2, seq_lens, raw_out_loc)
        # Even causal length completes a pair; its latent lives at loc // 2.
        self.assertEqual(out_loc.tolist(), [-1, 128, -1, 129, -1, 259])
        self.assertEqual(lens.tolist(), [1, 1, 1, 2, 3, 4])  # (pos + 1) // 2, min 1
        self.assertEqual(out_loc.dtype, torch.int64)
        self.assertEqual(lens.dtype, torch.int32)

        out_loc, lens = _low_ratio_compression_metadata(1, seq_lens, raw_out_loc)
        self.assertEqual(out_loc.tolist(), raw_out_loc.tolist())
        self.assertEqual(lens.tolist(), seq_lens.tolist())

    def test_write_tokens_may_be_fewer_than_metadata_rows(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_compression_metadata,
        )

        # CP padding: metadata rows outnumber the real write tokens.
        seq_lens = torch.tensor([2, 4, 6, 8], **INT32)
        raw_out_loc = torch.tensor([10, 11], **INT32)
        out_loc, lens = _low_ratio_compression_metadata(2, seq_lens, raw_out_loc)
        self.assertEqual(out_loc.tolist(), [5, 5])
        self.assertEqual(lens.tolist(), [1, 2, 3, 4])

    def test_sparse_buffers(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_sparse_buffers,
        )

        lens_clamp1 = torch.tensor([1, 300, 600], **INT32)
        topk_lengths, page_indices, raw = _low_ratio_sparse_buffers(
            lens_clamp1, 512, is_prefill=False
        )
        self.assertEqual(topk_lengths.tolist(), [1, 300, 512])
        self.assertEqual(page_indices.shape, (3, 512))
        self.assertEqual(page_indices.dtype, torch.int32)
        self.assertTrue((page_indices == -1).all())
        self.assertIsNone(raw)
        _, page_indices, raw = _low_ratio_sparse_buffers(lens_clamp1, 512, True)
        self.assertEqual(raw.shape, page_indices.shape)


class TestLowRatioAttnMetadataFields(CustomTestCase):
    def _make(self, base: int, low_ratios):
        from sglang.srt.layers.attention.deepseek_v4_backend import DSV4AttnMetadata

        md = DSV4AttnMetadata(
            page_size=256,
            page_table=torch.tensor([[base + 1, base + 2]], **INT32),
            raw_out_loc=torch.tensor([base + 5], **INT32),
            cuda_int32_kwargs={"dtype": torch.int32},
            seq_lens_casual=torch.tensor([base + 7], **INT32),
            positions_casual=torch.tensor([base + 9], **INT32),
            swa_page_indices=torch.tensor([[base + 11, base + 12]], **INT32),
            swa_topk_lengths=torch.tensor([base + 15], **INT32),
            index_topk=128,
            low_ratios=low_ratios,
        )
        for name in (
            "c4_out_loc",
            "c128_out_loc",
            "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1",
            "c4_sparse_topk_lengths",
            "c128_topk_lengths_clamp1",
        ):
            setattr(md, name, torch.tensor([base + 20], **INT32))
        for name in (
            "c4_sparse_page_indices",
            "c4_sparse_raw_indices",
            "c128_page_indices",
        ):
            setattr(md, name, torch.tensor([[base + 30, base + 31]], **INT32))
        for ratio in low_ratios:
            setattr(md, f"c{ratio}_out_loc", torch.tensor([base + 40 + ratio], **INT32))
            setattr(
                md, f"c{ratio}_topk_lengths_clamp1", torch.tensor([base + 50], **INT32)
            )
            setattr(
                md, f"c{ratio}_sparse_topk_lengths", torch.tensor([base + 60], **INT32)
            )
            setattr(
                md,
                f"c{ratio}_sparse_page_indices",
                torch.tensor([[base + 70, base + 71]], **INT32),
            )
            setattr(
                md,
                f"c{ratio}_sparse_raw_indices",
                torch.tensor([[base + 80, base + 81]], **INT32),
            )
            setattr(md, f"c{ratio}_flashmla_metadata", object())
        for name in (
            "c0_flashmla_metadata",
            "c4_flashmla_metadata",
            "c128_flashmla_metadata",
        ):
            setattr(md, name, object())
        return md

    def test_copy_moves_low_ratio_fields(self):
        dst, src = self._make(0, (1, 2)), self._make(1000, (1, 2))
        dst.copy_(src)
        for ratio in (1, 2):
            self.assertEqual(dst.sparse_page_indices(ratio).tolist(), [[1070, 1071]])
            self.assertEqual(dst.sparse_raw_indices(ratio).tolist(), [[1080, 1081]])
            self.assertEqual(dst.sparse_topk_lengths(ratio).tolist(), [1060])
            self.assertIs(
                getattr(dst, f"c{ratio}_flashmla_metadata"),
                getattr(src, f"c{ratio}_flashmla_metadata"),
            )
        self.assertEqual(dst.c2_out_loc.tolist(), [1042])

    def test_copy_rejects_mismatched_low_ratios(self):
        dst, src = self._make(0, (1,)), self._make(1000, (1, 2))
        with self.assertRaises(AssertionError):
            dst.copy_(src)

    def test_absent_ratio_fields_stay_none(self):
        md = self._make(0, ())
        for ratio in (1, 2):
            self.assertIsNone(md.sparse_page_indices(ratio))
            self.assertIsNone(md.sparse_raw_indices(ratio))
            self.assertIsNone(md.get_flashmla_metadata(ratio))
        md.copy_(self._make(1000, ()))  # None / None fields are skipped

    def test_refresh_keeps_low_ratio_tensor_storage(self):
        capture, replay = self._make(0, (2,)), self._make(1000, (2,))
        kept = capture.c2_sparse_topk_lengths
        kept_pages = capture.c2_sparse_page_indices
        capture.refresh_for_breakable_cuda_graph_replay_(replay)
        # Lengths are copied into the captured storage; the top-k pages are the
        # indexer's in-graph output and are left alone.
        self.assertIs(capture.c2_sparse_topk_lengths, kept)
        self.assertEqual(kept.tolist(), [1060])
        self.assertIs(capture.c2_sparse_page_indices, kept_pages)
        self.assertEqual(kept_pages.tolist(), [[70, 71]])
        self.assertIs(capture.c2_flashmla_metadata, replay.c2_flashmla_metadata)


class TestLowRatioChunkCache(CustomTestCase):
    @staticmethod
    def _cache(max_seq_len):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillChunkCache,
        )

        return SparsePrefillChunkCache(
            num_reqs=2,
            num_qo_tokens=2,
            max_seq_len=max_seq_len,
            swa_window_size=128,
            swa_page_size=128,
            seq_lens=torch.tensor([max_seq_len, max_seq_len], **INT32),
            query_start_loc=torch.tensor([0, 1, 2], **INT32),
            swa_token_ids=torch.empty(0, **INT32),
            swa_first_pos=torch.zeros(2, **INT32),
            swa_gather_lens=torch.zeros(2, **INT32),
            swa_offsets=torch.zeros(3, **INT32),
        )

    def test_low_ratio_extents_and_slots(self):
        page_table = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], **INT32)
        for ratio, rows_per_page in ((1, 256), (2, 128)):
            for max_seq_len in (1, 2, 3, 255, 256, 259, 600):
                with self.subTest(ratio=ratio, max_seq_len=max_seq_len):
                    cache = self._cache(max_seq_len)
                    gather = cache.ensure_compressed(ratio, page_table, rows_per_page)
                    extent = max(max_seq_len // ratio, 1)
                    self.assertEqual(gather.flat_token_ids.numel(), 2 * extent)
                    self.assertEqual(gather.compressed_base.tolist(), [0, extent])
                    # Compressed position j of request 1 sits at page * rows + j % rows.
                    ids = gather.flat_token_ids[extent:].tolist()
                    expected = [
                        int(page_table[1, j // rows_per_page]) * rows_per_page
                        + j % rows_per_page
                        for j in range(extent)
                    ]
                    self.assertEqual(ids, expected)
                    self.assertIs(
                        cache.ensure_compressed(ratio, page_table, rows_per_page),
                        gather,
                    )


class TestLowRatioPoolMapping(CustomTestCase):
    @staticmethod
    def _pool(ratios, sources):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.compression_ratios = ratios
        pool.kv_source_layers = sources
        pool._stage_start, pool._stage_end = 0, len(ratios)
        pool.sources_by_ratio = pool._collect_sources_by_ratio()
        pool.kv_pools = {
            r: SimpleNamespace(name=f"c{r}") for r in pool.sources_by_ratio
        }
        pool._init_compressed_layer_mapping()
        return pool

    def test_layers_read_the_nearest_source_of_their_ratio(self):
        # Mirrors the released layout in miniature: two ratio-2 sources, one ratio-1.
        pool = self._pool([0, 0, 2, 2, 2, 2, 1, 1, 1], sources=[2, 4, 6])
        self.assertEqual(pool.source_layer_of(3), 2)
        self.assertEqual(pool.source_layer_of(4), 4)
        self.assertEqual(pool.source_layer_of(5), 4)
        self.assertEqual(pool.source_layer_of(8), 6)
        # compress_layer_id is the source's index inside its ratio's pool.
        self.assertEqual(pool.layer_mapping[3].compress_layer_id, 0)
        self.assertEqual(pool.layer_mapping[5].compress_layer_id, 1)
        self.assertIs(pool.layer_mapping[5].compress_kv_pool, pool.kv_pools[2])
        self.assertEqual(pool.layer_mapping[7].compress_layer_id, 0)
        self.assertIs(pool.layer_mapping[7].compress_kv_pool, pool.kv_pools[1])
        self.assertIsNone(pool.layer_mapping[0].compress_kv_pool)

    def test_layer_before_any_source_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._pool([0, 2, 2], sources=[2])


class _FakeCompressor:
    """project: kv = x, gate score = 0 (pairs average); finish: identity."""

    def __init__(self, ratio):
        self.compress_ratio = ratio

    def project(self, x):
        return x.float(), (
            torch.zeros_like(x, dtype=torch.float32)
            if self.compress_ratio > 1
            else None
        )

    def finish(self, kv):
        return kv.to(torch.bfloat16)

    @staticmethod
    def pool_pairs(kv2, score2):
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)


class _FakeIndexer:
    """Scores prefer the newest compressed position, so top-k is the last k positions."""

    def __init__(self, topk, *, candidate_source=False, uses_candidates=False):
        self.index_topk = topk
        self.is_candidate_source = candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks = 1
        self.candidate_block_size = 2
        self.owns_k = False

    def queries(self, q_lora, freqs):
        return q_lora

    def head_weights(self, x):
        return x

    def scores(self, q, k, weights):
        # [t, n]: score grows with the position index (k row order = position).
        return (
            torch.arange(k.shape[0], dtype=torch.float32).expand(q.shape[0], -1).clone()
        )


def _backend_with(core, pool, req_to_token):
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    backend = object.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(core_metadata=core)
    backend.token_to_kv_pool = pool
    backend.req_to_token = req_to_token
    backend.candidate_masks = None
    return backend


class TestLowRatioTorchIndexer(CustomTestCase):
    """The torch indexer must fill the c1/c2 top-k buffers with the contract the
    FlashMLA path and the sparse-prefill combine kernel assume: row t holds exactly
    min((pos + 1) // ratio, topk) valid slots first, -1 after."""

    def _run(self, ratio, positions, req, topk):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_sparse_buffers,
        )

        dim = 4
        T = len(positions)
        pos = torch.tensor(positions, dtype=torch.int64)
        req = torch.tensor(req, dtype=torch.int64)
        lens_clamp1 = ((pos + 1) // ratio).clamp_min(1).to(torch.int32)
        topk_lengths, page_indices, raw_indices = _low_ratio_sparse_buffers(
            lens_clamp1, topk, is_prefill=True
        )
        core = SimpleNamespace(
            sparse_page_indices=lambda r: page_indices,
            sparse_raw_indices=lambda r: raw_indices,
        )
        # Request r occupies full locs [1000 * r, ...) page-aligned (page 256).
        req_to_token = torch.zeros(4, 512, dtype=torch.int32)
        for r in range(4):
            req_to_token[r] = torch.arange(512) + 1024 * r
        pool = SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda layer_id, slots: torch.zeros(
                slots.numel(), dim, dtype=torch.bfloat16
            ),
        )
        backend = _backend_with(core, pool, req_to_token)
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=ratio,
            indexer=_FakeIndexer(topk),
            freqs_cis=torch.ones(1024, 2, dtype=torch.complex64),
        )
        x = torch.zeros(T, dim, dtype=torch.bfloat16)
        reqs, counts = torch.unique_consecutive(req, return_counts=True)
        backend._low_ratio_index_topk(
            layer,
            x,
            x,
            req,
            pos,
            _extend_forward_batch(
                req_pool_indices=reqs.tolist(), extend_seq_lens=counts.tolist()
            ),
        )
        return page_indices, raw_indices, topk_lengths, req_to_token

    def test_valid_prefix_matches_metadata_lengths(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                req = [0] * 5 + [1] * 8
                topk = 4
                pages, raws, lengths, req_to_token = self._run(
                    ratio, positions, req, topk
                )
                for t, (p, r) in enumerate(zip(positions, req)):
                    n = min((p + 1) // ratio, topk)
                    row = pages[t].tolist()
                    self.assertTrue(all(v >= 0 for v in row[:n]), (ratio, t, row[:n]))
                    self.assertTrue(all(v == -1 for v in row[n:]), (ratio, t))
                    self.assertEqual(int(lengths[t]), max(n, 1) if n else 1)
                    # Newest-first fake scores: the last n compressed positions.
                    visible = (p + 1) // ratio
                    expected_pos = list(range(max(visible - n, 0), visible))
                    self.assertEqual(raws[t, :n].tolist(), expected_pos)
                    expected_slots = [
                        int(req_to_token[r, j * ratio]) // ratio for j in expected_pos
                    ]
                    self.assertEqual(row[:n], expected_slots)


class TestLowRatioTorchCompressor(CustomTestCase):
    def test_ratio_two_pairs_across_chunks_and_writes_metadata_slots(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_compression_metadata,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4
        from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

        dim = 64
        state = CompressStatePool(
            size=4 * 2,
            ring_size=2,
            overlap=False,
            head_dim=dim,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=2,
        )
        writes = []
        pool = SimpleNamespace(
            get_attention_compress_states=lambda layer_id: state,
            source_index_k={},
            set_extra_key_buffer_fused=lambda layer_id, loc, cache_k: writes.append(
                (loc.clone(), cache_k.clone())
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=2,
            compressor=_FakeCompressor(2),
            indexer=None,
            rope_head_dim=2,
            freqs_cis=torch.polar(torch.ones(64, 1), torch.zeros(64, 1)),
        )
        torch.manual_seed(0)
        x_all = torch.randn(7, dim, dtype=torch.bfloat16)
        base_loc = 512  # one request, page-aligned

        def chunk(positions):
            pos = torch.tensor(positions, dtype=torch.int64)
            raw_out_loc = (base_loc + pos).to(torch.int32)
            out_loc, _ = _low_ratio_compression_metadata(
                2, (pos + 1).to(torch.int32), raw_out_loc
            )
            core = SimpleNamespace(
                raw_out_loc=raw_out_loc, c1_out_loc=None, c2_out_loc=out_loc
            )
            backend = _backend_with(core, pool, req_to_token=None)
            backend._low_ratio_compress(
                layer,
                x_all[pos],
                torch.zeros_like(pos),
                pos,
                _extend_forward_batch(
                    req_pool_indices=[0], extend_seq_lens=[len(positions)]
                ),
            )

        def pending(position):
            # Request 0 parks position p at ring slot p % 2.
            return state.kv_score_buffer[position % state.ring_size].kv

        # Chunk 1 ends on an even position: pairs (0,1) and (2,3) complete, 4 waits.
        # Every row writes -- an even one lands on the reserved dummy slot 0.
        chunk([0, 1, 2, 3, 4])
        self.assertEqual(len(writes), 1)
        slots, latent = writes[0]
        self.assertEqual(
            slots.tolist(), [0, (base_loc + 1) // 2, 0, (base_loc + 3) // 2, 0]
        )
        expected = torch.stack(
            [x_all[0:2].float().mean(0), x_all[2:4].float().mean(0)]
        ).to(torch.bfloat16)
        torch.testing.assert_close(latent[1::2], fake_quant_fp4(expected))
        torch.testing.assert_close(pending(4), x_all[4].float())

        # Chunk 2 starts on the odd partner: (4,5) pairs through the ring, 6 waits.
        chunk([5, 6])
        self.assertEqual(len(writes), 2)
        slots, latent = writes[1]
        self.assertEqual(slots.tolist(), [(base_loc + 5) // 2, 0])
        expected = x_all[4:6].float().mean(0, keepdim=True).to(torch.bfloat16)
        torch.testing.assert_close(latent[:1], fake_quant_fp4(expected))
        torch.testing.assert_close(pending(6), x_all[6].float())

    def test_ratio_one_writes_every_token(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_compression_metadata,
        )

        dim = 64
        writes = []
        pool = SimpleNamespace(
            source_index_k={},
            set_extra_key_buffer_fused=lambda layer_id, loc, cache_k: writes.append(
                loc.clone()
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=1,
            compressor=_FakeCompressor(1),
            indexer=None,
            rope_head_dim=2,
            freqs_cis=torch.polar(torch.ones(64, 1), torch.zeros(64, 1)),
        )
        pos = torch.tensor([3, 4, 5], dtype=torch.int64)
        raw_out_loc = (256 + pos).to(torch.int32)
        out_loc, _ = _low_ratio_compression_metadata(
            1, (pos + 1).to(torch.int32), raw_out_loc
        )
        core = SimpleNamespace(
            raw_out_loc=raw_out_loc, c1_out_loc=out_loc, c2_out_loc=None
        )
        backend = _backend_with(core, pool, req_to_token=None)
        backend._low_ratio_compress(
            layer,
            torch.randn(3, dim, dtype=torch.bfloat16),
            torch.zeros_like(pos),
            pos,
            _extend_forward_batch(req_pool_indices=[0], extend_seq_lens=[3]),
        )
        self.assertEqual(writes[0].tolist(), [259, 260, 261])


if __name__ == "__main__":
    unittest.main()


class _RowHashIndexer(_FakeIndexer):
    """Scores are a deterministic function of each query row, so chunking rows must
    not change them."""

    def scores(self, q, k, weights):
        seed = q[:, 0, 0].float().unsqueeze(-1)  # [t, 1]
        pos = torch.arange(k.shape[0], dtype=torch.float32)
        return torch.sin(seed * 7.0 + pos * 0.37) + 0.01 * pos


class TestLowRatioTorchIndexerChunking(CustomTestCase):
    """The score budget only bounds peak memory: chunked and unchunked runs must fill
    the top-k buffers and the candidate masks identically."""

    def _run(self, budget, *, ratio, topk, candidate_source, uses_candidates, masks):
        from sglang.srt.layers.attention import deepseek_v4_backend as be
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_sparse_buffers,
        )

        T, dim, heads = 37, 4, 3
        pos = torch.arange(T, dtype=torch.int64)
        req = torch.zeros(T, dtype=torch.int64)
        lens_clamp1 = ((pos + 1) // ratio).clamp_min(1).to(torch.int32)
        _, page_indices, raw_indices = _low_ratio_sparse_buffers(
            lens_clamp1, topk, is_prefill=True
        )
        core = SimpleNamespace(
            sparse_page_indices=lambda r: page_indices,
            sparse_raw_indices=lambda r: raw_indices,
        )
        req_to_token = torch.arange(512, dtype=torch.int32).unsqueeze(0)
        pool = SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda layer_id, slots: torch.zeros(
                slots.numel(), dim, dtype=torch.bfloat16
            ),
        )
        backend = _backend_with(core, pool, req_to_token)
        backend.candidate_masks = masks
        indexer = _RowHashIndexer(
            topk, candidate_source=candidate_source, uses_candidates=uses_candidates
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=ratio,
            indexer=indexer,
            freqs_cis=torch.zeros(4096),
        )
        g = torch.Generator().manual_seed(0)
        q = torch.randn(T, heads, dim, generator=g).to(torch.bfloat16)
        x = torch.randn(T, heads, generator=g).to(torch.bfloat16)
        saved = be._TORCH_INDEXER_SCORE_BUDGET_BYTES
        be._TORCH_INDEXER_SCORE_BUDGET_BYTES = budget
        try:
            backend._low_ratio_index_topk_torch(layer, x, q, req, pos)
        finally:
            be._TORCH_INDEXER_SCORE_BUDGET_BYTES = saved
        return page_indices, raw_indices, backend.candidate_masks

    def test_chunked_equals_unchunked(self):
        for ratio in (1, 2):
            for budget in (1, 200, 10**12):  # 1 row, a few rows, everything at once
                with self.subTest(ratio=ratio, budget=budget):
                    full = self._run(
                        10**12,
                        ratio=ratio,
                        topk=8,
                        candidate_source=True,
                        uses_candidates=False,
                        masks=None,
                    )
                    got = self._run(
                        budget,
                        ratio=ratio,
                        topk=8,
                        candidate_source=True,
                        uses_candidates=False,
                        masks=None,
                    )
                    self.assertTrue(torch.equal(got[0], full[0]))
                    self.assertTrue(torch.equal(got[1], full[1]))
                    self.assertEqual(len(got[2]), 1)
                    self.assertTrue(torch.equal(got[2][0], full[2][0]))
                    # Consumer layer: the published masks are sliced per chunk.
                    full_c = self._run(
                        10**12,
                        ratio=ratio,
                        topk=8,
                        candidate_source=False,
                        uses_candidates=True,
                        masks=full[2],
                    )
                    got_c = self._run(
                        budget,
                        ratio=ratio,
                        topk=8,
                        candidate_source=False,
                        uses_candidates=True,
                        masks=full[2],
                    )
                    self.assertTrue(torch.equal(got_c[0], full_c[0]))
                    self.assertTrue(torch.equal(got_c[1], full_c[1]))
