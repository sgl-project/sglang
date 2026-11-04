import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    BuildDsparkSwaRingIndices,
    build_block_seq_lens_causal,
    build_dspark_swa_page_indices,
    compute_dspark_window_gather,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
    SparsePrefillKVBuffer,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    _CompressedPoolConfig,
    _num_dsv4_physical_kv_pages,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDSV4CompressedPools(CustomTestCase):
    def test_physical_kv_pages_cover_reserved_logical_page(self):
        size = 8192
        self.assertEqual(_num_dsv4_physical_kv_pages(size, 256, 256), 33)
        self.assertEqual(_num_dsv4_physical_kv_pages(size, 64, 256), 132)
        self.assertGreaterEqual(
            _num_dsv4_physical_kv_pages(size, 64, 256) * 64,
            size + 256,
        )

    def test_state_buf_item_covers_one_logical_swa_page(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.swa_page_size = 256
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        for physical_page_size in (256, 64):
            with self.subTest(physical_page_size=physical_page_size):
                row_bytes = physical_page_size * 4
                buf = torch.empty((8, row_bytes), dtype=torch.uint8)
                pool.swa_kv_pool = SimpleNamespace(
                    page_size=physical_page_size, kv_buffer=[buf]
                )
                data_ptrs, data_lens, item_lens = pool.get_state_buf_infos()
                self.assertEqual(data_ptrs, [buf.data_ptr()])
                self.assertEqual(data_lens, [buf.nbytes])
                self.assertEqual(item_lens, [256 * 4])

    def test_state_buf_infos_without_paged_swa(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = None
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_pp_mapping_and_pd_buffer_order(self):
        for unified, fp8, stage_ratios in product(
            (False, True), (False, True), ([4, 0, 128, 4], [128], [0])
        ):
            if fp8 and not unified:
                continue
            with self.subTest(unified=unified, fp8=fp8, stage_ratios=stage_ratios):
                pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
                pool._unified_kv = unified
                pool._unified_kv_fp8 = fp8
                pool.uniform_fp8 = False
                pool.kv_layout = KVLayout.V4
                pool.compressed_kv_layout_option = None
                pool.compressed_pool_configs = {
                    4: _CompressedPoolConfig(
                        256, 64, torch.bfloat16, indexer_size=1024
                    ),
                    128: _CompressedPoolConfig(512, 8, torch.float32),
                }
                pool.indexer_head_dim = 128
                pool.page_size = 256
                pool.compression_ratios = [128] + stage_ratios + [4]
                pool._stage_start = 1
                pool._stage_end = 1 + len(stage_ratios)

                def kv_factory(**kwargs):
                    return SimpleNamespace(
                        kv_buffer=[
                            torch.empty((3, kwargs["page_size"]), dtype=torch.uint8)
                            for _ in range(kwargs["layer_num"])
                        ]
                    )

                # Separate payload/scale buffers exercise the FP4 transfer contract.
                indexer_buffers = [
                    torch.empty((3, width), dtype=torch.uint8)
                    for _ in range(stage_ratios.count(4))
                    for width in (32, 2)
                ]
                indexer = SimpleNamespace(
                    contiguous_page_row_buffers=lambda: indexer_buffers
                )
                with (
                    patch.object(pool, "_make_kv_pool", side_effect=kv_factory),
                    patch.object(pool, "_make_indexer_pool", return_value=indexer),
                ):
                    pool._init_compressed_pools(
                        stage_ratios=stage_ratios,
                        page_size=256,
                        dtype=torch.float8_e4m3fn,
                        device="cpu",
                        enable_memory_saver=False,
                        enable_hisparse=False,
                        kv_pool_cls=DeepSeekV4SingleKVPool,
                    )
                pool._init_compressed_layer_mapping()
                self.assertIsNone(pool.layer_mapping[0])
                self.assertIsNone(pool.layer_mapping[-1])
                for local_id, ratio in enumerate(stage_ratios):
                    item = pool.layer_mapping[local_id + 1]
                    self.assertEqual(
                        item.compress_layer_id, stage_ratios[:local_id].count(ratio)
                    )
                    self.assertIs(item.compress_kv_pool, pool.kv_pools.get(ratio))
                self.assertIs(pool.c4_kv_pool, pool.kv_pools[4])
                self.assertIs(pool.c128_kv_pool, pool.kv_pools[128])
                self.assertIs(pool.c4_indexer_kv_pool, pool.index_pools[4])

                if unified:
                    buffers = [
                        torch.empty((9, 8), dtype=torch.uint8) for _ in stage_ratios
                    ]
                    rope_buffers = [
                        torch.empty((9, 16), dtype=torch.uint8) for _ in stage_ratios
                    ]
                    pool.unified_kv_pool = SimpleNamespace(
                        swa_pages=2,
                        kv_buffer=buffers,
                        kv_buffer_rope=rope_buffers,
                    )

                    def kv_entries(ratio):
                        return [
                            (buf.data_ptr() + 16, 56, 256 // ratio * 8)
                            for buf, r in zip(buffers, stage_ratios)
                            if r == ratio
                        ]

                    def rope_entries(ratio):
                        return [
                            (buf.data_ptr() + 32, 112, 256 // ratio * 16)
                            for buf, r in zip(rope_buffers, stage_ratios)
                            if r == ratio
                        ]

                else:

                    def kv_entries(ratio):
                        return [
                            (b.data_ptr(), b.nbytes, b[0].nbytes)
                            for b in pool.kv_pools[ratio].kv_buffer
                        ]

                    def rope_entries(ratio):
                        return []

                indexer_entries = [
                    (b.data_ptr(), b.nbytes, b[0].nbytes) for b in indexer_buffers
                ]
                if fp8:
                    expected = (
                        kv_entries(4)
                        + rope_entries(4)
                        + indexer_entries
                        + kv_entries(128)
                        + rope_entries(128)
                    )
                else:
                    expected = kv_entries(4) + indexer_entries + kv_entries(128)
                actual = list(zip(*pool.get_contiguous_buf_infos()))
                self.assertEqual(actual, expected)

    def test_shared_state_factory_preserves_layouts(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.compressed_pool_configs = {
            4: _CompressedPoolConfig(256, 64, torch.bfloat16, indexer_size=1024),
            128: _CompressedPoolConfig(512, 8, torch.float32),
        }
        pool.compression_ratios = [0, 4, 128]
        pool._stage_start, pool._stage_end = 0, 3
        pool.index_pools = {4: object()}
        pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64
        pool.indexer_head_dim = 128
        pool.device = "cpu"
        pool.swa_page_size = 128
        pool.online_mtp_max_draft_tokens = 3
        for online in (False, True):
            with (
                self.subTest(online=online),
                patch(
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", online
                ),
                patch.object(
                    pool,
                    "get_ring_size",
                    side_effect=lambda r: 8 if r == 4 else (1 if online else 128),
                ),
            ):
                pool._init_paged_compress_states(False)
            c4 = pool.compress_state_pools[1].kv_score_buffer.kv_score
            indexer = pool.indexer_compress_state_pools[1].kv_score_buffer.kv_score
            c128 = pool.compress_state_pools[2].kv_score_buffer.kv_score
            self.assertEqual(c4.shape, (76, 2048))
            self.assertEqual(indexer.shape, (76, 512))
            self.assertEqual(c128.shape, (40, 1536) if online else (256, 1024))
            self.assertEqual(c4.dtype, torch.bfloat16)
            self.assertEqual(indexer.dtype, torch.bfloat16)
            self.assertEqual(c128.dtype, torch.float32)
            self.assertNotEqual(c4.data_ptr(), indexer.data_ptr())
            self.assertIsNone(pool.compress_state_pools[0])
            self.assertIsNone(pool.indexer_compress_state_pools[2])

    def test_indexer_access_uses_layer_ratio_and_waits_only_for_reads(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.kv_pools = {4: None, 128: None}
        pool.compression_ratios = [4, 128, 4]
        pool._stage_start, pool._stage_end = 0, 3
        pool._init_compressed_layer_mapping()
        indexer = MagicMock(page_size=32)
        pool.index_pools = {4: indexer}
        trace = MagicMock()
        trace.attach_mock(indexer, "indexer")
        with patch.object(pool, "wait_layer_transfer") as wait:
            trace.attach_mock(wait, "wait")
            pool.get_index_k_fp4_payload_buffer(2)
            pool.get_low_ratio_index_k_dequant(2, "slots")
            pool.get_low_ratio_index_k_fp4(2, "slots")
            pool.set_index_k_fp4(2, "loc", "cache")
        self.assertEqual(
            trace.mock_calls,
            [
                unittest.mock.call.wait(2),
                unittest.mock.call.indexer.get_index_k_fp4_payload_buffer(1),
                unittest.mock.call.wait(2),
                unittest.mock.call.indexer.get_index_k_dequant(1, "slots"),
                unittest.mock.call.wait(2),
                unittest.mock.call.indexer.get_index_k_fp4(1, "slots"),
                unittest.mock.call.indexer.set_index_fp4(1, "loc", "cache"),
            ],
        )
        self.assertEqual(pool.get_index_k_page_size(4), 32)
        with self.assertRaisesRegex(AssertionError, "No indexer pool"):
            pool.get_index_k_page_size(128)


HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE


class TestV41KVPoolLayouts(CustomTestCase):
    """A V4.1-layout pool hands the attention kernel page-aligned buffers and
    picks the compressed layout each ratio asks for."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(page_size=PAGE_SIZE)
        override.install()
        self.addCleanup(override.restore)

    def make_pool(self, ratios, kv_source_layers, kv_layout, compressed=None, **sizes):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=16,
            swa_size=FULL_SIZE,
            c4_size=sizes.get("c4_size", 0),
            c128_size=sizes.get("c128_size", 0),
            c4_state_pool_size=sizes.get("c4_state_pool_size", 0),
            c128_state_pool_size=sizes.get("c128_state_pool_size", 0),
            page_size=PAGE_SIZE,
            swa_page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            indexer_head_dim=128,
            layer_num=len(ratios),
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=ratios,
            kv_source_layers=kv_source_layers,
            full_size=FULL_SIZE,
            kv_layout=kv_layout,
            compressed_kv_layout=compressed,
        )

    def assert_kernel_requirements(self, pool, layout):
        """Pages start on the kernel's alignment, and its
        (num_pages, page_size, 1, bytes_per_token) view walks one token per row."""
        for buf in pool.kv_buffer:
            self.assertEqual(buf.stride(0) % layout.page_align, 0)
            bpt = layout.bytes_per_token
            view = buf[:, : pool.page_size * bpt].view(
                buf.shape[0], pool.page_size, 1, bpt
            )
            self.assertEqual(view.stride(1), bpt)
            self.assertEqual(view.stride(0), pool.bytes_per_page_padded)

    def test_v41_pool_buffers(self):
        for main in (KVLayout.V41, KVLayout.V41_NEW):
            for option, encoding in (
                (None, KVLayout.V41_FP4),
                ("fp8", KVLayout.V41),
            ):
                expect = encoding.with_interleaved(main.is_interleaved)
                with self.subTest(main=main, compressed=option):
                    pool = self.make_pool([0, 0, 2, 1, 1], [2, 3], main, option)
                    self.assert_kernel_requirements(pool.swa_kv_pool, main)
                    self.assertIs(pool.get_swa_key_layout(), main)
                    self.assertEqual(pool.get_swa_key_bytes_per_token(), 528)
                    for ratio in (1, 2):
                        layer_id = pool.sources_by_ratio[ratio][0]
                        self.assertIs(pool.get_extra_key_layout(layer_id), expect)
                        self.assertEqual(
                            pool.get_extra_key_bytes_per_token(layer_id),
                            expect.bytes_per_token,
                        )
                        self.assertTrue(is_valid_kv_layout_pair(pool.kv_layout, expect))
                        self.assert_kernel_requirements(pool.kv_pools[ratio], expect)
        # A pool of an fp4 layout cannot be the main cache.
        for fp4 in (KVLayout.V41_FP4, KVLayout.V41_FP4_NEW):
            with self.assertRaises(AssertionError):
                self.make_pool([0], [], fp4)

    def test_v41_pool_with_c4_c128(self):
        # The 2-token c128 page is the only production page the block-split
        # layout pads; the interleaved one never pads.
        for main, c128_page_bytes in ((KVLayout.V41, 1536), (KVLayout.V41_NEW, 1056)):
            with self.subTest(main=main):
                pool = self.make_pool(
                    [0, 4, 128],
                    [],
                    main,
                    c4_size=PAGE_SIZE,
                    c128_size=PAGE_SIZE,
                    c4_state_pool_size=16,
                    c128_state_pool_size=16,
                )
                for ratio in (4, 128):
                    self.assertEqual(pool.kv_pools[ratio].page_size, PAGE_SIZE // ratio)
                    self.assertIs(pool.kv_pools[ratio].kv_layout, main)
                self.assertEqual(
                    pool.kv_pools[128].bytes_per_page_padded, c128_page_bytes
                )

    def test_pd_layout_names_the_arrangement(self):
        """Prefill and decode servers must refuse to pair the two arrangements:
        their pages have the same size, so a mismatch would read garbage."""
        names = {
            main: self.make_pool([0, 0, 2, 1, 1], [2, 3], main).pd_kv_layout()
            for main in (KVLayout.V41, KVLayout.V41_NEW)
        }
        self.assertNotEqual(names[KVLayout.V41], names[KVLayout.V41_NEW])
        self.assertIn("v41_fp4_new", names[KVLayout.V41_NEW])

    def test_pd_entries_cover_the_c4_kv_page_count(self):
        """The PD transfer walks every registered entry with the c4 KV pool's
        page count, so the c4 indexer must reserve the same FULL logical page."""
        pool = self.make_pool(
            [0, 4, 128],
            [],
            KVLayout.V4,
            c4_size=PAGE_SIZE,
            c128_size=PAGE_SIZE // 32,
            c4_state_pool_size=16,
            c128_state_pool_size=16,
        )
        self.assertEqual(pool.c4_indexer_kv_pool.size, pool.c4_kv_pool.size)
        _, data_lens, item_lens = pool.get_contiguous_buf_infos()
        rows = [n // item for n, item in zip(data_lens, item_lens)]
        c4_kv_rows = _num_dsv4_physical_kv_pages(PAGE_SIZE, PAGE_SIZE // 4, PAGE_SIZE)
        self.assertEqual(rows[:2], [c4_kv_rows] * 2)  # c4 KV, then its indexer
        self.assertEqual(min(rows), c4_kv_rows)


class TestDSparkDraftRingWithEncoderReplay(CustomTestCase):
    RING = 256  # sliding window + the largest draft token count, in whole pages

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
            speculative_dspark_block_size=5,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
        )
        override.install()
        self.addCleanup(override.restore)

    def make_pool(self, *, draft):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=2,
            num_req_slots=3,
            swa_size=1024,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.bfloat16,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 0, 0],
            online_mtp_max_draft_tokens=6,
            full_size=2048,
            is_draft_worker=draft,
        )

    def test_no_swa_state_needs_the_paged_allocator(self):
        target = self.make_pool(draft=False)
        draft = self.make_pool(draft=True)
        self.assertFalse(target.needs_paged_swa_allocator)
        self.assertFalse(draft.needs_paged_swa_allocator)
        self.assertEqual(draft.swa_req_ring_size, self.RING)
        self.assertGreaterEqual(draft.swa_kv_pool.size, 3 * self.RING)

    def test_pd_swa_state_pairs_with_a_paged_decode(self):
        target = self.make_pool(draft=False)
        draft = self.make_pool(draft=True)
        self.assertEqual(target.request_window.capacity, draft.swa_req_ring_size)
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=False
        )
        override.install()
        try:
            decode_target = self.make_pool(draft=False)
            decode_draft = self.make_pool(draft=True)
        finally:
            override.restore()
        for prefill, decode in ((target, decode_target), (draft, decode_draft)):
            self.assertEqual(
                prefill.get_state_buf_infos()[2], decode.get_state_buf_infos()[2]
            )

    def test_dspark_draft_keeps_the_mark_for_either_storage(self):
        # Ring or paged, a DSpark draft sends one request-row item, so the PD
        # roles' state pairs; another algorithm's draft and the target keep none.
        item = torch.zeros(1, dtype=torch.int64).nbytes
        for algorithm, encoder_replay, expected in (
            ("DSPARK", True, [item]),
            ("DSPARK", False, [item]),
            ("EAGLE", False, []),
        ):
            with self.subTest(algorithm=algorithm, encoder_replay=encoder_replay):
                override = get_context().override_server_args(
                    speculative_algorithm=algorithm,
                    enable_encoder_swa_bounded_replay=encoder_replay,
                )
                override.install()
                try:
                    target = self.make_pool(draft=False)
                    draft = self.make_pool(draft=True)
                finally:
                    override.restore()
                self.assertEqual(draft.get_swa_valid_from_buf_infos()[2], expected)
                self.assertEqual(target.get_swa_valid_from_buf_infos(), ([], [], []))

    def test_pd_window_pages_carry_the_last_window(self):
        window = self.make_pool(draft=False).request_window
        page, slot, sliding = window.page_size, 2, 128
        for seq_len in (300, 1000, 1030):
            with self.subTest(seq_len=seq_len):
                # A row holds the newest committed position with its residue.
                held = torch.full((window.rows + page,), -1)
                newest = torch.arange(seq_len - window.capacity, seq_len)
                held[window.row_loc(slot, newest)] = newest
                start = (seq_len - sliding) // page * page
                pages = torch.from_numpy(window.page_indices(slot, start, seq_len))
                # Decode lays the received pages out by position from start.
                received = held.view(-1, page)[pages.long()].flatten()
                last = torch.arange(seq_len - sliding, seq_len)
                self.assertEqual(received[last - start].tolist(), last.tolist())

    def _block_inputs(self, prefix_lens, slots, block_size=5):
        seq_lens_casual = build_block_seq_lens_causal(
            seq_lens=torch.tensor(prefix_lens),
            block_size=block_size,
            device="cpu",
        )
        req_pool_indices_repeated = torch.tensor(slots).repeat_interleave(block_size)
        return seq_lens_casual, req_pool_indices_repeated

    def test_ring_indices_match_paged_indices_over_ring_rows(self):
        prefix_lens, slots, block_size = [5, 200, 300], [2, 0, 1], 5
        seq_lens_casual, req_rep = self._block_inputs(prefix_lens, slots)
        # A paged mapping that sends every (slot, position) to its ring row.
        positions = torch.arange(512)
        req_to_token = torch.arange(3).view(-1, 1) * 1000 + positions
        mapping = torch.full((3000,), -1, dtype=torch.int64)
        for slot in range(3):
            mapping[req_to_token[slot]] = slot * self.RING + positions % self.RING
        out_loc = torch.cat(
            [req_to_token[s, p : p + block_size] for s, p in zip(slots, prefix_lens)]
        )
        # Per slot: a fresh request, and PD decode's marks from two prefix hits,
        # one inside the window and one past its start.
        for valid_from in ([0, 0, 0], [150, 290, 0]):
            with self.subTest(valid_from=valid_from):
                valid_from = torch.tensor(valid_from, dtype=torch.int64)
                gather = compute_dspark_window_gather(
                    seq_lens_casual=seq_lens_casual,
                    req_pool_indices_repeated=req_rep,
                    block_size=block_size,
                    swa_window=128,
                    valid_from=valid_from,
                )
                paged = build_dspark_swa_page_indices(
                    req_to_token=req_to_token,
                    full_to_swa_mapping=mapping,
                    req_pool_indices_per_request=gather.req_pool_indices_per_request,
                    offsets=gather.offsets,
                    invalid=gather.invalid,
                    out_loc=out_loc,
                    context_lens=gather.context_lens,
                    block_size=block_size,
                    swa_window=128,
                    page_index_aligned_size=64,
                )
                ring = BuildDsparkSwaRingIndices.torch(
                    seq_lens_casual=seq_lens_casual,
                    req_pool_indices_repeated=req_rep,
                    valid_from=valid_from,
                    ring_size=self.RING,
                    block_size=block_size,
                    swa_window=128,
                    page_index_aligned_size=64,
                )
                for got, want in zip(ring, paged):
                    torch.testing.assert_close(got, want)

    def test_ring_window_starts_at_valid_from(self):
        draft = self.make_pool(draft=True)
        # Slot 1 was placed at a 290-token prefix hit; 10 positions are its own.
        draft.set_swa_valid_from(torch.tensor([1]), torch.tensor([290]))
        seq_lens_casual, req_rep = self._block_inputs([300], [1])
        indices, lengths = BuildDsparkSwaRingIndices.torch(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_rep,
            valid_from=draft.swa_valid_from,
            ring_size=self.RING,
            block_size=5,
            swa_window=128,
            page_index_aligned_size=64,
        )
        self.assertEqual(lengths.tolist(), [15] * 5)
        rows = [self.RING + p % self.RING for p in range(290, 305)]
        self.assertEqual(indices[0, :15].tolist(), rows)
        self.assertTrue((indices[:, 15:] == -1).all())

    def test_injection_writes_only_the_last_window(self):
        pool = SimpleNamespace(swa_req_ring_size=self.RING, sliding_window=128)
        positions = torch.arange(300)
        loc = TargetHiddenKvInjector._ring_inject_loc(
            None,
            pool=pool,
            positions=positions,
            cache_loc_2d=None,
            commit_lens=None,
            state_slot=torch.full((300,), 2),
            final_pos=torch.full((300,), 299),
        )
        self.assertTrue((loc[:172] == -1).all())
        self.assertEqual(
            loc[172:].tolist(), (2 * self.RING + positions[172:] % self.RING).tolist()
        )


class TestEncoderReplayWithoutSpeculation(CustomTestCase):
    """Encoder replay without speculation has no paged SWA allocator, yet the
    sparse-prefill path still reads the pool's full-to-SWA mapping."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
        )
        override.install()
        self.addCleanup(override.restore)

    def _make_pool(self):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=2,
            num_req_slots=3,
            swa_size=0,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.bfloat16,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 0, 0],
            full_size=2048,
        )

    def test_workspace_covers_the_largest_prefill_graph_bucket(self):
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.BREAKABLE, bs=[64, 1024])
            ),
        )
        override.install()
        self.addCleanup(override.restore)
        history = (3 + 1) * 128  # (num_req_slots + 1) * sliding_window
        workspace = self._make_pool().request_window.workspace
        self.assertGreaterEqual(workspace.size, history + 1024)

    def test_sparse_prefill_chunk_cache_builds_without_mapping(self):
        pool = self._make_pool()
        self.assertIsNotNone(pool.request_window)
        self.assertFalse(pool.needs_paged_swa_allocator)
        self.assertIsNone(pool.full_to_swa_index_mapping)

        # A 300-token cold prefill of request slot 1 on the sparse-prefill path.
        rows = 300
        layout = window_layout(
            torch.ones(rows, dtype=torch.int64),
            torch.arange(rows),
            zero_row=pool.request_window.zero_row,
            capacity=pool.request_window.capacity,
            num_groups=1,
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.token_to_kv_pool = pool
        backend.req_to_token = torch.zeros((3, 2048), dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(late_layer_tail=None)
        backend.sparse_prefill_kv_buffer = SparsePrefillKVBuffer(torch.device("cpu"))
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens=torch.tensor([rows]),
            seq_lens_cpu=torch.tensor([rows]),
            extend_seq_lens=torch.tensor([rows], dtype=torch.int32),
            extend_seq_lens_cpu=[rows],
            req_pool_indices=torch.tensor([1]),
        )
        core = SimpleNamespace(
            request_window_layout=layout, seq_lens_casual=torch.arange(1, rows + 1)
        )
        build = SparsePrefillChunkCache.build
        empty = torch.empty(0, dtype=torch.int32)
        with (
            patch.object(SparsePrefillChunkCache, "build", wraps=build) as spy,
            # The combine is a Triton kernel; the request-window inputs are the contract.
            patch(
                "sglang.srt.layers.attention.dsv4.sparse_prefill_utils."
                "combine_topk_swa_indices",
                return_value=(empty, empty),
            ) as combine,
        ):
            cache = backend._build_sparse_prefill_chunk_cache(
                forward_batch, core, num_qo_tokens=rows
            )

        self.assertIsNone(spy.call_args.kwargs["full_to_swa"])
        self.assertIs(cache.swa_indices, layout.indices)
        self.assertIs(cache.swa_lengths, layout.lengths)
        self.assertIs(combine.call_args.kwargs["swa_indices"], layout.indices)
        torch.testing.assert_close(cache.query_pos, layout.pos.to(torch.int32))


if __name__ == "__main__":
    unittest.main()
