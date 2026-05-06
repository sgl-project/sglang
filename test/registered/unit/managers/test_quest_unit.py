"""Unit tests for QuestAlgorithm and its HiSparseCoordinator integration.

Two layers of tests:

1. ``TestQuestAlgorithm`` — pure-algorithm tests using synthetic K data,
   no HiSparseCoordinator.  Verifies that bounds are stored correctly
   for prefill (full + partial pages) and decode (incremental + page
   finalization), that ``invalidate_request`` resets all per-request
   state, and that ``retrieve_topk`` honours the contract (exactly
   ``top_k`` entries in ``[0, seq_len)``).

2. ``TestHiSparseQuestHooks`` — verifies that the coordinator wires
   QuestAlgorithm's hooks at the right call sites (admit, eager-backup,
   request_finished) by using a recording subclass.
"""

import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu

# Mirror config from test_hisparse_unit.py so we can reuse the same shape
# choices and stress similar cases.
SIZE = 2048
PAGE_SIZE_DEVICE = 64  # device-pool page size on CUDA (1 on ROCm)
TOP_K = 256
QUEST_PAGE_SIZE = 64
DEVICE_BUFFER_SIZE = 512
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048

# Quest-side dimensions for the synthetic pure-algorithm tests.
KV_HEADS = 4
HEAD_DIM = 32

# MHA pool dimensions for the hooks/integration tests.
MHA_HEAD_NUM = 4
MHA_HEAD_DIM = 64


def _make_req(rid="quest-req-0", origin_input_ids=None, output_ids=None):
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        req_pool_idx=None,
        kv_allocated_len=0,
        kv_committed_len=0,
        finished_reason=None,
        hisparse_staging=False,
        staging=False,
        is_chunked=0,
    )
    req.finished = lambda: req.finished_reason is not None
    return req


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest unit tests require CUDA/ROCm.",
)
class TestQuestAlgorithm(unittest.TestCase):
    """Pure-algorithm tests using synthetic K data."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")

    def _make_quest(self, top_k=TOP_K, page_size=QUEST_PAGE_SIZE, kv_heads=KV_HEADS,
                    head_dim=HEAD_DIM, max_reqs=MAX_NUM_REQS,
                    max_context_len=MAX_CONTEXT_LEN, num_layers=LAYER_NUM):
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        q = QuestAlgorithm(top_k=top_k, page_size=page_size, device=self.device)
        q.init_storage(
            start_layer=0,
            end_layer=num_layers,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            kv_heads=kv_heads,
            head_dim=head_dim,
        )
        return q

    def _make_k_buffer(self, pool_size, kv_heads=KV_HEADS, head_dim=HEAD_DIM, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(
            pool_size, kv_heads, head_dim,
            device=self.device, dtype=torch.bfloat16, generator=gen,
        )

    # ---------------------------------------------------------------- config

    def test_invalid_top_k_page_size_combo_rejected(self):
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        with self.assertRaises(ValueError):
            QuestAlgorithm(top_k=100, page_size=64, device=self.device)

    def test_init_storage_shapes_and_dtypes(self):
        q = self._make_quest()
        self.assertEqual(
            q.page_k_min.shape,
            (LAYER_NUM, MAX_NUM_REQS, MAX_CONTEXT_LEN // QUEST_PAGE_SIZE, KV_HEADS, HEAD_DIM),
        )
        self.assertEqual(q.page_k_min.dtype, torch.bfloat16)
        self.assertEqual(q.page_valid.shape, (LAYER_NUM, MAX_NUM_REQS,
                                              MAX_CONTEXT_LEN // QUEST_PAGE_SIZE))
        self.assertEqual(q.page_valid.dtype, torch.bool)
        self.assertEqual(q.running_k_min.shape,
                         (LAYER_NUM, MAX_NUM_REQS, KV_HEADS, HEAD_DIM))
        # Running buffers should start at (+inf, -inf) so the first observed
        # K initialises correctly via min/max accumulation.
        self.assertTrue(torch.all(q.running_k_min == torch.finfo(torch.bfloat16).max))
        self.assertTrue(torch.all(q.running_k_max == torch.finfo(torch.bfloat16).min))
        self.assertTrue(torch.all(q.running_token_count == 0))
        self.assertTrue(torch.all(q.running_page_idx == 0))

    # ----------------------------------------------------- prefill bounds

    def test_prefill_bounds_full_pages_only(self):
        """prefill_len exactly fills N pages → all bounds populated, no partial state."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=512)
        prefill_len = 192  # exactly 3 pages of 64
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)

        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=2,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        # First 3 pages valid, the rest still False.
        self.assertTrue(torch.all(q.page_valid[:, 2, :3]))
        self.assertFalse(torch.any(q.page_valid[:, 2, 3:]))
        # Bounds match a direct min/max over the slab.
        for layer_id in range(LAYER_NUM):
            for p in range(3):
                slab = pool[p * QUEST_PAGE_SIZE : (p + 1) * QUEST_PAGE_SIZE].float()
                expected_min = slab.amin(dim=0)
                expected_max = slab.amax(dim=0)
                actual_min = q.page_k_min[layer_id, 2, p].float()
                actual_max = q.page_k_max[layer_id, 2, p].float()
                # bf16 precision tolerance
                torch.testing.assert_close(actual_min, expected_min, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(actual_max, expected_max, atol=1e-2, rtol=1e-2)

        # Running counters: token_count == 0, page_idx == 3 (next page to fill).
        self.assertEqual(int(q.running_token_count[2]), 0)
        self.assertEqual(int(q.running_page_idx[2]), 3)

    def test_prefill_bounds_partial_last_page_seeds_running(self):
        """prefill_len = N*page_size + r → N pages valid, running primed with r tokens."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=512)
        prefill_len = 200  # 3 full pages + 8 partial tokens
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)

        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=1,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        self.assertTrue(torch.all(q.page_valid[:, 1, :3]))
        self.assertFalse(torch.any(q.page_valid[:, 1, 3:]))

        # Running buffers seeded from the partial 8 tokens (positions 192..199)
        for layer_id in range(LAYER_NUM):
            partial = pool[192:200].float()
            torch.testing.assert_close(
                q.running_k_min[layer_id, 1].float(), partial.amin(dim=0),
                atol=1e-2, rtol=1e-2,
            )
            torch.testing.assert_close(
                q.running_k_max[layer_id, 1].float(), partial.amax(dim=0),
                atol=1e-2, rtol=1e-2,
            )
        self.assertEqual(int(q.running_token_count[1]), 8)
        self.assertEqual(int(q.running_page_idx[1]), 3)

    def test_prefill_bounds_short_seq_no_full_page(self):
        """prefill_len < page_size → no pages valid, running seeded with all tokens."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=128)
        prefill_len = 30
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)

        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=0,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        self.assertFalse(torch.any(q.page_valid[:, 0, :]))
        self.assertEqual(int(q.running_token_count[0]), 30)
        self.assertEqual(int(q.running_page_idx[0]), 0)

    # ----------------------------------------------------- decode bounds

    def test_decode_bounds_finalise_completed_page(self):
        """Decode adds tokens until a page completes; bounds get copied from running."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=512, seed=11)
        # Prefill exactly 64 tokens (1 full page).  Then decode 64 more — at
        # token 128 the second page completes.
        prefill_len = 64
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=3,
                                    k_buffer=pool, prefill_indices=prefill_indices)
        self.assertEqual(int(q.running_token_count[3]), 0)
        self.assertEqual(int(q.running_page_idx[3]), 1)

        # Drive decode for 64 steps: each step "previous token" is at pool index 64+i.
        req_indices = torch.tensor([3], dtype=torch.int64, device=self.device)
        for step in range(QUEST_PAGE_SIZE):
            tok_pool_idx = 64 + step
            device_locs = torch.tensor([tok_pool_idx], dtype=torch.int64, device=self.device)
            for layer_id in range(LAYER_NUM):
                q.update_decode_representations(layer_id, req_indices, k_buffer=pool,
                                       device_locs=device_locs)
            q.maybe_finalize_decode_representations(req_indices)

        # Page 1 should now be valid; page 0 already valid from prefill.
        self.assertTrue(torch.all(q.page_valid[:, 3, :2]))
        # Bounds for page 1 should match a direct min/max over the 64 decode K.
        for layer_id in range(LAYER_NUM):
            slab = pool[64:128].float()
            torch.testing.assert_close(
                q.page_k_min[layer_id, 3, 1].float(), slab.amin(dim=0),
                atol=1e-2, rtol=1e-2,
            )
            torch.testing.assert_close(
                q.page_k_max[layer_id, 3, 1].float(), slab.amax(dim=0),
                atol=1e-2, rtol=1e-2,
            )
        # Running state: counter wrapped to 0, page_idx advanced to 2,
        # buffers reset to (+inf, -inf).
        self.assertEqual(int(q.running_token_count[3]), 0)
        self.assertEqual(int(q.running_page_idx[3]), 2)
        self.assertTrue(torch.all(q.running_k_min[:, 3] == torch.finfo(torch.bfloat16).max))
        self.assertTrue(torch.all(q.running_k_max[:, 3] == torch.finfo(torch.bfloat16).min))

    def test_decode_bounds_continues_partial_prefill(self):
        """Prefill leaves a partial page; decode tops it up to a full page."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=256, seed=22)
        prefill_len = 50  # 0 full pages, 50 partial
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=4,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        # Decode 14 more tokens to fill page 0 (50 + 14 == 64).
        req_indices = torch.tensor([4], dtype=torch.int64, device=self.device)
        for i in range(14):
            tok_pool_idx = 50 + i
            device_locs = torch.tensor([tok_pool_idx], dtype=torch.int64, device=self.device)
            for layer_id in range(LAYER_NUM):
                q.update_decode_representations(layer_id, req_indices, k_buffer=pool,
                                       device_locs=device_locs)
            q.maybe_finalize_decode_representations(req_indices)

        self.assertTrue(torch.all(q.page_valid[:, 4, 0]))
        # Bounds = min/max over all 64 tokens (50 partial-prefill + 14 decode).
        for layer_id in range(LAYER_NUM):
            slab = pool[:64].float()
            torch.testing.assert_close(
                q.page_k_min[layer_id, 4, 0].float(), slab.amin(dim=0),
                atol=1e-2, rtol=1e-2,
            )
        self.assertEqual(int(q.running_page_idx[4]), 1)
        self.assertEqual(int(q.running_token_count[4]), 0)

    # --------------------------------------------------------- invalidate

    def test_invalidate_request_resets_all_per_req_state(self):
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=512)
        prefill_indices = torch.arange(150, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=5,
                                    k_buffer=pool, prefill_indices=prefill_indices)
        self.assertTrue(torch.any(q.page_valid[:, 5, :]))
        self.assertNotEqual(int(q.running_token_count[5]), 0)

        q.invalidate_request(5)

        self.assertFalse(torch.any(q.page_valid[:, 5, :]))
        self.assertEqual(int(q.running_token_count[5]), 0)
        self.assertEqual(int(q.running_page_idx[5]), 0)
        self.assertTrue(torch.all(q.running_k_min[:, 5] == torch.finfo(torch.bfloat16).max))
        self.assertTrue(torch.all(q.running_k_max[:, 5] == torch.finfo(torch.bfloat16).min))

    def test_invalidate_does_not_affect_other_reqs(self):
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=512)
        for r in (1, 2, 3):
            for layer_id in range(LAYER_NUM):
                q.update_prefill_representations(layer_id, req_pool_idx=r,
                                        k_buffer=pool,
                                        prefill_indices=torch.arange(128, dtype=torch.int32,
                                                                     device=self.device))
        q.invalidate_request(2)
        self.assertTrue(torch.all(q.page_valid[:, 1, :2]))
        self.assertFalse(torch.any(q.page_valid[:, 2, :]))
        self.assertTrue(torch.all(q.page_valid[:, 3, :2]))

    # ---------------------------------------------------------- retrieve

    def test_retrieve_topk_contract(self):
        """retrieve_topk returns (positions [bs, top_k], actual_lens [bs]).
        Positions in [0, seq_len). actual_len = min(seq_len, top_k)."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=2048, seed=99)
        prefill_len = 800  # > top_k (256) — long-seq path
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=0,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=self.device)
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        queries = torch.randn(1, KV_HEADS, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        q.prepare_step(seq_lens)

        positions, actual_lens = q.retrieve_topk(
            queries, layer_id=0,
            req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        )

        self.assertEqual(positions.shape, (1, TOP_K))
        self.assertEqual(positions.dtype, torch.int32)
        self.assertEqual(actual_lens.shape, (1,))
        self.assertEqual(actual_lens.dtype, torch.int32)
        self.assertTrue(torch.all(positions >= 0))
        self.assertTrue(torch.all(positions < prefill_len))
        # Long-seq: actual_len == top_k.
        self.assertEqual(int(actual_lens[0]), TOP_K)

    def test_retrieve_topk_long_seq_includes_recent_window(self):
        """Long-seq layout: last `page_size` slots are the recent window
        covering the most recent page_size token positions."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=2048, seed=99)
        prefill_len = 800  # > top_k
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=0,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=self.device)
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        queries = torch.randn(1, KV_HEADS, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        q.prepare_step(seq_lens)

        positions, _ = q.retrieve_topk(
            queries, layer_id=0,
            req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        )
        # Last QUEST_PAGE_SIZE slots == positions [seq_len - page_size, seq_len).
        recent_slice = positions[0, -QUEST_PAGE_SIZE:].cpu().tolist()
        expected = list(range(prefill_len - QUEST_PAGE_SIZE, prefill_len))
        self.assertEqual(recent_slice, expected,
                         f"recent window mismatch: {recent_slice} != {expected}")

    def test_retrieve_topk_short_seq_dense_with_actual_len(self):
        """When seq_len < top_k, layout is dense [0..seq_len), actual_len=seq_len.
        No duplicate-position over-weighting in softmax."""
        q = self._make_quest()
        pool = self._make_k_buffer(pool_size=256, seed=77)
        prefill_len = 128  # 2 full pages, < top_k (256)
        prefill_indices = torch.arange(prefill_len, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=0,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=self.device)
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        queries = torch.randn(1, KV_HEADS, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
        q.prepare_step(seq_lens)

        positions, actual_lens = q.retrieve_topk(
            queries, layer_id=0,
            req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        )
        # Short-seq actual_len equals seq_len.
        self.assertEqual(int(actual_lens[0]), prefill_len)
        # First seq_len entries are the dense [0..seq_len) sequence.
        prefix = positions[0, :prefill_len].cpu().tolist()
        self.assertEqual(prefix, list(range(prefill_len)))
        # Beyond actual_len: clamped to seq_len - 1 (caller respects actual_len).
        tail_max = positions[0, prefill_len:].max().item()
        self.assertEqual(int(tail_max), prefill_len - 1)

    def test_retrieve_topk_picks_high_score_pages(self):
        """When one page has dramatically larger K bounds along the query
        direction, retrieve_topk must include it."""
        q = self._make_quest()
        pool = torch.zeros(2048, KV_HEADS, HEAD_DIM,
                           device=self.device, dtype=torch.bfloat16)
        # Put very large K at positions belonging to page 5 (320..383).
        pool[320:384] = 5.0
        # 10 pages = 640 positions. top_k=256 so this is short-seq path
        # (seq_len 640 > top_k 256? Actually seq_len > top_k → long-seq path).
        # Wait: seq_len = 640, top_k = 256, so seq_len > top_k → long-seq.
        prefill_indices = torch.arange(640, dtype=torch.int32, device=self.device)
        for layer_id in range(LAYER_NUM):
            q.update_prefill_representations(layer_id, req_pool_idx=0,
                                    k_buffer=pool, prefill_indices=prefill_indices)

        seq_lens = torch.tensor([640], dtype=torch.int32, device=self.device)
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        # Strictly positive query ⇒ formula uses k_max ⇒ page 5 wins big.
        queries = torch.ones(1, KV_HEADS, HEAD_DIM,
                             device=self.device, dtype=torch.bfloat16)
        q.prepare_step(seq_lens)

        positions, _ = q.retrieve_topk(
            queries, layer_id=0,
            req_pool_indices=req_pool_indices, seq_lens=seq_lens,
        )
        # Page 5's token range must appear in the output (in the
        # selected-pages portion, i.e. before the recent-window tail).
        page5_positions = set(range(320, 384))
        emitted = set(int(p) for p in positions[0].cpu().tolist())
        self.assertTrue(page5_positions.issubset(emitted),
                        f"page 5 not covered; emitted={sorted(emitted)}")


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest hooks tests require CUDA/ROCm.",
)
class TestHiSparseQuestHooks(unittest.TestCase):
    """Verify HiSparseCoordinator wires QuestAlgorithm hooks at the right sites."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29699")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.memory_pool_host import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        global_page_size = 1 if is_hip() else PAGE_SIZE_DEVICE

        from sglang.srt.mem_cache.hisparse_memory_pool import (
            HiSparseMHATokenToKVPool,
            HiSparseNSATokenToKVPool,
            HiSparseTokenToKVPoolAllocator,
        )

        # NSA pool + allocator for the constructor-rejection tests (which
        # pass mode=quest with NSA pool to verify TypeError dispatch).
        cls.nsa_device_pool = HiSparseNSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.nsa_allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.nsa_device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        # MHA pool + allocator for the actual quest-mode coord (the hooks
        # tests).  HiSparseCoordinator(mode=quest) requires this pool type.
        cls.mha_device_pool = HiSparseMHATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            head_num=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.mha_allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.mha_device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )
        cls.page_size_device = global_page_size

        # Backward-compat aliases for the constructor tests (NSA-based).
        cls.allocator = cls.nsa_allocator
        cls.device_pool = cls.nsa_device_pool

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _make_quest_and_coord(self):
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            MODE_QUEST,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        # MHA pool's get_key_buffer returns [size, head_num, head_dim].
        quest = QuestAlgorithm(
            top_k=TOP_K, page_size=QUEST_PAGE_SIZE, device=torch.device("cuda"),
        )
        quest.init_storage(
            start_layer=0,
            end_layer=LAYER_NUM,
            max_reqs=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            kv_heads=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
        )
        coord = HiSparseCoordinator(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.mha_allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=self.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            mode=MODE_QUEST,
            quest_algorithm=quest,
        )
        return quest, coord

    def setUp(self):
        # Reset shared allocator/pool state between tests.
        self.allocator.clear()
        self.req_to_token_pool.clear()

    # ----------------------------------------------------- constructor

    def test_constructor_rejects_unknown_mode(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        with self.assertRaises(ValueError):
            HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                device="cuda",
                tp_group=self.tp_group,
                mode="not_a_mode",
            )

    def test_constructor_quest_mode_requires_algorithm(self):
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            MODE_QUEST,
        )

        with self.assertRaises(ValueError):
            HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                device="cuda",
                tp_group=self.tp_group,
                mode=MODE_QUEST,
            )

    def test_constructor_quest_algorithm_must_be_initialised(self):
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            MODE_QUEST,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        unprimed = QuestAlgorithm(
            top_k=TOP_K, page_size=QUEST_PAGE_SIZE, device=torch.device("cuda"),
        )
        with self.assertRaises(RuntimeError):
            HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                device="cuda",
                tp_group=self.tp_group,
                mode=MODE_QUEST,
                quest_algorithm=unprimed,
            )

    def test_constructor_quest_top_k_must_match(self):
        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            MODE_QUEST,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        mismatched = QuestAlgorithm(
            top_k=128, page_size=QUEST_PAGE_SIZE, device=torch.device("cuda"),
        )
        mismatched.init_storage(0, LAYER_NUM, MAX_NUM_REQS,
                                MAX_CONTEXT_LEN, 1, KV_CACHE_DIM)
        with self.assertRaises(ValueError):
            HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                device="cuda",
                tp_group=self.tp_group,
                mode=MODE_QUEST,
                quest_algorithm=mismatched,
            )

    def test_constructor_rejects_quest_algo_in_dsa_mode(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        q = QuestAlgorithm(top_k=TOP_K, page_size=QUEST_PAGE_SIZE,
                           device=torch.device("cuda"))
        q.init_storage(0, LAYER_NUM, MAX_NUM_REQS, MAX_CONTEXT_LEN, 1, KV_CACHE_DIM)
        with self.assertRaises(ValueError):
            HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                top_k=TOP_K,
                device_buffer_size=DEVICE_BUFFER_SIZE,
                device="cuda",
                tp_group=self.tp_group,
                # mode defaults to dsa_native
                quest_algorithm=q,
            )

    # ----------------------------------------------------- request_finished

    def test_request_finished_invalidates_quest_state(self):
        quest, coord = self._make_quest_and_coord()
        # Manually populate quest state for slot 3 so we can verify it gets cleared.
        # Replace sentinel bounds with real values to mark pages "valid"
        # (page_valid property derives from k_max != -inf).
        quest.page_k_max[:, 3, :5] = 0.0
        quest.page_k_min[:, 3, :5] = 0.0
        quest.running_token_count[3] = 7
        quest.running_page_idx[3] = 12
        quest.running_k_min[:, 3] = 0.5

        # Build a Req that will pass through request_finished's free path.
        req = _make_req()
        req.req_pool_idx = 3
        req.kv_allocated_len = 0  # nothing actually allocated, free is a no-op

        coord.request_finished(req)

        self.assertFalse(torch.any(quest.page_valid[:, 3, :]))
        self.assertEqual(int(quest.running_token_count[3]), 0)
        self.assertEqual(int(quest.running_page_idx[3]), 0)
        self.assertTrue(torch.all(quest.running_k_min[:, 3]
                                  == torch.finfo(torch.bfloat16).max))


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "Quest+FlashInfer integration requires CUDA/ROCm.",
)
class TestQuestFlashInferIntegration(unittest.TestCase):
    """Real components (no mocks) end-to-end: MHA pool + Coord + Quest +
    FlashInfer backend.  Exercises the full call chain through
    forward_decode for a small batch in the FAST PATH (seq_len <=
    device_buffer_size, so the swap-in kernel doesn't need host DMA)."""

    @classmethod
    def setUpClass(cls):
        try:
            import flashinfer  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"flashinfer not available: {e}")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29799")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.memory_pool_host import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        global_page_size = 1 if is_hip() else PAGE_SIZE_DEVICE

        from sglang.srt.mem_cache.hisparse_memory_pool import (
            HiSparseMHATokenToKVPool,
            HiSparseTokenToKVPoolAllocator,
        )

        cls.device_pool = HiSparseMHATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            head_num=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device="cuda",
            enable_memory_saver=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import (
            HiSparseCoordinator,
            MODE_QUEST,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import (
            QuestAlgorithm,
        )

        cls.quest = QuestAlgorithm(
            top_k=TOP_K, page_size=QUEST_PAGE_SIZE, device=torch.device("cuda"),
        )
        cls.quest.init_storage(
            start_layer=0,
            end_layer=LAYER_NUM,
            max_reqs=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            kv_heads=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
        )
        cls.coord = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
            mode=MODE_QUEST,
            quest_algorithm=cls.quest,
        )

        from sglang.srt.layers.attention.flashinfer_hisparse_backend import (
            FlashInferHiSparseDecodeBackend,
        )

        cls.backend = FlashInferHiSparseDecodeBackend(
            quest_algorithm=cls.quest,
            coordinator=cls.coord,
            num_qo_heads=MHA_HEAD_NUM,  # MHA: q_heads == kv_heads
            num_kv_heads=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
            max_bs=MAX_NUM_REQS,
            kv_data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
            device=torch.device("cuda"),
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.memory_pool_host import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        # Class-level fixtures are shared across tests; wipe per-test state
        # so each test starts from a clean slate.
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coord.mem_pool_host.clear()
        self.coord.req_to_device_buffer.zero_()
        self.coord.req_device_buffer_size.zero_()
        self.coord.req_to_host_pool.fill_(-1)
        self.coord.req_device_buffer_tokens.fill_(-1)
        self.coord.req_device_buffer_token_locs.fill_(-1)
        self.coord.lru_slots[:] = self.coord._lru_init.view(1, 1, -1)
        self.coord.ack_staging_queue.clear()
        self.coord._has_pending_backup = False
        for i in range(len(self.coord._skip_first_backup)):
            self.coord._skip_first_backup[i] = False
        # Quest per-request state — reset to sentinels (post-init state).
        self.quest.page_k_min.fill_(torch.finfo(torch.bfloat16).max)
        self.quest.page_k_max.fill_(torch.finfo(torch.bfloat16).min)
        self.quest.running_token_count.zero_()
        self.quest.running_page_idx.zero_()
        self.quest.running_k_min.fill_(torch.finfo(torch.bfloat16).max)
        self.quest.running_k_max.fill_(torch.finfo(torch.bfloat16).min)

    # ----------------------------------------------------- helpers

    def _make_quest_req(self, prefill_len: int, rid: str = "quest-req-0"):
        """Allocate a req slot + KV; write random K/V to the device pool.

        Mirrors what real prefill would have done before staging admits.
        Returns the req SimpleNamespace and the allocated kv_loc tensor.
        """
        req = SimpleNamespace(
            rid=rid,
            origin_input_ids=list(range(prefill_len)),
            output_ids=[],
            fill_ids=list(range(prefill_len)),
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            finished_reason=None,
            hisparse_staging=False,
            staging=False,
            is_chunked=0,
        )
        req.finished = lambda: req.finished_reason is not None

        idx = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(idx)
        device = self.allocator.device

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([prefill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([prefill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=prefill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(0, len(kv_loc))), kv_loc
        )
        req.kv_allocated_len = prefill_len
        req.kv_committed_len = prefill_len

        # Write random K/V at the just-allocated device locations.
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        gen = torch.Generator(device="cuda").manual_seed(hash(rid) & 0xFFFF)
        for layer_id in range(LAYER_NUM):
            rand_k = torch.randn(
                prefill_len, MHA_HEAD_NUM, MHA_HEAD_DIM,
                dtype=torch.bfloat16, device="cuda", generator=gen,
            )
            rand_v = torch.randn(
                prefill_len, MHA_HEAD_NUM, MHA_HEAD_DIM,
                dtype=torch.bfloat16, device="cuda", generator=gen,
            )
            self.coord.mem_pool_device.k_buffer[layer_id][hisparse_locs] = rand_k
            self.coord.mem_pool_device.v_buffer[layer_id][hisparse_locs] = rand_v

        return req, kv_loc

    def _admit_and_complete(self, req):
        """Run the full admit pipeline: stage + wait + collect_ready_reqs."""
        self.coord.admit_request_into_staging(req)
        self.coord.write_staging_stream.synchronize()
        ready = self.coord.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertIs(ready[0], req)

    # ---------------------------------------------------- construction smoke

    def test_full_stack_constructed(self):
        """All components materialised — no exceptions on init."""
        self.assertIsNotNone(self.coord.mem_pool_host)
        # Coordinator picked the MHA host pool because mode=quest.
        from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
        self.assertIsInstance(self.coord.mem_pool_host, MHATokenToKVPoolHost)
        # Quest is wired into coord.
        self.assertIs(self.coord.quest_algorithm, self.quest)

    def test_prepare_wrappers_for_bs_real_flashinfer(self):
        """Real flashinfer wrapper construction at multiple bs values."""
        bs_list = [1, 2, 4]
        self.backend.prepare_wrappers_for_bs(bs_list)
        for bs in bs_list:
            self.assertIn(bs, self.backend._wrappers)
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
            self.assertIsInstance(
                self.backend._wrappers[bs], BatchDecodeWithPagedKVCacheWrapper,
            )

    def test_init_forward_metadata_real_flashinfer(self):
        """init_forward_metadata creates+plans wrapper at bs."""
        bs = 3
        forward_batch = SimpleNamespace(
            batch_size=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device="cuda"),
            seq_lens=torch.full((bs,), 128, dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.arange(bs, dtype=torch.int64, device="cuda"),
            token_to_kv_pool=self.coord.mem_pool_device,
            hisparse_coordinator=self.coord,
        )
        self.backend.init_forward_metadata(forward_batch)
        self.assertEqual(self.backend._current_bs, bs)
        self.assertIs(self.backend._wrapper, self.backend._wrappers[bs])

    # ---------------------------------------------- lifecycle integration

    def test_admit_triggers_quest_prefill_bounds(self):
        """admit_request_into_staging populates Quest's page_valid + bounds.

        Verifies the wiring at hisparse_coordinator.py:_update_quest_prefill_representations:
        the prefill K rows on device are turned into per-page min/max for every
        layer, and the page_valid bits flip True for full pages only.
        """
        prefill_len = 384  # 6 full pages of 64
        req, _ = self._make_quest_req(prefill_len=prefill_len, rid="prefill-bounds")
        self._admit_and_complete(req)

        num_full_pages = prefill_len // QUEST_PAGE_SIZE
        valid_slice = self.quest.page_valid[:, req.req_pool_idx, :num_full_pages]
        self.assertTrue(torch.all(valid_slice),
                        f"expected all {LAYER_NUM} layers x {num_full_pages} pages "
                        f"valid, got {valid_slice}")
        # Bounds should reflect the random K values (not still zero).
        bounds_min = self.quest.page_k_min[:, req.req_pool_idx, :num_full_pages]
        bounds_max = self.quest.page_k_max[:, req.req_pool_idx, :num_full_pages]
        self.assertTrue(torch.any(bounds_min != 0))
        self.assertTrue(torch.any(bounds_max != 0))
        # Sanity: max >= min everywhere.
        self.assertTrue(torch.all(bounds_max >= bounds_min))

        self.coord.request_finished(req)

    def test_multi_step_decode_finalises_page_via_eager_backup(self):
        """Drive the production decode-bounds hook (``_eager_backup_previous_token``)
        enough times to complete a Quest page; verify ``page_valid`` flips and
        the running counters wrap.

        This is the integration coverage gap from the audit: the decode hook
        is wired in but no other test pipes K through the real coord path."""
        # Partial last page so decode is what completes it, not prefill.
        prefill_len = 200  # 3 full pages of 64 + 8 partial tokens
        req, _ = self._make_quest_req(prefill_len=prefill_len, rid="multi-decode")
        self._admit_and_complete(req)

        slot = req.req_pool_idx
        initial_partial = prefill_len % QUEST_PAGE_SIZE  # 8
        self.assertEqual(int(self.quest.running_token_count[slot]), initial_partial)
        self.assertEqual(
            int(self.quest.running_page_idx[slot]),
            prefill_len // QUEST_PAGE_SIZE,
        )

        # The very first call to _eager_backup_previous_token after staging
        # SKIPS the backup (the prefill staging already covered everything).
        # Real production decode count after which page 3 finalises:
        #   skipped first + (page_size - initial_partial) actual updates.
        steps = (QUEST_PAGE_SIZE - initial_partial) + 1  # 64 - 8 + 1 = 57

        req_pool_indices = torch.tensor([slot], dtype=torch.int64, device="cuda")
        req_pool_indices_cpu = req_pool_indices.cpu()

        for step in range(steps):
            # Production convention (per schedule_batch.prepare_for_decode):
            # seq_lens has already been incremented for the upcoming token,
            # so actual_token_pos = seq_lens - 2 = prefill_len + step - 1.
            upcoming_seq_len = prefill_len + step + 1
            seq_lens = torch.tensor(
                [upcoming_seq_len], dtype=torch.int32, device="cuda",
            )
            seq_lens_cpu = seq_lens.cpu()
            self.coord._eager_backup_previous_token(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu,
            )
            # _eager_backup_previous_token launches updates on
            # decode_backup_stream; sync so we can inspect Quest state.
            self.coord.decode_backup_stream.synchronize()

        # Page 3 completed: page_valid for that page, all layers, must be True.
        page_done = prefill_len // QUEST_PAGE_SIZE  # 3
        valid_after = self.quest.page_valid[:, slot, page_done]
        self.assertTrue(torch.all(valid_after),
                        f"page {page_done} not all-valid after {steps} steps: "
                        f"{valid_after}")
        # And counters wrapped + advanced for the next page.
        self.assertEqual(int(self.quest.running_token_count[slot]), 0)
        self.assertEqual(int(self.quest.running_page_idx[slot]), page_done + 1)
        # Bounds for the just-finalised page should be non-zero (random K
        # was written to those device slots).
        bounds_min = self.quest.page_k_min[:, slot, page_done]
        self.assertTrue(torch.any(bounds_min != 0))

        self.coord.request_finished(req)

    def test_request_finished_releases_quest_state(self):
        """Coord's request_finished hook clears Quest state for that slot."""
        prefill_len = 256  # 4 full pages
        req, _ = self._make_quest_req(prefill_len=prefill_len, rid="cleanup")
        self._admit_and_complete(req)
        slot = req.req_pool_idx
        # Sanity: state is populated after admit.
        self.assertTrue(torch.any(self.quest.page_valid[:, slot, :]))

        self.coord.request_finished(req)

        # Quest state for this slot should be wiped.
        self.assertFalse(torch.any(self.quest.page_valid[:, slot, :]))
        self.assertEqual(int(self.quest.running_token_count[slot]), 0)
        self.assertEqual(int(self.quest.running_page_idx[slot]), 0)

    def test_forward_decode_e2e(self):
        """End-to-end forward_decode: real Quest scoring → real swap_in →
        real FlashInfer attention.  Smoke-test for output shape and no NaN."""
        prefill_len = 384  # fits in DEVICE_BUFFER_SIZE (512), fast-path swap-in
        req, _ = self._make_quest_req(prefill_len=prefill_len, rid="fwd-e2e")
        self._admit_and_complete(req)

        bs = 1
        forward_batch = SimpleNamespace(
            batch_size=bs,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device="cuda",
            ),
            seq_lens=torch.tensor([prefill_len], dtype=torch.int32, device="cuda"),
            # save_kv_cache=False below ⇒ out_cache_loc is unused.
            out_cache_loc=torch.zeros(bs, dtype=torch.int64, device="cuda"),
            token_to_kv_pool=self.coord.mem_pool_device,
            hisparse_coordinator=self.coord,
        )

        # The swap-in kernel reads num_real_reqs to gate padded blocks; we set
        # it explicitly so the kernel processes our one request.  In production
        # this is updated by ModelRunner before each forward.
        self.coord.num_real_reqs[0] = bs

        self.backend.init_forward_metadata(forward_batch)

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=MHA_HEAD_NUM,
            tp_k_head_num=MHA_HEAD_NUM,
            tp_v_head_num=MHA_HEAD_NUM,
            head_dim=MHA_HEAD_DIM,
            v_head_dim=MHA_HEAD_DIM,
            scaling=1.0 / (MHA_HEAD_DIM ** 0.5),
            logit_cap=0.0,
            k_scale=None, v_scale=None,
            k_scale_float=None, v_scale_float=None,
            is_cross_attention=False,
        )
        q = torch.randn(bs, MHA_HEAD_NUM * MHA_HEAD_DIM,
                        dtype=torch.bfloat16, device="cuda")
        k = torch.randn(bs, MHA_HEAD_NUM * MHA_HEAD_DIM,
                        dtype=torch.bfloat16, device="cuda")
        v = torch.randn(bs, MHA_HEAD_NUM * MHA_HEAD_DIM,
                        dtype=torch.bfloat16, device="cuda")

        # save_kv_cache=False so we don't need the coordinator's hot-buffer
        # slot reservation (map_last_loc_to_buffer) for this smoke test.
        out = self.backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=False,
        )

        self.assertEqual(out.shape, (bs, MHA_HEAD_NUM * MHA_HEAD_DIM))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.any(torch.isnan(out.float())),
                         "forward_decode produced NaN")
        # Output should depend on K/V — should not be exactly the zero tensor.
        self.assertGreater(out.abs().sum().item(), 0.0)

        self.coord.request_finished(req)


if __name__ == "__main__":
    unittest.main()
