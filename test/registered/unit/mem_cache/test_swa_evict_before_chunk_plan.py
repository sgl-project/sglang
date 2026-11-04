"""Chunk sizes of a long chunked prefill at the capped SWA pool.

compute_swa_request_cap sizes the SWA pool for the chunks in flight plus one
window and a small slack. PrefillAdder sizes a continuing chunk from the SWA
free count, but maybe_evict_swa frees the previous chunk's out-of-window slots
only afterwards, in alloc_for_extend, so every chunk after the second
alternated between the slack and chunk - window tokens.
free_chunked_swa_before_plan frees them before the chunk is sized.
"""

import unittest
from array import array
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import (
    checkpoint_kv_cache,
    free_chunked_swa_before_plan,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.pool_configurator import compute_swa_request_cap
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 4096
WINDOW = 128
PROMPT_LEN = 8 * CHUNK + 876


class _ChunkedPrefill:
    """One request's chunked prefill, driven through the scheduler's steps on CPU:
    stash the previous chunk, optionally free before planning, size the chunk
    with PrefillAdder, then alloc_for_extend's maybe_evict_swa and allocation."""

    def __init__(self, *, enable_overlap: bool, evict_before_plan: bool):
        self.enable_overlap = enable_overlap
        self.evict_before_plan = evict_before_plan
        self.swa_size = compute_swa_request_cap(
            page_size=1, window=WINDOW, attn_dp_size=1
        )
        kvcache = SWAKVPool.__new__(SWAKVPool)
        kvcache.full_kv_pool = None
        kvcache.swa_kv_pool = None
        kvcache.register_mapping = lambda mapping: None
        self.allocator = SWATokenToKVPoolAllocator(
            size=PROMPT_LEN,
            size_swa=self.swa_size,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=PROMPT_LEN,
            device="cpu",
            enable_memory_saver=False,
        )
        # What --disable-radix-cache builds for a hybrid SWA model.
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                sliding_window_size=WINDOW,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
            )
        )
        self.req = Req(
            rid="long-prompt",
            origin_input_text=None,
            origin_input_ids=array("q", [1] * PROMPT_LEN),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        self.req.init_next_round_input()
        self.req.kv.req_pool_idx = 1
        self.batch = SimpleNamespace(
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_to_token_pool,
            reqs=[self.req],
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=enable_overlap,
        )
        self.batch._swa_eviction_trigger = MethodType(
            ScheduleBatch._swa_eviction_trigger, self.batch
        )
        self.batch._evict_swa = MethodType(ScheduleBatch._evict_swa, self.batch)
        # The start of the chunk planned last; under overlap it may still be running.
        self.last_start = None
        # SWA eviction cursor before and after each alloc-time maybe_evict_swa.
        self.alloc_evictions = []

    def evicted(self) -> int:
        return self.req.kv.get_evicted_seqlen(ComponentType.SWA)

    def run(self, after_early_free=None, chunk_limits=None):
        # chunk_limits: {chunk index: limit}, as shortest_prefill_chunk_limit sets.
        req = self.req
        chunks = []
        while True:
            if chunks:
                checkpoint_kv_cache(req, self.cache)
                if self.evict_before_plan:
                    free_chunked_swa_before_plan(
                        req,
                        enable_overlap=self.enable_overlap,
                        tree_cache=self.cache,
                        token_to_kv_pool_allocator=self.allocator,
                    )
                    if after_early_free is not None:
                        after_early_free(self)

            adder = PrefillAdder(
                page_size=1,
                tree_cache=self.cache,
                token_to_kv_pool_allocator=self.allocator,
                running_batch=None,
                new_token_ratio=1.0,
                rem_input_tokens=2 * PROMPT_LEN,
                rem_chunk_tokens=CHUNK,
            )
            adder.chunked_req_limit = (chunk_limits or {}).get(len(chunks))
            req.init_next_round_input()
            unfinished = adder.add_chunked_req(req)
            assert adder.can_run_list == [req], f"no chunk planned after {chunks}"

            start, end = req.prefix_len, req.extend_end
            self.batch.prefix_lens = [start]
            before = self.evicted()
            ScheduleBatch.maybe_evict_swa(self.batch)
            self.alloc_evictions.append((before, self.evicted()))
            loc = self.allocator.alloc(end - start)
            assert loc is not None
            self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end] = loc
            req.extend_batch_idx += 1
            self.last_start = start
            chunks.append(end - start)
            if unfinished is None:
                return chunks


class TestSWAEvictBeforeChunkPlan(CustomTestCase):
    @staticmethod
    def _continuing_req(*, extend_batch_idx: int = 2):
        # Past the request-level guards, so only the gate under test can stop the free.
        return SimpleNamespace(
            prefix_len=2 * CHUNK,
            extend_batch_idx=extend_batch_idx,
            kv=SimpleNamespace(holds_kv=True),
        )

    def _assert_last_chunk_window_mapped(self, p, *, enable_overlap: bool) -> int:
        # Under overlap the last chunk may still be running.
        req = p.req
        live_from = max(
            (p.last_start if enable_overlap else req.prefix_len) - WINDOW, 0
        )
        rows = p.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        live = rows[live_from : req.prefix_len]
        self.assertTrue(bool((p.allocator.full_to_swa_index_mapping[live] > 0).all()))
        self.assertEqual(p.allocator.full_available_size(), PROMPT_LEN - req.prefix_len)
        return live_from

    def _server_args(self, *, enable_overlap: bool):
        override = get_context().override_server_args(
            chunked_prefill_size=CHUNK,
            max_running_requests=1,
            disable_overlap_schedule=not enable_overlap,
        )
        override.install()
        self.addCleanup(override.restore)

    def _prefill(self, *, enable_overlap: bool, evict_before_plan: bool):
        self._server_args(enable_overlap=enable_overlap)
        return _ChunkedPrefill(
            enable_overlap=enable_overlap, evict_before_plan=evict_before_plan
        )

    def test_old_order_shrinks_chunks_at_the_cap(self):
        # Control: without the early free the harness reproduces the short chunks.
        prefill = self._prefill(enable_overlap=True, evict_before_plan=False)
        self.assertEqual(prefill.swa_size, 2 * CHUNK + 259)
        self.assertEqual(
            prefill.run(), [CHUNK, CHUNK] + [258, CHUNK - WINDOW] * 6 + [96]
        )

    def test_chunks_stay_full_at_the_cap(self):
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap, evict_before_plan=True
                )
                self.assertEqual(prefill.run(), [CHUNK] * 8 + [876])

    def test_early_free_keeps_what_the_last_chunk_reads(self):
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap, evict_before_plan=True
                )
                bounds = []

                def check(p):
                    live_from = self._assert_last_chunk_window_mapped(
                        p, enable_overlap=enable_overlap
                    )
                    bounds.append((p.evicted(), live_from))

                prefill.run(after_early_free=check)
                self.assertEqual(len(bounds), 8)
                for evicted, live_from in bounds:
                    self.assertEqual(evicted, live_from)

    def test_chunks_stay_full_after_a_short_chunk(self):
        # After a chunk cut short, maybe_evict_swa's overlap bound (one full chunk
        # back) trails the chunk in flight; what it frees is still enough.
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap, evict_before_plan=True
                )
                chunks = prefill.run(
                    after_early_free=lambda p: self._assert_last_chunk_window_mapped(
                        p, enable_overlap=enable_overlap
                    ),
                    chunk_limits={2: 1000},
                )
                self.assertEqual(
                    chunks,
                    [CHUNK, CHUNK, 1000]
                    + [CHUNK] * 5
                    + [PROMPT_LEN - 7 * CHUNK - 1000],
                )

    def test_alloc_for_extend_finds_nothing_left_to_free(self):
        # The early free takes maybe_evict_swa's bound, so the alloc-time pass
        # that used to free these slots now frees nothing.
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap, evict_before_plan=True
                )
                prefill.run()
                self.assertGreater(prefill.evicted(), 0)
                for before, after in prefill.alloc_evictions:
                    self.assertEqual(after, before)

    def test_radix_cache_is_left_alone(self):
        req = self._continuing_req()
        cache = SimpleNamespace(
            supports_swa=lambda: True, supports_prefix_sharing=lambda: True
        )
        allocator = SimpleNamespace(
            free_group_begin=lambda: self.fail("freed from a radix cache")
        )
        free_chunked_swa_before_plan(
            req,
            enable_overlap=False,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
        )

    def test_swa_req_ring_is_left_alone(self):
        # A per-request ring is not sized from the SWA free count.
        req = self._continuing_req()
        cache = SimpleNamespace(
            supports_swa=lambda: True, supports_prefix_sharing=lambda: False
        )
        allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
        allocator._swa_req_ring = True
        allocator.free_group_begin = lambda: self.fail("freed from an SWA ring")
        free_chunked_swa_before_plan(
            req,
            enable_overlap=False,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
        )

    def test_request_without_kv_is_left_alone(self):
        req = self._continuing_req()
        req.kv.holds_kv = False
        cache = SimpleNamespace(
            supports_swa=lambda: True, supports_prefix_sharing=lambda: False
        )
        allocator = SimpleNamespace(
            free_group_begin=lambda: self.fail("freed for a request without KV")
        )
        free_chunked_swa_before_plan(
            req,
            enable_overlap=False,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
        )

    def test_first_chunk_in_flight_is_left_alone(self):
        # Under overlap the first chunk may still be running, as in maybe_evict_swa.
        self._server_args(enable_overlap=True)
        req = self._continuing_req(extend_batch_idx=1)
        cache = SimpleNamespace(
            supports_swa=lambda: True, supports_prefix_sharing=lambda: False
        )
        allocator = SimpleNamespace(
            free_group_begin=lambda: self.fail("freed under the first chunk")
        )
        free_chunked_swa_before_plan(
            req,
            enable_overlap=True,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
        )


if __name__ == "__main__":
    unittest.main()
