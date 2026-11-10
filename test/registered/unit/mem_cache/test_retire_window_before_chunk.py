"""Chunk sizes of a long chunked prefill on a capped window pool.

compute_swa_request_cap sizes the SWA pool for the chunks in flight, one window
and a small slack. PrefillAdder sizes a continuing chunk from the SWA free rows,
but ScheduleBatch.maybe_evict_swa retires the rows behind the window only once
that chunk is placed (alloc_for_extend). At the cap every chunk after the second
then alternated between the slack and chunk - window tokens; below two chunks
plus a window no chunk was planned after the second, so nothing retired and the
request never ran again. retire_window_before_chunk retires them before the
chunk is sized.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocation import retire_window_before_chunk
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.pool_configurator import compute_swa_request_cap
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CHUNK = 4096
WINDOW = 128
PROMPT_LEN = 8 * CHUNK + 876


def _page_ceil(tokens: int, page_size: int) -> int:
    return -(-tokens // page_size) * page_size


class _Stalled(Exception):
    """The adder planned no chunk: nothing is placed, so nothing retires."""

    def __init__(self, chunks):
        super().__init__(f"no chunk planned after {chunks}")
        self.chunks = chunks


class _ChunkedPrefill:
    """One request's chunked prefill, driven through the scheduler's steps on CPU:
    checkpoint the previous chunk, optionally retire before planning, size the
    chunk with PrefillAdder, then alloc_for_extend's maybe_evict_swa and the
    allocation. What --disable-radix-cache builds for a hybrid SWA model."""

    def __init__(
        self,
        *,
        enable_overlap: bool,
        retire_before_chunk: bool,
        page_size: int = 1,
        swa_size: int | None = None,
    ):
        self.enable_overlap = enable_overlap
        self.retire_before_chunk = retire_before_chunk
        self.page_size = page_size
        self.swa_size = swa_size or compute_swa_request_cap(
            page_size=page_size, window=WINDOW, attn_dp_size=1
        )
        full_size = _page_ceil(PROMPT_LEN, page_size)
        kvcache = SWAKVPool(
            size=full_size,
            size_swa=self.swa_size,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=full_size,
            size_swa=self.swa_size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        if page_size > 1:
            # The pools are on CPU even when a GPU is visible: no Triton.
            for paged in (
                self.allocator.full_attn_allocator,
                self.allocator.swa_attn_allocator,
            ):
                paged.use_triton_kernels = False
        self.full_free_at_start = self.allocator.full_available_size()
        self.req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=full_size,
            device="cpu",
            enable_memory_saver=False,
        )
        kv_storage_of(self.req_to_token_pool, self.allocator).bind_window_retention(
            WINDOW, None
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=page_size,
                sliding_window_size=WINDOW,
                tree_slots=tree_slots_of(
                    (ComponentType.FULL, ComponentType.SWA), window=WINDOW
                ),
            )
        )
        self.req = Req(
            rid="long-prompt",
            origin_input_text="",
            origin_input_ids=array("q", [1] * PROMPT_LEN),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        self.req_to_token_pool.alloc([self.req])
        self.req.init_next_round_input()
        self.batch = ScheduleBatch(
            reqs=[self.req],
            tree_cache=self.cache,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.allocator,
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=enable_overlap,
        )
        # The start of the chunk planned last; under overlap it may still be running.
        self.last_start = None
        # The window cursor before and after each alloc-time maybe_evict_swa.
        self.alloc_retirements = []

    def evicted(self) -> int:
        return self.req.kv.get_evicted_seqlen(BaseSWAKVPool.WINDOW_SUB_POOL)

    def row(self) -> torch.Tensor:
        return self.req_to_token_pool.req_to_token[self.req.kv.req_pool_idx]

    def _place(self, start: int, end: int) -> None:
        if self.page_size == 1:
            loc = self.allocator.alloc(end - start)
        else:
            row = self.row()
            last_loc = (
                row[start - 1 : start].to(torch.int64)
                if start
                else torch.tensor([-1], dtype=torch.int64)
            )
            prefix_lens = torch.tensor([start], dtype=torch.int64)
            seq_lens = torch.tensor([end], dtype=torch.int64)
            loc = self.allocator.alloc_extend(
                prefix_lens, prefix_lens, seq_lens, seq_lens, last_loc, end - start
            )
        assert loc is not None, f"chunk [{start}, {end}) was planned but not placed"
        self.row()[start:end] = loc

    def run(self, after_retire=None, chunk_limits=None):
        # chunk_limits: {chunk index: limit}, as shortest_prefill_chunk_limit sets.
        req = self.req
        chunks = []
        while True:
            if chunks:
                checkpoint_kv_cache(req, self.cache)
                if self.retire_before_chunk:
                    retire_window_before_chunk(
                        req, self.cache, enable_overlap=self.enable_overlap
                    )
                    if after_retire is not None:
                        after_retire(self)

            adder = PrefillAdder(
                page_size=self.page_size,
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
            if adder.can_run_list != [req]:
                raise _Stalled(chunks)

            start, end = req.prefix_len, req.extend_end
            self.batch.prefix_lens = [start]
            before = self.evicted()
            self.batch.maybe_evict_swa()
            self.alloc_retirements.append((before, self.evicted()))
            self._place(start, end)
            req.extend_batch_idx += 1
            self.last_start = start
            chunks.append(end - start)
            if unfinished is None:
                return chunks


class TestRetireWindowBeforeChunk(CustomTestCase):
    @staticmethod
    def _continuing_req(*, extend_batch_idx: int = 2):
        # Past the request-level guards, so only the check under test can stop it.
        return SimpleNamespace(
            prefix_len=2 * CHUNK,
            extend_batch_idx=extend_batch_idx,
            kv=SimpleNamespace(holds_kv=True),
        )

    @staticmethod
    def _cache(*, allocator, swa=True, shares_prefixes=False):
        return SimpleNamespace(
            supports_swa=lambda: swa,
            supports_prefix_sharing=lambda: shares_prefixes,
            token_to_kv_pool_allocator=allocator,
        )

    def _assert_last_chunk_window_mapped(self, p, *, enable_overlap: bool) -> int:
        # Under overlap the last chunk may still be running.
        req = p.req
        live_from = max(
            (p.last_start if enable_overlap else req.prefix_len) - WINDOW, 0
        )
        live = p.row()[live_from : req.prefix_len]
        self.assertTrue(bool((p.allocator.full_to_swa_index_mapping[live] > 0).all()))
        self.assertEqual(
            p.allocator.full_available_size(),
            p.full_free_at_start - _page_ceil(req.prefix_len, p.page_size),
        )
        return live_from

    def _server_args(self, *, enable_overlap: bool):
        override = get_context().override_server_args(
            chunked_prefill_size=CHUNK,
            max_running_requests=1,
            disable_overlap_schedule=not enable_overlap,
        )
        override.install()
        self.addCleanup(override.restore)

    def _prefill(self, *, enable_overlap: bool, retire_before_chunk: bool, **kw):
        self._server_args(enable_overlap=enable_overlap)
        return _ChunkedPrefill(
            enable_overlap=enable_overlap, retire_before_chunk=retire_before_chunk, **kw
        )

    def test_old_order_shrinks_chunks_at_the_cap(self):
        # Control: without the early retirement the harness reproduces the short
        # chunks (page 1: what MiMo-V2.6-Flash showed; page 64: gpt-oss-20b).
        for page_size, swa_size, tail in (
            (1, 2 * CHUNK + 259, [258, CHUNK - WINDOW] * 6 + [96]),
            (64, 2 * CHUNK + 448, [384, CHUNK - WINDOW] * 5 + [384, 3308]),
        ):
            with self.subTest(page_size=page_size):
                prefill = self._prefill(
                    enable_overlap=True, retire_before_chunk=False, page_size=page_size
                )
                self.assertEqual(prefill.swa_size, swa_size)
                self.assertEqual(prefill.run(), [CHUNK, CHUNK] + tail)

    def test_chunks_stay_full_at_the_cap(self):
        for page_size in (1, 64):
            for enable_overlap in (True, False):
                with self.subTest(page_size=page_size, enable_overlap=enable_overlap):
                    prefill = self._prefill(
                        enable_overlap=enable_overlap,
                        retire_before_chunk=True,
                        page_size=page_size,
                    )
                    self.assertEqual(prefill.run(), [CHUNK] * 8 + [876])

    def test_early_retirement_keeps_what_the_last_chunk_reads(self):
        for page_size in (1, 64):
            for enable_overlap in (True, False):
                with self.subTest(page_size=page_size, enable_overlap=enable_overlap):
                    prefill = self._prefill(
                        enable_overlap=enable_overlap,
                        retire_before_chunk=True,
                        page_size=page_size,
                    )
                    cursors = []

                    def check(p):
                        live_from = self._assert_last_chunk_window_mapped(
                            p, enable_overlap=enable_overlap
                        )
                        cursors.append((p.evicted(), live_from))

                    prefill.run(after_retire=check)
                    self.assertEqual(len(cursors), 8)
                    for evicted, live_from in cursors:
                        self.assertEqual(evicted, live_from)

    def test_chunks_stay_full_after_a_short_chunk(self):
        # After a chunk cut short, maybe_evict_swa's overlap frontier (one full
        # chunk back) trails the chunk in flight; what it retires is still enough.
        for enable_overlap in (True, False):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap, retire_before_chunk=True
                )
                chunks = prefill.run(
                    after_retire=lambda p: self._assert_last_chunk_window_mapped(
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

    def test_alloc_for_extend_finds_nothing_left_to_retire(self):
        # Both take prefill_window_frontier, so the alloc-time pass that used to
        # retire these rows now retires nothing for this request.
        for page_size in (1, 64):
            for enable_overlap in (True, False):
                with self.subTest(page_size=page_size, enable_overlap=enable_overlap):
                    prefill = self._prefill(
                        enable_overlap=enable_overlap,
                        retire_before_chunk=True,
                        page_size=page_size,
                    )
                    prefill.run()
                    self.assertGreater(prefill.evicted(), 0)
                    for before, after in prefill.alloc_retirements:
                        self.assertEqual(after, before)

    def test_pool_below_two_chunks_and_a_window(self):
        # Overlap keeps the first two chunks until the third is placed, and the
        # second took what the first left, so no third chunk fits and the
        # alloc-time retirement never runs: the request never runs again.
        swa_size = 6200
        prefill = self._prefill(
            enable_overlap=True, retire_before_chunk=False, swa_size=swa_size
        )
        with self.assertRaises(_Stalled) as stalled:
            prefill.run()
        self.assertEqual(stalled.exception.chunks, [CHUNK, swa_size - CHUNK - 1])

        # Retired first, each chunk gets what one chunk back leaves (overlap), or
        # a full chunk.
        short = swa_size - CHUNK - WINDOW - 1
        rest = PROMPT_LEN - CHUNK - (swa_size - CHUNK - 1)
        for enable_overlap, chunks in (
            (
                True,
                [CHUNK, swa_size - CHUNK - 1]
                + [short] * (rest // short)
                + [rest % short],
            ),
            (False, [CHUNK] * 8 + [876]),
        ):
            with self.subTest(enable_overlap=enable_overlap):
                prefill = self._prefill(
                    enable_overlap=enable_overlap,
                    retire_before_chunk=True,
                    swa_size=swa_size,
                )
                self.assertEqual(prefill.run(), chunks)

    def test_cache_without_swa_is_left_alone(self):
        cache = self._cache(
            allocator=SimpleNamespace(
                free_group_begin=lambda: self.fail("retired without a window")
            ),
            swa=False,
        )
        retire_window_before_chunk(self._continuing_req(), cache, enable_overlap=False)

    def test_prefix_sharing_cache_is_left_alone(self):
        cache = self._cache(
            allocator=SimpleNamespace(
                free_group_begin=lambda: self.fail("retired from a radix cache")
            ),
            shares_prefixes=True,
        )
        retire_window_before_chunk(self._continuing_req(), cache, enable_overlap=False)

    def test_swa_req_ring_is_left_alone(self):
        # A per-request ring is not sized from the SWA free rows.
        allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
        allocator._swa_req_ring = True
        allocator.free_group_begin = lambda: self.fail("retired from an SWA ring")
        retire_window_before_chunk(
            self._continuing_req(),
            self._cache(allocator=allocator),
            enable_overlap=False,
        )

    def test_request_without_kv_is_left_alone(self):
        req = self._continuing_req()
        req.kv.holds_kv = False
        cache = self._cache(
            allocator=SimpleNamespace(
                free_group_begin=lambda: self.fail("retired for a request without KV")
            )
        )
        retire_window_before_chunk(req, cache, enable_overlap=False)

    def test_first_chunk_in_flight_is_left_alone(self):
        # Under overlap the first chunk may still be running, as in maybe_evict_swa.
        self._server_args(enable_overlap=True)
        cache = self._cache(
            allocator=SimpleNamespace(
                free_group_begin=lambda: self.fail("retired under the first chunk")
            )
        )
        retire_window_before_chunk(
            self._continuing_req(extend_batch_idx=1), cache, enable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
