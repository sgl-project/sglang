"""CPU tests for one-slot exact token-id continuation on ChunkCache."""

from __future__ import annotations

from array import array
from unittest import mock

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, ReqKvInfo
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    kv_private_tokens,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.sticky_last_sequence import StickyLastSequenceCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _FakeAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, page_size: int = 1):
        super().__init__(
            size=1024,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def clear(self):
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token
        self.free_slots = []

    def free(self, req):
        self.free_slots.append(req.kv.req_pool_idx)
        req.kv.req_pool_idx = None


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size=1):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.disable = True
        self.finished = []
        self.matches = 0
        self.reset_calls = 0

    def supports_prefix_sharing(self):
        return False

    def match_prefix(self, params):
        self.matches += 1
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def checkpoint(self, req, *, up_to):
        if req.finished():
            self.finished.append(req)

    def reset(self):
        self.reset_calls += 1

    def evict(self, params):
        return None

    def evict_for_alloc(self, params):
        return None

    def inc_lock_ref(self, node):
        raise AssertionError("inner lock should not run for sticky sentinel")

    def dec_lock_ref(self, node, params=None):
        raise AssertionError("inner unlock should not run for sticky sentinel")

    def protected_size(self):
        return 0

    def evictable_size(self):
        return 0

    def pretty_print(self):
        return ""

    def supports_swa(self):
        return False

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeReq:
    def __init__(
        self, req_pool_idx, committed, allocated, origin, output=None, rid=None
    ):
        self.rid = rid
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
            cache_protected_len=0,
        )
        self.origin_input_ids = list(origin)
        self.output_ids = list(output or [])
        self.extra_key = None
        self.cache_salt = None
        self.finished_reason = None
        self.finished_len = None
        self.to_finish = None
        self.skip_radix_cache_insert = False
        self.extend_range = None

    def finished(self):
        return self.finished_reason is not None

    @property
    def output_ids_through_stop(self):
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def detach_kv(self):
        kv, self.kv = self.kv, ReqKvInfo()
        return kv


class _Range:
    def __init__(self, end):
        self.end = end


def _release(cache, req):
    """The tree calls release_kv_cache makes for a finished request."""
    if not cache.claim_kv_row(req):
        cache.checkpoint(req, up_to=req.kv.kv_allocated_len)


def _key(ids, extra_key=None, cache_salt=None, limit=None):
    return RadixKey(
        token_ids=array("q", ids),
        extra_key=extra_key,
        cache_salt=cache_salt,
        limit=limit,
    )


class TestStickyLastSequence(CustomTestCase):
    def _make(self, page_size=1, row_len=128):
        req_to_token = torch.arange(row_len, dtype=torch.int32).reshape(1, row_len)
        pool = _FakeReqToTokenPool(req_to_token)
        allocator = _FakeAllocator(page_size=page_size)
        inner = _FakeInnerCache(pool, allocator, page_size=page_size)
        return StickyLastSequenceCache(inner), inner, allocator, pool

    def _pin(self, cache, origin, output, allocated=None):
        ids = list(origin) + list(output)
        n = len(ids)
        req = _FakeReq(
            req_pool_idx=0,
            committed=n,
            allocated=allocated if allocated is not None else n,
            origin=origin,
            output=output,
        )
        _release(cache, req)
        return req, ids

    def test_exact_continuation_hits(self):
        cache, inner, allocator, pool = self._make()
        _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
        (kv,) = cache.session_records().values()
        self.assertEqual(kv_private_tokens(kv, cache.page_size), 10)
        self.assertFalse(allocator.freed)
        self.assertEqual(pool.free_slots, [])

        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(last + [50, 51]), req=nxt)
        )
        self.assertEqual(inner.matches, 0)
        self.assertEqual(len(result.device_indices), 10)
        self.assertEqual(result.device_indices.tolist(), list(range(10)))
        self.assertEqual(nxt.kv.req_pool_idx, 0)
        self.assertEqual(result.cache_protected_len, 0)

    def test_pin_stops_at_the_csa2_image(self):
        # Without the overlap loop the image ends before the last token.
        cache, inner, allocator, pool = self._make()
        target = "sglang.srt.mem_cache.sticky_last_sequence.csa2_image_len"
        with mock.patch(target, return_value=9):
            _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
        self.assertEqual(cache._last_ids, tuple(last[:9]))
        self.assertEqual(cache._slot.kv.kv_committed_len, 9)

        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(last + [50, 51]), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 9)

    def test_pin_ignores_an_image_outside_the_output(self):
        # An image past the stop (overlap loop) or before the prompt end
        # (another request's) keeps the full pin.
        for image in (12, 5, None):
            cache, *_ = self._make()
            target = "sglang.srt.mem_cache.sticky_last_sequence.csa2_image_len"
            with mock.patch(target, return_value=image):
                _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
            self.assertEqual(cache._last_ids, tuple(last))

    def test_shorter_prompt_misses_and_drops(self):
        cache, inner, allocator, pool = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(8))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(pool.free_slots, [0])
        self.assertTrue(allocator.freed)

    def test_partial_prefix_misses(self):
        cache, inner, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        # Same first 6 tokens, then diverges — not an exact continuation.
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(6)) + [99, 100]), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(cache._cuts, [])

    def test_recorded_stop_hits_a_shorter_shared_prefix(self):
        cache, inner, _, _ = self._make(row_len=16)
        partial = _FakeReq(
            req_pool_idx=0,
            committed=8,
            allocated=8,
            origin=list(range(8)),
        )
        partial.extend_range = _Range(8)
        checkpoint_kv_cache(partial, cache)
        self.assertEqual(cache._cuts, [8])
        _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
        self.assertEqual(last, list(range(10)))
        self.assertEqual(cache._cuts, [8, 10])

        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(8)) + [99, 100]), req=nxt)
        )
        self.assertEqual(inner.matches, 0)
        self.assertEqual(len(result.device_indices), 8)
        self.assertEqual(result.device_indices.tolist(), list(range(8)))
        self.assertEqual(nxt.kv.req_pool_idx, 0)
        self.assertEqual(cache._last_ids, tuple(last[:8]))
        self.assertEqual(cache._cuts, [8])

    def test_each_tool_turn_resumes_at_the_previous_prompt_end(self):
        """A turn's prompt-end stop must survive its own pin.

        The next request re-renders the assistant reply, so it diverges inside
        the pinned output and can only resume at that prompt end. Losing it
        sent every turn back to the first prompt end.
        """
        cache, _, _, _ = self._make(row_len=64)
        first = list(range(8))
        prefill = _FakeReq(req_pool_idx=0, committed=8, allocated=8, origin=first)
        prefill.extend_range = _Range(8)
        checkpoint_kv_cache(prefill, cache)
        self._pin(cache, origin=first, output=[8, 9])

        second = first + [40, 41, 42, 43]
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(MatchPrefixParams(key=_key(second), req=nxt))
        self.assertEqual(len(result.device_indices), 8)
        nxt.origin_input_ids = second
        nxt.kv.kv_committed_len = nxt.kv.kv_allocated_len = len(second)
        nxt.extend_range = _Range(len(second))
        checkpoint_kv_cache(nxt, cache)
        nxt.output_ids = [44, 45]
        nxt.kv.kv_committed_len = nxt.kv.kv_allocated_len = len(second) + 2
        _release(cache, nxt)
        self.assertIn(len(second), cache._cuts)

        third = second + [50, 51]
        last = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(MatchPrefixParams(key=_key(third), req=last))
        self.assertEqual(len(result.device_indices), len(second))

    def test_unfinished_stop_is_extend_end_not_the_sampled_token(self):
        cache, _, _, _ = self._make()
        # Prefill result processing appends the sampled token first. The CSA2
        # image still ends at the prompt.
        req = _FakeReq(
            req_pool_idx=0,
            committed=8,
            allocated=8,
            origin=list(range(8)),
            output=[8],
        )
        req.extend_range = _Range(8)
        checkpoint_kv_cache(req, cache)
        self.assertEqual(cache._cuts, [8])

    def test_chunk_stop_is_extend_end(self):
        cache, _, _, _ = self._make()
        req = _FakeReq(
            req_pool_idx=0,
            committed=4,
            allocated=4,
            origin=list(range(10)),
        )
        req.extend_range = _Range(4)
        checkpoint_kv_cache(req, cache)
        self.assertEqual(cache._cuts, [4])

    def test_cuts_keep_the_newest_stops(self):
        cache, _, _, _ = self._make()
        for length in range(1, 40):
            cache._last_ids = None
            cache._note_cut(list(range(length)))
        self.assertEqual(len(cache._cuts), 32)
        self.assertEqual(cache._cuts[0], 8)
        self.assertEqual(cache._cuts[-1], 39)

    def test_different_tokens_miss(self):
        cache, inner, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key([7] + list(range(1, 12))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)

    def test_miss_forgets_old_last_ids(self):
        cache, _, _, _ = self._make()
        self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        cache.match_prefix(MatchPrefixParams(key=_key([1, 2, 3]), req=nxt))
        # A later request that *would* have continued the dropped sequence
        # must not resurrect the pin (pos-0 prefill already overwrote CSA2).
        later = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(8)) + [8, 9, 50]), req=later)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertIsNone(later.kv.req_pool_idx)

    def test_extra_key_mismatch_misses(self):
        cache, inner, _, _ = self._make()
        _, last = self._pin(cache, origin=list(range(4)), output=[4])
        cache._last_extra_key = "lora-a"
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(last + [9], extra_key="lora-b"), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)

    def test_abort_does_not_save(self):
        cache, inner, _, _ = self._make()
        req = _FakeReq(
            req_pool_idx=0,
            committed=4,
            allocated=4,
            origin=list(range(4)),
        )
        req.finished_reason = FINISH_ABORT("too long")
        _release(cache, req)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(inner.finished, [req])

    def test_abort_pins_last_completed_chunk(self):
        cache, inner, allocator, pool = self._make(row_len=16)
        req = _FakeReq(
            req_pool_idx=0,
            committed=8,
            allocated=12,
            origin=list(range(12)),
        )
        req.extend_range = _Range(4)
        checkpoint_kv_cache(req, cache)
        req.extend_range = _Range(8)
        checkpoint_kv_cache(req, cache)
        req.finished_reason = FINISH_ABORT("client")
        _release(cache, req)

        self.assertEqual(cache._last_ids, tuple(range(8)))
        self.assertEqual(inner.finished, [])
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertEqual(cache._slot.kv.kv_allocated_len, 8)
        self.assertEqual(pool.free_slots, [])
        self.assertEqual(allocator.freed[0].tolist(), list(range(8, 12)))

        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(12))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 8)
        self.assertEqual(result.device_indices.tolist(), list(range(8)))
        self.assertEqual(inner.matches, 0)

        other = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        missed = cache.match_prefix(
            MatchPrefixParams(key=_key([99, 100, 101]), req=other)
        )
        self.assertEqual(len(missed.device_indices), 0)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(inner.matches, 1)

    def test_abort_pins_unstashed_extend_end(self):
        # The chunk that just finished is snapshotted in the worker before
        # stash, so the abort has to pin extend_range.end on its own.
        cache, inner, _, _ = self._make(row_len=16)
        req = _FakeReq(
            req_pool_idx=0,
            committed=8,
            allocated=8,
            origin=list(range(12)),
        )
        req.extend_range = _Range(8)
        req.finished_reason = FINISH_ABORT("client")
        _release(cache, req)
        self.assertEqual(cache._last_ids, tuple(range(8)))
        self.assertEqual(inner.finished, [])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        result = cache.match_prefix(
            MatchPrefixParams(key=_key(list(range(12))), req=nxt)
        )
        self.assertEqual(len(result.device_indices), 8)

    def test_reset_frees_pin(self):
        cache, inner, allocator, pool = self._make()
        self._pin(cache, origin=list(range(4)), output=[4, 5])
        cache.reset()
        self.assertIsNone(cache._last_ids)
        self.assertEqual(pool.free_slots, [0])
        self.assertTrue(allocator.freed)
        self.assertEqual(inner.reset_calls, 1)
        self.assertEqual(cache.session_records(), {})

    def test_second_finish_updates_last_ids(self):
        cache, inner, _, _ = self._make()
        _, last = self._pin(cache, origin=list(range(6)), output=[6])
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        cache.match_prefix(MatchPrefixParams(key=_key(last + [20, 21]), req=nxt))
        nxt.origin_input_ids = last + [20]
        nxt.output_ids = [21]
        nxt.kv.kv_committed_len = 9
        nxt.kv.kv_allocated_len = 9
        _release(cache, nxt)
        self.assertEqual(cache._last_ids, tuple(last + [20, 21]))
        self.assertEqual(inner.finished, [])
        self.assertFalse(nxt.kv.holds_kv)

    def test_request_on_the_pin_runs_on_the_session_record(self):
        # Pool accounting counts the pin's row once: as session-held, and not
        # again for a request whose kv record is that session record.
        cache, _, _, _ = self._make()
        _, last = self._pin(cache, origin=list(range(5)), output=[5])
        (kv,) = cache.session_records().values()
        self.assertEqual(kv_private_tokens(kv, cache.page_size), 6)
        nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
        cache.match_prefix(MatchPrefixParams(key=_key(last + [6]), req=nxt))
        self.assertIs(nxt.kv, kv)

    def test_health_check_miss_does_not_drop_pin(self):
        cache, inner, allocator, pool = self._make()
        _, last = self._pin(cache, origin=list(range(8)), output=[8, 9])
        nxt = _FakeReq(
            req_pool_idx=None,
            committed=0,
            allocated=0,
            origin=[],
            rid="HEALTH_CHECK_abc",
        )
        result = cache.match_prefix(MatchPrefixParams(key=_key([0]), req=nxt))
        self.assertEqual(len(result.device_indices), 0)
        self.assertEqual(inner.matches, 1)
        self.assertEqual(cache._last_ids, tuple(last))
        self.assertEqual(pool.free_slots, [])
        self.assertFalse(allocator.freed)

    def test_health_check_finish_does_not_pin(self):
        cache, inner, _, _ = self._make()
        req = _FakeReq(
            req_pool_idx=0,
            committed=1,
            allocated=1,
            origin=[0],
            rid="HEALTH_CHECK_abc",
        )
        req.finished_reason = FINISH_LENGTH(1)
        _release(cache, req)
        self.assertIsNone(cache._last_ids)
        self.assertEqual(inner.finished, [req])

    def test_saved_session_resumes_after_the_pin_is_gone(self):
        import tempfile

        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            remember,
            reset_handoff,
            save_resident,
            session_key,
            take_handoff,
        )

        class _Rows:
            def __init__(self):
                self.swa_ring = {0: torch.zeros(2, dtype=torch.uint8)}
                self.pending_kv = {}
                self.pending_score = {}
                self.kv_rows = {2: torch.ones(3, dtype=torch.uint8)}
                self.index_rows = {}

        class _Pool:
            def __init__(self):
                self.req_to_token = torch.arange(64, dtype=torch.int32).reshape(2, 32)
                self._free = [1]

            def alloc_rows(self, need):
                rows = self._free[-need:]
                del self._free[-need:]
                return rows

            def free_rows(self, indices):
                self._free.extend(indices)

            def free(self, req):
                self.free_rows([req.kv.req_pool_idx])
                req.kv.req_pool_idx = None

        class _Alloc(_FakeAllocator):
            def alloc(self, need_size):
                return torch.arange(need_size, dtype=torch.int32)

        directory = tempfile.mkdtemp()
        envs.SGLANG_DSV41_CSA2_SESSION_DIR.set(directory)
        envs.SGLANG_DSV41_CSA2_SESSION_KEEP.set(2)
        try:
            ids = list(range(6))
            key = session_key(ids, None, None)
            remember(key, ids, [4, 6], None, None)
            backend = type("B", (), {"_sm70_csa2": _Rows()})()
            from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
                csa2_finish_forward,
                csa2_prepare_decode,
            )

            csa2_finish_forward([backend], 6, from_extend=True)
            csa2_prepare_decode([backend])
            save_resident([("target", backend._sm70_csa2)], backend._csa2_boundary, key)

            pool = _Pool()
            allocator = _Alloc()
            inner = _FakeInnerCache(pool, allocator)
            cache = StickyLastSequenceCache(inner)
            reset_handoff()
            nxt = _FakeReq(req_pool_idx=None, committed=0, allocated=0, origin=[])
            result = cache.match_prefix(
                MatchPrefixParams(key=_key(ids + [20, 21]), req=nxt)
            )
            self.assertEqual(inner.matches, 0)
            self.assertEqual(len(result.device_indices), 6)
            self.assertEqual(nxt.kv.req_pool_idx, 1)
            spill, load = take_handoff()
            self.assertIsNone(spill)
            self.assertEqual(load, key)
        finally:
            envs.SGLANG_DSV41_CSA2_SESSION_DIR.clear()
            envs.SGLANG_DSV41_CSA2_SESSION_KEEP.clear()
            reset_handoff()
