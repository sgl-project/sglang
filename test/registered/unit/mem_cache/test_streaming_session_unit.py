from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheFinishedReqResult,
    MatchResult,
)
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self, page_size: int = 1, uses_legacy_real_length_alloc: bool = False):
        self.freed = []
        self.freed_pages = []
        self.page_size = page_size
        self.uses_legacy_real_length_alloc = uses_legacy_real_length_alloc

    def free(self, free_index: torch.Tensor) -> None:
        if not self.uses_legacy_real_length_alloc:
            assert free_index.numel() % self.page_size == 0, (
                f"free expects a concatenation of whole pages: "
                f"{free_index.numel()=}, {self.page_size=}"
            )
        self.freed.append(free_index.clone())
        self.freed_pages.append(torch.unique(free_index // self.page_size))


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeDelegatingInnerCache(_FakeInnerCache):
    def __init__(self, *args, result: CacheFinishedReqResult, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.result = result
        self.cache_finished_req_reqs: list[object] = []

    def cache_finished_req(
        self, req: object, is_insert: bool = True, **kwargs
    ) -> CacheFinishedReqResult:
        self.cache_finished_req_reqs.append(req)
        return self.result


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
            _inflight=False,
        )
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
        )
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None


def test_preabort_detaches_session_and_preserves_slot():
    """Pre-aborted req (to_finish set before match_prefix) is detached from
    the session: session=None, abort_req() called. Slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator(page_size=16)
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )
        ],
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv=SimpleNamespace(kv_allocated_len=48, swa_evicted_seqlen=0),
        cache_protected_len=16,
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    # Req detached from session.
    assert req.session is None
    # Slot untouched.
    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_first_mid_abort_nukes_ephemeral_slot():
    """First-request mid-processing abort: no slot exists yet, ephemeral
    slot is created from req state and nuked via release_session."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    req.finished_reason = FINISH_ABORT("input too long")

    result = tree_cache.cache_finished_req(req)

    assert result.unhandled_kv_start is None
    assert req.kv is None
    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))


def test_nth_mid_abort_nukes_session_slot():
    """Nth-request mid-processing abort: slot exists, restore_to_req ran.
    ALL KV is wiped (release_session). Slot is deleted. Token IDs stay
    in req_nodes for next turn's re-prefill."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # Session already has a slot from a previous turn.
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=None,
        cache_protected_len=0,
    )

    # Mid-processing abort: req has the SESSION slot's pool_idx (restore_to_req ran).
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    # Slot wiped — deleted from slots dict.
    assert "session-a" not in tree_cache.slots
    # All KV freed: [0, 65) from release_session (slot extended to req's allocated).
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(65))
    # Pool slot returned.
    assert req_to_token_pool.free_slots == [0]
    assert req.req_pool_idx is None


# Shrink tests removed: streaming sessions are append-only after the
# rollback fix in session_controller (rollback_aborted_req).  The shrink
# code path in cache_finished_req no longer exists.


def test_trim_overshoot_postcondition():
    """`_trim_overshoot` postcondition: every per-req KV field is capped at
    target = origin+finished_len, output_ids is truncated, and the tail
    KV slots are freed. Covers both non-SWA fields (kv_committed_len,
    kv_allocated_len, output_ids) and SWA bookkeeping (swa_evicted_seqlen)
    in one shot — same invariant `_free_tail` enforces on the match_prefix
    path.
    """
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    # Overshoot scenario: origin=26, finished_len=12 -> target=38.
    # committed=40 (overshoot 2), allocated=44, swa_evicted=42 (> target),
    # output_ids extended to 14 by the overshoot round.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 42

    tree_cache._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.swa_evicted_seqlen == target
    assert len(req.output_ids) == 12
    # Tail [38, 44) freed by _free_kv_aligned.
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(38, 44))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


def test_cache_finished_req_delegates_result_from_inner_cache():
    """A non-streaming req reaches inner, whose result must arrive at the caller unchanged."""
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    inner = _FakeDelegatingInnerCache(
        req_to_token_pool,
        _FakeAllocator(),
        1,
        result=CacheFinishedReqResult(unhandled_kv_start=16),
    )
    tree_cache = StreamingSession(inner)

    req = _FakeReq("session-a", req_pool_idx=0, committed=20, allocated=24)
    req.session.streaming = False

    result = tree_cache.cache_finished_req(req, kv_len_to_handle=20)

    assert inner.cache_finished_req_reqs == [req]
    assert result.unhandled_kv_start == 16


def _make_paged_session(
    allocator: _FakeAllocator, inner_page_size: int
) -> StreamingSession:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    return StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, inner_page_size)
    )


def test_trim_overshoot_frees_only_the_whole_tail_pages():
    """With page=4, trimming to target=38 keeps the half page [36, 40) and frees [40, 44)."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))

    tree_cache._trim_overshoot(req, finished_len=12)

    assert req.kv.kv_allocated_len == 40
    assert req.kv_committed_len == 38
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(40, 44))


def test_trim_overshoot_aligns_swa_evicted_seqlen_up_to_the_page():
    """Page>1 sibling of the page=1 clamp case: swa_evicted=44 is capped at ceil(38)=40, not 38."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 44

    tree_cache._trim_overshoot(req, finished_len=12)

    assert req.kv.swa_evicted_seqlen == 40


def test_trim_overshoot_keeps_a_water_mark_below_the_aligned_target():
    """allocated=36 is below ceil(38)=40, so the min must keep 36 rather than claim unallocated pages."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    req = _FakeReq("session-a", req_pool_idx=0, committed=36, allocated=36)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))

    tree_cache._trim_overshoot(req, finished_len=12)

    assert req.kv.kv_allocated_len == 36
    assert allocator.freed == []


def test_free_tail_writes_back_the_aligned_prefix_len():
    """prefix_len=38 with page=4: the half page [36, 40) stays owned, so the water mark is 40, not 38."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    slot = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=44,
        kv=SimpleNamespace(kv_allocated_len=44, swa_evicted_seqlen=44),
        last_node=None,
        cache_protected_len=0,
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=44, allocated=44)
    req.kv.swa_evicted_seqlen = 44

    tree_cache._free_tail(slot, req, prefix_len=38)

    assert slot.kv.kv_allocated_len == 40
    assert req.kv.kv_allocated_len == 40
    assert slot.kv.swa_evicted_seqlen == 40
    assert req.kv.swa_evicted_seqlen == 40
    assert slot.kv_committed_len == 38
    assert req.kv_committed_len == 38
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(40, 44))


def test_bookkeeping_page_reads_the_allocator_not_the_inner_cache():
    """The inner cache's page (1) must not win over the allocator's page (4)."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=1)

    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))

    tree_cache._trim_overshoot(req, finished_len=12)

    assert req.kv.kv_allocated_len == 40
    assert allocator.freed[0].tolist() == list(range(40, 44))


def test_legacy_allocator_bookkeeping_page_cannot_align_a_physical_free():
    """A legacy real-length allocator bookkeeps at page 1, so target=38 cannot ceil-align to the physical page 4; the resulting unaligned physical free is rejected loudly instead of silently corrupting the page holding [36, 40)."""
    allocator = _FakeAllocator(page_size=4, uses_legacy_real_length_alloc=True)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 42

    with pytest.raises(AssertionError):
        tree_cache._trim_overshoot(req, finished_len=12)


def test_free_kv_aligned_rejects_an_unaligned_end():
    """The free end is a page-aligned water mark by invariant; an unaligned one must fail loudly."""
    allocator = _FakeAllocator(page_size=4)
    tree_cache = _make_paged_session(allocator, inner_page_size=4)

    with pytest.raises(AssertionError):
        tree_cache._free_kv_aligned(0, 38, 43)
