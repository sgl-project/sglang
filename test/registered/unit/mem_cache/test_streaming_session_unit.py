from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, ReqKvInfo
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    IncLockRefResult,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeAllocator(BaseTokenToKVPoolAllocator):
    """Single-pool double. Subclassing the base routes free_full / free_segment /
    free_segments into free(), so a new free API cannot slip past the recorder."""

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
    """Stands in for UnifiedRadixCache: owns the session and tries it first."""

    free_kv_row = BasePrefixCache.free_kv_row

    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.unlocked = []
        self.session = StreamingSession(self)

    def checkpoint(self, req, *, up_to):
        pass

    def maybe_hand_to_session(self, req):
        self.session.take(req)

    def match_prefix(self, params):
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def unlock(self, lock):
        if lock is not None:
            self.unlocked.append(lock)


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
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
            cache_protected_len=0,
        )
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.cache_salt = None
        self.last_node = None
        self.swa_branching_seqlen = None
        self.lock = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None

    def detach_kv(self):
        kv, self.kv = self.kv, ReqKvInfo()
        return kv


def _single_row_cache():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    return _FakeInnerCache(
        _FakeReqToTokenPool(req_to_token), _FakeAllocator(), page_size=1
    )


def _finished_turn(tree_cache, req):
    """The session took the record at allocation; the turn then finished."""
    tree_cache.maybe_hand_to_session(req)
    req.finished_reason = FINISH_LENGTH(length=0)
    assert tree_cache.session.try_cache_finished_req(req)
    return tree_cache.session.slots[req.session.session_id]


def test_finished_turn_leaves_its_record_in_the_slot():
    req = _FakeReq("session-a", req_pool_idx=0, committed=4, allocated=4)
    record = req.kv
    record.mamba_next_track_idx = 1
    record.set_evicted_seqlen(ComponentType.SWA, 2)
    req.swa_branching_seqlen = 8

    slot = _finished_turn(_single_row_cache(), req)

    assert slot.kv is record
    assert slot.kv.mamba_next_track_idx == 1
    assert slot.kv.component_evicted_seqlens == {ComponentType.SWA: 2}
    assert req.kv is not record
    assert req.swa_branching_seqlen is None


def test_preabort_detaches_session_and_preserves_slot():
    """A request aborted before its match is detached from the session; the
    slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
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
    tree_cache = inner
    tree_cache.session.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=48,
            kv_allocated_len=48,
            cache_protected_len=16,
        ),
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    assert req.session is None
    slot = tree_cache.session.slots["session-a"]
    assert slot.kv.req_pool_idx == 0
    assert slot.kv.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


@pytest.mark.parametrize("uuid", [None, 17])
def test_release_session_preserves_component_lock_receipt(uuid):
    """Closing a session releases only the component locks it acquired."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = inner

    lock_node = SimpleNamespace(id=42)
    acquired = IncLockRefResult(
        node_id=42,
        skipped_lock_components=(ComponentType.MAMBA,),
    )
    acquired.set_lock_uuid(ComponentType.SWA, uuid)
    acquired.set_lock_uuid(ComponentType.SWA, 19, lock_host=True)
    acquired.set_lock_uuid(ComponentType.AUXILIARY_SWA, 23)
    acquired.set_lock_uuid(ComponentType.AUXILIARY_SWA, None, lock_host=True)
    tree_cache.session.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=50,
            kv_allocated_len=50,
            cache_protected_len=0,
        ),
        lock=TreeLock(lock_node, acquired.to_dec_params()),
    )

    acquired.set_lock_uuid(ComponentType.SWA, 99)
    acquired.set_lock_uuid(ComponentType.SWA, 99, lock_host=True)
    acquired.set_lock_uuid(ComponentType.AUXILIARY_SWA, 99)
    acquired.set_lock_uuid(ComponentType.AUXILIARY_SWA, 99, lock_host=True)
    tree_cache.session.release_session("session-a")

    (lock,) = inner.unlocked
    assert lock.node is lock_node
    params = lock.receipt
    assert params.skipped_lock_components == (ComponentType.MAMBA,)
    assert params.get_lock_uuid(ComponentType.SWA) == uuid
    assert params.get_lock_uuid(ComponentType.SWA, lock_host=True) == 19
    assert params.get_lock_uuid(ComponentType.AUXILIARY_SWA) == 23
    assert params.get_lock_uuid(ComponentType.AUXILIARY_SWA, lock_host=True) is None
    for lock_host in (False, True):
        with pytest.raises(KeyError):
            params.get_lock_uuid(ComponentType.MAMBA, lock_host=lock_host)
        with pytest.raises(KeyError):
            DecLockRefParams().get_lock_uuid(ComponentType.SWA, lock_host=lock_host)


def test_trim_overshoot_postcondition():
    """Every per-request KV cursor is capped at origin + finished_len and the
    tail past it is freed."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    tree_cache = _FakeInnerCache(req_to_token_pool, allocator, page_size)

    # target = 26 + 12 = 38; the overshoot committed 40 and allocated 44.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.set_evicted_seqlen(ComponentType.SWA, 42)

    tree_cache.session._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == target
    assert len(req.output_ids) == 12
    # [38, 42) already gave its SWA back, so the free splits at 42.
    assert [t.tolist() for t in allocator.freed] == [[38, 39, 40, 41], [42, 43]]


@pytest.mark.parametrize("operation", ["trim", "match"])
@pytest.mark.parametrize("component", [ComponentType.SWA, ComponentType.AUXILIARY_SWA])
def test_session_rewind_keeps_component_cursors_page_aligned(operation, component):
    """Rewinding below any component cursor must free whole pages."""
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator(page_size=page_size)
    tree_cache = _FakeInnerCache(req_to_token_pool, allocator, page_size)

    # origin=26, finished=12 -> raw target 38 (mid-page); cursor 48 > target.
    req = _FakeReq("session-a", req_pool_idx=0, committed=52, allocated=64)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.set_evicted_seqlen(ComponentType.SWA, 16)
    req.kv.set_evicted_seqlen(component, 48)

    if operation == "trim":
        tree_cache.session._trim_overshoot(req, finished_len=12)
        assert len(req.output_ids) == 12
    else:
        tree_cache.session.slots["session-a"] = SessionSlot(kv=req.detach_kv())
        req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
        result = tree_cache.match_prefix(SimpleNamespace(req=req, key=list(range(38))))
        assert result.device_indices.tolist() == list(range(32))

    # Rewound to floor_align(38) = 32; every cursor lands page-aligned.
    assert req.kv.kv_allocated_len == 32
    assert req.kv.kv_committed_len == 32
    assert req.kv.get_evicted_seqlen(component) == 32
    assert req.kv.max_evicted_seqlen == 32
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == (
        32 if component == ComponentType.SWA else 16
    )
    assert [t.tolist() for t in allocator.freed] == (
        [list(range(32, 48)), list(range(48, 64))]
        if component == ComponentType.SWA
        else [list(range(32, 64))]
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
