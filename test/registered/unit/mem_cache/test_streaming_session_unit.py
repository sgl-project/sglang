from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache, SessionSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


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


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.session = SimpleNamespace(session_id=session_id, streaming=True)
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv_allocated_len = allocated
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.swa_evicted_seqlen = 0
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.pop_overallocated_calls = 0
        self.to_finish = None
        self.finished_reason = None

    def pop_committed_kv_cache(self):
        assert not self.kv_committed_freed
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        assert not self.kv_overallocated_freed
        self.pop_overallocated_calls += 1
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


def test_streaming_release_kv_cache_trims_overallocated_tail(monkeypatch):
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = SessionAwareCache(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=17, allocated=40)

    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.get_global_server_args",
        lambda: SimpleNamespace(page_size=page_size, speculative_algorithm="eagle"),
    )

    release_kv_cache(req, tree_cache)

    slot = tree_cache.slots["session-a"]
    assert req.pop_overallocated_calls == 1
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True
    assert req.req_pool_idx is None
    assert slot.kv_committed_len == 17
    assert slot.kv_allocated_len == 17
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(32, 40))


def test_release_session_recomputes_current_tree_owned_prefix():
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()

    full_match = MatchResult(
        device_indices=torch.tensor(list(range(16)) + list(range(64, 96))),
        last_device_node="stale-expanded",
        last_host_node="stale-expanded",
    )
    protected_match = MatchResult(
        device_indices=torch.tensor(list(range(16))),
        last_device_node="current-protected",
        last_host_node="current-protected",
    )
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size,
        match_results=[full_match, protected_match],
    )
    tree_cache = SessionAwareCache(inner)

    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
        last_node="outdated-node",
        cache_protected_len=32,
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=48, allocated=48)

    tree_cache.release_session("session-a", req)

    assert inner.dec_lock_ref_calls == ["current-protected"]
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(16, 48))


def test_release_session_never_grows_tree_owned_prefix():
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()

    overmatched = MatchResult(
        device_indices=torch.tensor(list(range(48))),
        last_device_node="overmatched-node",
        last_host_node="overmatched-node",
    )
    capped_match = MatchResult(
        device_indices=torch.tensor(list(range(16))),
        last_device_node="original-lock-node",
        last_host_node="original-lock-node",
    )
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size,
        match_results=[overmatched, capped_match],
    )
    tree_cache = SessionAwareCache(inner)

    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
        last_node="outdated-node",
        cache_protected_len=16,
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=48, allocated=48)

    tree_cache.release_session("session-a", req)

    assert inner.dec_lock_ref_calls == ["original-lock-node"]
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(16, 48))


def test_match_prefix_abort_does_not_restore_live_session_slot():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
            )
        ],
    )
    tree_cache = SessionAwareCache(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
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

    slot = tree_cache.slots["session-a"]
    assert req.req_pool_idx == 1
    assert req.kv_committed_len == 1
    assert req.kv_allocated_len == 1
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_aborted_streaming_turn_preserves_slot_and_accounting(monkeypatch):
    page_size = 16
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = SessionAwareCache(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv_allocated_len=48,
        cache_protected_len=16,
        swa_evicted_seqlen=8,
        last_node="lock-node",
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=5, allocated=23)
    req.finished_reason = FINISH_ABORT("too long")

    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.get_global_server_args",
        lambda: SimpleNamespace(page_size=page_size, speculative_algorithm="eagle"),
    )

    release_kv_cache(req, tree_cache)

    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv_allocated_len == 48
    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True
    assert req.req_pool_idx is None
    assert req.pop_overallocated_calls == 1
    assert tree_cache.session_held_tokens() == 32
    assert tree_cache.session_held_full_tokens() == 32
    assert tree_cache.session_held_swa_tokens() == 32
    assert tree_cache.session_held_req_count() == 1
    assert req_to_token_pool.free_slots == [1]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(128, 151))

    tree_cache.release_session("session-a")

    assert tree_cache.session_held_tokens() == 0
    assert tree_cache.session_held_swa_tokens() == 0
    assert tree_cache.session_held_req_count() == 0
    assert req_to_token_pool.free_slots == [1, 0]
    assert len(allocator.freed) == 2
    assert allocator.freed[1].tolist() == list(range(16, 48))


def test_session_shrink_frees_orphaned_tail():
    """When a session's KV shrinks (client retried with shorter prompt),
    the orphaned tail pages must be freed before save_from_req overwrites
    the slot."""
    page_size = 16
    pool_size = 256
    req_to_token = torch.arange(pool_size, dtype=torch.int32).reshape(1, pool_size)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = SessionAwareCache(inner)

    # Session slot has 128 tokens committed
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=128,
        kv_allocated_len=128,
        last_node="lock-node",
        cache_protected_len=16,
    )

    # New request finished with only 48 tokens (client truncated)
    req = _FakeReq("session-a", req_pool_idx=0, committed=48, allocated=48)

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    # Slot should now reflect the shrunk state
    assert slot.kv_committed_len == 48
    assert slot.kv_allocated_len == 48
    # The tail [48:128] should have been freed (page-aligned: [48:128])
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(48, 128))


def test_session_shrink_page_aligns_free_start():
    """The shrink free should page-align the start to avoid freeing
    tokens that are still part of the new committed prefix."""
    page_size = 16
    pool_size = 256
    req_to_token = torch.arange(pool_size, dtype=torch.int32).reshape(1, pool_size)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = SessionAwareCache(inner)

    # Session slot has 128 tokens
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=128,
        kv_allocated_len=128,
        last_node="lock-node",
        cache_protected_len=16,
    )

    # New request committed 50 tokens (not page-aligned)
    req = _FakeReq("session-a", req_pool_idx=0, committed=50, allocated=50)

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 50
    # Free start should be ceil_align(50, 16) = 64, not 50
    # So freed range is [64:128]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(64, 128))
