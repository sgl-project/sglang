import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import zmq

from sglang.srt.managers.io_struct import AbortReq, SessionReapPlan
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, MatchResult
from sglang.srt.session.session_controller import Session, SessionController
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
    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []
        self.dec_lock_ref_params = []
        self.dec_lock_ref_skip_swa = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)
        self.dec_lock_ref_params.append(args[0] if args else kwargs.get("params"))
        self.dec_lock_ref_skip_swa.append(kwargs.get("skip_swa", False))

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeSessionTreeCache:
    def __init__(self):
        self.released = []

    def release_session(self, session_id):
        self.released.append(session_id)

    def release_radix_session(self, session_id):
        pass


class _FinishedReq:
    multimodal_inputs = None

    def finished(self):
        return True


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.rid = session_id
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            _inflight=False,
            _inflight_rid=None,
        )

        def abort_req(rid):
            if self.session._inflight_rid == rid:
                self.session._inflight = False
                self.session._inflight_rid = None

        self.session.abort_req = abort_req
        self.kv = ReqKvInfo(
            req_pool_idx=req_pool_idx,
            kv_committed_len=committed,
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
            cache_protected_len=0,
        )
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.cache_salt = None
        self.last_node = None
        self.swa_branching_seqlen = None
        self.lock_receipt = DecLockRefParams()
        self.swa_prefix_lock_released = False
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None

    def detach_kv(self):
        kv, self.kv = self.kv, ReqKvInfo()
        return kv


def test_session_slot_round_trip_preserves_mamba_state():
    # The mamba state rides in the shared ReqKvInfo record. mamba_branching_seqlen
    # is a per-turn match observation on the Req and is not preserved by the slot.
    req = _FakeReq("session-a", req_pool_idx=0, committed=4, allocated=4)
    req.kv.mamba_next_track_idx = 1
    req.kv.mamba_last_track_idx = 0
    req.kv.mamba_last_track_seqlen = 3

    slot = SessionSlot()
    slot.save_from_req(req, is_first=True)

    next_req = _FakeReq("session-a", req_pool_idx=1, committed=0, allocated=0)
    slot.restore_to_req(next_req)

    assert next_req.kv.mamba_next_track_idx == 1
    assert next_req.kv.mamba_last_track_idx == 0
    assert next_req.kv.mamba_last_track_seqlen == 3


def test_preabort_detaches_session_and_preserves_slot():
    """Pre-aborted req (to_finish set before match_prefix) is detached from
    the session: session=None, abort_req(rid) called. Slot stays intact."""
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
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=48,
            kv_allocated_len=48,
            swa_evicted_seqlen=0,
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

    # Req detached from session.
    assert req.session is None
    # Slot untouched.
    slot = tree_cache.slots["session-a"]
    assert slot.kv.req_pool_idx == 0
    assert slot.kv.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_preabort_detaches_without_slot():
    """Pre-aborted req detaches even when its session has no active slot."""
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    raw_result = MatchResult(
        device_indices=torch.tensor([], dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
    )
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[raw_result],
    )
    tree_cache = StreamingSession(inner)
    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
    req.session._inflight_rid = req.rid
    aborted = []
    original_abort_req = req.session.abort_req

    def record_abort_req(rid):
        aborted.append(rid)
        original_abort_req(rid)

    req.session.abort_req = record_abort_req
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    assert req.session is None
    assert aborted == ["session-a"]
    assert result is raw_result
    assert len(result.device_indices) == 0
    assert tree_cache.slots == {}


def test_preabort_of_non_inflight_req_keeps_inflight():
    """Only the matching in-flight request may clear session state."""

    def match_preaborted_req(rid):
        req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
        req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
        raw_result = MatchResult(
            device_indices=torch.tensor([], dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )
        inner = _FakeInnerCache(
            req_to_token_pool,
            _FakeAllocator(),
            page_size=16,
            match_results=[raw_result],
        )
        tree_cache = StreamingSession(inner)
        req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
        req.rid = rid
        req.session._inflight = True
        req.session._inflight_rid = "turn-1"
        abort_req = Mock()
        real_abort_req = req.session.abort_req

        def record_abort_req(request_rid):
            if req.session._inflight_rid == request_rid:
                abort_req(request_rid)
            real_abort_req(request_rid)

        req.session.abort_req = record_abort_req
        req.to_finish = FINISH_ABORT("too long")

        tree_cache.match_prefix(
            SimpleNamespace(
                req=req,
                key=SimpleNamespace(token_ids=list(range(64))),
            )
        )
        return req, abort_req

    non_inflight_req, abort_req = match_preaborted_req("stub")
    assert non_inflight_req.session is None
    abort_req.assert_not_called()

    inflight_req, abort_req = match_preaborted_req("turn-1")
    assert inflight_req.session is None
    abort_req.assert_called_once_with("turn-1")


def test_first_mid_abort_nukes_ephemeral_slot():
    """First-request mid-processing abort: no slot exists yet, ephemeral
    slot is created from req state and nuked via release_session."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    req.finished_reason = FINISH_ABORT("input too long")

    tree_cache.cache_finished_req(req)

    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.kv.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))


def test_nth_mid_abort_nukes_session_slot():
    """Nth-request mid-processing abort: slot exists, restore_to_req ran.
    ALL KV is wiped (release_session). Slot is deleted. Token IDs stay
    in req_nodes for next turn's re-prefill."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # Mid-processing abort: restore_to_req ran, so the req runs on the slot's
    # record, which this turn has grown to committed=60 / allocated=65.
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    req.finished_reason = FINISH_ABORT("client disconnected")
    tree_cache.slots["session-a"] = SessionSlot(kv=req.kv, last_node=None)

    tree_cache.cache_finished_req(req)

    # Slot wiped — deleted from slots dict.
    assert "session-a" not in tree_cache.slots
    # All KV freed: [0, 65) from release_session.
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(65))
    # Pool slot returned.
    assert req_to_token_pool.free_slots == [0]
    assert req.kv.req_pool_idx is None


def test_release_session_threads_mamba_lock_receipt():
    """release_session must forward the slot's mamba lock receipt to
    dec_lock_ref. The first req's last_node may be full-only-locked (mamba
    not taken at inc), so without the receipt the release would drop a mamba
    lock the session never took -- another request's, on a shared node."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = StreamingSession(inner)

    lock_node = SimpleNamespace(id=42)
    tree_cache.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=50,
            kv_allocated_len=50,
            swa_evicted_seqlen=0,
            cache_protected_len=0,
        ),
        last_node=lock_node,
    )

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [lock_node]
    params = inner.dec_lock_ref_params[0]
    assert params is not None
    assert params.skipped_lock_components == ()
    assert inner.dec_lock_ref_skip_swa == [False]


def test_release_session_skips_swa_after_early_release():
    """A slot saved from a req that early-released its SWA lock
    (swa_prefix_lock_released) must release with skip_swa, or the session
    close double-releases the SWA segment."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = StreamingSession(inner)

    lock_node = SimpleNamespace(id=42)
    tree_cache.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            req_pool_idx=0,
            kv_committed_len=50,
            kv_allocated_len=50,
            swa_evicted_seqlen=0,
            cache_protected_len=0,
        ),
        last_node=lock_node,
        lock_receipt=DecLockRefParams(node_id=42, swa_uuid_for_lock=7),
        swa_prefix_lock_released=True,
    )

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [lock_node]
    assert inner.dec_lock_ref_params[0].swa_uuid_for_lock == 7
    assert inner.dec_lock_ref_skip_swa == [True]


def test_session_slot_does_not_restore_swa_branching_seqlen():
    req = _FakeReq("session-a", req_pool_idx=0, committed=4, allocated=4)
    req.swa_branching_seqlen = 8

    slot = SessionSlot()
    slot.save_from_req(req, is_first=True)

    next_req = _FakeReq("session-a", req_pool_idx=1, committed=0, allocated=0)
    slot.restore_to_req(next_req)

    assert req.swa_branching_seqlen is None
    assert next_req.swa_branching_seqlen is None


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
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
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
    assert req.kv.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.swa_evicted_seqlen == target
    assert len(req.output_ids) == 12
    # Tail [38, 44) freed by _free_kv_aligned, split at the pre-trim eviction
    # floor 42: [38, 42) gave its SWA peers back already, so it goes back full-only.
    assert [t.tolist() for t in allocator.freed] == [[38, 39, 40, 41], [42, 43]]


def test_trim_overshoot_keeps_cursor_page_aligned_on_paged():
    """A mid-page trim target must not become the SWA eviction cursor (the
    dead/alive split there frees the shared page twice); rewind to the boundary."""
    page_size = 16
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = _FakeReqToTokenPool(req_to_token)
    allocator = _FakeAllocator(page_size=page_size)
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    # origin=26, finished=12 -> raw target 38 (mid-page); cursor 48 > target.
    req = _FakeReq("session-a", req_pool_idx=0, committed=52, allocated=64)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 48

    tree_cache._trim_overshoot(req, finished_len=12)

    # Rewound to floor_align(38) = 32; every cursor lands page-aligned.
    assert req.kv.kv_allocated_len == 32
    assert req.kv.kv_committed_len == 32
    assert req.kv.swa_evicted_seqlen == 32
    assert len(req.output_ids) == 12
    # Freed [32, 64): [32, 48) below the old cursor goes back full-only,
    # [48, 64) both halves.
    assert [t.tolist() for t in allocator.freed] == [
        list(range(32, 48)),
        list(range(48, 64)),
    ]


def test_release_session_skips_lazy_ping_pong_sentinels():
    allocator = MambaSlotAllocator(size=8, device="cpu")
    allocator.alloc(8)
    req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
    inner = _FakeInnerCache(
        req_to_token_pool,
        _FakeAllocator(),
        page_size=1,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            mamba_pool_idx=torch.tensor(3),
            mamba_ping_pong_track_buffer=torch.tensor([5, -1]),
        ),
    )

    tree_cache.release_session("session-a")

    assert set(allocator.free_slots.tolist()) == {3, 5}
    assert -1 not in allocator.free_slots.tolist()
    assert allocator.available_size() == 2


def test_session_held_mamba_slots_ignores_sentinels():
    req_to_token_pool = SimpleNamespace(mamba_allocator=_FakeAllocator())
    inner = _FakeInnerCache(
        req_to_token_pool,
        _FakeAllocator(),
        page_size=1,
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        kv=ReqKvInfo(
            mamba_pool_idx=torch.tensor(3),
            mamba_ping_pong_track_buffer=torch.tensor([5, -1]),
        ),
    )
    assert tree_cache.session_held_mamba_slots() == 2

    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-b"] = SessionSlot(
        kv=ReqKvInfo(
            mamba_pool_idx=torch.tensor(4),
            mamba_ping_pong_track_buffer=torch.tensor([-1, -1]),
        ),
    )
    assert tree_cache.session_held_mamba_slots() == 1


def test_session_controller_plan_reap_defers_application():
    tree_cache = _FakeSessionTreeCache()
    controller = SessionController(tree_cache)
    ready = Session(16, "ready")
    ready.close_on_finish = True
    ready.req_nodes["req"] = SimpleNamespace(req=_FinishedReq())
    timed_out = Session(16, "timed-out", timeout=1)
    timed_out.last_active_time = time.monotonic() - 2
    controller.sessions.update({"ready": ready, "timed-out": timed_out})

    controller._last_reap_time = 10
    assert controller.plan_reap(10.5) is None
    plan = controller.plan_reap(12)

    assert plan == SessionReapPlan(
        deferred=["ready"],
        timed_out=["timed-out"],
    )
    assert set(controller.sessions) == {"ready", "timed-out"}
    assert tree_cache.released == []


def test_session_controller_apply_reap_filters_stale_sessions():
    tree_cache = _FakeSessionTreeCache()
    controller = SessionController(tree_cache)
    deferred = Session(16, "deferred")
    deferred.close_on_finish = True
    not_deferred = Session(16, "not-deferred")
    timed_out = Session(16, "timed-out")
    controller.sessions.update(
        {
            "deferred": deferred,
            "not-deferred": not_deferred,
            "timed-out": timed_out,
        }
    )

    controller.apply_reap(
        SessionReapPlan(
            deferred=["deferred", "missing", "not-deferred"],
            timed_out=["timed-out", "missing"],
        )
    )

    assert tree_cache.released == ["deferred", "timed-out"]
    assert set(controller.sessions) == {"not-deferred"}


def test_session_controller_apply_reap_is_rank_symmetric():
    controllers = []
    for _ in range(2):
        tree_cache = _FakeSessionTreeCache()
        controller = SessionController(tree_cache)
        deferred = Session(16, "deferred")
        deferred.close_on_finish = True
        controller.sessions.update(
            {"deferred": deferred, "untouched": Session(16, "untouched")}
        )
        controllers.append(controller)

    plan = SessionReapPlan(deferred=["deferred"], timed_out=[])
    for controller in controllers:
        controller.apply_reap(plan)

    assert set(controllers[0].sessions) == set(controllers[1].sessions) == {"untouched"}


def test_request_receiver_classifies_session_reap_plan_as_work():
    receiver = object.__new__(SchedulerRequestReceiver)
    plan = SessionReapPlan(deferred=[], timed_out=[])

    work, control = receiver._split_work_and_control_reqs([plan])

    assert work == [plan]
    assert control == []


def _controller_with_deferred_session():
    tree_cache = _FakeSessionTreeCache()
    controller = SessionController(tree_cache)
    deferred = Session(16, "deferred", streaming=True)
    deferred.close_on_finish = True
    deferred.req_nodes["req"] = SimpleNamespace(req=_FinishedReq())
    controller.sessions["deferred"] = deferred
    controller._last_reap_time = -1.0  # the first plan_reap call plans
    return controller, tree_cache


def _make_single_rank_receiver(
    controller, skipper, pp_rank=0, pp_size=1, plan_session_reap=None
):
    """Receiver wired as the scheduler wires it, with the collective plumbing
    faked for one non-dp rank (broadcast is the identity at tp_size=1)."""
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=skipper,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dp_rank=0,
        ),
        tp_group=None,
        tp_cpu_group=None,
        attn_tp_group=None,
        attn_tp_cpu_group=None,
        attn_cp_group=None,
        attn_cp_cpu_group=None,
        world_group=SimpleNamespace(cpu_group=None),
        server_args=None,
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
        plan_session_reap=(
            plan_session_reap if plan_session_reap is not None else controller.plan_reap
        ),
    )


def test_skipped_receive_cycle_still_reaps_deferred_close():
    """recv_skipper declining the poll must not strand deferred closes: the
    leader still plans the reap and the plan still rides the per-step
    broadcast, so every rank applies it at the same loop position."""
    controller, tree_cache = _controller_with_deferred_session()
    receiver = _make_single_rank_receiver(
        controller, SimpleNamespace(handle=lambda _last_batch: False)
    )

    with patch(
        "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
        return_value=SimpleNamespace(enable_dp_attention=False, pp_rank=0),
    ):
        recv_reqs = receiver.recv_requests()

    # Planning ran despite the skipped receive, and nothing else rode along.
    assert recv_reqs == [SessionReapPlan(deferred=["deferred"], timed_out=[])]

    # Applied at process_input_requests' head, the deferred close completes.
    controller.apply_reap(recv_reqs[0])
    assert tree_cache.released == ["deferred"]
    assert controller.sessions == {}


def test_skipped_receive_cycle_broadcasts_local_aborts():
    """Timeout aborts the caller already polled still ride the per-step
    broadcast on a receive-skipped cycle instead of being dropped."""
    controller, _ = _controller_with_deferred_session()
    receiver = _make_single_rank_receiver(
        controller, SimpleNamespace(handle=lambda _last_batch: False)
    )
    abort = AbortReq(rid="stuck")

    with patch(
        "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
        return_value=SimpleNamespace(enable_dp_attention=False, pp_rank=0),
    ):
        recv_reqs = receiver.recv_requests(local_reqs=[abort])

    assert recv_reqs == [
        SessionReapPlan(deferred=["deferred"], timed_out=[]),
        abort,
    ]


def test_receive_cycle_appends_reap_plan_to_pulled_reqs():
    """On a receive cycle the leader's plan is appended to the pulled reqs and
    rides the same broadcast (the pre-fix behavior, preserved)."""
    controller, _ = _controller_with_deferred_session()
    receiver = _make_single_rank_receiver(
        controller, SimpleNamespace(handle=lambda _last_batch: True)
    )

    with (
        patch(
            "sglang.srt.managers.scheduler_components.request_receiver.sock_recv",
            side_effect=zmq.ZMQError(),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
            return_value=SimpleNamespace(enable_dp_attention=False, pp_rank=0),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.request_receiver.get_disagg",
            return_value=SimpleNamespace(
                language_only=False, encoder_transfer_backend=None
            ),
        ),
    ):
        recv_reqs = receiver.recv_requests()

    assert recv_reqs == [SessionReapPlan(deferred=["deferred"], timed_out=[])]


def test_skipped_receive_cycle_relays_reap_plan_across_pp_stages():
    """PP>1: a skipped cycle must keep the point-to-point relay so the
    leader's plan reaches every stage at the same iteration; later stages
    must not plan locally (single-planner invariant)."""
    controller0, tree_cache0 = _controller_with_deferred_session()
    controller1, tree_cache1 = _controller_with_deferred_session()
    skipper = SimpleNamespace(handle=lambda _last_batch: False)
    stage0 = _make_single_rank_receiver(controller0, skipper)
    stage1_plan = Mock(side_effect=controller1.plan_reap)
    stage1 = _make_single_rank_receiver(
        controller1, skipper, pp_rank=1, pp_size=2, plan_session_reap=stage1_plan
    )

    with patch(
        "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
        return_value=SimpleNamespace(enable_dp_attention=False, pp_rank=0),
    ):
        recv0 = stage0.recv_requests()
    with (
        patch(
            "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
            return_value=SimpleNamespace(enable_dp_attention=False, pp_rank=1),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.request_receiver."
            "point_to_point_pyobj",
            side_effect=lambda *args, **kwargs: recv0,
        ) as relay,
    ):
        # The event loop forwards stage 0's list; stage 1's pull receives it.
        recv1 = stage1.recv_requests()

    plan = SessionReapPlan(deferred=["deferred"], timed_out=[])
    assert recv0 == [plan]
    assert recv1 == [plan]
    relay.assert_called_once()
    stage1_plan.assert_not_called()

    # Every stage applies the same plan.
    controller0.apply_reap(recv0[0])
    controller1.apply_reap(recv1[0])
    assert tree_cache0.released == tree_cache1.released == ["deferred"]
    assert controller0.sessions == controller1.sessions == {}


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
