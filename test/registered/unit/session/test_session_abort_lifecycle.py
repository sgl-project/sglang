"""Regression tests for aborting a streaming-session turn that is still queued.

`Session.create_req` marks a streaming session in-flight so that only one turn
at a time may borrow the session's `SessionSlot`. A turn that is removed from
`Scheduler.waiting_queue` before it ever runs must clear that flag, must not be
committed as a finished turn, and must leave the slot (and the prefix it holds)
intact for the next turn.

These tests drive the real `Scheduler.abort_request` and
`Scheduler._abort_on_waiting_timeout` against a real `Session`, so they fail if
either removal path stops finalizing the session correctly.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq, SessionParams
from sglang.srt.managers.schedule_batch import get_parallel
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


def _make_recv_req(rid, input_ids, session_id):
    """Minimal stand-in for TokenizedGenerateReqInput.

    Only carries the attributes `Session.create_req` reads; the request object
    it builds is a real `Req`.
    """
    sampling_params = SamplingParams(max_new_tokens=8)
    sampling_params.normalize(None)
    return SimpleNamespace(
        rid=rid,
        input_ids=list(input_ids),
        session_params=SessionParams(id=session_id),
        sampling_params=sampling_params,
        lora_id=None,
        custom_logit_processor=None,
        stream=True,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=None,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=None,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class _RecordingTreeCache:
    """Tree cache that records release calls instead of performing them.

    A queued session turn borrows its req-pool / KV / Mamba state from the
    `SessionSlot`, so any release reaching this object would be operating on
    state the slot still owns.
    """

    def __init__(self):
        self.cache_finished_req_calls = []
        self.released_aborted_rids = []

    def cache_finished_req(self, req, **kwargs):
        self.cache_finished_req_calls.append(req.rid)

    def release_aborted_request(self, rid):
        self.released_aborted_rids.append(rid)


def _make_scheduler(tree_cache, *, waiting_queue):
    """Smallest `Scheduler` exposing the real queued-abort paths."""
    s = Scheduler.__new__(Scheduler)
    s.chunked_req = None
    s._pending_chunked_abort_req = None
    s.waiting_queue = waiting_queue
    s.tree_cache = tree_cache
    s.enable_hicache_storage = False
    s.enable_hierarchical_cache = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.ipc_channels = MagicMock()
    s.grammar_manager = MagicMock()
    s.ps = SimpleNamespace(pp_size=1)
    s.running_batch = None
    s.last_batch = None
    s.dllm_config = None
    return s


def _queue_turn(session, scheduler, rid, input_ids, session_id="s0"):
    req = session.create_req(
        _make_recv_req(rid, input_ids, session_id), None, VOCAB_SIZE
    )
    req.mamba_pool_idx = None
    req.req_pool_idx = None
    req.kv = None
    scheduler.waiting_queue.append(req)
    return req


def _finish_turn(session, req, output_ids):
    """Mark a turn successfully finished the way the streaming cache does."""
    req.output_ids = list(output_ids)
    session.finish_req(req)


class TestQueuedSessionAbortLifecycle(CustomTestCase):
    def setUp(self):
        self._parallel_override = get_parallel().override(tp_rank=0)
        self._parallel_override.__enter__()
        self.addCleanup(self._parallel_override.__exit__, None, None, None)

    # -- explicit abort -------------------------------------------------

    def test_queued_abort_releases_streaming_session(self):
        """Real abort_request must clear _inflight so the session continues."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        first = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        _finish_turn(session, first, [91, 92])
        scheduler.waiting_queue.clear()

        queued = _queue_turn(session, scheduler, "r2", [44, 55, 66])
        self.assertTrue(session._inflight)

        scheduler.abort_request(AbortReq(rid="r2"))

        self.assertNotIn(queued, scheduler.waiting_queue)
        self.assertFalse(session._inflight)

        follow_up = session.create_req(
            _make_recv_req("r3", [77, 88], "s0"), None, VOCAB_SIZE
        )
        self.assertIsNone(follow_up.to_finish)

    def test_queued_abort_does_not_commit_the_turn(self):
        """The aborted prompt must not enter the session's committed history."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        first = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        _finish_turn(session, first, [91, 92])
        scheduler.waiting_queue.clear()
        committed_len = session.committed_origin_len

        _queue_turn(session, scheduler, "r2", [44, 55, 66])
        scheduler.abort_request(AbortReq(rid="r2"))

        # finish_req would have advanced the rollback point over the
        # never-executed prompt.
        self.assertEqual(session.committed_origin_len, committed_len)

        follow_up = session.create_req(
            _make_recv_req("r3", [77, 88], "s0"), None, VOCAB_SIZE
        )
        context = list(follow_up.origin_input_ids)
        self.assertEqual(context, [11, 12, 13, 91, 92, 77, 88])
        for token in (44, 55, 66):
            self.assertNotIn(token, context)

    def test_queued_abort_preserves_committed_prefix(self):
        """The slot's committed prefix must survive a queued abort."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        first = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        _finish_turn(session, first, [91, 92])
        scheduler.waiting_queue.clear()

        _queue_turn(session, scheduler, "r2", [44, 55, 66])
        scheduler.abort_request(AbortReq(rid="r2"))

        # req_nodes still points at the last successful turn, and nothing was
        # handed to the cache for release.
        self.assertEqual(list(session.req_nodes), ["r1"])
        self.assertEqual(tree_cache.cache_finished_req_calls, [])

        follow_up = session.create_req(
            _make_recv_req("r3", [77, 88], "s0"), None, VOCAB_SIZE
        )
        self.assertEqual(list(follow_up.origin_input_ids)[:5], [11, 12, 13, 91, 92])

    def test_first_turn_queued_abort_is_safe(self):
        """Aborting before any turn committed must leave the session reusable."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        _queue_turn(session, scheduler, "r1", [11, 12, 13])
        scheduler.abort_request(AbortReq(rid="r1"))

        self.assertFalse(session._inflight)
        self.assertEqual(dict(session.req_nodes), {})
        self.assertIsNone(session.committed_origin_len)
        self.assertEqual(tree_cache.cache_finished_req_calls, [])

        retry = session.create_req(
            _make_recv_req("r2", [21, 22], "s0"), None, VOCAB_SIZE
        )
        self.assertIsNone(retry.to_finish)
        self.assertEqual(list(retry.origin_input_ids), [21, 22])

    # -- waiting timeout ------------------------------------------------

    def test_waiting_timeout_releases_streaming_session(self):
        """The timeout path must finalize the session like an explicit abort."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        first = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        _finish_turn(session, first, [91, 92])
        scheduler.waiting_queue.clear()

        timed_out = _queue_turn(session, scheduler, "r2", [44, 55, 66])
        # Entry time far enough in the past that any positive timeout fires.
        timed_out.time_stats = SimpleNamespace(wait_queue_entry_time=1e-6)

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get",
            return_value=1,
        ):
            scheduler._abort_on_waiting_timeout()

        self.assertNotIn(timed_out, scheduler.waiting_queue)
        self.assertFalse(session._inflight)

        follow_up = session.create_req(
            _make_recv_req("r3", [77, 88], "s0"), None, VOCAB_SIZE
        )
        self.assertIsNone(follow_up.to_finish)
        context = list(follow_up.origin_input_ids)
        self.assertEqual(context, [11, 12, 13, 91, 92, 77, 88])

    # -- non-session traffic --------------------------------------------

    def test_non_session_queued_abort_unchanged(self):
        """Requests without a session must keep their existing behaviour."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        plain = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        plain.session = None

        scheduler.abort_request(AbortReq(rid="r1"))

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(tree_cache.cache_finished_req_calls, [])
        # The session was never notified, because the request no longer owned one.
        self.assertTrue(session._inflight)


class _RawCachePathGuard:
    """Inner cache standing behind a real `StreamingSession`.

    Delegation to `cache_finished_req` here means the request stopped looking
    like a streaming request and reached the raw cache, which would free state
    the `SessionSlot` still owns.
    """

    def __init__(self, req_to_token_pool, allocator, page_size):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.disable = False
        self.metrics_collector = None

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("session-owned request reached the raw cache path")

    def dec_lock_ref(self, node, *args, **kwargs):
        raise AssertionError("queued abort must not unlock the session's tree node")

    def supports_mamba(self):
        return True

    def sanity_check(self):
        return None


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, index):
        self.freed.append(index)


def _make_streaming_cache(num_slots=8, max_context=64, page_size=1):
    """Real `StreamingSession` over a fake inner cache and pools."""
    mamba_allocator = _RecordingAllocator()
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((num_slots, max_context), dtype=torch.int32),
        mamba_allocator=mamba_allocator,
        free_slots=[],
    )
    inner = _RawCachePathGuard(req_to_token_pool, _RecordingAllocator(), page_size)
    return StreamingSession(inner), mamba_allocator


class TestQueuedSessionAbortMambaOwnership(CustomTestCase):
    """A restored turn borrows its Mamba/req-pool state from the SessionSlot.

    These drive the real `StreamingSession`, so the pre-fix path really did
    reach `Session.finish_req`.
    """

    def setUp(self):
        self._parallel_override = get_parallel().override(tp_rank=0)
        self._parallel_override.__enter__()
        self.addCleanup(self._parallel_override.__exit__, None, None, None)

    def _build(self):
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        streaming_cache, mamba_allocator = _make_streaming_cache()
        scheduler = _make_scheduler(streaming_cache, waiting_queue=[])

        first = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        _finish_turn(session, first, [91, 92])
        scheduler.waiting_queue.clear()

        # The slot the finished turn left behind, holding the committed prefix.
        slot = SessionSlot(
            req_pool_idx=3,
            kv_committed_len=5,
            kv=SimpleNamespace(kv_allocated_len=5, swa_evicted_seqlen=0),
            mamba_pool_idx=torch.tensor([2], dtype=torch.int64),
        )
        streaming_cache.slots["s0"] = slot
        return session, scheduler, streaming_cache, slot, mamba_allocator

    def _restored_turn(self, session, scheduler, slot, rid, input_ids):
        """Queue a turn holding slot-borrowed state, as `restore_to_req` leaves it."""
        req = _queue_turn(session, scheduler, rid, input_ids)
        slot.restore_to_req(req)
        return req

    def test_slot_owned_state_is_not_released(self):
        session, scheduler, streaming_cache, slot, mamba_allocator = self._build()
        self._restored_turn(session, scheduler, slot, "r2", [44, 55, 66])

        scheduler.abort_request(AbortReq(rid="r2"))

        # The slot keeps every resource it lent out.
        self.assertIn("s0", streaming_cache.slots)
        self.assertEqual(slot.req_pool_idx, 3)
        self.assertIsNotNone(slot.mamba_pool_idx)
        self.assertEqual(mamba_allocator.freed, [])
        self.assertEqual(streaming_cache.inner.token_to_kv_pool_allocator.freed, [])
        self.assertEqual(streaming_cache.req_to_token_pool.free_slots, [])
        self.assertFalse(session._inflight)

    def test_mamba_turn_history_is_not_committed(self):
        session, scheduler, _cache, slot, _alloc = self._build()
        committed_len = session.committed_origin_len

        self._restored_turn(session, scheduler, slot, "r2", [44, 55, 66])
        scheduler.abort_request(AbortReq(rid="r2"))

        self.assertEqual(session.committed_origin_len, committed_len)
        self.assertEqual(list(session.req_nodes), ["r1"])

        follow_up = session.create_req(
            _make_recv_req("r3", [77, 88], "s0"), None, VOCAB_SIZE
        )
        self.assertEqual(list(follow_up.origin_input_ids), [11, 12, 13, 91, 92, 77, 88])

    def test_waiting_timeout_preserves_slot_owned_state(self):
        session, scheduler, streaming_cache, slot, mamba_allocator = self._build()
        timed_out = self._restored_turn(session, scheduler, slot, "r2", [44, 55, 66])
        timed_out.time_stats = SimpleNamespace(wait_queue_entry_time=1e-6)

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get",
            return_value=1,
        ):
            scheduler._abort_on_waiting_timeout()

        self.assertIn("s0", streaming_cache.slots)
        self.assertEqual(mamba_allocator.freed, [])
        self.assertFalse(session._inflight)
        self.assertEqual(list(session.req_nodes), ["r1"])

    def test_waiting_timeout_releases_first_turn_mamba_state(self):
        """The timeout path must free request-owned Mamba state like abort does.

        A streaming session's first turn has no slot yet, so a Mamba index the
        cache allocated during match_prefix belongs to the request. The
        admission-rejection path deliberately keeps it for session requests, so
        a request can sit in the waiting queue holding it until it times out.
        """
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        req = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        req.req_pool_idx = None
        req.kv = None
        req.mamba_pool_idx = torch.tensor([2], dtype=torch.int64)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=1e-6)

        released = []
        tree_cache.supports_mamba = lambda: True
        tree_cache.req_to_token_pool = SimpleNamespace(
            mamba_allocator=SimpleNamespace(free=released.append)
        )

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get",
            return_value=1,
        ):
            scheduler._abort_on_waiting_timeout()

        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual([t.tolist() for t in released], [[[2]]])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(tree_cache.cache_finished_req_calls, [])
        self.assertFalse(session._inflight)

    def test_first_turn_exclusive_mamba_state_is_released(self):
        """Without a slot the Mamba state is the request's own and must be freed."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        req = _queue_turn(session, scheduler, "r1", [11, 12, 13])
        # Early Mamba allocation happens before any req-pool slot is taken.
        req.req_pool_idx = None
        req.kv = None
        req.mamba_pool_idx = torch.tensor([2], dtype=torch.int64)

        released = []
        tree_cache.supports_mamba = lambda: True
        tree_cache.req_to_token_pool = SimpleNamespace(
            mamba_allocator=SimpleNamespace(free=released.append)
        )

        scheduler.abort_request(AbortReq(rid="r1"))

        self.assertEqual([t.tolist() for t in released], [[[2]]])
        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(tree_cache.cache_finished_req_calls, [])
        self.assertFalse(session._inflight)


class TestQueuedSessionAbortIdempotence(CustomTestCase):
    def setUp(self):
        self._parallel_override = get_parallel().override(tp_rank=0)
        self._parallel_override.__enter__()
        self.addCleanup(self._parallel_override.__exit__, None, None, None)

    def test_repeated_abort_does_not_clear_a_newer_turn(self):
        """A second abort for a removed rid must not disturb the next turn."""
        session = Session(capacity_of_str_len=0, session_id="s0", streaming=True)
        tree_cache = _RecordingTreeCache()
        scheduler = _make_scheduler(tree_cache, waiting_queue=[])

        _queue_turn(session, scheduler, "r1", [11, 12, 13])
        scheduler.abort_request(AbortReq(rid="r1"))
        self.assertFalse(session._inflight)

        # The next turn is admitted and is genuinely active.
        newer = _queue_turn(session, scheduler, "r2", [21, 22])
        self.assertTrue(session._inflight)

        # Re-aborting the already removed request must be inert.
        scheduler.abort_request(AbortReq(rid="r1"))
        self.assertTrue(session._inflight)
        self.assertIn(newer, scheduler.waiting_queue)


if __name__ == "__main__":
    unittest.main()
