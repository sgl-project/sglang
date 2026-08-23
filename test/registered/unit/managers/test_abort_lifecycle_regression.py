"""
CPU-only regression tests for the request-abort lifecycle (PR #35936 review).

Encodes the two invariants requested by the reviewer:

  1. Disconnect cleanup: AbortReq must be dispatched BEFORE the
     rid_to_state entry is popped (zombie-request fix).
  2. Public abort API: an unknown/stale rid must NOT be dispatched to the
     scheduler when tokenizer_worker_num == 1, because scheduler abort
     matching is prefix-based (`req.rid.startswith(recv_req.rid)`) and a
     prefix-like unknown rid could cancel unrelated live requests.

Also characterizes:
  - GenerateReqInput._normalize_rid batch rid scheme ("batch" -> batch_0..N)
  - scheduler prefix-matching semantics (documented via the exact predicate)
  - exact known rid abort still dispatches
  - duplicate/late aborts are tolerated

Branch-sensitive behavior matrix (tokenizer_worker_num == 1):
  - test_unknown_rid_not_dispatched_* / test_exact_finished_rid_not_dispatched /
    test_duplicate_abort_after_finish_no_dispatch /
    test_cancelled_error_dispatches_abort_before_pop
      * base (guard + pop-only cleanup): invariant 1 FAILS (zombie), invariant 2 holds
      * PR head (no guard + abort-and-discard): invariant 1 holds, invariant 2 FAILS
      * candidate fix (guard restored + abort-and-discard): both hold

Run against a source tree with:
  PYTHONPATH=<tree>/python python -m pytest test_abort_lifecycle_regression.py -q
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.runtime_context import get_context  # noqa: E402

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _EventLog:
    def __init__(self):
        self.events = []

    def record(self, kind, detail):
        self.events.append((kind, detail))

    def kinds(self, kind):
        return [e for e in self.events if e[0] == kind]


class _RecordingSock:
    """Stands in for the zmq socket; records objects handed to sock_send."""

    def __init__(self, log: _EventLog):
        self.log = log

    def send_pyobj(self, obj, *a, **kw):
        self.log.record("sent", (type(obj).__name__, getattr(obj, "rid", None)))


class _TracingDict(dict):
    """dict that records pop/del of keys into an event log."""

    def __init__(self, log: _EventLog, *a, **kw):
        super().__init__(*a, **kw)
        self.log = log

    def pop(self, key, *args):
        self.log.record("pop", key)
        return super().pop(key, *args)

    def __delitem__(self, key):
        self.log.record("del", key)
        super().__delitem__(key)


def _make_tm(case, tokenizer_worker_num: int = 1) -> TokenizerManager:
    """TokenizerManager with mocked dependencies, bypassing __init__.

    abort_request() reads get_serving().tokenizer_worker_num from the config
    bags, so a context with the desired value must be published per case.
    """
    override = get_context().override_server_args(
        speculative_algorithm=None, tokenizer_worker_num=tokenizer_worker_num
    )
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.enable_trace = False
    tm.server_args.enable_metrics = False
    tm.server_args.enable_lora = False
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.weight_version = "1"
    tm.server_args.crash_dump_folder = ""
    tm.server_args.dp_size = 1
    tm.server_args.language_only = False
    tm.server_args.enable_strict_thinking = False
    tm.tokenizer_ipc_name = None
    tm.disaggregation_mode = "none"
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.allow_auto_truncate = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    return tm


def _make_tm_with_log(case, tokenizer_worker_num: int = 1):
    log = _EventLog()
    tm = _make_tm(case, tokenizer_worker_num)
    tm.rid_to_state = _TracingDict(log)
    tm.log = log
    tm.send_to_scheduler = _RecordingSock(log)
    return tm


def _make_state(rid: str) -> ReqState:
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _aborts_sent(tm) -> list:
    """Rids of AbortReq objects handed to sock_send so far."""
    return [
        rid
        for _, (obj_name, rid) in tm.log.kinds("sent")
        if obj_name == "AbortReq"
    ]


# ---------------------------------------------------------------------------
# Invariant 2: public abort API must not dispatch unknown/stale rids.
# ---------------------------------------------------------------------------


class TestPublicAbortGuard(CustomTestCase):
    def test_unknown_rid_not_dispatched_single_worker(self):
        """Reviewer reproducer: empty rid_to_state + abort_request('batch')."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        self.assertEqual(len(tm.rid_to_state), 0)
        tm.abort_request(rid="batch")
        self.assertEqual(
            _aborts_sent(tm),
            [],
            f"unknown public rid must be dropped by the guard; sent={_aborts_sent(tm)}",
        )

    def test_prefix_rid_not_dispatched_while_similar_live(self):
        """abort_request('batch_1') with only 'batch_10' live must not dispatch."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        tm.rid_to_state["batch_10"] = _make_state("batch_10")
        tm.abort_request(rid="batch_1")
        self.assertEqual(_aborts_sent(tm), [])

    def test_exact_known_rid_dispatched(self):
        """Exact, still-pending rid must reach the scheduler."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        tm.rid_to_state["req-abc"] = _make_state("req-abc")
        tm.abort_request(rid="req-abc")
        self.assertEqual(_aborts_sent(tm), ["req-abc"])

    def test_exact_finished_rid_not_dispatched(self):
        """Stale-but-exact rid (already finished & popped) stays a no-op."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        tm.rid_to_state["req-gone"] = _make_state("req-gone")
        del tm.rid_to_state["req-gone"]  # normal completion removed it
        tm.abort_request(rid="req-gone")
        self.assertEqual(_aborts_sent(tm), [])

    def test_duplicate_abort_after_finish_no_second_dispatch(self):
        """Duplicate abort for an already-finished rid must not re-dispatch."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        tm.rid_to_state["dup"] = _make_state("dup")
        tm.abort_request(rid="dup")  # first: dispatched
        self.assertEqual(_aborts_sent(tm), ["dup"])
        tm.rid_to_state.pop("dup")  # cleanup happened afterwards
        tm.abort_request(rid="dup")  # duplicate: suppressed
        self.assertEqual(_aborts_sent(tm), ["dup"])

    def test_abort_all_still_dispatches(self):
        """abort_all=True bypasses the guard by design."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=1)
        tm.abort_request(abort_all=True)
        self.assertEqual(len(_aborts_sent(tm)), 1)

    def test_multi_worker_still_dispatches_unknown(self):
        """tokenizer_worker_num > 1 has no local guard (pre-existing semantics)."""
        tm = _make_tm_with_log(self, tokenizer_worker_num=4)
        tm.abort_request(rid="no-such-rid")
        self.assertEqual(len(_aborts_sent(tm)), 1)


# ---------------------------------------------------------------------------
# Prefix scheme + scheduler matching characterization.
# ---------------------------------------------------------------------------


class TestPrefixScheme(CustomTestCase):
    def test_normalize_rid_generates_batch_prefix_scheme(self):
        """User-supplied rid='batch' with batch_size=11 yields batch_0..batch_10."""
        obj = GenerateReqInput(
            text=["a"] * 11,
            sampling_params=[{}] * 11,
            rid="batch",
        )
        obj.batch_size = 11
        obj.parallel_sample_num = 1
        obj._normalize_rid(obj.batch_size)
        self.assertEqual(obj.rid[0], "batch_0")
        self.assertEqual(obj.rid[1], "batch_1")
        self.assertEqual(obj.rid[10], "batch_10")

    def test_scheduler_predicate_prefix_fanout(self):
        """Exact scheduler predicate fan-outs over live rids.

        Mirrors `recv_req.abort_all or req.rid.startswith(recv_req.rid)`
        from Scheduler.abort_request (8 call sites).
        """

        def matches(live_rid: str, abort_rid: str, abort_all: bool = False) -> bool:
            return abort_all or live_rid.startswith(abort_rid)

        live = ["batch_0", "batch_1", "batch_10"]
        aborted = [r for r in live if matches(r, "batch")]
        self.assertEqual(aborted, live, "'batch' fans out to all three")

        aborted = [r for r in live if matches(r, "batch_1")]
        self.assertEqual(
            aborted, ["batch_1", "batch_10"], "'batch_1' also kills 'batch_10'"
        )


# ---------------------------------------------------------------------------
# Invariant 1: disconnect cleanup dispatches AbortReq before popping state.
# ---------------------------------------------------------------------------


class _DummyAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_generate_tm(case):
    tm = _make_tm_with_log(case)
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    return tm


def _make_obj(rid):
    obj = MagicMock(spec=GenerateReqInput)
    obj.routed_dp_rank = None
    obj.is_single = True
    obj.rid = rid
    obj.received_time = 0.0
    obj.external_trace_header = None
    obj.bootstrap_room = None
    obj.max_thinking_tokens = None
    obj.normalize_batch_and_arguments = Mock()
    return obj


class TestDisconnectCleanupOrdering(CustomTestCase):
    def test_cancelled_error_dispatches_abort_before_pop(self):
        """Client disconnect => CancelledError => AbortReq sent, then state popped."""
        tm = _make_generate_tm(self)
        rid = "streaming-rid"
        obj = _make_obj(rid)


        tm._tokenize_one_request = AsyncMock(return_value=Mock(input_ids=[1]))
        tm._send_one_request = Mock()

        async def wait_then_cancel(*args, **kwargs):
            yield {"chunk": 1}
            raise asyncio.CancelledError()

        tm._wait_one_response = wait_then_cancel

        async def drive():
            try:
                async for _ in tm.generate_request(obj):
                    pass
            except asyncio.CancelledError:
                pass

        asyncio.run(drive())

        abort_events = [
            i for i, e in enumerate(tm.log.events) if e == ("sent", ("AbortReq", rid))
        ]
        pop_events = [
            i for i, e in enumerate(tm.log.events) if e == ("pop", rid)
        ]
        self.assertEqual(
            len(abort_events),
            1,
            f"exactly one AbortReq expected during disconnect cleanup; "
            f"log={tm.log.events}",
        )
        self.assertEqual(len(pop_events), 1, f"log={tm.log.events}")
        self.assertLess(
            abort_events[0],
            pop_events[0],
            f"AbortReq must be dispatched before state pop; log={tm.log.events}",
        )
        self.assertNotIn(rid, tm.rid_to_state)

    def test_normal_completion_does_not_dispatch_extra_abort(self):
        """Normal completion removes state via batch-output path, no AbortReq."""
        tm = _make_generate_tm(self)
        rid = "normal-rid"
        obj = _make_obj(rid)

        tm._tokenize_one_request = AsyncMock(return_value=Mock(input_ids=[1]))
        tm._send_one_request = Mock()

        async def wait_finish(*args, **kwargs):
            yield {"chunk": 1}

        tm._wait_one_response = wait_finish

        async def drive():
            async for _ in tm.generate_request(obj):
                pass

        asyncio.run(drive())
        self.assertEqual(
            _aborts_sent(tm),
            [],
            f"no AbortReq expected on normal completion; log={tm.log.events}",
        )
        self.assertIn(rid, tm.rid_to_state)  # still waiting for scheduler output

    def test_delayed_abort_task_after_cleanup_is_noop(self):
        """create_abort_task firing after disconnect cleanup must not re-dispatch.

        Timeline: cleanup dispatched the AbortReq and popped state; 2s later
        the background task calls abort_request(rid). With the public guard
        intact this is a no-op => exactly one AbortReq in total.
        """
        tm = _make_generate_tm(self)
        rid = "delayed-rid"
        obj = _make_obj(rid)

        tm._tokenize_one_request = AsyncMock(return_value=Mock(input_ids=[1]))
        tm._send_one_request = Mock()

        async def wait_then_cancel(*args, **kwargs):
            yield {"chunk": 1}
            raise asyncio.CancelledError()

        tm._wait_one_response = wait_then_cancel

        async def drive():
            try:
                async for _ in tm.generate_request(obj):
                    pass
            except asyncio.CancelledError:
                pass

        asyncio.run(drive())

        # Delayed create_abort_task body (after its 2s sleep)
        tm.abort_request(obj.rid)

        self.assertEqual(
            _aborts_sent(tm),
            [rid],
            f"cleanup abort expected exactly once; delayed task must be a "
            f"no-op; log={tm.log.events}",
        )


class TestBatchDisconnectOrdering(CustomTestCase):
    def test_batch_cleanup_dispatches_before_pop_per_rid(self):
        """For a batch, each sub-rid gets AbortReq before its own pop."""
        log = _EventLog()
        tm = _make_tm(self)
        tm.rid_to_state = _TracingDict(log)
        tm.log = log
        tm.send_to_scheduler = _RecordingSock(log)

        rids = ["b_0", "b_1", "b_2"]
        for r in rids:
            tm.rid_to_state[r] = _make_state(r)
        obj = Mock(spec=GenerateReqInput)
        obj.is_single = False
        obj.rid = list(rids)

        tm._abort_and_discard_pending_req_states(obj)

        for r in rids:
            sent_idx = log.events.index(("sent", ("AbortReq", r)))
            pop_idx = log.events.index(("pop", r))
            self.assertLess(
                sent_idx,
                pop_idx,
                f"AbortReq for {r} must precede its pop; log={log.events}",
            )
            self.assertNotIn(r, tm.rid_to_state)


class TestCleanupDispatchFailure(CustomTestCase):
    """A dispatch failure during cleanup must not leak state or mask the
    original cancellation exception."""

    def test_dispatch_failure_still_cleans_all_batch_rids(self):
        """Middle dispatch raises => loop continues, every state still popped."""
        log = _EventLog()
        tm = _make_tm(self)
        tm.rid_to_state = _TracingDict(log)
        tm.log = log

        rids = ["f_0", "f_1", "f_2"]
        for r in rids:
            tm.rid_to_state[r] = _make_state(r)

        class FailingSock:
            def send_pyobj(self, obj, *a, **kw):
                rid = getattr(obj, "rid", None)
                if rid == "f_1":
                    raise RuntimeError("scheduler socket broken")
                log.record("sent", ("AbortReq", rid))

        tm.send_to_scheduler = FailingSock()

        obj = Mock(spec=GenerateReqInput)
        obj.is_single = False
        obj.rid = list(rids)

        tm._abort_and_discard_pending_req_states(obj)  # must not raise

        # Loop continued past the failing rid: f_0 and f_2 were still aborted.
        self.assertEqual(
            [e for e in log.kinds("sent")],
            [("sent", ("AbortReq", "f_0")), ("sent", ("AbortReq", "f_2"))],
            f"log={log.events}",
        )
        # No state leaked, including the rid whose dispatch failed.
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)
            self.assertIn(("pop", r), log.events)

    def test_dispatch_failure_does_not_mask_cancelled_error(self):
        """generate_request must re-raise CancelledError, not the dispatch error."""
        tm = _make_generate_tm(self)
        rid = "broken-dispatch-rid"
        obj = _make_obj(rid)

        class BrokenSock:
            def send_pyobj(self, o, *a, **k):
                raise RuntimeError("scheduler socket broken")

        tm.send_to_scheduler = BrokenSock()
        tm._tokenize_one_request = AsyncMock(return_value=Mock(input_ids=[1]))
        tm._send_one_request = Mock()

        async def wait_then_cancel(*args, **kwargs):
            yield {"chunk": 1}
            raise asyncio.CancelledError()

        tm._wait_one_response = wait_then_cancel

        raised = None

        async def drive():
            nonlocal raised
            try:
                async for _ in tm.generate_request(obj):
                    pass
            except BaseException as e:  # assert exception identity is preserved
                raised = e

        asyncio.run(drive())

        self.assertIsInstance(
            raised,
            asyncio.CancelledError,
            f"original CancelledError must not be masked by a dispatch "
            f"failure; got {raised!r}",
        )
        self.assertNotIn(rid, tm.rid_to_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
