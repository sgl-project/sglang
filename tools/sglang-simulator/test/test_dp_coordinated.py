"""Real Coordinator/loopback control with a bounded two-rank transport double.

The queue/load/native-request doubles test protocol ownership, not native Gloo,
GPU execution, source attestation, qualified costs or public ADP2 activation.
"""

import copy
import json
import queue
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from types import SimpleNamespace as NS

import msgspec
import pytest
import zmq
from sglang_simulator.simulation.coordinator import (
    Cancel,
    ProtocolError,
    Ready,
    Submit,
    decode_grant,
    encode_message,
)
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang import dp_coordinated as dp
from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE,
    DSparkGroup,
    DSparkPhaseCost,
    DSparkProgress,
    DSparkStep,
    DSparkStepResult,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.token_events import TokenEventRecorder
from test_dspark_step import request


class Load(msgspec.Struct):
    dp_rank: int
    timestamp: float = 99999.0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_total_tokens: int = 0


class TokenizedGenerateReqInput:
    def __init__(self, rid, routed_dp_rank=None):
        self.rid = rid
        self.routed_dp_rank = routed_dp_rank
        self.input_ids = [1, 2]
        self.sampling_params = NS(max_new_tokens=1)


class AbortReq:
    def __init__(self, *, rid):
        self.rid, self.abort_all = rid, False


class ProfileReq:
    def __init__(self, action="START_PROFILE"):
        self.req_type = NS(name=action)


class LocalRequest:
    def __init__(self, rid):
        self.rid, self.output_ids, self.done = rid, [], False
        self.origin_input_ids, self.cached_tokens = [1, 2], 0

    def finished(self):
        return self.done


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return "tcp://127.0.0.1:" + str(sock.getsockname()[1])


def drain(values):
    result = []
    while True:
        try:
            result.append(values.get_nowait())
        except queue.Empty:
            return result


@pytest.fixture
def owners(monkeypatch):
    for name, fields in (
        ("sglang.srt.managers.load_snapshot", {"LoadSnapshot": Load}),
        ("sglang.srt.managers.io_struct", {"AbortReq": AbortReq}),
        ("sglang.srt.managers.schedule_batch", {"FINISH_ABORT": lambda: NS()}),
    ):
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dp, "_native_group", lambda group: None)
    StateManager.reset()
    request_stats_manager.reset()
    condition, exchanges = threading.Condition(), {}

    class Group:
        def __init__(self, rank):
            self.rank_in_group, self.round = rank, 0

        def all_gather_object(self, value):
            with condition:
                rows = exchanges.setdefault(self.round, {})
                self.round += 1
                rows[self.rank_in_group] = copy.deepcopy(value)
                condition.notify_all()
                assert condition.wait_for(lambda: len(rows) == 2, timeout=3)
                return copy.deepcopy([rows[0], rows[1]])

    incoming, queues, events = queue.Queue(), [queue.Queue(), queue.Queue()], []
    controller = NS(
        soft_watchdog=NS(feed=lambda: None, is_active=lambda: True),
        _last_refresh_time=0.0,
    )

    def dispatch(req):
        events.append(("dispatch", type(req).__name__, getattr(req, "rid", None)))
        if type(req) is TokenizedGenerateReqInput:
            rank = req.routed_dp_rank if req.routed_dp_rank is not None else 0
            queues[rank].put(req)
        else:
            for values in queues:
                values.put(req)

    controller._request_dispatcher = dispatch
    external, internal = endpoint(), endpoint()
    owner = dp.DPCoordinatedController(
        controller=controller,
        endpoint=external,
        peer_endpoint=internal,
        receive=lambda: drain(incoming),
        abort_output=lambda req: events.append(("abort", req.rid)),
    )
    owner.timeout_s = 2
    local = []
    for rank in range(2):
        scheduler = NS(
            result_queue=[],
            waiting_queue=[],
            chunked_req=None,
            _pending_chunked_abort_req=None,
            inflight=[],
            pending=False,
        )
        scheduler.collect_inflight_reqs = lambda scheduler=scheduler: scheduler.inflight
        scheduler.process_pending_chunked_abort = lambda: None
        scheduler.abort_request = lambda req: None
        scheduler.load_inquirer = NS(get_loads=lambda rank=rank: Load(rank))
        bridge = dp.DPCoordinatedScheduler(
            scheduler=scheduler,
            tp_group=Group(rank),
            endpoint=internal,
            pending_work=lambda scheduler=scheduler: scheduler.pending,
            timeout_s=3,
        )
        local.append(NS(scheduler=scheduler, bridge=bridge, rank=rank))
    failures = []

    def serve():
        try:
            owner.serve()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    control_context = zmq.Context()
    control = control_context.socket(zmq.REQ)
    control.setsockopt(zmq.LINGER, 0)
    control.setsockopt(zmq.RCVTIMEO, 3000)
    control.connect(external)
    instance = NS(
        owner=owner,
        local=local,
        incoming=incoming,
        queues=queues,
        control=control,
        events=events,
        failures=failures,
        external=external,
    )
    try:
        yield instance
    finally:
        owner._closed = True
        thread.join(timeout=4)
        assert not thread.is_alive()
        for run in local:
            run.bridge.close()
        # close() must still release its resources after stopping the loop.
        owner._closed = False
        owner.close()
        control.close(linger=0)
        control_context.term()
        request_stats_manager.reset()
        StateManager.reset()


def both(owners, call):
    def invoke(run):
        try:
            return call(run)
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, run) for run in owners.local]
        return [future.result(timeout=5) for future in futures]


def receive(owners):
    return both(
        owners, lambda run: run.bridge.receive(lambda: drain(owners.queues[run.rank]))
    )


def test_bootstrap_reports_reply_before_first_ready(owners):
    # A preparing client has no Ready yet; the healthy internal exchange still
    # has a finite deadline, independent of the external admission timeout.
    owners.local[0].bridge._socket.setsockopt(zmq.RCVTIMEO, 500)
    try:
        for _ in range(3):
            assert receive(owners) == [[], []]
            assert both(owners, lambda run: run.bridge.on_idle()) == [None, None]
            assert owners.owner.clock.run_id is None
            assert owners.owner.clock.now_ns == 0
            assert owners.events == []
            assert all(run.bridge.clock.now_ns == 0 for run in owners.local)
        owners.control.send(encode_message(Ready("run", 0, [Submit("first")])))
        request = TokenizedGenerateReqInput("first", 1)
        owners.incoming.put(request)
        assert receive(owners) == [[], [request]]
        assert owners.owner.clock.run_id == "run"
        assert owners.owner._delivered == {"first": 1}
        assert not owners.failures
    finally:
        if owners.owner.clock.run_id is None:
            # Release the old blocking implementation when this regression is
            # run against it, so a failing assertion cannot strand its thread.
            owners.control.send(encode_message(Ready("cleanup", 0, [], finished=True)))


def test_first_ready_missing_payload_retains_active_timeout(owners):
    owners.owner.timeout_s = 0.02
    owners.control.send(encode_message(Ready("run", 0, [Submit("missing")])))
    results = receive(owners)
    assert all(isinstance(result, ProtocolError) for result in results)
    assert all("driver/payload timeout" in str(result) for result in results)
    assert owners.owner.clock.run_id == "run"
    assert owners.owner.clock.now_ns == 0
    assert decode_grant(owners.control.recv()).status == "ERROR"
    assert all("driver/payload timeout" in run.bridge._failure for run in owners.local)


@pytest.mark.parametrize(
    "kind", ["GetInternalStateReq", "ConfigureLoggingReq", "FreezeGCReq", "ShutdownReq"]
)
@pytest.mark.parametrize("second_rank_poll", [1, 3])
def test_bootstrap_admin_applies_before_first_ready(
    owners, monkeypatch, kind, second_rank_poll
):
    admin = type(kind, (), {})()
    settle = owners.owner._settle
    bootstrap_settles = []

    def bounded_settle():
        if owners.owner.clock.run_id is None:
            # A second bootstrap wait would deadlock the server-info reply and
            # the client which sends Ready only after receiving that reply.
            assert not bootstrap_settles, "bootstrap ACK waited for Ready"
            bootstrap_settles.append(True)
        return settle()

    monkeypatch.setattr(owners.owner, "_settle", bounded_settle)
    owners.incoming.put(admin)
    polls = [0, 0]

    def native_receive(run):
        def values():
            polls[run.rank] += 1
            if run.rank == 1 and polls[1] < second_rank_poll:
                return []
            return drain(owners.queues[run.rank])

        return run.bridge.receive(values)

    assert both(owners, native_receive) == [[admin], [admin]]
    assert polls[1] >= second_rank_poll
    assert bootstrap_settles == [True]
    assert owners.owner._inflight is None
    assert owners.owner.clock.run_id is None
    assert owners.owner.clock.now_ns == 0
    assert owners.events == [("dispatch", kind, None)]

    monkeypatch.setattr(owners.owner, "_settle", settle)
    # Native administration has returned on both ranks; ordinary coordinated
    # admission still needs the unchanged first Ready and its matching payload.
    owners.control.send(encode_message(Ready("run", 0, [Submit("after-admin")])))
    request = TokenizedGenerateReqInput("after-admin", 1)
    owners.incoming.put(request)
    assert receive(owners) == [[], [request]]
    assert owners.owner._delivered == {"after-admin": 1}
    assert owners.owner.clock.run_id == "run"
    assert not owners.failures


def test_native_ingress_is_acknowledged_before_local_delivery(owners):
    owners.control.send(
        encode_message(Ready("run", 0, [Submit("a"), Submit("b")], next_wakeup_ns=100))
    )
    a, b = TokenizedGenerateReqInput("a", 1), TokenizedGenerateReqInput("b", 0)
    owners.incoming.put(a)
    owners.incoming.put(b)
    delivered = receive(owners)
    assert delivered == [[b], [a]]
    assert owners.owner._delivered == {"a": 1, "b": 0}
    assert owners.owner._inflight is None
    assert owners.owner.clock.now_ns == 0
    assert owners.events == [
        ("dispatch", "TokenizedGenerateReqInput", "a"),
        ("dispatch", "TokenizedGenerateReqInput", "b"),
    ]
    assert all(
        run.bridge.clock.request_arrival_ns("a" if run.rank else "b") == 0
        for run in owners.local
    )


@pytest.mark.parametrize("idle_rank", [None, 0, 1])
def test_one_joint_duration_waits_for_both_native_callbacks_and_accounting(
    owners, monkeypatch, idle_rank, tmp_path
):
    ledger = tmp_path / "tokens.jsonl"
    owners.owner._token_events = TokenEventRecorder(ledger)
    active = [rank for rank in (0, 1) if rank != idle_rank]
    owners.control.send(
        encode_message(Ready("run", 0, [Submit(str(rank)) for rank in active]))
    )
    for rank in active:
        owners.incoming.put(TokenizedGenerateReqInput(str(rank), rank))
    assert not any(isinstance(row, BaseException) for row in receive(owners))
    groups = tuple(
        DSparkGroup(
            rank,
            "IDLE" if rank == idle_rank else "TARGET_VERIFY",
            () if rank == idle_rank else (request(str(rank)),),
            '{"fixture":true}',
        )
        for rank in (0, 1)
    )
    step = DSparkStep(0, DSparkTopology(2, 2), groups, '{"fixture":true}')
    timing = DSparkStepTiming(
        0, (DSparkPhaseCost("whole_worker", 0.01, "protocol fixture; not measured"),)
    )
    progress = DSparkProgress(
        0, tuple((req.rid, 1) for req in step.requests), "protocol fixture"
    )
    processed = []

    def execute(run):
        reqs = [LocalRequest(req.rid) for req in step.groups[run.rank].requests]
        batch = NS(reqs=reqs.copy())
        run.scheduler.inflight = reqs
        run.scheduler.pending = True
        result = NS(auxiliary_host_output=None)
        setattr(
            result, STEP_ATTRIBUTE, DSparkStepResult(step, timing, progress, run.rank)
        )

        def process():
            assert owners.owner.clock.now_ns == 10_000_000
            assert owners.owner._joint_result is not None
            assert not owners.owner.clock.has_terminals
            for req in reqs:
                req.output_ids.append(35)
                req.done = True
            result.auxiliary_host_output.consume(
                NS(reqs=reqs), [NS(output_index=0, token_ids=(35,)) for req in reqs]
            )
            batch.reqs.clear()  # Ownership must survive a callback filtering its batch.
            assert all(req._simulator_timing["commit_ns"] == 10_000_000 for req in reqs)
            run.scheduler.pending = False
            processed.append(run.rank)  # Native callback + accounting, before ACK.

        return run.bridge.process_batch_result(
            batch, result, process=process, duration=0.01
        )

    original_grant = owners.owner._grant

    def grant(**kwargs):
        assert sorted(processed) == [0, 1]
        assert not any(run.scheduler.pending for run in owners.local)
        return original_grant(**kwargs)

    monkeypatch.setattr(owners.owner, "_grant", grant)
    assert both(owners, execute) == [None, None]
    grant = decode_grant(owners.control.recv())
    assert grant.now_ns == 10_000_000
    assert {row.rid for row in grant.terminals} == {str(rank) for rank in active}
    assert all(run.bridge.clock.now_ns == 10_000_000 for run in owners.local)
    owners.control.send(encode_message(Ready("run", 1, [], finished=True)))
    assert receive(owners) == [[], []]
    assert decode_grant(owners.control.recv()).status == "FINISHED"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len([row for row in records if row["event"] == "output_commit"]) == 1
    assert records[-1]["total_output_tokens"] == len(active)


def test_one_sided_report_failure_reaches_both_ranks_without_advancing(owners):
    def execute(run):
        def make():
            if run.rank:
                raise ValueError("local source failure")
            return run.bridge._report("idle")

        return run.bridge._exchange("idle", make)

    results = both(owners, execute)
    assert isinstance(results[0], ProtocolError)
    assert isinstance(results[1], ValueError)
    assert all(run.bridge._failure for run in owners.local)
    assert owners.owner.clock.now_ns == 0


def test_timer_cancel_of_admitted_waiting_request_reaches_native_before_next_selection(
    owners,
):
    """collect_inflight_reqs excludes waiting B while A owns an active result."""

    def driver():
        context = zmq.Context()
        control = context.socket(zmq.REQ)
        control.setsockopt(zmq.LINGER, 0)
        control.setsockopt(zmq.RCVTIMEO, 3000)
        control.connect(owners.external)
        try:
            control.send(
                encode_message(
                    Ready(
                        "run", 0, [Submit("a"), Submit("b")], next_wakeup_ns=5_000_000
                    )
                )
            )
            timer = decode_grant(control.recv())
            assert timer.now_ns == 5_000_000 and not timer.terminals
            control.send(encode_message(Ready("run", 1, [Cancel("b")], finished=True)))
            cancelled = decode_grant(control.recv())
            assert [(row.rid, row.kind) for row in cancelled.terminals] == [
                ("b", "CANCELLED")
            ]
            control.send(encode_message(Ready("run", 2, [], finished=True)))
            completed = decode_grant(control.recv())
            assert [(row.rid, row.kind) for row in completed.terminals] == [
                ("a", "COMPLETED")
            ]
            control.send(encode_message(Ready("run", 3, [], finished=True)))
            return decode_grant(control.recv())
        finally:
            control.close(linger=0)
            context.term()

    owners.incoming.put(TokenizedGenerateReqInput("a", 0))
    owners.incoming.put(TokenizedGenerateReqInput("b", 0))
    a, b = LocalRequest("a"), LocalRequest("b")
    owners.local[0].scheduler.inflight = [a]
    owners.local[0].scheduler.waiting_queue = [b]
    aborted = []
    for run in owners.local:

        def abort(req, run=run):
            aborted.append((run.rank, req.rid))
            run.scheduler.waiting_queue[:] = [
                r for r in run.scheduler.waiting_queue if r.rid != req.rid
            ]

        run.scheduler.abort_request = abort
    step = DSparkStep(
        0,
        DSparkTopology(2, 2),
        (
            DSparkGroup(0, "TARGET_VERIFY", (request("a"),), '{"fixture":true}'),
            DSparkGroup(1, "IDLE", (), '{"fixture":true}'),
        ),
        '{"fixture":true}',
    )
    timing = DSparkStepTiming(
        0, (DSparkPhaseCost("whole_worker", 0.01, "protocol fixture; not measured"),)
    )
    progress = DSparkProgress(0, (("a", 1),), "protocol fixture")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(driver)
        assert [len(row) for row in receive(owners)] == [2, 0]

        def execute(run):
            reqs = [a] if run.rank == 0 else []
            result = NS(auxiliary_host_output=None)
            setattr(
                result,
                STEP_ATTRIBUTE,
                DSparkStepResult(step, timing, progress, run.rank),
            )

            def process():
                assert "b" in owners.owner._cancel_cleanup
                assert not owners.local[0].scheduler.waiting_queue
                for req in reqs:
                    req.done = True

            return run.bridge.process_batch_result(
                NS(reqs=reqs), result, process=process, duration=0.01
            )

        assert both(owners, execute) == [None, None]
        assert sorted(aborted) == [(0, "b"), (1, "b")]
        assert not owners.owner._cancel_cleanup
        assert receive(owners) == [[], []]
        assert future.result(timeout=4).status == "FINISHED"


@pytest.mark.parametrize("failure", ["receive", "timeout", "load_rank"])
def test_ingress_failure_rendezvous_does_not_leave_peer_waiting(owners, failure):
    owners.control.send(
        encode_message(Ready("run", 0, [Submit("a")], next_wakeup_ns=100))
    )
    owners.incoming.put(TokenizedGenerateReqInput("a", 1))

    def execute(run):
        if failure == "timeout":
            run.bridge.timeout_s = 0
        if failure == "load_rank" and run.rank == 1:
            run.scheduler.load_inquirer.get_loads = lambda: Load(0)

        def native_receive():
            if failure == "receive" and run.rank:
                raise OSError("native socket failed")
            return drain(owners.queues[run.rank])

        return run.bridge.receive(native_receive)

    assert all(isinstance(row, BaseException) for row in both(owners, execute))
    assert owners.owner.clock.now_ns == 0
    assert not owners.owner._finished


def test_cadence_idle_does_not_advance_tool_timer_while_native_work_remains(owners):
    owners.control.send(
        encode_message(Ready("run", 0, [Submit("a")], next_wakeup_ns=100))
    )
    owners.incoming.put(TokenizedGenerateReqInput("a", 0))
    received = receive(owners)
    assert [len(row) for row in received] == [1, 0]
    for run in owners.local:
        run.bridge._cadence_only_pass = True
    owners.local[0].scheduler.pending = True
    assert both(owners, lambda run: run.bridge.on_idle()) == [None, None]
    assert owners.owner.clock.now_ns == 0
    assert not owners.control.poll(0)


def test_finished_requires_both_native_owners_drained(owners):
    owners.control.send(encode_message(Ready("run", 0, [], finished=True)))
    owners.local[1].scheduler.pending = True
    assert receive(owners) == [[], []]
    assert not owners.control.poll(0)
    owners.local[1].scheduler.pending = False
    assert receive(owners) == [[], []]
    assert decode_grant(owners.control.recv()).status == "FINISHED"
    assert all(run.bridge._finished for run in owners.local)


def test_profile_receipts_and_one_writer_wait_for_both_rank_resets(owners):
    req = ProfileReq()
    owners.incoming.put(req)
    assert receive(owners) == [[req], [req]]
    writes, resets = [], []

    def profile(run):
        def execute(payload, writer):
            assert [row.rid for row in payload["profile_stats"]] == ["a", "b"]
            if writer:
                writes.append(run.rank)
            resets.append(run.rank)
            return NS(success=True, message="reset")

        return run.bridge.profile_boundary(
            req,
            snapshot=lambda: {
                "profile_stats": [NS(rid="ab"[run.rank])],
                "iterations": [run.rank],
                "predictor_metrics": {"rank": run.rank},
            },
            execute=execute,
        )

    assert all(row.success for row in both(owners, profile))
    assert writes == [0]
    assert sorted(resets) == [0, 1]
    assert owners.owner._profile is None


@pytest.mark.parametrize("bad", ["pending", "snapshot", "writer"])
def test_profile_failure_is_paired_and_sticky(owners, bad):
    req = ProfileReq()
    owners.incoming.put(req)
    assert receive(owners) == [[req], [req]]
    owners.local[1].scheduler.pending = bad == "pending"

    def profile(run):
        def snapshot():
            if bad == "snapshot" and run.rank:
                raise ValueError("cost sink failed")
            return {"profile_stats": [], "iterations": [], "predictor_metrics": {}}

        def execute(payload, writer):
            return NS(success=not (bad == "writer" and writer), message="disk failure")

        return run.bridge.profile_boundary(req, snapshot=snapshot, execute=execute)

    assert all(isinstance(row, BaseException) for row in both(owners, profile))
    assert owners.owner._profile is not None


def test_native_budget_uses_virtual_cadence_and_preserves_native_counts(owners):
    updated = []
    owners.owner.controller.dp_budget = NS(
        update_budget=lambda rows: updated.append(rows)
    )
    loads = [
        Load(0, num_running_reqs=3, num_total_tokens=100),
        Load(1, num_waiting_reqs=2),
    ]
    owners.owner._loads = loads
    owners.owner.refresh_load_budget()
    assert not updated
    owners.owner.clock.accept_ready(Ready("run", 0, [], next_wakeup_ns=100_000_000))
    owners.owner.clock.advance(20_000_000)
    owners.owner.refresh_load_budget()
    assert [row.timestamp for row in updated[0]] == [0.02, 0.02]
    assert [
        msgspec.structs.replace(row, timestamp=99999.0) for row in updated[0]
    ] == loads
    owners.owner.refresh_load_budget()
    assert len(updated) == 1


@pytest.mark.parametrize("value", ["", "../path", "A" * 12, None])
def test_internal_namespace_requires_actual_native_instance_token(value):
    with pytest.raises(ProtocolError, match="instance_id"):
        dp.internal_endpoint(NS(instance_id=value))


def test_both_native_portargs_share_instance_despite_different_ingress():
    left = NS(instance_id="abcdef012345", scheduler_input_ipc_name="tcp://127.0.0.1:1")
    right = NS(instance_id="abcdef012345", scheduler_input_ipc_name="tcp://127.0.0.1:2")
    assert dp.internal_endpoint(left) == dp.internal_endpoint(right)
