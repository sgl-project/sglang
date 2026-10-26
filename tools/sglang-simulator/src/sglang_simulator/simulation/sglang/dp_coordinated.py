"""Internal native attention-DP clock bridge, beneath the closed public guards.

The DPC owns the existing external Coordinator. Scheduler copies are read-only
timing views; native routing, local result processing and reference ownership
remain with their original owners. No cost or progress is predicted here.
"""

import re
import tempfile
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import zmq
from sglang_simulator.hook import BaseHook
from sglang_simulator.hook.utils import get_obj_from_args
from sglang_simulator.simulation.coordinator import ProtocolError
from sglang_simulator.simulation.manager import Envs, StateManager
from sglang_simulator.simulation.sglang.coordinated import (
    CoordinatedScheduler,
    _validate_prefill_cadence_runtime,
)
from sglang_simulator.simulation.sglang.dspark import (
    DSparkTopology,
    dspark_result_from_native,
)
from sglang_simulator.simulation.sglang.dspark_native_peers import _native_group
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.timing_metadata import attach_timing_output


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def internal_endpoint(port_args):
    """All native attention-DP PortArgs retain this one server-instance token."""
    instance = port_args.instance_id
    _require(
        type(instance) is str and re.fullmatch(r"[a-f0-9]{12}", instance) is not None,
        "native DP clock requires the original server instance_id",
    )
    return "ipc://" + str(
        Path(tempfile.gettempdir()) / ("sglang-simulator-dp-" + instance)
    )


def native_dp_clock_enabled():
    """Select this internal bridge only for the actual two-singleton topology.

    This does not relax the public compatibility or factory checks.
    """
    if not Envs.coordinator_endpoint():
        return False
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not parallel.attn_dp_enabled:
        return False
    _require(
        parallel.tp_size == 2
        and parallel.attn_dp_size == 2
        and parallel.attn_tp_size == 1
        and parallel.pp_size == 1
        and parallel.nnodes == 1
        and parallel.attn_cp_size == 1,
        "internal native DP clock requires TP2/ADP2/attention-TP1/PP1 on one node",
    )
    return True


def _receipt(req):
    kind = type(req).__name__
    if kind == "TokenizedGenerateReqInput":
        return kind, req.rid
    if kind == "ProfileReq":
        return kind, req.req_type.name
    return kind, getattr(req, "rid", None)


@dataclass(frozen=True)
class _ClockView:
    """A read-only projection used by the existing timing-output consumer."""

    now_ns: int
    arrivals: dict
    cancelled: frozenset

    def request_arrival_ns(self, rid):
        return self.arrivals[rid]

    def is_cancelled(self, rid):
        return rid in self.cancelled


class DPCoordinatedController(CoordinatedScheduler):
    """One native DPC, external Ready/Grant socket and paired internal reports."""

    def __init__(self, *, controller, endpoint, peer_endpoint, receive, abort_output):
        self.controller = controller
        self._abort_output = abort_output
        self._reports = None
        self._generation = 0
        self._inflight = None
        self._inflight_routes = {}
        self._profile = None
        self._joint_result = None
        self._last_forward_id = -1
        self._report_round = 0
        self._delivered = {}
        self._loads = None
        super().__init__(scheduler=controller, endpoint=endpoint)
        self._raw_receive = receive
        self._peer_socket = self._context.socket(zmq.REP)
        self._peer_socket.setsockopt(zmq.LINGER, 0)
        self._peer_socket.bind(peer_endpoint)

    def close(self):
        if hasattr(self, "_peer_socket"):
            self._peer_socket.close(linger=0)
        super().close()

    def _has_native_work(self):
        return bool(
            self._inflight
            or self._admin
            or self._pending
            or self._cancel_cleanup
            or self._profile
            or self._joint_result
            or self._reports is None
            or any(row["pending"] for row in self._reports)
        )

    def _abort_payload(self, req):
        self._abort_output(req)

    def _cleanup_cancelled(self):
        # Active local native batches receive the same cancellation set in the
        # result-release view before their original processors run.
        if self._batch is None:
            from sglang.srt.managers.io_struct import AbortReq

            for rid in self._cancel_cleanup:
                # Use the same paired receipt barrier as ordinary admission.
                self._admin.append(AbortReq(rid=rid))
            self._cancel_cleanup.clear()

    def _dispatch_pending(self):
        values = self._pending + self._admin
        self._pending, self._admin = [], []
        if not values:
            return
        _require(
            self._inflight is None, "native admission generation is not acknowledged"
        )
        self._generation += 1
        expected = [_receipt(req) for req in values]
        profiles = [item for item in expected if item[0] == "ProfileReq"]
        _require(len(profiles) <= 1, "multiple profile transitions in one native poll")
        if profiles:
            _require(self._profile is None, "profile transition still owns both ranks")
            self._profile = profiles[0]
        self._inflight = expected
        self._inflight_routes = {
            req.rid: req.routed_dp_rank
            for req in values
            if type(req).__name__ == "TokenizedGenerateReqInput"
        }
        for req in values:
            self.controller._request_dispatcher(req)

    def _ack_ingress(self):
        if self._inflight is None:
            return
        _require(
            all(row["generation"] == self._generation for row in self._reports),
            "native ingress ACK has a stale dispatch generation",
        )
        expected = self._inflight
        ready = True
        for item in expected:
            ranks = [
                rank
                for rank, row in enumerate(self._reports)
                if item in row["received"]
            ]
            if item[0] == "TokenizedGenerateReqInput":
                _require(
                    len(ranks) <= 1,
                    "native routing delivered one RID to both partitions",
                )
                if ranks:
                    route = self._inflight_routes[item[1]]
                    _require(
                        route is None or route == ranks[0],
                        "native ingress differs from explicit routed_dp_rank",
                    )
                    old = self._delivered.setdefault(item[1], ranks[0])
                    _require(
                        old == ranks[0], "native ingress changed its request owner"
                    )
                else:
                    ready = False
            elif len(ranks) != 2:
                ready = False
        if ready:
            self._inflight = None

    def _view(self):
        # A registered request can be waiting rather than in any current/result
        # batch. Keep it in the release until both native abort owners ACK.
        rids = {
            rid for row in self._reports for rid in row["rids"]
        } | self._cancel_cleanup
        return {
            "now_ns": self.clock.now_ns,
            "arrivals": {rid: self.clock.request_arrival_ns(rid) for rid in rids},
            "cancelled": tuple(rid for rid in rids if self.clock.is_cancelled(rid)),
            "generation": self._generation,
            "again": self._inflight is not None,
            "finished": self._finished,
        }

    def handle(self, message):
        """Handle a paired native round; no worker independently advances time."""
        _require(
            type(message) is dict
            and set(message) == {"round", "phase", "reports"}
            and message["round"] == self._report_round
            and type(message["round"]) is int,
            "stale native DP clock report",
        )
        reports = message["reports"]
        _require(
            type(reports) is list
            and len(reports) == 2
            and all(
                type(row) is dict and row["rank"] == rank
                for rank, row in enumerate(reports)
            )
            and reports[0]["phase"] == reports[1]["phase"] == message["phase"],
            "native DP clock requires the same ordered report phase",
        )
        _require(
            all(row["error"] is None for row in reports), "native DP local event failed"
        )
        self._reports = reports
        phase = message["phase"]
        if phase == "receive":
            # Counts and routing belong to the native load inquirer/budget.
            # Only the refresh clock is replaced; no queue-size approximation.
            from sglang.srt.managers.load_snapshot import LoadSnapshot

            loads = [row["load"] for row in reports]
            _require(
                all(
                    type(load) is LoadSnapshot
                    and type(load.dp_rank) is int
                    and load.dp_rank == rank
                    for rank, load in enumerate(loads)
                ),
                "native load snapshots do not identify both actual DP ranks",
            )
            self._loads = loads
            bootstrap_ingress = self.clock.run_id is None and self._inflight is not None
            self._ack_ingress()
            # Let both native handlers apply acknowledged bootstrap administration
            # before waiting for the first Ready (which can depend on its reply).
            if (
                self._inflight is None
                and self._profile is None
                and not bootstrap_ingress
            ):
                with self._guard():
                    self._settle()
                    self._dispatch_pending()
        elif phase == "idle":
            _require(
                self._inflight is None and self._profile is None,
                "native idle before ingress/admin ACK",
            )
            self._cadence_only_pass = any(row["cadence_only"] for row in reports)
            self.on_idle()
        elif phase == "result_ready":
            keys = [row["result_key"] for row in reports]
            _require(
                keys[0] == keys[1] and self._joint_result is None,
                "native DP results have different global ownership",
            )
            _require(
                reports[0]["duration"] == reports[1]["duration"],
                "native DP result durations differ; no implicit sum/max is defined",
            )
            _require(
                keys[0][0].forward_id > self._last_forward_id,
                "native DP clock received an already processed forward",
            )
            self._joint_result = keys[0]
            self._prepare_result_time(keys[0], reports[0]["duration"])
        elif phase == "result_done":
            _require(
                all(row["result_key"] == self._joint_result for row in reports)
                and self._joint_result is not None,
                "native DP result ACK has no owner",
            )
            completed = [rid for row in reports for rid in row["completed"]]
            _require(
                len(completed) == len(set(completed)),
                "native result completed one RID twice",
            )
            if self._token_events is not None:
                self._token_events.record(
                    self.clock.now_ns,
                    [event for row in reports for event in row["token_events"]],
                )
            for rid in completed:
                if not self.clock.is_terminal(rid):
                    self.clock.terminal(rid, "COMPLETED")
            self._last_forward_id = self._joint_result[0].forward_id
            self._joint_result = None
            self._cancel_cleanup.clear()
            self._finish_result_time()
        elif phase == "profile_prepare":
            _require(
                self._profile is not None
                and self._inflight is None
                and self._joint_result is None
                and self._batch is None
                and all(not row["pending"] for row in reports),
                "native profile requires both drained source owners",
            )
            _require(
                all(row["profile_action"] == self._profile for row in reports),
                "native profile ACK differs from dispatched action",
            )
            stats = [stat for row in reports for stat in row["profile_stats"]]
            _require(
                len({stat.rid for stat in stats}) == len(stats),
                "duplicate DP request statistics",
            )
            self._report_round += 1
            return {
                **self._view(),
                "profile_stats": stats,
                "iterations": reports[0]["iterations"],
                "local_predictor_metrics": [
                    row["predictor_metrics"] for row in reports
                ],
            }
        elif phase == "profile_done":
            _require(
                self._profile is not None
                and all(row["profile_action"] == self._profile for row in reports),
                "native profile completion has no paired action",
            )
            self._profile = None
        else:
            raise ProtocolError("unsupported native DP report phase: " + str(phase))
        self._report_round += 1
        return self._view()

    def refresh_load_budget(self):
        """Run the native budget update at its unchanged 20ms virtual cadence."""
        import msgspec

        now = self.clock.now_ns / 1e9
        if now - self.controller._last_refresh_time < 0.02:
            return
        _require(self._loads is not None, "native routing has no paired load snapshot")
        self.controller._last_refresh_time = now
        self.controller.dp_budget.update_budget(
            [msgspec.structs.replace(load, timestamp=now) for load in self._loads]
        )

    def serve(self):
        """Replace only the native DPC poll loop, retaining its dispatcher."""
        while not self._closed:
            self.controller.soft_watchdog.feed()
            if not self._peer_socket.poll(5):
                continue
            try:
                with self._guard():
                    value = self.handle(self._peer_socket.recv_pyobj())
                self._peer_socket.send_pyobj({"value": value, "error": None})
            except BaseException as error:
                self._peer_socket.send_pyobj({"value": None, "error": str(error)})
                raise


class DPCoordinatedScheduler:
    """Paired native source events and read-only mirrors of the one DPC clock."""

    def __init__(self, *, scheduler, tp_group, endpoint, pending_work, timeout_s=30.0):
        _native_group(tp_group)
        _validate_prefill_cadence_runtime(scheduler)
        self.scheduler = scheduler
        self.group = tp_group
        self.rank = tp_group.rank_in_group
        self._pending_work = pending_work
        self.timeout_s = timeout_s
        self._round = 0
        self._generation = 0
        self._failure = None
        self._buffer = []
        self._arrivals = {}
        self._cancelled = frozenset()
        self.clock = _ClockView(0, {}, frozenset())
        self._finished = False
        self._context = self._socket = None
        if self.rank == 0:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
            self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
            self._socket.connect(endpoint)

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._context.term()

    def _report(self, phase, **extra):
        scheduler = self.scheduler
        return {
            "rank": self.rank,
            "phase": phase,
            "error": None,
            "generation": self._generation,
            "pending": bool(
                self._buffer
                or self._pending_work()
                or scheduler.result_queue
                or scheduler.waiting_queue
                or scheduler.chunked_req is not None
                or scheduler._pending_chunked_abort_req is not None
                or any(not req.finished() for req in scheduler.collect_inflight_reqs())
            ),
            "received": tuple(_receipt(req) for req in self._buffer),
            "cadence_only": bool(getattr(self, "_cadence_only_pass", False)),
            "rids": tuple(
                dict.fromkeys(
                    [
                        req.rid
                        for req in self._buffer
                        if type(req).__name__ == "TokenizedGenerateReqInput"
                    ]
                    + [req.rid for req in scheduler.collect_inflight_reqs()]
                    + list(extra.pop("rids", ()))
                )
            ),
            **(
                {"load": scheduler.load_inquirer.get_loads()}
                if phase == "receive"
                else {}
            ),
            **extra,
        }

    def _exchange(self, phase, make):
        failure = None
        try:
            _require(self._failure is None, "native DP clock failed; restart the run")
            report = make()
        except BaseException as error:  # noqa: BLE001 - rendezvous, then re-raise
            failure = error
            report = {"rank": self.rank, "phase": phase, "error": str(error)}
        reports = self.group.all_gather_object(report)
        response = None
        if self.rank == 0:
            try:
                self._socket.send_pyobj(
                    {"round": self._round, "phase": phase, "reports": reports}
                )
                response = self._socket.recv_pyobj()
            except BaseException as error:  # noqa: BLE001 - broadcast failure first
                response = {"value": None, "error": str(error)}
        replies = self.group.all_gather_object(response)
        response = replies[0]
        if failure is not None or response["error"] is not None:
            self._failure = response["error"] or str(failure)
            if failure is not None:
                raise failure
            raise ProtocolError(self._failure)
        self._round += 1
        value = response["value"]
        self._generation = value["generation"]
        self._arrivals = dict(value["arrivals"])
        self._cancelled = frozenset(value["cancelled"])
        self.clock = _ClockView(value["now_ns"], dict(self._arrivals), self._cancelled)
        self._finished = value["finished"]
        StateManager.set_global_clock(value["now_ns"] / 1e9)
        return value

    def receive(self, receive):
        started = time.monotonic()

        def report():
            self._buffer.extend(receive())
            _require(
                time.monotonic() - started < self.timeout_s,
                "native DP admission delivery timeout",
            )
            return self._report("receive")

        while True:
            response = self._exchange("receive", report)
            if not response["again"]:
                break
            time.sleep(0.001)
        result, self._buffer = self._buffer, []
        for req in result:
            if type(req).__name__ == "TokenizedGenerateReqInput":
                stats = request_stats_manager.get_req_stats(req.rid)
                stats.input_length = len(req.input_ids)
                stats.output_length = req.sampling_params.max_new_tokens
                stats.created_time = stats.queue_start = stats.last_event_time = (
                    self.clock.request_arrival_ns(req.rid) / 1e9
                )
        return result

    def process_batch_result(self, batch, result, *, process, duration):
        key = rids = requests = None

        def prepare():
            nonlocal key, rids, requests
            prediction = dspark_result_from_native(result)
            _require(
                prediction is not None,
                "native DP clock requires its owned joint result",
            )
            _require(
                prediction.step.topology == DSparkTopology(2, 2)
                and prediction.local_attention_dp_rank == self.rank
                and tuple(cost.phase for cost in prediction.timing.phases)
                == ("whole_worker",),
                "native DP clock requires one whole-worker result bound to this local rank",
            )
            key = (prediction.step, prediction.timing, prediction.progress)
            requests = tuple(batch.reqs)
            rids = tuple(req.rid for req in requests)
            _require(
                rids == tuple(req.rid for req in prediction.local_group.requests),
                "native local result differs from the joint request owner",
            )
            _require(
                type(duration) in (int, float)
                and duration == prediction.timing.seconds,
                "native DP duration differs from its single owned whole-worker cost",
            )
            return self._report(
                "result_ready", result_key=key, duration=duration, rids=rids
            )

        self._exchange("result_ready", prepare)
        ret = None

        def finish():
            nonlocal ret
            from sglang.srt.managers.io_struct import AbortReq
            from sglang.srt.managers.schedule_batch import FINISH_ABORT

            for rid in self._cancelled:
                self.scheduler.abort_request(AbortReq(rid=rid))
                for req in requests:
                    if req.rid == rid:
                        req.to_finish = FINISH_ABORT()
            events = _TokenCommits()
            attach_timing_output(result, self.clock, events)
            before = {
                id(req): (len(req.output_ids), req.finished()) for req in requests
            }
            # The existing scheduler supplies process-and-account, so neither
            # global terminals nor FINISHED can precede that exact consumption.
            ret = process()
            self.scheduler.process_pending_chunked_abort()
            for req in requests:
                count, done = before[id(req)]
                if (
                    not done
                    and not self.clock.is_cancelled(req.rid)
                    and len(req.output_ids) > count
                ):
                    stats = request_stats_manager.get_req_stats(req.rid)
                    stats.gen_token_latencies.append(
                        self.clock.now_ns / 1e9 - stats.last_event_time
                    )
                    stats.last_event_time = self.clock.now_ns / 1e9
            return self._report(
                "result_done",
                result_key=key,
                rids=rids,
                completed=tuple(req.rid for req in requests if req.finished()),
                token_events=events.events,
            )

        self._exchange("result_done", finish)
        return ret

    def on_idle(self):
        self._exchange("idle", lambda: self._report("idle"))

    def profile_boundary(self, req, *, snapshot, execute):
        action = None

        def prepare():
            nonlocal action
            action = _receipt(req)
            return self._report("profile_prepare", profile_action=action, **snapshot())

        response = self._exchange("profile_prepare", prepare)
        ret = None

        def finish():
            nonlocal ret
            ret = execute(response, self.rank == 0)
            _require(ret.success, "native profile failed: " + ret.message)
            return self._report("profile_done", profile_action=action)

        self._exchange("profile_done", finish)
        return ret


class _TokenCommits:
    """Forward the existing timing consumer's actual commits to its sole writer."""

    def __init__(self):
        self.events = []

    def record(self, at_ns, events):
        self.events.extend(events)


class C_DPCoordinatedHook(BaseHook):
    HOOK_MODULE_NAME = "sglang.srt.managers.data_parallel_controller"
    HOOK_CLASS_NAME = "DataParallelController"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_loop = target.event_loop
        original_refresh = target.refresh_load_budget

        @wraps(original_init)
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not native_dp_clock_enabled():
                return
            from sglang.srt.managers.io_struct import AbortReq, sock_recv
            from sglang.srt.managers.scheduler_components.output_sender import (
                SenderWrapper,
            )
            from sglang.srt.utils.network import get_zmq_socket

            ports = get_obj_from_args(
                "sglang.srt.server_args.PortArgs", *args, **kwargs
            )
            output = SenderWrapper(
                get_zmq_socket(self.context, zmq.PUSH, ports.tokenizer_ipc_name, False)
            )

            def receive():
                values = []
                while True:
                    try:
                        values.append(
                            sock_recv(self.recv_from_tokenizer, flags=zmq.NOBLOCK)
                        )
                    except zmq.Again:
                        return values

            self._simulator_dp_clock = DPCoordinatedController(
                controller=self,
                endpoint=Envs.coordinator_endpoint(),
                peer_endpoint=internal_endpoint(ports),
                receive=receive,
                abort_output=lambda req: output.send_output(AbortReq(rid=req.rid), req),
            )

        @wraps(original_loop)
        def event_loop(self):
            bridge = getattr(self, "_simulator_dp_clock", None)
            if bridge is None:
                return original_loop(self)
            try:
                return bridge.serve()
            finally:
                bridge.close()

        @wraps(original_refresh)
        def refresh(self):
            bridge = getattr(self, "_simulator_dp_clock", None)
            if bridge is None:
                return original_refresh(self)
            return bridge.refresh_load_budget()

        target.__init__ = initialize
        target.event_loop = event_loop
        target.refresh_load_budget = refresh
