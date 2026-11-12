"""Health probes are acknowledged only by completed inference.

A prefill engine whose model execution stalled kept passing /health_generate
because any scheduler message counted. Now every probe is registered with the
scheduler's health monitor and acknowledged, by rid, only after a batch with
real work has been fully processed: a positive prefill chunk, a decode step or
a verify step whose device results were synchronized. Control and abort
traffic, launches, idle housekeeping, aborted-only or zero-token batches and
failed result processing acknowledge nothing.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.health_monitor import (
    SchedulerHealthMonitor,
)
from sglang.srt.managers.scheduler_components.request_list import ListKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PROBE_RID = f"{HEALTH_CHECK_RID_PREFIX}_probe"
PROBE_IPC = "ipc:///tmp/http_worker_0"


def _recv(rid, http_worker_ipc=None):
    return SimpleNamespace(rid=rid, http_worker_ipc=http_worker_ipc)


def _req(rid, http_worker_ipc=None, aborted=False):
    return SimpleNamespace(
        rid=rid,
        http_worker_ipc=http_worker_ipc,
        to_finish=FINISH_ABORT() if aborted else None,
        finished_reason=None,
    )


def _batch(*reqs, forward_mode=ForwardMode.EXTEND, extend_num_tokens=512):
    return SimpleNamespace(
        reqs=list(reqs),
        forward_mode=forward_mode,
        extend_num_tokens=extend_num_tokens,
        is_dllm=lambda: False,
    )


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestSchedulerHealthMonitor(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.sent = []
        self.clock = _Clock()
        self.monitor = SchedulerHealthMonitor(
            self.sent.append, dp_rank=3, max_pending=4, ttl_s=40.0, now=self.clock
        )

    def acks(self):
        for output in self.sent:
            self.assertIsInstance(output, HealthCheckOutput)
            self.assertEqual(output.incarnation, self.monitor.incarnation)
            self.assertEqual(output.dp_rank, 3)
        return [(o.rid, o.http_worker_ipc, o.completion_seq) for o in self.sent]

    def test_prefill_chunk_acks_every_pending_probe_once(self):
        self.monitor.register(_recv(PROBE_RID, PROBE_IPC), admitted=False)
        self.monitor.register(
            _recv(f"{PROBE_RID}2", "ipc:///tmp/http_worker_1"), admitted=True
        )

        self.assertTrue(self.monitor.on_batch_completed(_batch(_req("long"))))
        self.assertEqual(
            self.acks(),
            [
                (PROBE_RID, PROBE_IPC, 1),
                (f"{PROBE_RID}2", "ipc:///tmp/http_worker_1", 1),
            ],
        )
        self.assertEqual(self.monitor.pending_rids, [])

        # The rest of a 47-chunk chain does not repeat the acknowledgement.
        for _ in range(46):
            self.monitor.on_batch_completed(_batch(_req("long")))
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.monitor.completion_seq, 47)

    def test_decode_and_verify_steps_count_as_progress(self):
        self.monitor.register(_recv(PROBE_RID), admitted=False)
        self.assertTrue(
            self.monitor.on_batch_completed(
                _batch(
                    _req("r"), forward_mode=ForwardMode.DECODE, extend_num_tokens=None
                )
            )
        )
        self.monitor.register(_recv(PROBE_RID), admitted=False)
        self.assertTrue(
            self.monitor.on_batch_completed(
                _batch(_req("r"), forward_mode=ForwardMode.TARGET_VERIFY)
            )
        )
        self.assertEqual([seq for _, _, seq in self.acks()], [1, 2])

    def test_probe_only_forward_is_real_inference(self):
        self.monitor.register(_recv(PROBE_RID, PROBE_IPC), admitted=True)
        self.assertTrue(
            self.monitor.on_batch_completed(
                _batch(_req(PROBE_RID, PROBE_IPC), extend_num_tokens=1)
            )
        )
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])

    def test_batches_that_prove_nothing_do_not_ack(self):
        self.monitor.register(_recv(PROBE_RID), admitted=False)
        nothing = [
            _batch(),
            _batch(forward_mode=ForwardMode.IDLE, extend_num_tokens=None),
            _batch(
                _req("r"), forward_mode=ForwardMode.PREBUILT, extend_num_tokens=None
            ),
            _batch(_req("r"), forward_mode=ForwardMode.DRAFT_EXTEND_V2),
            _batch(_req("r"), extend_num_tokens=0),
            _batch(_req("r"), extend_num_tokens=None),
            _batch(_req("a", aborted=True), _req("b", aborted=True)),
        ]
        for batch in nothing:
            self.assertFalse(self.monitor.on_batch_completed(batch), batch)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.monitor.completion_seq, 0)
        self.assertEqual(self.monitor.pending_rids, [PROBE_RID])

    def test_sequence_is_monotonic_across_probes(self):
        for i in range(3):
            self.monitor.on_batch_completed(_batch(_req("r")))
        self.monitor.register(_recv(PROBE_RID), admitted=False)
        self.monitor.on_batch_completed(_batch(_req("r")))
        self.assertEqual(self.acks(), [(PROBE_RID, None, 4)])

    def test_deferred_probes_are_handed_back_once(self):
        busy = _recv(PROBE_RID, PROBE_IPC)
        admitted = _recv(f"{PROBE_RID}2")
        self.monitor.register(busy, admitted=False)
        self.monitor.register(admitted, admitted=True)

        self.assertEqual(self.monitor.pop_deferred(), [busy])
        self.assertEqual(self.monitor.pop_deferred(), [])
        # Still pending: the admitted forward acknowledges it later.
        self.assertEqual(self.monitor.pending_rids, [PROBE_RID, f"{PROBE_RID}2"])
        self.monitor.on_batch_completed(_batch(_req(PROBE_RID, PROBE_IPC)))
        self.assertEqual(len(self.sent), 2)

    def test_expired_probes_are_dropped_not_acked(self):
        self.monitor.register(_recv(f"{PROBE_RID}_old"), admitted=False)
        self.clock.now += 39
        new = _recv(f"{PROBE_RID}_new")
        self.monitor.register(new, admitted=False)
        self.clock.now += 2

        self.assertEqual(self.monitor.pop_deferred(), [new])
        self.monitor.on_batch_completed(_batch(_req("r")))
        self.assertEqual(self.acks(), [(f"{PROBE_RID}_new", None, 1)])

    def test_pending_is_bounded(self):
        for i in range(6):
            self.monitor.register(_recv(f"{PROBE_RID}_{i}"), admitted=False)
        self.assertEqual(
            self.monitor.pending_rids, [f"{PROBE_RID}_{i}" for i in range(2, 6)]
        )

    def test_ack_wire_format_round_trips_and_reads_legacy_shape(self):
        import msgspec

        self.monitor.register(_recv(PROBE_RID, PROBE_IPC), admitted=False)
        self.monitor.on_batch_completed(_batch(_req("r")))
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(HealthCheckOutput)
        self.assertEqual(decoder.decode(encoder.encode(self.sent[0])), self.sent[0])

        class Legacy(
            msgspec.Struct, tag="HealthCheckOutput", kw_only=True, array_like=True
        ):
            rid: str = None
            http_worker_ipc: str = None

        legacy = decoder.decode(encoder.encode(Legacy(rid=PROBE_RID)))
        self.assertIsNone(legacy.incarnation)
        self.assertIsNone(legacy.completion_seq)


class TestSchedulerHealthWiring(CustomTestCase):
    """The real Scheduler paths around the monitor."""

    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="scheduler"))

    def _scheduler(self, mode=DisaggregationMode.PREFILL):
        scheduler = Scheduler.__new__(Scheduler)
        self.sent = []
        scheduler.scheduler_stage_metrics = None
        scheduler.disaggregation_mode = mode
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = (
            SimpleNamespace(queue=[]) if mode == DisaggregationMode.PREFILL else None
        )
        # State read by the real is_fully_idle(for_health_check=True).
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._engine_paused = False
        scheduler._pp_microbatches_drained = lambda: True
        scheduler.request_lists = lambda: {
            ListKind.WAITING: scheduler.waiting_queue,
            ListKind.CHUNKED: [],
            **(
                {ListKind.PD_BOOTSTRAP: scheduler.disagg_prefill_bootstrap_queue.queue}
                if scheduler.disagg_prefill_bootstrap_queue is not None
                else {}
            ),
        }
        scheduler.health_monitor = SchedulerHealthMonitor(self.sent.append, dp_rank=0)
        # process_input_requests collaborators outside the code under test.
        scheduler.session_controller = SimpleNamespace(maybe_reap=lambda now: None)
        scheduler.rust_server = None
        scheduler.flush_wrapper = SimpleNamespace(check_pending=lambda: None)
        scheduler.external_corpus_manager = None
        scheduler._request_dispatcher = lambda recv_req: self._enqueue(
            scheduler, recv_req
        )
        # process_batch_result collaborators.
        scheduler.publish_load_snapshot = Mock(return_value=None)
        scheduler.load_publisher = Mock()
        scheduler.load_inquirer = SimpleNamespace(get_loads=Mock())
        scheduler.batch_result_processor = Mock()
        scheduler.process_batch_result_disagg_prefill = Mock()
        scheduler._on_step_processed = Mock()
        scheduler.deferred_responses = Mock()
        scheduler.tree_cache = Mock()
        scheduler._record_step_counters = Mock()
        scheduler.metrics_reporter = Mock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = Mock()
        return scheduler

    @staticmethod
    def _enqueue(scheduler, recv_req):
        # handle_generate_request -> _add_request_to_queue: a PD prefill parks
        # every new request in the bootstrap queue first.
        req = _req(recv_req.rid, recv_req.http_worker_ipc)
        if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            scheduler.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            scheduler.waiting_queue.append(req)

    def acks(self):
        for output in self.sent:
            self.assertIsInstance(output, HealthCheckOutput)
            self.assertIsNotNone(output.incarnation)
        return [(o.rid, o.http_worker_ipc, o.completion_seq) for o in self.sent]

    def test_busy_engine_acks_with_the_running_batch_not_a_forward(self):
        scheduler = self._scheduler()
        running = _req("running")
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: False)

        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])
        self.assertEqual(scheduler.disagg_prefill_bootstrap_queue.queue, [])
        self.assertEqual(self.sent, [])  # registration alone proves nothing

        scheduler.process_batch_result(
            _batch(running, forward_mode=ForwardMode.DECODE, extend_num_tokens=None),
            Mock(),
        )
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])

    def test_control_abort_and_error_traffic_during_stall_acks_nothing(self):
        scheduler = self._scheduler()
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: False)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])

        # Aborts, flushes and other control requests keep flowing while the
        # forward never completes; so do result-less idle ticks.
        scheduler._request_dispatcher = lambda recv_req: SimpleNamespace(ok=True)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )
        scheduler.process_input_requests([_recv("abort-1"), _recv("flush")])
        scheduler.admit_deferred_health_probes()
        self.assertEqual(self.sent, [])
        self.assertEqual(scheduler.health_monitor.pending_rids, [PROBE_RID])

    def test_failed_result_processing_acks_nothing(self):
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        scheduler.running_batch = SimpleNamespace(is_empty=lambda: False)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])
        # The launch happened; the device-to-host copy reports failure.
        scheduler.batch_result_processor.process_batch_result_prefill.side_effect = (
            RuntimeError("async copy failed")
        )

        with self.assertRaises(RuntimeError):
            scheduler.process_batch_result(_batch(_req("r")), Mock())
        self.assertEqual(self.sent, [])

    def test_probe_queued_behind_chunked_prompt_is_acked_by_first_chunk(self):
        scheduler = self._scheduler()
        scheduler.process_input_requests([_recv("long"), _recv(PROBE_RID, PROBE_IPC)])
        self.assertEqual(
            [r.rid for r in scheduler.disagg_prefill_bootstrap_queue.queue],
            ["long", PROBE_RID],
        )
        long_req = scheduler.disagg_prefill_bootstrap_queue.queue[0]

        scheduler.process_batch_result(_batch(long_req), Mock())
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])
        for _ in range(46):
            scheduler.process_batch_result(_batch(long_req), Mock())
        self.assertEqual(len(self.sent), 1)

    def test_probe_deferred_after_last_batch_runs_on_idle(self):
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        # The iteration right after the last real batch: last_batch is still
        # set, so the probe is deferred, and no completion will follow.
        scheduler.last_batch = SimpleNamespace(is_empty=lambda: False)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])
        self.assertEqual(scheduler.waiting_queue, [])

        scheduler.admit_deferred_health_probes()  # what on_idle does
        self.assertEqual([r.rid for r in scheduler.waiting_queue], [PROBE_RID])
        scheduler.admit_deferred_health_probes()
        self.assertEqual(len(scheduler.waiting_queue), 1)  # admitted once
        self.assertEqual(self.sent, [])

        probe = scheduler.waiting_queue.pop()
        scheduler.process_batch_result(_batch(probe, extend_num_tokens=1), Mock())
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])

    def test_dp_rank_running_lockstep_idle_batches_admits_deferred_probe(self):
        # Under DP attention a rank with no work of its own still runs IDLE
        # forwards every iteration while a sibling is busy, so on_idle is
        # never reached; those passes prove nothing and must not ack.
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        scheduler.last_batch = SimpleNamespace(is_empty=lambda: False)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])

        scheduler.process_batch_result(
            _batch(forward_mode=ForwardMode.IDLE, extend_num_tokens=None), Mock()
        )
        self.assertEqual(self.sent, [])
        self.assertEqual([r.rid for r in scheduler.waiting_queue], [PROBE_RID])

        probe = scheduler.waiting_queue.pop()
        scheduler.process_batch_result(_batch(probe, extend_num_tokens=1), Mock())
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])

    def test_idle_probe_is_admitted_and_acked_by_its_own_forward(self):
        scheduler = self._scheduler(mode=DisaggregationMode.NULL)
        scheduler.process_input_requests([_recv(PROBE_RID, PROBE_IPC)])
        (probe,) = scheduler.waiting_queue
        self.assertEqual(self.sent, [])

        scheduler.waiting_queue.clear()
        scheduler.process_batch_result(_batch(probe, extend_num_tokens=1), Mock())
        self.assertEqual(self.acks(), [(PROBE_RID, PROBE_IPC, 1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
