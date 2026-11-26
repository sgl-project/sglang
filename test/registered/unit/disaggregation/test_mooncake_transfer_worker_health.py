"""A per-request error in the Mooncake prefill transfer worker fails only that
request, and a dead or stuck worker fails the health check.

Each prefill rank shards rooms over a few transfer queues, one worker each. A
worker that exits strands every later room hashed onto its queue, so it has to
survive one bad request and, when it really cannot go on, be visible.
"""

import asyncio
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill as prefill_module  # noqa: E402
from sglang.srt.disaggregation.base.conn import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.transfer_worker_health import (  # noqa: E402
    TRANSFER_WORKER_DEATHS,
    TRANSFER_WORKER_REQUEST_FAILURES,
    TransferWorkerHealth,
    is_worker_fatal_error,
)
from sglang.srt.disaggregation.common.utils import TransferKVChunk  # noqa: E402
from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import (  # noqa: E402
    MooncakeKVManager,
    TransferInfo,
)
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import HealthCheckOutput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ServerStatus,
    TokenizerManager,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BAD_ROOM = 201
GOOD_ROOM = 202
DECODE_IP = "10.0.0.2"
DECODE_PORT = 6000
SESSION = "decode-session-0"
# The error the incident raised: prefill and decode disagreed on the request
# length, so a positional state component had a different page count.
STATE_MISMATCH = RuntimeError(
    "C128_STATE state index length mismatch: prefill=1, dst=0"
)
CUDA_ERROR = RuntimeError("CUDA error: an illegal memory access was encountered")


def _counter(counter) -> float:
    return counter._value.get()


def _transfer_info(room: int, num_dst_pages: int = 1) -> TransferInfo:
    return TransferInfo(
        room=room,
        endpoint=DECODE_IP,
        dst_port=DECODE_PORT,
        mooncake_session_id=SESSION,
        dst_kv_indices=np.arange(num_dst_pages, dtype=np.int32),
        dst_aux_index=0,
        dst_state_indices=[[1]],
        required_dst_info_num=1,
        is_dummy=False,
    )


def _worker_manager(rooms) -> MooncakeKVManager:
    """A prefill manager whose sends succeed unless a test makes one raise."""
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.customized_info_enabled = False
    manager.exchange_aux_item_lens = False
    manager.enable_trace = False
    manager.enable_staging = False
    manager.enable_deferred_decode_kv_release = False
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.bootstrap_port = 8998
    manager.attn_tp_rank = manager.pp_rank = manager.attn_cp_rank = 0
    manager.pp_size = manager.attn_cp_size = 1
    manager._stopped = False
    manager._staging_outstanding = defaultdict(int)
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager.session_lock = threading.Lock()
    manager.session_failures = defaultdict(int)
    manager.failed_session_probes = {}
    manager.failed_sessions = set()
    manager.state_layout_rejections = {}
    manager.req_to_decode_prefix_len = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.kv_args = MagicMock(
        kv_data_ptrs=[0x1000], aux_data_ptrs=[0] * 8, aux_item_lens=[64] * 8
    )
    manager._get_dsa_cache_transfer_skip_flags = lambda info: (False, False)
    manager.decode_kv_args_table = {
        SESSION: MagicMock(
            requires_dcp_relayout=False, dst_aux_ptrs=[0] * 8, dst_aux_item_lens=None
        )
    }
    manager.request_status = {room: KVPoll.WaitingForInput for room in rooms}
    manager.transfer_infos = {
        room: {(DECODE_IP, DECODE_PORT): _transfer_info(room)} for room in rooms
    }
    manager.check_status = lambda room: manager.request_status[room]
    manager.update_status = lambda room, value: manager.request_status.__setitem__(
        room, value
    )
    manager.send_kvcache = MagicMock(return_value=0)
    manager.maybe_send_extra = MagicMock(return_value=0)
    manager.send_aux = MagicMock(return_value=0)
    manager.conclude_failure = MagicMock(
        side_effect=lambda **kw: manager.update_status(
            kw["bootstrap_room"], KVPoll.Failed
        )
    )
    manager.conclude_transfer = MagicMock(
        side_effect=lambda **kw: manager.update_status(
            kw["bootstrap_room"], kw["status"]
        )
    )
    manager.transfer_worker_health = TransferWorkerHealth()
    return manager


def _last_chunk(room: int, num_pages: int = 1) -> TransferKVChunk:
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.arange(num_pages, dtype=np.int32),
        index_slice=slice(0, num_pages),
        is_last_chunk=True,
        prefill_aux_index=0,
        state_indices=[[1]],
    )


def _queue(*chunks) -> MagicMock:
    """One shard queue holding chunks, then teardown's stop sentinel."""
    queue = MagicMock()
    queue.get.side_effect = [*chunks, None]
    return queue


def _raise_for(room: int, exc: Exception):
    def send_extra(req, *args, **kwargs):
        if req.room == room:
            raise exc
        return 0

    return send_extra


class TestTransferWorkerSurvivesRequestErrors(CustomTestCase):
    def test_request_error_fails_its_room_and_the_next_chunk_is_sent(self):
        manager = _worker_manager([BAD_ROOM, GOOD_ROOM])
        manager.maybe_send_extra.side_effect = _raise_for(BAD_ROOM, STATE_MISMATCH)
        failures_before = _counter(TRANSFER_WORKER_REQUEST_FAILURES)

        # Both rooms are on the same queue; returning means the worker drained
        # it up to the stop sentinel instead of dying on the first chunk.
        manager.transfer_worker(
            _queue(_last_chunk(BAD_ROOM), _last_chunk(GOOD_ROOM)), MagicMock()
        )

        self.assertEqual(manager.request_status[BAD_ROOM], KVPoll.Failed)
        self.assertIn("state index length mismatch", manager.failure_records[BAD_ROOM])
        # Decode is told now rather than at its waiting timeout.
        manager.conclude_failure.assert_any_call(
            bootstrap_room=BAD_ROOM, failure_reason=manager.failure_records[BAD_ROOM]
        )
        self.assertEqual(manager.request_status[GOOD_ROOM], KVPoll.Success)
        manager.send_aux.assert_called_once()
        self.assertIsNone(manager.transfer_worker_unhealthy_reason())
        self.assertEqual(
            _counter(TRANSFER_WORKER_REQUEST_FAILURES), failures_before + 1
        )

    def test_failed_room_withholds_the_drained_abort_ack(self):
        # The failing chunk may have left a write in flight.
        manager = _worker_manager([BAD_ROOM])
        manager.maybe_send_extra.side_effect = STATE_MISMATCH

        manager.transfer_worker(_queue(_last_chunk(BAD_ROOM)), MagicMock())

        self.assertIn(BAD_ROOM, manager._deferred_ack_poisoned_rooms)

    def test_error_after_success_keeps_the_room_successful(self):
        manager = _worker_manager([GOOD_ROOM])
        manager.note_send_finished = MagicMock(side_effect=KeyError("bookkeeping"))

        manager.transfer_worker(_queue(_last_chunk(GOOD_ROOM)), MagicMock())

        self.assertEqual(manager.request_status[GOOD_ROOM], KVPoll.Success)

    def test_error_on_a_cleared_room_records_nothing(self):
        # clear() popped the room mid-send: a record now would be adopted by
        # a later request that reuses this bootstrap_room.
        manager = _worker_manager([BAD_ROOM, GOOD_ROOM])
        failures_before = _counter(TRANSFER_WORKER_REQUEST_FAILURES)

        def clear_then_raise(req, *args, **kwargs):
            if req.room == BAD_ROOM:
                manager.request_status.pop(BAD_ROOM)
                raise STATE_MISMATCH
            return 0

        manager.maybe_send_extra.side_effect = clear_then_raise
        manager.transfer_worker(
            _queue(_last_chunk(BAD_ROOM), _last_chunk(GOOD_ROOM)), MagicMock()
        )

        self.assertNotIn(BAD_ROOM, manager.failure_records)
        self.assertNotIn(BAD_ROOM, manager.request_status)
        self.assertEqual(manager.request_status[GOOD_ROOM], KVPoll.Success)
        self.assertEqual(_counter(TRANSFER_WORKER_REQUEST_FAILURES), failures_before)

    def test_kv_page_count_mismatch_fails_instead_of_truncating(self):
        # Decode registered one page where prefill sends four: sending the first
        # page and reporting Success would hand decode a truncated prompt.
        manager = _worker_manager([BAD_ROOM])

        manager.transfer_worker(_queue(_last_chunk(BAD_ROOM, 4)), MagicMock())

        manager.send_kvcache.assert_not_called()
        manager.send_aux.assert_not_called()
        self.assertEqual(manager.request_status[BAD_ROOM], KVPoll.Failed)
        reason = manager.conclude_failure.call_args.kwargs["failure_reason"]
        self.assertIn("KV page count mismatch", reason)


class TestTransferWorkerDeath(CustomTestCase):
    def test_fatal_error_kills_the_worker_and_reports_it(self):
        manager = _worker_manager([BAD_ROOM, GOOD_ROOM])
        manager.maybe_send_extra.side_effect = _raise_for(BAD_ROOM, CUDA_ERROR)
        deaths_before = _counter(TRANSFER_WORKER_DEATHS)
        queue = _queue(_last_chunk(BAD_ROOM), _last_chunk(GOOD_ROOM))

        with self.assertRaisesRegex(RuntimeError, "Transfer thread failed"):
            manager.transfer_worker(queue, MagicMock(), worker_index=2)

        self.assertEqual(manager.request_status[BAD_ROOM], KVPoll.Failed)
        reason = manager.transfer_worker_unhealthy_reason()
        self.assertIn("worker 2 is dead", reason)
        self.assertEqual(_counter(TRANSFER_WORKER_DEATHS), deaths_before + 1)

    def test_rooms_for_a_dead_shard_fail_at_enqueue(self):
        manager = _worker_manager([GOOD_ROOM])
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.transfer_infos = {GOOD_ROOM: {f"{DECODE_IP}:{DECODE_PORT}": None}}
        manager.transfer_queues = [MagicMock() for _ in range(4)]
        shard = DECODE_PORT % 4
        manager.transfer_worker_health.mark_dead(shard, "RuntimeError()")

        manager.add_transfer_request(
            GOOD_ROOM, np.array([1], dtype=np.int32), slice(0, 1), False
        )

        manager.transfer_queues[shard].put.assert_not_called()
        self.assertEqual(manager.request_status[GOOD_ROOM], KVPoll.Failed)

    def test_teardown_is_not_a_death(self):
        manager = _worker_manager([BAD_ROOM])
        manager._stopped = True
        manager.maybe_send_extra.side_effect = RuntimeError(
            "cannot schedule new futures after shutdown"
        )

        with self.assertRaises(RuntimeError):
            manager.transfer_worker(_queue(_last_chunk(BAD_ROOM)), MagicMock())

        self.assertIsNone(manager.transfer_worker_unhealthy_reason())

    def test_only_device_and_pool_errors_are_fatal(self):
        self.assertTrue(is_worker_fatal_error(CUDA_ERROR))
        self.assertTrue(
            is_worker_fatal_error(RuntimeError("HIP error: an illegal memory access"))
        )
        self.assertTrue(
            is_worker_fatal_error(
                RuntimeError("cannot schedule new futures after shutdown")
            )
        )
        self.assertFalse(is_worker_fatal_error(STATE_MISMATCH))
        self.assertFalse(is_worker_fatal_error(KeyError("decode-session-0")))
        self.assertFalse(is_worker_fatal_error(ValueError("bad aux buffer")))


class TestTransferWorkerStall(CustomTestCase):
    def test_a_worker_stuck_on_one_chunk_is_reported_until_it_finishes(self):
        now = [100.0]
        health = TransferWorkerHealth(stall_timeout_s=60, clock=lambda: now[0])

        health.mark_busy(1)
        now[0] += 60
        self.assertIsNone(health.unhealthy_reason())
        now[0] += 1
        self.assertIn("worker 1 has been on one chunk", health.unhealthy_reason())

        health.mark_idle(1)
        self.assertIsNone(health.unhealthy_reason())

    def test_the_worker_reports_a_stall_on_the_chunk_it_is_sending(self):
        now = [0.0]
        manager = _worker_manager([GOOD_ROOM])
        manager.transfer_worker_health = TransferWorkerHealth(
            stall_timeout_s=60, clock=lambda: now[0]
        )
        seen = []

        def slow_send(*args, **kwargs):
            now[0] += 61
            seen.append(manager.transfer_worker_unhealthy_reason())
            return 0

        manager.maybe_send_extra.side_effect = slow_send
        manager.transfer_worker(
            _queue(_last_chunk(GOOD_ROOM)), MagicMock(), worker_index=3
        )

        self.assertIn("worker 3 has been on one chunk", seen[0])
        self.assertIsNone(manager.transfer_worker_unhealthy_reason())

    def test_zero_timeout_disables_the_stall_check(self):
        now = [0.0]
        health = TransferWorkerHealth(stall_timeout_s=0, clock=lambda: now[0])
        health.mark_busy(0)
        now[0] += 1e6
        self.assertIsNone(health.unhealthy_reason())


def _prefill_scheduler(local_reason):
    scheduler = object.__new__(scheduler_module.Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.session_controller = SimpleNamespace(maybe_reap=MagicMock())
    scheduler._request_dispatcher = MagicMock(return_value=None)
    scheduler.flush_wrapper = SimpleNamespace(check_pending=MagicMock())
    scheduler.external_corpus_manager = None
    scheduler.rust_server = None
    scheduler.ipc_channels = MagicMock()
    scheduler._event_loop_should_restart = False
    scheduler.return_health_check_ipcs = []
    scheduler.is_fully_idle = MagicMock(return_value=False)
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.attn_tp_cpu_group = scheduler.attn_cp_cpu_group = "group"
    kv_manager = SimpleNamespace(transfer_worker_unhealthy_reason=lambda: local_reason)
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=kv_manager)
    return scheduler


class TestPrefillHealthProbe(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def _probe(self, scheduler, peer_flag=0):
        probe = SimpleNamespace(rid="HEALTH_CHECK_abc", http_worker_ipc="ipc://w0")

        def all_reduce_flags(flags, group, op):
            return [max(flags[0], peer_flag)]

        with (
            patch.object(prefill_module, "all_reduce_flags", all_reduce_flags),
            patch.object(
                scheduler_module,
                "get_mm",
                return_value=SimpleNamespace(mm_feature_transport=""),
            ),
        ):
            scheduler.process_input_requests([probe])

    def _sent_reason(self, scheduler):
        (output,) = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args
        self.assertIsInstance(output, HealthCheckOutput)
        self.assertEqual(output.http_worker_ipc, "ipc://w0")
        return output.unhealthy_reason

    def test_a_dead_worker_fails_the_probe_even_under_load(self):
        # Busy: the probe would otherwise be acknowledged from ongoing traffic.
        scheduler = _prefill_scheduler("KV transfer worker 0 is dead")

        self._probe(scheduler)

        self.assertEqual(self._sent_reason(scheduler), "KV transfer worker 0 is dead")
        self.assertEqual(scheduler.return_health_check_ipcs, [])

    def test_a_dead_worker_on_another_rank_fails_the_probe(self):
        scheduler = _prefill_scheduler(None)

        self._probe(scheduler, peer_flag=1)

        self.assertIn("another rank", self._sent_reason(scheduler))

    def test_healthy_workers_leave_the_probe_alone(self):
        scheduler = _prefill_scheduler(None)

        self._probe(scheduler)

        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
        self.assertEqual(scheduler.return_health_check_ipcs, ["ipc://w0"])


def _tokenizer_manager():
    tm = object.__new__(TokenizerManager)
    tm._engine_state_changed_callback = None
    tm._server_status = ServerStatus.Up
    tm._gracefully_exit = False
    tm.scheduler_unhealthy_reason = None
    tm.last_receive_tstamp = 0.0
    tm.rid_to_state = {}
    tm.is_generation = True
    return tm


class TestHealthEndpoint(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def test_tokenizer_manager_keeps_the_first_failure(self):
        tm = _tokenizer_manager()

        tm._handle_health_check_output(HealthCheckOutput())
        self.assertIsNone(tm.scheduler_unhealthy_reason)
        self.assertTrue(tm.is_healthy())

        tm._handle_health_check_output(HealthCheckOutput(unhealthy_reason="dead"))
        tm._handle_health_check_output(HealthCheckOutput(unhealthy_reason="later"))
        tm._handle_health_check_output(HealthCheckOutput())
        self.assertEqual(tm.scheduler_unhealthy_reason, "dead")
        self.assertFalse(tm.is_healthy())

    def test_health_returns_503_once_a_scheduler_reported_a_failure(self):
        from sglang.srt.entrypoints import http_server

        tm = _tokenizer_manager()
        request = SimpleNamespace(url=SimpleNamespace(path="/health"))

        async def generate_request(obj, request):
            # The probe's failure and another request's output land together.
            tm._handle_health_check_output(HealthCheckOutput(unhealthy_reason="dead"))
            tm.last_receive_tstamp = float("inf")
            yield {}

        tm.generate_request = generate_request
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
        ):
            first = asyncio.run(http_server.health_generate(request))
            # Sticky: later probes fail without reaching a scheduler.
            tm.generate_request = MagicMock()
            second = asyncio.run(http_server.health_generate(request))

        self.assertEqual(first.status_code, 503)
        self.assertEqual(second.status_code, 503)
        tm.generate_request.assert_not_called()

    def test_probes_rotate_the_fake_room_to_reach_every_dp_rank(self):
        # PD prefill dispatches DP ranks by bootstrap_room % dp_size.
        from sglang.srt.entrypoints import http_server

        tm = _tokenizer_manager()
        rooms = []

        async def generate_request(obj, request):
            rooms.append(obj.bootstrap_room)
            tm.last_receive_tstamp = float("inf")
            yield {}

        tm.generate_request = generate_request
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        with (
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
            ),
            patch.object(
                http_server,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
        ):
            for _ in range(2):
                response = asyncio.run(http_server.health_generate(request))
                self.assertEqual(response.status_code, 200)

        self.assertEqual(rooms[1], rooms[0] + 1)


if __name__ == "__main__":
    unittest.main()
