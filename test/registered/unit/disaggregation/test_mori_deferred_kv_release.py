"""Deferred decode-side KV release on the Mori backend."""

import ctypes
import struct
import threading
import unittest
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.test.mori_stub import mori_stub

_mori_stub = mori_stub()
if _mori_stub is not None:
    _mori_stub.start()

from mori.io import StatusCode
from test_deferred_decode_kv_release import (
    ABORT_GENERATION,
    DeferredAbortNotificationScenarios,
)

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import AbortNotification, AckTarget
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mori.conn import (
    MoriKVManager,
    _MoriTransferSubmissionError,
    _SubmissionLocal,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _manager(enabled: bool = True) -> MoriKVManager:
    manager = MoriKVManager.__new__(MoriKVManager)
    manager.enable_deferred_decode_kv_release = enabled
    manager.request_status = {}
    manager.transfer_lock = threading.Lock()
    manager._abort_ack_lock = threading.Lock()
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._deferred_abort_ack_tracker = {}
    manager._staging_outstanding = defaultdict(int)
    manager._drain_queue = Queue(maxsize=1)
    manager._submission_local = _SubmissionLocal()
    manager.req_to_decode_prefix_len = {}
    manager.transfer_infos = {}
    manager.failure_lock = threading.Lock()
    manager.failure_records = {}
    manager._sent = []
    manager._send_abort_ack = lambda *args: manager._sent.append(args)
    manager._wait_poll_ms = 0
    manager._transfer_timeout_ms = 0
    return manager


def _abort_message(room: int = 11) -> list[bytes]:
    return AbortNotification(room, "10.0.0.3", 6000, ABORT_GENERATION).to_zmq()


def _chunk(room: int = 11) -> TransferKVChunk:
    return TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )


class TestMoriAbortAck(DeferredAbortNotificationScenarios, CustomTestCase):
    room = 11
    decode_ip = "10.0.0.3"
    decode_port = 6000

    def _make_abort_manager(self, status: KVPoll | None):
        manager = _manager()
        if status is not None:
            manager.request_status[self.room] = status
        return manager

    def _dispatch_abort(self, manager) -> None:
        manager._handle_abort_message(self._abort_message())

    def _start_test_transfer(self, manager) -> None:
        manager._mark_transfer_started(_chunk(self.room))

    def _drain_test_transfer(self, manager) -> None:
        manager._mark_transfer_quiescent(_chunk(self.room))

    def test_abort_after_submit_waits_for_terminal_status(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        status.InProgress.return_value = False
        status.Failed.return_value = False
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        wait_results = iter((StatusCode.IN_PROGRESS, StatusCode.SUCCESS))

        def wait_all(*args, **kwargs):
            result = next(wait_results)
            if result == StatusCode.IN_PROGRESS:
                manager._handle_abort_message(_abort_message())
                self.assertEqual(manager._sent, [])
            return result

        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = wait_all
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        is_quiescent = manager._process_transfer_chunk(chunk)

        self.assertTrue(is_quiescent)
        self.assertEqual(manager.engine.wait_all.call_count, 2)
        self.assertEqual(manager._sent, [])
        manager._mark_transfer_quiescent(chunk)
        self.assertEqual(
            manager._sent,
            [(11, AckTarget("10.0.0.3", 6000, ABORT_GENERATION))],
        )

    def test_sla_failure_acks_after_status_drains(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._transfer_timeout_ms = 1
        manager.engine = MagicMock()
        manager.engine.wait_all.return_value = StatusCode.IN_PROGRESS
        status = MagicMock()
        status.InProgress.return_value = True
        status.Failed.return_value = False
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        with patch(
            "sglang.srt.disaggregation.mori.conn.time.perf_counter",
            side_effect=(0.0, 0.002),
        ):
            is_quiescent = manager._process_transfer_chunk(chunk)

        self.assertFalse(is_quiescent)
        manager.engine.wait_all.assert_called_once()
        self.assertEqual(manager._sent, [])
        self.assertEqual(manager._staging_outstanding[11], 1)
        # Concluded at detection; only the ACK waits for the drain.
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11, failure_reason="KV transfer exceeded SLA 1ms"
        )
        manager._handle_abort_message(_abort_message())
        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        manager._drain_transfer_statuses(queued_chunk, queued_statuses)

        status.Wait.assert_called_once_with()
        self.assertEqual(
            manager._sent,
            [(11, AckTarget("10.0.0.3", 6000, ABORT_GENERATION))],
        )
        self.assertNotIn(11, manager._staging_outstanding)
        manager.conclude_failure.assert_called_once()

    def test_failed_status_never_acks(self):
        """A write that ended Failed() may still land in decode's pages, so
        neither the pending ABORT nor a later one may be ACKed."""
        for via_drainer in (False, True):
            with self.subTest(via_drainer=via_drainer):
                manager = _manager()
                manager.request_status[11] = KVPoll.Transferring
                manager.conclude_failure = MagicMock()
                status = MagicMock()
                status.InProgress.return_value = via_drainer
                status.Failed.return_value = True
                manager._submit_kv_transfer = MagicMock(return_value=[status])
                manager.engine = MagicMock()
                manager.engine.wait_all.return_value = (
                    StatusCode.IN_PROGRESS if via_drainer else StatusCode.FAILED
                )
                manager._transfer_timeout_ms = 1 if via_drainer else 0
                chunk = _chunk()
                manager._mark_transfer_started(chunk)

                with patch(
                    "sglang.srt.disaggregation.mori.conn.time.perf_counter",
                    side_effect=(0.0, 0.002),
                ):
                    is_quiescent = manager._process_transfer_chunk(chunk)
                manager._handle_abort_message(_abort_message())
                if via_drainer:
                    self.assertFalse(is_quiescent)
                    queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
                    manager._drain_transfer_statuses(queued_chunk, queued_statuses)
                else:
                    self.assertTrue(is_quiescent)
                    manager._mark_transfer_quiescent(chunk)
                manager._handle_abort_message(_abort_message())

                self.assertEqual(manager._sent, [])
                self.assertIn(11, manager._deferred_ack_poisoned_rooms)
                self.assertNotIn(11, manager._staging_outstanding)

    def test_room_cleared_after_drain_handoff_keeps_single_owner(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager._transfer_timeout_ms = 1
        status = MagicMock()
        status.InProgress.return_value = True
        manager._submit_kv_transfer = MagicMock(return_value=[status])

        def wait_all(*args, **kwargs):
            manager.request_status.pop(11)
            return StatusCode.IN_PROGRESS

        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = wait_all
        manager.conclude_failure = MagicMock()
        queue = MagicMock()
        queue.get.side_effect = (_chunk(), KeyboardInterrupt())

        with patch(
            "sglang.srt.disaggregation.mori.conn.time.perf_counter",
            side_effect=(0.0, 0.002),
        ):
            with self.assertRaises(KeyboardInterrupt):
                manager._transfer_worker(queue)

        self.assertEqual(manager._staging_outstanding[11], 1)
        self.assertEqual(manager._drain_queue.qsize(), 1)
        manager.conclude_failure.assert_called_once()

    def test_wait_event_failure_releases_outstanding_count(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        chunk.wait_event = MagicMock()
        chunk.wait_event.synchronize.side_effect = RuntimeError("event failed")
        queue = MagicMock()
        queue.get.side_effect = (chunk, KeyboardInterrupt())

        with patch("sglang.srt.disaggregation.mori.conn.logger.exception"):
            with self.assertRaises(KeyboardInterrupt):
                manager._transfer_worker(queue)

        self.assertNotIn(11, manager._staging_outstanding)
        manager.conclude_failure.assert_called_once()

    def test_wait_all_exception_keeps_status_for_draining(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()
        manager._submit_kv_transfer = MagicMock(return_value=[status])
        manager.engine = MagicMock()
        manager.engine.wait_all.side_effect = RuntimeError("wait failed")
        manager.conclude_failure = MagicMock()

        is_quiescent = manager._process_transfer_chunk(_chunk())

        self.assertFalse(is_quiescent)
        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        self.assertEqual(queued_chunk.room, 11)
        self.assertEqual(queued_statuses, [status])
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11,
            failure_reason="Transfer completion failed: RuntimeError('wait failed')",
        )

    def test_partial_submission_failure_transfers_ownership_to_drainer(self):
        manager = _manager()
        manager.request_status[11] = KVPoll.Transferring
        status = MagicMock()

        def submit(*args, **kwargs):
            manager._record_submitted_statuses([status])
            raise RuntimeError("later target failed")

        manager._submit_kv_transfer = submit
        manager.conclude_failure = MagicMock()
        chunk = _chunk()
        manager._mark_transfer_started(chunk)

        with self.assertRaises(_MoriTransferSubmissionError) as raised:
            manager._process_transfer_chunk(chunk)
        manager._handle_submission_failure(chunk, raised.exception)

        queued_chunk, queued_statuses = manager._drain_queue.get_nowait()
        self.assertIs(queued_chunk, chunk)
        self.assertEqual(queued_statuses, [status])
        self.assertEqual(manager._staging_outstanding[11], 1)
        manager.conclude_failure.assert_called_once_with(
            bootstrap_room=11,
            failure_reason="Transfer submission failed: later target failed",
        )

    def test_drain_worker_retries_same_item_after_exception(self):
        manager = _manager()
        manager._wait_poll_ms = 0
        chunk = _chunk()
        status = MagicMock()
        manager._drain_queue = MagicMock()
        manager._drain_queue.get.side_effect = (
            (chunk, [status]),
            KeyboardInterrupt(),
        )
        manager._drain_transfer_statuses = MagicMock(
            side_effect=(RuntimeError("transient wait failure"), None)
        )

        with patch("sglang.srt.disaggregation.mori.conn.logger.exception") as log:
            with patch("sglang.srt.disaggregation.mori.conn.time.sleep"):
                with self.assertRaises(KeyboardInterrupt):
                    manager._drain_worker()

        self.assertEqual(manager._drain_transfer_statuses.call_count, 2)
        log.assert_called_once()


class TestMoriAuxLayout(CustomTestCase):
    def test_prefill_refuses_a_decode_with_another_aux_layout(self):
        """Mori pairs metadata buffers by position. Where the layout differs
        across builds, a decode that reports other buffers, or none because its
        build predates the report, must fail the transfer before any KV is
        written. A buffer count alone cannot tell: an older decode's KV checksum
        buffer makes up for the buffer it lacks."""
        item_lens = [8] * 15 + [64]
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.transfer_lock = threading.Lock()
        manager.kv_args = SimpleNamespace(
            aux_data_ptrs=[0] * 16, aux_item_lens=item_lens
        )
        manager.customized_info_enabled = False
        manager.pp_group = SimpleNamespace(is_last_rank=True)
        manager.state_mem_descs = []
        manager.update_status = MagicMock()
        info = SimpleNamespace(
            engine_key="decode",
            is_dummy=False,
            dst_kv_indices=np.arange(4, dtype=np.int32),
            dst_aux_index=0,
        )
        manager.transfer_infos = {11: {"decode": info}}

        def submit(dst_aux_item_lens, num_remote=16, exchange_aux_item_lens=True):
            manager.exchange_aux_item_lens = exchange_aux_item_lens
            manager.request_status = {11: KVPoll.WaitingForInput}
            manager.decode_kv_args_table = {
                "decode": SimpleNamespace(
                    dst_aux_mem_descs=[object()] * num_remote,
                    dst_aux_item_lens=dst_aux_item_lens,
                    requires_dcp_relayout=False,
                )
            }
            manager.send_kvcache = MagicMock(return_value=[])
            manager.send_aux = MagicMock(return_value=[])
            manager._submit_kv_transfer(
                11, np.arange(4, dtype=np.int32), slice(0, 4), True, aux_index=0
            )

        with get_context().override_server_args(
            disaggregation_enable_kv_checksum=False
        ):
            for name, kwargs in (
                ("older decode, same count", {"dst_aux_item_lens": None}),
                ("shorter row", {"dst_aux_item_lens": [8] * 14 + [4, 64]}),
                (
                    "one buffer fewer",
                    {"dst_aux_item_lens": item_lens[:15], "num_remote": 15},
                ),
                (
                    "two buffers more",
                    {"dst_aux_item_lens": item_lens + [64, 64], "num_remote": 18},
                ),
            ):
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "layout mismatch"),
                ):
                    submit(**kwargs)
                manager.send_kvcache.assert_not_called()
                manager.send_aux.assert_not_called()
            for name, kwargs in (
                ("same layout", {"dst_aux_item_lens": item_lens}),
                (
                    "decode adds the KV checksum buffer",
                    {"dst_aux_item_lens": item_lens + [64], "num_remote": 17},
                ),
                (
                    "nothing exchanged",
                    {"dst_aux_item_lens": None, "exchange_aux_item_lens": False},
                ),
            ):
                with self.subTest(name=name):
                    submit(**kwargs)
                    manager.send_aux.assert_called_once()

    def test_rdma_aux_write_lands_on_the_decode_row(self):
        """A decode may register longer rows than the prefill sends (a larger
        --disaggregation-max-top-logprobs-num), so the remote offset of a row is
        the decode's row size times the row index, not the prefill's."""
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.aux_mem_descs = ["src0", "src1"]
        manager.kv_args = SimpleNamespace(aux_item_lens=[8, 16])
        manager.engine = MagicMock()
        manager.engine.batch_write.return_value = []
        manager._record_submitted_statuses = MagicMock()

        def remote_offsets_and_sizes(dst_aux_item_lens):
            manager.engine.batch_write.reset_mock()
            manager.send_aux_rdma(
                SimpleNamespace(
                    dst_aux_mem_descs=["dst0", "dst1"],
                    dst_aux_item_lens=dst_aux_item_lens,
                ),
                prefill_aux_index=2,
                dst_aux_index=3,
                room=11,
            )
            args = manager.engine.batch_write.call_args.args
            self.assertEqual(args[1], [[16], [32]])
            return args[3], args[4]

        self.assertEqual(
            remote_offsets_and_sizes([8, 40]), ([[24], [120]], [[8], [16]])
        )
        self.assertEqual(remote_offsets_and_sizes(None), ([[24], [48]], [[8], [16]]))

    def test_decode_drops_an_aux_item_longer_than_its_buffer(self):
        """A prefill with another buffer layout sends another buffer's item to this
        position. A longer one must not be written: it would run into the next
        request's row, or past the buffer."""
        rows = (ctypes.c_ubyte * 16)()
        manager = MoriKVManager.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(
            aux_data_ptrs=[ctypes.addressof(rows)], aux_item_lens=[8]
        )

        def receive(data):
            manager._handle_aux_data(
                [
                    MoriKVManager.AUX_DATA_HEADER,
                    b"11",
                    b"0",
                    b"0",
                    struct.pack(">I", len(data)),
                    data,
                ]
            )

        with patch("sglang.srt.disaggregation.mori.conn.logger.error") as log:
            receive(b"\x01" * 16)
        log.assert_called_once()
        self.assertEqual(bytes(rows), bytes(16))
        receive(b"\x02" * 8)
        self.assertEqual(bytes(rows), b"\x02" * 8 + bytes(8))


if __name__ == "__main__":
    unittest.main()
