"""CPU unit tests for fatal Mooncake transfer-worker failures."""

import errno
import os
import signal
import unittest
from unittest.mock import MagicMock, patch

import zmq

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeTransferWorkerFailure(CustomTestCase):
    @staticmethod
    def _make_manager():
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.bootstrap_port = 8998
        return manager

    @patch("sglang.srt.disaggregation.mooncake.conn.psutil.Process")
    def test_worker_failure_signals_original_parent(self, mock_process):
        parent = MagicMock()
        adoptive_parent = MagicMock()
        mock_process.return_value.parent.return_value = parent
        queue = MagicMock()

        def fail_after_reparenting():
            mock_process.return_value.parent.return_value = adoptive_parent
            raise RuntimeError("transfer failed")

        queue.get.side_effect = fail_after_reparenting

        self._make_manager().transfer_worker(queue, MagicMock(), worker_index=2)

        parent.send_signal.assert_called_once_with(signal.SIGQUIT)
        adoptive_parent.send_signal.assert_not_called()

    @patch("sglang.srt.disaggregation.mooncake.conn.os.kill")
    @patch("sglang.srt.disaggregation.mooncake.conn.psutil.Process")
    def test_worker_failure_signals_self_without_parent(self, mock_process, mock_kill):
        mock_process.return_value.parent.return_value = None
        queue = MagicMock()
        queue.get.side_effect = RuntimeError("transfer failed")

        self._make_manager().transfer_worker(queue, MagicMock(), worker_index=1)

        mock_kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)


class TestDecodeNotificationFailure(CustomTestCase):
    """Decode-notification timeouts must fail the request, not the rank.

    Regression guard: these notify sends used to propagate ``zmq.Again`` once
    a dead decode endpoint's socket queue filled into ``transfer_worker``'s
    fatal except-block, so one unreachable decode endpoint could SIGQUIT every
    prefill rank that kept syncing status to it.
    """

    @staticmethod
    def _make_manager(send_error):
        manager = object.__new__(MooncakeKVManager)
        manager._send_multipart_locked = MagicMock(side_effect=send_error)
        return manager

    def test_sync_status_swallows_zmq_send_timeout(self):
        manager = self._make_manager(zmq.Again())

        manager.sync_status_to_decode_endpoint(
            remote="127.0.0.1", dst_port=30000, room=7, status=2, prefill_rank=0
        )

        manager._send_multipart_locked.assert_called_once()

    def test_sync_status_propagates_process_wide_zmq_error(self):
        # Resource/context failures are not evidence of a dead endpoint and
        # must still reach transfer_worker's fatal handler.
        manager = self._make_manager(zmq.ZMQError(errno.EMFILE))

        with self.assertRaises(zmq.ZMQError) as error:
            manager.sync_status_to_decode_endpoint(
                remote="127.0.0.1", dst_port=30000, room=7, status=2, prefill_rank=0
            )
        self.assertEqual(error.exception.errno, errno.EMFILE)

    def test_chunk_ready_zmq_send_timeout_reports_failure(self):
        manager = self._make_manager(zmq.Again())
        req = MagicMock(
            endpoint="127.0.0.1",
            dst_port=30000,
            room=7,
            mooncake_session_id="session-0",
        )
        kv_chunk = MagicMock(index_slice=slice(0, 4), prefill_kv_indices=[0, 1, 2, 3])

        self.assertFalse(
            manager._send_chunk_ready(req, 0, kv_chunk, prefill_unique_rank=0)
        )

    def test_send_aux_tcp_zmq_send_timeout_reports_failure(self):
        manager = self._make_manager(zmq.Again())
        manager.kv_args = MagicMock(aux_data_ptrs=[1], aux_item_lens=[8])
        req = MagicMock(endpoint="127.0.0.1", dst_port=30000, room=7, dst_aux_index=0)

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.AuxDataCodec"
        ) as mock_codec:
            mock_codec.serialize_data_from_buffer.return_value = b"payload"
            self.assertEqual(
                manager.send_aux_tcp(req, prefill_aux_index=0, dst_aux_ptrs=[1]), -1
            )


if __name__ == "__main__":
    unittest.main()
