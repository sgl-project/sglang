"""CPU unit tests for fatal Mooncake transfer-worker failures."""

import os
import signal
import unittest
from unittest.mock import MagicMock, patch

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
    def test_worker_failure_signals_parent(self, mock_process):
        parent = MagicMock()
        mock_process.return_value.parent.return_value = parent
        queue = MagicMock()
        queue.get.side_effect = RuntimeError("transfer failed")

        self._make_manager().transfer_worker(queue, MagicMock(), worker_index=2)

        parent.send_signal.assert_called_once_with(signal.SIGQUIT)

    @patch("sglang.srt.disaggregation.mooncake.conn.os.kill")
    @patch("sglang.srt.disaggregation.mooncake.conn.psutil.Process")
    def test_worker_failure_signals_self_without_parent(self, mock_process, mock_kill):
        mock_process.return_value.parent.return_value = None
        queue = MagicMock()
        queue.get.side_effect = RuntimeError("transfer failed")

        self._make_manager().transfer_worker(queue, MagicMock(), worker_index=1)

        mock_kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)


if __name__ == "__main__":
    unittest.main()
