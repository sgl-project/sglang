"""Unit tests for best-effort HealthCheckOutput ZMQ send."""

import unittest
from collections import deque
from unittest.mock import MagicMock

import zmq

from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _maybe_send_health_check_signal(scheduler) -> None:
    """Mirror of Scheduler.maybe_send_health_check_signal for unit testing."""
    if not scheduler.return_health_check_ipcs:
        return
    http_worker_ipc = scheduler.return_health_check_ipcs[0]
    if scheduler.ipc_channels.send_to_tokenizer.try_send_output(
        HealthCheckOutput(http_worker_ipc=http_worker_ipc)
    ):
        scheduler.return_health_check_ipcs.popleft()


class TestSenderWrapperTrySendOutput(CustomTestCase):
    def test_try_send_output_returns_false_on_again(self):
        socket = MagicMock()
        socket.send_pyobj.side_effect = zmq.Again
        wrapper = SenderWrapper(socket)

        self.assertFalse(
            wrapper.try_send_output(HealthCheckOutput(http_worker_ipc="ipc://test"))
        )

    def test_try_send_output_returns_true_on_success(self):
        socket = MagicMock()
        wrapper = SenderWrapper(socket)

        self.assertTrue(
            wrapper.try_send_output(HealthCheckOutput(http_worker_ipc="ipc://test"))
        )
        socket.send_pyobj.assert_called_once()
        _, kwargs = socket.send_pyobj.call_args
        self.assertEqual(kwargs.get("flags"), zmq.NOBLOCK)

    def test_send_output_remains_blocking(self):
        socket = MagicMock()
        wrapper = SenderWrapper(socket)

        wrapper.send_output(HealthCheckOutput(http_worker_ipc="ipc://test"))
        socket.send_pyobj.assert_called_once()
        _, kwargs = socket.send_pyobj.call_args
        self.assertEqual(kwargs.get("flags", 0), 0)

    def test_try_send_output_noop_when_socket_is_none(self):
        wrapper = SenderWrapper(None)
        self.assertTrue(wrapper.try_send_output(HealthCheckOutput()))

    def test_try_send_output_propagates_non_again_errors(self):
        socket = MagicMock()
        socket.send_pyobj.side_effect = zmq.ZMQError()
        wrapper = SenderWrapper(socket)

        with self.assertRaises(zmq.ZMQError):
            wrapper.try_send_output(HealthCheckOutput(http_worker_ipc="ipc://test"))


class TestMaybeSendHealthCheckSignal(CustomTestCase):
    def test_skips_tick_without_dropping_ipc_on_again(self):
        scheduler = MagicMock()
        scheduler.return_health_check_ipcs = deque(["ipc://worker-a"])
        scheduler.ipc_channels.send_to_tokenizer.try_send_output.return_value = False

        _maybe_send_health_check_signal(scheduler)

        self.assertEqual(list(scheduler.return_health_check_ipcs), ["ipc://worker-a"])
        scheduler.ipc_channels.send_to_tokenizer.try_send_output.assert_called_once()

    def test_pops_ipc_only_after_successful_send(self):
        scheduler = MagicMock()
        scheduler.return_health_check_ipcs = deque(["ipc://worker-a", "ipc://worker-b"])
        scheduler.ipc_channels.send_to_tokenizer.try_send_output.return_value = True

        _maybe_send_health_check_signal(scheduler)

        self.assertEqual(list(scheduler.return_health_check_ipcs), ["ipc://worker-b"])


if __name__ == "__main__":
    unittest.main()
