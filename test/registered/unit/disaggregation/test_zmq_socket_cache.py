"""CPU unit tests for the disaggregation ZMQ endpoint cache."""

import errno
import threading
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import zmq

from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    _validate_zmq_socket_limits,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestZmqSocketLimits(CustomTestCase):
    def test_monitored_endpoint_uses_three_context_sockets(self):
        context = zmq.Context()
        context.set(zmq.MAX_SOCKETS, 6)
        pull = context.socket(zmq.PULL)
        first = context.socket(zmq.PUSH)
        first_monitor = first.get_monitor_socket(
            zmq.EVENT_DISCONNECTED, addr="inproc://socket-accounting-first"
        )
        second = context.socket(zmq.PUSH)

        try:
            # PULL + the first monitored endpoint use four sockets. The second
            # PUSH and its internal monitor use the remaining two; creating
            # pyzmq's PAIR monitor receiver must exceed the cap.
            with self.assertRaises(zmq.ZMQError) as error:
                second.get_monitor_socket(
                    zmq.EVENT_DISCONNECTED,
                    addr="inproc://socket-accounting-second",
                )
            self.assertEqual(error.exception.errno, errno.EMFILE)
        finally:
            CommonKVManager._close_monitored_socket(second, None)
            CommonKVManager._close_monitored_socket(first, first_monitor)
            pull.close(linger=0)
            context.destroy(linger=0)

    def test_rejects_nonpositive_limits(self):
        for zmq_max_sockets, cache_bound, expected in (
            (0, 1, "ZMQ_MAX_SOCKETS must be greater than zero"),
            (32, 0, "SOCKET_CACHE_MAX_ENDPOINTS must be greater than zero"),
            (32, -1, "SOCKET_CACHE_MAX_ENDPOINTS must be greater than zero"),
        ):
            with self.subTest(zmq_max_sockets=zmq_max_sockets, cache_bound=cache_bound):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_zmq_socket_limits(zmq_max_sockets, cache_bound)

    def test_rejects_cap_below_actual_steady_state_requirement(self):
        with (
            envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.override(6),
            envs.SGLANG_DISAGGREGATION_SOCKET_CACHE_MAX_ENDPOINTS.override(2),
        ):
            with self.assertRaisesRegex(ValueError, "needs at least 7"):
                _validate_zmq_socket_limits(
                    envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.get(),
                    envs.SGLANG_DISAGGREGATION_SOCKET_CACHE_MAX_ENDPOINTS.get(),
                )

    def test_warns_when_cap_lacks_full_replacement_headroom(self):
        with self.assertLogs(
            "sglang.srt.disaggregation.common.conn", level="WARNING"
        ) as logs:
            _validate_zmq_socket_limits(7, 2)
        self.assertIn("full cache replacement (13 sockets)", logs.output[0])

    def test_defaults_cover_full_cache_replacement(self):
        zmq_max_sockets = envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.default
        cache_bound = envs.SGLANG_DISAGGREGATION_SOCKET_CACHE_MAX_ENDPOINTS.default
        self.assertEqual(zmq_max_sockets, 32768)
        self.assertEqual(cache_bound, 4096)
        with self.assertNoLogs(
            "sglang.srt.disaggregation.common.conn", level="WARNING"
        ):
            _validate_zmq_socket_limits(zmq_max_sockets, cache_bound)


class TestZmqSocketCache(CustomTestCase):
    def _make_manager(self, *, zmq_max_sockets: int, cache_bound: int):
        manager = object.__new__(CommonKVManager)
        manager._zmq_ctx = zmq.Context()
        manager._zmq_ctx.set(zmq.MAX_SOCKETS, zmq_max_sockets)
        manager.server_socket = manager._zmq_ctx.socket(zmq.PULL)
        manager._socket_cache_max_endpoints = cache_bound
        manager._socket_cache = OrderedDict()
        manager._monitor_cache = {}
        manager._socket_send_locks = {}
        manager._socket_lock = threading.Lock()
        manager._monitor_endpoint_seq = 0
        self.addCleanup(self._destroy_manager, manager)
        return manager

    @staticmethod
    def _destroy_manager(manager):
        with manager._socket_lock:
            for endpoint in list(manager._socket_cache):
                manager._drop_endpoint_locked(endpoint)
        manager.server_socket.close(linger=0)
        manager._zmq_ctx.destroy(linger=0)

    def test_rapid_endpoint_churn_stays_bounded(self):
        # Seven is the exact steady-state requirement for a PULL socket and
        # two monitored endpoints. Queue a message on every endpoint so an
        # eviction would retain context slots if it used nonzero linger.
        manager = self._make_manager(zmq_max_sockets=7, cache_bound=2)
        endpoints = [f"tcp://127.0.0.1:{20000 + index}" for index in range(200)]

        for endpoint in endpoints:
            manager._send_multipart_locked(endpoint, [b"payload"])

        self.assertEqual(list(manager._socket_cache), endpoints[-2:])
        self.assertEqual(set(manager._monitor_cache), set(endpoints[-2:]))
        self.assertEqual(set(manager._socket_send_locks), set(endpoints[-2:]))
        self.assertGreaterEqual(manager._monitor_endpoint_seq, len(endpoints))

    @patch("sglang.srt.disaggregation.common.conn.time.sleep")
    def test_partial_monitor_creation_is_cleaned_and_retried(self, mock_sleep):
        manager = object.__new__(CommonKVManager)
        manager._zmq_ctx = MagicMock()
        manager._socket_cache_max_endpoints = 4
        manager._socket_cache = OrderedDict()
        manager._monitor_cache = {}
        manager._socket_send_locks = {}
        manager._socket_lock = threading.Lock()
        manager._monitor_endpoint_seq = 0

        failed_socket = MagicMock()
        good_monitor = MagicMock()
        good_socket = MagicMock()
        manager._zmq_ctx.socket.side_effect = [
            failed_socket,
            zmq.ZMQError(errno.EMFILE),
            good_socket,
            good_monitor,
        ]

        returned_socket, _ = manager._connect("tcp://127.0.0.1:29999")

        self.assertIs(returned_socket, good_socket)
        failed_socket.disable_monitor.assert_called_once_with()
        failed_socket.close.assert_called_once_with(linger=0)
        mock_sleep.assert_called_once_with(0.001)
        self.assertEqual(list(manager._socket_cache.values()), [good_socket])
        self.assertEqual(list(manager._monitor_cache.values()), [good_monitor])
        good_monitor.connect.assert_called_once_with(
            good_socket.monitor.call_args.args[0]
        )
        first_addr = failed_socket.monitor.call_args.args[0]
        second_addr = good_socket.monitor.call_args.args[0]
        self.assertNotEqual(first_addr, second_addr)

    def test_send_retries_only_a_socket_closed_by_eviction(self):
        manager = object.__new__(CommonKVManager)
        evicted_socket = MagicMock()
        evicted_socket.send_multipart.side_effect = zmq.ZMQError(errno.ENOTSOCK)
        replacement_socket = MagicMock()
        manager._connect = MagicMock(
            side_effect=[
                (evicted_socket, threading.Lock()),
                (replacement_socket, threading.Lock()),
            ]
        )

        manager._send_multipart_locked("tcp://127.0.0.1:30000", [b"payload"])

        self.assertEqual(manager._connect.call_count, 2)
        replacement_socket.send_multipart.assert_called_once_with([b"payload"])

    def test_send_does_not_retry_unrelated_zmq_error(self):
        manager = object.__new__(CommonKVManager)
        socket = MagicMock()
        socket.send_multipart.side_effect = zmq.ZMQError(errno.EAGAIN)
        manager._connect = MagicMock(return_value=(socket, threading.Lock()))

        with self.assertRaises(zmq.ZMQError):
            manager._send_multipart_locked("tcp://127.0.0.1:30000", [b"payload"])

        manager._connect.assert_called_once_with("tcp://127.0.0.1:30000", is_ipv6=False)


if __name__ == "__main__":
    unittest.main()
