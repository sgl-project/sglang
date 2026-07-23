"""Unit tests for Mooncake control-plane message handling."""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import (
    _parse_transfer_status_message,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ConcurrentSendDetector:
    def __init__(self):
        self._state_lock = threading.Lock()
        self.active_sends = 0
        self.max_active_sends = 0
        self.messages = []

    def send_multipart(self, frames):
        with self._state_lock:
            self.active_sends += 1
            self.max_active_sends = max(self.max_active_sends, self.active_sends)

        time.sleep(0.005)

        with self._state_lock:
            self.messages.append(frames)
            self.active_sends -= 1


class TestMooncakeControlMessages(CustomTestCase):
    def test_cached_socket_multipart_sends_are_serialized_per_endpoint(self):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager._socket_lock = threading.Lock()
        manager._socket_send_locks = {}

        socket = _ConcurrentSendDetector()
        manager._connect = MagicMock(return_value=socket)
        start = threading.Event()

        def send(index):
            start.wait()
            manager._send_multipart(
                "tcp://decode:1234",
                [str(index).encode("ascii"), b"1", b"0"],
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(send, index) for index in range(8)]
            start.set()
            for future in futures:
                future.result()

        self.assertEqual(socket.max_active_sends, 1)
        self.assertEqual(len(socket.messages), 8)

    def test_parse_valid_transfer_status(self):
        self.assertEqual(
            _parse_transfer_status_message([b"123", b"2", b"7"]),
            (123, 2, 7),
        )

    def test_rejects_merged_transfer_status_messages(self):
        with self.assertLogs("sglang.srt.disaggregation.mooncake.conn", level="ERROR"):
            parsed = _parse_transfer_status_message(
                [b"123", b"2", b"7", b"124", b"2", b"6"]
            )

        self.assertIsNone(parsed)

    def test_rejects_non_numeric_transfer_status(self):
        with self.assertLogs("sglang.srt.disaggregation.mooncake.conn", level="ERROR"):
            parsed = _parse_transfer_status_message([b"123", b"success", b"7"])

        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
