import threading
import unittest
from unittest.mock import MagicMock, call

import zmq

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCommonKVManagerConnect(unittest.TestCase):
    def test_status_socket_keeps_automatic_reconnect_enabled(self):
        manager = object.__new__(CommonKVManager)
        manager._socket_lock = threading.Lock()
        manager._socket_cache = {}
        manager._monitor_cache = {}
        manager._zmq_ctx = MagicMock()
        socket = manager._zmq_ctx.socket.return_value

        manager._connect("tcp://127.0.0.1:12345")

        self.assertNotIn(call(zmq.RECONNECT_IVL, -1), socket.setsockopt.call_args_list)


if __name__ == "__main__":
    unittest.main()
