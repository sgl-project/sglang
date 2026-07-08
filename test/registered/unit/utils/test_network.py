import unittest
from unittest.mock import patch

import zmq

from sglang.srt.utils.network import config_socket
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeSocket:
    def __init__(self):
        self.options = []

    def setsockopt(self, option, value):
        self.options.append((option, value))


class TestConfigSocket(CustomTestCase):
    def test_uses_default_buffer_when_memory_probe_fails(self):
        socket = FakeSocket()

        with patch(
            "sglang.srt.utils.network.psutil.virtual_memory",
            side_effect=ValueError("invalid literal for int() with base 10: b'kB'"),
        ):
            config_socket(socket, zmq.DEALER)

        self.assertIn((zmq.SNDHWM, 0), socket.options)
        self.assertIn((zmq.RCVHWM, 0), socket.options)
        self.assertIn((zmq.SNDBUF, -1), socket.options)
        self.assertIn((zmq.RCVBUF, -1), socket.options)


if __name__ == "__main__":
    unittest.main()
