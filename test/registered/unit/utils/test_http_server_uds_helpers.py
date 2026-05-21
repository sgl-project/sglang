"""Unit tests for UDS-related private helpers in http_server.

Usage:
    python3 -m pytest test/registered/unit/utils/test_http_server_uds_helpers.py -v
"""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.http_server import (
    _format_listen_addr,
    _uvicorn_bind_kwargs,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

# Mock get_device() so the test imports ServerArgs on CPU-only runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestUvicornBindKwargs(unittest.TestCase):
    def test_returns_host_port_when_uds_unset(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(
            _uvicorn_bind_kwargs(args),
            {"host": args.host, "port": args.port},
        )

    def test_returns_uds_when_set(self):
        args = ServerArgs(model_path="dummy", uds="/tmp/x.sock")
        self.assertEqual(_uvicorn_bind_kwargs(args), {"uds": "/tmp/x.sock"})


class TestFormatListenAddr(unittest.TestCase):
    def test_formats_host_port(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(
            _format_listen_addr(args), f"{args.host}:{args.port}"
        )

    def test_formats_uds(self):
        args = ServerArgs(model_path="dummy", uds="/run/sglang.sock")
        self.assertEqual(_format_listen_addr(args), "unix:/run/sglang.sock")


if __name__ == "__main__":
    unittest.main()
