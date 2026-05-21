"""Unit tests for UDS-related private helpers in http_server.

Usage:
    python3 -m pytest test/registered/unit/utils/test_http_server_uds_helpers.py -v
"""

import errno
import os
import socket
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.http_server import (
    _format_listen_addr,
    _prepare_uds_path,
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
        self.assertEqual(_format_listen_addr(args), f"{args.host}:{args.port}")

    def test_formats_uds(self):
        args = ServerArgs(model_path="dummy", uds="/run/sglang.sock")
        self.assertEqual(_format_listen_addr(args), "unix:/run/sglang.sock")


class TestPrepareUdsPath(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="sglang-uds-")
        self.path = os.path.join(self._tmpdir, "test.sock")

    def tearDown(self):
        # Best-effort cleanup; tests may already have unlinked.
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        os.rmdir(self._tmpdir)

    def test_missing_path_is_noop(self):
        # No file at path -> returns cleanly, nothing created.
        _prepare_uds_path(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_regular_file_refused(self):
        with open(self.path, "w") as f:
            f.write("not a socket")
        with self.assertRaises(FileExistsError):
            _prepare_uds_path(self.path)
        # File preserved (we refuse to overwrite).
        self.assertTrue(os.path.exists(self.path))

    def test_live_socket_refused(self):
        # Bind a real UDS listener at self.path and leave it open.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.listen(1)
        try:
            with self.assertRaises(OSError) as cm:
                _prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)
            # Socket file still present.
            self.assertTrue(os.path.exists(self.path))
        finally:
            srv.close()

    def test_stale_socket_unlinked(self):
        # Create a socket file and then close it WITHOUT unlinking.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()
        # The socket file exists but nobody is listening.
        self.assertTrue(os.path.exists(self.path))
        _prepare_uds_path(self.path)
        # Stale file removed.
        self.assertFalse(os.path.exists(self.path))

    def test_timeout_treated_as_live(self):
        # Create a real stale socket so lstat/S_ISSOCK pass, then mock
        # socket.socket to return a probe whose connect() raises TimeoutError.
        # _prepare_uds_path must refuse rather than silently unlink the file
        # of a possibly-live server we couldn't probe in time.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_socket = socket.socket
        connect_called = []

        class _TimeoutSocket:
            def __init__(self, *args, **kwargs):
                self._inner = original_socket(*args, **kwargs)

            def settimeout(self, t):
                self._inner.settimeout(t)

            def connect(self, addr):
                connect_called.append(True)
                raise TimeoutError("simulated probe timeout")

            def close(self):
                self._inner.close()

        with patch("socket.socket", side_effect=_TimeoutSocket):
            with self.assertRaises(OSError) as cm:
                _prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)

        # Socket file preserved (we refused to unlink).
        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(connect_called, "probe.connect was not invoked")


if __name__ == "__main__":
    unittest.main()
