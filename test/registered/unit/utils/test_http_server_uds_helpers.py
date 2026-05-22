"""Unit tests for UDS-related private helpers in http_server.

Usage:
    python3 -m pytest test/registered/unit/utils/test_http_server_uds_helpers.py -v
"""

import errno
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.entrypoints.http_server import (
    _format_listen_addr,
    _prepare_uds_path,
    _uvicorn_bind_kwargs,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

try:
    import granian  # noqa: F401

    _HAS_GRANIAN = True
except ImportError:
    _HAS_GRANIAN = False

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
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        os.rmdir(self._tmpdir)

    def test_missing_path_is_noop(self):
        _prepare_uds_path(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_regular_file_refused(self):
        with open(self.path, "w") as f:
            f.write("not a socket")
        with self.assertRaises(FileExistsError):
            _prepare_uds_path(self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_symlink_to_socket_refused(self):
        # lstat reports the symlink itself (S_IFLNK), not the target. This
        # blocks the symlink-to-arbitrary-path attack vector: a hostile
        # --uds /tmp/foo.sock that points at /etc/passwd would otherwise be
        # at risk of being unlinked when treated as a "stale socket".
        target_dir = tempfile.mkdtemp(prefix="sglang-uds-target-")
        target_sock_path = os.path.join(target_dir, "real.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(target_sock_path)
        try:
            os.symlink(target_sock_path, self.path)
            with self.assertRaises(FileExistsError):
                _prepare_uds_path(self.path)
            self.assertTrue(os.path.islink(self.path))
        finally:
            srv.close()
            try:
                os.unlink(target_sock_path)
            except FileNotFoundError:
                pass
            os.rmdir(target_dir)

    def test_live_socket_refused(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.listen(1)
        try:
            with self.assertRaises(OSError) as cm:
                _prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)
            self.assertTrue(os.path.exists(self.path))
        finally:
            srv.close()

    def test_stale_socket_unlinked(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()
        self.assertTrue(os.path.exists(self.path))
        _prepare_uds_path(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_timeout_treated_as_live(self):
        # Stale socket file present; probe times out. Conservative refusal
        # (raise EADDRINUSE) is preferable to clobbering a slow listener.
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

        # Patch the http_server module's view of socket.socket, not the stdlib
        # root: future refactors that change how http_server names socket
        # won't silently bypass this test.
        with patch(
            "sglang.srt.entrypoints.http_server.socket.socket",
            side_effect=_TimeoutSocket,
        ):
            with self.assertRaises(OSError) as cm:
                _prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)

        self.assertTrue(os.path.exists(self.path))
        self.assertTrue(connect_called, "probe.connect was not invoked")

    def test_probe_permission_error_propagates(self):
        # Probe-time PermissionError surfaces to the operator unchanged; we
        # refuse to clobber-or-classify a socket we can't even connect to.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_socket = socket.socket

        class _PermSocket:
            def __init__(self, *args, **kwargs):
                self._inner = original_socket(*args, **kwargs)

            def settimeout(self, t):
                self._inner.settimeout(t)

            def connect(self, addr):
                raise PermissionError(errno.EACCES, "permission denied")

            def close(self):
                self._inner.close()

        with patch(
            "sglang.srt.entrypoints.http_server.socket.socket",
            side_effect=_PermSocket,
        ):
            with self.assertRaises(PermissionError):
                _prepare_uds_path(self.path)

        self.assertTrue(os.path.exists(self.path))

    def test_unlink_failure_preserves_path_context(self):
        # Stale-socket detection succeeds; the unlink itself fails (e.g. the
        # parent directory is on a read-only FS or we lost permission since
        # lstat). The error must name the path and the underlying errno so
        # the operator can recover without re-reading the trace.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_unlink = os.unlink

        def _failing_unlink(p):
            if p == self.path:
                raise PermissionError(errno.EACCES, "permission denied", p)
            return original_unlink(p)

        with patch(
            "sglang.srt.entrypoints.http_server.os.unlink",
            side_effect=_failing_unlink,
        ):
            with self.assertRaises(OSError) as cm:
                _prepare_uds_path(self.path)

        self.assertEqual(cm.exception.errno, errno.EACCES)
        self.assertIn(self.path, str(cm.exception))
        self.assertIn("stale UDS file", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, PermissionError)
        self.assertTrue(os.path.exists(self.path))

    def test_unlink_race_tolerated(self):
        # A concurrent process can win the unlink race after our liveness
        # probe says "stale". Treat that FileNotFoundError as success.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_unlink = os.unlink

        def _vanishing_unlink(p):
            if p == self.path:
                # Pretend a peer process already removed it.
                original_unlink(p)
                raise FileNotFoundError(errno.ENOENT, "no such file", p)
            return original_unlink(p)

        with patch(
            "sglang.srt.entrypoints.http_server.os.unlink",
            side_effect=_vanishing_unlink,
        ):
            _prepare_uds_path(self.path)

        self.assertFalse(os.path.exists(self.path))


@unittest.skipUnless(_HAS_GRANIAN, "granian not installed (pip install sglang[http2])")
class TestRunGranianServerKwargs(unittest.TestCase):
    # Granian-side regression for the same UDS plumbing the uvicorn paths
    # use. Without this, the `--enable-http2 --uds /path` combination has
    # only manual-verification coverage.

    def test_passes_uds_when_set(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        args = ServerArgs(model_path="dummy", uds="/tmp/x.sock", enable_http2=True)
        with patch("granian.Granian") as MockGranian:
            _run_granian_server(args)

        self.assertEqual(MockGranian.call_count, 1)
        kwargs = MockGranian.call_args.kwargs
        self.assertEqual(kwargs.get("uds"), Path("/tmp/x.sock"))
        self.assertNotIn("address", kwargs)
        self.assertNotIn("port", kwargs)
        MockGranian.return_value.serve.assert_called_once()

    def test_passes_host_port_when_uds_unset(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        args = ServerArgs(model_path="dummy", enable_http2=True)
        with patch("granian.Granian") as MockGranian:
            _run_granian_server(args)

        kwargs = MockGranian.call_args.kwargs
        self.assertEqual(kwargs.get("address"), args.host)
        self.assertEqual(kwargs.get("port"), args.port)
        self.assertNotIn("uds", kwargs)


if __name__ == "__main__":
    unittest.main()
