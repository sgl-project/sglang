"""Unit tests for the UDS (--uds) support: ServerArgs validation and helpers.

Usage:
    python3 -m pytest test/registered/unit/utils/test_uds.py -v
"""

import errno
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.uds import (
    format_listen_addr,
    prepare_uds_path,
    uvicorn_bind_kwargs,
)
from sglang.test.ci.ci_register import register_cpu_ci

try:
    import granian  # noqa: F401

    _HAS_GRANIAN = True
except ImportError:
    _HAS_GRANIAN = False

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestServerArgsUDSValidation(unittest.TestCase):
    def test_default_host_port_accepted(self):
        args = ServerArgs(model_path="dummy", uds="/tmp/sglang-test.sock")
        self.assertEqual(args.uds, "/tmp/sglang-test.sock")
        self.assertEqual(args.host, ServerArgs.host)
        self.assertEqual(args.port, ServerArgs.port)

    def test_explicit_host_rejected(self):
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="/tmp/x.sock", host="0.0.0.0")
        self.assertIn("--uds", str(cm.exception))
        self.assertIn("--host", str(cm.exception))

    def test_explicit_port_rejected(self):
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="/tmp/x.sock", port=31000)
        self.assertIn("--uds", str(cm.exception))
        self.assertIn("--port", str(cm.exception))

    def test_windows_rejected(self):
        with patch.object(sys, "platform", "win32"):
            with self.assertRaises(ValueError) as cm:
                ServerArgs(model_path="dummy", uds="/tmp/x.sock")
            self.assertIn("Linux", str(cm.exception))

    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="")
        self.assertIn("non-empty", str(cm.exception))

    def test_relative_path_rejected(self):
        # AF_UNIX accepts relative paths but the cwd is service-launcher
        # dependent (systemd unit's WorkingDirectory, Docker WORKDIR, etc.).
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="sglang.sock")
        self.assertIn("absolute path", str(cm.exception))

    def test_path_too_long_rejected(self):
        # AF_UNIX sun_path is ~104 bytes (macOS) / 108 bytes (Linux). A
        # too-long path otherwise fails with cryptic ENAMETOOLONG from bind.
        long_path = "/tmp/" + "x" * 200 + ".sock"
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds=long_path)
        self.assertIn("sun_path", str(cm.exception))

    def test_parent_dir_missing_rejected(self):
        # bind() does not create the parent directory; pre-check so the
        # operator sees a clear error instead of a bare ENOENT.
        with self.assertRaises(ValueError) as cm:
            ServerArgs(
                model_path="dummy",
                uds="/nonexistent-uds-parent-12345/sglang.sock",
            )
        self.assertIn("parent directory", str(cm.exception))

    def test_grpc_mode_rejected(self):
        # gRPC server binds via its own listener and never reads server_args.uds.
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="/tmp/x.sock", grpc_mode=True)
        self.assertIn("--grpc-mode", str(cm.exception))

    def test_multi_tokenizer_rejected(self):
        # Multi-tokenizer mode binds host/port per worker; UDS would be
        # silently ignored. Reject the combination at config-parse time.
        with self.assertRaises(ValueError) as cm:
            ServerArgs(model_path="dummy", uds="/tmp/x.sock", tokenizer_worker_num=4)
        self.assertIn("--tokenizer-worker-num", str(cm.exception))

    def test_ssl_rejected(self):
        # uvicorn wraps the UDS listener in TLS but the in-process warmup
        # self-call speaks plain HTTP and would fail the TLS handshake.
        # Use real (empty) cert/key files so we hit the UDS+SSL combo check
        # rather than the earlier SSL "file not found" check.
        with tempfile.NamedTemporaryFile(
            suffix=".pem"
        ) as cert, tempfile.NamedTemporaryFile(suffix=".pem") as key:
            with self.assertRaises(ValueError) as cm:
                ServerArgs(
                    model_path="dummy",
                    uds="/tmp/x.sock",
                    ssl_certfile=cert.name,
                    ssl_keyfile=key.name,
                )
            self.assertIn("--ssl-certfile", str(cm.exception))

    def test_cli_flag_parsed(self):
        from sglang.srt.server_args import prepare_server_args
        from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

        args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--uds",
                "/tmp/sglang-cli.sock",
            ]
        )
        self.assertEqual(args.uds, "/tmp/sglang-cli.sock")


class TestBindHelpers(unittest.TestCase):
    def test_uvicorn_bind_kwargs_host_port(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(
            uvicorn_bind_kwargs(args), {"host": args.host, "port": args.port}
        )

    def test_uvicorn_bind_kwargs_uds(self):
        args = ServerArgs(model_path="dummy", uds="/tmp/x.sock")
        self.assertEqual(uvicorn_bind_kwargs(args), {"uds": "/tmp/x.sock"})

    def test_format_listen_addr_host_port(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(format_listen_addr(args), f"{args.host}:{args.port}")

    def test_format_listen_addr_uds(self):
        args = ServerArgs(model_path="dummy", uds="/tmp/sglang.sock")
        self.assertEqual(format_listen_addr(args), "unix:/tmp/sglang.sock")


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
        prepare_uds_path(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_regular_file_refused(self):
        with open(self.path, "w") as f:
            f.write("not a socket")
        with self.assertRaises(FileExistsError):
            prepare_uds_path(self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_symlink_to_socket_refused(self):
        # lstat reports the symlink itself (S_IFLNK), not the target. This
        # blocks a hostile --uds /tmp/foo.sock pointing at /etc/passwd from
        # being treated as a "stale socket" and unlinked.
        target_dir = tempfile.mkdtemp(prefix="sglang-uds-target-")
        target_sock_path = os.path.join(target_dir, "real.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(target_sock_path)
        try:
            os.symlink(target_sock_path, self.path)
            with self.assertRaises(FileExistsError):
                prepare_uds_path(self.path)
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
                prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)
            self.assertTrue(os.path.exists(self.path))
        finally:
            srv.close()

    def test_stale_socket_unlinked(self):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()
        self.assertTrue(os.path.exists(self.path))
        prepare_uds_path(self.path)
        self.assertFalse(os.path.exists(self.path))

    def test_timeout_treated_as_live(self):
        # Stale socket file present; probe times out. Conservative refusal
        # (raise EADDRINUSE) is preferable to clobbering a slow listener.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_socket = socket.socket

        class _TimeoutSocket:
            def __init__(self, *args, **kwargs):
                self._inner = original_socket(*args, **kwargs)

            def settimeout(self, t):
                self._inner.settimeout(t)

            def connect(self, addr):
                raise TimeoutError("simulated probe timeout")

            def close(self):
                self._inner.close()

        with patch("sglang.srt.utils.uds.socket.socket", side_effect=_TimeoutSocket):
            with self.assertRaises(OSError) as cm:
                prepare_uds_path(self.path)
            self.assertEqual(cm.exception.errno, errno.EADDRINUSE)

        self.assertTrue(os.path.exists(self.path))

    def test_probe_permission_error_propagates(self):
        # PermissionError surfaces unchanged so the operator can fix
        # ownership/mode rather than have us silently clobber or refuse.
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

        with patch("sglang.srt.utils.uds.socket.socket", side_effect=_PermSocket):
            with self.assertRaises(PermissionError):
                prepare_uds_path(self.path)

        self.assertTrue(os.path.exists(self.path))

    def test_unlink_failure_preserves_path_context(self):
        # Stale-socket detection succeeds; unlink fails. The error must name
        # the path so the operator can recover.
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.path)
        srv.close()

        original_unlink = os.unlink

        def _failing_unlink(p):
            if p == self.path:
                raise PermissionError(errno.EACCES, "permission denied", p)
            return original_unlink(p)

        with patch("sglang.srt.utils.uds.os.unlink", side_effect=_failing_unlink):
            with self.assertRaises(OSError) as cm:
                prepare_uds_path(self.path)

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

        with patch("sglang.srt.utils.uds.os.unlink", side_effect=_vanishing_unlink):
            prepare_uds_path(self.path)

        self.assertFalse(os.path.exists(self.path))


@unittest.skipUnless(_HAS_GRANIAN, "granian not installed (pip install sglang[http2])")
class TestRunGranianServerKwargs(unittest.TestCase):
    # Granian-side regression for the same UDS plumbing the uvicorn paths
    # use. Without this, --enable-http2 --uds /path has only manual coverage.
    # tokenizer_worker_num=2 forces the multi-worker Granian path (the
    # embedded server is only used when tokenizer_worker_num=1).

    def test_passes_uds_when_set(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                uds="/tmp/x.sock",
                tokenizer_worker_num=2,
            )

        self.assertEqual(MockGranian.call_count, 1)
        kwargs = MockGranian.call_args.kwargs
        self.assertEqual(kwargs.get("uds"), Path("/tmp/x.sock"))
        self.assertNotIn("address", kwargs)
        self.assertNotIn("port", kwargs)
        MockGranian.return_value.serve.assert_called_once()

    def test_passes_host_port_when_uds_unset(self):
        from sglang.srt.entrypoints.http_server import _run_granian_server

        with patch("granian.Granian") as MockGranian:
            _run_granian_server(
                host="127.0.0.1",
                port=30000,
                log_level="info",
                tokenizer_worker_num=2,
            )

        kwargs = MockGranian.call_args.kwargs
        self.assertEqual(kwargs.get("address"), "127.0.0.1")
        self.assertEqual(kwargs.get("port"), 30000)
        self.assertNotIn("uds", kwargs)


if __name__ == "__main__":
    unittest.main()
