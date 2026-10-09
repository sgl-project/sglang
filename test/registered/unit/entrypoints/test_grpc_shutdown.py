"""Regression test for native gRPC shutdown while a Python RPC is active."""

import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_SUBPROCESS_TEST = textwrap.dedent(
    r"""
    import socket
    import threading
    import time
    from types import SimpleNamespace

    import grpc

    from sglang.srt.rust_extensions import load_rust_extension


    _entered_health_check = threading.Event()
    _release_health_check = threading.Event()


    class _FakeRuntime:
        def __init__(self):
            self.tokenizer_manager = SimpleNamespace(
                server_args=SimpleNamespace(),
                model_config=SimpleNamespace(context_len=0),
            )

        def health_check(self):
            _entered_health_check.set()
            _release_health_check.wait(timeout=10)
            return True


    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


    port = _free_port()
    grpc_native = load_rust_extension(
        "sglang.srt.rust_extensions._grpc",
        mode="never",
    )
    handle = grpc_native.start_server(
        host="127.0.0.1",
        port=port,
        runtime_handle=_FakeRuntime(),
        worker_threads=1,
        response_timeout_secs=10,
    )
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        health_check = channel.unary_unary(
            "/sglang.runtime.v1.SglangService/HealthCheck",
            request_serializer=lambda payload: payload,
            response_deserializer=lambda payload: payload,
        )
        response = health_check.future(b"")
        if not _entered_health_check.wait(timeout=5):
            raise RuntimeError("HealthCheck did not reach the Python runtime")

        def release_health_check():
            time.sleep(0.1)
            _release_health_check.set()

        threading.Thread(target=release_health_check, daemon=True).start()
        handle.shutdown()
        assert response.result(timeout=5) == b"\x08\x01"
    finally:
        channel.close()
        if handle.is_alive():
            handle.shutdown()
    """
).strip()


class TestNativeGrpcShutdown(CustomTestCase):
    def test_shutdown_releases_gil_before_join(self):
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_TEST],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                "native gRPC shutdown subprocess timed out; likely GIL deadlock\n"
                f"--- stdout ---\n{exc.stdout}\n"
                f"--- stderr ---\n{exc.stderr}"
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"native gRPC shutdown regression subprocess failed\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
