"""Exercise standard health checks against the native Rust gRPC listener on CPU."""

import socket
import subprocess
import sys
import unittest
from types import SimpleNamespace

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from sglang.srt.rust_extensions import load_rust_extension
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SERVICE = "sglang.runtime.v1.SglangService"


class _Runtime:
    def __init__(self):
        self.tokenizer_manager = SimpleNamespace()
        self.healthy = False
        self.fail = False
        self.check_calls = 0

    def health_check(self):
        self.check_calls += 1
        if self.fail:
            raise RuntimeError("runtime unavailable")
        return self.healthy


class TestNativeGrpcHealth(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.grpc_native = load_rust_extension("sglang.srt.rust_extensions._grpc")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.runtime = _Runtime()
        cls.server = cls.grpc_native.start_server(
            host="127.0.0.1", port=port, runtime_handle=cls.runtime
        )
        cls.addClassCleanup(cls.server.shutdown)
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        cls.addClassCleanup(cls.channel.close)
        grpc.channel_ready_future(cls.channel).result(timeout=5)
        cls.health = health_pb2_grpc.HealthStub(cls.channel)

    def setUp(self):
        self.runtime.healthy = False
        self.runtime.fail = False
        self.runtime.check_calls = 0

    def test_check_tracks_readiness_and_preserves_native_rpc(self):
        native_check = self.channel.unary_unary(f"/{SERVICE}/HealthCheck")
        for ready in (False, True, False):
            self.runtime.healthy = ready
            expected = (
                health_pb2.HealthCheckResponse.SERVING
                if ready
                else health_pb2.HealthCheckResponse.NOT_SERVING
            )
            for name in ("", SERVICE):
                response = self.health.Check(
                    health_pb2.HealthCheckRequest(service=name), timeout=5
                )
                self.assertEqual(response.status, expected)
            # Native HealthCheck has an empty request and bool healthy at field 1.
            self.assertEqual(
                native_check(b"", timeout=5), b"\x08\x01" if ready else b""
            )

    def test_check_handles_runtime_error_and_unknown_service(self):
        self.runtime.fail = True
        response = self.health.Check(health_pb2.HealthCheckRequest(), timeout=5)
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.NOT_SERVING)
        with self.assertRaises(grpc.RpcError) as caught:
            self.health.Check(
                health_pb2.HealthCheckRequest(service="unknown"), timeout=5
            )
        self.assertEqual(caught.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_watch_is_unimplemented_without_calling_runtime(self):
        for name in ("", SERVICE, "unknown"):
            with self.assertRaises(grpc.RpcError) as caught:
                next(
                    self.health.Watch(
                        health_pb2.HealthCheckRequest(service=name), timeout=5
                    )
                )
            self.assertEqual(caught.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        self.assertEqual(self.runtime.check_calls, 0)

    def test_shutdown_releases_gil_for_inflight_health_check(self):
        # Isolate a potential GIL deadlock so a regression fails with a timeout
        # instead of hanging the entire test process.
        code = r"""
import importlib.util
import socket
import sys
import threading
from types import SimpleNamespace

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

spec = importlib.util.spec_from_file_location("_grpc", sys.argv[1])
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
started, release = threading.Event(), threading.Event()

def health_check():
    started.set()
    release.wait()
    return True

runtime = SimpleNamespace(
    tokenizer_manager=SimpleNamespace(), health_check=health_check
)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
server = native.start_server(host="127.0.0.1", port=port, runtime_handle=runtime)
with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
    stub = health_pb2_grpc.HealthStub(channel)
    pending = stub.Check.future(health_pb2.HealthCheckRequest(), timeout=5)
    assert started.wait(5), "health check did not start"
    threading.Timer(0.1, release.set).start()
    server.shutdown()
    assert not server.is_alive()
"""
        result = subprocess.run(
            [sys.executable, "-c", code, self.grpc_native.__file__],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
