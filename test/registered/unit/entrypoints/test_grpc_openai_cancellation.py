"""Regression test for native gRPC OpenAI cancellation propagation."""

import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_SUBPROCESS_TEST = textwrap.dedent(
    r"""
    import json
    import socket
    import sys
    import threading
    import time
    from types import SimpleNamespace

    import grpc

    from sglang.srt.rust_extensions import load_rust_extension


    _started = threading.Event()
    _disconnected = threading.Event()
    _abort_rids = []


    class _FakeRuntime:
        def __init__(self):
            self.tokenizer_manager = SimpleNamespace(
                server_args=SimpleNamespace(),
                model_config=SimpleNamespace(context_len=0),
            )

        def submit_openai_chat(
            self,
            *,
            json_body,
            chunk_callback,
            trace_headers=None,
            is_disconnected_fn=None,
        ):
            _started.set()

            def wait_for_disconnect():
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if is_disconnected_fn is not None and is_disconnected_fn():
                        _disconnected.set()
                        return
                    time.sleep(0.01)

            threading.Thread(target=wait_for_disconnect, daemon=True).start()

        def abort(self, rid="", abort_all=False):
            _abort_rids.append((rid, abort_all))


    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


    def _openai_request_serializer(body):
        encoded = json.dumps(body).encode()
        size = len(encoded)
        prefix = bytearray()
        while size >= 128:
            prefix.append((size & 127) | 128)
            size >>= 7
        prefix.append(size)
        return b"\x0a" + prefix + encoded


    scenario = sys.argv[1]
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
        response_timeout_secs=1 if scenario == "server-timeout" else 10,
    )
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        call = channel.unary_stream(
            "/sglang.runtime.v1.SglangService/ChatComplete",
            request_serializer=_openai_request_serializer,
            response_deserializer=lambda payload: payload,
        )(
            {
                "model": "fake",
                "messages": [{"role": "user", "content": "hello"}],
                "rid": "engine-request-A",
                "stream": True,
            },
            timeout=10,
        )
        if not _started.wait(timeout=5):
            raise RuntimeError("OpenAI request did not reach the Python runtime")
        if scenario == "client-cancel":
            if not call.cancel():
                raise RuntimeError("Client cancellation did not take effect")
        else:
            try:
                next(call)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise AssertionError(
                        f"unexpected server timeout status: {exc.code()}"
                    ) from exc
            else:
                raise AssertionError("server timeout unexpectedly returned a response")
        if not _disconnected.wait(timeout=5):
            raise AssertionError(
                f"native {scenario} did not reach is_disconnected"
            )
        if _abort_rids:
            raise AssertionError(
                f"native cancellation used the wrong Python abort path: {_abort_rids!r}"
            )
    finally:
        channel.close()
        if handle.is_alive():
            handle.shutdown()
    """
).strip()


class TestNativeGrpcOpenAICancellation(CustomTestCase):
    def _run_subprocess(self, scenario):
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_TEST, scenario],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                f"native gRPC OpenAI {scenario} subprocess timed out\n"
                f"--- stdout ---\n{exc.stdout}\n"
                f"--- stderr ---\n{exc.stderr}"
            )
        self.assertEqual(
            completed.returncode,
            0,
            f"native gRPC OpenAI {scenario} regression subprocess failed\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}",
        )

    def test_client_disconnect_reaches_openai_python_request(self):
        self._run_subprocess("client-cancel")

    def test_response_timeout_reaches_openai_python_request(self):
        self._run_subprocess("server-timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
