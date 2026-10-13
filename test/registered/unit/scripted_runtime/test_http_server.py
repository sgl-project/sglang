from __future__ import annotations

import threading
import unittest
import uuid

import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.http_server import (
    CANARY_LAUNCH_DEFAULTS,
    DEFAULT_RUN_TIMEOUT_S,
    ScriptedHttpServer,
    _canary_decode_graph_override,
)
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _sample_script(ctx, *args):
    yield


_EXPECTED_FN_PATH = f"{_sample_script.__module__}:{_sample_script.__qualname__}"
_NO_REPLY = object()


class _PairSocketHarness:
    def __init__(self, *, reply: object = _NO_REPLY) -> None:
        self._ctx = zmq.Context()
        self.server_socket = self._ctx.socket(zmq.PAIR)
        self._peer_socket = self._ctx.socket(zmq.PAIR)
        self._endpoint = f"inproc://scripted-http-server-{uuid.uuid4().hex}"
        self.server_socket.bind(self._endpoint)
        self._peer_socket.connect(self._endpoint)
        self._reply = reply
        self.sent: list = []
        self._thread = None
        if reply is not _NO_REPLY:
            self._thread = threading.Thread(target=self._reply_once, daemon=True)
            self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.server_socket.close(0)
        self._peer_socket.close(0)
        self._ctx.term()

    def _reply_once(self):
        self.sent.append(self._peer_socket.recv_pyobj())
        self._peer_socket.send_pyobj(self._reply)

    def assert_no_sent_message(self, test_case: unittest.TestCase) -> None:
        test_case.assertFalse(self._peer_socket.poll(50))


class _FakeProcess:
    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _make_server(socket: zmq.Socket, process: _FakeProcess) -> ScriptedHttpServer:
    server = ScriptedHttpServer.__new__(ScriptedHttpServer)
    server._socket = socket
    server._server_process = process
    server._dirty = None
    server._run_timeout_s = DEFAULT_RUN_TIMEOUT_S
    return server


class TestExecuteScriptReplyMatching(CustomTestCase):
    def test_returns_on_script_succeeded(self):
        with _PairSocketHarness(reply=ScriptSucceeded()) as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))

            server.execute_script(_sample_script)

            self.assertEqual(pair.sent, [RunScript(fn_path=_EXPECTED_FN_PATH, args=())])

    def test_forwards_args_in_run_script(self):
        with _PairSocketHarness(reply=ScriptSucceeded()) as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))

            server.execute_script(_sample_script, args=(1, "two"))

            self.assertEqual(
                pair.sent, [RunScript(fn_path=_EXPECTED_FN_PATH, args=(1, "two"))]
            )

    def test_script_failed_reply_raises_assertion_with_traceback(self):
        with _PairSocketHarness(
            reply=ScriptFailed(traceback="REMOTE-TB-MARKER")
        ) as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))

            with self.assertRaisesRegex(AssertionError, "REMOTE-TB-MARKER"):
                server.execute_script(_sample_script)

    def test_unexpected_reply_raises_runtime_error(self):
        with _PairSocketHarness(reply=HookReady()) as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))

            with self.assertRaisesRegex(RuntimeError, "unexpected message"):
                server.execute_script(_sample_script)


class TestExecuteScriptNoReply(CustomTestCase):
    def test_timeout_when_process_still_alive(self):
        with _PairSocketHarness() as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))

            with self.assertRaisesRegex(TimeoutError, "timed out"):
                server.execute_script(_sample_script, timeout_s=0.01)
            self.assertIn("timed out", server._dirty)

    def test_runtime_error_when_process_died(self):
        with _PairSocketHarness() as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=False))

            with self.assertRaisesRegex(RuntimeError, "died before responding"):
                server.execute_script(_sample_script, timeout_s=0.01)
            self.assertIn("died before responding", server._dirty)


class TestExecuteScriptDirtyGuard(CustomTestCase):
    def test_refuses_to_run_when_already_dirty(self):
        with _PairSocketHarness() as pair:
            server = _make_server(pair.server_socket, _FakeProcess(alive=True))
            server._dirty = "prior timeout"

            with self.assertRaisesRegex(RuntimeError, "dirty"):
                server.execute_script(_sample_script)
            pair.assert_no_sent_message(self)


class TestCanaryLaunchKwargs(CustomTestCase):
    def test_canary_is_enabled_without_consulting_the_device(self):
        self.assertEqual(CANARY_LAUNCH_DEFAULTS["kv_canary"], "raise")
        self.assertEqual(CANARY_LAUNCH_DEFAULTS["kv_canary_real_data"], "partial")

    def test_cuda_keeps_its_decode_graph(self):
        self.assertEqual(
            _canary_decode_graph_override(kv_canary="raise", device="cuda"), {}
        )

    def test_a_disabled_canary_keeps_the_decode_graph(self):
        for device in ("cuda", "xpu", "cpu"):
            with self.subTest(device=device):
                self.assertEqual(
                    _canary_decode_graph_override(kv_canary="none", device=device), {}
                )


if __name__ == "__main__":
    unittest.main()
