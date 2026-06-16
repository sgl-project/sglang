from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def _sample_script(ctx, *args):
    yield


_EXPECTED_FN_PATH = f"{_sample_script.__module__}:{_sample_script.__qualname__}"


class _FakePairSocket:

    def __init__(self, *, poll_result: bool, reply: object = None) -> None:
        self._poll_result = poll_result
        self._reply = reply
        self.sent: list = []

    def send_pyobj(self, obj: object) -> None:
        self.sent.append(obj)

    def poll(self, timeout_ms: int) -> bool:
        return self._poll_result

    def recv_pyobj(self) -> object:
        return self._reply


class _FakeProcess:

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _make_server(socket: _FakePairSocket, process: _FakeProcess) -> ScriptedHttpServer:
    server = ScriptedHttpServer.__new__(ScriptedHttpServer)
    server._socket = socket
    server._server_process = process
    server._dirty = None
    return server


class TestExecuteScriptReplyMatching(CustomTestCase):

    def test_returns_on_script_succeeded(self):
        socket = _FakePairSocket(poll_result=True, reply=ScriptSucceeded())
        server = _make_server(socket, _FakeProcess(alive=True))

        server.execute_script(_sample_script)

        self.assertEqual(socket.sent, [RunScript(fn_path=_EXPECTED_FN_PATH, args=())])

    def test_forwards_args_in_run_script(self):
        socket = _FakePairSocket(poll_result=True, reply=ScriptSucceeded())
        server = _make_server(socket, _FakeProcess(alive=True))

        server.execute_script(_sample_script, args=(1, "two"))

        self.assertEqual(
            socket.sent, [RunScript(fn_path=_EXPECTED_FN_PATH, args=(1, "two"))]
        )

    def test_script_failed_reply_raises_assertion_with_traceback(self):
        socket = _FakePairSocket(
            poll_result=True, reply=ScriptFailed(traceback="REMOTE-TB-MARKER")
        )
        server = _make_server(socket, _FakeProcess(alive=True))

        with self.assertRaisesRegex(AssertionError, "REMOTE-TB-MARKER"):
            server.execute_script(_sample_script)

    def test_unexpected_reply_raises_runtime_error(self):
        socket = _FakePairSocket(poll_result=True, reply=HookReady())
        server = _make_server(socket, _FakeProcess(alive=True))

        with self.assertRaisesRegex(RuntimeError, "unexpected message"):
            server.execute_script(_sample_script)


class TestExecuteScriptNoReply(CustomTestCase):

    def test_timeout_when_process_still_alive(self):
        socket = _FakePairSocket(poll_result=False)
        server = _make_server(socket, _FakeProcess(alive=True))

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            server.execute_script(_sample_script, timeout_s=0.01)
        self.assertIn("timed out", server._dirty)

    def test_runtime_error_when_process_died(self):
        socket = _FakePairSocket(poll_result=False)
        server = _make_server(socket, _FakeProcess(alive=False))

        with self.assertRaisesRegex(RuntimeError, "died before responding"):
            server.execute_script(_sample_script, timeout_s=0.01)
        self.assertIn("died before responding", server._dirty)


class TestExecuteScriptDirtyGuard(CustomTestCase):

    def test_refuses_to_run_when_already_dirty(self):
        socket = _FakePairSocket(poll_result=True, reply=ScriptSucceeded())
        server = _make_server(socket, _FakeProcess(alive=True))
        server._dirty = "prior timeout"

        with self.assertRaisesRegex(RuntimeError, "dirty"):
            server.execute_script(_sample_script)
        self.assertEqual(socket.sent, [])


if __name__ == "__main__":
    unittest.main()
