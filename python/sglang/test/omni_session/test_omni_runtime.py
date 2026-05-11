# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang.srt.omni_session.runtime import OmniSessionRuntime


def test_omni_session_runtime_opens_streaming_srt_session():
    controller = _FakeSessionController()
    runtime = OmniSessionRuntime(
        model_policy=_FakeModelPolicy(),
        session_controller=controller,
        capacity_of_str_len=8192,
    )

    runtime._ensure_srt_session("session-a")

    assert controller.opened.session_id == "session-a"
    assert controller.opened.capacity_of_str_len == 8192
    assert controller.opened.streaming is True


def test_omni_session_runtime_drains_srt_executor_after_close():
    controller = _FakeSessionController()
    controller.sessions.add("session-a")
    executor = _FakeSRTExecutor(controller)
    runtime = OmniSessionRuntime(
        model_policy=_FakeModelPolicy(),
        session_controller=controller,
        srt_request_executor=executor,
    )

    runtime.close_session("session-a")

    assert controller.closed == ["session-a"]
    assert executor.idle_cleanup_count == 1


class _FakeSessionController:
    def __init__(self):
        self.opened = None
        self.sessions = set()
        self.closed = []

    def __contains__(self, session_id):
        return session_id in self.sessions

    def open(self, req):
        self.opened = req
        self.sessions.add(req.session_id)
        return _FakeOpenResult(success=True)

    def close(self, req):
        self.closed.append(req.session_id)
        self.sessions.remove(req.session_id)


class _FakeOpenResult:
    def __init__(self, *, success):
        self.success = success


class _FakeModelPolicy:
    def close_session(self, *, session_id):
        self.closed_session_id = session_id


class _FakeSRTExecutor:
    def __init__(self, session_controller):
        self.session_controller = session_controller
        self.idle_cleanup_count = 0

    def close_session_on_scheduler_thread(self, session_id):
        self.session_controller.close(SimpleNamespace(session_id=session_id))

    def run_idle_cleanup(self):
        self.idle_cleanup_count += 1
