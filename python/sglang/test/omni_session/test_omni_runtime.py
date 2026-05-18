# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang.srt.omni_session.runtime import (
    OmniSegmentState,
    OmniSessionRecord,
    OmniSessionRuntime,
)


def test_omni_session_runtime_opens_streaming_srt_session():
    controller = _FakeSessionController()
    runtime = OmniSessionRuntime(
        model_hooks=_FakeModelHooks(),
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
        model_hooks=_FakeModelHooks(),
        session_controller=controller,
        srt_request_executor=executor,
    )

    runtime.close_session("session-a")

    assert controller.closed == ["session-a"]
    assert executor.idle_cleanup_count == 1


def test_append_ar_input_tokens_preserves_explicit_positions():
    controller = _FakeSessionController()
    srt_session = _FakeSRTSession(prefix_input_ids=[101, 102])
    controller.session_objects["session-a"] = srt_session
    controller.sessions.add("session-a")
    runtime = OmniSessionRuntime(
        model_hooks=_FakeModelHooks(),
        session_controller=controller,
        tokenizer=_FakeTokenizer(),
    )
    record = OmniSessionRecord(
        session_id="session-a",
        state=OmniSegmentState.AR_DECODE,
        anchor_request_id="session-a:ar1",
        context_length=2,
        context_version=1,
    )
    runtime._records["session-a"] = record

    handle = runtime.append_ar_input_tokens(
        record.handle(),
        token_ids=[10, 10, 123],
        position_ids=[34, 35, 36],
        model_state_updates={
            "u1": {
                "open_image_marker": True,
                "generation_position_start": 37,
            }
        },
    )
    req = srt_session.created_requests[-1]

    assert handle.context_length == 5
    assert req.origin_input_ids == [101, 102, 10, 10, 123]
    # 1. session-internal prefix cache must stay reusable for custom-position KV
    assert req.extra_key is None
    assert req.custom_position_ids == [0, 1, 34, 35, 36]
    assert req.omni_srt_position_count == 37
    assert record.omni_model_state["u1"]["generation_position_start"] == 37


class _FakeSessionController:
    def __init__(self):
        self.opened = None
        self.sessions = set()
        self.closed = []
        self.session_objects = {}

    def __contains__(self, session_id):
        return session_id in self.sessions

    def open(self, req):
        self.opened = req
        self.sessions.add(req.session_id)
        return _FakeOpenResult(success=True)

    def close(self, req):
        self.closed.append(req.session_id)
        self.sessions.remove(req.session_id)

    def get(self, session_id):
        return self.session_objects.get(session_id)


class _FakeOpenResult:
    def __init__(self, *, success):
        self.success = success


class _FakeModelHooks:
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


class _FakeSRTSession:
    def __init__(self, *, prefix_input_ids):
        self.prefix_input_ids = list(prefix_input_ids)
        self.created_requests = []

    def create_req(self, recv_req, *, tokenizer, vocab_size):
        req = SimpleNamespace(
            origin_input_ids=self.prefix_input_ids + list(recv_req.input_ids),
            output_ids=[],
            multimodal_inputs=recv_req.mm_inputs,
            sampling_params=recv_req.sampling_params,
            extra_key=recv_req.extra_key,
            to_finish=None,
            custom_position_ids=None,
            omni_srt_position_count=None,
        )
        self.created_requests.append(req)
        return req


class _FakeTokenizer:
    eos_token_id = 0
