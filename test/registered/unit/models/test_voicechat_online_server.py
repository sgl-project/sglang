import asyncio
from types import SimpleNamespace

import pytest

from examples.voicechat import online_server


class FakeSidecar:
    def __init__(self, events, fail_encode=False):
        self.events = events
        self.fail_encode = fail_encode

    def start(self):
        self.events.append("audio.start")
        return "audio-session"

    def encode(self, session_id, pcm16):
        assert session_id == "audio-session"
        assert len(pcm16) > 0
        self.events.append("audio.encode")
        if self.fail_encode:
            raise RuntimeError("encode failed")
        return [[0.0] * 4480]

    def decode(self, session_id, codes):
        assert session_id == "audio-session"
        assert codes == [1, 2, 3]
        self.events.append("audio.decode")

    def close(self, session_id):
        assert session_id == "audio-session"
        self.events.append("audio.close")


class FakeModelSession:
    def __init__(self, events):
        self.events = events

    async def start(self, prompt_ids, speaker, pad_token_id):
        assert prompt_ids == [10, 11]
        assert speaker.shape == (4, 1152)
        assert pad_token_id == 12
        self.events.append("model.start")

    async def step(self, embedding):
        assert len(embedding[0]) == 4480
        self.events.append("model.step")
        return SimpleNamespace(audio_codes=[1, 2, 3])

    async def close(self):
        self.events.append("model.close")


def make_runtime(events, frames=2, fail_encode=False):
    runtime = online_server.VoiceChatRuntime.__new__(online_server.VoiceChatRuntime)
    runtime.sidecar = FakeSidecar(events, fail_encode=fail_encode)
    runtime.duplex = object()
    runtime.eartts = object()
    runtime.speaker = SimpleNamespace(shape=(4, 1152))
    runtime.config = SimpleNamespace(pad_token_id=12)
    runtime.prompt_ids = lambda _prompt: [10, 11]
    runtime.warmup_frames = frames
    runtime.warmup_duration_ms = None
    runtime.ready = False
    runtime.session_capacity = 8192
    return runtime


def test_warmup_exercises_all_stages_and_releases_sessions(monkeypatch):
    events = []
    model_session = FakeModelSession(events)

    async def create(*_args, **_kwargs):
        events.append("model.create")
        return model_session

    monkeypatch.setattr(online_server.AsyncSGLangVoiceChatSession, "create", create)
    runtime = make_runtime(events)

    asyncio.run(runtime.warmup())

    assert events == [
        "audio.start",
        "model.create",
        "model.start",
        "audio.encode",
        "model.step",
        "audio.decode",
        "audio.encode",
        "model.step",
        "audio.decode",
        "model.close",
        "audio.close",
    ]
    assert runtime.ready
    assert runtime.warmup_duration_ms is not None


def test_warmup_failure_releases_sessions_and_does_not_mark_ready(monkeypatch):
    events = []
    model_session = FakeModelSession(events)

    async def create(*_args, **_kwargs):
        events.append("model.create")
        return model_session

    monkeypatch.setattr(online_server.AsyncSGLangVoiceChatSession, "create", create)
    runtime = make_runtime(events, fail_encode=True)

    with pytest.raises(RuntimeError, match="encode failed"):
        asyncio.run(runtime.warmup())

    assert events[-2:] == ["model.close", "audio.close"]
    assert not runtime.ready


def test_warmup_can_be_disabled():
    runtime = make_runtime([], frames=0)

    asyncio.run(runtime.warmup())

    assert runtime.ready
    assert runtime.warmup_duration_ms is None


def test_runtime_arguments_warm_up_realtime_by_default():
    parser = online_server.argparse.ArgumentParser()
    online_server.add_runtime_arguments(parser)

    args = parser.parse_args(
        ["--duplex-model", "/models/duplex", "--eartts-model", "/models/eartts"]
    )

    assert not args.skip_warmup


def test_app_exposes_realtime_discovery_and_alias_routes():
    app = online_server.create_app(make_runtime([]))
    http_paths = {
        route.path for route in app.routes if "GET" in getattr(route, "methods", set())
    }
    websocket_paths = {
        route.path
        for route in app.routes
        if route.__class__.__name__ == "APIWebSocketRoute"
    }

    assert {"/", "/health", "/v1/realtime/health"} <= http_paths
    assert {"/realtime", "/v1/realtime"} <= websocket_paths


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([], False),
        ([12] * 20, False),
        ([99] + [12] * 12, False),
        ([12] * 8 + [99] + [12] * 11, True),
    ],
)
def test_reply_truncation_guard_only_checks_active_tail(tokens, expected):
    assert online_server._reply_may_be_truncated(tokens, pad_token_id=12) is expected


def test_reply_truncation_guard_requires_positive_window():
    with pytest.raises(ValueError, match="guard_frames must be positive"):
        online_server._reply_may_be_truncated([12], pad_token_id=12, guard_frames=0)
