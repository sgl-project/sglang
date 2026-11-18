import asyncio
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

# The session's inference imports are unrelated to event validation. Isolate
# only the manager type and Qwen processor's prompt constant; event parsing,
# dispatch, buffer reset and SDK error serialization below run unmodified.
_manager = types.ModuleType("sglang.srt.managers.tokenizer_manager")
_manager.TokenizerManager = object
_processor = types.ModuleType("sglang.srt.multimodal.processors.qwen3_asr")
_processor.DEFAULT_ASR_PROMPT = ""
with patch.dict(
    sys.modules,
    {
        _manager.__name__: _manager,
        _processor.__name__: _processor,
    },
):
    from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _WebSocket:
    def __init__(self, event):
        self.messages = iter(
            [
                {"type": "websocket.receive", "text": json.dumps(event)},
                {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "input_audio_buffer.clear"}),
                },
                {"type": "websocket.disconnect"},
            ]
        )
        self.sent = []

    async def receive(self):
        return next(self.messages)

    async def send_text(self, text):
        self.sent.append(json.loads(text))


class TestRealtimeEventValidation(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.server_args = ServerArgs(model_path="dummy")
        publish(self.server_args, role="tokenizer")

    def _drive(self, event):
        websocket = _WebSocket(event)
        adapter = SimpleNamespace(
            model_sample_rate=16000,
            chunked_streaming_config={
                "chunk_size_sec": 1.0,
                "unfixed_chunk_num": 1,
                "unfixed_token_num": 1,
            },
        )
        session = RealtimeConnection(
            websocket, SimpleNamespace(), adapter, self.server_args
        )
        asyncio.run(session._run_loop())
        return websocket.sent

    def test_non_string_event_type_reports_error_and_keeps_session_usable(self):
        for event_type in ([], {}, None, 1, True):
            with self.subTest(event_type=event_type):
                events = self._drive({"type": event_type, "event_id": "client-1"})
                self.assertEqual(events[0]["type"], "error")
                self.assertEqual(events[0]["error"]["code"], "invalid_value")
                self.assertEqual(events[0]["error"]["param"], "type")
                self.assertEqual(events[0]["error"]["event_id"], "client-1")
                self.assertEqual(events[1]["type"], "input_audio_buffer.cleared")

    def test_unknown_string_type_preserves_error_contract(self):
        events = self._drive({"type": "unknown", "event_id": "client-1"})
        self.assertEqual(events[0]["error"]["code"], "unknown_event")
        self.assertEqual(events[0]["error"]["event_id"], "client-1")
        self.assertEqual(events[1]["type"], "input_audio_buffer.cleared")

    def test_non_string_event_id_can_be_reported_without_serialization_failure(self):
        for event_id in ([], {}, 123, True):
            with self.subTest(event_id=event_id):
                events = self._drive({"type": "unknown", "event_id": event_id})
                self.assertEqual(events[0]["type"], "error")
                self.assertEqual(events[0]["error"]["code"], "invalid_value")
                self.assertEqual(events[0]["error"]["param"], "event_id")
                self.assertIsNone(events[0]["error"].get("event_id"))
                self.assertEqual(events[1]["type"], "input_audio_buffer.cleared")


if __name__ == "__main__":
    unittest.main()
