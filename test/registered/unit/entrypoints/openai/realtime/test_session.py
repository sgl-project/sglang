from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []
        self.receive_count = 0

    async def receive(self):
        self.receive_count += 1
        return next(self.messages)

    async def send_text(self, text):
        self.sent.append(json.loads(text))


class _FakeAdapter:
    model_sample_rate = 16000
    chunked_streaming_config = {
        "chunk_size_sec": 1.0,
        "unfixed_chunk_num": 2,
        "unfixed_token_num": 5,
    }

    def __init__(self):
        self.sampling_requests = []

    def build_sampling_params(self, request):
        self.sampling_requests.append(request)
        return {"temperature": 0.0}


class TestRealtimeConnection(CustomTestCase):
    def _make_connection(self, websocket):
        server_args = SimpleNamespace(asr_max_buffer_seconds=60)
        return RealtimeConnection(
            websocket=websocket,
            tokenizer_manager=SimpleNamespace(served_model_name="test-asr-model"),
            adapter=_FakeAdapter(),
            server_args=server_args,
        )

    async def _run_text_event(self, text, *, initial_config=None):
        websocket = _FakeWebSocket(
            [
                {"type": "websocket.receive", "text": text},
                {"type": "websocket.disconnect", "code": 1000},
            ]
        )
        connection = self._make_connection(websocket)
        for name, value in (initial_config or {}).items():
            setattr(connection.config, name, value)

        await connection._run_loop()

        self.assertEqual(websocket.receive_count, 2)
        self.assertEqual(len(websocket.sent), 1)
        return connection, websocket.sent[0]

    def test_client_errors_are_correlated_and_nonfatal(self):
        """Client input errors must not close the session, and parseable
        events must retain enough context for the client to identify the
        rejected event and field.
        """
        cases = [
            ("invalid JSON", "{not valid json", "invalid_payload", None, None),
            (
                "unknown event",
                json.dumps({"type": "made.up.event", "event_id": "client_event_1"}),
                "unknown_event",
                "client_event_1",
                None,
            ),
            (
                "invalid known event",
                json.dumps({"type": "session.update", "event_id": "client_event_2"}),
                "invalid_value",
                "client_event_2",
                "session",
            ),
        ]

        for name, text, code, event_id, param in cases:
            with self.subTest(name=name):
                _, error_event = asyncio.run(self._run_text_event(text))

                self.assertEqual(error_event["type"], "error")
                self.assertEqual(error_event["error"]["code"], code)
                self.assertEqual(error_event["error"]["event_id"], event_id)
                self.assertEqual(error_event["error"]["param"], param)

    def test_session_update_configures_connection(self):
        """An accepted update must commit every negotiated field and
        acknowledge the configuration used for subsequent audio.
        """
        connection, response_event = asyncio.run(
            self._run_text_event(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "client_event_3",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": 48000,
                                    },
                                    "transcription": {
                                        "model": "test-asr-model",
                                        "language": "en",
                                    },
                                }
                            },
                        },
                    }
                )
            )
        )

        self.assertTrue(connection.config.configured)
        self.assertEqual(connection.config.input_sample_rate, 48000)
        self.assertEqual(connection.config.client_model, "test-asr-model")
        self.assertEqual(connection.config.language, "en")
        self.assertEqual(len(connection.adapter.sampling_requests), 1)
        self.assertEqual(connection.adapter.sampling_requests[0].language, "en")
        self.assertEqual(connection.config.sampling_params, {"temperature": 0.0})
        self.assertEqual(response_event["type"], "session.updated")

    def test_rejected_session_update_does_not_mutate_config(self):
        """Rejecting one field must not silently retain other fields from the
        same update, which would desynchronize client and server state.
        """
        connection, error_event = asyncio.run(
            self._run_text_event(
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "client_event_4",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": 48000,
                                    },
                                    "transcription": {
                                        "model": "test-asr-model",
                                        "language": "de",
                                        "prompt": "unsupported prompt",
                                    },
                                }
                            },
                        },
                    }
                ),
                initial_config={
                    "input_sample_rate": 16000,
                    "client_model": "existing-model",
                    "language": "fr",
                    "sampling_params": {"language": "fr"},
                    "configured": True,
                },
            )
        )

        self.assertEqual(connection.config.input_sample_rate, 16000)
        self.assertEqual(connection.config.client_model, "existing-model")
        self.assertEqual(connection.config.language, "fr")
        self.assertEqual(connection.config.sampling_params, {"language": "fr"})
        self.assertTrue(connection.config.configured)
        self.assertEqual(error_event["type"], "error")
        self.assertEqual(error_event["error"]["code"], "not_supported")
        self.assertEqual(
            error_event["error"]["param"],
            "session.audio.input.transcription.prompt",
        )
        self.assertEqual(error_event["error"]["event_id"], "client_event_4")


if __name__ == "__main__":
    unittest.main()
