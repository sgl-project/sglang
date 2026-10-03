"""Tests for the Anthropic error and fake-SSE helpers used by external frontends."""

import ast
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.anthropic import utils  # noqa: E402
from sglang.srt.entrypoints.anthropic.serving import (  # noqa: E402
    ERROR_TYPE_MAP as SERVING_ERROR_TYPE_MAP,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeOpenAIServingChat:
    """Just enough of OpenAIServingChat for ``AnthropicServing.__init__``."""

    def __init__(self):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=None)
        )


def _real_serving() -> AnthropicServing:
    return AnthropicServing(_FakeOpenAIServingChat())


def _chat_response(
    message: dict, finish_reason: str = "stop", usage: dict = None
) -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "served-alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", **message},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage
            or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


_ERROR_BODIES = [
    b'{"error": "boom"}',
    b'{"error": {"message": "m", "type": "custom_type"}}',
    b'{"message": "top-level"}',
    b"<html>gateway</html>",
    b"",
]


class TestErrorConversion(CustomTestCase):
    def test_matches_serving_for_shared_statuses(self):
        # 409 is unlisted in both maps; 413/422 exist only in utils (below).
        for status in (*SERVING_ERROR_TYPE_MAP, 409):
            for body in _ERROR_BODIES:
                with self.subTest(status=status, body=body):
                    serving_response = _real_serving()._convert_openai_error_response(
                        SimpleNamespace(status_code=status, body=body)
                    )
                    self.assertEqual(serving_response.status_code, status)
                    envelope = utils.to_anthropic_error(status, body)
                    self.assertEqual(
                        envelope.model_dump(), json.loads(bytes(serving_response.body))
                    )

    def test_composite_statuses_follow_http_layer_policy(self):
        env_413 = utils.to_anthropic_error(413, b'{"error": "big"}')
        self.assertEqual(
            (env_413.error.type, env_413.error.message), ("request_too_large", "big")
        )
        env_422 = utils.to_anthropic_error(422, b'{"error": "bad"}')
        self.assertEqual(
            (env_422.error.type, env_422.error.message),
            ("invalid_request_error", "bad"),
        )

    def test_message_policy_goldens(self):
        """Absolute pins for the envelope message policy, which the parity
        matrix above cannot catch if serving.py and utils drift together."""
        # 4xx keeps the parsed upstream message.
        env = utils.to_anthropic_error(400, b'{"error": {"message": "m"}}')
        self.assertEqual(
            (env.error.type, env.error.message), ("invalid_request_error", "m")
        )
        # 5xx never echoes upstream detail or type.
        env = utils.to_anthropic_error(
            500, b'{"error": {"message": "secret", "type": "custom_type"}}'
        )
        self.assertEqual(
            (env.error.type, env.error.message), ("api_error", "Internal server error")
        )
        # Empty-body fallbacks.
        self.assertEqual(
            utils.to_anthropic_error(400, b"").error.message, "Request failed"
        )
        self.assertEqual(
            utils.to_anthropic_error(502, b"").error.message, "Internal server error"
        )
        # Non-JSON 4xx body passes through as a bounded hint.
        self.assertEqual(
            utils.to_anthropic_error(400, b"<html>gateway</html>").error.message,
            "<html>gateway</html>",
        )

    def test_malformed_body_still_yields_envelope(self):
        """An upstream body the parser cannot use must not raise out of the helper."""
        for body in (
            b'{"error": {"message": 7}}',
            b'{"message": {"nested": true}}',
            b"[" * 100_000,
            b"1" * 5_000,
        ):
            with self.subTest(body=body[:40]):
                envelope = utils.to_anthropic_error(400, body)
                self.assertEqual(envelope.error.type, "invalid_request_error")

    def test_composite_map_matches_http_server_source(self):
        """The /v1/messages exception handler's status map is not importable without
        the server app, so parse http_server.py to catch a policy change there."""
        origin = importlib.util.find_spec("sglang.srt.entrypoints.http_server").origin
        tree = ast.parse(Path(origin).read_text())
        handler_maps = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or not node.keys:
                continue
            if not all(
                key is not None
                and isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                for key in node.keys
            ):
                continue
            mapping = {
                key.value: value.value
                for key, value in zip(node.keys, node.values)
                if isinstance(value, ast.Constant)
            }
            if 413 in mapping and 422 in mapping:
                handler_maps.append(mapping)
        self.assertTrue(
            handler_maps, "no status->type dict with 413/422 found in http_server.py"
        )
        for mapping in handler_maps:
            for status, error_type in mapping.items():
                self.assertEqual(
                    utils.ERROR_TYPE_MAP.get(status, "api_error"),
                    error_type,
                    f"status {status} diverged from http_server.py policy",
                )

    def test_4xx_scrub_strips_traceback_lines_and_truncates(self):
        upstream = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/app/handler.py", line 3, in run',
                "real cause: " + "x" * 600,
            ]
        )
        body = json.dumps({"error": {"message": upstream}}).encode()
        message = utils.to_anthropic_error(400, body).error.message
        self.assertNotIn("Traceback", message)
        self.assertNotIn('File "/', message)
        self.assertTrue(message.startswith("real cause: "))
        self.assertEqual(len(message), 501)
        self.assertTrue(message.endswith("\u2026"))


def _collect_blocks(events) -> list:
    """Fold fake-SSE start/delta events back into per-index content blocks."""
    blocks = {}
    for event in events:
        if event.type == "content_block_start":
            block = event.content_block
            blocks[event.index] = {"type": block.type, "id": None, "name": None}
            if block.type == "tool_use":
                blocks[event.index].update(id=block.id, name=block.name, json="")
            else:
                blocks[event.index]["text"] = ""
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "input_json_delta":
                blocks[event.index]["json"] += delta.partial_json
            else:
                blocks[event.index]["text"] += (
                    delta.thinking if delta.type == "thinking_delta" else delta.text
                )
    return [blocks[index] for index in sorted(blocks)]


class TestFakeSse(CustomTestCase):
    def test_matches_non_streaming_conversion(self):
        """For a populated response with valid tool arguments, blocks, stop_reason
        and usage match the server's non-streaming ``/v1/messages`` conversion."""
        message = {
            "content": "txt",
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "g", "arguments": "{}"},
                },
            ],
        }
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        for finish_reason in ("stop", "length", "tool_calls", "content_filter"):
            with self.subTest(finish_reason=finish_reason):
                response = _chat_response(message, finish_reason, usage)
                expected = _real_serving()._convert_response(response)
                events = utils.to_anthropic_fake_sse_events(
                    response, model="claude-test"
                )
                self.assertRegex(events[0].message.id, r"^msg_[0-9a-f]{32}$")

                blocks = _collect_blocks(events)
                self.assertEqual(
                    [block["type"] for block in blocks],
                    [block.type for block in expected.content],
                )
                for block, expected_block in zip(blocks, expected.content):
                    if expected_block.type == "tool_use":
                        self.assertEqual(
                            (block["id"], block["name"], json.loads(block["json"])),
                            (
                                expected_block.id,
                                expected_block.name,
                                expected_block.input,
                            ),
                        )
                    elif expected_block.type == "thinking":
                        self.assertEqual(block["text"], expected_block.thinking)
                    else:
                        self.assertEqual(block["text"], expected_block.text)

                start_usage = events[0].message.usage
                message_delta = events[-2]
                self.assertEqual(message_delta.delta.stop_reason, expected.stop_reason)
                self.assertEqual(
                    (
                        start_usage.input_tokens,
                        start_usage.cache_read_input_tokens,
                        message_delta.usage.output_tokens,
                    ),
                    (
                        expected.usage.input_tokens,
                        expected.usage.cache_read_input_tokens,
                        expected.usage.output_tokens,
                    ),
                )

    def test_text_event_sequence_uses_request_model(self):
        events = utils.to_anthropic_fake_sse_events(
            _chat_response({"content": "hello"}),
            model="claude-test",
            id_factory=lambda: "msg_fixed",
        )
        self.assertEqual(
            [e.type for e in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        start = events[0].message
        # Model comes from the Anthropic request, not the backend alias.
        self.assertEqual(start.model, "claude-test")
        self.assertEqual(start.id, "msg_fixed")
        self.assertEqual(start.content, [])
        self.assertEqual(start.usage.input_tokens, 10)
        self.assertEqual(start.usage.output_tokens, 0)
        self.assertEqual(events[2].delta.text, "hello")
        self.assertEqual(events[4].delta.stop_reason, "end_turn")
        self.assertIsNone(events[4].usage.input_tokens)
        self.assertEqual(events[4].usage.output_tokens, 5)

    def test_multi_block_index_accounting(self):
        events = utils.to_anthropic_fake_sse_events(
            _chat_response(
                {
                    "content": "txt",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"a": 1}'},
                        },
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "g", "arguments": ""},
                        },
                    ],
                },
                finish_reason="tool_calls",
            ),
            model="claude-test",
            id_factory=lambda: "m",
        )
        starts = [e for e in events if e.type == "content_block_start"]
        self.assertEqual(
            [(e.index, e.content_block.type) for e in starts],
            [(0, "thinking"), (1, "text"), (2, "tool_use"), (3, "tool_use")],
        )
        deltas = [e for e in events if e.type == "content_block_delta"]
        # A zero-argument tool call emits no input_json_delta, as in the live stream.
        self.assertEqual(
            [(e.index, e.delta.type) for e in deltas],
            [(0, "thinking_delta"), (1, "text_delta"), (2, "input_json_delta")],
        )
        self.assertEqual(deltas[2].delta.partial_json, '{"a": 1}')
        stops = [e.index for e in events if e.type == "content_block_stop"]
        self.assertEqual(stops, [0, 1, 2, 3])
        self.assertEqual(events[-2].delta.stop_reason, "tool_use")
        self.assertEqual(events[-1].type, "message_stop")

    def test_empty_choices_emits_bare_envelope(self):
        no_choices = ChatCompletionResponse.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
        events = utils.to_anthropic_fake_sse_events(
            no_choices, model="claude-test", id_factory=lambda: "m"
        )
        self.assertEqual(
            [e.type for e in events], ["message_start", "message_delta", "message_stop"]
        )
        self.assertEqual(events[1].delta.stop_reason, "end_turn")


class TestImportHygiene(CustomTestCase):
    def test_utils_import_loads_no_serving_runtime(self):
        # External frontends import utils without a serving runtime; serving.py
        # keeps OpenAIServingChat under TYPE_CHECKING, so none of these may load.
        code = (
            "import sys\n"
            "import sglang.srt.entrypoints.anthropic.utils\n"
            "banned = ('sglang.srt.entrypoints.openai.serving_chat',\n"
            "          'sglang.srt.managers.tokenizer_manager',\n"
            "          'sglang.srt.entrypoints.engine')\n"
            "loaded = [m for m in sys.modules for b in banned "
            "if m == b or m.startswith(b + '.')]\n"
            "assert not loaded, loaded\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
