"""
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_simple_messages
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_simple_messages_stream
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_multi_turn_messages
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_system_message_string
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_system_message_blocks
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_max_tokens
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_temperature
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_stop_sequences
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_error_invalid_max_tokens
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_error_empty_messages
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_raw_http_non_streaming
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_raw_http_streaming
python3 -m unittest openai_server.basic.test_anthropic_server.TestAnthropicServer.test_tool_result_image_content_conversion
"""

import json
import unittest

import requests

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=140, suite="stage-b-test-small-1-gpu-amd")


class TestAnthropicServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.messages_url = cls.base_url + "/v1/messages"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _make_request(self, payload, stream=False):
        """Send a request to the /v1/messages endpoint."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        return requests.post(
            self.messages_url,
            headers=headers,
            json=payload,
            stream=stream,
        )

    def _default_payload(self, **overrides):
        """Build a default Anthropic Messages request payload."""
        payload = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                }
            ],
        }
        payload.update(overrides)
        return payload

    # ---- Non-streaming tests ----

    def test_tool_result_image_content_conversion(self):
        """Tool-result image blocks should be preserved as OpenAI image_url content."""
        anthropic_request = AnthropicMessagesRequest(
            model=self.model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": "I have called read_file to get an image. What color is it?",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "read_file",
                            "input": {"file_path": "/test.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abcd",
                                    },
                                }
                            ],
                        }
                    ],
                },
            ],
        )

        serving = AnthropicServing(openai_serving_chat=object())
        chat_request = serving._convert_to_chat_completion_request(anthropic_request)
        converted = chat_request.model_dump()

        tool_messages = [m for m in converted["messages"] if m.get("role") == "tool"]
        self.assertEqual(
            len(tool_messages),
            1,
            f"Expected one tool message, got: {converted['messages']}",
        )

        tool_message = tool_messages[0]
        self.assertEqual(tool_message["tool_call_id"], "call_123")
        self.assertIsInstance(tool_message["content"], list)
        self.assertEqual(len(tool_message["content"]), 1)
        self.assertEqual(tool_message["content"][0]["type"], "image_url")
        self.assertEqual(
            tool_message["content"][0]["image_url"]["url"],
            "data:image/png;base64,abcd",
        )

    def test_simple_messages(self):
        """Test basic non-streaming message request."""
        payload = self._default_payload()
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")
        self.assertIn("content", body)
        self.assertIsInstance(body["content"], list)
        self.assertTrue(len(body["content"]) > 0)
        self.assertEqual(body["content"][0]["type"], "text")
        self.assertIsInstance(body["content"][0]["text"], str)
        self.assertTrue(len(body["content"][0]["text"]) > 0)

        # Verify stop reason
        self.assertIn(body["stop_reason"], ["end_turn", "max_tokens", "stop_sequence"])

        # Verify usage
        self.assertIn("usage", body)
        self.assertIsInstance(body["usage"]["input_tokens"], int)
        self.assertIsInstance(body["usage"]["output_tokens"], int)
        self.assertGreater(body["usage"]["input_tokens"], 0)
        self.assertGreater(body["usage"]["output_tokens"], 0)

        # Verify id format (must be msg_*) and model
        self.assertIn("id", body)
        self.assertIsInstance(body["id"], str)
        self.assertTrue(
            body["id"].startswith("msg_"),
            f"ID should start with 'msg_', got: {body['id']}",
        )
        self.assertIn("model", body)

    def test_multi_turn_messages(self):
        """Test multi-turn conversation."""
        payload = self._default_payload(
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ]
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)
        self.assertEqual(body["content"][0]["type"], "text")
        self.assertIsInstance(body["content"][0]["text"], str)

    def test_system_message_string(self):
        """Test system message as a string."""
        payload = self._default_payload(
            system="You are a helpful assistant. Always respond in French.",
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)

    def test_system_message_blocks(self):
        """Test system message as content blocks."""
        payload = self._default_payload(
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": "Always be concise."},
            ],
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)

    def test_max_tokens(self):
        """Test max_tokens limits output length."""
        payload = self._default_payload(
            max_tokens=5,
            messages=[
                {"role": "user", "content": "Tell me a long story about a dragon."}
            ],
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        # With very small max_tokens the model should hit the limit
        self.assertIn(body["stop_reason"], ["max_tokens", "end_turn"])
        self.assertGreater(body["usage"]["output_tokens"], 0)

    def test_temperature(self):
        """Test temperature parameter is accepted."""
        payload = self._default_payload(temperature=0.0)
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)

    def test_stop_sequences(self):
        """Test stop_sequences parameter is accepted."""
        payload = self._default_payload(
            stop_sequences=["\n"],
            max_tokens=128,
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")

    def test_top_p_and_top_k(self):
        """Test top_p and top_k parameters."""
        payload = self._default_payload(top_p=0.9, top_k=40)
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)

    # ---- Streaming tests ----

    def test_simple_messages_stream(self):
        """Test basic streaming message request."""
        payload = self._default_payload(stream=True)
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200, f"Status: {resp.status_code}")

        events = self._parse_sse_events(resp)

        # Verify event sequence
        event_types = [e["type"] for e in events]
        self.assertIn("message_start", event_types)
        self.assertIn("message_stop", event_types)

        # Verify message_start
        message_start = next(e for e in events if e["type"] == "message_start")
        self.assertIn("message", message_start)
        self.assertEqual(message_start["message"]["type"], "message")
        self.assertEqual(message_start["message"]["role"], "assistant")
        self.assertIn("usage", message_start["message"])

        # Verify we got content deltas
        content_deltas = [e for e in events if e["type"] == "content_block_delta"]
        self.assertTrue(
            len(content_deltas) > 0, "Expected at least one content_block_delta event"
        )

        # Verify all text deltas have correct structure
        for delta_event in content_deltas:
            self.assertIn("delta", delta_event)
            self.assertEqual(delta_event["delta"]["type"], "text_delta")
            self.assertIn("text", delta_event["delta"])

        # Reconstruct the full text
        full_text = "".join(
            e["delta"]["text"]
            for e in content_deltas
            if e["delta"].get("type") == "text_delta"
        )
        self.assertTrue(len(full_text) > 0, "Reconstructed text should not be empty")

        # Verify content_block_start/stop
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        block_stops = [e for e in events if e["type"] == "content_block_stop"]
        self.assertTrue(len(block_starts) > 0, "Expected content_block_start")
        self.assertTrue(len(block_stops) > 0, "Expected content_block_stop")
        self.assertEqual(block_starts[0]["content_block"]["type"], "text")

        # Verify message_delta with stop_reason
        message_deltas = [e for e in events if e["type"] == "message_delta"]
        self.assertTrue(len(message_deltas) > 0, "Expected message_delta event")
        last_delta = message_deltas[-1]
        self.assertIn("delta", last_delta)
        self.assertIn("stop_reason", last_delta["delta"])
        self.assertIn(
            last_delta["delta"]["stop_reason"],
            ["end_turn", "max_tokens", "stop_sequence", "tool_use"],
        )

        # Verify usage in message_delta
        self.assertIn("usage", last_delta)
        self.assertIsInstance(last_delta["usage"]["output_tokens"], int)

    def test_stream_multi_turn(self):
        """Test streaming with multi-turn conversation."""
        payload = self._default_payload(
            stream=True,
            messages=[
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Say goodbye."},
            ],
        )
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        events = self._parse_sse_events(resp)
        event_types = [e["type"] for e in events]
        self.assertIn("message_start", event_types)
        self.assertIn("message_stop", event_types)

    def test_stream_with_system(self):
        """Test streaming with system message."""
        payload = self._default_payload(
            stream=True,
            system="You are a pirate. Respond in pirate speak.",
        )
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        events = self._parse_sse_events(resp)
        event_types = [e["type"] for e in events]
        self.assertIn("message_start", event_types)
        self.assertIn("message_stop", event_types)

    # ---- Error handling tests ----

    def test_error_invalid_max_tokens(self):
        """Test error response for invalid max_tokens."""
        payload = self._default_payload(max_tokens=-1)
        resp = self._make_request(payload)
        self.assertIn(resp.status_code, [400, 422])

    def test_error_empty_messages(self):
        """Test error response for empty messages list."""
        payload = self._default_payload(messages=[])
        resp = self._make_request(payload)
        self.assertIn(resp.status_code, [400, 422])

    def test_error_missing_content_type(self):
        """Test error when Content-Type is not application/json."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            self.messages_url,
            headers=headers,
            data="not json",
        )
        self.assertIn(resp.status_code, [400, 415, 422])

    # ---- Raw HTTP tests ----

    def test_raw_http_non_streaming(self):
        """Test raw HTTP request/response format for non-streaming."""
        payload = self._default_payload(temperature=0)
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200)

        # Verify response content type
        self.assertIn("application/json", resp.headers.get("content-type", ""))

        body = resp.json()
        # Verify all required fields per Anthropic spec
        required_fields = ["id", "type", "role", "content", "model", "usage"]
        for field in required_fields:
            self.assertIn(field, body, f"Missing required field: {field}")

        self.assertEqual(body["type"], "message")
        self.assertEqual(body["role"], "assistant")

    def test_raw_http_streaming(self):
        """Test raw HTTP request/response format for streaming."""
        payload = self._default_payload(stream=True, temperature=0)
        resp = self._make_request(payload, stream=True)
        self.assertEqual(resp.status_code, 200)

        # Verify streaming content type
        self.assertIn("text/event-stream", resp.headers.get("content-type", ""))

        # Verify we get proper SSE events
        events = self._parse_sse_events(resp)
        self.assertTrue(len(events) > 0, "Expected at least some SSE events")

        # Verify event ordering: message_start should be first
        self.assertEqual(
            events[0]["type"], "message_start", "First event should be message_start"
        )

        # Verify message_stop is last data event
        data_events = [e for e in events if e["type"] != "ping"]
        self.assertEqual(
            data_events[-1]["type"],
            "message_stop",
            "Last data event should be message_stop",
        )

    # ---- Content block tests ----

    def test_content_blocks_message(self):
        """Test sending messages with explicit content blocks."""
        payload = self._default_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 2+2?"},
                    ],
                }
            ],
        )
        resp = self._make_request(payload)
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertEqual(body["type"], "message")
        self.assertTrue(len(body["content"]) > 0)
        self.assertEqual(body["content"][0]["type"], "text")

    # ---- Count tokens tests ----

    def test_count_tokens(self):
        """Test /v1/messages/count_tokens endpoint."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
        }
        resp = requests.post(
            self.base_url + "/v1/messages/count_tokens",
            headers=headers,
            json=payload,
        )
        self.assertEqual(resp.status_code, 200, f"Response: {resp.text}")

        body = resp.json()
        self.assertIn("input_tokens", body)
        self.assertIsInstance(body["input_tokens"], int)
        self.assertGreater(body["input_tokens"], 0)

    def test_count_tokens_with_system(self):
        """Test count_tokens with system message."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload_no_system = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }
        payload_with_system = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "system": "You are a helpful assistant with a very long system prompt that adds tokens.",
        }
        resp1 = requests.post(
            self.base_url + "/v1/messages/count_tokens",
            headers=headers,
            json=payload_no_system,
        )
        resp2 = requests.post(
            self.base_url + "/v1/messages/count_tokens",
            headers=headers,
            json=payload_with_system,
        )
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)

        # System message should increase the token count
        tokens_no_system = resp1.json()["input_tokens"]
        tokens_with_system = resp2.json()["input_tokens"]
        self.assertGreater(
            tokens_with_system,
            tokens_no_system,
            "Adding system message should increase token count",
        )

    # ---- Helpers ----

    def _parse_sse_events(self, response):
        """Parse SSE events from a streaming response."""
        events = []

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                    events.append(data)
                except json.JSONDecodeError:
                    pass

        return events


if __name__ == "__main__":
    unittest.main()
