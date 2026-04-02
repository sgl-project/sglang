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

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicMessagesRequest,
    AnthropicThinkingParam,
)
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

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=140, suite="stage-b-test-1-gpu-small-amd")


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


class TestAnthropicThinkingConversion(unittest.TestCase):
    """Unit tests for thinking/reasoning support in request/response conversion.

    These tests verify the translation logic without requiring a running server.
    """

    def _make_serving(self):
        return AnthropicServing(openai_serving_chat=object())

    def test_thinking_enabled_sets_reasoning(self):
        """thinking.type='enabled' should set reasoning={enabled: True},
        separate_reasoning/stream_reasoning=True, and
        chat_template_kwargs.thinking=True on the chat request."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Think about this."}],
            thinking=AnthropicThinkingParam(type="enabled", budget_tokens=4096),
        )
        serving = self._make_serving()
        chat_req = serving._convert_to_chat_completion_request(request)

        self.assertTrue(chat_req.separate_reasoning)
        self.assertTrue(chat_req.stream_reasoning)
        # The normalize_reasoning_inputs validator should set this
        self.assertIsNotNone(chat_req.chat_template_kwargs)
        self.assertTrue(chat_req.chat_template_kwargs.get("thinking"))

    def test_thinking_disabled_sets_reasoning_effort_none(self):
        """thinking.type='disabled' should set reasoning_effort='none',
        separate_reasoning/stream_reasoning=False, and
        chat_template_kwargs.thinking=False."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "No thinking please."}],
            thinking=AnthropicThinkingParam(type="disabled"),
        )
        serving = self._make_serving()
        chat_req = serving._convert_to_chat_completion_request(request)

        self.assertEqual(chat_req.reasoning_effort, "none")
        self.assertFalse(chat_req.separate_reasoning)
        self.assertFalse(chat_req.stream_reasoning)
        # reasoning_effort="none" should set thinking=False
        self.assertIsNotNone(chat_req.chat_template_kwargs)
        self.assertFalse(chat_req.chat_template_kwargs.get("thinking"))
        self.assertFalse(chat_req.chat_template_kwargs.get("enable_thinking"))

    def test_no_thinking_param_preserves_defaults(self):
        """When thinking is not specified, separate_reasoning and stream_reasoning
        should remain at their defaults (True)."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello."}],
        )
        serving = self._make_serving()
        chat_req = serving._convert_to_chat_completion_request(request)

        self.assertTrue(chat_req.separate_reasoning)
        self.assertTrue(chat_req.stream_reasoning)
        self.assertIsNone(chat_req.reasoning_effort)

    def test_multi_turn_thinking_blocks_reconstructed(self):
        """Thinking blocks in assistant messages should be reconstructed as
        <think>...</think> tags for the chat template."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Let me calculate: 2+2=4",
                        },
                        {"type": "text", "text": "The answer is 4."},
                    ],
                },
                {"role": "user", "content": "And 3+3?"},
            ],
        )
        serving = self._make_serving()
        chat_req = serving._convert_to_chat_completion_request(request)
        converted = chat_req.model_dump()

        # Find the assistant message
        assistant_msgs = [
            m for m in converted["messages"] if m.get("role") == "assistant"
        ]
        self.assertEqual(len(assistant_msgs), 1)

        assistant_content = assistant_msgs[0]["content"]
        # Should be a list with <think> block + text
        self.assertIsInstance(assistant_content, list)
        texts = [p["text"] for p in assistant_content if p.get("type") == "text"]
        # First part should contain the think tags
        self.assertTrue(
            any("<think>" in t for t in texts),
            f"Expected <think> tag in assistant content, got: {texts}",
        )
        # Second part should be the actual response
        self.assertTrue(
            any("The answer is 4." in t for t in texts),
            f"Expected response text in assistant content, got: {texts}",
        )

    def test_redacted_thinking_blocks_skipped(self):
        """Redacted thinking blocks should be skipped in conversion."""
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking"},
                        {"type": "text", "text": "Hi there!"},
                    ],
                },
                {"role": "user", "content": "Bye"},
            ],
        )
        serving = self._make_serving()
        chat_req = serving._convert_to_chat_completion_request(request)
        converted = chat_req.model_dump()

        assistant_msgs = [
            m for m in converted["messages"] if m.get("role") == "assistant"
        ]
        self.assertEqual(len(assistant_msgs), 1)
        # Content should only have the text, not redacted_thinking
        content = assistant_msgs[0]["content"]
        if isinstance(content, str):
            self.assertEqual(content, "Hi there!")
        else:
            texts = [p["text"] for p in content if p.get("type") == "text"]
            self.assertNotIn("<think>", " ".join(texts))

    def test_convert_response_with_reasoning_content(self):
        """Non-streaming response with reasoning_content should include a
        thinking content block before text."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="The answer is 4.",
                        reasoning_content="Let me think step by step: 2+2=4",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        serving = self._make_serving()
        result = serving._convert_response(response)

        # Should have thinking block + text block
        self.assertEqual(len(result.content), 2)
        self.assertEqual(result.content[0].type, "thinking")
        self.assertEqual(result.content[0].thinking, "Let me think step by step: 2+2=4")
        self.assertEqual(result.content[0].signature, "")
        self.assertEqual(result.content[1].type, "text")
        self.assertEqual(result.content[1].text, "The answer is 4.")

    def test_convert_response_without_reasoning_content(self):
        """Non-streaming response without reasoning_content should only have
        text block (no thinking block)."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Hello world.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        serving = self._make_serving()
        result = serving._convert_response(response)

        self.assertEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")
        self.assertEqual(result.content[0].text, "Hello world.")

    def test_convert_response_cache_tokens(self):
        """Cache token info should be populated in the Anthropic response."""
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            PromptTokensDetails,
            UsageInfo,
        )

        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Cached response."),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=80),
            ),
        )

        serving = self._make_serving()
        result = serving._convert_response(response)

        self.assertEqual(result.usage.input_tokens, 100)
        self.assertEqual(result.usage.output_tokens, 10)
        self.assertEqual(result.usage.cache_read_input_tokens, 80)


class TestAnthropicThinkingStreaming(unittest.TestCase):
    """Unit tests for streaming thinking/reasoning event translation.

    These tests mock the OpenAI stream output and verify the Anthropic
    event sequence without requiring a running server.
    """

    def _collect_events(self, sse_output: list[str]) -> list[dict]:
        """Parse Anthropic SSE events from raw output strings.

        Each string from _wrap_sse_event is 'event: ...\ndata: ...\n\n'.
        """
        events = []
        for chunk in sse_output:
            for line in chunk.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        events.append(json.loads(data_str))
        return events

    async def _run_stream(self, openai_chunks: list[str]) -> list[str]:
        """Run the Anthropic stream generator with mocked OpenAI chunks."""
        from unittest.mock import MagicMock

        # Create mock serving
        mock_chat = MagicMock()

        async def mock_stream(*args, **kwargs):
            for chunk in openai_chunks:
                yield chunk

        mock_chat._generate_chat_stream = mock_stream

        serving = AnthropicServing(openai_serving_chat=mock_chat)

        anthropic_request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
            thinking=AnthropicThinkingParam(type="enabled"),
        )

        lines = []
        async for line in serving._generate_anthropic_stream(
            adapted_request=MagicMock(),
            processed_request=MagicMock(),
            anthropic_request=anthropic_request,
            raw_request=MagicMock(),
        ):
            lines.append(line)
        return lines

    def _make_openai_chunk(
        self,
        content=None,
        reasoning_content=None,
        finish_reason=None,
        role=None,
        usage=None,
        tool_calls=None,
    ):
        """Build a mock OpenAI SSE data line."""
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        delta = chunk["choices"][0]["delta"]
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        if usage is not None:
            # Ensure total_tokens is present (required by UsageInfo)
            if "total_tokens" not in usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                    "completion_tokens", 0
                )
            chunk["usage"] = usage
            if finish_reason is None and content is None and reasoning_content is None:
                # Usage-only chunk has no choices
                chunk["choices"] = []
        return f"data: {json.dumps(chunk)}"

    def test_thinking_then_text_stream(self):
        """Streaming: thinking deltas → text deltas should produce correct
        Anthropic event sequence with signature_delta."""
        import asyncio

        chunks = [
            # First chunk with role + usage
            self._make_openai_chunk(
                role="assistant",
                content="",
                usage={"prompt_tokens": 10, "completion_tokens": 0},
            ),
            # Thinking deltas
            self._make_openai_chunk(reasoning_content="Let me "),
            self._make_openai_chunk(reasoning_content="think..."),
            # Text deltas
            self._make_openai_chunk(content="The answer"),
            self._make_openai_chunk(content=" is 4."),
            # Finish
            self._make_openai_chunk(finish_reason="stop"),
            # Usage chunk
            self._make_openai_chunk(
                usage={"prompt_tokens": 10, "completion_tokens": 20}
            ),
            "data: [DONE]",
        ]

        lines = asyncio.run(self._run_stream(chunks))
        events = self._collect_events(lines)

        event_types = [e["type"] for e in events]

        # Verify event sequence
        self.assertEqual(event_types[0], "message_start")

        # Should have: thinking block_start, thinking_deltas, signature_delta,
        # thinking block_stop, text block_start, text_deltas, text block_stop
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_delta", event_types)
        self.assertIn("content_block_stop", event_types)

        # Find thinking block events
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(block_starts), 2)
        self.assertEqual(block_starts[0]["content_block"]["type"], "thinking")
        self.assertEqual(block_starts[0]["index"], 0)
        self.assertEqual(block_starts[1]["content_block"]["type"], "text")
        self.assertEqual(block_starts[1]["index"], 1)

        # Find thinking deltas
        thinking_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        self.assertEqual(len(thinking_deltas), 2)
        self.assertEqual(thinking_deltas[0]["delta"]["thinking"], "Let me ")
        self.assertEqual(thinking_deltas[1]["delta"]["thinking"], "think...")

        # Find signature_delta (should appear before thinking block_stop)
        sig_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(len(sig_deltas), 1)
        self.assertEqual(sig_deltas[0]["delta"]["signature"], "")

        # Find text deltas
        text_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertEqual(len(text_deltas), 2)
        full_text = "".join(d["delta"]["text"] for d in text_deltas)
        self.assertEqual(full_text, "The answer is 4.")

        # Verify message_delta with stop_reason
        msg_deltas = [e for e in events if e["type"] == "message_delta"]
        self.assertEqual(len(msg_deltas), 1)
        self.assertEqual(msg_deltas[0]["delta"]["stop_reason"], "end_turn")

    def test_thinking_only_stream(self):
        """Streaming: thinking deltas only (no text) should still produce
        correct events with signature_delta before block_stop."""
        import asyncio

        chunks = [
            self._make_openai_chunk(
                role="assistant",
                content="",
                usage={"prompt_tokens": 5, "completion_tokens": 0},
            ),
            self._make_openai_chunk(reasoning_content="Deep thought..."),
            self._make_openai_chunk(finish_reason="stop"),
            self._make_openai_chunk(
                usage={"prompt_tokens": 5, "completion_tokens": 10}
            ),
            "data: [DONE]",
        ]

        lines = asyncio.run(self._run_stream(chunks))
        events = self._collect_events(lines)

        # Should have: message_start, thinking block_start, thinking_delta,
        # signature_delta, thinking block_stop, message_delta, message_stop
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(block_starts), 1)
        self.assertEqual(block_starts[0]["content_block"]["type"], "thinking")

        # Only one block stop (for thinking)
        block_stops = [e for e in events if e["type"] == "content_block_stop"]
        self.assertEqual(len(block_stops), 1)

        # Signature delta should be present
        sig_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(len(sig_deltas), 1)

    def test_text_only_stream_unchanged(self):
        """Streaming: text-only should produce standard events with no
        thinking blocks (regression test)."""
        import asyncio

        chunks = [
            self._make_openai_chunk(
                role="assistant",
                content="",
                usage={"prompt_tokens": 5, "completion_tokens": 0},
            ),
            self._make_openai_chunk(content="Hello world."),
            self._make_openai_chunk(finish_reason="stop"),
            self._make_openai_chunk(usage={"prompt_tokens": 5, "completion_tokens": 3}),
            "data: [DONE]",
        ]

        lines = asyncio.run(self._run_stream(chunks))
        events = self._collect_events(lines)

        block_starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(len(block_starts), 1)
        self.assertEqual(block_starts[0]["content_block"]["type"], "text")

        # No thinking or signature deltas
        thinking_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        self.assertEqual(len(thinking_deltas), 0)

        sig_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(len(sig_deltas), 0)


if __name__ == "__main__":
    unittest.main()
