"""Anthropic 4.x / 4.7 protocol forward-compatibility tests.

These tests do NOT launch an SGLang server. They only exercise the
Pydantic models and the request-conversion logic in the Anthropic
translation layer, so they are CPU-only and finish in seconds.

Run:
    python3 -m pytest test/registered/openai_server/basic/test_anthropic_4_7_compat.py -v
    # or
    python3 -m unittest openai_server.basic.test_anthropic_4_7_compat -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from sglang.srt.entrypoints.anthropic.serving import (
    AnthropicServing,
    _extract_thinking_config,
)

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_serving() -> AnthropicServing:
    """Build an AnthropicServing with a dummy OpenAIServingChat."""
    return AnthropicServing(openai_serving_chat=MagicMock())


class TestAnthropicRequest47Fields(unittest.TestCase):
    """Verify Anthropic 4.x/4.7 top-level request fields are accepted."""

    def test_adaptive_thinking_field(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "summarized"},
        )
        self.assertEqual(req.thinking["type"], "adaptive")

    def test_legacy_enabled_thinking_field(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 2048},
        )
        self.assertEqual(req.thinking["budget_tokens"], 2048)

    def test_output_config_field(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            output_config={
                "effort": "xhigh",
                "task_budget": {"type": "tokens", "total": 40000},
            },
        )
        self.assertEqual(req.output_config["effort"], "xhigh")

    def test_betas_service_tier_mcp_container(self):
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            betas=["task-budgets-2026-03-13"],
            service_tier="standard_only",
            mcp_servers=[{"type": "url", "url": "https://example.com/mcp"}],
            container={"id": "c_123"},
        )
        self.assertEqual(req.betas, ["task-budgets-2026-03-13"])
        self.assertEqual(req.service_tier, "standard_only")
        self.assertEqual(len(req.mcp_servers), 1)
        self.assertEqual(req.container["id"], "c_123")

    def test_unknown_top_level_field_is_ignored(self):
        # Forward-compatibility: new Anthropic SDK keys must not 422 us.
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            some_future_field={"foo": "bar"},
        )
        self.assertFalse(hasattr(req, "some_future_field"))

    def test_extract_thinking_config_variants(self):
        # adaptive
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
            thinking={"type": "adaptive", "display": "omitted"},
        )
        enabled, display = _extract_thinking_config(req)
        self.assertTrue(enabled)
        self.assertEqual(display, "omitted")

        # enabled (legacy)
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        enabled, display = _extract_thinking_config(req)
        self.assertTrue(enabled)
        self.assertEqual(display, "summarized")

        # unknown type
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
            thinking={"type": "something-new"},
        )
        enabled, _ = _extract_thinking_config(req)
        self.assertFalse(enabled)

        # missing
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        )
        enabled, _ = _extract_thinking_config(req)
        self.assertFalse(enabled)


class TestAnthropicContentBlock47Types(unittest.TestCase):
    """Verify new content block types parse without error."""

    def test_server_tool_use_block(self):
        AnthropicContentBlock(
            type="server_tool_use",
            id="srvtoolu_1",
            name="web_search",
            input={"query": "foo"},
        )

    def test_web_search_tool_result_block(self):
        AnthropicContentBlock(
            type="web_search_tool_result",
            tool_use_id="srvtoolu_1",
            content=[{"type": "web_search_result", "url": "https://x"}],
        )

    def test_code_execution_tool_result_block(self):
        AnthropicContentBlock(
            type="code_execution_tool_result",
            tool_use_id="srvtoolu_2",
            content="stdout",
        )

    def test_mcp_tool_use_and_result(self):
        AnthropicContentBlock(
            type="mcp_tool_use",
            id="mcptool_1",
            name="fs.read",
            input={"path": "/tmp/x"},
            server_name="fs",
        )
        AnthropicContentBlock(
            type="mcp_tool_result",
            tool_use_id="mcptool_1",
            content=[{"type": "text", "text": "ok"}],
        )

    def test_container_upload_block(self):
        AnthropicContentBlock(
            type="container_upload",
            id="upload_1",
            container={"id": "c_1"},
        )

    def test_thinking_block_still_supported(self):
        blk = AnthropicContentBlock(
            type="thinking",
            thinking="I should add two numbers.",
            signature="sig_abc",
        )
        self.assertEqual(blk.thinking, "I should add two numbers.")
        self.assertEqual(blk.signature, "sig_abc")


class TestAnthropicDelta47(unittest.TestCase):
    """Verify new streaming delta types parse without error."""

    def test_thinking_delta(self):
        d = AnthropicDelta(type="thinking_delta", thinking="step 1...")
        self.assertEqual(d.type, "thinking_delta")
        self.assertEqual(d.thinking, "step 1...")

    def test_signature_delta(self):
        d = AnthropicDelta(type="signature_delta", signature="sig_xyz")
        self.assertEqual(d.signature, "sig_xyz")

    def test_citations_delta(self):
        d = AnthropicDelta(
            type="citations_delta",
            citation={"type": "web_search_result", "url": "https://a"},
        )
        self.assertEqual(d.type, "citations_delta")

    def test_new_stop_reasons(self):
        for sr in ("pause_turn", "refusal", "model_context_window_exceeded"):
            AnthropicDelta(stop_reason=sr)


class TestAnthropicUsage47(unittest.TestCase):
    def test_new_usage_fields(self):
        u = AnthropicUsage(
            input_tokens=1,
            output_tokens=2,
            server_tool_use={"web_search_requests": 3},
            service_tier="standard",
            cache_creation={"ephemeral_5m_input_tokens": 10},
        )
        self.assertEqual(u.server_tool_use["web_search_requests"], 3)
        self.assertEqual(u.service_tier, "standard")


class TestAnthropicResponse47(unittest.TestCase):
    def test_new_stop_reasons_in_response(self):
        for sr in ("pause_turn", "refusal", "model_context_window_exceeded"):
            AnthropicMessagesResponse(
                model="m",
                content=[AnthropicContentBlock(type="text", text="hi")],
                stop_reason=sr,
            )


class TestConversionForwardCompat(unittest.TestCase):
    """`_convert_to_chat_completion_request` must tolerate new blocks."""

    def test_thinking_block_in_assistant_history_does_not_raise(self):
        serving = _make_serving()
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicContentBlock(
                            type="thinking",
                            thinking="let me think...",
                            signature="sig",
                        ),
                        AnthropicContentBlock(type="text", text="Hi!"),
                    ],
                ),
                AnthropicMessage(role="user", content="Continue."),
            ],
        )
        chat_req = serving._convert_to_chat_completion_request(req)

        # assistant turn should carry at least the visible text.
        # `m.content` may be either a plain string (when there's only one
        # text part after flattening) or a list of part-objects/dicts
        # (when thinking is preserved alongside the text). Accept both,
        # and tolerate Pydantic objects in the list (use duck-typing on
        # the `text` field rather than `isinstance(dict)`).
        def _part_text(p):
            if isinstance(p, dict):
                return p.get("text", "")
            return getattr(p, "text", "") or ""

        def _msg_has_hi(m):
            content = m.content
            if content is None:
                return False
            if isinstance(content, str):
                return "Hi!" in content
            return any("Hi!" in _part_text(p) for p in content)

        assistant_msgs = [m for m in chat_req.messages if m.role == "assistant"]
        self.assertTrue(
            any(_msg_has_hi(m) for m in assistant_msgs),
            f"assistant message did not carry visible text 'Hi!': "
            f"{[m.content for m in assistant_msgs]}",
        )

    def test_server_tool_use_block_does_not_raise(self):
        serving = _make_serving()
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[
                AnthropicMessage(role="user", content="search for x"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        AnthropicContentBlock(
                            type="server_tool_use",
                            id="srvtoolu_1",
                            name="web_search",
                            input={"query": "x"},
                        ),
                        AnthropicContentBlock(
                            type="web_search_tool_result",
                            tool_use_id="srvtoolu_1",
                            content=[{"type": "text", "text": "result body"}],
                        ),
                        AnthropicContentBlock(type="text", text="Done."),
                    ],
                ),
                AnthropicMessage(role="user", content="Thanks."),
            ],
        )
        # Must not raise.
        chat_req = serving._convert_to_chat_completion_request(req)
        self.assertIsNotNone(chat_req)

    def test_thinking_request_injects_chat_template_kwargs(self):
        serving = _make_serving()
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[AnthropicMessage(role="user", content="2+2?")],
            thinking={"type": "adaptive", "display": "summarized"},
        )
        chat_req = serving._convert_to_chat_completion_request(req)
        self.assertIsNotNone(chat_req.chat_template_kwargs)
        self.assertTrue(chat_req.chat_template_kwargs.get("enable_thinking"))
        self.assertTrue(chat_req.separate_reasoning)

    def test_non_thinking_request_leaves_chat_template_kwargs_alone(self):
        serving = _make_serving()
        req = AnthropicMessagesRequest(
            model="m",
            max_tokens=16,
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        chat_req = serving._convert_to_chat_completion_request(req)
        # No implicit enable_thinking for classic (non-4.7) callers.
        self.assertFalse(
            (chat_req.chat_template_kwargs or {}).get("enable_thinking", False)
        )


class TestResponseWithThinking(unittest.TestCase):
    """Non-streaming response should emit a thinking block when applicable."""

    def _fake_openai_response(self, reasoning: str, text: str):
        message = MagicMock()
        message.content = text
        message.reasoning_content = reasoning
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"

        usage = MagicMock()
        usage.prompt_tokens = 5
        usage.completion_tokens = 7

        response = MagicMock()
        response.choices = [choice]
        response.model = "m"
        response.usage = usage
        return response

    def test_thinking_block_emitted_before_text(self):
        serving = _make_serving()
        anthropic_req = AnthropicMessagesRequest(
            model="m",
            max_tokens=32,
            messages=[AnthropicMessage(role="user", content="2+2?")],
            thinking={"type": "adaptive", "display": "summarized"},
        )
        fake = self._fake_openai_response("2+2=4", "4")

        # Patch isinstance check by using the real type
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

        # Make the mock pass isinstance checks used inside the method.
        fake.__class__ = ChatCompletionResponse  # type: ignore[assignment]

        resp = serving._convert_response(fake, anthropic_req)
        types = [b.type for b in resp.content]
        self.assertIn("thinking", types)
        self.assertIn("text", types)
        self.assertLess(types.index("thinking"), types.index("text"))

    def test_thinking_omitted_when_display_is_omitted(self):
        serving = _make_serving()
        anthropic_req = AnthropicMessagesRequest(
            model="m",
            max_tokens=32,
            messages=[AnthropicMessage(role="user", content="2+2?")],
            thinking={"type": "adaptive", "display": "omitted"},
        )
        fake = self._fake_openai_response("2+2=4", "4")

        from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

        fake.__class__ = ChatCompletionResponse  # type: ignore[assignment]

        resp = serving._convert_response(fake, anthropic_req)
        types = [b.type for b in resp.content]
        self.assertNotIn("thinking", types)
        self.assertIn("text", types)

    def test_thinking_skipped_without_thinking_field(self):
        serving = _make_serving()
        anthropic_req = AnthropicMessagesRequest(
            model="m",
            max_tokens=32,
            messages=[AnthropicMessage(role="user", content="hi")],
        )
        fake = self._fake_openai_response("some reasoning", "hi there")

        from sglang.srt.entrypoints.openai.protocol import ChatCompletionResponse

        fake.__class__ = ChatCompletionResponse  # type: ignore[assignment]

        resp = serving._convert_response(fake, anthropic_req)
        types = [b.type for b in resp.content]
        self.assertNotIn("thinking", types)


if __name__ == "__main__":
    unittest.main()
