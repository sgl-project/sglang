# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Tests for Anthropic-compatible serving conversion."""

import unittest

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestAnthropicServingConversion(unittest.TestCase):
    def setUp(self):
        self.serving = AnthropicServing(openai_serving_chat=object())

    def test_thinking_and_output_config_convert_to_openai_request(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=4096,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "visible answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={
                "effort": "xhigh",
                "task_budget": {"type": "tokens", "total": 20000},
            },
            betas=["task-budgets-2026-03-13"],
        )

        chat_request = self.serving._convert_to_chat_completion_request(request)

        self.assertEqual(chat_request.reasoning_effort, "max")
        self.assertEqual(
            chat_request.chat_template_kwargs,
            {"thinking": True, "enable_thinking": True},
        )
        self.assertEqual(chat_request.custom_params["task_budget"]["total"], 20000)
        self.assertEqual(
            chat_request.custom_params["anthropic_betas"],
            ["task-budgets-2026-03-13"],
        )
        assistant_message = chat_request.messages[0]
        self.assertEqual(assistant_message.content, "visible answer")
        self.assertEqual(assistant_message.reasoning_content, "private reasoning")

    def test_enabled_thinking_budget_maps_to_custom_params(self):
        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": "hello"}],
            thinking={"type": "enabled", "budget_tokens": 2048},
        )

        chat_request = self.serving._convert_to_chat_completion_request(request)

        self.assertEqual(chat_request.reasoning_effort, "high")
        self.assertEqual(chat_request.custom_params["thinking_budget"], 2048)
        self.assertTrue(chat_request.chat_template_kwargs["thinking"])
        self.assertTrue(chat_request.chat_template_kwargs["enable_thinking"])

    def test_disabled_thinking_maps_to_reasoning_effort_none(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=4096,
            messages=[{"role": "user", "content": "hello"}],
            thinking={"type": "disabled"},
        )

        chat_request = self.serving._convert_to_chat_completion_request(request)

        self.assertEqual(chat_request.reasoning_effort, "none")
        self.assertFalse(chat_request.chat_template_kwargs["thinking"])
        self.assertFalse(chat_request.chat_template_kwargs["enable_thinking"])

    def test_reasoning_content_converts_to_anthropic_thinking_block(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=4096,
            messages=[{"role": "user", "content": "hello"}],
            thinking={"type": "adaptive", "display": "summarized"},
        )
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        reasoning_content="reason first",
                        content="answer",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        )

        anthropic_response = self.serving._convert_response(response, request)

        self.assertEqual(anthropic_response.content[0].type, "thinking")
        self.assertEqual(anthropic_response.content[0].thinking, "reason first")
        self.assertEqual(anthropic_response.content[1].type, "text")
        self.assertEqual(anthropic_response.content[1].text, "answer")


if __name__ == "__main__":
    unittest.main()
