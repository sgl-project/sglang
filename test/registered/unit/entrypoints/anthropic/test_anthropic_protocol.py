# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Tests for Anthropic-compatible protocol models."""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicDelta,
    AnthropicMessagesRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestAnthropicProtocol(unittest.TestCase):
    def test_claude_47_request_fields(self):
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=4096,
            messages=[{"role": "user", "content": "hello"}],
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={
                "effort": "xhigh",
                "task_budget": {"type": "tokens", "total": 20000},
            },
            betas=["task-budgets-2026-03-13"],
        )

        self.assertEqual(request.thinking.type, "adaptive")
        self.assertEqual(request.thinking.display, "summarized")
        self.assertEqual(request.output_config.effort, "xhigh")
        self.assertEqual(request.output_config.task_budget.total, 20000)
        self.assertEqual(request.betas, ["task-budgets-2026-03-13"])

    def test_enabled_thinking_requires_budget(self):
        with self.assertRaises(ValidationError):
            AnthropicMessagesRequest(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": "hello"}],
                thinking={"type": "enabled"},
            )

    def test_thinking_delta_fields(self):
        delta = AnthropicDelta(type="thinking_delta", thinking="reasoning")
        self.assertEqual(delta.type, "thinking_delta")
        self.assertEqual(delta.thinking, "reasoning")


if __name__ == "__main__":
    unittest.main()
