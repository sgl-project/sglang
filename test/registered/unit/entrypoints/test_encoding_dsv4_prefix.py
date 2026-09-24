"""Unit tests for encoding_dsv4 generation-prefix behavior — no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

ASSISTANT_TOKEN = "<｜Assistant｜>"
THINKING_END = "</think>"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def _render(messages):
    msgs = [dict(m) for m in messages]
    if msgs and "tools" not in msgs[0]:
        msgs[0]["tools"] = TOOLS
    return encode_messages(msgs, thinking_mode="chat")


class TestAssistantGenerationPrefix(unittest.TestCase):
    """Conversations ending in system / latest_reminder must still get the
    explicit assistant generation prompt (regression: bare instruction text
    caused repetition / empty outputs in agentic multi-turn flows)."""

    def test_user_terminated_gets_prefix(self):
        prompt = _render(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        self.assertTrue(prompt.rstrip().endswith(THINKING_END))
        self.assertIn(ASSISTANT_TOKEN, prompt)

    def test_system_terminated_gets_prefix(self):
        prompt = _render(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query": "q"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "c1"},
                {"role": "system", "content": "Please continue with the tool result."},
            ]
        )
        self.assertTrue(prompt.rstrip().endswith(THINKING_END))
        # the trailing system text is still rendered (as context), and the
        # generation prompt follows it
        self.assertLess(prompt.rfind("Please continue"), prompt.rfind(ASSISTANT_TOKEN))

    def test_latest_reminder_terminated_gets_prefix(self):
        prompt = _render(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "latest_reminder", "content": "Reminder: use tools."},
            ]
        )
        self.assertTrue(prompt.rstrip().endswith(THINKING_END))
        self.assertLess(prompt.rfind("Reminder: use tools."), prompt.rfind(ASSISTANT_TOKEN))


if __name__ == "__main__":
    unittest.main()
