"""Unit tests for the DeepSeek-V4.1 chat encoder -- no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    encode_messages,
)
from sglang.test.test_utils import CustomTestCase


class TestNullContent(CustomTestCase):
    """OpenAI content is `str | parts | None`; None must not surface as the
    literal string "None", crash the encoder, or escape as an AssertionError
    (which serving_base maps to a 500 instead of a 400)."""

    def test_tool_message_null_content_renders_empty_result(self):
        prompt = encode_messages(
            [
                {"role": "user", "content": "question"},
                {"role": "tool", "tool_call_id": "t1", "content": None},
            ],
            thinking_mode="thinking",
        )
        self.assertIn("<tool_result></tool_result>", prompt)
        self.assertNotIn("<tool_result>None</tool_result>", prompt)

    def test_user_message_null_content_encodes(self):
        prompt = encode_messages(
            [{"role": "user", "content": None}], thinking_mode="thinking"
        )
        self.assertIn("<｜User｜><｜Assistant｜>", prompt)

    def test_developer_empty_content_raises_value_error(self):
        # ValueError so serving_base answers 400 BadRequest, matching the dsv32
        # encoder's DS32EncodingError; an AssertionError would surface as a 500.
        for content in ("", None):
            with self.subTest(content=content):
                with self.assertRaises(ValueError) as ctx:
                    encode_messages(
                        [{"role": "developer", "content": content}],
                        thinking_mode="thinking",
                    )
                self.assertIn("developer", str(ctx.exception))


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


class TestDsmlTags(CustomTestCase):
    def test_multiturn_prompt_preserves_reasoning_and_tool_results(self):
        messages = [
            {"role": "system", "content": "system", "tools": [_tool()]},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "summary",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"query": query}},
                    }
                    for call_id, query in (("a", "first"), ("b", "second"))
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "second result"},
            {"role": "tool", "tool_call_id": "a", "content": "first result"},
        ]
        prompt = encode_messages(messages, thinking_mode="thinking")
        expected = (
            "<｜User｜>question<｜Assistant｜><think>reason</think>summary\n\n"
            "<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">first</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            '<｜DSML｜ invoke name="lookup">\n'
            '<｜DSML｜ parameter name="query" string="true">second</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n"
            "</｜DSML｜ calls><｜end▁of▁sentence｜>"
            "<｜User｜><tool_result>first result</tool_result>\n\n"
            "<tool_result>second result</tool_result><｜Assistant｜><think>"
        )
        self.assertEqual(prompt[prompt.index("<｜User｜>") :], expected)


if __name__ == "__main__":
    unittest.main()
