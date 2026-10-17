"""Unit tests for the DeepSeek-V4.1 chat encoder -- no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER_ESCAPED,
    encode_messages,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestImagePlaceholderEscaping(CustomTestCase):
    """A literal '<｜deepseek_image｜>' typed by the user must never be mistaken
    for a real image placeholder: it is escaped to the ASCII-pipe spelling
    instead of being rejected, and it must never produce an image token.
    """

    def test_tool_result_literal_placeholder_is_escaped_not_rejected(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"query": "q"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": f"result mentions {IMAGE_PLACEHOLDER} literally",
            },
        ]
        prompt, mm = encode_messages(
            messages, thinking_mode="chat", return_multi_modal_data=True
        )
        self.assertEqual(mm["images"], [])
        self.assertNotIn(IMAGE_PLACEHOLDER, prompt)
        self.assertIn(IMAGE_PLACEHOLDER_ESCAPED, prompt)

    def test_user_text_literal_placeholder_is_escaped_not_rejected(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"please explain {IMAGE_PLACEHOLDER}"}
                ],
            },
        ]
        prompt, mm = encode_messages(
            messages, thinking_mode="chat", return_multi_modal_data=True
        )
        self.assertEqual(mm["images"], [])
        self.assertNotIn(IMAGE_PLACEHOLDER, prompt)
        self.assertIn(IMAGE_PLACEHOLDER_ESCAPED, prompt)

    def test_string_content_literal_placeholder_is_escaped_not_rejected(self):
        messages = [
            {
                "role": "user",
                "content": f"What does the token {IMAGE_PLACEHOLDER} mean?",
            },
        ]
        prompt, mm = encode_messages(
            messages, thinking_mode="thinking", return_multi_modal_data=True
        )
        self.assertEqual(mm["images"], [])
        self.assertNotIn(IMAGE_PLACEHOLDER, prompt)
        self.assertIn(IMAGE_PLACEHOLDER_ESCAPED, prompt)

    def test_reasoning_content_literal_placeholder_is_escaped(self):
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": f"thinking about {IMAGE_PLACEHOLDER}",
                "content": "answer",
            },
        ]
        prompt = encode_messages(
            messages, thinking_mode="thinking", drop_thinking=False
        )
        self.assertNotIn(IMAGE_PLACEHOLDER, prompt)
        self.assertIn(IMAGE_PLACEHOLDER_ESCAPED, prompt)

    def test_real_image_block_still_inserts_the_actual_placeholder(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                    {"type": "text", "text": "what is this?"},
                ],
            },
        ]
        prompt, mm = encode_messages(
            messages, thinking_mode="chat", return_multi_modal_data=True
        )
        self.assertEqual(len(mm["images"]), 1)
        self.assertIn(IMAGE_PLACEHOLDER, prompt)


if __name__ == "__main__":
    unittest.main()
