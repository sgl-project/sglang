"""Focused CPU-only tests for DeepSeek-V4 encoding protocol properties."""

import json
import unittest

from sglang.srt.entrypoints.openai import encoding_dsv4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _tool_result(tool_use_id: str, content: str) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


class TestDSV4Encoding(CustomTestCase):
    def test_context_tool_call_order_controls_new_message_results(self):
        """Tool results in a new message slice follow calls declared in context.

        This guards against simplifying encode_messages() to sort only the new
        messages, which would lose the preceding assistant's call order.
        """
        context = [
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("call_a", "tool_a", {"value": "A"}),
                    _tool_call("call_b", "tool_b", {"value": "B"}),
                ],
            }
        ]
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "content": "result B",
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "content": "result A",
            },
        ]

        prompt = encoding_dsv4.encode_messages(
            messages=messages,
            thinking_mode="chat",
            context=context,
        )

        self.assertLess(prompt.index("result A"), prompt.index("result B"))

    def test_sorting_tool_results_preserves_non_tool_slots(self):
        """Sorting tool results must not move text blocks between their slots.

        This guards against sorting the entire content_blocks list instead of
        replacing only the positions occupied by tool_result blocks.
        """
        text_block = {
            "type": "text",
            "text": "Compare both tool results.",
        }
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    _tool_call("call_a", "tool_a", {}),
                    _tool_call("call_b", "tool_b", {}),
                ],
            },
            {
                "role": "user",
                "content_blocks": [
                    _tool_result("call_b", "result B"),
                    text_block,
                    _tool_result("call_a", "result A"),
                ],
            },
        ]

        sorted_messages = encoding_dsv4.sort_tool_results_by_call_order(messages)
        blocks = sorted_messages[1]["content_blocks"]

        self.assertEqual(
            [block["type"] for block in blocks],
            ["tool_result", "text", "tool_result"],
        )
        self.assertEqual(blocks[0]["tool_use_id"], "call_a")
        self.assertEqual(blocks[1], text_block)
        self.assertEqual(blocks[2]["tool_use_id"], "call_b")

    def test_drop_thinking_respects_user_or_developer_boundary(self):
        """Historical reasoning is removed only before the latest input boundary.

        The latest developer message is itself the boundary, so it and the
        current assistant reasoning must remain while older reasoning is
        removed and its visible answer is retained.
        """
        messages = [
            {
                "role": "user",
                "content": "Initial question",
            },
            {
                "role": "assistant",
                "reasoning_content": "old private reasoning",
                "content": "old visible answer",
            },
            {
                "role": "developer",
                "content": "New instruction for the current turn",
            },
            {
                "role": "assistant",
                "reasoning_content": "current private reasoning",
                "content": "current visible answer",
            },
        ]

        result = encoding_dsv4._drop_thinking_messages(messages)

        self.assertEqual(len(result), 4)

        self.assertNotIn("reasoning_content", result[1])
        self.assertEqual(result[1]["content"], "old visible answer")

        self.assertEqual(result[2], messages[2])

        self.assertEqual(
            result[3]["reasoning_content"],
            "current private reasoning",
        )
        self.assertEqual(result[3]["content"], "current visible answer")

    def test_assistant_round_trip_preserves_semantics(self):
        """Rendering and parsing preserve assistant-message semantics.

        This guards against encoder and parser format changes drifting apart
        while each side still appears locally valid.
        """
        message = {
            "role": "assistant",
            "reasoning_content": "Compare weather and accommodation options.",
            "content": "I will check both sources.",
            "tool_calls": [
                _tool_call(
                    "call_weather",
                    "get_weather",
                    {
                        "city": "北京",
                        "unit": "celsius",
                    },
                ),
                _tool_call(
                    "call_search",
                    "search_hotels",
                    {
                        "query": "Singapore hotels",
                        "limit": 3,
                        "filters": {
                            "stars": [4, 5],
                            "available": True,
                        },
                    },
                ),
            ],
        }

        encoded = encoding_dsv4.render_message(
            index=0,
            messages=[message],
            thinking_mode="thinking",
            drop_thinking=False,
        )
        parsed = encoding_dsv4.parse_message_from_completion_text(
            encoded,
            thinking_mode="thinking",
        )

        self.assertEqual(parsed["role"], "assistant")
        self.assertEqual(
            parsed["reasoning_content"],
            message["reasoning_content"],
        )
        self.assertEqual(parsed["content"], message["content"])

        original_calls = message["tool_calls"]
        parsed_calls = parsed["tool_calls"]

        self.assertEqual(len(parsed_calls), len(original_calls))
        self.assertEqual(
            [call["function"]["name"] for call in parsed_calls],
            [call["function"]["name"] for call in original_calls],
        )

        for original, recovered in zip(original_calls, parsed_calls):
            self.assertEqual(
                json.loads(recovered["function"]["arguments"]),
                json.loads(original["function"]["arguments"]),
            )


if __name__ == "__main__":
    unittest.main()
