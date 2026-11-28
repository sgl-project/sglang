import unittest

from sglang.benchmark.serving import _normalize_round_messages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBenchmarkToolMessageFields(CustomTestCase):
    def test_multi_turn_preserves_tool_message_fields(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_1",
                "name": "lookup",
            },
        ]
        self.assertEqual(_normalize_round_messages(messages), messages)


if __name__ == "__main__":
    unittest.main()
