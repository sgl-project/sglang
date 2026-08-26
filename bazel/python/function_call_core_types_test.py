import unittest

from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
)


class FunctionCallCoreTypesTest(unittest.TestCase):
    def test_streaming_result_serializes_tool_calls(self) -> None:
        result = StreamingParseResult(
            normal_text="before",
            calls=[
                ToolCallItem(
                    tool_index=2,
                    name="weather",
                    parameters='{"city":"Paris"}',
                )
            ],
        )

        self.assertEqual(
            result.model_dump(),
            {
                "normal_text": "before",
                "calls": [
                    {
                        "tool_index": 2,
                        "name": "weather",
                        "parameters": '{"city":"Paris"}',
                    }
                ],
            },
        )

    def test_structure_info_is_a_value_object(self) -> None:
        self.assertEqual(
            StructureInfo(begin="<call>", end="</call>", trigger="<call>"),
            StructureInfo(begin="<call>", end="</call>", trigger="<call>"),
        )


if __name__ == "__main__":
    unittest.main()
