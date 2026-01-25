import unittest

from sglang.srt.function_call.state_machine import (
    ParseResult,
    ToolParserStateMachine,
    UniversalToolParserState,
)


class MockStateMachine(ToolParserStateMachine):
    def parse(self, data: str) -> ParseResult:
        return ParseResult(
            state=UniversalToolParserState.IDLE, completed_tools=[], remaining=data
        )


class TestStateMachine(unittest.TestCase):
    def test_interface(self):
        sm = MockStateMachine()
        result = sm.parse("test")
        self.assertEqual(result.state, UniversalToolParserState.IDLE)
        self.assertEqual(result.remaining, "test")


if __name__ == "__main__":
    unittest.main()
