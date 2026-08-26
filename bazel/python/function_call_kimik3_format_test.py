import unittest

from sglang.srt.function_call.kimik3_format import (
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    partial_suffix_len,
    strip_response_wrappers,
)


class FunctionCallKimiK3FormatTest(unittest.TestCase):
    def test_partial_suffix_len_finds_longest_marker_prefix(self) -> None:
        self.assertEqual(
            partial_suffix_len("answer<|cl", [RESPONSE_OPEN, RESPONSE_CLOSE]),
            len("<|cl"),
        )

    def test_strip_response_wrappers_handles_complete_and_partial_close(self) -> None:
        self.assertEqual(
            strip_response_wrappers(f"{RESPONSE_OPEN}answer{RESPONSE_CLOSE}"),
            "answer",
        )
        self.assertEqual(
            strip_response_wrappers(f"{RESPONSE_OPEN}answer<|close|>"),
            "answer",
        )


if __name__ == "__main__":
    unittest.main()
