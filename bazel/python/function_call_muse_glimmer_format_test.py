import unittest

from sglang.srt.function_call.muse_glimmer_format import (
    FUNCTION_CALLS_OPEN,
    MESSAGE,
    could_start_header,
    has_atem_markers,
    partial_marker_len,
)


class FunctionCallMuseGlimmerFormatTest(unittest.TestCase):
    def test_header_detection_accepts_streaming_prefixes(self) -> None:
        self.assertTrue(could_start_header("to=weather<|mes"))
        self.assertTrue(could_start_header(f"to=weather{MESSAGE}"))
        self.assertFalse(could_start_header("ordinary response"))

    def test_atem_and_partial_marker_detection(self) -> None:
        self.assertTrue(has_atem_markers(f"prefix{FUNCTION_CALLS_OPEN}"))
        self.assertFalse(has_atem_markers("ordinary response"))
        self.assertEqual(
            partial_marker_len("prefix<|me", [MESSAGE], len(MESSAGE)),
            len("<|me"),
        )


if __name__ == "__main__":
    unittest.main()
