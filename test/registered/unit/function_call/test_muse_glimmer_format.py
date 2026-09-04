"""Unit tests for muse_glimmer_format helpers — no server, no model loading.

These functions decide, while a Muse Glimmer stream is still arriving, whether
to emit text now or hold it back because it might still become a channel header
or ATEM marker. A wrong True/False here either leaks markup into assistant
content or swallows real text until the next chunk (which may never come).
"""

from sglang.srt.function_call.muse_glimmer_format import (
    EOM,
    EOT,
    FUNCTION_CALLS_CLOSE,
    FUNCTION_CALLS_OPEN,
    INVOKE_CLOSE,
    INVOKE_OPEN,
    MAX_CHANNEL_MARKER,
    MAX_MARKER,
    MESSAGE,
    START,
    could_start_header,
    has_atem_markers,
    partial_marker_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_CHANNEL_MARKERS = (MESSAGE, EOM, EOT, START)
_HELD_BACK_MARKERS = (MESSAGE, EOM, EOT, START, FUNCTION_CALLS_OPEN, INVOKE_OPEN)


class TestHasAtemMarkers(CustomTestCase):
    def test_invoke_open_is_a_marker(self):
        self.assertTrue(has_atem_markers('<atem:invoke name="get_weather">'))

    def test_function_calls_open_is_a_marker(self):
        self.assertTrue(has_atem_markers("<atem:function_calls>"))

    def test_invoke_inside_prose_is_still_detected(self):
        self.assertTrue(
            has_atem_markers('Sure.\n<atem:invoke name="search"></atem:invoke>')
        )

    def test_plain_text_is_not_a_marker(self):
        self.assertFalse(has_atem_markers("The weather in Beijing is sunny."))

    def test_close_tags_alone_are_not_open_markers(self):
        # Close tags must not trip the detector; only the opens start a call.
        self.assertFalse(has_atem_markers(INVOKE_CLOSE))
        self.assertFalse(has_atem_markers(FUNCTION_CALLS_CLOSE))

    def test_similar_atem_tag_is_not_an_open_marker(self):
        self.assertFalse(has_atem_markers('<atem:parameter name="city">SF</atem:parameter>'))


class TestPartialMarkerLen(CustomTestCase):
    def test_empty_text_holds_nothing(self):
        self.assertEqual(partial_marker_len("", _CHANNEL_MARKERS, MAX_CHANNEL_MARKER), 0)

    def test_ordinary_text_holds_nothing(self):
        self.assertEqual(
            partial_marker_len("hello world", _CHANNEL_MARKERS, MAX_CHANNEL_MARKER),
            0,
        )

    def test_complete_max_len_marker_is_not_held_back(self):
        # k only goes up to max_len-1, so a finished marker whose length
        # equals max_len cannot be treated as a partial suffix.
        self.assertEqual(len(MESSAGE), MAX_CHANNEL_MARKER)
        self.assertEqual(
            partial_marker_len(MESSAGE, _CHANNEL_MARKERS, MAX_CHANNEL_MARKER),
            0,
        )

    def test_complete_shorter_marker_is_held_until_more_text(self):
        # EOM/EOT/START are shorter than MAX_CHANNEL_MARKER, so a buffer
        # that ends on the finished token is still a prefix of itself.
        # The next non-marker characters release it (see below).
        for marker in (EOM, EOT, START):
            with self.subTest(marker=marker):
                self.assertLess(len(marker), MAX_CHANNEL_MARKER)
                self.assertEqual(
                    partial_marker_len(marker, _CHANNEL_MARKERS, MAX_CHANNEL_MARKER),
                    len(marker),
                )

    def test_complete_marker_followed_by_text_is_released(self):
        self.assertEqual(
            partial_marker_len(EOM + "hello", _CHANNEL_MARKERS, MAX_CHANNEL_MARKER),
            0,
        )

    def test_holds_back_the_longest_matching_suffix(self):
        text = "Sure.<|mess"
        held = partial_marker_len(text, _CHANNEL_MARKERS, MAX_CHANNEL_MARKER)
        self.assertEqual(held, len("<|mess"))
        self.assertTrue(MESSAGE.startswith(text[-held:]))

    def test_single_angle_bracket_is_held(self):
        # The first byte of every channel / ATEM marker.
        self.assertEqual(
            partial_marker_len("price is 10<", _HELD_BACK_MARKERS, MAX_MARKER),
            1,
        )

    def test_shared_prefix_of_eom_and_eot_is_held(self):
        text = "done<|eo"
        held = partial_marker_len(text, _CHANNEL_MARKERS, MAX_CHANNEL_MARKER)
        self.assertEqual(held, len("<|eo"))
        self.assertTrue(EOM.startswith(text[-held:]))
        self.assertTrue(EOT.startswith(text[-held:]))

    def test_lookalike_suffix_is_not_held(self):
        # "<|eox" cannot grow into EOM, EOT, MESSAGE, or START.
        self.assertEqual(
            partial_marker_len("done<|eox", _CHANNEL_MARKERS, MAX_CHANNEL_MARKER),
            0,
        )

    def test_partial_function_calls_open_uses_max_marker(self):
        prefix = FUNCTION_CALLS_OPEN[:-1]  # missing the closing '>'
        held = partial_marker_len("hi" + prefix, _HELD_BACK_MARKERS, MAX_MARKER)
        self.assertEqual(held, len(prefix))

    def test_max_len_one_never_holds(self):
        # range(min(n, 0), 0, -1) is empty, so the helper is a no-op.
        self.assertEqual(partial_marker_len("<|message", _CHANNEL_MARKERS, 1), 0)

    def test_held_length_never_reaches_max_len(self):
        # A complete marker of length max_len must not be treated as partial.
        self.assertEqual(
            partial_marker_len(FUNCTION_CALLS_OPEN, _HELD_BACK_MARKERS, MAX_MARKER),
            0,
        )


class TestCouldStartHeader(CustomTestCase):
    def test_empty_or_whitespace_can_still_become_a_header(self):
        self.assertTrue(could_start_header(""))
        self.assertTrue(could_start_header("   "))
        self.assertTrue(could_start_header("\t\n"))

    def test_prefixes_of_to_equal_are_still_possible(self):
        self.assertTrue(could_start_header("t"))
        self.assertTrue(could_start_header("to"))
        self.assertTrue(could_start_header("to="))

    def test_leading_whitespace_before_to_equal_is_ignored(self):
        self.assertTrue(could_start_header("  to=functions.get_weather"))

    def test_recipient_without_message_marker_is_still_growing(self):
        self.assertTrue(could_start_header("to=functions.get_weather"))

    def test_complete_header_is_accepted(self):
        self.assertTrue(could_start_header("to=functions.get_weather" + MESSAGE))

    def test_partial_message_marker_after_recipient_is_accepted(self):
        self.assertTrue(could_start_header("to=functions.get_weather<|mess"))

    def test_unrelated_text_cannot_start_a_header(self):
        self.assertFalse(could_start_header("hello"))
        self.assertFalse(could_start_header("Sure, let me check."))

    def test_to_lookalike_is_rejected(self):
        self.assertFalse(could_start_header("too="))
        self.assertFalse(could_start_header("to!"))

    def test_whitespace_in_recipient_is_rejected(self):
        # A space means this is prose, not `to=<recipient><|message|>`.
        self.assertFalse(could_start_header("to=foo bar"))
        self.assertFalse(could_start_header("to= functions.get_weather"))

    def test_wrong_angle_marker_cannot_become_message(self):
        self.assertFalse(could_start_header("to=foo<|eom"))
        self.assertFalse(could_start_header("to=foo<x"))

    def test_complete_message_wins_over_whitespace_in_recipient(self):
        # The finished header is already present; do not drop it because the
        # recipient also happens to contain a space.
        self.assertTrue(could_start_header("to=foo bar" + MESSAGE))


class TestMarkerLengthInvariants(CustomTestCase):
    def test_max_channel_marker_tracks_the_longest_framing_token(self):
        self.assertEqual(MAX_CHANNEL_MARKER, max(len(m) for m in _CHANNEL_MARKERS))

    def test_max_marker_covers_function_calls_open(self):
        self.assertEqual(
            MAX_MARKER, max(MAX_CHANNEL_MARKER, len(FUNCTION_CALLS_OPEN))
        )
        self.assertGreaterEqual(MAX_MARKER, len(INVOKE_OPEN))


if __name__ == "__main__":
    import unittest

    unittest.main()
