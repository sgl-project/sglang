"""Unit tests for the reasoning-side MuseGlimmerDetector (srt/parser/reasoning_parser.py).

The function-call detector of the same name (srt/function_call/
muse_glimmer_detector.py) is covered separately; these tests exercise the
BaseReasoningFormatDetector subclass that splits streaming output into
reasoning (``to=self`` channels) and normal text (everything else).
"""

import pytest

from sglang.srt.function_call.muse_glimmer_format import (
    EOM,
    EOT,
    FUNCTION_CALLS_OPEN,
    MESSAGE,
    START,
)
from sglang.srt.parser.reasoning_parser import MuseGlimmerDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SELF_OPEN = f"{START}to=self{MESSAGE}"
USER_OPEN = f"{START}to=user{MESSAGE}"
FUNCTIONS_OPEN = f"{START}to=functions{MESSAGE}"


def _stream(detector: MuseGlimmerDetector, chunks: list[str]) -> tuple[str, str]:
    reasoning = ""
    normal = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk)
        reasoning += result.reasoning_text
        normal += result.normal_text
    end = detector.finish()
    reasoning += end.reasoning_text
    normal += end.normal_text
    return reasoning, normal


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize(
    ("text", "reasoning", "normal"),
    [
        # A thinking channel followed by a user channel: framing stripped from user.
        (
            f"{SELF_OPEN}thinking about it{EOM}{USER_OPEN}the answer{EOM}",
            "thinking about it",
            "the answer",
        ),
        # Unframed prose has no header at all and streams as normal text.
        ("just talking", "", "just talking"),
        # Two reasoning blocks are joined with a newline separator.
        (f"{SELF_OPEN}first{EOM}{SELF_OPEN}second{EOM}", "first\nsecond", ""),
        # Header without to= defaults to the user recipient.
        (f"{START}{MESSAGE}hi{EOM}", "", "hi"),
        # Non-user, non-self recipients pass through with framing intact
        # so the function-call detector downstream can read them.
        (f"{FUNCTIONS_OPEN}PAYLOAD{EOM}", "", f"{FUNCTIONS_OPEN}PAYLOAD{EOM}"),
        # eot terminates a channel the same way eom does.
        (f"{SELF_OPEN}final thought{EOT}", "final thought", ""),
        # Leading whitespace before the first header is normal text.
        (f"  {USER_OPEN}hi{EOM}", "", "  hi"),
    ],
)
def test_one_shot_channel_routing(text: str, reasoning: str, normal: str) -> None:
    detector = MuseGlimmerDetector()
    result = detector.detect_and_parse(text)
    assert result.reasoning_text == reasoning
    assert result.normal_text == normal


@pytest.mark.parametrize(
    "text",
    [
        f"{SELF_OPEN}thinking about it{EOM}{USER_OPEN}the answer{EOM}",
        "just talking",
    ],
)
def test_streaming_matches_one_shot(text: str) -> None:
    one_shot = MuseGlimmerDetector().detect_and_parse(text)
    streamed = _stream(MuseGlimmerDetector(), _chunks(text, 3))
    assert streamed == (one_shot.reasoning_text, one_shot.normal_text)


def test_streaming_partial_marker_is_held_not_leaked() -> None:
    """A <|message|> split across chunks must never surface in normal text."""
    detector = MuseGlimmerDetector()
    held = detector.parse_streaming_increment(f"{START}to=self<|mes")
    assert held.normal_text == ""
    assert held.reasoning_text == ""
    done = detector.parse_streaming_increment(f"sage|>hold test{EOM}")
    assert done.reasoning_text == "hold test"
    assert done.normal_text == ""


def test_no_stream_reasoning_holds_until_block_closes() -> None:
    detector = MuseGlimmerDetector(stream_reasoning=False)
    first = detector.parse_streaming_increment(f"{SELF_OPEN}quiet deduction")
    assert first.reasoning_text == ""
    second = detector.parse_streaming_increment(f"{EOM}{USER_OPEN}the answer{EOM}")
    assert second.reasoning_text == "quiet deduction"
    assert second.normal_text == "the answer"


def test_tool_call_parser_active_preserves_channel_framing() -> None:
    """With a function-call parser attached, user-channel framing is kept too."""
    text = f"{USER_OPEN}ok{EOM}{FUNCTIONS_OPEN}{FUNCTION_CALLS_OPEN}x{EOM}"

    plain = MuseGlimmerDetector().detect_and_parse(text)
    assert plain.normal_text == f"ok{FUNCTIONS_OPEN}{FUNCTION_CALLS_OPEN}x{EOM}"

    preserved = MuseGlimmerDetector(tool_call_parser_active=True).detect_and_parse(text)
    assert (
        preserved.normal_text
        == f"{USER_OPEN}ok{EOM}{FUNCTIONS_OPEN}{FUNCTION_CALLS_OPEN}x{EOM}"
    )


def test_force_nonempty_content_one_shot_promotes_reasoning() -> None:
    result = MuseGlimmerDetector(force_nonempty_content=True).detect_and_parse(
        f"{SELF_OPEN}only thinking{EOM}"
    )
    assert result.normal_text == "only thinking"
    assert result.reasoning_text == ""


def test_force_nonempty_content_streaming_promotes_at_finish() -> None:
    detector = MuseGlimmerDetector(force_nonempty_content=True)
    streamed_reasoning = ""
    for chunk in _chunks(f"{SELF_OPEN}quiet turn{EOM}", 4):
        streamed_reasoning += detector.parse_streaming_increment(chunk).reasoning_text
    assert streamed_reasoning == "quiet turn"
    end = detector.finish()
    assert end.normal_text == "quiet turn"
    assert end.reasoning_text == ""


def test_force_nonempty_content_backup_dropped_once_content_arrives() -> None:
    detector = MuseGlimmerDetector(force_nonempty_content=True)
    reasoning, normal = _stream(
        detector, _chunks(f"{SELF_OPEN}thought{EOM}{USER_OPEN}answer{EOM}", 4)
    )
    assert reasoning == "thought"
    assert normal == "answer"
    # Nothing left to promote once real content showed up.
    assert detector.finish().normal_text == ""


def test_finish_after_complete_stream_is_noop() -> None:
    detector = MuseGlimmerDetector()
    result = detector.parse_streaming_increment(f"{USER_OPEN}hi{EOM}")
    assert result.normal_text == "hi"
    assert detector.finish().normal_text == ""
    assert detector.finish().normal_text == ""


def test_stream_start_header_candidate_is_held() -> None:
    """A lone "to" could still grow into a header, so it must not emit yet."""
    detector = MuseGlimmerDetector()
    held = detector.parse_streaming_increment("to")
    assert held.normal_text == ""
    assert held.reasoning_text == ""
    reasoning, normal = _stream(detector, [f"=self{MESSAGE}late header{EOM}"])
    assert reasoning == "late header"
    assert normal == ""


def test_truncated_header_flushes_as_normal_text() -> None:
    detector = MuseGlimmerDetector()
    detector.parse_streaming_increment(f"{START}to")
    end = detector.finish()
    assert end.normal_text == f"{START}to"
    assert end.reasoning_text == ""


def test_no_stream_reasoning_pending_released_at_finish() -> None:
    """A reasoning block that never closes still surfaces via finish()."""
    detector = MuseGlimmerDetector(stream_reasoning=False)
    streamed = detector.parse_streaming_increment(f"{SELF_OPEN}truncated")
    assert streamed.reasoning_text == ""
    end = detector.finish()
    assert end.reasoning_text == "truncated"
    assert end.normal_text == ""
