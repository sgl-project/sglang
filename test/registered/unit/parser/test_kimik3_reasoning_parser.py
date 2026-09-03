import sys

import pytest

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.parser.reasoning_parser import KimiK3Detector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _stream(detector: KimiK3Detector, chunks: list[str]) -> tuple[str, str]:
    reasoning = ""
    content = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk)
        reasoning += result.reasoning_text
        content += result.normal_text
    return reasoning, content


def _chunks(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


@pytest.mark.parametrize(
    ("text", "reasoning", "content"),
    [
        (
            f"{THINK_OPEN}deep thought{THINK_CLOSE}"
            f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
            "deep thought",
            "the answer",
        ),
        (
            f"thinking...{THINK_CLOSE}{RESPONSE_OPEN}done{RESPONSE_CLOSE}",
            "thinking...",
            "done",
        ),
        (
            f"{RESPONSE_OPEN}plain reply{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
            "",
            "plain reply",
        ),
        ("still going", "still going", ""),
    ],
)
def test_non_stream_reasoning_channels(text: str, reasoning: str, content: str) -> None:
    detector = KimiK3Detector(force_reasoning=True)
    result = detector.detect_and_parse(text)
    assert result.reasoning_text == reasoning
    assert result.normal_text == content


def test_non_stream_tools_channel_passthrough() -> None:
    tools_channel = (
        f'{TOOLS_OPEN}<|open|>call tool="python" index="1"<|sep|>'
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    detector = KimiK3Detector(force_reasoning=True)
    result = detector.detect_and_parse(
        f"thought{THINK_CLOSE}{RESPONSE_OPEN}reply{RESPONSE_CLOSE}{tools_channel}"
    )
    assert result.reasoning_text == "thought"
    assert result.normal_text == f"reply{tools_channel}"


def test_non_stream_recovers_missing_think_separator() -> None:
    detector = KimiK3Detector(force_reasoning=True)
    result = detector.detect_and_parse(
        f"thought{THINK_CLOSE.removesuffix('<|sep|>')}{RESPONSE_OPEN}"
        f"reply{RESPONSE_CLOSE}"
    )
    assert result.reasoning_text == "thought"
    assert result.normal_text == "reply"


@pytest.mark.parametrize(
    ("text", "reasoning", "content"),
    [
        ("deep thought<|close|>", "deep thought", ""),
        ("deep thought<|close|>think", "deep thought", ""),
        (f"{THINK_CLOSE}<|open|>", "", ""),
        (f"{THINK_CLOSE}<|open|>response", "", ""),
        (
            f"{THINK_CLOSE}{RESPONSE_OPEN}the answer<|close|>response",
            "",
            "the answer",
        ),
        (
            f"{THINK_CLOSE}{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}<|close|>message",
            "",
            "the answer",
        ),
    ],
)
def test_non_stream_strips_partial_marker_suffixes(
    text: str, reasoning: str, content: str
) -> None:
    result = KimiK3Detector(force_reasoning=True).detect_and_parse(text)
    assert result.reasoning_text == reasoning
    assert result.normal_text == content


def test_non_stream_preserves_non_marker_angle_bracket_suffix() -> None:
    result = KimiK3Detector(force_reasoning=True).detect_and_parse(
        f"{THINK_CLOSE}{RESPONSE_OPEN}answer <3"
    )
    assert result.normal_text == "answer <3"


@pytest.mark.parametrize("chunk_size", [1, 4, 13])
def test_streaming_split_markers(chunk_size: int) -> None:
    detector = KimiK3Detector(force_reasoning=True)
    text = (
        f"{THINK_OPEN}deep thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
    )
    reasoning, content = _stream(detector, _chunks(text, chunk_size))
    assert reasoning == "deep thought"
    assert content == "the answer"


def test_streaming_tools_channel_passthrough() -> None:
    tools_channel = (
        f'{TOOLS_OPEN}<|open|>call tool="python" index="1"<|sep|>'
        "<|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    detector = KimiK3Detector(force_reasoning=True)
    text = f"thought{THINK_CLOSE}{RESPONSE_OPEN}reply{RESPONSE_CLOSE}{tools_channel}"
    reasoning, content = _stream(detector, _chunks(text, 5))
    assert reasoning == "thought"
    assert content == f"reply{tools_channel}"


def test_streaming_recovers_missing_think_separator() -> None:
    detector = KimiK3Detector(force_reasoning=True)
    text = (
        f"thought{THINK_CLOSE.removesuffix('<|sep|>')}{RESPONSE_OPEN}"
        f"reply{RESPONSE_CLOSE}"
    )
    reasoning, content = _stream(detector, _chunks(text, 3))
    assert reasoning == "thought"
    assert content == "reply"


_TOOLS_CHANNEL = (
    f'{TOOLS_OPEN}<|open|>call tool="python" index="1"<|sep|>'
    "<|close|>call<|sep|>"
    f"{TOOLS_CLOSE}"
)


def test_non_stream_tools_channel_before_think_close_is_not_reasoning() -> None:
    """A tools channel emitted inside the think block, with think closing after
    it, must still reach the tool-call parser.

    The reasoning grammar is deferred until think closes, so nothing stops the
    model from opening the tools channel first. Splitting on think_end put the
    whole call in reasoning_text, dropping it with no log line.
    """
    detector = KimiK3Detector(force_reasoning=True)
    result = detector.detect_and_parse(
        f"thought{_TOOLS_CHANNEL}{THINK_CLOSE}{MESSAGE_CLOSE}"
    )
    assert result.reasoning_text == "thought"
    assert TOOLS_OPEN in result.normal_text


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
def test_streaming_tools_channel_before_think_close(chunk_size: int) -> None:
    detector = KimiK3Detector(force_reasoning=True)
    text = f"thought{_TOOLS_CHANNEL}{THINK_CLOSE}{MESSAGE_CLOSE}"
    reasoning, content = _stream(detector, _chunks(text, chunk_size))
    assert reasoning == "thought"
    assert TOOLS_OPEN in content


def test_reasoning_parser_registration() -> None:
    assert isinstance(ReasoningParser("kimi_k3").detector, KimiK3Detector)


def _stream_with_finish(detector: KimiK3Detector, chunks: list[str]) -> tuple[str, str]:
    reasoning, content = _stream(detector, chunks)
    result = detector.finish()
    return reasoning + result.reasoning_text, content + result.normal_text


@pytest.mark.parametrize(
    ("text", "reasoning", "content"),
    [
        (
            f"bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
            "",
            "bare answer",
        ),
        (
            f"{RESPONSE_OPEN}bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
            "",
            "bare answer",
        ),
        ("still going", "still going", ""),
        ("deep thought<|close|>", "deep thought", ""),
    ],
)
def test_fnc_non_stream_skipped_think_vs_truncated_reasoning(
    text: str, reasoning: str, content: str
) -> None:
    detector = KimiK3Detector(force_reasoning=True, force_nonempty_content=True)
    result = detector.detect_and_parse(text)
    assert result.reasoning_text == reasoning
    assert result.normal_text == content


@pytest.mark.parametrize("chunk_size", [1, 5, 13])
def test_fnc_streaming_skipped_think_answer(chunk_size: int) -> None:
    detector = KimiK3Detector(force_reasoning=True, force_nonempty_content=True)
    text = f"bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
    reasoning, content = _stream_with_finish(detector, _chunks(text, chunk_size))
    # Streamed as reasoning in real time; finish() re-emits the cleaned
    # payload as content once the channel close proves skipped-think.
    assert reasoning == text
    assert content == "bare answer"


@pytest.mark.parametrize("chunk_size", [1, 5, 13])
def test_fnc_streaming_truncated_reasoning_stays_reasoning(chunk_size: int) -> None:
    detector = KimiK3Detector(force_reasoning=True, force_nonempty_content=True)
    reasoning, content = _stream_with_finish(
        detector, _chunks("still going", chunk_size)
    )
    assert reasoning == "still going"
    assert content == ""


@pytest.mark.parametrize("chunk_size", [5, 13])
def test_fnc_streaming_long_think_streams_without_close(chunk_size: int) -> None:
    detector = KimiK3Detector(force_reasoning=True, force_nonempty_content=True)
    text = "x" * 20000
    reasoning = ""
    for chunk in _chunks(text, chunk_size):
        # Live chunks flow immediately — no hold-back to starve SSE idle timeouts.
        reasoning += detector.parse_streaming_increment(chunk).reasoning_text
    assert reasoning == text
    result = detector.finish()
    assert result.reasoning_text == ""
    assert result.normal_text == ""


def test_fnc_streaming_response_open_not_reemitted() -> None:
    detector = KimiK3Detector(force_reasoning=True, force_nonempty_content=True)
    text = f"{RESPONSE_OPEN}bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
    reasoning, content = _stream_with_finish(detector, _chunks(text, 5))
    # The channel switch already streamed the answer as content; finish()
    # must not re-emit it.
    assert reasoning == ""
    assert content == "bare answer"


def test_fnc_streaming_force_reasoning_off_not_reemitted() -> None:
    detector = KimiK3Detector(force_reasoning=False, force_nonempty_content=True)
    text = f"bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
    reasoning, content = _stream_with_finish(detector, _chunks(text, 5))
    assert reasoning == ""
    assert content == "bare answer"


@pytest.mark.parametrize("force_nonempty_content", [False, True])
def test_stream_reasoning_off_truncation_flushes_reasoning(
    force_nonempty_content: bool,
) -> None:
    detector = KimiK3Detector(
        force_reasoning=True,
        stream_reasoning=False,
        force_nonempty_content=force_nonempty_content,
    )
    reasoning, content = _stream_with_finish(detector, _chunks("still going", 5))
    assert reasoning == "still going"
    assert content == ""


def test_fnc_stream_reasoning_off_skipped_think_reemits_content() -> None:
    detector = KimiK3Detector(
        force_reasoning=True, stream_reasoning=False, force_nonempty_content=True
    )
    text = f"bare answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
    reasoning, content = _stream_with_finish(detector, _chunks(text, 5))
    assert reasoning == ""
    assert content == "bare answer"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
