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

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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


def test_reasoning_parser_registration() -> None:
    assert isinstance(ReasoningParser("kimi_k3").detector, KimiK3Detector)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
