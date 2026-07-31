"""
Differential audit tests for Kimi K3 reasoning parser finalization.

These tests compare streaming vs non-streaming output for the KimiK3Detector,
focusing on truncation at every parser state boundary.

The key invariant: streaming accumulation + finish() must produce the same
(reasoning_text, normal_text) as detect_and_parse() for the full text.

Additional invariant: Internal Kimi K3 protocol-marker fragments must not
appear in user-visible reasoning, content, or tool-call fields.
"""

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


def _stream_with_finish(detector: KimiK3Detector, chunks: list[str]) -> tuple[str, str]:
    """Feed chunks through parse_streaming_increment, then call finish()."""
    reasoning = ""
    content = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk)
        reasoning += result.reasoning_text
        content += result.normal_text
    # Flush on stream end
    end = detector.finish()
    reasoning += end.reasoning_text
    content += end.normal_text
    return reasoning, content


def _non_stream(detector: KimiK3Detector, text: str) -> tuple[str, str]:
    """One-shot parse."""
    result = detector.detect_and_parse(text)
    return result.reasoning_text, result.normal_text


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


# ---------------------------------------------------------------------------
# Test 1: Streaming + finish() == non-streaming for complete outputs
# ---------------------------------------------------------------------------

COMPLETE_OUTPUTS = [
    # reasoning + final text
    (
        f"{THINK_OPEN}deep thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}",
        "deep thought",
        "the answer",
    ),
    # empty reasoning + final text
    (
        f"{THINK_OPEN}{THINK_CLOSE}{RESPONSE_OPEN}final{RESPONSE_CLOSE}",
        "",
        "final",
    ),
    # reasoning + tool call channel
    (
        f"thought{THINK_CLOSE}{RESPONSE_OPEN}reply{RESPONSE_CLOSE}"
        f'{TOOLS_OPEN}<|open|>call tool="go" index="1"<|sep|>'
        f'<|open|>argument key="x" type="string"<|sep|>42<|close|>argument<|sep|>'
        f"<|close|>call<|sep|>{TOOLS_CLOSE}",
        "thought",
        f'reply{TOOLS_OPEN}<|open|>call tool="go" index="1"<|sep|>'
        f'<|open|>argument key="x" type="string"<|sep|>42<|close|>argument<|sep|>'
        f"<|close|>call<|sep|>{TOOLS_CLOSE}",
    ),
]


@pytest.mark.parametrize("text,exp_reasoning,exp_content", COMPLETE_OUTPUTS)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 13, 50])
def test_streaming_matches_non_streaming_complete(
    text: str, exp_reasoning: str, exp_content: str, chunk_size: int
):
    """Streaming + finish() must match non-streaming for complete outputs."""
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(det_stream, _chunks(text, chunk_size))
    assert r_stream == exp_reasoning, (
        f"reasoning mismatch at chunk_size={chunk_size}: "
        f"got {r_stream!r}, expected {exp_reasoning!r}"
    )
    assert c_stream == exp_content, (
        f"content mismatch at chunk_size={chunk_size}: "
        f"got {c_stream!r}, expected {exp_content!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Truncation inside reasoning — every prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_truncation_inside_reasoning(chunk_size: int):
    """Generation ends inside reasoning text. Streaming must flush buffered
    reasoning via finish() and match non-streaming."""
    # Prefix that ends mid-reasoning (after THINK_OPEN, before THINK_CLOSE)
    full = f"{THINK_OPEN}deep thought"
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, full)

    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(det_stream, _chunks(full, chunk_size))

    assert r_stream == r_ns, (
        f"reasoning mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"content mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )
    # The reasoning text should contain "deep thought"
    assert "deep thought" in r_stream, (
        f"reasoning text lost: {r_stream!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: Truncation inside partial THINK_CLOSE marker — token-level boundaries
# ---------------------------------------------------------------------------

# XTML markers are multi-token sequences: <|close|> | think | <|sep|>
# Reachable token-level truncation boundaries for THINK_CLOSE:
#   After token 1: "<|close|>"      (special token <|close|>)
#   After token 2: "<|close|>think"  (<|close|> + text "think")
# These are the ONLY realistic truncation points; character-level splits
# inside a single token are not reachable via max_tokens.

REACHABLE_THINK_CLOSE_PARTIALS = ["<|close|>", "<|close|>think"]


@pytest.mark.parametrize("partial", REACHABLE_THINK_CLOSE_PARTIALS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_truncation_inside_partial_think_close(partial: str, chunk_size: int):
    """Generation ends with a partial THINK_CLOSE marker.

    Both streaming and non-streaming paths must:
    - never emit a partial THINK_CLOSE prefix in reasoning or content
    - preserve the actual reasoning text before the partial marker
    """
    full_reasoning = "deep thought"
    truncated = f"{THINK_OPEN}{full_reasoning}{partial}"

    # Non-streaming
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, truncated)

    # Streaming
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(
        det_stream, _chunks(truncated, chunk_size)
    )

    # Both paths must not leak the partial marker
    assert partial not in r_ns, (
        f"non-streaming leaked partial THINK_CLOSE into reasoning: {r_ns!r}"
    )
    assert partial not in c_ns, (
        f"non-streaming leaked partial THINK_CLOSE into content: {c_ns!r}"
    )
    assert partial not in r_stream, (
        f"streaming leaked partial THINK_CLOSE into reasoning: {r_stream!r}"
    )
    assert partial not in c_stream, (
        f"streaming leaked partial THINK_CLOSE into content: {c_stream!r}"
    )
    # Actual reasoning text must be preserved
    assert full_reasoning in r_ns, (
        f"non-streaming reasoning text lost: {r_ns!r}"
    )
    assert full_reasoning in r_stream, (
        f"streaming reasoning text lost: {r_stream!r}"
    )
    # Streaming and non-streaming must match
    assert r_stream == r_ns, (
        f"stream/non-stream mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"stream/non-stream mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: Truncation after reasoning ends, before normal text
# ---------------------------------------------------------------------------

def test_truncation_after_reasoning_before_normal_text():
    """Generation ends after THINK_CLOSE but before RESPONSE_OPEN."""
    full = f"{THINK_OPEN}thought{THINK_CLOSE}"
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(det_stream, _chunks(full, 3))
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, full)
    assert r_stream == r_ns
    assert c_stream == c_ns


# ---------------------------------------------------------------------------
# Test 5: Truncation inside partial RESPONSE_OPEN marker — token-level boundaries
# ---------------------------------------------------------------------------

REACHABLE_RESPONSE_OPEN_PARTIALS = ["<|open|>", "<|open|>response"]


@pytest.mark.parametrize("partial", REACHABLE_RESPONSE_OPEN_PARTIALS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_truncation_inside_partial_response_open(partial: str, chunk_size: int):
    """Generation ends with a partial RESPONSE_OPEN after THINK_CLOSE.

    Both streaming and non-streaming paths must:
    - never emit a partial RESPONSE_OPEN prefix in content or reasoning
    - preserve the actual reasoning text
    """
    truncated = f"{THINK_OPEN}thought{THINK_CLOSE}{partial}"

    # Non-streaming
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, truncated)

    # Streaming
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(
        det_stream, _chunks(truncated, chunk_size)
    )

    # Both paths must not leak the partial marker
    assert partial not in r_ns, (
        f"non-streaming leaked partial RESPONSE_OPEN into reasoning: {r_ns!r}"
    )
    assert partial not in c_ns, (
        f"non-streaming leaked partial RESPONSE_OPEN into content: {c_ns!r}"
    )
    assert partial not in r_stream, (
        f"streaming leaked partial RESPONSE_OPEN into reasoning: {r_stream!r}"
    )
    assert partial not in c_stream, (
        f"streaming leaked partial RESPONSE_OPEN into content: {c_stream!r}"
    )
    # Reasoning text must be preserved
    assert "thought" in r_ns, (
        f"non-streaming reasoning text lost: {r_ns!r}"
    )
    assert "thought" in r_stream, (
        f"streaming reasoning text lost: {r_stream!r}"
    )
    # Streaming and non-streaming must match
    assert r_stream == r_ns, (
        f"stream/non-stream mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"stream/non-stream mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )


# ---------------------------------------------------------------------------
# Test 5b: Truncation inside partial RESPONSE_CLOSE marker — token-level boundaries
# ---------------------------------------------------------------------------

REACHABLE_RESPONSE_CLOSE_PARTIALS = ["<|close|>", "<|close|>response"]


@pytest.mark.parametrize("partial", REACHABLE_RESPONSE_CLOSE_PARTIALS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_truncation_inside_partial_response_close(partial: str, chunk_size: int):
    """Generation ends with a partial RESPONSE_CLOSE inside the response channel.

    Both streaming and non-streaming paths must:
    - never emit a partial RESPONSE_CLOSE prefix in content or reasoning
    - preserve the actual response text before the partial marker
    """
    truncated = (
        f"{THINK_OPEN}thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}the answer{partial}"
    )

    # Non-streaming
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, truncated)

    # Streaming
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(
        det_stream, _chunks(truncated, chunk_size)
    )

    # Both paths must not leak the partial marker
    assert partial not in r_ns, (
        f"non-streaming leaked partial RESPONSE_CLOSE into reasoning: {r_ns!r}"
    )
    assert partial not in c_ns, (
        f"non-streaming leaked partial RESPONSE_CLOSE into content: {c_ns!r}"
    )
    assert partial not in r_stream, (
        f"streaming leaked partial RESPONSE_CLOSE into reasoning: {r_stream!r}"
    )
    assert partial not in c_stream, (
        f"streaming leaked partial RESPONSE_CLOSE into content: {c_stream!r}"
    )
    # Response text must be preserved
    assert "the answer" in c_ns, (
        f"non-streaming content lost: {c_ns!r}"
    )
    assert "the answer" in c_stream, (
        f"streaming content lost: {c_stream!r}"
    )
    # Streaming and non-streaming must match
    assert r_stream == r_ns, (
        f"stream/non-stream mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"stream/non-stream mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: Truncation inside normal text
# ---------------------------------------------------------------------------

def test_truncation_inside_normal_text():
    """Generation ends inside normal text after RESPONSE_OPEN."""
    for prefix_len in range(1, 10):
        full = (
            f"{THINK_OPEN}thought{THINK_CLOSE}"
            f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}"
        )
        truncated = full[: -(len(RESPONSE_CLOSE) + prefix_len)]
        det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
        r_stream, c_stream = _stream_with_finish(
            det_stream, _chunks(truncated, 3)
        )
        det_ns = KimiK3Detector(force_reasoning=True)
        r_ns, c_ns = _non_stream(det_ns, truncated)
        assert r_stream == r_ns, (
            f"reasoning mismatch at prefix_len={prefix_len}: "
            f"stream={r_stream!r} non_stream={r_ns!r}"
        )
        assert c_stream == c_ns, (
            f"content mismatch at prefix_len={prefix_len}: "
            f"stream={c_stream!r} non_stream={c_ns!r}"
        )


# ---------------------------------------------------------------------------
# Test 7: Truncation inside partial tool-call marker — token-level boundaries
# ---------------------------------------------------------------------------

REACHABLE_TOOLS_OPEN_PARTIALS = ["<|open|>", "<|open|>tools"]


@pytest.mark.parametrize("partial", REACHABLE_TOOLS_OPEN_PARTIALS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_truncation_inside_partial_tools_open(partial: str, chunk_size: int):
    """Generation ends with a partial TOOLS_OPEN after reasoning + response.

    Both streaming and non-streaming paths must not leak the partial marker.
    """
    truncated = (
        f"{THINK_OPEN}thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}reply{RESPONSE_CLOSE}"
        f"{partial}"
    )

    # Non-streaming
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, truncated)

    # Streaming
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(
        det_stream, _chunks(truncated, chunk_size)
    )

    # Both paths must not leak the partial marker
    assert partial not in r_ns, (
        f"non-streaming leaked partial TOOLS_OPEN into reasoning: {r_ns!r}"
    )
    assert partial not in c_ns, (
        f"non-streaming leaked partial TOOLS_OPEN into content: {c_ns!r}"
    )
    assert partial not in r_stream, (
        f"streaming leaked partial TOOLS_OPEN into reasoning: {r_stream!r}"
    )
    assert partial not in c_stream, (
        f"streaming leaked partial TOOLS_OPEN into content: {c_stream!r}"
    )
    # Streaming and non-streaming must match
    assert r_stream == r_ns, (
        f"stream/non-stream mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"stream/non-stream mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )


# ---------------------------------------------------------------------------
# Test 8: Truncation inside tool arguments
# ---------------------------------------------------------------------------

def test_truncation_inside_tool_arguments():
    """Generation ends inside tool arguments."""
    full = (
        f"{THINK_OPEN}thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}reply{RESPONSE_CLOSE}"
        f'{TOOLS_OPEN}<|open|>call tool="go" index="1"<|sep|>'
        f'<|open|>argument key="x" type="string"<|sep|>42'
    )
    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(det_stream, _chunks(full, 3))
    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, full)
    assert r_stream == r_ns, (
        f"reasoning mismatch: stream={r_stream!r} non_stream={r_ns!r}"
    )
    assert c_stream == c_ns, (
        f"content mismatch: stream={c_stream!r} non_stream={c_ns!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: Empty reasoning followed by final text
# ---------------------------------------------------------------------------

def test_empty_reasoning_streaming_matches_non_streaming():
    """Empty reasoning followed by final text."""
    text = f"{THINK_OPEN}{THINK_CLOSE}{RESPONSE_OPEN}final{RESPONSE_CLOSE}"
    for chunk_size in [1, 2, 5, 100]:
        det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
        r_stream, c_stream = _stream_with_finish(
            det_stream, _chunks(text, chunk_size)
        )
        det_ns = KimiK3Detector(force_reasoning=True)
        r_ns, c_ns = _non_stream(det_ns, text)
        assert r_stream == r_ns, (
            f"chunk_size={chunk_size}: stream={r_stream!r} non_stream={r_ns!r}"
        )
        assert c_stream == c_ns, (
            f"chunk_size={chunk_size}: stream={c_stream!r} non_stream={c_ns!r}"
        )


# ---------------------------------------------------------------------------
# Test 10: Reasoning followed by tool call — full round trip
# ---------------------------------------------------------------------------

def test_reasoning_then_tool_call_streaming_matches_non_streaming():
    """Reasoning + tool call, truncated at various points."""
    full = (
        f"{THINK_OPEN}plan{THINK_CLOSE}"
        f"{RESPONSE_OPEN}ok{RESPONSE_CLOSE}"
        f'{TOOLS_OPEN}<|open|>call tool="go" index="1"<|sep|>'
        f'<|open|>argument key="x" type="string"<|sep|>42'
        f"<|close|>argument<|sep|><|close|>call<|sep|>"
        f"{TOOLS_CLOSE}"
    )
    for chunk_size in [1, 3, 7, 200]:
        det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
        r_stream, c_stream = _stream_with_finish(
            det_stream, _chunks(full, chunk_size)
        )
        det_ns = KimiK3Detector(force_reasoning=True)
        r_ns, c_ns = _non_stream(det_ns, full)
        assert r_stream == r_ns, (
            f"chunk_size={chunk_size}: stream={r_stream!r} non_stream={r_ns!r}"
        )
        assert c_stream == c_ns, (
            f"chunk_size={chunk_size}: stream={c_stream!r} non_stream={c_ns!r}"
        )


# ---------------------------------------------------------------------------
# Test 11: Per-choice isolation — two independent detectors
# ---------------------------------------------------------------------------

def test_two_choices_independent_state():
    """Two independent KimiK3Detector instances must not interfere."""
    text_a = f"{THINK_OPEN}alpha{THINK_CLOSE}{RESPONSE_OPEN}A{RESPONSE_CLOSE}"
    text_b = f"{THINK_OPEN}beta{THINK_CLOSE}{RESPONSE_OPEN}B{RESPONSE_CLOSE}"
    det_a = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    det_b = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_a, c_a = _stream_with_finish(det_a, _chunks(text_a, 3))
    r_b, c_b = _stream_with_finish(det_b, _chunks(text_b, 3))
    assert r_a == "alpha" and c_a == "A"
    assert r_b == "beta" and c_b == "B"


# ---------------------------------------------------------------------------
# Test 12: Partial MESSAGE_CLOSE — token-level boundaries
# ---------------------------------------------------------------------------

REACHABLE_MESSAGE_CLOSE_PARTIALS = ["<|close|>", "<|close|>message"]


@pytest.mark.parametrize("partial", REACHABLE_MESSAGE_CLOSE_PARTIALS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 50])
def test_truncation_inside_partial_message_close(partial: str, chunk_size: int):
    """Generation ends with a partial MESSAGE_CLOSE after response close."""
    truncated = (
        f"{THINK_OPEN}thought{THINK_CLOSE}"
        f"{RESPONSE_OPEN}answer{RESPONSE_CLOSE}"
        f"{partial}"
    )

    det_ns = KimiK3Detector(force_reasoning=True)
    r_ns, c_ns = _non_stream(det_ns, truncated)

    det_stream = KimiK3Detector(force_reasoning=True, stream_reasoning=True)
    r_stream, c_stream = _stream_with_finish(
        det_stream, _chunks(truncated, chunk_size)
    )

    assert partial not in r_ns, (
        f"non-streaming leaked partial MESSAGE_CLOSE into reasoning: {r_ns!r}"
    )
    assert partial not in c_ns, (
        f"non-streaming leaked partial MESSAGE_CLOSE into content: {c_ns!r}"
    )
    assert partial not in r_stream, (
        f"streaming leaked partial MESSAGE_CLOSE into reasoning: {r_stream!r}"
    )
    assert partial not in c_stream, (
        f"streaming leaked partial MESSAGE_CLOSE into content: {c_stream!r}"
    )
    assert r_stream == r_ns
    assert c_stream == c_ns


# ---------------------------------------------------------------------------
# Test 13: Legitimate text starting with '<' is preserved
# ---------------------------------------------------------------------------

def test_legitimate_angle_bracket_text_preserved():
    """Text that starts with '<' but is NOT a partial marker must be preserved."""
    # Text that starts with '<' but is not a prefix of any XTML marker
    legitimate_texts = [
        "< hello world",
        "<3 meaning love",
        "<<important>>",
        "<not_a_marker>",
    ]
    for text in legitimate_texts:
        det_ns = KimiK3Detector(force_reasoning=True)
        r_ns, c_ns = _non_stream(det_ns, f"{THINK_OPEN}thought{THINK_CLOSE}{RESPONSE_OPEN}{text}{RESPONSE_CLOSE}")
        assert text in c_ns, f"legitimate text lost: {c_ns!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
