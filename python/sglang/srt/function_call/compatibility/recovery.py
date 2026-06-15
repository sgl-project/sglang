"""Fail-open recovery helpers for tool-call parsing."""

import json
import logging
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.function_call.compatibility.mode import CompatibilityEvent
from sglang.srt.function_call.core_types import StreamingParseResult

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.base_format_detector import BaseFormatDetector

logger = logging.getLogger(__name__)


def recover_stream(
    detector: "BaseFormatDetector",
    error: Exception,
    chunk_text: str,
    tools: List["Tool"],
) -> StreamingParseResult:
    """Recover after ``parse_streaming_increment`` raised.

    The caller (the boundary) latches afterwards, passing the rest of the
    stream through as normal text.
    """
    detector.compatibility.note(CompatibilityEvent.FAIL_OPEN, detail=str(error))
    logger.exception(
        "Tool-call detector %s raised in parse_streaming_increment; "
        "failing open to normal text (compatibility events: %s).",
        type(detector).__name__,
        detector.compatibility.summary(),
    )
    try:
        return detector.fail_open_stream(error, chunk_text, tools)
    except Exception:
        logger.exception(
            "fail_open_stream of %s raised; passing the chunk through.",
            type(detector).__name__,
        )
        return StreamingParseResult(normal_text=chunk_text)


def recover_nonstream(
    detector: "BaseFormatDetector",
    error: Exception,
    full_text: str,
    tools: List["Tool"],
) -> StreamingParseResult:
    """Recover after ``detect_and_parse`` raised.

    The detector's hook salvages what it can; when no call survived, the
    caller falls back to returning the full original text.
    """
    detector.compatibility.note(CompatibilityEvent.FAIL_OPEN, detail=str(error))
    logger.exception(
        "Tool-call detector %s raised in detect_and_parse; "
        "failing open to normal text (compatibility events: %s).",
        type(detector).__name__,
        detector.compatibility.summary(),
    )
    try:
        return detector.fail_open_nonstream(error, full_text, tools)
    except Exception:
        logger.exception(
            "fail_open_nonstream of %s raised; returning the full text.",
            type(detector).__name__,
        )
        return StreamingParseResult(normal_text=full_text)


def synthesize_json_close(partial: str) -> Optional[str]:
    """Return the minimal suffix that turns ``partial`` into valid JSON.

    Scope-4 mechanism: when a tool call fails mid-stream, the argument
    fragments already sent to the client cannot be unsent, so
    ``BaseFormatDetector._close_mid_stream_call`` appends this suffix to keep
    them concatenating to valid JSON.

    ``partial`` is a prefix of a JSON object as streamed by a detector
    (object / array nesting, possibly cut inside a string, right after a
    ``"key": `` separator, or right after a ``,``). Returns ``None`` when no
    such suffix exists (the caller then falls back to best-effort
    bookkeeping).
    """
    stack: List[str] = []
    in_string = False
    escape = False
    last_significant = ""
    for ch in partial:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                last_significant = ch
        else:
            if ch == '"':
                in_string = True
                last_significant = ch
            elif ch == "{":
                stack.append("}")
                last_significant = ch
            elif ch == "[":
                stack.append("]")
                last_significant = ch
            elif ch in "}]":
                if not stack:
                    return None
                stack.pop()
                last_significant = ch
            elif not ch.isspace():
                last_significant = ch

    parts: List[str] = []
    if in_string:
        if escape:
            # A dangling backslash would escape our closing quote.
            parts.append("\\")
        parts.append('"')
    elif last_significant == ":":
        # Cut right after a `"key": ` separator.
        parts.append("null")
    elif last_significant == ",":
        # Cut right after a separator; emit a placeholder element/entry.
        parts.append('"": null' if (stack and stack[-1] == "}") else "null")
    parts.extend(reversed(stack))
    closing = "".join(parts)

    try:
        json.loads(partial + closing)
    except json.JSONDecodeError:
        return None
    return closing
