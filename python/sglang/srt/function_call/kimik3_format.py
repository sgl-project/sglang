from typing import List

THINK_OPEN = "<|open|>think<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"
RESPONSE_OPEN = "<|open|>response<|sep|>"
RESPONSE_CLOSE = "<|close|>response<|sep|>"
TOOLS_OPEN = "<|open|>tools<|sep|>"
TOOLS_CLOSE = "<|close|>tools<|sep|>"
MESSAGE_CLOSE = "<|close|>message<|sep|>"
CALL_OPEN = "<|open|>call"
CALL_CLOSE = "<|close|>call<|sep|>"
ARGUMENT_CLOSE = "<|close|>argument<|sep|>"

# max_tokens can stop after an XTML control token or channel name, before <|sep|>.
_PARTIAL_MARKER_SUFFIXES = (
    "<|open|>",
    "<|close|>",
    THINK_CLOSE.removesuffix("<|sep|>"),
    RESPONSE_OPEN.removesuffix("<|sep|>"),
    RESPONSE_CLOSE.removesuffix("<|sep|>"),
    TOOLS_OPEN.removesuffix("<|sep|>"),
    TOOLS_CLOSE.removesuffix("<|sep|>"),
    MESSAGE_CLOSE.removesuffix("<|sep|>"),
)


def partial_suffix_len(text: str, markers: List[str]) -> int:
    best = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best


def strip_partial_marker_suffix(text: str) -> str:
    for suffix in _PARTIAL_MARKER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def closing_markers_len(text: str) -> int:
    """Length of the response and message closers, complete or cut off, that
    `text` ends with."""
    end = len(text) - partial_suffix_len(text, [RESPONSE_CLOSE, MESSAGE_CLOSE])
    while True:
        for marker in (MESSAGE_CLOSE, RESPONSE_CLOSE):
            if text.endswith(marker, 0, end):
                end -= len(marker)
                break
        else:
            return len(text) - end


_CHANNEL_MARKER_PREFIXES = ("<|open|>", "<|close|>")


def _closer_run_end(text: str, start: int) -> int:
    end = start
    while True:
        for marker in (RESPONSE_CLOSE, MESSAGE_CLOSE):
            if text.startswith(marker, end):
                end += len(marker)
                break
        else:
            return end


def resolve_closers(text: str) -> tuple[str, int]:
    """`text` without the closer runs that end the answer, and where the first
    such run started (len(text) if none). A run ends the answer when it ends
    the message or nothing but another channel marker follows it; a response
    closer followed by more answer text is quoted."""
    out, pos, answer_end = [], 0, len(text)
    while True:
        starts = [
            i for m in (RESPONSE_CLOSE, MESSAGE_CLOSE) if (i := text.find(m, pos)) != -1
        ]
        if not starts:
            out.append(text[pos:])
            return "".join(out), answer_end
        start = min(starts)
        end = _closer_run_end(text, start)
        out.append(text[pos:start])
        if (
            end == len(text)
            or MESSAGE_CLOSE in text[start:end]
            or text.startswith(_CHANNEL_MARKER_PREFIXES, end)
        ):
            answer_end = min(answer_end, sum(map(len, out)))
        else:
            out.append(text[start:end])
        pos = end


def strip_response_wrappers(text: str) -> str:
    """The response channel's answer without its opener and the closers that
    end it; markers quoted inside the answer are answer text."""
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        text = text[open_idx + len(RESPONSE_OPEN) :]
    resolved, answer_end = resolve_closers(strip_partial_marker_suffix(text))
    return resolved[:answer_end]
