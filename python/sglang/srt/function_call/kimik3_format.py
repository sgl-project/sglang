from typing import Sequence

THINK_OPEN = "<|open|>think<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"
RESPONSE_OPEN = "<|open|>response<|sep|>"
RESPONSE_CLOSE = "<|close|>response<|sep|>"
TOOLS_OPEN = "<|open|>tools<|sep|>"
TOOLS_CLOSE = "<|close|>tools<|sep|>"
MESSAGE_CLOSE = "<|close|>message<|sep|>"
CALL_OPEN = "<|open|>call"
CALL_CLOSE = "<|close|>call<|sep|>"
ARGUMENT_OPEN = "<|open|>argument"
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


ALL_MARKERS = (
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    MESSAGE_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    ARGUMENT_OPEN,
    ARGUMENT_CLOSE,
)


def partial_suffix_len(text: str, markers: Sequence[str], min_len: int = 1) -> int:
    best = min_len - 1
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best if best >= min_len else 0


def strip_partial_marker_suffix(text: str) -> str:
    for suffix in _PARTIAL_MARKER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _strip_response_wrappers(text: str) -> tuple[str, bool]:
    deleted = False
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(RESPONSE_CLOSE, open_idx + len(RESPONSE_OPEN))
        if close_idx != -1:
            text = text[open_idx + len(RESPONSE_OPEN) : close_idx]
        else:
            text = text[open_idx + len(RESPONSE_OPEN) :]
        deleted = True
    elif RESPONSE_CLOSE in text:
        text = text.replace(RESPONSE_CLOSE, "")
        deleted = True
    if MESSAGE_CLOSE in text:
        text = text.replace(MESSAGE_CLOSE, "")
        deleted = True
    stripped = strip_partial_marker_suffix(text)
    if stripped != text:
        text = stripped
        deleted = True
    return text, deleted


def strip_response_wrappers(text: str) -> str:
    return _strip_response_wrappers(text)[0]
