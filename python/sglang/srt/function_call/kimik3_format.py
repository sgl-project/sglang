import re
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


class _SpanTracker:
    def __init__(self, text: str):
        self._chars = list(zip(text, range(len(text))))
        self._deleted = False

    @property
    def text(self) -> str:
        return "".join(char for char, _ in self._chars)

    def delete(self, current_start: int, current_end: int) -> None:
        if current_start < current_end and self._chars[current_start:current_end]:
            self._deleted = True
        del self._chars[current_start:current_end]

    def truncate_at(self, current_end: int) -> None:
        self.delete(current_end, len(self._chars))

    def delete_prefix(self, current_end: int) -> None:
        self.delete(0, current_end)

    def delete_suffix(self, current_start: int) -> None:
        self.delete(current_start, len(self._chars))

    def delete_regex_matches(self, pattern: re.Pattern[str]) -> None:
        for match in reversed(list(pattern.finditer(self.text))):
            self.delete(match.start(), match.end())

    def result(self, collapse_blank: bool = False) -> tuple[str, list[tuple[int, int]]]:
        text = self.text
        if collapse_blank and self._deleted and not text.strip():
            return "", []
        spans = []
        for _, original_index in self._chars:
            if spans and spans[-1][1] == original_index:
                spans[-1] = (spans[-1][0], original_index + 1)
            else:
                spans.append((original_index, original_index + 1))
        return text, spans


def strip_response_wrappers_in_place(tracker: _SpanTracker) -> None:
    text = tracker.text
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(RESPONSE_CLOSE, open_idx + len(RESPONSE_OPEN))
        tracker.delete_prefix(open_idx + len(RESPONSE_OPEN))
        if close_idx != -1:
            tracker.delete_suffix(close_idx - open_idx - len(RESPONSE_OPEN))
    else:
        tracker.delete_regex_matches(re.compile(re.escape(RESPONSE_CLOSE)))
    tracker.delete_regex_matches(re.compile(re.escape(MESSAGE_CLOSE)))
    stripped = strip_partial_marker_suffix(tracker.text)
    if stripped != tracker.text:
        tracker.delete_suffix(len(stripped))


def strip_response_wrappers(text: str) -> str:
    tracker = _SpanTracker(text)
    strip_response_wrappers_in_place(tracker)
    return tracker.text
