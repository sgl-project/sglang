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


def strip_response_wrappers(text: str) -> str:
    open_idx = text.find(RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(RESPONSE_CLOSE, open_idx + len(RESPONSE_OPEN))
        if close_idx != -1:
            text = text[open_idx + len(RESPONSE_OPEN) : close_idx]
        else:
            text = text[open_idx + len(RESPONSE_OPEN) :]
    else:
        text = text.replace(RESPONSE_CLOSE, "")
    text = text.replace(MESSAGE_CLOSE, "")
    return strip_partial_marker_suffix(text)


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


# The grammar opens the answer at the start of the output or right after the
# first think closer, which ends the thinking.
THINK_THEN_RESPONSE = THINK_CLOSE + RESPONSE_OPEN


def response_opener_idx(text: str, after_thinking: bool) -> int:
    """Where the response channel's own opener starts in `text`, or -1; an
    opener anywhere else is quoted text."""
    # Thinking comes through the reasoning parser's grammar wrapper unconstrained,
    # so the output may skip it and start with the opener too.
    if text.startswith(RESPONSE_OPEN):
        return 0
    if not after_thinking:
        return -1
    idx = text.find(THINK_CLOSE)
    if idx == -1:
        return -1
    idx += len(THINK_CLOSE)
    return idx if text.startswith(RESPONSE_OPEN, idx) else -1


def strip_response_markers(
    text: str, json_answer_channel: bool = False, after_thinking: bool = False
) -> str:
    """`text` without the opener and closers of its response channel, keeping
    the text before the opener and any marker quoted inside the answer."""
    # Only the JSON-answer grammar pins the opener to the channel boundary;
    # unconstrained output may open the response without closing its thinking.
    idx = (
        response_opener_idx(text, after_thinking)
        if json_answer_channel
        else text.find(RESPONSE_OPEN)
    )
    if idx == -1:
        head, answer = "", text
    else:
        head, answer = text[:idx], text[idx + len(RESPONSE_OPEN) :]
    answer = strip_partial_marker_suffix(answer).removesuffix(MESSAGE_CLOSE)
    return head + strip_partial_marker_suffix(answer).removesuffix(RESPONSE_CLOSE)
