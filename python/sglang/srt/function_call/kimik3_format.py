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

# XTML markers are multi-token sequences composed of special tokens
# (<|open|>, <|close|>, <|sep|>) and text (think, response, tools, message).
# Each marker has the structure: <control> + text + <sep>
# Reachable token-level truncation boundaries for each marker:
#   After token 1: "<|open|>" or "<|close|>"  (special token)
#   After token 2: "<|open|>text" or "<|close|>text"  (special + text token)
#   After token 3: complete marker
# We must strip these partial suffixes from non-streaming output.

# All partial marker suffixes that can appear at a token boundary
_PARTIAL_MARKER_SUFFIXES = [
    "<|close|>",
    "<|close|>think",
    "<|open|>",
    "<|open|>response",
    "<|close|>response",
    "<|open|>tools",
    "<|close|>tools",
    "<|close|>message",
]


def partial_suffix_len(text: str, markers: List[str]) -> int:
    best = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best


def strip_partial_marker_suffix(text: str) -> str:
    """Strip a reachable partial XTML marker suffix from *text*.

    Only suffixes that correspond to token-level truncation boundaries are
    removed; arbitrary character prefixes (e.g. ``<`` or ``<|c``) are not
    touched because they cannot appear in detokenized output.
    """
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
