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


def partial_suffix_len(text: str, markers: List[str]) -> int:
    best = 0
    for marker in markers:
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best


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
    return text.replace(MESSAGE_CLOSE, "")
