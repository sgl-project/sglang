"""Muse Glimmer wire format, shared by its reasoning and function-call detectors."""

import re
from typing import Sequence

# Channel framing.
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
START = "<|start|>"

# ATEM payload markers.
FUNCTION_CALLS_OPEN = "<atem:function_calls>"
FUNCTION_CALLS_CLOSE = "</atem:function_calls>"
INVOKE_OPEN = "<atem:invoke"
INVOKE_CLOSE = "</atem:invoke>"

RECIPIENT_RE = re.compile(r"to=([^\s<]+)")

# Longest marker that could straddle a chunk boundary while streaming.
MAX_CHANNEL_MARKER = max(len(m) for m in (MESSAGE, EOM, EOT, START))
MAX_MARKER = max(MAX_CHANNEL_MARKER, len(FUNCTION_CALLS_OPEN))


def could_start_header(text: str) -> bool:
    """Whether the tail could still grow into a header."""
    stripped = text.lstrip()
    if not stripped:
        return True
    if not (stripped.startswith("to=") or "to=".startswith(stripped[:3])):
        return False
    if MESSAGE in stripped:
        return True
    recipient, angle, marker = stripped[3:].partition("<")
    if any(c.isspace() for c in recipient):
        return False
    return not angle or MESSAGE.startswith("<" + marker)


def has_atem_markers(text: str) -> bool:
    return INVOKE_OPEN in text or FUNCTION_CALLS_OPEN in text


def partial_marker_len(text: str, markers: Sequence[str], max_len: int) -> int:
    """Length of the longest suffix of ``text`` that could still become a marker.

    Returns 0 when nothing is held back, so ordinary text streams out immediately
    instead of waiting for a terminator that may never arrive.
    """
    for k in range(min(len(text), max_len - 1), 0, -1):
        tail = text[-k:]
        if any(m.startswith(tail) for m in markers):
            return k
    return 0
