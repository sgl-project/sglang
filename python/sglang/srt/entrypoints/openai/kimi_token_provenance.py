"""Map K3 parser output back to generated tokens, without re-tokenizing it."""

import re
from bisect import bisect_right

from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)

_CHANNEL_MARKERS = re.compile(
    b"|".join(
        re.escape(s.encode())
        for s in (
            THINK_OPEN,
            THINK_CLOSE,
            RESPONSE_OPEN,
            RESPONSE_CLOSE,
            MESSAGE_CLOSE,
        )
    )
)


class KimiTokenProvenance:
    def __init__(self, tokenizer):
        # Both the HF Kimi tokenizer and SGLang's tiktoken wrapper expose the
        # actual Encoding. Its byte API preserves partial UTF-8 tokens.
        encoding = getattr(tokenizer, "model", None)
        if not callable(getattr(encoding, "decode_single_token_bytes", None)):
            encoding = getattr(tokenizer, "tokenizer", None)
        self.decode = getattr(encoding, "decode_single_token_bytes", None)
        if not callable(self.decode):
            raise ValueError("K3 internal_content requires a tiktoken byte decoder")
        self.raw = bytearray()
        self.ids = []
        self.ends = []
        self.cursor = 0
        self.reasoning_ranges = []

    def append(self, token_ids):
        for token_id in token_ids:
            value = self.decode(token_id)
            if not isinstance(value, bytes) or not value:
                raise ValueError("Invalid token bytes for K3 internal_content")
            self.raw.extend(value)
            self.ids.append(token_id)
            self.ends.append(len(self.raw))

    def _ranges(self, source, strip_channels):
        value = source.encode("utf-8")
        if not value:
            return []
        tail = bytes(self.raw[self.cursor :])
        segments = []
        offset = 0
        for marker in _CHANNEL_MARKERS.finditer(tail) if strip_channels else []:
            segments.append((self.cursor + offset, self.cursor + marker.start()))
            offset = marker.end()
        segments.append((self.cursor + offset, len(self.raw)))
        visible = b"".join(self.raw[a:b] for a, b in segments)
        start = visible.find(value)
        if start < 0:
            raise ValueError("K3 increment has no matching generated token source")
        end = start + len(value)
        ranges = []
        position = 0
        for a, b in segments:
            lo, hi = max(start, position), min(end, position + b - a)
            if lo < hi:
                ranges.append((a + lo - position, a + hi - position))
            position += b - a
        self.cursor = ranges[-1][1]
        return ranges

    def take(self, source, *, text=False, reasoning=False, replay_chars=0):
        ranges = []
        if replay_chars:
            prefix = source[:replay_chars].encode("utf-8")
            remaining = len(prefix)
            for a, b in self.reasoning_ranges:
                if remaining <= 0:
                    break
                end = min(b, a + remaining)
                ranges.append((a, end))
                remaining -= end - a
            if remaining or b"".join(self.raw[a:b] for a, b in ranges) != prefix:
                raise ValueError("K3 reasoning replay does not match its token source")
            source = source[replay_chars:]
        ranges.extend(self._ranges(source, strip_channels=text))
        if reasoning:
            self.reasoning_ranges.extend(ranges)
        # Deduplicate token *positions*, not IDs: repeated generated tokens are
        # distinct. A token spanning two increments legitimately appears in both.
        positions = []
        for a, b in ranges:
            index = bisect_right(self.ends, a)
            while index < len(self.ids) and (self.ends[index - 1] if index else 0) < b:
                if not positions or positions[-1] != index:
                    positions.append(index)
                index += 1
        if not positions:
            raise ValueError("K3 non-empty increment has no generated tokens")
        return [{"token_ids": [self.ids[i] for i in positions]}]
