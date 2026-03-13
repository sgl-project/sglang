"""
TokenCounterFilter — accumulates prompt and completion token counts.

Non-blocking: counting never delays the response to the caller.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.iochain.base import IOContext, IOFilter


class TokenCounterFilter(IOFilter):
    """
    Accumulates token usage across all completed non-streaming requests.

    blocking = False: fires as a background task; zero added latency.

    Thread safety: Python's GIL makes integer increments atomic; no lock needed.
    """

    blocking = False

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._request_count: int = 0

    async def on_request(self, ctx: IOContext) -> None:
        # Token counts are only available after inference; nothing to do here.
        pass

    async def on_response(self, ctx: IOContext) -> None:
        usage = getattr(ctx.response, "usage", None)
        if usage is None:
            return
        self._prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self._request_count += 1

    def get_stats(self) -> dict[str, Any]:
        return {
            "request_count": self._request_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }

    def reset(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0
