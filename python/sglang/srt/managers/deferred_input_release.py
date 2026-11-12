from __future__ import annotations

import asyncio
from typing import Any, List, Optional


class DeferredInputRelease:
    """Keeps finished requests' prompt token ids alive past their responses.

    A prompt given as token ids is a Python list with one int object per token,
    and freeing it takes milliseconds at long context lengths. When the
    request's generator holds the last reference, that cost lands between the
    final output and the end of the response stream. Held here, the lists are
    freed once the next request has been dispatched (while it prefills), or
    ``idle_delay_s`` after the first one was held, whichever comes first.
    """

    def __init__(self, idle_delay_s: float = 1.0):
        self.idle_delay_s = idle_delay_s
        self._held: List[Any] = []
        self._timer: Optional[asyncio.TimerHandle] = None

    def hold(self, input_ids: Any) -> None:
        if input_ids is None:
            return
        self._held.append(input_ids)
        if self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(
                self.idle_delay_s, self.release
            )

    def release(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._held.clear()

    def __len__(self) -> int:
        return len(self._held)
