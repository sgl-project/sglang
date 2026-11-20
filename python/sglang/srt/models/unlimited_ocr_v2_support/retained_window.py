"""KV-row holder for a finished Unlimited-OCR document window."""

from __future__ import annotations

import msgspec

from sglang.srt.managers.schedule_batch import ReqKvInfo


class RetainedWindow(msgspec.Struct):
    """Owns a retained window's KV row until it is released exactly once.

    Mutable: ``kv`` is replaced by the detached row and ``released`` flips once.
    """

    kv: ReqKvInfo = msgspec.field(default_factory=ReqKvInfo)
    released: bool = False

    def mark_released(self) -> bool:
        """Mark ownership released and return whether this was the first call."""
        if self.released:
            return False
        self.released = True
        return True
