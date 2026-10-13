"""The running rows: requests past prefill that decode one token per step.

``DecodingBatch.rows`` is the running ``ScheduleBatch``. The scheduler changes
the rows only through the methods below; readers use ``rows`` directly.
``join`` and ``park_for_mix`` may replace the rows object; the other methods
change it in place, so a reference to ``rows`` stays valid across them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from sglang.srt.managers.schedule_batch import ScheduleBatch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DecodingBatch:
    def __init__(self, rows: ScheduleBatch):
        self.rows = rows

    def join(self, batch: ScheduleBatch) -> None:
        """Append batch's rows. Empty rows become batch itself."""
        if self.rows.is_empty():
            self.rows = batch
        else:
            self.rows.merge_batch(batch)

    def leave(self, reqs: List[Req]) -> None:
        """Drop reqs and every finished row."""
        self.rows.filter_batch(reqs_to_exclude=reqs)

    def leave_finished(self) -> None:
        """Drop the rows whose finish a commit already observed."""
        if not self.rows.is_empty():
            self.rows.filter_batch()

    def retract(self) -> Tuple[List[Req], float, List[Req]]:
        """Retract rows until the rest fit the KV pool for one decode step."""
        return self.rows.retract_decode()

    def advance(self) -> None:
        """Prepare every row for its next decode step."""
        self.rows.prepare_for_decode()

    def park_for_mix(self) -> None:
        """Hand the rows to a mixed step: they run inside that step's batch,
        which the next plan joins back."""
        self.rows = ScheduleBatch(reqs=[])

    def clear_for_pause(self) -> None:
        """Forget the rows after a pause retracted them."""
        self.rows.reqs = []
