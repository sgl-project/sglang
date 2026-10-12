"""The running rows: requests past prefill that decode one token per step.

``DecodingBatch.rows`` is the running ``ScheduleBatch``; readers use it
directly. The scheduler changes the rows only through the methods below, with
one exception: PrefillAdder's priority preemption releases and filters the rows
object it was given in place. ``join`` and ``park_for_mix`` may replace the
rows object; the other methods change it in place, so a reference to ``rows``
stays valid across them.

The rows are never a step. A decode step is a copy of the rows
(``decode_step``), and the rows take back the fields its run and commit
rebound (``absorb``) before the next plan reads them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.managers.schedule_batch import ScheduleBatch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# The fields convert_decode_to_extend sets on a decode step for the DP extend
# view. They describe that one step; the rows keep decoding.
_EXTEND_VIEW_FIELDS = frozenset(
    {
        "forward_mode",
        "chunked_req",
        "prefill_stats",
        "prefix_lens",
        "extend_lens",
        "extend_num_tokens",
        "extend_logprob_start_lens",
        "num_decode_rows",
        "is_prefill_only",
    }
)


class DecodingBatch:
    def __init__(self, rows: ScheduleBatch):
        self.rows = rows
        # The decode step copied from the rows and not yet absorbed.
        self._step: Optional[ScheduleBatch] = None

    def join(self, batch: ScheduleBatch) -> None:
        """Append batch's rows. Empty rows become batch, and its penalizers
        then read their requests from it."""
        if self.rows.is_empty():
            self.rows = batch
            # A weak reference; a row copy still points at the batch it was
            # copied from.
            batch.sampling_info.penalizer_orchestrator.batch = batch
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

    def decode_step(self) -> ScheduleBatch:
        """The step that decodes every row: a copy of the advanced rows."""
        self._step = self.rows.shallow_copy()
        return self._step

    def absorb(self) -> None:
        """Take back the fields the last decode step's run and commit rebound,
        except the DP extend view's."""
        step, self._step = self._step, None
        if step is None:
            return
        fields = vars(step)
        if step.forward_mode.is_extend():
            fields = {k: v for k, v in fields.items() if k not in _EXTEND_VIEW_FIELDS}
        vars(self.rows).update(fields)
        self.rows.reqs = step.reqs[:]

    def park_for_mix(self) -> None:
        """Hand the rows to a mixed step: they run inside that step's batch,
        which the next plan joins back."""
        self.rows = ScheduleBatch(reqs=[])

    def clear_for_pause(self) -> None:
        """Forget the rows, and the decode step launched from them, after a
        pause retracted them."""
        self.rows.reqs = []
        self._step = None
