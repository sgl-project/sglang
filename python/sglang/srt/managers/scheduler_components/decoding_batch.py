"""The running rows: requests past prefill that decode one token per step.

The scheduler changes ``DecodingBatch.rows`` only through the methods below,
except PrefillAdder's priority preemption, which filters the rows in place.
``join``, ``join_launched`` and ``park_for_mix`` may replace the rows object;
the other methods keep it, so a reference to ``rows`` stays valid across them.

The rows are never a step, except in pdmux, which still launches the rows
itself. A decode step is a copy of the rows, and ``absorb`` takes back the
fields it rebound before the next plan reads them. The pipeline PD decode loop
takes it back at the ring commit of its slot, where the output relay rebinds
the step's fields; the other pipeline loop and the MLX loop at the top of the
next plan. The normal and overlap loops, PD decode's included, take it back
right after its launch, before its commit, so a commit must not rebind a field
of the step that the rows need.

The normal and overlap loops, PD decode's included, change the rows at launch
and commit: right after a step's run the rows absorb it and its finished
prefills join (``join_launched``; PD decode joins the prebuilt batch at the
plan instead), and after a commit the rows it finished leave. Between steps the
rows are then exactly the requests still decoding. The other loops join and
leave at the top of the next plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DecodingBatch:
    def __init__(self, rows: ScheduleBatch):
        self.rows = rows
        # The decode step copied from the rows and not yet absorbed.
        self._step: Optional[ScheduleBatch] = None
        # Under SGLANG_DEBUG_INFLIGHT_STEPS: the rows object and its requests
        # when the decode step was copied.
        self._rows_at_step: Optional[Tuple[ScheduleBatch, List[Req]]] = None

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

    def join_launched(self, step: ScheduleBatch) -> None:
        """Join a copy of the rows whose last prefill chunk step ran: all of
        its rows but the request still chunking. A mixed step ran the rows as
        its tail, so its rows, new before old, replace them.

        The copy shares the step's sampling_info and spec_info, and its filter
        changes them in place right after the launch, before the step's
        commit: a commit must not read either of them by row."""
        if not step.has_prefill_rows():
            return
        still_chunking = [] if step.chunked_req is None else [step.chunked_req]
        joined = step.shallow_copy()
        joined.filter_batch(reqs_to_exclude=still_chunking)
        if step.forward_mode.is_mixed():
            if envs.SGLANG_DEBUG_INFLIGHT_STEPS.get():
                tail = step.reqs[step.first_decode_row :]
                assert len(tail) == len(self.rows.reqs) and all(
                    a is b for a, b in zip(tail, self.rows.reqs)
                ), "a mixed step's decode rows are not the running rows"
            self.rows = ScheduleBatch(reqs=[])
        if not joined.is_empty():
            self.join(joined)

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
        if envs.SGLANG_DEBUG_INFLIGHT_STEPS.get():
            self._rows_at_step = (self.rows, self.rows.reqs[:])
        return self._step

    def absorb(self) -> None:
        """Take back the fields the last decode step rebound since it was
        copied: its run's, and its commit's when this runs after the commit.
        The DP extend view's fields stay out: they describe that one step, and
        the rows keep decoding."""
        step, self._step = self._step, None
        if step is None:
            return
        if envs.SGLANG_DEBUG_INFLIGHT_STEPS.get():
            self._check_absorb(step)
        fields = vars(step)
        if step.forward_mode.is_extend():
            fields = {
                k: v
                for k, v in fields.items()
                if k not in ScheduleBatch.EXTEND_VIEW_FIELDS
            }
        vars(self.rows).update(fields)
        self.rows.reqs = step.reqs[:]

    def _check_absorb(self, step: ScheduleBatch) -> None:
        # The update below overwrites every rows field, so a rows write made
        # since decode_step would be lost silently.
        rows, reqs = self._rows_at_step
        self._rows_at_step = None
        assert self.rows is rows, "the rows were replaced while a decode step ran"
        assert len(self.rows.reqs) == len(reqs) and all(
            a is b for a, b in zip(self.rows.reqs, reqs)
        ), "the rows changed while a decode step ran"
        assert len(step.reqs) == len(reqs) and all(
            a is b for a, b in zip(step.reqs, reqs)
        ), "the decode step's requests changed"
        # The step is still a decode step, or the DP extend view of one;
        # an extend mode a worker leaked would turn the rows into a prefill.
        assert step.forward_mode == ForwardMode.DECODE or (
            step.forward_mode == ForwardMode.EXTEND
            and step.num_decode_rows == len(step.reqs)
        ), f"decode step came back as {step.forward_mode}"

    def park_for_mix(self) -> None:
        """Hand the rows to a mixed step, for a loop that joins at the next
        plan: they run inside that step's batch, which that plan joins back."""
        self.rows = ScheduleBatch(reqs=[])

    def clear_for_pause(self) -> None:
        """Forget the rows, and the decode step launched from them, after a
        pause retracted them."""
        self.rows.reqs = []
        self._step = None
        self._rows_at_step = None
