"""Explicit reference lifecycle and cost preparation; no scheduler activation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from sglang_simulator.simulation.sglang.dspark import DSparkStep
from sglang_simulator.simulation.sglang.dspark_reference import (
    DSparkReferenceLedger,
    ReferenceCPUView,
    _require,
)
from sglang_simulator.time_predictor.base import PredictorError


@dataclass(frozen=True)
class DSparkReferenceAdapter:
    """Consume source-owned events, deferring retirement until ownership drains.

    G's synchronous finish/idle callbacks cannot supply reference event order or
    membership. No factory or scheduler hook constructs this adapter.
    """

    ledger: DSparkReferenceLedger = field(default_factory=DSparkReferenceLedger)
    retiring_request_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _require(
            type(self.ledger) is DSparkReferenceLedger
            and type(self.retiring_request_ids) is tuple
            and all(type(rid) is str and bool(rid) for rid in self.retiring_request_ids)
            and len(set(self.retiring_request_ids)) == len(self.retiring_request_ids),
            "reference adapter requires a typed ledger and unique retirement intents",
        )
        _require(
            all(
                not self.ledger.request_for(rid).active
                for rid in self.retiring_request_ids
            ),
            "reference retirement intent requires an inactive request",
        )

    def _install(self, ledger):
        owned = {
            request.rid for result in ledger.pending for request in result.step.requests
        }
        if ledger.prepared is not None:
            owned.update(request.rid for request in ledger.prepared.step.requests)
        remaining = []
        for rid in self.retiring_request_ids:
            if rid in owned:
                remaining.append(rid)
            else:
                ledger = ledger.forget(rid)
        return replace(self, ledger=ledger, retiring_request_ids=tuple(remaining))

    def prepare_current(self, step, *, page_size):
        return self._install(self.ledger.prepare(step, page_size=page_size))

    def resolve_current(self, *, future_indices_ready, needs_cpu_seq_lens):
        _require(
            self.ledger.prepared is not None, "reference relay has no prepared batch"
        )
        return self.ledger.prepared.resolve(
            future_indices_ready=future_indices_ready,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
        )

    def publish_current(self, result):
        return self._install(self.ledger.publish(result))

    def process_head(self, forward_id):
        _require(
            bool(self.ledger.pending), "reference processing has no pending result"
        )
        head = self.ledger.pending[0]
        skipped = tuple(
            request.rid
            for request in head.step.requests
            if request.rid in self.retiring_request_ids
        )
        return self._install(
            self.ledger.process_result(forward_id, skipped_request_ids=skipped)
        )

    def retire_request(self, rid):
        """Record source retirement, including prefill finish and retraction.

        Process a naturally finishing head before issuing its retirement event.
        Retirement before processing means an already finished/retracted/aborted
        request whose pending work must be skipped.
        """
        ledger = self.ledger.mark_inactive(rid)
        intents = self.retiring_request_ids
        if rid not in intents:
            intents = (*intents, rid)
        return replace(self, ledger=ledger, retiring_request_ids=intents)._install(
            ledger
        )

    def drained_reset(self):
        return replace(self, ledger=self.ledger.clear(), retiring_request_ids=())


def _previous_plan(ledger, process_previous_before, skipped_previous_ids):
    _require(
        type(ledger) is DSparkReferenceLedger, "reference bridge needs a typed ledger"
    )
    _require(
        type(process_previous_before) is bool,
        "reference previous-result synchronization policy is unknown",
    )
    _require(
        type(skipped_previous_ids) is tuple
        and all(type(rid) is str and bool(rid) for rid in skipped_previous_ids)
        and len(set(skipped_previous_ids)) == len(skipped_previous_ids),
        "reference previous-result skips must be unique explicit request IDs",
    )
    previous = ledger.pending[0] if ledger.pending else None
    _require(
        (not process_previous_before or previous is not None)
        and set(skipped_previous_ids)
        <= (set() if previous is None else {r.rid for r in previous.step.requests}),
        "reference synchronization/skips have no matching previous result",
    )
    return None if previous is None else previous.step.forward_id


@dataclass(frozen=True)
class ReferenceStepTransaction:
    """A pure candidate; the owner supplies its current ledger at completion."""

    base_ledger: DSparkReferenceLedger
    prepared_ledger: DSparkReferenceLedger
    step: DSparkStep
    view: ReferenceCPUView
    previous_forward_id: int | None
    process_previous_before: bool
    skipped_previous_ids: tuple[str, ...]

    def __post_init__(self):
        previous = _previous_plan(
            self.base_ledger, self.process_previous_before, self.skipped_previous_ids
        )
        _require(
            type(self.prepared_ledger) is DSparkReferenceLedger
            and self.prepared_ledger.prepared is not None
            and self.prepared_ledger.prepared.step is self.step
            and type(self.view) is ReferenceCPUView
            and self.previous_forward_id == previous
            and type(self.previous_forward_id) is type(previous),
            "reference transaction does not bind its exact prepared step and prior result",
        )

    def complete(self, result, *, current_ledger):
        """Publish current, then retire only the frozen prior FIFO entry.

        Calculate this candidate before committing external publication/session
        state. Assign it only after those operations succeed; this method cannot
        roll back another owner or an external callback.
        """
        _require(
            current_ledger is self.base_ledger,
            "reference transaction is stale relative to its current ledger owner",
        )
        candidate = self.prepared_ledger.publish(result)
        if self.previous_forward_id is not None and not self.process_previous_before:
            candidate = candidate.process_result(
                self.previous_forward_id,
                skipped_request_ids=self.skipped_previous_ids,
            )
        return candidate


def prepare_reference_step(
    ledger,
    unresolved_step,
    *,
    page_size,
    future_indices_ready,
    needs_cpu_seq_lens,
    process_previous_before,
    skipped_previous_ids=(),
    execution_resolver,
    native_batch,
    buffer_capacity,
    buffer_device_matches,
):
    """Bind one explicit reference preparation to the existing resolver inputs.

    The source driver supplies reference membership, relay/backend policy and
    synchronization decisions. Resolver output changes execution JSON only;
    runtime bytes and the caller's ordered worker-before request facts survive.
    No current cost/progress is requested here. This is not a source projector.
    """
    previous = _previous_plan(ledger, process_previous_before, skipped_previous_ids)
    _require(
        callable(execution_resolver), "reference execution resolver is unavailable"
    )
    _require(
        type(buffer_capacity) is int
        and buffer_capacity >= 0
        and type(buffer_device_matches) is bool,
        "reference bridge requires explicit before-proposal buffer metadata",
    )
    prepared = ledger.prepare(unresolved_step, page_size=page_size)
    if process_previous_before:
        prepared = prepared.process_result(
            previous, skipped_request_ids=skipped_previous_ids
        )
    view = prepared.prepared.resolve(
        future_indices_ready=future_indices_ready,
        needs_cpu_seq_lens=needs_cpu_seq_lens,
    )
    execution = execution_resolver(
        group=unresolved_step.groups[0],
        runtime_profile_json=unresolved_step.runtime_profile_json,
        native_batch=native_batch,
        buffer_capacity=buffer_capacity,
        buffer_device_matches=buffer_device_matches,
        reference_view=view,
    )
    _require(
        type(execution) is dict,
        "reference execution resolver must return a JSON object",
    )
    execution = {
        **execution,
        "draft_buffer_capacity": buffer_capacity,
        "draft_buffer_device_matches": buffer_device_matches,
    }
    try:
        encoded = json.dumps(
            execution, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise PredictorError(
            "invalid_reference_state",
            "reference execution resolver returned invalid JSON",
        ) from error
    step = replace(
        unresolved_step,
        groups=(replace(unresolved_step.groups[0], execution_json=encoded),),
    )
    prepared = replace(prepared, prepared=replace(prepared.prepared, step=step))
    return ReferenceStepTransaction(
        ledger,
        prepared,
        step,
        view,
        previous,
        process_previous_before,
        skipped_previous_ids,
    )
