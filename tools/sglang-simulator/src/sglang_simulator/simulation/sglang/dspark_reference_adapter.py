"""Reference lifecycle and transactional source-event cost preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from sglang_simulator.simulation.sglang.dspark import DSparkStep, DSparkStepResult
from sglang_simulator.simulation.sglang.dspark_reference import (
    DSparkReferenceLedger,
    ReferenceCPUView,
    _require,
)
from sglang_simulator.time_predictor.base import PredictorError


@dataclass(frozen=True)
class DSparkReferenceAdapter:
    """Consume source-owned events, deferring retirement until ownership drains.

    The source event driver supplies native selection and settlement order.
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
            request.rid
            for result in ledger.pending
            for request in result.local_group.requests
        }
        if ledger.prepared is not None:
            owned.update(
                request.rid for request in ledger.prepared.local_group.requests
            )
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
            for request in head.local_group.requests
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
        <= (
            set()
            if previous is None
            else {r.rid for r in previous.local_group.requests}
        ),
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
            and self.base_ledger.local_attention_dp_rank
            == self.prepared_ledger.local_attention_dp_rank
            == self.view.local_attention_dp_rank
            and self.view.is_idle
            == (self.prepared_ledger.prepared.local_group.forward_mode == "IDLE")
            and self.previous_forward_id == previous
            and type(self.previous_forward_id) is type(previous),
            "reference transaction does not bind its exact prepared step and prior result",
        )

    def with_resolved_executions(self, step):
        """Accept agreed execution JSON before global prediction.

        The peer exchange and source validator remain external owners. Only
        execution JSON may differ; all before-work request, topology, runtime
        and forward facts stay frozen. CPU vectors and prior ownership survive.
        """
        _require(
            type(step) is DSparkStep
            and step.forward_id == self.step.forward_id
            and step.topology == self.step.topology
            and step.runtime_profile_json == self.step.runtime_profile_json
            and all(
                replace(group, execution_json=before.execution_json) == before
                for group, before in zip(step.groups, self.step.groups, strict=True)
            ),
            "reference execution resolution changed frozen before-work",
        )
        prepared = replace(
            self.prepared_ledger,
            prepared=replace(self.prepared_ledger.prepared, step=step),
        )
        return replace(self, prepared_ledger=prepared, step=step)

    def bind_result(self, result):
        """Bind shared global prediction data to this receiving native owner.

        Both owners must already agree on the complete resolved global step.
        This does not merge independently resolved peer steps, run prediction,
        publish or advance a clock. Call before building this owner's native result.
        """
        _require(
            type(result) is DSparkStepResult and result.step == self.step,
            "reference local result requires the complete matching global step",
        )
        return replace(
            result,
            step=self.step,
            local_attention_dp_rank=self.base_ledger.local_attention_dp_rank,
        )

    def complete(self, result, *, current_ledger, process_previous_after=True):
        """Publish current, then retire only the frozen prior FIFO entry.

        Calculate this candidate before committing external publication/session
        state. Assign it only after those operations succeed; this method cannot
        roll back another owner or an external callback.
        """
        _require(
            current_ledger is self.base_ledger,
            "reference transaction is stale relative to its current ledger owner",
        )
        _require(
            type(process_previous_after) is bool,
            "reference post-publication settlement policy is unknown",
        )
        candidate = self.prepared_ledger.publish(result)
        if (
            process_previous_after
            and self.previous_forward_id is not None
            and not self.process_previous_before
        ):
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
    allocation_ledger=None,
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
    if allocation_ledger is None:
        prepared = ledger.prepare(unresolved_step, page_size=page_size)
    else:
        # Selection/allocation happened before a possible native sync boundary.
        # Only output count may advance between that event and worker entry.
        _require(
            type(allocation_ledger) is DSparkReferenceLedger
            and allocation_ledger.prepared is not None
            and allocation_ledger.local_attention_dp_rank
            == ledger.local_attention_dp_rank
            and type(unresolved_step) is DSparkStep
            and unresolved_step.topology == allocation_ledger.prepared.step.topology,
            "reference allocation snapshot is absent",
        )
        selected = allocation_ledger.prepared.step
        rank = ledger.local_attention_dp_rank
        group = unresolved_step.groups[rank]
        selected_group = allocation_ledger.prepared.local_group
        rebound = replace(
            unresolved_step,
            groups=tuple(
                replace(
                    group,
                    requests=tuple(
                        replace(now, output_tokens_before=then.output_tokens_before)
                        for now, then in zip(
                            group.requests, selected_group.requests, strict=True
                        )
                    ),
                )
                if peer.attention_dp_rank == rank
                else peer
                for peer in unresolved_step.groups
            ),
        )
        _require(
            rebound == selected
            and allocation_ledger.pending == ledger.pending
            and allocation_ledger.last_forward_id == ledger.last_forward_id,
            "reference worker input differs from its frozen selection",
        )
        prepared = replace(
            allocation_ledger,
            prepared=replace(allocation_ledger.prepared, step=unresolved_step),
        )
    if process_previous_before:
        prepared = prepared.process_result(
            previous, skipped_request_ids=skipped_previous_ids
        )
    view = prepared.prepared.resolve(
        future_indices_ready=future_indices_ready,
        needs_cpu_seq_lens=needs_cpu_seq_lens,
    )
    execution = execution_resolver(
        group=prepared.prepared.local_group,
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
        groups=tuple(
            replace(group, execution_json=encoded)
            if group.attention_dp_rank == ledger.local_attention_dp_rank
            else group
            for group in unresolved_step.groups
        ),
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
