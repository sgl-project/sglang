"""Source-owned native scheduler events for the existing DSpark reference ledger.

No membership selection, forward cost or acceptance policy lives here. The
native scheduler selects/prepares, decides synchronization, publishes through
FutureMap, and processes results. This owner freezes those events for the SDK.
"""

from dataclasses import dataclass

from sglang_simulator.simulation.sglang.dspark import dspark_result_from_native
from sglang_simulator.simulation.sglang.dspark_reference import _require
from sglang_simulator.simulation.sglang.dspark_reference_adapter import (
    DSparkReferenceAdapter,
    prepare_reference_step,
)
from sglang_simulator.simulation.sglang.dspark_runtime import cpu_int, cpu_vector
from sglang_simulator.time_predictor.base import PredictorError


@dataclass
class _Selection:
    batch: object
    base: DSparkReferenceAdapter
    allocation: object
    future_indices_ready: bool
    needs_cpu_seq_lens: bool | None
    simulator_needs_cpu_seq_lens: bool | None
    page_size: int
    sync: bool = False
    decision_known: bool = False
    processed_previous: bool = False
    skipped_previous: tuple[str, ...] = ()
    retirements: tuple[str, ...] = ()


class DSparkReferenceDriver:
    """One scheduler/worker run; failure after external effects is terminal."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.adapter = DSparkReferenceAdapter()
        self.selection = None
        self.processing = None
        self.failure = None
        self.owner = None
        self.retired_request_ids = set()
        self.callback_wall_seconds = 0.0

    def check(self):
        if self.failure is not None:
            raise PredictorError(
                "invalid_reference_state",
                "reference source event driver failed; restart the run",
                failure=self.failure,
            )

    def fail(self, event, error):
        if self.failure is None:
            self.failure = {"event": event, "error": str(error)}

    def take_callback_wall_seconds(self):
        wall = self.callback_wall_seconds
        self.callback_wall_seconds = 0.0
        return wall

    def bind(self, scheduler):
        self.check()
        _require(
            self.owner is None or self.owner is scheduler,
            "reference run changed scheduler owner",
        )
        _require(
            scheduler.enable_overlap is True,
            "native reference events require the original overlap scheduler and relay",
        )
        self.owner = scheduler

    def selected(self, scheduler, batch, *, prior_requests=()):
        self.bind(scheduler)
        _require(
            self.selection is None and self.processing is None,
            "reference selection still owns work",
        )
        requests = (
            *prior_requests,
            *getattr(scheduler, "waiting_queue", ()),
            *(() if batch is None else batch.reqs),
        )
        _require(
            not any(
                getattr(req, "is_retracted", False)
                or getattr(req, "retraction_count", 0) > 0
                for req in requests
            ),
            "reference retraction/re-entry with retained output is unsupported",
        )
        if batch is None:
            return
        before = self.runtime.capture(batch, resolve=False, reference=True)
        page_size = cpu_int(batch.token_to_kv_pool_allocator.page_size, "page_size")
        allocation = self.adapter.ledger.prepare(before, page_size=page_size)
        prepared = allocation.prepared
        _require(
            cpu_vector(batch.seq_lens_cpu, len(batch.reqs), "selection seq_lens_cpu")
            == prepared.seed_seq_lens_cpu,
            "native selection seed differs from reference committed KV",
        )
        _require(
            tuple(
                cpu_int(req.kv.kv_committed_len, "kv_committed_len")
                for req in batch.reqs
            )
            == prepared.seed_seq_lens_cpu,
            "native selected request KV differs from reference committed KV",
        )
        draft = batch.spec_info
        indices = None if draft is None else draft.future_indices
        relay = indices is not None
        needs_cpu = None
        simulator_needs_cpu = None
        if relay:
            _require(
                cpu_vector(indices, len(batch.reqs), "future_indices")
                == cpu_vector(
                    batch.req_pool_indices, len(batch.reqs), "req_pool_indices"
                ),
                "native relay indices differ from selected request slots",
            )
            simulator_needs_cpu = scheduler.future_map.needs_cpu_seq_lens
            _require(
                type(simulator_needs_cpu) is bool,
                "simulator backend CPU mirror policy is unknown",
            )
            # The physical HIP backend's OR/TBO policy is a distinct source
            # contract. torch_native on the CPU engine cannot establish it.
            policy = getattr(
                self.runtime.session.cost_model, "native_source_policy", None
            )
            needs_cpu = getattr(policy, "reference_needs_cpu_seq_lens", None)
            if type(needs_cpu) is not bool:
                raise PredictorError(
                    "data_unavailable",
                    "verified source policy omits the reference backend CPU mirror policy",
                    missing_source_facts=["reference_backend_cpu_mirror_policy"],
                )
        if prepared.nxt_kv_lens_cpu is not None:
            _require(
                draft is not None
                and cpu_vector(
                    draft.nxt_kv_lens_cpu, len(batch.reqs), "nxt_kv_lens_cpu"
                )
                == prepared.nxt_kv_lens_cpu,
                "native allocation differs from frozen reference reservation",
            )
        self.selection = _Selection(
            batch,
            self.adapter,
            allocation,
            relay,
            needs_cpu,
            simulator_needs_cpu,
            page_size,
        )

    def overlap_decision(self, batch, disable_overlap):
        self.check()
        if batch is None:
            return
        selected = self.selection
        _require(
            selected is not None and selected.batch is batch,
            "reference overlap decision has no selection",
        )
        _require(not selected.decision_known, "reference overlap decision repeated")
        selected.sync = bool(disable_overlap)
        selected.decision_known = True
        _require(
            not selected.sync or bool(selected.base.ledger.pending),
            "reference sync has no previous result",
        )

    def prepare_worker(self, batch):
        self.check()
        selected = self.selection
        _require(
            selected is not None
            and selected.batch is batch
            and selected.decision_known,
            "native reference worker has no source-selected batch/overlap decision",
        )
        _require(
            selected.sync == selected.processed_previous,
            "native previous-result event order differs from its overlap decision",
        )
        prepared = selected.allocation.prepared
        native_seq = cpu_vector(batch.seq_lens, len(batch.reqs), "worker seq_lens")
        _require(
            native_seq
            == (
                prepared.published_seq_lens
                if selected.future_indices_ready
                else prepared.seed_seq_lens_cpu
            ),
            "native worker device lengths differ from the frozen source relay",
        )
        before = self.runtime.capture(batch, resolve=False, reference=True)
        _require(
            before.groups[0].forward_mode != "TARGET_VERIFY"
            or tuple(req.target_prefix_len for req in before.requests) == native_seq,
            "native worker lengths differ from previously published causal work",
        )
        transaction = prepare_reference_step(
            selected.base.ledger,
            before,
            page_size=selected.page_size,
            future_indices_ready=selected.future_indices_ready,
            needs_cpu_seq_lens=selected.needs_cpu_seq_lens,
            process_previous_before=selected.sync,
            skipped_previous_ids=selected.skipped_previous,
            execution_resolver=self.runtime.execution_resolver,
            native_batch=batch,
            buffer_capacity=self.runtime._draft_buffer_capacity,
            buffer_device_matches=self.runtime._draft_buffer_capacity > 0,
            allocation_ledger=selected.allocation,
        )
        # Validate the actual CPU engine relay with its own backend policy;
        # reference query inputs retain the separately verified physical policy.
        view = selected.allocation.prepared.resolve(
            future_indices_ready=selected.future_indices_ready,
            needs_cpu_seq_lens=selected.simulator_needs_cpu_seq_lens,
        )
        _require(
            batch.seq_lens_cpu is None
            if view.seq_lens_cpu is None
            else cpu_vector(batch.seq_lens_cpu, len(batch.reqs), "worker seq_lens_cpu")
            == view.seq_lens_cpu,
            "native worker mirror differs from its source reference event",
        )
        return transaction

    def validate_owner(self, transaction):
        selected = self.selection
        _require(
            selected is not None
            and self.adapter is selected.base
            and self.adapter.ledger is transaction.base_ledger,
            "reference transaction is stale relative to its source owner",
        )

    def complete_candidate(self, transaction, prediction):
        selected = self.selection
        _require(
            selected is not None and self.adapter is selected.base,
            "reference adapter owner changed during worker call",
        )
        ledger = transaction.complete(
            prediction,
            current_ledger=self.adapter.ledger,
            process_previous_after=False,
        )
        candidate = selected.base._install(ledger)
        for rid in selected.retirements:
            candidate = candidate.retire_request(rid)
        return candidate

    def install(self, candidate):
        self.adapter = candidate
        self.retired_request_ids.update(self.selection.retirements)
        self.selection = None
        self._forget_drained()

    def _forget_drained(self):
        live = {row.rid for row in self.adapter.ledger.requests}
        # Session is the existing owner of draft history. Retirement is deferred
        # until every prepared/pending successor has released this RID.
        for rid in tuple(self.retired_request_ids):
            if rid not in live:
                self.runtime.session.forget(rid)
                self.retired_request_ids.remove(rid)

    def before_result(self, batch, native_result):
        self.check()
        prediction = dspark_result_from_native(native_result)
        _require(
            prediction is not None and self.processing is None,
            "reference result has no unique native modeled owner",
        )
        reqs = tuple(batch.reqs)
        _require(
            tuple(req.rid for req in reqs)
            == tuple(req.rid for req in prediction.step.requests),
            "reference result membership differs from its native FIFO entry",
        )
        _require(
            not any(getattr(req, "is_retracted", False) for req in reqs),
            "reference retraction with retained output is unsupported",
        )
        selected = self.selection
        if selected is not None:
            _require(
                selected.sync
                and selected.decision_known
                and not selected.processed_previous,
                "reference previous result processed before current publication without source sync",
            )
            current = selected.base._install(selected.allocation)
        else:
            current = self.adapter
        skipped = tuple(
            req.rid
            for req in reqs
            if req.finished() or req.rid in current.retiring_request_ids
        )
        for rid in skipped:
            current = current.retire_request(rid)
        candidate = current.process_head(prediction.step.forward_id)
        self.processing = (candidate, reqs, selected, skipped)

    def after_result(self):
        candidate, reqs, selected, skipped = self.processing
        for req in reqs:
            if (
                req.finished()
                and req.rid not in candidate.retiring_request_ids
                and any(row.rid == req.rid for row in candidate.ledger.requests)
            ):
                candidate = candidate.retire_request(req.rid)
            if req.finished():
                self.retired_request_ids.add(req.rid)
        if selected is not None:
            selected.processed_previous = True
            selected.skipped_previous = skipped
            selected.retirements = tuple(req.rid for req in reqs if req.finished())
        else:
            self.adapter = candidate
            self._forget_drained()
        self.processing = None

    def retire(self, rid):
        self.check()
        self.retired_request_ids.add(rid)
        if self.processing is not None:
            candidate, reqs, selected, skipped = self.processing
            if any(row.rid == rid for row in candidate.ledger.requests):
                candidate = candidate.retire_request(rid)
                self.processing = (candidate, reqs, selected, skipped)
            return
        _require(
            self.selection is None,
            "reference retirement during selection requires its source result event",
        )
        # Never-admitted queued cancellation has neither a reference row nor
        # draft history. It must not retire another request's FIFO ownership.
        if any(row.rid == rid for row in self.adapter.ledger.requests):
            self.adapter = self.adapter.retire_request(rid)
        self._forget_drained()

    def drained_reset(self):
        self.check()
        _require(
            self.selection is None and self.processing is None,
            "reference reset still owns a source event",
        )
        self.adapter = self.adapter.drained_reset()
