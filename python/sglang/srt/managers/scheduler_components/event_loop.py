"""The flat scheduler event loop and the stage choices of each mode that runs it.

Every non-pipeline CUDA loop runs one step order:

    intake -> plan -> launch -> commit -> step end

A ``LoopSpec`` names how each mode fills those stages; ``run_flat_loop`` walks
them. The in-flight queue the mode installs sets the commit timing: depth 0
commits a step in the iteration that launched it, depth 1 in the next one.

PD prefill overlap keeps its own loop (``event_loop_overlap_disagg_prefill``):
with continuous input polling it repeats intake while the due step's result
is unfinished, which no other mode does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.inflight_queue import InflightQueue

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler


@dataclass(frozen=True)
class LoopSpec:
    # Writes the mode's flags and installs its in-flight queue.
    enter: Callable[[Scheduler], None]
    # Plans the next step.
    plan: Callable[[Scheduler], Optional[ScheduleBatch]]
    # Hands the PD queues' new work over after intake; returns what arrived.
    take_arrivals: Optional[Callable[[Scheduler], Optional[List]]] = None
    # PD decode: query the prefill DP ranks before intake.
    prefetch_prefill_dp_ranks: bool = False
    # Commits the due step before a launch that must not overlap it; returns
    # whether it disabled the overlap.
    commit_before_launch: Optional[Callable[[Scheduler, ScheduleBatch], bool]] = None
    prefetch_staging: bool = False
    # Updates the running rows at a launch; a mode that sets it also drops
    # the finished rows after the commit.
    rows_at_launch: Optional[Callable[[Scheduler, ScheduleBatch], None]] = None
    after_commit: Optional[Callable[[Scheduler], None]] = None
    # Delayed sampling of an overlapped step runs only for generation models.
    sample_only_for_generation: bool = False
    strict_mem_check: bool = False


def run_flat_loop(self: Scheduler, spec: LoopSpec) -> None:
    spec.enter(self)
    overlapped = self.inflight_steps.depth > 0

    while True:
        if self.gracefully_exit:
            break

        # Intake
        if spec.prefetch_prefill_dp_ranks and not self._engine_paused:
            # Pending rooms from the prior cycle can overlap request intake and
            # the tail of the in-flight decode graph.
            self.disagg_decode_prealloc_queue.prefetch_prefill_dp_rank_queries()
        self.ingest_requests()
        if self._engine_paused:
            self._record_scheduler_state_for_paused_engine()
            continue
        if spec.take_arrivals:
            spec.take_arrivals(self)

        # Plan and launch
        batch = spec.plan(self)
        self.cur_batch_for_debug = batch
        disable_overlap_for_batch = False
        if spec.commit_before_launch:
            disable_overlap_for_batch = spec.commit_before_launch(self, batch)
        batch_result = _launch(self, spec, batch, overlapped)

        # Commit the due step
        if overlapped:
            self._debug_check_commit_gate(
                self.last_batch is not None and not disable_overlap_for_batch
            )
        if self.inflight_steps.has_due():
            # Process the oldest batch result.
            self._commit_due_step()
        elif batch is None:
            # When the server is idle, do self-check and re-init some states
            self.on_idle()
        # Rows whose finish a commit observed leave here, after the launch: a
        # decode step copied from the rows hands back its run writes first.
        # This also covers the commit made before the launch; a depth-0 loop
        # commits only the step it launched. The readers between steps (in PD
        # decode, preallocation and the prebuilt join) then see only the rows
        # still decoding.
        if spec.rows_at_launch and (overlapped or batch):
            self.leave_finished_rows(self.decoding_batch)
        if spec.after_commit:
            spec.after_commit(self)

        # Run sample of the current batch
        # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
        if overlapped and (not spec.sample_only_for_generation or self.is_generation):
            self.launch_batch_sample_if_needed(batch_result, batch)

        # Update last_batch
        self.last_batch = batch
        self.inflight_steps.end_step(batch)
        self._debug_check_last_launched()

        if (
            spec.strict_mem_check
            and envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        ):
            self.invariant_checker.self_check_during_busy()


def _launch(
    self: Scheduler,
    spec: LoopSpec,
    batch: Optional[ScheduleBatch],
    overlapped: bool,
):
    if batch:
        if spec.prefetch_staging and self.enable_staging:
            self.maybe_prefetch_staging_for_batch(batch)
        batch_result = self.run_batch(batch)
        if overlapped:
            # Fence result processing behind this forward's shared reads.
            self._apply_war_barrier()
            self.inflight_steps.push(batch.copy(), batch_result)
            if spec.rows_at_launch:
                spec.rows_at_launch(self, batch)
        else:
            if spec.rows_at_launch:
                spec.rows_at_launch(self, batch)
            self.inflight_steps.push(batch, batch_result)
    else:
        batch_result = None
        self._sched_idled = True
    return batch_result


# Entry: the flags each mode sets and the queue it installs.


def _enter_normal(self: Scheduler) -> None:
    self.inflight_steps = InflightQueue(depth=0)
    self.rows_join_before_plan = True


def _enter_overlap(self: Scheduler) -> None:
    self.enable_skip_finishing_decode = True
    self.inflight_steps = InflightQueue(depth=1)
    self.rows_join_before_plan = True


def _enter_pd(self: Scheduler) -> None:
    self.inflight_steps = InflightQueue(depth=0)


def _enter_pd_decode_overlap(self: Scheduler) -> None:
    self.inflight_steps = InflightQueue(depth=1)
    self.last_batch: Optional[ScheduleBatch] = None


# Arrivals: the work the PD queues hand over after intake.


def _take_bootstrapped(self: Scheduler) -> List:
    bootstrapped_reqs = self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
    self.waiting_queue.extend(bootstrapped_reqs)
    return bootstrapped_reqs


def _process_decode_queue(self: Scheduler) -> None:
    self.process_decode_queue()


# Plans: each mode's schedule entry point.


def _plan(self: Scheduler) -> Optional[ScheduleBatch]:
    plan = self.get_next_batch_to_run(
        decoding_batch=self.decoding_batch, last_batch=self.last_batch
    )
    return plan.batch_to_run


def _plan_pd_prefill(self: Scheduler) -> Optional[ScheduleBatch]:
    plan = self.get_next_disagg_prefill_batch_to_run(
        decoding_batch=self.decoding_batch, last_batch=self.last_batch
    )
    batch = plan.batch_to_run
    return self.ngram_embedding_manager.prepare_for_forward(
        batch, chunked_req=self.chunked_req
    )


def _plan_pd_decode(self: Scheduler) -> Optional[ScheduleBatch]:
    plan = self.get_next_disagg_decode_batch_to_run(decoding_batch=self.decoding_batch)
    batch = plan.batch_to_run
    return self.ngram_embedding_manager.prepare_for_forward(
        batch, chunked_req=self.chunked_req
    )


# Commits before a launch: the due step a launch must not overlap.


def _commit_if_overlap_disabled(self: Scheduler, batch: ScheduleBatch) -> bool:
    disable_overlap_for_batch = self.is_disable_overlap_for_batch(
        batch, last_batch=self.last_batch
    )

    # If we do not need to overlap the current batch with the last batch,
    # we can process the last batch immediately.
    if disable_overlap_for_batch:
        self._commit_due_step()
        # Opportunistic flush at the disable_overlap sync boundary:
        # forward_stream is idle (prev forward drained, next not launched),
        # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
        if self.enable_unified_memory:
            try:
                self.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass
    return disable_overlap_for_batch


def _commit_if_overlap_disabled_pd_decode(
    self: Scheduler, batch: ScheduleBatch
) -> bool:
    # overlap + spec + grammar is unsupported (would desync DP ranks).
    disable_overlap_for_batch = self.is_disable_overlap_for_batch(
        batch, last_batch=self.last_batch
    )

    if disable_overlap_for_batch and self.inflight_steps.has_due():
        self._commit_due_step()
    return disable_overlap_for_batch


# Running rows at a launch.


def _update_rows(self: Scheduler, batch: ScheduleBatch) -> None:
    self._update_rows_at_launch(batch)


def _absorb_decode_step(self: Scheduler, batch: ScheduleBatch) -> None:
    # The rows take back what their decode step's run rebound.
    self.decoding_batch.absorb()


def _process_prefill_inflight_queue(self: Scheduler) -> None:
    self.process_disagg_prefill_inflight_queue()


NORMAL_LOOP = LoopSpec(
    enter=_enter_normal,
    plan=_plan,
    rows_at_launch=_update_rows,
    strict_mem_check=True,
)

# Overlaps the CPU processing and GPU computation.
OVERLAP_LOOP = LoopSpec(
    enter=_enter_overlap,
    plan=_plan,
    commit_before_launch=_commit_if_overlap_disabled,
    rows_at_launch=_update_rows,
    sample_only_for_generation=True,
    strict_mem_check=True,
)

PD_PREFILL_LOOP = LoopSpec(
    enter=_enter_pd,
    plan=_plan_pd_prefill,
    take_arrivals=_take_bootstrapped,
    prefetch_staging=True,
    after_commit=_process_prefill_inflight_queue,
)

PD_DECODE_LOOP = LoopSpec(
    enter=_enter_pd,
    plan=_plan_pd_decode,
    take_arrivals=_process_decode_queue,
    prefetch_prefill_dp_ranks=True,
    rows_at_launch=_absorb_decode_step,
)

PD_DECODE_OVERLAP_LOOP = LoopSpec(
    enter=_enter_pd_decode_overlap,
    plan=_plan_pd_decode,
    take_arrivals=_process_decode_queue,
    prefetch_prefill_dp_ranks=True,
    commit_before_launch=_commit_if_overlap_disabled_pd_decode,
    rows_at_launch=_absorb_decode_step,
)
