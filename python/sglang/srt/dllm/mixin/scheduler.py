from __future__ import annotations

import logging
from array import array
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, List, Optional, Set

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    release_req,
    unresumable_abort_message,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(
            dllm_config=self.dllm_config,
            abort=self._abort_dllm_req,
        )

    def _abort_dllm_req(self: Scheduler, req: Req) -> None:
        self.request_retirer.abort(req)
        logger.debug(f"Abort dLLM queued request. {req.rid=}")

    def validate_dllm_request(self: Scheduler, req: Req) -> Optional[str]:
        if self.dllm_config is None:
            return None
        return self.dllm_config.validate_request(req)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch, staged: List[Req]
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self._ack_waiting_health_checks()
        self.waiting_queue.reorder(self.policy.calc_priority(self.waiting_queue))

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # The last round's requests start their next block.
        for req in staged:
            req.init_next_round_input()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode, staging_result = self._process_dllm_batches(
            adder, running_batch=running_batch
        )

        can_run_list = adder.can_run_list
        if not can_run_list:
            self._maybe_break_dllm_staging_stall(staging_result, running_batch)
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if (
            self.dllm_config.requires_separate_context_encoding
            and not batch.forward_mode.is_dllm_extend()
        ):
            self.metrics_reporter.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=result.can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            return

        fdfo_mode = self.dllm_config.first_done_first_out_mode
        assert not fdfo_mode or result.accept_length_per_req_cpu is not None, (
            "FDFO dLLM result is missing accept lengths."
        )

        # FDFO also commits unresolved blocks so their KV can be reused.
        if fdfo_mode or result.next_token_ids:
            block_size = self.dllm_config.block_size
            algo_states = result.dllm_algo_state

            self.token_to_kv_pool_allocator.free_group_begin()
            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                if not fdfo_mode:
                    next_token_ids = result.next_token_ids[idx].tolist()
                    new_tokens = len(next_token_ids)
                    if new_tokens == 0:
                        continue

                    req.full_untruncated_fill_ids[
                        req.extend_end - new_tokens : req.extend_end
                    ] = array("q", next_token_ids)
                    self.metrics_reporter.num_generated_tokens += new_tokens

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)
                    self._finish_dllm_request_if_needed(req)
                    continue

                next_token_ids = result.next_token_ids[idx]
                assert len(next_token_ids) == block_size

                if result.accept_length_per_req_cpu[idx] == 0:
                    # Unresolved: keep partial state and KV for the next FDFO round.
                    req.dllm_incomplete_ids = array("q", next_token_ids)
                    req.dllm_algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm_incomplete_ids = array("q")
                req.dllm_algo_state = None

                # Mirror the resolved block into the committed fill ids so the
                # prefix cache keys on the real tokens, not the mask block, next
                # round. Index relative to extend_end (the truncated/
                # committed length), which can be shorter than
                # full_untruncated_fill_ids when the staging adder truncates the
                # block to the KV budget.
                req.full_untruncated_fill_ids[
                    req.extend_end - block_size : req.extend_end
                ] = array("q", next_token_ids)

                len_input = len(req.origin_input_ids)
                len_fill = req.extend_end
                if len_fill <= len_input:
                    continue

                if len_fill - len(next_token_ids) < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                self.metrics_reporter.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))
                self._finish_dllm_request_if_needed(req)

            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _finish_dllm_request_if_needed(self: Scheduler, req: Req) -> None:
        if (
            not req.finished()
            and req.seqlen + self.dllm_config.block_size > self.model_config.context_len
        ):
            req.finish_outside_step(FINISH_LENGTH(length=len(req.output_ids)))

        if req.finished():
            self.request_retirer.release(
                req,
                checkpoint=not self.dllm_config.requires_separate_context_encoding,
            )
            req.time_stats.set_completion_time()

    def _stash_dllm_context(self: Scheduler, req: Req) -> None:
        context_len = req.dllm_block_offset
        block_size = self.dllm_config.block_size
        assert req.extend_end == context_len + block_size
        assert req.kv is not None and req.kv.req_pool_idx is not None

        allocated_len = req.kv.kv_allocated_len
        page_size = self.token_to_kv_pool_allocator.page_size
        free_start = -(-context_len // page_size) * page_size
        if free_start < allocated_len:
            slots = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, free_start:allocated_len
            ]
            self.token_to_kv_pool_allocator.free_segment(slots, start_pos=free_start)

        req.kv.set_row_len(context_len)
        assert req.kv.max_evicted_seqlen <= context_len
        req.extend_end = context_len
        self.stash_chunked_request(req)

    def finish_dllm_forward(self: Scheduler, req: Req) -> None:
        """Commit or discard KV after one scheduler-visible dLLM forward."""
        fdfo_mode = self.dllm_config.first_done_first_out_mode
        if fdfo_mode and req.dllm_incomplete_ids:
            return

        if req.is_dllm_prefill():
            self.stash_chunked_request(req)
        elif self.dllm_config.requires_separate_context_encoding:
            self._stash_dllm_context(req)
        else:
            # The row stays with the request between blocks: a later abort
            # releases its KV and tree lock through it.
            self.stash_chunked_request(req)

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            self.dllm_manager.add_waiting_reqs(self.waiting_queue[:num_requests_to_add])

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
            request_retirer=self.request_retirer,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> tuple[ForwardMode, AddReqResult]:
        """Process prefill or decode batches for DLLM.

        Returns the forward mode and the staging pass's result, so the caller
        can tell a round refused by the staging capacity gate (NO_TOKEN) from
        one that simply had nothing to admit."""
        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            staging_result = self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
            forward_mode = (
                ForwardMode.EXTEND
                if self.dllm_config.requires_separate_context_encoding
                else ForwardMode.DLLM_EXTEND
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            staging_result = self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )
            forward_mode = ForwardMode.DLLM_EXTEND
        return forward_mode, staging_result

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process a batch, separating staging and incoming requests.

        Returns NO_TOKEN when either pass refused a row for lack of capacity
        (the stall breaker's signal), the staging pass's non-CONTINUE result
        otherwise, and CONTINUE when nothing was refused."""
        staging_result = AddReqResult.CONTINUE
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return staging_result

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            incoming_result = self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )
            if incoming_result == AddReqResult.NO_TOKEN:
                # Retained rows can starve incoming rows the same way they
                # starve staged fresh rows: the pool they hold refuses every
                # admission while this branch keeps winning the
                # prefill-over-decode priority, so nothing ever runs.
                staging_result = incoming_result
        return staging_result

    def _maybe_break_dllm_staging_stall(
        self: Scheduler, staging_result: AddReqResult, running_batch: ScheduleBatch
    ) -> None:
        """Break a staging livelock after a round that admitted nothing.

        Retained FDFO rows keep their KV between blocks, so they can fill the
        pool while every one of them needs a *fresh* block (each row's last
        block resolved mid-generation). The capacity gate in
        `add_dllm_staging_req` then refuses every row, the round admits
        nothing, no forward runs, and nothing can ever resolve, finish, or
        free KV again — a permanent livelock on a small pool.

        Fire only in that exact state: the round reported NO_TOKEN (a
        capacity refusal, not an empty queue), the round admitted nothing
        (the caller checked `can_run_list`), and no running batch holds
        in-flight work that could still free KV (dLLM disables the overlap
        schedule, so a zero-admission round really is quiescent). Any admitted
        row makes progress that frees KV eventually, so the breaker never
        fires while progress is possible.

        Retract one retained row — the newest, which has the least committed
        work to recompute — releasing its KV uncheckpointed and requeueing it
        as an incoming request to re-prefill once the stall clears. One victim
        per stalled round: if its KV does not unstick the next round, that
        round fires again.
        """
        if staging_result != AddReqResult.NO_TOKEN or running_batch.reqs:
            return
        victims = [req for req in self.dllm_manager.waiting_queue if req.kv.holds_kv]
        if not victims:
            return
        victim = victims[-1]
        # The retraction resets the request's dLLM progress
        # (`Req.reset_dllm_progress`): the unresolved block, re-stageable only
        # on top of the KV released here, is dropped and redone after the
        # re-prefill, and the phase goes back to incoming, so the request
        # re-matches the cache and re-initializes its fill ids. It stays in
        # the dLLM waiting queue.
        if release_req(
            req=victim,
            remaing_req_count=1,
            tree_cache=self.tree_cache,
            request_retirer=self.request_retirer,
        ):
            logger.warning(
                "dLLM staging stalled: retained rows hold the KV pool and no "
                f"row could be admitted. Retracted {victim.rid} to free its KV."
            )
        else:
            victim.request_finish(
                FINISH_ABORT(
                    unresumable_abort_message(),
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            )
            # Leaves the dLLM waiting queue there.
            self.request_retirer.abort_retracted(victim, reason=victim.to_finish)
            logger.warning(
                "dLLM staging stalled and the retained request cannot be "
                f"resumed. Aborted {victim.rid} to free its KV."
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(running_batch.reqs)
            num_held_rows = sum(r.kv.holds_kv for r in adder.can_run_list)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(
                running_bs, num_held_rows=num_held_rows
            ):
                running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation.

        NO_TOKEN is a per-row refusal, not batch exhaustion: the refused row
        is simply not admitted this round (it stays staged and is retried next
        round), while rows behind it — e.g. retained-KV reuse rows that
        allocate zero fresh tokens — must still get their own admission check.
        Returning on the first NO_TOKEN would let one fresh-needing row at the
        head starve every zero-cost row behind it. NO_TOKEN from the
        post-admission recheck in `add_dllm_staging_req` leaves that row
        admitted (already in can_run_list), so continuing is correct there
        too. Still report NO_TOKEN if any row was refused, so incoming reqs
        don't jump ahead of starved staging rows (`_process_batch_by_phase`).
        """
        result = AddReqResult.CONTINUE
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                result = res
                continue

        return result


class DllmManager:
    """Manager for Diffusion LLM request scheduling: `waiting_queue` holds every
    dLLM request until it finishes, the ones a round runs included."""

    def __init__(
        self,
        dllm_config: Optional[DllmConfig],
        abort: Callable[[Req], None],
    ):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue = IndexedReqList(ListKind.DLLM, abort=self._abort)
        self._abort_one = abort

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: List[Req]) -> None:
        """Move requests into the waiting queue, with a redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        move(reqs, self.waiting_queue)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def staged_reqs(self, last_batch: Optional[ScheduleBatch]) -> List[Req]:
        """The requests the last round ran that are still waiting, in its order."""
        if last_batch is None:
            return []
        return [req for req in last_batch.reqs if req in self.waiting_queue]

    def is_empty(self) -> bool:
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def _abort(self, recv_req: AbortReq) -> None:
        for req in [r for r in self.waiting_queue if recv_req.matches(r.rid)]:
            self._abort_one(req)
