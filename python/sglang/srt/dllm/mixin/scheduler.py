from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    @staticmethod
    def _truncate_dllm_tokens_for_finish(req: Req, token_ids: List[int]) -> List[int]:
        remaining = req.sampling_params.max_new_tokens - len(req.output_ids)
        if remaining <= 0:
            return []

        token_ids = token_ids[:remaining]
        if req.sampling_params.ignore_eos:
            return token_ids

        stop_ids: Set[int] = set(req.sampling_params.stop_token_ids or [])
        stop_ids.update(req.eos_token_ids or [])

        tokenizer = req.tokenizer
        if tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                stop_ids.add(eos_token_id)
            stop_ids.update(getattr(tokenizer, "additional_stop_token_ids", []) or [])

        for i, token_id in enumerate(token_ids):
            if token_id in stop_ids:
                return token_ids[: i + 1]

        return token_ids

    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)
        # Fine-grained DLLM scheduler profiling
        self._dllm_sched_prof = (
            self.dllm_config is not None
            and self.dllm_config.algorithm_config.get("profile", False)
        )
        if self._dllm_sched_prof:
            import time

            self._dllm_time = time
            self._dsp_n = 0
            # process_batch_result_dllm sub-timings
            self._dsp_sync = 0.0
            self._dsp_tolist = 0.0
            self._dsp_kv_free = 0.0
            self._dsp_stream = 0.0
            self._dsp_stats = 0.0
            # get_new_batch_dllm sub-timings
            self._dsp_prepare = 0.0
            self._dsp_process = 0.0
            self._dsp_create = 0.0

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Build the next DLLM batch (EXTEND for prompt caching or DLLM_EXTEND for denoising)."""
        _prof = self._dllm_sched_prof
        if _prof:
            _gt0 = self._dllm_time.perf_counter()

        if self.enable_priority_scheduling:
            self.running_batch.batch_is_full = False

        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        adder = self._create_dllm_prefill_adder(running_bs)

        self._prepare_staging_reqs()
        self._fetch_waiting_reqs()

        if _prof:
            _gt1 = self._dllm_time.perf_counter()

        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        if _prof:
            _gt2 = self._dllm_time.perf_counter()

        batch = self._create_dllm_batch(can_run_list, forward_mode)

        if _prof:
            _gt3 = self._dllm_time.perf_counter()
            self._dsp_prepare += _gt1 - _gt0
            self._dsp_process += _gt2 - _gt1
            self._dsp_create += _gt3 - _gt2
            self._dsp_n += 1
            if self._dsp_n % 500 == 0:
                n = self._dsp_n
                logger.info(
                    "DLLM SCHED PROFILE (%d iters): "
                    "process_result[sync=%.2f tolist+kv=%.2f stream=%.2f stats=%.2f]ms  "
                    "get_batch[prepare=%.2f process=%.2f create=%.2f]ms",
                    n,
                    self._dsp_sync / n * 1000,
                    self._dsp_tolist / n * 1000,
                    self._dsp_stream / n * 1000,
                    self._dsp_stats / n * 1000,
                    self._dsp_prepare / n * 1000,
                    self._dsp_process / n * 1000,
                    self._dsp_create / n * 1000,
                )

        return batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        _prof = self._dllm_sched_prof
        if _prof:
            _pt0 = self._dllm_time.perf_counter()

        if result.copy_done is not None:
            result.copy_done.synchronize()

        if _prof:
            _pt1 = self._dllm_time.perf_counter()

        if not batch.forward_mode.is_dllm_extend():
            for req in batch.reqs:
                req.fill_ids = []
            self.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=False,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            return

        if result.next_token_ids:
            self.token_to_kv_pool_allocator.free_group_begin()

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                next_token_ids = result.next_token_ids[idx].tolist()
                next_token_ids = self._truncate_dllm_tokens_for_finish(
                    req, next_token_ids
                )
                new_tokens = len(next_token_ids)
                if new_tokens == 0:
                    # Free the entire allocated block to prevent kv_committed_len
                    # inflation. Without this, cache_finished_req frees only
                    # len(origin_input_ids + output_ids) positions which is less
                    # than kv_committed_len, permanently leaking block_size tokens.
                    rejected = self.dllm_config.block_size
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv_allocated_len = free_start
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.metrics_reporter.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)

                if new_tokens < self.dllm_config.block_size:
                    rejected = self.dllm_config.block_size - new_tokens
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv_allocated_len = free_start

                req.check_finished_stop_before_length(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                    req.time_stats.set_completion_time()

            if _prof:
                _pt2 = self._dllm_time.perf_counter()

            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)

            if _prof:
                _pt3 = self._dllm_time.perf_counter()

            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

        if _prof and batch.forward_mode.is_dllm_extend():
            _pt4 = self._dllm_time.perf_counter()
            self._dsp_sync += _pt1 - _pt0
            self._dsp_tolist += _pt2 - _pt1
            self._dsp_stream += _pt3 - _pt2
            self._dsp_stats += _pt4 - _pt3

    def _prepare_staging_reqs(self: Scheduler) -> None:
        """Rebuild fill_ids and set prefix_indices for the next scheduling round.

        For each staged request, append a fresh block of mask tokens to fill_ids
        and set prefix_indices from the committed KV in req_to_token_pool so the
        next DLLM_EXTEND forward can attend to the full previously-denoised prefix.

        Note: the model writes correct KV for every position during DLLM_EXTEND
        (save_kv_cache=True in nemotron_labs_dllm.py), so no separate KV-update EXTEND
        pass is needed between blocks.
        """
        for req in self.dllm_manager.staging_queue:
            req.init_next_round_input()
            if req.req_pool_idx is not None and req.kv_committed_len > 0:
                kv_len = req.kv_committed_len
                # Convert to int64: req_to_token_pool uses int32, but the Triton
                # write_req_to_token_pool_triton kernel casts prefix_tensor to
                # int64* — reading int32 data as int64 corrupts prefix positions.
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len
                ].to(torch.int64)
                req.determine_dllm_phase()
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        self.dllm_manager.staging_queue = []

    def _fetch_waiting_reqs(self: Scheduler):
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))
        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler) -> bool:
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True
        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_scheduling
        ):
            self.running_batch.batch_is_full = True
            return True
        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Decide batch type and populate adder.

        Priority:
          1. INCOMING_PREFILL requests → causal EXTEND to cache prompt KV.
          2. STAGING_DECODE / INCOMING_DECODE requests → DLLM_EXTEND denoising.
        """
        incoming_prefill = [
            req
            for req in self.dllm_manager.waiting_queue
            if req.dllm_phase == DllmReqPhase.INCOMING_PREFILL
        ]
        if incoming_prefill:
            self._process_incoming_prefill_reqs(adder, incoming_prefill)
            return ForwardMode.EXTEND

        # Try prefill batch (STAGING_PREFILL only)
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return ForwardMode.DLLM_EXTEND

    def _process_incoming_prefill_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> None:
        """Schedule INCOMING_PREFILL requests as a causal EXTEND (prompt caching)."""
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_scheduling
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            req.init_prompt_cache_input()
            # Ensure last_node is set so dec_lock_ref is safe.
            if req.last_node is None and hasattr(self.tree_cache, "root_node"):
                req.last_node = self.tree_cache.root_node
            res = adder.add_dllm_prompt_cache_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
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
            self.adder,
            self.running_batch.reqs,
            getattr(self, "enable_priority_scheduling", False),
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        res = AddReqResult.CONTINUE
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_scheduling
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res
        return AddReqResult.CONTINUE


class DllmManager:
    """Manages DLLM request queues.

    waiting_queue: all active DLLM requests (persists across rounds).
    staging_queue: requests scheduled this round (cleared after each round).
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        assert self.dllm_config is not None
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")
        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]
