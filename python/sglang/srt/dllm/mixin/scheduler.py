from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

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
        self._dllm_block_buffers: Optional[dict] = None
        self._dllm_cached_sampling_info = None
        self._dllm_cached_sampling_info_reqs_id: Optional[int] = None
        self._dllm_inner_k_blocks = (
            self.dllm_config.algorithm_config.get("inner_k_blocks", 1)
            if self.dllm_config is not None
            else 1
        )
        self._dllm_stats_skip_count = 0
        self._dllm_tier_hist: dict[int, int] = {}
        self._dllm_tier_blocks: int = 0
        self._dllm_pending_tier_bs: Optional[int] = None

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Build the next DLLM batch."""
        if self.enable_priority_scheduling:
            running_batch.batch_is_full = False

        if self._should_skip_prefill(running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        adder = self._create_dllm_prefill_adder(running_bs, running_batch)

        self._prepare_staging_reqs()
        self._fetch_waiting_reqs()

        forward_mode = self._process_dllm_batches(adder, running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        batch = self._create_dllm_batch(can_run_list, forward_mode, adder, running_batch)

        pending_tier_bs = self._dllm_pending_tier_bs
        if (
            getattr(self.dllm_config, "block_size_tiers", None)
            and batch is not None
            and batch.reqs
            and pending_tier_bs is not None
        ):
            batch.dllm_block_size = pending_tier_bs

        if getattr(self.dllm_config, "block_size_tiers", None):
            # Histogram reports the block_size actually dispatched (from
            # _prepare_staging_reqs / _dllm_pending_tier_bs). Falling back to
            # select_block_size(running_bs) here is wrong because running_bs
            # is sampled at function entry, before the new batch is built,
            # so it's always the *previous* batch's running count.
            picked = (
                pending_tier_bs
                if pending_tier_bs is not None
                else self.dllm_config.select_block_size(running_bs)
            )
            dispatched_bs = (
                batch.batch_size() if batch is not None and batch.reqs else 0
            )
            self._dllm_tier_hist[picked] = self._dllm_tier_hist.get(picked, 0) + 1
            self._dllm_tier_blocks += 1
            if self._dllm_tier_blocks % 200 == 0:
                tot = self._dllm_tier_blocks
                breakdown = ", ".join(
                    f"bs={k}: {v} ({v / tot * 100:.1f}%)"
                    for k, v in sorted(self._dllm_tier_hist.items())
                )
                logger.info(
                    "DLLM TIER POLICY (%d blocks): %s "
                    "[last batch_size=%d -> block_size=%d]",
                    tot,
                    breakdown,
                    dispatched_bs,
                    picked,
                )

        return batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if not batch.forward_mode.is_dllm_extend():
            self.metrics_reporter.report_prefill_stats(
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
                block_bs = (
                    req.dllm_active_block_size
                    if req.dllm_active_block_size is not None
                    else self.dllm_config.block_size
                )
                if new_tokens == 0:
                    # No accepted tokens: release the speculative block now.
                    rejected = block_bs
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv.kv_allocated_len = free_start
                    continue

                req.full_untruncated_fill_ids[
                    req.extend_range.end - new_tokens : req.extend_range.end
                ] = array("q", next_token_ids)
                self.metrics_reporter.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)

                if new_tokens < block_bs:
                    rejected = block_bs - new_tokens
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv.kv_allocated_len = free_start

                req.check_finished_stop_before_length(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                    req.time_stats.set_completion_time()

            need_stream = any(
                r.finished() or getattr(r, "stream", False) for r in batch.reqs
            )
            if need_stream:
                self.output_streamer.stream_output(batch.reqs, batch.return_logprob)

            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        should_report_stats = True
        if (
            batch.forward_mode.is_dllm_extend()
            and not self.metrics_reporter.current_scheduler_metrics_enabled
        ):
            self._dllm_stats_skip_count += 1
            should_report_stats = self._dllm_stats_skip_count >= 20
            if should_report_stats:
                self._dllm_stats_skip_count = 0

        if should_report_stats:
            self.metrics_reporter.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )


    def maybe_run_dllm_inner_loop(self: Scheduler, batch: ScheduleBatch) -> None:
        inner_k = getattr(self, "_dllm_inner_k_blocks", 1)
        if (
            inner_k > 1
            and self.dllm_config is not None
            and batch.forward_mode == ForwardMode.DLLM_EXTEND
            and batch.batch_size() == 1
            and not batch.reqs[0].finished()
            and not self.waiting_queue
            and len(self.dllm_manager.waiting_queue) == 1
        ):
            self._dllm_run_inner_k_blocks(batch, inner_k - 1)

    def _dllm_run_inner_k_blocks(self: Scheduler, batch: ScheduleBatch, k: int) -> None:
        """Run extra single-request DLLM_EXTEND blocks in place."""
        if not batch.forward_mode.is_dllm_extend() or batch.batch_size() != 1:
            raise RuntimeError(
                "_dllm_run_inner_k_blocks called with invalid batch state: "
                f"forward_mode={batch.forward_mode}, "
                f"batch_size={batch.batch_size()}"
            )
        req = batch.reqs[0]
        for _ in range(k):
            self._set_active_block_size_for(batch)
            req.init_next_round_input()
            if req.req_pool_idx is not None and req.kv_committed_len > 0:
                kv_len = req.kv_committed_len
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len
                ].to(torch.int64)
                req.determine_dllm_phase()
                req._set_dllm_extend_range_to_fill_len()

            batch.prepare_for_dllm_block_extend(buffers=self._dllm_block_buffers)
            self._dllm_block_buffers = getattr(batch, "_dllm_block_buffers", None)
            batch.forward_mode = ForwardMode.DLLM_EXTEND
            batch.decoding_reqs = None

            result = self.run_batch(batch)
            self.process_batch_result(batch, result)

            if req.finished():
                break

    def _set_active_block_size_for(self: Scheduler, batch: ScheduleBatch) -> None:
        if not getattr(self.dllm_config, "block_size_tiers", None):
            return
        running_bs = max(1, batch.batch_size())
        bs = self.dllm_config.select_block_size(running_bs)
        batch.dllm_block_size = bs
        for req in batch.reqs:
            req.dllm_active_block_size = bs

    def _prepare_staging_reqs(self: Scheduler) -> None:
        if getattr(self.dllm_config, "block_size_tiers", None):
            staging_set = set(id(r) for r in self.dllm_manager.staging_queue)
            decode_reqs = [
                r
                for r in self.dllm_manager.waiting_queue
                if getattr(r, "dllm_phase", None)
                in (DllmReqPhase.STAGING_DECODE, DllmReqPhase.INCOMING_DECODE)
                or id(r) in staging_set
            ]
            running_bs = max(1, len(decode_reqs))
            tier_bs = self.dllm_config.select_block_size(running_bs)
            self._dllm_pending_tier_bs = tier_bs
            for req in decode_reqs:
                old_bs = req.dllm_active_block_size
                req.dllm_active_block_size = tier_bs
                if (
                    id(req) not in staging_set
                    and req.dllm_phase == DllmReqPhase.STAGING_DECODE
                    and old_bs != tier_bs
                ):
                    req.init_next_round_input()
                    if req.req_pool_idx is not None and req.kv_committed_len > 0:
                        kv_len = req.kv_committed_len
                        req.prefix_indices = self.req_to_token_pool.req_to_token[
                            req.req_pool_idx, :kv_len
                        ].to(torch.int64)
                        req.determine_dllm_phase()
                        req._set_dllm_extend_range_to_fill_len()
        for req in self.dllm_manager.staging_queue:
            req.init_next_round_input()
            if req.req_pool_idx is not None and req.kv_committed_len > 0:
                kv_len = req.kv_committed_len
                # The Triton writer reads prefix tensors as int64.
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len
                ].to(torch.int64)
                req.determine_dllm_phase()
                req._set_dllm_extend_range_to_fill_len()
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

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True
        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_scheduling
        ):
            running_batch.batch_is_full = True
            return True
        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
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
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        incoming_prefill = [
            req
            for req in self.dllm_manager.waiting_queue
            if req.dllm_phase == DllmReqPhase.INCOMING_PREFILL
        ]
        if incoming_prefill:
            self._process_incoming_prefill_reqs(adder, incoming_prefill)
            return ForwardMode.EXTEND

        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
        else:
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )

        return ForwardMode.DLLM_EXTEND

    def _process_incoming_prefill_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> None:
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
        running_batch: ScheduleBatch,
    ) -> None:
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
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

        if forward_mode == ForwardMode.DLLM_EXTEND:
            new_batch.prepare_for_dllm_block_extend(
                buffers=self._dllm_block_buffers,
            )
            self._dllm_block_buffers = getattr(new_batch, "_dllm_block_buffers", None)

            reqs_signature = tuple(id(r) for r in can_run_list)
            if (
                self._dllm_cached_sampling_info is not None
                and self._dllm_cached_sampling_info_reqs_id == reqs_signature
            ):
                new_batch.sampling_info = self._dllm_cached_sampling_info
                penalizer_orchestrator = new_batch.sampling_info.penalizer_orchestrator
                if penalizer_orchestrator is not None:
                    penalizer_orchestrator.batch = new_batch
            else:
                new_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
                    new_batch,
                    self.model_config.vocab_size,
                )
                self._dllm_cached_sampling_info = new_batch.sampling_info
                self._dllm_cached_sampling_info_reqs_id = reqs_signature
        else:
            new_batch.prepare_for_extend()

        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_batch.reqs,
            getattr(self, "enable_priority_scheduling", False),
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        res = AddReqResult.CONTINUE
        pending_tier_bs = self._dllm_pending_tier_bs
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_scheduling
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if pending_tier_bs is not None:
                req.dllm_active_block_size = pending_tier_bs
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
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res
        return AddReqResult.CONTINUE


class DllmManager:
    """Tracks active DLLM requests between scheduler rounds."""

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
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]
