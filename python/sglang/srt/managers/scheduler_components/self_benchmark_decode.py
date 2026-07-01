from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import alloc_for_extend, release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SyntheticDecodeBatchBuilder:
    """Build and clean up decode-ready batches for the self-benchmark."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def build(self, reqs: list[Req], context_length: int) -> ScheduleBatch:
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.scheduler.model_config,
            enable_overlap=self.scheduler.enable_overlap,
            spec_algorithm=self.scheduler.spec_algorithm,
            dllm_config=self.scheduler.dllm_config,
        )
        if getattr(self.scheduler, "enable_hisparse", False):
            batch.hisparse_coordinator = self.scheduler.hisparse_coordinator

        self._place_context_cache(batch, context_length)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )

        prefill_output_tokens = torch.tensor(
            [req.output_ids[-1] for req in reqs],
            dtype=torch.int64,
            device=batch.device,
        )
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(bonus_tokens=prefill_output_tokens),
        )
        batch.input_ids = None
        return batch

    def _place_context_cache(self, batch: ScheduleBatch, context_length: int) -> None:
        """Allocate synthetic context through the scheduler's canonical path."""
        batch_size = len(batch.reqs)
        total_context_tokens = context_length * batch_size
        batch.forward_mode = ForwardMode.EXTEND
        batch.prefix_lens = [0] * batch_size
        batch.extend_lens = [context_length] * batch_size
        batch.extend_num_tokens = total_context_tokens
        root_node = getattr(batch.tree_cache, "root_node", None)
        if root_node is not None:
            for req in batch.reqs:
                req.last_node = root_node
                req.last_host_node = root_node
                req.best_match_node = root_node
        batch.seq_lens_cpu = torch.full(
            (batch_size,), context_length, dtype=torch.int64
        )
        batch.seq_lens = batch.seq_lens_cpu.to(batch.device, non_blocking=True)
        batch.orig_seq_lens = torch.full(
            (batch_size,), context_length, dtype=torch.int32, device=batch.device
        )
        batch.seq_lens_sum = total_context_tokens
        (
            batch.out_cache_loc,
            batch.req_pool_indices,
            batch.req_pool_indices_cpu,
        ) = alloc_for_extend(batch)

        for req in batch.reqs:
            req.synthetic_benchmark_kv_placed = True

    def cleanup(self, reqs: list[Req]) -> None:
        for req in reqs:
            if getattr(req, "synthetic_benchmark_kv_placed", False):
                release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)
            elif req.req_pool_idx is not None:
                self.scheduler.req_to_token_pool.free(req)
