import logging
from typing import List, Optional

import numpy as np
import torch
import triton
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    release_kv_cache,
    alloc_token_slots
)
from sglang.srt.speculative.spec_utils import (
    TREE_SPEC_KERNEL_AVAILABLE,
    assign_req_to_token_pool,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)

logger = logging.getLogger(__name__)

USE_FULL_MASK = True

class LlguidanceWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

    def clear_cache_pool(self):
        pass

    def forward_batch_generation(self, batch: ScheduleBatch, **kwargs) -> GenerationBatchResult:
        # Update fields
        if not batch.forward_mode.is_extend():
            batch.input_ids = batch.output_ids
            batch.output_ids = None
            batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

            for req in batch.reqs:
                req.kv_committed_len += 1
                req.kv_allocated_len += 1

            bs = len(batch.reqs)

            # Update seq_lens after allocation
            if batch.enable_overlap:
                # Do not use in-place operations in the overlap mode
                batch.seq_lens = batch.seq_lens + 1
                batch.seq_lens_cpu = batch.seq_lens_cpu + 1
                batch.orig_seq_lens = batch.orig_seq_lens + 1
            else:
                # A faster in-place version
                batch.seq_lens.add_(1)
                batch.seq_lens_cpu.add_(1)
                batch.orig_seq_lens.add_(1)
            batch.seq_lens_sum += bs

            # if get_global_server_args().enable_mamba_extra_buffer():
            #     self.mamba_track_indices = torch.tensor(
            #         [
            #             req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
            #             for req in self.reqs
            #         ],
            #         dtype=torch.int64,
            #         device=self.device,
            #     )
            #     self.mamba_track_mask = torch.tensor(
            #         [
            #             sl % get_global_server_args().mamba_track_interval == 0
            #             for sl in self.seq_lens_cpu
            #         ],
            #         dtype=torch.bool,
            #         device=self.device,
            #     )

            batch.prepare_for_decode()

        model_worker_batch = batch.get_model_worker_batch()
        num_accepted_tokens = 0
        accept_lens = None        

        kwargs = {"pp_proxy_tensors": None}
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, **kwargs
        )        

        # logits_output, next_token_ids, can_run_cuda_graph = (
        #     batch_result.logits_output,
        #     batch_result.next_token_ids,
        #     batch_result.can_run_cuda_graph,
        # )
        # # batch.forward_mode = ForwardMode.DECODE

        # return GenerationBatchResult(
        #     logits_output=logits_output,
        #     next_token_ids=next_token_ids,
        #     num_accepted_tokens=num_accepted_tokens,
        #     can_run_cuda_graph=can_run_cuda_graph,
        #     accept_lens=accept_lens,
        # )

        return batch_result