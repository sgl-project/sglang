from typing import Optional

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.build_eagle_tree import TreeMaskMode
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import build_tree_kernel_efficient_tmp
from sglang.srt.speculative.eagle_worker import EAGLEWorker

# TODO: [ ] add related fields in spec info (EagleDraftInput)
# TODO: [ ] add future map related logic
# TODO: [ ] rename "spec_info" -> something else


class EAGLEWorkerV2(EAGLEWorker):
    def __call__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__call__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self.tree_mask_mode = TreeMaskMode.FULL_MASK
        self.plan_stream: CudaStream = torch.get_device_module(self.device).Stream()
        self.plan_stream_ctx = torch.cuda.stream(self.plan_stream)

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if model_worker_batch.forward_mode.is_decode():
            # FIXME(lsyin): why shall we use spec_info for both draft and verify?
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            assert draft_input.is_draft_input()
            verify_input: EagleVerifyInput = self.draft(model_worker_batch)
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch, draft_input.allocate_lens)
            return batch_output
        else:
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_output.next_draft_input = self.forward_draft_extend(
                model_worker_batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient_tmp(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            num_steps=self.speculative_num_steps,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
        )

    def verify(
        self, model_worker_batch: ModelWorkerBatch, allocate_lens: torch.Tensor
    ) -> GenerationBatchResult:
        raise NotImplementedError()

    def forward_draft_extend(self, model_worker_batch, hidden_states, next_token_ids):
        raise NotImplementedError()


def free_spec_dec_tokens_page_size_1(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    req: Req,
    allocate_len: int,
    new_seq_len: int,
):
    # FIXME(lsyin): move this function elsewhere

    # free extra allocated tokens
    if new_seq_len is None:
        # True only for overlap eagle and the current batch is decode. This seq will be part of the decode, so the final iteration's allocation is not used (i.e. this case).
        start_len = allocate_len - EagleDraftInput.ALLOC_LEN_PER_DECODE
    else:
        # True for 1) non-overlap; 2) overlap eagle and the current batch is prefill. This seq will not run extra iteration, so start_lens is passed in.
        start_len = new_seq_len
    indices_to_free = req_to_token_pool.req_to_token[req.req_pool_idx][
        start_len:allocate_len
    ]
    token_to_kv_pool_allocator.free(indices_to_free)
