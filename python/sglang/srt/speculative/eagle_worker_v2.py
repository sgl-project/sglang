import logging
from typing import List, Optional

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.build_eagle_tree import TreeMaskMode
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    build_tree_kernel_efficient_tmp,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens_tmp,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils.common import fast_topk, next_power_of_2

logger = logging.getLogger(__name__)


class EAGLEWorkerV2(EAGLEWorker):
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
        """
        Initialize an EAGLEWorkerV2 and configure speculative-drafting and CUDA plan stream state.
        
        Sets up per-decode allocation sizing for draft inputs, selects the tree mask mode for drafting, and creates a dedicated CUDA plan stream and its context used for verification/draft planning and synchronization.
        """
        super().__init__(
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
        # TODO(lsyin): potential bugs with a separate plan stream
        self.plan_stream_ctx = torch.cuda.stream(self.plan_stream)

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        """
        Dispatches the forward pass for a generation batch, using the speculative draft/verify flow for decode and a two-stage prefill for non-decode.
        
        For decode mode: expects model_worker_batch.spec_info to be an EagleDraftInput, runs a draft to produce an EagleVerifyInput, replaces model_worker_batch.spec_info with that verify input, and performs verification; returns the verification batch result. For non-decode mode: runs a target prefill with full hidden capture, then runs a draft prefill with last-hidden capture, attaches the resulting draft input to the returned batch output as `next_draft_input`.
        
        Parameters:
            model_worker_batch: the ModelWorkerBatch to process; in decode mode its `spec_info` must be an EagleDraftInput and will be replaced with an EagleVerifyInput as a side effect.
        
        Returns:
            The batch processing result containing logits and generation state; in non-decode mode the result will include `next_draft_input`.
        """
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
        """
        Prepare and execute a speculative draft pass for the provided batch and build an EagleVerifyInput for the subsequent verification stage.
        
        Runs the draft forward (using a CUDA graph replay when available or the draft_forward path), builds tree-based verify buffers (mask and positions), and returns an EagleVerifyInput populated with the selected draft tokens, mask, positions, and retrieval metadata required by the verifier.
        
        Returns:
            EagleVerifyInput: Contains `draft_token`, `custom_mask`, `positions`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling`, and related speculative parameters used for verification.
        """
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
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        """
        Selects speculative draft tokens and their parent indices across multiple speculative steps.
        
        Parameters:
            forward_batch (ForwardBatch): Input batch for draft extension; must include a populated `spec_info` (EagleDraftInput) and `out_cache_loc`.
        
        Returns:
            parent_list (torch.Tensor): Concatenated parent indices for each selected draft token (shape: batch_size x num_parents).
            top_scores_index (torch.Tensor): Sorted indices of the chosen top-scoring entries per batch used to pick draft tokens (shape: batch_size x (speculative_num_draft_tokens - 1)).
            draft_tokens (torch.Tensor): Selected draft token IDs per batch (shape: batch_size x (speculative_num_draft_tokens - 1)).
        """
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens_tmp(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def verify(
        self,
        batch: ModelWorkerBatch,
        pre_draft_allocate_lens: torch.Tensor,
    ):
        # Parse args
        """
        Run target verification for a generation batch, move accepted tokens into the target KV cache, perform a draft extension, and return results needed for the next speculative step.
        
        Parameters:
            batch (ModelWorkerBatch): The batch to verify; must contain an EagleVerifyInput in batch.spec_info.
            pre_draft_allocate_lens (torch.Tensor): Allocation lengths used for the draft stage prior to verification; preserved and returned as last_batch_allocate_lens.
        
        Returns:
            GenerationBatchResult: Contains:
                - logits_output: the target model's logits and hidden states from verification,
                - next_token_ids: sampled token ids from verification,
                - can_run_cuda_graph: whether the prepared verify forward batch can use a CUDA graph,
                - next_draft_input: an EagleDraftInput prepopulated with top-k probabilities/indices, hidden states, verified ids, updated sequence lengths, allocate_lens, and a CUDA event marking verify completion,
                - accept_lens: lengths accepted by verification for each sequence,
                - last_batch_allocate_lens: the provided pre_draft_allocate_lens tensor.
        """
        verify_input: EagleVerifyInput = batch.spec_info
        seq_lens_backup = batch.seq_lens
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Sample
        self._detect_nan_if_needed(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output)
        new_seq_lens = seq_lens_backup + accept_length
        verify_done = torch.cuda.Event()

        # Move the accepted tokens to the target KV cache locations
        batch.seq_lens = seq_lens_backup
        self.move_accepted_tokens_to_target_kvcache(
            batch,
            accept_index,
            accept_length,
        )

        verify_done.record()

        all_verified_id = predict[accept_index]
        verified_id = torch.empty_like(accept_length, dtype=torch.int32)
        fill_new_verified_id[(bs,)](
            all_verified_id,
            accept_length,
            verified_id,
            self.speculative_num_draft_tokens,
        )

        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=logits_output.hidden_states,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + accept_length
            - 1
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                predict,
                self.speculative_num_draft_tokens,
                self.draft_model_runner,
            )

        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

        # Run draft extend batch in the main compute stream
        draft_logits_output = self.draft_model_runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Since seq_lens_backup's tensor is allocated in another stream, we
        # need record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        seq_lens_backup.record_stream(torch.cuda.current_stream())

        # Construct the return values
        next_draft_input = EagleDraftInput(
            topk_p=ret_topk_p,
            topk_index=ret_topk_index,
            hidden_states=ret_hidden_states,
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=pre_draft_allocate_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            last_batch_allocate_lens=pre_draft_allocate_lens,
        )

    def forward_draft_extend(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Extend the draft model by one step to populate the draft KV cache and produce the next draft input.
        
        Parameters:
            batch: ModelWorkerBatch whose input_ids are updated in-place to append the provided next tokens per sequence and whose spec_info is set to the returned draft input.
            target_hidden_states: Hidden states produced by the target model, aligned with the batch sequences.
            next_token_ids: 1-D tensor of next-token ids, one entry per sequence in the batch.
        
        Returns:
            EagleDraftInput: A draft input prepared for the next speculative step with updated `topk_p`, `topk_index`, `hidden_states`, `verified_id`, `new_seq_lens`, and `allocate_lens`.
        """
        # Construct input_ids
        pt = 0
        for i, extend_len in enumerate(batch.extend_seq_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], next_token_ids[i].reshape(1))
            )
            pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            allocate_lens=batch.seq_lens,
        )
        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        logits_output, _ = self.draft_model_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move KV-cache entries for accepted speculative draft tokens into the target model's KV cache.
        
        This computes per-request target cache locations and source (accepted) out-cache locations for all speculative draft tokens (using the batch, the worker's speculative_num_draft_tokens and req-to-token pool mapping), then instructs the token-to-KV allocator to move those KV cache entries into the target KV cache.
        
        Parameters:
            batch (ModelWorkerBatch): Batch containing sequence lengths, request pool indices, and out-cache locations used to compute cache addresses.
            accept_index (torch.Tensor): Indices of accepted tokens within the draft out-cache for each accepted position.
            accept_length (torch.Tensor): Number of accepted tokens per batch element; used to compute ranges of tokens to move.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        """
        Check the logits for NaN values when NaN detection is enabled and raise an error if any are found.
        
        When `self.enable_nan_detection` is true, inspects `logits_output.next_token_logits`. If any NaN values are present, an error is logged and a ValueError is raised.
        
        Parameters:
            logits_output (LogitsProcessorOutput): Object containing `next_token_logits` to be checked.
        
        Raises:
            ValueError: If any NaN values are found in `logits_output.next_token_logits`.
        """
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")


def free_spec_dec_tokens_page_size_1(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    req: Req,
    allocate_len: int,
    new_seq_len: int,
):
    # FIXME(lsyin): move this function elsewhere

    # free extra allocated tokens
    """
    Free extra speculative decode token slots allocated for a request by releasing their indices back to the KV pool allocator.
    
    Parameters:
        req_to_token_pool (ReqToTokenPool): Mapping from request pool indices to allocated token index lists.
        token_to_kv_pool_allocator (TokenToKVPoolAllocator): Allocator used to free token indices.
        req (Req): Request whose allocated tokens are to be trimmed; its `req_pool_idx` selects the allocation row.
        allocate_len (int): Total number of token slots that were allocated for this request.
        new_seq_len (int | None): The sequence length actually used. If `None`, computes the start of unused slots as
            `allocate_len - EagleDraftInput.ALLOC_LEN_PER_DECODE`; otherwise uses `new_seq_len` as the start.
    
    Side effects:
        Calls `token_to_kv_pool_allocator.free` with the slice of indices from the selected allocation between
        the computed start and `allocate_len`.
    """
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