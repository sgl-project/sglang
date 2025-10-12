from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import global_server_args_dict
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.build_eagle_tree import TreeMaskMode
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import fast_topk, is_cuda, is_hip, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
    from sgl_kernel.top_k import fast_topk
elif is_hip():
    from sgl_kernel import verify_tree_greedy


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    """
    Map and copy token cache entries for a single-page (page size 1) layout from a request-specific token pool into per-program output cache slots used for draft top-k speculative steps.
    
    Parameters:
        req_pool_indices: Pointer/array of indices mapping each program to its token pool index.
        req_to_token: Base pointer to concatenated token pools; per-request pool entries are offset from this base.
        seq_lens: Array of start offsets (kv start positions) into each request's token pool for the current sequence length.
        out_cache_loc: Output pointer/array where per-program cache locations are written; each program writes `topk * speculative_num_steps` entries.
        pool_len: Compile-time constant giving the length (stride) of each token pool in `req_to_token`.
        topk: Compile-time constant number of top tokens per speculative step.
        speculative_num_steps: Compile-time constant number of speculative steps to copy for each top-k.
    
    Notes:
        - Copies `topk * speculative_num_steps` consecutive entries starting at `req_to_token[req_pool_indices[pid] * pool_len + seq_lens[pid]]` into `out_cache_loc[pid * topk * speculative_num_steps : (pid+1) * topk * speculative_num_steps]`.
    """
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        """
        Prepare a forward batch and allocate per-draft-token cache locations for v2 speculative drafting.
        
        This sets up batch.out_cache_loc to hold cache locations for the top-k draft tokens across num_steps, configures capture mode and draft token positions, and returns a ForwardBatch ready for the draft forward pass along with whether a CUDA graph can be used.
        
        Parameters:
            req_to_token_pool (ReqToTokenPool): Pool mapping requests to token cache entries used to populate out_cache_loc.
            batch (ModelWorkerBatch): Batch to prepare; this function mutates batch.out_cache_loc, batch.capture_hidden_mode, and self.positions.
            cuda_graph_runner (EAGLEDraftCudaGraphRunner): Runner used to determine CUDA-graph viability for the prepared forward batch.
            draft_model_runner (ModelRunner): Model runner used to initialize the ForwardBatch.
            topk (int): Number of draft token candidates per request.
            num_steps (int): Number of speculative steps (draft length) to allocate for each candidate.
        
        Returns:
            tuple[ForwardBatch, bool]: A ForwardBatch initialized for the draft model runner and a boolean indicating whether a CUDA graph can be used.
        """
        bs = len(batch.seq_lens)

        # Assign cache locations
        batch.out_cache_loc = torch.empty(
            (bs * topk * num_steps,),
            dtype=torch.int64,
            device=batch.input_ids.device,
        )
        # FIXME(lsyin): align with the default code path
        assign_draft_cache_locs_page_size_1[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            topk,
            num_steps,
        )

        # Get a forward batch
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
    ):
        """
        Prepare the batch for extending draft tokens into the model KV cache and initialize forward metadata.
        
        This mutates the provided `batch` to reflect the appended draft tokens (updates input_ids, sequence lengths, extension bookkeeping, capture and forward modes) and returns a ForwardBatch configured for running the draft-extension forward pass. The batch is prepared for a full hidden-state capture and uses the DRAFT_EXTEND_V2 forward mode.
        
        Parameters:
            batch (ModelWorkerBatch): Batch object to modify and extend for draft tokens.
            predict (torch.Tensor): Token indices to use as the new input_ids for the extension.
            num_draft_tokens (int): Number of draft tokens appended per sequence in the batch.
            draft_model_runner (Any): Model runner whose attention backend will be used to initialize forward metadata.
        
        Returns:
            ForwardBatch: A forward batch initialized for the draft-extension pass with forward metadata prepared by the draft model runner.
        """
        seq_lens_cpu_backup = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_backup.tolist()
        batch.extend_prefix_lens_cpu = seq_lens_cpu_backup
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        # Assign cache locations
        """
        Prepare the batch and forward metadata for v2 draft verification.
        
        This assigns cache locations for draft tokens into batch.out_cache_loc, sets the batch into TARGET_VERIFY mode with full hidden capture, initializes a ForwardBatch for the target worker's model runner, and prepares or replays any CUDA graph/attention-backend metadata required to run the verification forward pass.
        
        Parameters:
            req_to_token_pool (ReqToTokenPool): Mapping from requests to token pools used to fill cache locations.
            batch (ModelWorkerBatch): The worker batch to prepare; its input_ids and out_cache_loc are updated in-place.
            target_worker (TpModelWorker): The worker whose ModelRunner will execute the verification forward pass.
        
        Returns:
            tuple[ForwardBatch, bool]: A tuple with the initialized ForwardBatch for verification and a boolean indicating whether a CUDA graph can be used (`True` if CUDA graph replay was prepared).
        """
        bs = len(batch.req_pool_indices)
        batch.input_ids = self.draft_token
        device = batch.input_ids.device
        batch.out_cache_loc = torch.empty(
            (bs * self.draft_token_num,),
            dtype=torch.int64,
            device=device,
        )

        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.draft_token_num,
            batch.out_cache_loc,
            req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

        # Get a forward batch
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
    ):
        """
        Verify draft tokens and produce selected predictions with per-request acceptance indices and acceptance lengths.
        
        Performs either deterministic (greedy) verification or probabilistic sampling according to the batch's sampling_info, optionally simulates additional accepted lengths, and increments each accept length by one to include a bonus token.
        
        Parameters:
            batch: ModelWorkerBatch containing speculative decoding state and sampling_info used to drive verification.
            logits_output: LogitsProcessorOutput whose `next_token_logits` are used to derive candidate probabilities or greedy targets.
        
        Returns:
            predict (torch.Tensor): Flattened predicted token ids for all programs and speculative steps with shape (bs * (spec_steps + 1),).
            accept_length (torch.Tensor): Per-request accepted token counts (after adding the bonus token) with shape (bs,).
            accept_index (torch.Tensor): Per-request indices of accepted draft tokens with shape (bs, spec_steps + 1), where invalid entries are -1.
        """
        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict = torch.zeros(
            (bs * (self.spec_steps + 1),), dtype=torch.int32, device=device
        )
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            # This is currently not used
            draft_probs = torch.empty_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=global_server_args_dict[
                    "speculative_accept_threshold_single"
                ],
                threshold_acc=global_server_args_dict[
                    "speculative_accept_threshold_acc"
                ],
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.draft_token_num,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


def build_tree_kernel_efficient_tmp(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    # TODO(lsyin): make it compatible with default code path
    # TODO(lsyin): support cuda graph graph padding for eagle
    """
    Construct tree-related buffers used by the efficient tree-building kernel and return them together with the flattened draft token sequence.
    
    Parameters:
        verified_id (torch.Tensor): Tensor of verified token ids with shape (bs,).
        parent_list (List[torch.Tensor]): List of parent indices per draft token used to build tree adjacency.
        top_scores_index (torch.Tensor): Top-score token indices for each candidate used to populate retrieval buffers.
        draft_tokens (torch.Tensor): Draft token ids with shape (bs, num_draft_tokens).
        seq_lens (torch.Tensor): Sequence lengths for each batch element (without draft tokens).
        seq_lens_sum (int): Sum of values in `seq_lens`.
        topk (int): Number of top candidates per step.
        spec_steps (int): Number of speculative steps used for tree construction.
        num_verify_tokens (int): Number of tokens to verify per batch element.
        tree_mask_mode (TreeMaskMode): Mode controlling layout/packing of the tree attention mask.
        tree_mask_buf (Optional[torch.Tensor]): Optional preallocated buffer to write the tree mask into.
        position_buf (Optional[torch.Tensor]): Optional preallocated buffer to write per-token positions into.
    
    Returns:
        tuple: (tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling, draft_tokens)
            - tree_mask (torch.Tensor): Attention mask for draft tokens in the layout determined by `tree_mask_mode`.
            - positions (torch.Tensor): Flattened position indices indicating where each draft/verify token is placed in the sequence.
            - retrive_index (torch.Tensor): Retrieval index buffer of shape (bs, num_verify_tokens) used to locate nodes.
            - retrive_next_token (torch.Tensor): Retrieval buffer containing next-token indices for each node.
            - retrive_next_sibling (torch.Tensor): Retrieval buffer containing next-sibling indices for each node.
            - draft_tokens (torch.Tensor): Flattened tensor of draft tokens with the verified id prepended.
    """
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )

    sgl_build_tree_kernel_efficient(
        parent_list,
        top_scores_index,
        seq_lens,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
        tree_mask_mode,
    )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


@torch.compile(dynamic=True)
def select_top_k_tokens_tmp(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    # FIXME(lsyin): remove this duplicate code
    """
    Selects top-k token candidates for a decoding step and prepares corresponding input ids, hidden states, scores, and tree metadata for tree-based speculative decoding.
    
    Parameters:
        i (int): Decoding step index since extension; 0 indicates the first step after extension.
        topk_p (torch.Tensor): Per-step top-k probabilities for each batch (shape: (batch, topk) for i==0 or shape compatible for later steps).
        topk_index (torch.Tensor): Token indices corresponding to top-k probabilities (shape: (batch, topk) for i==0; reshaped internally for later steps).
        hidden_states (torch.Tensor): Candidate hidden states aligned with current top-k candidates; repeated or reindexed depending on step.
        scores (torch.Tensor): Current scores for candidate paths (shape: (batch, topk) for i>0).
        topk (int): The top-k value used for selection and reshaping.
    
    Returns:
        tuple:
            input_ids (torch.Tensor): Flattened token ids selected for the next forward pass.
            hidden_states (torch.Tensor): Hidden states aligned with the returned input_ids.
            scores (torch.Tensor): Updated scores for the selected top-k candidates (shape: (batch, topk)).
            tree_info (tuple): Metadata for reconstructing/speculative tree traversal containing:
                - probabilities tensor used for tree expansion (shape varies by step),
                - flattened token-index buffer used for gathering,
                - integer indices used for tree position mapping (shape: (batch, topk)).
    """
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
            0, hidden_states.shape[0], step=topk, device=hidden_states.device
        ).repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    """
    Copy each program's verified token id at the current accepted length into the corresponding slot of `new_verified_id`.
    
    Parameters:
        verified_id (Tensor): Flattened tensor of verified token ids organized as [program0_tokens..., program1_tokens..., ...].
        accept_lens (Tensor): 1-D tensor of per-program accepted lengths; the kernel reads the value for the current program.
        new_verified_id (Tensor): 1-D output tensor where the selected verified id for each program will be stored (written in-place).
        num_draft_tokens (tl.constexpr): Compile-time constant number of draft tokens allocated per program.
    
    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    """
    Place the cache location corresponding to the program's accepted token into the compacted accepted cache array.
    
    For the current program (pid), count how many entries in `accept_index` at positions 0..pid-1 are valid (not -1); if `accept_index[pid]` is valid, load the corresponding entry from `out_cache_loc` and store it into `accepted_out_cache_loc` at the slot indexed by that count.
    
    Parameters:
        accept_index (tensor): 1-D tensor of accepted indices per program where -1 indicates an invalid/missing entry.
        out_cache_loc (tensor): Tensor of cache-location entries to copy from.
        accepted_out_cache_loc (tensor): Destination tensor to receive compacted accepted cache locations.
        size_upper (tl.constexpr): Compile-time upper bound for the length scanned in `accept_index`.
    """
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    """
    Assign cache locations for extended draft tokens by copying token data from per-request token pools into a contiguous output cache.
    
    For each program (batch element) this kernel copies the token entries in the range [start_offset[pid], end_offset[pid]) from the token pool referenced by req_pool_indices[pid] into out_cache_loc. Copies are concatenated across earlier batch elements to compute the destination offset so the result is a packed, per-batch contiguous block of extended token cache locations.
    
    Parameters:
        req_pool_indices (tensor): Per-request indices selecting which token pool to use for each program.
        req_to_token (tensor): Base pointer/array containing all token pools; an index into this plus req_pool_indices selects the pool start.
        start_offset (tensor): Per-program start indices (inclusive) into the selected token pool.
        end_offset (tensor): Per-program end indices (exclusive) into the selected token pool.
        out_cache_loc (tensor): Output buffer that will be filled with the copied token entries; writes are packed per-batch.
        pool_len (tl.constexpr): Length (stride) of a single token pool used to compute pool base pointers.
        bs_upper (tl.constexpr): Upper bound on batch size used for internal indexing and masking.
    """
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE