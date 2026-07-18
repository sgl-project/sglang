from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class InputLogprobsResult:
    input_token_logprobs: torch.Tensor
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    return get_top_logprobs_raw(
        all_logprobs,
        logits_metadata.top_logprobs_nums,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            if token_ids is None:
                # The sequence's rows still occupy logprobs; step over them.
                vals.append([])
                idxs.append([])
                pt += pruned_len
                continue
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                logprobs.device, non_blocking=True
            )
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
            vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, no_copy_to_cpu=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_k_nums: List[int],
    pruned_lens: List[int],
    input_top_logprobs_val: List,
    input_top_logprobs_idx: List,
    split_pruned_len: int,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_top_logprobs_val: List to store top-k logprob values
        input_top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # Empty chunks still walk the slice to emit placeholder entries.
    max_k = max(logits_metadata.top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        # A continuation of a sequence split by the previous chunk extends its
        # entry; every other sequence owns a fresh entry.
        if split_pruned_len > 0:
            input_top_logprobs_val[-1].extend(val)
            input_top_logprobs_idx[-1].extend(idx)
        else:
            input_top_logprobs_val.append(val)
            input_top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    input_token_ids_logprobs_val: List,
    input_token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing token IDs and pruned length info
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_token_ids_logprobs_val: List to store token logprob values
        input_token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # Empty chunks still walk the slice to emit placeholder entries.
    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                val.append(logprobs[pt + j, token_ids].tolist())
                idx.append(token_ids)

        # A continuation of a sequence split by the previous chunk extends its
        # entry; every other sequence owns a fresh entry (empty for
        # token_ids=None sequences, matching the non-chunked reference).
        if split_pruned_len > 0:
            input_token_ids_logprobs_val[-1].extend(val)
            input_token_ids_logprobs_idx[-1].extend(idx)
        else:
            input_token_ids_logprobs_val.append(val)
            input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def compute_spec_v2_logprobs(
    batch,
    logits_output,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    speculative_num_steps: int,
):
    """Compute logprobs for accepted tokens after spec v2 verify sampling.

    Gathers logits at accepted positions, applies log_softmax (temperature-scaled
    if not greedy), and populates logits_output.next_token_logprobs (plus optional
    top-k / token-ids logprobs) so they flow through copy_to_cpu().
    """
    bs = len(batch.seq_lens)
    max_accept = speculative_num_steps + 1
    device = predict.device

    flat_accept_idx = accept_index.long().reshape(-1)
    gathered_logits = logits_output.next_token_logits[flat_accept_idx]

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    accepted_token_ids = predict[flat_accept_idx]
    token_logprobs = gathered_logprobs[
        torch.arange(bs * max_accept, device=device),
        accepted_token_ids.long(),
    ]
    logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )


class InputLogprobProcessor:
    """Input (prefill) logprob processing: single-pass or chunked.

    Logits are computed through the injected ``get_logits_fn(hidden_states,
    lm_head, logits_metadata)`` callable, so this class stays decoupled from
    the lm_head / TP-gather machinery in LogitsProcessor.
    """

    def __init__(self):
        # enable chunked logprobs processing
        self.enable_logprobs_chunk = envs.SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK.get()
        # chunk size for logprobs processing
        self.logprobs_chunk_size = envs.SGLANG_LOGITS_PROCESSER_CHUNK_SIZE.get()

    def forward(
        self,
        pruned_states: torch.Tensor,
        sample_indices: Optional[torch.Tensor],
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        skip_chunking_for_dp_attn: bool = False,
    ) -> Tuple[InputLogprobsResult, torch.Tensor]:
        # Start to process input logprobs
        # Determine whether to use chunked or non-chunked logits processing.
        # Skip chunking if:
        # 1. Chunking is disabled
        # 2. Total count is below chunk size threshold
        # 3. DP attention all-gather is enabled (can use "enable_dp_lm_head" to enable chunking)
        should_skip_chunking = (
            not self.enable_logprobs_chunk
            or pruned_states.shape[0] <= self.logprobs_chunk_size
            or skip_chunking_for_dp_attn
        )

        if should_skip_chunking:
            # Compute logits for both input and sampled tokens.
            logits = get_logits_fn(pruned_states, lm_head, logits_metadata)
            sampled_logits = (
                logits[sample_indices] if sample_indices is not None else logits
            )
            input_logits = logits[input_logprob_indices]
            del logits

            logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)
        else:
            logprobs_result, sampled_logits = self.process_input_logprobs_by_chunk(
                pruned_states,
                sample_indices,
                input_logprob_indices,
                token_to_seq_idx,
                lm_head,
                get_logits_fn,
                logits_metadata,
            )

        return logprobs_result, sampled_logits

    def process_input_logprobs(self, input_logits, logits_metadata: LogitsMetadata):
        input_logprobs = torch.nn.functional.log_softmax(input_logits, dim=-1)

        # Get the logprob of top-k tokens
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_top_logprobs_val = input_top_logprobs_idx = None

        # Get the logprob of given token id
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

        input_token_logprobs = input_logprobs[
            torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
            logits_metadata.extend_input_logprob_token_ids_gpu,
        ]

        return InputLogprobsResult(
            input_token_logprobs=input_token_logprobs,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
        )

    def process_input_logprobs_by_chunk(
        self,
        pruned_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head: VocabParallelEmbedding,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[InputLogprobsResult, torch.Tensor]:
        """
        compute logprobs for the output token from the hidden states.
        To avoid using too much memory, we split pruned_states into chunks of
        rows to compute input_logprobs separately, then concatenate the results.

        Returns:
            InputLogprobsResult: logprobs result
            torch.Tensor: sampled logits
        """

        # The peak memory usage is proportional to the chunk size.
        chunk_size = self.logprobs_chunk_size
        total_size = pruned_states.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        input_token_logprobs = []
        if logits_metadata.extend_return_top_logprob:
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
        else:
            input_top_logprobs_val = None
            input_top_logprobs_idx = None
        if logits_metadata.extend_token_ids_logprob:
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
        else:
            input_token_ids_logprobs_val = None
            input_token_ids_logprobs_idx = None

        # If a single sequence is split into multiple chunks, we need to keep track
        # of the pruned length of the sequences in the previous chunks.
        split_len_topk = 0
        split_len_token_ids = 0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)

            # Notify lm_head LoRA about the current chunk so it can swap
            # to the precomputed per-chunk batch_info.  This is a no-op
            # for non-LoRA lm_head modules.
            if hasattr(lm_head, "set_lm_head_pass"):
                lm_head.set_lm_head_pass(i)

            # Get indices for this chunk
            chunk_mask = (input_logprob_indices >= start_idx) & (
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]
            chunk_indices = global_indices - start_idx
            # Get the positions in the original array where chunk_mask is True
            # This is needed to correctly index into extend_input_logprob_token_ids_gpu
            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]

            # Get the logits for this chunk. Each chunk must own its output:
            # writing through the shared graph logits buffer would alias
            # chunks whose shape happens to match the buffer.
            chunk_states = pruned_states[start_idx:end_idx]
            chunk_logits = get_logits_fn(
                chunk_states, lm_head, logits_metadata, use_logits_buffer=False
            )

            # Initialize sampled_logits on first chunk
            if i == 0:
                sampled_logits = torch.empty(
                    (sample_indices.shape[0], chunk_logits.shape[1]),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )

            # Handle sampled logits for the chunk if needed
            # This must be done before the continue statement to ensure all sampled_logits are filled
            chunk_sample_mask = (sample_indices >= start_idx) & (
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx
                sampled_logits[chunk_sample_mask] = chunk_logits[chunk_sample_indices]

            # Zero-logprob-row chunks still need the per-sequence bookkeeping below.
            # Compute the logprobs of the chunk
            chunk_input_logprobs = chunk_logits[chunk_indices]
            chunk_input_logprobs = torch.nn.functional.log_softmax(
                chunk_input_logprobs, dim=-1
            )

            # End at the last row inside the chunk; token_to_seq_idx[end_idx]
            # belongs to the next chunk and would emit its sequence twice.
            chunk_slice = slice(
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx - 1] + 1
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(
                    chunk_input_logprobs,
                    logits_metadata,
                    top_k_nums,
                    pruned_lens,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    split_len_topk,
                )

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(
                    chunk_input_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    split_len_token_ids,
                )

            # Get the logprob of the requested token ids
            chunk_input_token_logprobs = chunk_input_logprobs[
                torch.arange(
                    chunk_input_logprobs.shape[0], device=chunk_input_logprobs.device
                ),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],
            ]
            input_token_logprobs.append(chunk_input_token_logprobs)

        # Restore the full-pruned lm_head batch_info after chunk iteration.
        if hasattr(lm_head, "reset_lm_head_pass"):
            assert hasattr(
                lm_head, "set_lm_head_pass"
            ), "lm_head must have set_lm_head_pass method and reset_lm_head_pass method at the same time"
            lm_head.reset_lm_head_pass()

        # Concatenate the results
        input_token_logprobs = torch.cat(input_token_logprobs, dim=0)

        return (
            InputLogprobsResult(
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            ),
            sampled_logits,
        )
