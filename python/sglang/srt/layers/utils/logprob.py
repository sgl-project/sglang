from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyOutput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput


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


def compute_temp_top_p_normalized_logprobs(
    last_logits: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_p: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    compute logprobs for the output token from the given logits.

    Returns:
        torch.Tensor: logprobs from logits
    """
    if top_p is None:
        top_p = logits_metadata.top_p
    if temperature is None:
        temperature = logits_metadata.temperature

    # Scale logits if temperature scaling is enabled
    if logits_metadata.temp_scaled_logprobs:
        last_logits = last_logits / temperature

    # Normalize logprobs if top_p normalization is enabled
    # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
    if logits_metadata.top_p_normalized_logprobs and (top_p != 1.0).any():
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = torch.softmax(last_logits, dim=-1)
        del last_logits
        probs = top_p_normalize_probs_torch(probs, top_p)
        return torch.log(probs)
    else:
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
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
):
    return get_top_logprobs_raw(logprobs, top_logprobs_nums, stage=LogprobStage.DECODE)


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    delay_cpu_copy: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                vals.append(logprobs[i, token_ids].tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for token_ids, pruned_len in zip(
            token_ids_logprobs, extend_logprob_pruned_lens_cpu
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids]
            vals.append(pos_logprobs if delay_cpu_copy else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, delay_cpu_copy=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        delay_cpu_copy=delay_cpu_copy,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs):
    return get_token_ids_logprobs_raw(
        logprobs, token_ids_logprobs, stage=LogprobStage.DECODE
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
    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

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
        if len(val) > 0:
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

    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

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

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_token_ids_logprobs_val[-1].extend(val)
                input_token_ids_logprobs_idx[-1].extend(idx)
            else:
                input_token_ids_logprobs_val.append(val)
                input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def add_output_logprobs_for_spec_v1(
    batch: ScheduleBatch,
    res: Union[EagleVerifyOutput, NgramVerifyInput],
    logits_output: Optional[LogitsProcessorOutput] = None,
):
    # Extract args
    if logits_output is None:
        logits_output = res.logits_output

    if hasattr(res, "accept_length_per_req_cpu"):
        accept_length_per_req_cpu = res.accept_length_per_req_cpu
    else:
        # FIXME: Get a NgramVerifyOutput class and use that instead of this hack.
        accept_length_per_req_cpu = res.accept_length.tolist()

    top_logprobs_nums = batch.top_logprobs_nums
    token_ids_logprobs = batch.token_ids_logprobs
    accepted_indices = res.accepted_indices
    assert len(accepted_indices) == len(logits_output.next_token_logits)

    temperatures = batch.sampling_info.temperatures
    num_draft_tokens = batch.spec_info.draft_token_num
    # acceptance indices are the indices in a "flattened" batch.
    # dividing it to num_draft_tokens will yield the actual batch index.
    temperatures = temperatures[accepted_indices // num_draft_tokens]
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
    else:
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits / temperatures, dim=-1
        )
    batch_next_token_ids = res.verified_id
    num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

    # We should repeat top_logprobs_nums to match num_tokens_per_req.
    top_logprobs_nums_repeat_interleaved = [
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    token_ids_logprobs_repeat_interleaved = [
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    # Extract logprobs
    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)
    if should_top_logprobs:
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            logprobs,
            top_logprobs_nums_repeat_interleaved,
        )

    if should_token_ids_logprobs:
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            logprobs,
            token_ids_logprobs_repeat_interleaved,
        )

    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
        batch_next_token_ids,
    ]

    # Add output logprobs to the request
    pt = 0
    next_token_logprobs = logits_output.next_token_logprobs.tolist()
    verified_ids = batch_next_token_ids.tolist()
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
        for _ in range(num_tokens):
            # TODO: add token_ids_logprobs to each request
            if req.return_logprob:
                req.output_token_logprobs_val.append(next_token_logprobs[pt])
                req.output_token_logprobs_idx.append(verified_ids[pt])
                if req.top_logprobs_num > 0:
                    assert (
                        should_top_logprobs
                    ), "Inconsistent state: should_top_logprobs is False"
                    req.output_top_logprobs_val.append(token_top_logprobs_val[pt])
                    req.output_top_logprobs_idx.append(token_top_logprobs_idx[pt])
            pt += 1
