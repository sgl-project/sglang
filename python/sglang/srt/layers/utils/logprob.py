from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    stage: str = "decode",
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    values = values.tolist()
    indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == "decode":
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
        logits_metadata.extend_logprob_pruned_lens_cpu,
        stage="prefill",
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
):
    return get_top_logprobs_raw(logprobs, top_logprobs_nums, stage="decode")


def get_token_ids_logprobs_prefill(
    all_logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    delay_cpu_copy: bool = False,
):
    input_token_ids_logprobs_val, input_token_ids_logprobs_idx = [], []
    pt = 0
    for token_ids, pruned_len in zip(
        logits_metadata.token_ids_logprobs,
        logits_metadata.extend_logprob_pruned_lens_cpu,
    ):
        if pruned_len <= 0:
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        position_logprobs = all_logprobs[
            pt : pt + pruned_len, token_ids
        ]  # Shape: [pruned_len, num_tokens]

        if delay_cpu_copy:
            # Keep as tensor to delay GPU-to-CPU transfer
            input_token_ids_logprobs_val.append(position_logprobs)
        else:
            # Convert to list immediately (default behavior)
            input_token_ids_logprobs_val.append(position_logprobs.tolist())

        input_token_ids_logprobs_idx.append([token_ids for _ in range(pruned_len)])
        pt += pruned_len

    return input_token_ids_logprobs_val, input_token_ids_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return (
        output_token_ids_logprobs_val,
        output_token_ids_logprobs_idx,
    )
