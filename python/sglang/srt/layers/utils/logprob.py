from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata


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


def process_input_logprobs(input_logits, logits_metadata: LogitsMetadata):
    input_logprobs = compute_temp_top_p_normalized_logprobs(
        input_logits, logits_metadata
    )

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


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[Optional[List[int]]],
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    stage: str = "decode",
    delay_cpu_copy: bool = False,
):
    vals, idxs = [], []
    if stage == "decode":
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


def get_token_ids_logprobs_prefill(all_logprobs, logits_metadata, delay_cpu_copy=False):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        logits_metadata.extend_logprob_pruned_lens_cpu,
        stage="prefill",
        delay_cpu_copy=delay_cpu_copy,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs):
    return get_token_ids_logprobs_raw(logprobs, token_ids_logprobs, stage="decode")
