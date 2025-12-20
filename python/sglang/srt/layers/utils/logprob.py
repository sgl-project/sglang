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
