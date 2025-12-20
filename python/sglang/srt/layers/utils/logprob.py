from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    max_k = max(logits_metadata.top_logprobs_nums)
    ret = all_logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    input_top_logprobs_val, input_top_logprobs_idx = [], []

    pt = 0
    for k, pruned_len in zip(
        logits_metadata.top_logprobs_nums,
        logits_metadata.extend_logprob_pruned_lens_cpu,
    ):
        if pruned_len <= 0:
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        input_top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
        input_top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
        pt += pruned_len

    return input_top_logprobs_val, input_top_logprobs_idx


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
):
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])

    return (
        output_top_logprobs_val,
        output_top_logprobs_idx,
    )


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
