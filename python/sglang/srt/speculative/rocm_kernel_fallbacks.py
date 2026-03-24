"""PyTorch fallback implementations of sgl_kernel sampling ops for ROCm/HIP.

These replace the CUDA C++ kernels that are not compiled for HIP:
- top_k_renorm_prob: top-k probability renormalization
- top_p_renorm_prob: top-p (nucleus) probability renormalization
- tree_speculative_sampling_target_only: tree-based rejection sampling for Eagle3

Algorithm for tree_speculative_sampling_target_only translated from:
  sgl-kernel/csrc/speculative/speculative_sampling.cuh
"""

from typing import Union

import torch


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Renormalize probabilities by top-k thresholding.

    Keeps the top-k probabilities per row, zeros the rest, and renormalizes.
    Handles per-row k values when top_k is a tensor.
    """
    if isinstance(top_k, (int, float)):
        top_k = torch.full(
            (probs.shape[0],), int(top_k), device=probs.device, dtype=torch.int64
        )

    top_k = top_k.to(dtype=torch.int64, device=probs.device)
    batch_size, vocab_size = probs.shape

    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)

    k_indices = (top_k - 1).clamp(min=0, max=vocab_size - 1).unsqueeze(1)
    thresholds = sorted_probs.gather(1, k_indices)

    renorm_probs = probs.clone()
    renorm_probs[renorm_probs < thresholds] = 0.0
    row_sums = renorm_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    renorm_probs.div_(row_sums)

    return renorm_probs


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Renormalize probabilities by top-p (nucleus) thresholding.

    Keeps the smallest set of tokens whose cumulative probability >= top_p,
    zeros the rest, and renormalizes. Handles per-row p values.
    """
    if isinstance(top_p, (int, float)):
        top_p = torch.full(
            (probs.shape[0],), float(top_p), device=probs.device, dtype=torch.float32
        )

    top_p = top_p.to(device=probs.device, dtype=torch.float32)

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = (cumsum_probs - sorted_probs) >= top_p.unsqueeze(1)
    sorted_probs[sorted_mask] = 0.0

    renorm_probs = torch.zeros_like(probs)
    renorm_probs.scatter_(1, sorted_indices, sorted_probs)

    row_sums = renorm_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    renorm_probs.div_(row_sums)

    return renorm_probs


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Tree-based speculative sampling (target-only) for Eagle3.

    Traverses the draft token tree per batch element, accepting or rejecting
    tokens via probability threshold comparison against uniform random coins.
    After traversal, samples a bonus token from max(target_probs - draft_probs, 0).

    Modifies predicts, accept_index, accept_token_num, draft_probs in place.
    """
    batch_size = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_spec_tokens = accept_index.shape[1]
    d = target_probs.shape[2]
    capped_threshold_acc = max(threshold_acc, 1e-9)

    # Transfer small index tensors to CPU for serial tree traversal.
    # These are (bs, num_draft_tokens) ~ a few KB total.
    candidates_cpu = candidates.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()
    uniform_samples_cpu = uniform_samples.cpu()

    for bx in range(batch_size):
        prob_acc = 0.0
        cur_prob_row = 0
        coin = uniform_samples_cpu[bx, 0].item()
        last_accepted_retrive_idx = retrive_index_cpu[bx, 0].item()
        accept_index[bx, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_idx = 0

        for j in range(1, num_spec_tokens):
            cur_idx = retrive_next_token_cpu[bx, cur_idx].item()

            while cur_idx != -1:
                draft_index = retrive_index_cpu[bx, cur_idx].item()
                draft_token_id = candidates_cpu[bx, cur_idx].item()
                target_prob_single = target_probs[
                    bx, cur_prob_row, draft_token_id
                ].item()
                prob_acc += target_prob_single

                if (
                    coin <= prob_acc / capped_threshold_acc
                    or target_prob_single >= threshold_single
                ):
                    prob_acc = 0.0
                    cur_prob_row = cur_idx
                    coin = uniform_samples_cpu[bx, cur_idx].item()
                    predicts[last_accepted_retrive_idx] = draft_token_id
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    draft_probs[bx, cur_prob_row, draft_token_id] = target_probs[
                        bx, cur_prob_row, draft_token_id
                    ]
                    cur_idx = retrive_next_sibling_cpu[bx, cur_idx].item()

            if cur_idx == -1:
                break

        accept_token_num[bx] = num_accepted

        # Sample bonus token from max(target_probs - draft_probs, 0).
        target_row = target_probs[bx, cur_prob_row]
        if num_accepted != num_spec_tokens - 1:
            draft_row = draft_probs[bx, cur_prob_row]
        else:
            draft_row = torch.zeros_like(target_row)

        relu_diff = torch.clamp(target_row - draft_row, min=0.0)
        sum_relu = relu_diff.sum()

        if sum_relu.item() > 0:
            coin_final = uniform_samples_for_final_sampling[bx]
            u = coin_final * sum_relu
            cumsum = torch.cumsum(relu_diff, dim=0)
            sampled_id = torch.searchsorted(cumsum, u, right=True).clamp(max=d - 1)
            predicts[last_accepted_retrive_idx] = sampled_id.to(torch.int32)
        else:
            predicts[last_accepted_retrive_idx] = d - 1
