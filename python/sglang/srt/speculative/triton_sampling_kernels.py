"""Triton kernel implementations of sampling ops for ROCm/HIP.

Replaces CUDA C++ sgl_kernel ops unavailable on HIP:
- top_k_renorm_prob: top-k probability renormalization
- top_p_renorm_prob: top-p (nucleus) probability renormalization
- tree_speculative_sampling_target_only: tree-based speculative sampling

Semantics are identical to the sgl_kernel CUDA implementations.
Algorithm for tree_speculative_sampling_target_only ported from
sgl-kernel/csrc/speculative/speculative_sampling.cuh.
"""

from typing import Union

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# top_k_renorm_prob
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["vocab_size"],
)
@triton.jit
def _top_k_mask_kernel(
    probs_ptr,
    sorted_probs_ptr,
    top_k_ptr,
    out_ptr,
    row_sums_ptr,
    vocab_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.load(top_k_ptr + row).to(tl.int64)
    k = tl.minimum(tl.maximum(k, 1), vocab_size)
    threshold = tl.load(sorted_probs_ptr + row * stride + k - 1)

    row_sum = tl.zeros([], dtype=tl.float32)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)

    for i in range(num_blocks):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < vocab_size
        val = tl.load(probs_ptr + row * stride + offset, mask=mask, other=0.0)
        masked = tl.where(val >= threshold, val, 0.0)
        tl.store(out_ptr + row * stride + offset, masked, mask=mask)
        row_sum += tl.sum(masked)

    tl.store(row_sums_ptr + row, tl.maximum(row_sum, 1e-8))


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    if isinstance(top_k, (int, float)):
        top_k = torch.full(
            (probs.shape[0],), int(top_k), device=probs.device, dtype=torch.int64
        )
    top_k = top_k.to(dtype=torch.int64, device=probs.device)

    batch_size, vocab_size = probs.shape
    probs = probs.float()

    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    out = torch.empty_like(probs)
    row_sums = torch.empty(batch_size, device=probs.device, dtype=torch.float32)

    grid = (batch_size,)
    _top_k_mask_kernel[grid](
        probs,
        sorted_probs,
        top_k,
        out,
        row_sums,
        vocab_size,
        probs.stride(0),
    )
    out.div_(row_sums.unsqueeze(1))
    return out


# ---------------------------------------------------------------------------
# top_p_renorm_prob
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["vocab_size"],
)
@triton.jit
def _top_p_scatter_kernel(
    sorted_probs_ptr,
    sorted_indices_ptr,
    top_p_ptr,
    out_ptr,
    row_sums_ptr,
    vocab_size,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    p = tl.load(top_p_ptr + row)

    cumsum = tl.zeros([], dtype=tl.float32)
    row_sum = tl.zeros([], dtype=tl.float32)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    done = tl.zeros([], dtype=tl.int32)

    for i in range(num_blocks):
        if done == 0:
            offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offset < vocab_size
            sorted_val = tl.load(
                sorted_probs_ptr + row * stride + offset, mask=mask, other=0.0
            )

            prev_cumsum = cumsum
            block_cumsum = tl.cumsum(sorted_val, axis=0)
            global_cumsum_before = block_cumsum - sorted_val + prev_cumsum
            keep = global_cumsum_before < p

            filtered = tl.where(keep & mask, sorted_val, 0.0)
            orig_idx = tl.load(
                sorted_indices_ptr + row * stride + offset, mask=mask, other=0
            ).to(tl.int64)
            tl.store(out_ptr + row * stride + orig_idx, filtered, mask=mask & keep)

            row_sum += tl.sum(filtered)
            cumsum = prev_cumsum + tl.sum(sorted_val)

            if cumsum >= p:
                done = 1

    tl.store(row_sums_ptr + row, tl.maximum(row_sum, 1e-8))


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    if isinstance(top_p, (int, float)):
        top_p = torch.full(
            (probs.shape[0],), float(top_p), device=probs.device, dtype=torch.float32
        )
    top_p = top_p.to(device=probs.device, dtype=torch.float32)

    batch_size, vocab_size = probs.shape
    probs = probs.float()

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    out = torch.zeros_like(probs)
    row_sums = torch.empty(batch_size, device=probs.device, dtype=torch.float32)

    grid = (batch_size,)
    _top_p_scatter_kernel[grid](
        sorted_probs,
        sorted_indices,
        top_p,
        out,
        row_sums,
        vocab_size,
        probs.stride(0),
    )
    out.div_(row_sums.unsqueeze(1))
    return out


# ---------------------------------------------------------------------------
# tree_speculative_sampling_target_only
# ---------------------------------------------------------------------------


@triton.jit
def _tree_spec_sampling_kernel(
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    candidates_ptr,
    retrive_index_ptr,
    retrive_next_token_ptr,
    retrive_next_sibling_ptr,
    uniform_samples_ptr,
    uniform_samples_final_ptr,
    target_probs_ptr,
    draft_probs_ptr,
    num_speculative_tokens,
    num_draft_tokens,
    vocab_size,
    threshold_single,
    threshold_acc,
    BLOCK_D: tl.constexpr,
):
    bx = tl.program_id(0)
    base = (bx * num_draft_tokens).to(tl.int64)

    # --- Phase 1: serial tree traversal (scalar ops) ---
    capped_threshold_acc = tl.maximum(threshold_acc, 1e-9)
    prob_acc = tl.zeros([], dtype=tl.float32)
    cur_prob_offset = (base * vocab_size).to(tl.int64)
    coin = tl.load(uniform_samples_ptr + base)
    last_accepted_idx = tl.load(retrive_index_ptr + base).to(tl.int32)
    tl.store(accept_index_ptr + bx * num_speculative_tokens, last_accepted_idx)
    num_accepted = tl.zeros([], dtype=tl.int32)
    cur_index = tl.zeros([], dtype=tl.int64)
    tree_done = tl.zeros([], dtype=tl.int32)

    for _j in range(num_speculative_tokens - 1):
        if tree_done == 0:
            cur_index = tl.load(retrive_next_token_ptr + base + cur_index)
            step_accepted = tl.zeros([], dtype=tl.int32)

            for _s in range(num_draft_tokens):
                if cur_index != -1 and step_accepted == 0:
                    draft_idx = tl.load(retrive_index_ptr + base + cur_index).to(
                        tl.int32
                    )
                    draft_token = tl.load(candidates_ptr + base + cur_index)
                    target_prob = tl.load(
                        target_probs_ptr + cur_prob_offset + draft_token
                    )
                    prob_acc += target_prob

                    accept = (coin <= prob_acc / capped_threshold_acc) | (
                        target_prob >= threshold_single
                    )

                    if accept:
                        prob_acc = 0.0
                        cur_prob_offset = ((base + cur_index) * vocab_size).to(tl.int64)
                        coin = tl.load(uniform_samples_ptr + base + cur_index)
                        tl.store(
                            predicts_ptr + last_accepted_idx, draft_token.to(tl.int32)
                        )
                        num_accepted += 1
                        tl.store(
                            accept_index_ptr
                            + bx * num_speculative_tokens
                            + num_accepted,
                            draft_idx,
                        )
                        last_accepted_idx = draft_idx
                        step_accepted = 1
                    else:
                        tl.store(
                            draft_probs_ptr + cur_prob_offset + draft_token,
                            target_prob,
                        )
                        cur_index = tl.load(retrive_next_sibling_ptr + base + cur_index)

            if step_accepted == 0:
                tree_done = 1

    tl.store(accept_token_num_ptr + bx, num_accepted)

    # --- Phase 2: bonus token sampling from relu(target - draft) ---
    coin_final = tl.load(uniform_samples_final_ptr + bx)

    sum_relu = tl.zeros([], dtype=tl.float32)
    num_vocab_blocks = tl.cdiv(vocab_size, BLOCK_D)

    for i in range(num_vocab_blocks):
        offset = i * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offset < vocab_size
        q = tl.load(target_probs_ptr + cur_prob_offset + offset, mask=mask, other=0.0)
        p = tl.zeros([BLOCK_D], dtype=tl.float32)
        if num_accepted < num_speculative_tokens - 1:
            p = tl.load(
                draft_probs_ptr + cur_prob_offset + offset, mask=mask, other=0.0
            )
        relu_diff = tl.maximum(q - p, 0.0)
        sum_relu += tl.sum(relu_diff)

    u = coin_final * sum_relu

    cumulative = tl.zeros([], dtype=tl.float32)
    sampled_id = vocab_size - 1
    found = tl.zeros([], dtype=tl.int32)

    for i in range(num_vocab_blocks):
        if found == 0:
            offset = i * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = offset < vocab_size
            q = tl.load(
                target_probs_ptr + cur_prob_offset + offset, mask=mask, other=0.0
            )
            p = tl.zeros([BLOCK_D], dtype=tl.float32)
            if num_accepted < num_speculative_tokens - 1:
                p = tl.load(
                    draft_probs_ptr + cur_prob_offset + offset, mask=mask, other=0.0
                )
            relu_diff = tl.maximum(q - p, 0.0)
            block_cumsum = tl.cumsum(relu_diff, axis=0) + cumulative

            exceeds = block_cumsum > u
            if tl.sum(exceeds.to(tl.int32)) > 0:
                candidate_ids = tl.where(exceeds, offset, vocab_size)
                sampled_id = tl.min(candidate_ids)
                found = 1

            cumulative += tl.sum(relu_diff)

    tl.store(predicts_ptr + last_accepted_idx, sampled_id.to(tl.int32))


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
    batch_size = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_speculative_tokens = accept_index.shape[1]
    vocab_size = target_probs.shape[2]

    grid = (batch_size,)
    _tree_spec_sampling_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        num_speculative_tokens,
        num_draft_tokens,
        vocab_size,
        threshold_single,
        threshold_acc,
        BLOCK_D=4096,
        num_warps=16,
    )
