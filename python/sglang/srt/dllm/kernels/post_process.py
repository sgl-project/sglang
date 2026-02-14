"""Fused Triton kernels for dLLM post-processing."""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _dllm_post_process_kernel(
    logits_ptr,
    input_ids_ptr,
    transfer_out_ptr,
    confidence_out_ptr,
    argmax_out_ptr,
    num_transfers_ptr,
    mask_id: tl.constexpr,
    threshold: tl.constexpr,
    block_size,
    vocab_size,
    logits_stride,
    BLOCK_V: tl.constexpr,
):
    """Fused softmax + argmax + threshold transfer for each masked position.
    Unmasked positions are skipped. Uses online softmax for numerical stability.
    """
    row_idx = tl.program_id(0)

    if row_idx >= block_size:
        return

    row_start = row_idx * logits_stride

    # Load current input_id
    input_id = tl.load(input_ids_ptr + row_idx)
    is_masked = input_id == mask_id

    # Skip non-masked positions
    if not is_masked:
        tl.store(transfer_out_ptr + row_idx, 0)
        tl.store(confidence_out_ptr + row_idx, -float("inf"))
        tl.store(argmax_out_ptr + row_idx, input_id)
        return

    # Online softmax + argmax in single pass
    max_val = -float("inf")
    exp_sum = 0.0
    argmax_idx = 0

    for v_start in range(0, vocab_size, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offsets < vocab_size

        logits = tl.load(
            logits_ptr + row_start + v_offsets, mask=v_mask, other=-float("inf")
        )
        logits = logits.to(tl.float32)

        tile_max = tl.max(logits)

        if tile_max > max_val:
            exp_sum = exp_sum * tl.exp(max_val - tile_max)
            max_val = tile_max

            is_max = logits == tile_max
            tile_indices = tl.where(is_max, v_offsets, vocab_size)
            argmax_idx = tl.min(tile_indices)

        exp_vals = tl.exp(logits - max_val)
        exp_sum += tl.sum(tl.where(v_mask, exp_vals, 0.0))

    prob = 1.0 / exp_sum
    transfer = prob > threshold

    tl.store(argmax_out_ptr + row_idx, argmax_idx)
    tl.store(confidence_out_ptr + row_idx, prob)
    tl.store(transfer_out_ptr + row_idx, transfer.to(tl.int8))

    if transfer:
        tl.store(input_ids_ptr + row_idx, argmax_idx)
        tl.atomic_add(num_transfers_ptr, 1)


@triton.jit
def _dllm_fallback_kernel(
    input_ids_ptr,
    confidence_ptr,
    argmax_out_ptr,
    transfer_out_ptr,
    num_transfers_ptr,
    block_size,
):
    """If no transfers occurred, force-accept the highest-confidence position.
    Single-thread kernel (grid=1), exits immediately when num_transfers > 0.
    """
    num_transfers = tl.load(num_transfers_ptr)
    if num_transfers > 0:
        return

    best_idx = 0
    best_conf = -float("inf")
    for i in range(block_size):
        c = tl.load(confidence_ptr + i)
        if c > best_conf:
            best_conf = c
            best_idx = i

    argmax_token = tl.load(argmax_out_ptr + best_idx)
    tl.store(input_ids_ptr + best_idx, argmax_token)
    tl.store(transfer_out_ptr + best_idx, tl.cast(1, tl.int8))
    tl.store(num_transfers_ptr, 1)


_dllm_post_process_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_V": 8192}, num_warps=8, num_stages=4),
    ],
    key=["vocab_size"],
)(_dllm_post_process_kernel)


def dllm_post_process_fused(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_id: int,
    threshold: float,
    autotune: bool = True,
) -> None:
    """Fused dLLM post-processing. Modifies input_ids in-place: masked positions
    whose argmax probability exceeds *threshold* are replaced with the predicted
    token. A fallback kernel guarantees at least one transfer per call.
    """
    block_size, vocab_size = logits.shape
    device = logits.device

    transfer_index = torch.empty(block_size, dtype=torch.int8, device=device)
    confidence = torch.empty(block_size, dtype=torch.float32, device=device)
    argmax_tokens = torch.empty(block_size, dtype=torch.int64, device=device)
    num_transfers_dev = torch.zeros(1, dtype=torch.int32, device=device)

    grid = (block_size,)

    if autotune:
        _dllm_post_process_kernel_autotuned[grid](
            logits,
            input_ids,
            transfer_index,
            confidence,
            argmax_tokens,
            num_transfers_dev,
            mask_id,
            threshold,
            block_size,
            vocab_size,
            logits.stride(0),
        )
    else:
        if vocab_size <= 2048:
            BLOCK_V = triton.next_power_of_2(vocab_size)
        elif vocab_size <= 8192:
            BLOCK_V = 2048
        else:
            BLOCK_V = 4096

        _dllm_post_process_kernel[grid](
            logits,
            input_ids,
            transfer_index,
            confidence,
            argmax_tokens,
            num_transfers_dev,
            mask_id,
            threshold,
            block_size,
            vocab_size,
            logits.stride(0),
            BLOCK_V=BLOCK_V,
            num_warps=4,
        )

    # Fallback: exits immediately if num_transfers > 0
    _dllm_fallback_kernel[(1,)](
        input_ids,
        confidence,
        argmax_tokens,
        transfer_index,
        num_transfers_dev,
        block_size,
    )


dllm_post_process = dllm_post_process_fused


def dllm_post_process_pytorch(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_id: int,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Reference PyTorch implementation for correctness verification."""
    import torch.nn.functional as F

    block_mask_index = input_ids == mask_id

    x = torch.argmax(logits, dim=-1)

    probs = F.softmax(logits.float(), dim=-1)
    p = torch.squeeze(
        torch.gather(probs, dim=-1, index=torch.unsqueeze(x, -1)),
        -1,
    )

    x = torch.where(block_mask_index, x, input_ids)
    confidence = torch.where(
        block_mask_index, p, torch.tensor(-float("inf"), device=logits.device)
    )

    transfer_index = confidence > threshold

    num_transfers = transfer_index.sum().item()
    if num_transfers == 0:
        _, select_index = torch.topk(confidence, k=1)
        transfer_index[select_index] = True
        num_transfers = 1

    input_ids = input_ids.clone()
    input_ids[transfer_index] = x[transfer_index]

    return input_ids, transfer_index, confidence, num_transfers
