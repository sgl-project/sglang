import torch
import triton
import triton.language as tl

QUEST_FINALIZE_MAX_WIDTH = 1024


@triton.jit
def _quest_finalize_selected_pages_kernel(
    topk_scores_ptr,
    topk_indices_ptr,
    k_per_req_ptr,
    recent_indices_ptr,
    recent_valid_ptr,
    output_indices_ptr,
    output_lengths_ptr,
    TOPK_WIDTH: tl.constexpr,
    RECENT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    topk_mask = offsets < TOPK_WIDTH
    sentinel = 0x7FFFFFFF
    topk_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    topk_valid = offsets < 0
    if TOPK_WIDTH > 0:
        safe_topk_offsets = tl.where(topk_mask, offsets, 0)
        topk_scores = tl.load(
            topk_scores_ptr + batch_idx * TOPK_WIDTH + safe_topk_offsets,
            mask=topk_mask,
            other=-float("inf"),
        ).to(tl.float32)
        topk_indices = tl.load(
            topk_indices_ptr + batch_idx * TOPK_WIDTH + safe_topk_offsets,
            mask=topk_mask,
            other=-1,
        ).to(tl.int64)
        k_per_req = tl.load(k_per_req_ptr + batch_idx)
        topk_valid = (
            topk_mask
            & (offsets < k_per_req)
            & (topk_scores > -float("inf"))
            & (topk_scores < float("inf"))
            & (topk_indices >= 0)
            & (topk_indices < sentinel)
        )

    recent_offsets = offsets - TOPK_WIDTH
    recent_mask = (recent_offsets >= 0) & (recent_offsets < RECENT_WIDTH)
    recent_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    recent_valid = offsets < 0
    if RECENT_WIDTH > 0:
        safe_recent_offsets = tl.where(recent_mask, recent_offsets, 0)
        recent_indices = tl.load(
            recent_indices_ptr + batch_idx * RECENT_WIDTH + safe_recent_offsets,
            mask=recent_mask,
            other=-1,
        ).to(tl.int64)
        recent_valid = tl.load(
            recent_valid_ptr + batch_idx * RECENT_WIDTH + safe_recent_offsets,
            mask=recent_mask,
            other=0,
        ).to(tl.int1)
        recent_valid &= (recent_indices >= 0) & (recent_indices < sentinel)

    selected = tl.where(
        topk_valid,
        topk_indices.to(tl.int32),
        tl.where(recent_mask & recent_valid, recent_indices.to(tl.int32), sentinel),
    )
    sorted_indices = tl.sort(selected, descending=False)
    valid = sorted_indices != sentinel

    output_mask = offsets < OUTPUT_WIDTH
    safe_output_offsets = tl.where(output_mask, offsets, 0)
    output_offsets = batch_idx * OUTPUT_WIDTH + safe_output_offsets
    tl.store(
        output_indices_ptr + output_offsets,
        tl.where(valid, sorted_indices, -1),
        mask=output_mask,
    )
    tl.store(output_lengths_ptr + batch_idx, tl.sum(valid.to(tl.int32), axis=0))


def quest_finalize_selected_pages(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    k_per_req: torch.Tensor,
    recent_indices: torch.Tensor,
    recent_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finalize fixed-width Quest selections in one CUDA kernel."""
    if not topk_scores.is_cuda or torch.version.hip is not None:
        raise ValueError("Quest Triton finalize requires NVIDIA CUDA tensors")
    if topk_scores.dim() != 2 or topk_indices.shape != topk_scores.shape:
        raise ValueError("Quest top-k scores and indices must have matching 2D shapes")
    if not topk_scores.dtype.is_floating_point:
        raise ValueError("Quest top-k scores must use a floating-point dtype")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest top-k indices must use int32 or int64")

    batch_size, topk_width = topk_scores.shape
    if k_per_req.shape != (batch_size,) or k_per_req.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Quest per-request k must be an int32/int64 [batch] tensor")
    if recent_indices.dim() != 2 or recent_indices.shape[0] != batch_size:
        raise ValueError("Quest recent indices must have shape [batch, recent]")
    if recent_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Quest recent indices must use int32 or int64")
    if recent_valid.shape != recent_indices.shape or recent_valid.dtype != torch.bool:
        raise ValueError("Quest recent validity must be a matching bool tensor")

    tensors = (topk_indices, k_per_req, recent_indices, recent_valid)
    if any(tensor.device != topk_scores.device for tensor in tensors):
        raise ValueError("Quest finalize tensors must be on the same CUDA device")
    if not all(tensor.is_contiguous() for tensor in (topk_scores, *tensors)):
        raise ValueError("Quest finalize tensors must be contiguous")

    recent_width = recent_indices.shape[1]
    output_width = topk_width + recent_width
    if output_width <= 0 or output_width > QUEST_FINALIZE_MAX_WIDTH:
        raise ValueError(
            f"Quest Triton finalize width must be in [1, "
            f"{QUEST_FINALIZE_MAX_WIDTH}], got {output_width}"
        )

    output_indices = torch.empty(
        (batch_size, output_width), dtype=torch.int32, device=topk_scores.device
    )
    output_lengths = torch.empty(
        batch_size, dtype=torch.int32, device=topk_scores.device
    )
    if batch_size == 0:
        return output_indices, output_lengths

    block_size = triton.next_power_of_2(output_width)
    _quest_finalize_selected_pages_kernel[(batch_size,)](
        topk_scores,
        topk_indices,
        k_per_req,
        recent_indices,
        recent_valid,
        output_indices,
        output_lengths,
        TOPK_WIDTH=topk_width,
        RECENT_WIDTH=recent_width,
        OUTPUT_WIDTH=output_width,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output_indices, output_lengths
