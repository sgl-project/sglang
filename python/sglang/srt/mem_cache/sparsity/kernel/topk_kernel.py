import torch
import triton
import triton.language as tl


@triton.jit
def _bitwise_sort_with_indices(scores, indices, BLOCK_SIZE: tl.constexpr):
    """
    Sorts (score, index) pairs based on scores in descending order.

    The implementation maps float32 scores to int32 preserving order,
    packs score (high 32 bits) and index (low 32 bits) into int64,
    sorts the packed array, and finally unpacks to retrieve sorted indices.
    """
    # Map float32 scores to int32
    # Positive floats map directly. Negative floats need inversion and sign-bit flipping.
    scores_i32 = scores.to(tl.int32, bitcast=True)
    SIGN_BIT_MASK = -0x80000000
    neg_mapped = (~scores_i32) | SIGN_BIT_MASK
    scores_mapped = tl.where(scores_i32 >= 0, scores_i32, neg_mapped)

    # Pack: [High 32 bits: Score] | [Low 32 bits: Index]
    packed = (scores_mapped.to(tl.int64) << 32) | indices.to(tl.int64)

    # Sort descending
    sorted_packed = tl.sort(packed, dim=0, descending=True)

    # Unpack indices
    sorted_indices = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    return sorted_indices


@triton.jit
def topk_selection_triton_kernel(
    scores,
    num_pages,
    out_indices,
    out_lengths,
    scores_s0,
    scores_s1,
    out_indices_s0,
    out_indices_s1,
    num_recent,
    sparsity_ratio,
    fixed_k,
    max_pages,
    max_out_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Load page count and handle early exit for short sequences
    current_num_pages = tl.load(num_pages + pid)
    if current_num_pages <= num_recent:
        tl.store(out_lengths + pid, 0)
        return

    # Determine the number of top-k pages to select (k)
    # The selection pool excludes the most recent pages which are always kept
    recent_start = current_num_pages - num_recent
    history_len = recent_start

    k = 0
    if fixed_k > 0:
        k = fixed_k - num_recent
    else:
        k = (history_len * sparsity_ratio).to(tl.int32)

    # Clamp k to valid range [1, history_len]
    if k < 1:
        k = 1
    if k > history_len:
        k = history_len.to(tl.int32)

    # Load scores and apply masking
    offs = tl.arange(0, BLOCK_SIZE)
    score_ptrs = scores + pid * scores_s0 + offs * scores_s1

    # Create masks for out-of-bounds pages and recent pages
    # (recent pages are automatically kept, so excluded from top-k sort)
    mask_load = offs < max_pages
    is_invalid = offs >= current_num_pages
    is_recent = (offs >= recent_start) & (offs < current_num_pages)

    scores_val = tl.load(score_ptrs, mask=mask_load, other=float("-inf"))

    # Ensure scores are float32 for bitwise sorting stability
    scores_val = scores_val.to(tl.float32)
    scores_val = tl.where(scores_val != scores_val, float("-inf"), scores_val)  # Handle NaNs
    scores_val = tl.where(is_invalid | is_recent, float("-inf"), scores_val)

    # Perform Top-K selection using Bitwise Radix Sort
    indices = tl.arange(0, BLOCK_SIZE)
    sorted_indices = _bitwise_sort_with_indices(scores_val, indices, BLOCK_SIZE)

    # Construct final output indices
    # First k elements are from the Top-K sort
    # Remaining elements are the forced recent pages
    out_range = tl.arange(0, BLOCK_SIZE)

    topk_vals = sorted_indices
    recent_vals = recent_start + (out_range - k)

    is_topk_slot = out_range < k
    is_recent_slot = (out_range >= k) & (out_range < (k + num_recent))

    # Use INT32_MAX padding so invalid slots are sorted to the end
    INT32_MAX = 0x7FFFFFFF
    merged = tl.full([BLOCK_SIZE], INT32_MAX, dtype=tl.int32)
    merged = tl.where(is_topk_slot, topk_vals, merged)
    merged = tl.where(is_recent_slot, recent_vals, merged)

    # Final sort: ensure output indices are in ascending order
    final_indices = tl.sort(merged, dim=0, descending=False)

    # Write back results
    total_len = k + num_recent
    tl.store(out_lengths + pid, total_len)

    # Replace placeholders with -1 padding
    out_ptrs = out_indices + pid * out_indices_s0 + out_range * out_indices_s1
    final_val_to_store = tl.where(final_indices == INT32_MAX, -1, final_indices)
    tl.store(out_ptrs, final_val_to_store, mask=out_range < max_out_len)


def invoke_topk_selection(
    scores: torch.Tensor,
    num_pages: torch.Tensor,
    num_recent: int,
    sparsity_ratio: float,
    fixed_k: int = -1,
    max_out_len: int = None,
):
    bs, max_num_pages = scores.shape

    # Configure kernel block size
    block_size = triton.next_power_of_2(max_num_pages)
    if block_size < 128:
        block_size = 128

    if max_out_len is None:
        max_out_len = max_num_pages

    out_indices = torch.full(
        (bs, max_out_len), -1, dtype=torch.int32, device=scores.device
    )
    out_lengths = torch.zeros(bs, dtype=torch.int32, device=scores.device)

    # Ensure contiguous memory for kernel access
    if not num_pages.is_contiguous():
        num_pages = num_pages.contiguous()

    grid = (bs,)
    topk_selection_triton_kernel[grid](
        scores,
        num_pages,
        out_indices,
        out_lengths,
        scores.stride(0),
        scores.stride(1),
        out_indices.stride(0),
        out_indices.stride(1),
        num_recent,
        sparsity_ratio,
        fixed_k if fixed_k is not None else -1,
        max_num_pages,
        max_out_len,
        BLOCK_SIZE=block_size,
    )

    return out_indices, out_lengths


def _reference_topk(
    scores: torch.Tensor,
    num_pages: torch.Tensor,
    num_recent: int,
    sparsity_ratio: float,
    fixed_k: int,
    max_out_len: int,
):
    """Reference implementation for verification."""
    bs, max_p = scores.shape
    out_indices = torch.full(
        (bs, max_out_len), -1, dtype=torch.int32, device=scores.device
    )
    out_lengths = torch.zeros(bs, dtype=torch.int32, device=scores.device)

    for i in range(bs):
        n = int(num_pages[i].item())
        if n <= num_recent:
            continue

        recent_start = n - num_recent
        scores_i = scores[i].clone()

        # Mask recent and invalid pages
        scores_i[recent_start:n] = float("-inf")
        scores_i[n:] = float("-inf")

        history_pages = max(recent_start, 1)
        if fixed_k > 0:
            k = max(fixed_k - num_recent, 1)
        else:
            k = max(int(history_pages * sparsity_ratio), 1)
        k = min(k, history_pages)

        # Select Top-K
        topk_idx = torch.topk(scores_i, k=k, dim=0, sorted=False)[1]

        # Merge with recent pages
        recent_idx = torch.arange(
            recent_start, recent_start + num_recent, device=scores.device
        )
        combined = torch.cat([topk_idx, recent_idx], dim=0).sort()[0].to(torch.int32)

        length = int(combined.numel())
        out_indices[i, :length] = combined
        out_lengths[i] = length

    return out_indices, out_lengths


def test_kernel():
    print("Running kernel tests...")
    torch.manual_seed(0)

    # Test Parameters
    page_size = 64
    num_recent = 4
    sparsity_ratio = 0.5
    fixed_k = 12
    device = "cuda"

    # 1. Real Scenario
    seq_lens = torch.tensor([2400], dtype=torch.int32, device=device)
    num_pages = (seq_lens + page_size - 1) // page_size
    max_pages = 64
    scores = torch.arange(max_pages, device=device, dtype=torch.float32).unsqueeze(0)

    out_idx, out_len = invoke_topk_selection(
        scores, num_pages, num_recent, sparsity_ratio, -1, max_out_len=max_pages
    )
    ref_idx, ref_len = _reference_topk(
        scores, num_pages, num_recent, sparsity_ratio, -1, max_out_len=max_pages
    )
    assert torch.all(out_idx == ref_idx) and torch.all(out_len == ref_len)

    # 2. Random Data
    num_pages_rand = torch.tensor([10, 4, 20], dtype=torch.int32, device=device)
    max_pages_rand = 32
    scores_rand = torch.randn((3, max_pages_rand), device=device, dtype=torch.float32)

    out_idx, out_len = invoke_topk_selection(
        scores_rand,
        num_pages_rand,
        num_recent,
        sparsity_ratio,
        fixed_k,
        max_out_len=max_pages_rand,
    )
    ref_idx, ref_len = _reference_topk(
        scores_rand,
        num_pages_rand,
        num_recent,
        sparsity_ratio,
        fixed_k,
        max_out_len=max_pages_rand,
    )
    assert torch.all(out_idx == ref_idx) and torch.all(out_len == ref_len)

    print("All tests passed.")


if __name__ == "__main__":
    test_kernel()
