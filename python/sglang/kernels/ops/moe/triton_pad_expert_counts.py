import torch
import triton
import triton.language as tl


@triton.jit
def _pad_expert_counts_kernel(
    counts_ptr,
    out_ptr,
    num_experts,
    all_tokens,
    BLOCK_E: tl.constexpr,
    NE_POW2: tl.constexpr,
):
    i = tl.arange(0, NE_POW2)
    m = i < num_experts
    c = tl.load(counts_ptr + i, mask=m, other=0).to(tl.int32)
    padded = ((c + BLOCK_E - 1) // BLOCK_E) * BLOCK_E
    padded = tl.where(m, padded, 0)
    # The trailing segment absorbs the difference so the total stays
    # graph-static; its m_indices remain -1 and DeepGEMM skips those rows.
    slack = all_tokens - tl.sum(padded, axis=0)
    padded = tl.where(i == num_experts - 1, padded + slack, padded)
    tl.store(out_ptr + i, padded, mask=m)


def pad_expert_counts(
    counts: torch.Tensor, block_e: int, all_tokens: int
) -> torch.Tensor:
    ne = counts.numel()
    out = torch.empty(ne, dtype=torch.int32, device=counts.device)
    _pad_expert_counts_kernel[(1,)](
        counts,
        out,
        ne,
        all_tokens,
        BLOCK_E=block_e,
        NE_POW2=triton.next_power_of_2(ne),
        num_warps=4,
    )
    return out
