"""
Index streams for the grouped target-verify decode.

The per-token HCA stream is [swa][compressed tail], one per draft, so a
request's kv is read once per draft. The asm kernel can take four drafts
in one tile, but only if the stream is laid out the way its masking
already works:

[compressed tail][swa]
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# asm v4 only support specified max_seqlen_q
_ALLOWED_BLOCK_Q = (1, 2, 4)


@triton.jit
def _grouped_stream_kernel(
    out_ptr,  # [*] int32
    indptr_ptr,  # [G+1] int32
    slot_ptr,  # [G] int32, state slot of the group's request
    pos_ptr,  # [G] int32, position of the group's LAST draft
    tail_len_ptr,  # [G] int32, compressed entries, all drafts see them
    tail_page_ptr,  # [G, TAIL_W] int32
    ring_stride,
    swa_pages,
    TAIL_W: tl.constexpr,
    TAIL_B: tl.constexpr,
    WIN_B: tl.constexpr,
):
    g = tl.program_id(0)
    base = tl.load(indptr_ptr + g)
    n_tail = tl.load(tail_len_ptr + g).to(tl.int32)

    # The tail goes first. Every draft in the group sees all of it.
    for off in tl.range(0, TAIL_W, TAIL_B):
        j = off + tl.arange(0, TAIL_B)
        m = j < n_tail
        jc = tl.minimum(j, TAIL_W - 1)
        pi = tl.load(tail_page_ptr + g * TAIL_W + jc, mask=m, other=-1).to(tl.int32)
        tl.store(out_ptr + base + j, tl.where(pi >= 0, pi + swa_pages, -1), mask=m)

    # Then the swa from this group's first draft to its last.
    slot = tl.load(slot_ptr + g)
    pos = tl.load(pos_ptr + g)
    n = tl.load(indptr_ptr + g + 1) - base - n_tail
    i = tl.arange(0, WIN_B)
    m = i < n
    abs_pos = pos - n + 1 + i
    val = slot * ring_stride + abs_pos % ring_stride
    tl.store(out_ptr + base + n_tail + i, val, mask=m)


def _group_index(R, num_draft, block_q, device):
    """
    Per-group gather index and group size, built on the device.

    With R=3, num_draft=7, block_q=4 the six groups come out as

        last_token = [3, 6, 10, 13, 17, 20]
        group_size = [4, 3,  4,  3,  4,  3]
    """
    n_full = num_draft // block_q
    remainder = num_draft % block_q
    n_groups = n_full + (1 if remainder else 0)
    ends = (
        torch.arange(1, n_groups + 1, dtype=torch.int64, device=device) * block_q
    ).clamp(max=num_draft)
    base = torch.arange(R, dtype=torch.int64, device=device) * num_draft
    last_token = (base[:, None] + (ends - 1)).reshape(-1)
    group_size = torch.diff(ends, prepend=ends.new_zeros(1))
    return last_token, group_size.to(torch.int32).repeat(R)


def build_grouped_verify_streams(
    *,
    state_slot: torch.Tensor,  # [N] int, per query token
    positions: torch.Tensor,  # [N] int, per query token
    hca_len: torch.Tensor,  # [N] int, compressed entries per token
    hca_page_indices: torch.Tensor,  # [N, Wc] int32
    win: int,
    ring_stride: int,
    swa_pages: int,
    num_draft: int,
    block_q: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (indices, kv_indptr, qo_indptr) for the grouped call.
    """
    assert block_q in _ALLOWED_BLOCK_Q, (
        f"block_q={block_q} is not one of {_ALLOWED_BLOCK_Q}; aiter's v4 "
        "dispatcher would not route it to the tile this path needs"
    )
    dev = state_slot.device
    N = state_slot.shape[0]
    assert N % num_draft == 0, f"{N} tokens is not a whole number of drafts"
    R = N // num_draft
    last_token, group_size = _group_index(R, num_draft, block_q, dev)
    G = last_token.numel()

    slot_g = state_slot.to(torch.int32)[last_token].contiguous()
    pos_g = positions.to(torch.int32)[last_token].contiguous()
    tail_g = hca_len.to(torch.int32)[last_token].contiguous()
    page_g = hca_page_indices[last_token].contiguous()

    lens = tail_g + torch.minimum(pos_g + 1, win + group_size - 1)
    kv_indptr = torch.zeros((G + 1,), dtype=torch.int32, device=dev)
    kv_indptr[1:] = lens.cumsum(0).to(torch.int32)

    Wc = page_g.shape[1]
    # Sized to the worst case rather than to kv_indptr[-1], which would need a
    # device read and so could not be captured in a graph. The streams are
    # packed by kv_indptr, so the slack past the last one is never touched.
    indices = torch.empty(
        (G * (Wc + win + block_q - 1),), dtype=torch.int32, device=dev
    )
    _grouped_stream_kernel[(G,)](
        indices,
        kv_indptr,
        slot_g,
        pos_g,
        tail_g,
        page_g,
        ring_stride,
        swa_pages,
        TAIL_W=max(Wc, 1),
        TAIL_B=min(triton.next_power_of_2(max(Wc, 1)), 1024),
        WIN_B=triton.next_power_of_2(win + block_q - 1),
        num_warps=4,
    )

    qo_indptr = torch.zeros((G + 1,), dtype=torch.int32, device=dev)
    qo_indptr[1:] = group_size.cumsum(0).to(torch.int32)
    return indices, kv_indptr, qo_indptr
