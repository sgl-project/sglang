from __future__ import annotations

import torch
import triton
import triton.language as tl


def build_ngram_full_tree_mask_ref(
    draft_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    """Reference implementation (same semantics as the old Python loop)."""
    assert draft_mask.dim() == 3
    bs, d1, d2 = draft_mask.shape
    assert d1 == d2 == draft_token_num
    device = draft_mask.device
    seq_lens_list = seq_lens.tolist()

    pieces = []
    for i in range(bs):
        seq_len = int(seq_lens_list[i])
        left = torch.ones((draft_token_num, seq_len), dtype=torch.bool, device=device)
        right = draft_mask[i].to(dtype=torch.bool)
        pieces.append(torch.cat((left, right), dim=1).reshape(-1))
    if not pieces:
        return torch.empty(0, dtype=torch.bool, device=device)
    return torch.cat(pieces, dim=0)


@triton.jit
def _build_ngram_full_tree_mask_kernel(
    draft_mask_ptr,  # (bs, D, D) bool/uint8, contiguous
    seq_lens_ptr,  # (bs,) int32
    offsets_ptr,  # (bs,) int32, start of each req in out
    out_ptr,  # (total,) bool/uint8
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    bid = tl.program_id(0)  # request
    rid = tl.program_id(1)  # draft row

    seq_len = tl.load(seq_lens_ptr + bid)
    offset = tl.load(offsets_ptr + bid)
    row_width = seq_len + D
    row_base = offset + rid * row_width

    # ---- left: history visible (all True) ----
    cols = tl.arange(0, BLOCK_S)
    for start in tl.range(0, seq_len, BLOCK_S):
        c = start + cols
        tl.store(
            out_ptr + row_base + c,
            tl.full((BLOCK_S,), 1, dtype=out_ptr.dtype.element_ty),
            mask=c < seq_len,
        )

    # ---- right: draft_mask[bid, rid, :] ----
    dcols = tl.arange(0, BLOCK_D)
    in_range = dcols < D
    vals = tl.load(
        draft_mask_ptr + (bid * D + rid) * D + dcols,
        mask=in_range,
        other=0,
    )
    tl.store(
        out_ptr + row_base + seq_len + dcols,
        vals,
        mask=in_range,
    )


def build_ngram_full_tree_mask(
    draft_mask: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    *,
    block_s: int = 128,
) -> torch.Tensor:
    """Build flattened FULL custom_mask for an ngram verify batch.

    Args:
        draft_mask: (bs, D, D) bool CUDA tensor. draft-to-draft visibility.
        seq_lens: (bs,) int tensor (CPU or CUDA). history length per req.
        draft_token_num: D.
        block_s: Triton block size for the history fill loop.

    Returns:
        (total,) bool CUDA tensor, total = sum_b D * (seq_lens[b] + D).
    """
    if draft_mask.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=draft_mask.device)

    assert draft_mask.is_cuda, "draft_mask must be on CUDA"
    assert draft_mask.dim() == 3, draft_mask.shape
    bs, d1, d2 = draft_mask.shape
    assert d1 == d2 == draft_token_num, (draft_mask.shape, draft_token_num)
    assert seq_lens.numel() == bs, (seq_lens.shape, bs)

    device = draft_mask.device
    # Kernel loads bytes; bool is 1-byte. Force contiguous bool.
    draft_mask = draft_mask.to(dtype=torch.bool).contiguous()
    seq = seq_lens.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()

    # sizes[b] = D * (seq_lens[b] + D)
    sizes = draft_token_num * (seq + draft_token_num)
    offsets = torch.empty(bs, dtype=torch.int32, device=device)
    offsets[0] = 0
    if bs > 1:
        torch.cumsum(sizes[:-1], dim=0, out=offsets[1:])

    total = int(sizes.sum().item())
    out = torch.empty(total, dtype=torch.bool, device=device)

    block_d = triton.next_power_of_2(draft_token_num)
    grid = (bs, draft_token_num)
    _build_ngram_full_tree_mask_kernel[grid](
        draft_mask,
        seq,
        offsets,
        out,
        D=draft_token_num,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
    )
    return out
