from __future__ import annotations

from typing import Optional

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
    seq_lens_list = seq_lens.detach().cpu().tolist()

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
    offsets_ptr,  # (bs,) int32 — exclusive prefix sum
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

    # history: all True
    cols = tl.arange(0, BLOCK_S)
    for start in tl.range(0, seq_len, BLOCK_S):
        c = start + cols
        tl.store(
            out_ptr + row_base + c,
            tl.full((BLOCK_S,), 1, dtype=out_ptr.dtype.element_ty),
            mask=c < seq_len,
        )

    # draft block
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
    offsets: torch.Tensor,
    draft_token_num: int,
    *,
    required_numel: int,
    tree_mask_buf: Optional[torch.Tensor] = None,
    block_s: int = 128,
) -> torch.Tensor:
    """Build a flattened FULL custom mask.

    Caller provides:
    - ``seq_lens``: GPU sequence lengths, shape ``(bs,)``.
    - ``offsets``: GPU exclusive prefix sum of
        ``D * (seq_lens[b] + D)``, shape ``(bs,)``.
    - ``required_numel``: exact number of valid output elements.
    - ``tree_mask_buf``: optional preallocated output buffer.

    Returns:
        A flattened output buffer. If ``tree_mask_buf`` is provided,
        the returned tensor may be larger than ``required_numel``.
    """
    if draft_mask.numel() == 0:
        if tree_mask_buf is not None:
            return tree_mask_buf
        return torch.empty(0, dtype=torch.bool, device=draft_mask.device)

    assert (
        draft_mask.dim() == 3
        and draft_mask.shape[1] == draft_mask.shape[2] == draft_token_num
    ), draft_mask.shape
    bs = draft_mask.shape[0]

    draft_mask = draft_mask.to(dtype=torch.bool).contiguous()
    seq = seq_lens.to(dtype=torch.int32).contiguous()
    offsets = offsets.to(dtype=torch.int32).contiguous()

    if tree_mask_buf is None:
        out = torch.empty(
            required_numel,
            dtype=torch.bool,
            device=draft_mask.device,
        )
    else:
        assert tree_mask_buf.numel() >= required_numel, (
            tree_mask_buf.shape,
            required_numel,
        )
        assert tree_mask_buf.dtype in (torch.bool, torch.uint8), tree_mask_buf.dtype
        out = tree_mask_buf.contiguous()

    block_s = triton.next_power_of_2(max(1, block_s))
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
