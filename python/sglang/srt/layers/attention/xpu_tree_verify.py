"""Graph-safe expand pass for topk > 1 target verify on intel_xpu.

Tree verify is a cascade: a paged pass over the committed prefix, merged with
an expand pass in which every draft query row attends to its own tree branch.
Branch slots are scattered token ids and the paged decode kernel reads only
64/128-token pages, so each row's branch KV is copied into one scratch page;
every shape is fixed, so XPU graphs can record the pass.
"""

from __future__ import annotations

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

# The paged decode kernel only takes 64/128-token pages, and one scratch page
# must hold a row's whole tree branch (at most Q tokens).
TREE_EXPAND_MAX_DRAFT_TOKENS = 128


def tree_expand_scratch_page(num_draft_tokens: int) -> int:
    return 64 if num_draft_tokens <= 64 else 128


def tree_verify_expand_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    custom_mask: torch.Tensor,
    num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand-pass metadata for topk > 1 target verify: a ``[bs * Q, Q]``
    int32 token-slot table per query row, valid tree-branch slots first, and
    each row's valid count. Only fixed-shape ops, so XPU graphs can record it.

    Valid slots are moved first by sorting on ``col`` (valid) / ``col + Q``
    (masked); the kernel reads only the first ``cache_seqlens[row]`` entries.
    """
    q = num_draft_tokens
    bs = seq_lens.numel()
    offsets = torch.arange(q, device=seq_lens.device).unsqueeze(0)
    cols = offsets.expand(bs, -1) + seq_lens.unsqueeze(1)
    # custom_mask holds one (seq_len + Q)-wide row per query; read its Q
    # draft columns.
    row_starts = torch.nn.functional.pad(
        torch.cumsum((seq_lens + q).repeat_interleave(q), dim=0), (1, 0)
    )[:-1]
    mask_indices = cols.repeat_interleave(q, dim=0) + row_starts[:, None]
    # Graph replay passes a padded batch with the real custom_mask: keep pad
    # rows in range. Their outputs are discarded.
    mask_indices = mask_indices.clamp_(max=custom_mask.numel() - 1)
    mask = custom_mask[mask_indices.view(-1)].view(-1, q)

    col_indices = offsets.expand(mask.shape[0], q)
    keys = torch.where(mask, col_indices, col_indices + q)
    _, sort_order = torch.sort(keys, dim=1)
    slots = req_to_token[req_pool_indices.unsqueeze(1), cols].repeat_interleave(
        q, dim=0
    )
    page_table = slots.gather(1, sort_order).to(torch.int32)
    # A real row always attends to itself; one slot keeps pad rows non-empty.
    cache_seqlens = mask.sum(dim=1).clamp_(min=1).to(torch.int32)
    return page_table, cache_seqlens


def tree_expand_attention(
    *,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    scratch_k: torch.Tensor,
    scratch_v: torch.Tensor,
    scratch_page_table: torch.Tensor,
    scale: float,
    softcap: float,
    window_size: tuple[int, int],
) -> tuple[torch.Tensor, ...]:
    """Expand-pass attention: each query row attends to its tree-branch slots.

    Each row's branch KV is copied into its own scratch page and read by the
    paged decode kernel. Unlike a page_size=1 view, which sgl-kernel-xpu
    serves with a boolean-mask pack and ``.item()``, every shape is fixed, so
    XPU graphs capture it. Returns ``(out, softmax_lse)`` like the kernel.
    """
    rows, width = page_table.shape
    slots = page_table.reshape(-1).long()
    k_flat = key_cache.reshape(-1, *key_cache.shape[-2:])
    v_flat = value_cache.reshape(-1, *value_cache.shape[-2:])
    scratch_k[:rows, :width].copy_(
        k_flat.index_select(0, slots).view(rows, width, *k_flat.shape[1:])
    )
    scratch_v[:rows, :width].copy_(
        v_flat.index_select(0, slots).view(rows, width, *v_flat.shape[1:])
    )
    return flash_attn_with_kvcache(
        q=q,
        k_cache=scratch_k[:rows],
        v_cache=scratch_v[:rows],
        page_table=scratch_page_table[:rows],
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=None,
        max_seqlen_q=1,
        softmax_scale=scale,
        causal=False,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=True,
    )
