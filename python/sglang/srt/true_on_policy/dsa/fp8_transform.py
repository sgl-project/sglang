"""Ordered DSA selection with the native cache-coordinate transforms.

Keeps logical-token order through physical cache relocation.
"""

import torch

from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk


def ordered_transform(
    logits,
    lengths,
    topk,
    method,
    metadata,
    *,
    cu_seqlens_q=None,
    offsets=None,
    row_starts=None,
    batch_idx_list=None,
    force_unfused=False,
    return_raw=False,
):
    raw = ordered_dsa_topk(logits, lengths, topk, row_starts)
    if force_unfused:
        return (raw, raw) if return_raw else raw
    if method.name == "RAGGED":
        if offsets is None:
            raise ValueError("RAGGED selection requires per-row KV offsets")
        result = torch.where(raw >= 0, raw + offsets[:, None], -1).int()
    elif method.name == "PAGED":
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            _build_flashinfer_paged_args,
        )

        row_to_batch, starts = _build_flashinfer_paged_args(
            attn_metadata=metadata,
            row_starts=row_starts,
            cu_seqlens_q_topk=cu_seqlens_q,
            batch_idx_list=batch_idx_list,
            device=logits.device,
            num_rows=logits.shape[0],
        )
        table = metadata.real_page_table
        if row_to_batch is not None:
            table = table[row_to_batch.long()]
        logical = raw if starts is None else raw + starts[:, None]
        valid = raw >= 0
        safe = logical.clamp_min(0).long()
        # Invalid padded entries must never address beyond the compact table.
        page_indices = torch.where(valid, safe // metadata.page_size, 0)
        physical = (
            table.gather(1, page_indices) * metadata.page_size
            + safe % metadata.page_size
        )
        result = torch.where(valid, physical, -1).int()
    else:
        raise ValueError(f"Unsupported DSA transform: {method}")
    return (result, raw) if return_raw else result
