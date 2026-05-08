"""DS-specific FA3 metadata adaptor.

Differs from `FlashAttentionAdaptor` in three CUDA-graph-relevant ways:

1. **In-place writes only.** The base adaptor rebinds attributes
   (`current_metadata.cu_seqlens_k = torch.nn.functional.pad(...)`),
   which orphans FA3's pre-captured graph buffers and breaks replay.
   We write through `.copy_()` / `out=` everywhere.

2. **`max_seq_len_k` is static.** The base adaptor calls
   `int(cache_seqlens.max())` per step — a host sync that is illegal
   inside a captured graph. DS knows the cap upfront
   (`max_selected_per_request`), so we set it once at construction and
   never touch it afterwards.

3. **`scheduler_metadata = None` for the entire server lifetime when
   DS is enabled.** A device-tensor "any DS row active?" branch is not
   legal under graph replay; making this a Python-time decision keeps
   capture+replay correct and isn't measurably slower in v1 (v1.1 may
   recompute scheduler metadata into a graph-safe buffer).

Plus the page_size=1 simplification: `cache_seqlens` is just
`valid_lengths` (no in-page offset math), and the logical→physical
mapping is `req_to_token[req_idx, logical]` (page id == token id).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import FlashAttentionAdaptor

logger = logging.getLogger(__name__)


class DSFlashAttentionAdaptor(FlashAttentionAdaptor):
    """FA3 adaptor specialized for Double Sparsity v1.

    Args:
      device:                CUDA device.
      max_selected_per_request: hard cap (= captured `max_seq_len_k`).
    """

    def __init__(self, device: torch.device, *, max_selected_per_request: int):
        super().__init__(device)
        self.max_selected_per_request = int(max_selected_per_request)

    def save_original_metadata(self, metadata: Any) -> None:
        # Stash a reference to the captured-graph slots — same as the parent
        # but we only need page_table / cache_seqlens / cu_seqlens (we do
        # NOT touch max_seq_len_k after capture, so no need to save it).
        self._original_metadata = {
            "page_table": metadata.page_table.clone(),
            "cache_seqlens_int32": metadata.cache_seqlens_int32.clone(),
            "cu_seqlens_k": metadata.cu_seqlens_k.clone(),
        }
        # scheduler_metadata is statically None for the lifetime of the
        # captured graph — set here once.
        if hasattr(metadata, "scheduler_metadata"):
            metadata.scheduler_metadata = None

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch,
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        if page_size != 1:
            raise ValueError(
                f"DSFlashAttentionAdaptor requires page_size=1, got {page_size}. "
                "Pass --page-size 1 when --enable-double-sparsity is set."
            )
        if self._original_metadata is None:
            return current_metadata
        # Note: no `if sparse_mask.any()` short-circuit here — that branches on
        # device-tensor state, illegal under CUDA-graph capture/replay. If
        # sparse_mask is all-False, `update_mask` becomes all-False below and
        # the `torch.where` calls leave everything at the originals (correct
        # no-op).

        # Restore originals into the captured slots in-place (so unmodified
        # rows fall back to the dense layout for FA3).
        current_metadata.page_table.copy_(self._original_metadata["page_table"])
        current_metadata.cache_seqlens_int32.copy_(
            self._original_metadata["cache_seqlens_int32"]
        )

        # Page id == token id under page_size=1.
        physical = self._logical_to_physical_pages_batch(
            selected_indices,
            forward_batch.req_pool_indices,
            req_to_token,
            page_size,
        )
        # Clamp to the captured page_table width — selected_indices is sized
        # to `max_selected_per_request` (the static graph cap), but the
        # page_table buffer is sized to whatever `max_seq_len_k` was when
        # captured. They aren't always equal: FA3's metadata uses the
        # actual max-pages-per-request seen at this step, which can be
        # smaller than our cap.
        max_selected = min(physical.shape[1], current_metadata.page_table.shape[1])
        physical = physical[:, :max_selected]

        # Per-row mask: only update [: max_selected] for sparsified rows
        # whose entries are within their valid_length window.
        col_range = torch.arange(max_selected, device=physical.device).unsqueeze(0)
        valid_col = col_range < valid_lengths.unsqueeze(1)
        update_mask = sparse_mask.unsqueeze(1) & valid_col

        current_metadata.page_table[:, :max_selected] = torch.where(
            update_mask, physical, current_metadata.page_table[:, :max_selected]
        )

        # cache_seqlens for sparsified rows = valid_lengths (page_size=1).
        # In-place via index_copy_ on the rows that are sparsified.
        new_cache_seqlens = torch.where(
            sparse_mask,
            valid_lengths.to(torch.int32),
            current_metadata.cache_seqlens_int32,
        )
        current_metadata.cache_seqlens_int32.copy_(new_cache_seqlens)

        # cu_seqlens_k is a [bs+1] prefix sum starting at 0. Write [1:] in-place
        # via `cumsum(..., out=)` — no allocation. Index 0 is initialized to 0
        # at buffer construction time and never modified here (a `cs[0] = 0`
        # assignment from a Python int isn't stream-capture-safe).
        cs = current_metadata.cu_seqlens_k
        torch.cumsum(
            current_metadata.cache_seqlens_int32, dim=0, dtype=cs.dtype, out=cs[1:]
        )

        # max_seq_len_k is static: set at construction, never updated here.
        # (Plan: dense-fallback rows fit because min_seq_len <= max_selected_per_request.)

        # scheduler_metadata: statically None for DS lifetime — set once in
        # save_original_metadata. No-op here.
        return current_metadata
