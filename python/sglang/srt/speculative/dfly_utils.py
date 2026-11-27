from __future__ import annotations

from typing import Optional, Tuple

import torch


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class DFlyRequestState:
    """Per-request persistent state carried between target forwards for DFly2.

    Stores:
      - ``hidden``: post-final-norm target hidden state at each request's last
        accepted token (``[pool_size, hidden_size]``).
      - ``topk``: per-draft-layer target sparse-indexer top-k pool slots
        captured at each request's last accepted token
        (``[num_layers, pool_size, target_topk]`` int32, ``-1`` padded).
    """

    def __init__(
        self,
        *,
        pool_size: int,
        hidden_size: int,
        num_layers: int,
        target_topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.hidden = torch.zeros((pool_size, hidden_size), dtype=dtype, device=device)
        self.topk = torch.full(
            (num_layers, pool_size, target_topk), -1, dtype=torch.int32, device=device
        )

    def write(
        self,
        *,
        req_pool_indices: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        self.hidden[req_pool_indices] = hidden
        self.topk[:, req_pool_indices] = topk


def _pack_valid_indices(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack non-negative entries of ``indices`` (``[..., K]`` int32) to the front,
    padding unused tail entries with ``-1``. Returns ``(packed, counts)``."""
    valid = indices >= 0
    counts = valid.sum(dim=-1, dtype=torch.int32)
    # Stable descending sort on valid puts valid entries first; no host sync.
    order = torch.argsort(valid.to(torch.int8), dim=-1, descending=True, stable=True)
    packed = torch.gather(indices, -1, order)
    return packed, counts


def build_dfly_extra_topk(
    *,
    state_topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_size: int,
    out_rows: torch.Tensor,
    out_lengths: torch.Tensor,
    align: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-layer sparse top-k extra-KV indices for DFly2 draft attention.

    Each request's captured top-k ``c1`` pool slots (for the token preceding the
    anchor) are packed to the front and repeated across all ``block_size`` rows
    of that request's draft block.

    Returns:
      - ``block_rows``: ``[num_layers, bs * block_size, aligned_topk]`` int32
      - ``block_lengths``: ``[num_layers, bs * block_size]`` int32 (clamped >= 1)
      - ``ctx_rows``: ``[num_layers, bs, aligned_topk]`` int32 (for slot-by-slot
        attentive correction)
      - ``ctx_lengths``: ``[num_layers, bs]`` int32 (true valid count, may be 0)
    """
    num_layers, _, target_topk = state_topk.shape
    bs = req_pool_indices.shape[0]
    num_rows = bs * block_size
    aligned_topk = _align_up(target_topk, align)

    req_idx = req_pool_indices.to(torch.int64)
    gathered = state_topk[:, req_idx, :]  # [L, bs, target_topk]
    packed, raw_counts = _pack_valid_indices(gathered)  # [L, bs, target_topk], [L, bs]

    if aligned_topk > target_topk:
        ctx_rows = torch.full(
            (num_layers, bs, aligned_topk),
            -1,
            dtype=torch.int32,
            device=state_topk.device,
        )
        ctx_rows[:, :, :target_topk].copy_(packed)
    else:
        ctx_rows = packed

    repeated_rows = ctx_rows.repeat_interleave(block_size, dim=1)
    repeated_lengths = raw_counts.clamp_min(1).repeat_interleave(block_size, dim=1)

    out_rows[:, :num_rows, :aligned_topk].copy_(repeated_rows)
    if out_rows.shape[-1] > aligned_topk:
        out_rows[:, :num_rows, aligned_topk:].fill_(-1)
    out_lengths[:, :num_rows].copy_(repeated_lengths)

    return (
        out_rows[:, :num_rows, :aligned_topk],
        out_lengths[:, :num_rows],
        ctx_rows,
        raw_counts,
    )


def build_dfly_swa_indices(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    causal_block: bool,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: Optional[torch.Tensor],
    swa_out_cache_loc: Optional[torch.Tensor],
    request_window_layout=None,
    swa_valid_from: Optional[torch.Tensor] = None,
    align: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build DFly2 SWA page-index tables for both backbone draft attention and
    context-only attentive correction.

    Backbone draft attention reads, for every row in request ``b``'s draft block:
      1. The target layer's sliding-window context ``[prefix_len - swa_window, prefix_len)``.
      2. The draft block's own ``block_size`` tokens written to the verify slots
         (all ``block_size`` tokens when ``causal_block=False``, or ``0..j`` when
         ``causal_block=True``).

    Attentive correction reads ONLY segment (1) (the target's context window).

    Supports both paged SWA (``request_window_layout is None``) and the
    per-request workspace SWA layout (``request_window_layout is not None``).
    Zero host syncs — safe inside CUDA graph capture.

    Returns:
      - ``block_swa_indices``: ``[bs * block_size, aligned_block_width]`` int32
      - ``block_swa_lengths``: ``[bs * block_size]`` int32 (clamped >= 1)
      - ``ctx_swa_indices``: ``[bs, aligned_ctx_width]`` int32
      - ``ctx_swa_lengths``: ``[bs]`` int32 (true valid count, may be 0)
    """
    device = req_pool_indices.device
    bs = req_pool_indices.shape[0]
    num_rows = bs * block_size
    ctx_width = _align_up(swa_window, align)
    block_width = _align_up(swa_window + block_size, align)

    if request_window_layout is not None:
        # RequestWindow.workspace layout for TARGET_VERIFY:
        #   history_loc: [bs * swa_window] holding request b's history at
        #                workspace[b * swa_window : (b + 1) * swa_window]
        #   write_loc:   [bs * block_size] holding request b's block slots
        hist_valid = request_window_layout.history_valid[: bs * swa_window].view(
            bs, swa_window
        )
        hist_loc = (
            request_window_layout.history_loc[: bs * swa_window]
            .to(torch.int32)
            .view(bs, swa_window)
        )
        ctx_raw = torch.where(hist_valid, hist_loc, torch.full_like(hist_loc, -1))
        block_slots = (
            request_window_layout.write_loc[:num_rows]
            .to(torch.int32)
            .view(bs, block_size)
        )
    else:
        assert full_to_swa_mapping is not None and swa_out_cache_loc is not None, (
            "Paged DFly SWA indices require full_to_swa_mapping and swa_out_cache_loc."
        )
        req_idx = req_pool_indices.to(torch.int64)
        prefix = prefix_lens[:bs].to(torch.int64)
        offsets = torch.arange(swa_window, dtype=torch.int64, device=device)
        ctx_pos = prefix[:, None] - swa_window + offsets[None, :]  # [bs, swa_window]
        valid = ctx_pos >= 0
        if swa_valid_from is not None:
            valid = valid & (
                ctx_pos >= swa_valid_from[req_idx].to(torch.int64)[:, None]
            )
        max_col = req_to_token.shape[1] - 1
        safe_pos = ctx_pos.clamp(min=0, max=max_col)
        full_slots = req_to_token[req_idx[:, None], safe_pos].to(torch.int64)
        swa_slots = full_to_swa_mapping[full_slots.clamp(min=0)].to(torch.int32)
        ctx_raw = torch.where(valid, swa_slots, torch.full_like(swa_slots, -1))
        block_slots = swa_out_cache_loc[:num_rows].to(torch.int32).view(bs, block_size)

    # Pack valid context window slots to the front for FlashMLA scanning.
    ctx_packed, ctx_counts = _pack_valid_indices(ctx_raw)  # [bs, swa_window], [bs]
    if ctx_width > swa_window:
        ctx_swa_indices = torch.full(
            (bs, ctx_width), -1, dtype=torch.int32, device=device
        )
        ctx_swa_indices[:, :swa_window].copy_(ctx_packed)
    else:
        ctx_swa_indices = ctx_packed

    # Build per-query block rows: [packed valid context | visible draft block slots].
    block_swa_indices = torch.full(
        (bs, block_size, block_width), -1, dtype=torch.int32, device=device
    )
    # Place packed context slots at [0 .. ctx_counts[b] - 1]
    block_swa_indices[:, :, :swa_window] = ctx_packed[:, None, :]

    # Place draft block slots starting at col = ctx_counts[b]
    slot_idx = torch.arange(block_size, dtype=torch.int64, device=device)
    dst_cols = ctx_counts.to(torch.int64)[:, None, None] + slot_idx[None, None, :]
    dst_cols = dst_cols.expand(bs, block_size, block_size)
    src_slots = block_slots[:, None, :].expand(bs, block_size, block_size)
    if causal_block:
        q_idx = slot_idx[None, :, None]
        k_idx = slot_idx[None, None, :]
        src_slots = torch.where(
            k_idx <= q_idx, src_slots, torch.full_like(src_slots, -1)
        )
        visible_block = (slot_idx + 1).to(torch.int32)[None, :]  # [1, block_size]
    else:
        visible_block = torch.full(
            (1, block_size), block_size, dtype=torch.int32, device=device
        )

    block_swa_indices.scatter_(2, dst_cols, src_slots)
    block_swa_lengths = (
        (ctx_counts[:, None] + visible_block).clamp_min(1).reshape(num_rows)
    )
    return (
        block_swa_indices.reshape(num_rows, block_width),
        block_swa_lengths,
        ctx_swa_indices,
        ctx_counts,
    )
