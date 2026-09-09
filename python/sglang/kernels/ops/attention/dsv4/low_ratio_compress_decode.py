"""Triton decode kernel for the DeepSeek V4.1 ratio-1/2 compressor.

Replaces the eager-torch body of
``DeepSeekV4AttnBackend._low_ratio_compress_decode`` (pairing + pooling +
RMSNorm latent) with one Triton program per decode row, in the style of
``_kpool_decode_update_and_maybe_write_cache_bf16_kernel``:

* ratio 1: the projected key is the latent, ``group_pos = pos``.
* ratio 2: an odd position completes a group — its partner (the previous
  token) is read from the per-request pair state; an even position parks
  itself in the state and completes no group. Padded CUDA-graph rows
  (``raw_out_loc == 0``) are routed to the spare pad row so a live request's
  pair state is never touched. The pair is pooled with a per-dimension
  softmax over ``[partner_score, score]``.
* Every row gets a static output: latent (pre-RoPE, RMSNormed, bf16) is
  returned for the index module, and rows completing no group carry
  ``slots = 0`` (the reserved dummy slot).

The kernel does NOT write the KV pool or index caches; it returns the latent
plus ``group_pos``/``slots`` so the caller (and the index module) consume them
exactly as before. The per-request pair state is updated in place.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _low_ratio_compress_decode_kernel(
    kv_ptr,  # [batch, head_dim] fp32 (ratio 2) or bf16 (ratio 1)
    score_ptr,  # [batch, head_dim] fp32 (ratio 2 only)
    req_ptr,  # [batch] int64 request pool indices (pre pad-routing)
    pos_ptr,  # [batch] int64 positions
    raw_out_loc_ptr,  # [batch] int (ratio 2 only; 0 marks a padded row)
    out_loc_ptr,  # [batch] int, core.c{ratio}_out_loc
    state_kv_ptr,  # [num_req_slots + 1, head_dim] fp32 (ratio 2 only)
    state_score_ptr,  # [num_req_slots + 1, head_dim] fp32 (ratio 2 only)
    norm_w_ptr,  # [head_dim] RMSNorm weight (any float dtype)
    latent_ptr,  # [batch, head_dim] bf16 output (pre-RoPE latent)
    group_pos_ptr,  # [batch] int64 output
    slots_ptr,  # [batch] int64 output
    pad_row,  # spare state row for padded graph rows
    eps,
    head_dim,
    RATIO: tl.constexpr,  # 1 or 2
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < head_dim

    pos = tl.load(pos_ptr + row)

    if RATIO == 2:
        # -- Load current key & score (fp32 projection output) --
        kv = tl.load(kv_ptr + row * head_dim + offs, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        score = tl.load(
            score_ptr + row * head_dim + offs, mask=dim_mask, other=0.0
        ).to(tl.float32)

        # Padded graph rows carry req_pool_idx 0, possibly a live request;
        # route their pair state to the spare row.
        raw = tl.load(raw_out_loc_ptr + row)
        req = tl.where(raw == 0, pad_row, tl.load(req_ptr + row))

        odd = (pos % 2) == 1

        state_base = req * head_dim
        partner_kv = tl.load(
            state_kv_ptr + state_base + offs, mask=dim_mask, other=0.0
        )
        partner_score = tl.load(
            state_score_ptr + state_base + offs, mask=dim_mask, other=0.0
        )

        # Even rows park themselves for their partner; odd rows leave the
        # state untouched (decode rows carry a unique req per program).
        if odd == 0:
            tl.store(state_kv_ptr + state_base + offs, kv, mask=dim_mask)
            tl.store(state_score_ptr + state_base + offs, score, mask=dim_mask)

        # -- Pooling: per-dimension softmax over [partner_score, score] --
        smax = tl.maximum(partner_score, score)
        e0 = tl.exp(partner_score - smax)
        e1 = tl.exp(score - smax)
        pooled = (partner_kv * e0 + kv * e1) / (e0 + e1)

        group_pos = tl.where(odd, pos - 1, pos)
    else:
        pooled = tl.load(kv_ptr + row * head_dim + offs, mask=dim_mask, other=0.0).to(
            tl.float32
        )
        group_pos = pos

    # -- Latent: mirror Compressor.finish — norm the bf16-cast pooled value --
    x = pooled.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    rstd = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(norm_w_ptr + offs, mask=dim_mask, other=0.0).to(tl.float32)
    tl.store(
        latent_ptr + row * head_dim + offs, (x * rstd * w).to(tl.bfloat16), mask=dim_mask
    )

    # Rows completing no group (even positions, padding) have out_loc -1 and
    # write the reserved dummy slot 0 instead, so every shape stays static.
    out_loc = tl.load(out_loc_ptr + row)
    tl.store(group_pos_ptr + row, group_pos)
    tl.store(slots_ptr + row, tl.where(out_loc >= 0, out_loc, 0))


def low_ratio_compress_decode(
    kv: torch.Tensor,
    score: Optional[torch.Tensor],
    req: torch.Tensor,
    pos: torch.Tensor,
    out_loc: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    ratio: int,
    raw_out_loc: Optional[torch.Tensor] = None,
    pair_kv_state: Optional[torch.Tensor] = None,
    pair_score_state: Optional[torch.Tensor] = None,
    pad_row: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode low-ratio compress (pairing + pooling + norm) in one Triton launch.

    Args:
        kv: projected keys, ``[batch, head_dim]``. fp32 when ``ratio == 2``
            (``compressor.project`` output), bf16 when ``ratio == 1``.
        score: gating scores, ``[batch, head_dim]`` fp32; required iff
            ``ratio == 2`` (``None`` for ratio 1).
        req: request pool indices, ``[batch]`` int64. Must be unique per row;
            padded CUDA-graph rows carry 0 and are re-routed via
            ``raw_out_loc``.
        pos: token positions, ``[batch]`` int64.
        out_loc: ``core.c{ratio}_out_loc``, ``[batch]`` int32/int64. Rows
            completing no group carry -1 and get slot 0.
        norm_weight: RMSNorm weight, ``[head_dim]`` (bf16 or fp32).
        eps: RMSNorm epsilon.
        ratio: compress ratio, 1 or 2 (constexpr-selected kernel variant).
        raw_out_loc: ``core.raw_out_loc``, ``[batch]`` int32/int64; 0 marks a
            padded row. Required iff ``ratio == 2``.
        pair_kv_state: pending-partner keys, ``[num_req_slots + 1, head_dim]``
            fp32 (``pool.c2_pair_kv_state[layer_id]``); updated in place.
            Required iff ``ratio == 2``.
        pair_score_state: pending-partner scores, same layout; updated in
            place. Required iff ``ratio == 2``.
        pad_row: spare state row for padded rows
            (``pool.c2_pair_pad_row``).

    Returns:
        latent: ``[batch, head_dim]`` bf16, RMSNormed pre-RoPE latent; the
            index module consumes it directly (``compressor.finish`` output
            equivalent).
        group_pos: ``[batch]`` int64; ``pos - 1`` for odd rows, else ``pos``.
        slots: ``[batch]`` int64; destination cache slot per row (0 for
            no-group rows).
    """
    assert ratio in (1, 2)
    assert kv.ndim == 2 and kv.is_contiguous()
    batch, head_dim = kv.shape
    device = kv.device
    if batch == 0:
        z64 = pos.new_empty(0, dtype=torch.int64)
        return (
            torch.empty((0, head_dim), dtype=torch.bfloat16, device=device),
            z64,
            z64.clone(),
        )

    latent = torch.empty((batch, head_dim), dtype=torch.bfloat16, device=device)
    group_pos = torch.empty((batch,), dtype=torch.int64, device=device)
    slots = torch.empty((batch,), dtype=torch.int64, device=device)

    if ratio == 2:
        assert score is not None and score.shape == kv.shape
        assert raw_out_loc is not None and raw_out_loc.numel() == batch
        assert pair_kv_state is not None and pair_score_state is not None
        assert pair_kv_state.shape == pair_score_state.shape
        assert pair_kv_state.shape[1] == head_dim
        score = score.contiguous()
        raw_out_loc = raw_out_loc.contiguous()
        pair_kv_state = pair_kv_state.contiguous()
        pair_score_state = pair_score_state.contiguous()
    else:
        # Placeholder pointers for the unused ratio-2 operands.
        score = kv
        raw_out_loc = pos
        pair_kv_state = norm_weight
        pair_score_state = norm_weight

    assert pos.is_contiguous() and out_loc.is_contiguous() and req.is_contiguous()
    assert norm_weight.is_contiguous() and norm_weight.numel() == head_dim

    _low_ratio_compress_decode_kernel[(batch,)](
        kv,
        score,
        req,
        pos,
        raw_out_loc,
        out_loc,
        pair_kv_state,
        pair_score_state,
        norm_weight,
        latent,
        group_pos,
        slots,
        pad_row,
        eps,
        head_dim,
        RATIO=ratio,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return latent, group_pos, slots


def low_ratio_compress_decode_torch(
    kv: torch.Tensor,
    score: Optional[torch.Tensor],
    req: torch.Tensor,
    pos: torch.Tensor,
    out_loc: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    ratio: int,
    raw_out_loc: Optional[torch.Tensor] = None,
    pair_kv_state: Optional[torch.Tensor] = None,
    pair_score_state: Optional[torch.Tensor] = None,
    pad_row: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native-torch twin of ``low_ratio_compress_decode`` (reference/bring-up).

    Mirrors ``_low_ratio_compress_decode`` + ``Compressor.finish`` exactly:
    pairing against the per-request state, per-dimension softmax pooling,
    RMSNorm of the bf16-cast pooled value. State tensors are updated in place.
    Signature matches the Triton wrapper; returns ``(latent, group_pos,
    slots)``.
    """
    assert ratio in (1, 2)
    if ratio == 1:
        pooled = kv
        group_pos = pos.clone()
    else:
        assert score is not None and raw_out_loc is not None
        assert pair_kv_state is not None and pair_score_state is not None
        # Padded graph rows carry req_pool_idx 0, possibly a live request;
        # route their pair state to the spare row.
        is_pad = raw_out_loc == 0
        req = torch.where(is_pad, torch.full_like(req, pad_row), req)
        odd = pos % 2 == 1

        partner_kv = pair_kv_state[req]
        partner_score = pair_score_state[req]
        keep = odd.unsqueeze(-1)
        pair_kv_state[req] = torch.where(keep, partner_kv, kv)
        pair_score_state[req] = torch.where(keep, partner_score, score)

        # Compressor.pool_pairs: per-dimension softmax over the pair.
        kv2 = torch.stack([partner_kv, kv], dim=1)
        score2 = torch.stack([partner_score, score], dim=1)
        pooled = (kv2 * score2.softmax(dim=1)).sum(dim=1)
        group_pos = torch.where(odd, pos - 1, pos)

    # finish(): RMSNorm over the bf16-cast pooled value.
    x = pooled.to(torch.bfloat16).to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    latent = (x * norm_weight.to(torch.float32)).to(torch.bfloat16)

    slots = torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc))
    return latent, group_pos.to(torch.int64), slots.to(torch.int64)
