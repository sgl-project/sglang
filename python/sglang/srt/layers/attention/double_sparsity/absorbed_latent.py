"""Absorbed-latent Double Sparsity scoring.

Double Sparsity selects tokens by ``score = query · signature`` where the
signature is ``channel_select(W_UK · c_kv)`` — the per-head K_nope
(``k_nope[h] = W_UK[h] · c_kv``) sliced to the offline-mask channels ``S_h``,
with the channel weights ``w_c`` applied on the query side (query =
``w_c · q_{S_h}``).

Substituting ``k_nope[t,h,c] = Σ_l W_UK[h][c,l] · c_kv[t,l]`` collapses the score
to an inner product against the resident latent::

    score[b,t] = max_h Σ_{c∈S_h} (w_c[h,c] · q[b,h,c]) · k_nope[t,h,c]
               = max_h Σ_l ( Σ_{c∈S_h} w_c[h,c]·q[b,h,c]·W_UK[h][c,l] ) · c_kv[t,l]
               = max_h ( v_h[b] · c_kv[t] )                         # scorer_norm="off"

so the per-token signature IS the latent: the ``v_h`` projection (a few MACs per
head per step) is built query-side from the bind-time-selected ``W_UK`` rows, and
the key side is the fp8 latent the KV pool already stores. No separate signature
store and no prefill label-write hook are needed; the score reads the resident
latent directly (``scorer_norm="off"``).

This module owns the production absorbed-latent scoring math: the bind-time
``build_absorbed_projection`` and the per-step query-side ``absorbed_latent_v``
build (the raw-dot ``scorer_norm="off"`` numerator), plus the resident-latent key
norm (``key_norms_from_latent`` / ``key_norms_from_resident_fp8``) and the cosine
reference (``absorbed_latent_cosine_logical`` / ``_fp8``, ``scorer_norm="cosine"``)
— and the CPU reference scorers used as the exact oracle for the Triton kernel.
"""

from __future__ import annotations

import torch


def build_absorbed_projection(
    kv_b_proj_weight: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    channel_selection: torch.Tensor,
    weight_scale_inv: torch.Tensor = None,
    weight_block_size=None,
) -> torch.Tensor:
    """Bind-time absorbed projection: dequantize the real ``kv_b_proj`` weight and
    return the SELECTED ``W_UK`` rows ``[H, label_dim, kv_lora_rank]`` (fp32).

    Mirrors the model's own ``w_kc`` extraction (deepseek_weight_loader): the
    block-fp8 ``[out, in]`` weight (``out = H·(qk_nope+v_head)``, ``in =
    kv_lora_rank``) is dequantized with the SAME block-fp8 semantics attention
    uses, reshaped to ``[H, qk_nope+v_head, lora]``, sliced to the K-noPE rows
    ``[:qk_nope_head_dim]`` (rope dims excluded by construction), then GATHERED at
    each head's mask channels ``S_h``. Built once at bind; the result is exactly
    the per-mask-channel ``W_UK`` rows the absorbed score consumes, so the
    per-token signature then IS the resident latent.

    Args:
        kv_b_proj_weight: ``[H·(qk_nope+v_head), kv_lora_rank]``.
        channel_selection: ``[H, label_dim]`` int — mask channel indices into
            ``qk_nope_head_dim``.
        weight_scale_inv / weight_block_size: block-fp8 dequant inputs; when both
            present, dequantize via ``block_quant_dequant``, else ``.float()``.

    Returns:
        ``[H, label_dim, kv_lora_rank]`` fp32 — ``W_UK[h][S_h[d], :]``.
    """
    if weight_scale_inv is not None and weight_block_size is not None:
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        w = block_quant_dequant(
            kv_b_proj_weight, weight_scale_inv, list(weight_block_size), torch.float32
        )
    else:
        w = kv_b_proj_weight.to(torch.float32)
    out, lora = w.shape
    head_width = qk_nope_head_dim + v_head_dim
    assert (
        out == num_heads * head_width
    ), f"kv_b_proj out {out} != num_heads*(qk_nope+v_head) {num_heads * head_width}"
    # [H, qk_nope+v_head, lora] -> K-noPE rows -> W_UK [H, qk_nope, lora]
    w_kc = w.view(num_heads, head_width, lora)[:, :qk_nope_head_dim, :]
    # gather the mask channels per head -> [H, label_dim, lora]
    sel = channel_selection.long().to(w_kc.device)
    return torch.gather(w_kc, 1, sel.unsqueeze(-1).expand(-1, -1, lora)).contiguous()


def absorbed_latent_v(
    queries: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-head query-side latent projection ``v_h`` (the absorbed score query),
    fp32 to match the production ``_logical_score`` accumulation.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]`` — no-PE query, BEFORE channel
            projection (same input as ``project_query_onto_channels``).
        w_sel: ``[H, label_dim, kv_lora_rank]`` — the bind-time-selected ``W_UK``
            rows from :func:`build_absorbed_projection`.
        channel_selection: ``[H, label_dim]`` int — the query channels ``S_h``.
        channel_weights: ``[H, label_dim]`` float — the per-channel weights ``w_c``.

    Returns:
        ``[bs, H, kv_lora_rank]`` fp32 — ``v_h[b] = Σ_{c∈S_h} w_c·q_c · W_UK[h][c,:]``.
    """
    bs = queries.shape[0]
    sel = channel_selection.long()
    # weighted query at the selected channels: w_c · q_{S_h}  -> [bs, H, label_dim]
    q_sel = torch.gather(
        queries.to(torch.float32), 2, sel.unsqueeze(0).expand(bs, -1, -1)
    ) * channel_weights.to(torch.float32).unsqueeze(0)
    # v_h = Σ_d (w_c·q_c) · w_sel[h, d, :]  -> [bs, H, lora]
    return torch.einsum("bhd,hdl->bhl", q_sel, w_sel.to(torch.float32))


def absorbed_latent_v_into(
    out: torch.Tensor,
    queries: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    scratch_qsel: torch.Tensor,
    scratch_q: torch.Tensor,
) -> torch.Tensor:
    """Allocation-free :func:`absorbed_latent_v` — writes ``v_h`` into a caller-owned
    ``out[:bs]`` (fp32 ``[bs_buf, H, kv_lora_rank]``) for the graph-safe path.

    Same value as :func:`absorbed_latent_v`: ``v_h[b,h,l] = Σ_d (w_c·q_c)·w_sel[h,d,l]``.
    Every step writes into caller-owned scratch (an in-place fp32 cast of ``queries``,
    ``gather(out=)``, ``mul_``, then a head-batched ``bmm(out=)`` over a transposed
    view of ``out``), so after warmup the caching-allocator counter does not grow —
    CUDA-graph safe. ``scratch_qsel`` is fp32 ``[bs_buf, H, label_dim]`` for the
    weighted-gather; ``scratch_q`` is fp32 ``[bs_buf, H, qk_nope_head_dim]`` that the
    bf16/fp16 served ``queries`` are cast into in place (``copy_`` does the dtype cast
    with no allocation), so the hot path never calls ``queries.to(torch.float32)``.
    ``channel_selection`` must be int64 here (the caller pre-copies the int32 layer
    mask into an int64 scratch so ``gather`` does no per-step ``.long()`` allocation).
    """
    bs = queries.shape[0]
    out_v = out[:bs]  # [bs, H, lora]
    label_dim = int(channel_selection.shape[1])
    q_sel = scratch_qsel[:bs, :, :label_dim]  # [bs, H, label_dim]
    # Cast the served bf16/fp16 query into the fp32 scratch in place (copy_ does
    # the dtype conversion without allocating), so the gather reads fp32 without a
    # per-step queries.to(torch.float32).
    q_f32 = scratch_q[:bs, :, : queries.shape[2]]  # [bs, H, qk_nope_head_dim]
    q_f32.copy_(queries)
    # weighted query at the selected channels: w_c · q_{S_h}  -> [bs, H, label_dim].
    torch.gather(
        q_f32,
        2,
        channel_selection.unsqueeze(0).expand(bs, -1, -1),
        out=q_sel,
    )
    q_sel.mul_(channel_weights.to(torch.float32).unsqueeze(0))
    # v_h = Σ_d (w_c·q_c) · w_sel[h, d, :] as a per-head bmm: batch the head axis so
    # the contraction is [H, bs, label_dim] @ [H, label_dim, lora] -> [H, bs, lora],
    # written straight into a head-major (transposed) view of out_v with out=.
    q_hbd = q_sel.transpose(0, 1)  # [H, bs, label_dim] (view)
    w_sel_f = w_sel.to(torch.float32)  # [H, label_dim, lora]
    v_hbl = out_v.transpose(0, 1)  # [H, bs, lora] (view of out_v)
    torch.bmm(q_hbd, w_sel_f, out=v_hbl)
    return out_v


def absorbed_latent_score(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    head_agg: str = "max",
) -> torch.Tensor:
    """Per-(query, token) selection score from the latent, no table.

    ``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` for ``scorer_norm="off"``.

    Args:
        queries: ``[bs, H, qk_nope_head_dim]``.
        c_kv: ``[T, kv_lora_rank]`` — the resident MLA KV latent (dequantized).
        w_sel, channel_selection, channel_weights: see :func:`absorbed_latent_v`.
        head_agg: ``"max"`` (default) or ``"mean"``.

    Returns:
        ``[bs, T]`` fp32 scores.
    """
    v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    dots = torch.einsum("bhl,tl->bht", v, c_kv.to(torch.float32))  # [bs, H, T]
    if head_agg == "mean":
        return dots.mean(dim=1)
    return dots.amax(dim=1)


def absorbed_latent_score_logical(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: torch.Tensor = None,
    head_agg: str = "max",
) -> torch.Tensor:
    """Logical-domain paged absorbed score — scores the resident latent
    directly (no materialized signatures).

    For each request, walks logical positions ``0..max_seq_len`` through
    ``req_to_token[pool, pos] -> physical slot``, gathers ``c_kv`` at the physical
    slot, scores ``agg_h (v_h · c_kv[slot])``, and masks unwritten slots (if
    ``written`` given) then positions ``>= seq_len`` to ``-inf`` — same order as
    the production logical scorer. Returns ``[bs, max_seq_len]`` fp32 (feed to
    ``select_topk_sequence_order``).

    Args:
        queries: ``[bs, H, qk_nope_head_dim]``.
        c_kv: ``[max_tokens, kv_lora_rank]`` physical-slot latent (dequantized).
        w_sel: ``[H, label_dim, kv_lora_rank]`` (from :func:`build_absorbed_projection`).
        req_pool_indices: ``[bs]`` int; req_to_token: ``[num_pools, max_seqlen]`` int;
        seq_lens: ``[bs]`` int; written: optional ``[max_tokens]`` bool.
    """
    bs = queries.shape[0]
    device = queries.device
    if max_seq_len <= 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)
    v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    safe_pool = req_pool_indices.clamp(0, max(req_to_token.shape[0] - 1, 0)).long()
    logical_positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(bs, -1)
    )
    safe_positions = logical_positions.clamp(0, max(req_to_token.shape[1] - 1, 0))
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_seq_len)
    physical_slots = req_to_token[pool_expanded, safe_positions.long()]
    max_tokens = c_kv.shape[0]
    safe_phys = physical_slots.long().clamp(0, max(max_tokens - 1, 0))
    gathered = c_kv[safe_phys].to(torch.float32)  # [bs, max_seq_len, lora]
    dots = torch.einsum("bhl,bil->bih", v, gathered)  # [bs, max_seq_len, H]
    scores = dots.mean(dim=-1) if head_agg == "mean" else dots.amax(dim=-1)
    if written is not None:
        scores = scores.masked_fill(~written[safe_phys], float("-inf"))
    seq_len_mask = logical_positions < seq_lens.unsqueeze(1).to(device)
    return scores.masked_fill(~seq_len_mask, float("-inf"))


def dequantize_resident_latent(
    latent_fp8: torch.Tensor, latent_scales: torch.Tensor
) -> torch.Tensor:
    """fp8-e4m3 resident latent + per-block fp32 scales -> fp32 ``c_kv`` ``[T, lora]``.

    The KV pool stores the MLA noPE latent as fp8 with one fp32 scale per
    128-channel block. This reverses that exactly in fp32 — the dequant the cosine
    reference and the key-norm populate both score against (the resident bytes the
    score kernel reads, NOT the pre-quant ``k``). ``latent_fp8`` is ``[T, lora]``
    viewed as ``float8_e4m3fn``; ``latent_scales`` is ``[T, nblk]`` fp32.
    """
    t, lora = latent_fp8.shape
    nblk = latent_scales.shape[-1]
    block = lora // nblk
    deq = latent_fp8.to(torch.float32).view(t, nblk, block)
    deq = deq * latent_scales.to(torch.float32).view(t, nblk, 1)
    return deq.view(t, lora)


def key_norms_from_latent(
    w_sel: torch.Tensor,
    c_kv: torch.Tensor,
) -> torch.Tensor:
    """Per-(token, head) key norm ``||K_label_h[t]|| = ||w_sel[h] @ c_kv[t]||`` for
    the cosine scorer, computed from the resident dequantized latent ``c_kv``.

    ``c_kv[t]`` must be the resident latent the score kernel actually reads
    (fp8-dequantized on the FP8 KV path, or the bf16 ``k_nope`` as fp32 on the BF16
    path) — NOT the pre-quant ``k`` — so the norm matches the absorbed raw-dot
    numerator's denominator. ``K_label_h[t] = w_sel[h] @ c_kv[t]`` is the per-head
    signature; its L2 norm over the ``label_dim`` axis is the per-head key norm.

    Args:
        w_sel: ``[H, label_dim, lora]`` fp32 — bind-time K-noPE ``W_UK`` rows.
        c_kv: ``[T, lora]`` — the dequantized resident latent.

    Returns:
        ``[T, H]`` fp32 key norms.
    """
    w = w_sel.to(torch.float32)  # [H, label_dim, lora]
    # K_label[t, h, d] = sum_l w[h, d, l] * c_kv[t, l]
    k_label = torch.einsum("hdl,tl->thd", w, c_kv.to(torch.float32))  # [T, H, label_dim]
    return k_label.norm(dim=-1)  # [T, H]


def key_norms_from_resident_fp8(
    w_sel: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
) -> torch.Tensor:
    """:func:`key_norms_from_latent` over the fp8-dequantized resident latent (the
    FP8 KV path). The BF16-path analogue is :func:`key_norms_from_latent` called
    with the bf16 ``k_nope`` (as fp32) directly.
    """
    return key_norms_from_latent(
        w_sel, dequantize_resident_latent(latent_fp8, latent_scales)
    )


def absorbed_latent_cosine_logical(
    queries: torch.Tensor,
    c_kv: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: torch.Tensor = None,
    head_agg: str = "max",
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """Direction-normalized (cosine) absorbed score oracle — the exact reference
    the served cosine kernel must match (``scorer_norm="cosine"``).

    Mirrors :func:`absorbed_latent_score_logical` (same paged walk through
    ``req_to_token``, same ``written`` / ``seq_len`` masking) but divides each
    per-head dot by the per-head query and key norms BEFORE the head aggregation::

        score[b, t] = agg_h ( Q_label_h · K_label_h[t] )
                            / ( ||Q_label_h|| · ||K_label_h[t]|| + eps )

    where ``Q_label_h = w_c ⊙ q_{S_h}`` (the weighted query at the mask channels,
    in label-dim space) and ``K_label_h[t] = w_sel[h] @ c_kv[t]``. The numerator is
    exactly the raw-dot absorbed score; cosine adds only the per-head division,
    taken AFTER the mask-channel gather (label-dim space, NOT lora space). With
    ``normalize=False`` it returns the raw-dot numerator through this same path —
    the materialized-raw control, which must equal
    :func:`absorbed_latent_score_logical`. ``c_kv`` is the dequantized resident
    latent (fp8-dequant on the FP8 path, bf16-as-fp32 on the BF16 path). fp32
    throughout. Returns ``[bs, max_seq_len]`` fp32.
    """
    bs = queries.shape[0]
    device = queries.device
    if max_seq_len <= 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)
    cs = channel_selection.long()
    cw = channel_weights.to(torch.float32)
    # Q_label_h = w_c ⊙ q_{S_h}  -> [bs, H, label_dim]
    q_sel = torch.gather(
        queries.to(torch.float32), 2, cs.unsqueeze(0).expand(bs, -1, -1)
    ) * cw.unsqueeze(0)
    q_norm = q_sel.norm(dim=-1)  # [bs, H]
    safe_pool = req_pool_indices.clamp(0, max(req_to_token.shape[0] - 1, 0)).long()
    logical_positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(bs, -1)
    )
    safe_positions = logical_positions.clamp(0, max(req_to_token.shape[1] - 1, 0))
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_seq_len)
    physical_slots = req_to_token[pool_expanded, safe_positions.long()]  # [bs, S]
    max_tokens = c_kv.shape[0]
    safe_phys = physical_slots.long().clamp(0, max(max_tokens - 1, 0))
    gathered = c_kv[safe_phys].to(torch.float32)  # [bs, S, lora]
    w_sel_f = w_sel.to(torch.float32)  # [H, label_dim, lora]
    # K_label[b,s,h,d] = w_sel[h,d,:]·gathered[b,s,:] -> [bs, S, H, label_dim]
    k_label = torch.einsum("hdl,bsl->bshd", w_sel_f, gathered)
    k_norm = k_label.norm(dim=-1)  # [bs, S, H]
    dots = torch.einsum("bhd,bshd->bsh", q_sel, k_label)  # [bs, S, H]
    if normalize:
        per_head = dots / (q_norm.unsqueeze(1) * k_norm + eps)  # cosine [bs, S, H]
    else:
        per_head = dots  # materialized-raw control (cosine numerator, no division)
    scores = per_head.mean(dim=-1) if head_agg == "mean" else per_head.amax(dim=-1)
    if written is not None:
        scores = scores.masked_fill(~written[safe_phys], float("-inf"))
    seq_len_mask = logical_positions < seq_lens.unsqueeze(1).to(device)
    return scores.masked_fill(~seq_len_mask, float("-inf"))


def absorbed_latent_cosine_logical_fp8(
    queries: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    w_sel: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: torch.Tensor = None,
    head_agg: str = "max",
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """:func:`absorbed_latent_cosine_logical` over the fp8-dequantized resident
    latent (the FP8 KV path): dequantizes ``latent_fp8`` / ``latent_scales`` to
    ``c_kv`` via :func:`dequantize_resident_latent`, then scores. The BF16-path
    analogue is :func:`absorbed_latent_cosine_logical` called with the bf16
    ``k_nope`` (as fp32) directly.
    """
    c_kv = dequantize_resident_latent(latent_fp8, latent_scales)
    return absorbed_latent_cosine_logical(
        queries,
        c_kv,
        w_sel,
        channel_selection,
        channel_weights,
        req_pool_indices,
        req_to_token,
        seq_lens,
        max_seq_len,
        written=written,
        head_agg=head_agg,
        eps=eps,
        normalize=normalize,
    )
