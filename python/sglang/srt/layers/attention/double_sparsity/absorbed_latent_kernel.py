"""GPU paged absorbed-latent score kernel (production DS selection score).

Companion to ``absorbed_latent.py``: the same identity
``score[b, t] = agg_h ( v_h[b] · c_kv[t] )`` (``scorer_norm="off"``), but the key
side is read from the RESIDENT paged fp8 MLA latent instead of a materialized
signature table, with per-128-channel-block dequantization done in-register. The
query-side projection ``v_h`` is built once per step on the host
(``absorbed_latent.absorbed_latent_v``) and handed to the kernel, so the kernel
is a paged ``max_h Σ_l v_h[b,h,l] · dequant(latent[slot,l])`` reduction.

The kernel uses a persistent-worker topology: a static ``(bs, WORKERS)`` grid,
each worker striding over the token blocks it owns, loop bound = the LIVE block
count, written-then-``seq_len`` masking in selection order. The per-element
dequant-then-dot matches the CPU reference value-for-value (only fp32 summation
order reassociates), so the CPU ``absorbed_latent_score_logical`` is its exact
oracle. This is the production DS selection score path. Value-affecting (the fp8
latent vs the bf16-pre-quant label), declared, recall-gated — not a bit-identity
claim.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only import path
    _HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def quantize_latent_fp8(
    c_kv: torch.Tensor, *, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-128-channel-block fp8 quantization of the MLA nope latent.

    Matches the pool's ``quantize_k_cache_separate`` scheme: per block
    ``s = max|tile| / FP8_MAX; q = clamp(tile / s, ±FP8_MAX).to(fp8)``.

    Args:
        c_kv: ``[T, lora]`` fp32/bf16 latent. ``lora % block_size == 0``.

    Returns:
        ``(fp8 [T, lora] float8_e4m3fn, scales [T, lora//block_size] fp32)`` —
        the two tensors ``get_mla_kv_buffer`` exposes after unpacking the
        ``[512 fp8 | 4 fp32 scales]`` pool bytes.
    """
    T, lora = c_kv.shape
    assert lora % block_size == 0, f"lora {lora} not a multiple of block {block_size}"
    nblk = lora // block_size
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    src = c_kv.to(torch.float32)
    tiles = src.view(T, nblk, block_size)
    scales = tiles.abs().amax(dim=2) / fp8_max  # [T, nblk]
    safe = scales.clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.clamp(tiles / safe.unsqueeze(2), -fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    return q.view(T, lora).contiguous(), scales.contiguous()


def dequantize_latent_fp8(
    fp8: torch.Tensor, scales: torch.Tensor, *, block_size: int = 128
) -> torch.Tensor:
    """Inverse of :func:`quantize_latent_fp8` — the value-for-value latent the
    kernel scores against; feed this to the CPU reference as the oracle's input."""
    T, lora = fp8.shape
    nblk = lora // block_size
    deq = fp8.to(torch.float32).view(T, nblk, block_size) * scales.unsqueeze(2)
    return deq.view(T, lora)


if _HAS_TRITON:

    @triton.jit
    def _absorbed_score_kernel(
        v_ptr,  # [bs, H, lora] fp32 (precomputed v_h)
        fp8_ptr,  # [max_tokens, lora] float8_e4m3fn (paged nope latent)
        scale_ptr,  # [max_tokens, nblk] fp32 (per-128-block scales)
        written_ptr,  # [max_tokens] bool
        rpi_ptr,  # [bs] int32
        rtt_ptr,  # [num_pools, max_pool_len] int32
        sl_ptr,  # [bs] int32
        out_ptr,  # [bs, max_seq_len] fp32 (pre-allocated)
        q_norm_ptr,  # [bs, H] fp32 — per (batch, head) query norm (COSINE only)
        k_norm_ptr,  # [max_tokens, H] fp32 — per (slot, head) key norm (COSINE only)
        qpe_ptr,  # [bs, H, rope_dim] fp32 — post-RoPE query (HAS_ROPE only)
        kpe_ptr,  # [max_tokens, rope_dim] bf16 — resident RoPE key (HAS_ROPE only)
        qpe_stride_b: tl.constexpr,
        qpe_stride_h: tl.constexpr,
        qpe_stride_r: tl.constexpr,
        kpe_stride_t: tl.constexpr,
        num_heads: tl.constexpr,
        max_seq_len: tl.constexpr,
        lora: tl.constexpr,
        block_size: tl.constexpr,
        max_pool_len: tl.constexpr,
        max_tokens: tl.constexpr,
        v_stride_b: tl.constexpr,
        v_stride_h: tl.constexpr,
        fp8_stride_t: tl.constexpr,
        scale_stride_t: tl.constexpr,
        rtt_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr,
        qn_stride_b: tl.constexpr,
        kn_stride_t: tl.constexpr,
        kn_stride_h: tl.constexpr,
        eps: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        H_POW2: tl.constexpr,
        HEAD_AGG_MEAN: tl.constexpr,
        STORE_DEAD_NEG_INF: tl.constexpr,
        HAS_WRITTEN: tl.constexpr,
        COSINE: tl.constexpr,
        BF16_LATENT: tl.constexpr,
        HAS_ROPE: tl.constexpr,
        ROPE_DIM: tl.constexpr,
        WORKERS: tl.constexpr,
    ):
        # Persistent-worker layout: static (bs, WORKERS) grid; each worker
        # strides its token blocks; the
        # loop bound is the LIVE block count (device-computed from seq_len). The
        # full-width torch.topk consumer scans the whole scratch, so dead blocks
        # are filled with -inf when STORE_DEAD_NEG_INF is set.
        batch_id = tl.program_id(0)
        worker = tl.program_id(1)

        seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
        n_live = tl.minimum(seq_len_i, max_seq_len)
        live_blocks = (n_live + TOKEN_BLOCK - 1) // TOKEN_BLOCK
        if STORE_DEAD_NEG_INF:
            nblk = (max_seq_len + TOKEN_BLOCK - 1) // TOKEN_BLOCK
        else:
            nblk = live_blocks

        pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)
        h_offs = tl.arange(0, H_POW2)
        h_mask = h_offs < num_heads
        c_offs = tl.arange(0, block_size)
        nblk_lat = lora // block_size  # number of 128-channel latent blocks

        for tok_blk in range(worker, nblk, WORKERS):
            tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
            in_range = tok_offs < max_seq_len
            if tok_blk * TOKEN_BLOCK >= seq_len_i:
                tl.store(
                    out_ptr + batch_id * out_stride_b + tok_offs,
                    tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32),
                    mask=in_range,
                )
            else:
                pos_valid = in_range & (tok_offs < seq_len_i)

                safe_tok = tl.minimum(tok_offs, max_pool_len - 1)
                phys = tl.load(
                    rtt_ptr + pool_idx * rtt_stride_p + safe_tok,
                    mask=in_range,
                    other=0,
                ).to(tl.int64)
                safe_phys = tl.minimum(tl.maximum(phys, 0), max_tokens - 1)

                if HAS_WRITTEN:
                    written = tl.load(
                        written_ptr + safe_phys, mask=in_range, other=0
                    ).to(tl.int1)
                    valid = pos_valid & written
                else:
                    valid = pos_valid

                # Block-scale reassociation with a tensor-core MMA: per 128-wide
                # latent block, partial[tok, h] = Σ_{l∈blk} v[h,l]·fp8(latent[tok,l])
                # via tl.dot, then weight by that token's fp32 block scale and
                # accumulate fp32 per head — never materializing a [TOKEN_BLOCK, lora]
                # dequant tile (what the spilled per-head vector loop did). Real
                # arithmetic is identical to dequant-then-dot; only the tf32 MMA and
                # the fp32 reassociation differ (declared value-affecting, recall-gated).
                acc = tl.zeros((TOKEN_BLOCK, H_POW2), dtype=tl.float32)
                for blk in range(nblk_lat):
                    cols = blk * block_size + c_offs
                    lat_blk = tl.load(
                        fp8_ptr + safe_phys[:, None] * fp8_stride_t + cols[None, :],
                        mask=in_range[:, None],
                        other=0.0,
                    ).to(
                        tl.float32
                    )  # [TOKEN_BLOCK, block_size]
                    v_blk_t = tl.load(
                        v_ptr
                        + batch_id * v_stride_b
                        + h_offs[None, :] * v_stride_h
                        + cols[:, None],
                        mask=h_mask[None, :],
                        other=0.0,
                    ).to(
                        tl.float32
                    )  # [block_size, H_POW2]
                    partial = tl.dot(lat_blk, v_blk_t, allow_tf32=True)  # [TB, H_POW2]
                    if BF16_LATENT:
                        # bf16 resident k_nope is already the dequantized latent
                        # value — no per-128-block fp8 scale to reapply.
                        acc += partial
                    else:
                        sc_blk = tl.load(
                            scale_ptr + safe_phys * scale_stride_t + blk,
                            mask=in_range,
                            other=0.0,
                        ).to(tl.float32)
                        acc += partial * sc_blk[:, None]

                if COSINE:
                    # Direction-normalize each per-head dot into a cosine score
                    # BEFORE the head reduce: acc[t,h] /= (|Q_label_h|·|K_label_h[t]|
                    # + eps). q_norm is per (batch, head); k_norm is per (physical
                    # slot, head), gathered with the SAME safe_phys physical-slot
                    # index the latent uses. Pad heads (h >= num_heads) load 1.0 so
                    # their already-zero dot stays 0 before the h_mask reduce. This
                    # is the only difference from the raw dot — COSINE=False elides
                    # the whole block, so scorer_norm="off" stays byte-identical.
                    qn = tl.load(
                        q_norm_ptr + batch_id * qn_stride_b + h_offs,
                        mask=h_mask,
                        other=1.0,
                    ).to(tl.float32)
                    kn = tl.load(
                        k_norm_ptr
                        + safe_phys[:, None] * kn_stride_t
                        + h_offs[None, :] * kn_stride_h,
                        mask=in_range[:, None] & h_mask[None, :],
                        other=1.0,
                    ).to(tl.float32)
                    acc = acc / (qn[None, :] * kn + eps)

                if HAS_ROPE:
                    # Add the RoPE term q_pe_h·k_pe[t] per head BEFORE the head
                    # reduce — rank tokens by the full attention logit (no-PE
                    # absorbed dot + rope), not no-PE alone. k_pe[t] is the
                    # resident post-RoPE rope key (shared across heads in MLA);
                    # q_pe[h] the post-RoPE query. rope[t,h] = Σ_r k_pe[t,r]·q_pe[h,r]
                    # via a [TOKEN_BLOCK, ROPE_DIM] @ [ROPE_DIM, H_POW2] MMA. Pad
                    # heads load 0 (h_mask) so their rope add is 0 before the reduce.
                    r_offs = tl.arange(0, ROPE_DIM)
                    kpe_blk = tl.load(
                        kpe_ptr + safe_phys[:, None] * kpe_stride_t + r_offs[None, :],
                        mask=in_range[:, None],
                        other=0.0,
                    ).to(
                        tl.float32
                    )  # [TOKEN_BLOCK, ROPE_DIM]
                    qpe_t = tl.load(
                        qpe_ptr
                        + batch_id * qpe_stride_b
                        + h_offs[None, :] * qpe_stride_h
                        + r_offs[:, None] * qpe_stride_r,
                        mask=h_mask[None, :],
                        other=0.0,
                    ).to(
                        tl.float32
                    )  # [ROPE_DIM, H_POW2]
                    acc += tl.dot(kpe_blk, qpe_t, allow_tf32=True)

                if HEAD_AGG_MEAN:
                    acc = tl.where(h_mask[None, :], acc, 0.0)
                    score = tl.sum(acc, axis=1) / num_heads
                else:
                    acc = tl.where(h_mask[None, :], acc, float("-inf"))
                    score = tl.max(acc, axis=1)

                out_score = tl.where(
                    valid,
                    score,
                    tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32),
                )
                tl.store(
                    out_ptr + batch_id * out_stride_b + tok_offs,
                    out_score,
                    mask=in_range,
                )


def absorbed_score_paged_fp8(
    v: torch.Tensor,
    latent_fp8: torch.Tensor,
    latent_scales: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    written: Optional[torch.Tensor] = None,
    *,
    block_size: int = 128,
    token_block: int = 128,
    workers: int = 128,
    num_warps: int = 2,
    num_stages: int = 2,
    head_agg: str = "max",
    out: Optional[torch.Tensor] = None,
    q_norm: Optional[torch.Tensor] = None,
    key_norm_cache: Optional[torch.Tensor] = None,
    cosine: bool = False,
    q_pe: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Paged absorbed score from the resident fp8 latent — GPU.

    ``token_block=128`` + ``num_warps=2`` + ``num_stages=2`` is the measured-best
    config (captured-replay sweep: 19.4 µs/call ≈ 15.2k µs/window at the bs=29/H=8/
    width=5120/seq=4608 op point, under the ~23.1k logical-score bucket). The MMA
    inner loop tiles ``[TOKEN_BLOCK, 128] @ [128, H_pow2]`` per latent block, so both
    ``TOKEN_BLOCK`` and the head tile are floored at 16 for tensor cores. Triton's
    default ``num_warps=4`` over-subscribes this shape (~2× slower) — hence the knob.

    Args:
        v: ``[bs, H, lora]`` fp32 — the per-head query projection ``v_h`` (from
            ``absorbed_latent.absorbed_latent_v``).
        latent_fp8: ``[max_tokens, lora]`` float8_e4m3fn — the resident nope latent.
        latent_scales: ``[max_tokens, lora//block_size]`` fp32 — per-block scales.
        req_pool_indices / req_to_token / seq_lens: paging, as in the production
            logical scorer.
        written: optional ``[max_tokens]`` bool; ``None`` treats all slots written.
        out: optional pre-allocated ``[bs_buf, max_seq_len]`` fp32 destination. When
            given, the kernel writes into ``out[:bs, :max_seq_len]`` (the graph-safe
            zero-alloc path) instead of a fresh ``torch.empty`` (the diagnostic path).

    Returns:
        ``[bs, max_seq_len]`` fp32 scores (``-inf`` on unwritten / out-of-range).
    """
    if not _HAS_TRITON:
        raise RuntimeError("absorbed_score_paged_fp8 requires Triton/CUDA")
    bs, num_heads, lora = v.shape
    max_tokens = latent_fp8.shape[0]
    assert lora % block_size == 0, f"lora {lora} not a multiple of block {block_size}"
    nblk = lora // block_size
    # BF16 resident path: the latent IS the dequantized k_nope, so there is no
    # per-128-block fp8 scale (latent_scales is a harmless unused dummy).
    bf16_latent = latent_fp8.dtype == torch.bfloat16
    if not bf16_latent:
        assert latent_scales.shape == (
            max_tokens,
            nblk,
        ), f"scales {tuple(latent_scales.shape)} != {(max_tokens, nblk)}"
    device = v.device
    if max_seq_len <= 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)
    # `written is None` means treat every slot as written — drive the kernel via the
    # HAS_WRITTEN=False constexpr instead of materializing a per-call ones tensor
    # (a per-call torch.ones would break the graph-safe zero-alloc contract).
    has_written = written is not None
    written_ptr = written if has_written else v

    if out is None:
        out = torch.empty((bs, max_seq_len), dtype=torch.float32, device=device)
    else:
        out = out[:bs, :max_seq_len]
    max_pool_len = int(req_to_token.shape[1])
    # tl.dot needs ≥16 on every tile dim; pad both the token tile and the head tile.
    desired_block = min(token_block, max(max_seq_len, 1))
    token_block_pow2 = max(16, _next_pow2(desired_block))
    h_pow2 = max(16, _next_pow2(num_heads))
    num_token_blocks = (max_seq_len + token_block_pow2 - 1) // token_block_pow2
    num_workers = max(1, min(int(workers), num_token_blocks))
    grid = (bs, num_workers)

    # Cosine division tensors. When cosine, q_norm is [bs, H] and key_norm_cache is
    # the layer's [max_tokens, H] resident key-norm cache (gathered by physical
    # slot). When off, pass v as a harmless non-null pointer (COSINE=False makes the
    # kernel never dereference it) and zero strides, so the raw-dot launch is
    # byte-identical.
    if cosine:
        assert q_norm is not None and key_norm_cache is not None, (
            "cosine absorbed score requires q_norm [bs, H] and key_norm_cache "
            "[max_tokens, H]; one is None."
        )
        q_norm_ptr = q_norm
        k_norm_ptr = key_norm_cache
        qn_stride_b = q_norm.stride(0)
        kn_stride_t = key_norm_cache.stride(0)
        kn_stride_h = key_norm_cache.stride(1)
    else:
        q_norm_ptr = v
        k_norm_ptr = v
        qn_stride_b = 0
        kn_stride_t = 0
        kn_stride_h = 0

    # RoPE term. When off, pass v as a harmless non-null pointer with zero strides —
    # HAS_ROPE=False makes the kernel never dereference it, so the no-rope launch is
    # byte-identical. q_pe is [bs, H, rope_dim] post-RoPE; k_pe is the resident
    # [max_tokens, rope_dim] bf16 RoPE key.
    # rope inputs are a PAIR: exactly-one-present is a wiring bug — fail closed rather
    # than silently launching no-PE (which would drop the rope term unnoticed).
    if (q_pe is None) != (k_pe is None):
        raise ValueError(
            "absorbed_score_paged_fp8: q_pe and k_pe must be provided together; got "
            f"q_pe={'set' if q_pe is not None else 'None'}, "
            f"k_pe={'set' if k_pe is not None else 'None'}."
        )
    has_rope = q_pe is not None and k_pe is not None
    if has_rope:
        rope_dim = int(q_pe.shape[-1])
        assert (
            rope_dim & (rope_dim - 1) == 0
        ), f"rope_dim {rope_dim} must be a power of 2"
        qpe_ptr, kpe_ptr = q_pe, k_pe
        qpe_stride_b = q_pe.stride(0)
        qpe_stride_h = q_pe.stride(1)
        qpe_stride_r = q_pe.stride(2)
        kpe_stride_t = k_pe.stride(0)
    else:
        rope_dim = 16  # unused (HAS_ROPE=False); a valid power-of-2 constexpr
        qpe_ptr = kpe_ptr = v
        qpe_stride_b = qpe_stride_h = qpe_stride_r = kpe_stride_t = 0

    # For bf16 the scale pointer is never dereferenced (BF16_LATENT branch); pass
    # the latent itself as a harmless non-null pointer when scales are absent.
    scale_arg = latent_fp8 if (bf16_latent or latent_scales is None) else latent_scales
    _absorbed_score_kernel[grid](
        v,
        latent_fp8,
        scale_arg,
        written_ptr,
        req_pool_indices,
        req_to_token,
        seq_lens,
        out,
        q_norm_ptr,
        k_norm_ptr,
        qpe_ptr,
        kpe_ptr,
        qpe_stride_b=qpe_stride_b,
        qpe_stride_h=qpe_stride_h,
        qpe_stride_r=qpe_stride_r,
        kpe_stride_t=kpe_stride_t,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        lora=lora,
        block_size=block_size,
        max_pool_len=max_pool_len,
        max_tokens=max_tokens,
        v_stride_b=v.stride(0),
        v_stride_h=v.stride(1),
        fp8_stride_t=latent_fp8.stride(0),
        scale_stride_t=(0 if bf16_latent else latent_scales.stride(0)),
        rtt_stride_p=req_to_token.stride(0),
        out_stride_b=out.stride(0),
        qn_stride_b=qn_stride_b,
        kn_stride_t=kn_stride_t,
        kn_stride_h=kn_stride_h,
        eps=eps,
        TOKEN_BLOCK=token_block_pow2,
        H_POW2=h_pow2,
        HEAD_AGG_MEAN=head_agg == "mean",
        STORE_DEAD_NEG_INF=True,
        HAS_WRITTEN=has_written,
        COSINE=cosine,
        BF16_LATENT=bf16_latent,
        HAS_ROPE=has_rope,
        ROPE_DIM=rope_dim,
        WORKERS=num_workers,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def absorbed_latent_score_logical_paged(
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
    written: Optional[torch.Tensor] = None,
    *,
    block_size: int = 128,
    token_block: int = 128,
    head_agg: str = "max",
    out: Optional[torch.Tensor] = None,
    scratch_v: Optional[torch.Tensor] = None,
    scratch_qsel: Optional[torch.Tensor] = None,
    channel_selection_i64: Optional[torch.Tensor] = None,
    scratch_q: Optional[torch.Tensor] = None,
    cosine: bool = False,
    key_norm_cache: Optional[torch.Tensor] = None,
    scratch_qnorm: Optional[torch.Tensor] = None,
    q_pe: Optional[torch.Tensor] = None,
    k_pe: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GPU equivalent of ``absorbed_latent.absorbed_latent_score_logical`` reading
    the paged fp8 latent. Builds ``v_h`` host-side then launches the paged kernel.

    Diagnostic path (default): ``v_h`` and the score are freshly allocated.
    Graph-safe path: ``scratch_v`` / ``scratch_qsel`` / ``channel_selection_i64``
    (the int64 layer mask) / ``scratch_q`` (the fp32 query cast scratch) build
    ``v_h`` allocation-free, and ``out`` receives the score in place — so after
    warmup the call grows no caching-allocator counter.
    """
    q_norm = None
    if (
        scratch_v is not None
        and scratch_qsel is not None
        and channel_selection_i64 is not None
        and scratch_q is not None
    ):
        from .absorbed_latent import absorbed_latent_v_into

        v = absorbed_latent_v_into(
            scratch_v,
            queries,
            w_sel,
            channel_selection_i64,
            channel_weights,
            scratch_qsel=scratch_qsel,
            scratch_q=scratch_q,
        )
        if cosine:
            # Cosine denominator's query side: q_norm = ||q_sel|| over the label-dim
            # axis. absorbed_latent_v_into just wrote q_sel (the weighted channel
            # gather) into scratch_qsel[:bs, :, :label_dim], so norm it in place
            # (out=) into scratch_qnorm — 0-alloc, exactly the reference's
            # q_sel.norm(dim=-1). The kernel reads q_norm during the same launch.
            assert scratch_qnorm is not None and key_norm_cache is not None, (
                "cosine graph-safe score requires scratch_qnorm and the layer's "
                "key_norm_cache; one is None."
            )
            bs = queries.shape[0]
            label_dim = int(channel_selection_i64.shape[1])
            q_norm = scratch_qnorm[:bs]
            torch.linalg.vector_norm(
                scratch_qsel[:bs, :, :label_dim], dim=-1, out=q_norm
            )
    else:
        from .absorbed_latent import absorbed_latent_v

        if cosine:
            raise NotImplementedError(
                "cosine absorbed score requires the graph-safe scratch path "
                "(scratch_v/scratch_qsel/scratch_q/channel_selection_i64); the "
                "eager fallback does not expose q_sel for the query norm."
            )
        v = absorbed_latent_v(queries, w_sel, channel_selection, channel_weights)
    return absorbed_score_paged_fp8(
        v,
        latent_fp8,
        latent_scales,
        req_pool_indices,
        req_to_token,
        seq_lens,
        max_seq_len,
        written=written,
        block_size=block_size,
        token_block=token_block,
        head_agg=head_agg,
        out=out,
        q_norm=q_norm,
        key_norm_cache=key_norm_cache,
        cosine=cosine,
        q_pe=q_pe,
        k_pe=k_pe,
        eps=eps,
    )
