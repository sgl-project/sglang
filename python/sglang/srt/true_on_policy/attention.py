"""Shared dense TOP attention forward for packed training and paged serving.

The vendored FA4 forward is intentional: its batch-invariant masking policy is
part of the numerical program. Packed training retains FA4 autograd through a
separate backward adapter; serving continues to use the same forward directly.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

# FA's combine kernel merges at most this many pieces.
MAX_DECODE_KV_PIECES = 256


# Decode reads one query row against its whole prefix. The shared TOP forward gives that row one sequential pass over
# its keys (TOP forces num_splits=1), which leaves most SMs idle on long contexts. Serving decode therefore splits the
# keys at fixed absolute positions (multiples of the split), runs the same forward on each piece, and merges the pieces
# with FA's combine kernel; the trainer scores decoded rows with the same split, so both sides
# keep one program. A row whose prefix fits in one piece is unchanged. The split is part of the contract of the
# architectures whose trainer reproduces it (the Qwen GDN programs); every other TOP family keeps the one-pass decode.
def decode_kv_split(model_type: str) -> int:
    """TOP decode KV split of a text architecture.

    SGLANG_TOP_DECODE_KV_SPLIT for the GDN TOP architectures whose full attention is FA4, else 0 (one pass);
    no other TOP trainer reproduces the split. Engine and trainer read the same variable.
    """
    from sglang.srt.true_on_policy.gdn_contract import (
        GDN_TOP_ARCHITECTURES,
        GDN_TOP_QSA_ARCHITECTURES,
    )

    if (
        model_type not in GDN_TOP_ARCHITECTURES
        or model_type in GDN_TOP_QSA_ARCHITECTURES
    ):
        return 0
    return envs.SGLANG_TOP_DECODE_KV_SPLIT.get()


def flash_attn_varlen_func(
    *, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
):
    # Defer optional CUDA/CuTe imports until the attention operation is used.
    from sglang.kernels.ops.attention.flash_attn.cute import (
        flash_attn_varlen_func as fa4_varlen,
    )
    from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import (
        set_batch_invariant,
    )

    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("TOP attention requires BF16 Q/K/V")
    if torch.cuda.get_device_capability(q.device)[0] not in (9, 10):
        raise NotImplementedError("TOP FA4 forward supports SM90 and SM10x")
    # The SM100 head-256 implementation requires dense Q/K/V strides. Apply
    # the same layout boundary to packed trainer inputs and paged serving.
    if q.shape[-1] == k.shape[-1] == v.shape[-1] == 256:
        q, k, v = (t.contiguous() for t in (q, k, v))
    set_batch_invariant(True)
    # Keep packing explicit on both sides; automatic splitting changes reductions.
    kwargs.update(causal=True, num_splits=1, pack_gqa=True)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        # Training-only dependency; ordinary serving needs no backward package.
        from sglang.srt.true_on_policy.attention_backward import dense_varlen_forward

        return dense_varlen_forward(q=q, k=k, v=v, **kwargs)
    return fa4_varlen(q=q, k=k, v=v, **kwargs)


def _forward(*, q, k, v, causal: bool, **kwargs):
    """The shared FA4 forward (no autograd) with an explicit causal flag, for key pieces a row sees in full."""
    from sglang.kernels.ops.attention.flash_attn.cute import (
        flash_attn_varlen_func as fa4_varlen,
    )
    from sglang.kernels.ops.attention.flash_attn.cute.batch_invariance import (
        set_batch_invariant,
    )

    if q.shape[-1] == k.shape[-1] == v.shape[-1] == 256:
        q, k, v = (t.contiguous() for t in (q, k, v))
    set_batch_invariant(True)
    kwargs.update(causal=causal, num_splits=1, pack_gqa=True, return_lse=True)
    with torch.no_grad():
        return fa4_varlen(q=q, k=k, v=v, **kwargs)


def noncausal_varlen_forward(*, q, k, v, cu_seqlens, max_seqlen, softmax_scale):
    """Packed self-attention where every row sees its whole sequence (a frozen vision tower), on the shared FA4
    forward. q/k/v [T, H, D] BF16; cu_seqlens [B + 1] int32 -> [T, H, D] BF16."""
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise ValueError("TOP attention requires BF16 Q/K/V")
    if torch.cuda.get_device_capability(q.device)[0] not in (9, 10):
        raise NotImplementedError("TOP FA4 forward supports SM90 and SM10x")
    return _forward(
        q=q,
        k=k,
        v=v,
        causal=False,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale,
    )[0]


def combine_kv_splits(
    out: torch.Tensor, lse: torch.Tensor, splits: int
) -> torch.Tensor:
    """Merge key pieces: ``out`` [n * splits, H, D] (each row's pieces consecutive, in key order), ``lse`` [H, n *
    splits] -> [n, H, D] BF16 via FA's combine kernel. A piece with lse -inf (and zero output) contributes nothing;
    the result does not depend on how many such pieces pad a row or on the other rows."""
    n = out.shape[0] // splits
    h, d = out.shape[1], out.shape[2]
    o = out.view(n, splits, h, d).transpose(0, 1).float().contiguous()
    l = lse.view(h, n, splits).permute(2, 1, 0).contiguous()
    return _combine_pieces(o, l)


def _combine_pieces(o: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """FA's combine over pieces laid out piece-major: ``o`` [splits, n, H, D] FP32, ``l`` [splits, n, H] -> [n, H, D]."""
    from sglang.kernels.ops.attention.flash_attn.cute.interface import (
        flash_attn_combine,
    )

    res = flash_attn_combine(
        o[:, :, None], l[:, :, None], out_dtype=torch.bfloat16, return_lse=False
    )
    res = res[0] if isinstance(res, tuple) else res
    return res[:, 0]


class SplitPrefillPlan:
    """Entries of a split prefill (``split_prefill_paged``) over a batch of extend rows, built once per forward.

    Row t of a request (absolute position) takes its keys in pieces at fixed positions, as its decode would
    (``decode_kv_split``): its own piece [s * j, t] causal (j = t // s) and every earlier piece [s * p, s * (p + 1)) in
    full. ``own`` entries hold, per request and piece, the request's new rows inside the piece; ``full`` entries the
    new rows past the piece's end."""

    def __init__(self, *, prefix, lens, split, page_size, page_table, device):
        per = split // page_size
        own_cu, own_used, own_pages, own_piece = [0], [], [], []
        full_rows, full_cu, full_pages, full_piece = [], [0], [], []
        n = 1
        row0 = 0
        for i, (p0, m) in enumerate(zip(prefix, lens)):
            end = p0 + m
            last = (end - 1) // split
            n = max(n, last + 1)
            for j in range(p0 // split, last + 1):
                a, b = max(p0, j * split), min(end, (j + 1) * split)
                own_cu.append(own_cu[-1] + b - a)
                own_used.append(b - j * split)
                own_pages.append((i, j * per))
                own_piece.extend([j] * (b - a))
            for p in range(last):
                a = max(p0, (p + 1) * split)
                full_rows.extend(range(row0 + a - p0, row0 + m))
                full_cu.append(full_cu[-1] + end - a)
                full_pages.append((i, p * per))
                full_piece.extend([p] * (end - a))
            row0 += m
        self.pieces = n
        self.rows = row0

        def ints(values, dtype=torch.int32):
            from sglang.srt.utils import is_pin_memory_available

            host = torch.tensor(values, dtype=dtype)
            if is_pin_memory_available(device):
                host = host.pin_memory()
            return host.to(device, non_blocking=True)

        width = page_table.shape[1]
        cols = torch.arange(per, device=device)

        def pages(entries):
            req = ints([i for i, _ in entries], torch.int64)
            start = ints([c for _, c in entries], torch.int64)
            idx = (start[:, None] + cols[None, :]).clamp_(max=width - 1)
            return page_table.index_select(0, req).gather(1, idx).to(torch.int32)

        self.own = dict(
            cu=ints(own_cu),
            used=ints(own_used),
            pages=pages(own_pages),
            max_q=max(b - a for a, b in zip(own_cu, own_cu[1:])),
            piece=ints(own_piece, torch.int64),
        )
        self.full = None
        if full_rows:
            self.full = dict(
                rows=ints(full_rows, torch.int64),
                cu=ints(full_cu),
                pages=pages(full_pages),
                max_q=max(b - a for a, b in zip(full_cu, full_cu[1:])),
                piece=ints(full_piece, torch.int64),
                split=split,
            )


def split_prefill_paged(q, key_cache, value_cache, plan: SplitPrefillPlan, scale):
    """Prefill attention with every row's keys split at fixed positions, the arithmetic of the split decode
    (one TOP FA4 forward per piece with num_splits 1, FA's combine): a row's bytes are those of its decode.
    q [T, H, D]; paged caches as the TOP decode reads them. Returns [T, H, D] BF16."""
    T, h, d = q.shape
    own_o, own_l = _forward(
        q=q,
        k=key_cache,
        v=value_cache,
        causal=True,
        cu_seqlens_q=plan.own["cu"],
        max_seqlen_q=plan.own["max_q"],
        page_table=plan.own["pages"],
        seqused_k=plan.own["used"],
        softmax_scale=scale,
    )[:2]
    n = plan.pieces
    if n == 1:
        return combine_kv_splits(own_o, own_l, 1)
    out = own_o.new_zeros(T, n, h, d)
    lse = own_l.new_full((h, T, n), float("-inf"))
    rows = torch.arange(T, device=q.device)
    out[rows, plan.own["piece"]] = own_o
    lse[:, rows, plan.own["piece"]] = own_l
    full = plan.full
    if full is not None:
        used = torch.full_like(full["cu"][1:], full["split"])
        full_o, full_l = _forward(
            q=q.index_select(0, full["rows"]),
            k=key_cache,
            v=value_cache,
            causal=False,
            cu_seqlens_q=full["cu"],
            max_seqlen_q=full["max_q"],
            page_table=full["pages"],
            seqused_k=used,
            softmax_scale=scale,
        )[:2]
        out[full["rows"], full["piece"]] = full_o
        lse[:, full["rows"], full["piece"]] = full_l
    return combine_kv_splits(out.view(T * n, h, d), lse.reshape(h, T * n), n)


def split_decode_paged(
    q, key_cache, value_cache, page_table, cache_seqlens, split, pieces, scale
):
    """Reference for tests; serving runs ``split_decode``.

    TOP decode with the keys split at fixed positions: one entry per request and piece (``pieces`` follow the
    page table's width, so shapes are static under CUDA graphs), the shared FA4 forward over each piece's pages,
    FA's combine across pieces. q [B, H, D]; caches [pages, page size, KV heads, D] -> [B, H, D] BF16."""
    per = split // key_cache.shape[1]
    b = q.shape[0]
    pages = torch.nn.functional.pad(page_table, (0, pieces * per - page_table.shape[1]))
    pages = pages.view(b * pieces, per)
    starts = torch.arange(pieces, device=q.device, dtype=torch.int32) * split
    used = (
        (cache_seqlens.view(b, 1).to(torch.int32) - starts).clamp_(0, split).reshape(-1)
    )
    # A dense [entries, 1] query, not varlen: FA4's varlen scheduler scans every entry per tile, so the mostly
    # empty pieces of a full-width table cost quadratically; the dense scheduler maps tiles in O(1). Same bytes.
    out, lse = flash_attn_varlen_func(
        q=q.repeat_interleave(pieces, dim=0).unsqueeze(1),
        k=key_cache,
        v=value_cache,
        page_table=pages,
        seqused_k=used,
        softmax_scale=scale,
        return_lse=True,
    )[:2]
    out, lse = out.squeeze(1), lse.squeeze(-1).transpose(0, 1)
    empty = used == 0
    out = out.masked_fill(empty.view(-1, 1, 1), 0)
    lse = lse.masked_fill(empty.view(1, -1), float("-inf"))
    return combine_kv_splits(out, lse, pieces)


def _piece_major_pages(page_table, pieces, per):
    """[B, width] page table -> [pieces * B, per]: entry p * B + i holds request i's pages of piece p."""
    b = page_table.shape[0]
    pages = torch.nn.functional.pad(page_table, (0, pieces * per - page_table.shape[1]))
    return pages.view(b, pieces, per).transpose(0, 1).reshape(pieces * b, per)


class SplitDecodePlan:
    """Entries of a split decode (``split_decode``) for one forward: the same tensors for every attention layer, so a
    forward builds it once (inside its CUDA graph when captured). Entries are piece-major (entry p * B + i is piece p of
    request i), the layout FA's combine reads, so the per-piece outputs need no transpose."""

    def __init__(self, *, page_table, cache_seqlens, split, pieces, page_size):
        per = split // page_size
        self.batch, self.pieces = page_table.shape[0], pieces
        self.pages = _piece_major_pages(page_table, pieces, per)
        starts = (
            torch.arange(pieces, device=page_table.device, dtype=torch.int32) * split
        )
        self.used = (
            (cache_seqlens.view(1, -1).to(torch.int32) - starts.view(-1, 1))
            .clamp_(0, split)
            .view(-1)
        )


def split_decode(q, key_cache, value_cache, plan: SplitDecodePlan, scale):
    """``split_decode_paged`` bit for bit, with the plan built once per forward. FA4 writes an empty piece (no key) as
    output 0 and LSE -inf, the values the reference masks in. q [B, H, D] -> [B, H, D] BF16."""
    b, h, d = q.shape
    n = plan.pieces
    out, lse = flash_attn_varlen_func(
        q=q.unsqueeze(0).expand(n, b, h, d).reshape(n * b, 1, h, d),
        k=key_cache,
        v=value_cache,
        page_table=plan.pages,
        seqused_k=plan.used,
        softmax_scale=scale,
        return_lse=True,
    )[:2]
    return _combine_pieces(out.view(n, b, h, d).float(), lse.view(n, b, h))


class SplitVerifyPlan:
    """Entries of a split chain verify (``split_verify``) for one forward, built once for every attention layer.

    Row j of request i sits at position t = cache_seqlens[i] - d + j, so a request's d rows are a d-row extend and take
    ``split_prefill_paged``'s arithmetic: every piece before a row's own one in full (one non-causal entry per request
    and piece, all d rows as queries) and the row's own piece causal (one entry per request for the first rows' piece,
    one for the next, since d <= split rows span at most two). A row's own piece result then replaces that piece's
    full-entry slot, and every piece a row does not take is an empty entry (0, -inf), as in its per-row split."""

    def __init__(
        self, *, page_table, cache_seqlens, rows_per_request, split, pieces, page_size
    ):
        d = rows_per_request
        if d > split:
            raise ValueError(
                f"[top] {d} verify rows exceed the decode KV split {split}"
            )
        per = split // page_size
        b = page_table.shape[0]
        dev = page_table.device
        self.batch, self.rows_per_request, self.pieces = b, d, pieces
        r = b * d
        pos = (
            cache_seqlens.view(b, 1).to(torch.int32)
            - d
            + torch.arange(d, device=dev, dtype=torch.int32)
        )
        own = torch.div(pos, split, rounding_mode="floor")  # -1 for a row with no key
        first = own[:, 0].clamp(min=0)
        # Own entry 0 (slot 0): rows 0..c0-1 bottom-right causal over piece `first`, rows with no key seeing none.
        c0 = (own <= first.view(b, 1)).sum(1, dtype=torch.int32)
        end0 = pos.gather(1, (c0 - 1).view(b, 1).long()).view(b) + 1
        # Own entry 1 (slot 1): all d rows over piece first + 1; rows before it see no key there.
        end1 = pos[:, -1] + 1
        self.own_used_q = torch.cat((c0, torch.full_like(c0, d)))
        self.own_used_k = torch.cat(
            (
                (end0 - first * split).clamp(0, split),
                (end1 - (first + 1) * split).clamp(0, split),
            )
        )
        pages = torch.nn.functional.pad(
            page_table, (0, pieces * per - page_table.shape[1])
        ).view(b, pieces, per)
        own_piece = torch.stack((first, (first + 1).clamp(max=pieces - 1))).long()
        self.own_pages = (
            pages.gather(
                1, own_piece.transpose(0, 1).reshape(b, 2, 1).expand(b, 2, per)
            )
            .transpose(0, 1)
            .reshape(2 * b, per)
        )
        self.full_pages = _piece_major_pages(page_table, pieces, per)
        piece = torch.arange(pieces, device=dev, dtype=torch.int32)
        self.full_used_k = (
            torch.where(piece.view(-1, 1) < own[:, -1].view(1, -1), split, 0)
            .to(torch.int32)
            .view(-1)
        )
        # Row (i, j) -> its own piece's slot in the piece-major full output, and its row in the slot-major own output.
        # A row with no key maps to piece 0 and slot 1, both empty for its request.
        row = torch.arange(r, device=dev)
        slot = (own != first.view(b, 1)).view(-1).long()
        self.dst = own.clamp(min=0).view(-1).long() * r + row
        self.src = slot * r + row
        self.take_own = (piece.view(-1, 1) == own.view(1, -1)).view(pieces, b, d, 1)


def split_verify(q, key_cache, value_cache, plan: SplitVerifyPlan, scale):
    """Every chain-verify row through its own decode's key split (``split_decode_paged``), bit for bit, with each
    request's pieces read once for its d rows. q [B * d, H, D] -> [B * d, H, D] BF16."""
    r, h, hd = q.shape
    b, d, n = plan.batch, plan.rows_per_request, plan.pieces
    qq = q.view(1, b, d, h, hd).expand(n + 2, b, d, h, hd).contiguous()
    full_o, full_l = _forward(
        q=qq[:n].view(n * b, d, h, hd),
        k=key_cache,
        v=value_cache,
        causal=False,
        page_table=plan.full_pages,
        seqused_k=plan.full_used_k,
        softmax_scale=scale,
    )[:2]
    own_o, own_l = _forward(
        q=qq[n:].view(2 * b, d, h, hd),
        k=key_cache,
        v=value_cache,
        causal=True,
        page_table=plan.own_pages,
        seqused_q=plan.own_used_q,
        seqused_k=plan.own_used_k,
        softmax_scale=scale,
    )[:2]
    o = full_o.view(n * r, h, hd)
    o.index_copy_(0, plan.dst, own_o.view(2 * r, h, hd).index_select(0, plan.src))
    own_rows_l = own_l.view(2, b, h, d).transpose(2, 3).reshape(2 * r, h)[plan.src]
    l = torch.where(
        plan.take_own,
        own_rows_l.view(1, b, d, h),
        full_l.view(n, b, h, d).transpose(2, 3),
    )
    return _combine_pieces(o.view(n, r, h, hd).float(), l.reshape(n, r, h))
