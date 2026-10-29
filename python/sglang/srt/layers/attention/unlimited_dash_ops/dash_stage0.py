"""DASH Stage-0 summarization for ``head_cls`` checkpoints.

A masked softmax summarizes each KV page before it is scored against index
queries; both stages apply ``1 / sqrt(d)``, matching training.
"""

from __future__ import annotations

import math

import torch

from sglang.srt.layers.attention.unlimited_dash_ops.dash_router import SCORE_NEG

# Same activation bound training clamps with.
ACT_CLAMP = 1.0e4


def resolve_stage0(sparse_cfg) -> str:
    """Return the checkpoint's Stage-0 mode, defaulting to ``indexer``."""
    cfg = sparse_cfg.get("dash_stage0") if isinstance(sparse_cfg, dict) else None
    return str(cfg or "indexer").lower()


def _summary_keys(
    keys: torch.Tensor, valid: torch.Tensor, head_cls: torch.Tensor, inv_sqrt_d: float
) -> torch.Tensor:
    """[nb, bs, d] index keys -> [nb, d] head_cls summary keys.

    NaN-safe masked softmax matching training: row max over VALID tokens only,
    exp, multiplicative padding zeroing, clamped denominator.
    """
    aw = (keys * head_cls).sum(-1) * inv_sqrt_d  # [nb, bs]
    aw = aw.clamp(-ACT_CLAMP, ACT_CLAMP)
    neg = torch.finfo(aw.dtype).min
    row_max = torch.where(valid, aw, aw.new_full((), neg)).amax(dim=-1, keepdim=True)
    row_max = torch.where(
        valid.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max)
    )
    e = torch.exp(aw - row_max) * valid.to(aw.dtype)
    w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [nb, bs]
    k_bar = (w.unsqueeze(-1) * keys).sum(dim=1)  # [nb, d]
    return torch.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)


def _score_rows(
    q: torch.Tensor,
    k_bar: torch.Tensor,
    q_pos: torch.Tensor,
    nb: int,
    block_size: int,
    local_blocks: int,
    inv_sqrt_d: float,
    q_chunk: int,
) -> torch.Tensor:
    """[Lq, d] queries x [nb, d] summaries -> [Lq, nb] remote-only scores."""
    out = q.new_full((q.shape[0], nb), SCORE_NEG)
    blk = torch.arange(nb, device=q.device)
    q_blk = q_pos // block_size
    for c0 in range(0, q.shape[0], q_chunk):
        c1 = min(c0 + q_chunk, q.shape[0])
        z = torch.matmul(q[c0:c1], k_bar.transpose(0, 1)) * inv_sqrt_d
        z = z.clamp(-ACT_CLAMP, ACT_CLAMP)
        # chunk c is a remote candidate iff c <= qblk - local_blocks
        keep = blk[None, :] <= (q_blk[c0:c1, None] - local_blocks)
        out[c0:c1] = torch.where(keep, z, z.new_full((), SCORE_NEG))
    return out


@torch.no_grad()
def headcls_scores_prefill(
    *,
    idx_q,
    kv_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    context_lens,
    head_cls,
    max_block,
    block_size,
    local_blocks,
    q_chunk=2048,
    cache=None,
):
    """Prefill block scores [1, total_q, max_block] fp32 (SCORE_NEG elsewhere).

    ``idx_q`` [total_q, d] (single DASH router head), ``kv_cache``
    [num_pages, block_size, d]. When ``cache`` is given every summary computed
    here is stored in it, so later decode steps score off the cache instead of
    re-summarizing the whole KV.
    """
    device = idx_q.device
    d = idx_q.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(d)
    q = idx_q.float()
    hc = head_cls.to(device=device, dtype=torch.float32)
    sc = q.new_full((1, q.shape[0], max_block), SCORE_NEG)

    cu = cu_seqlens_q.tolist()
    lens = seq_lens.tolist()
    ctxs = context_lens.tolist()
    bt_w = int(block_table.shape[1])
    off = torch.arange(block_size, device=device)
    for b in range(len(lens)):
        q0, q1 = int(cu[b]), int(cu[b + 1])
        if q1 <= q0:
            continue
        kv_len = int(lens[b])
        nb = min(max_block, bt_w, (kv_len + block_size - 1) // block_size)
        if nb <= 0:
            continue
        pages = block_table[b, :nb].to(torch.int64)
        keys = kv_cache[pages].to(torch.float32)  # [nb, bs, d]
        pos = torch.arange(nb, device=device)[:, None] * block_size + off[None, :]
        k_bar = _summary_keys(keys, pos < kv_len, hc, inv_sqrt_d)
        if cache is not None:
            cache.put(pages, k_bar)
        q_pos = int(ctxs[b]) + torch.arange(q1 - q0, device=device)
        sc[0, q0:q1, :nb] = _score_rows(
            q[q0:q1],
            k_bar,
            q_pos,
            nb,
            block_size,
            local_blocks,
            inv_sqrt_d,
            q_chunk,
        )
    return sc


class KBarCache:
    """Cache one fp32 ``head_cls`` summary per KV page.

    Prefill initializes used pages; decode refreshes the mutable tail pages.
    Prefix caching is unsupported: a sequence could read an uninitialized
    shared-page summary.
    """

    def __init__(self, num_pages: int, dim: int, device, dtype=torch.float32):
        self.k_bar = torch.zeros(num_pages, dim, dtype=dtype, device=device)

    def put(self, pages: torch.Tensor, k_bar: torch.Tensor) -> None:
        """Write the stage-0 summary ``k_bar`` for the given ``pages``."""
        self.k_bar[pages] = k_bar.to(self.k_bar.dtype)

    @torch.no_grad()
    def refresh_tail(
        self,
        *,
        kv_cache,
        block_table,
        seq_lens,
        head_cls,
        block_size: int,
        tail_pages: int = 2,
    ) -> None:
        """Recompute the last ``tail_pages`` summaries of every request.

        Sync-free: page ids and token counts come from ``seq_lens`` on device,
        so this stays CUDA-graph friendly.
        """
        device = self.k_bar.device
        d = self.k_bar.shape[-1]
        sl = seq_lens.to(device=device, dtype=torch.int64)  # [R]
        nb = (sl + block_size - 1) // block_size  # pages in use
        last = (nb - 1).clamp(min=0)
        offs = torch.arange(tail_pages, device=device)  # 0,1,...
        blk = (last[:, None] - offs[None, :]).clamp(min=0)  # [R, T]
        w = int(block_table.shape[1])
        pages = (
            block_table[:, :w].to(torch.int64).gather(1, blk.clamp(max=w - 1))
        )  # [R, T]
        tok = (sl[:, None] - blk * block_size).clamp(0, block_size)  # [R, T]

        flat_pages = pages.reshape(-1)
        keys = kv_cache[flat_pages].to(torch.float32)  # [R*T, bs, d]
        valid = torch.arange(block_size, device=device)[None, :] < tok.reshape(
            -1, 1
        )  # [R*T, bs]
        k_bar = _summary_keys(
            keys,
            valid,
            head_cls.to(device=device, dtype=torch.float32),
            1.0 / math.sqrt(d),
        )
        self.k_bar[flat_pages] = k_bar


@torch.no_grad()
def headcls_scores_decode(
    *,
    idx_q,
    cache,
    block_table,
    seq_lens,
    max_block,
    block_size,
    local_blocks,
    decode_query_len,
    req_chunk=64,
):
    """Decode block scores [1, nd, max_block] fp32 (SCORE_NEG elsewhere).

    Scores straight off ``KBarCache`` (one dot per query/page, no key re-read),
    with no host sync on the decode path. Row layout is request-major,
    ``nd == R * dql``; ``req_chunk`` bounds the peak of the [R, W, d] gather.
    """
    device = idx_q.device
    d = idx_q.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(d)
    dql = max(1, int(decode_query_len))
    R = int(seq_lens.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    q = idx_q.float().view(R, dql, d)
    sl = seq_lens.to(device=device, dtype=torch.int64)
    blk = torch.arange(W, device=device)
    q_off = torch.arange(dql, device=device)
    out = q.new_full((1, R * dql, max_block), SCORE_NEG)

    for r0 in range(0, R, req_chunk):
        r1 = min(r0 + req_chunk, R)
        pages = block_table[r0:r1, :W].to(torch.int64)  # [r, W]
        kb = cache.k_bar[pages].to(torch.float32)  # [r, W, d]
        z = torch.bmm(q[r0:r1], kb.transpose(1, 2)) * inv_sqrt_d  # [r, dql, W]
        z = z.clamp(-ACT_CLAMP, ACT_CLAMP)
        # the dql query positions are the LAST dql tokens of the sequence
        pos = sl[r0:r1, None] - dql + q_off[None, :]  # [r, dql]
        q_blk = pos // block_size
        keep = blk[None, None, :] <= (q_blk[:, :, None] - local_blocks)
        keep &= (blk[None, None, :] * block_size) < sl[r0:r1, None, None]
        z = torch.where(keep, z, z.new_full((), SCORE_NEG))
        out[0, r0 * dql : r1 * dql, :W] = z.reshape(-1, W)
    return out
