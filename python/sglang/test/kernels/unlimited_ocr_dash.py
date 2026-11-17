"""PyTorch references for the Unlimited-OCR v2 DASH attention kernels."""

import math

import torch

# Same finite sentinels as the kernels and training.
SCORE_NEG = -1.0e9
ACT_CLAMP = 1.0e4


def sparse_attn_reference(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    blk_bias: torch.Tensor | None,
    block_table: torch.Tensor,
    query_pos: list[int],
    row_req: list[int],
    local_blocks: int,
    block_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """fp32 ``[T, H, D]`` bias-aware block-sparse GQA attention.

    A selected block contributes its keys at or before ``pos - local_blocks *
    block_size`` with the slot bias; the unbiased local window holds the last
    ``local_blocks * block_size`` keys up to the query.
    """
    num_tokens, num_heads, _ = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads
    window = local_blocks * block_size
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for t in range(num_tokens):
        pos, req = query_pos[t], row_req[t]
        for g in range(num_kv_heads):
            key_pos, bias = [], []
            slots = topk_idx[g, t].tolist()
            for j, blk in enumerate(slots[: sum(b >= 0 for b in slots)]):
                for p in range(blk * block_size, (blk + 1) * block_size):
                    if p <= pos - window:
                        key_pos.append(p)
                        bias.append(
                            0.0 if blk_bias is None else float(blk_bias[g, t, j])
                        )
            if window > 0:
                for p in range(max(0, pos - window + 1), pos + 1):
                    key_pos.append(p)
                    bias.append(0.0)
            if not key_pos:
                continue
            kp = torch.tensor(key_pos, device=q.device)
            pages = block_table[req].long()[kp // block_size]
            keys = k_cache[pages, kp % block_size, g].float()
            vals = v_cache[pages, kp % block_size, g].float()
            qg = q[t, g * group : (g + 1) * group].float()
            logits = qg @ keys.T * sm_scale + torch.tensor(bias, device=q.device)
            out[t, g * group : (g + 1) * group] = logits.softmax(-1) @ vals
    return out


def block_scores_reference(
    *,
    idx_q: torch.Tensor,
    k_bar: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    row_req: torch.Tensor,
    query_pos: torch.Tensor,
    width: int,
    block_size: int,
    local_blocks: int,
) -> torch.Tensor:
    """fp32 ``[N, width]`` remote block scores, ``SCORE_NEG`` where not a candidate."""
    d = idx_q.shape[-1]
    pages = block_table[row_req, :width].long()
    z = (k_bar[pages].float() * idx_q.float()[:, None, :]).sum(-1) / math.sqrt(d)
    z = z.clamp(-ACT_CLAMP, ACT_CLAMP)
    blk = torch.arange(width, device=idx_q.device)[None, :]
    keep = (blk <= (query_pos // block_size - local_blocks)[:, None]) & (
        blk * block_size < seq_lens[row_req][:, None]
    )
    return torch.where(keep, z, torch.full_like(z, SCORE_NEG))


def _entmax15_threshold(z: torch.Tensor) -> torch.Tensor:
    """Exact tau with ``sum(clamp(z - tau, 0) ** 2) == 1`` (sort-based)."""
    zs, _ = z.sort(descending=True)
    rho = torch.arange(1, z.numel() + 1, dtype=z.dtype, device=z.device)
    mean = zs.cumsum(0) / rho
    mean_sq = (zs**2).cumsum(0) / rho
    delta = (1 - rho * (mean_sq - mean**2)) / rho
    tau = mean - delta.clamp(min=0).sqrt()
    support = int((tau <= zs).sum().clamp(min=1))
    return tau[support - 1]


def entmax_route_reference(
    *,
    scores: torch.Tensor,
    image_allow: torch.Tensor | None,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    out_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: 1.5-entmax prior over candidates, hard top-``budget`` by score,
    centered log-prior bias; selected blocks packed ascending, ``-1`` / ``0`` after.
    """
    rows = scores.shape[0]
    idx = torch.full((rows, out_slots), -1, dtype=torch.int32)
    bias = torch.zeros(rows, out_slots, dtype=torch.float32)
    for r in range(rows):
        s = scores[r].double().cpu()
        cand = s > SCORE_NEG * 0.5
        if image_allow is not None:
            with_img = cand & image_allow[r].bool().cpu()
            if with_img.any():
                cand = with_img
        cols = cand.nonzero().flatten()
        if cols.numel() == 0:
            continue
        x = s[cols] * idx_scale
        z = (x - x.max()) * 0.5
        prior = (z - _entmax15_threshold(z)).clamp(min=0) ** 2
        keep = z.argsort(descending=True)[:budget]
        keep = keep[cols[keep].argsort()]
        logp = prior[keep].clamp(min=prior_floor).log()
        n = min(keep.numel(), out_slots)
        idx[r, :n] = cols[keep][:n].to(torch.int32)
        bias[r, :n] = ((logp - logp.mean()) / sigma)[:n].float()
    return idx, bias


def win_bounds_reference(
    *,
    allow: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per decode row: ``[lo, hi)`` span of allowed live remote blocks and ``fit``."""
    lo, hi = [], []
    for row in range(allow.shape[0]):
        seq = int(seq_lens[row // decode_query_len])
        q_blk = (seq - decode_query_len + row % decode_query_len) // block_size
        live = [
            w
            for w in range(min(width, allow.shape[1]))
            if allow[row, w] and w <= q_blk - local_blocks and w * block_size < seq
        ]
        if live:
            lo.append(min(live))
            hi.append(max(live) + 1)
        elif seq <= 0 or q_blk - local_blocks < 0:
            lo.append(0)
            hi.append(0)
        else:
            lo.append(0)
            hi.append(min(width, max(-(-seq // block_size), 1)))
    lo, hi = torch.tensor(lo, dtype=torch.int32), torch.tensor(hi, dtype=torch.int32)
    return lo, hi, ((hi - lo) <= tile).to(torch.int8)


def kbar_summary_reference(
    *,
    kv_cache: torch.Tensor,
    pages: torch.Tensor,
    toks: torch.Tensor,
    head_cls: torch.Tensor,
) -> torch.Tensor:
    """fp32 ``[N, d]`` head_cls softmax summary of the first ``toks`` keys of each page."""
    keys = kv_cache[pages.long()].float()
    d = keys.shape[-1]
    valid = torch.arange(keys.shape[1], device=keys.device)[None, :] < toks[:, None]
    aw = ((keys * head_cls.float()).sum(-1) / math.sqrt(d)).clamp(-ACT_CLAMP, ACT_CLAMP)
    row_max = torch.where(valid, aw, torch.full_like(aw, -math.inf)).amax(-1, True)
    row_max = torch.where(valid.any(-1, True), row_max, torch.zeros_like(row_max))
    e = torch.exp(aw - row_max) * valid
    w = e / e.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num((w[..., None] * keys).sum(1), nan=0.0)


def local_union_reference(
    *,
    allow: torch.Tensor,
    seq_lens: torch.Tensor,
    max_block: int,
    block_size: int,
    local_blocks: int,
) -> torch.Tensor:
    """int8 ``[R, max_block]``: ``allow`` plus each request's last ``local_blocks``
    blocks, clamped to the table width."""
    out = torch.zeros(allow.shape[0], max_block, dtype=torch.int8)
    width = min(max_block, allow.shape[1])
    out[:, :width] = (allow[:, :width] != 0).to(torch.int8).cpu()
    for r, seq in enumerate(seq_lens.tolist()):
        q_blk = (seq - 1) // block_size
        for j in range(local_blocks):
            out[r, min(max(q_blk - j, 0), max_block - 1)] = 1
    return out


def cos_sin_2d_reference(
    *,
    inv_tok: torch.Tensor,
    inv_page: torch.Tensor,
    tok_pos: torch.Tensor,
    page_pos: torch.Tensor,
    page_mod: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, 2 * (n_tok + n_page)]`` cos/sin of the token- then page-axis angles."""
    if page_mod > 0:
        page_pos = torch.remainder(page_pos.long(), page_mod)
    ang = torch.cat(
        (tok_pos.float()[:, None] * inv_tok, page_pos.float()[:, None] * inv_page), -1
    )
    emb = torch.cat((ang, ang), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def page2d_rope_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """NeoX rotate-half RoPE on ``[T, H * head_dim]``, multiplied in ``x.dtype``."""
    v = x.view(x.shape[0], -1, head_dim)
    half = head_dim // 2
    rot = torch.cat((-v[..., half:], v[..., :half]), dim=-1)
    return (v * cos[:, None, :] + rot * sin[:, None, :]).view_as(x)


def torch_decode_page_meta(
    *,
    page_state,
    coord_state,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_slots: int,
    max_block: int,
    num_blocks: int,
):
    """One-token decode step through the PyTorch page / coordinate state path.

    Returns ``(q_page, allow, (tok_pos, page_pos))``.
    """
    from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
        BatchLayout,
    )

    num_reqs = int(seq_lens.shape[0])
    rows = torch.arange(num_reqs, device=seq_lens.device, dtype=torch.int64)
    layout = BatchLayout(
        num_tokens=num_reqs,
        num_reqs=num_reqs,
        q_start=torch.arange(num_reqs + 1, device=seq_lens.device, dtype=torch.int64),
        start_pos=seq_lens.to(torch.int64) - 1,
        block_table=block_table,
        req_of_token=rows,
        num_decode_tokens=num_reqs,
    )
    q_page = page_state.step(input_ids, layout, slot_mapping, num_slots).clone()
    allow = page_state.build_query_allow(layout, q_page, max_block, num_blocks).clone()
    return q_page, allow, coord_state.step(input_ids, layout)
