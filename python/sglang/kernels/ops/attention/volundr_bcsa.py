"""Fused BCSA kernels for Volundr decode: the block-key write and the far-field branch.

BCSA (models/volundr.py VolundrAttention) merges a sliding window with a mean-pooled, indexer-selected
far field under one softmax. At decode every query has one absolute position, so the far branch has
static shapes; its index arithmetic and gathers are fused here, while its three einsums (cuBLAS),
torch.topk (tie-breaking must not move) and the merge stay torch. Every kernel is bit-exact with the
torch reference beside it (ref_*): FMA contraction is off and every bf16 intermediate torch
materialises is rounded at the same point.
"""

import torch
import triton
import triton.language as tl


def merge_local_far(sf, lse, o_loc, vb, num_kv_heads: int, kv_first: bool):
    """One softmax over [local | selected far], given the local branch's natural-log LSE.

    sf: [t, H, n] far scores (-inf where a block is not selected), fp32.
    vb: pooled far values, [n, KV, D] (shared, extend) or [t, n, KV, D] (per query, decode).
    Returns [t, H, D] fp32.
    """
    t, H, n = sf.shape
    D = o_loc.shape[-1]
    G = H // num_kv_heads
    m = torch.maximum(lse, sf.amax(-1))  # [t, H]
    w = torch.exp(lse - m)
    pf = torch.exp(sf - m[..., None])
    z = w + pf.sum(-1)
    pf = (pf / z[..., None]).to(o_loc.dtype).view(t, num_kv_heads, G, n)
    if kv_first:
        o_far = torch.einsum("tkgn,nkd->tkgd", pf, vb)
    else:
        o_far = torch.einsum("tkgn,tnkd->tkgd", pf, vb)
    return (w / z)[..., None] * o_loc.float() + o_far.reshape(t, H, D).float()


# ---------------------------------------------------------------- BCSA decode block-key write
@triton.jit
def _bcsa_block_keys_decode_kernel(
    POS_ptr,
    REQ_ptr,
    LOC_ptr,
    R2T_ptr,
    r2t_w,
    IDX_ptr,
    BLK_ptr,
    hi,
    C: tl.constexpr,
    DI: tl.constexpr,
):
    """One request per program: pool the C indexer keys of the block holding this step's position and
    write the mean to blk_cache at this step's KV slot if the position closes the block, else to the
    trash row `hi` — VolundrAttention._write_block_keys (decode branch), op for op."""
    b = tl.program_id(0)
    pos = tl.load(POS_ptr + b).to(tl.int64)
    req = tl.load(REQ_ptr + b).to(tl.int64)
    base = (pos // C) * C
    d = tl.arange(0, DI)
    acc = tl.zeros([DI], dtype=tl.float32)
    for c in tl.static_range(C):
        col = tl.minimum(tl.maximum(base + c, 0), r2t_w - 1)
        slot = tl.load(R2T_ptr + req * r2t_w + col).to(tl.int64)
        slot = tl.minimum(tl.maximum(slot, 0), hi)
        acc += tl.load(IDX_ptr + slot * DI + d).to(
            tl.float32
        )  # serial fp32 sum, c = 0..C-1
    pooled = (acc * (1.0 / C)).to(tl.bfloat16)
    closes = (pos % C) == (C - 1)
    dst = tl.where(closes, tl.load(LOC_ptr + b).to(tl.int64), hi)
    tl.store(BLK_ptr + dst * DI + d, pooled)


def bcsa_block_keys_decode(
    positions, req_pool_indices, out_cache_loc, r2t, idx_cache, blk_cache, C
):
    bs = positions.shape[0]
    if bs == 0:
        return
    DI = idx_cache.shape[1]
    assert (
        idx_cache.is_contiguous() and blk_cache.is_contiguous() and r2t.is_contiguous()
    )
    assert (C & (C - 1)) == 0, "the 1/C scale must be exact (C a power of two)"
    _bcsa_block_keys_decode_kernel[(bs,)](
        positions,
        req_pool_indices,
        out_cache_loc,
        r2t,
        r2t.shape[1],
        idx_cache,
        blk_cache,
        idx_cache.shape[0] - 1,
        C=C,
        DI=DI,
        num_warps=1,
        enable_fp_fusion=False,
    )


def ref_block_keys_decode(
    positions, req_pool_indices, out_cache_loc, r2t, idx_cache, blk_cache, C
):
    """Torch reference of bcsa_block_keys_decode."""
    Di = idx_cache.shape[1]
    hi = int(idx_cache.shape[0]) - 1
    bs = positions.shape[0]
    pos = positions.long()
    reqs = req_pool_indices.long()
    cols = (pos // C * C)[:, None] + torch.arange(C, device=pos.device)
    cols.clamp_(0, r2t.shape[1] - 1)
    slots = r2t.view(-1)[reqs[:, None] * r2t.shape[1] + cols].long().clamp_(0, hi)
    pooled = idx_cache[slots].view(bs, C, Di).mean(1)
    closes = (pos % C) == (C - 1)
    dst = torch.where(closes, out_cache_loc.long(), torch.full_like(pos, hi))
    blk_cache.index_copy_(0, dst, pooled.to(blk_cache.dtype))


# ---------------------------------------------------------------- BCSA decode far branch
# ref_far_sparse below is the torch definition. The fused path keeps its three cuBLAS einsums, its torch.topk (tie-breaking at the 512th
# block must not move) and its merge (a 512-wide fp32 softmax sum); the index arithmetic and gathers
# around them -- ~30 small kernels per BCSA layer -- become four Triton kernels, each bit-exact.
@triton.jit
def _far_kbi_kernel(
    POS_ptr,
    REQ_ptr,
    R2T_ptr,
    r2t_w,
    BLK_ptr,
    KBI_ptr,
    VALID_ptr,
    NB,
    hi,
    W,
    C: tl.constexpr,
    DI: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """kbi[b, n] = blk_cache[r2t[req, n*C + C-1]] for every far-valid block n, zeros elsewhere (an
    invalid block's score is overwritten with -inf, so its key row never reaches the output);
    valid[b, n] = (n+1)*C - 1 <= pos - W."""
    b = tl.program_id(0)
    pn = tl.program_id(1)
    pos = tl.load(POS_ptr + b).to(tl.int64)
    req = tl.load(REQ_ptr + b).to(tl.int64)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    inb = n < NB
    num = pos - W - (C - 1)
    nmax = tl.where(num >= 0, num // C, -1)  # floor division for negative numerators
    valid = (n.to(tl.int64) <= nmax) & inb
    slot = tl.load(
        R2T_ptr + req * r2t_w + n.to(tl.int64) * C + (C - 1), mask=valid, other=0
    ).to(tl.int64)
    slot = tl.minimum(tl.maximum(slot, 0), hi)
    d = tl.arange(0, DI)
    rows = tl.load(
        BLK_ptr + slot[:, None] * DI + d[None, :], mask=valid[:, None], other=0.0
    )
    tl.store(
        KBI_ptr + (b * NB + n.to(tl.int64))[:, None] * DI + d[None, :],
        rows,
        mask=inb[:, None],
    )
    tl.store(VALID_ptr + b * NB + n, valid.to(tl.int8), mask=inb)


@triton.jit
def _far_score_kernel(E_ptr, VALID_ptr, S_ptr, NB, scale, BLOCK_N: tl.constexpr):
    """s_idx = (e.mean(1) * scale).masked_fill(~valid, -inf) for e = einsum("bhd,bnd->bhn") [bs, 4, NB]
    bf16: serial fp32 head sum, x0.25, rounded; x scale in fp32, rounded."""
    b = tl.program_id(0)
    pn = tl.program_id(1)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    inb = n < NB
    base = E_ptr + b * 4 * NB + n
    e0 = tl.load(base, mask=inb, other=0.0).to(tl.float32)
    e1 = tl.load(base + NB, mask=inb, other=0.0).to(tl.float32)
    e2 = tl.load(base + 2 * NB, mask=inb, other=0.0).to(tl.float32)
    e3 = tl.load(base + 3 * NB, mask=inb, other=0.0).to(tl.float32)
    m = ((((e0 + e1) + e2) + e3) * 0.25).to(tl.bfloat16).to(tl.float32)
    s = (m * scale).to(tl.bfloat16)
    v = tl.load(VALID_ptr + b * NB + n, mask=inb, other=0)
    s = tl.where(v != 0, s, float("-inf"))
    tl.store(S_ptr + b * NB + n, s, mask=inb)


@triton.jit
def _far_pool_kernel(
    TOP_ptr,
    POS_ptr,
    REQ_ptr,
    R2T_ptr,
    r2t_w,
    K_ptr,
    V_ptr,
    KB_ptr,
    VB_ptr,
    OK_ptr,
    KSEL,
    hi,
    W,
    C: tl.constexpr,
    ROW: tl.constexpr,
):
    """For selected block t = top[b, j]: kb/vb[b, j] = mean of the C K/V cache rows of block t (serial
    fp32 sum, x 1/C, rounded) and sel_ok[b, j] = block t is far-valid."""
    b = tl.program_id(0)
    j = tl.program_id(1)
    t = tl.load(TOP_ptr + b * KSEL + j).to(tl.int64)
    pos = tl.load(POS_ptr + b).to(tl.int64)
    req = tl.load(REQ_ptr + b).to(tl.int64)
    num = pos - W - (C - 1)
    nmax = tl.where(num >= 0, num // C, -1)
    tl.store(OK_ptr + b * KSEL + j, (t <= nmax).to(tl.int8))
    r = tl.arange(0, ROW)
    acck = tl.zeros([ROW], dtype=tl.float32)
    accv = tl.zeros([ROW], dtype=tl.float32)
    for c in tl.static_range(C):
        col = tl.minimum(tl.maximum(t * C + c, 0), r2t_w - 1)
        slot = tl.load(R2T_ptr + req * r2t_w + col).to(tl.int64)
        slot = tl.minimum(tl.maximum(slot, 0), hi)
        acck += tl.load(K_ptr + slot * ROW + r).to(tl.float32)
        accv += tl.load(V_ptr + slot * ROW + r).to(tl.float32)
    o = (b * KSEL + j) * ROW + r
    tl.store(KB_ptr + o, (acck * (1.0 / C)).to(tl.bfloat16))
    tl.store(VB_ptr + o, (accv * (1.0 / C)).to(tl.bfloat16))


@triton.jit
def _far_sf_kernel(E_ptr, OK_ptr, SF_ptr, H, KSEL, scale, BLOCK: tl.constexpr):
    """sf = (e * scale).reshape(bs, H, ksel).float().masked_fill(~sel_ok[:, None, :], -inf), e =
    einsum("bkgd,bnkd->bkgn") bf16 [bs, KV, G, ksel] (contiguous, so [bs, H, ksel])."""
    b = tl.program_id(0)
    hp = tl.program_id(1)
    j = tl.arange(0, BLOCK)
    inb = j < KSEL
    e = tl.load(E_ptr + (b * H + hp) * KSEL + j, mask=inb, other=0.0).to(tl.float32)
    s = (e * scale).to(tl.bfloat16).to(tl.float32)
    ok = tl.load(OK_ptr + b * KSEL + j, mask=inb, other=0)
    tl.store(
        SF_ptr + (b * H + hp) * KSEL + j, tl.where(ok != 0, s, float("-inf")), mask=inb
    )


def far_sparse_fused(attn, hidden_states, qh, o_loc, lse, pos, reqs, kcache, vcache):
    """ref_far_sparse with the index arithmetic and gathers fused; bit-exact with it."""
    C, W, NB = attn.compress, attn.window, attn.max_blocks
    H, KV, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
    G = H // KV
    bs = qh.shape[0]
    dev = qh.device
    hi = int(attn.idx_cache.shape[0]) - 1
    r2t = attn.r2t
    Di = attn.idx_dim
    assert (C & (C - 1)) == 0 and attn.idx_heads == 4
    pos = pos.contiguous()
    reqs = reqs.contiguous()
    kbi = torch.empty(bs, NB, Di, dtype=attn.blk_cache.dtype, device=dev)
    valid = torch.empty(bs, NB, dtype=torch.int8, device=dev)
    BN = 64
    _far_kbi_kernel[(bs, triton.cdiv(NB, BN))](
        pos,
        reqs,
        r2t,
        r2t.shape[1],
        attn.blk_cache,
        kbi,
        valid,
        NB,
        hi,
        W,
        C=C,
        DI=Di,
        BLOCK_N=BN,
        num_warps=4,
    )
    qi = attn.idx_norm(attn.idx_q(hidden_states)[0].reshape(-1, Di))
    qi = qi.view(bs, attn.idx_heads, Di)
    e = torch.einsum("bhd,bnd->bhn", qi, kbi)
    s_idx = torch.empty(bs, NB, dtype=e.dtype, device=dev)
    BS = 1024
    _far_score_kernel[(bs, triton.cdiv(NB, BS))](
        e,
        valid,
        s_idx,
        NB,
        float(attn.idx_scale),
        BLOCK_N=BS,
        num_warps=4,
        enable_fp_fusion=False,
    )
    ksel = min(attn.bcsa_topk, NB)
    top = s_idx.topk(ksel, dim=-1).indices.contiguous()
    kb = torch.empty(bs, ksel, KV, D, dtype=kcache.dtype, device=dev)
    vb = torch.empty(bs, ksel, KV, D, dtype=vcache.dtype, device=dev)
    sel_ok = torch.empty(bs, ksel, dtype=torch.int8, device=dev)
    _far_pool_kernel[(bs, ksel)](
        top,
        pos,
        reqs,
        r2t,
        r2t.shape[1],
        kcache,
        vcache,
        kb,
        vb,
        sel_ok,
        ksel,
        hi,
        W,
        C=C,
        ROW=KV * D,
        num_warps=4,
        enable_fp_fusion=False,
    )
    e2 = torch.einsum("bkgd,bnkd->bkgn", qh.view(bs, KV, G, D), kb).contiguous()
    sf = torch.empty(bs, H, ksel, dtype=torch.float32, device=dev)
    _far_sf_kernel[(bs, H)](
        e2,
        sel_ok,
        sf,
        H,
        ksel,
        float(attn.scaling),
        BLOCK=triton.next_power_of_2(ksel),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return merge_local_far(sf, lse, o_loc, vb, KV, kv_first=False)


def r2t_gather(
    r2t: torch.Tensor, req_idx: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """r2t[req_idx[b], cols[b, j]] as one flat gather. req_idx [bs], cols [bs, n] -> [bs, n] int64."""
    return r2t.view(-1)[req_idx[:, None] * r2t.shape[1] + cols].long()


def ref_far_sparse(attn, hidden_states, qh, o_loc, lse, pos, reqs, kcache, vcache):
    """Torch reference of far_sparse_fused: the BCSA far branch at one position per query token.

    Every shape is static (safe to capture in a CUDA graph). `pos` and `reqs` are per token. The far
    set of a query is read only from complete blocks whose last position is <= pos - W.
    """
    C, W, NB = attn.compress, attn.window, attn.max_blocks
    H, KV, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
    G = H // KV
    bs = qh.shape[0]
    dev = qh.device
    r2t = attn.r2t
    hi = int(attn.idx_cache.shape[0]) - 1  # trash row for pads
    n = torch.arange(NB, device=dev)
    # Block n is far iff its last position is at or before pos - W; floor division keeps short
    # sequences at nmax < 0, i.e. an empty far set.
    nmax = (pos - W - (C - 1)).div(C, rounding_mode="floor")  # [bs]
    valid = n[None, :] <= nmax[:, None]  # [bs, NB]

    # Pooled indexer keys of every block, read at the slot of the block's last position. Blocks
    # past nmax read a stale or trash row and are masked below.
    last = r2t_gather(r2t, reqs, (n * C + (C - 1))[None, :].expand(bs, NB)).clamp_(
        0, hi
    )
    kbi = attn.blk_cache[last]  # [bs, NB, Di]

    qi = attn.idx_norm(attn.idx_q(hidden_states)[0].reshape(-1, attn.idx_dim))
    qi = qi.view(bs, attn.idx_heads, attn.idx_dim)
    s_idx = torch.einsum("bhd,bnd->bhn", qi, kbi).mean(1) * attn.idx_scale
    s_idx = s_idx.masked_fill(~valid, float("-inf"))
    ksel = min(attn.bcsa_topk, NB)
    top = s_idx.topk(ksel, dim=-1).indices  # [bs, ksel]
    sel_ok = valid.gather(1, top)

    # Pooled K/V of the selected blocks, straight out of the KV cache.
    cols = (top[:, :, None] * C + torch.arange(C, device=dev)).view(bs, ksel * C)
    loc = r2t_gather(r2t, reqs, cols.clamp_(0, r2t.shape[1] - 1)).clamp_(0, hi)
    kb = kcache[loc].view(bs, ksel, C, KV, D).mean(2)  # [bs, ksel, KV, D]
    vb = vcache[loc].view(bs, ksel, C, KV, D).mean(2)

    sf = torch.einsum("bkgd,bnkd->bkgn", qh.view(bs, KV, G, D), kb) * attn.scaling
    sf = sf.reshape(bs, H, ksel).float().masked_fill(~sel_ok[:, None, :], float("-inf"))
    return merge_local_far(sf, lse, o_loc, vb, KV, kv_first=False)
