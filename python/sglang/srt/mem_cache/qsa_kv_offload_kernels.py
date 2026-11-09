"""CUDA-graph-safe resolve for the QSA K/V host offload.

Per QSA layer and forward, with every table on the device and no host
synchronization, so decode and target-verify forwards can be captured:

``_resolve_kernel``, one program per request (its ``Q`` query rows):
  1. tail ring: blocks ``b0 = q0 // CR`` and ``b0 + 1`` take their ring slots
     (``b % 2``); committed rows (``< q0``) of a block that newly enters a ring
     slot, or whose logical slots changed, are planned as host loads. A block
     re-entering the ring may be rewritten there, so its LRU copy turns stale.
  2. ``U``: the sorted unique complete blocks before ``b0`` the rows select, one
     representative column per block (the one holding its first token).
  3. hits and misses; misses take the least recently used slots this forward
     does not read (an exact LRU list, double buffered). A hit whose rows were
     filled from logical slots ``req_to_token`` no longer maps to (radix dedup
     of a running request) is refreshed from the host in place.
  4. a copy plan: one host block per miss or refresh.
  5. the device row of every selected column (``-1`` where it is not attended).
``_copy_plan_kernel`` and ``_ring_load_kernel`` then copy the planned host rows
(read through their pinned UVA addresses). Requests at or past ``num_real``
(CUDA-graph padding) change no state and get ``-1`` rows.
"""

import torch
import triton
import triton.language as tl

_BIG = tl.constexpr(1 << 30)  # sort key of "no block"
_RING = 2
_PLAN_PER_PROGRAM = 8


@triton.jit
def _resolve_kernel(
    topk_ptr,
    stride_topk,
    seq_ptr,
    req_ptr,
    qpos0_ptr,
    blk_slot_ptr,
    slot_blk_ptr,
    mark_ptr,
    lru_ptr,
    lru_par_ptr,
    ring_ptr,
    slot_src_ptr,
    ring_src_ptr,
    r2t_ptr,
    stride_r2t,
    rows_ptr,
    stride_rows,
    plan_src_ptr,
    plan_dst_ptr,
    plan_n_ptr,
    rplan_src_ptr,
    rplan_dst_ptr,
    rplan_n_ptr,
    uniq_ptr,
    vic_ptr,
    err_ptr,
    step_ptr,
    num_real_ptr,
    MAXB,
    H,
    NR,
    REGION,
    Q: tl.constexpr,
    W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    KB: tl.constexpr,
    BLOCK_U: tl.constexpr,
    CR: tl.constexpr,
    NRING: tl.constexpr,
    BH: tl.constexpr,
):
    g = tl.program_id(0)
    cols = tl.arange(0, BLOCK_W)
    if g >= tl.load(num_real_ptr):
        for q in tl.static_range(Q):
            tl.store(
                rows_ptr + (g * Q + q) * stride_rows + cols,
                tl.full([BLOCK_W], -1, tl.int32),
                mask=cols < W,
            )
        tl.store(plan_n_ptr + g, 0)
        tl.store(rplan_n_ptr + g, 0)
        return
    step = tl.load(step_ptr)
    req = tl.load(req_ptr + g * Q).to(tl.int64)
    q0 = tl.load(qpos0_ptr + g).to(tl.int64)
    b0 = q0 // CR
    base = req * REGION
    ring_rows = NRING * CR

    # 1. tail ring
    nring = tl.full([], 0, tl.int32)
    jv = tl.arange(0, CR)
    for t in tl.static_range(NRING):
        b = b0 + t
        rs = b % NRING
        old = tl.load(ring_ptr + req * NRING + rs).to(tl.int64)
        changed = old != b
        rec = tl.load(ring_src_ptr + (req * NRING + rs) * CR + jv)
        if changed:
            tl.store(ring_ptr + req * NRING + rs, b.to(tl.int32))
            stale_slot = tl.load(blk_slot_ptr + req * MAXB + b).to(tl.int64)
            if stale_slot >= 0:
                tl.store(
                    slot_src_ptr + (req * H + stale_slot) * CR + jv,
                    tl.full([CR], -2, tl.int32),
                )
        pv = b * CR + jv
        curv = tl.load(r2t_ptr + req * stride_r2t + pv, mask=pv < q0 + Q, other=-1)
        need_load = (pv < q0) & (changed | (rec != curv))
        nl = need_load.to(tl.int32)
        idx = nring + tl.cumsum(nl, axis=0) - nl
        tl.store(rplan_src_ptr + g * (NRING * CR) + idx, curv, mask=need_load)
        tl.store(
            rplan_dst_ptr + g * (NRING * CR) + idx, base + rs * CR + jv, mask=need_load
        )
        nring += tl.sum(nl, axis=0)
        tl.store(
            ring_src_ptr + (req * NRING + rs) * CR + jv,
            tl.where(pv < q0 + Q, curv, tl.where(changed, -1, rec)),
        )
    tl.store(rplan_n_ptr + g, nring)

    # 2. U: sorted unique needed blocks
    u = tl.arange(0, BLOCK_U)
    uq = u // KB
    uj = u % KB
    in_rows = uq < Q
    pos = tl.load(
        topk_ptr + (g * Q + uq) * stride_topk + uj * CR, mask=in_rows, other=-1
    ).to(tl.int64)
    sq = tl.load(seq_ptr + g * Q + uq, mask=in_rows, other=0).to(tl.int64)
    rep = in_rows & (pos >= 0) & (pos < sq) & (pos % CR == 0) & (pos // CR < b0)
    ub = tl.sort(tl.where(rep, (pos // CR).to(tl.int32), _BIG))
    tl.store(uniq_ptr + g * BLOCK_U + u, ub)
    tl.debug_barrier()
    prev = tl.load(uniq_ptr + g * BLOCK_U + u - 1, mask=u > 0, other=-1)
    uniq = (ub != prev) & (ub < _BIG)
    ubc = tl.where(uniq, ub, 0).to(tl.int64)

    # 3. hits, misses and the exact LRU order
    slot = tl.load(blk_slot_ptr + req * MAXB + ubc, mask=uniq, other=-1).to(tl.int64)
    hit = uniq & (slot >= 0)
    miss = uniq & (slot < 0)
    stale = tl.zeros([BLOCK_U], dtype=tl.int1)
    for j in tl.static_range(CR):
        cj = tl.load(r2t_ptr + req * stride_r2t + ubc * CR + j, mask=uniq, other=-1)
        rj = tl.load(
            slot_src_ptr + (req * H + tl.where(hit, slot, 0)) * CR + j,
            mask=hit,
            other=-2,
        )
        stale = stale | (hit & (rj != cj))
    nh = tl.sum(hit.to(tl.int32), axis=0)
    m = tl.sum(miss.to(tl.int32), axis=0)
    tl.store(mark_ptr + req * H + slot, step, mask=hit)
    tl.debug_barrier()
    par = tl.load(lru_par_ptr + req)
    cur_lru = lru_ptr + (par * NR + req) * H
    nxt_lru = lru_ptr + ((1 - par) * NR + req) * H
    carry_n = tl.full([], 0, tl.int32)
    carry_h = tl.full([], 0, tl.int32)
    for c0 in range(0, H, BH):
        i = c0 + tl.arange(0, BH)
        mi = i < H
        s = tl.load(cur_lru + i, mask=mi, other=0)
        h = (tl.load(mark_ptr + req * H + s, mask=mi, other=-1) == step) & mi
        n = (~h) & mi
        hi = h.to(tl.int32)
        ni = n.to(tl.int32)
        ex_n = tl.cumsum(ni, axis=0) - ni + carry_n
        ex_h = tl.cumsum(hi, axis=0) - hi + carry_h
        victim = n & (ex_n < m)
        # New order, least to most recent: untouched | hits | newly loaded.
        newpos = tl.where(
            h, H - nh - m + ex_h, tl.where(victim, H - m + ex_n, ex_n - m)
        )
        tl.store(nxt_lru + newpos, s, mask=mi)
        tl.store(vic_ptr + g * BLOCK_U + ex_n, s, mask=victim)
        carry_n += tl.sum(ni, axis=0)
        carry_h += tl.sum(hi, axis=0)
    tl.store(lru_par_ptr + req, 1 - par)
    tl.debug_barrier()

    # 4. assign victims, update the tables, plan the copies
    mi32 = miss.to(tl.int32)
    mr = tl.cumsum(mi32, axis=0) - mi32
    v = tl.load(vic_ptr + g * BLOCK_U + mr, mask=miss, other=0).to(tl.int64)
    old_b = tl.load(slot_blk_ptr + req * H + v, mask=miss, other=-1).to(tl.int64)
    tl.store(
        blk_slot_ptr + req * MAXB + tl.maximum(old_b, 0),
        -1,
        mask=miss & (old_b >= 0),
    )
    tl.debug_barrier()
    tl.store(slot_blk_ptr + req * H + v, ub, mask=miss)
    tl.store(blk_slot_ptr + req * MAXB + ubc, v.to(tl.int32), mask=miss)
    tl.store(mark_ptr + req * H + v, step, mask=miss)
    si32 = stale.to(tl.int32)
    ns = tl.sum(si32, axis=0)
    planned = miss | stale
    k = tl.where(miss, mr, m + tl.cumsum(si32, axis=0) - si32)
    dslot = tl.where(miss, v, slot)
    for j in tl.static_range(CR):
        cj = tl.load(r2t_ptr + req * stride_r2t + ubc * CR + j, mask=planned, other=0)
        tl.store(
            plan_src_ptr + (g * BLOCK_U + k) * CR + j, cj.to(tl.int64), mask=planned
        )
        tl.store(
            slot_src_ptr + (req * H + dslot) * CR + j, cj.to(tl.int32), mask=planned
        )
    tl.store(
        plan_dst_ptr + g * BLOCK_U + k, base + ring_rows + dslot * CR, mask=planned
    )
    tl.store(plan_n_ptr + g, m + ns)
    tl.debug_barrier()

    # 5. device row of every selected column
    nerr = tl.full([], 0, tl.int32)
    for q in tl.static_range(Q):
        pc = tl.load(
            topk_ptr + (g * Q + q) * stride_topk + cols, mask=cols < W, other=-1
        ).to(tl.int64)
        sqq = tl.load(seq_ptr + g * Q + q).to(tl.int64)
        valid = (cols < W) & (pc >= 0) & (pc < sqq)
        blk = tl.where(valid, pc // CR, 0)
        off = tl.where(valid, pc % CR, 0)
        in_ring = valid & (blk >= b0)
        sl = tl.load(
            blk_slot_ptr + req * MAXB + blk, mask=valid & ~in_ring, other=-1
        ).to(tl.int64)
        err = valid & ~in_ring & (sl < 0)
        row = tl.where(
            in_ring,
            base + (blk % NRING) * CR + off,
            base + ring_rows + sl * CR + off,
        )
        row = tl.where(valid & ~err, row, -1)
        tl.store(
            rows_ptr + (g * Q + q) * stride_rows + cols, row.to(tl.int32), mask=cols < W
        )
        nerr += tl.sum(err.to(tl.int32), axis=0)
    tl.atomic_add(err_ptr, nerr.to(tl.int64))


@triton.jit
def _copy_plan_kernel(
    plan_src_ptr,
    plan_dst_ptr,
    plan_n_ptr,
    num_real_ptr,
    host_k_addr,
    host_v_addr,
    dev_k_ptr,
    dev_v_ptr,
    BLOCK_U: tl.constexpr,
    PER_PROGRAM: tl.constexpr,
    CR: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    """Plan entries of one request: CR host rows (logical slots, read through
    UVA) each, into CR consecutive device rows."""
    g = tl.program_id(0)
    c = tl.program_id(1)
    if g >= tl.load(num_real_ptr):
        return
    n = tl.load(plan_n_ptr + g)
    if c * PER_PROGRAM >= n:
        return
    r = tl.arange(0, PER_PROGRAM * CR)
    entry = c * PER_PROGRAM + r // CR
    ok = entry < n
    src = tl.load(plan_src_ptr + (g * BLOCK_U + entry) * CR + r % CR, mask=ok, other=0)
    dst = tl.load(plan_dst_ptr + g * BLOCK_U + entry, mask=ok, other=0) + r % CR
    col = tl.arange(0, ROW_WORDS)
    hk = (host_k_addr + src * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    hv = (host_v_addr + src * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    out = dst.to(tl.int64)[:, None] * ROW_WORDS + col[None, :]
    k = tl.load(hk[:, None] + col[None, :], mask=ok[:, None], other=0)
    v = tl.load(hv[:, None] + col[None, :], mask=ok[:, None], other=0)
    tl.store(dev_k_ptr + out, k, mask=ok[:, None])
    tl.store(dev_v_ptr + out, v, mask=ok[:, None])


@triton.jit
def _ring_load_kernel(
    rplan_src_ptr,
    rplan_dst_ptr,
    rplan_n_ptr,
    num_real_ptr,
    host_k_addr,
    host_v_addr,
    dev_k_ptr,
    dev_v_ptr,
    SLOTS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    g = tl.program_id(0)
    if g >= tl.load(num_real_ptr):
        return
    r = tl.arange(0, SLOTS)
    ok = r < tl.load(rplan_n_ptr + g)
    src = tl.load(rplan_src_ptr + g * SLOTS + r, mask=ok, other=0).to(tl.int64)
    dst = tl.load(rplan_dst_ptr + g * SLOTS + r, mask=ok, other=0)
    col = tl.arange(0, ROW_WORDS)
    hk = (host_k_addr + src * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    hv = (host_v_addr + src * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    out = dst[:, None] * ROW_WORDS + col[None, :]
    k = tl.load(hk[:, None] + col[None, :], mask=ok[:, None], other=0)
    v = tl.load(hv[:, None] + col[None, :], mask=ok[:, None], other=0)
    tl.store(dev_k_ptr + out, k, mask=ok[:, None])
    tl.store(dev_v_ptr + out, v, mask=ok[:, None])


@triton.jit
def _write_through_kernel(
    dev_rows_ptr,
    host_rows_ptr,
    host_k_addr,
    host_v_addr,
    dev_k_ptr,
    dev_v_ptr,
    ROW_WORDS: tl.constexpr,
):
    """Row i: device row ``dev_rows[i]`` -> host logical slot ``host_rows[i]``."""
    i = tl.program_id(0)
    col = tl.arange(0, ROW_WORDS)
    d = tl.load(dev_rows_ptr + i).to(tl.int64)
    h = tl.load(host_rows_ptr + i).to(tl.int64)
    hk = (host_k_addr + h * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    hv = (host_v_addr + h * (ROW_WORDS * 8)).to(tl.pointer_type(tl.int64))
    tl.store(hk + col, tl.load(dev_k_ptr + d * ROW_WORDS + col))
    tl.store(hv + col, tl.load(dev_v_ptr + d * ROW_WORDS + col))


class QSAKVFusedResolve:
    """Device tables and launchers of the CUDA-graph-safe resolve."""

    def __init__(
        self,
        *,
        num_request_slots: int,
        hot_blocks: int,
        compress_ratio: int,
        token_topk: int,
        width: int,
        region_rows: int,
        dev_k,
        dev_v,
        host_k: torch.Tensor,
        host_v: torch.Tensor,
        max_rows_per_request: int,
    ):
        device = dev_k[0].device
        row_bytes = dev_k[0][0].numel() * dev_k[0].element_size()
        if row_bytes % 8:
            raise ValueError(f"K/V row of {row_bytes} B is not a multiple of 8 B")
        layers = len(dev_k)
        self.row_words = row_bytes // 8
        self.dev_k = [b.view(b.shape[0], -1).view(torch.int64) for b in dev_k]
        self.dev_v = [b.view(b.shape[0], -1).view(torch.int64) for b in dev_v]
        self.host_k_addr = [host_k[i].data_ptr() for i in range(layers)]
        self.host_v_addr = [host_v[i].data_ptr() for i in range(layers)]
        nr, hot, cr = num_request_slots, hot_blocks, compress_ratio
        self.num_request_slots, self.hot_blocks, self.cr = nr, hot, cr
        self.region_rows = region_rows
        self.block_topk = token_topk // cr
        self.max_blocks = width // cr + _RING
        i32 = dict(dtype=torch.int32, device=device)
        self.blk_slot = torch.full((layers, nr, self.max_blocks), -1, **i32)
        self.slot_blk = torch.full((layers, nr, hot), -1, **i32)
        self.mark = torch.full((layers, nr, hot), -1, **i32)
        self.lru = torch.arange(hot, **i32).repeat(layers, 2, nr, 1).contiguous()
        self.lru_par = torch.zeros((layers, nr), **i32)
        self.ring = torch.full((layers, nr, _RING), -1, **i32)
        self.slot_src = torch.full((layers, nr, hot, cr), -1, **i32)
        self.ring_src = torch.full((layers, nr, _RING, cr), -1, **i32)
        self.step = torch.zeros(1, **i32)
        self.num_real = torch.zeros(1, **i32)
        self.errors = torch.zeros(1, dtype=torch.int64, device=device)
        self.block_u = {
            q: triton.next_power_of_2(q * self.block_topk)
            for q in range(1, max_rows_per_request + 1)
        }
        max_u = max(self.block_u.values())
        self.plan_src = torch.zeros((nr, max_u, cr), dtype=torch.int64, device=device)
        self.plan_dst = torch.zeros((nr, max_u), dtype=torch.int64, device=device)
        self.plan_n = torch.zeros(nr, **i32)
        self.rplan_src = torch.zeros((nr, _RING * cr), **i32)
        self.rplan_dst = torch.zeros((nr, _RING * cr), dtype=torch.int64, device=device)
        self.rplan_n = torch.zeros(nr, **i32)
        self.uniq = torch.zeros((nr, max_u), **i32)
        self.vic = torch.zeros((nr, max_u), **i32)

    def reset(self, req_pool_indices: torch.Tensor) -> None:
        idx = req_pool_indices.long()
        for table in (
            self.blk_slot,
            self.slot_blk,
            self.mark,
            self.ring,
            self.ring_src,
        ):
            table.index_fill_(1, idx, -1)

    def resolve(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        row_seq_lens: torch.Tensor,
        row_req: torch.Tensor,
        rows_per_request: int,
        req_to_token: torch.Tensor,
    ) -> torch.Tensor:
        num_rows, width = positions.shape
        groups = num_rows // rows_per_request
        if layer_idx == 0:
            self.step.add_(1)
        qpos0 = (row_seq_lens.view(groups, rows_per_request)[:, 0] - 1).contiguous()
        rows = torch.empty(
            (num_rows, width), dtype=torch.int32, device=positions.device
        )
        block_u = self.block_u[rows_per_request]
        _resolve_kernel[(groups,)](
            positions,
            positions.stride(0),
            row_seq_lens,
            row_req,
            qpos0,
            self.blk_slot[layer_idx],
            self.slot_blk[layer_idx],
            self.mark[layer_idx],
            self.lru[layer_idx],
            self.lru_par[layer_idx],
            self.ring[layer_idx],
            self.slot_src[layer_idx],
            self.ring_src[layer_idx],
            req_to_token,
            req_to_token.stride(0),
            rows,
            rows.stride(0),
            self.plan_src,
            self.plan_dst,
            self.plan_n,
            self.rplan_src,
            self.rplan_dst,
            self.rplan_n,
            self.uniq,
            self.vic,
            self.errors,
            self.step,
            self.num_real,
            self.max_blocks,
            self.hot_blocks,
            self.num_request_slots,
            self.region_rows,
            Q=rows_per_request,
            W=width,
            BLOCK_W=triton.next_power_of_2(width),
            KB=self.block_topk,
            BLOCK_U=block_u,
            CR=self.cr,
            NRING=_RING,
            BH=1024,
            num_warps=8,
        )
        _copy_plan_kernel[(groups, triton.cdiv(block_u, _PLAN_PER_PROGRAM))](
            self.plan_src,
            self.plan_dst,
            self.plan_n,
            self.num_real,
            self.host_k_addr[layer_idx],
            self.host_v_addr[layer_idx],
            self.dev_k[layer_idx],
            self.dev_v[layer_idx],
            BLOCK_U=block_u,
            PER_PROGRAM=_PLAN_PER_PROGRAM,
            CR=self.cr,
            ROW_WORDS=self.row_words,
            num_warps=8,
        )
        _ring_load_kernel[(groups,)](
            self.rplan_src,
            self.rplan_dst,
            self.rplan_n,
            self.num_real,
            self.host_k_addr[layer_idx],
            self.host_v_addr[layer_idx],
            self.dev_k[layer_idx],
            self.dev_v[layer_idx],
            SLOTS=_RING * self.cr,
            ROW_WORDS=self.row_words,
            num_warps=2,
        )
        return rows

    def write_through(
        self, layer_idx: int, rows: torch.Tensor, loc: torch.Tensor
    ) -> None:
        if rows.numel() == 0:
            return
        _write_through_kernel[(rows.numel(),)](
            rows,
            loc,
            self.host_k_addr[layer_idx],
            self.host_v_addr[layer_idx],
            self.dev_k[layer_idx],
            self.dev_v[layer_idx],
            ROW_WORDS=self.row_words,
            num_warps=2,
        )
