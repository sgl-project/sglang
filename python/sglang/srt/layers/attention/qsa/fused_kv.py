import triton
import triton.language as tl


@triton.jit
def _qsa_fused_kv_prepare(
    K,
    V,
    NK,
    NV,
    LOC,
    MAP,
    REQ,
    IDX,
    LENGTH,
    QPOS,
    COUNTS,
    OK,
    OV,
    H: tl.constexpr,
    D: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    NK_STRIDE: tl.constexpr,
    NV_STRIDE: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
    BT: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
    EXPAND: tl.constexpr,
    RATIO: tl.constexpr,
    IB: tl.constexpr,
    CHAIN: tl.constexpr,
):
    tiles = tl.cdiv(S, B) + 2
    pid = tl.program_id(0)
    row = (pid // (G * tiles * H)) * G + pid % G
    tile = (pid // G) % tiles
    head = (pid // (G * tiles)) % H
    if row >= R:
        return
    req = tl.load(REQ + row)
    row_active = tl.load(LOC + row) != 0
    d = tl.arange(0, D)
    if tile < tl.cdiv(S, B):
        c = tile * B + tl.arange(0, B)
        if EXPAND:
            n = tl.minimum((tl.load(LENGTH + row) // RATIO), IB) * RATIO
            blocks = tl.load(IDX + row * IB + c // RATIO, c < IB * RATIO, -1)
            expanded = blocks * RATIO + c % RATIO
            qpos = tl.load(QPOS + row)
            tail_start = (qpos + 1) // RATIO * RATIO
            tail_off = c - n
            tail = tail_start + tail_off
            pos = tl.where(
                (c < n) & (blocks >= 0),
                expanded,
                tl.where(
                    (tail_off >= 0) & (tail_off < RATIO - 1) & (tail < qpos + 1),
                    tail,
                    -1,
                ),
            )
        else:
            pos = tl.load(IDX + row * T + c, c < T, -1)
        length = tl.load(LENGTH + row)
        valid = (pos >= 0) & (pos < length) & (c < T)
        valid = valid & row_active
        slot = tl.load(MAP + req.to(tl.int64) * MAP_STRIDE + pos, valid, 0).to(tl.int64)
        if EXPAND and CHAIN and W > 1:
            group_row = row // W * W
            first_pos = length - 1 - row % W
            relative_pos = pos - first_pos
            fresh = valid & (relative_pos >= 0)
            input_row = tl.where(fresh, group_row + relative_pos, 0)
        else:
            input_row = tl.full((B,), 0, tl.int32)
            fresh = tl.full((B,), False, tl.int1)
            for j in tl.static_range(W):
                candidate_row = row // W * W + j
                candidate_slot = tl.load(LOC + candidate_row)
                matches = valid & (candidate_slot != 0) & (slot == candidate_slot)
                input_row = tl.where(matches, candidate_row, input_row)
                fresh = fresh | matches
        oldoff = slot[:, None] * H * D + head * D + d[None, :]
        nkoff = input_row.to(tl.int64)[:, None] * NK_STRIDE + head * D + d[None, :]
        nvoff = input_row.to(tl.int64)[:, None] * NV_STRIDE + head * D + d[None, :]
        kp = tl.where(fresh[:, None], NK + nkoff, K + oldoff)
        vp = tl.where(fresh[:, None], NV + nvoff, V + oldoff)
        k = tl.load(kp, valid[:, None], 0.0)
        v = tl.load(vp, valid[:, None], 0.0)
        outoff = (row.to(tl.int64) * S + c[:, None]) * H * D + head * D + d[None, :]
        tl.store(OK + outoff, k, (c < S)[:, None])
        tl.store(OV + outoff, v, (c < S)[:, None])
    elif tile == tl.cdiv(S, B):
        slot = tl.load(LOC + row).to(tl.int64)
        k = tl.load(NK + row.to(tl.int64) * NK_STRIDE + head * D + d).to(
            K.dtype.element_ty
        )
        v = tl.load(NV + row.to(tl.int64) * NV_STRIDE + head * D + d).to(
            V.dtype.element_ty
        )
        tl.store(K + slot * H * D + head * D + d, k, slot != 0)
        tl.store(V + slot * H * D + head * D + d, v, slot != 0)
    elif head == 0:
        if EXPAND:
            n = tl.minimum((tl.load(LENGTH + row) // RATIO), IB) * RATIO
            qpos = tl.load(QPOS + row)
            tail_start = (qpos + 1) // RATIO * RATIO
            length = tl.load(LENGTH + row)
            tail_count = tl.maximum(0, tl.minimum(qpos + 1, length) - tail_start)
            tl.store(COUNTS + row, n + tail_count)
        else:
            c = tl.arange(0, BT)
            pos = tl.load(IDX + row * T + c, c < T, -1)
            length = tl.load(LENGTH + row)
            tl.store(COUNTS + row, tl.sum(((pos >= 0) & (pos < length)).to(tl.int32)))


def fused_kv_prepare(
    k_cache,
    v_cache,
    k,
    v,
    loc,
    req_to_token,
    row_requests,
    indices,
    sequence_lengths,
    counts,
    packed_k,
    packed_v,
    width,
    compress_ratio=1,
    query_positions=None,
    chain_positions=False,
):
    rows, input_topk = indices.shape
    topk = input_topk * compress_ratio + compress_ratio - 1
    if rows == 0:
        return
    heads, dim = k_cache.shape[1:]
    stride = packed_k.shape[0] // rows
    block = 8 if compress_ratio > 1 and rows <= 4 else 16
    _qsa_fused_kv_prepare[(rows * heads * (triton.cdiv(stride, block) + 2),)](
        k_cache,
        v_cache,
        k,
        v,
        loc,
        req_to_token,
        row_requests,
        indices,
        sequence_lengths,
        query_positions,
        counts,
        packed_k,
        packed_v,
        heads,
        dim,
        req_to_token.stride(0),
        k.stride(0),
        v.stride(0),
        topk,
        stride,
        width,
        block,
        triton.next_power_of_2(topk),
        rows,
        width,
        compress_ratio > 1,
        compress_ratio,
        input_topk,
        chain_positions,
        num_warps=4,
    )
