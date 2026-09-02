from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _select_group_value(group, v0, v1, v2, v3):
    return tl.where(
        group == 0,
        v0,
        tl.where(group == 1, v1, tl.where(group == 2, v2, v3)),
    )


@triton.jit
def _ceil_ue8m0_exp(x):
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp += mantissa != 0
    return tl.minimum(tl.maximum(exp, 1), 254)


@triton.jit
def _fp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax > 0.25).to(tl.uint8)
    idx += (ax > 0.75).to(tl.uint8)
    idx += (ax > 1.25).to(tl.uint8)
    idx += (ax > 1.75).to(tl.uint8)
    idx += (ax > 2.5).to(tl.uint8)
    idx += (ax > 3.5).to(tl.uint8)
    idx += (ax > 5.0).to(tl.uint8)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _quantize_fp4_indexer_kernel(
    x,
    x_fp4,
    x_sf,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    values = tl.load(x + token_id * BLOCK_N + offs).to(tl.float32)
    abs_values = tl.abs(values)

    amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)

    sf0 = tl.maximum(amax0 / 6.0, 1.0e-4)
    sf1 = tl.maximum(amax1 / 6.0, 1.0e-4)
    sf2 = tl.maximum(amax2 / 6.0, 1.0e-4)
    sf3 = tl.maximum(amax3 / 6.0, 1.0e-4)

    exp0 = _ceil_ue8m0_exp(sf0)
    exp1 = _ceil_ue8m0_exp(sf1)
    exp2 = _ceil_ue8m0_exp(sf2)
    exp3 = _ceil_ue8m0_exp(sf3)

    packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    tl.store(x_sf + token_id, packed_sf)

    pair_offsets = tl.arange(0, BLOCK_N // 2)
    offs0 = pair_offsets * 2
    offs1 = offs0 + 1
    group0 = offs0 // GROUP_N
    group1 = offs1 // GROUP_N
    scale_exp0 = _select_group_value(group0, exp0, exp1, exp2, exp3)
    scale_exp1 = _select_group_value(group1, exp0, exp1, exp2, exp3)
    scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
    scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)

    v0 = tl.load(x + token_id * BLOCK_N + offs0).to(tl.float32) / scale0
    v1 = tl.load(x + token_id * BLOCK_N + offs1).to(tl.float32) / scale1
    code0 = _fp4_e2m1_code(v0)
    code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    tl.store(x_fp4 + token_id * (BLOCK_N // 2) + pair_offsets, packed)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


@triton.jit
def _store_fp4_index_k_cache_fused_kernel(
    x,            # [n_tokens, 128] bf16 k
    cache,        # [pages, page_size*(64+4)] uint8
    loc,          # [n_tokens] int64
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,   # 128
    GROUP_N: tl.constexpr,   # 32
):
    # Fused quantize + paged store: quantize bf16 k to fp4 (E2M1 + UE8M0 group
    # scale) and write it straight into the paged index-k cache in one pass.
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    abs_values = tl.abs(tl.load(x + token_id * BLOCK_N + offs).to(tl.float32))
    amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
    amax1 = tl.max(tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0), axis=0)
    amax2 = tl.max(tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0), axis=0)
    amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)
    exp0 = _ceil_ue8m0_exp(tl.maximum(amax0 / 6.0, 1.0e-4))
    exp1 = _ceil_ue8m0_exp(tl.maximum(amax1 / 6.0, 1.0e-4))
    exp2 = _ceil_ue8m0_exp(tl.maximum(amax2 / 6.0, 1.0e-4))
    exp3 = _ceil_ue8m0_exp(tl.maximum(amax3 / 6.0, 1.0e-4))

    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size
    HALF: tl.constexpr = BLOCK_N // 2

    pair = tl.arange(0, HALF)
    o0 = pair * 2
    o1 = o0 + 1
    s0 = (_select_group_value(o0 // GROUP_N, exp0, exp1, exp2, exp3) << 23).to(tl.float32, bitcast=True)
    s1 = (_select_group_value(o1 // GROUP_N, exp0, exp1, exp2, exp3) << 23).to(tl.float32, bitcast=True)
    c0 = _fp4_e2m1_code(tl.load(x + token_id * BLOCK_N + o0).to(tl.float32) / s0)
    c1 = _fp4_e2m1_code(tl.load(x + token_id * BLOCK_N + o1).to(tl.float32) / s1)
    packed = (c0 & 0x0F) | ((c1 & 0x0F) << 4)
    tl.store(cache + page * cache_stride + page_offset * HALF + pair, packed)

    sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)
    sfo = tl.arange(0, 4)
    tl.store(
        cache + page * cache_stride + page_size * HALF + page_offset * 4 + sfo,
        (sf >> (sfo * 8)) & 0xFF,
    )


def quantize_fp4_indexer_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == 128
    x = x.contiguous().view(-1, x.shape[-1])
    x_fp4 = torch.empty((x.shape[0], 64), device=x.device, dtype=torch.int8)
    x_sf = torch.empty((x.shape[0],), device=x.device, dtype=torch.int32)
    if x.shape[0] > 0:
        _quantize_fp4_indexer_kernel[(x.shape[0],)](
            x,
            x_fp4,
            x_sf,
            BLOCK_N=128,
            GROUP_N=32,
        )
    return x_fp4, x_sf


def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
) -> None:
    assert input.shape[-1] == 128
    assert cache.shape[1] == page_size * (64 + 4)
    x = input.contiguous().view(-1, 128)
    n_tokens = x.shape[0]
    if n_tokens == 0:
        return
    _store_fp4_index_k_cache_fused_kernel[(n_tokens,)](
        x,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK_N=128,
        GROUP_N=32,
    )


@triton.jit
def _fp4_paged_mqa_logits_kernel(
    q_fp8,        # [B, H, HEAD] fp8_e4m3 (q kept fp8; only KV is fp4)
    kv,           # [pages, page_stride] uint8, blocked [PAGE_SIZE*64 fp4 | PAGE_SIZE*4 sf]
    weight,       # [B, H] fp32
    seq_lens,     # [B] int32
    page_table,   # [B, pt_stride] int32
    out,          # [B, out_stride] fp32
    H,
    max_seq_len,
    page_stride,
    pt_stride,
    out_stride,
    PAGE_SIZE: tl.constexpr,
    HEAD: tl.constexpr,
    HALF: tl.constexpr,
    GROUP: tl.constexpr,
    BH: tl.constexpr,
    NP: tl.constexpr,
):
    # One program = NP pages (BK = NP*PAGE_SIZE kv positions) x all heads.
    # Score = tl.dot_scaled over the block: fp4 KV fed straight to the CDNA4
    # mxfp4 MFMA (no dequant), q kept fp8. Coalesced block store.
    BK: tl.constexpr = NP * PAGE_SIZE
    NG: tl.constexpr = HEAD // GROUP  # e8m0 scale groups along K (=4)
    t = tl.program_id(0)
    kv0 = tl.program_id(1) * BK
    seq_len = tl.load(seq_lens + t)
    if kv0 >= seq_len:
        return

    h = tl.arange(0, BH)
    d = tl.arange(0, HEAD)
    rr = tl.arange(0, BK)
    half = tl.arange(0, HALF)
    hmask = h < H

    # a = q fp8 [BH, HEAD] with unit e8m0 scale [BH, NG]
    a = tl.load(
        q_fp8 + t * H * HEAD + h[:, None] * HEAD + d[None, :],
        mask=hmask[:, None], other=0.0,
    )
    a_scale = tl.full((BH, NG), 127, dtype=tl.uint8)  # 1.0 in e8m0
    w = tl.load(weight + t * H + h, mask=hmask, other=0.0)

    # b = k fp4 loaded transposed to [HALF, BK] (K//2 rows, positions cols);
    # b_scale = k e8m0 [NG, BK]. Raw bytes, NO dequant.
    pos = kv0 + rr
    pmask = pos < seq_len
    phys = tl.load(page_table + t * pt_stride + pos // PAGE_SIZE, mask=pmask, other=0)
    row = pos % PAGE_SIZE
    kbase = phys * page_stride + row * HALF
    b = tl.load(
        kv + half[:, None] + kbase[None, :], mask=pmask[None, :], other=0
    )  # [HALF, BK]
    scbase = phys * page_stride + PAGE_SIZE * HALF + row * 4
    b_scale = tl.load(
        kv + scbase[:, None] + tl.arange(0, NG)[None, :],
        mask=pmask[:, None], other=0,
    )  # [BK, NG] (N-major, K-groups last)

    # [BH, BK] = q(fp8) @ k^T(fp4) via native mxfp4 MFMA (aiter a8wfp4 shapes)
    scores = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e2m1")
    scores = tl.maximum(scores, 0.0) * w[:, None]
    logits = tl.sum(tl.where(hmask[:, None], scores, 0.0), axis=0)  # [BK]
    tl.store(out + t * out_stride + pos, logits, mask=pmask)


def fp4_paged_mqa_logits_triton(
    q,
    kvcache_raw: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata=None,
    max_seq_len: int = 0,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Fused fp4-KV paged-MQA-logits for gfx950. q is kept fp8 (fed straight from
    the fused rope+hadamard op, no extra q fp4-quant kernel); only KV is fp4."""
    q_fp8 = q[0] if isinstance(q, tuple) else q  # [B, 1, H, HEAD] fp8
    B, _, H, HEAD = q_fp8.shape
    PAGE_SIZE = 64
    # Callers pass the pool as [pages, page_size*68] or a reshaped [pages, 64, 1,
    # 68] view; flatten to per-page uint8 bytes so page_stride is the true stride
    # (page_size*(64+4)), not an inner view dim.
    kv_u8 = kvcache_raw.reshape(kvcache_raw.shape[0], -1).view(torch.uint8)
    page_stride = kv_u8.shape[1]
    out = torch.zeros(B, max_seq_len, device=q_fp8.device, dtype=torch.float32)
    BH = max(16, triton.next_power_of_2(H))
    # Occupancy-aware tiling: pick the largest NP (biggest mxfp4 MFMA per program)
    # that still launches >= 2*CUs programs, so low-batch decode fills the device
    # (what gluon's SplitKV does). NUM_CU=256 for MI355X (gfx950).
    NP = 8
    while NP > 1 and B * triton.cdiv(max_seq_len, PAGE_SIZE * NP) < 2 * 256:
        NP //= 2
    grid = (B, triton.cdiv(max_seq_len, PAGE_SIZE * NP))
    _fp4_paged_mqa_logits_kernel[grid](
        q_fp8.reshape(B, H, HEAD),
        kv_u8,
        weight.reshape(B, H).to(torch.float32),
        seq_lens.reshape(B).to(torch.int32),
        page_table,
        out,
        H, max_seq_len, page_stride, page_table.shape[1], max_seq_len,
        PAGE_SIZE=PAGE_SIZE, HEAD=HEAD, HALF=64, GROUP=32, BH=BH, NP=NP,
    )
    return out
