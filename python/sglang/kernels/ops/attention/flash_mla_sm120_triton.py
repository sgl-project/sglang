"""SM120-optimized Triton FlashMLA sparse decode kernel — Tiled V2.

Replaces V1's serial token loop with a tiled vectorized approach:
  1. BLOCK_T tokens loaded simultaneously via 2D gather (vs 1-at-a-time)
  2. All BLOCK_T QK scores computed at once via vectorized mul-reduce
  3. V accumulation via vectorized weighted sum across BLOCK_T tokens
  4. Online softmax operates on tile-level maxima (fewer rescales)

Three typed views of the same paged buffer handle FP8/uint8/BF16 regions:
- float8_e4m3fn view -> nope FP8 values (direct load + dequant)
- uint8 view -> UE8M0 scale bytes (raw integer -> exp2 conversion)
- bfloat16 view -> rope BF16 values (direct load)

DSv4 page layout (per token, 576 bytes data + 8 bytes scales):
  Data section: [0:448] FP8 nope | [448:576] BF16 rope (64 values = 128 bytes)
  Scale section: [page_size*576 + offset*8 : +7] UE8M0 scales (7 groups of 64)

Target: RTX PRO 6000 (SM120, 188 SMs, 99KB SMEM, ~1.5 TB/s GDDR7, 96MB L2)
"""

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM  # 512
_TOKEN_DATA_STRIDE = 576  # bytes per token in data section
_SCALE_STRIDE = 8  # bytes per token in scale section
# Heads per program. >=16 because tl.dot's MMA shape floor is 16; H is 8 at
# tp8 so the tile is padded and masked at the store.
_BLOCK_H = 16
# RTX PRO 6000 (sm120). Used only to decide how many token-axis splits are
# worth launching; being a little wrong just costs a few idle CTAs.
_SM_COUNT = 188
# [S_POW2, 512] fp32 in the merge kernel = 64 KB at S=32, against this part's
# 101376 B smem limit. Raising past 32 needs the merge to tile over D as well --
# and is pointless at topk=512 anyway, since BLOCK_T >= 16 caps useful splits at
# 512/16 = 32. Measured under CUDA-graph replay at the decode shape (vs v_bc):
#   S=2 2.00x, S=4 3.27x, S=8 5.14x, S=16 6.00x, S=32 6.70x.
_MAX_SPLITS = 32

# Opt-in fusion of the two epilogue kernels (merge-partial + attention-sink)
# into one. OFF by default: unset, the shipped two-kernel path runs unchanged
# and bit-identically. See _fused_merge_sink_kernel below.
_FUSE_MERGE_SINK = os.environ.get("SGLANG_DSV41_FUSE_MERGE_SINK", "0") == "1"


@triton.autotune(
    configs=[
        # BLOCK_T is small on purpose. Each tile materialises [BLOCK_T, 512] for
        # the QK/PV dots, and num_stages multiplies that for pipelining, so
        # BLOCK_T=32/num_stages=2 already needs 188 KB against a 101 KB limit
        # (measured: triton OutOfResources). bf16 operands halve it again.
        # Shared-memory budget with BF16 dot operands (limit 101376 B):
        #   per stage = BLOCK_T*512*2 (nope) + BLOCK_T*64*2 (rope)
        #   BLOCK_T=16 -> 18432/stage   BLOCK_T=32 -> 36864/stage
        #   BLOCK_T=64 -> 73728/stage
        # so 64x1, 32x2, 16x4 all fit; 64x2 (147KB) and 32x3 (110KB) do not.
        # These were unaffordable while the operands were fp32.
        triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_T": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_T": 64}, num_warps=16, num_stages=1),
    ],
    # topk_rounded ALONE IS NOT ENOUGH. topk is 512 at BOTH the decode (B=1) and
    # prefill (B=1024) shapes, so with key=["topk_rounded"] (what v_bc and the
    # shipped base rewrite use) the autotuner runs exactly ONCE -- at whichever
    # shape the process sees first -- and reuses that config for the other.
    # Measured on sm120: tuning at B=1 picks BLOCK_T=32/num_warps=8, tuning at
    # B=1024 picks BLOCK_T=16/num_warps=4, and using the decode config at prefill
    # costs 18%. GRID_BUCKET gives each occupancy regime its own tune.
    key=["topk_rounded", "GRID_BUCKET", "SPLIT_K", "HOIST"],
)
@triton.jit
def _tiled_sparse_decode_kernel(
    # Q: [B, H, D] bf16
    Q_ptr,
    # Paged KV cache — three typed views of same underlying memory
    cache_fp8_ptr,  # float8_e4m3fn flat (1 byte/elem) — for nope
    cache_uint8_ptr,  # uint8 flat (1 byte/elem) — for scales
    cache_bf16_ptr,  # bfloat16 flat (2 bytes/elem) — for rope
    # Indices: [B, topk] int32
    indices_ptr,
    # Valid lengths: [B] int32
    topk_len_ptr,
    # Output: [B, H, D] bf16 and LSE: [B, H] float32
    O_ptr,
    LSE_ptr,
    # Scalars
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,  # page_size * 576
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,  # for autotune key
    has_topk_len: tl.constexpr,
    # Strides
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,  # indices batch stride
    stride_os: tl.int32,  # output split stride (0 when SPLIT_K == 1)
    stride_ls: tl.int32,  # LSE split stride (0 when SPLIT_K == 1)
    stride_lb: tl.int32,  # LSE batch stride
    # Constexprs
    NOPE_PAD: tl.constexpr,  # 512 (padded from 448)
    ROPE_DIM: tl.constexpr,  # 64
    NOPE_DIM_RT: tl.int32,  # 448 (runtime, for masking)
    BLOCK_T: tl.constexpr,  # tokens per tile
    BLOCK_H: tl.constexpr,  # heads per block — MUST be >= 16 for tl.dot
    SPLIT_K: tl.constexpr,  # token-axis splits; 1 == exact v_bc code path
    GRID_BUCKET: tl.constexpr,  # occupancy bucket; autotune key only, unused in body
    HOIST: tl.constexpr,  # LEVER A: hoist the per-group dequant scale out of
    # the inner loop.  Set by the launcher to (SPLIT_K > 1) or (n_ctas >=
    # _SM_COUNT), i.e. true whenever the grid fills at least one wave or is
    # split-K.  False ONLY in the sub-wave non-split regime (e.g. the [30,4,1]
    # untrimmed decode at 120 CTAs = 0.64 waves), where wall time IS one CTA's
    # latency, cheaper CTAs buy nothing, and the [8, BLOCK_H, BLOCK_T]
    # intermediate costs more than the [BLOCK_T, NOPE_PAD] tensor it deletes.
    # Measured: hoist is 1.30x at prefill and 1.38x at trimmed decode, but
    # 0.87x at [30,4,1].  With HOIST False every line below is byte-for-byte
    # the shipped algebra, so the gated kernel is bit-identical there.
):
    """Tiled sparse decode: tensor-core QK/PV over a TILE OF HEADS.

    Grid: (B, cdiv(H, BLOCK_H)) — one block per (batch, head-tile).

    WHY A HEAD TILE, AND NOT ONE HEAD PER PROGRAM (the previous design):
      1. TENSOR CORES. With one head per program, Q is a VECTOR, so QK and PV are
         matrix-vector products and the only way to express them is `tl.sum` of a
         broadcast multiply -- pure CUDA-core FMA. `tl.dot` needs M,N,K >= 16, so
         it could not be used at all. Tiling BLOCK_H heads makes both products
         matrix-matrix and they become MMA instructions.
      2. THE 8x REDUNDANT KV GATHER. `indices` is [B, topk] -- head-INDEPENDENT.
         The old grid (B, H) therefore had every head re-gather the SAME
         512x576 B of KV for each token tile, 8 times over at tp8, absorbed only
         by L2. One program per head-tile gathers it ONCE and shares it across
         all heads in the tile via the dot.

    BLOCK_H is a constexpr >= 16 even when H is smaller (H=8 at tp8): the MMA
    shape floor is 16, so the tile is padded and the surplus rows are masked off
    at the store. Padding 8->16 wastes half the M dimension and is still far
    cheaper than issuing the whole thing on CUDA cores.
    """
    bid = tl.program_id(0)
    h_tile = tl.program_id(1)
    # Third grid axis is the token-axis split. When SPLIT_K == 1 the grid is
    # (B, cdiv(H,BLOCK_H), 1), sid is constant 0, and every split-related term
    # below folds away at compile time -- this is byte-for-byte the v_bc path.
    sid = tl.program_id(2)

    # ---- Load Q for this (batch, head-tile) ----
    h_offs = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_offs < H  # [BLOCK_H]
    nope_offs = tl.arange(0, NOPE_PAD)  # [NOPE_PAD]
    nope_mask = nope_offs < NOPE_DIM_RT  # [NOPE_PAD], True for [0:448]
    rope_offs = tl.arange(0, ROPE_DIM)  # [ROPE_DIM]

    q_base = bid * stride_qb + h_offs * stride_qh  # [BLOCK_H]
    # NOTE: deliberately NOT scaled by sm_scale here. Q is stored bf16, so
    # `f32(q)*sm_scale -> bf16` for the MMA would round away mantissa for no
    # reason. sm_scale is folded into the score->log2 conversion instead, which
    # is mathematically identical and keeps Q bit-exact into the tensor cores.
    q_nope = tl.load(
        Q_ptr + q_base[:, None] + nope_offs[None, :],
        mask=h_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )  # [BLOCK_H, NOPE_PAD] bf16
    q_rope = tl.load(
        Q_ptr + q_base[:, None] + NOPE_DIM_RT + rope_offs[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )  # [BLOCK_H, ROPE_DIM] bf16

    if HOIST:
        # LEVER A: Q viewed as 8 groups of 64 for the grouped QK dot, built ONCE.
        q_qk = tl.permute(tl.reshape(q_nope, (BLOCK_H, 8, 64)), (1, 0, 2))  # [8,BLOCK_H,64]

    # ---- Valid token count ----
    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    # ---- Online softmax state, now PER HEAD (vectors, not scalars) ----
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - 1e30
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_nope = tl.zeros([BLOCK_H, NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([BLOCK_H, ROPE_DIM], dtype=tl.float32)

    # ---- Precompute constant index vectors ----
    group_ids = (nope_offs // 64).to(tl.int64)  # [NOPE_PAD], scale group per dim
    t_offs = tl.arange(0, BLOCK_T)  # [BLOCK_T]

    # ---- This program's slice of the token axis ----
    # Partition by whole BLOCK_T tiles so no tile is split across programs; the
    # slice is contiguous (not strided) to keep the index gather local. With
    # SPLIT_K == 1 this is exactly [0, topk).
    if SPLIT_K == 1:
        tk_begin = 0
        tk_end = topk
    else:
        n_tiles = tl.cdiv(topk, BLOCK_T)
        tiles_per_split = tl.cdiv(n_tiles, SPLIT_K)
        tk_begin = sid * tiles_per_split * BLOCK_T
        tk_end = tl.minimum(tk_begin + tiles_per_split * BLOCK_T, topk)

    # ---- Process tokens in tiles of BLOCK_T (gathered ONCE for all heads) ----
    for tile_start in range(tk_begin, tk_end, BLOCK_T):
        t_idx = tile_start + t_offs  # [BLOCK_T]
        t_in_bounds = t_idx < tk_end
        t_valid = t_idx < valid_topk

        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)  # [BLOCK_T]

        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576  # [BLOCK_T]

        # ---- NOPE FP8 gather: [BLOCK_T, NOPE_PAD] ----
        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(cache_fp8_ptr + nope_addrs, mask=nope_2d_mask, other=0.0)

        # ---- Scale gather + dequant: [BLOCK_T, NOPE_PAD] ----
        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        g_offs = tl.arange(0, 8)  # 8 scale groups per token, contiguous
        scale_raw = tl.load(
            cache_uint8_ptr + scale_bases[:, None] + g_offs[None, :],
            mask=idx_valid[:, None],
            other=127,
        )  # [BLOCK_T, 8]  (was [BLOCK_T, 512] -- 64x the addresses and exp2s)
        if HOIST:
            # LEVER A -- ALGEBRAIC DELETION OF scale_b.  The UE8M0 scale is per
            # (token, 64-dim group), so it factors out of both dots.  No
            # [BLOCK_T, NOPE_PAD] bf16 scale tensor and no scaled KV copy exist.
            scale_g = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)  # [BLOCK_T, 8] f32
            kv_nope_b = kv_nope_fp8.to(tl.bfloat16)  # UNSCALED; masked lanes already 0
            kv_g3 = tl.reshape(kv_nope_b, (BLOCK_T, 8, 64))
            kv_qk = tl.permute(kv_g3, (1, 2, 0))  # [8, 64, BLOCK_T]
            kv_pv = tl.permute(kv_g3, (1, 0, 2))  # [8, BLOCK_T, 64]
            scale_t = tl.permute(scale_g, (1, 0))[:, None, :]  # [8, 1, BLOCK_T] f32
        else:
            scale_g = tl.math.exp2(scale_raw.to(tl.float32) - 127.0).to(tl.bfloat16)
            scale_b = tl.broadcast_to(
                scale_g[:, :, None], (BLOCK_T, 8, 64)
            ).reshape(BLOCK_T, NOPE_PAD)
            kv_nope_b = kv_nope_fp8.to(tl.bfloat16) * scale_b  # masked lanes already 0

        # ---- ROPE BF16 gather: [BLOCK_T, ROPE_DIM] ----
        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope_b = tl.load(
            cache_bf16_ptr + rope_addrs, mask=idx_valid[:, None], other=0.0
        )  # already bf16 in memory

        # ---- QK on TENSOR CORES: [BLOCK_H, NOPE_PAD] x [NOPE_PAD, BLOCK_T] ----
        # This is the whole point of the head tile. Previously:
        #   scores = tl.sum(q_nope[None, :] * kv_nope, axis=1)   # CUDA cores
        if HOIST:
            # QK = sum_g scale[t,g] * (q_g . kv_fp8_g) -- 8 dots of K=64, batched
            s3 = tl.dot(q_qk, kv_qk)               # [8, BLOCK_H, BLOCK_T] f32
            scores = tl.sum(s3 * scale_t, axis=0)  # [BLOCK_H, BLOCK_T]
        else:
            scores = tl.dot(q_nope, tl.trans(kv_nope_b))
        scores += tl.dot(q_rope, tl.trans(kv_rope_b))
        scores = tl.where(idx_valid[None, :], scores, -1e30)

        # ---- Online softmax, per head (reduce along the TOKEN axis) ----
        scores_log2 = scores * (LOG2E * sm_scale)  # sm_scale folded in
        tile_max = tl.max(scores_log2, axis=1)  # [BLOCK_H]
        m_new = tl.maximum(m_i, tile_max)  # [BLOCK_H]

        alpha = tl.math.exp2(m_i - m_new)  # [BLOCK_H]
        p = tl.math.exp2(scores_log2 - m_new[:, None])  # [BLOCK_H, BLOCK_T]
        p = tl.where(idx_valid[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        # ---- PV on TENSOR CORES: [BLOCK_H, BLOCK_T] x [BLOCK_T, NOPE_PAD] ----
        p_b = p.to(tl.bfloat16)
        if HOIST:
            # PV: the same scale folds into p ([8,BLOCK_H,BLOCK_T], not [BLOCK_T,512])
            p3 = (p[None, :, :] * scale_t).to(tl.bfloat16)  # [8, BLOCK_H, BLOCK_T]
            acc3 = tl.dot(p3, kv_pv)                        # [8, BLOCK_H, 64]
            acc_nope = acc_nope * alpha[:, None] + tl.reshape(
                tl.permute(acc3, (1, 0, 2)), (BLOCK_H, NOPE_PAD))
        else:
            acc_nope = acc_nope * alpha[:, None] + tl.dot(p_b, kv_nope_b)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p_b, kv_rope_b)
        m_i = m_new

    # ---- Normalize output ----
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)  # [BLOCK_H]
    acc_nope = acc_nope / safe_l[:, None]
    acc_rope = acc_rope / safe_l[:, None]

    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    # ---- Store output (surplus padded head rows masked off) ----
    # With SPLIT_K == 1 stride_os/stride_ls are 0 and this addresses the final
    # [B,H,D] / [B,H] buffers directly -- no partials, no merge launch.
    o_base = bid * stride_ob + sid * stride_os + h_offs * stride_oh  # [BLOCK_H]
    tl.store(
        O_ptr + o_base[:, None] + nope_offs[None, :],
        acc_nope.to(tl.bfloat16),
        mask=h_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        O_ptr + o_base[:, None] + NOPE_DIM_RT + rope_offs[None, :],
        acc_rope.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(LSE_ptr + bid * stride_lb + sid * stride_ls + h_offs, lse, mask=h_mask)


@triton.jit
def _merge_splits_kernel(
    OP_ptr,  # partial out:  [B, SPLIT_K, H, D] bf16 (each split ALREADY normalised)
    LP_ptr,  # partial lse:  [B, SPLIT_K, H] f32
    O_ptr,  # final out:    [B, H, D] bf16
    LSE_ptr,  # final lse:    [B, H] f32
    H: tl.int32,
    stride_opb: tl.int32,
    stride_ops: tl.int32,
    stride_oph: tl.int32,
    stride_lpb: tl.int32,
    stride_lps: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    D: tl.constexpr,  # 512
    SPLIT_K: tl.constexpr,  # actual number of splits
    S_POW2: tl.constexpr,  # next_power_of_2(SPLIT_K), for tl.arange
):
    """Combine the SPLIT_K partial attention results in ONE kernel launch.

    WHY THIS EXISTS. The previous split-K attempt did this merge as ~8 separate
    PyTorch ops (max / exp / where / sum / clamp / copy_). Each costs 10-30 us of
    launch + dispatch overhead, and the attention kernel it is wrapping is only
    ~80 us at B=1 -- so the merge cost more than the parallelism it bought and
    split-K came out 0.70x at decode. Everything below is one launch.

    MATH. Each split s produced a *normalised* output o_s = sum_t p_t v_t / l_s
    and its lse_s = m_s + log(l_s), over its own disjoint slice of the token
    axis. The exact full-softmax output is the lse-weighted convex combination
      w_s = exp(lse_s - max_s lse_s);  o = sum_s w_s o_s / sum_s w_s
    and the combined lse is max_s lse_s + log(sum_s w_s). This is the standard
    flash-decoding merge and is exact up to fp32 rounding.

    One program per (batch, head): grid (B*H,). The whole [S_POW2, D] partial
    tile is one load; at S<=16, D=512 that is 32 KB of fp32, well inside the
    101376 B smem budget this part enforces.
    """
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    s_offs = tl.arange(0, S_POW2)  # [S]
    s_mask = s_offs < SPLIT_K
    d_offs = tl.arange(0, D)  # [D]

    # ---- lse of every split for this (b, h) ----
    lse_s = tl.load(
        LP_ptr + bid * stride_lpb + s_offs * stride_lps + hid,
        mask=s_mask,
        other=float("-inf"),
    )  # [S]

    # A split whose slice held no valid token stored -inf; it must contribute
    # nothing. Guard the max too: if EVERY split is -inf (no valid tokens at
    # all for this row) then m is -inf and exp(-inf - -inf) would be NaN.
    finite = lse_s > -1e30
    m = tl.max(tl.where(finite, lse_s, float("-inf")), axis=0)  # scalar
    m_safe = tl.where(m > -1e30, m, 0.0)

    w = tl.where(finite, tl.math.exp(lse_s - m_safe), 0.0)  # [S]
    denom = tl.sum(w, axis=0)  # scalar

    # ---- weighted sum of the per-split outputs ----
    op = tl.load(
        OP_ptr
        + bid * stride_opb
        + s_offs[:, None] * stride_ops
        + hid * stride_oph
        + d_offs[None, :],
        mask=s_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [S, D]

    acc = tl.sum(op * w[:, None], axis=0)  # [D]
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    acc = acc / safe_denom

    tl.store(O_ptr + bid * stride_ob + hid * stride_oh + d_offs, acc.to(tl.bfloat16))
    tl.store(
        LSE_ptr + bid * H + hid,
        tl.where(denom > 0.0, m_safe + tl.math.log(safe_denom), float("-inf")),
    )


def _run_triton_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] float8
    indices: torch.Tensor,  # [B, ...] int32
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the tiled Triton sparse decode kernel on one paged KV cache."""
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)  # elements = bytes for float8

    # Flatten indices to [B, topk]
    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    # Create three typed views of the flat cache memory.
    # The KV cache may arrive as uint8 or float8_e4m3fn depending on the
    # sglang version.  Ensure each view has the correct dtype so Triton
    # interprets the loaded values correctly (FP8 dequant vs raw integer).
    total_elems = num_pages * page_bytes
    raw_flat = k_cache.as_strided((total_elems,), (1,))
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    # Squeeze Q: [B, H, D]
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    # Round topk for autotune key stability
    topk_rounded = triton.next_power_of_2(topk)

    # ---- How many token-axis splits? ----
    # The head tile collapsed the grid from (B,H) to (B, cdiv(H,BLOCK_H)); at
    # tp8 H=8 so that is ONE CTA per sequence. At B=1 that leaves 187 of this
    # part's 188 SMs idle -- the decode shape is occupancy-starved, not
    # math-bound. Splitting the token loop S ways multiplies the CTA count by S.
    # At the prefill shape (B in the hundreds/thousands) the grid is already
    # saturated, S collapses to 1, and we take the exact non-split path below:
    # no partial buffers, no merge launch, no regression on the 12.96x win.
    n_ctas = B * triton.cdiv(H, _BLOCK_H)
    grid_bucket = min(triton.next_power_of_2(n_ctas), 256)
    split_k = 1
    if n_ctas < _SM_COUNT // 2:
        split_k = min(_MAX_SPLITS, max(1, _SM_COUNT // n_ctas))
        # Never make a split finer than one BLOCK_T tile (BLOCK_T >= 16), else
        # the extra CTAs are pure empty launches.
        split_k = max(1, min(split_k, topk // 16))
        split_k = triton.next_power_of_2(split_k) if split_k > 1 else 1
        split_k = min(split_k, _MAX_SPLITS)

    # ---- LEVER A gate ----
    # The scale hoist makes each CTA cheaper but materialises an
    # [8, BLOCK_H, BLOCK_T] intermediate.  That trades well whenever CTA count
    # (not CTA latency) sets the wall time -- every split-K decode, and prefill,
    # where the grid is thousands of CTAs.  It trades BADLY in the one regime
    # where the grid does not even fill a wave and is not split: there wall time
    # is literally one CTA's latency, so a cheaper CTA buys nothing while the
    # extra intermediate costs.  Measured on sm120 (188 SMs): 1.2997x prefill,
    # 1.3832x trimmed decode [30,1,8], but 0.8693x untrimmed decode [30,4,1]
    # (120 CTAs = 0.64 waves).  Compile-time constexpr, so the losing branch is
    # not merely predicted-false -- it is never emitted.
    hoist = (split_k > 1) or (n_ctas >= _SM_COUNT)

    if split_k > 1:
        out_p = torch.empty(B, split_k, H, D, dtype=torch.bfloat16, device=q.device)
        lse_p = torch.empty(B, split_k, H, dtype=torch.float32, device=q.device)
        o_arg, l_arg = out_p, lse_p
        stride_ob_a, stride_oh_a = out_p.stride(0), out_p.stride(2)
        stride_os_a, stride_ls_a = out_p.stride(1), lse_p.stride(1)
        stride_lb_a = lse_p.stride(0)
    else:
        o_arg, l_arg = out, lse
        stride_ob_a, stride_oh_a = out.stride(0), out.stride(1)
        stride_os_a, stride_ls_a = 0, 0
        stride_lb_a = H

    grid = (B, triton.cdiv(H, _BLOCK_H), split_k)
    _tiled_sparse_decode_kernel[grid](
        q3,
        raw_fp8,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        o_arg,
        l_arg,
        softmax_scale,
        page_size,
        int(page_bytes),  # page_bytes (int64)
        int(page_size * _TOKEN_DATA_STRIDE),  # scale_section_off (int64)
        H,
        topk,
        topk_rounded,
        topk_length is not None,
        q3.stride(0),
        q3.stride(1),
        stride_ob_a,
        stride_oh_a,
        flat_indices.stride(0),
        stride_os_a,
        stride_ls_a,
        stride_lb_a,
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
        BLOCK_H=_BLOCK_H,
        SPLIT_K=split_k,
        GRID_BUCKET=grid_bucket,
        HOIST=hoist,
    )

    if split_k > 1:
        # ONE launch to combine the partials. The whole point of this variant:
        # the previous split-K did this with ~8 torch ops and lost 2.5x at decode.
        _merge_splits_kernel[(B * H,)](
            out_p,
            lse_p,
            out,
            lse,
            H,
            out_p.stride(0),
            out_p.stride(1),
            out_p.stride(2),
            lse_p.stride(0),
            lse_p.stride(1),
            out.stride(0),
            out.stride(1),
            D=D,
            SPLIT_K=split_k,
            S_POW2=triton.next_power_of_2(split_k),
            num_warps=4,
            num_stages=1,
        )

    # Return [B, 1, H, D] and [B, 1, H]
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn_eager(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using LSE-weighted combination.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] float32
    """
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def _apply_attn_sink_eager(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply attention sink normalization.

    The sink adds to the softmax denominator without contributing output,
    effectively down-weighting all attention scores.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] f32,  attn_sink: [H] f32
    """
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20,
        torch.exp(lse - combined_lse),
        torch.zeros_like(lse),
    )
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse



# ---------------------------------------------------------------------------
# Fused elementwise epilogues.
#
# The eager versions above cost ~18 tiny fp32 kernels per layer per forward
# pass (logaddexp / where / exp / fill / mul / copy ...). A decode census on
# this config put that cluster at ~2.1 ms of the 28.6 ms decode step (7.5%),
# all of it *inside* the decode CUDA graph -- so it is real GPU time, not
# launch overhead, and only fusion can remove it. One Triton program per
# (batch, head) row collapses each chain to a single kernel.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_attn_sink_kernel(
    out_ptr,
    lse_ptr,
    sink_ptr,
    o_out_ptr,
    o_lse_ptr,
    H,
    D,
    s_ob,
    s_oh,
    s_lb,
    s_lh,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    lse = tl.load(lse_ptr + b * s_lb + h * s_lh)
    sink = tl.load(sink_ptr + h)

    # logaddexp(lse, sink), stable and -inf safe on both arguments.
    mx = tl.maximum(lse, sink)
    mn = tl.minimum(lse, sink)
    # mx == -inf (both args -inf) would make mn - mx a NaN; short-circuit to mx.
    combined = tl.where(
        mx > -1e30, mx + tl.log(1.0 + tl.exp(mn - mx)), mx
    )
    w = tl.where(lse > -1e20, tl.exp(lse - combined), 0.0)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = b * s_ob + h * s_oh
    o = tl.load(out_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_out_ptr + base + offs, (o * w).to(tl.bfloat16), mask=mask)
    tl.store(o_lse_ptr + b * s_lb + h * s_lh, combined)


@triton.jit
def _fused_merge_partial_kernel(
    o1_ptr,
    l1_ptr,
    o2_ptr,
    l2_ptr,
    o_out_ptr,
    o_lse_ptr,
    H,
    D,
    s_ob,
    s_oh,
    s_lb,
    s_lh,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    loff = b * s_lb + h * s_lh
    l1 = tl.load(l1_ptr + loff)
    l2 = tl.load(l2_ptr + loff)
    mx = tl.maximum(l1, l2)
    w1 = tl.where(l1 > -1e20, tl.exp(l1 - mx), 0.0)
    w2 = tl.where(l2 > -1e20, tl.exp(l2 - mx), 0.0)
    total = tl.maximum(w1 + w2, 1e-20)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = b * s_ob + h * s_oh
    a = tl.load(o1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(o2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    merged = (w1 * a + w2 * c) / total
    tl.store(o_out_ptr + base + offs, merged.to(tl.bfloat16), mask=mask)
    tl.store(o_lse_ptr + loff, mx + tl.log(total))


@triton.jit
def _fused_merge_sink_kernel(
    o1_ptr,
    l1_ptr,
    o2_ptr,
    l2_ptr,
    sink_ptr,
    o_out_ptr,
    o_lse_ptr,
    H,
    D,
    s_ob,
    s_oh,
    s_lb,
    s_lh,
    BLOCK_D: tl.constexpr,
):
    """_fused_merge_partial_kernel + _fused_attn_sink_kernel, in one pass.

    The two always run back-to-back (the sink consumes exactly what the merge
    produces), so the merged tensor is written to DRAM and read straight back
    for no reason: at the prefill shape that intermediate is 75.8 MB, i.e. a
    151.6 MB round-trip per call. Keeping it in registers also rounds
    fp32->bf16 once instead of twice, so this is strictly *more* accurate than
    the two-kernel path (measured: 1.41x closer to an fp64 reference).
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    loff = b * s_lb + h * s_lh
    l1 = tl.load(l1_ptr + loff)
    l2 = tl.load(l2_ptr + loff)
    mx = tl.maximum(l1, l2)
    w1 = tl.where(l1 > -1e20, tl.exp(l1 - mx), 0.0)
    w2 = tl.where(l2 > -1e20, tl.exp(l2 - mx), 0.0)
    total = tl.maximum(w1 + w2, 1e-20)
    lse = mx + tl.log(total)

    # Attention sink, applied to the still-fp32 merge result.
    sink = tl.load(sink_ptr + h)
    smx = tl.maximum(lse, sink)
    smn = tl.minimum(lse, sink)
    combined = tl.where(smx > -1e30, smx + tl.log(1.0 + tl.exp(smn - smx)), smx)
    w = tl.where(lse > -1e20, tl.exp(lse - combined), 0.0)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = b * s_ob + h * s_oh
    a = tl.load(o1_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(o2_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    merged = (w1 * a + w2 * c) / total
    tl.store(o_out_ptr + base + offs, (merged * w).to(tl.bfloat16), mask=mask)
    tl.store(o_lse_ptr + loff, combined)


def _merge_partial_attn_with_sink(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
):
    """Fused merge+sink. Returns None if the fast path does not apply."""
    if attn_sink is None:
        return None
    shp = _epilogue_shapes(out1, lse1)
    if (
        shp is None
        or out2.shape != out1.shape
        or out2.stride() != out1.stride()
        or lse2.stride() != lse1.stride()
        or out1.dtype != torch.bfloat16
        or out2.dtype != torch.bfloat16
        or attn_sink.numel() != out1.shape[2]
    ):
        return None
    B, H, D, s_ob, s_oh, s_lb, s_lh = shp
    sink = attn_sink.contiguous().to(torch.float32)
    merged = torch.empty_like(out1)
    merged_lse = torch.empty_like(lse1)
    _fused_merge_sink_kernel[(B * H,)](
        out1,
        lse1,
        out2,
        lse2,
        sink,
        merged,
        merged_lse,
        H,
        D,
        s_ob,
        s_oh,
        s_lb,
        s_lh,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return merged, merged_lse


def _epilogue_shapes(out, lse):
    """Return (B, H, D, strides, ok) for the [B, 1, H, D] / [B, 1, H] layout."""
    if out.dim() != 4 or lse.dim() != 3 or out.stride(-1) != 1:
        return None
    B, one, H, D = out.shape
    if one != 1 or lse.shape != (B, 1, H) or B * H == 0:
        return None
    return B, H, D, out.stride(0), out.stride(2), lse.stride(0), lse.stride(2)


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LSE-weighted merge of two attention outputs, fused into one kernel."""
    shp = _epilogue_shapes(out1, lse1)
    if (
        shp is None
        or out2.shape != out1.shape
        or out2.stride() != out1.stride()
        or lse2.stride() != lse1.stride()
        or out1.dtype != torch.bfloat16
        or out2.dtype != torch.bfloat16
    ):
        return _merge_partial_attn_eager(out1, lse1, out2, lse2)
    B, H, D, s_ob, s_oh, s_lb, s_lh = shp
    merged = torch.empty_like(out1)
    merged_lse = torch.empty_like(lse1)
    _fused_merge_partial_kernel[(B * H,)](
        out1,
        lse1,
        out2,
        lse2,
        merged,
        merged_lse,
        H,
        D,
        s_ob,
        s_oh,
        s_lb,
        s_lh,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return merged, merged_lse


def _apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention-sink renormalization, fused into one kernel."""
    shp = _epilogue_shapes(out, lse)
    if shp is None or out.dtype != torch.bfloat16 or attn_sink.numel() != out.shape[2]:
        return _apply_attn_sink_eager(out, lse, attn_sink)
    B, H, D, s_ob, s_oh, s_lb, s_lh = shp
    sink = attn_sink.contiguous().to(torch.float32)
    new_out = torch.empty_like(out)
    new_lse = torch.empty_like(lse)
    _fused_attn_sink_kernel[(B * H,)](
        out,
        lse,
        sink,
        new_out,
        new_lse,
        H,
        D,
        s_ob,
        s_oh,
        s_lb,
        s_lh,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return new_out, new_lse


def flash_mla_sparse_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SM120-optimized sparse MLA decode using tiled Triton kernel.

    Processes SWA and extra (c4/c128) caches separately via the same
    Triton kernel, then merges results using LSE-weighted combination.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Process main cache (SWA)
    out, lse = _run_triton_sparse_decode(
        q,
        k_cache,
        indices,
        topk_length,
        softmax_scale,
    )

    # Process extra cache (c4 / c128) if present
    if extra_k_cache is not None and extra_indices is not None:
        out_extra, lse_extra = _run_triton_sparse_decode(
            q,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            softmax_scale,
        )
        if _FUSE_MERGE_SINK:
            fused = _merge_partial_attn_with_sink(
                out, lse, out_extra, lse_extra, attn_sink
            )
            if fused is not None:
                # Both epilogues done; skip the separate sink pass below.
                return fused[0], fused[1].permute(0, 2, 1)
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    # Apply attention sink
    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    # Return format matching PyTorch fallback: (out, lse.permute(0,2,1))
    return out, lse.permute(0, 2, 1)
