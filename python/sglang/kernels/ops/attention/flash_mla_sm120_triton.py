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


# Multi-head variant: one program handles BLOCK_H heads and reads each KV tile
# once. The original grid=(B, H) made all 64 query heads re-read the same KV --
# MLA has a single KV head -- so KV traffic was amplified 64x. Measured on A800
# that lands at 0.07% of peak.
# ────────────────────────────────────────────────────────────────────────────
@triton.jit
def _e4m3_scaled(b, sc):
    """Decode an E4M3 byte and its E8M0 scale byte to float32 in one step.

    The two-step form is: (1) assemble 2^(e-7) * (1 + m/8), (2) multiply by
    exp2(sc - 127). Both steps only move the exponent. The product's exponent is
    (e - 7) + (sc - 127), so writing e + sc - 7 straight into the float32
    exponent field already gives the final value -- one integer add, and both
    the exp2 and the multiply disappear.

      e > 0 : exponent field e + sc - 7, mantissa m << 20
      e == 0: subnormal, value = m * 2^-9 * 2^(sc-127) = m * 2^(sc-136).
              Also done by assembling an exponent: let the float multiply do the
              normalisation of m, against a pure power of two.
    """
    e = (b >> 3) & 0xF
    m = b & 0x7
    # Clamp the assembled exponent to the finite float32 range: garbage scale
    # bytes (e.g. from a masked/never-written slot) would otherwise assemble
    # exponent <= 0 (sign-bit garbage) or >= 255 (inf/NaN). The true value in
    # both regimes underflows/overflows anyway, and masked lanes are zeroed by
    # the caller, so clamping is exact for all reachable inputs.
    exp_n = tl.minimum(tl.maximum(e + sc - 7, 1), 254)
    v = ((exp_n << 23) | (m << 20)).to(tl.float32, bitcast=True)
    # Subnormal: m * 2^(sc-136). Exponent field is sc-136+127 = sc-9; build that
    # power of two and scale it by m. For sc <= 9 the true value underflows to
    # zero; clamp instead of assembling a garbage exponent.
    exp_s = tl.minimum(tl.maximum(sc - 9, 0), 254)
    sub = m.to(tl.float32) * (exp_s << 23).to(tl.float32, bitcast=True)
    sub = tl.where(sc <= 9, 0.0, sub)
    v = tl.where(e == 0, sub, v)
    return tl.where((b >> 7) & 1 == 1, -v, v)


@triton.jit
def _e4m3_to_f32(b):
    """Decode E4M3 (float8_e4m3fn) to float32 with bit arithmetic, no LUT.

    A 256-entry lookup table cost 65% of kernel time (15.02 ms -> 5.28 ms once
    removed): one gather per element, with every lane hitting a different
    address, serialises on shared-memory bank conflicts.

    e4m3fn is 1 sign / 4 exponent (bias 7) / 3 mantissa bits, with no inf.
      e>0 : (-1)^s * 2^(e-7) * (1 + m/8)
      e == 0: (-1)^s * 2^-6 * (m/8) = (-1)^s * m * 2^-9   (subnormal)

    Normals are assembled directly: an exponent field of e + 120 gives
    2^(e+120-127) = 2^(e-7), and the mantissa shifts left by 20 to sit in the
    top 3 bits of float32's 23-bit field. The sign is applied as a float
    negation to avoid overflowing 1 << 31 in int32.

    Note the one behavioural difference from a table: the two NaN encodings
    (0x7F / 0xFF) decode to +-480 rather than NaN. A KV cache should not contain
    NaN, and every other bit pattern is exact.
    """
    e = (b >> 3) & 0xF
    m = b & 0x7
    mag = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    mag = tl.where(e == 0, m.to(tl.float32) * 1.953125e-3, mag)  # 2^-9
    return tl.where((b >> 7) & 1 == 1, -mag, mag)


@triton.autotune(
    configs=[
        # The candidate set follows measurement, not intuition. Broadcasting the
        # scales freed enough shared memory that the best BLOCK_T moved from 32
        # to 16 (9.31 vs 13.42 ms, same cold-cache methodology) -- worth
        # re-sweeping after every change.
        # BLOCK_T=8 does not compile: tl.dot requires a minimum dimension of 16.
        # On num_warps: acc[64, 512] in fp32 is 32768 floats, which over 128
        # threads is 256 registers/thread, past Ampere's 255 limit. warps=4 still
        # measures fastest -- the spill costs less than doubling the threads.
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 64, "LOOP_STAGES": 2}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 64, "LOOP_STAGES": 4}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 64, "LOOP_STAGES": 6}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 64, "LOOP_STAGES": 2}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 64, "LOOP_STAGES": 3}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 16, "BLOCK_H": 32, "LOOP_STAGES": 3}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_T": 32, "BLOCK_H": 32, "LOOP_STAGES": 3}, num_warps=8, num_stages=2
        ),
    ],
    key=["topk_rounded"],
)
@triton.jit
def _tiled_sparse_decode_kernel_mh(
    Q_ptr,
    cache_i32_ptr,
    cache_uint8_ptr,
    cache_bf16_ptr,
    indices_ptr,
    topk_len_ptr,
    O_ptr,
    LSE_ptr,
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    scale_section_off: tl.int64,
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,
    has_topk_len: tl.constexpr,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,
    NOPE_PAD: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM_RT: tl.int32,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    """Grid is (B, ceil(H / BLOCK_H)). Each KV tile is read once; QK and PV both
    reuse it across the head dimension through tl.dot."""
    bid = tl.program_id(0)
    hblk = tl.program_id(1)

    h_offs = hblk * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_offs < H

    nope_offs = tl.arange(0, NOPE_PAD)
    rope_offs = tl.arange(0, ROPE_DIM)

    # ---- Q: [BLOCK_H, NOPE_PAD] / [BLOCK_H, ROPE_DIM] ----
    # sm_scale is applied in fp32 after the dot rather than folded into q, which
    # would round it in bf16 first.
    q_base = bid * stride_qb
    q_ptrs = Q_ptr + q_base + h_offs[:, None] * stride_qh
    # KV is loaded as int32 (4 bytes at a time), which yields an interleaved
    # layout grouped every 4 columns. Rather than de-interleave the KV, q is
    # split the same way into 4 slices and the 4 dots are summed: a dot product
    # does not care about the order of the contracted dimension, so applying the
    # same permutation to both sides is exactly equivalent.
    w_offs = tl.arange(0, NOPE_PAD // 4)  # [128]
    q_n0 = tl.load(q_ptrs + (w_offs * 4 + 0)[None, :], mask=h_mask[:, None], other=0.0)
    q_n1 = tl.load(q_ptrs + (w_offs * 4 + 1)[None, :], mask=h_mask[:, None], other=0.0)
    q_n2 = tl.load(q_ptrs + (w_offs * 4 + 2)[None, :], mask=h_mask[:, None], other=0.0)
    q_n3 = tl.load(q_ptrs + (w_offs * 4 + 3)[None, :], mask=h_mask[:, None], other=0.0)
    q_rope = tl.load(
        q_ptrs + NOPE_DIM_RT + rope_offs[None, :], mask=h_mask[:, None], other=0.0
    )

    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    m_i = tl.full([BLOCK_H], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_n0 = tl.zeros([BLOCK_H, NOPE_PAD // 4], dtype=tl.float32)
    acc_n1 = tl.zeros([BLOCK_H, NOPE_PAD // 4], dtype=tl.float32)
    acc_n2 = tl.zeros([BLOCK_H, NOPE_PAD // 4], dtype=tl.float32)
    acc_n3 = tl.zeros([BLOCK_H, NOPE_PAD // 4], dtype=tl.float32)
    acc_rope = tl.zeros([BLOCK_H, ROPE_DIM], dtype=tl.float32)

    t_offs = tl.arange(0, BLOCK_T)

    # 循环界用 valid_topk(真实有效数)而非全宽 topk:mask 语义与原先逐位一致
    # (t_valid 本就按 valid_topk 截断),只是不再为纯 masked 尾部空转。
    # 全宽 32K token 而实际内容几百 token 时,该内核吃掉 74% GPU 时间。
    loop_end = ((valid_topk + BLOCK_T - 1) // BLOCK_T) * BLOCK_T
    for tile_start in tl.range(0, loop_end, BLOCK_T, num_stages=LOOP_STAGES):
        t_idx = tile_start + t_offs
        t_in_bounds = t_idx < topk
        t_valid = t_idx < valid_topk

        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx, mask=t_in_bounds, other=-1
        )
        idx_valid = t_valid & (raw_indices >= 0)

        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = page_ids * page_bytes + page_offs_t * 576

        # ---- KV nope: dequantise ----
        # Add the int64 base to the pointer first to get a [BLOCK_T] pointer
        # vector, then broadcast the int32 inner offsets. Doing it the other way
        # materialises an int64 [BLOCK_T, 512] address tensor (64 x 512 x 8B =
        # 256 KB of registers), and int64 arithmetic is slow on top of that.
        # Reading 4 bytes at a time cuts the load count to a quarter; both the
        # per-token base and the 576-byte stride are multiples of 4, so the
        # alignment holds.
        w_ptrs = cache_i32_ptr + (token_data_bases // 4)
        kv_w = tl.load(
            w_ptrs[:, None] + w_offs[None, :], mask=idx_valid[:, None], other=0
        )  # [BLOCK_T, 128] int32
        # Zero the tail past NOPE_DIM_RT on the packed word instead of on the four
        # decoded slices: one select replaces four. A zero byte decodes through the
        # subnormal branch to exactly 0.0, so this is not an approximation -- and it
        # must stay a mask rather than be dropped, because the bytes past the nope
        # section are rope bf16 reinterpreted as E4M3 and can decode to inf, which
        # would turn a 0 * inf product into NaN. The row mask above matters for the
        # same reason: invalid lanes are clamped to slot 0, and if that slot was
        # never written the garbage bytes can decode to inf/NaN, which p=0 cannot
        # cancel in the PV dot (0 * inf = NaN).
        kv_w = tl.where((w_offs < (NOPE_DIM_RT // 4))[None, :], kv_w, 0)
        b0 = kv_w & 0xFF
        b1 = (kv_w >> 8) & 0xFF
        b2 = (kv_w >> 16) & 0xFF
        b3 = (kv_w >> 24) & 0xFF

        # Scales: element 4m+j belongs to group (4m+j)//64 = m//16, since j < 4
        # never crosses a 64-element boundary -- so all 4 slices share one index.
        scale_bases = page_ids * page_bytes + scale_section_off + page_offs_t * 8
        scale_base_ptrs = cache_uint8_ptr + scale_bases
        # There are only 8 distinct scale bytes per token, but indexing by
        # (w_offs // 16) issued 128 loads to fetch them. Loading 8 and
        # broadcasting cuts that to a sixteenth.
        # The broadcast expands [BLOCK_T, 8] -> [BLOCK_T, 8, 16] -> [BLOCK_T, 128].
        s8 = tl.load(
            scale_base_ptrs[:, None] + tl.arange(0, 8)[None, :],
            mask=idx_valid[:, None],
            other=127,
        )  # [BLOCK_T, 8]
        scale_raw = tl.reshape(
            tl.broadcast_to(s8[:, :, None], (BLOCK_T, 8, NOPE_PAD // 4 // 8)),
            (BLOCK_T, NOPE_PAD // 4),
        )
        # Tried replacing exp2 with bit assembly here (2^(s-127) is just s in the
        # exponent field): measurably slower, 10.58 -> 12.05 ms. exp2 is a single
        # ex2.approx instruction in hardware, so assembling bits adds work.
        # The scale goes straight into the exponent field, no exp2.
        sc_i = scale_raw.to(tl.int32)  # [BLOCK_T, 128]

        # Zero the tail past 448: element 4m+j < 448 iff m < 112 (448/4).
        # Folding this into the scale to save 4 wheres was also slower; reverted.
        kv0 = _e4m3_scaled(b0, sc_i).to(tl.bfloat16)
        kv1 = _e4m3_scaled(b1, sc_i).to(tl.bfloat16)
        kv2 = _e4m3_scaled(b2, sc_i).to(tl.bfloat16)
        kv3 = _e4m3_scaled(b3, sc_i).to(tl.bfloat16)

        rope_base_ptrs = cache_bf16_ptr + ((token_data_bases + 448) // 2)
        kv_rope = tl.load(
            rope_base_ptrs[:, None] + rope_offs[None, :],
            mask=idx_valid[:, None],
            other=0.0,
        )

        # ---- QK: [BLOCK_H, BLOCK_T], 4 nope slices plus 1 rope slice ----
        scores = tl.dot(q_n0, tl.trans(kv0))
        scores += tl.dot(q_n1, tl.trans(kv1))
        scores += tl.dot(q_n2, tl.trans(kv2))
        scores += tl.dot(q_n3, tl.trans(kv3))
        scores += tl.dot(q_rope, tl.trans(kv_rope))
        scores = scores * sm_scale
        scores = tl.where(idx_valid[None, :], scores, -1e30)

        scores_log2 = scores * LOG2E
        tile_max = tl.max(scores_log2, axis=1)  # [BLOCK_H]
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.math.exp2(m_i - m_new)  # [BLOCK_H]
        p = tl.math.exp2(scores_log2 - m_new[:, None])  # [BLOCK_H, BLOCK_T]
        p = tl.where(idx_valid[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_b = p.to(tl.bfloat16)
        acc_n0 = acc_n0 * alpha[:, None] + tl.dot(p_b, kv0)
        acc_n1 = acc_n1 * alpha[:, None] + tl.dot(p_b, kv1)
        acc_n2 = acc_n2 * alpha[:, None] + tl.dot(p_b, kv2)
        acc_n3 = acc_n3 * alpha[:, None] + tl.dot(p_b, kv3)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(p_b, kv_rope)
        m_i = m_new

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_rope = acc_rope / safe_l[:, None]
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    o_ptrs = O_ptr + bid * stride_ob + h_offs[:, None] * stride_oh
    w_keep_o = (w_offs < (NOPE_DIM_RT // 4))[None, :]
    om = h_mask[:, None] & w_keep_o
    tl.store(
        o_ptrs + (w_offs * 4 + 0)[None, :],
        (acc_n0 / safe_l[:, None]).to(tl.bfloat16),
        mask=om,
    )
    tl.store(
        o_ptrs + (w_offs * 4 + 1)[None, :],
        (acc_n1 / safe_l[:, None]).to(tl.bfloat16),
        mask=om,
    )
    tl.store(
        o_ptrs + (w_offs * 4 + 2)[None, :],
        (acc_n2 / safe_l[:, None]).to(tl.bfloat16),
        mask=om,
    )
    tl.store(
        o_ptrs + (w_offs * 4 + 3)[None, :],
        (acc_n3 / safe_l[:, None]).to(tl.bfloat16),
        mask=om,
    )
    tl.store(
        o_ptrs + NOPE_DIM_RT + rope_offs[None, :],
        acc_rope.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(LSE_ptr + bid * H + h_offs, lse, mask=h_mask)


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
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    # Squeeze Q: [B, H, D]
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    # Round topk for autotune key stability
    topk_rounded = triton.next_power_of_2(topk)

    # Multi-head kernel: a batch of heads per program, KV read once per tile.
    # The first argument becomes an int32 view of the same buffer.
    raw_i32 = raw_uint8.view(torch.int32)
    grid = lambda META: (B, triton.cdiv(H, META["BLOCK_H"]))
    _tiled_sparse_decode_kernel_mh[grid](
        q3,
        raw_i32,
        raw_uint8,
        raw_bf16,
        flat_indices,
        (
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q.device, dtype=torch.int32)
        ),
        out,
        lse,
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
        out.stride(0),
        out.stride(1),
        flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )

    # Return [B, 1, H, D] and [B, 1, H]
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
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


def _apply_attn_sink(
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
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    # Apply attention sink
    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    # Return format matching PyTorch fallback: (out, lse.permute(0,2,1))
    return out, lse.permute(0, 2, 1)
