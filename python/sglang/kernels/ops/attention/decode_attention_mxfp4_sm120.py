# SPDX-License-Identifier: Apache-2.0
"""MXFP4 native decode kernels.

A standalone Triton kernel for the OCP MXFP4 KV cache (block-32 E2M1 data
+ per-block E8M0 scale), consumed directly from the packed pool buffers --
no PLAIN materialization. ``mxfp4_decode_attention_fwd`` is a two-stage
split-KV decode attention that mirrors the bf16/fp8 kernel structure in
``decode_attention.py`` (``_fwd_kernel_stage1`` + ``_fwd_kernel_stage2``) but
loads the packed bytes per token/head and dequantizes inline (fp32 math
throughout).

Codec semantics (must stay bit-compatible with the production codec and the
oracle): low nibble of a packed byte = even-index element, high nibble =
odd-index element; E2M1 code magnitude table (0, .5, 1, 1.5, 2, 3, 4, 6)
with bit 3 = sign; E8M0 scale = 2^(byte - 127). The code x scale product is
applied through a 4096-entry lookup table (``e2m1_scale_lut``) whose values
are produced by the exact bit-construction routine ``_e2m1_scale_to_bf16``.
A NaN scale byte (0xFF) propagates NaN exactly like the oracle.

Intentionally NOT implemented: lean attention, score_mod,
DCP, sliding window, logit cap, xai temperature, MLA, extend/prefill.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides

MXFP4_BLOCK_SIZE = 32
# Kernel-side mirror of MXFP4_BLOCK_SIZE: @triton.jit functions may only read
# tl.constexpr globals. The half-block form is the same bound counted in packed
# bytes, used by the nibble-interleaved loads.
_MXFP4_BLOCK = tl.constexpr(MXFP4_BLOCK_SIZE)
_MXFP4_HALF_BLOCK = tl.constexpr(MXFP4_BLOCK_SIZE // 2)
_MIN_BLOCK_KV = 32
_BLOCK_N = 64
_GROUPED_BLOCK_H = 16

# Production launch config for the two stage-1 paths, plus the compiled
# footprint that decides how many CTAs reside per SM. These live next to the
# kernels because the attention backend derives its kv-split schedule from
# them (it aims for a whole number of CTA waves), so retuning a kernel without
# updating these would silently desync that schedule.
#
# Measured on SM120 / triton 3.7.1 (24Q/4KV, head_dim 256, BLOCK_N 64):
#   grouped, nibble-interleaved : 4 warps, 178 regs, 0 spills, 43,008 B smem
#   per-head (MHA, fp32 LUT)    : 8 warps, 255 regs, 16 spills, 43,008 B smem
# The grouped path needs only half the threads of the per-head one and stays
# well under the 255-register ISA ceiling, so two of its CTAs fit per SM.
_GROUPED_NUM_WARPS = 4
_GROUPED_REGS_PER_THREAD = 178
_PERHEAD_NUM_WARPS = 8
_PERHEAD_REGS_PER_THREAD = 255
_STAGE1_SMEM_BYTES = 43008


def stage1_launch_config(kv_group_num: int, device=None) -> tuple[int, int]:
    """``(num_warps, ctas_per_sm)`` for the stage-1 path ``kv_group_num`` picks.

    ``ctas_per_sm`` is the residency the compiled kernel actually achieves on
    this device, derived from the measured register/shared-memory footprint
    above. Callers use it to size a full CTA wave; it is a scheduling hint
    only, never a correctness input.
    """
    if kv_group_num > 1:
        num_warps, regs = _GROUPED_NUM_WARPS, _GROUPED_REGS_PER_THREAD
    else:
        num_warps, regs = _PERHEAD_NUM_WARPS, _PERHEAD_REGS_PER_THREAD
    props = torch.cuda.get_device_properties(device)
    by_regs = props.regs_per_multiprocessor // (regs * num_warps * 32)
    by_smem = props.shared_memory_per_multiprocessor // _STAGE1_SMEM_BYTES
    return num_warps, max(1, min(by_regs, by_smem))


@triton.jit
def _e2m1_scale_to_bf16(code, sbytes):
    """Bit-exact E2M1 code x E8M0 scale -> bf16 via pure integer ops.

    value = mag(code) * 2^(b - 127). For normal E2M1 (e >= 1) the bf16 bit
    pattern is exp field = e + b - 1, mantissa = m << 6; the E2M1 subnormal
    (e == 0, m == 1) is 2^(b - 128), exp field b - 1, mantissa 0; e == 0 and
    m == 0 is a signed zero. The product crosses bf16's normal/subnormal
    boundary when the exponent field is 0 (value 2^-127) -> subnormal
    encoding mant = 1 << (E + 6). Out-of-range scales encode inf; an E8M0
    NaN byte (0xFF) encodes a quiet NaN (highest priority, matching the
    oracle's all-NaN block). Verified against the OCP oracle over all
    16 codes x 256 scale bytes: element-exact except scale byte 0 with
    NONZERO codes (fp32-subnormal domain where the oracle's torch.exp2
    carries a 1-ULP error; production pairs byte 0 only with all-zero
    codes, where both sides give exact zeros). Pure int32 ALU: no SFU exp2,
    no fp32 multiply in the dequant hot path.
    """
    e = (code >> 1) & 3
    m = (code & 1).to(tl.int32)
    sgn = (code & 8).to(tl.int32) << 12  # sign bit 3 -> bf16 bit 15
    b = sbytes.to(tl.int32)

    is_zero = (e == 0) & (m == 0)
    E = tl.where(e == 0, b - 1, e + b - 1)
    mant = tl.where(e == 0, 0, m << 6)

    is_sub = E < 1  # value 2^(E-127) < 2^-126: bf16 subnormal encoding
    sub_mant = tl.where(E >= -6, 1 << tl.minimum(E + 6, 31), 0)
    is_inf = E >= 255
    is_nan = b == 255

    normal_bits = tl.where(E > 254, 0x7F80, (E << 7) | mant)
    bits = tl.where(is_sub, sub_mant, normal_bits)
    bits = tl.where(is_inf, 0x7F80, bits)
    bits = tl.where(is_zero, 0, bits)
    bits = tl.where(is_nan, 0x7FC0, bits)
    bits = bits | sgn
    return (bits & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)


# ---------------------------------------------------------------------------
# E2M1 code x E8M0 scale lookup table
# ---------------------------------------------------------------------------
# The dequantized value is a pure function of 4 code bits + 8 scale bits, so
# the entire domain is 4096 entries (8 KiB in bf16 / 16 KiB in fp32 -- hot in
# L1 for every program). One gather replaces the ~20-op bit-construction chain
# above, of which the online-softmax decode loop runs two instances per
# (token, block) for K and V.
#
# The table is built by ``_e2m1_scale_to_bf16`` itself, so it is bit-identical
# to the ALU path by construction: the E8M0 NaN byte, the fp32-subnormal scale
# (2^-127) and the bf16 subnormal/inf encodings all come from the same code.
# Index layout ``(scale_byte << 4) | code`` keeps the 16 codes of one block in
# one 32 B sector (bf16), i.e. blocks read one sector per gather.
_LUT_SIZE = 1 << 12
_LUT_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


@triton.jit
def _build_e2m1_scale_lut_kernel(LUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    code = offs & 0x0F
    sbyte = (offs >> 4).to(tl.uint8)
    tl.store(LUT + offs, _e2m1_scale_to_bf16(code, sbyte))


def e2m1_scale_lut(device=None, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``(scale_byte << 4) | e2m1_code`` -> dequantized value, built on first use.

    Cached per (device index, dtype); the fp32 variant is an exact widening of
    the bf16 one (E2M1 x power-of-two never needs more than 8 significand bits).
    """
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    key = (dev.index, dtype)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        table = torch.empty(_LUT_SIZE, dtype=torch.bfloat16, device=dev)
        _build_e2m1_scale_lut_kernel[(1,)](table, BLOCK=_LUT_SIZE, num_warps=4)
        lut = table.to(dtype)
        _LUT_CACHE[key] = lut
    return lut


# ---------------------------------------------------------------------------
# Native MXFP4 decode attention (two-stage split-KV)
# ---------------------------------------------------------------------------


@triton.jit
def _mxfp4_decode_stage1_kernel(
    Q,
    K_Packed,
    V_Packed,
    K_Scales,
    V_Scales,
    E2M1_LUT,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kp_bs,
    stride_kp_h,
    stride_kp_page,
    stride_kp_tok,
    stride_vp_bs,
    stride_vp_h,
    stride_vp_page,
    stride_vp_tok,
    stride_ks_bs,
    stride_ks_h,
    stride_vs_bs,
    stride_vs_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    PACKED_K_DIM: tl.constexpr,
    PACKED_V_DIM: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    # int64 to mirror the stock kernel's flat-offset overflow guard.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    byte_idx_k = offs_d // 2
    byte_idx_v = offs_dv // 2
    blk_idx_k = offs_d // _MXFP4_BLOCK
    blk_idx_v = offs_dv // _MXFP4_BLOCK

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0).to(tl.float32)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=tok_mask,
                other=0,
            )
            # Page-aware address math, identical to the stock stage-1 kernel
            # (for these buffers it reduces to kv_loc * slot_stride).
            if PAGE_SIZE == 1:
                k_row = kv_loc * stride_kp_bs + cur_kv_head * stride_kp_h
                v_row = kv_loc * stride_vp_bs + cur_kv_head * stride_vp_h
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                k_row = (
                    page_id * stride_kp_page
                    + tok_in_p * stride_kp_tok
                    + cur_kv_head * stride_kp_h
                )
                v_row = (
                    page_id * stride_vp_page
                    + tok_in_p * stride_vp_tok
                    + cur_kv_head * stride_vp_h
                )
            ks_row = kv_loc * stride_ks_bs + cur_kv_head * stride_ks_h
            vs_row = kv_loc * stride_vs_bs + cur_kv_head * stride_vs_h

            # --- K tile: packed bytes -> E2M1 code -> fused dequant LUT (fp32)
            kbyte = tl.load(
                K_Packed + k_row[:, None] + byte_idx_k[None, :],
                mask=tok_mask[:, None] & (byte_idx_k[None, :] < PACKED_K_DIM),
                other=0,
            )
            is_odd_k = (offs_d % 2) == 1
            kcode = tl.where(is_odd_k[None, :], (kbyte >> 4) & 0x0F, kbyte & 0x0F)

            ksbytes = tl.load(
                K_Scales + ks_row[:, None] + blk_idx_k[None, :],
                mask=tok_mask[:, None] & (blk_idx_k[None, :] < NUM_K_BLOCKS),
                other=0,
            )
            # E2M1 code x E8M0 scale -> fp32 in one gather (see e2m1_scale_lut)
            k = tl.load(E2M1_LUT + ((ksbytes.to(tl.int32) << 4) | kcode))
            k = tl.where(mask_d[None, :], k, 0.0)

            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale
            qk = tl.where(tok_mask, qk, float("-inf"))

            # --- V tile: same inline dequant
            vbyte = tl.load(
                V_Packed + v_row[:, None] + byte_idx_v[None, :],
                mask=tok_mask[:, None] & (byte_idx_v[None, :] < PACKED_V_DIM),
                other=0,
            )
            is_odd_v = (offs_dv % 2) == 1
            vcode = tl.where(is_odd_v[None, :], (vbyte >> 4) & 0x0F, vbyte & 0x0F)

            vsbytes = tl.load(
                V_Scales + vs_row[:, None] + blk_idx_v[None, :],
                mask=tok_mask[:, None] & (blk_idx_v[None, :] < NUM_V_BLOCKS),
                other=0,
            )
            v = tl.load(E2M1_LUT + ((vsbytes.to(tl.int32) << 4) | vcode))
            v = tl.where(mask_dv[None, :], v, 0.0)

            # --- online softmax, fp32 (same recurrence as the stock kernel)
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _mxfp4_grouped_decode_stage1_kernel(
    Q,
    K_Packed,
    V_Packed,
    K_Scales,
    V_Scales,
    E2M1_LUT,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kp_bs,
    stride_kp_h,
    stride_kp_page,
    stride_kp_tok,
    stride_vp_bs,
    stride_vp_h,
    stride_vp_page,
    stride_vp_tok,
    stride_ks_bs,
    stride_ks_h,
    stride_vs_bs,
    stride_vs_h,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    q_head_num: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HALF_D: tl.constexpr,
    HALF_DV: tl.constexpr,
    PACKED_K_DIM: tl.constexpr,
    PACKED_V_DIM: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_V_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    # Grouped-head variant of ``_mxfp4_decode_stage1_kernel`` (GQA/MQA): one
    # program owns one kv head and serves its whole query group, so the
    # packed K/V bytes are unpacked ONCE per (token, kv_head) instead of once
    # per query head. The per-head grid above re-reads them kv_group_num
    # times, which dominated TPOT at long context. Math follows the stock
    # grouped kernel: Q/K/V tiles in bf16 (EXACT for dequantized E2M1 values:
    # 2-bit mantissa x power-of-two scale always fits bf16's 8-bit
    # significand), tl.dot with fp32 accumulation, online softmax in fp32.
    #
    # Tiles are nibble-interleaved: each packed byte is loaded once and both of
    # its nibbles are expanded in place, instead of addressing elements by
    # ``offs_d // 2`` (which reads every byte twice and needs a select to pick
    # the nibble). See the comment at the index setup below.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    heads_per_kv = tl.cdiv(kv_group_num, BLOCK_H)
    cur_kv_head = cur_head_id // heads_per_kv
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # Nibble-interleaved element order: lane j of the low half holds logical
    # element 2j, lane j of the high half holds 2j + 1. Both halves index the
    # SAME packed byte and the SAME E8M0 block -- a 32-element block spans 16
    # packed bytes, and 2j + 1 (odd) can never cross a block boundary that 2j
    # did not -- so one byte load plus one scale load serve both nibbles.
    # The contraction in ``tl.dot(q, k^T)`` is invariant under a permutation of
    # the K dim, so Q is simply loaded in the same even/odd order. For V the
    # permutation lands on the OUTPUT dim instead, and is undone by the two
    # strided stores at the end of the kernel.
    offs_half = tl.arange(0, HALF_D)
    offs_halfv = tl.arange(0, HALF_DV)
    offs_k_lo = offs_half * 2
    offs_k_hi = offs_k_lo + 1
    offs_v_lo = offs_halfv * 2
    offs_v_hi = offs_v_lo + 1
    mask_k_lo = offs_k_lo < Lk
    mask_k_hi = offs_k_hi < Lk
    mask_v_lo = offs_v_lo < Lv
    mask_v_hi = offs_v_hi < Lv
    byte_mask_k = offs_half < PACKED_K_DIM
    byte_mask_v = offs_halfv < PACKED_V_DIM
    blk_idx_k = offs_half // _MXFP4_HALF_BLOCK
    blk_idx_v = offs_halfv // _MXFP4_HALF_BLOCK

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q_base = cur_batch * stride_qbs + cur_head[:, None] * stride_qh
    offs_q_lo = off_q_base + offs_k_lo[None, :]
    offs_q_hi = off_q_base + offs_k_hi[None, :]

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_H, HALF_DV], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_H, HALF_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q_lo = tl.load(
            Q + offs_q_lo, mask=mask_h[:, None] & mask_k_lo[None, :], other=0.0
        )
        q_hi = tl.load(
            Q + offs_q_hi, mask=mask_h[:, None] & mask_k_hi[None, :], other=0.0
        )

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=tok_mask,
                other=0,
            )
            if PAGE_SIZE == 1:
                k_row = kv_loc * stride_kp_bs + cur_kv_head * stride_kp_h
                v_row = kv_loc * stride_vp_bs + cur_kv_head * stride_vp_h
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                k_row = (
                    page_id * stride_kp_page
                    + tok_in_p * stride_kp_tok
                    + cur_kv_head * stride_kp_h
                )
                v_row = (
                    page_id * stride_vp_page
                    + tok_in_p * stride_vp_tok
                    + cur_kv_head * stride_vp_h
                )
            ks_row = kv_loc * stride_ks_bs + cur_kv_head * stride_ks_h
            vs_row = kv_loc * stride_vs_bs + cur_kv_head * stride_vs_h

            # --- K: one byte load + one scale load feed two LUT gathers (see
            # e2m1_scale_lut); the group shares the resulting bf16 tiles.
            kbyte = tl.load(
                K_Packed + k_row[:, None] + offs_half[None, :],
                mask=tok_mask[:, None] & byte_mask_k[None, :],
                other=0,
            )
            ksbytes = tl.load(
                K_Scales + ks_row[:, None] + blk_idx_k[None, :],
                mask=tok_mask[:, None] & (blk_idx_k[None, :] < NUM_K_BLOCKS),
                other=0,
            )
            k_scale_base = ksbytes.to(tl.int32) << 4
            k_lo = tl.load(E2M1_LUT + (k_scale_base | (kbyte & 0x0F)))
            k_hi = tl.load(E2M1_LUT + (k_scale_base | ((kbyte >> 4) & 0x0F)))
            k_lo = tl.where(mask_k_lo[None, :], k_lo, 0.0)
            k_hi = tl.where(mask_k_hi[None, :], k_hi, 0.0)

            # The two halves are disjoint slices of the same contraction, so
            # their partial dots simply add.
            qk = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))
            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                qk,
                float("-inf"),
            )

            # --- V: same single-load expansion; p goes to bf16 for the dot
            vbyte = tl.load(
                V_Packed + v_row[:, None] + offs_halfv[None, :],
                mask=tok_mask[:, None] & byte_mask_v[None, :],
                other=0,
            )
            vsbytes = tl.load(
                V_Scales + vs_row[:, None] + blk_idx_v[None, :],
                mask=tok_mask[:, None] & (blk_idx_v[None, :] < NUM_V_BLOCKS),
                other=0,
            )
            v_scale_base = vsbytes.to(tl.int32) << 4
            v_lo = tl.load(E2M1_LUT + (v_scale_base | (vbyte & 0x0F)))
            v_hi = tl.load(E2M1_LUT + (v_scale_base | ((vbyte >> 4) & 0x0F)))
            v_lo = tl.where(mask_v_lo[None, :], v_lo, 0.0)
            v_hi = tl.where(mask_v_hi[None, :], v_hi, 0.0)

            # --- online softmax (fp32), same recurrence as the stock kernel
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            p_cast = p.to(v_lo.dtype)
            acc_lo *= re_scale[:, None]
            acc_lo += tl.dot(p_cast, v_lo)
            acc_hi *= re_scale[:, None]
            acc_hi += tl.dot(p_cast, v_hi)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # Undo the V-side permutation: the two accumulators own the even and
        # odd output dims, so they go out as two strided stores.
        offs_mid_base = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        tl.store(
            Att_Out + offs_mid_base + offs_v_lo[None, :],
            acc_lo / e_sum[:, None],
            mask=mask_h[:, None] & mask_v_lo[None, :],
        )
        tl.store(
            Att_Out + offs_mid_base + offs_v_hi[None, :],
            acc_hi / e_sum[:, None],
            mask=mask_h[:, None] & mask_v_hi[None, :],
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _mxfp4_decode_stage2_kernel(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    # Codec-agnostic split merge; mirrors the stock ``_fwd_kernel_stage2``
    # (v_scale fixed at 1.0: MXFP4 scales are applied inline in stage 1).
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _host_num_kv_splits(seq_lens_cpu, max_kv_splits):
    """Test/prototype split schedule: fill up to max_kv_splits with at least
    MIN_BLOCK_KV tokens per split. The production backend brings its own
    on-device schedule (Phase 2); correctness holds for any split count."""
    splits = []
    for seq_len in seq_lens_cpu:
        seq_len = int(seq_len)
        splits.append(max(1, min(max_kv_splits, triton.cdiv(seq_len, _MIN_BLOCK_KV))))
    return splits


def mxfp4_decode_attention_fwd(
    q: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    page_size: int = 1,
    max_kv_splits: int = 8,
    attn_logits: torch.Tensor | None = None,
    attn_lse: torch.Tensor | None = None,
    num_kv_splits: torch.Tensor | None = None,
    block_n: int = _BLOCK_N,
    num_warps: int | None = None,
):
    """Native MXFP4 decode attention over packed paged KV.

    Args:
        q: (batch, q_heads, head_dim), bf16/fp16/fp32.
        k_packed / v_packed: (slots, kv_heads, ceil(head_dim/2)) uint8.
        k_scales / v_scales: (slots, kv_heads, head_dim//32) E8M0 bytes.
        o: (batch, q_heads, v_head_dim) output buffer; written in its dtype
            (fp32 for oracle comparison).
        kv_indptr / kv_indices: standard ragged page-table metadata (indices
            are physical slot ids; page_size > 1 uses the same page/tok
            address math as the stock triton decode kernel).
        attn_logits / attn_lse / num_kv_splits: optional preallocated
            split-KV scratch (CUDA-graph reuse in the backend integration).

    Note: ``num_kv_splits=None`` derives the schedule on the host (prototype
    convenience; involves a device->host sync). The backend passes its
    on-device schedule instead. ``num_warps=None`` takes the per-path
    production value from ``stage1_launch_config``; pass an explicit value
    only to sweep configs.
    """
    batch, head_num, head_dim = q.shape
    kv_head_num = k_packed.shape[1]
    if head_num % kv_head_num != 0:
        raise ValueError(
            f"GQA requires q_heads % kv_heads == 0, got {head_num} % {kv_head_num}"
        )
    kv_group_num = head_num // kv_head_num
    if num_warps is None:
        num_warps, _ = stage1_launch_config(kv_group_num, q.device)
    logical_k = head_dim
    logical_v = o.shape[-1]
    packed_k_dim = k_packed.shape[-1]
    packed_v_dim = v_packed.shape[-1]
    if packed_k_dim != (logical_k + 1) // 2 or packed_v_dim != (logical_v + 1) // 2:
        raise ValueError(
            f"packed dims {packed_k_dim}/{packed_v_dim} do not match logical "
            f"dims {logical_k}/{logical_v}"
        )
    if q.shape[0] != kv_indptr.shape[0] - 1:
        raise ValueError("kv_indptr must have batch + 1 entries")

    k_packed = k_packed.view(torch.uint8).contiguous()
    v_packed = v_packed.view(torch.uint8).contiguous()
    k_scales = k_scales.view(torch.uint8).contiguous()
    v_scales = v_scales.view(torch.uint8).contiguous()
    num_k_blocks = k_scales.shape[-1]
    num_v_blocks = v_scales.shape[-1]

    if num_kv_splits is None:
        seq_lens_cpu = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
        splits = _host_num_kv_splits(seq_lens_cpu, max_kv_splits)
        num_kv_splits = torch.tensor(splits, dtype=torch.int32, device=q.device)
    # Same order as ``attn_logits``/``attn_lse`` liveness: the LUT is a tiny
    # module-level cache, materialized on first decode of the process.
    lut_dtype = torch.float32 if kv_group_num == 1 else torch.bfloat16
    e2m1_lut = e2m1_scale_lut(q.device, lut_dtype)
    if attn_logits is None:
        attn_logits = torch.empty(
            (batch, head_num, max_kv_splits, logical_v),
            dtype=torch.float32,
            device=q.device,
        )
    if attn_lse is None:
        attn_lse = torch.empty(
            (batch, head_num, max_kv_splits), dtype=torch.float32, device=q.device
        )
    if attn_logits.shape[-1] != logical_v:
        raise ValueError("attn_logits last dim must equal the V head dim")

    BLOCK_DMODEL = triton.next_power_of_2(logical_k)
    BLOCK_DV = triton.next_power_of_2(logical_v)

    (
        kp_slot_stride,
        kp_head_stride,
        kp_page_stride,
        kp_tok_stride,
    ) = _extract_kv_strides(k_packed, page_size)
    (
        vp_slot_stride,
        vp_head_stride,
        vp_page_stride,
        vp_tok_stride,
    ) = _extract_kv_strides(v_packed, page_size)
    ks_slot_stride, ks_head_stride, _, _ = _extract_kv_strides(k_scales, 1)
    vs_slot_stride, vs_head_stride, _, _ = _extract_kv_strides(v_scales, 1)

    if kv_group_num == 1:
        # MHA: one query head per kv head -> no redundant K/V reads; the
        # per-head kernel keeps exact element-wise fp32 math.
        grid = (batch, head_num, max_kv_splits)
        _mxfp4_decode_stage1_kernel[grid](
            q,
            k_packed,
            v_packed,
            k_scales,
            v_scales,
            e2m1_lut,
            sm_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            kp_slot_stride,
            kp_head_stride,
            kp_page_stride,
            kp_tok_stride,
            vp_slot_stride,
            vp_head_stride,
            vp_page_stride,
            vp_tok_stride,
            ks_slot_stride,
            ks_head_stride,
            vs_slot_stride,
            vs_head_stride,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            PACKED_K_DIM=packed_k_dim,
            PACKED_V_DIM=packed_v_dim,
            NUM_K_BLOCKS=num_k_blocks,
            NUM_V_BLOCKS=num_v_blocks,
            BLOCK_N=block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            Lk=logical_k,
            Lv=logical_v,
            PAGE_SIZE=page_size,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        # GQA/MQA: grouped kernel unpacks the packed K/V once per kv head and
        # serves the whole query group with tl.dot (see kernel docstring).
        valid_block_h = min(_GROUPED_BLOCK_H, kv_group_num)
        head_tiles = triton.cdiv(head_num, valid_block_h)
        grid = (batch, head_tiles, max_kv_splits)
        _mxfp4_grouped_decode_stage1_kernel[grid](
            q,
            k_packed,
            v_packed,
            k_scales,
            v_scales,
            e2m1_lut,
            sm_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            kp_slot_stride,
            kp_head_stride,
            kp_page_stride,
            kp_tok_stride,
            vp_slot_stride,
            vp_head_stride,
            vp_page_stride,
            vp_tok_stride,
            ks_slot_stride,
            ks_head_stride,
            vs_slot_stride,
            vs_head_stride,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            q_head_num=head_num,
            kv_group_num=kv_group_num,
            BLOCK_H=_GROUPED_BLOCK_H,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            # Nibble-interleaved tiles address packed bytes, so each lane owns
            # two logical elements.
            HALF_D=max(1, BLOCK_DMODEL // 2),
            HALF_DV=max(1, BLOCK_DV // 2),
            PACKED_K_DIM=packed_k_dim,
            PACKED_V_DIM=packed_v_dim,
            NUM_K_BLOCKS=num_k_blocks,
            NUM_V_BLOCKS=num_v_blocks,
            BLOCK_N=block_n,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            Lk=logical_k,
            Lv=logical_v,
            PAGE_SIZE=page_size,
            num_warps=num_warps,
            num_stages=2,
        )

    _mxfp4_decode_stage2_kernel[(batch, head_num)](
        attn_logits,
        attn_lse,
        o,
        kv_indptr,
        num_kv_splits,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=logical_v,
    )
