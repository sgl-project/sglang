# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# UltraQuant D=256 decode (FlyDSL) — gfx950 / CDNA4
#
# Scaled FP4×E4M3 QK MFMA, native V CVT, HW V-transpose, strided tile-groups,
# and an in-kernel Walsh-Hadamard query rotation for GQA 6/8/16.
#
# Reads four SoA buffers per (slot, head), D=256, group_size=32:
#   k_code/v_code  [size, Hk, 128]  FP4 E2M1 nibbles, 2 head-dims per byte
#   k_scale/v_scale[size, Hk,   8]  UE8M0 exponent byte per group of 32
# Slot ids come from kv_indices (page_size 1), sequence extents from kv_indptr.
# Byte offsets are 32-bit buffer voffsets; the launcher enforces the bound.
# QK MFMA: A=K, B=Q → C[token, query]; PV MFMA: A=V_T, B=P → C[head_dim, query]

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops, vector
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import scf as _scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    gpu,
    rocdl,
)
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def _vector_insert(value, dest, *, static_position):
    """Insert a DSL scalar using FlyDSL 0.3's raw vector dialect binding."""
    return vector.insert(
        vector.as_ir_value(value),
        vector.as_ir_value(dest),
        dynamic_position=[],
        static_position=static_position,
    )


# === Constants (256-wide-head ultraquant decode profile) =======================
HEAD_SIZE = 256
TILE_SIZE = 16  # MFMA tile = 16 tokens
WARP_SIZE = 64

FP8_GROUP_SIZE = 32
N_GROUPS = HEAD_SIZE // FP8_GROUP_SIZE  # 8 UE8M0 scale bytes per head
KEY_CODE_BYTES = HEAD_SIZE // 2  # 128

MFMA_N = 16
PV_N_CHUNKS = HEAD_SIZE // MFMA_N  # 16 for HEAD_SIZE=256

# --- Per-lane data-movement geometry ----------------------------------------
# Each token's packed code bytes are split across the 4 ``chunk_in_tok``
# sub-lanes (lane % 4): 32 B per lane, loaded as ``HALVES``=2 dwordx4 loads,
# each covering exactly one UE8M0 group of 32 head-dims.
LANE_CODE_BYTES = KEY_CODE_BYTES // 4  # 32 code bytes per lane per K-tile
HALVES = LANE_CODE_BYTES // 16  # 2 for HEAD_SIZE=256
HALF_HDIMS = 32  # head-dims per half (= one group)
SUBCHUNK_HDIMS = HEAD_SIZE // 4  # head-dims per chunk_in_tok sub-lane (64)

# KV LDS rows are padded by 16 B to reduce bank conflicts while preserving
# ds_read/ds_write_b128 alignment.
KV_ROW_PAD_ELEMS = 8
assert KV_ROW_PAD_ELEMS % 8 == 0, "KV row pad must be a multiple of 8 bf16 (16 B)"
KV_ROW_ELEMS = HEAD_SIZE + KV_ROW_PAD_ELEMS  # 264 @ pad 8
KV_ROW_BYTES = KV_ROW_ELEMS * 2  # 528
KFP4_ROW_PAD_I32 = 4
assert KFP4_ROW_PAD_I32 % 4 == 0, "FP4-K row pad must be a multiple of 4 i32 (16 B)"
KFP4_ROW_I32 = HEAD_SIZE // 8 + KFP4_ROW_PAD_I32
KV_TILE_LDS_BYTES_PADDED = TILE_SIZE * KV_ROW_BYTES  # 8448 (V dominates)
SCALE_LDS_BYTES = TILE_SIZE * N_GROUPS * 4  # 16*8*4 = 512 for 256

LOG2E = 1.4426950408889634
NEG_INF_VAL = float("-inf")


def _vsplat_mul(vec, scalar):
    s = scalar.ir_value() if hasattr(scalar, "ir_value") else scalar
    return vec * vector.broadcast(T.f32x4, s)


@functools.cache
def create_ultraquant_decode_hd256_kernel(
    num_kv_heads: int,
    num_partitions: int,
    block_kv: int,
    softmax_scale: float,
    query_group_size: int,
    stride_q_seq: int,
    stride_q_head: int,
    split_stride: int,
    work_budget: int = 0,
    seqs_per_lane: int = 1,
):
    """Build the kernel and wrap it in a launchable entry point.

    With ``work_budget`` zero, each sequence runs on ``num_partitions``
    partitions and partition p walks the ``block_kv``-token blocks p, p + P,
    p + 2P, ... of it; the launch grid is (batch, Hk, P).

    With a nonzero ``work_budget`` W the batch's tokens are cut into chunks of
    C = ceil(batch_tokens / ((W - batch) * block_kv)) * block_kv tokens (at
    least one block), and sequence s owns workgroups g(s) .. g(s) +
    ceil(len_s / C) - 1, with g(s) = (kv_indptr[s] - kv_indptr[0]) // C + s.
    Rounding each sequence up to whole chunks costs at most one workgroup per
    sequence, so g stays below W. Workgroup g walks its chunk's blocks in order
    and writes split row g of ``[rows, Hq, 1, D]`` buffers, so every workgroup
    has the same work and each sequence's chunks sit next to each other in
    launch order. The grid is (batch + W, Hk, 1), keeping W a build constant;
    the last workgroups find no chunk.

    Batch size is a launch argument rather than a build parameter, so every
    CUDA-graph batch bucket reuses one compiled variant.
    """
    # GQA 6 reuses the 8-row WHT lane map; rows past QG are discarded by the
    # `mfma_row < QG` output gate.
    assert query_group_size in (6, 8, 16), (
        f"query_group_size must be 6, 8 or 16; got {query_group_size}"
    )
    assert split_stride >= num_partitions
    assert block_kv > 0 and block_kv % TILE_SIZE == 0
    assert work_budget >= 0

    NUM_PARTS_C = num_partitions
    CHUNKED = work_budget > 0
    assert not CHUNKED or (num_partitions == 1 and split_stride == 1)
    SEQS_PER_LANE = seqs_per_lane
    TILES_PER_BLOCK = block_kv // TILE_SIZE
    MFMA_SCALED_K = 128
    MFMA_ISSUES = HEAD_SIZE // MFMA_SCALED_K  # 2 for HEAD_SIZE=256
    GRPS_PER_ISSUE = MFMA_SCALED_K // FP8_GROUP_SIZE  # 4
    # MFMA operand format codes.
    A_FMT_FP4 = 4
    B_FMT_FP8 = 0

    QG = query_group_size
    _Q_LDS_BYTES = QG * HEAD_SIZE * 2

    arch = get_hip_arch()
    _qk_scale = float(softmax_scale)

    # --- Strides ---
    # Q is a view of the fused QKV projection, so its strides are passed in.
    _Hq = num_kv_heads * QG
    _stride_q_seq = stride_q_seq
    _stride_q_head = stride_q_head
    _stride_code_slot = num_kv_heads * KEY_CODE_BYTES
    _stride_code_head = KEY_CODE_BYTES
    _stride_scale_slot = num_kv_heads * N_GROUPS
    _stride_scale_head = N_GROUPS

    # Split-K outputs use the decode reducer's layout: attn_logits
    # [B, Hq, split_stride, D] and attn_lse [B, Hq, split_stride], both fp32.
    _stride_out_split = HEAD_SIZE
    _stride_out_qhead = split_stride * HEAD_SIZE
    _stride_out_seq = _Hq * split_stride * HEAD_SIZE
    _stride_lse_qhead = split_stride
    _stride_lse_seq = _Hq * split_stride

    # --- LDS layout ---
    # The allocator must stay local to this build: flyc.jit aborts when a module
    # global it references changes between compiles, and several variants can
    # be resident at once.
    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=(
            f"ultraquant_hd256_smem_p{num_partitions}_b{block_kv}_s{split_stride}"
            f"_w{work_budget}_k{seqs_per_lane}"
        ),
    )
    q_off = 0
    allocator.ptr = _Q_LDS_BYTES
    kv_off = allocator.ptr
    allocator.ptr += KV_TILE_LDS_BYTES_PADDED
    scale_off = allocator.ptr
    allocator.ptr += SCALE_LDS_BYTES

    @flyc.kernel
    def ultraquant_decode_hd256_kernel(
        out_ptr: fx.Tensor,
        lse_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        k_code_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        v_code_ptr: fx.Tensor,
        v_scale_ptr: fx.Tensor,
        kv_indptr_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
    ):
        # ---- IDs ---------------------------------------------------------
        tid = gpu.thread_idx.x
        kv_h = gpu.block_idx.y
        lane = tid  # 0..63
        mfma_row = lane & fx.Int32(15)
        mfma_col_grp = lane >> fx.Int32(4)  # 0..3, K-group dim

        # ---- Buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        kc_rsrc = buffer_ops.create_buffer_resource(k_code_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(k_scale_ptr, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(v_code_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(v_scale_ptr, max_size=True)
        kip_rsrc = buffer_ops.create_buffer_resource(kv_indptr_ptr, max_size=True)
        kvi_rsrc = buffer_ops.create_buffer_resource(kv_indices_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        lse_rsrc = buffer_ops.create_buffer_resource(lse_ptr, max_size=True)

        # ---- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(_Q_LDS_BYTES // 4,)).get()
        q_lds_i64 = SmemPtr(base, q_off, T.i64, shape=(_Q_LDS_BYTES // 8,)).get()
        kv_lds_i32 = SmemPtr(
            base, kv_off, T.i32, shape=(KV_TILE_LDS_BYTES_PADDED // 4,)
        ).get()
        kv_lds_i64 = SmemPtr(
            base, kv_off, T.i64, shape=(KV_TILE_LDS_BYTES_PADDED // 8,)
        ).get()
        scale_lds_i32 = SmemPtr(base, scale_off, T.i32, shape=(TILE_SIZE * N_GROUPS,))

        # ---- Constants ---------------------------------------------------
        c_sq = fx.Int32(_stride_q_seq)
        c_qh = fx.Int32(_stride_q_head)
        c_qg = fx.Int32(QG)
        c_code_slot = fx.Int32(_stride_code_slot)
        c_code_head = fx.Int32(_stride_code_head)
        c_scale_slot = fx.Int32(_stride_scale_slot)
        c_scale_head = fx.Int32(_stride_scale_head)
        c_w = fx.Int32(WARP_SIZE)

        NEG_INF = arith.constant(NEG_INF_VAL, type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        QK_SCALE = arith.constant(_qk_scale, type=T.f32)

        def _ival(v):
            return v.ir_value() if hasattr(v, "ir_value") else v

        c_zero_i32 = arith.constant(0, type=T.i32)

        def _f32x8_to_fp8_i64(f):
            # 8 f32 -> 8 E4M3 bytes packed as i64 (MFMA fp8 operand).
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[0], f[1], c_zero_i32, 0)
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, f[2], f[3], w0, 1)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[4], f[5], c_zero_i32, 0)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, f[6], f[7], w1, 1)
            pv = vector.from_elements(T.vec(2, T.i32), [w0, w1])
            return vector.extract(
                vector.bitcast(T.vec(1, T.i64), pv), static_position=[0]
            )

        def _bf16x8_to_fp8_i64(v_bf16):
            f = [
                arith.extf(T.f32, vector.extract(v_bf16, static_position=[i]))
                for i in range(8)
            ]
            return _f32x8_to_fp8_i64(f)

        # ===== Sequence, length and the blocks this workgroup walks ========
        # kv_indptr is the ragged row-pointer over kv_indices: a sequence owns
        # kv_indices[kv_base : kv_base + seq_len]. Taking the length from the
        # same tensor as the base keeps the two from ever disagreeing, and
        # matches how the Triton ultraquant kernel reads it.
        c_kcb = fx.Int32(block_kv)
        c_one_i32 = fx.Int32(1)
        c_zero = fx.Int32(0)

        def _load_indptr(i):
            return buffer_ops.buffer_load(kip_rsrc, i, vec_width=1, dtype=T.i32)

        if const_expr(CHUNKED):
            # ultraquant_decode_reduce derives the same chunking; keep the two equal.
            chunk = fx.Int32(gpu.block_idx.x)
            num_seqs = fx.Int32(gpu.grid_dim.x) - fx.Int32(work_budget)
            kv_start = _load_indptr(c_zero)
            batch_tokens = _load_indptr(num_seqs) - kv_start
            # Lane l holds kv_indptr[l*K .. l*K + K], clamped to the batch end,
            # all fetched in the same round as the batch extent.
            lane_first = lane * fx.Int32(SEQS_PER_LANE)
            ptrs = []
            for i in range_constexpr(SEQS_PER_LANE + 1):
                idx = lane_first + fx.Int32(i)
                ptrs.append(_load_indptr((idx < num_seqs).select(idx, num_seqs)))
            c_budget_tokens = (fx.Int32(work_budget) - num_seqs) * c_kcb
            chunk_tokens = (
                (batch_tokens + c_budget_tokens - c_one_i32) // c_budget_tokens * c_kcb
            )
            # An empty batch still needs a nonzero divisor.
            chunk_tokens = (chunk_tokens > c_kcb).select(chunk_tokens, c_kcb)

            # The sequence is the last s with g(s) <= chunk. g rises with s, so
            # the passing sequences are a prefix and counting them finds s.
            num_hits = c_zero
            for i in range_constexpr(SEQS_PER_LANE):
                s_i = lane_first + fx.Int32(i)
                hit = (s_i < num_seqs) & (
                    (ptrs[i] - kv_start) // chunk_tokens + s_i <= chunk
                )
                mask = rocdl.ballot(T.i64, hit)
                num_hits = num_hits + fx.Int32(arith.trunci(T.i32, _math.ctpop(mask)))
            seq = num_hits - c_one_i32
            owner = seq // fx.Int32(SEQS_PER_LANE)
            owner_slot = seq - owner * fx.Int32(SEQS_PER_LANE)
            kv_base = c_zero
            kv_end = c_zero
            for i in range_constexpr(SEQS_PER_LANE):
                is_slot = owner_slot == fx.Int32(i)
                kv_base = is_slot.select(
                    fx.Int32(rocdl.readlane(T.i32, ptrs[i], owner)), kv_base
                )
                kv_end = is_slot.select(
                    fx.Int32(rocdl.readlane(T.i32, ptrs[i + 1], owner)), kv_end
                )
            seq_len = kv_end - kv_base
            total_tgs = (seq_len + c_kcb - c_one_i32) // c_kcb
            local = chunk - ((kv_base - kv_start) // chunk_tokens + seq)
            n_chunks = (seq_len + chunk_tokens - c_one_i32) // chunk_tokens
            n_chunks = (n_chunks > c_one_i32).select(n_chunks, c_one_i32)
            active = local < n_chunks
            blocks_per_chunk = chunk_tokens // c_kcb
            blk0 = local * blocks_per_chunk
            blk_stride = c_one_i32
            trip = total_tgs - blk0
            trip = (trip < blocks_per_chunk).select(trip, blocks_per_chunk)
            trip = (trip > c_zero).select(trip, c_zero)
            out_row = chunk
            out_part = c_zero
        else:
            # Workgroups go to the XCDs round robin by linear id. Rotating the
            # sequence by the partition spreads each sequence over every XCD;
            # otherwise one XCD takes all of sequence s whenever the batch is a
            # multiple of the XCD count.
            part = fx.Int32(gpu.block_idx.z)
            seq = (fx.Int32(gpu.block_idx.x) + part) % fx.Int32(gpu.grid_dim.x)
            kv_base = _load_indptr(seq)
            seq_len = _load_indptr(seq + c_one_i32) - kv_base
            total_tgs = (seq_len + c_kcb - c_one_i32) // c_kcb
            active = part < fx.Int32(NUM_PARTS_C)
            blk0 = part
            blk_stride = fx.Int32(NUM_PARTS_C)
            # part < P, so the trip count is >= 0 and 0 when the sequence ends first.
            trip = (total_tgs - part + blk_stride - c_one_i32) // blk_stride
            out_row = seq
            out_part = part

        # ===== STEP A: in-kernel Q rotation (fused WHT) ===================
        # Q @ PiT is a Walsh-Hadamard transform (PiT is the Sylvester Hadamard
        # with a column permutation folded in), so 8 butterfly stages replace a
        # 256x256 matmul. After the E4M3 haircut it matches the matmul exactly.
        #
        # Lane map: lane L owns row r = L>>3 and the 32 CONSECUTIVE head-dims
        # d = (L&7)*32 + t. That puts butterfly stages 0..4 (strides 1..16)
        # entirely inside one lane's registers, and leaves only stages 5..7
        # (strides 32/64/128 == chunk XOR 1/2/4) needing a cross-lane exchange.
        _wrow = lane >> fx.Int32(3)  # 0..7  row within the group
        _wchk = lane & fx.Int32(7)  # 0..7  32-dim chunk
        # inv(qperm) is a pure bit permutation of the head-dim index:
        #   dest = (d & 0x8F) | ((d>>2)&0x10) | ((d<<1)&0x60)
        # which splits additively into a compile-time part in t and a
        # per-lane part in the chunk index (verified exhaustively):
        #   ct(t) = (t & 0x0F) | ((t & 0x10) << 1)
        #   rt(c) = ((c&2)<<3) | ((c&1)<<6) | ((c&4)<<5)
        _perm_base = (
            ((_wchk & fx.Int32(2)) << fx.Int32(3))
            | ((_wchk & fx.Int32(1)) << fx.Int32(6))
            | ((_wchk & fx.Int32(4)) << fx.Int32(5))
        )
        _inv_sqrtD = arith.constant(1.0 / math.sqrt(HEAD_SIZE), type=T.f32)
        for _qit in range_constexpr((QG + 7) // 8):
            _qrow = _wrow + fx.Int32(_qit * 8)
            if (_qrow < fx.Int32(QG)) & active:
                _qb = seq * c_sq + (kv_h * c_qg + _qrow) * c_qh + _wchk * fx.Int32(32)
                # ---- load this lane's 32 raw-Q head-dims (4 x dwordx4) ----
                _v = []
                for _j in range_constexpr(4):
                    _raw = buffer_ops.buffer_load(
                        q_rsrc,
                        (_qb + fx.Int32(_j * 8)) // fx.Int32(2),
                        vec_width=4,
                        dtype=T.i32,
                    )
                    _rb = vector.bitcast(T.vec(8, T.bf16), _raw)
                    for _e in range_constexpr(8):
                        _v.append(
                            arith.extf(T.f32, vector.extract(_rb, static_position=[_e]))
                        )
                # ---- stages 0..4: in-register, no LDS, no cross-lane ----
                for _s in range_constexpr(5):
                    _st = 1 << _s
                    _nx = list(_v)
                    for _g in range_constexpr(32 // (2 * _st)):
                        for _k in range_constexpr(_st):
                            _i0 = _g * 2 * _st + _k
                            _i1 = _i0 + _st
                            _a, _b = _v[_i0], _v[_i1]
                            _nx[_i0] = _a + _b
                            _nx[_i1] = _a - _b
                    _v = _nx
                # ---- stages 5..7: chunk XOR 1/2/4 via ds_swizzle ----
                # ds_swizzle_b32 offset (bit15=0, 32-lane groups):
                #   [4:0]=and_mask [9:5]=or_mask [14:10]=xor_mask
                # pure XOR k -> and_mask=0x1F, or_mask=0, xor_mask=k. The
                # partner is always within the same 32-lane group (k<=4).
                for _s in range_constexpr(3):
                    _k = 1 << _s
                    _swz = arith.constant((_k << 10) | 0x1F, type=T.i32)
                    # High half of the pair computes (partner - mine), the low
                    # half (mine + partner). Fold that into a single sign
                    # computed ONCE per stage, so each element costs one
                    # multiply-add rather than a branch or a per-element select.
                    _chk_bit = _wchk & fx.Int32(_k)
                    _is_hi = arith.cmpi(
                        arith.CmpIPredicate.ne,
                        _ival(_chk_bit),
                        fx.Int32(0).ir_value(),
                    )
                    _sf = arith.select(
                        _is_hi,
                        arith.constant(-1.0, type=T.f32),
                        arith.constant(1.0, type=T.f32),
                    )
                    _nx = []
                    for _t in range_constexpr(32):
                        _mi = arith.bitcast(T.i32, _v[_t])
                        _pt = arith.bitcast(T.f32, rocdl.ds_swizzle(T.i32, _mi, _swz))
                        _nx.append(_v[_t] * _sf + _pt)
                    _v = _nx
                # ---- 1/sqrt(D) normalization + E4M3 haircut -> bf16 ----
                # cvt_pk_fp8_f32 is the SAME instruction the QK operand build
                # uses, so this reproduces the host haircut exactly. E4M3 is a
                # subset of bf16, so the bf16 store below is lossless and every
                # downstream consumer stays byte-identical.
                _hc = []
                for _p in range_constexpr(16):
                    _w = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        _v[2 * _p] * _inv_sqrtD,
                        _v[2 * _p + 1] * _inv_sqrtD,
                        fx.Int32(0),
                        0,
                    )
                    _u = rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), _w, 0)
                    _hc.append(
                        arith.truncf(T.bf16, vector.extract(_u, static_position=[0]))
                    )
                    _hc.append(
                        arith.truncf(T.bf16, vector.extract(_u, static_position=[1]))
                    )
                # ---- permuted store ----
                # ct(t) maps t=0..15 -> 0..15 and t=16..31 -> 32..47, i.e. TWO
                # contiguous 16-element runs, so the scatter is still 4 aligned
                # 16-byte vector stores rather than 32 scalar writes.
                _qrow_elem = _qrow * fx.Int32(HEAD_SIZE)
                for _half in range_constexpr(2):
                    for _q4 in range_constexpr(2):
                        _elems = [_hc[_half * 16 + _q4 * 8 + _e] for _e in range(8)]
                        _vec = vector.from_elements(T.vec(8, T.bf16), _elems)
                        _dst = _qrow_elem + _perm_base + fx.Int32(_half * 32 + _q4 * 8)
                        vector.store(
                            vector.bitcast(T.vec(4, T.i32), _vec),
                            q_lds_i32,
                            [arith.index_cast(T.index, _dst // fx.Int32(2))],
                        )
        gpu.barrier()

        # Scaled-MFMA Q operands (loop-invariant): one per MFMA issue.
        q_op_hoisted = []
        for h in range_constexpr(MFMA_ISSUES):
            _q_hoist_words = []
            for j in range_constexpr(4):
                _q_hoist_idx = (
                    mfma_row * fx.Int32(HEAD_SIZE * 2 // 8)
                    + (fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp) * fx.Int32(8)
                    + fx.Int32(j * 2)
                )
                _q_hoist_v = vector.load_op(
                    T.vec(2, T.i64),
                    q_lds_i64,
                    [arith.index_cast(T.index, _q_hoist_idx)],
                )
                _q_hoist_words.append(
                    _bf16x8_to_fp8_i64(vector.bitcast(T.vec(8, T.bf16), _q_hoist_v))
                )
            q_op_hoisted.append(
                vector.bitcast(
                    T.vec(8, T.i32),
                    vector.from_elements(T.vec(4, T.i64), _q_hoist_words),
                )
            )

        # ===== STEP B: Online softmax + PV state =========================
        running_max = NEG_INF
        running_sum = ZERO_F
        zero_v4 = arith.constant_vector(0.0, T.f32x4)
        acc_pv = [zero_v4 for _ in range(PV_N_CHUNKS)]

        # Per-K-tile dequant lane assignment: lane t -> token = t/4,
        # chunk_in_tok = t%4 (each chunk = SUBCHUNK_HDIMS=64 head-dims = 2 groups).
        tok_in_tile = lane >> fx.Int32(2)
        chunk_in_tok = lane & fx.Int32(3)
        # Both UE8M0 group bytes for this lane's chunk (groups 2c, 2c+1) live in
        # the SAME 4-byte scale word at offset (chunk_in_tok>>1)*4 within the
        # 8-byte scale region. Load once; extract per-half below.
        c_scale_word_off = (chunk_in_tok >> fx.Int32(1)) * fx.Int32(4)

        # ===== STEP C: K-tile loop =======================================
        c_zero_idx = arith.constant(0, index=True)
        c_one_idx = arith.constant(1, index=True)
        trip_idx = arith.index_cast(T.index, _ival(trip))
        _init_iter = [
            _ival(running_max),
            _ival(running_sum),
            *[_ival(p) for p in acc_pv],
        ]
        _for_op = _scf.ForOp(
            c_zero_idx,
            trip_idx,
            c_one_idx,
            _init_iter,
        )
        with ir.InsertionPoint(_for_op.body):
            tg_idx = _for_op.induction_variable
            tg_i32 = fx.Int32(arith.index_cast(T.i32, tg_idx))
            partition_start = (blk0 + tg_i32 * blk_stride) * c_kcb
            running_max = _for_op.inner_iter_args[0]
            running_sum = _for_op.inner_iter_args[1]
            acc_pv = list(_for_op.inner_iter_args[2:])

            # Software prefetch: issue tile N+1's HBM loads (K/V codes and
            # scale words) before tile N's dequant + MFMA so they overlap.
            def _emit_tile_loads(n_tile):
                tile_start_tok = partition_start + fx.Int32(n_tile * TILE_SIZE)
                tok_pos = tile_start_tok + tok_in_tile
                # Tail lanes reload the last token; the `kv_tok < seq_len` mask
                # discards them. The loop has no trips when seq_len == 0.
                tok_safe = (tok_pos < seq_len).select(tok_pos, seq_len - fx.Int32(1))
                # kv_indices is int64, so each entry spans two dwords; slot ids
                # are far below 2^31 (the launcher bounds the pool), so reading
                # the low dword is exact.
                slot_id = buffer_ops.buffer_load(
                    kvi_rsrc,
                    (kv_base + tok_safe) * fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )

                # SoA: each field is its own tensor, so the slot id scales by
                # that tensor's own row stride. Both code rows are 128 B and
                # land 128 B-aligned, so each lane's 16 B loads never straddle
                # a cache line.
                code_base_byte = slot_id * c_code_slot + kv_h * c_code_head
                scale_base_byte = slot_id * c_scale_slot + kv_h * c_scale_head

                # K codes: LANE_CODE_BYTES (=32) at code_base + chunk*32, in
                # HALVES x 16-byte buffer_loads.
                k_byte0 = code_base_byte + chunk_in_tok * fx.Int32(LANE_CODE_BYTES)
                _kpl = []
                for hf in range_constexpr(HALVES):
                    _kpl.append(
                        buffer_ops.buffer_load(
                            kc_rsrc,
                            (k_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )

                # ---- V codes + K/V scale HBM loads ------------------------
                # V codes sit at the same in-row offset as K.
                _vpl = []
                for hf in range_constexpr(HALVES):
                    _vpl.append(
                        buffer_ops.buffer_load(
                            vc_rsrc,
                            (k_byte0 + fx.Int32(hf * 16)) // fx.Int32(4),
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                # UE8M0 group scale words (one 4-byte word covers this lane's
                # two groups). scale = 2^(byte-127) = bitcast_f32(byte<<23);
                # byte==0 -> +0.0 (zero sentinel).
                _ksw = buffer_ops.buffer_load(
                    ks_rsrc,
                    (scale_base_byte + c_scale_word_off) // fx.Int32(4),
                    vec_width=1,
                    dtype=T.i32,
                )
                _vsw = buffer_ops.buffer_load(
                    vs_rsrc,
                    (scale_base_byte + c_scale_word_off) // fx.Int32(4),
                    vec_width=1,
                    dtype=T.i32,
                )
                return (_kpl, _vpl, _ksw, _vsw)

            next_loads = _emit_tile_loads(0)
            for n_tile in range_constexpr(TILES_PER_BLOCK):
                _ntile_tok = fx.Int32(n_tile * TILE_SIZE)
                (k_packed_list, v_packed_list, kscale_word, vscale_word) = next_loads
                if const_expr(n_tile + 1 < TILES_PER_BLOCK):
                    next_loads = _emit_tile_loads(n_tile + 1)
                # Store raw FP4 codes (natural contiguous order,
                # 16 B = 32 nibbles per UE8M0 group) straight to KV LDS as
                # the scaled-MFMA A operand — no dequant here.
                for hf in range_constexpr(HALVES):
                    grp = chunk_in_tok * fx.Int32(2) + fx.Int32(hf)
                    grp_shift = (grp & fx.Int32(3)) * fx.Int32(8)
                    kscale_byte = (kscale_word >> grp_shift) & fx.Int32(0xFF)
                    k_packed = k_packed_list[hf]
                    vector.store(
                        k_packed,
                        kv_lds_i32,
                        [
                            arith.index_cast(
                                T.index,
                                tok_in_tile * fx.Int32(KFP4_ROW_I32)
                                + grp * fx.Int32(4),
                            )
                        ],
                    )
                    scale_lds_i32.store(
                        kscale_byte,
                        [
                            arith.index_cast(
                                T.index,
                                tok_in_tile * fx.Int32(N_GROUPS) + grp,
                            )
                        ],
                    )
                gpu.barrier()
                # QK: MFMA_ISSUES native scaled MFMAs over K=128,
                # chained through the accumulator.
                #   A = K raw FP4 codes (vec4 i32 = 16 B) per (token, group)
                #   B = Q E4M3 (vec8 i32 = 32 B) for the same group
                #   scaleA = per-(token,group) UE8M0 byte at op_sel byte 0
                #   scaleB = 0x7F identity (Q carries no scale)
                IDENT = fx.Int32(0x7F)
                qk_acc = zero_v4
                for h in range_constexpr(MFMA_ISSUES):
                    grp = fx.Int32(h * GRPS_PER_ISSUE) + mfma_col_grp
                    k_op = vector.load_op(
                        T.vec(4, T.i32),
                        kv_lds_i32,
                        [
                            arith.index_cast(
                                T.index,
                                mfma_row * fx.Int32(KFP4_ROW_I32) + grp * fx.Int32(4),
                            )
                        ],
                    )
                    q_op = q_op_hoisted[h]
                    scbyte = fx.Int32(
                        scale_lds_i32.load(
                            [
                                arith.index_cast(
                                    T.index,
                                    mfma_row * fx.Int32(N_GROUPS) + grp,
                                )
                            ]
                        )
                    )
                    kscale = fx.Int32(0x7F7F7F00) | scbyte
                    qk_acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32),
                        [
                            k_op,
                            q_op,
                            qk_acc,
                            A_FMT_FP4,
                            B_FMT_FP8,
                            0,
                            kscale,
                            0,
                            IDENT,
                        ],
                    )

                # Scale + mask out-of-context tokens.
                qk_acc = _vsplat_mul(qk_acc, QK_SCALE)
                for elem in range_constexpr(4):
                    kv_tok = (
                        partition_start
                        + _ntile_tok
                        + mfma_col_grp * fx.Int32(4)
                        + fx.Int32(elem)
                    )
                    in_b = kv_tok < seq_len
                    v = vector.extract(qk_acc, static_position=[elem])
                    qk_acc = _vector_insert(
                        in_b.select(v, NEG_INF),
                        qk_acc,
                        static_position=[elem],
                    )

                # FA2 online softmax: per-query-row reduce.
                local_max = vector.reduction(T.f32, "maxnumf", qk_acc)
                r1 = local_max.shuffle_xor(fx.Int32(16), c_w)
                local_max = local_max.maximumf(r1)
                r2 = local_max.shuffle_xor(fx.Int32(32), c_w)
                tile_max = local_max.maximumf(r2)

                new_max = running_max.maximumf(tile_max)
                max_diff = running_max - new_max
                safe_diff = (running_max > NEG_INF).select(max_diff, ZERO_F)
                scale = (safe_diff * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                running_sum = running_sum * scale
                for h in range_constexpr(PV_N_CHUNKS):
                    acc_pv[h] = _vsplat_mul(acc_pv[h], scale)
                running_max = new_max

                tile_sum = ZERO_F
                for elem in range_constexpr(4):
                    s = vector.extract(qk_acc, static_position=[elem])
                    d = s - new_max
                    d = (new_max > NEG_INF).select(d, NEG_INF)
                    p = (d * LOG2E_C).exp2(fastmath=arith.FastMathFlags.fast)
                    tile_sum = tile_sum + p
                    qk_acc = _vector_insert(p, qk_acc, static_position=[elem])

                ts1 = tile_sum.shuffle_xor(fx.Int32(16), c_w)
                tile_sum = tile_sum + ts1
                ts2 = tile_sum.shuffle_xor(fx.Int32(32), c_w)
                tile_sum = tile_sum + ts2
                running_sum = running_sum + tile_sum
                # V dequant → LDS [token][head_dim] (ROW-MAJOR, no transpose)
                # written as ds_write_b128 (vec(2,i64)); read back via the
                # ds_read_tr16_b64 HW transpose in the PV block below.
                v_lds_elem_base = tok_in_tile * fx.Int32(
                    KV_ROW_ELEMS
                ) + chunk_in_tok * fx.Int32(SUBCHUNK_HDIMS)
                for hf in range_constexpr(HALVES):
                    v_packed = v_packed_list[hf]
                    half_hd = fx.Int32(hf * HALF_HDIMS)
                    grp_shift = (
                        (chunk_in_tok * fx.Int32(2) + fx.Int32(hf)) & fx.Int32(3)
                    ) * fx.Int32(8)
                    vscale_byte = (vscale_word >> grp_shift) & fx.Int32(0xFF)
                    vscale_f32 = arith.bitcast(
                        T.f32, _ival(vscale_byte << fx.Int32(23))
                    )
                    for w in range_constexpr(4):
                        word_i32 = vector.extract(v_packed, static_position=[w])
                        # Native CDNA4 scaled convert: 2-wide
                        # cvt_scalef32_pk_bf16_fp4. srcSel s picks byte s
                        # of the word (nibbles 2s, 2s+1) → 2 bf16 with the
                        # UE8M0 group scale fused. 4 cvt calls/word.
                        bf16_elems = []
                        for s in range_constexpr(4):
                            v2 = rocdl.cvt_scalef32_pk_bf16_fp4(
                                T.vec(2, T.bf16),
                                _ival(word_i32),
                                _ival(vscale_f32),
                                int(s),
                            )
                            bf16_elems.append(vector.extract(v2, static_position=[0]))
                            bf16_elems.append(vector.extract(v2, static_position=[1]))
                        v_bf16 = vector.from_elements(T.vec(8, T.bf16), bf16_elems)
                        v_i64 = vector.bitcast(T.vec(2, T.i64), v_bf16)
                        v_lds_i64_idx = (
                            v_lds_elem_base + half_hd + fx.Int32(w * 8)
                        ) // fx.Int32(4)
                        vector.store(
                            v_i64,
                            kv_lds_i64,
                            [arith.index_cast(T.index, v_lds_i64_idx)],
                        )
                # HW V transpose cross-lane LDS fence.
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(0xC07F)
                gpu.barrier()

                # ---- PV MFMA: A=V[head_dim, token], B=P (=qk_acc bf16) -----
                # V_lds holds V[token][head_dim] row-major, so the HW transpose
                # on read is what supplies A in the order the MFMA wants.
                p_bf16 = arith.trunc_f(T.vec(4, T.bf16), qk_acc)
                p_op = vector.bitcast(T.vec(4, T.i16), p_bf16)
                token_idx = lane >> fx.Int32(2)
                hd_sub = (lane & fx.Int32(3)) * fx.Int32(4)
                v_lane_byte = (
                    fx.Int32(kv_off)
                    + token_idx * fx.Int32(KV_ROW_BYTES)
                    + hd_sub * fx.Int32(2)
                )
                for h in range_constexpr(PV_N_CHUNKS):
                    v_byte_off = v_lane_byte + fx.Int32(h * 32)
                    v_byte_i64 = fx.Int64(v_byte_off)
                    v_ptr = buffer_ops.create_llvm_ptr(
                        v_byte_i64,
                        address_space=3,
                    )
                    v_op_raw = rocdl.ds_read_tr16_b64(
                        T.vec(4, T.i16),
                        v_ptr,
                    ).result
                    acc_pv[h] = rocdl.mfma_f32_16x16x16bf16_1k(
                        T.f32x4, [v_op_raw, p_op, acc_pv[h], 0, 0, 0]
                    )

            _scf.YieldOp(
                [
                    _ival(running_max),
                    _ival(running_sum),
                    *[_ival(p) for p in acc_pv],
                ]
            )
        running_max = _for_op.results[0]
        running_sum = _for_op.results[1]
        acc_pv = list(_for_op.results[2:])

        # ===== STEP D: Output ===========================================
        safe_sum = (running_sum > ZERO_F).select(running_sum, ONE_F)
        rcp = ONE_F / safe_sum

        # sglang indexes the split-K buffers by query head, so fold the GQA
        # group row into the head index here (q_head = kv_h * QG + row).
        q_head = kv_h * fx.Int32(QG) + mfma_row
        out_base = (
            out_row * fx.Int32(_stride_out_seq)
            + q_head * fx.Int32(_stride_out_qhead)
            + out_part * fx.Int32(_stride_out_split)
        )

        valid_row_pred = arith.andi(
            arith.cmpi(
                arith.CmpIPredicate.ult,
                _ival(mfma_row),
                arith.constant(QG, type=T.i32),
            ),
            _ival(active),
        )
        _if = _scf.IfOp(valid_row_pred)
        with ir.InsertionPoint(_if.then_block):
            for h in range_constexpr(PV_N_CHUNKS):
                pv_norm = _vsplat_mul(acc_pv[h], rcp)
                pv_i32x4 = vector.bitcast(T.vec(4, T.i32), pv_norm)
                head_dim_start = fx.Int32(h * 16) + mfma_col_grp * fx.Int32(4)
                out_off_elem = out_base + head_dim_start
                buffer_ops.buffer_store(
                    pv_i32x4,
                    out_rsrc,
                    out_off_elem * fx.Int32(4),
                    offset_is_bytes=True,
                )

            # An empty partition has running_sum == 0, so its lse is -inf and
            # it drops out of the reducer's merge.
            lse = running_max + fx.log(running_sum)
            lse_off = (
                out_row * fx.Int32(_stride_lse_seq)
                + q_head * fx.Int32(_stride_lse_qhead)
                + out_part
            )
            buffer_ops.buffer_store(lse, lse_rsrc, lse_off)
            _scf.YieldOp([])

    @flyc.jit
    def launch(
        out_mem: fx.Tensor,
        lse_mem: fx.Tensor,
        query_mem: fx.Tensor,
        k_code_mem: fx.Tensor,
        k_scale_mem: fx.Tensor,
        v_code_mem: fx.Tensor,
        v_scale_mem: fx.Tensor,
        kv_indptr_mem: fx.Tensor,
        kv_indices_mem: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        ultraquant_decode_hd256_kernel(
            out_mem,
            lse_mem,
            query_mem,
            k_code_mem,
            k_scale_mem,
            v_code_mem,
            v_scale_mem,
            kv_indptr_mem,
            kv_indices_mem,
        ).launch(
            grid=(batch_size, num_kv_heads, num_partitions),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return launch
