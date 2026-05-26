"""FlyDSL sparse NSA prefill attention kernel for AMD gfx950 (MI350X).

Online flash-attention over NSA sparse KV using MFMA 16x16x32 FP8 wave64.

LLVM type constraint: LLVM has NO knowledge of f8E4M3FN. Any MLIR op that
keeps f8E4M3FN alive until LLVM lowering crashes with "unknown LLVM dialect
type". This means:
  ❌ memref<Nxf8>            (load/store/GEP → LLVM fp8 type)
  ❌ arith.truncf f32 → f8   (fx.Float32.to(fp8_t))
  ❌ arith.bitcast f8 → i8   (scalar fp8 bitcast)
  ❌ vector<Nxf8>            (any fp8 vector op)

Safe pattern (used here):
  ✓ _llvm.LoadOp(T.i64, gep_with_T_i8_elem)   # 8 raw FP8 bytes as i64
  ✓ memref<T.i64> for KV LDS                  # i64 memref, LLVM-safe
  ✓ memref<T.f32> for P LDS                   # f32 memref, LLVM-safe
  ✓ _memref.store/load on f32 memref          # f32 scalar, LLVM-safe
  ✓ rocdl.cvt_f32_fp8 to decode V bytes       # AMD hardware fp8→f32
  ✓ mfma_f32_16x16x32_fp8_fp8 with i64 A/B   # MFMA takes i64, no fp8 type

Kernel structure:
  GEMM1: Q[16,512]@K[32,512]^T — fp8 MFMA with Q/K as raw i64
  Softmax: f32 online softmax
  P→LDS: store f32 attention weights to f32 LDS (transposed)
  GEMM2: P[16,32]@V[32,512] — fp8 MFMA; P packed via rocdl.cvt_pk_fp8_f32, V bytes strided from KV LDS

MFMA 16x16x32 FP8 layout (wave64):
  A/B: thread t → row t%16, cols t//16*8:+8  (8 FP8 as i64)
  C/D: thread t, reg r → C[(t//16)*4+r][t%16]  (v4f32)
"""

from __future__ import annotations

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import (
    arith as _mlir_arith,
    fly as _fly,
    llvm as _llvm,
    memref as _memref,
)

_LOG2E = host_math.log2(host_math.e)


def _is_available() -> bool:
    try:
        return str(get_rocm_arch()).startswith("gfx950")
    except Exception:
        return False


@functools.lru_cache(maxsize=64)
def build_nsa_prefill_kernel(
    h_q: int,
    head_dim: int = 512,
    topk: int = 2048,
    tile_m: int = 16,
    block_n: int = 32,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
):
    assert head_dim == 512
    assert tile_m == 16
    assert block_n == 32
    assert topk % block_n == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    gpu_arch = get_rocm_arch()

    WARP_SIZE  = 64
    TILE_M     = tile_m
    BLOCK_N    = block_n
    HEAD_DIM   = head_dim
    TOPK       = topk
    BLOCK_SIZE = WARP_SIZE

    MFMA_N      = 16
    MFMA_K      = 32
    K_STEPS_QK  = HEAD_DIM // MFMA_K    # 16
    N_BLKS_S    = BLOCK_N  // MFMA_N    # 2
    D_BLKS      = HEAD_DIM // MFMA_N    # 32
    KV_TILES    = TOPK // BLOCK_N

    # LDS layout:
    #   KV  : T.i64 memref, LDS_KV_I64 entries (8 FP8 per i64)
    #   P   : T.f32 memref, LDS_P_F32  entries (1 f32 per attention weight)
    LDS_KV_I64    = BLOCK_N * HEAD_DIM // 8    # 32*512//8 = 2048
    LDS_KV_BYTES  = LDS_KV_I64 * 8             # 16384
    LDS_P_F32     = TILE_M * BLOCK_N            # 16*32 = 512
    LDS_P_BYTES   = LDS_P_F32 * 4              # 2048
    LDS_TOTAL     = LDS_KV_BYTES + LDS_P_BYTES

    Q_STOK  = h_q * HEAD_DIM
    IDX_STR = TOPK

    alloc      = SmemAllocator(None, arch=gpu_arch, global_sym_name="nsa_smem")
    base_off   = alloc._align(alloc.ptr, 16)
    alloc.ptr  = base_off + LDS_TOTAL
    kv_lds_off = base_off
    p_lds_off  = base_off + LDS_KV_BYTES

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def nsa_prefill_kernel(
        Q:       fx.Tensor,   # [total_tokens, h_q, HEAD_DIM]  fp8
        KV:      fx.Tensor,   # [num_pages, HEAD_DIM]           fp8
        Indices: fx.Tensor,   # [total_tokens, TOPK]            int32
        Out:     fx.Tensor,   # [total_tokens, h_q, HEAD_DIM]  bf16
        total_tokens: fx.Int32,
    ):
        v4f32_type = Vec.make_type(4, fx.Float32)

        fm = arith.FastMathFlags.fast
        def _fadd(a, b): return arith.addf(_raw(a), _raw(b), fastmath=fm)
        def _fmul(a, b): return arith.mulf(_raw(a), _raw(b), fastmath=fm)
        def _fsub(a, b): return arith.subf(_raw(a), _raw(b), fastmath=fm)
        def _fmax(a, b): return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result

        c_neg_inf    = fx.Float32(float("-inf"))
        c_zero_f     = fx.Float32(0.0)
        c_one_f      = fx.Float32(1.0)
        c_sm_log2e   = fx.Float32(sm_scale * _LOG2E)
        c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)

        # ── LLVM pointers ────────────────────────────────────────────────────
        def _ptr_ty():  return ir.Type.parse("!llvm.ptr")
        def _as_ptr(t):
            v = t
            if hasattr(v, "ir_value") and not isinstance(v, ir.Value):
                v = v.ir_value()
            return _fly.extract_aligned_pointer_as_index(_ptr_ty(), v)

        q_ptr   = _as_ptr(Q)
        kv_ptr  = _as_ptr(KV)
        idx_ptr = _as_ptr(Indices)
        out_ptr = _as_ptr(Out)

        # ── Global load helpers (NO fp8 type in result) ────────────────────
        # Use T.i8 as GEP elem_type (byte addressing, LLVM-safe — avoids f8 in GEP).
        # Load 8 bytes as i64: T.i64 is LLVM-native.
        def load_8bytes_as_i64(ptr, byte_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(byte_off_index), elem_type=T.i8)
            return _llvm.LoadOp(T.i64, gep).result

        def load_i32_elem(ptr, elem_off_index):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(elem_off_index), elem_type=T.i32)
            return _llvm.LoadOp(T.i32, gep).result

        # ── MFMA (fp8 MFMA takes i64 operands — no fp8 type in LLVM) ─────
        def mfma_fp8(acc, a_i64, b_i64):
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                v4f32_type, [a_i64, b_i64, acc, 0, 0, 0]
            )

        # ── Decode FP8 byte (in low 8 bits of i32) to f32 ─────────────────
        # Uses AMD hardware instruction rocdl.cvt_f32_fp8 (gfx940+).
        # Input: i32 with fp8 byte in bits [7:0]; byte_sel=0 decodes byte 0.
        def fp8_byte_to_f32(byte_i32):
            # Signature: cvt_f32_fp8(result_type, src_i32, byte_sel_i32)
            return rocdl.cvt_f32_fp8(T.f32, byte_i32, fx.Int32(0))

        # ── Thread / block ids ────────────────────────────────────────────
        tid     = fx.Index(gpu.thread_idx.x)
        bid_m   = fx.Index(gpu.block_idx.x)
        bid_h   = fx.Index(gpu.block_idx.y)
        lane    = tid % 16
        k_group = tid // 16
        q_start = bid_m * fx.Index(TILE_M)

        # V-extraction lane decomposition (const per thread, outside d-loop)
        # d_col = d*16+lane; (d*16)%8=0 so d_col//8 = d*2+lane//8, d_col%8 = lane%8
        lane_mod8      = lane % fx.Index(8)
        lane_div8      = lane // fx.Index(8)
        lane_shift_i64 = fx.Int64(lane_mod8) * fx.Int64(8)

        # ── LDS ──────────────────────────────────────────────────────────
        lds_base   = alloc.get_base()
        lds_kv_i64 = SmemPtr(lds_base, kv_lds_off, T.i64, shape=(LDS_KV_I64,)).get()
        lds_p_f32  = SmemPtr(lds_base, p_lds_off,  T.f32, shape=(LDS_P_F32,)).get()

        # ── Pre-load Q into registers (K_STEPS_QK × i64) ────────────────
        q_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_byte_off = (
                (q_start + lane) * fx.Index(Q_STOK)
                + bid_h * fx.Index(HEAD_DIM)
                + fx.Index(ks * MFMA_K)
                + k_group * fx.Index(8)
            )
            q_packs.append(load_8bytes_as_i64(q_ptr, q_byte_off))

        # ── Online softmax init ──────────────────────────────────────────
        _init = (
            [_raw(c_neg_inf)] * 4
            + [_raw(c_zero_f)]  * 4
            + [_raw(c_zero_v4f32)] * D_BLKS
        )
        anchor_tok = q_start

        # ── Main KV-tile loop ────────────────────────────────────────────
        loop_results = _init
        for kv_tile, _carry in range(0, KV_TILES, 1, init=_init):
            m_run = [_carry[r]     for r in range_constexpr(4)]
            l_run = [_carry[4 + r] for r in range_constexpr(4)]
            o_acc = [_carry[8 + d] for d in range_constexpr(D_BLKS)]

            kv_pos_base = kv_tile * fx.Index(BLOCK_N)

            # ── Gather KV → LDS i64 ──────────────────────────────────────
            kv_row  = tid % fx.Index(BLOCK_N)
            d_group = tid // fx.Index(BLOCK_N)

            idx_flat = anchor_tok * fx.Index(IDX_STR) + kv_pos_base + kv_row
            page_i32 = load_i32_elem(idx_ptr, idx_flat)
            page_idx = fx.Index(page_i32)

            D_HALF   = HEAD_DIM // 2
            D_CHUNKS = D_HALF // 8
            for dc in range_constexpr(D_CHUNKS):
                d_byte_in_half = fx.Index(dc * 8)
                d_byte_abs     = d_group * fx.Index(D_HALF) + d_byte_in_half
                kv_byte_off    = page_idx * fx.Index(HEAD_DIM) + d_byte_abs
                raw_i64        = load_8bytes_as_i64(kv_ptr, kv_byte_off)
                i64_off = (
                    kv_row * fx.Index(HEAD_DIM // 8)
                    + d_group * fx.Index(D_HALF // 8)
                    + fx.Index(dc)
                )
                _memref.store(raw_i64, lds_kv_i64, [_raw(i64_off)])

            gpu.barrier()

            # ── GEMM1: S = Q @ K^T (fp8 MFMA) ───────────────────────────
            s_acc = [_raw(c_zero_v4f32) for _ in range(N_BLKS_S)]
            for ks in range_constexpr(K_STEPS_QK):
                q_a = q_packs[ks]
                for nb in range_constexpr(N_BLKS_S):
                    k_i64_off = (
                        (fx.Index(nb * MFMA_N) + lane) * fx.Index(HEAD_DIM // 8)
                        + fx.Index(ks * 4)
                        + k_group
                    )
                    k_pack    = _memref.load(lds_kv_i64, [_raw(k_i64_off)])
                    s_acc[nb] = mfma_fp8(s_acc[nb], q_a, k_pack)

            # ── Online softmax ───────────────────────────────────────────
            s_scaled = []
            for nb in range_constexpr(N_BLKS_S):
                sv = Vec(s_acc[nb])
                s_scaled.append([_fmul(sv[r], c_sm_log2e) for r in range_constexpr(4)])

            local_max = [_fmax(s_scaled[0][r], s_scaled[1][r]) for r in range_constexpr(4)]

            shfl_w  = fx.Int32(WARP_SIZE)
            row_max = list(local_max)
            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    row_max[r] = _fmax(row_max[r], fx.Float32(row_max[r]).shuffle_xor(so, shfl_w))

            m_new = [_fmax(m_run[r], row_max[r]) for r in range_constexpr(4)]
            corr  = [rocdl.exp2(T.f32, _raw(_fsub(m_run[r], m_new[r]))) for r in range_constexpr(4)]

            p_vals   = [[None] * 4 for _ in range(N_BLKS_S)]
            tile_sum = [_raw(c_zero_f) for _ in range(4)]
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_val = rocdl.exp2(T.f32, _raw(_fsub(s_scaled[nb][r], m_new[r])))
                    p_vals[nb][r] = p_val
                    tile_sum[r]   = _fadd(tile_sum[r], p_val)

            for xor_off in [8, 4, 2, 1]:
                so = fx.Int32(xor_off)
                for r in range_constexpr(4):
                    tile_sum[r] = _fadd(tile_sum[r], fx.Float32(tile_sum[r]).shuffle_xor(so, shfl_w))

            l_new = [_fadd(_fmul(corr[r], l_run[r]), tile_sum[r]) for r in range_constexpr(4)]

            for d in range_constexpr(D_BLKS):
                ov = Vec(o_acc[d])
                o_acc[d] = Vec.from_elements(
                    [_fmul(ov[r], corr[r]) for r in range_constexpr(4)], fx.Float32
                ).ir_value()

            m_run = m_new
            l_run = l_new

            # ── Store P → f32 LDS (transposed) ──────────────────────────
            # P[k_group*4+r][nb*16+lane] stored at row*BLOCK_N+col
            for nb in range_constexpr(N_BLKS_S):
                for r in range_constexpr(4):
                    p_row  = k_group * fx.Index(4) + fx.Index(r)
                    p_col  = fx.Index(nb * MFMA_N) + lane
                    p_off  = p_row * fx.Index(BLOCK_N) + p_col
                    _memref.store(p_vals[nb][r], lds_p_f32, [_raw(p_off)])

            gpu.barrier()

            # -- GEMM2: O += P @ V (fp8 MFMA) ----------------------------------------
            # mfma_f32_16x16x32_fp8_fp8: M=16(TILE_M), K=32(BLOCK_N), N=16(per d-block)
            # A (P): thread t -> P[lane][k_group*8:k_group*8+8] as i64 (8 FP8 bytes)
            # B (V): thread t -> V[k_group*8:k_group*8+8][d*16+lane] as i64 (8 FP8 bytes)
            # C/D: o_acc[d] (v4f32), same layout as before

            # Pack P -> i64: read 8 consecutive f32 from f32 LDS, convert via cvt_pk_fp8_f32
            # P[lane][k_group*8+j] at flat offset lane*BLOCK_N + k_group*8 + j
            p_base_off = lane * fx.Index(BLOCK_N) + k_group * fx.Index(8)
            p_f32_vals = [
                _memref.load(lds_p_f32, [_raw(p_base_off + fx.Index(j))])
                for j in range(8)
            ]

            # rocdl.cvt_pk_fp8_f32(result_i32, src1_f32, src0_f32, old_i32, op_sel)
            # op_sel=0: writes 2 FP8 bytes to bits [15:0]; op_sel=1: bits [31:16]
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[1], p_f32_vals[0], fx.Int32(0), fx.Int32(0))
            i32_lo = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[3], p_f32_vals[2], i32_lo,       fx.Int32(1))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[5], p_f32_vals[4], fx.Int32(0), fx.Int32(0))
            i32_hi = rocdl.cvt_pk_fp8_f32(T.i32, p_f32_vals[7], p_f32_vals[6], i32_hi,       fx.Int32(1))

            lo64    = _mlir_arith.ExtUIOp(T.i64, i32_lo).result
            hi64    = _mlir_arith.ShLIOp(
                _mlir_arith.ExtUIOp(T.i64, i32_hi).result, _raw(fx.Int64(32))
            ).result
            p_a_i64 = _mlir_arith.OrIOp(lo64, hi64).result

            for d in range_constexpr(D_BLKS):
                # Pack V -> i64: V[k_group*8+j][d*16+lane] for j=0..7 (strided KV rows)
                # i64_off(j) = (k_group*8+j) * (HEAD_DIM//8) + d*2 + lane//8
                # byte within i64 = lane%8
                hd_off_f = fx.Index(d * 2) + lane_div8
                v_bytes_i32 = []
                for j in range_constexpr(8):
                    kv_i64_off = (
                        (k_group * fx.Index(8) + fx.Index(j)) * fx.Index(HEAD_DIM // 8)
                        + hd_off_f
                    )
                    v_i64  = _memref.load(lds_kv_i64, [_raw(kv_i64_off)])
                    v_sh   = _mlir_arith.ShRUIOp(_raw(v_i64), _raw(lane_shift_i64)).result
                    v_b64  = _mlir_arith.AndIOp(v_sh, _raw(fx.Int64(0xFF))).result
                    v_bytes_i32.append(_mlir_arith.TruncIOp(T.i32, v_b64).result)

                # Pack 8 extracted bytes into i64 (byte j at bit position j*8)
                v_b_i64 = _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[0]).result
                for j in range_constexpr(7):
                    bj = _mlir_arith.ShLIOp(
                        _mlir_arith.ExtUIOp(T.i64, v_bytes_i32[j + 1]).result,
                        _raw(fx.Int64((j + 1) * 8))
                    ).result
                    v_b_i64 = _mlir_arith.OrIOp(v_b_i64, bj).result

                o_acc[d] = mfma_fp8(o_acc[d], p_a_i64, v_b_i64)

            gpu.barrier()
            loop_results = yield list(m_run) + list(l_run) + list(o_acc)

        # ── Normalize and write output ────────────────────────────────────
        l_final = [loop_results[4 + r] for r in range_constexpr(4)]
        o_final = [loop_results[8 + d] for d in range_constexpr(D_BLKS)]

        for d in range_constexpr(D_BLKS):
            ov = Vec(o_final[d])
            for r in range_constexpr(4):
                inv_l  = arith.divf(_raw(c_one_f), _raw(l_final[r]), fastmath=fm)
                o_norm = _fmul(ov[r], inv_l)
                o_bf16 = fx.Float32(o_norm).to(fx.BFloat16).ir_value()
                out_row = k_group * fx.Index(4) + fx.Index(r)
                out_col = fx.Index(d * MFMA_N) + lane
                out_idx = fx.Int64(
                    (q_start + out_row) * fx.Index(Q_STOK)
                    + bid_h * fx.Index(HEAD_DIM)
                    + out_col
                )
                out_gep = buffer_ops.get_element_ptr(out_ptr, out_idx, elem_type=T.bf16)
                _llvm.StoreOp(o_bf16, out_gep)

    # ── JIT launcher ────────────────────────────────────────────────────
    @flyc.jit
    def launch_nsa_prefill(
        Q: fx.Tensor, KV: fx.Tensor, Indices: fx.Tensor, Out: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc.finalize()

        tokens_idx = fx.Index(total_tokens)
        grid_m     = (tokens_idx + TILE_M - 1) // TILE_M
        launcher   = nsa_prefill_kernel(Q, KV, Indices, Out, total_tokens)

        passthrough = []
        for pair in [
            ("denormal-fp-math-f32", "preserve-sign,preserve-sign"),
            ("no-nans-fp-math",      "true"),
            ("unsafe-fp-math",       "true"),
        ]:
            passthrough.append(
                ir.ArrayAttr.get([ir.StringAttr.get(pair[0]), ir.StringAttr.get(pair[1])])
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"]        = ir.ArrayAttr.get(passthrough)
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, int(waves_per_eu))

        launcher.launch(
            grid=(grid_m, fx.Index(h_q), 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_hints):
            return launch_nsa_prefill(*args, **kwargs)

    return _launch


def flydsl_nsa_prefill(
    q:       "torch.Tensor",
    kv:      "torch.Tensor",
    indices: "torch.Tensor",
    sm_scale: float,
) -> "torch.Tensor":
    """FlyDSL sparse NSA prefill for gfx950.
    q:       [total_tokens, h_q, 512] float8_e4m3fn
    kv:      [num_pages, 512]          float8_e4m3fn
    indices: [total_tokens, topk]      int32
    Returns: [total_tokens, h_q, 512] bfloat16
    """
    import torch

    total_tokens, h_q, head_dim = q.shape
    topk = indices.shape[1]
    assert q.dtype   == torch.float8_e4m3fn
    assert kv.dtype  == q.dtype
    assert indices.dtype == torch.int32

    out    = torch.empty((total_tokens, h_q, head_dim), dtype=torch.bfloat16, device=q.device)
    kernel = build_nsa_prefill_kernel(h_q=h_q, head_dim=head_dim, topk=topk, sm_scale=sm_scale)
    stream = torch.cuda.current_stream()
    kernel(q, kv, indices, out, total_tokens=total_tokens,
           stream=fx.Stream(stream.cuda_stream))
    return out
