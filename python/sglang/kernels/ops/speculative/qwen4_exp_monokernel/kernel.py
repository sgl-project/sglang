# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next (Qwen4-Exp) MonoKernel: speculative-decoding verify (and MTP draft) of R requests x L tokens.

One persistent launch of 256 CTAs x 512 threads runs whole decoder layers (3 GDN + 1 QSA per launch, 13 launches per
verify); stages hand values over through tagged-pair mailboxes in a global scratch buffer.

R requests x L tokens = S rows (compile-time, S <= 32; row t = request t // L, draft step t % L):
  * FP8 attention / GDN projections (W8A8): ``mfma_scale_f32_16x16x128_f8f6f4`` on sglang's AITER-shuffled FP8
    weights (per-channel scales) and per-token dynamic FP8 activations (AITER's quantization), with the S rows as
    MFMA N-columns in NCT = ceil(S / 16) column tiles that share the weight registers;
  * MXFP4 experts read in place from sglang's FusedMoE params (AITER layout, shared expert fused at index 512);
  * bf16 skinny GEMMs (``mfma_f32_16x16x32_bf16``) for the hyper-connection mixers, router, PLE and MTP input fusion;
  * per-request state: positions / RoPE, KV locations (req_to_token row), mamba slot, verify-scratch row
    (row r of intermediate_ssm / intermediate_conv_window / PLE intermediate state / n-gram context);
  * GDN: the gated delta rule runs each request's L tokens in order on its own register-resident state and writes
    the state / conv window after every token into scratch row r;
  * QSA: per-row causal sparse attention inside its request (row t sees its request's positions <= pos_t, its own
    top-512 blocks); index-key ring / compression per request; two-level exact top-512 per row;
  * MoE: per-row top-10, the union of the routed experts of all rows (+ shared) is computed once; only the routed
    (row, expert) pairs are handed to the down projection.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sglang.kernels.ops.speculative.qwen4_exp_monokernel import buffer_ops as bo
from sglang.kernels.ops.speculative.qwen4_exp_monokernel import layout as LY
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.layout import (
    BLOCKS,
    CM_DEV,
    THREADS,
    WAVES,
    scratch_layout,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import exp as _exp
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    fp8_roundtrip as _fp8_roundtrip,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    mem_realtime,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    mxfp4_to_bf16x8 as _mxfp4_to_bf16x8,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import rcp as _rcp
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import rsq as _rsq
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import rsrc as _rsrc
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    spin_pause,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    uniform as _uniform,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    wave_umax_dpp as _wave_umax,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import xred as _xred
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.primitives import (
    xshfl as _xshfl,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.weights import (
    EPS,
    GDN_DK,
    GDN_HV,
    GDN_KEY_DIM,
    GDN_VAL_DIM,
    HC,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    TOPK,
)

POLL_MAX = 24
NEG_INF = -3.0e38
LDS_WORDS = 40448  # 158 KB arena (gfx950: 160 KB per CU)
RED_OFF = 0  # MFMA accumulator exchange: [tile][column tile][wave][lane][4] f32


def act_stride(n):
    """LDS row stride (words) of a [T][n] bf16 activation: n/2 + 4 keeps (stride / 4) odd, so the 16 MFMA
    B-operand rows (one per token) of a ds_read_b128 land on distinct bank quads."""
    s = n // 2 + 4
    assert (s // 4) % 2 == 1, n
    return s


def act_stride8(n):
    """LDS row stride (words) of a [S][n] FP8 activation: n/4 + 4 keeps (stride / 4) odd (conflict-free B reads)."""
    s = n // 4 + 4
    assert (s // 4) % 2 == 1, n
    return s


def _opaque_i32(v):
    """Side-effecting copy: blocks CSE/hoisting of per-thread values across stage boundaries."""
    from flydsl.expr.typing import as_ir_value

    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [as_ir_value(fx.Int32(v))],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        ).result
    )


def _ordered_i32(v):
    """Side-effecting copy with a memory clobber: memory operations are not reordered across it (bounds how far
    the scheduler hoists independent weight loads)."""
    from flydsl.expr.typing import as_ir_value

    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [as_ir_value(fx.Int32(v))],
            "v_mov_b32 $0, $1",
            "=v,v,~{memory}",
            has_side_effects=True,
        ).result
    )


def _pin(vals, mem=False):
    """Fake use of f32 values by a side-effecting asm: the instructions producing them (MFMAs) stay before the
    next side-effecting operation (barrier) instead of being sunk past it with their operands live. mem: also a
    compiler memory barrier (later LDS / global loads are not hoisted above it)."""
    from flydsl.expr.typing import as_ir_value

    for i in range(0, len(vals), 8):
        part = vals[i : i + 8]
        llvm.InlineAsmOp(
            None,
            [as_ir_value(fx.Float32(v)) for v in part],
            "",
            ",".join(["v"] * len(part)) + (",~{memory}" if mem else ""),
            has_side_effects=True,
        )


def _gload_v4(addr):
    """16-byte global load at a 64-bit byte address (per-lane addresses, no 32-bit buffer offset limit)."""
    from flydsl._mlir import ir

    pt = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), fx.Int64(addr).ir_value()
    ).result
    return fx.Vector(llvm.LoadOp(T.vec(4, T.i32), pt, alignment=16).result)


def _log2(v):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.log.f32", [fx.Float32(v).ir_value()], [], []
        )
    )


TL_LABELS = []


@functools.cache
def _bitonic_pairs(n):
    """Compare-exchange pairs (a, b) of a descending bitonic sort of n = 2^k elements: a receives the larger."""
    out = []
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            for i in range(n):
                partner = i ^ j
                if partner > i:
                    out.append((i, partner) if (i & k) == 0 else (partner, i))
            j //= 2
        k *= 2
    return tuple(out)


@functools.cache
def build_qwen4_exp_monokernel(
    plan: tuple,
    nreq: int,
    ntok: int,
    rtt_stride: int = 0,
    pos_stride: int = 1,
    map_i64: bool = False,
    timeline: bool = False,
    ple_first: bool = False,
    final_mix: bool = False,
    prelude: bool = False,
    eos_token_id: int = 0,
    iconv_strides: tuple = (30720, 3, 1, 4 * 30720),
    issm_row: int = 4 * LY.SSM_STEP_WORDS,
    ple_ist_row: int = 4 * HC_DIM * LY.PLE_STATE_LEN,
    ictx_row: int = 8,
    first_src: str = "global",
    ssm_bf16: bool = False,
):
    """plan: tuple of layer kinds ("gdn" | "qsa"); nreq: requests R; ntok: verify tokens per request L (S = R L
    rows); rtt_stride: req_to_token row stride (int32 elements); pos_stride: element stride between the 3 MRoPE
    position axes (0: 1-D text positions); map_i64: dtype of the req -> mamba slot map; iconv_strides: element
    strides (step, channel, window, request row) of the GDN intermediate conv-window view; issm_row: elements
    per request row of intermediate_ssm; ssm_bf16: GDN SSM states (committed + intermediate_ssm) are bf16
    (SGLANG_MAMBA_SSM_DTYPE=bfloat16; the recurrence stays f32, states are rounded when stored, as sglang); ple_ist_row: bf16 elements per request row of the PLE intermediate
    short-conv state; ictx_row: int64 elements per request row of the n-gram intermediate context. The pointer
    table has one row per layer plus a global row."""
    R, L = nreq, ntok
    NT = R * L
    assert 1 <= R <= LY.MAX_R and 1 <= NT <= LY.MAX_T
    n_layers = len(plan)
    regions, _ = scratch_layout(n_layers + 2, NT, R)
    G = BLOCKS
    NCT = (NT + 15) // 16  # MFMA column tiles of the S rows
    KPH = LY.kphases(NT)  # K phases of the K=2560 activation GEMMs
    KH = HIDDEN // KPH  # K elements staged per phase
    S_ACT = act_stride(KH)  # [S][2560 / KPH] activation rows
    S_A8 = act_stride8(
        HIDDEN
    )  # [S][2560] FP8 activation rows (fit the LDS for every S <= 32)
    # QB (8 <= S <= 16): activations staged in LDS as bf16 (raw mailbox copy, row amax reduced on the fly or from the
    # producers' partials) and quantized to FP8 per token while the MFMA B operands are built (no re-poll, no FP8
    # staging pass). S < 8: FP8 staging (one poll round); S = 32: FP8 staging (two-pass, bf16 rows do not fit the LDS)
    QB = 8 <= NT <= 16
    if QB:
        OP_KS = (
            1 if NT <= 8 else 2
        )  # out_proj K parts staged one after the other (bf16 og [S][6144 / OP_KS])
        S_OG = act_stride(GDN_VAL_DIM // OP_KS)
        OG_BATCH = (
            12  # og polls per round (next to the out_proj weight vectors, no spills)
        )
    else:
        OP_KS = (
            1 if NT <= 16 else 3
        )  # FP8 og [S][6144 / OP_KS] (S = 32: three K parts keep the weights in registers)
        S_OG = act_stride8(GDN_VAL_DIM // OP_KS)
        OG_BATCH = (
            24 if NT <= 4 else (12 if NT <= 16 else 8)
        )  # og polls per round (quantized as they land)
    A8_BATCH = (
        12 if QB or NT > 16 else POLL_MAX
    )  # in_proj / QSA proj activation polls per round
    G8_PD = (
        2 if QB else 1
    )  # in_proj / QSA proj weight tiles in flight (issued before the activation is staged)
    S_HNS = act_stride(1024)  # [S][1024] Hn K-slice of a mix
    S_TACT = act_stride(HC_LOWRANK)  # [S][320]
    SG = LY.dn_sg(NT)  # expert slots per down K-group
    S_MID = act_stride(SG * MOE_INTER)  # [S][SG * 640]
    NS_MAX = LY.max_slots(NT)
    NG_MAX = LY.max_groups(NT, SG)
    NPAIR = LY.MOE_PAIR  # (row, expert) pairs per row
    CN = LY.cnew_slots(L)
    IC_STEP, IC_DIM, IC_WIN, IC_ROW = iconv_strides
    CSPLIT = LY.core_split(NT)
    C_ROWS, C_TASKS, C_RPT, C_KPT, C_RED = LY.core_geom(CSPLIT)
    RED_SLOT = WAVES * 64 * 4  # words of one (tile, column tile) accumulator slot
    KPG = LY.gu_kphases(NT)  # K phases (CTA halves) of the MoE gate/up
    S_GU = act_stride(HIDDEN // KPG)  # [S][2560 / KPG] gate/up activation rows
    W0 = (
        4 if NCT == 1 else NCT
    ) * RED_SLOT  # work area above the accumulator slots of tile 0 (NCT=1: 4 tiles)

    # ---- LDS map (f32 words); stage work areas alias each other above W0
    # tail (every CTA, whole launch): batch tokens, rope positions, per-request tables, work prefix sums
    EMB_TOK = LDS_WORDS - 32  # [S] token ids (prelude / MTP launches)
    POS3 = EMB_TOK - 3 * 32  # [3][S] rope positions of the batch rows
    RQ_REQ = POS3 - 8  # [R] req pool index
    RQ_MS = RQ_REQ - 8  # [R] mamba slot
    RQ_P0 = RQ_MS - 8  # [R] position of the request's first verify token
    SC_PFX = RQ_P0 - 16  # [R + 1] prefix sums of the index-score tiles of the requests
    SL_PFX = (
        SC_PFX - 48
    )  # [S + 1] prefix sums of the top-512 selection tasks (segments) of the rows
    ASC = (
        SL_PFX - 32
    )  # [S] FP8 activation scale of row t (per-token dynamic quantization of the current GEMM)
    AINV = ASC - 32  # [S] its reciprocal (448 / amax)
    TAIL_OFF = AINV
    # persistent across the MoE stages of a layer: route tables
    PERS_WORDS = NS_MAX + 512 + 2 * NT * 16 + NT + 16 + 2 * SG * NT
    PERS_OFF = TAIL_OFF - PERS_WORDS
    SLOT_E = PERS_OFF  # [NS] expert id of slot
    SLOT_OF = SLOT_E + NS_MAX  # [512] slot of expert (valid where flagged)
    FLAG = SLOT_OF  # aliases SLOT_OF before it is written: [512] flags
    TOPI = SLOT_OF + 512  # [S][16] top-10 ids (int)
    TOPW = TOPI + NT * 16  # [S][16] weights
    SHG = TOPW + NT * 16  # [S] shared gate
    NSV = SHG + NT  # [1] number of slots, [8] wave totals
    WTOT = NSV + 4
    RNK = NSV + 16  # [SG][S] down: pair rank of (group slot, row) (-1: not routed)
    SWG = RNK + SG * NT  # [SG][S] down: route weight of (group slot, row)
    assert SWG + SG * NT <= TAIL_OFF
    # mix
    MIX_RSTD = W0  # [S][4]
    MIX_HNS = MIX_RSTD + 4 * NT  # [S][S_HNS]
    MIX_TACT = MIX_HNS + NT * S_HNS  # [S][S_TACT]
    MIX_OUTS = MIX_TACT + NT * S_TACT  # [S][64]
    assert MIX_OUTS + 128 * NT + THREADS <= TAIL_OFF
    # GEMV activation
    OUTS = W0  # small per-stage outputs ([S][16] words, merge scratch) after the GEMM (ACT dead)
    ACT = W0 + 256
    assert ACT + NT * S_OG + 48 * NT <= TAIL_OFF, "out_proj activation does not fit LDS"
    assert ACT + NT * S_ACT <= TAIL_OFF
    RSTD = ACT + NT * S_OG  # [S][48] GDN gated-norm rstd per head
    # MoE (wave-level tasks, no accumulator exchange): small outputs at 0, then the gate/up activation (live with
    # the route tables) / the staged mid of a down group
    MOE_OUTS = 0  # down: routed pair list [SG S], then the group sums [S][16]
    GU_ACT = 512
    assert GU_ACT + NT * S_GU <= PERS_OFF, (
        "gate/up activation + route tables do not fit LDS"
    )
    # GDN core
    C_Q = W0  # [S][288] f32 conv outputs (q 128, k 128, v 32)
    C_Z = C_Q + NT * 288  # [S][32]
    C_O = C_Z + NT * 32  # [S][32]
    C_OG = C_O + NT * 32  # [S][16] words (bf16 og)
    C_SC = C_OG + NT * 16  # [S][4] beta, decay, k.q
    C_BA = C_SC + 4 * NT  # [2][2560] bf16 b/a projection rows of the head
    assert C_BA + HIDDEN <= TAIL_OFF
    # MoE down: staged mid of a slot group
    MID = 512
    assert MID + NT * S_MID <= PERS_OFF and 16 * NT <= MID
    # QSA attention (per task) / select
    ATT_OFF = W0
    ATT_KS = 132
    ATT_VS = 72
    ATT_KV_WORDS = max(64 * ATT_KS, 256 * ATT_VS // 2)
    ATT_Q_OFF = (
        ATT_OFF + 2 * ATT_KV_WORDS
    )  # per kv head: K [64][ATT_KS] words, then V^T [256][ATT_VS] bf16
    ATT_S_OFF = ATT_Q_OFF + 2 * 16 * ATT_KS  # Q [2][16][264] bf16 (rows 12..15 zero)
    ATT_P_OFF = ATT_S_OFF + 2 * 16 * 64  # S [2][16][64] f32
    ATT_M_OFF = (
        ATT_P_OFF + 2 * 16 * ATT_VS // 2
    )  # P [2][16][72] bf16; then m[24], l[24], token flags [64]
    assert ATT_M_OFF + 128 <= TAIL_OFF
    KEYS_OFF = W0
    QI_OFF = W0 + 64  # scores: [4 S][QS] words of index q rows
    PREP_OFF = W0  # prep scratch
    PREP_KEYS = W0 + 1024  # index-key ring of a request's tokens: [L][128] f32
    assert QI_OFF + 4 * NT * 68 <= TAIL_OFF and PREP_KEYS + L * 128 <= TAIL_OFF

    # MTP input fusion (first_src "mtp"): B rows = S normalized embedding rows + 4 S (row, branch) hidden rows,
    # staged in MF_KP K phases; E / H column tiles reduced through red slots after the last phase (B dead)
    MF_ROWS = 5 * NT
    MF_KP = 1
    while MF_ROWS * act_stride(HIDDEN // MF_KP) > 27000:
        MF_KP *= 2
    MF_KC = HIDDEN // MF_KP
    S_MF = act_stride(MF_KC)
    MF_B = 256
    MF_RS = MF_B + MF_ROWS * S_MF  # [2][S] rstd of the embedding / hidden rows
    NCTE = (NT + 15) // 16
    NCTH = (4 * NT + 15) // 16
    assert MF_RS + 2 * NT <= TAIL_OFF and (NCTE + NCTH) * RED_SLOT <= TAIL_OFF

    @fx.struct
    class Smem:
        work: fx.Array[fx.Float32, LDS_WORDS, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def qwen4_exp_monokernel(
        h_in: Int64,
        h_out: Int64,
        ptab: Int64,
        scratch: Int64,
        step: Int64,
        launch_id: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.work.ptr
        xb = fx.recast_iter(
            fx.BFloat16, xs
        )  # bf16 view of the arena (element index = 2 * word)
        xi = fx.recast_iter(fx.Int32, xs)
        red = xs + RED_OFF
        v4f = fx.Vector.make_type(4, fx.Float32)

        step_value = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        tag = step_value * LY.LAUNCH_SLOTS + launch_id + 1
        r_ptab = _rsrc(ptab)

        # ------------------------------------------------------------ helpers
        def ptr(layer, slot):
            pv = fx.Vector(
                bo.buffer_load(
                    r_ptab, (layer * LY.N_PTR + slot) * 2, vec_width=2, dtype=T.i32
                )
            )
            return (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(_uniform(pv[0]))
            )

        def rsrc_if(addr, live, nbytes):
            """Buffer descriptor that is empty (loads return 0, no traffic) when ``live`` is false."""
            return bo.create_buffer_resource_from_addr(
                addr, num_records_bytes=live.select(fx.Int32(nbytes), fx.Int32(0))
            )

        def lds_ld(p, i):
            return fx.ptr_load(p + i)

        def lds_st(p, i, v):
            fx.ptr_store(v, p + i)

        def bf16_pair(a, b):
            return (
                fx.Vector.from_elements([a, b], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Float32)[0]
            )

        def bf2_f32(w):
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(
                fx.Float32
            )

        def bf_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        def sigmoid(v):
            return _rcp(fx.Float32(1.0) + _exp(-v))

        def mb(layer, name):
            return scratch + fx.Int64(regions[f"L{layer}.{name}"])

        def put(base_addr, i, v):
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(
                fx.Vector.from_elements([bits, tag], fx.Int32),
                _rsrc(base_addr),
                i * 2,
                cache_modifier=CM_DEV,
            )

        def put_bf(base_addr, i, a, b):
            put(base_addr, i, bf16_pair(a, b))

        def poll(specs, batch=POLL_MAX):
            """specs = [(base_addr, pair index, npairs in {1, 2}[, ok])] -> [[Int32 bits]]. Entries with a false
            (per-lane) ok are not waited for: they read pair 0 of their mailbox and return 0."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):
                return poll(specs[:batch], batch) + poll(specs[batch:], batch)
            oks = [sp[3] if len(sp) > 3 else None for sp in specs]

            def load_all():
                words = []
                for sp, ok in zip(specs, oks):
                    b, i, n = sp[0], sp[1], sp[2]
                    if const_expr(ok is not None):
                        i = ok.select(fx.Int32(i), fx.Int32(0))
                    w = fx.Vector(
                        bo.buffer_load(
                            _rsrc(b),
                            fx.Int32(i) * 2,
                            vec_width=2 * n,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * sp[2] for sp in specs)
            okw = []
            for sp, ok in zip(specs, oks):
                okw += [ok] * sp[2]

            def pending(v):
                bad = None
                for q in range_constexpr(nw // 2):
                    bq = v[2 * q + 1] != tag
                    if const_expr(okw[q] is not None):
                        bq = bq & okw[q]
                    bad = bq if bad is None else (bad | bq)
                return bad

            v = load_all()
            while pending(v):
                spin_pause()
                v = load_all()
            res, e = [], 0
            for sp, ok in zip(specs, oks):
                vals = [v[e + 2 * q] for q in range(sp[2])]
                if const_expr(ok is not None):
                    vals = [ok.select(x, fx.Int32(0)) for x in vals]
                res.append(vals)
                e += 2 * sp[2]
            return res

        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def block_sums(vs):
            ws = [wave_sum(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = t + lds_ld(red, i * WAVES + w)
                tots.append(t)
            gpu.barrier()
            return tots

        def block_sums_many(vs, chunk=64):
            """block_sums over long lists, in chunks that fit the reduction area."""
            out = []
            for c0 in range_constexpr(0, len(vs), chunk):
                out += block_sums(vs[c0 : c0 + chunk])
            return out

        # ------------------------------------------------ MFMA skinny-GEMM machinery (tokens = N columns)
        def tcol(ct=0):
            """Row of this lane's MFMA B / C column in column tile ct (clamped: columns >= S duplicate the last row)."""
            return fx.min(lane % 16 + 16 * ct, NT - 1)

        def bcol(act_off, stride, k_word):
            """-> (ct -> LDS word of this lane's B operand (8 bf16) for the K chunk at k_word of [S][..] rows)."""
            return lambda ct: act_off + tcol(ct) * stride + k_word + (lane // 16) * 4

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, coef=None, subs=(0, 1)):
            """16 rows x 64 K (chunk kc; subs: its 32-K halves to use) of a packed bf16 weight."""
            wv = [
                (
                    sp,
                    fx.Vector(
                        bo.buffer_load(
                            w_rsrc,
                            (((rg * NKC + kc) * 2 + sp) * 64 + lane) * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    ),
                )
                for sp in subs
            ]
            return ("bf16", wv, coef, b_word)

        def unit_f8(w_rsrc, rg, kc, NKC, b_word, coef=None, q=False):
            """16 rows x 128 K (chunk kc of NKC) of a k128-packed FP8 weight (two 1 KB halves). q: the activation is
            bf16 in LDS (b_word from bcolq), quantized with AINV of the lane's row when the B operand is built."""
            # sglang's AITER layout (shuffle_weight (16, 16) of the FP8 [N, K] weight): the (rg, kc) tile is 2 KB at
            # word (rg NKC + kc) 512, lane (r, g) half h holds K 128 kc + 32 g + 16 h .. +16 at word (2 g + h) 64 + 4 r
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        w_rsrc,
                        (rg * NKC + kc) * 512
                        + ((lane // 16) * 2 + h) * 64
                        + (lane % 16) * 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for h in range(2)
            ]
            return ("f8q" if q else "f8", wv, coef, b_word)

        def bcolq(act_off, stride, k_word):
            """-> (ct -> LDS word of this lane's 32 bf16 activation values for the 128-K chunk at k_word (bf16 words))."""
            return lambda ct: act_off + tcol(ct) * stride + k_word + (lane // 16) * 16

        def bcol8(act_off, stride, k_word):
            """-> (ct -> LDS word of this lane's 32 FP8 activation bytes for the 128-K chunk at k_word of [S][..] rows)."""
            return lambda ct: act_off + tcol(ct) * stride + k_word + (lane // 16) * 8

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None):
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * (K // 128) + kc) * 64 + lane) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            packed = fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    (rg * (K // 128) + kc) * 16 + lane % 16,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            return ("mxfp4", (raw, packed), coef, b_word)

        def mxfp4a_sdw(s_rsrc, rg, d, K):
            """The lane's scale dword of 8-group block d (128-K chunks 2 d, 2 d + 1) of row tile rg: bytes
            (kc % 2) * 2 + rg % 2."""
            D = (K // 32 + 7) // 8
            return fx.Int32(
                bo.buffer_load(
                    s_rsrc,
                    ((((rg // 2) * D + d) * 4 + lane // 16) * 16 + lane % 16),
                    vec_width=1,
                    dtype=T.i32,
                )
            )

        def unit_mxfp4a(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, sdw=None):
            """16 rows x 128 K (chunk kc) of an expert matrix in sglang's AITER MXFP4 layout. Weights
            (shuffle_weight (16, 16)): the tile's 1 KB is contiguous, lane l = row l % 16, K [32 (l / 16), +32) as 4
            words of 8 values; the MFMA K32 step s takes word s, so step s / lane group j covers K 32 j + 8 s + e (the
            B operand from bcolq, + 4 words per step). Scales (e8m0_shuffle, 32-row x 8-group blocks, byte
            ((((a D + kc / 2) 4 + j) 16 + c) 2 + kc % 2) 2 + b for expert row 32 a + 16 b + c, group 4 kc + j): one
            scale per lane and chunk (group 4 kc + j)."""
            raw = fx.Vector(
                bo.buffer_load(
                    w_rsrc,
                    ((rg * (K // 128) + kc) * 64 + lane) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            D = (K // 32 + 7) // 8
            dw = (((rg // 2) * D + kc // 2) * 4 + lane // 16) * 16 + lane % 16
            if const_expr(
                sdw is None
            ):  # else: the dword preloaded by the task (mxfp4a_sdw)
                sdw = fx.Int32(bo.buffer_load(s_rsrc, dw, vec_width=1, dtype=T.i32))
            sw = sdw.shrui(
                fx.Int32(8) * (fx.Int32(2) * (fx.Int32(kc) % 2) + fx.Int32(rg) % 2)
            )
            return ("mxfp4a", (raw, sw), coef, b_word)

        def mma_units(acc, units, nct=NCT):
            """acc: nct x 4 f32 (column tile major); every unit's weights feed the MFMAs of all column tiles."""
            acc = list(acc)
            invs = None
            if const_expr(any(u[0] == "f8q" for u in units)):
                invs = [lds_ld(xs, AINV + tcol(ct)) for ct in range(nct)]
            for fmt, wv, coef, bw in units:
                if const_expr(
                    fmt in ("f8", "f8q")
                ):  # 16 x 128 FP8 weight tile x 128 x 16 FP8 activation (W8A8)
                    a8 = fx.Vector.from_elements(
                        [wv[h][e] for h in range(2) for e in range(4)], fx.Int32
                    )
                    for ct in range_constexpr(nct):
                        cf = coef(ct) if const_expr(callable(coef)) else coef
                        if const_expr(fmt == "f8"):
                            bv = [
                                fx.Vector(
                                    fx.ptr_load(xs + (bw(ct) + 4 * h), result_type=v4f)
                                ).bitcast(fx.Int32)
                                for h in range(2)
                            ]
                            b8 = fx.Vector.from_elements(
                                [bv[h][e] for h in range(2) for e in range(4)], fx.Int32
                            )
                        else:  # 32 bf16 (16 words) -> per-token FP8 (AITER: cvt(x * (1 / (amax / 448))))
                            bv = [
                                fx.Vector(
                                    fx.ptr_load(xs + (bw(ct) + 4 * h), result_type=v4f)
                                ).bitcast(fx.Int32)
                                for h in range(4)
                            ]
                            inv = invs[ct]
                            words = []
                            for j in range_constexpr(8):
                                l0, h0 = bf2_f32(bv[j // 2][(j % 2) * 2])
                                l1, h1 = bf2_f32(bv[j // 2][(j % 2) * 2 + 1])
                                w01 = fx.Int32(
                                    rocdl.cvt_pk_fp8_f32(
                                        T.i32, l0 * inv, h0 * inv, fx.Int32(0), False
                                    )
                                )
                                words.append(
                                    fx.Int32(
                                        rocdl.cvt_pk_fp8_f32(
                                            T.i32, l1 * inv, h1 * inv, w01, True
                                        )
                                    )
                                )
                            b8 = fx.Vector.from_elements(words, fx.Int32)
                        one = fx.Int32(
                            127
                        )  # E8M0 1.0: per-channel / per-token scales are applied to the outputs
                        c = fx.Vector(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    a8,
                                    b8,
                                    fx.Vector.filled(4, 0.0, fx.Float32),
                                    0,
                                    0,
                                    0,
                                    one,
                                    0,
                                    one,
                                ],
                            )
                        )
                        for e in range_constexpr(4):
                            if const_expr(cf is None):
                                acc[ct * 4 + e] = acc[ct * 4 + e] + c[e]
                            else:
                                acc[ct * 4 + e] = acc[ct * 4 + e] + c[e] * cf
                else:
                    if const_expr(fmt == "bf16"):
                        av = [(sp, w.bitcast(fx.BFloat16)) for sp, w in wv]
                    else:
                        raw, packed = wv
                        av = []
                        if const_expr(
                            fmt == "mxfp4a"
                        ):  # one scale per lane (group 4 kc + lane / 16)
                            sc = ((packed & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(
                                fx.Float32
                            )
                        for sp in range_constexpr(4):
                            if const_expr(fmt != "mxfp4a"):
                                sc = (
                                    (packed.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF))
                                    << fx.Int32(23)
                                ).bitcast(fx.Float32)
                            av.append((sp, _mxfp4_to_bf16x8(raw[sp], sc)))
                    BSTEP = (
                        4 if fmt == "mxfp4a" else 16
                    )  # words between the B operands of K32 steps
                    for ct in range_constexpr(nct):
                        cf = (
                            coef(ct) if const_expr(callable(coef)) else coef
                        )  # per-lane factor (e.g. per-row rstd)
                        c = fx.Vector.filled(4, 0.0, fx.Float32)
                        for sp, a in av:
                            b = fx.ptr_load(
                                xs + (bw(ct) + sp * BSTEP), result_type=v4f
                            ).bitcast(fx.BFloat16)
                            c = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c])
                            )
                        for e in range_constexpr(4):
                            if const_expr(cf is None):
                                acc[ct * 4 + e] = acc[ct * 4 + e] + c[e]
                            else:
                                acc[ct * 4 + e] = acc[ct * 4 + e] + c[e] * cf
            return acc

        def zero_acc():
            return [fx.Float32(0.0)] * (4 * NCT)

        def store_acc(acc, tile=0):
            for ct in range_constexpr(NCT):
                fx.ptr_store(
                    fx.Vector.from_elements(acc[4 * ct : 4 * ct + 4], fx.Float32),
                    red + (((tile * NCT + ct) * WAVES + wave) * 64 + lane) * 4,
                )

        def red_rows(w0, nw, r, t, tile=0):
            """Row r (0..15), batch row (column) t of the summed MFMA tiles of waves [w0, w0 + nw)."""
            tot = fx.Float32(0.0)
            base = (
                tile * NCT * RED_SLOT
                + (t // 16) * RED_SLOT
                + 16 * (r // 4) * 4
                + (t % 16) * 4
                + r % 4
            )
            for j in range_constexpr(nw):
                tot = tot + lds_ld(red, base + (w0 + j) * 256)
            return tot

        def gemm_tile(units, tile=0):
            store_acc(mma_units(zero_acc(), units), tile)

        # ------------------------------------------------ activation staging
        def stage_bf16_mb(
            base_addr, n_elems, act_off, stride, batch=POLL_MAX, row_elems=None, elem0=0
        ):
            """LDS bf16 [S][n_elems] (row stride words) := elements [elem0, elem0 + n_elems) of the packed bf16
            mailbox [S][row_elems]; poll rounds of `batch` 16-byte polls per thread (4 VGPRs each, live next to
            any prefetched weights), each round stored before the next is issued."""
            row_elems = n_elems if row_elems is None else row_elems
            nq = n_elems // 4  # 16-byte polls per row
            tot = NT * nq
            full = tot // THREADS
            for i0 in range_constexpr(0, full, batch):
                qs = [tid + i * THREADS for i in range(i0, min(i0 + batch, full))]
                vals = poll(
                    [
                        (
                            base_addr,
                            (q // nq) * (row_elems // 2) + elem0 // 2 + (q % nq) * 2,
                            2,
                        )
                        for q in qs
                    ],
                    batch,
                )
                for i in range_constexpr(len(qs)):
                    q = qs[i]
                    w = act_off + (q // nq) * stride + (q % nq) * 2
                    lds_st(xi, w, vals[i][0])
                    lds_st(xi, w + 1, vals[i][1])
            if const_expr(tot % THREADS):
                q = tid + full * THREADS
                if q < tot:
                    v = poll(
                        [
                            (
                                base_addr,
                                (q // nq) * (row_elems // 2)
                                + elem0 // 2
                                + (q % nq) * 2,
                                2,
                            )
                        ]
                    )[0]
                    w = act_off + (q // nq) * stride + (q % nq) * 2
                    lds_st(xi, w, v[0])
                    lds_st(xi, w + 1, v[1])
            gpu.barrier()

        def stage_bf16_global(addr, n_elems, act_off, stride, row_elems=None, elem0=0):
            """LDS bf16 [S][n_elems] := elements [elem0, elem0 + n_elems) of plain global bf16 [S][row_elems]."""
            row_elems = n_elems if row_elems is None else row_elems
            nq = n_elems // 4
            tot = NT * nq
            r = _rsrc(addr)
            for i in range_constexpr((tot + THREADS - 1) // THREADS):
                q = fx.min(tid + i * THREADS, tot - 1)
                w = fx.Vector(
                    bo.buffer_load(
                        r,
                        (q // nq) * (row_elems // 2) + elem0 // 2 + (q % nq) * 2,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                o = act_off + (q // nq) * stride + (q % nq) * 2
                lds_st(xi, o, w[0])
                lds_st(xi, o + 1, w[1])
            gpu.barrier()

        # ---------------------------------------------- residual-stream sources
        # src = (kind, addr, ssq_kind, ssq_addr, nparts): kind "global" (bf16 [T][HC_DIM]), "mb" (pairs),
        # "embed" (token embedding rows, 4 identical branches; tokens in LDS EMB_TOK)

        def h_words(src, t, word_idx):
            """[(t, w) for w in word_idx] -> Int32 bf16 pairs of H (one batched poll for mailboxes)."""
            kind, addr = src[0], src[1]
            if const_expr(kind == "embed"):
                out = []
                for w in word_idx:
                    tok = lds_ld(xi, EMB_TOK + t)
                    row = addr + fx.Int64(tok) * (HIDDEN * 2)
                    out.append(
                        fx.Int32(
                            bo.buffer_load(
                                _rsrc(row), w % (HIDDEN // 2), vec_width=1, dtype=T.i32
                            )
                        )
                    )
                return out
            if const_expr(kind == "global"):
                r = _rsrc(addr)
                return [
                    fx.Int32(
                        bo.buffer_load(
                            r, t * (HC_DIM // 2) + w, vec_width=1, dtype=T.i32
                        )
                    )
                    for w in word_idx
                ]
            return [
                v[0] for v in poll([(addr, t * (HC_DIM // 2) + w, 1) for w in word_idx])
            ]

        def h_quads(src, tq):
            """[(t, q)] -> f32 x4 of H elements 4q..4q+3 of token t."""
            kind, addr = src[0], src[1]
            if const_expr(kind == "embed"):
                ws = []
                for t, q in tq:
                    tok = lds_ld(xi, EMB_TOK + t)
                    row = addr + fx.Int64(tok) * (HIDDEN * 2)
                    ws.append(
                        fx.Vector(
                            bo.buffer_load(
                                _rsrc(row),
                                (q % (HIDDEN // 4)) * 2,
                                vec_width=2,
                                dtype=T.i32,
                            )
                        )
                    )
            elif const_expr(kind == "global"):
                r = _rsrc(addr)
                ws = [
                    fx.Vector(
                        bo.buffer_load(
                            r, t * (HC_DIM // 2) + q * 2, vec_width=2, dtype=T.i32
                        )
                    )
                    for t, q in tq
                ]
            else:
                ws = [
                    fx.Vector.from_elements(v, fx.Int32)
                    for v in poll([(addr, t * (HC_DIM // 2) + q * 2, 2) for t, q in tq])
                ]
            return [w.bitcast(fx.BFloat16).to(fx.Float32) for w in ws]

        def mix_rstd(src):
            """LDS MIX_RSTD[t][b] = rsqrt(mean(H[t, branch b]^2) + eps) (every CTA)."""
            kind = src[0]
            if const_expr(kind == "embed"):
                # one embedding row per token: sum of squares over HIDDEN (identical for the 4 branches)
                NQ = HIDDEN // 4
                vals = []
                for t in range_constexpr(NT):
                    acc = fx.Float32(0.0)
                    for i in range_constexpr((NQ + THREADS - 1) // THREADS):
                        q = tid + i * THREADS
                        hv = h_quads(src, [(t, fx.min(q, NQ - 1))])[0]
                        s = (
                            hv[0] * hv[0]
                            + hv[1] * hv[1]
                            + hv[2] * hv[2]
                            + hv[3] * hv[3]
                        )
                        acc = acc + (q < NQ).select(s, fx.Float32(0.0))
                    vals.append(acc)
                tots = block_sums_many(vals)
                if tid < 4 * NT:
                    tsel = tots[0]
                    for t in range_constexpr(1, NT):
                        tsel = (tid // 4 == t).select(tots[t], tsel)
                    lds_st(xs, MIX_RSTD + tid, _rsq(tsel * (1.0 / HIDDEN) + EPS))
                gpu.barrier()
                return
            _, _, skind, saddr, npart = src
            # wave w sums (t, b) combos c = w, w + 8, ...; lane l reads partials l, l + 64, ...
            NC = 4 * NT
            for c0 in range_constexpr(0, NC, WAVES):
                c = c0 + wave
                cc = fx.min(c, NC - 1)
                idx = [
                    cc * npart + fx.min(lane + 64 * i, npart - 1)
                    for i in range((npart + 63) // 64)
                ]
                if const_expr(skind == "global"):
                    r = _rsrc(saddr)
                    vs = [
                        fx.Int32(
                            bo.buffer_load(r, j, vec_width=1, dtype=T.i32)
                        ).bitcast(fx.Float32)
                        for j in idx
                    ]
                else:
                    vs = [
                        v[0].bitcast(fx.Float32)
                        for v in poll([(saddr, j, 1) for j in idx])
                    ]
                acc = fx.Float32(0.0)
                for i in range_constexpr(len(vs)):
                    acc = acc + (lane + 64 * i < npart).select(vs[i], fx.Float32(0.0))
                tot = wave_sum(acc)
                if (lane == 0) & (c < NC):
                    lds_st(xs, MIX_RSTD + c, _rsq(tot * (1.0 / HIDDEN) + EPS))
            gpu.barrier()

        def hn_value(src, t, k, hv, wn):
            """bf16(Hn[t, k]) from H value hv and norm weight wn."""
            r = lds_ld(xs, MIX_RSTD + t * 4 + k // HIDDEN)
            return bf_round(hv * r * (fx.Float32(1.0) + wn))

        # ---------------------------------------------- hyper-connection mix
        def stage_mix(layer, hc, src, p_norm, p_down, p_up, p_inj, x_out=None):
            """x[t] = mean_b sigmoid(W_up silu(W_down Hn[t] / HC))_b * Hn[t]_b -> mailbox mixx [T][2560];
            combine-gate partials (per K-slice) -> injp."""
            NKC_D = HC_DIM // 64
            CPT = NKC_D // LY.MIX_KSPLIT  # 16 chunks = 1024 K per slice
            CPW = CPT // WAVES
            is_dn = bid < LY.MIX_DOWN_TASKS
            r_wd = _rsrc(ptr(layer, p_down))
            rg_d = fx.min(bid, LY.MIX_DOWN_TASKS - 1) // LY.MIX_KSPLIT
            ks_d = fx.min(bid, LY.MIX_DOWN_TASKS - 1) % LY.MIX_KSPLIT
            k0 = ks_d * (CPT * 64)
            dunits = [
                unit_bf16(
                    r_wd,
                    rg_d,
                    ks_d * CPT + wave * CPW + c,
                    NKC_D,
                    bcol(MIX_HNS, S_HNS, (wave * CPW + c) * 32),
                )
                for c in range(CPW)
            ]
            r_wn = _rsrc(ptr(layer, p_norm))
            mixd = mb(layer, hc + ".mixd")
            mixx = mb(layer, hc + ".mixx")
            NQ = 256 * NT  # quads of the Hn K-slice [k0, k0 + 1024) of all tokens
            NIQ = (NQ + THREADS - 1) // THREADS
            hitems = []
            for i in range_constexpr(NIQ):
                if const_expr(
                    NQ % THREADS == 0
                ):  # affine in i: the poll addresses share base registers
                    hitems.append((tid // 256 + (THREADS // 256) * i, tid % 256))
                else:
                    q = fx.min(tid + i * THREADS, NQ - 1)
                    hitems.append((q // 256, q % 256))
            if const_expr(src[0] == "embed"):
                mix_rstd(src)
                hvs = h_quads(src, [(t, k0 // 4 + j) for t, j in hitems])
            else:
                # one poll round: this wave's sum-of-squares partials (rstd of its (t, b) combos) + the H quads
                _, haddr, skind, saddr, npart = src
                NC = 4 * NT
                NPL = (npart + 63) // 64
                combos = [c0 + wave for c0 in range(0, NC, WAVES)]
                sidx = [
                    (c if c0_ + WAVES <= NC else fx.min(c, NC - 1)) * npart
                    + (
                        lane + 64 * i
                        if 64 * i + 64 <= npart
                        else fx.min(lane + 64 * i, npart - 1)
                    )
                    for c0_, c in zip(range(0, NC, WAVES), combos)
                    for i in range(NPL)
                ]
                hspecs = [(t, k0 // 4 + j) for t, j in hitems]
                if const_expr(skind == "mb" and src[0] == "mb"):
                    got = poll(
                        [(saddr, j_, 1) for j_ in sidx]
                        + [(haddr, t * (HC_DIM // 2) + q * 2, 2) for t, q in hspecs],
                        32,
                    )
                    svals = [g_[0].bitcast(fx.Float32) for g_ in got[: len(sidx)]]
                    hvs = [
                        fx.Vector.from_elements(g_, fx.Int32) for g_ in got[len(sidx) :]
                    ]  # bf16, widened at use
                else:
                    if const_expr(skind == "global"):
                        r_ss = _rsrc(saddr)
                        svals = [
                            fx.Int32(
                                bo.buffer_load(r_ss, j_, vec_width=1, dtype=T.i32)
                            ).bitcast(fx.Float32)
                            for j_ in sidx
                        ]
                    else:
                        svals = [
                            g_[0].bitcast(fx.Float32)
                            for g_ in poll([(saddr, j_, 1) for j_ in sidx])
                        ]
                    hvs = h_quads(src, hspecs)
                for ci in range_constexpr(len(combos)):
                    acc = fx.Float32(0.0)
                    for i in range_constexpr(NPL):
                        acc = acc + (lane + 64 * i < npart).select(
                            svals[ci * NPL + i], fx.Float32(0.0)
                        )
                    tot = wave_sum(acc)
                    if (lane == 0) & (combos[ci] < NC):
                        lds_st(
                            xs, MIX_RSTD + combos[ci], _rsq(tot * (1.0 / HIDDEN) + EPS)
                        )
                gpu.barrier()
            stamp(f"L{layer}.{hc}.rstd")
            if is_dn:
                for n_ in range_constexpr(NIQ):
                    t, j = hitems[n_]
                    k = k0 + 4 * j
                    wv = (
                        fx.Vector(
                            bo.buffer_load(r_wn, k // 2, vec_width=2, dtype=T.i32)
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    hq = hvs[n_]
                    if const_expr(src[0] == "mb" and skind == "mb"):
                        hq = hq.bitcast(fx.BFloat16).to(fx.Float32)
                    o = [hn_value(src, t, k, hq[e], wv[e]) for e in range(4)]
                    lds_st(xs, MIX_HNS + t * S_HNS + j * 2, bf16_pair(o[0], o[1]))
                    lds_st(xs, MIX_HNS + t * S_HNS + j * 2 + 1, bf16_pair(o[2], o[3]))
                gpu.barrier()
                gemm_tile(dunits)
                gpu.barrier()
                if rg_d < LY.MIX_DOWN_RG - 1:
                    if tid < 16 * NT:
                        r = tid % 16
                        t = tid // 16
                        put(
                            mixd,
                            (ks_d * NT + t) * HC_LOWRANK + rg_d * 16 + r,
                            red_rows(0, WAVES, r, t),
                        )
                else:  # combine-gate tile: rows 0..3 = Hn[t, slice] . Winj_b[slice]
                    if tid < 4 * NT:
                        b = tid % 4
                        t = tid // 4
                        put(
                            mb(layer, hc + ".injp"),
                            (ks_d * NT + t) * 4 + b,
                            red_rows(0, WAVES, b, t),
                        )
            stamp(f"L{layer}.{hc}.down")
            # ---- up + branch mean: task u covers outputs [16u, 16u+16) of all 4 branches, all tokens
            NKC_U = HC_LOWRANK // 64
            is_up = bid < LY.MIX_UP_TASKS
            r_wu = rsrc_if(ptr(layer, p_up), is_up, HC_DIM * HC_LOWRANK * 2)
            u = fx.min(bid, LY.MIX_UP_TASKS - 1)
            b = wave // 2
            h = wave % 2
            pre = []
            for c in range_constexpr(3):
                kk = h + 2 * c
                kc = fx.min(kk, NKC_U - 1)
                pre.append(
                    unit_bf16(
                        r_wu,
                        b * (HIDDEN // 16) + u,
                        kc,
                        NKC_U,
                        bcol(MIX_TACT, S_TACT, kc * 32),
                        coef=(kk < NKC_U).select(fx.Float32(1.0), fx.Float32(0.0)),
                    )
                )
            if is_up:
                NPP = NT * (HC_LOWRANK // 2)
                NI = (NPP + THREADS - 1) // THREADS
                qs = [fx.min(tid + i * THREADS, NPP - 1) for i in range(NI)]
                # + the residual words of this thread's output elements o = tid + p THREADS < 64 T:
                #   token o // 64, k = bb * HIDDEN + 16 u + r
                NOP = (64 * NT + THREADS - 1) // THREADS
                hspec = []
                if const_expr(src[0] == "mb"):
                    for p_ in range_constexpr(NOP):
                        o_ = fx.min(tid + p_ * THREADS, 64 * NT - 1)
                        ko = ((o_ % 64) // 16) * HIDDEN + u * 16 + o_ % 16
                        hspec.append((src[1], (o_ // 64) * (HC_DIM // 2) + ko // 2, 1))
                ICH = 5  # output pairs per poll round (their 10 K-slice partials live together)
                for i0 in range_constexpr(0, NI, ICH):
                    ii = list(range(i0, min(i0 + ICH, NI)))
                    extra = hspec if i0 == 0 else []
                    parts = poll(
                        [
                            (
                                mixd,
                                (s_ * NT + qs[i] // (HC_LOWRANK // 2)) * HC_LOWRANK
                                + (qs[i] % (HC_LOWRANK // 2)) * 2,
                                2,
                            )
                            for i in ii
                            for s_ in range(LY.MIX_KSPLIT)
                        ]
                        + extra,
                        32 if NT <= 8 else 16,
                    )
                    if const_expr(src[0] == "mb" and i0 == 0):
                        for p_ in range_constexpr(NOP):
                            lds_st(
                                xi,
                                MIX_OUTS + 64 * NT + tid + p_ * THREADS,
                                parts[len(parts) - NOP + p_][0],
                            )
                    for k_ in range_constexpr(len(ii)):
                        i = ii[k_]
                        q = qs[i]
                        d0 = fx.Float32(0.0)
                        d1 = fx.Float32(0.0)
                        for s_ in range_constexpr(LY.MIX_KSPLIT):
                            d0 = d0 + parts[k_ * LY.MIX_KSPLIT + s_][0].bitcast(
                                fx.Float32
                            )
                            d1 = d1 + parts[k_ * LY.MIX_KSPLIT + s_][1].bitcast(
                                fx.Float32
                            )
                        d0 = d0 * (1.0 / HC)
                        d1 = d1 * (1.0 / HC)
                        if tid + i * THREADS < NPP:
                            lds_st(
                                xs,
                                MIX_TACT
                                + (q // (HC_LOWRANK // 2)) * S_TACT
                                + q % (HC_LOWRANK // 2),
                                bf16_pair(d0 * sigmoid(d0), d1 * sigmoid(d1)),
                            )
            gpu.barrier()
            if is_up:
                gemm_tile(pre)
                gpu.barrier()
                for p_ in range_constexpr((64 * NT + THREADS - 1) // THREADS):
                    o_ = tid + p_ * THREADS
                    if o_ < 64 * NT:
                        t = o_ // 64
                        bb = (o_ % 64) // 16
                        r = o_ % 16
                        uval = red_rows(bb * 2, 2, r, t)
                        k = bb * HIDDEN + u * 16 + r
                        if const_expr(src[0] == "mb"):
                            w = lds_ld(xi, MIX_OUTS + 64 * NT + o_)
                        else:
                            w = h_words(src, t, [k // 2])[0]
                        lo, hi = bf2_f32(w)
                        hv = (k % 2 == 0).select(lo, hi)
                        wn = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(r_wn, k, vec_width=1, dtype=T.bf16)
                            )
                        )
                        lds_st(
                            xs,
                            MIX_OUTS + o_,
                            sigmoid(uval) * hn_value(src, t, k, hv, wn),
                        )
                gpu.barrier()
                if tid < 8 * NT:
                    t = tid // 8
                    p = tid % 8
                    xv = []
                    for e in range_constexpr(2):
                        r = p * 2 + e
                        o_ = MIX_OUTS + t * 64 + r
                        s_ = (
                            lds_ld(xs, o_)
                            + lds_ld(xs, o_ + 16)
                            + lds_ld(xs, o_ + 32)
                            + lds_ld(xs, o_ + 48)
                        )
                        xv.append(s_ * (1.0 / HC))
                    put_bf(mixx, t * (HIDDEN // 2) + u * 8 + p, xv[0], xv[1])
                    if const_expr(x_out is not None):
                        bo.buffer_store(
                            bf16_pair(xv[0], xv[1]).bitcast(fx.Int32),
                            _rsrc(x_out),
                            t * (HIDDEN // 2) + u * 8 + p,
                        )
                gpu.barrier()

        def inj_gate(layer, hc, t, b):
            """Combine gate 2 sigmoid(Hn[t] . Winj_b / HC) from the mix's per-slice partials."""
            g = poll(
                [
                    (mb(layer, hc + ".injp"), (s * NT + t) * 4 + b, 1)
                    for s in range(LY.MIX_KSPLIT)
                ]
            )
            tot = g[0][0].bitcast(fx.Float32)
            for s in range_constexpr(1, LY.MIX_KSPLIT):
                tot = tot + g[s][0].bitcast(fx.Float32)
            return fx.Float32(2.0) * sigmoid(tot * (1.0 / HC))

        def combine_rows(layer, hc, src, rg, block_val, dst, dst_ssq, out_global=False):
            """dst[t, b*HIDDEN + 16 rg + r] = H + block_out[t, 16 rg + r] * gate[t, b]  (tid < 32 T);
            plus the per-(t, b) sum-of-squares partial rg of the new stream. block_val(p, t) -> (v0, v1)."""
            NCT = 32 * NT
            for i in range_constexpr((NCT + THREADS - 1) // THREADS):
                q = tid + i * THREADS
                if q < NCT:
                    t = q // 32
                    b = (q % 32) // 8
                    p = q % 8
                    row = b * HIDDEN + rg * 16 + p * 2
                    injp = mb(layer, hc + ".injp")
                    gspecs = [
                        (injp, (s_ * NT + t) * 4 + b, 1) for s_ in range(LY.MIX_KSPLIT)
                    ]
                    if const_expr(
                        src[0] == "mb"
                    ):  # one batched poll: residual word + combine-gate partials
                        got = poll([(src[1], t * (HC_DIM // 2) + row // 2, 1)] + gspecs)
                        hw = got[0][0]
                        gp = got[1:]
                    else:
                        hw = h_words(src, t, [row // 2])[0]
                        gp = poll(gspecs)
                    gsum = gp[0][0].bitcast(fx.Float32)
                    for s_ in range_constexpr(1, LY.MIX_KSPLIT):
                        gsum = gsum + gp[s_][0].bitcast(fx.Float32)
                    gt = fx.Float32(2.0) * sigmoid(gsum * (1.0 / HC))
                    h0, h1 = bf2_f32(hw)
                    v0, v1 = block_val(p, t)
                    o0 = bf_round(h0 + v0 * gt)
                    o1 = bf_round(h1 + v1 * gt)
                    put_bf(dst, t * (HC_DIM // 2) + row // 2, o0, o1)
                    if const_expr(out_global):
                        bo.buffer_store(
                            bf16_pair(o0, o1).bitcast(fx.Int32),
                            _rsrc(h_out),
                            t * (HC_DIM // 2) + row // 2,
                        )
                    ss = o0 * o0 + o1 * o1
                    for off in (4, 2, 1):
                        ss = _xred(ss, off, lambda a, c: a + c)
                    if p == 0:
                        put(dst_ssq, (t * 4 + b) * LY.NPART + rg, ss)
                        if const_expr(out_global):
                            bo.buffer_store(
                                ss.bitcast(fx.Int32),
                                _rsrc(ptr(GI, LY.G_HOUT_SSQ)),
                                (t * 4 + b) * LY.NPART + rg,
                            )

        # ---------------------------------------------- GDN: input projection
        def act_gemm(stage_phase, r_w, NTI, rg_of, emit):
            """Row tiles rg_of(i) (i < NTI) of a packed bf16 [*, 2560] weight times the S activation rows.
            stage_phase(h): stage K phase h of the activation into ACT [S][2560 / KPH] (+ barrier); emit(i, rg):
            outputs of tile i from the accumulator slot (tile 0). KPH = 1: tiles one after the other, next weights in
            flight during the MMAs; KPH = 2: per K half all tiles (accumulators kept), then the outputs."""
            NKC = HIDDEN // 64
            if const_expr(KPH == 1):
                CPW = NKC // WAVES

                def units_for(i):
                    rg = rg_of(i)
                    return [
                        unit_bf16(
                            r_w,
                            rg,
                            wave * CPW + c,
                            NKC,
                            bcol(ACT, S_ACT, (wave * CPW + c) * 32),
                        )
                        for c in range(CPW)
                    ]

                cur = units_for(0)
                stage_phase(0)
                for i in range_constexpr(NTI):
                    nxt = units_for(i + 1) if i + 1 < NTI else None
                    gemm_tile(cur)
                    gpu.barrier()
                    emit(i, rg_of(i))
                    gpu.barrier()
                    cur = nxt
            else:
                NQ = 2 * NKC // KPH  # 32-K sub-chunks per phase
                QPW = NQ // WAVES

                def units_for(i, h):
                    # opaque row group: the loads stay behind the previous tile's MMAs (one tile of weights in flight)
                    rg = _ordered_i32(rg_of(i))
                    return [
                        unit_bf16(
                            r_w,
                            rg,
                            0,
                            NKC,
                            bcol(ACT, S_ACT, -h * (KH // 2)),
                            subs=(h * NQ + wave * QPW + j,),
                        )
                        for j in range(QPW)
                    ]

                accs = [zero_acc() for _ in range(NTI)]
                for h in range_constexpr(KPH):
                    if const_expr(h > 0):
                        gpu.barrier()  # every wave is done with the previous K phase
                    cur = units_for(0, h)
                    stage_phase(h)
                    for i in range_constexpr(NTI):
                        nxt = units_for(i + 1, h) if i + 1 < NTI else None
                        accs[i] = mma_units(accs[i], cur)
                        cur = nxt
                    if const_expr(h + 1 < KPH):
                        for i in range_constexpr(NTI):
                            _pin(
                                accs[i]
                            )  # phase h MMAs complete before the next K phase is staged
                for i in range_constexpr(NTI):
                    store_acc(accs[i])
                    gpu.barrier()
                    emit(i, rg_of(i))
                    gpu.barrier()

        def stage_mb_phase(addr):
            return lambda h: stage_bf16_mb(
                addr, KH, ACT, S_ACT, row_elems=HIDDEN, elem0=h * KH
            )

        def emit_bf_rows(mbx, row_words, valid=None):
            """emit(i, rg): rows [16 rg, 16 rg + 16) of every batch row -> bf16 pairs of mailbox [S][row_words]."""

            def emit(i, rg):
                for p0 in range_constexpr(0, 8 * NT, THREADS):
                    q = tid + p0
                    ok = q < 8 * NT
                    if const_expr(valid is not None):
                        ok = ok & valid(i)
                    if ok:
                        t = q // 8
                        p = q % 8
                        put_bf(
                            mbx,
                            t * row_words + rg * 8 + p,
                            red_rows(0, WAVES, p * 2, t),
                            red_rows(0, WAVES, p * 2 + 1, t),
                        )

            return emit

        def stage_f8_mb(
            base_addr,
            n_elems,
            act_off,
            stride,
            row_elems=None,
            elem0=0,
            amax_known=False,
            xform=None,
            batch=POLL_MAX,
        ):
            """LDS FP8 [S][n_elems] (row stride words, 4 values per word) := per-token dynamic FP8 quantization of
            elements [elem0, elem0 + n_elems) of the packed bf16 mailbox [S][row_elems] (optionally transformed by
            xform(row, element, f32) first). amax_known: ASC / AINV already hold the rows' scales; else the row amax
            is reduced here (a wave's 64 quads lie in one row) and ASC[t] = amax / 448, AINV[t] = 448 / amax."""
            row_elems = n_elems if row_elems is None else row_elems
            nq = n_elems // 4
            assert nq % 64 == 0
            tot = NT * nq
            NI = (tot + THREADS - 1) // THREADS
            qs = [tid + i * THREADS for i in range(NI)]
            words = [None] * NI

            def poll_batch(i0):
                got = poll(
                    [
                        (
                            base_addr,
                            (fx.min(q, tot - 1) // nq) * (row_elems // 2)
                            + elem0 // 2
                            + (fx.min(q, tot - 1) % nq) * 2,
                            2,
                        )
                        for q in qs[i0 : i0 + batch]
                    ],
                    batch,
                )
                for k_ in range_constexpr(len(got)):
                    words[i0 + k_] = got[k_]

            def quant_store(i):
                q = fx.min(qs[i], tot - 1)
                v = vals_of(i)
                inv = lds_ld(xs, AINV + q // nq)
                w01 = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(
                        T.i32, v[0] * inv, v[1] * inv, fx.Int32(0), False
                    )
                )
                w = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(T.i32, v[2] * inv, v[3] * inv, w01, True)
                )
                if qs[i] < tot:
                    lds_st(xi, act_off + (q // nq) * stride + q % nq, w)

            def vals_of(i):
                q = fx.min(qs[i], tot - 1)
                lo0, hi0 = bf2_f32(words[i][0])
                lo1, hi1 = bf2_f32(words[i][1])
                v = [lo0, hi0, lo1, hi1]
                if const_expr(xform is not None):
                    v = [
                        xform(q // nq, elem0 + (q % nq) * 4 + e, v[e]) for e in range(4)
                    ]
                return v

            if const_expr(
                amax_known
            ):  # quantize each poll round as it lands (no values held across rounds)
                for i0 in range_constexpr(0, NI, batch):
                    poll_batch(i0)
                    for i in range_constexpr(i0, min(i0 + batch, NI)):
                        quant_store(i)
                gpu.barrier()
                return
            TWO_PASS = (
                NI > batch
            )  # more than one poll round: reduce the amax round by round, re-poll to quantize
            for i0 in range_constexpr(0, NI, batch):
                poll_batch(i0)
                if const_expr(TWO_PASS):
                    for i in range_constexpr(i0, min(i0 + batch, NI)):
                        v = vals_of(i)
                        m = fx.max(
                            fx.max(fx.max(v[0], -v[0]), fx.max(v[1], -v[1])),
                            fx.max(fx.max(v[2], -v[2]), fx.max(v[3], -v[3])),
                        )
                        for off in (32, 16, 8, 4, 2, 1):
                            m = _xred(m, off, fx.max)
                        if lane == 0:
                            lds_st(red, i * WAVES + wave, m)
            if const_expr(not TWO_PASS):
                for i in range_constexpr(NI):
                    v = vals_of(i)
                    m = fx.max(
                        fx.max(fx.max(v[0], -v[0]), fx.max(v[1], -v[1])),
                        fx.max(fx.max(v[2], -v[2]), fx.max(v[3], -v[3])),
                    )
                    for off in (32, 16, 8, 4, 2, 1):
                        m = _xred(m, off, fx.max)
                    if lane == 0:
                        lds_st(red, i * WAVES + wave, m)
            if const_expr(True):
                gpu.barrier()
                if tid < NT:
                    NPR = nq // 64  # wave partials per row
                    am = lds_ld(red, tid * NPR)
                    for k_ in range_constexpr(1, NPR):
                        am = fx.max(am, lds_ld(red, tid * NPR + k_))
                    am = fx.max(am, fx.Float32(1e-12))
                    lds_st(xs, ASC + tid, am * (1.0 / 448.0))
                    lds_st(xs, AINV + tid, fx.Float32(1.0) / (am * (1.0 / 448.0)))
                gpu.barrier()
            if const_expr(TWO_PASS):
                for i0 in range_constexpr(0, NI, batch):
                    poll_batch(i0)  # the data has landed: one L2 round per batch
                    for i in range_constexpr(i0, min(i0 + batch, NI)):
                        quant_store(i)
            else:
                for i in range_constexpr(NI):
                    quant_store(i)
            gpu.barrier()

        def stage_bf16_amax(base_addr, n_elems, act_off, stride, batch=POLL_MAX):
            """stage_bf16_mb of a whole [S][n_elems] mailbox + per-token FP8 scales: the row amax is reduced from the
            polled words as they land (a wave's 64 quads lie in one row) -> ASC[t] = amax / 448, AINV[t] = 1 / ASC[t]."""
            nq = n_elems // 4
            assert nq % 64 == 0
            tot = NT * nq
            NI = (tot + THREADS - 1) // THREADS
            for i0 in range_constexpr(0, NI, batch):
                ii = list(range(i0, min(i0 + batch, NI)))
                qs = [fx.min(tid + i * THREADS, tot - 1) for i in ii]
                vals = poll(
                    [
                        (base_addr, (q // nq) * (n_elems // 2) + (q % nq) * 2, 2)
                        for q in qs
                    ],
                    batch,
                )
                for k_ in range_constexpr(len(ii)):
                    i = ii[k_]
                    q = qs[k_]
                    w = act_off + (q // nq) * stride + (q % nq) * 2
                    if const_expr((i + 1) * THREADS <= tot):
                        lds_st(xi, w, vals[k_][0])
                        lds_st(xi, w + 1, vals[k_][1])
                    else:
                        if tid + i * THREADS < tot:
                            lds_st(xi, w, vals[k_][0])
                            lds_st(xi, w + 1, vals[k_][1])
                    lo0, hi0 = bf2_f32(vals[k_][0])
                    lo1, hi1 = bf2_f32(vals[k_][1])
                    m = fx.max(
                        fx.max(fx.max(lo0, -lo0), fx.max(hi0, -hi0)),
                        fx.max(fx.max(lo1, -lo1), fx.max(hi1, -hi1)),
                    )
                    for off in (32, 16, 8, 4, 2, 1):
                        m = _xred(m, off, fx.max)
                    if lane == 0:
                        lds_st(red, i * WAVES + wave, m)
            gpu.barrier()
            if tid < NT:
                NPR = nq // 64  # wave partials per row
                am = lds_ld(red, tid * NPR)
                for k_ in range_constexpr(1, NPR):
                    am = fx.max(am, lds_ld(red, tid * NPR + k_))
                am = fx.max(am, fx.Float32(1e-12))
                lds_st(xs, ASC + tid, am * (1.0 / 448.0))
                lds_st(xs, AINV + tid, fx.Float32(1.0) / (am * (1.0 / 448.0)))
            gpu.barrier()

        def stage_act8(addr):
            """The [S][2560] activation of an FP8 GEMM: bf16 + scales (QB) or quantized FP8 rows."""
            if const_expr(QB):
                stage_bf16_amax(addr, HIDDEN, ACT, S_ACT, batch=A8_BATCH)
            else:
                stage_f8_mb(addr, HIDDEN, ACT, S_A8, batch=A8_BATCH)

        def act_gemm8(stage_phase, r_w, r_s, NTI, rg_of, emit, split=None):
            """Row tiles rg_of(i) (i < NTI) of a k128-packed FP8 [*, 2560] weight (F32 row scales r_s) times the S
            FP8 activation rows staged by stage_phase() into ACT [S][S_A8]; tiles one after the other, next weights in
            flight during the MMAs. emit(i, rg, val(r, t)) with val = dequantized output of row r, batch row t."""
            NKC = HIDDEN // 128  # 20 chunks of 128 K
            UPW = (NKC + WAVES - 1) // WAVES

            def rs_of(rg):
                """(weight, scale) descriptors of tile rg; split = (RG0, w, s, w2, s2): tiles >= RG0 come from the
                second param (base moved back RG0 tiles so tile rg keeps its offset)."""
                if const_expr(split is None):
                    return r_w, r_s
                RG0, pw, ps, pw2, ps2 = split
                lo = rg < RG0
                return (
                    _rsrc(lo.select(pw, pw2 - fx.Int64(RG0 * NKC * 2048))),
                    _rsrc(lo.select(ps, ps2 - fx.Int64(RG0 * 16 * 4))),
                )

            def units_for(i):
                rg = rg_of(i)
                rw_, _ = rs_of(rg)
                out = []
                for j in range_constexpr(UPW):
                    kc = wave + WAVES * j
                    kcc = fx.min(kc, NKC - 1)
                    out.append(
                        unit_f8(
                            rw_,
                            rg,
                            kcc,
                            NKC,
                            bcolq(ACT, S_ACT, kcc * 64)
                            if QB
                            else bcol8(ACT, S_A8, kcc * 32),
                            coef=None
                            if NKC % WAVES == 0 or j + 1 < UPW
                            else (kc < NKC).select(fx.Float32(1.0), fx.Float32(0.0)),
                            q=QB,
                        )
                    )
                return out

            PD = min(
                G8_PD, NTI
            )  # weight tiles in flight (issued before the activation is staged)
            q_ = [units_for(i) for i in range(PD)]
            stage_phase()
            stamp("g8.stage")
            for i in range_constexpr(NTI):
                if const_expr(i + PD < NTI):
                    q_.append(units_for(i + PD))
                cur = q_.pop(0)
                gemm_tile(cur)
                gpu.barrier()
                rg = rg_of(i)

                _, rs_ = rs_of(rg)

                def val(r, t, rg=rg, rs_=rs_):
                    ws = fx.Float32(
                        bo.buffer_load(rs_, rg * 16 + r, vec_width=1, dtype=T.f32)
                    )
                    return red_rows(0, WAVES, r, t) * ws * lds_ld(xs, ASC + t)

                emit(i, rg, val)
                gpu.barrier()

        def emit8_bf_rows(mbx, row_words, valid=None):
            """emit(i, rg, val): rows [16 rg, 16 rg + 16) of every batch row -> bf16 pairs of mailbox [S][row_words]."""

            def emit(i, rg, val):
                for p0 in range_constexpr(0, 8 * NT, THREADS):
                    q = tid + p0
                    ok = q < 8 * NT
                    if const_expr(valid is not None):
                        ok = ok & valid(i)
                    if ok:
                        t = q // 8
                        p = q % 8
                        put_bf(
                            mbx,
                            t * row_words + rg * 8 + p,
                            val(p * 2, t),
                            val(p * 2 + 1, t),
                        )

            return emit

        def stage_in_proj(layer):
            act_gemm8(
                lambda: stage_act8(mb(layer, "a.mixx")),
                _rsrc(ptr(layer, LY.P_IP)),
                _rsrc(ptr(layer, LY.P_IP_S)),
                LY.IP_RG_MAIN // G,
                lambda i: bid + i * G,
                emit8_bf_rows(mb(layer, "ip"), LY.IP_ROWS // 2),
            )

        # ---------------------------------------------- GDN: conv + recurrence over the T tokens
        def stage_core(layer):
            """Task (value head h, V-row split s), all R requests: conv1d over each request's L tokens (+ per-step
            conv windows), b/a projections, l2norm, gated delta rule token by token on the register-resident state
            of the request (state after every token -> intermediate_ssm[request row][step]); og[t] = o * w *
            sigmoid(z) and per-split sum of squares."""
            is_core = bid < C_TASKS
            ip = mb(layer, "ip")
            bc = fx.min(bid, C_TASKS - 1)
            h = bc // CSPLIT
            s = bc % CSPLIT
            kh = h // (GDN_HV // (GDN_KEY_DIM // GDN_DK))
            RPT, KPT, ROWS = C_RPT, C_KPT, C_ROWS
            j = tid // RPT
            c = tid % RPT
            srow = (h * GDN_DK + s * ROWS + j) * GDN_DK + c * KPT
            msl = [_uniform(lds_ld(xi, RQ_MS + r_)) for r_ in range(R)]
            sts = []
            SSB = (
                LY.SSM_SLOT_BYTES // 2 if ssm_bf16 else LY.SSM_SLOT_BYTES
            )  # bytes per mamba slot
            for r_ in range_constexpr(R):
                r_ssm = rsrc_if(
                    ptr(layer, LY.P_SSM_STATE) + fx.Int64(msl[r_]) * SSB, is_core, SSB
                )
                if const_expr(ssm_bf16):
                    sts.append(
                        [
                            fx.Vector(
                                bo.buffer_load(
                                    r_ssm, (srow + 4 * q) // 2, vec_width=2, dtype=T.i32
                                )
                            )
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                            for q in range(KPT // 4)
                        ]
                    )
                else:
                    sts.append(
                        [
                            fx.Vector(
                                bo.buffer_load(
                                    r_ssm, srow + 4 * q, vec_width=4, dtype=T.f32
                                )
                            )
                            for q in range(KPT // 4)
                        ]
                    )
            r_cw = _rsrc(ptr(layer, LY.P_CONV_W))
            NCH = 256 + ROWS  # conv channels of this task: q, k of head kh + its v rows
            tc = fx.min(tid, NCH - 1)
            ch = (tc < 128).select(
                kh * GDN_DK + tc,
                (tc < 256).select(
                    GDN_KEY_DIM + kh * GDN_DK + (tc - 128),
                    2 * GDN_KEY_DIM + h * GDN_DK + s * ROWS + (tc - 256),
                ),
            )
            cw = (
                fx.Vector(bo.buffer_load(r_cw, ch * 2, vec_width=2, dtype=T.i32))
                .bitcast(fx.BFloat16)
                .to(fx.Float32)
            )
            cst = []  # per request: committed conv window (3) of channel ch
            for r_ in range_constexpr(R):
                r_cs = _rsrc(
                    ptr(layer, LY.P_CONV_STATE) + fx.Int64(msl[r_]) * LY.CONV_SLOT_BYTES
                )
                cst.append(
                    [
                        fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    r_cs, ch * 3 + k_, vec_width=1, dtype=T.bf16
                                )
                            )
                        )
                        for k_ in range(3)
                    ]
                )
            # b/a rows of head h (row-major FP8) -> LDS C_BA [2][2560] FP8 (640 words per row) + their scales
            r_ba = rsrc_if(ptr(layer, LY.P_BA), is_core, LY.BA_ROWS * HIDDEN)
            NW_BA = HIDDEN // 4  # words per row
            bav = []
            for i in range_constexpr((2 * NW_BA + THREADS - 1) // THREADS):
                q = fx.min(tid + i * THREADS, 2 * NW_BA - 1)
                hr = (q < NW_BA).select(h, GDN_HV + h)
                w_ = (
                    q % NW_BA
                )  # K bytes 4 w_ .. of row hr in sglang's AITER-shuffled in_proj_ba (16 x 32-byte pieces)
                wa = (
                    ((hr // 16) * (HIDDEN // 32) + w_ // 8) * 128
                    + ((w_ % 8) // 4) * 64
                    + (hr % 16) * 4
                    + w_ % 4
                )
                bav.append(
                    (q, fx.Int32(bo.buffer_load(r_ba, wa, vec_width=1, dtype=T.i32)))
                )
            r_bas = _rsrc(ptr(layer, LY.P_BA_S))
            sb = fx.Float32(bo.buffer_load(r_bas, h, vec_width=1, dtype=T.f32))
            sa = fx.Float32(bo.buffer_load(r_bas, GDN_HV + h, vec_width=1, dtype=T.f32))
            # q/k channels are shared by the 12 tasks of a k head: (h % 3 == 0, s == 0) writes their windows
            own = (tid >= 256) | ((h % 3 == 0) & (s == 0))
            r_ic = rsrc_if(
                ptr(layer, LY.P_ICONV), is_core & (tid < NCH) & own, 0x7FFFFFFF
            )
            r_is = rsrc_if(
                ptr(layer, LY.P_ISSM), is_core, R * issm_row * (2 if ssm_bf16 else 4)
            )
            if is_core:
                mixx = mb(layer, "a.mixx")
                for q, wba in bav:
                    lds_st(xi, C_BA + q, wba)
                if tid < NCH:
                    xw = poll(
                        [(ip, t * (LY.IP_ROWS // 2) + ch // 2, 1) for t in range(NT)]
                    )
                    for r_ in range_constexpr(R):
                        win = list(cst[r_])
                        for j_ in range_constexpr(L):
                            xlo, xhi = bf2_f32(xw[r_ * L + j_][0])
                            win.append((ch % 2 == 0).select(xlo, xhi))
                        for j_ in range_constexpr(L):
                            t = r_ * L + j_
                            acc = (
                                cw[0] * win[j_]
                                + cw[1] * win[j_ + 1]
                                + cw[2] * win[j_ + 2]
                                + cw[3] * win[j_ + 3]
                            )
                            lds_st(
                                xs, C_Q + t * 288 + tid, bf_round(acc * sigmoid(acc))
                            )
                            for k_ in range_constexpr(3):
                                bo.buffer_store(
                                    fx.Float32(win[j_ + 1 + k_]).to(fx.BFloat16),
                                    r_ic,
                                    r_ * IC_ROW
                                    + j_ * IC_STEP
                                    + ch * IC_DIM
                                    + k_ * IC_WIN,
                                )
                if (tid >= NCH) & (tid < NCH + ROWS):
                    zr = (
                        GDN_KEY_DIM * 2
                        + GDN_VAL_DIM
                        + h * GDN_DK
                        + s * ROWS
                        + (tid - NCH)
                    )
                    zw = poll(
                        [(ip, t * (LY.IP_ROWS // 2) + zr // 2, 1) for t in range(NT)]
                    )
                    for t in range_constexpr(NT):
                        zlo, zhi = bf2_f32(zw[t][0])
                        lds_st(
                            xs,
                            C_Z + t * 32 + (tid - NCH),
                            (zr % 2 == 0).select(zlo, zhi),
                        )
                gpu.barrier()
                stamp("core.in")
                # wave t (per token): b/a dot products with x[t], q/k l2 norms and k.q -> LDS scalars
                dtb = fx.Float32(
                    bo.buffer_load(
                        _rsrc(ptr(layer, LY.P_DT_BIAS)), h, vec_width=1, dtype=T.f32
                    )
                )
                nega = fx.Float32(
                    bo.buffer_load(
                        _rsrc(ptr(layer, LY.P_NEG_A)), h, vec_width=1, dtype=T.f32
                    )
                )
                NQX = HIDDEN // 4 // 64  # 10 quads of x per lane
                for t0 in range_constexpr(0, NT, WAVES):
                    if wave + t0 < NT:
                        t = wave + t0
                        xq = poll(
                            [
                                (mixx, t * (HIDDEN // 2) + (lane + 64 * i) * 2, 2)
                                for i in range(NQX)
                            ]
                        )
                        # W8A8 b/a: per-token dynamic FP8 quantization of x[t] (as for in_proj), FP8 rows, f32 sums
                        xvs = [
                            fx.Vector.from_elements(xq[i], fx.Int32)
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                            for i in range(NQX)
                        ]
                        am = fx.Float32(0.0)
                        for i in range_constexpr(NQX):
                            for e in range_constexpr(4):
                                am = fx.max(am, fx.max(xvs[i][e], -xvs[i][e]))
                        for off in (32, 16, 8, 4, 2, 1):
                            am = _xred(am, off, fx.max)
                        am = fx.max(am, fx.Float32(1e-12))
                        xinv = fx.Float32(1.0) / (am * (1.0 / 448.0))
                        pb = fx.Float32(0.0)
                        pa = fx.Float32(0.0)
                        f2 = fx.Vector.make_type(2, fx.Float32)
                        for i in range_constexpr(NQX):
                            qd = lane + 64 * i
                            xq8 = []
                            for e in range_constexpr(2):
                                d0, d1 = _fp8_roundtrip(
                                    xvs[i][2 * e] * xinv, xvs[i][2 * e + 1] * xinv
                                )
                                xq8 += [d0, d1]
                            for rw, ab in ((C_BA + qd, 0), (C_BA + NW_BA + qd, 1)):
                                wd = lds_ld(xi, rw)
                                wf = []
                                for half in range_constexpr(2):
                                    pr = fx.Vector(
                                        rocdl.cvt_pk_f32_fp8(
                                            res=f2, src=wd, word_sel=bool(half)
                                        )
                                    )
                                    wf += [pr[0], pr[1]]
                                acc_ = (
                                    xq8[0] * wf[0]
                                    + xq8[1] * wf[1]
                                    + xq8[2] * wf[2]
                                    + xq8[3] * wf[3]
                                )
                                if const_expr(ab == 0):
                                    pb = pb + acc_
                                else:
                                    pa = pa + acc_
                        qv = [
                            lds_ld(xs, C_Q + t * 288 + lane + 64 * i) for i in range(2)
                        ]
                        kv = [
                            lds_ld(xs, C_Q + t * 288 + 128 + lane + 64 * i)
                            for i in range(2)
                        ]
                        sqq = qv[0] * qv[0] + qv[1] * qv[1]
                        sqk = kv[0] * kv[0] + kv[1] * kv[1]
                        dqk = qv[0] * kv[0] + qv[1] * kv[1]
                        sqq, sqk, dqk, pb, pa = (
                            wave_sum(v_) for v_ in (sqq, sqk, dqk, pb, pa)
                        )
                        rq = _rsq(sqq + 1e-6) * (GDN_DK**-0.5)
                        rk = _rsq(sqk + 1e-6)
                        for i in range_constexpr(2):
                            lds_st(xs, C_Q + t * 288 + lane + 64 * i, qv[i] * rq)
                            lds_st(xs, C_Q + t * 288 + 128 + lane + 64 * i, kv[i] * rk)
                        pb = (
                            pb * (am * (1.0 / 448.0)) * sb
                        )  # aiter's per-token quant: s = amax / 448, q = x * (1 / s)
                        pa = pa * (am * (1.0 / 448.0)) * sa
                        if lane == 0:
                            xa = bf_round(pa) + dtb
                            sp = (xa > 20.0).select(
                                xa,
                                _log2(fx.Float32(1.0) + _exp(xa)) * 0.6931471805599453,
                            )
                            lds_st(xs, C_SC + t * 4, sigmoid(bf_round(pb)))
                            lds_st(xs, C_SC + t * 4 + 1, _exp(nega * sp))
                            lds_st(xs, C_SC + t * 4 + 2, dqk * rq * rk)
                gpu.barrier()
                stamp("core.norm")
                NQ4 = KPT // 4
                nw = fx.Float32(
                    bo.buffer_load(
                        _rsrc(ptr(layer, LY.P_NORM_W)),
                        s * ROWS + j,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )

                def rowsum2(a_, b_):
                    for off in C_RED:
                        a_ = _xred(a_, off, lambda x_, y_: x_ + y_)
                        b_ = _xred(b_, off, lambda x_, y_: x_ + y_)
                    return a_, b_

                # gated delta rule, token by token: o_t = decay (S q) + vnew (k . q) needs only S of the previous
                # step, so the two row reductions of a token run side by side; the R requests' chains are
                # independent and run interleaved (step j of every request, then step j + 1)
                svs = [
                    [sts[r_][q][e] for q in range(NQ4) for e in range(4)]
                    for r_ in range(R)
                ]
                own_o = c == 0
                o_junk = (
                    C_BA + tid
                )  # b/a rows are dead: lanes c != 0 store their (unused) o here, no branch
                for j_ in range_constexpr(L):
                    for r_ in range_constexpr(R):
                        t = r_ * L + j_
                        sv = svs[r_]
                        qq = [
                            fx.Vector(
                                fx.ptr_load(
                                    xs + (C_Q + t * 288 + c * KPT + 4 * q),
                                    result_type=v4f,
                                )
                            )
                            for q in range(NQ4)
                        ]
                        kk = [
                            fx.Vector(
                                fx.ptr_load(
                                    xs + (C_Q + t * 288 + 128 + c * KPT + 4 * q),
                                    result_type=v4f,
                                )
                            )
                            for q in range(NQ4)
                        ]
                        vj = lds_ld(xs, C_Q + t * 288 + 256 + j)
                        beta = lds_ld(xs, C_SC + t * 4)
                        decay = lds_ld(xs, C_SC + t * 4 + 1)
                        kq = lds_ld(xs, C_SC + t * 4 + 2)
                        kf = [kk[q][e] for q in range(NQ4) for e in range(4)]
                        qf = [qq[q][e] for q in range(NQ4) for e in range(4)]
                        sk = sv[0] * kf[0]
                        sq_ = sv[0] * qf[0]
                        for e in range_constexpr(1, KPT):
                            sk = sk + sv[e] * kf[e]
                            sq_ = sq_ + sv[e] * qf[e]
                        sk, sq_ = rowsum2(sk, sq_)
                        vnew = (vj - decay * sk) * beta
                        o = decay * sq_ + vnew * kq
                        sv = [sv[e] * decay + kf[e] * vnew for e in range(KPT)]
                        svs[r_] = sv
                        for q in range_constexpr(NQ4):
                            st4 = fx.Vector.from_elements(
                                sv[4 * q : 4 * q + 4], fx.Float32
                            )
                            off = r_ * issm_row + j_ * LY.SSM_STEP_WORDS + srow + 4 * q
                            if const_expr(ssm_bf16):
                                bo.buffer_store(
                                    st4.to(fx.BFloat16).bitcast(fx.Int32),
                                    r_is,
                                    off // 2,
                                )
                            else:
                                bo.buffer_store(st4, r_is, off)
                        lds_st(xs, own_o.select(C_O + t * 32 + j, o_junk), o)
                        fx.ptr_store(
                            (o * nw * sigmoid(lds_ld(xs, C_Z + t * 32 + j))).to(
                                fx.BFloat16
                            ),
                            xb
                            + own_o.select(
                                (C_OG + t * 16) * 2 + j, (C_BA + 512) * 2 + tid
                            ),
                        )
                gpu.barrier()
                stamp("core.rec")
                # outputs: per-(row, task) sum of squares of o; then, with the head's 4 split partials, the gated
                # RMS-normalized og = bf16(o rstd w sigmoid(z)) (one rounding, as sglang's fused gated norm) and its
                # per-(row, task) max |og| (the out_proj FP8 activation scale)
                for t0 in range_constexpr(0, NT, WAVES):
                    if wave + t0 < NT:
                        t = wave + t0
                        ov = (lane < ROWS).select(
                            lds_ld(xs, C_O + t * 32 + fx.min(lane, ROWS - 1)),
                            fx.Float32(0.0),
                        )
                        ss = wave_sum(ov * ov)
                        if lane == 0:
                            put(mb(layer, "ssq"), t * C_TASKS + bc, ss)
                nwl = fx.Float32(
                    bo.buffer_load(
                        _rsrc(ptr(layer, LY.P_NORM_W)),
                        s * ROWS + fx.min(lane, ROWS - 1),
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                for t0 in range_constexpr(0, NT, WAVES):
                    if wave + t0 < NT:
                        t = wave + t0
                        sq = poll(
                            [
                                (mb(layer, "ssq"), t * C_TASKS + h * CSPLIT + p, 1)
                                for p in range(CSPLIT)
                            ]
                        )
                        tot = sq[0][0].bitcast(fx.Float32)
                        for p in range_constexpr(1, CSPLIT):
                            tot = tot + sq[p][0].bitcast(fx.Float32)
                        rs = _rsq(tot * (1.0 / GDN_DK) + EPS)
                        jl = fx.min(lane, ROWS - 1)
                        on = lds_ld(xs, C_O + t * 32 + jl) * rs * nwl
                        og = bf_round(on * sigmoid(lds_ld(xs, C_Z + t * 32 + jl)))
                        og = (lane < ROWS).select(og, fx.Float32(0.0))
                        gm = fx.max(og, -og)
                        for off in (32, 16, 8, 4, 2, 1):
                            gm = _xred(gm, off, fx.max)
                        nb_ = _xshfl(og, 1)
                        if lane == 0:
                            put(mb(layer, "og_amax"), t * C_TASKS + bc, gm)
                        if (lane < ROWS) & (lane % 2 == 0):
                            put_bf(
                                mb(layer, "og"),
                                t * (GDN_VAL_DIM // 2)
                                + (h * GDN_DK + s * ROWS + lane) // 2,
                                og,
                                nb_,
                            )
                gpu.barrier()

        # ---------------------------------------------- norm + out_proj + combine
        def stage_out_proj(layer, src, kind):
            """W8A8 out_proj (GDN) / o_proj (QSA) + combine. The activation (GDN: the gated-RMS-normalized og, QSA: the
            merged attention output) is quantized per row with the scale from the producers' max |og| partials;
            output = acc x weight row scale x activation scale."""
            is_op = bid < LY.OUT_RG
            rg = fx.min(bid, LY.OUT_RG - 1)
            r_w = rsrc_if(ptr(layer, LY.P_OUT), is_op, HIDDEN * GDN_VAL_DIM)
            r_ws = _rsrc(ptr(layer, LY.P_OUT_S))
            NKC = GDN_VAL_DIM // 128  # 48 chunks of 128 K
            NKH = NKC // OP_KS  # chunks per staged K part
            CPW = NKH // WAVES
            KP = GDN_VAL_DIM // OP_KS

            def part_units(hh):
                part = []
                rg_p = (
                    rg if hh == 0 else _ordered_i32(rg)
                )  # part hh + 1 loads stay behind part hh's MMAs
                for c in range_constexpr(CPW):
                    kc = hh * NKH + wave * CPW + c
                    part.append(
                        unit_f8(
                            r_w,
                            rg_p,
                            kc,
                            NKC,
                            bcolq(ACT, S_OG, (kc - hh * NKH) * 64)
                            if QB
                            else bcol8(ACT, S_OG, (kc - hh * NKH) * 32),
                            q=QB,
                        )
                    )
                return part

            xform = None
            units0 = part_units(0)
            if is_op:
                if const_expr(
                    QB
                ):  # og part 0 lands while the producers' amax partials are reduced below
                    stage_bf16_mb(
                        mb(layer, "og"),
                        KP,
                        ACT,
                        S_OG,
                        batch=OG_BATCH,
                        row_elems=GDN_VAL_DIM,
                    )
                og_amax = mb(layer, "og_amax")
                if const_expr(kind == "gdn"):
                    # row amax = max over the core tasks' max |og| (a wave's 64 tasks lie in one row)
                    NA = C_TASKS * NT
                    assert C_TASKS % 64 == 0
                    NIA = (NA + THREADS - 1) // THREADS
                    # S <= 16: one poll round; S = 32: one partial at a time (next to the out_proj weights)
                    vam = (
                        poll(
                            [
                                (og_amax, fx.min(tid + i * THREADS, NA - 1), 1)
                                for i in range(NIA)
                            ]
                        )
                        if NT <= 16
                        else None
                    )
                    for i in range_constexpr(NIA):
                        if const_expr(NT <= 16):
                            v = vam[i][0].bitcast(fx.Float32)
                        else:
                            v = poll([(og_amax, fx.min(tid + i * THREADS, NA - 1), 1)])[
                                0
                            ][0].bitcast(fx.Float32)
                        for off in (32, 16, 8, 4, 2, 1):
                            v = _xred(v, off, fx.max)
                        if lane == 0:
                            lds_st(red, i * WAVES + wave, v)
                    gpu.barrier()
                    if tid < NT:
                        NPR = C_TASKS // 64
                        am = lds_ld(red, tid * NPR)
                        for k_ in range_constexpr(1, NPR):
                            am = fx.max(am, lds_ld(red, tid * NPR + k_))
                        am = fx.max(am, fx.Float32(1e-12))
                        lds_st(xs, ASC + tid, am * (1.0 / 448.0))
                        lds_st(xs, AINV + tid, fx.Float32(1.0) / (am * (1.0 / 448.0)))
                else:
                    if tid < NT:  # merge tasks published max |og| per (row, head)
                        got = poll([(og_amax, tid * 24 + h_, 1) for h_ in range(24)])
                        am = got[0][0].bitcast(fx.Float32)
                        for h_ in range_constexpr(1, 24):
                            am = fx.max(am, got[h_][0].bitcast(fx.Float32))
                        am = fx.max(am, fx.Float32(1e-12))
                        lds_st(xs, ASC + tid, am * (1.0 / 448.0))
                        lds_st(xs, AINV + tid, fx.Float32(1.0) / (am * (1.0 / 448.0)))
                gpu.barrier()
            stamp("op.amax")
            if is_op:
                if const_expr(not QB):
                    stage_f8_mb(
                        mb(layer, "og"),
                        KP,
                        ACT,
                        S_OG,
                        row_elems=GDN_VAL_DIM,
                        amax_known=True,
                        xform=xform,
                        batch=OG_BATCH,
                    )
            stamp("op.prep")
            if is_op:
                # K part hh + 1: weights issued before the MMA of part hh (the og polls of part hh are done)
                nxt = part_units(1) if OP_KS > 1 else None
                acc = mma_units(zero_acc(), units0)
                for hh in range_constexpr(1, OP_KS):
                    cur = nxt
                    nxt = part_units(hh + 1) if hh + 1 < OP_KS else None
                    _pin(acc)  # part hh - 1 MMAs complete before part hh is staged
                    gpu.barrier()
                    if const_expr(QB):
                        stage_bf16_mb(
                            mb(layer, "og"),
                            KP,
                            ACT,
                            S_OG,
                            batch=OG_BATCH,
                            row_elems=GDN_VAL_DIM,
                            elem0=hh * KP,
                        )
                    else:
                        stage_f8_mb(
                            mb(layer, "og"),
                            KP,
                            ACT,
                            S_OG,
                            row_elems=GDN_VAL_DIM,
                            elem0=hh * KP,
                            amax_known=True,
                            xform=xform,
                            batch=OG_BATCH,
                        )
                    acc = mma_units(acc, cur)
                store_acc(acc)
                gpu.barrier()
                for p0 in range_constexpr(0, 16 * NT, THREADS):
                    q = tid + p0
                    if q < 16 * NT:
                        r = q % 16
                        t = q // 16
                        ws = fx.Float32(
                            bo.buffer_load(r_ws, rg * 16 + r, vec_width=1, dtype=T.f32)
                        )
                        ov_ = red_rows(0, WAVES, r, t) * ws * lds_ld(xs, ASC + t)
                        lds_st(xs, OUTS + q, ov_)
                gpu.barrier()
                combine_rows(
                    layer,
                    "a",
                    src,
                    rg,
                    lambda p, t: (
                        lds_ld(xs, OUTS + t * 16 + 2 * p),
                        lds_ld(xs, OUTS + t * 16 + 2 * p + 1),
                    ),
                    mb(layer, "h1"),
                    mb(layer, "h1ssq"),
                )
                gpu.barrier()

        # ---------------------------------------------- MoE
        def stage_router(layer):
            if bid < LY.ROUTER_RG:
                logit = mb(layer, "logit")

                def emit(i, rg):
                    for p0 in range_constexpr(0, 16 * NT, THREADS):
                        q = tid + p0
                        if q < 16 * NT:
                            put(
                                logit,
                                (q // 16) * LY.ROUTER_ROWS + rg * 16 + q % 16,
                                red_rows(0, WAVES, q % 16, q // 16),
                            )

                act_gemm(
                    stage_mb_phase(mb(layer, "m.mixx")),
                    _rsrc(ptr(layer, LY.P_ROUTER)),
                    1,
                    lambda i: bid,
                    emit,
                )

        def topk_token(logit, t):
            """One wave: top-10 of token t's router logits (softmax renormalized) + shared gate -> LDS."""
            NI = N_EXPERTS // 64
            got = poll(
                [(logit, t * LY.ROUTER_ROWS + lane + 64 * i, 1) for i in range(NI)]
                + [(logit, t * LY.ROUTER_ROWS + N_EXPERTS, 1)]
            )
            ks = []
            for i in range_constexpr(NI):
                kb = got[i][0]
                ks.append((kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb))
            ids = [lane + 64 * i for i in range(NI)]
            for a_, b_ in (
                (0, 1),
                (2, 3),
                (4, 5),
                (6, 7),
                (0, 2),
                (1, 3),
                (4, 6),
                (5, 7),
                (1, 2),
                (5, 6),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
                (2, 4),
                (3, 5),
                (1, 2),
                (3, 4),
                (5, 6),
            ):
                first = (fx.Uint32(ks[a_]) > fx.Uint32(ks[b_])) | (
                    (ks[a_] == ks[b_]) & (ids[a_] < ids[b_])
                )
                ka, kb2, ia, ib = ks[a_], ks[b_], ids[a_], ids[b_]
                ks[a_], ks[b_] = first.select(ka, kb2), first.select(kb2, ka)
                ids[a_], ids[b_] = first.select(ia, ib), first.select(ib, ia)
            ks = ks + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]
            sel_k, sel_i = [], []
            for r in range_constexpr(TOPK):
                m = _wave_umax(ks[0])
                winner = _wave_umax(
                    (ks[0] == m).select((N_EXPERTS - 1) - ids[0], fx.Int32(0))
                )
                expert = (N_EXPERTS - 1) - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(NI)] + [ks[NI]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(NI)] + [ids[NI]]
                sel_k.append(m)
                sel_i.append(expert)
            lg = [
                (sel_k[r] < 0)
                .select(sel_k[r] ^ fx.Int32(-(2**31)), ~sel_k[r])
                .bitcast(fx.Float32)
                for r in range(TOPK)
            ]
            ws = [_exp(lg[r] - lg[0]) for r in range(TOPK)]
            tot = ws[0]
            for r in range_constexpr(1, TOPK):
                tot = tot + ws[r]
            inv = _rcp(tot)
            if lane == 0:
                for r in range_constexpr(TOPK):
                    lds_st(xi, TOPI + t * 16 + r, fx.Int32(sel_i[r]))
                    lds_st(xs, TOPW + t * 16 + r, ws[r] * inv)
                    lds_st(xi, FLAG + sel_i[r], fx.Int32(1))
                lds_st(xs, SHG + t, sigmoid(got[NI][0].bitcast(fx.Float32)))

        def bitonic_desc(ks, ids):
            """Sort (key, id) lists of a power-of-2 length per lane: keys descending (unsigned), ties lower id first."""
            for a_, b_ in _bitonic_pairs(len(ks)):
                first = (fx.Uint32(ks[a_]) > fx.Uint32(ks[b_])) | (
                    (ks[a_] == ks[b_]) & (ids[a_] < ids[b_])
                )
                ka, kb2, ia, ib = ks[a_], ks[b_], ids[a_], ids[b_]
                ks[a_], ks[b_] = first.select(ka, kb2), first.select(kb2, ka)
                ids[a_], ids[b_] = first.select(ia, ib), first.select(ib, ia)
            return ks, ids

        def topk_half(logit, t, ok):
            """One half wave (32 lanes): top-10 of row t's router logits (softmax renormalized) + shared gate -> LDS."""
            NI = N_EXPERTS // 32
            lh = lane % 32
            got = poll(
                [(logit, t * LY.ROUTER_ROWS + lh + 32 * i, 1, ok) for i in range(NI)]
                + [(logit, t * LY.ROUTER_ROWS + N_EXPERTS, 1, ok)]
            )
            ks = []
            for i in range_constexpr(NI):
                kb = got[i][0]
                ks.append((kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb))
            ids = [lh + 32 * i for i in range(NI)]
            ks, ids = bitonic_desc(ks, ids)
            ks = ks + [fx.Int32(0)]
            ids = ids + [fx.Int32(N_EXPERTS)]

            def umax(a, b):
                return (fx.Uint32(a) > fx.Uint32(b)).select(a, b)

            def hmax(v):
                for off in (16, 8, 4, 2, 1):
                    v = _xred(v, off, umax)
                return v

            sel_k, sel_i = [], []
            for r in range_constexpr(TOPK):
                m = hmax(ks[0])
                winner = hmax(
                    (ks[0] == m).select((N_EXPERTS - 1) - ids[0], fx.Int32(0))
                )
                expert = (N_EXPERTS - 1) - winner
                hit = (ks[0] == m) & (ids[0] == expert)
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(NI)] + [ks[NI]]
                ids = [hit.select(ids[i + 1], ids[i]) for i in range(NI)] + [ids[NI]]
                sel_k.append(m)
                sel_i.append(expert)
            lg = [
                (sel_k[r] < 0)
                .select(sel_k[r] ^ fx.Int32(-(2**31)), ~sel_k[r])
                .bitcast(fx.Float32)
                for r in range(TOPK)
            ]
            ws = [_exp(lg[r] - lg[0]) for r in range(TOPK)]
            tot = ws[0]
            for r in range_constexpr(1, TOPK):
                tot = tot + ws[r]
            inv = _rcp(tot)
            if ok & (lh == 0):
                for r in range_constexpr(TOPK):
                    lds_st(xi, TOPI + t * 16 + r, fx.Int32(sel_i[r]))
                    lds_st(xs, TOPW + t * 16 + r, ws[r] * inv)
                    lds_st(xi, FLAG + sel_i[r], fx.Int32(1))
                lds_st(xs, SHG + t, sigmoid(got[NI][0].bitcast(fx.Float32)))

        def route(layer):
            """Every CTA: per-row top-10, union of the selected experts (sorted by id) + shared expert as slots."""
            logit = mb(layer, "logit")
            lds_st(xi, FLAG + tid, fx.Int32(0))
            gpu.barrier()
            if const_expr(NT <= WAVES):
                if wave < NT:
                    topk_token(logit, wave)
            else:  # two rows per wave (half waves), S / 16 rounds
                for t0 in range_constexpr(0, NT, 2 * WAVES):
                    t = t0 + 2 * wave + lane // 32
                    topk_half(logit, fx.min(t, NT - 1), t < NT)
            gpu.barrier()
            f = lds_ld(xi, FLAG + tid)
            incl = fx.coop.warp_inclusive_scan(f, fx.ReductionOp.ADD, width=64)
            wtot = fx.Int32(rocdl.readlane(T.i32, fx.Int32(incl), fx.Int32(63)))
            if lane == 63:
                lds_st(xi, WTOT + wave, wtot)
            gpu.barrier()
            before = fx.Int32(0)
            total = fx.Int32(0)
            for w in range_constexpr(WAVES):
                wc = lds_ld(xi, WTOT + w)
                before = before + (wave > w).select(wc, fx.Int32(0))
                total = total + wc
            slot = before + incl - f
            gpu.barrier()  # FLAG (aliasing SLOT_OF) fully read
            if f != 0:
                lds_st(xi, SLOT_E + slot, tid)
                lds_st(xi, SLOT_OF + tid, slot)
            if tid == 0:
                lds_st(xi, SLOT_E + total, fx.Int32(LY.SHARED_EXPERT))
                lds_st(xi, NSV, total + 1)
            gpu.barrier()

        def n_slots():
            return _uniform(lds_ld(xi, NSV))

        def pair_rank(t, sl, e, ns):
            """Rank of expert e (slot sl) in row t's top-10 (10: the shared slot; -1: not routed)."""
            rk = (sl == ns - 1).select(fx.Int32(TOPK), fx.Int32(-1))
            for r in range_constexpr(TOPK):
                rk = (lds_ld(xi, TOPI + t * 16 + r) == e).select(fx.Int32(r), rk)
            return rk

        def pair_weight(t, rk):
            w = lds_ld(xs, SHG + t)
            for r in range_constexpr(TOPK):
                w = (rk == r).select(lds_ld(xs, TOPW + t * 16 + r), w)
            return w

        def put4(base_addr, i, vals):
            """Tagged pairs i .. i + 3 := vals (two 16-byte stores)."""
            for h_ in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [
                            vals[2 * h_].bitcast(fx.Int32),
                            tag,
                            vals[2 * h_ + 1].bitcast(fx.Int32),
                            tag,
                        ],
                        fx.Int32,
                    ),
                    _rsrc(base_addr),
                    (i + 2 * h_) * 2,
                    cache_modifier=CM_DEV,
                )

        def wave_mma(unit_fn, n_units, piece, acc_of=None, nacc=1):
            """One wave's accumulator(s) over units 0 .. n_units - 1 (unit u -> accumulator acc_of(u)), issued in
            pieces (piece k + 1's weights in flight during piece k's MMAs)."""
            accs = [zero_acc() for _ in range(nacc)]
            starts = list(range(0, n_units, piece))
            cur = [unit_fn(u) for u in range(0, min(piece, n_units))]
            for i_, s0 in enumerate(starts):
                nxt = None
                if i_ + 1 < len(starts):
                    nxt = [
                        unit_fn(u)
                        for u in range(
                            starts[i_ + 1], min(starts[i_ + 1] + piece, n_units)
                        )
                    ]
                for a_ in range_constexpr(nacc):
                    mine = [
                        cur[k_]
                        for k_ in range(len(cur))
                        if (acc_of(s0 + k_) if acc_of else 0) == a_
                    ]
                    if const_expr(len(mine) > 0):
                        accs[a_] = mma_units(accs[a_], mine)
                pins = []
                for a_ in range_constexpr(nacc):
                    pins += list(accs[a_])
                _pin(
                    pins, mem=True
                )  # piece k's operands are consumed before piece k + 1's LDS reads
                for a_ in range_constexpr(nacc):
                    accs[a_] = pins[a_ * len(accs[a_]) : (a_ + 1) * len(accs[a_])]
                cur = nxt
            return accs[0] if nacc == 1 else accs

        def stage_gate_up(layer):
            """Wave-level tasks (slot, 16-row gate/up tile) over all CTAs, all S rows as MFMA columns, no barriers in
            the task loop; only the routed (row, expert) pairs p = t * 11 + rank are published. KPG 1: mid[p][640] =
            silu(g) * u (bf16, the up rows of a tile are lane ^ 32); KPG 2: CTA parity = K half, f32 partials
            mid[half][p][tile * 16 + r] (the down staging sums the halves)."""
            if const_expr(KPG == 1):
                kh = 0
                cta, NCTA = bid, G
                stage_bf16_mb(
                    mb(layer, "m.mixx"), HIDDEN, GU_ACT, S_GU
                )  # x2 lands before the router logits
            else:
                kh = bid % 2
                cta, NCTA = bid // 2, G // 2
                stage_bf16_mb(
                    mb(layer, "m.mixx"),
                    HIDDEN // 2,
                    GU_ACT,
                    S_GU,
                    row_elems=HIDDEN,
                    elem0=kh * (HIDDEN // 2),
                )
            stamp("gu.x2")
            route(layer)
            stamp("gu.routed")
            gu = ptr(layer, LY.P_GU)
            gus = ptr(layer, LY.P_GU_S)
            mid = mb(layer, "mid")
            gupart = mb(layer, "gupart") if KPG == 2 else None
            ns = n_slots()
            TPS = LY.GU_PAIRS  # tasks per slot
            ntask = ns * TPS
            gw = cta * WAVES + wave
            NWV = NCTA * WAVES
            nloop = _uniform((ntask - gw + NWV - 1) // NWV)
            for it, _c in range(fx.Int32(0), nloop, 1, init=[fx.Int32(0)]):
                it = fx.Int32(it)
                task = gw + it * NWV
                sl = task // TPS
                tile = task % TPS
                e = _uniform(lds_ld(xi, SLOT_E + sl))
                r_g = _rsrc(gu + fx.Int64(e) * LY.GU_BYTES)
                r_s = _rsrc(gus + fx.Int64(e) * LY.GU_SBYTES)
                KTP = (HIDDEN // 128) // KPG  # 128-K chunks of this CTA's K range
                # units [0, KTP): gate tile `tile`; [KTP, 2 KTP): up tile `tile + 48` (w13 rows 768 + 16 tile ..)
                accg, accu = wave_mma(
                    lambda u: unit_mxfp4a(
                        r_g,
                        r_s,
                        tile + (0 if u < KTP else LY.UP_TILE0),
                        kh * KTP + u % KTP,
                        HIDDEN,
                        bcolq(GU_ACT, S_GU, (u % KTP) * 64),
                    ),
                    2 * KTP,
                    KTP // 2,
                    acc_of=lambda u: 0 if u < KTP else 1,
                    nacc=2,
                )
                for ct in range_constexpr(NCT):
                    t = lane % 16 + 16 * ct
                    tc = fx.min(t, NT - 1)
                    rk = pair_rank(tc, sl, e, ns)
                    ok = (t < NT) & (rk >= 0)
                    rows = 4 * (
                        lane // 16
                    )  # this lane's 4 rows of the gate tile (and of the up tile)
                    if const_expr(KPG == 1):
                        hv = []
                        for i in range_constexpr(4):
                            g = accg[ct * 4 + i]
                            hv.append(g * sigmoid(g) * accu[ct * 4 + i])
                        if ok:
                            base = (
                                (tc * NPAIR + rk) * MOE_INTER + tile * 16 + rows
                            ) // 2
                            put_bf(mid, base, hv[0], hv[1])
                            put_bf(mid, base + 1, hv[2], hv[3])
                    else:
                        # K half 0 publishes its f32 gate (pairs [0, 640)) / up (pairs [640, 1280)) partials; the
                        # sibling CTA's wave (half 1, same task at the same loop step) adds them
                        pidx = (
                            (tc * NPAIR + fx.max(rk, fx.Int32(0))) * (2 * MOE_INTER)
                            + tile * 16
                            + rows
                        )
                        if kh == 0:
                            if ok:
                                put4(gupart, pidx, accg[ct * 4 : ct * 4 + 4])
                                put4(
                                    gupart, pidx + MOE_INTER, accu[ct * 4 : ct * 4 + 4]
                                )
                        else:
                            got_h0 = poll(
                                [(gupart, pidx + 2 * hf, 2, ok) for hf in range(2)]
                                + [
                                    (gupart, pidx + MOE_INTER + 2 * hf, 2, ok)
                                    for hf in range(2)
                                ]
                            )
                            hv_h = []
                            for i in range_constexpr(4):
                                g_h = accg[ct * 4 + i] + got_h0[i // 2][i % 2].bitcast(
                                    fx.Float32
                                )
                                u_h = accu[ct * 4 + i] + got_h0[2 + i // 2][
                                    i % 2
                                ].bitcast(fx.Float32)
                                hv_h.append(g_h * sigmoid(g_h) * u_h)
                            if ok:
                                base_h = (
                                    (tc * NPAIR + rk) * MOE_INTER + tile * 16 + rows
                                ) // 2
                                put_bf(mid, base_h, hv_h[0], hv_h[1])
                                put_bf(mid, base_h + 1, hv_h[2], hv_h[3])
                _ = yield [fx.Int32(0)]
            gpu.barrier()

        def stage_down(layer, h1_src, out_global):
            """Expert-group split down projection: CTA block of group g stages mid of slots [SG g, SG g + SG)
            (routed pairs; zero for unrouted rows) once and loops over its output row groups; partials
            dpart[g][t][2560]; then CTA rg < 160 sums the groups, applies the combine and publishes h2 (+ sum-of-squares
            partials)."""
            KT = MOE_INTER // 128  # 128-K MXFP4 chunks per slot
            NU = SG * KT
            dn = ptr(layer, LY.P_DN)
            dns = ptr(layer, LY.P_DN_S)
            mid = mb(layer, "mid")
            dpart = mb(layer, "dpart")
            ns = n_slots()
            ng = (ns + SG - 1) // SG
            cpg = G // ng  # CTAs per group
            g = bid // cpg
            jj = bid % cpg
            live_g = g < ng
            gc = fx.min(g, ng - 1)
            # pair rank / route weight of (group slot sj, row t); zero-filled mid tile
            for q0 in range_constexpr(0, SG * NT, THREADS):
                q = tid + q0
                if q < SG * NT:
                    sj = q // NT
                    t = q % NT
                    sl = gc * SG + sj
                    slc = fx.min(sl, ns - 1)
                    rk = pair_rank(t, slc, lds_ld(xi, SLOT_E + slc), ns)
                    rk = (sl < ns).select(rk, fx.Int32(-1))
                    lds_st(xi, RNK + q, rk)
                    lds_st(
                        xs,
                        SWG + q,
                        (rk >= 0).select(pair_weight(t, rk), fx.Float32(0.0)),
                    )
            NZ = NT * S_MID
            for z0 in range_constexpr(0, NZ, THREADS):
                if tid + z0 < NZ:
                    lds_st(xi, MID + tid + z0, fx.Int32(0))
            gpu.barrier()
            # routed (slot, row) pairs of the group, compacted -> PLIST (wave 0)
            PLIST = MOE_OUTS  # [SG * S] (dead until the combine)
            if wave == 0:
                cnt = fx.Int32(0)
                for q0 in range_constexpr(0, SG * NT, 64):
                    q = lane + q0
                    f = (
                        (q < SG * NT) & (lds_ld(xi, RNK + fx.min(q, SG * NT - 1)) >= 0)
                    ).select(fx.Int32(1), fx.Int32(0))
                    incl = fx.coop.warp_inclusive_scan(f, fx.ReductionOp.ADD, width=64)
                    if f != 0:
                        lds_st(xi, PLIST + cnt + incl - 1, q)
                    cnt = cnt + fx.Int32(
                        rocdl.readlane(T.i32, fx.Int32(incl), fx.Int32(63))
                    )
                if lane == 0:
                    lds_st(xi, NSV + 1, cnt)
            gpu.barrier()
            npr = _uniform(lds_ld(xi, NSV + 1))
            if live_g:
                # stage the routed rows' mid of the group's slots: LDS [S][SG * 640] (zero elsewhere)
                NQP = MOE_INTER // 4  # 16-byte output quads per pair
                IB = 4  # items per thread per poll round
                nit = npr * NQP
                nround = (nit + THREADS * IB - 1) // (THREADS * IB)
                for it, _c in range(fx.Int32(0), nround, 1, init=[fx.Int32(0)]):
                    it = fx.Int32(it)
                    items = []
                    for k_ in range_constexpr(IB):
                        i = (it * IB + k_) * THREADS + tid
                        ok = i < nit
                        ic = fx.min(i, nit - 1)
                        pq = lds_ld(xi, PLIST + ic // NQP)
                        kq = ic % NQP
                        sj = pq // NT
                        t = pq % NT
                        rk = fx.max(lds_ld(xi, RNK + pq), fx.Int32(0))
                        items.append((ok, t, sj, kq, rk))
                    hw = poll(
                        [
                            (mid, ((t * NPAIR + rk) * MOE_INTER + 4 * kq) // 2, 2, ok)
                            for ok, t, sj, kq, rk in items
                        ]
                    )
                    for n_ in range_constexpr(len(items)):
                        ok, t, sj, kq, rk = items[n_]
                        if ok:
                            o = MID + t * S_MID + sj * (MOE_INTER // 2) + kq * 2
                            lds_st(xi, o, hw[n_][0])
                            lds_st(xi, o + 1, hw[n_][1])
                    _ = yield [fx.Int32(0)]
            gpu.barrier()
            stamp("dn.mid")
            # wave-level row-group tasks: wave w of CTA jj takes rg = jj * 8 + w + k * cpg * 8 (all of the group's K)
            jw = jj * WAVES + wave
            nrg = live_g.select(
                (LY.DN_RG - jw + cpg * WAVES - 1) // (cpg * WAVES), fx.Int32(0)
            )
            nrg = _uniform(nrg)
            rds = []
            for sj in range_constexpr(SG):
                sl = gc * SG + sj
                slc = fx.min(sl, ns - 1)
                e = _uniform(lds_ld(xi, SLOT_E + slc))
                okl = sl < ns
                rds.append(
                    (
                        rsrc_if(dn + fx.Int64(e) * LY.DN_BYTES, okl, LY.DN_BYTES),
                        rsrc_if(dns + fx.Int64(e) * LY.DN_SBYTES, okl, LY.DN_SBYTES),
                    )
                )

            def dn_unit(u, rg, sdws=None):
                sj, kt = u // KT, u % KT
                coef = lambda ct, sj=sj: lds_ld(xs, SWG + sj * NT + tcol(ct))
                return unit_mxfp4a(
                    rds[sj][0],
                    rds[sj][1],
                    rg,
                    kt,
                    LY.MOE_INTER_PAD,  # w2 rows: padded K
                    bcolq(MID, S_MID, sj * (MOE_INTER // 2) + kt * 64),
                    coef=coef,
                    sdw=None if sdws is None else sdws[sj][kt // 2],
                )

            for ir, _c in range(fx.Int32(0), nrg, 1, init=[fx.Int32(0)]):
                ir = fx.Int32(ir)
                rg = jw + ir * (cpg * WAVES)
                sdws = None
                # the task's scale dwords (3 blocks of 8 groups per slot) ahead of its weight stream: an AITER scale
                # block is 256 B per 32 rows x 8 groups, of which one unit uses 64 B
                sdws = [
                    [
                        mxfp4a_sdw(rds[sj][1], rg, d, LY.MOE_INTER_PAD)
                        for d in range((KT + 1) // 2)
                    ]
                    for sj in range(SG)
                ]
                acc = wave_mma(
                    lambda u: dn_unit(u, rg, sdws), NU, KT if NU > 2 * KT else NU
                )
                for ct in range_constexpr(NCT):
                    t = lane % 16 + 16 * ct
                    if t < NT:
                        put4(
                            dpart,
                            (gc * NT + t) * HIDDEN + rg * 16 + 4 * (lane // 16),
                            acc[ct * 4 : ct * 4 + 4],
                        )
                _ = yield [fx.Int32(0)]
            gpu.barrier()
            stamp("dn.gemm")
            if bid < LY.DN_RG:
                rg = bid
                # sum the group partials of rows [16 rg, 16 rg + 16) for all rows -> LDS MOE_OUTS[t][16]
                # (poll rounds of NGC groups, only the ng live groups)
                NGC = 16
                nround = (ng + NGC - 1) // NGC
                for p0 in range_constexpr(0, 16 * NT, THREADS):
                    q = tid + p0
                    qok = q < 16 * NT
                    qc = fx.min(q, 16 * NT - 1)
                    t = qc // 16
                    r = qc % 16
                    res = None
                    for it, st in range(fx.Int32(0), nround, 1, init=[fx.Float32(0.0)]):
                        it = fx.Int32(it)
                        g0 = it * NGC
                        vs = poll(
                            [
                                (
                                    dpart,
                                    ((g0 + i) * NT + t) * HIDDEN + rg * 16 + r,
                                    1,
                                    qok & (g0 + i < ng),
                                )
                                for i in range(NGC)
                            ]
                        )
                        tot = st[0]
                        for i in range_constexpr(NGC):
                            tot = tot + vs[i][0].bitcast(fx.Float32)
                        res = yield [tot]
                    tot = fx.Float32(res[0] if isinstance(res, (list, tuple)) else res)
                    if qok:
                        lds_st(xs, MOE_OUTS + q, tot)
                gpu.barrier()
                combine_rows(
                    layer,
                    "m",
                    h1_src,
                    rg,
                    lambda p, t: (
                        lds_ld(xs, MOE_OUTS + t * 16 + 2 * p),
                        lds_ld(xs, MOE_OUTS + t * 16 + 2 * p + 1),
                    ),
                    mb(layer, "h2"),
                    mb(layer, "h2ssq"),
                    out_global,
                )
                gpu.barrier()

        # ================================================ QSA (sparse full attention)
        GI = n_layers  # global pointer-table row

        tl_count = [0]

        def stamp(label=""):
            """Debug timeline: CTA's realtime clock at a stage boundary -> timeline[bid][k]."""
            if const_expr(timeline):
                k = tl_count[0]
                tl_count[0] += 1
                TL_LABELS.append(label)
                if tid == 0:
                    t_ = mem_realtime()
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [fx.Int32(t_), fx.Int32(t_ >> 32)], fx.Int32
                        ),
                        _rsrc(ptr(GI, LY.G_TIMELINE)),
                        (bid * LY.TL_STAMPS + k) * 2,
                    )

        def rtt_at(rtt_row, tok):
            return fx.Int32(bo.buffer_load(rtt_row, tok, vec_width=1, dtype=T.i32))

        def req_p0(r):
            """Position of request r's first verify token (uniform for a uniform r)."""
            return _uniform(lds_ld(xi, RQ_P0 + r))

        def rtt_row_of(r):
            """req_to_token row of request r (r uniform)."""
            return _rsrc(
                ptr(GI, LY.G_RTT)
                + fx.Int64(_uniform(lds_ld(xi, RQ_REQ + r))) * (rtt_stride * 4)
            )

        def find_bucket(pfx_off, n, i):
            """Bucket b < n with pfx[b] <= i < pfx[b + 1] of a prefix-sum table in LDS."""
            b = fx.Int32(0)
            for k_ in range_constexpr(1, n):
                b = b + (i >= lds_ld(xi, pfx_off + k_)).select(fx.Int32(1), fx.Int32(0))
            return b

        def rope_axis(i):
            """MRoPE (interleaved, section 11/11/10) axis of rotary pair i < 32."""
            m3 = i % 3
            return (m3 == 1).select(
                fx.Int32(1), ((m3 == 2) & (i < 30)).select(fx.Int32(2), fx.Int32(0))
            )

        def rope_elem(x, e, p3, r_cos):
            """Neox rope of element e (< 64, lanes = elements within one wave) with partner x[e ^ 32].
            p3: callable axis -> position."""
            partner = _xshfl(x, 32)
            i = e % 32
            ax = rope_axis(i)
            p = (ax == 0).select(p3(0), (ax == 1).select(p3(1), p3(2)))
            c = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(r_cos, p * 64 + i, vec_width=1, dtype=T.bf16)
                )
            )
            sn = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(r_cos, p * 64 + 32 + i, vec_width=1, dtype=T.bf16)
                )
            )
            return (e < 32).select(x * c - partner * sn, x * c + partner * sn)

        def tok_p3(t):
            return lambda a: lds_ld(xi, POS3 + a * NT + t)

        def qp_elems(qp, specs):
            """[(t, e)] -> elements e of token t of the fused projection mailbox (one batched poll)."""
            ws = poll([(qp, t * (LY.QP_ROWS // 2) + e // 2, 1) for t, e in specs])
            out = []
            for i in range_constexpr(len(specs)):
                lo, hi = bf2_f32(ws[i][0])
                out.append((specs[i][1] % 2 == 0).select(lo, hi))
            return out

        def stage_qsa_proj(layer):
            act_gemm8(
                lambda: stage_act8(mb(layer, "a.mixx")),
                None,
                None,
                (LY.QP_RG + G - 1) // G,
                lambda i: fx.min(bid + i * G, LY.QP_RG - 1),
                emit8_bf_rows(
                    mb(layer, "qp"),
                    LY.QP_ROWS // 2,
                    valid=lambda i: bid + i * G < LY.QP_RG,
                ),
                split=(
                    LY.QP_RG_QKV,
                    ptr(layer, LY.P_IP),
                    ptr(layer, LY.P_IP_S),
                    ptr(layer, LY.P_IP2),
                    ptr(layer, LY.P_IP2_S),
                ),
            )

        def heads_norm_rope(qp, base_of, n, w_addr, r_cos, rope, out_off, ntok_heads):
            """Elements of ntok_heads (token, head) rows of width n (one per 256- or 128-thread group):
            bf16(gemma_rmsnorm) then (e < 64) neox rope -> LDS bf16 at out_off + row * n/2 words.
            base_of(row) -> (token, element offset in qp). All passes (THREADS // n rows each) share one poll
            round and one block reduction."""
            RPI = THREADS // n  # rows per pass
            e = tid % n
            passes = []
            for r0 in range_constexpr(0, ntok_heads, RPI):
                row = r0 + tid // n
                rowc = fx.min(row, ntok_heads - 1)
                t, base = base_of(rowc)
                passes.append((row, rowc, t, base))
            xv = qp_elems(qp, [(t, base + e) for _, _, t, base in passes])
            xx = [
                (passes[i][0] < ntok_heads).select(xv[i], fx.Float32(0.0))
                for i in range(len(passes))
            ]
            # row sums: one wave sum per pass, then the n / 64 waves of a row combine through LDS
            WPR = n // 64
            for i in range_constexpr(len(passes)):
                ws = wave_sum(xx[i] * xx[i])
                if lane == 0:
                    lds_st(red, i * WAVES + wave, ws)
            gpu.barrier()
            wv = fx.Float32(
                fx.BFloat16(bo.buffer_load(_rsrc(w_addr), e, vec_width=1, dtype=T.bf16))
            )
            w0 = (wave // WPR) * WPR
            for i in range_constexpr(len(passes)):
                row, rowc, t, base = passes[i]
                tsel = lds_ld(red, i * WAVES + w0)
                for w_ in range_constexpr(1, WPR):
                    tsel = tsel + lds_ld(red, i * WAVES + w0 + w_)
                rstd = _rsq(tsel * (1.0 / n) + EPS)
                y = bf_round(xx[i] * rstd * (fx.Float32(1.0) + wv))
                if const_expr(rope):
                    yr = bf_round(rope_elem(y, e, tok_p3(t), r_cos))
                    y = (e < 64).select(yr, y)
                if row < ntok_heads:
                    fx.ptr_store(y.to(fx.BFloat16), xb + (out_off * 2 + rowc * n + e))
            gpu.barrier()

        def stage_qsa_prep(layer):
            qp = mb(layer, "qp")
            r_cos = _rsrc(ptr(GI, LY.G_COS))
            if bid < LY.PREP_Q_TASKS:
                h = bid
                heads_norm_rope(
                    qp,
                    lambda row: (row, h * 512),
                    256,
                    ptr(layer, LY.P_QNORM),
                    r_cos,
                    True,
                    PREP_OFF,
                    NT,
                )
                NW = NT * 128
                for i in range_constexpr((NW + THREADS - 1) // THREADS):
                    q = tid + i * THREADS
                    if q < NW:
                        t = q // 128
                        put(
                            mb(layer, "q"),
                            t * (24 * 128) + h * 128 + q % 128,
                            lds_ld(xi, PREP_OFF + q),
                        )
                gpu.barrier()
            if (bid >= LY.PREP_K_CTA) & (bid < LY.PREP_K_CTA + 2):
                kvh = bid - LY.PREP_K_CTA
                heads_norm_rope(
                    qp,
                    lambda row: (row, LY.QP_K_OFF + kvh * 256),
                    256,
                    ptr(layer, LY.P_KNORM),
                    r_cos,
                    True,
                    PREP_OFF,
                    NT,
                )
                kvnew = mb(layer, "kvnew")
                r_loc = _rsrc(ptr(GI, LY.G_LOC))
                NW = NT * 128
                NIW = (NW + THREADS - 1) // THREADS
                qcs = [fx.min(tid + i * THREADS, NW - 1) for i in range(NIW)]
                vall = qp_elems(
                    qp,
                    [
                        (qc // 128, LY.QP_V_OFF + kvh * 256 + 2 * (qc % 128) + d_)
                        for qc in qcs
                        for d_ in range(2)
                    ],
                )
                for i in range_constexpr(NIW):
                    q = tid + i * THREADS
                    qc = qcs[i]
                    t = qc // 128
                    e = qc % 128
                    v = vall[2 * i : 2 * i + 2]
                    # 128 threads (2 waves) per token: the KV row base is wave-uniform -> 64-bit descriptor base
                    loc = _uniform(
                        fx.Int32(bo.buffer_load(r_loc, t * 2, vec_width=1, dtype=T.i32))
                    )
                    r_k = _rsrc(
                        ptr(layer, LY.P_KBUF)
                        + fx.Int64(loc) * 1024
                        + fx.Int64(kvh * 512)
                    )
                    r_v = _rsrc(
                        ptr(layer, LY.P_VBUF)
                        + fx.Int64(loc) * 1024
                        + fx.Int64(kvh * 512)
                    )
                    if q < NW:
                        kw = lds_ld(xi, PREP_OFF + q)
                        vw = bf16_pair(v[0], v[1]).bitcast(fx.Int32)
                        put(kvnew, t * 512 + kvh * 128 + e, kw)
                        put(kvnew, t * 512 + 256 + kvh * 128 + e, vw)
                        bo.buffer_store(kw, r_k, e)
                        bo.buffer_store(vw, r_v, e)
                gpu.barrier()
            if (bid >= LY.PREP_IDX_CTA0) & (bid < LY.PREP_IDX_CTA0 + R):
                # request r: index q (4 heads x 128) of its L tokens, then its index-key ring / compression
                r = bid - LY.PREP_IDX_CTA0
                tok0 = r * L
                pos0 = req_p0(r)
                req = _uniform(lds_ld(xi, RQ_REQ + r))
                rtt_row = rtt_row_of(r)
                heads_norm_rope(
                    qp,
                    lambda row: (tok0 + row // 4, LY.QP_IDX_OFF + (row % 4) * 128),
                    128,
                    ptr(layer, LY.P_IQNORM),
                    r_cos,
                    True,
                    PREP_OFF,
                    4 * L,
                )
                NW = L * 256
                for i in range_constexpr((NW + THREADS - 1) // THREADS):
                    q = tid + i * THREADS
                    if q < NW:
                        put(mb(layer, "qi"), tok0 * 256 + q, lds_ld(xi, PREP_OFF + q))
                gpu.barrier()
                # index k of every token -> LDS (f32) and the pending ring; in token order, a completed group
                # (pos % 4 == 3) is compressed from its 4 members (batch members from LDS, older ones from the ring)
                r_ring = _rsrc(ptr(layer, LY.P_RING))
                r_rp = _rsrc(ptr(GI, LY.G_ROPEPOS))
                NK = L * 128
                NIK = (NK + THREADS - 1) // THREADS
                qks = [fx.min(tid + i * THREADS, NK - 1) for i in range(NIK)]
                kall = qp_elems(
                    qp, [(tok0 + q // 128, LY.QP_IDX_OFF + 512 + q % 128) for q in qks]
                )
                for i in range_constexpr(NIK):
                    if tid + i * THREADS < NK:
                        lds_st(xs, PREP_KEYS + qks[i], kall[i])
                gpu.barrier()
                wk = fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            _rsrc(ptr(layer, LY.P_IKNORM)),
                            tid % 128,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                )
                gfirst = (
                    pos0 // 4
                )  # first group that can complete in this request's tokens
                for j in range_constexpr(L):
                    pos = pos0 + j
                    if tid < 128:
                        bo.buffer_store(
                            fx.Float32(lds_ld(xs, PREP_KEYS + j * 128 + tid)).to(
                                fx.BFloat16
                            ),
                            r_ring,
                            (req * 4 + pos % 4) * 128 + tid,
                        )
                    if tid < 3:
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [lds_ld(xi, POS3 + tid * NT + tok0 + j), fx.Int32(0)],
                                fx.Int32,
                            ),
                            r_rp,
                            ((req * 4 + pos % 4) * 3 + tid) * 2,
                        )
                    if (pos + 1) % 4 == 0:
                        acc = fx.Float32(0.0)
                        if tid < 128:
                            for g in range_constexpr(4):
                                pm = pos - 3 + g
                                src_batch = pm >= pos0
                                tb = fx.max(pm - pos0, fx.Int32(0))
                                from_lds = bf_round(
                                    lds_ld(xs, PREP_KEYS + tb * 128 + tid)
                                )
                                from_ring = fx.Float32(
                                    fx.BFloat16(
                                        bo.buffer_load(
                                            r_ring,
                                            (req * 4 + g) * 128 + tid,
                                            vec_width=1,
                                            dtype=T.bf16,
                                        )
                                    )
                                )
                                acc = acc + src_batch.select(from_lds, from_ring)
                        m = bf_round(acc * 0.25)
                        mm = (tid < 128).select(m, fx.Float32(0.0))
                        ss = block_sums([mm * mm])[0]
                        rstd = _rsq(ss * (1.0 / 128) + EPS)
                        yk = bf_round(mm * rstd * (fx.Float32(1.0) + wk))
                        ps = pos - 3  # group start: rope position of the first member
                        tsb = ps - pos0

                        def sp3(a, tsb=tsb, ps=ps):
                            from_lds = lds_ld(
                                xi, POS3 + a * NT + tok0 + fx.max(tsb, fx.Int32(0))
                            )
                            from_buf = _uniform(
                                fx.Int32(
                                    bo.buffer_load(
                                        r_rp,
                                        ((req * 4 + ps % 4) * 3 + a) * 2,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                            )
                            return (tsb >= 0).select(from_lds, from_buf)

                        ykr = bf_round(rope_elem(yk, tid, sp3, r_cos))
                        yk = (tid < 64).select(ykr, yk)
                        if tid < 128:
                            fx.ptr_store(yk.to(fx.BFloat16), xb + (PREP_OFF * 2 + tid))
                        gpu.barrier()
                        cslot = rtt_at(rtt_row, ps) // 4
                        if tid < 64:
                            w = lds_ld(xi, PREP_OFF + tid)
                            put(
                                mb(layer, "cnew"),
                                (r * CN + pos // 4 - gfirst) * 64 + tid,
                                w,
                            )
                            bo.buffer_store(
                                w, _rsrc(ptr(layer, LY.P_COMP)), cslot * 64 + tid
                            )
                        gpu.barrier()

        QS = 68  # LDS row stride (words) of the index q rows [4 S][128] bf16 ((stride / 4) odd)

        def stage_qsa_scores(layer):
            """Index logits of every complete block for every row: ilog[t][b] for b < nb_t = (pos_t + 1) // 4.
            Work item = (request r, tile of 16 blocks of its context), handled by the global wave item % (256 * 8).
            MFMA: A = index q rows (row, head) of request r from LDS, B = 16 compressed keys (blocks), k = 128 dims;
            a lane's 4 accumulator rows are the 4 heads of one row -> logit = sum relu / sqrt(128) in-lane."""
            ntile = _uniform(lds_ld(xi, SC_PFX + R))
            if bid * WAVES < ntile:
                qi = mb(layer, "qi")
                NW = 256 * NT
                NIW = (NW + THREADS - 1) // THREADS
                qv = poll(
                    [(qi, fx.min(tid + i * THREADS, NW - 1), 1) for i in range(NIW)]
                )
                for i in range_constexpr(NIW):
                    w_ = tid + i * THREADS
                    if (
                        w_ < NW
                    ):  # word w_ of row t = w_ // 256: head (w_ % 256) // 64, word w_ % 64
                        lds_st(
                            xi,
                            QI_OFF
                            + ((w_ // 256) * 4 + (w_ % 256) // 64) * QS
                            + w_ % 64,
                            qv[i][0],
                        )
                gpu.barrier()
                gw = bid * WAVES + wave
                nloop = (ntile - gw + G * WAVES - 1) // (G * WAVES)
                r_comp = _rsrc(ptr(layer, LY.P_COMP))
                cnew = mb(layer, "cnew")
                ilog = mb(layer, "ilog")
                for it, _c in range(fx.Int32(0), nloop, 1, init=[fx.Int32(0)]):
                    it = fx.Int32(it)
                    item = gw + it * (G * WAVES)
                    r = _uniform(find_bucket(SC_PFX, R, item))
                    tl = item - lds_ld(xi, SC_PFX + r)
                    pos0 = req_p0(r)
                    nb_all = (pos0 + L) // 4
                    gfirst = pos0 // 4
                    rtt_row = rtt_row_of(r)
                    blk = tl * 16 + lane % 16
                    bc = fx.min(blk, nb_all - 1)
                    cslot = rtt_at(rtt_row, bc * 4) // 4
                    kb = [
                        fx.Vector(
                            bo.buffer_load(
                                r_comp,
                                cslot * 64 + ks * 16 + (lane // 16) * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for ks in range(4)
                    ]
                    if bc >= gfirst:  # compressed inside this batch: from the mailbox
                        v = poll(
                            [
                                (
                                    cnew,
                                    (r * CN + bc - gfirst) * 64
                                    + ks * 16
                                    + (lane // 16) * 4
                                    + 2 * hf,
                                    2,
                                )
                                for ks in range(4)
                                for hf in range(2)
                            ]
                        )
                        kb = [
                            fx.Vector.from_elements(v[2 * ks] + v[2 * ks + 1], fx.Int32)
                            for ks in range(4)
                        ]
                    for rt in range_constexpr((4 * L + 15) // 16):
                        row = r * (4 * L) + fx.min(rt * 16 + lane % 16, 4 * L - 1)
                        cacc = fx.Vector.filled(4, 0.0, fx.Float32)
                        for ks in range_constexpr(4):
                            a_ = fx.ptr_load(
                                xs + (QI_OFF + row * QS + ks * 16 + (lane // 16) * 4),
                                result_type=v4f,
                            ).bitcast(fx.BFloat16)
                            cacc = fx.Vector(
                                rocdl.mfma_f32_16x16x32_bf16(
                                    T.vec(4, T.f32),
                                    [a_, kb[ks].bitcast(fx.BFloat16), cacc],
                                )
                            )
                        j = rt * 4 + lane // 16  # verify token of the request
                        logit = fx.Float32(0.0)
                        for e in range_constexpr(4):
                            logit = logit + fx.max(cacc[e], fx.Float32(0.0))
                        nb_t = (pos0 + j + 1) // 4
                        if (j < L) & (blk < nb_t):
                            put(
                                ilog,
                                (r * L + j) * LY.MAX_BLOCKS + blk,
                                logit * (1.0 / (128**0.5)),
                            )
                    _ = yield [fx.Int32(0)]
                gpu.barrier()

        def radix_topk(item_keys, item_ok, topk, emit):
            """Exact radix top-k of the CTA's items (orderable u32 keys, per-thread lists); emit(slot, j) for the
            selected items of this thread (per-wave private 256-bin LDS histograms, ties: lowest scan order)."""
            keys = xi + KEYS_OFF
            SCR = WAVES * 256
            ITEMS = len(item_keys)
            # prefilter: if every thread holds an item and topk == THREADS, at least topk items are >= the smallest
            # per-thread maximum m, so items < m cannot be selected (cuts the contended histogram atomics ~10x)
            imin = fx.Int32(-(2**31))
            smax = imin
            has = fx.Int32(0)
            for jj in range_constexpr(ITEMS):
                sk = fx.Int32(
                    item_keys[jj] ^ fx.Uint32(2**31)
                )  # signed order == unsigned key order
                smax = item_ok[jj].select(fx.max(smax, sk), smax)
                has = has | item_ok[jj].select(fx.Int32(1), fx.Int32(0))
            mval = (has != 0).select(smax, fx.Int32(2**31 - 1))
            for off in (32, 16, 8, 4, 2, 1):
                mval = _xred(mval, off, fx.min)
                has = _xred(has, off, fx.min)
            if lane == 0:
                lds_st(keys, SCR + 16 + wave, mval)
                lds_st(keys, SCR + 24 + wave, has)
            gpu.barrier()
            m_all = lds_ld(keys, SCR + 16)
            h_all = lds_ld(keys, SCR + 24)
            for w in range_constexpr(1, WAVES):
                m_all = fx.min(m_all, lds_ld(keys, SCR + 16 + w))
                h_all = fx.min(h_all, lds_ld(keys, SCR + 24 + w))
            use_f = (h_all != 0) & (topk == THREADS)
            thr = use_f.select(m_all, imin)
            item_ok = [
                ok & (fx.Int32(k ^ fx.Uint32(2**31)) >= thr)
                for ok, k in zip(item_ok, item_keys)
            ]
            gpu.barrier()

            def clear_hist():
                for q in range_constexpr(WAVES * 256 // THREADS):
                    lds_st(keys, tid + q * THREADS, fx.Int32(0))

            def select_digit(shift, prefix, remain):
                digit = 255 - fx.min(tid, 255)
                count = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    count = count + lds_ld(keys, w * 256 + digit)
                count = (tid < 256).select(count, fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    count, fx.ReductionOp.ADD, width=64
                )
                wave_total = fx.Int32(
                    rocdl.readlane(T.i32, fx.Int32(inclusive), fx.Int32(63))
                )
                if (wave < 4) & (lane == 63):
                    lds_st(keys, SCR + wave, wave_total)
                gpu.barrier()
                before_wave = fx.Int32(0)
                for w in range_constexpr(4):
                    before_wave = before_wave + (wave > w).select(
                        lds_ld(keys, SCR + w), fx.Int32(0)
                    )
                above = before_wave + inclusive - count
                hit = (tid < 256) & (above < remain) & (above + count >= remain)
                if hit:
                    lds_st(
                        keys, SCR + 8, fx.Int32(prefix | (fx.Uint32(digit) << shift))
                    )
                    lds_st(keys, SCR + 9, remain - above)
                gpu.barrier()
                return fx.Uint32(lds_ld(keys, SCR + 8)), lds_ld(keys, SCR + 9)

            prefix = fx.Uint32(0)
            remain = topk
            prefix_mask = 0
            for shift in (24, 16, 8, 0):
                clear_hist()
                gpu.barrier()
                for jj in range_constexpr(ITEMS):
                    k = item_keys[jj]
                    if item_ok[jj] & ((k & fx.Uint32(prefix_mask)) == prefix):
                        digit = fx.Int32((k >> shift) & fx.Uint32(255))
                        fx.atomic_add(
                            keys + (wave * 256 + digit),
                            fx.Int32(1),
                            syncscope="workgroup",
                        )
                gpu.barrier()
                prefix, remain = select_digit(shift, prefix, remain)
                prefix_mask |= 255 << shift
            threshold = prefix
            gt = [ok & (k > threshold) for ok, k in zip(item_ok, item_keys)]
            eq = [ok & (k == threshold) for ok, k in zip(item_ok, item_keys)]

            def scan_flags(flags):
                local = fx.Int32(0)
                offs = []
                for f in flags:
                    offs.append(local)
                    local = local + f.select(fx.Int32(1), fx.Int32(0))
                inclusive = fx.coop.warp_inclusive_scan(
                    local, fx.ReductionOp.ADD, width=64
                )
                wave_total = fx.Int32(
                    rocdl.readlane(T.i32, fx.Int32(inclusive), fx.Int32(63))
                )
                if lane == 63:
                    lds_st(keys, SCR + wave, wave_total)
                gpu.barrier()
                before = fx.Int32(0)
                total = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    wc = lds_ld(keys, SCR + w)
                    before = before + (wave > w).select(wc, fx.Int32(0))
                    total = total + wc
                base = before + inclusive - local
                return [base + o for o in offs], total

            gt_off, n_gt = scan_flags(gt)
            gpu.barrier()
            eq_off, _ = scan_flags(eq)
            gpu.barrier()
            need_eq = topk - n_gt
            for jj in range_constexpr(ITEMS):
                if gt[jj]:
                    emit(gt_off[jj], jj)
                if eq[jj] & (eq_off[jj] < need_eq):
                    emit(n_gt + eq_off[jj], jj)

        def okey(bits):
            return fx.Uint32((bits >= 0).select(bits ^ fx.Int32(-(2**31)), ~bits))

        def put_sel(sel, slot, blk, loc):
            bo.buffer_store(
                fx.Vector.from_elements(
                    [fx.Int32(blk), tag, fx.Int32(loc), tag], fx.Int32
                ),
                _rsrc(sel),
                slot * 4,
                cache_modifier=CM_DEV,
            )

        def stage_qsa_select(layer):
            """Top-512 blocks per row. Task (row t, segment s), tasks from the top CTA down: exact radix top-512 of
            the blocks [8192 s, 8192 s + 8192) (16 per thread). One segment (contexts < 32k): the result is final
            -> sel[t]; else candidates -> cand[t][s], merged by CTA sel2_cta0 + t (stage_qsa_merge_select)."""
            ntask = _uniform(lds_ld(xi, SL_PFX + NT))
            ci = (G - 1) - bid
            nloop = (ntask - ci + G - 1) // G
            for it, _c in range(fx.Int32(0), nloop, 1, init=[fx.Int32(0)]):
                it = fx.Int32(it)
                i_task = ci + it * G
                t = _uniform(find_bucket(SL_PFX, NT, i_task))
                sg = i_task - lds_ld(xi, SL_PFX + t)
                r = t // L
                pos0 = req_p0(r)
                rtt_row = rtt_row_of(r)
                nb = (pos0 + t % L + 1) // 4
                nseg = fx.max((nb + LY.SEG_BLOCKS - 1) // LY.SEG_BLOCKS, fx.Int32(1))
                ITEMS = LY.SEG_BLOCKS // THREADS  # 16
                b0 = sg * LY.SEG_BLOCKS
                nbs = fx.min(fx.max(nb - b0, fx.Int32(0)), fx.Int32(LY.SEG_BLOCKS))
                topk = fx.min(fx.Int32(512), nbs)
                item_idx = [
                    b0 + tid + THREADS * j_ for j_ in range_constexpr(ITEMS)
                ]  # interleaved
                item_ok = [i < nb for i in item_idx]
                last = fx.max(nb - 1, fx.Int32(0))
                item_loc = [rtt_at(rtt_row, fx.min(i, last) * 4) for i in item_idx]
                # nb == 0: nothing to select; poll a pair this layer already published instead of ilog[-1]
                ilog = (nb > 0).select(
                    mb(layer, "ilog") + fx.Int64(t) * (LY.MAX_BLOCKS * 8),
                    mb(layer, "qi"),
                )
                got = poll([(ilog, fx.min(i, last), 1) for i in item_idx], ITEMS)
                item_keys = [okey(got[jj][0]) for jj in range(ITEMS)]
                sel = mb(layer, "sel") + fx.Int64(t) * (LY.SEL_STRIDE * 8)
                cs = t * LY.NSEG_MAX + sg
                if nseg == 1:
                    radix_topk(
                        item_keys,
                        item_ok,
                        topk,
                        lambda slot, jj: put_sel(sel, slot, item_idx[jj], item_loc[jj]),
                    )
                    if tid == 0:
                        put(sel, 1024, topk)
                else:
                    ck = mb(layer, "cand_k")
                    cbl = mb(layer, "cand_bl")

                    def emit(slot, jj):
                        put(ck, cs * 512 + slot, got[jj][0])
                        put_sel(
                            cbl + fx.Int64(cs * 512 * 16),
                            slot,
                            item_idx[jj],
                            item_loc[jj],
                        )

                    radix_topk(item_keys, item_ok, topk, emit)
                gpu.barrier()
                _ = yield [fx.Int32(0)]

        def stage_qsa_merge_select(layer):
            """Long contexts: CTA sel2_cta0 + t merges the segment top-512s of row t (<= 9 x 512 candidates,
            one per (segment, thread)) into its top-512 -> sel[t]."""
            SEL2 = LY.sel2_cta0(NT)
            if bid >= SEL2:
                t = bid - SEL2
                nb = (req_p0(t // L) + t % L + 1) // 4
                nseg = (nb + LY.SEG_BLOCKS - 1) // LY.SEG_BLOCKS
                if nseg > 1:
                    # segment i holds min(512, its block count) candidates (known without polling cand_n)
                    cnt = [
                        fx.min(
                            fx.Int32(512), fx.max(nb - i * LY.SEG_BLOCKS, fx.Int32(0))
                        )
                        for i in range(LY.NSEG_MAX)
                    ]
                    item_ok = [(i < nseg) & (tid < cnt[i]) for i in range(LY.NSEG_MAX)]
                    # invalid items poll candidate 0 of segment 0 (always present)
                    cidx = [
                        item_ok[i].select(
                            (t * LY.NSEG_MAX + i) * 512 + tid, t * LY.NSEG_MAX * 512
                        )
                        for i in range(LY.NSEG_MAX)
                    ]
                    ck = mb(layer, "cand_k")
                    cbl = mb(layer, "cand_bl")
                    got = poll(
                        [(ck, c, 1) for c in cidx] + [(cbl, c * 2, 2) for c in cidx],
                        2 * LY.NSEG_MAX,
                    )
                    item_keys = [okey(got[i][0]) for i in range(LY.NSEG_MAX)]
                    blks = [got[LY.NSEG_MAX + i][0] for i in range(LY.NSEG_MAX)]
                    locs = [got[LY.NSEG_MAX + i][1] for i in range(LY.NSEG_MAX)]
                    sel = mb(layer, "sel") + fx.Int64(t) * (LY.SEL_STRIDE * 8)
                    radix_topk(
                        item_keys,
                        item_ok,
                        fx.Int32(512),
                        lambda slot, jj: put_sel(sel, slot, blks[jj], locs[jj]),
                    )
                    if tid == 0:
                        put(sel, 1024, fx.Int32(512))
                    gpu.barrier()

        def stage_qsa_attn(layer):
            """Split flash-decode on MFMA: task (token t, chunk of 64 selected tokens) -> partial (o, m, l) of all
            24 heads; waves [4 kvh, 4 kvh + 4) serve kv head kvh. Batch tokens of the row's request (pos >= pos0 of the
            request) read K/V from kvnew."""
            NTASK = NT * LY.ATT_TASKS
            nloop = (NTASK - bid + G - 1) // G
            for it, _c in range(fx.Int32(0), nloop, 1, init=[fx.Int32(0)]):
                it = fx.Int32(it)
                task = bid + it * G
                t = task // LY.ATT_TASKS
                c = task % LY.ATT_TASKS
                r = t // L
                pos0 = req_p0(r)
                rtt_row = rtt_row_of(r)
                pos = pos0 + t % L
                seq = pos + 1
                sel = mb(layer, "sel") + fx.Int64(t) * (LY.SEL_STRIDE * 8)
                j = tid // 8
                ch = tid % 8
                qmb = mb(layer, "q")
                got = poll(
                    [(qmb, t * (24 * 128) + tid + i * THREADS, 1) for i in range(6)]
                    + [(sel, 1024, 1)]
                )
                for i in range_constexpr(6):
                    pq = tid + i * THREADS
                    hd = pq // 128
                    lds_st(
                        xi,
                        ATT_Q_OFF + ((hd // 12) * 16 + hd % 12) * ATT_KS + pq % 128,
                        got[i][0],
                    )
                for i in range_constexpr(2):
                    z = tid + i * THREADS
                    lds_st(
                        xi,
                        ATT_Q_OFF
                        + ((z // 512) * 16 + 12 + (z % 512) // 128) * ATT_KS
                        + z % 128,
                        fx.Int32(0),
                    )
                nsel = got[6][0]
                stamp("att.q")
                tail_start = (seq // 4) * 4
                is_blk = c < LY.ATT_CHUNKS - 1
                si = c * 16 + j // 4
                blk = fx.Int32(0)
                locb = fx.Int32(0)
                if is_blk & (si < nsel):
                    bl = poll([(sel, fx.min(si, 511) * 2, 2)])[0]
                    blk = bl[0]
                    locb = bl[1]
                tok_t = tail_start + j
                tok = is_blk.select(blk * 4 + j % 4, tok_t)
                valid = is_blk.select(si < nsel, tok_t < seq)
                tokc = fx.min(fx.max(tok, fx.Int32(0)), seq - 1)
                loc_t = rtt_at(rtt_row, fx.min(tok_t, seq - 1))
                locv = is_blk.select(locb + j % 4, loc_t)
                # thread ch holds the 16 B chunks ch + 8 q (dims 8 (ch + 8 q) .. + 8) of its K/V rows; 64-bit
                # addresses: the KV pools may exceed 4 GB
                rowk = ptr(layer, LY.P_KBUF) + fx.Int64(locv) * 1024
                rowv = ptr(layer, LY.P_VBUF) + fx.Int64(locv) * 1024
                kw = [
                    [
                        _gload_v4(rowk + (kvh * 512 + (ch + 8 * q) * 16))
                        for q in range(4)
                    ]
                    for kvh in range(2)
                ]
                vw = [
                    [
                        _gload_v4(rowv + (kvh * 512 + (ch + 8 * q) * 16))
                        for q in range(4)
                    ]
                    for kvh in range(2)
                ]
                if valid & (tokc >= pos0):
                    kvnew = mb(layer, "kvnew")
                    tb = r * L + tokc - pos0
                    for kvh in range_constexpr(2):
                        kn = poll(
                            [
                                (
                                    kvnew,
                                    tb * 512 + kvh * 128 + (ch + 8 * q) * 4 + 2 * hf,
                                    2,
                                )
                                for q in range(4)
                                for hf in range(2)
                            ]
                        )
                        vn = poll(
                            [
                                (
                                    kvnew,
                                    tb * 512
                                    + 256
                                    + kvh * 128
                                    + (ch + 8 * q) * 4
                                    + 2 * hf,
                                    2,
                                )
                                for q in range(4)
                                for hf in range(2)
                            ]
                        )
                        kw[kvh] = [
                            fx.Vector.from_elements(kn[2 * q] + kn[2 * q + 1], fx.Int32)
                            for q in range(4)
                        ]
                        vw[kvh] = [
                            fx.Vector.from_elements(vn[2 * q] + vn[2 * q + 1], fx.Int32)
                            for q in range(4)
                        ]
                for kvh in range_constexpr(2):
                    for q in range_constexpr(4):
                        for e in range_constexpr(4):
                            lds_st(
                                xi,
                                ATT_OFF
                                + kvh * ATT_KV_WORDS
                                + j * ATT_KS
                                + (ch + 8 * q) * 4
                                + e,
                                kw[kvh][q][e],
                            )
                if ch == 0:
                    lds_st(
                        xi, ATT_M_OFF + 64 + j, valid.select(fx.Int32(1), fx.Int32(0))
                    )
                gpu.barrier()
                stamp("att.kv")
                # ---- S = Q K^T (MFMA): wave -> (kv head, 16-token tile), full 256 dims
                kvw = wave // 4
                nt_ = wave % 4
                cacc = fx.Vector.filled(4, 0.0, fx.Float32)
                for ks in range_constexpr(8):
                    dw = ks * 16 + (lane // 16) * 4
                    a = fx.ptr_load(
                        xs + (ATT_Q_OFF + (kvw * 16 + lane % 16) * ATT_KS + dw),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    bk = fx.ptr_load(
                        xs
                        + (
                            ATT_OFF
                            + kvw * ATT_KV_WORDS
                            + (nt_ * 16 + lane % 16) * ATT_KS
                            + dw
                        ),
                        result_type=v4f,
                    ).bitcast(fx.BFloat16)
                    cacc = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, bk, cacc])
                    )
                for i in range_constexpr(4):
                    lds_st(
                        xs,
                        ATT_S_OFF
                        + kvw * 1024
                        + (4 * (lane // 16) + i) * 64
                        + nt_ * 16
                        + lane % 16,
                        cacc[i],
                    )
                gpu.barrier()
                stamp("att.qk")
                # ---- V^T into the K regions (K is dead), softmax of the 2 x 16 (padded) head rows -> P (bf16)
                for kvh in range_constexpr(2):
                    for q in range_constexpr(4):
                        vv = vw[kvh][q].bitcast(fx.BFloat16)
                        for e in range_constexpr(8):
                            fx.ptr_store(
                                vv[e],
                                xb
                                + (
                                    (ATT_OFF + kvh * ATT_KV_WORDS) * 2
                                    + ((ch + 8 * q) * 8 + e) * ATT_VS
                                    + j
                                ),
                            )
                okv = lds_ld(xi, ATT_M_OFF + 64 + lane) != 0
                for r in range_constexpr(4):
                    row = wave + 8 * r  # (kvh, head) row: kvh = row // 16
                    hl = row % 16
                    sv = lds_ld(xs, ATT_S_OFF + row * 64 + lane) * (1.0 / 16.0)
                    live = okv & (hl < 12)
                    sv = live.select(sv, fx.Float32(NEG_INF))
                    mx = sv
                    for off in (32, 16, 8, 4, 2, 1):
                        mx = _xred(mx, off, fx.max)
                    pv = live.select(_exp(sv - mx), fx.Float32(0.0))
                    sm = wave_sum(pv)
                    fx.ptr_store(
                        pv.to(fx.BFloat16), xb + (ATT_P_OFF * 2 + row * ATT_VS + lane)
                    )
                    if (lane == 0) & (hl < 12):
                        hh = (row // 16) * 12 + hl
                        lds_st(xs, ATT_M_OFF + hh, mx)
                        lds_st(xs, ATT_M_OFF + 24 + hh, sm)
                gpu.barrier()
                stamp("att.sm")
                # ---- O = P V (MFMA): wave -> (kv head, 64 dims)
                part = mb(layer, "part")
                pbase = task * LY.PART_STRIDE
                for t2 in range_constexpr(4):
                    n0 = nt_ * 64 + t2 * 16
                    oacc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(2):
                        ke = ks * 32 + (lane // 16) * 8
                        a = fx.ptr_load(
                            xs
                            + (ATT_P_OFF + ((kvw * 16 + lane % 16) * ATT_VS + ke) // 2),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        bv = fx.ptr_load(
                            xs
                            + (
                                ATT_OFF
                                + kvw * ATT_KV_WORDS
                                + ((n0 + lane % 16) * ATT_VS + ke) // 2
                            ),
                            result_type=v4f,
                        ).bitcast(fx.BFloat16)
                        oacc = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, bv, oacc])
                        )
                    if lane < 48:  # rows (heads) 4 (lane // 16) + i < 12
                        for i in range_constexpr(4):
                            put(
                                part,
                                pbase
                                + (kvw * 12 + 4 * (lane // 16) + i) * 256
                                + n0
                                + lane % 16,
                                oacc[i],
                            )
                if tid < 48:
                    put(part, pbase + 24 * 256 + tid, lds_ld(xs, ATT_M_OFF + tid))
                gpu.barrier()
                _ = yield [fx.Int32(0)]

        def stage_qsa_merge(layer):
            """Task (token, head): combine the chunk partials, sigmoid gate -> og."""
            qp = mb(layer, "qp")
            NTASK = NT * LY.MERGE_TASKS
            MG = G
            nloop = (NTASK - bid + MG - 1) // MG
            for it, _c in range(fx.Int32(0), nloop, 1, init=[fx.Int32(0)]):
                it = fx.Int32(it)
                task = bid + it * MG
                t = task // LY.MERGE_TASKS
                h = task % LY.MERGE_TASKS
                part = mb(layer, "part")
                NC = LY.ATT_CHUNKS
                tb = t * LY.ATT_TASKS * LY.PART_STRIDE
                if tid < NC:
                    pb = tb + tid * LY.PART_STRIDE + 24 * 256
                    ml = poll([(part, pb + h, 1), (part, pb + 24 + h, 1)])
                    lds_st(xs, OUTS + tid, ml[0][0].bitcast(fx.Float32))
                    lds_st(xs, OUTS + 64 + tid, ml[1][0].bitcast(fx.Float32))
                gpu.barrier()
                if tid < 256:
                    d = tid
                    M = lds_ld(xs, OUTS)
                    for cc in range_constexpr(1, NC):
                        M = fx.max(M, lds_ld(xs, OUTS + cc))
                    L = fx.Float32(0.0)
                    O = fx.Float32(0.0)
                    got = poll(
                        [
                            (part, tb + cc * LY.PART_STRIDE + h * 256 + d, 1)
                            for cc in range(NC)
                        ]
                        + [(qp, t * (LY.QP_ROWS // 2) + (h * 512 + 256 + d) // 2, 1)],
                        NC + 1,
                    )
                    for cc in range_constexpr(NC):
                        mc = lds_ld(xs, OUTS + cc)
                        wgt = (mc > NEG_INF * 0.5).select(_exp(mc - M), fx.Float32(0.0))
                        L = L + wgt * lds_ld(xs, OUTS + 64 + cc)
                        O = O + wgt * got[cc][0].bitcast(fx.Float32)
                    glo, ghi = bf2_f32(got[NC][0])
                    gt = (d % 2 == 0).select(glo, ghi)
                    og = O * _rcp(L) * sigmoid(gt)
                    fx.ptr_store(og.to(fx.BFloat16), xb + ((OUTS + 128) * 2 + tid))
                gpu.barrier()
                if tid < 128:
                    put(
                        mb(layer, "og"),
                        t * (GDN_VAL_DIM // 2) + h * 128 + tid,
                        lds_ld(xi, OUTS + 128 + tid),
                    )
                # max |og| of (row, head) for the o_proj FP8 activation scale (waves 0, 1 hold the 256 values)
                g0_, g1_ = bf2_f32(lds_ld(xi, OUTS + 128 + fx.min(tid, 127)))
                gm = fx.max(fx.max(g0_, -g0_), fx.max(g1_, -g1_))
                for off in (32, 16, 8, 4, 2, 1):
                    gm = _xred(gm, off, fx.max)
                if lane == 0:
                    lds_st(red, wave, gm)
                gpu.barrier()
                if tid == 0:
                    put(
                        mb(layer, "og_amax"),
                        t * 24 + h,
                        fx.max(lds_ld(red, 0), lds_ld(red, 1)),
                    )
                gpu.barrier()
                _ = yield [fx.Int32(0)]

        # ================================================ PLE (per-layer n-gram embedding, layer 1)
        def stage_ple_proj(layer):
            """[key; value] = [W_key; W_value] @ emb[t]  (800 row groups, <= 4 per CTA), all rows."""
            emb = ptr(GI, LY.G_PLE_EMB)
            act_gemm(
                lambda h: stage_bf16_global(
                    emb, KH, ACT, S_ACT, row_elems=HIDDEN, elem0=h * KH
                ),
                _rsrc(ptr(layer, LY.P_PLE_W)),
                (LY.PLE_RG + G - 1) // G,
                lambda i: fx.min(bid + i * G, LY.PLE_RG - 1),
                emit_bf_rows(
                    mb(layer, "ple_kv"),
                    LY.PLE_ROWS // 2,
                    valid=lambda i: bid + i * G < LY.PLE_RG,
                ),
            )

        def stage_ple_apply(layer, src):
            """Every CTA, token by token: gates of the 4 branches (redundant), gated value + its group norm;
            then the dilated short conv of channels [40 bid, 40 bid + 40) over the T tokens (window after every
            token -> intermediate_conv_state[step]) -> mailbox ple_h[t] = H + gated + silu(conv) (+ ssq partials)."""
            kv = mb(layer, "ple_kv")
            NQ = HC_DIM // 4 // THREADS  # 5 quads per thread
            js = [tid + i * THREADS for i in range(NQ)]
            grp = [j // (HIDDEN // 4) for j in js]
            r_nk, r_nq, r_nc = (
                _rsrc(ptr(layer, sl)) for sl in (LY.P_PLE_NK, LY.P_PLE_NQ, LY.P_PLE_NC)
            )
            c0 = bid * LY.PLE_CH_PER_CTA
            PO = W0  # [T][3][64] gv, gvn, h of this CTA's 40 channels

            def ldw(r, j):
                return (
                    fx.Vector(bo.buffer_load(r, j * 2, vec_width=2, dtype=T.i32))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )

            wk = [ldw(r_nk, j) for j in js]
            wq = [ldw(r_nq, j) for j in js]
            wc = [ldw(r_nc, j) for j in js]

            def per_group(vals):
                acc = [fx.Float32(0.0) for _ in range(HC)]
                for i in range_constexpr(NQ):
                    for g in range_constexpr(HC):
                        acc[g] = acc[g] + (grp[i] == g).select(vals[i], fx.Float32(0.0))
                return acc

            def gsel(vals, i):
                r = vals[0]
                for g in range_constexpr(1, HC):
                    r = (grp[i] == g).select(vals[g], r)
                return r

            sq = lambda v: v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]  # noqa: E731
            for t in range_constexpr(NT):
                hv = h_quads(src, [(t, j) for j in js])
                got = poll(
                    [(kv, t * (LY.PLE_ROWS // 2) + j * 2, 2) for j in js]
                    + [
                        (
                            kv,
                            t * (LY.PLE_ROWS // 2)
                            + HC_DIM // 2
                            + (j % (HIDDEN // 4)) * 2,
                            2,
                        )
                        for j in js
                    ]
                )
                key = [
                    fx.Vector.from_elements(got[i], fx.Int32)
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                    for i in range(NQ)
                ]
                val = [
                    fx.Vector.from_elements(got[NQ + i], fx.Int32)
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                    for i in range(NQ)
                ]
                tots = block_sums(
                    per_group([sq(k) for k in key]) + per_group([sq(h_) for h_ in hv])
                )
                rk = [_rsq(tt * (1.0 / HIDDEN) + EPS) for tt in tots[:HC]]
                rq = [_rsq(tt * (1.0 / HIDDEN) + EPS) for tt in tots[HC:]]
                prods = []
                for i in range_constexpr(NQ):
                    a, b = gsel(rk, i), gsel(rq, i)
                    pr = fx.Float32(0.0)
                    for e in range_constexpr(4):
                        kn = bf_round(key[i][e] * a * (fx.Float32(1.0) + wk[i][e]))
                        qn = bf_round(hv[i][e] * b * (fx.Float32(1.0) + wq[i][e]))
                        pr = pr + bf_round(kn * qn)
                    prods.append(pr)
                gs = block_sums(per_group(prods))
                act = []
                for g in range_constexpr(HC):
                    gt = bf_round(bf_round(gs[g]) * (1.0 / HIDDEN**0.5))
                    mag = fx.max(fx.max(gt, -gt), fx.Float32(1.0e-6))
                    root = bf_round(_rcp(_rsq(mag)))
                    sgn = (gt > 0.0).select(
                        fx.Float32(1.0),
                        (gt < 0.0).select(fx.Float32(-1.0), fx.Float32(0.0)),
                    )
                    act.append(bf_round(sigmoid(bf_round(root * sgn))))
                gv = [
                    [bf_round(gsel(act, i) * val[i][e]) for e in range(4)]
                    for i in range(NQ)
                ]
                tc = block_sums(
                    per_group(
                        [
                            g_[0] * g_[0]
                            + g_[1] * g_[1]
                            + g_[2] * g_[2]
                            + g_[3] * g_[3]
                            for g_ in gv
                        ]
                    )
                )
                rc = [_rsq(tt * (1.0 / HIDDEN) + EPS) for tt in tc]
                for i in range_constexpr(NQ):
                    for e in range_constexpr(4):
                        el = js[i] * 4 + e
                        li_ = el - c0
                        if (li_ >= 0) & (li_ < LY.PLE_CH_PER_CTA):
                            gvn = bf_round(
                                gv[i][e] * gsel(rc, i) * (fx.Float32(1.0) + wc[i][e])
                            )
                            lds_st(xs, PO + t * 192 + li_, gv[i][e])
                            lds_st(xs, PO + t * 192 + 64 + li_, gvn)
                            lds_st(xs, PO + t * 192 + 128 + li_, hv[i][e])
            gpu.barrier()
            r_cw = _rsrc(ptr(layer, LY.P_PLE_CW))
            r_ist = _rsrc(ptr(layer, LY.P_PLE_ISTATE))
            PH = PO + NT * 192  # [S][40] ple_h values
            if tid < LY.PLE_CH_PER_CTA:
                c = c0 + tid
                cw = (
                    fx.Vector(bo.buffer_load(r_cw, c * 2, vec_width=2, dtype=T.i32))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                for r_ in range_constexpr(R):
                    msr = _uniform(lds_ld(xi, RQ_MS + r_))
                    r_st = _rsrc(
                        ptr(layer, LY.P_PLE_STATE)
                        + fx.Int64(msr) * LY.PLE_STATE_SLOT_BYTES
                    )
                    win = [
                        fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    r_st,
                                    c * LY.PLE_STATE_LEN + k,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            )
                        )
                        for k in range(LY.PLE_STATE_LEN)
                    ]
                    for j_ in range_constexpr(L):
                        t = r_ * L + j_
                        win.append(lds_ld(xs, PO + t * 192 + 64 + tid))
                        w = win[j_ : j_ + LY.PLE_STATE_LEN + 1]
                        cv = bf_round(
                            cw[0] * w[0] + cw[1] * w[3] + cw[2] * w[6] + cw[3] * w[9]
                        )
                        cv = bf_round(cv * sigmoid(cv))
                        out = bf_round(lds_ld(xs, PO + t * 192 + tid) + cv)
                        lds_st(
                            xs,
                            PH + t * 40 + tid,
                            bf_round(lds_ld(xs, PO + t * 192 + 128 + tid) + out),
                        )
                        for k in range_constexpr(LY.PLE_STATE_LEN):
                            bo.buffer_store(
                                fx.Float32(w[k + 1]).to(fx.BFloat16),
                                r_ist,
                                r_ * ple_ist_row
                                + (j_ * HC_DIM + c) * LY.PLE_STATE_LEN
                                + k,
                            )
            gpu.barrier()
            NPP = NT * (LY.PLE_CH_PER_CTA // 2)
            for q0 in range_constexpr(0, NPP, THREADS):
                q = tid + q0
                if q < NPP:
                    t = q // (LY.PLE_CH_PER_CTA // 2)
                    p = q % (LY.PLE_CH_PER_CTA // 2)
                    a = lds_ld(xs, PH + t * 40 + 2 * p)
                    b = lds_ld(xs, PH + t * 40 + 2 * p + 1)
                    put_bf(mb(layer, "ple_h"), t * (HC_DIM // 2) + c0 // 2 + p, a, b)
            # sum-of-squares partial of this CTA's 40 channels (one branch) per token
            if wave == 0:
                for t in range_constexpr(NT):
                    v = (lane < LY.PLE_CH_PER_CTA).select(
                        lds_ld(xs, PH + t * 40 + fx.min(lane, 39)), fx.Float32(0.0)
                    )
                    ss = wave_sum(v * v)
                    if lane == 0:
                        br = c0 // HIDDEN
                        put(
                            mb(layer, "ple_ssq"),
                            (t * 4 + br) * LY.PLE_PARTS
                            + (c0 % HIDDEN) // LY.PLE_CH_PER_CTA,
                            ss,
                        )
            gpu.barrier()

        # ================================================ verify prelude (embeddings + n-gram history)
        def stage_prelude():
            """CTA PRELUDE_CTA0 + t (row t of request r = t // L, step j = t % L): 16 PLE n-gram ids from (the
            request's history, its batch tokens) -> fp8 rows * scale -> emb[t] (global bf16); the n-gram context
            after step j -> intermediate_context[r][j]."""
            if (bid >= LY.PRELUDE_CTA0) & (bid < LY.PRELUDE_CTA0 + NT):
                t = bid - LY.PRELUDE_CTA0
                r = t // L
                j = t % L
                msr = _uniform(lds_ld(xi, RQ_MS + r))
                r_ctx = _rsrc(ptr(GI, LY.G_NGRAM_CTX) + fx.Int64(msr) * 16)
                r_meta = _rsrc(ptr(GI, LY.G_NGRAM_META))
                r_ids = _rsrc(ptr(GI, LY.G_INPUT_IDS))

                def ld64(r_, i):
                    w = fx.Vector(bo.buffer_load(r_, i * 2, vec_width=2, dtype=T.i32))
                    return (fx.Int64(w[1]) << 32) | fx.Int64(fx.Uint32(w[0]))

                def c_at(k):
                    """Token k of the request's sequence (k < 0: history c[-2], c[-1])."""
                    from_ids = ld64(r_ids, r * L + fx.max(k, fx.Int32(0)))
                    from_ctx = ld64(
                        r_ctx, fx.min(fx.max(k + 2, fx.Int32(0)), fx.Int32(1))
                    )
                    return (k >= 0).select(from_ids, from_ctx)

                t0 = c_at(j - 2)
                t1 = c_at(j - 1)
                t2 = c_at(j)
                eos = fx.Int64(eos_token_id)
                NH = LY.NGRAM_HEADS
                if tid < NH:
                    hd = tid
                    p2 = ((t0 == eos) | (t1 == eos)).select(eos, t0)
                    mixed = (t2 * ld64(r_meta, 0)) ^ (t1 * ld64(r_meta, 1))
                    mixed3 = mixed ^ (p2 * ld64(r_meta, 2))
                    m = (hd < LY.HEADS_PER_NGRAM).select(mixed, mixed3)
                    vsz = ld64(r_meta, 3 + hd)
                    off = ld64(r_meta, 3 + NH + hd)
                    nid = fx.Int64(fx.Uint64(m) % fx.Uint64(vsz)) + off
                    lds_st(xi, PREP_OFF + 2 * tid, fx.Int32(nid))
                    lds_st(xi, PREP_OFF + 2 * tid + 1, fx.Int32(nid >> 32))
                gpu.barrier()
                scale = fx.Float32(
                    fx.BFloat16(
                        bo.buffer_load(
                            _rsrc(ptr(GI, LY.G_NGRAM_SCALE)),
                            0,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                )
                r_emb = _rsrc(ptr(GI, LY.G_PLE_EMB))
                DW = LY.NGRAM_DIM // 4  # 40 dwords per head row
                for rr in range_constexpr((NH + WAVES - 1) // WAVES):
                    hd = wave + WAVES * rr
                    if hd < NH:
                        lo = _uniform(lds_ld(xi, PREP_OFF + 2 * hd))
                        hi = _uniform(lds_ld(xi, PREP_OFF + 2 * hd + 1))
                        rowaddr = (
                            ptr(GI, LY.G_NGRAM_W)
                            + ((fx.Int64(hi) << 32) | fx.Int64(fx.Uint32(lo)))
                            * LY.NGRAM_DIM
                        )
                        if lane < DW:
                            word = fx.Int32(
                                bo.buffer_load(
                                    _rsrc(rowaddr), lane, vec_width=1, dtype=T.i32
                                )
                            )
                            one = fx.Float32(1.0).ir_value()
                            vals = []
                            for half in range_constexpr(2):
                                pr = fx.Vector(
                                    rocdl.cvt_scalef32_pk_bf16_fp8(
                                        T.vec(2, T.bf16),
                                        word.ir_value(),
                                        one,
                                        bool(half),
                                    )
                                )
                                vals += [
                                    fx.Float32(pr[0]) * scale,
                                    fx.Float32(pr[1]) * scale,
                                ]
                            e0 = (
                                (t * NH + hd) * LY.NGRAM_DIM + lane * 4
                            )  # = t * HIDDEN + hd * 160 + 4 lane
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [
                                        bf16_pair(vals[0], vals[1]).bitcast(fx.Int32),
                                        bf16_pair(vals[2], vals[3]).bitcast(fx.Int32),
                                    ],
                                    fx.Int32,
                                ),
                                r_emb,
                                e0 // 2,
                            )
                if tid == 0:
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [
                                fx.Int32(t1),
                                fx.Int32(t1 >> 32),
                                fx.Int32(t2),
                                fx.Int32(t2 >> 32),
                            ],
                            fx.Int32,
                        ),
                        _rsrc(ptr(GI, LY.G_NGRAM_ICTX)),
                        r * (2 * ictx_row) + j * 4,
                    )
                gpu.barrier()

        # ================================================ MTP draft input fusion
        def stage_mtp_fusion():
            """H[t] = fc_embedding(gemma_norm(embed(tok_t)))  (broadcast to the 4 branches)
                      + fc_hidden(gemma_norm(hc_hidden[t]) per branch)   -> mailbox mtp_h (+ sum-of-squares partials).
            Every CTA computes the row norms; CTA rg < 160 runs both GEMMs for rows [16 rg, 16 rg + 16): fc_embedding
            with the S row columns, fc_hidden with the 4 S (row, branch) columns, the normalized B rows staged in MF_KP
            K phases (accumulators kept across the phases)."""
            r_hin = _rsrc(ptr(GI, LY.G_MTP_HIN))
            r_emb = ptr(GI, LY.G_EMBED_W)
            r_pe = _rsrc(ptr(GI, LY.G_MTP_PE))
            r_ph = _rsrc(ptr(GI, LY.G_MTP_PH))
            NQE = HIDDEN // 4
            NQH = HC_DIM // 4

            def emb_quad(t, q):
                tok = lds_ld(xi, EMB_TOK + t)
                w = fx.Vector(
                    bo.buffer_load(
                        _rsrc(r_emb + fx.Int64(tok) * (HIDDEN * 2)),
                        q * 2,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
                return w.bitcast(fx.BFloat16).to(fx.Float32)

            def hin_quad(t, q):
                w = fx.Vector(
                    bo.buffer_load(
                        r_hin, t * (HC_DIM // 2) + q * 2, vec_width=2, dtype=T.i32
                    )
                )
                return w.bitcast(fx.BFloat16).to(fx.Float32)

            # rstd of every embedding row and hidden row: wave per row, loop over the row's 16-byte quads
            for t0 in range_constexpr(0, NT, WAVES):
                t = t0 + wave
                if t < NT:
                    ae = fx.Float32(0.0)
                    for i in range_constexpr(NQE // 64):
                        v = emb_quad(t, lane + 64 * i)
                        ae = ae + v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
                    ae = wave_sum(ae)
                    if lane == 0:
                        lds_st(xs, MF_RS + t, _rsq(ae * (1.0 / HIDDEN) + EPS))
                    res = None
                    for it, st in range(
                        fx.Int32(0), fx.Int32(NQH // 256), 1, init=[fx.Float32(0.0)]
                    ):
                        it = fx.Int32(it)
                        ah = st[0]
                        for i in range_constexpr(4):
                            v = hin_quad(t, (it * 4 + i) * 64 + lane)
                            ah = (
                                ah
                                + v[0] * v[0]
                                + v[1] * v[1]
                                + v[2] * v[2]
                                + v[3] * v[3]
                            )
                        res = yield [ah]
                    ah = wave_sum(
                        fx.Float32(res[0] if isinstance(res, (list, tuple)) else res)
                    )
                    if lane == 0:
                        lds_st(xs, MF_RS + NT + t, _rsq(ah * (1.0 / HC_DIM) + EPS))
            gpu.barrier()
            is_g = bid < LY.OUT_RG
            rg = fx.min(bid, LY.OUT_RG - 1)
            NKC = HIDDEN // 64
            NQP = MF_KC // 32  # 32-K sub-chunks per phase
            QPW = (NQP + WAVES - 1) // WAVES
            r_fe = rsrc_if(ptr(GI, LY.G_MTP_FCE), is_g, HIDDEN * HIDDEN * 2)
            r_fh = rsrc_if(ptr(GI, LY.G_MTP_FCH), is_g, HIDDEN * HIDDEN * 2)
            acc_e = zero_acc()
            acc_h = [fx.Float32(0.0)] * (4 * NCTH)
            NQC = MF_KC // 4  # staged quads per row per phase
            NIT = (MF_ROWS * NQC + THREADS - 1) // THREADS
            for ph in range_constexpr(MF_KP):
                k0 = ph * MF_KC
                if const_expr(ph > 0):
                    _pin(acc_e + acc_h)
                    gpu.barrier()
                # normalized B rows of this K phase -> LDS (bf16): embedding rows, then (row, branch) hidden rows
                NE_ = NT * NQC
                for i in range_constexpr((NE_ + THREADS - 1) // THREADS):
                    q = fx.min(tid + i * THREADS, NE_ - 1)
                    row = q // NQC
                    kq = q % NQC
                    k = k0 + 4 * kq
                    v = emb_quad(row, k // 4)
                    w = (
                        fx.Vector(
                            bo.buffer_load(r_pe, k // 2, vec_width=2, dtype=T.i32)
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    re_ = lds_ld(xs, MF_RS + row)
                    o = [
                        bf_round(v[e] * re_ * (fx.Float32(1.0) + w[e]))
                        for e in range(4)
                    ]
                    if tid + i * THREADS < NE_:
                        lds_st(xs, MF_B + row * S_MF + 2 * kq, bf16_pair(o[0], o[1]))
                        lds_st(
                            xs, MF_B + row * S_MF + 2 * kq + 1, bf16_pair(o[2], o[3])
                        )
                NH_ = 4 * NT * NQC
                for i in range_constexpr((NH_ + THREADS - 1) // THREADS):
                    q = fx.min(tid + i * THREADS, NH_ - 1)
                    c = q // NQC
                    kq = q % NQC
                    t = c // 4
                    kk = (c % 4) * HIDDEN + k0 + 4 * kq
                    v = hin_quad(t, kk // 4)
                    w = (
                        fx.Vector(
                            bo.buffer_load(r_ph, kk // 2, vec_width=2, dtype=T.i32)
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    rh_ = lds_ld(xs, MF_RS + NT + t)
                    o = [
                        bf_round(v[e] * rh_ * (fx.Float32(1.0) + w[e]))
                        for e in range(4)
                    ]
                    if tid + i * THREADS < NH_:
                        lds_st(
                            xs, MF_B + (NT + c) * S_MF + 2 * kq, bf16_pair(o[0], o[1])
                        )
                        lds_st(
                            xs,
                            MF_B + (NT + c) * S_MF + 2 * kq + 1,
                            bf16_pair(o[2], o[3]),
                        )
                    if const_expr(i % 4 == 3):
                        _pin(
                            acc_e + acc_h, mem=True
                        )  # bounds the staging loads in flight
                gpu.barrier()
                # every CTA runs the MMAs (CTAs >= 160 read zero-size weight descriptors: no traffic)
                units_e, units_h = [], []
                for j in range_constexpr(QPW):
                    ql = wave + WAVES * j
                    qlc = fx.min(ql, NQP - 1)
                    gq = ph * NQP + qlc
                    cf = (ql < NQP).select(fx.Float32(1.0), fx.Float32(0.0))
                    units_e.append(
                        unit_bf16(
                            r_fe,
                            rg,
                            0,
                            NKC,
                            lambda ct, ph=ph: (
                                MF_B
                                + fx.min(lane % 16 + 16 * ct, NT - 1) * S_MF
                                - ph * NQP * 16
                                + (lane // 16) * 4
                            ),
                            coef=cf,
                            subs=(gq,),
                        )
                    )
                    units_h.append(
                        unit_bf16(
                            r_fh,
                            rg,
                            0,
                            NKC,
                            lambda ct, ph=ph: (
                                MF_B
                                + (NT + fx.min(lane % 16 + 16 * ct, 4 * NT - 1)) * S_MF
                                - ph * NQP * 16
                                + (lane // 16) * 4
                            ),
                            coef=cf,
                            subs=(gq,),
                        )
                    )
                acc_e = mma_units(acc_e, units_e)
                acc_h = mma_units(acc_h, units_h, NCTH)
            _pin(acc_e + acc_h)
            gpu.barrier()
            stamp("mtp.gemm")
            for ct in range_constexpr(NCTE):
                fx.ptr_store(
                    fx.Vector.from_elements(acc_e[4 * ct : 4 * ct + 4], fx.Float32),
                    red + ((ct * WAVES + wave) * 64 + lane) * 4,
                )
            for ct in range_constexpr(NCTH):
                fx.ptr_store(
                    fx.Vector.from_elements(acc_h[4 * ct : 4 * ct + 4], fx.Float32),
                    red + (((NCTE + ct) * WAVES + wave) * 64 + lane) * 4,
                )
            gpu.barrier()

            def red_col(slot, r, c):
                tot = fx.Float32(0.0)
                for w_ in range_constexpr(WAVES):
                    tot = tot + lds_ld(
                        red, ((slot * WAVES + w_) * 64 + 16 * (r // 4) + c) * 4 + r % 4
                    )
                return tot

            if is_g:
                for q0 in range_constexpr(0, 32 * NT, THREADS):
                    q = tid + q0
                    if q < 32 * NT:
                        t = q // 32
                        b = (q % 32) // 8
                        p = q % 8
                        ch = 4 * t + b
                        o = []
                        for e in range_constexpr(2):
                            r = 2 * p + e
                            ye = bf_round(red_col(t // 16, r, t % 16))
                            yh = bf_round(red_col(NCTE + ch // 16, r, ch % 16))
                            o.append(bf_round(ye + yh))
                        put_bf(
                            mb(0, "mtp_h"),
                            t * (HC_DIM // 2) + (b * HIDDEN + rg * 16) // 2 + p,
                            o[0],
                            o[1],
                        )
                        ss = o[0] * o[0] + o[1] * o[1]
                        for off in (4, 2, 1):
                            ss = _xred(ss, off, lambda a, c: a + c)
                        if p == 0:
                            put(mb(0, "mtp_ssq"), (t * 4 + b) * LY.NPART + rg, ss)
            gpu.barrier()

        # ---------------------------------------------- layer program
        # per-request tables (every CTA): rope positions of the rows, req pool index / mamba slot / first position
        # of the requests, prefix sums of the index-score tiles and of the selection tasks
        r_pos = _rsrc(ptr(GI, LY.G_POS))
        if tid < 3 * NT:
            a_ = tid // NT
            t_ = tid % NT
            lds_st(
                xi,
                POS3 + tid,
                fx.Int32(
                    bo.buffer_load(
                        r_pos, (a_ * pos_stride + t_) * 2, vec_width=1, dtype=T.i32
                    )
                ),
            )
        if tid < R:
            rq_ = fx.Int32(
                bo.buffer_load(
                    _rsrc(ptr(GI, LY.G_REQ)), tid * 2, vec_width=1, dtype=T.i32
                )
            )
            lds_st(xi, RQ_REQ + tid, rq_)
            lds_st(
                xi,
                RQ_MS + tid,
                fx.Int32(
                    bo.buffer_load(
                        _rsrc(ptr(GI, LY.G_MAMBA_MAP)),
                        rq_ * (2 if map_i64 else 1),
                        vec_width=1,
                        dtype=T.i32,
                    )
                ),
            )
            # logical position = text position
            lds_st(
                xi,
                RQ_P0 + tid,
                fx.Int32(bo.buffer_load(r_pos, tid * L * 2, vec_width=1, dtype=T.i32)),
            )
        gpu.barrier()
        if tid == 0:
            acc_ = fx.Int32(0)
            for r_ in range_constexpr(R):
                lds_st(xi, SC_PFX + r_, acc_)
                acc_ = acc_ + ((lds_ld(xi, RQ_P0 + r_) + L) // 4 + 15) // 16
            lds_st(xi, SC_PFX + R, acc_)
            acc_ = fx.Int32(0)
            for t_ in range_constexpr(NT):
                lds_st(xi, SL_PFX + t_, acc_)
                nb_ = (lds_ld(xi, RQ_P0 + t_ // L) + t_ % L + 1) // 4
                acc_ = acc_ + fx.max(
                    (nb_ + LY.SEG_BLOCKS - 1) // LY.SEG_BLOCKS, fx.Int32(1)
                )
            lds_st(xi, SL_PFX + NT, acc_)
        gpu.barrier()
        tid0 = fx.thread_idx.x
        stamp("start")
        if const_expr(first_src == "global"):
            src = ("global", h_in, "global", ptr(GI, LY.G_HIN_SSQ), LY.NPART)
        if const_expr(first_src == "mtp"):
            if tid < NT:
                lds_st(
                    xi,
                    EMB_TOK + tid,
                    fx.Int32(
                        bo.buffer_load(
                            _rsrc(ptr(GI, LY.G_INPUT_IDS)),
                            tid * 2,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
            gpu.barrier()
            stage_mtp_fusion()
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            src = ("mb", mb(0, "mtp_h"), "mb", mb(0, "mtp_ssq"), LY.NPART)
        if const_expr(prelude):
            if tid < NT:
                lds_st(
                    xi,
                    EMB_TOK + tid,
                    fx.Int32(
                        bo.buffer_load(
                            _rsrc(ptr(GI, LY.G_INPUT_IDS)),
                            tid * 2,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
            gpu.barrier()
            src = ("embed", ptr(GI, LY.G_EMBED_W))
            stage_prelude()
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
        gpu.barrier()
        if const_expr(ple_first):
            stage_ple_proj(0)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp("ple_proj")
            stage_ple_apply(0, src)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp("ple_apply")
            src = ("mb", mb(0, "ple_h"), "mb", mb(0, "ple_ssq"), LY.PLE_PARTS)
        for li in range_constexpr(n_layers):
            kind = plan[li]
            stage_mix(
                li, "a", src, LY.P_HCA_NORM, LY.P_HCA_DOWN, LY.P_HCA_UP, LY.P_HCA_INJ
            )
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.mixA")
            if const_expr(kind == "gdn"):
                stage_in_proj(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.in_proj")
                stage_core(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.core")
            else:
                stage_qsa_proj(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_proj")
                stage_qsa_prep(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_prep")
                stage_qsa_scores(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_scores")
                stage_qsa_select(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_select")
                stage_qsa_merge_select(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_merge_sel")
                stage_qsa_attn(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_attn")
                stage_qsa_merge(li)
                tid = _opaque_i32(tid0)
                lane = tid % 64
                wave = tid // 64
                stamp(f"L{li}.qsa_merge")
            stage_out_proj(li, src, kind)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.out_proj")
            h1 = ("mb", mb(li, "h1"), "mb", mb(li, "h1ssq"), LY.NPART)
            stage_mix(
                li, "m", h1, LY.P_HCM_NORM, LY.P_HCM_DOWN, LY.P_HCM_UP, LY.P_HCM_INJ
            )
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.mixM")
            stage_router(li)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.router")
            stage_gate_up(li)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.gate_up")
            stage_down(li, h1, li == n_layers - 1)
            tid = _opaque_i32(tid0)
            lane = tid % 64
            wave = tid // 64
            stamp(f"L{li}.down")
            src = ("mb", mb(li, "h2"), "mb", mb(li, "h2ssq"), LY.NPART)
        if const_expr(final_mix):
            # model-level hyper-connection mixer (no combine): row n_layers + 1 holds its weights
            stage_mix(
                n_layers + 1,
                "a",
                src,
                LY.P_HCA_NORM,
                LY.P_HCA_DOWN,
                LY.P_HCA_UP,
                LY.P_HCA_INJ,
                x_out=ptr(GI, LY.G_FIN_OUT),
            )
            stamp("final_mix")

    @flyc.jit
    def launch(
        h_in: Int64,
        h_out: Int64,
        ptab: Int64,
        scratch: Int64,
        step: Int64,
        launch_id: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        qwen4_exp_monokernel(h_in, h_out, ptab, scratch, step, launch_id).launch(
            grid=(G,), block=(THREADS,), stream=stream
        )

    return launch
