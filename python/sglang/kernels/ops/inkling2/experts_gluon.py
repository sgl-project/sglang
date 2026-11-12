"""NVFP4 and MXFP8 expert GEMMs for decode and verify sizes on gfx950, in Gluon, on
weights stored in MFMA order: each lane loads its B-operand values with 16-byte loads
and dequantizes them in registers, and only the activations go through LDS. At BM 16
and BK 256 with the same K split the result is bitwise _grouped_dequant_gemm's (the
same FP16 / BF16 products and the same chain of 16x16x32 MFMAs).

MFMA order, per expert and per 16-row group of the weights' 2-value units (NVFP4
bytes, MXFP8 16-bit pairs; K // 2 per row): [K // 128][lane group g][row][32-K
block a][4 units], so lane 16 * g + row reads one 16-unit run per 128 K: the 4
units (8 values) of each 32-K block that the 16x16x32 B operand gives its lane
group. The NVFP4 (N, K // 16) scales are stored as [K // 256][16-K half][row][32-K
block], one 8-byte run per lane and K tile; MXFP8's row-major (N, K // 32) scales
already are."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BM, BN, BK = 16, 64, 256


@gluon.jit
def _cvt_fp4(p, scale):
    """v_cvt_scalef32_pk_f16_fp4 on each lane: two e2m1 values -> (low, high) FP16.
    It applies only the exponent of its FP32 scale, so the e4m3 scale is applied
    after, in FP16 (exact: the NVFP4 values lie in [2^-10, 2688])."""
    packed = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f16_fp4 $0, $1, $2",
        "=v,v,v",
        [p, scale],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    low = (packed & 0xFFFF).to(gl.int16).to(gl.float16, bitcast=True)
    high = (packed >> 16).to(gl.int16).to(gl.float16, bitcast=True)
    return low, high


@gluon.jit
def _dequant(
    p,
    s,
    LP: gl.constexpr,
    LB: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
):
    """(BK // 2, BN) packed bytes in LP and [column group, 16-K half, row, 32-K block]
    scales -> the (BK, BN) FP16 B operand in LB, without moving data between lanes."""
    s = s.to(gl.float16)
    s = gl.join(s, s)
    s = gl.join(s, s)  # one scale per packed byte: [.., 32-K block, 2, 2]
    s = gl.reshape(gl.permute(s, (3, 1, 4, 5, 0, 2)), [BK // 2, BN])
    s = gl.convert_layout(s, LP)
    one = gl.full(p.shape, 1.0, gl.float32, layout=LP)
    low, high = _cvt_fp4(p.to(gl.int32), one)
    w = gl.reshape(gl.permute(gl.join(low * s, high * s), (0, 2, 1)), [BK, BN])
    return gl.convert_layout(w, LB, assert_trivial=True)


@gluon.jit
def _dequant_mxfp8(
    p, s, LP: gl.constexpr, LB: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr
):
    """_dequant for MXFP8: (BK // 2, BN) e4m3 pairs in LP and their ue8m0 scales ->
    the (BK, BN) BF16 B operand.
    v_cvt_scalef32_pk_bf16_fp8 applies the exponent of its FP32 scale: exactly the
    UE8M0 scale."""
    s = gl.where(s == 0, 2.0**-127, (s.to(gl.int32) << 23).to(gl.float32, bitcast=True))
    s = gl.join(s, s)
    s = gl.join(s, s)  # one scale per pair: [.., 32-K block, 2, 2]
    s = gl.reshape(gl.permute(s, (3, 1, 4, 5, 0, 2)), [BK // 2, BN])
    s = gl.convert_layout(s, LP)
    packed = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2",
        "=v,v,v",
        [p.to(gl.int32) & 0xFFFF, s],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    low = (packed & 0xFFFF).to(gl.int16).to(gl.bfloat16, bitcast=True)
    high = (packed >> 16).to(gl.int16).to(gl.bfloat16, bitcast=True)
    w = gl.reshape(gl.permute(gl.join(low, high), (0, 2, 1)), [BK, BN])
    return gl.convert_layout(w, LB, assert_trivial=True)


@gluon.jit
def _units_tile(t, LP: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    """The loaded [group, half, lane group * 16 + row, block * 4 + unit] weight tile
    as the (BK // 2, BN) units in LP."""
    p = gl.reshape(t, [BN // 16, 2, 4, 16, 4, 4])
    p = gl.reshape(gl.permute(p, (1, 4, 2, 5, 0, 3)), [BK // 2, BN])
    return gl.convert_layout(p, LP)


@gluon.jit
def _b_operand(
    p,
    s,
    LP: gl.constexpr,
    LB: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    NVFP4: gl.constexpr,
):
    if NVFP4:
        b = _dequant(p, s, LP, LB, BN, BK)
    else:
        b = _dequant_mxfp8(p, s, LP, LB, BN, BK)
    return b


@gluon.jit
def _grouped_gemm(
    X,
    W,
    WS,
    Y,
    Offsets,
    Dst2Src,
    Schedule,
    rows,
    N,
    K: gl.constexpr,
    GATHER_DIV: gl.constexpr,
    K_SPLIT: gl.constexpr,
    NVFP4: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
):
    """_grouped_dequant_gemm (BM 16, BN 64, BK 256) on MFMA-order weights; W is
    MXFP8's as int16 pairs."""
    gl.static_assert((BM == 16) & (BN == 64) & (BK == 256))
    entry = gl.load(Schedule + gl.program_id(0))
    if entry == -1:
        return
    expert = (entry & 0xFFFF).to(gl.int64)
    block = entry >> 16
    lo = gl.load(Offsets + expert)
    hi = gl.load(Offsets + expert + 1)
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    LA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    LB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    # LB with K halved: the 2-value units each lane converts into its B operand.
    LP: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [16, 0], [32, 0], [64, 0]],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        warp_bases=[[0, 16], [0, 32]],
        block_bases=[],
        shape=[BK // 2, BN],
    )
    # Weight tile as [column group, 128-K half, lane, 16 units]: one warp per group.
    LW: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 1, 0, 0]],
        lane_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 32, 0]],
        warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
        block_bases=[],
        shape=[BN // 16, 2, 64, 16],
    )  # fmt: skip
    # Scales as [column group, lane group, row, 32-K block]; NVFP4 lane groups g and
    # g ^ 1 read the same 8 bytes (their 16-K half), MXFP8's all read their row's.
    LS: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4]],
        lane_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 1, 0, 0], [0, 2, 0, 0]],
        warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
        block_bases=[],
        shape=[BN // 16, 4, 16, 8],
    )  # fmt: skip
    LX: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[2, 32], warps_per_cta=[4, 1], order=[1, 0]
    )  # fmt: skip
    SX: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    n0 = gl.program_id(1) * BN
    split = gl.program_id(2)
    W = W + expert * N * (K // 2)
    WS = WS + expert * N * (K // 16 if NVFP4 else K // 32)

    w_g = gl.arange(
        0, BN // 16, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, LW)))
    )
    w_h = gl.arange(
        0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, LW)))
    )
    w_l = gl.arange(
        0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, LW)))
    )
    w_b = gl.arange(
        0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, LW)))
    )
    w_off = (
        gl.expand_dims(gl.expand_dims(gl.expand_dims(n0 // 16 + w_g, 1), 2), 3) * (K // 2) * 16
        + gl.expand_dims(gl.expand_dims(gl.expand_dims(w_h, 0), 2), 3) * 1024
        + gl.expand_dims(gl.expand_dims(gl.expand_dims(w_l, 0), 1), 3) * 16
        + gl.expand_dims(gl.expand_dims(gl.expand_dims(w_b, 0), 1), 2)
    )  # fmt: skip
    s_g = gl.arange(
        0, BN // 16, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, LS)))
    )
    s_q = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, LS)))
    )
    s_r = gl.arange(
        0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, LS)))
    )
    s_b = gl.arange(
        0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, LS)))
    )
    if NVFP4:
        s_off = (
            gl.expand_dims(gl.expand_dims(gl.expand_dims(n0 // 16 + s_g, 1), 2), 3) * K
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_q // 2, 0), 2), 3) * 128
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_r, 0), 1), 3) * 8
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_b, 0), 1), 2)
        )  # fmt: skip
    else:  # ue8m0 scales row-major: every lane group reads its row's 8 bytes
        s_off = (
            gl.expand_dims(gl.expand_dims(gl.expand_dims(n0 // 16 + s_g, 1), 2), 3) * 16 * (K // 32)
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_q * 0, 0), 2), 3)
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_r, 0), 1), 3) * (K // 32)
            + gl.expand_dims(gl.expand_dims(gl.expand_dims(s_b, 0), 1), 2)
        )  # fmt: skip
    xm = gl.arange(0, BM, layout=gl.SliceLayout(1, LX))
    xk = gl.arange(0, BK, layout=gl.SliceLayout(0, LX))
    x_pos = lo + block * BM + xm
    x_mask = x_pos < hi
    x_rows = (
        gl.amd.cdna4.buffer_load(Dst2Src, x_pos, mask=x_mask, other=0) // GATHER_DIV
    )
    acc = gl.zeros([BM, BN], gl.float32, layout=mfma)
    xs = gl.allocate_shared_memory(X.dtype.element_ty, [BM, BK], layout=SX)
    for k0 in range(split * K_SPLIT, (split + 1) * K_SPLIT, BK):
        t = gl.amd.cdna4.buffer_load(W, w_off + (k0 // 128) * 1024)
        s = gl.amd.cdna4.buffer_load(WS, s_off + (k0 if NVFP4 else k0 // 32))
        x = gl.amd.cdna4.buffer_load(
            X,
            x_rows[:, None] * K + (k0 + xk)[None, :],
            mask=x_mask[:, None],
            other=0.0,
        )
        p = _units_tile(t, LP, BN, BK)
        gl.barrier()
        xs.store(x)
        gl.barrier()
        # long K loops read x from LDS before the dequant, whose ALU work hides
        # the latency; a split K's few tiles read it after
        if K_SPLIT // BK > 4:
            a = xs.load(LA)
            b = _b_operand(p, s, LP, LB, BN, BK, NVFP4)
        else:
            b = _b_operand(p, s, LP, LB, BN, BK, NVFP4)
            a = xs.load(LA)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    cm = gl.arange(0, BM, layout=gl.SliceLayout(1, mfma))
    cn = gl.arange(0, BN, layout=gl.SliceLayout(0, mfma))
    c_pos = lo + block * BM + cm
    c_mask = c_pos < hi
    src = gl.amd.cdna4.buffer_load(Dst2Src, c_pos, mask=c_mask, other=0)
    # With a split K the partials go to slab `split` of Y, reduced in split order.
    gl.amd.cdna4.buffer_store(
        acc.to(Y.dtype.element_ty),
        Y + split.to(gl.int64) * rows * N,
        src[:, None] * N + (n0 + cn)[None, :],
        mask=c_mask[:, None],
    )


def fits(n, k):
    """Whether (n, k) expert weights can take the MFMA order."""
    return n % BN == 0 and k % BK == 0


def mfma_order_weights(weight):
    """(experts, n, k // 2) packed NVFP4 or (experts, n, k) e4m3 MXFP8 weights in
    MFMA order (same shape): one permutation of their 2-value units (bytes, pairs)."""
    units = weight.view(torch.int16) if weight.dtype != torch.uint8 else weight
    e, n, kb = units.shape
    units = units.view(e * n // 16, 16, kb // 64, 4, 4, 4)
    return units.permute(0, 2, 4, 1, 3, 5).reshape(e, n, kb).view(weight.dtype)


def mfma_order_scales(scales):
    """(experts, n, k // 16) e4m3 block scales in MFMA order (same shape); MXFP8's
    row-major ue8m0 scales already give each lane its 8 bytes."""
    if scales.dtype == torch.uint8:
        return scales
    e, n, k16 = scales.shape
    raw = scales.view(torch.uint8).view(e * n // 16, 16, k16 // 16, 8, 2)
    return raw.permute(0, 2, 4, 1, 3).reshape(e, n, k16).view(scales.dtype)


def grouped_gemm(x, weight, scales, partials, route, rows, gather_div, splits, blocks):
    """_grouped_gemm into partials ((splits, rows, n), or (rows, n) unsplit)."""
    n, k = weight.shape[1], x.shape[1]
    assert n % BN == 0 and k % (BK * splits) == 0, (n, k, splits)
    nvfp4 = weight.dtype == torch.uint8
    _grouped_gemm[(blocks, n // BN, splits)](
        x,
        weight if nvfp4 else weight.view(torch.int16),
        scales,
        partials,
        route.offsets,
        route.dst2src,
        route.schedule,
        rows,
        n,
        k,
        gather_div,
        k // splits,
        nvfp4,
        BM,
        BN,
        BK,
        num_warps=4,
    )
