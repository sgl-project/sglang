"""Fused Qwen gated-residual HC (hc_per_branch_norm, HC=4) for decode-sized M on gfx950.

Extends AITER's FlyDSL ``hc_mix`` (down split-K -> reduce + SiLU -> up + gate + branch
mean) so an HC site needs no separate RMSNorm and its combine needs no GEMV:

* The per-branch Gemma RMSNorm is folded into the down projection. ``(1 + w)`` is folded
  into the packed down / inject weights, the down kernel multiplies the raw residual
  ``H``, and the reduce scales each split-K partial by its branch's
  ``rsqrt(mean(H_b^2) + eps)`` (a split never straddles two branches). The up kernel
  rebuilds the normed gate operand as ``H * s_b * (1 + w)`` in FP32.
* The combine's inject projection ``normed @ W_inj^T`` rides along as extra down rows,
  so the reduce also returns the inject logits ``G`` and the combine is elementwise.
* The per-branch sums of squares come either from the combine that produced ``H``
  (``hc_apply_inject`` returns them) or, when ``H`` has another origin, from extra
  down-kernel CTAs.

Numerics differ from the unfused path only by rounding points: the normed input is no
longer rounded to BF16 before the GEMMs, and ``(1 + w)`` is rounded into the weights.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl

# TODO(aiter#5993): the down / reduce / up kernel structure, the helpers and the weight
# packing are adapted from AITER's FlyDSL hc_mix (ROCm/aiter#5993, not in the aiter pin);
# import its shared pieces once it is.
_HC = 4
# Rows in a 16x16x32 MFMA tile.
_MFMA_M = 16
# Elements of one packed 16x32 B fragment (see _pack): 64 lanes x 8 BF16.
_FRAG = 16 * 32
# The up kernel stages its A operand in LDS; gfx950 has 160 KiB of LDS per CU.
_LDS_ROW_PAD = 8
_LDS_BYTES = 160 * 1024
_REDUCE_THREADS = 256
# B fragments in flight per wave.
_PREFETCH = 16
# Measured on Qwen K=10240, R=320 with per-layer (cold) weights at M=16 and 64: split-K 32
# fills the GPU with single-wave down CTAs; 4 waves per up CTA.
_SPLIT_K = 32
_UP_WAVES = 4
# Columns per combine wave (64 lanes x 8); HS splits into HS / _INJECT_BLOCK partial
# sums of squares per branch.
_INJECT_BLOCK = 512
# The up kernel stages up to this many rows of its [M, R] operand in LDS.
_MAX_ROWS = 128


def _bucket_rows(m):
    """M is a Constexpr, so each distinct row count is a compile. Round up to a power of
    two within the first MFMA tile, then to whole tiles: the kernels already compute
    whole tiles, so padding adds no MFMA work and M = 1 stays exact."""
    if m <= _MFMA_M:
        return 1 << (m - 1).bit_length()
    return -(-m // _MFMA_M) * _MFMA_M


HC_FUSED_ROW_BUCKETS = tuple(sorted({_bucket_rows(m) for m in range(1, _MAX_ROWS + 1)}))

# Calling a JitFunction re-traces it to MLIR on every call, which costs far more than the
# kernels outside CUDA graphs; dispatch a CompiledFunction per Constexpr set instead.
_compiled = {}


def _dispatch(launcher, constexprs, *args):
    key = (launcher, constexprs)
    compiled = _compiled.get(key)
    if compiled is not None:
        compiled(*args)
        return
    # flyc.compile also runs the launch. Tensor layouts stay dynamic, so one compile
    # serves every row count that shares these Constexprs.
    compiled = flyc.compile(launcher, *args)
    if compiled is not None:
        _compiled[key] = compiled


def _load(tensor, offset, width, dtype):
    """Buffer load of `width` elements at `offset`; OOB reads return zero."""
    if width == 8:
        kind = fx.rocdl.BufferCopy128b()
    elif dtype == fx.Float32:
        kind = fx.rocdl.BufferCopy32b()
    else:
        kind = fx.rocdl.BufferCopy16b()
    view = fx.make_view(
        fx.get_iter(tensor) + fx.Int32(offset), fx.make_layout(width, 1)
    )
    frag = fx.make_rmem_tensor(width, dtype)
    fx.copy(fx.make_copy_atom(kind, dtype), view, frag)
    return fx.Vector(frag.load())


def _store(tensor, offset, value, dtype):
    view = fx.make_view(fx.get_iter(tensor) + fx.Int32(offset), fx.make_layout(1, 1))
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy32b() if dtype == fx.Float32 else fx.rocdl.BufferCopy16b(),
        dtype,
    )
    frag = fx.make_rmem_tensor(1, dtype)
    frag.store(fx.Vector.from_elements([value], dtype))
    fx.copy(atom, frag, view)


def _store8(tensor, offset, vec):
    """128-bit store of 8 BF16 at `offset`."""
    view = fx.make_view(fx.get_iter(tensor) + fx.Int32(offset), fx.make_layout(8, 1))
    frag = fx.make_rmem_tensor(8, fx.BFloat16)
    frag.store(vec)
    fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), frag, view)


def _mfma(av, bv, acc):
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_bf16(
            acc.ir_value().type,
            [av.ir_value(), bv.ir_value(), acc.ir_value(), 0, 0, 0],
        )
    )


def _sigmoid(v):
    return fx.Float32(1.0) / (fx.Float32(1.0) + math.exp(-v))


@functools.cache
def _up_shared(rows, rpad):
    @fx.struct
    class UpShared:
        t: fx.Array[fx.Int32, rows * rpad // 2, 16]

    return UpShared


@flyc.kernel
def _hc_down_fused(
    A: fx.Tensor,
    W: fx.Tensor,
    P: fx.Tensor,
    SS: fx.Tensor,
    M: fx.Constexpr[int],
    R: fx.Constexpr[int],
    K: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    SS_OUT: fx.Constexpr[bool],
):
    """H[M,K] against packed down[R,K] -> P[SPLIT,M,R] FP32 split-K partials.

    One wave per CTA; grid (R / 16 [+ row tiles], SPLIT). With SS_OUT, CTA R / 16 + t
    also writes row tile t's sums of squares over the split's columns, SS[M, SPLIT].
    """
    lane = gpu.thread_id("x") % 64
    nt = gpu.block_id("x")
    sp = gpu.block_id("y")
    a = fx.rocdl.make_buffer_tensor(A, max_size=False)
    w = fx.rocdl.make_buffer_tensor(W, max_size=False)
    out = fx.rocdl.make_buffer_tensor(P, max_size=False)
    row_a = lane % 16
    mt = (M + _MFMA_M - 1) // _MFMA_M
    steps = K // SPLIT // 32
    k0 = sp * steps

    def a_offset(row, kcol):
        # Descriptor-bound OOB loads return zero for the padded M rows.
        return (row < M).select(fx.Int32(row * K + kcol), fx.Int32(M * K))

    if nt < R // 16:
        accs = [fx.Vector.filled(4, 0.0, fx.Float32)] * mt
        for g in range_constexpr(0, steps, _PREFETCH):
            bvs = []
            for ki in range_constexpr(g, min(g + _PREFETCH, steps)):
                bvs.append(
                    _load(
                        w, (nt * (K // 32) + k0 + ki) * _FRAG + lane * 8, 8, fx.BFloat16
                    )
                )
            for j in range_constexpr(len(bvs)):
                kcol = (k0 + g + j) * 32 + lane // 16 * 8
                for t in range_constexpr(mt):
                    row = t * _MFMA_M + row_a
                    av = _load(a, a_offset(row, kcol), 8, fx.BFloat16)
                    accs[t] = _mfma(av, bvs[j], accs[t])
        for t in range_constexpr(mt):
            for i in range_constexpr(4):
                row = t * _MFMA_M + lane // 16 * 4 + i
                col = nt * 16 + lane % 16
                if row < M:
                    _store(out, (sp * M + row) * R + col, accs[t][i], fx.Float32)
    if const_expr(SS_OUT):
        ss = fx.rocdl.make_buffer_tensor(SS, max_size=False)
        for t in range_constexpr(mt):
            if nt == R // 16 + t:
                row = t * _MFMA_M + row_a
                sq = fx.Float32(0.0)
                for k in range_constexpr(steps):
                    kcol = (k0 + k) * 32 + lane // 16 * 8
                    af = _load(a, a_offset(row, kcol), 8, fx.BFloat16).to(fx.Float32)
                    for e in range_constexpr(8):
                        sq = sq + af[e] * af[e]
                # The 4 lanes holding a row's four 8-column groups are 16 apart.
                for shift in range_constexpr(2):
                    peer = gpu.shuffle(
                        sq, fx.Int32(16 << shift), fx.Int32(64), mode="xor"
                    )
                    sq = sq + fx.Float32(peer)
                if (lane < 16) & (row < M):
                    _store(ss, row * SPLIT + sp, sq, fx.Float32)


@flyc.kernel
def _hc_reduce_fused(
    P: fx.Tensor,
    SS: fx.Tensor,
    T: fx.Tensor,
    G: fx.Tensor,
    SC: fx.Tensor,
    M: fx.Constexpr[int],
    R: fx.Constexpr[int],
    RL: fx.Constexpr[int],
    NI: fx.Constexpr[int],
    K: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    NP: fx.Constexpr[int],
    EPS: fx.Constexpr[float],
):
    """Scale each split by its branch's rstd and sum: SiLU(down / HC) -> T[M,RL] BF16,
    inject logits -> G[M,NI] FP32, branch scales -> SC[M,HC] FP32.

    SS[M, HC * NP] holds NP partial sums of squares per branch.
    """
    i = gpu.block_id("x") * _REDUCE_THREADS + gpu.thread_id("x")
    p = fx.rocdl.make_buffer_tensor(P, max_size=False)
    ss = fx.rocdl.make_buffer_tensor(SS, max_size=False)
    t = fx.rocdl.make_buffer_tensor(T, max_size=False)
    g = fx.rocdl.make_buffer_tensor(G, max_size=False)
    sc = fx.rocdl.make_buffer_tensor(SC, max_size=False)
    row = i // R
    c = i % R
    per = SPLIT // _HC
    acc = fx.Float32(0.0)
    for b in range_constexpr(_HC):
        total = fx.Float32(0.0)
        for j in range_constexpr(NP):
            total = total + _load(ss, row * (_HC * NP) + (b * NP + j), 1, fx.Float32)[0]
        scale = fx.Float32(1.0) / math.sqrt(
            total * fx.Float32(_HC / K) + fx.Float32(EPS)
        )
        part = fx.Float32(0.0)
        for s in range_constexpr(b * per, (b + 1) * per):
            part = part + _load(p, fx.Int32(s * M * R) + i, 1, fx.Float32)[0]
        acc = acc + part * scale
        if (c == b) & (i < M * R):
            _store(sc, row * _HC + b, scale, fx.Float32)
    v = acc * fx.Float32(1.0 / _HC)
    v = v * _sigmoid(v)
    if (i < M * R) & (c < RL):
        _store(t, row * RL + c, v.to(fx.BFloat16), fx.BFloat16)
    if const_expr(NI > 0):
        if (i < M * R) & (c >= RL) & (c < RL + NI):
            _store(g, row * NI + c - RL, acc, fx.Float32)


@flyc.kernel
def _hc_up_fused(
    A: fx.Tensor,
    W: fx.Tensor,
    X: fx.Tensor,
    SC: fx.Tensor,
    W1: fx.Tensor,
    O: fx.Tensor,
    M: fx.Constexpr[int],
    N: fx.Constexpr[int],
    R: fx.Constexpr[int],
):
    """T[M,R] against interleaved up[N,R]; gate sigmoid * (H * s_b * (1 + w)), mean over HC."""
    tid = gpu.thread_id("x")
    lane = tid % 64
    nt = gpu.block_id("x") * _UP_WAVES + tid // 64
    a = fx.rocdl.make_buffer_tensor(A, max_size=False)
    w = fx.rocdl.make_buffer_tensor(W, max_size=False)
    x = fx.rocdl.make_buffer_tensor(X, max_size=False)
    sc = fx.rocdl.make_buffer_tensor(SC, max_size=False)
    w1 = fx.rocdl.make_buffer_tensor(W1, max_size=False)
    out = fx.rocdl.make_buffer_tensor(O, max_size=False)
    mt = (M + _MFMA_M - 1) // _MFMA_M

    bvs = []
    for ki in range_constexpr(min(_PREFETCH, R // 32)):
        bvs.append(_load(w, (nt * (R // 32) + ki) * _FRAG + lane * 8, 8, fx.BFloat16))
    col = nt * 16 + lane % 16
    # Interleaved up rows: col % HC is the branch, col // HC the hidden unit.
    n = (col % _HC) * (N // _HC) + col // _HC
    wn = _load(w1, n, 1, fx.Float32)[0]
    xvs = []
    for t in range_constexpr(mt):
        for i in range_constexpr(4):
            row = t * _MFMA_M + lane // 16 * 4 + i
            # Rows >= M sit past the end of X / SC, so these loads return zero.
            hv = _load(x, row * N + n, 1, fx.BFloat16)[0].to(fx.Float32)
            sv = _load(sc, row * _HC + col % _HC, 1, fx.Float32)[0]
            xvs.append(hv * sv * wn)

    # Stage A in LDS; rows >= M sit past the end of A, so their loads return zero.
    rows = mt * _MFMA_M
    rpad = R + _LDS_ROW_PAD
    threads = 64 * _UP_WAVES
    lds = fx.SharedAllocator().allocate(_up_shared(rows, rpad)).peek()
    base = fx.Int32(fx.ptrtoint(lds.t.ptr))
    chunks = rows * (R // 8)
    vals = []
    for c0 in range_constexpr(0, chunks, threads):
        vals.append(_load(a, (fx.Int32(c0) + fx.Int32(tid)) * 8, 8, fx.BFloat16))
    for j in range_constexpr(len(vals)):
        c = fx.Int32(j * threads) + fx.Int32(tid)
        ptr = lds_typed_ptr(
            base,
            fx.BFloat16.ir_type,
            align=16,
            byte_offset=(c + c // (R // 8)) * fx.Int32(16),
        )
        if const_expr((j + 1) * threads <= chunks):
            fx.ptr_store(vals[j].ir_value(), ptr)
        elif c < chunks:
            fx.ptr_store(vals[j].ir_value(), ptr)
    gpu.barrier()

    row_a = lane % 16
    steps = R // 32
    accs = [fx.Vector.filled(4, 0.0, fx.Float32)] * mt
    for g in range_constexpr(0, steps, _PREFETCH):
        if const_expr(g > 0):
            bvs = []
            for ki in range_constexpr(g, min(g + _PREFETCH, steps)):
                bvs.append(
                    _load(w, (nt * steps + ki) * _FRAG + lane * 8, 8, fx.BFloat16)
                )
        for j in range_constexpr(len(bvs)):
            ki = g + j
            for t in range_constexpr(mt):
                row = t * _MFMA_M + row_a
                av = fx.Vector(
                    lds_vec_load(
                        base,
                        fx.Int32(row * rpad + ki * 32 + lane // 16 * 8) * fx.Int32(2),
                        fx.Vector.make_type(8, fx.BFloat16),
                        fx.BFloat16,
                        align=16,
                    )
                )
                accs[t] = _mfma(av, bvs[j], accs[t])
    for t in range_constexpr(mt):
        for i in range_constexpr(4):
            row = t * _MFMA_M + lane // 16 * 4 + i
            value = xvs[t * 4 + i] * _sigmoid(accs[t][i])
            # Four adjacent lanes hold the four branches of one hidden unit.
            for shift in range_constexpr(2):
                peer = gpu.shuffle(
                    value, fx.Int32(1 << shift), fx.Int32(64), mode="xor"
                )
                value = value + fx.Float32(peer)
            if (row < M) & (lane % 4 == 0):
                _store(
                    out,
                    row * (N // _HC) + col // _HC,
                    (value * fx.Float32(1.0 / _HC)).to(fx.BFloat16),
                    fx.BFloat16,
                )


@flyc.jit
def _launch(
    H: fx.Tensor,
    D: fx.Tensor,
    U: fx.Tensor,
    W1: fx.Tensor,
    P: fx.Tensor,
    SS: fx.Tensor,
    T: fx.Tensor,
    G: fx.Tensor,
    SC: fx.Tensor,
    O: fx.Tensor,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    R: fx.Constexpr[int],
    RL: fx.Constexpr[int],
    NI: fx.Constexpr[int],
    S: fx.Constexpr[int],
    NP: fx.Constexpr[int],
    SS_OUT: fx.Constexpr[bool],
    EPS: fx.Constexpr[float],
    stream: fx.Stream,
):
    # With SS_OUT, one extra CTA per row tile computes the sums of squares.
    ss_ctas = (M + _MFMA_M - 1) // _MFMA_M if SS_OUT else 0
    _hc_down_fused(H, D, P, SS, M, R, K, S, SS_OUT).launch(
        grid=(R // 16 + ss_ctas, S), block=(64,), stream=stream
    )
    _hc_reduce_fused(P, SS, T, G, SC, M, R, RL, NI, K, S, NP, EPS).launch(
        grid=((M * R + _REDUCE_THREADS - 1) // _REDUCE_THREADS,),
        block=(_REDUCE_THREADS,),
        stream=stream,
    )
    _hc_up_fused(T, U, H, SC, W1, O, M, K, RL).launch(
        grid=(K // (16 * _UP_WAVES),), block=(64 * _UP_WAVES,), stream=stream
    )


def _pack(w):
    n, k = w.shape
    return w.reshape(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous()


class HcFusedWeights:
    """Packed weights of one HC site; build after weight loading."""

    def __init__(self, down, up, inject, norm_weight, eps):
        r, k = down.shape
        self.r, self.eps = r, float(eps)
        self.w1 = (1.0 + norm_weight.float()).contiguous()
        rows = [down.float() * self.w1]
        self.ni = 0
        if inject is not None:
            self.ni = inject.shape[0]
            rows.append(inject.float() * self.w1)
        d = torch.cat(rows)
        pad = -d.shape[0] % 16
        if pad:
            d = torch.cat([d, d.new_zeros(pad, k)])
        self.rp = d.shape[0]
        self.down = _pack(d.to(torch.bfloat16))
        interleaved = up.reshape(_HC, k // _HC, r).permute(1, 0, 2).reshape(k, r)
        self.up = _pack(interleaved.contiguous())


def hc_fused_shapes_supported(k, r):
    """Weight-shape half of the check; the per-call half is ``flydsl_hc_mix_supported``
    (bf16, contiguous, 1 to 128 rows)."""
    return (
        k % (32 * _SPLIT_K) == 0
        and k % (_HC * _INJECT_BLOCK) == 0
        and r % 32 == 0
        and _MAX_ROWS * (r + _LDS_ROW_PAD) * 2 <= _LDS_BYTES
    )


def hc_fused_mix(h, weights: HcFusedWeights, sumsq=None):
    """One HC site on residual h [M, K]: (mixed [M, K/HC] BF16, inject logits [M, NI] FP32
    or None). ``sumsq`` is the [M, HC * NP] partial sums of squares ``hc_apply_inject``
    returned for h; None makes the down kernel compute them."""
    m, k = h.shape
    # Buffers hold the padded rows; past m, h and sumsq read as zero.
    mb = _bucket_rows(m)
    dev = h.device
    f32 = torch.float32
    p = torch.empty((_SPLIT_K, mb, weights.rp), device=dev, dtype=f32)
    t = torch.empty((mb, weights.r), device=dev, dtype=h.dtype)
    g = torch.empty((mb, max(weights.ni, 1)), device=dev, dtype=f32)
    sc = torch.empty((mb, _HC), device=dev, dtype=f32)
    out = torch.empty((mb, k // _HC), device=dev, dtype=h.dtype)
    if sumsq is None:
        sumsq = torch.empty((mb, _SPLIT_K), device=dev, dtype=f32)
        np_, ss_out = _SPLIT_K // _HC, True
    else:
        if sumsq.shape != (m, k // _INJECT_BLOCK) or sumsq.dtype != f32:
            raise ValueError(
                f"sumsq must be FP32 [{m}, {k // _INJECT_BLOCK}], "
                f"got {sumsq.dtype} {list(sumsq.shape)}"
            )
        np_, ss_out = sumsq.shape[1] // _HC, False
    constexprs = (
        mb,
        k,
        weights.rp,
        weights.r,
        weights.ni,
        _SPLIT_K,
        np_,
        ss_out,
        weights.eps,
    )
    _dispatch(
        _launch, constexprs,
        h, weights.down, weights.up, weights.w1, p, sumsq, t, g, sc, out, *constexprs,
        torch.cuda.current_stream(dev),
    )  # fmt: skip
    return out[:m], (g[:m] if weights.ni else None)


@flyc.kernel
def _hc_inject_fused(
    H: fx.Tensor,
    Y: fx.Tensor,
    G: fx.Tensor,
    O: fx.Tensor,
    SS: fx.Tensor,
    K: fx.Constexpr[int],
    NI: fx.Constexpr[int],
):
    """O = H + Y * 2 sigmoid(G / HC) per branch, plus each wave's sum of squares of O.

    One wave per CTA owns _INJECT_BLOCK columns of one branch of one row: grid
    (M, HC * HS / _INJECT_BLOCK). SS[M, HC * NP] gets NP = HS / _INJECT_BLOCK partials
    per branch, the layout ``_hc_reduce_fused`` sums.
    """
    lane = gpu.thread_id("x") % 64
    row = gpu.block_id("x")
    part = gpu.block_id("y")
    hs = K // _HC
    nblk = hs // _INJECT_BLOCK
    branch = part // nblk
    col = (part % nblk) * _INJECT_BLOCK + lane * 8
    h = fx.rocdl.make_buffer_tensor(H, max_size=False)
    y = fx.rocdl.make_buffer_tensor(Y, max_size=False)
    g = fx.rocdl.make_buffer_tensor(G, max_size=False)
    o = fx.rocdl.make_buffer_tensor(O, max_size=False)
    ss = fx.rocdl.make_buffer_tensor(SS, max_size=False)
    gl = _load(g, row * NI + branch, 1, fx.Float32)[0]
    gate = fx.Float32(2.0) * _sigmoid(gl * fx.Float32(1.0 / _HC))
    off = row * K + branch * hs + col
    hv = _load(h, off, 8, fx.BFloat16).to(fx.Float32)
    yv = _load(y, row * hs + col, 8, fx.BFloat16).to(fx.Float32)
    ov = (hv + yv * fx.Vector.from_elements([gate] * 8, fx.Float32)).to(fx.BFloat16)
    _store8(o, off, ov)
    of = ov.to(fx.Float32)
    sq = fx.Float32(0.0)
    for e in range_constexpr(8):
        sq = sq + of[e] * of[e]
    for shift in range_constexpr(6):
        peer = gpu.shuffle(sq, fx.Int32(1 << shift), fx.Int32(64), mode="xor")
        sq = sq + fx.Float32(peer)
    if lane == 0:
        _store(ss, row * (_HC * nblk) + part, sq, fx.Float32)


@flyc.jit
def _launch_inject(
    H: fx.Tensor,
    Y: fx.Tensor,
    G: fx.Tensor,
    O: fx.Tensor,
    SS: fx.Tensor,
    M: fx.Constexpr[int],
    K: fx.Constexpr[int],
    NI: fx.Constexpr[int],
    stream: fx.Stream,
):
    _hc_inject_fused(H, Y, G, O, SS, K, NI).launch(
        grid=(M, K // _INJECT_BLOCK), block=(64,), stream=stream
    )


def hc_apply_inject(h, y, g):
    """Elementwise combine h + y * 2 sigmoid(g / HC) per branch, given the inject logits.

    Returns (combined [M, K] BF16, partial sums of squares [M, HC * HS / 512] FP32 of the
    combined residual, for the next ``hc_fused_mix``)."""
    m, k = h.shape
    if k % (_HC * _INJECT_BLOCK) or g.shape != (m, _HC) or y.shape != (m, k // _HC):
        raise ValueError("unsupported fused HC combine shape")
    mb = _bucket_rows(m)
    out = torch.empty((mb, k), device=h.device, dtype=h.dtype)
    ss = torch.empty((mb, k // _INJECT_BLOCK), device=h.device, dtype=torch.float32)
    constexprs = (mb, k, g.shape[1])
    _dispatch(
        _launch_inject, constexprs,
        h, y, g, out, ss, *constexprs, torch.cuda.current_stream(h.device),
    )  # fmt: skip
    return out[:m], ss[:m]


def prewarm_hc_fused(weights: HcFusedWeights, k, device):
    """Compile one weight shape for every row bucket, with both sums-of-squares sources
    and, given inject rows, the combine. FlyDSL caches the kernels on disk."""
    f32 = torch.float32
    for m in HC_FUSED_ROW_BUCKETS:
        h = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
        _, g = hc_fused_mix(h, weights)
        sumsq = torch.zeros((m, k // _INJECT_BLOCK), device=device, dtype=f32)
        hc_fused_mix(h, weights, sumsq=sumsq)
        if g is not None:
            hc_apply_inject(h, h.new_zeros((m, k // _HC)), g)
