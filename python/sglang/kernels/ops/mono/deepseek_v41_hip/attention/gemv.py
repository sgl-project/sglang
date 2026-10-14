# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's MXFP8 GEMV of a row-major e4m3 weight on the gfx950 scaled MFMA, and K2's
GEMV of the same weight on bf16 rows (the weight widened to bf16 in registers)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from sglang.kernels.ops.mono.common_hip.ops import mfma_bf16, rsrc, traced
from sglang.kernels.ops.mono.deepseek_v41_hip.stages.gemv import lds_row

from .device import THREADS, WAVES, fp8x8_bf16, mfma_scaled, pow2, tile_map

RED = WAVES * 64 * 4  # one task's accumulators in ``red`` (f32)


def _steps(c, nsteps, split, tiled):
    """This wave's steps of the window: [(local step, live)]."""
    wave = c["wave"]
    if const_expr(split is None):
        per = (nsteps + WAVES - 1) // WAVES
        ragged = nsteps % WAVES != 0
        return [
            (wave + WAVES * i, (wave + WAVES * i < nsteps) if ragged else None)
            for i in range(per)
        ]
    assert split <= WAVES and nsteps % split == 0
    per = nsteps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def _step_load(c, r_w, r_ws, k, row0, st, pred=None):
    """Load step ``st``'s weight operand of rows row0 .. +16 and its E8M0 byte (masked when not ``pred``)."""
    lane = c["lane"]
    j = lane // 16
    row = row0 + lane % 16
    base = row * k + 128 * st + 16 * j
    lo = fx.Vector(bo.buffer_load(r_w, base // 4, vec_width=4, dtype=T.i32))
    hi = fx.Vector(bo.buffer_load(r_w, (base + 64) // 4, vec_width=4, dtype=T.i32))
    kg = k // 32
    wsi = (row0 // 32) * kg + st * 4 + j
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def gemv_loads(
    c, w, ws, k, row0, split=None, tiled=False, kbeg=0, nsteps=None, pred=None
):
    """Issue this wave's weight loads for steps kbeg .. kbeg + nsteps."""
    nsteps = k // 128 if nsteps is None else nsteps
    if const_expr(pred is None):
        r_w, r_ws = rsrc(w), rsrc(ws)
    else:
        # a masked-off CTA reads through a zero-size buffer: zeros, no traffic
        nb = pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
        r_w, r_ws = rsrc(w, nb), rsrc(ws, nb)
    if const_expr(tiled):
        row0 = row0 + 16 * (c["wave"] // split)
    return [
        _step_load(
            c,
            r_w,
            r_ws,
            k,
            row0,
            kbeg + (st if live is None else fx.min(st, nsteps - 1)),
            pred,
        )
        for st, live in _steps(c, nsteps, split, tiled)
    ]


def _step_mfma(c, nsteps, xl, xsl, st, ops, acc, live, rows, halves=False):
    lane = c["lane"]
    lo, hi, sa = ops
    x_row, s_row = lds_row(nsteps * 32), lds_row(nsteps * 4)
    j = lane // 16
    col = fx.min(lane % 16, rows - 1)
    if const_expr(live is not None):
        st = fx.min(st, nsteps - 1)
    av = fx.Vector.from_elements(
        [lo[d] for d in range(4)] + [hi[d] for d in range(4)], fx.Int32
    )
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xv = fx.Vector.from_elements(
        [xlo[d] for d in range(4)] + [xhi[d] for d in range(4)], fx.Int32
    )
    sb = fx.ptr_load(xsl + (col * s_row + st * 4 + j))
    if const_expr(live is not None):
        # a step past the end adds zero at unit scale
        av = fx.Vector.from_elements(
            [live.select(av[d], fx.Int32(0)) for d in range(8)], fx.Int32
        )
        sa = live.select(sa, fx.Int32(127))
        sb = live.select(sb, fx.Int32(127))
    if const_expr(halves):
        # K 0..63, then 64..127, the other half zeroed: bit for bit two 32x32x64 MFMAs
        z = fx.Int32(0)
        x0 = fx.Vector.from_elements(
            [xv[d] if d < 4 else z for d in range(8)], fx.Int32
        )
        x1 = fx.Vector.from_elements(
            [z if d < 4 else xv[d] for d in range(8)], fx.Int32
        )
        return mfma_scaled(av, x1, mfma_scaled(av, x0, acc, sa, sb), sa, sb)
    return mfma_scaled(av, xv, acc, sa, sb)


def gemv_mfmas(c, k, xl, xsl, red, ops, split=None, tiled=False, rows=16, nsteps=None):
    """Multiply ``gemv_loads``' operands with ``rows`` tokens' MXFP8 rows in LDS into ``red``."""
    lane, wave = c["lane"], c["wave"]
    nsteps = k // 128 if nsteps is None else nsteps
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_steps(c, nsteps, split, tiled)):
        acc = _step_mfma(c, nsteps, xl, xsl, st, ops[i], acc, live, rows)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements(
            [(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(4)],
            fx.Float32,
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)


# SGLang's mxfp8_gemv chains per wave and sums in wave order; a chain here plays one wave


def chain_slots(c, nsteps, waves):
    """This wave's chains (w, w + WAVES, ...): [(chain, [(step, live)])]."""
    per = (nsteps + waves - 1) // waves
    out = []
    for i in range((waves + WAVES - 1) // WAVES):
        ch = c["wave"] + WAVES * i
        out.append((ch, [(ch * per + j, ch * per + j < nsteps) for j in range(per)]))
    return out


def chain_loads(c, w, ws, k, row0, slots, pred):
    """Issue the weight loads of every step of ``slots`` (a dead step reads a zero-size buffer)."""
    out = []
    for _, steps in slots:
        ops = []
        for st, live in steps:
            nb = (live & pred).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
            ops.append(
                _step_load(
                    c, rsrc(w, nb), rsrc(ws, nb), k, row0, fx.min(st, k // 128 - 1)
                )
            )
        out.append(ops)
    return out


def chain_mfmas(c, xl, xsl, ops, rows, nsteps, slots, halves=False):
    """Run the chains against ``rows`` tokens' MXFP8 rows, each from zero in step order."""
    out = []
    for (_, steps), sops in zip(slots, ops):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for (st, live), op in zip(steps, sops):
            loc = fx.min(st, nsteps - 1)
            new = _step_mfma(c, nsteps, xl, xsl, loc, op, acc, None, rows, halves)
            acc = fx.Vector.from_elements(
                [live.select(new[e], acc[e]) for e in range(4)], fx.Float32
            )
        out.append(acc)
    return out


def chain_store(c, red, accs, slots):
    """Each chain's accumulator -> ``red[chain][lane]`` (an empty chain's is +0)."""
    for (ch, _), acc in zip(slots, accs):
        fx.ptr_store(acc, red + (ch * 64 + c["lane"]) * 4)


def chain_sum(red, r, t, order):
    """Row r, token t: the chains' partials summed one by one in ``order``."""
    at = lambda w: red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4)  # noqa: E731
    tot = fx.ptr_load(at(order[0]))
    for w in order[1:]:
        tot = tot + fx.ptr_load(at(w))
    return tot


def span_items(width, n):
    """A [n, width] span of 16 B items: rows a round of THREADS threads, rounds."""
    wq = width // 4
    rpr = THREADS // wq
    assert rpr >= 1, width
    return wq, rpr, (n + rpr - 1) // rpr


def rows_loads(
    c,
    words_base,
    codes_base,
    stride_w,
    off_w,
    words,
    stride_g,
    off_g,
    groups,
    t0,
    n,
    pred=None,
):
    """Issue loads of the published MXFP8 rows of tokens t0 .. t0 + n (masked when not ``pred``)."""
    tid = c["tid"]
    nb = None if pred is None else pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    out = []
    for base, stride, off, width in (
        (words_base, stride_w, off_w, words),
        (codes_base, stride_g, off_g, groups),
    ):
        wq, rpr, rounds = span_items(width, n)
        row0, col = tile_map(tid, wq)
        a0 = col * 4 + off + t0 * stride
        r = rsrc(base, nb)
        regs = []
        for i in range(rounds):
            row = row0 + rpr * i
            if (i + 1) * rpr > n or rpr * wq < THREADS:
                row = fx.min(row, n - 1)
            regs.append(
                fx.Vector(
                    bo.buffer_load(r, row * stride + a0, vec_width=4, dtype=T.i32)
                )
            )
        out.append(regs)
    return out


@traced
def _store_span(c, regs, width, dst, n, lrow=None):
    tid = c["tid"]
    wq, rpr, _ = span_items(width, n)
    row0, col = tile_map(tid, wq)
    lrow = lds_row(width) if lrow is None else lrow
    base = dst + (row0 * lrow + col * 4)
    for i in range_constexpr(len(regs)):
        if const_expr((i + 1) * rpr <= n and rpr * wq == THREADS):
            fx.ptr_store(regs[i], base + i * rpr * lrow)
        else:
            if (row0 + rpr * i < n) & (tid < rpr * wq):
                fx.ptr_store(regs[i], base + i * rpr * lrow)


def rows_store(c, regs, words, groups, xl, xsl, n, row_w=None, row_g=None):
    """Store ``rows_loads``' registers into LDS rows 0 .. n."""
    _store_span(c, regs[0], words, xl, n, row_w)
    _store_span(c, regs[1], groups, xsl, n, row_g)
    gpu.barrier()


# ---- the GEMV of a bf16 activation: the FP8 weight widened to bf16 in registers, at
# its 32 x 32 block scales (exact), on the bf16 MFMA. The rows stage through LDS one K
# half at a time, and each wave of a row tile takes a quarter of each half.


def _half_steps(c, nsteps, split):
    """This wave's 128-K steps of each K half: [[step, ..] for the two halves]."""
    half = nsteps // 2
    per = half // split
    part = c["wave"] % split
    return [[h * half + part * per + i for i in range(per)] for h in range(2)]


def bf16_gemv_loads(c, w, ws, k, row0, split, pred=None):
    """Issue this wave's weight loads for ``bf16_gemv_mfmas``, rows row0 + 16 (wave //
    split) ..: per K half, each step's (step, e4m3 K 16j .., K 64 + 16j .., the
    E8M0 word of its four 32-blocks); masked when not ``pred``."""
    if const_expr(pred is None):
        r_w, r_ws = rsrc(w), rsrc(ws)
    else:
        nb = pred.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
        r_w, r_ws = rsrc(w, nb), rsrc(ws, nb)
    lane = c["lane"]
    row0 = row0 + 16 * (c["wave"] // split)
    row = row0 + lane % 16
    j = lane // 16
    kg = k // 32
    out = []
    for steps in _half_steps(c, k // 128, split):
        half = []
        for st in steps:
            base = row * k + 128 * st + 16 * j
            lo = fx.Vector(bo.buffer_load(r_w, base // 4, vec_width=4, dtype=T.i32))
            hi = fx.Vector(
                bo.buffer_load(r_w, (base + 64) // 4, vec_width=4, dtype=T.i32)
            )
            word = fx.Int32(
                bo.buffer_load(
                    r_ws, ((row0 // 32) * kg + 4 * st) // 4, vec_width=1, dtype=T.i32
                )
            )
            half.append((st, lo, hi, word))
        out.append(half)
    return out


def bf16_gemv_mfmas(c, k, xl, ops, acc, h, rows):
    """Add K half ``h``'s products into ``acc``: ``bf16_gemv_loads``' weights widened
    to bf16 at their block scales, against ``rows`` tokens' bf16 rows of that half in
    LDS (natural K order, ``k // 4`` words a row)."""
    lane = c["lane"]
    j = lane // 16
    col = fx.min(lane % 16, rows - 1)
    x_row = lds_row(k // 4)
    half = k // 128 // 2
    for st, lo, hi, word in ops[h]:
        k0 = 128 * (st - h * half)
        # the 32-blocks of this lane's K 16j .. and K 64 + 16j ..
        s_lo = pow2((word >> (8 * (j // 2))) & 0xFF)
        s_hi = pow2((word >> (8 * (2 + j // 2))) & 0xFF)
        for w0, w1, sc, kk in (
            (lo[0], lo[1], s_lo, 16 * j),
            (lo[2], lo[3], s_lo, 16 * j + 8),
            (hi[0], hi[1], s_hi, 64 + 16 * j),
            (hi[2], hi[3], s_hi, 64 + 16 * j + 8),
        ):
            xv = fx.Vector(
                fx.ptr_load(
                    xl + (col * x_row + (k0 + kk) // 2),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            wv = fp8x8_bf16(w0, w1, sc)
            acc = mfma_bf16(wv.bitcast(fx.BFloat16), xv.bitcast(fx.BFloat16), acc)
    return acc


def words_loads(c, base, stride, off, words, t0, n):
    """Issue loads of ``words`` 32-bit words of tokens t0 .. t0 + n's rows at ``off``."""
    tid = c["tid"]
    wq, rpr, rounds = span_items(words, n)
    row0, col = tile_map(tid, wq)
    a0 = col * 4 + off + t0 * stride
    r = rsrc(base)
    regs = []
    for i in range(rounds):
        row = row0 + rpr * i
        if (i + 1) * rpr > n or rpr * wq < THREADS:
            row = fx.min(row, n - 1)
        regs.append(
            fx.Vector(bo.buffer_load(r, row * stride + a0, vec_width=4, dtype=T.i32))
        )
    return regs


def words_store(c, regs, words, xl, n):
    """Store ``words_loads``' registers into LDS rows 0 .. n."""
    _store_span(c, regs, words, xl, n)
    gpu.barrier()
