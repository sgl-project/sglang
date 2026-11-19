"""Muse Spark's attention boundary on FlyDSL's role-split flagless one-shot all-reduce (gfx1250, TP8, 1-16 rows).

One kernel: the role-split one-shot (sglang_meta_common flydsl_ar.allreduce_kernels_gfx1250.make_one_shot_role_launcher,
form "role": sender waves push their line to every peer, receiver waves poll their own scratch with the adaptive wait,
re-zero, fixed-order fp32 FTZ sum, one rounding), then the boundary -- weight-free post-norm, per-channel gated
residual, weighted pre-norm, optional fp32 router copy -- on every row, on every rank.

One single-wave sender block and one receiver block per 256-column piece of a row; the senders take the lowest block
ids and never wait. A row's norms need the whole row: a receiver posts its piece's x.x, a.a, a.c, c.c (a = alpha *
stream, c = beta * x) in this GPU's sum lines ({sum, flag, sum, flag}, device scope) and folds the row's 32 pieces in
one butterfly; the pre-norm's sum of squares uses x * post_rstd before its bf16 rounding. A receiver waits only on
peers' senders and on its row's other receivers (adjacent block ids), so in-order dispatch keeps it deadlock-free. The
fixed summation order gives every rank the same bits.
"""

from __future__ import annotations

import functools
import os

import torch

_WAVE = 32
_LINE_BYTES = 16
# gfx12 cache policy SCOPE_DEV (bits [4:3]): the sum lines never leave this GPU
_CPOL_SCOPE_DEV = 2 << 3


def _scratch_lines(world_size: int, max_rows: int, hidden: int) -> int:
    """Lines of the scratch: [bank 2][slot ws][max_rows * hidden / 8]."""
    return 2 * world_size * max_rows * (hidden // 8)


def _sum_lines(max_rows: int, hidden: int) -> int:
    """Lines of this GPU's sum lines: [bank 2][row max_rows][piece][2]."""
    return 2 * max_rows * (hidden // 8 // _WAVE) * 2


@functools.lru_cache(maxsize=None)
def make_ar_boundary_one_shot_launcher(
    world_size: int,
    rank: int,
    hidden: int,
    max_rows: int,
    post_eps: float,
    pre_eps: float,
    with_router: bool,
    wait_tight: int = 4,
    debug_cap: int = 0,
):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as std_arith
    from flydsl._mlir.dialects import gpu as gpu_d
    from flydsl._mlir.dialects import math as math_d
    from flydsl._mlir.dialects import vector as vec_d
    from flydsl.expr import T, arith, const_expr, range_constexpr
    from flydsl.expr.rocdl import sched_barrier
    from flydsl.expr.typing import Stream
    from sglang_meta_common.ops.comm.flydsl_ar import allreduce_kernels_gfx1250 as k
    from sglang_meta_common.ops.flydsl_common.buffer_ops import (
        buffer_load,
        buffer_store,
        create_buffer_resource_from_addr,
    )

    ws = world_size
    row_lines = hidden // 8
    pieces = row_lines // _WAVE
    assert hidden % (8 * _WAVE) == 0 and pieces == _WAVE, (
        "one lane per piece when folding a row's sums: hidden must be 8 * 32 * 32"
    )
    peers = [(rank + r) % ws for r in range(1, ws)]
    srcs = [q for q in range(ws) if q != rank]
    src_index = {q: i for i, q in enumerate(srcs)}
    n = ws - 1
    sl = max_rows * row_lines
    max_pos = max_rows * pieces
    cpol = k.LL_STORE_CPOL_SCOPE_SYS
    inv_hidden = 1.0 / hidden

    def f32():
        return ir.F32Type.get()

    def f32c(v: float):
        return std_arith.ConstantOp(f32(), ir.FloatAttr.get(f32(), v)).result

    def vf(m):
        return ir.VectorType.get([m], f32())

    def fadd(a, b):
        return std_arith.AddFOp(a, b).result

    def fmul(a, b):
        return std_arith.MulFOp(a, b).result

    def splat(m, s):
        return vec_d.broadcast(vf(m), s)

    def cat8(a, b):
        return vec_d.from_elements(vf(8), [k._extract(a, j) for j in range(4)] + [k._extract(b, j) for j in range(4)])

    def half(v8, h):
        return vec_d.from_elements(vf(4), [k._extract(v8, 4 * h + j) for j in range(4)])

    def hsum(v8):
        acc = k._extract(v8, 0)
        for j in range(1, 8):
            acc = fadd(acc, k._extract(v8, j))
        return acc

    def wave_sum(s):
        """Butterfly over the wave's 32 lanes: every lane gets the same bits (fp add commutes)."""
        for off in (1, 2, 4, 8, 16):
            other = gpu_d.ShuffleOp(
                s, arith.unwrap(fx.Int32(off)), arith.unwrap(fx.Int32(_WAVE)), mode="xor"
            ).shuffleResult
            s = fadd(s, other)
        return s

    def as_i32(s):
        return std_arith.BitcastOp(ir.IntegerType.get_signless(32), s).result

    def as_f32(i):
        return std_arith.BitcastOp(f32(), arith.unwrap(i)).result

    @flyc.kernel(known_block_size=[_WAVE, 1, 1])
    def ar_boundary_one_shot_gfx1250(
        base0: fx.Int64,
        base1: fx.Int64,
        base2: fx.Int64,
        base3: fx.Int64,
        base4: fx.Int64,
        base5: fx.Int64,
        base6: fx.Int64,
        base7: fx.Int64,
        out: fx.Int64,
        new_stream: fx.Int64,
        router: fx.Int64,
        inp: fx.Int64,
        stream: fx.Int64,
        alpha: fx.Int64,
        beta: fx.Int64,
        pre_w: fx.Int64,
        rows: fx.Int32,
        cells: fx.Int64,
        sums: fx.Int64,
        dbg: fx.Int64,
    ):
        lane = fx.Uint32(fx.thread_idx.x)
        blk = fx.Uint32(fx.block_idx.x)
        nw = fx.Uint32(rows) * fx.Uint32(pieces)
        live = (blk < nw * fx.Uint32(2)) & (fx.Uint64(inp) != fx.Uint64(0))
        live = live & (fx.Uint64(cells) != fx.Uint64(0)) & (fx.Uint64(out) != fx.Uint64(0))
        if live:
            bases = [base0, base1, base2, base3, base4, base5, base6, base7][:ws]
            is_sender = blk < nw
            pos = fx.Uint32(k._select(is_sender, blk, blk - nw))
            # sender cells first; both strided by the largest batch's positions, so a position keeps its cells
            cell = fx.Uint32(k._select(is_sender, pos, pos + fx.Uint32(max_pos)))
            row = pos // fx.Uint32(pieces)
            piece = pos % fx.Uint32(pieces)
            idx = pos * fx.Uint32(_WAVE) + lane
            col = (piece * fx.Uint32(_WAVE) + lane) * fx.Uint32(8)

            in_rsrc = create_buffer_resource_from_addr(inp)
            cell_rsrc = create_buffer_resource_from_addr(cells)
            d = buffer_load(in_rsrc, idx * fx.Uint32(4), vec_width=4, dtype=T.i32)
            epoch = buffer_load(cell_rsrc, cell, vec_width=1, dtype=T.i32)
            sched_barrier(0)
            flag = k._next_flag(fx.Uint32(epoch))
            bank = flag & fx.Uint32(1)
            my_rsrc = create_buffer_resource_from_addr(bases[rank])

            def line_el(slot, i):
                return ((bank * fx.Uint32(ws) + slot) * fx.Uint32(sl) + i) * fx.Uint32(4)

            if is_sender:
                enc = k._flagless_encode(d)
                for qi in range_constexpr(n):
                    buffer_store(enc, create_buffer_resource_from_addr(bases[peers[qi]]),
                                 line_el(fx.Uint32(rank), idx), cache_modifier=cpol)
            else:
                a_rsrc = create_buffer_resource_from_addr(alpha)
                b_rsrc = create_buffer_resource_from_addr(beta)
                w_rsrc = create_buffer_resource_from_addr(pre_w)
                s_rsrc = create_buffer_resource_from_addr(stream)
                sum_rsrc = create_buffer_resource_from_addr(sums)
                zero4 = k._from_elements(4, [fx.Uint32(0)] * 4)
                ones4 = k._splat_i32(k._vec_i32(4), k._I32_MINUS1)
                e = row * fx.Uint32(hidden) + col
                av = cat8(buffer_load(a_rsrc, col, vec_width=4), buffer_load(a_rsrc, col + fx.Uint32(4), vec_width=4))
                bv = cat8(buffer_load(b_rsrc, col, vec_width=4), buffer_load(b_rsrc, col + fx.Uint32(4), vec_width=4))
                wv = k._upcast_n(buffer_load(w_rsrc, col // fx.Uint32(2), vec_width=4, dtype=T.i32), 4)
                sv = cat8(buffer_load(s_rsrc, e, vec_width=4), buffer_load(s_rsrc, e + fx.Uint32(4), vec_width=4))

                def capped(pend, it):
                    return pend & (it < fx.Uint32(debug_cap)) if const_expr(debug_cap > 0) else pend

                def record(stage, pend):
                    if const_expr(debug_cap > 0):
                        if pend:
                            buffer_store(fx.Uint32(1), create_buffer_resource_from_addr(dbg),
                                         cell * fx.Uint32(3) + fx.Uint32(stage))

                def ladder(kk):
                    # FlyDSL's low-traffic poll ladder: 2 (x3), 8 (x4), then 16 x 64 clocks
                    if kk > fx.Uint32(0):
                        k._s_sleep(2)
                    if kk > fx.Uint32(3):
                        k._s_sleep(6)
                    if kk > fx.Uint32(7):
                        k._s_sleep(8)

                # FlyDSL's adaptive wait: ``wait_tight`` full reloads, then lanes 0..n-1 each poll one source's line
                # of the wave's first index under the sleep ladder, then full reloads until complete
                els = [line_el(fx.Uint32(s), idx) for s in srcs]

                def load_all():
                    return k._ll_pack_lines([buffer_load(my_rsrc, el, vec_width=4, dtype=T.i32, cache_modifier=cpol)
                                             for el in els])

                lines = load_all()
                pend = k._flagless_lines_pending(lines, n)
                it = fx.Uint32(0)
                while pend & (it < fx.Uint32(wait_tight)):
                    k._fence("acquire", "wavefront")
                    lines = load_all()
                    pend = k._flagless_lines_pending(lines, n)
                    it = it + fx.Uint32(1)
                if k._wave_any(pend):
                    sslot = fx.Uint32(srcs[0])
                    for i in range_constexpr(1, n):
                        sslot = fx.Uint32(k._select(lane == fx.Uint32(i), fx.Uint32(srcs[i]), sslot))
                    saddr = bases[rank] + fx.Uint64(line_el(sslot, pos * fx.Uint32(_WAVE)) * fx.Uint32(4))
                    want = lane < fx.Uint32(n)
                    kk = fx.Uint32(0)
                    while capped(k._wave_any(want), kk):
                        ladder(kk)
                        sline = ones4
                        if want:
                            sline = k._gptr_load_v4(saddr)
                        want = want & k._flagless_any_zero(sline)
                        kk = kk + fx.Uint32(1)
                it = fx.Uint32(0)
                while capped(pend, it):
                    k._fence("acquire", "wavefront")
                    lines = load_all()
                    pend = k._flagless_lines_pending(lines, n)
                    it = it + fx.Uint32(1)
                record(0, pend)
                for i in range_constexpr(n):
                    buffer_store(zero4, my_rsrc, els[i], cache_modifier=cpol)
                for i in range_constexpr(ws):
                    if const_expr(i == rank):
                        v = k._ftz_n(k._upcast_n(k._flagless_canon(d), 4), 8)
                    else:
                        v = k._ftz_n(k._upcast_n(k._flagless_payload(lines, src_index[i]), 4), 8)
                    acc = v if i == 0 else k._ftz_n(fadd(acc, v), 8)
                # the all-reduce returns bf16
                x8 = k._upcast_n(k._downcast_hw_n(acc, 8), 4)

                def sum_el(pc, h):
                    line = ((bank * fx.Uint32(max_rows) + row) * fx.Uint32(pieces) + pc) * fx.Uint32(2) + fx.Uint32(h)
                    return line * fx.Uint32(4)

                fl = arith.unwrap(flag)
                a8 = fmul(av, sv)
                c8 = fmul(bv, x8)
                part = [wave_sum(hsum(fmul(x8, x8))), wave_sum(hsum(fmul(a8, a8))),
                        wave_sum(hsum(fmul(a8, c8))), wave_sum(hsum(fmul(c8, c8)))]
                post = [vec_d.from_elements(k._vec_i32(4), [as_i32(part[0]), fl, as_i32(part[1]), fl]),
                        vec_d.from_elements(k._vec_i32(4), [as_i32(part[2]), fl, as_i32(part[3]), fl])]
                for h in range_constexpr(2):
                    if lane == fx.Uint32(h):
                        buffer_store(post[h], sum_rsrc, sum_el(piece, h), cache_modifier=_CPOL_SCOPE_DEV)

                # the row's folded sums: lane p reads piece p's sum lines
                sels = [sum_el(lane, h) for h in range(2)]

                def load_sums():
                    return [buffer_load(sum_rsrc, el, vec_width=4, dtype=T.i32, cache_modifier=_CPOL_SCOPE_DEV)
                            for el in sels]

                def stale(ls):
                    p = fx.Uint32(k._extract(ls[0], 1)) != flag
                    for li in range_constexpr(2):
                        for dw in range_constexpr(1, 4, 2):
                            p = p | (fx.Uint32(k._extract(ls[li], dw)) != flag)
                    return p

                ls = load_sums()
                spend = k._wave_any(stale(ls))
                it = fx.Uint32(0)
                while capped(spend, it):
                    ladder(it)
                    k._fence("acquire", "wavefront")
                    ls = load_sums()
                    spend = k._wave_any(stale(ls))
                    it = it + fx.Uint32(1)
                record(2, spend)
                xx, aa, ac, cc = [wave_sum(as_f32(k._extract(ls[li], dw))) for li in range(2) for dw in (0, 2)]
                post_r = math_d.rsqrt(fadd(fmul(xx, f32c(inv_hidden)), f32c(post_eps)))
                pre_ss = fadd(fadd(aa, fmul(fmul(f32c(2.0), post_r), ac)), fmul(fmul(post_r, post_r), cc))
                pre_r = math_d.rsqrt(fadd(fmul(pre_ss, f32c(inv_hidden)), f32c(pre_eps)))
                # the post-norm returns bf16
                normed = k._upcast_n(k._downcast_hw_n(fmul(x8, splat(8, post_r)), 8), 4)
                s_new = fadd(fmul(av, sv), fmul(bv, normed))
                y = fmul(fmul(s_new, splat(8, pre_r)), wv)
                ns_rsrc = create_buffer_resource_from_addr(new_stream)
                buffer_store(half(s_new, 0), ns_rsrc, e)
                buffer_store(half(s_new, 1), ns_rsrc, e + fx.Uint32(4))
                if const_expr(with_router):
                    r_rsrc = create_buffer_resource_from_addr(router)
                    buffer_store(half(y, 0), r_rsrc, e)
                    buffer_store(half(y, 1), r_rsrc, e + fx.Uint32(4))
                buffer_store(k._downcast_hw_n(y, 8), create_buffer_resource_from_addr(out), e // fx.Uint32(2))
            if lane == fx.Uint32(0):
                buffer_store(flag, cell_rsrc, cell)

    @flyc.jit
    def launch(
        base0: fx.Int64,
        base1: fx.Int64,
        base2: fx.Int64,
        base3: fx.Int64,
        base4: fx.Int64,
        base5: fx.Int64,
        base6: fx.Int64,
        base7: fx.Int64,
        out: fx.Int64,
        new_stream: fx.Int64,
        router: fx.Int64,
        inp: fx.Int64,
        stream: fx.Int64,
        alpha: fx.Int64,
        beta: fx.Int64,
        pre_w: fx.Int64,
        rows: fx.Int32,
        cells: fx.Int64,
        sums: fx.Int64,
        dbg: fx.Int64,
        nblocks: fx.Int32,
        stream_: Stream = Stream(None),
    ):
        _ = ("muse_ar_boundary_one_shot", ws, rank, hidden, max_rows, post_eps, pre_eps, with_router, wait_tight,
             debug_cap)
        ar_boundary_one_shot_gfx1250(
            base0, base1, base2, base3, base4, base5, base6, base7,
            out, new_stream, router, inp, stream, alpha, beta, pre_w,
            rows, cells, sums, dbg,
        ).launch(grid=(fx.Index(nblocks), 1, 1), block=(_WAVE, 1, 1), stream=stream_)

    return launch


class FlyDSLArBoundaryOneShot:
    """Peer scratch, epoch cells and the compiled kernel of one TP group, for 1 to ``max_rows`` rows.

    Built collectively (every TP rank in the same order); the kernel compiles on its first call, which must not be
    under graph capture."""

    def __init__(self, tp_group, hidden: int, max_rows: int, post_eps: float, pre_eps: float):
        from sglang_meta_common.ops.comm.flydsl_ar.peer_memory import make_peer_buffer, select_backend

        self.world_size = ws = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.device = tp_group.device
        self.hidden = hidden
        self.max_rows = max_rows
        self.post_eps, self.pre_eps = float(post_eps), float(pre_eps)
        arch = torch.cuda.get_device_properties(self.device).gcnArchName
        self._buf = make_peer_buffer(select_backend(arch), _scratch_lines(ws, max_rows, hidden) * _LINE_BYTES,
                                     tp_group.cpu_group, self.device)
        self._buf.local.zero_()
        torch.cuda.synchronize(self.device)
        torch.distributed.barrier(group=tp_group.cpu_group)
        bases = [int(a) for a in self._buf.addresses.cpu().tolist()]
        self._bases = tuple(bases + [0] * (8 - len(bases)))
        self._pieces = hidden // 8 // _WAVE
        self._cells = torch.zeros(2 * max_rows * self._pieces, dtype=torch.int32, device=self.device)
        self._sums = torch.zeros(_sum_lines(max_rows, hidden) * 4, dtype=torch.int32, device=self.device)
        # > 0: every spin wait gives up after this many rounds and marks its wave's (cell, stage) in ``dbg``
        self._debug_cap = int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_FLYDSL_DEBUG_CAP", "0"))
        self.dbg = (torch.zeros(self._cells.numel() * 3, dtype=torch.int32, device=self.device)
                    if self._debug_cap > 0 else None)
        self._compiled = {}

    def supports(self, num_rows: int) -> bool:
        return 0 < num_rows <= self.max_rows

    def _launch(self, args, with_router: bool):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        cu_stream = torch.cuda.current_stream(self.device)
        fn = self._compiled.get(with_router)
        if fn is not None:
            fn(*args, cu_stream)
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("AR-boundary FlyDSL one-shot kernel was not compiled before graph capture")
        launcher = make_ar_boundary_one_shot_launcher(
            self.world_size, self.rank, self.hidden, self.max_rows, self.post_eps, self.pre_eps, with_router,
            int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_FLYDSL_TIGHT", "4")), self._debug_cap,
        )
        ints = (0,) * 16 + (1, 0, 0, 0, 1)
        typed = tuple(fx.Int32(a) if is32 else fx.Int64(a) for a, is32 in zip(args, ints))
        self._compiled[with_router] = flyc.compile(launcher, *typed, cu_stream)

    def warm(self):
        """Compile both forms now (outside graph capture): a null input makes the compile's launch a no-op."""
        for with_router in (False, True):
            if with_router not in self._compiled:
                self._launch(self._bases + (0,) * 8 + (1, 0, 0, 0, 1), with_router)

    def __call__(self, stream, partial, pre_weight, alpha, beta, with_router: bool):
        rows = partial.shape[0]
        out = torch.empty_like(partial)
        new_stream = torch.empty_like(stream)
        router = torch.empty_like(stream) if with_router else None
        self._launch(self._bases + (
            out.data_ptr(), new_stream.data_ptr(), router.data_ptr() if with_router else 0,
            partial.data_ptr(), stream.data_ptr(), alpha.data_ptr(), beta.data_ptr(), pre_weight.data_ptr(),
            rows, self._cells.data_ptr(), self._sums.data_ptr(), 0 if self.dbg is None else self.dbg.data_ptr(),
            2 * rows * self._pieces,
        ), with_router)
        return out, new_stream, router
