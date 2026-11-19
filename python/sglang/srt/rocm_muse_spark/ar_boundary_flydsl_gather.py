"""Muse Spark's FFN boundary on a FlyDSL all-gather of an output-column-sharded Unpooler (gfx1250, TP decode).

Each TP rank computes its hidden / ws output columns of the MoE Unpooler (post_expert_proj) from the reduced expert
output; one kernel gathers every rank's columns and runs the FFN boundary -- input-scale post-norm
(bf16(w * x), then rms-normalized), per-channel gated residual, the next layer's weighted pre-norm -- on every row, on
every rank.

The exchange is the role-split flagless one-shot's (sglang_meta_common flydsl_ar.allreduce_kernels_gfx1250): 16-byte
self-validating lines, per-wave epochs, two banks, the adaptive wait. One single-wave sender block per 256-column
piece of this rank's columns pushes it to every peer; one receiver block per 256-column piece of a row reads it (its
owner's line, or this rank's own input) without any reduction. A row's norms need the whole row: a receiver posts its
piece's x.x, a.a, a.c, c.c (x the post-norm input, a = alpha * stream, c = beta * x) in this GPU's sum lines
({sum, flag, sum, flag}, device scope) and folds the row's 32 pieces in one butterfly; the pre-norm's sum of squares
uses x * post_rstd before its bf16 rounding. Senders take the lowest block ids and never wait; a receiver waits only
on one peer's sender and on its row's other receivers (adjacent block ids), so in-order dispatch keeps it
deadlock-free. Every rank holds the same columns and stream, so every rank gets the same bits.

Package all-gather (``package=True``; world size 4 or 8, compact DPX placement: ranks 2g, 2g + 1 are the two
partitions of package g): a sender pushes to its sibling (rank ^ 1) and to the same-parity ranks of the other packages
only, half the cross-package lines of the flat push; a receiver of a same-parity owner's piece forwards the line, still
encoded, to the same slot of its sibling's scratch, which is where the sibling's receiver of that piece waits
(sglang_meta_common cpu_reference.all_gather_plan "package"). A forwarding receiver waits only on a sender, so the
forwards never wait on the sibling. The bytes are the flat gather's, so the outputs are bit-identical to it.
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
    """Lines of the scratch: [bank 2][slot ws][max_rows * hidden / ws / 8]."""
    return 2 * world_size * max_rows * (hidden // world_size // 8)


def _sum_lines(max_rows: int, hidden: int) -> int:
    """Lines of this GPU's sum lines: [bank 2][row max_rows][piece][2]."""
    return 2 * max_rows * (hidden // 8 // _WAVE) * 2


@functools.lru_cache(maxsize=None)
def make_gather_ffn_boundary_launcher(
    world_size: int,
    rank: int,
    hidden: int,
    max_rows: int,
    post_eps: float,
    pre_eps: float,
    post_weight: bool,
    wait_tight: int = 4,
    debug_cap: int = 0,
    package: bool = False,
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
    assert pieces % ws == 0, "every rank owns whole 256-column pieces"
    own_pieces = pieces // ws
    part_lines = row_lines // ws
    from sglang_meta_common.ops.comm.flydsl_ar.cpu_reference import all_gather_plan

    plan = all_gather_plan(ws, rank, "package" if package else "flat")
    peers = plan["targets"]
    sib = plan["sibling"]
    sl = max_rows * part_lines
    max_send = max_rows * own_pieces
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
    def gather_ffn_boundary_gfx1250(
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
        inp: fx.Int64,
        stream: fx.Int64,
        alpha: fx.Int64,
        beta: fx.Int64,
        post_w: fx.Int64,
        pre_w: fx.Int64,
        rows: fx.Int32,
        cells: fx.Int64,
        sums: fx.Int64,
        dbg: fx.Int64,
    ):
        lane = fx.Uint32(fx.thread_idx.x)
        blk = fx.Uint32(fx.block_idx.x)
        ns = fx.Uint32(rows) * fx.Uint32(own_pieces)
        nr = fx.Uint32(rows) * fx.Uint32(pieces)
        live = (blk < ns + nr) & (fx.Uint64(inp) != fx.Uint64(0))
        live = live & (fx.Uint64(cells) != fx.Uint64(0)) & (fx.Uint64(out) != fx.Uint64(0))
        if live:
            bases = [base0, base1, base2, base3, base4, base5, base6, base7][:ws]
            is_sender = blk < ns
            # sender: (row, own piece q); receiver: (row, piece) of the whole row
            pos = fx.Uint32(k._select(is_sender, blk, blk - ns))
            per_row = fx.Uint32(k._select(is_sender, fx.Uint32(own_pieces), fx.Uint32(pieces)))
            row = pos // per_row
            piece = pos % per_row
            owner = fx.Uint32(k._select(is_sender, fx.Uint32(rank), piece // fx.Uint32(own_pieces)))
            q = fx.Uint32(k._select(is_sender, piece, piece % fx.Uint32(own_pieces)))
            # this lane's line of the owner's [rows, hidden / ws] columns
            idx = (row * fx.Uint32(own_pieces) + q) * fx.Uint32(_WAVE) + lane
            # both strided by the largest batch's positions, so a position keeps its cells at every batch size
            cell = fx.Uint32(k._select(is_sender, pos, pos + fx.Uint32(max_send)))

            in_rsrc = create_buffer_resource_from_addr(inp)
            cell_rsrc = create_buffer_resource_from_addr(cells)
            epoch = buffer_load(cell_rsrc, cell, vec_width=1, dtype=T.i32)
            mine = owner == fx.Uint32(rank)
            d = buffer_load(in_rsrc, fx.Uint32(k._select(mine, idx, fx.Uint32(0))) * fx.Uint32(4), vec_width=4,
                            dtype=T.i32)
            sched_barrier(0)
            flag = k._next_flag(fx.Uint32(epoch))
            bank = flag & fx.Uint32(1)
            my_rsrc = create_buffer_resource_from_addr(bases[rank])

            def line_el(slot, i):
                return ((bank * fx.Uint32(ws) + slot) * fx.Uint32(sl) + i) * fx.Uint32(4)

            if is_sender:
                enc = k._flagless_encode(d)
                for qi in range_constexpr(len(peers)):
                    buffer_store(enc, create_buffer_resource_from_addr(bases[peers[qi]]),
                                 line_el(fx.Uint32(rank), idx), cache_modifier=cpol)
            else:
                col = (piece * fx.Uint32(_WAVE) + lane) * fx.Uint32(8)
                e = row * fx.Uint32(hidden) + col
                a_rsrc = create_buffer_resource_from_addr(alpha)
                b_rsrc = create_buffer_resource_from_addr(beta)
                s_rsrc = create_buffer_resource_from_addr(stream)
                sum_rsrc = create_buffer_resource_from_addr(sums)
                av = cat8(buffer_load(a_rsrc, col, vec_width=4), buffer_load(a_rsrc, col + fx.Uint32(4), vec_width=4))
                bv = cat8(buffer_load(b_rsrc, col, vec_width=4), buffer_load(b_rsrc, col + fx.Uint32(4), vec_width=4))
                wv = k._upcast_n(buffer_load(create_buffer_resource_from_addr(pre_w), col // fx.Uint32(2),
                                             vec_width=4, dtype=T.i32), 4)
                if const_expr(post_weight):
                    pw = k._upcast_n(buffer_load(create_buffer_resource_from_addr(post_w), col // fx.Uint32(2),
                                                 vec_width=4, dtype=T.i32), 4)
                sv = cat8(buffer_load(s_rsrc, e, vec_width=4), buffer_load(s_rsrc, e + fx.Uint32(4), vec_width=4))
                zero4 = k._from_elements(4, [fx.Uint32(0)] * 4)
                ones4 = k._splat_i32(k._vec_i32(4), k._I32_MINUS1)

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

                x_i32 = k._flagless_canon(d)
                if owner != fx.Uint32(rank):
                    # FlyDSL's adaptive wait on the owner's line: ``wait_tight`` reloads, then lane 0 polls the
                    # wave's first line under the sleep ladder, then reloads until complete
                    el = line_el(owner, idx)

                    def load_line():
                        return buffer_load(my_rsrc, el, vec_width=4, dtype=T.i32, cache_modifier=cpol)

                    line = load_line()
                    pend = k._flagless_lines_pending(line, 1)
                    it = fx.Uint32(0)
                    while pend & (it < fx.Uint32(wait_tight)):
                        k._fence("acquire", "wavefront")
                        line = load_line()
                        pend = k._flagless_lines_pending(line, 1)
                        it = it + fx.Uint32(1)
                    if k._wave_any(pend):
                        saddr = bases[rank] + fx.Uint64(
                            line_el(owner, idx - lane) * fx.Uint32(4)
                        )
                        want = lane == fx.Uint32(0)
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
                        line = load_line()
                        pend = k._flagless_lines_pending(line, 1)
                        it = it + fx.Uint32(1)
                    record(0, pend)
                    if const_expr(package):
                        # a same-parity owner pushes only to this rank of the package: relay to the sibling
                        if ((owner ^ fx.Uint32(rank)) & fx.Uint32(1)) == fx.Uint32(0):
                            buffer_store(line, create_buffer_resource_from_addr(bases[sib]), el,
                                         cache_modifier=cpol)
                    buffer_store(zero4, my_rsrc, el, cache_modifier=cpol)
                    x_i32 = k._flagless_payload(line, 0)
                x8 = k._upcast_n(x_i32, 4)
                if const_expr(post_weight):
                    # the input-scale post-norm's scaled input, in bf16
                    x8 = k._upcast_n(k._downcast_hw_n(fmul(x8, pw), 8), 4)

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
                    return [buffer_load(sum_rsrc, s_el, vec_width=4, dtype=T.i32, cache_modifier=_CPOL_SCOPE_DEV)
                            for s_el in sels]

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
        inp: fx.Int64,
        stream: fx.Int64,
        alpha: fx.Int64,
        beta: fx.Int64,
        post_w: fx.Int64,
        pre_w: fx.Int64,
        rows: fx.Int32,
        cells: fx.Int64,
        sums: fx.Int64,
        dbg: fx.Int64,
        nblocks: fx.Int32,
        stream_: Stream = Stream(None),
    ):
        _ = ("muse_gather_ffn_boundary", ws, rank, hidden, max_rows, post_eps, pre_eps, post_weight, wait_tight,
             debug_cap, package)
        gather_ffn_boundary_gfx1250(
            base0, base1, base2, base3, base4, base5, base6, base7,
            out, new_stream, inp, stream, alpha, beta, post_w, pre_w,
            rows, cells, sums, dbg,
        ).launch(grid=(fx.Index(nblocks), 1, 1), block=(_WAVE, 1, 1), stream=stream_)

    return launch


class FlyDSLGatherFfnBoundary:
    """Peer scratch, epoch cells and the compiled kernel of one TP group, for 1 to ``max_rows`` rows.

    Built collectively (every TP rank in the same order); ``warm`` compiles it, which must not be under graph
    capture."""

    def __init__(self, tp_group, hidden: int, max_rows: int, post_eps: float, pre_eps: float, post_weight: bool):
        from sglang_meta_common.ops.comm.flydsl_ar.peer_memory import make_peer_buffer, select_backend

        self.world_size = ws = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.device = tp_group.device
        self.hidden = hidden
        self.max_rows = max_rows
        self.post_eps, self.pre_eps = float(post_eps), float(pre_eps)
        self.post_weight = post_weight
        arch = torch.cuda.get_device_properties(self.device).gcnArchName
        self._buf = make_peer_buffer(select_backend(arch), _scratch_lines(ws, max_rows, hidden) * _LINE_BYTES,
                                     tp_group.cpu_group, self.device)
        self._buf.local.zero_()
        torch.cuda.synchronize(self.device)
        torch.distributed.barrier(group=tp_group.cpu_group)
        bases = [int(a) for a in self._buf.addresses.cpu().tolist()]
        self._bases = tuple(bases + [0] * (8 - len(bases)))
        self._pieces = hidden // 8 // _WAVE
        self._own_pieces = self._pieces // ws
        self._cells = torch.zeros(max_rows * (self._own_pieces + self._pieces), dtype=torch.int32,
                                  device=self.device)
        self._sums = torch.zeros(_sum_lines(max_rows, hidden) * 4, dtype=torch.int32, device=self.device)
        # > 0: every spin wait gives up after this many rounds and marks its wave's (cell, stage) in ``dbg``
        self._debug_cap = int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_FLYDSL_DEBUG_CAP", "0"))
        self.dbg = (torch.zeros(self._cells.numel() * 3, dtype=torch.int32, device=self.device)
                    if self._debug_cap > 0 else None)
        # batches of at least this many rows take the package all-gather (0: never)
        self.package_min_rows = int(os.environ.get("SGLANG_ROCM_MUSE_GATHER_FFN_PACKAGE_MIN_ROWS", "0"))
        if self.package_min_rows > 0 and not self._compact_placement(tp_group):
            self.package_min_rows = 0
        self._fns = {}

    def _compact_placement(self, tp_group) -> bool:
        """Collective: whether every rank sees ranks 2g, 2g + 1 as the two partitions of package g."""
        import logging

        from sglang_meta_common.ops.comm.flydsl_ar.hier_allreduce import rank_packages
        from sglang_meta_common.ops.comm.flydsl_ar.hier_reference import check_compact

        err = None
        try:
            if self.world_size not in (4, 8):
                raise ValueError(f"world size {self.world_size}")
            check_compact(rank_packages(self.device, tp_group.cpu_group))
        except (RuntimeError, ValueError, OSError) as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        ok = torch.tensor([0 if err else 1], dtype=torch.int32)
        torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN, group=tp_group.cpu_group)
        if not int(ok.item()):
            logging.getLogger(__name__).warning(
                "FFN gather boundary: compact DPX placement not confirmed (%s); flat all-gather", err or "another rank")
            return False
        return True

    def supports(self, num_rows: int) -> bool:
        return 0 < num_rows <= self.max_rows

    def columns(self) -> slice:
        """This rank's output columns of the hidden dimension."""
        n = self.hidden // self.world_size
        return slice(self.rank * n, (self.rank + 1) * n)

    def unpool(self, x, weight):
        """This rank's ``columns()`` of ``x @ weight.T`` (weight [hidden, K] bf16), on FlyDSL GEMM tiles tuned for
        the [hidden / 8, 4096] slice at 1 to 64 rows."""
        from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

        m = x.shape[0]
        tile_m, m_warp, tile_k, split_k = (16, 1, 256, 4) if m <= 16 else (64, 2 if m <= 32 else 4, 128, 8)
        return gemm_a16w16(x, weight[self.columns()], dtype=torch.bfloat16, tile_m=tile_m, tile_n=32,
                           tile_k=tile_k, m_warp=m_warp, n_warp=2, num_buffers=4, split_k=split_k,
                           main_loop_unroll=True)

    def _launch(self, args, package: bool = False):
        import flydsl.compiler as flyc
        import flydsl.expr as fx

        cu_stream = torch.cuda.current_stream(self.device)
        fn = self._fns.get(package)
        if fn is not None:
            fn(*args, cu_stream)
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FFN gather-boundary FlyDSL kernel was not compiled before graph capture")
        launcher = make_gather_ffn_boundary_launcher(
            self.world_size, self.rank, self.hidden, self.max_rows, self.post_eps, self.pre_eps, self.post_weight,
            int(os.environ.get("SGLANG_ROCM_MUSE_AR_BOUNDARY_FLYDSL_TIGHT", "4")), self._debug_cap, package,
        )
        ints = (0,) * 16 + (1, 0, 0, 0, 1)
        typed = tuple(fx.Int32(a) if is32 else fx.Int64(a) for a, is32 in zip(args, ints))
        self._fns[package] = flyc.compile(launcher, *typed, cu_stream)

    def warm(self):
        """Compile now (outside graph capture): a null input makes the compile's launch a no-op."""
        for package in (False, True) if self.package_min_rows > 0 else (False,):
            if package not in self._fns:
                self._launch(self._bases + (0,) * 8 + (1, 0, 0, 0, 1), package)

    def __call__(self, stream, part, post_weight, pre_weight, alpha, beta):
        """(next pre-norm output in bf16, new fp32 stream) of the FFN boundary on the gathered ``part`` columns
        ([rows, hidden / ws] bf16, this rank's ``columns()``)."""
        rows = part.shape[0]
        out = torch.empty(rows, self.hidden, dtype=torch.bfloat16, device=part.device)
        new_stream = torch.empty_like(stream)
        self._launch(self._bases + (
            out.data_ptr(), new_stream.data_ptr(), part.data_ptr(), stream.data_ptr(), alpha.data_ptr(),
            beta.data_ptr(), post_weight.data_ptr() if self.post_weight else 0, pre_weight.data_ptr(),
            rows, self._cells.data_ptr(), self._sums.data_ptr(), 0 if self.dbg is None else self.dbg.data_ptr(),
            rows * (self._own_pieces + self._pieces),
        ), 0 < self.package_min_rows <= rows)
        return out, new_stream
