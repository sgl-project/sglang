# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Histogram-assisted exact FP32 TopK (K=2048) over a fixed 128-CTA grid.

Device primitives originated in NVIDIA GVR2 (TensorRT-LLM PR #17821, ed94d4cfbf).
An invalid certificate or a candidate-capacity overflow raises a device error;
selection within capacity is exact FP32.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator
from cutlass.cute.testing import assert_ as _device_assert

# Workspace bytes: [0, 1280) per-row arrival words (8 B), [1280, 2048) per-row slab cursors (4 B),
# [2048, 3072) per-row split pair words (8 B), [4096, ...) per-row candidate slabs.
WS_CURSOR_OFF = 1280
WS_PAIR_OFF = 2048
WS_SLAB_OFF = 4096
RANK_MAX = 288  # boundary sets up to this size are ranked in O(m^2)
RES_B, RES_M, RES_ABOVE, RES_TOT = 0, 1, 2, 3
U = 4  # float4 tiles per thread in flight during the row pass
NBS = 256  # fine histogram bins

ballot = cute.arch.vote_ballot_sync
popc = cute.arch.popc


def u32_of_f32(v):
    return Uint32(llvm.bitcast(Uint32.mlir_type, v.ir_value()))

def f32_of_i32(i):
    return Float32(llvm.bitcast(Float32.mlir_type, i.ir_value()))

def i32_of_f32(v):
    return Int32(llvm.bitcast(Int32.mlir_type, v.ir_value()))

def fkey_bits(u):
    """Order-preserving uint32 key of raw FP32 bits."""
    neg = Uint32(0) - (u >> Uint32(31))  # 0 or 0xFFFFFFFF
    return u ^ (neg | Uint32(0x80000000))

def clz_i32(x):
    return Int32(cute.arch.clz(x))

def ffs_m1(x):
    """Index of the lowest set bit of x != 0."""
    return Int32(cute.arch.popc((x & (Int32(0) - x)) - 1))

def atomic_add_cta(ptr, val):
    return Int32(cute.arch.atomic_add(ptr, val, sem="relaxed", scope="cta"))

def _inline(ptx, constraints, out=(), effects=True):
    """Inline-PTX op returning one value per cutlass type in out."""
    @dsl_user_op
    def op(*args, loc=None, ip=None):
        types = [t.mlir_type for t in out]
        result = None if not out else types[0] if len(out) == 1 else llvm.StructType.get_literal(types)
        value = llvm.inline_asm(
            result, [a.ir_value(loc=loc, ip=ip) for a in args], ptx, constraints,
            has_side_effects=effects, is_align_stack=None if result is None else False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
        if len(out) == 1:
            return out[0](value)
        return tuple(t(llvm.extractvalue(t.mlir_type, value, [i], loc=loc, ip=ip)) for i, t in enumerate(out))
    return op

def warp_incl_scan_add(val, width=32):
    """Inclusive sum in the first width lanes; other lanes are unspecified."""
    ptx = ["{", ".reg .pred valid;", ".reg .b32 other;", "mov.b32 $0, $1;"]
    for offset in (1, 2, 4, 8, 16):
        if offset < width:
            ptx += [f"shfl.sync.up.b32 other|valid, $0, {offset}, 0, 0xffffffff;", "@valid add.s32 $0, $0, other;"]
    return _inline("\n".join(ptx + ["}"]), "=r,r", (Int32,), effects=False)(val)

def warp_incl_count16(val):
    """Inclusive warp sum of per-lane counts in [0,16] from five bit ballots."""
    bits = range(5)
    ptx = ["{", ".reg .pred p0, p1, p2, p3, p4;", ".reg .b32 b0, b1, b2, b3, b4, t0, t1, t2, t3, t4, lm;"]
    ptx += [f"and.b32 t{b}, $1, {1 << b};" for b in bits]
    ptx += [f"setp.ne.u32 p{b}, t{b}, 0;" for b in bits]
    ptx += [f"vote.ballot.sync.b32 b{b}, p{b}, 0xffffffff;" for b in bits]
    ptx.append("mov.u32 lm, %lanemask_le;")
    ptx += [f"and.b32 b{b}, b{b}, lm;" for b in bits]
    ptx += [f"popc.b32 t{b}, b{b};" for b in bits]
    ptx += [f"mad.lo.u32 t0, t{b}, {1 << b}, t0;" for b in (1, 2, 3)]
    ptx += ["mad.lo.u32 $0, t4, 16, t0;", "}"]
    return _inline("\n".join(ptx), "=r,r", (Int32,), effects=False)(val)

def _prefetch4_uniform(base, start, end, padding, full):
    """Four float4 lines per thread (1024 threads) behind an explicit warp-uniform full/partial branch."""
    ptx = ["{", ".reg .pred clamp;", ".reg .b32 i0, i1, i2, i3;", ".reg .b64 a0, a1, a2, a3;",
           "TARGETS: .branchtargets PARTIAL, FULL;", "brx.idx.uni $20, TARGETS;", "FULL:"]
    for partial in (False, True):
        if partial:
            ptx.append("PARTIAL:")
        for u in range(4):
            ptx.append(f"add.s32 i{u}, $17, {u * 1024};")
            if partial:
                ptx += [f"setp.ge.s32 clamp, i{u}, $18;", f"selp.b32 i{u}, $19, i{u}, clamp;"]
            ptx += [f"mul.wide.s32 a{u}, i{u}, 16;", f"add.u64 a{u}, a{u}, $16;"]
        for u in range(4):
            dst = ", ".join(f"${u * 4 + q}" for q in range(4))
            ptx.append(f"ld.global.nc.v4.f32 {{{dst}}}, [a{u}];")
        if not partial:
            ptx.append("bra.uni END;")
    ptx += ["END:", "}"]
    return _inline("\n".join(ptx), ",".join(["=f"] * 16 + ["l", "r", "r", "r", "r"]),
                   (Float32,) * 16)(base, start, end, padding, full)

_prefetch_l2 = _inline('prefetch.global.L2 [$0];', 'l')
# The asm boundary keeps four scalar f32 results: NVVM would otherwise merge adjacent 128-bit
# loads into v2.b64 register pairs and spill at the 64-register limit.
_ld_g_nc_v4_f32 = _inline('ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];', '=f,=f,=f,=f,l', (Float32,) * 4, effects=False)
# Single-copy-atomic 8-byte slab entry -> (value bits, index).
_ld_slab = _inline('{ .reg .b64 t; ld.relaxed.gpu.global.b64 t, [$2]; mov.b64 {$0, $1}, t; }',
                   '=r,=r,l', (Int32, Int32))
# Split-mode slab values lie in [TF, HIC) with finite, nonzero, same-signed endpoints, so a
# published entry never has a zero value word; zero is the at-rest sentinel and is polled.
_ld_slab_poll = _inline('{ .reg .pred p; .reg .b64 t; SLAB_POLL: ld.relaxed.gpu.global.b64 t, [$2]; '
                        'mov.b64 {$0, $1}, t; setp.eq.u32 p, $0, 0; @p bra SLAB_POLL; }', '=r,=r,l', (Int32, Int32))
_ld_relaxed_u64 = _inline('ld.relaxed.gpu.global.b64 $0, [$1];', '=l,l', (Int64,))
_ld_relaxed_u64_until = _inline('{ .reg .pred p; .reg .b64 t; PAIR_POLL: ld.relaxed.gpu.global.b64 t, [$0]; '
                                'setp.ne.b64 p, t, $1; @p bra PAIR_POLL; }', 'l,l')
_st_relaxed_u64 = _inline('st.relaxed.gpu.global.b64 [$0], $1;', 'l,l')
_lds_v2_u64 = _inline('ld.shared.v2.u64 {$0, $1}, [$2];', '=l,=l,r', (Uint64, Uint64))
# Two 32-bit registers, coalesced by ptxas straight from the survivor walk's (value, index) pair.
_st_s_v2_u32 = _inline('st.shared.v2.u32 [$0], {$1, $2};', 'r,r,r')
_cp_async4 = _inline('cp.async.ca.shared.global [$0], [$1], 4;', 'r,l,~{memory}')
_cp_async_wait_all = _inline('cp.async.wait_all;', '~{memory}')
# Opaque identities: NVVM cannot rematerialize the value (or a shared-memory base) inside loops.
_pin_i64 = _inline('mov.b64 $0, $1;', '=l,l', (Int64,), effects=False)
_pin_i32 = _inline('mov.b32 $0, $1;', '=r,r', (Int32,), effects=False)
_red_shared_add1 = _inline('red.relaxed.cta.shared.add.u32 [$0], 1;', 'r')


def ld_g_f32x4(base_addr, v_idx, frag):
    """frag[0:4] = float4 #v_idx of a global row."""
    for q, v in enumerate(_ld_g_nc_v4_f32(base_addr + Int64(v_idx) * 16)):
        frag[q] = v

def ldg_f32(base_addr, idx, stride=4):
    """Read-only scalar FP32 gather. A byte stride loaded from shared memory stays in a
    register; an immediate one is re-materialized inside the 512-thread survivor walk."""
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), Float32, num_bits_per_copy=32, invariant=True)
    p = cute.make_ptr(Float32, base_addr + Int64(idx) * Int64(stride), cute.AddressSpace.gmem, assumed_align=4)
    frag = cute.make_rmem_tensor((1,), Float32)
    cute.copy(atom, cute.make_tensor(p, cute.make_layout((1,))), frag)
    return frag[0]

def _st_global(addr_i64, value):
    """Plain global store of a Uint64/Int32 value."""
    p = cute.make_ptr(type(value), addr_i64, cute.AddressSpace.gmem, assumed_align=type(value).width // 8)
    cute.make_tensor(p, cute.make_layout((1,)))[0] = value

def _alloc(smem, dtype, size, align):
    return smem.allocate_tensor(dtype, cute.make_ordered_layout((size,), order=(0,)), byte_alignment=align)

def _smem_view(dtype, addr, size):
    return cute.make_tensor(cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=16),
                            cute.make_layout((size,)))

@cute.jit
def scan_cross0(s_hist, target, tidx, s_res, zero: cutlass.Constexpr):
    """Warp 0 finds the 256-bin crossing of target, counting from the top bin.

    zero=True clears the bins; otherwise each bin becomes the count strictly above it."""
    BPT = NBS // 32  # bins per lane
    NV = BPT // 4  # 16B vectors per lane
    if tidx < 32:
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128)
        hbase = s_hist.iterator.toint()
        frags = [cute.make_rmem_tensor((4,), Int32) for _ in range(NV)]
        sm = Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (tidx * Int32(NV) + q) * 16
            cute.copy(atom, _smem_view(Int32, hbase + boff, 4), frags[q])
            sm = sm + frags[q][0] + frags[q][1] + frags[q][2] + frags[q][3]
        w = warp_incl_scan_add(sm)
        tot = cute.arch.shuffle_sync(w, Int32(31))
        after = tot - w  # bins strictly above my span
        if tidx == 0:
            s_res[RES_TOT] = tot
        base = tidx * Int32(BPT)
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            vv = frags[q]
            o4 = cute.make_rmem_tensor((4,), Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = vv[j]
                if cutlass.const_expr(zero):
                    o4[j] = Int32(0)
                else:
                    o4[j] = after
                gb = base + Int32(4 * q + j)
                cross = Int32(0)
                if after < target:
                    if (after + cq) >= target:
                        cross = Int32(1)
                    if gb == 0:
                        cross = Int32(1)
                if cross != 0:
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                after = after + cq
            boff = (tidx * Int32(NV) + q) * 16
            cute.copy(atom, o4, _smem_view(Int32, hbase + boff, 4))

@cute.jit
def coarse_floor(tau):
    # Exact inverse of FP16-RN low bins plus the unit-width high bins.
    result = Float32(float("inf"))
    if tau >= 1023:
        result = Float32(float("-inf"))
    elif tau >= 0:
        magnitude = Int32(511) - tau
        if tau > 511:
            magnitude = tau - 511
        value = Float32(0.0)
        if magnitude > 304:
            value = Float32(magnitude - 288)
        elif magnitude > 16:
            value = f32_of_i32(((magnitude + 1792) << 19) - 4096)
        elif magnitude > 0:
            value = Float32((magnitude << 7) - 1) * Float32(2.0 ** -25)
        if tau > 511:
            value = -value
            if magnitude <= 304:
                value = f32_of_i32(i32_of_f32(value) - 1)
        result = value
    return result


def fine_bin(x, TF, SC):
    return Int32(cute.arch.fmin((x - TF) * SC, Float32(NBS - 1)))

def fine_hist_add(hb_pin, xbits, TF, SC):
    _red_shared_add1(hb_pin + (fine_bin(f32_of_i32(xbits), TF, SC) << 2))

def physical(ptable, tbase, idv):
    """Physical KV slot of logical index idv (64-token pages)."""
    return ptable[tbase + (idv >> 6)] * 64 + (idv & 63)

@cute.jit
def staged(fromg, gbuf_row, s_cbuf2, i):
    """(FP32 bits, index) of staged candidate i, from the global slab or shared memory."""
    vx = Int32(0)
    vy = Int32(0)
    if fromg != 0:
        vx, vy = _ld_slab(gbuf_row + Int64(i) * 8)
    else:
        pk64 = s_cbuf2[i]
        vx = Int32(Uint32(pk64 & 0xFFFFFFFF))
        vy = Int32(pk64 >> 32)
    return vx, vy


class HistogramSelector:
    """One build per route: blk=512 owns 1..8 active rows, blk=1024 owns the rest. Both run on the
    same fixed 128-CTA grid; the route that does not own the active batch exits immediately."""

    def __init__(self, blk: int, capacity: int = 32768):
        assert blk in (512, 1024)
        self.blk = blk
        self.capacity = capacity
        self.scpb = 8192 if blk == 1024 else 4096  # staged candidates per CTA
        self.cmpb = 4096 if blk == 1024 else 1024  # boundary-bin candidates per row
        # Shared candidate blob: int2 staged candidates, then boundary keys.
        self.ck_off = 8 * (self.scpb + 4)
        self.dyn_bytes = self.ck_off + (self.cmpb + 1) * 8

    @cute.jit
    def _emitc(self, xv, idx, pos, cb2):
        """Stage (FP32 bits, index) at slot min(pos, scpb); returns pos + 1."""
        ps = pos
        if ps > self.scpb:
            ps = Int32(self.scpb)
        _st_s_v2_u32(cb2 + ps * 8, u32_of_f32(xv), Uint32(idx))
        return pos + 1

    @cute.jit
    def _zero_hist(self, histogram, hbase, tidx):
        hz = tidx
        while hz < 1024:
            histogram[hbase + hz] = Int32(0)
            hz = hz + self.blk

    @cute.jit
    def _load_tile(self, x_addr, base, tidx, c1, lim4, full, pf):
        """pf[u] = float4 line base + tidx + u*blk; unless full, lines at or past c1 read line lim4."""
        if full != 0:
            for uu in cutlass.range_constexpr(U):
                ld_g_f32x4(x_addr, base + tidx + uu * self.blk, pf[uu])
        else:
            for uu in cutlass.range_constexpr(U):
                j = base + tidx + uu * self.blk
                if j >= c1:
                    j = lim4
                ld_g_f32x4(x_addr, j, pf[uu])

    @cute.jit
    def _prime(self, x_addr, c0, c1, lim4, tidx, pf):
        """Issue this CTA's first row-pass tile into pf."""
        full = Int32((c1 - c0) >= Int32(self.blk * U))
        self._load_tile(x_addr, c0, tidx, c1, lim4, full, pf)

    @cute.jit
    def _split_partition(self, local: cutlass.Constexpr, myn, tidx, lane, s_cbuf2, HICs, ptable, tbase,
                         s_scal, out_row, k, s_ck64, pair_addr, gbuf_row):
        """Split mode: staged candidates >= HICs are strict and go straight to the output; the rest
        go to the boundary slab (s_ck64 for a single-CTA row, else the row's global slab)."""
        BLK = self.blk
        if cutlass.const_expr(not local):
            pgo64 = cute.make_ptr(Int64, pair_addr, cute.AddressSpace.gmem, assumed_align=8)
        it2 = (myn + BLK - 1) // BLK
        it = Int32(0)
        while it < it2:
            i = it * BLK + tidx
            p1 = Int32(0)
            p2 = Int32(0)
            w64 = Uint64(0)
            idv = Int32(0)
            if i < myn:
                w64 = Uint64(s_cbuf2[i])
                xb = Int32(Uint32(w64 & 0xFFFFFFFF))
                idv = Int32(Uint32(w64 >> 32))
                if f32_of_i32(xb) >= HICs:
                    p1 = Int32(1)
                else:
                    p2 = Int32(1)
            # Unconditional page gather (idv=0 reads this row's first page) before the reservation
            # atomic so both round trips overlap; a predicated load would stall next to pg*64.
            pg = ptable[tbase + (idv >> 6)]
            n1 = ballot(p1 != 0)
            n2 = ballot(p2 != 0)
            bhi = Int32(0)
            blo = Int32(0)
            if lane == 0:
                if cutlass.const_expr(local):
                    if n1 != 0:
                        bhi = atomic_add_cta(s_scal.iterator + 1, Int32(popc(n1)))
                    if n2 != 0:
                        blo = atomic_add_cta(s_scal.iterator + 2, Int32(popc(n2)))
                else:
                    # One u64 RMW per warp reserves strict (high word) and slab (low word) slots.
                    if (n1 | n2) != 0:
                        oldv = Int64(cute.arch.atomic_add(pgo64, (Int64(popc(n1)) << 32) + Int64(popc(n2))))
                        bhi = Int32(oldv >> 32)
                        blo = Int32(oldv & 0xFFFFFFFF)
                        if n2 != 0:
                            atomic_add_cta(s_scal.iterator + 2, Int32(popc(n2)))
            bhi = cute.arch.shuffle_sync(bhi, Int32(0))
            blo = cute.arch.shuffle_sync(blo, Int32(0))
            lm = Int32(cute.arch.lanemask_lt())
            if p1 != 0:
                p = bhi + Int32(popc(n1 & lm))
                if p < k:
                    out_row[p] = pg * 64 + (idv & 63)
            if p2 != 0:
                p = blo + Int32(popc(n2 & lm))
                if cutlass.const_expr(local):
                    if p < self.cmpb:
                        s_ck64[p] = w64
                else:
                    if p < self.capacity:
                        _st_relaxed_u64(gbuf_row + Int64(p) * 8, w64)
            it = it + 1

    @cute.jit
    def _narrow_and_emit(self, count, k, tidx, s_hist, s_res, s_scal, lane, out_row, ptable, tbase,
                         gbuf_row, s_cbuf2, fromg, s_ck64, s_kmm, soff, above, source_mode: cutlass.Constexpr):
        """Exact selection by key-space radix narrowing. source_mode 1: the whole staged list
        (FP32 keys, eight levels); 2: the packed boundary keys in s_ck64 (six levels)."""
        BLK = self.blk
        above2 = Int32(0)
        need2 = k
        m2 = count
        if cutlass.const_expr(source_mode == 2):
            rlo = s_kmm[0]
            rhi = s_kmm[1]
            ethr = Int64(Uint32(rlo))
        else:
            rlo = Uint32(0)
            rhi = Uint32(0xFFFFFFFF)
            ethr = Int64(0)
            tie_m = Int32(1)
            if tidx < NBS:
                s_hist[tidx] = Int32(0)
            cute.arch.barrier()
        brk = Int32(0)
        lev = Int32(0)
        while brk == 0:
            if need2 == m2:
                ethr = Int64(Uint32(rlo)) - 1
                above2 = above2 + m2
                need2 = Int32(0)
                if cutlass.const_expr(source_mode != 2):
                    tie_m = Int32(0)
                brk = Int32(1)
            elif Uint32(rlo) >= Uint32(rhi):
                ethr = Int64(Uint32(rlo))
                brk = Int32(1)
            elif lev >= Int32(6 if source_mode == 2 else 8):
                ethr = Int64(Uint32(rlo))
                brk = Int32(1)
            else:
                d2 = Uint32(rhi) - Uint32(rlo)
                sh2 = Int32(32) - clz_i32(Int32(d2 | Uint32(1))) - 8
                if sh2 < 0:
                    sh2 = Int32(0)
                sh2u = Uint32(sh2)
                i = tidx
                while i < count:
                    if cutlass.const_expr(source_mode == 1):
                        vx, vy = staged(fromg, gbuf_row, s_cbuf2, i)
                        uq = fkey_bits(Uint32(vx))
                    else:
                        uq = Uint32(s_ck64[i] >> 32)
                    if uq >= Uint32(rlo):
                        if uq <= Uint32(rhi):
                            du = (uq - Uint32(rlo)) >> sh2u
                            if du > Uint32(NBS - 1):
                                du = Uint32(NBS - 1)
                            atomic_add_cta(s_hist.iterator + Int32(du), Int32(1))
                    i = i + BLK
                cute.arch.barrier()
                scan_cross0(s_hist, need2, tidx, s_res, zero=True)
                cute.arch.barrier()
                above2 = above2 + s_res[RES_ABOVE]
                need2 = need2 - s_res[RES_ABOVE]
                m2 = s_res[RES_M]
                sB = s_res[RES_B]
                nlo = Uint32(rlo) + (Uint32(sB) << sh2u)
                if sB != NBS - 1:
                    rhi = nlo + ((Uint32(1) << sh2u) - Uint32(1))
                rlo = nlo
                lev = lev + 1
        if tidx == 0:
            s_scal[1] = Int32(0)
            s_scal[2] = Int32(0)
        cute.arch.barrier()
        # Strict survivors fill nA slots from base1; boundary ties fill the next nT.
        if cutlass.const_expr(source_mode == 2):
            base1 = soff + above
            nA = above2
            nT = need2
        else:
            base1 = soff
            nA = k
            nT = Int32(0)
            if tie_m != 0:
                nA = above2
                nT = need2
        it2 = (count + BLK - 1) // BLK
        it = Int32(0)
        while it < it2:
            i = it * BLK + tidx
            p1 = Int32(0)
            p2 = Int32(0)
            idv = Int32(0)
            if i < count:
                if cutlass.const_expr(source_mode == 1):
                    vx, idv = staged(fromg, gbuf_row, s_cbuf2, i)
                    iu = Int64(fkey_bits(Uint32(vx)))
                else:
                    w64 = s_ck64[i]
                    iu = Int64(Uint32(w64 >> 32))
                    idv = Int32(Uint32(w64 & 0xFFFFFFFF))
                if iu > ethr:
                    p1 = Int32(1)
                if cutlass.const_expr(source_mode == 2):
                    if iu == ethr:
                        p2 = Int32(1)
                else:
                    if tie_m != 0:
                        if iu == ethr:
                            p2 = Int32(1)
            n1 = ballot(p1 != 0)
            n2 = ballot(p2 != 0)
            b1 = Int32(0)
            b2 = Int32(0)
            if lane == 0:
                if n1 != 0:
                    b1 = atomic_add_cta(s_scal.iterator + 1, Int32(popc(n1)))
                if n2 != 0:
                    b2 = atomic_add_cta(s_scal.iterator + 2, Int32(popc(n2)))
            b1 = cute.arch.shuffle_sync(b1, Int32(0))
            b2 = cute.arch.shuffle_sync(b2, Int32(0))
            lm = Int32(cute.arch.lanemask_lt())
            if p1 != 0:
                p = b1 + Int32(popc(n1 & lm))
                if p < nA:
                    out_row[base1 + p] = physical(ptable, tbase, idv)
            if p2 != 0:
                p = b2 + Int32(popc(n2 & lm))
                if p < nT:
                    out_row[base1 + nA + p] = physical(ptable, tbase, idv)
            it = it + 1

    @cute.kernel
    def kern(self, logits: cute.Tensor, out: cute.Tensor, ws: cute.Tensor, n: Int32, kv_lens: cute.Tensor,
             histogram: cute.Tensor, diagnostics: cute.Tensor, ptable: cute.Tensor, pcols: Int32,
             active_batch: cute.Tensor):
        npad = Int32(logits.shape[1])
        k = Int32(2048)
        active = active_batch[0]
        route_small = Int32(0)
        if active > 0 and active <= 8:
            if active <= Int32(logits.shape[0]):
                route_small = Int32(1)
        if route_small == Int32(1 if self.blk == 512 else 0):
            BLK = self.blk
            SCPB = self.scpb
            CMPB = self.cmpb
            NW = BLK // 32
            tidx, _, _ = cute.arch.thread_idx()
            bx, _, _ = cute.arch.block_idx()
            lane = tidx & 31
            valid_active = active
            if active < 0 or active > Int32(logits.shape[0]):
                valid_active = Int32(0)
            # Each row owns 128 >> ceil(log2(active)) CTA slots and uses R <= that many parts,
            # chosen from its own length below.
            part_shift = Int32(0)
            if valid_active > 0:
                if valid_active <= 2:
                    part_shift = Int32(6)
                elif valid_active <= 8:
                    part_shift = Int32(4)
                elif valid_active <= 16:
                    part_shift = Int32(3)
                elif valid_active <= 32:
                    part_shift = Int32(2)
                elif valid_active <= 64:
                    part_shift = Int32(1)
                if cutlass.const_expr(self.blk == 512):
                    # One row splits 128 ways, 3..4 rows 32 ways.
                    if valid_active <= 1:
                        part_shift = Int32(7)
                    elif valid_active > 2 and valid_active <= 4:
                        part_shift = Int32(5)
            row = bx >> part_shift
            part = bx & ((Int32(1) << part_shift) - 1)
            if cutlass.const_expr(self.blk == 512):
                if part_shift == 5:
                    # Parts 0..15 of row r run on CTAs 16r..16r+15 (the 16-slot layout) and parts
                    # 16..31 on 64+16r..64+16r+15: spreading them makes the tail placement-sensitive.
                    row = (bx >> 4) & 3
                    part = ((bx >> 6) << 4) | (bx & 15)
            inactive = Int32(row >= valid_active)

            # Storage rows past the active batch are cleared; a CTA may also select an active row.
            if bx >= valid_active and bx < Int32(logits.shape[0]):
                clear_row = bx
                if tidx < 6:
                    diagnostics[clear_row, tidx] = Int32(-2)
                i = tidx
                while i < k:
                    out[clear_row, i] = Int32(-1)
                    i = i + BLK
                self._zero_hist(histogram, clear_row * 1024, tidx)
                if tidx == 0:
                    ws[clear_row * 2] = Int32(0)
                    ws[clear_row * 2 + 1] = Int32(0)
                    ws[Int32(WS_CURSOR_OFF // 4) + clear_row] = Int32(0)
                    tail_word = Int32(WS_PAIR_OFF // 4) + clear_row * 2
                    ws[tail_word] = Int32(0)
                    ws[tail_word + 1] = Int32(0)

            # Row length, clamped to the logical envelope n (npad is the allocation stride). Rows
            # with n <= k run a zero-work pass on part 0 and emit all indices in the epilogue.
            short = Int32(0)
            alive = Int32(0)
            n_row = Int32(0)
            if inactive == 0:
                n_row = kv_lens[row]
            if n_row < 0:
                n_row = Int32(0)
            if n_row > n:
                n_row = n
            Q = Int32(0)
            row_shift = Int32(0)
            if n_row <= k:
                short = Int32(1)
                n = Int32(0)
            else:
                n = n_row
                n4v = n >> 2
                # At most one part per 512 float4s (512 threads, at least 16 parts) or per 1024
                # float4s (1024 threads, one part below 2^14 float4s): thinner slices only add
                # arrivals, reservations and histogram reads to the tail.
                chunks = n4v >> Int32(9 if self.blk == 512 else 10)
                if chunks > 0:
                    row_shift = Int32(31) - clz_i32(chunks)
                if cutlass.const_expr(self.blk == 512):
                    if row_shift < 4:
                        row_shift = Int32(4)
                if row_shift > part_shift:
                    row_shift = part_shift
                if cutlass.const_expr(self.blk != 512):
                    if n4v < Int32(1 << 14) and row_shift == 1:
                        row_shift = Int32(0)
                Q = (n4v + (Int32(1) << row_shift) - 1) >> row_shift
            R = Int32(1) << row_shift
            if part >= R:
                inactive = Int32(1)

            smem = SmemAllocator()
            s_hist = _alloc(smem, Int32, NBS, 128)
            s_res = _alloc(smem, Int32, 8, 16)  # RES_B/M/ABOVE/TOT
            s_scal = _alloc(smem, Int32, 4, 16)  # [0] staged count, [1]/[2] emit cursors, [3] slab base
            s_pk = _alloc(smem, Int64, 1, 8)  # arrival word returned to the CTA
            if cutlass.const_expr(self.blk == 512):
                s_x4 = _alloc(smem, Int32, 1, 4)  # gather byte stride, read back from shared memory
            s_kmm = _alloc(smem, Uint32, 2, 8)  # boundary key min/max
            s_bound = _alloc(smem, Int32, 4, 16)  # certificate: valid, lower, upper, expected count
            s_cert = _alloc(smem, Int32, 4, 16)  # split flag, split need, strict count S, part count R
            sbase = _alloc(smem, cutlass.Int8, self.dyn_bytes, 16).iterator.toint()
            s_cbuf = _smem_view(Int32, sbase, SCPB + 4)
            s_cbuf2 = _smem_view(Uint64, sbase, SCPB + 4)
            ck_addr = sbase + Int32(self.ck_off)
            s_ck64 = _smem_view(Uint64, ck_addr, CMPB + 1)
            # Page bases of each thread's first two slab entries (split mode) reuse the top BLK
            # staged slots, which are free whenever listN <= PFMAX.
            PFMAX = SCPB + 4 - BLK
            EARLY = 4 if self.blk == 512 else 2  # slab entries per thread loaded right after arrival
            s_pfp = _smem_view(Int32, sbase + 8 * PFMAX, 2 * BLK)

            if inactive == 0:
                # Registers pinned outside the loops; otherwise NVVM re-derives each base (shared
                # window symbol, ld.param + ctaid chains) inside every region that uses it.
                hb_pin = _pin_i32(s_hist.iterator.toint())
                cb2_pin = _pin_i32(s_cbuf2.iterator.toint())
                row64 = Int64(row)
                x_addr = _pin_i64(logits.iterator.toint() + row64 * Int64(npad) * 4)
                out_row = out[row, None]
                tbase = row * pcols  # the producer has validated every active physical page
                ws_addr = ws.iterator.toint()
                goff_addr = ws_addr + Int64(WS_CURSOR_OFF)
                gbuf_row = _pin_i64(ws_addr + Int64(WS_SLAB_OFF) + row64 * Int64(self.capacity) * 8)
                hbase = row * 1024  # this row's slice of the producer's int32[B, 1024] histogram
                pair_addr = ws_addr + Int64(WS_PAIR_OFF) + row64 * 8

                # Histogram bins and the first row-pass tile depend only on the row geometry;
                # both loads start before any barrier.
                bins_per_thread = 1024 // BLK
                hh = cute.make_rmem_tensor((bins_per_thread,), Int32)
                for hi in cutlass.range_constexpr(bins_per_thread):
                    hh[hi] = histogram[hbase + tidx * bins_per_thread + hi]
                n4 = n >> 2
                c0 = part * Q
                c1 = c0 + Q
                if c1 > n4:
                    c1 = n4
                tail0 = n4 << 2
                tailn = Int32(0)
                if part == 0:
                    tailn = n - tail0
                lim4 = (npad >> 2) - 1
                if cutlass.const_expr(self.blk == 512):
                    # Out-of-slice loads reuse this CTA's last live line: the padded row end is
                    # never written by the producer, so clamping there costs a DRAM miss.
                    lim4 = c1 - 1
                    if lim4 < 0:
                        lim4 = Int32(0)
                pf = [cute.make_rmem_tensor((4,), Float32) for _ in range(U)]
                if cutlass.const_expr(self.blk == 512):
                    self._prime(x_addr, c0, c1, lim4, tidx, pf)
                else:
                    # Register-free L2 hints now and the tile loads after barrier B: holding pf
                    # through the certificate is slower with 1024 threads.
                    for uu in cutlass.range_constexpr(U):
                        pic = c0 + tidx + uu * BLK
                        if pic >= c1:
                            pic = lim4
                        _prefetch_l2(x_addr + Int64(pic) * 16)

                # Defaults published by barrier A; the thread owning the crossing bin overwrites
                # s_bound and s_cert[0:2] before barrier B.
                if cutlass.const_expr(self.blk == 512):
                    if tidx == 0:
                        s_x4[0] = Int32(4)
                if tidx == 0:
                    s_scal[0] = Int32(0)
                    s_scal[1] = Int32(0)
                    s_scal[2] = Int32(0)
                    s_bound[0] = Int32(0)
                    s_bound[1] = i32_of_f32(Float32(0.0))
                    s_bound[2] = i32_of_f32(Float32(0.0))
                    s_bound[3] = Int32(0)
                    s_cert[0] = Int32(0)
                    s_cert[1] = Int32(0)
                    s_cert[2] = Int32(0)
                    s_cert[3] = R
                if tidx < NBS:
                    s_hist[tidx] = Int32(0)

                # Exact certificate from the producer's coarse histogram; the candidate blob holds
                # the NW warp sums during the prologue.
                hs = Int32(0)
                for hi in cutlass.range_constexpr(bins_per_thread):
                    if short != 0:
                        hh[hi] = Int32(0)
                    hs = hs + hh[hi]
                hincl = warp_incl_scan_add(hs)
                if lane == 31:
                    s_cbuf[tidx >> 5] = hincl
                cute.arch.barrier()  # A
                x4_pin = 4
                if cutlass.const_expr(self.blk == 512):
                    x4_pin = s_x4[0]  # opaque to ptxas, so the stride stays in a register
                # Every warp scans the NW warp totals (exact: a row holds at most 1M scores).
                hwarp = tidx >> 5
                warp_total = Int32(0)
                if lane < NW:
                    warp_total = s_cbuf[lane]
                warp_inclusive = warp_incl_scan_add(warp_total, NW)
                hbefore = cute.arch.shuffle_sync(warp_inclusive - warp_total, hwarp)
                cert_total = cute.arch.shuffle_sync(warp_inclusive, Int32(NW - 1))
                # The crossing is the first bin whose inclusive prefix exceeds k.
                running = hbefore + hincl - hs
                has_cross = Int32(0)
                cert_bin = Int32(0)
                cert_strict = Int32(0)
                cert_count = Int32(0)
                if short == 0:
                    for hi in cutlass.range_constexpr(bins_per_thread):
                        after = running + hh[hi]
                        if running <= k and k < after:
                            has_cross = Int32(1)
                            cert_bin = tidx * bins_per_thread + hi
                            cert_strict = running
                            cert_count = hh[hi]
                        running = after
                if has_cross != 0:
                    eligible = Int32(0)
                    split_flag = Int32(0)
                    split_need = Int32(0)
                    cert_lower = Float32(0.0)
                    cert_upper = Float32(0.0)
                    cert_expected = Int32(0)
                    if cert_total == n:
                        if cert_bin >= 0 and cert_bin <= 1023 and cert_strict >= 0 and cert_strict <= k:
                            cert_quota = k - cert_strict
                            if (cert_count > 0 and cert_count <= n and cert_strict + cert_count > k
                                    and cert_strict + cert_count <= n):
                                cert_effective = cert_bin
                                cert_expected = cert_strict + cert_count
                                if cert_quota == 0:
                                    cert_effective = cert_bin - 1
                                    cert_expected = cert_strict
                                cert_lower = coarse_floor(cert_effective)
                                cert_upper = coarse_floor(cert_effective - 1)
                                lower_abs = u32_of_f32(cert_lower) & Uint32(0x7FFFFFFF)
                                upper_abs = u32_of_f32(cert_upper) & Uint32(0x7FFFFFFF)
                                # Infinite endpoints give SC=0 below: every candidate shares one
                                # fine bin and exact refinement handles it.
                                if cert_upper > cert_lower:
                                    eligible = Int32(1)
                                    # Signed-zero comparisons can admit extra ties, so the
                                    # strict/slab split is never used at a zero or infinite bound.
                                    if (lower_abs > 0 and upper_abs > 0 and lower_abs < Uint32(0x7F800000)
                                            and upper_abs < Uint32(0x7F800000)):
                                        if cert_expected <= SCPB and cert_quota > 0:
                                            split_flag = Int32(1)
                                            split_need = cert_quota
                    s_bound[0] = eligible
                    s_bound[1] = i32_of_f32(cert_lower)
                    s_bound[2] = i32_of_f32(cert_upper)
                    s_bound[3] = cert_expected
                    s_cert[0] = split_flag
                    s_cert[1] = split_need
                cute.arch.barrier()  # B
                cert_fast = s_bound[0]
                TF = f32_of_i32(s_bound[1])
                HIC = f32_of_i32(s_bound[2])
                expected_pass = s_bound[3]
                split_mode = s_cert[0]
                HICs = HIC  # split bound: candidates >= HICs are strict
                if s_cert[1] == 0:
                    HICs = TF
                soff = Int32(0)
                ksc = k
                if cutlass.const_expr(self.blk != 512):
                    self._prime(x_addr, c0, c1, lim4, tidx, pf)

                # Reject the coarse population before any candidate write; the first loads are in flight.
                if short == 0:
                    _device_assert(cert_fast != 0,
                        "LiteTopK coarse certificate is invalid (histogram/count mismatch, e.g. a NaN score)")
                    _device_assert(expected_pass <= self.capacity,
                        f"LiteTopK coarse candidate capacity ({self.capacity}) exceeded; rebuild with a larger --candidate-capacity")
                if tidx == 0 and part == 0:
                    diagnostics[row, 0] = cert_fast
                    diagnostics[row, 1] = n_row
                    diagnostics[row, 2] = expected_pass if cert_fast != 0 else Int32(-1)
                    diagnostics[row, 3] = i32_of_f32(TF)
                    diagnostics[row, 4] = i32_of_f32(HIC)
                    diagnostics[row, 5] = split_mode

                listN = Int32(0)
                above = Int32(0)
                m = Int32(0)
                need = Int32(0)
                B = Int32(0)
                valid = Int32(0)  # the staged population holds at least k candidates
                fromg = Int32(0)  # refinement reads the global slab
                alive = Int32(1)  # this CTA refines the row
                pf_lim = Int32(0)  # emit-pass entries below this have their page base in s_pfp
                WD = (HIC - TF) * Float32(1.0 / 256.0)
                if WD <= Float32(0.0):
                    WD = Float32(1e-30)
                if cutlass.const_expr(self.blk == 512):
                    SC = cute.arch.rcp_approx(WD)
                else:
                    SC = Float32(1.0) / WD

                # ---- Row pass: stage every score >= TF of this CTA's slice ----
                span = c1 - c0
                step = Int32(BLK * U)
                nFull = Int32(0)
                rem = Int32(0)
                if span > 0:
                    nFull = span // step
                    rem = span - nFull * step
                # Pinned: otherwise NVVM re-derives the division chain in the loop condition.
                nFull = _pin_i32(nFull)
                nIt = nFull
                if rem > 0:
                    nIt = nIt + 1
                nIt = _pin_i32(nIt)
                it = Int32(0)
                # 1024 threads peel the full tiles (phase 0) from the last one or two (phase 1).
                for phase in cutlass.range_constexpr(2 if self.blk == 1024 else 1):
                    phase_end = nIt
                    if cutlass.const_expr(self.blk == 1024 and phase == 0):
                        phase_end = nFull - 1
                    while it < phase_end:
                        i0 = c0 + it * step + tidx
                        M = Int32(0)
                        isfull = Int32(1)
                        if cutlass.const_expr(self.blk != 1024 or phase != 0):
                            isfull = Int32(it < nFull)
                        if isfull != 0:
                            for uu in cutlass.range_constexpr(U):
                                for q in cutlass.range_constexpr(4):
                                    M = M | (Int32(pf[uu][q] >= TF) << Int32(uu * 4 + q))
                        else:
                            for uu in cutlass.range_constexpr(U):
                                okq = Int32(i0 + uu * BLK < c1)
                                if okq != 0:
                                    for q in cutlass.range_constexpr(4):
                                        M = M | (Int32(pf[uu][q] >= TF) << Int32(uu * 4 + q))
                        # Next tile's loads go out before the reservation and the survivor walk.
                        hasnext = Int32(1)
                        if cutlass.const_expr(self.blk != 1024 or phase != 0):
                            hasnext = Int32(it + 1 < nIt)
                        if hasnext != 0:
                            j0 = i0 + step
                            infull = Int32(it + 1 < nFull)
                            if cutlass.const_expr(self.blk == 1024 and phase == 0):
                                for uu in cutlass.range_constexpr(U):
                                    ld_g_f32x4(x_addr, j0 + uu * BLK, pf[uu])
                            elif cutlass.const_expr(self.blk == 1024):
                                values = _prefetch4_uniform(x_addr, j0, c1, lim4, infull)
                                for uu in cutlass.range_constexpr(4):
                                    for qq in cutlass.range_constexpr(4):
                                        pf[uu][qq] = values[uu * 4 + qq]
                            else:
                                self._load_tile(x_addr, i0, step, c1, lim4, infull, pf)
                        # Warp-aggregated reservation.
                        cnt = Int32(popc(M))
                        if cutlass.const_expr(self.blk == 512):
                            inc = warp_incl_count16(cnt)
                        else:
                            inc = warp_incl_scan_add(cnt)
                        bpos = Int32(0)
                        if lane == 31:
                            if inc != 0:
                                bpos = atomic_add_cta(s_scal.iterator + 0, inc)
                        pos = cute.arch.shuffle_sync(bpos, Int32(31)) + (inc - cnt)
                        # Survivor walk, pipelined one deep; survivors are re-read from global
                        # memory because holding the U float4s spills.
                        if M != 0:
                            bp = ffs_m1(M)
                            M = M & (M - 1)
                            idx = ((i0 + (bp >> 2) * BLK) << 2) + (bp & 3)
                            xv = ldg_f32(x_addr, idx, x4_pin)
                            while M != 0:
                                bp2 = ffs_m1(M)
                                M = M & (M - 1)
                                idx2 = ((i0 + (bp2 >> 2) * BLK) << 2) + (bp2 & 3)
                                xv2 = ldg_f32(x_addr, idx2, x4_pin)
                                pos = self._emitc(xv, idx, pos, cb2_pin)
                                idx = idx2
                                xv = xv2
                            pos = self._emitc(xv, idx, pos, cb2_pin)
                        it = it + 1
                i = tidx  # scalar tail (part 0 only)
                while i < tailn:
                    x = ldg_f32(x_addr, tail0 + i)
                    if x >= TF:
                        post = atomic_add_cta(s_scal.iterator + 0, Int32(1))
                        post = self._emitc(x, tail0 + i, post, cb2_pin)
                    i = i + BLK
                cute.arch.barrier()
                myn = s_scal[0]

                local_fast = Int32(0)
                if s_cert[3] == 1:
                    if cert_fast != 0:
                        if expected_pass <= CMPB and myn <= CMPB:
                            local_fast = Int32(1)
                if local_fast != 0:
                    # Single-CTA row whose list and boundary subset fit in CMPB: no global hand-off.
                    # s_ck64 is free before the emit pass and serves as the second buffer.
                    listN = myn
                    if split_mode != 0:
                        self._split_partition(True, myn, tidx, lane, s_cbuf2, HICs, ptable, tbase,
                                              s_scal, out_row, k, s_ck64, pair_addr, gbuf_row)
                        # Every candidate is read before s_cbuf2 is overwritten (in-place compaction
                        # would race across warps).
                        cute.arch.barrier()
                        if tidx == 0:
                            s_cert[2] = s_scal[1]
                        listN = s_scal[2]
                        i = tidx
                        while i < listN:
                            s_cbuf2[i] = s_ck64[i]
                            i = i + BLK
                    self._zero_hist(histogram, hbase, tidx)
                    i = tidx
                    while i < listN:
                        w64 = Uint64(s_cbuf2[i])
                        gvx = Int32(Uint32(w64 & 0xFFFFFFFF))
                        fine_hist_add(hb_pin, gvx, TF, SC)
                        i = i + BLK
                    cute.arch.barrier()  # fine histogram, compacted list and strict count published
                    if split_mode != 0:
                        soff = s_cert[2]
                        ksc = k - soff
                    scan_cross0(s_hist, ksc, tidx, s_res, zero=False)
                    cute.arch.barrier()
                    if s_res[RES_TOT] >= ksc:
                        valid = Int32(1)
                        above = s_res[RES_ABOVE]
                        m = s_res[RES_M]
                        need = ksc - above
                        B = s_res[RES_B]
                elif short != 0:
                    # Part 0 alone owns the row; the row-pass barrier ordered its histogram reads.
                    self._zero_hist(histogram, hbase, tidx)
                else:
                    pubn = myn
                    if split_mode != 0:
                        if tidx == BLK - 32:
                            # Releases the prologue histogram reads while the other warps publish.
                            # Publication stores need no release: the last CTA validates slab entries
                            # by value, never reads strict outputs and resets the pair word only after
                            # observing every reservation.
                            cute.arch.fence_acq_rel_gpu()
                        self._split_partition(False, myn, tidx, lane, s_cbuf2, HICs, ptable, tbase,
                                              s_scal, out_row, k, s_ck64, pair_addr, gbuf_row)
                    else:
                        if tidx == 0:
                            pgo = cute.make_ptr(Int32, goff_addr + row64 * 4, cute.AddressSpace.gmem, assumed_align=4)
                            s_scal[3] = Int32(cute.arch.atomic_add(pgo, myn))
                        cute.arch.barrier()
                        base = s_scal[3]
                        if myn <= SCPB:
                            i = tidx
                            while i < myn:
                                p = base + i
                                if p < self.capacity:
                                    _st_global(gbuf_row + Int64(p) * 8, s_cbuf2[i])
                                i = i + BLK
                        else:
                            # Staging overflowed: re-sweep the slice, then the row's scalar tail.
                            if tidx == 0:
                                s_scal[0] = Int32(0)
                            cute.arch.barrier()
                            lo2 = c0 << 2
                            hi2 = c1 << 2
                            for sweep in cutlass.range_constexpr(2):
                                i = lo2 + tidx if sweep == 0 else tidx
                                while i < (hi2 if sweep == 0 else tailn):
                                    idx = i if sweep == 0 else tail0 + i
                                    x = ldg_f32(x_addr, idx)
                                    if x >= TF:
                                        p = base + atomic_add_cta(s_scal.iterator + 0, Int32(1))
                                        if p < self.capacity:
                                            _st_global(gbuf_row + Int64(p) * 8,
                                                       (Uint64(Uint32(idx)) << 32) | Uint64(u32_of_f32(x)))
                                    i = i + BLK
                    cute.arch.barrier()
                    if split_mode != 0:
                        pubn = s_scal[2]
                    # Arrival word: [63:56] arrivals, [55:32] strict count, [31:0] slab count. Every
                    # staged candidate is strict or slab, so the local strict count is myn - pubn.
                    # Row sums of both counts are below 2^24: no carries.
                    if tidx == BLK - 32:
                        if split_mode == 0:  # split mode fenced before publication
                            cute.arch.fence_acq_rel_gpu()
                        pdon = cute.make_ptr(Int64, ws_addr + row64 * 8, cute.AddressSpace.gmem, assumed_align=8)
                        s_pk[0] = Int64(cute.arch.atomic_add(
                            pdon, Int64(1 << 56) + (Int64(myn - pubn) << 32) + Int64(pubn),
                            sem="acquire", scope="gpu"))
                    cute.arch.barrier()
                    pk = s_pk[0]
                    alive = Int32(Int32(pk >> 56) == s_cert[3] - 1)  # the last arriving CTA refines the row
                    if alive != 0:
                        total = Int32(pk & 0xFFFFFFFF) + pubn
                        _device_assert(total <= self.capacity,
                            f"LiteTopK candidate capacity ({self.capacity}) exceeded; rebuild with a larger --candidate-capacity")
                        listN = total
                        if total > SCPB:
                            fromg = Int32(1)
                        # Split mode: every reservation adds positive strict and slab counts to the
                        # pair word, so it equals (S << 32) + total only after all of this row's
                        # reservations. Thread BLK-32 (idle during the scan) observes that value
                        # before storing zero, which orders the reset after all of them.
                        pair_v = Int64(0)
                        if tidx == BLK - 32 and split_mode != 0:
                            pair_v = _ld_relaxed_u64(pair_addr)
                        # Each thread's first EARLY slab entries are in flight before the restore stores.
                        gx = cute.make_rmem_tensor((EARLY,), Int32)
                        gy = cute.make_rmem_tensor((EARLY,), Int32)
                        for j in cutlass.range_constexpr(EARLY):
                            if tidx + j * BLK < listN:
                                gx[j], gy[j] = _ld_slab(gbuf_row + Int64(tidx + j * BLK) * 8)
                        if tidx == 0:
                            # Row strict count S: the earlier arrivals' counts plus this CTA's own.
                            s_cert[2] = Int32((pk >> 32) & 0xFFFFFF) + myn - pubn
                            _st_global(goff_addr + row64 * 4, Int32(0))
                            _st_global(ws_addr + row64 * 8, Uint64(0))
                        # Every CTA of the row read its bins before arriving.
                        self._zero_hist(histogram, hbase, tidx)
                        # Early entries: validate, stage and restore. In split mode the page-table
                        # words of the first two are asynchronous copies that overlap the scan.
                        pf_on = Int32(listN <= PFMAX and split_mode != 0)
                        for j in cutlass.range_constexpr(EARLY):
                            ij = tidx + j * BLK
                            if ij < listN:
                                gvx = gx[j]
                                gvy = gy[j]
                                if split_mode != 0:
                                    if gvx == 0:
                                        gvx, gvy = _ld_slab_poll(gbuf_row + Int64(ij) * 8)
                                if pf_on != 0 and j < 2:
                                    _cp_async4(s_pfp.iterator.toint() + ij * 4,
                                               ptable.iterator.toint() + Int64(tbase + (gvy >> 6)) * 4)
                                if fromg == 0:
                                    _st_s_v2_u32(cb2_pin + ij * 8, Uint32(gvx), Uint32(gvy))
                                    _st_relaxed_u64(gbuf_row + Int64(ij) * 8, Uint64(0))
                                fine_hist_add(hb_pin, gvx, TF, SC)
                        if pf_on != 0:
                            pf_lim = Int32(2 * BLK)
                        # Remaining entries, four reads in flight per thread. A split-mode entry
                        # whose value word is still zero is polled until its owner's store lands.
                        # Entries staged in shared memory are restored to zero here, the rest after refinement.
                        i = tidx + EARLY * BLK
                        while i < listN:
                            gv = []
                            for uu in cutlass.range_constexpr(4):
                                iu = i + uu * BLK
                                if iu >= listN:
                                    iu = i
                                gv.append(_ld_slab(gbuf_row + Int64(iu) * 8))
                            for uu in cutlass.range_constexpr(4):
                                iu = i + uu * BLK
                                if iu < listN:
                                    gvx, gvy = gv[uu]
                                    if split_mode != 0:
                                        if gvx == 0:
                                            gvx, gvy = _ld_slab_poll(gbuf_row + Int64(iu) * 8)
                                    if fromg == 0:
                                        s_cbuf2[iu] = (Uint64(Uint32(gvy)) << 32) | Uint64(Uint32(gvx))
                                        _st_relaxed_u64(gbuf_row + Int64(iu) * 8, Uint64(0))
                                    fine_hist_add(hb_pin, gvx, TF, SC)
                            i = i + 4 * BLK
                        cute.arch.barrier()
                        if split_mode != 0:
                            soff = s_cert[2]
                            ksc = k - soff
                        scan_cross0(s_hist, ksc, tidx, s_res, zero=False)
                        if tidx == BLK - 32 and split_mode != 0:
                            # (S << 32) + total: this CTA's arrival word after its own add, minus arrivals.
                            pair_want = (pk + (Int64(myn - pubn) << 32) + Int64(pubn)) & 0xFFFFFFFFFFFFFF
                            if pair_v != pair_want:
                                _ld_relaxed_u64_until(pair_addr, pair_want)
                            _st_relaxed_u64(pair_addr, Uint64(0))
                        cute.arch.barrier()
                        _cp_async_wait_all()
                        if s_res[RES_TOT] >= ksc:
                            valid = Int32(1)
                            above = s_res[RES_ABOVE]
                            m = s_res[RES_M]
                            need = ksc - above
                            B = s_res[RES_B]

                # ---- Refinement: fine bins above B are emitted, bin B is resolved exactly ----
                if alive != 0 and short == 0:
                    _device_assert(valid != 0,
                        "LiteTopK has fewer than K candidates; the candidate threshold is invalid")
                    whole = Int32(need >= m)
                    lim1 = above
                    if whole != 0:
                        lim1 = above + m
                    degen = Int32(m > CMPB)
                    mc = Int32(0)
                    if degen == 0:
                        mc = m
                    if degen == 0:
                        # Cursor emit; boundary-bin candidates are packed into s_ck64 as keys.
                        i = tidx
                        while i < listN:
                            vx, idv = staged(fromg, gbuf_row, s_cbuf2, i)
                            xv = f32_of_i32(vx)
                            bq = fine_bin(xv, TF, SC)
                            if bq >= B:
                                # The page base (prefetched or gathered) overlaps the cursor atomic.
                                pg = Int32(0)
                                if i < pf_lim:
                                    pg = s_pfp[i]
                                else:
                                    pg = ptable[tbase + (idv >> 6)]
                                p = atomic_add_cta(s_hist.iterator + bq, Int32(1))
                                if p < lim1:
                                    out_row[soff + p] = pg * 64 + (idv & 63)
                                else:
                                    if whole == 0:
                                        q2 = p - above
                                        if q2 < CMPB:
                                            kk = fkey_bits(u32_of_f32(xv))
                                            s_ck64[q2] = (Uint64(kk) << 32) | Uint64(Uint32(idv))
                            i = i + BLK
                        # Exact rank of the boundary bin.
                        if whole == 0:
                            cute.arch.barrier()
                            if mc <= RANK_MAX:
                                mc2 = mc & Int32(~1)
                                i = tidx
                                while i < mc:
                                    # Values crossing a dynamic loop come back signed: re-assert Uint64.
                                    u64v = s_ck64[i]
                                    r_ = Int32(0)
                                    jq = Int32(0)
                                    while jq < mc2:
                                        vlo, vhi = _lds_v2_u64(ck_addr + jq * 8)
                                        r_ = r_ + Int32(vlo > Uint64(u64v)) + Int32(vhi > Uint64(u64v))
                                        jq = jq + 2
                                    if mc2 < mc:
                                        r_ = r_ + Int32(Uint64(s_ck64[mc2]) > Uint64(u64v))
                                    if r_ < need:
                                        tie_id = Int32(Uint32(Uint64(u64v) & 0xFFFFFFFF))
                                        out_row[soff + above + r_] = physical(ptable, tbase, tie_id)
                                    i = i + BLK
                            else:
                                if tidx == 0:
                                    s_kmm[0] = Uint32(0xFFFFFFFF)
                                    s_kmm[1] = Uint32(0)
                                if tidx < NBS:
                                    s_hist[tidx] = Int32(0)
                                cute.arch.barrier()
                                i = tidx
                                while i < mc:
                                    kk = Uint32(s_ck64[i] >> 32)
                                    cute.arch.atomic_min(s_kmm.iterator + 0, kk, sem="relaxed", scope="cta")
                                    cute.arch.atomic_max(s_kmm.iterator + 1, kk, sem="relaxed", scope="cta")
                                    i = i + BLK
                                cute.arch.barrier()
                                self._narrow_and_emit(mc, need, tidx, s_hist, s_res, s_scal, lane, out_row, ptable,
                                                      tbase, gbuf_row, s_cbuf2, fromg, s_ck64, s_kmm, soff, above, 2)
                    else:
                        self._narrow_and_emit(listN, ksc, tidx, s_hist, s_res, s_scal, lane, out_row, ptable,
                                              tbase, gbuf_row, s_cbuf2, fromg, s_ck64, s_kmm, soff, above, 1)

                if alive != 0 and fromg != 0:
                    # Slab entries refined from global memory are restored to zero last.
                    i = tidx
                    while i < listN:
                        _st_relaxed_u64(gbuf_row + Int64(i) * 8, Uint64(0))
                        i = i + BLK
                if short != 0:
                    if part == 0:
                        i = tidx
                        while i < n_row:
                            out_row[i] = physical(ptable, tbase, i)
                            i = i + BLK
                        j = n_row + tidx
                        while j < k:
                            out_row[j] = Int32(-1)
                            j = j + BLK

    @cute.jit
    def __call__(self, logits: cute.Tensor, out: cute.Tensor, ws: cute.Tensor, n: Int32, kv_lens: cute.Tensor,
                 histogram: cute.Tensor, diagnostics: cute.Tensor, ptable: cute.Tensor, pcols: Int32,
                 active_batch: cute.Tensor, stream):
        self.kern(logits, out, ws, n, kv_lens, histogram, diagnostics, ptable, pcols, active_batch).launch(
            grid=(128, 1, 1), block=(self.blk, 1, 1), stream=stream, min_blocks_per_mp=1)
