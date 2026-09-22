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

"""Shared device helpers for the histogram-assisted exact FP32 selector.

Extracted from the main-family NVIDIA GVR2 implementation originally vendored
here (SHA256 dceaeec021126b7a72e4a9428d3e4d8e014e1d8d211f063396cf0f011f572195).
The base class is specialized to the two decode builds; device helper barrier
contracts and the qualified shared-memory layout are preserved.
Only GvrMainKernel.__init__ and _emitc are inherited by the final selector;
old kernel entry points, runtime compilation caches and cluster helpers are absent.
The original implementation was ported from TensorRT-LLM PR #17821, ed94d4cfbf.
"""

import sys

import cutlass

import cutlass.cute as cute

import cutlass.cute.math as cmath

from cutlass._mlir.dialects import arith as mlir_arith

from cutlass._mlir.dialects import llvm

from cutlass._mlir.dialects import llvm as mlir_llvm

from cutlass._mlir.dialects import math as mlir_math

from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass.utils.smem_allocator import SmemAllocator

C = sys.modules[__name__]

MAXC = 160  # multi-CTA SPLIT row cap

GCAP = 16384  # per-row slab capacity in int2

QUADC_CLUS = 288  # clus + gvr_main gate

GVR_WS_OFF_OFF = MAXC * 8  # workspace g_off byte offset

GVR_WS_BUF_OFF = 2048  # workspace g_buf byte offset

SENT_LO = -3.0e38

SENT_HI = 3.0e38

RES_B = 0

RES_M = 1

RES_ABOVE = 2

RES_TOT = 3

RES_B2 = 4

RES_B3 = 5

def u32_of_f32(v):
    """Raw fp32 bits as Uint32 (bit-cast, no conversion)."""
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, v.ir_value()))

def f32_of_u32(u):
    """Uint32 bit pattern as Float32 (bit-cast)."""
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, u.ir_value()))

def f32_of_i32(i):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, i.ir_value()))

def i32_of_f32(v):
    return cutlass.Int32(llvm.bitcast(cutlass.Int32.mlir_type, v.ir_value()))

def fkey_bits(u):
    """fkey on raw fp32 bits already held as Uint32."""
    neg = cutlass.Uint32(0) - (u >> cutlass.Uint32(31))  # 0 or 0xFFFFFFFF
    return u ^ (neg | cutlass.Uint32(0x80000000))

def fkey(x):
    """CUDA fkey(float). x: dynamic Float32 -> Uint32 key."""
    return fkey_bits(u32_of_f32(x))

def invkey_bits(K):
    """CUDA invkey without the final bitcast: key -> fp32 bits."""
    s = K >> cutlass.Uint32(31)  # 1 iff key top bit set
    m = (s - cutlass.Uint32(1)) | cutlass.Uint32(0x80000000)
    # s==1 -> m=0x80000000 (K^0x80000000); s==0 -> m=0xFFFFFFFF (~K)
    return K ^ m

def invkey(K):
    """CUDA invkey(uint32). K: dynamic Uint32 key -> Float32."""
    return f32_of_u32(invkey_bits(K))

def warp_min_u32(v):
    """__reduce_min_sync(FULLM, v) -> redux.sync.min.u32 (single inst)."""
    return cute.arch.warp_redux_sync(v, "min")

def warp_max_u32(v):
    """__reduce_max_sync(FULLM, v) -> redux.sync.max.u32."""
    return cute.arch.warp_redux_sync(v, "max")

def fmin_f32(a, b):
    """fminf -> native min.f32."""
    return cute.arch.fmin(a, b)

def fmax_f32(a, b):
    """fmaxf -> max.f32."""
    return cute.arch.fmax(a, b)

def ballot(pred):
    """__ballot_sync(FULLM, pred) -> Int32 mask."""
    return cute.arch.vote_ballot_sync(pred)

def popc(x):
    return cute.arch.popc(x)

def clz_i32(x):
    """__clz as Int32."""
    return cutlass.Int32(cute.arch.clz(x))

def ffs_m1(x):
    """__ffs(x) - 1 for x != 0 (bit index of lowest set bit).

    Spelled popc((x & -x) - 1). Caller must guarantee x != 0 (every use is
    inside a mask-walk loop).
    """
    return cutlass.Int32(cute.arch.popc((x & (cutlass.Int32(0) - x)) - cutlass.Int32(1)))

@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    """Inclusive-scan step: val += shfl_up(val, offset) gated lane >= offset.

    Native shfl.sync.up (mask_and_clamp=0, the __shfl_up_sync lowering):
    hardware clamps the source lane, deleting the VIMNMX+VIADD software
    clamp of the previous idx-kind spelling. Lanes < offset receive an
    undefined-but-discarded value (the gate keeps the result identical).
    """
    other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
    if lane >= cutlass.Int32(offset):
        val = val + other
    return val

@cute.jit
def warp_incl_scan_add(val, lane):
    """5-step inclusive __shfl_up_sync add scan."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val

def atomic_add_cta(ptr, val):
    """shared atomicAdd returning old value. ptr: cute Pointer

    (e.g. `s_hist.iterator + bin_idx`), val: Int32.
    """
    return cutlass.Int32(cute.arch.atomic_add(ptr, val, sem="relaxed", scope="cta"))

def atomic_min_cta(ptr, val):
    """shared atomicMin (s_kmin seeds). Unsigned iff val is Uint32."""
    return cute.arch.atomic_min(ptr, val, sem="relaxed", scope="cta")

def atomic_max_cta(ptr, val):
    """shared atomicMax (s_kmax seeds)."""
    return cute.arch.atomic_max(ptr, val, sem="relaxed", scope="cta")

def atomic_or_cta(ptr, val):
    """shared atomicOr (gvr_topk_reg bitmap path)."""
    return cute.arch.atomic_or(ptr, val, sem="relaxed", scope="cta")


def f2s_rz(v):
    """__float2int_rz: saturating (-inf -> INT_MIN, huge -> INT_MAX, NaN -> 0)."""
    return cutlass.Int32(v)

def _asm(result, operands, instruction, constraints, *, effects, align_stack, loc, ip):
    """Emit the shared inline-PTX convention without changing instruction flags."""
    return llvm.inline_asm(
        result, [value.ir_value(loc=loc, ip=ip) for value in operands],
        instruction, constraints, has_side_effects=effects, is_align_stack=align_stack,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )

@dsl_user_op
def _prefetch_l2(gaddr, *, loc=None, ip=None):
    """prefetch.global.L2 [gaddr]; gaddr is a byte address (Int64)."""
    _asm(
        None, [gaddr],
        'prefetch.global.L2 [$0];', 'l',
        effects=True, align_stack=None, loc=loc, ip=ip,
    )

def g2r_atom_f32(bits: int, invariant: bool = True):
    """CopyG2ROp atom: bits=128 -> LDG.E.128[.CONSTANT], bits=32 -> scalar."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=bits, invariant=invariant
    )

@dsl_user_op
def _ld_g_nc_v4_f32(gaddr, *, loc=None, ip=None):
    """Pinned `ld.global.nc.v4.f32` (CUDA `__ldg(const float4*)`).

    The asm boundary pins the four-scalar-f32 shape: NVVM otherwise rewrites
    adjacent 128-bit f32 copy-atom loads into v2.b64 register-pair loads,
    whose even-aligned pair constraint fragments allocation at the
    64-register wall and induces spills."""
    from cutlass._mlir import ir as _ir

    st = _ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>")
    r = _asm(
            st, [gaddr],
            'ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];', '=f,=f,=f,=f,l',
            effects=False, align_stack=False, loc=loc, ip=ip,
        )
    return tuple(
        cutlass.Float32(mlir_llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    )

def ld_g_f32x4(copy_atom, base_addr, v_idx, frag):
    """Load float4 #v_idx (16B units) from gmem byte base into frag[0..3].

    base_addr: Int64 byte address; frag: (4,) f32 fragment. Issue ALL batch
    members before consuming any. Pinned-asm form (see _ld_g_nc_v4_f32);
    copy_atom kept for call-site compatibility.
    """
    v0, v1, v2, v3 = _ld_g_nc_v4_f32(base_addr + cutlass.Int64(v_idx) * cutlass.Int64(16))
    frag[0] = v0
    frag[1] = v1
    frag[2] = v2
    frag[3] = v3

def ldg_f32(base_addr, idx):
    """__ldg(X + idx): scalar read-only 4B gather."""
    atom = g2r_atom_f32(32, invariant=True)
    p = cute.make_ptr(
        cutlass.Float32,
        base_addr + cutlass.Int64(idx) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    frag = cute.make_rmem_tensor((1,), cutlass.Float32)
    cute.copy(atom, cute.make_tensor(p, cute.make_layout((1,))), frag)
    return frag[0]

def ld_g_i32(base_addr, idx):
    """plain P[idx] scalar int32 load."""
    p = cute.make_ptr(
        cutlass.Int32,
        base_addr + cutlass.Int64(idx) * cutlass.Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    return cutlass.Int32(cute.arch.load(p, cutlass.Int32))

@dsl_user_op
def _ldcg_v2_i32(gaddr, *, loc=None, ip=None):
    """__ldcg on an int2 (8B slab word): ld.global.cg.v2.u32 -> (x, y).

    x = value bits, y = index (workspace g_buf layout).
    gaddr: Int64 byte address, 8B-aligned.
    """
    ret = _asm(
              llvm.StructType.get_literal([T.i32(), T.i32()]), [gaddr],
              'ld.global.cg.v2.u32 {$0, $1}, [$2];', '=r,=r,l',
              effects=True, align_stack=False, loc=loc, ip=ip,
          )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])),
    )

def smem_atom_i32_128():
    """CopyUniversalOp atom for ld/st.shared.v4.b32 on Int32 smem."""
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)

def _smem_v4_tensor(base_addr, byte_off):
    """4-elt Int32 smem tensor at 16B-aligned base_addr+byte_off (Int32 addr)."""
    p = cute.make_ptr(cutlass.Int32, base_addr + byte_off, cute.AddressSpace.smem, assumed_align=16)
    return cute.make_tensor(p, cute.make_layout((4,)))

def lds128_i32(copy_atom, base_addr, byte_off, frag):
    """ld.shared.v4.b32 -> frag(4, Int32)."""
    cute.copy(copy_atom, _smem_v4_tensor(base_addr, byte_off), frag)

def sts128_i32(copy_atom, frag, base_addr, byte_off):
    """st.shared.v4.b32 <- frag(4, Int32)."""
    cute.copy(copy_atom, frag, _smem_v4_tensor(base_addr, byte_off))

@dsl_user_op
def _lds_v2_u64(saddr, *, loc=None, ip=None):
    """ulonglong2 16B smem read (quad-rank path): (lo, hi)."""
    ret = _asm(
              llvm.StructType.get_literal([T.i64(), T.i64()]), [saddr],
              'ld.shared.v2.u64 {$0, $1}, [$2];', '=l,=l,r',
              effects=True, align_stack=False, loc=loc, ip=ip,
          )
    return (
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [0])),
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [1])),
    )

@cute.jit
def scan_cross0(
    s_hist,
    target,
    tidx,
    s_res,
    target2,
    target3,
    s_addv,
    nb: cutlass.Constexpr,
    zero: cutlass.Constexpr,
    two: cutlass.Constexpr = False,
    three: cutlass.Constexpr = False,
):
    assert nb == 256
    BPT = nb // 32  # bins per lane (trace-time int)
    NV = BPT // 4  # 16B vectors per lane
    if tidx < cutlass.Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        hbase = s_hist.iterator.toint()
        # pass 1: span sum via NV uint4 LDS.128
        frags = [cute.make_rmem_tensor((4,), cutlass.Int32) for _ in range(NV)]
        sm = cutlass.Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            lds128_i32(atom, hbase, boff, frags[q])
            sm = sm + frags[q][0] + frags[q][1] + frags[q][2] + frags[q][3]
        # 5-step inclusive shfl_up scan
        w = warp_incl_scan_add(sm, lane)
        tot = cute.arch.shuffle_sync(w, cutlass.Int32(31))
        after = tot - w  # bins strictly above my span
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = tot
        base = lane * cutlass.Int32(BPT)
        # pass 2: descending vector walk
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            vv = frags[q]
            o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = vv[j]
                if cutlass.const_expr(zero):
                    o4[j] = cutlass.Int32(0)
                else:
                    o4[j] = after
                gb = base + cutlass.Int32(4 * q + j)
                cross = cutlass.Int32(0)
                if after < target:
                    if (after + cq) >= target:
                        cross = cutlass.Int32(1)
                    if gb == cutlass.Int32(0):
                        cross = cutlass.Int32(1)
                if cross != cutlass.Int32(0):
                    s_res[RES_B] = gb
                    s_res[RES_ABOVE] = after
                    s_res[RES_M] = cq
                if cutlass.const_expr(two):
                    cross2 = cutlass.Int32(0)
                    if after < target2:
                        if (after + cq) >= target2:
                            cross2 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross2 = cutlass.Int32(1)
                    if cross2 != cutlass.Int32(0):
                        s_res[RES_B2] = gb
                if cutlass.const_expr(three):
                    cross3 = cutlass.Int32(0)
                    if after < target3:
                        if (after + cq) >= target3:
                            cross3 = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross3 = cutlass.Int32(1)
                    if cross3 != cutlass.Int32(0):
                        s_res[RES_B3] = gb
                after = after + cq
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, hbase, boff)

@cute.jit
def gather_hint(
    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk: cutlass.Constexpr, kpt: cutlass.Constexpr
):
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    # batch A: KPT coalesced pre_idx loads, predicated flat
    pvs = []
    for t in cutlass.range_constexpr(kpt):
        pv = cutlass.Int32(-1)
        j = tidx + cutlass.Int32(t * blk)
        if j < k:
            pv = ld_g_i32(p_addr, j)
        pvs.append(pv)
    # batch B: KPT scattered read-only gathers, predicated flat
    xs = []
    for t in cutlass.range_constexpr(kpt):
        xv = cutlass.Float32(0.0)
        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):  # (unsigned)p < (unsigned)n
            xv = ldg_f32(x_addr, pvs[t])
        xs.append(xv)
    # fold
    glmin = cutlass.Uint32(0xFFFFFFFF)
    glmax = cutlass.Uint32(0)
    for t in cutlass.range_constexpr(kpt):
        if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
            u2 = fkey(xs[t])
            if u2 < glmin:
                glmin = u2
            if u2 > glmax:
                glmax = u2
    # warp redux + staging
    glmin = warp_min_u32(glmin)
    glmax = warp_max_u32(glmax)
    if lane == cutlass.Int32(0):
        s_wmn[tidx >> cutlass.Int32(5)] = glmin
        s_wmx[tidx >> cutlass.Int32(5)] = glmax
    cute.arch.barrier()  # barrier 1/2
    # cross-warp redux by EVERY thread — block-uniform outputs
    a2 = cutlass.Uint32(0xFFFFFFFF)
    c2 = cutlass.Uint32(0)
    if lane < cutlass.Int32(NW):
        a2 = s_wmn[lane]
        c2 = s_wmx[lane]
    gm = invkey(warp_min_u32(a2))
    gx = invkey(warp_max_u32(c2))
    # NaN-safe degeneracy guard: !(GM < GX) — NaN compares false
    ok = cutlass.Int32(0)
    if gm < gx:
        ok = cutlass.Int32(1)
    if ok == cutlass.Int32(0):
        gm = cutlass.Float32(SENT_LO)
        gx = cutlass.Float32(SENT_HI)
    cute.arch.barrier()  # barrier 2/2
    return gm, gx

MAXC__main = C.MAXC

GCAP__main = C.GCAP

QUADC_CLUS__main = C.QUADC_CLUS

WS_BYTES = C.GVR_WS_BUF_OFF + MAXC__main * GCAP__main * 8  # 20,973,568

_NEG_INF = float("-inf")

@dsl_user_op
def _fmaf(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        mlir_math.fma(
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            fastmath=mlir_arith.FastMathFlags.none,
            loc=loc,
            ip=ip,
        )
    )

def _st_g_u64(addr_i64, val_u64):
    """plain st.global.u64 (slab publish, g_don restore)."""
    p = cute.make_ptr(cutlass.Uint64, addr_i64, cute.AddressSpace.gmem, assumed_align=8)
    t = cute.make_tensor(p, cute.make_layout((1,)))
    t[0] = val_u64

def _st_g_u32(addr_i64, val_i32):
    """plain st.global.u32 (g_off restore)."""
    p = cute.make_ptr(cutlass.Int32, addr_i64, cute.AddressSpace.gmem, assumed_align=4)
    t = cute.make_tensor(p, cute.make_layout((1,)))
    t[0] = val_i32

@dsl_user_op
def _st_s_v2_u32(saddr_i32, lo_u32, hi_u32, *, loc=None, ip=None):
    """st.shared.v2.u32 [saddr], {lo, hi} — the CUDA make_int2 STS.64 spelling.
    Byte-identical to the little-endian u64 pack ((hi << 32) | lo) but keeps
    the two words as independent 32-bit registers, so ptxas can coalesce the
    emission bit-walk's loop-carried (xv, idx) pair straight into the store
    pair."""
    _asm(
        None, [saddr_i32, lo_u32, hi_u32],
        'st.shared.v2.u32 [$0], {$1, $2};', 'r,r,r',
        effects=True, align_stack=None, loc=loc, ip=ip,
    )

@dsl_user_op
def _pin_i64(v, *, loc=None, ip=None):
    """Opaque identity mov.b64: pins a loop-invariant Int64 so NVVM cannot
    rematerialize its defining chain (param ld.const + %ctaid reads + mul/add)
    into every scf region body."""
    return cutlass.Int64(
        _asm(
            T.i64(), [v],
            'mov.b64 $0, $1;', '=l,l',
            effects=False, align_stack=False, loc=loc, ip=ip,
        )
    )

@dsl_user_op
def _pin_i32(v, *, loc=None, ip=None):
    """Opaque identity mov.b32 (Int32 twin of _pin_i64)."""
    return cutlass.Int32(
        _asm(
            T.i32(), [v],
            'mov.b32 $0, $1;', '=r,r',
            effects=False, align_stack=False, loc=loc, ip=ip,
        )
    )

def _ldg_f32_rs(base_addr, idx, sc4):
    """__ldg(X + idx) with the byte stride riding a register.

    Identical to ldg_f32 except `* 4` multiplies a caller-held Int32: the
    row base is uniformized into URx by ptxas, IMAD.WIDE cannot encode an
    immediate stride next to a UR addend, and a constant stride register
    would otherwise be re-materialized inside the survivor walk. The caller
    loads the 4 from smem (LDS results are opaque to ptxas value-tracking;
    asm movs and shfl are not), so the register stays live and the remat
    disappears."""
    atom = C.g2r_atom_f32(32, invariant=True)
    p = cute.make_ptr(
        cutlass.Float32,
        base_addr + cutlass.Int64(idx) * cutlass.Int64(sc4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    frag = cute.make_rmem_tensor((1,), cutlass.Float32)
    cute.copy(atom, cute.make_tensor(p, cute.make_layout((1,))), frag)
    return frag[0]

@dsl_user_op
def _smem_addr_reg(addr, *, loc=None, ip=None):
    """Pin a CTA-shared 32-bit byte address in ONE register.

    Identity `mov` behind an asm boundary: without it LLVM re-folds the
    `mov.b32 %r, __dynamic_shmem__0` symbol materialisation into EVERY use
    site inside the divergent emission bit-walk (one extra IMAD.MOV per
    survivor). The asm result is not duplicable, so the shared window is
    materialised exactly once. Value-identical: a plain register copy."""
    return cutlass.Int32(
        _asm(
            T.i32(), [addr],
            'mov.u32 $0, $1;', '=r,r',
            effects=False, align_stack=False, loc=loc, ip=ip,
        )
    )

@dsl_user_op
def _red_shared_add1(addr, *, loc=None, ip=None):
    """CUDA `atomicAdd(&hist[bin], 1u)` with the result unused.

    `red` (not `atom`) is the result-less spelling — ptxas lowers it to the
    same ATOMS.POPC.INC.32 RZ the CUDA arm emits. Same ordering contract as
    atomic_add_cta (.relaxed scope .cta). Takes the final shared byte
    address as a plain Int32 so ptxas fuses the shl+add into one LEA
    against the pinned `_smem_addr_reg` base."""
    _asm(
        None, [addr],
        'red.relaxed.cta.shared.add.u32 [$0], 1;', 'r',
        effects=True, align_stack=None, loc=loc, ip=ip,
    )

class GvrMainKernel:
    """Geometry and survivor staging for the two qualified decode selectors."""

    def __init__(self, blk: int, u: int):
        assert (blk, u) in ((512, 1), (1024, 4))
        self.blk = blk
        self.u = u
        self.minb = 1
        self.nbs = self.hb = 256
        self.kpt = 2
        self.vstg = self.tshg = True
        self.next_n = 1
        self.cr_shift = 0
        self.kbig = blk == 1024
        self.scpb = 8192 if self.kbig else 4096
        self.cmpb = 4096 if self.kbig else 1024
        self.shd = False
        self.pfd = u
        self.natt = 1
        # Preserve the qualified shared-memory layout: int2 candidates, then keys.
        self.ck_off = 8 * (self.scpb + 4)
        self.dyn_bytes = self.ck_off + (self.cmpb + 1) * 8
        self.lb = 8

    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, hb, cb2, s_hist, s_cbuf, s_cbuf2):
        ps = pos
        if ps > cutlass.Int32(self.scpb):
            ps = cutlass.Int32(self.scpb)
        # int2 {FP32 bits, index}; pinned base preserves the instruction sequence.
        _st_s_v2_u32(cb2 + ps * cutlass.Int32(8), C.u32_of_f32(xv), cutlass.Uint32(idx))
        return pos + cutlass.Int32(1)

def workspace_bytes() -> int:
    return WS_BYTES
