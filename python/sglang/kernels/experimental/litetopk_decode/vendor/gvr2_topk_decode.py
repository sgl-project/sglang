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

"""Self-sampling GVR top-K decode kernels (CuTe DSL, Blackwell sm_100a; the same
source compiles for sm_103a and Rubin sm_107a — it uses only family-portable
ops, no tcgen05 / block-scaled MMA — and the compile target follows the current
device).

Sample-calibrated threshold ladders for exact single-pass top-K: the kernel
derives its selection threshold from an in-kernel sample of the row itself
(one sample histogram yields a bracketed ladder of candidate thresholds,
resolved and harvested in a single streaming pass); exactness is guaranteed
by count-crossing invariants, never by the estimate; the temporal hint
(pre_idx) survives only as a degenerate-case anchor.

Four kernel families -- sampling-ladder slab/streaming (main),
register-resident (reg), cluster streaming (clus), clustered
register-resident (regclus) -- merged into a single device module;
collision symbols carry a ``__<family>`` suffix.

Provenance: ported near-verbatim from TensorRT-LLM
``tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling.py``
(PR NVIDIA/TensorRT-LLM#17821, commit ed94d4cfbf), plus two register-family
fixes ported verbatim: PR NVIDIA/TensorRT-LLM#18501 (``GvrTopkRegKernel.kern``:
count-crossing enforcement when the histogram total never reaches k,
radix-descent escape, -inf fill-lane bound) and PR NVIDIA/TensorRT-LLM#18625
(an infinite bracket width from a sampled +inf fails the collapse guard;
``GvrTopkRegKernel.kern`` then takes the key-space escape and
``GvrRegClusKernel.kern`` the whole-row ``degen`` fallback, instead of the
collapsed histogram dropping the +inf).  FlashInfer-local changes: module
rename, this note, the ``_persist`` hook that routes the four
``get_compiled*`` builders through FlashInfer's on-disk CuTe-DSL kernel
cache, ``_base_compile_opts`` (explicit ``--gpu-arch`` for the current
device), and the main-family varlen prologue clamping each row's length to
the envelope in launch slot ``n`` instead of the row stride.  Keep future
diffs against upstream mechanical.
"""

import contextlib
import sys

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass._mlir.dialects import math as mlir_math
from cutlass.cute import runtime as _crt
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

# `C.` references throughout resolve against this module itself.
C = sys.modules[__name__]


def _compile_arch_token() -> str:
    """Current compile target, mirroring the persistent cache's arch axis
    (CUTE_DSL_ARCH env override, else the current device's capability).

    Included in the in-process ``_COMPILE_CACHE*`` keys so a heterogeneous
    multi-GPU process (e.g. SM100 + SM103) never serves a callable compiled
    for another architecture — the on-disk cache is arch-namespaced, but
    these dicts sit above it."""
    import os

    env = os.environ.get("CUTE_DSL_ARCH")
    if env:
        return env
    try:
        import torch  # lazy: module stays torch-free at import

        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except Exception:  # noqa: BLE001 — no GPU context: single-arch fallback
        return "unknown"


def _base_compile_opts() -> str:
    """Base ``cute.compile`` options for every gvr2 family: the TVM-FFI ABI
    (the persistent cache reloads artifacts through it) plus an explicit
    ``--gpu-arch`` for the CURRENT device. Without the latter the DSL resolves
    its target once per process from device 0 (``detect_gpu_arch``), so in a
    heterogeneous process a kernel compiled while device 1 is current is built
    for device 0's architecture and fails to load
    (cudaErrorNoKernelImageForDevice). ``CUTE_DSL_ARCH`` still wins when set."""
    import os

    opts = "--enable-tvm-ffi"
    if os.environ.get("CUTE_DSL_ARCH"):
        return opts
    try:
        import torch  # lazy: module stays torch-free at import

        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 — no GPU context: let the DSL detect
        return opts
    return f"{opts} --gpu-arch sm_{major}{minor}{'a' if major >= 9 else ''}"


def _persist(kernel_name: str, compile_fn):
    """Route a ``cute.compile`` closure through FlashInfer's persistent
    CuTe-DSL kernel cache (one ``gvr2_topk`` module directory shared by all
    four families; artifacts reload via TVM-FFI, so the closures MUST compile
    with ``--enable-tvm-ffi`` — they all do). Falls back to a plain in-process
    compile when the FlashInfer JIT layer is unavailable (standalone use)."""
    try:
        from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    except ImportError:
        return compile_fn()
    return build_and_load_cute_dsl_kernel(
        "gvr2_topk", kernel_name, compile_fn, extra_key_files=(__file__,)
    )


# ===========================================================================
# ==== shared device units =====================================
# ===========================================================================
"""Device-helper library shared by the main / reg / clus / regclus families.

Conventions
-----------
* Crossing-scan helpers write their scalar outputs into an Int32 smem tensor
  `s_res` using the slot map RES_B=0, RES_M=1, RES_ABOVE=2, RES_TOT=3,
  RES_B2=4, RES_B3=5. Slots are written ONLY on a pin.
* Histograms are Int32 smem tensors: adds/scans are bit-identical mod 2^32,
  and totals stay below 2^31 by dispatch domain.
* Warp-0-only helpers (find_cross / scan_cross0 / merge_scan0) contain NO
  barrier; scan_cross and scan_cross_w contain EXACTLY ONE internal barrier;
  gather_hint contains EXACTLY TWO. Do not add or drop any.
* All warp collectives use the full mask FULLM = 0xffffffff.
"""


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
FULLM = 0xFFFFFFFF
NB = 1024  # register-family base bin count
SNB = 256  # streaming-path bin count — MUST stay 256
MAXC = 160  # multi-CTA SPLIT row cap
GCAP = 16384  # per-row slab capacity in int2
QUADC = 96  # O(mc^2) rank gate, streaming/reg
QUADC_CLUS = 288  # clus + gvr_main gate
IDXB = 22  # packed candidate index bits
IDXM = (1 << IDXB) - 1
GVR_WS_OFF_OFF = MAXC * 8  # workspace g_off byte offset
GVR_WS_BUF_OFF = 2048  # workspace g_buf byte offset

# degenerate-hint sentinels (exact-equality flag values)
SENT_LO = -3.0e38
SENT_HI = 3.0e38

# s_res slot map (see module docstring)
RES_B = 0
RES_M = 1
RES_ABOVE = 2
RES_TOT = 3
RES_B2 = 4
RES_B3 = 5


# ---------------------------------------------------------------------------
# float <-> u32 bitcasts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# fkey / invkey — order-preserving float->u32 radix key.
# fkey:   u ^ (((int32)u >> 31) | 0x80000000)  [arithmetic-shift sign trick,
#         spelled 0 - (u >> 31) on Uint32]
# invkey: (K & 0x80000000) ? K ^ 0x80000000 : ~K   [exact inverse]
# Monotone over all finite floats and +-inf; min identity 0xffffffff, max 0.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# warp redux wrappers. Values passed to the u32 forms MUST be genuine
# cutlass.Uint32 — an Int32 silently lowers to redux.sync.{min,max}.s32.
# ---------------------------------------------------------------------------
def warp_min_u32(v):
    """__reduce_min_sync(FULLM, v) -> redux.sync.min.u32 (single inst)."""
    return cute.arch.warp_redux_sync(v, "min")


def warp_max_u32(v):
    """__reduce_max_sync(FULLM, v) -> redux.sync.max.u32."""
    return cute.arch.warp_redux_sync(v, "max")


def warp_add_u32(v):
    """__reduce_add_sync(FULLM, v) -> redux.sync.add.s32 (bit-identical u32)."""
    return cute.arch.warp_redux_sync(v, "add")


def warp_add_i32(v):
    """__reduce_add_sync on Int32 (scan_cross_w two-redux stage)."""
    return cute.arch.warp_redux_sync(v, "add")


def fmin_f32(a, b):
    """fminf -> native min.f32."""
    return cute.arch.fmin(a, b)


def fmax_f32(a, b):
    """fmaxf -> max.f32."""
    return cute.arch.fmax(a, b)


# ---------------------------------------------------------------------------
# ballot / popc / clz / ffs wrappers
# ---------------------------------------------------------------------------
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
def hi_bit_or_zero(msk):
    """CUDA `msk ? (31 - __clz(msk)) : 0`."""
    r = cutlass.Int32(0)
    if msk != cutlass.Int32(0):
        r = cutlass.Int32(31) - clz_i32(msk)
    return r


# ---------------------------------------------------------------------------
# warp shfl scans (plus the TWO-interleaved variant gvr_topk_reg needs)
# ---------------------------------------------------------------------------
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
def _shfl_down_add(val, lane, offset: cutlass.Constexpr):
    """Suffix-scan step: val += shfl_down(val, offset) gated lane+offset < 32.

    Native shfl.sync.down (mask_and_clamp=31 = __shfl_down_sync lowering);
    hardware clamps, gate discards out-of-range lanes as before.
    """
    other = cute.arch.shuffle_sync_down(val, offset, mask_and_clamp=31)
    if lane + cutlass.Int32(offset) < cutlass.Int32(32):
        val = val + other
    return val


@cute.jit
def warp_incl_scan_add(val, lane):
    """5-step inclusive __shfl_up_sync add scan."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val


@cute.jit
def warp_incl_scan_add2(v1, v2, lane):
    """TWO interleaved inclusive shfl_up scans.

    Per step o: shfl(v1); gated add; shfl(v2); gated add — the two dependency
    chains interleave so the second scan hides under the first's shfl latency
    exactly as the CUDA dual-scan loop does.
    """
    for o in [1, 2, 4, 8, 16]:
        z1 = cute.arch.shuffle_sync_up(v1, o, mask_and_clamp=0)
        if lane >= cutlass.Int32(o):
            v1 = v1 + z1
        z2 = cute.arch.shuffle_sync_up(v2, o, mask_and_clamp=0)
        if lane >= cutlass.Int32(o):
            v2 = v2 + z2
    return v1, v2


@cute.jit
def warp_suffix_scan_add(val, lane):
    """5-step __shfl_down_sync suffix add scan."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_down_add(val, lane, o)
    return val


# ---------------------------------------------------------------------------
# CTA-scope shared-memory atomics (return the OLD value; never sys scope)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# gpu-scope fences + global u64 atomicAdd (SPLIT slab protocol)
# ---------------------------------------------------------------------------
def threadfence_gpu():
    """__threadfence() == fence.acq_rel.gpu — required on BOTH the release and
    acquire sides of the slab hand-off."""
    cute.arch.fence_acq_rel_gpu()


def atomic_add_u64_gpu(ptr, val):
    """atom.global.add.u64 returning the OLD value (arrival RMW).

    ptr: cute Pointer to an Int64 gmem word; val: cutlass.Int64.
    Packed arrival word: `cutlass.Int64(1 << 32) + cutlass.Int64(myn)`.
    """
    return cutlass.Int64(cute.arch.atomic_add(ptr, val))


# ---------------------------------------------------------------------------
# saturating converts (native ctors emit cvt.rzi.{u32,s32}.f32)
# ---------------------------------------------------------------------------
def f2u_rz(v):
    """__float2uint_rz: saturating (neg/-inf -> 0, huge -> 0xffffffff, NaN -> 0).

    Dynamic values only — host constants raise OverflowError on inf.
    """
    return cutlass.Uint32(v)


def f2s_rz(v):
    """__float2int_rz: saturating (-inf -> INT_MIN, huge -> INT_MAX, NaN -> 0)."""
    return cutlass.Int32(v)


# ---------------------------------------------------------------------------
# L2 prefetch escape hatch
# ---------------------------------------------------------------------------
@dsl_user_op
def _prefetch_l2(gaddr, *, loc=None, ip=None):
    """prefetch.global.L2 [gaddr]; gaddr is a byte address (Int64)."""
    llvm.inline_asm(
        res=None,
        operands_=[gaddr.ir_value(loc=loc, ip=ip)],
        asm_string="prefetch.global.L2 [$0];",
        constraints="l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# global loads: 128-bit ldg (read-only) / plain, scalar forms, and __ldcg
# (L2-direct) vector forms for the slab consume
# ---------------------------------------------------------------------------
def g2r_atom_f32(bits: int, invariant: bool = True):
    """CopyG2ROp atom: bits=128 -> LDG.E.128[.CONSTANT], bits=32 -> scalar."""
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=bits, invariant=invariant
    )


def g2r_atom_i32(bits: int, invariant: bool = False):
    return cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(), cutlass.Int32, num_bits_per_copy=bits, invariant=invariant
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
    r = mlir_llvm.inline_asm(
        st,
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
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
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])),
    )


@dsl_user_op
def _ldcg_v4_i32(gaddr, *, loc=None, ip=None):
    """ld.global.cg.v4.b32 (16B L2-direct load), returns 4 Int32."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(cutlass.Int32(llvm.extractvalue(T.i32(), ret, [i])) for i in range(4))


# ---------------------------------------------------------------------------
# 128-bit shared-memory ld/st (copy-atom spelling) + ulonglong2 read
# ---------------------------------------------------------------------------
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
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.i64()]),
        [saddr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u64 {$0, $1}, [$2];",
        "=l,=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [0])),
        cutlass.Uint64(llvm.extractvalue(T.i64(), ret, [1])),
    )


# ---------------------------------------------------------------------------
# DSMEM op set. mapa returns a byte-addressed Int32 in the PEER's shared
# window; offset arithmetic is applied after one mapa per rank.
# ---------------------------------------------------------------------------
@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None):
    """mapa.shared::cluster of a local smem Pointer -> Int32 peer byte addr."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mapa_shared_cluster_addr(addr_i32, peer_rank, *, loc=None, ip=None):
    """mapa of a raw Int32 shared-window byte address (already .toint()'d)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr_i32.ir_value(loc=loc, ip=ip), peer_rank.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_i32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_f32(mapped_addr, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [mapped_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_v4_u32(mapped_addr, *, loc=None, ip=None):
    """Single-shot remote 16B DSMEM load."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [mapped_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [2])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [3])),
    )


@dsl_user_op
def _st_shared_cluster_i32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u32 [$0], $1;",
        constraints="r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_cluster_f32(mapped_addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.f32 [$0], $1;",
        constraints="r,f",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_cluster_u64(mapped_addr, val, *, loc=None, ip=None):
    """ONE packed 8B DSMEM candidate push (never split into two 4B stores).

    val = (Uint64(key) << 32) | Uint64(idx_bits).
    """
    llvm.inline_asm(
        res=None,
        operands_=[mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        asm_string="st.shared::cluster.u64 [$0], $1;",
        constraints="r,l",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_shared_cluster_add_i32(mapped_addr, val, *, loc=None, ip=None):
    """Remote CTA smem atomicAdd (cluster scope), returns old value."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mapped_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.relaxed.cluster.shared::cluster.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# aligned cluster barrier (cg::cluster.sync() ==
# barrier.cluster.{arrive,wait}.aligned). Writers use the FULL (releasing)
# arrive — cluster_arrive_relaxed has NO release and races DSMEM. Never
# substitute the non-aligned cute.arch forms.
# ---------------------------------------------------------------------------
@dsl_user_op
def _cluster_arrive_aligned(*, loc=None, ip=None):
    nvvm.cluster_arrive(aligned=True, loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_aligned(*, loc=None, ip=None):
    nvvm.cluster_wait(aligned=True, loc=loc, ip=ip)


@cute.jit
def _cluster_sync_aligned():
    """cg::cluster_group::sync()."""
    _cluster_arrive_aligned()
    _cluster_wait_aligned()


# ===========================================================================
# find_cross<NB_=1024>
# highest bin B with sum_{j>=B} hist[j] >= target; also total, m = hist[B],
# above = sum_{j>B}. Warp-parallel (warp 0 only), bank-conflict free via the
# rotated indexing hist[lane*BPL + ((j+lane) & (BPL-1))] — DO NOT drop it.
# Non-destructive. NO barrier inside.
# Writes s_res[RES_B/RES_M/RES_ABOVE] from the single pinning lane and
# s_res[RES_TOT] from lane 0.
# ===========================================================================
@cute.jit
def find_cross(s_hist, target, tidx, s_res, nb: cutlass.Constexpr):
    BPL = nb // 32  # python int at trace time
    if tidx < cutlass.Int32(32):
        lane = tidx
        # per-lane span sum with rotated bank-skew indexing
        part = cutlass.Int32(0)
        for j in cutlass.range_constexpr(BPL):
            idx = lane * cutlass.Int32(BPL) + ((cutlass.Int32(j) + lane) & cutlass.Int32(BPL - 1))
            part = part + s_hist[idx]
        # 5-step suffix scan: v = sum of part over lanes >= lane
        v = warp_suffix_scan_add(part, lane)
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = v
        # level 1: highest lane whose suffix still reaches target
        msk = ballot(v >= target)
        L = hi_bit_or_zero(msk)
        aboveL = cute.arch.shuffle_sync(v - part, L)
        # level 2: one bin per lane inside lane L's span
        h = cutlass.Int32(0)
        if lane < cutlass.Int32(BPL):
            h = s_hist[L * cutlass.Int32(BPL) + lane]
        w = warp_suffix_scan_add(h, lane)
        msk2 = ballot((aboveL + w) >= target)
        J = hi_bit_or_zero(msk2)
        if lane == J:  # pinning lane
            s_res[RES_B] = L * cutlass.Int32(BPL) + J
            s_res[RES_M] = h
            s_res[RES_ABOVE] = aboveL + (w - h)


# ===========================================================================
# scan_cross0<NB_, ZERO, TWO, THREE, ADD>
# Warp-0-only vectorized suffix scan (streaming workhorse, NB_=256 at every
# production call site). Contains NO barrier — the caller pays exactly one
# after it. Leaves hist[j] = per-bin OUTPUT CURSOR (count strictly above
# bin j), or ZEROS when zero=True (folds the next phase's histogram clear).
# two/three pin extra crossing bins for target2/target3 into RES_B2/RES_B3.
# addf folds the per-rank bin-offset vector s_addv into the cursors.
# HOLD register guard: NV<=2 holds the span in regs across the scan; wider
# instantiations re-READ their span (no barrier needed — each lane only
# touches its own span).
# ===========================================================================
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
    addf: cutlass.Constexpr = False,
):
    BPT = nb // 32  # bins per lane (trace-time int)
    NV = BPT // 4  # 16B vectors per lane
    HOLD = NV <= 2  # register-pressure guard
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
            if cutlass.const_expr(HOLD):
                vv = frags[q]
            else:
                vv = cute.make_rmem_tensor((4,), cutlass.Int32)  # re-read span
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, hbase, boff, vv)
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
            if cutlass.const_expr(addf):  # fold per-rank bin offset
                av = cute.make_rmem_tensor((4,), cutlass.Int32)
                aoff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                lds128_i32(atom, s_addv.iterator.toint(), aoff, av)
                for j in cutlass.range_constexpr(4):
                    o4[j] = o4[j] + av[j]
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, hbase, boff)


# ===========================================================================
# scan_cross<BLK, NB_, TWO>
# Block-parallel suffix scan over NB_ (<= BLK) bins. Leaves hist[j] = OUTPUT
# CURSOR (count in bins > j) and pins the crossing bin. Warps that hold no
# bin skip the body. EXACTLY ONE internal barrier; the caller pays its usual
# publish barrier after. Used by the gvr_clus whole-row degenerate path.
# ===========================================================================
@cute.jit
def scan_cross(
    s_hist,
    s_ws,
    target,
    tidx,
    s_res,
    target2,
    blk: cutlass.Constexpr,
    nb: cutlass.Constexpr,
    two: cutlass.Constexpr = False,
):
    NWU = nb // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    c = cutlass.Int32(0)
    w = cutlass.Int32(0)
    if tidx < cutlass.Int32(nb):
        c = s_hist[tidx]
        w = warp_incl_scan_add(c, lane)
        if lane == cutlass.Int32(31):
            s_ws[wid] = w
    cute.arch.barrier()  # the ONE internal barrier
    if tidx < cutlass.Int32(nb):
        v2 = cutlass.Int32(0)
        if lane < cutlass.Int32(NWU):
            v2 = s_ws[lane]
        pre = warp_incl_scan_add(v2, lane)
        tot = cute.arch.shuffle_sync(pre, cutlass.Int32(31))
        off = cute.arch.shuffle_sync(pre - v2, wid)
        after = tot - (off + w)
        if tidx == cutlass.Int32(0):
            s_res[RES_TOT] = tot
        s_hist[tidx] = after  # output cursor
        cross = cutlass.Int32(0)
        if after < target:
            if (after + c) >= target:
                cross = cutlass.Int32(1)
            if tidx == cutlass.Int32(0):
                cross = cutlass.Int32(1)
        if cross != cutlass.Int32(0):
            s_res[RES_B] = tidx
            s_res[RES_ABOVE] = after
            s_res[RES_M] = c
        if cutlass.const_expr(two):
            cross2 = cutlass.Int32(0)
            if after < target2:
                if (after + c) >= target2:
                    cross2 = cutlass.Int32(1)
                if tidx == cutlass.Int32(0):
                    cross2 = cutlass.Int32(1)
            if cross2 != cutlass.Int32(0):
                s_res[RES_B2] = tidx


# ===========================================================================
# scan_cross_w<BLK, NB_>
# Register-path block-parallel suffix scan for NB_ >= BLK: every thread owns
# a private contiguous BPT = NB_/BLK span, so its read->write needs no
# barrier. EXACTLY ONE internal barrier. The second stage is TWO REDUCTIONS,
# not a scan: tot = redux_add(vv), off = redux_add((lane < wid) ? vv : 0) —
# wid is warp-uniform so the masked operand stays convergent.
# ===========================================================================
@cute.jit
def scan_cross_w(s_hist, s_ws, target, tidx, s_res, blk: cutlass.Constexpr, nb: cutlass.Constexpr):
    BPT = nb // blk
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    loc = cute.make_rmem_tensor((BPT,), cutlass.Int32)
    base = tidx * cutlass.Int32(BPT)
    sm = cutlass.Int32(0)
    for i in cutlass.range_constexpr(BPT):
        loc[i] = s_hist[base + cutlass.Int32(i)]
        sm = sm + loc[i]
    w = warp_incl_scan_add(sm, lane)
    if lane == cutlass.Int32(31):
        s_ws[wid] = w
    cute.arch.barrier()  # the ONE internal barrier
    vv = cutlass.Int32(0)
    if lane < cutlass.Int32(NW):
        vv = s_ws[lane]
    tot = cutlass.Int32(warp_add_i32(vv))
    sel = cutlass.Int32(0)
    if lane < wid:
        sel = vv
    off = cutlass.Int32(warp_add_i32(sel))
    after = tot - (off + w)
    if tidx == cutlass.Int32(0):
        s_res[RES_TOT] = tot
    for i in cutlass.range_constexpr(BPT - 1, -1, -1):
        cq = loc[i]
        s_hist[base + cutlass.Int32(i)] = after  # per-bin OUTPUT CURSOR
        gb = base + cutlass.Int32(i)
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
        after = after + cq


# ===========================================================================
# merge_scan0<NB_, CS>
# Warp-0-fused cluster merge + suffix scan: each lane reads its BPT-bin span
# from EVERY rank's hist via 16B DSMEM loads, sums the cluster totals (and
# the r<rank prefix that biases this rank's cursors) in registers, runs the
# suffix scan and writes the biased output cursors straight into mrg. ONE
# caller barrier (the post-scan publish) instead of two, and no hoff[]
# array at all. NO barrier inside.
# rank: this CTA's rank-in-cluster (dynamic Int32); cs: cluster size.
# ===========================================================================
@cute.jit
def merge_scan0(
    s_hist, s_mrg, rank, target, tidx, s_res, nb: cutlass.Constexpr, cs: cutlass.Constexpr
):
    BPT = nb // 32
    NV = BPT // 4
    if tidx < cutlass.Int32(32):
        lane = tidx
        atom = smem_atom_i32_128()
        # one mapa per rank on the hist base, offsets applied after
        mapped = [_mapa_shared_cluster(s_hist.iterator, cutlass.Int32(r)) for r in range(cs)]
        # pass 1: remote v4 accumulation of tot/pre per vector
        tot_r = []  # NV entries of [4 x Int32] cluster totals
        pre_r = []  # NV entries of [4 x Int32] r<rank prefixes
        sm = cutlass.Int32(0)
        for q in cutlass.range_constexpr(NV):
            boff = (lane * cutlass.Int32(BPT) + cutlass.Int32(4 * q)) * cutlass.Int32(4)
            t = [cutlass.Int32(0)] * 4
            p = [cutlass.Int32(0)] * 4
            for r in cutlass.range_constexpr(cs):
                v0, v1, v2, v3 = _ld_shared_cluster_v4_u32(mapped[r] + boff)
                t[0] = t[0] + v0
                t[1] = t[1] + v1
                t[2] = t[2] + v2
                t[3] = t[3] + v3
                if cutlass.Int32(r) < rank:  # predicated adds
                    p[0] = p[0] + v0
                    p[1] = p[1] + v1
                    p[2] = p[2] + v2
                    p[3] = p[3] + v3
            tot_r.append(t)
            pre_r.append(p)
            sm = sm + t[0] + t[1] + t[2] + t[3]
        # inclusive scan + totals
        w = warp_incl_scan_add(sm, lane)
        tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
        after = tt - w
        if lane == cutlass.Int32(0):
            s_res[RES_TOT] = tt
        base = lane * cutlass.Int32(BPT)
        # descending walk: crossing pin + prefix-biased cursors into mrg
        for q in cutlass.range_constexpr(NV - 1, -1, -1):
            o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for j in cutlass.range_constexpr(3, -1, -1):
                cq = tot_r[q][j]
                o4[j] = after + pre_r[q][j]
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
                after = after + cq
            boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
            sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)


# ===========================================================================
# gather_hint == GVR_GATHER_HINT(GM_, GX_, KPTV)
# LAZY block-wide (min,max) of logits[pre_idx[j]] over all k hint slots, in
# fkey space, returned as floats. Off the hot path by design: two dependent
# memory round trips (k coalesced P[j] words, then k scattered __ldg 4B
# gathers). Contains EXACTLY 2 barriers — call sites must be block-uniform.
# Outputs are block-uniform (every thread computes them).
# NaN-safe degeneracy guard: if !(GM < GX) both become sentinels.
#
# x_addr / p_addr: Int64 byte base addresses of THIS ROW of logits/pre_idx
# (pass `t.iterator.toint() + row * stride_bytes`). s_wmn/s_wmx: Uint32 smem
# tensors of >= blk//32 slots. Returns (gm, gx) Float32.
# Both round trips are issued as predicated flat batches.
# ===========================================================================
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


# ===========================================================================
# ==== family: main ============================================
# ===========================================================================
"""gvr_main — streaming self-sampling GVR top-K.

Ctor knobs (compile-time, mirror of the CUDA template params):
    BLK ∈ {1024, 512, 256}, U ∈ {1,2,4,8}, MINB ∈ {1,2,4}, NBS = 256,
    KPT ∈ {1,2,4,8}, SPLIT ∈ {True, False}
Derived constexprs (bit-identical to the CUDA):
    HB=NBS; KBIG=(KPT>=2 && KPT*BLK>=2048); SCPB=(BLK>=1024)?(SPLIT?8192:16384)
    :(KBIG?8192:4096); CMPB=(BLK>=1024)?(KBIG?4096:2048):1024; SHD=!SPLIT;
    VSTG=SPLIT||BLK>=512; PFD=(MINB<=2)?min(U,4):0; PF=PFD>0; NATT=SPLIT?1:3.

Signature (ABI parity with the CUDA form incl. dead SCAP_/CMP_):
    run(logits[b,npad] f32, pre_idx[b,k] i32, out[b,k] i32,
        n, npad, k, SCAP_, CMP_, R, SMP, TGT, Q, SS2, TGT2, ws)
Grid dim3(R, b) native 2-D; block BLK; min_blocks_per_mp=MINB is the
64-register wall; smem via one SmemAllocator blob (all extents compile-time),
dynamic-equivalent region byte-identical to the host dispatch formula:
(SCPB+4)*(VSTG?8:4) + (CMPB+1)*8.

int2 staging convention: an int2 (x=value bits, y=index) is ONE little-endian
Uint64 = (idx << 32) | value_bits, so cbuf2 / g_buf traffic is single u64
ld/st (__ldcg = _ldcg_v2_i32).

Barrier placement mirrors the CUDA source one-for-one (plus exactly 2 inside
each gather_hint expansion); scan_cross0 contains NO internal barrier. Do not
add or drop barriers.
"""


MAXC__main = C.MAXC
GCAP__main = C.GCAP
IDXB__main = C.IDXB
IDXM__main = C.IDXM
QUADC_CLUS__main = C.QUADC_CLUS
WS_BYTES = C.GVR_WS_BUF_OFF + MAXC__main * GCAP__main * 8  # 20,973,568

_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# single-rounding fma.rn.f32, used at every CUDA fmaf() site: T/Tk/T3 rung
# math, HIC, window terms. (x-T)*SC classify shapes stay plain sub+mul
# (structurally uncontractible).
# ---------------------------------------------------------------------------
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
    mlir_llvm.inline_asm(
        res=None,
        operands_=[
            saddr_i32.ir_value(loc=loc, ip=ip),
            lo_u32.ir_value(loc=loc, ip=ip),
            hi_u32.ir_value(loc=loc, ip=ip),
        ],
        asm_string="st.shared.v2.u32 [$0], {$1, $2};",
        constraints="r,r,r",
        has_side_effects=True,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _pin_i64(v, *, loc=None, ip=None):
    """Opaque identity mov.b64: pins a loop-invariant Int64 so NVVM cannot
    rematerialize its defining chain (param ld.const + %ctaid reads + mul/add)
    into every scf region body."""
    return cutlass.Int64(
        mlir_llvm.inline_asm(
            T.i64(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, $1;",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pin_i32(v, *, loc=None, ip=None):
    """Opaque identity mov.b32 (Int32 twin of _pin_i64)."""
    return cutlass.Int32(
        mlir_llvm.inline_asm(
            T.i32(),
            [v.ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
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
        mlir_llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "mov.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
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
    mlir_llvm.inline_asm(
        res=None,
        operands_=[addr.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cta.shared.add.u32 [$0], 1;",
        constraints="r",
        has_side_effects=True,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class GvrMainKernel:
    """gvr_main<BLK, U, MINB, NBS, KPT, SPLIT> — streaming self-sampling GVR."""

    def __init__(
        self,
        blk: int,
        u: int,
        minb: int,
        nbs: int,
        kpt: int,
        split: bool,
        tshg: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
        r_const: int = 1,
    ):
        assert nbs == 256, "SNB must stay 256"
        assert blk in (256, 512, 1024) and u in (1, 2, 4, 8)
        assert kpt in (1, 2, 4, 8) and minb in (1, 2, 4)
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.kpt = kpt
        self.split = bool(split)
        # ---- per-row varlen mode (production heuristicTopKDecode contract) --
        # n and the sampling-ladder scalars are re-derived PER ROW inside the
        # kernel from a device kv_lens tensor (route_dynamic formula mirror);
        # the scalar n/SMP/TGT/Q/SS2/TGT2 launch args become dead.  next_n /
        # cr_shift (log2 compressRatio: 0 = DSv3.2, 2 = DSv4) / r_const (the
        # frozen grid.x) are compile-time so their divisions strength-reduce.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        self.r_const = int(r_const)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2) and self.r_const >= 1
        # TSH-floor staging arm.  SPLIT-only compile-time key; the CUDA form
        # is a grid-uniform runtime gate over the same predicate
        # (b > 15 && k <= 1024 && n4 <= 32768).  varlen mode compiles the
        # machinery in whenever SPLIT and gates it per row at runtime
        # (tsh_en && n4 <= 32768) — mirroring the CUDA runtime gate.
        if self.varlen:
            self.tshg = bool(split)
        else:
            self.tshg = bool(tshg) and bool(split)
        # derived constexprs (bit-identical to the CUDA)
        self.hb = nbs
        self.kbig = (kpt >= 2) and (kpt * blk >= 2048)
        self.scpb = (8192 if split else 16384) if blk >= 1024 else (8192 if self.kbig else 4096)
        self.cmpb = (4096 if self.kbig else 2048) if blk >= 1024 else 1024
        self.shd = not split
        self.vstg = split or blk >= 512
        self.pfd = (u if u < 4 else 4) if minb <= 2 else 0
        self.pf = self.pfd > 0
        self.natt = 1 if split else 3
        # smem blob byte map: cbuf/cbuf2 alias @0,
        # ck64 @ 4*(VSTG ? 2*(SCPB+4) : SCPB+4), size (CMPB+1)*8
        self.ck_off = 4 * ((2 * (self.scpb + 4)) if self.vstg else (self.scpb + 4))
        assert self.ck_off % 16 == 0, "ck64 must stay 16B aligned (ulonglong2)"
        self.dyn_bytes = self.ck_off + (self.cmpb + 1) * 8
        self.lb = self.nbs.bit_length() - 1  # log2(NBS)=8

    # ------------------------------------------------------------------
    # GVR_EMITC: classify+stage one survivor.
    # Returns pos+1. Branchless trash slot min(pos, SCPB).
    # ------------------------------------------------------------------
    @cute.jit
    def _emitc(self, xv, idx, pos, TF, SC, hb, cb2, s_hist, s_cbuf, s_cbuf2):
        SCPB = self.scpb
        NBS = self.nbs
        if cutlass.const_expr(not self.split):
            bn_u = C.f2u_rz((xv - TF) * SC)  # saturating cvt.rzi
            if bn_u > cutlass.Uint32(NBS - 1):
                bn_u = cutlass.Uint32(NBS - 1)
            bn = cutlass.Int32(bn_u)
            if cutlass.const_expr(self.vstg):
                # result unused -> resultless red off the pinned hist base
                # (no per-site smem-base refold)
                _red_shared_add1(hb + (bn << cutlass.Int32(2)))
            else:
                # VSTG=False tuples sit at the 64-register wall: keep the
                # original spelling, no pinned base here
                C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
            if cutlass.const_expr(not self.vstg):
                ps = pos
                if ps > cutlass.Int32(SCPB):
                    ps = cutlass.Int32(SCPB)  # trash slot (IMNMX)
                s_cbuf[ps] = cutlass.Int32(
                    (bn_u << cutlass.Uint32(IDXB__main)) | cutlass.Uint32(idx)
                )
        if cutlass.const_expr(self.vstg):
            ps = pos
            if ps > cutlass.Int32(SCPB):
                ps = cutlass.Int32(SCPB)
            # int2 {value bits, idx} via st.shared.v2.u32 — same bytes as the
            # (idx << 32) | bits u64 pack (+0=bits, +4=idx), but no i64
            # materialization inside the bit-walk; address = one LEA off the
            # pinned cb2 base
            _st_s_v2_u32(cb2 + ps * cutlass.Int32(8), C.u32_of_f32(xv), cutlass.Uint32(idx))
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (shared by P6 and both degen
    # emits): q1 winners to out[base1+p] p<cap1, q2 ties to out[base2+p]
    # p<cap2. s_scal[1]=s_o1, s_scal[2]=s_o2.
    # ------------------------------------------------------------------
    @cute.jit
    def _ballot_pair_emit(self, p1, p2, idv, base1, cap1, base2, cap2, out_row, s_scal, lane):
        n1 = C.ballot(p1 != cutlass.Int32(0))
        n2 = C.ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = C.atomic_add_cta(s_scal.iterator + 1, cutlass.Int32(C.popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = C.atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(C.popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(C.popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(C.popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        KPT = self.kpt
        SCPB = self.scpb
        CMPB = self.cmpb
        PFD = self.pfd
        NATT = self.natt
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # 2-D grid (part, row)
        row = by
        part = cutlass.Int32(0)
        if cutlass.const_expr(self.split):
            part = bx
        lane = tidx & cutlass.Int32(31)

        # ================= per-row varlen prologue (varlen mode only) =========
        # Production contract: row r serves request r // next_n with
        # kv_len = kv_lens[r // next_n], n = (kv_len - next_n + r % next_n + 1)
        # >> cr_shift.  The sampling-ladder scalars are then re-derived from
        # this row's n by the EXACT route_dynamic() host formulas (the scalar
        # launch args are dead in this mode, except `n`, which carries the
        # envelope the per-row length is clamped to).  n <= k rows have no runtime
        # `return` in CuTe DSL: they run the body as a zero-work pass
        # (n = 0, TGT = INT_MAX so no rung ever accepts) and the identity/pad
        # emission happens in the epilogue at the end of the kernel.  Every
        # value below is a pure function of `row`, so all R split CTAs of a
        # row (and all threads) compute identical scalars — grid-uniform per
        # row by construction.
        short = cutlass.Int32(0)
        n_row = cutlass.Int32(0)
        tsh_run = cutlass.Int32(1)
        if cutlass.const_expr(self.varlen):
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            # Clamp to the logical envelope carried in the (otherwise dead)
            # `n` launch slot, NOT to the row stride `npad`: on a wider-stride
            # arena view the columns in [n, npad) belong to the arena, never
            # to this row, so an oversized kv_len must not classify them
            # (same clamp the reg / reg_clus / clus varlen prologues apply).
            if nv > n:
                nv = n
            n_row = nv
            if nv <= k:
                short = cutlass.Int32(1)
                n = cutlass.Int32(0)
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                # "never accepts" sentinels; 2^30-1 so the TGT*2 scan target
                # stays positive (0x7FFFFFFF would overflow to -2 and flip
                # every tot0 >= TGT*2 gate on the all-zero histogram)
                TGT = cutlass.Int32(0x3FFFFFFF)
                TGT2 = cutlass.Int32(0x3FFFFFFF)
                Q = cutlass.Int32(0)
            if short == cutlass.Int32(0):
                n = nv
                n4v = n >> cutlass.Int32(2)
                # Ladder-scalar baselines only: the real SMP/SS2/TGT/TGT2 are
                # derived by warp0 alone in the block below (bit-identical
                # formulas) and published through s_lad — every thread's local
                # copies here are overwritten by the post-barrier smem read.
                SMP = cutlass.Int32(0)
                SS2 = cutlass.Int32(1)
                TGT = cutlass.Int32(0)
                TGT2 = cutlass.Int32(0)
                if cutlass.const_expr(self.split):
                    Q = (n4v + cutlass.Int32(self.r_const - 1)) // cutlass.Int32(self.r_const)
                else:
                    Q = cutlass.Int32(0)
            # per-row TSH-floor runtime gate (CUDA parity: b>15 && k<=1024 in
            # tsh_en, n4 <= 32768 per row)
            tsh_run = cutlass.Int32(0)
            if tsh_en != cutlass.Int32(0):
                if (n >> cutlass.Int32(2)) <= cutlass.Int32(32768):
                    tsh_run = cutlass.Int32(1)

        # ---- shared memory (one blob, compile-time offsets) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        s_ws = smem.allocate_tensor(  # unused; byte parity  # noqa: F841
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # crossing-scan result slots (RES_B/M/ABOVE/TOT/B2/B3)
        s_res = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2 [3]=s_base
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_pk = smem.allocate_tensor(
            cutlass.Int64, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=8
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        # STATIC smem word for the walk's byte stride — kept out of the blob
        # so dyn_bytes stays equal to the CUDA dispatch's smem. blk==512 VSTG
        # only.
        if cutlass.const_expr(self.vstg and self.blk == 512):
            s_x4 = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
            )
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        if cutlass.const_expr(self.varlen):
            # ladder broadcast slots: [0]=SMP [1]=SS2 [2]=TGT [3]=TGT2
            # (static like s_x4, so dyn_bytes keeps CUDA dispatch parity)
            s_lad = smem.allocate_tensor(
                cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
            )
        blob = smem.allocate_tensor(  # dynamic-equivalent region
            cutlass.Int8, cute.make_ordered_layout((self.dyn_bytes,), order=(0,)), byte_alignment=16
        )
        sbase = blob.iterator.toint()
        s_cbuf = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        s_cbuf2 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((SCPB + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((CMPB + 1,)),
        )

        # emission smem bases pinned ONCE, outside the attempt/tile loops
        # (asm identity mov) — LLVM otherwise refolds the shared-window
        # materialisation into every _emitc site inside the divergent
        # bit-walk. VSTG-only: the VSTG=False tuples keep their original
        # spellings untouched (64-register wall).
        hb_pin = cutlass.Int32(0)
        cb2_pin = cutlass.Int32(0)
        if cutlass.const_expr(self.vstg):
            hb_pin = _smem_addr_reg(s_hist.iterator.toint())
            cb2_pin = _smem_addr_reg(s_cbuf2.iterator.toint())
        # park the stride 4 in the dedicated smem word and load it back —
        # the LDS result is opaque to ptxas, so the walk's stride register
        # cannot be re-materialized in-loop (asm-mov and shfl forms are
        # folded by ptxas value-tracking). blk==512 family ONLY: the other
        # arms sit at the 64-register wall. Threads are converged here
        # (kernel prologue), so the one extra barrier is safe.
        x4_pin = cutlass.Int32(4)
        if cutlass.const_expr(self.vstg and self.blk == 512):
            if tidx == cutlass.Int32(0):
                s_x4[0] = cutlass.Int32(4)
            cute.arch.barrier()
            x4_pin = s_x4[0]

        # ---- row bases ----
        row64 = cutlass.Int64(row)
        # _pin_i64: keep the row base a REGISTER across the attempt/tile scf
        # regions (NVVM otherwise re-derives ld.param+%ctaid.y+mul per region)
        x_addr = _pin_i64(logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4))
        # varlen: pre_idx is REQUEST-level [num_rows/next_n, k] — a request's
        # next_n rows share one hint row (production contract); legacy mode
        # keeps the per-row mapping (next_n == 1 makes them identical).
        prow64 = row64
        if cutlass.const_expr(self.varlen):
            prow64 = cutlass.Int64(row // cutlass.Int32(self.next_n))
        p_addr = pre_idx.iterator.toint() + prow64 * cutlass.Int64(k) * cutlass.Int64(4)
        out_row = out[row, None]
        ws_addr = ws.iterator.toint()
        gdon_addr = ws_addr  # slab views
        goff_addr = ws_addr + cutlass.Int64(C.GVR_WS_OFF_OFF)
        gbuf_addr = ws_addr + cutlass.Int64(C.GVR_WS_BUF_OFF)
        # SPLIT only: row-slab base pinned like x_addr above; the
        # publish/gather/P5/degen consumers spell gbuf_row + i*8 instead of
        # re-deriving gbuf_addr + (row64*GCAP__main + i)*8 per candidate
        # (value-identical by i64 distributivity).
        gbuf_row = cutlass.Int64(0)
        if cutlass.const_expr(self.split):
            gbuf_row = _pin_i64(gbuf_addr + row64 * cutlass.Int64(GCAP__main) * cutlass.Int64(8))

        n4 = n >> cutlass.Int32(2)
        c0 = cutlass.Int32(0)
        c1 = n4
        if cutlass.const_expr(self.split):
            c0 = part * Q
            c1 = c0 + Q
            if c1 > n4:
                c1 = n4
        tail0 = n4 << cutlass.Int32(2)
        tailn = cutlass.Int32(0)
        if part == cutlass.Int32(0):
            tailn = n - tail0

        if tidx == cutlass.Int32(0):
            s_scal[0] = cutlass.Int32(0)  # s_bufn
            s_res[C.RES_B2] = cutlass.Int32(-1)
            s_res[C.RES_B3] = cutlass.Int32(-1)
        if tidx < cutlass.Int32(self.hb):  # HB<=BLK always
            s_hist[tidx] = cutlass.Int32(0)

        # ===== varlen: warp0-only ladder mirror + register-free L2 hints =====
        # The sampling-ladder scalars are a pure function of the row; issuing
        # the mirror chain (runtime divides + isqrt fixups) per thread costs
        # more instructions than the rest of the kernel on 1-row launches.
        # warp0 alone walks the chain and publishes the four derived scalars
        # through s_lad; the other warps spend the wait issuing L2 prefetch
        # hints for this CTA's own P3 slice (register-free, so zero pressure
        # on the 64-register arms — the PRIME-LATE register loads below are
        # untouched and simply hit L2).  Values are bit-identical to the
        # per-thread derivation this replaces.
        if cutlass.const_expr(self.varlen):
            if tidx < cutlass.Int32(32):
                if short == cutlass.Int32(0):
                    # ---- aim ladder (cheap mirror) ----
                    # The ladder scalars steer the sampling rung only —
                    # exactness is schedule-invariant (retry/degen close every
                    # miss), so +-1 drift vs the host double form is allowed.
                    # Serial latency dominates (this chain sits in front of a
                    # barrier): runtime divides become MUFU.RCP multiplies and
                    # the isqrt fixup loops collapse to single steps (the f32
                    # sqrt of an exactly-representable int (6n <= 2^23) is
                    # within 1 of isqrt, so one correction per side suffices).
                    # Q (chunk ownership) stays exact — compile-time divisor.
                    x6 = cutlass.Int32(6) * n
                    ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                    if ri * ri > x6:
                        ri = ri - cutlass.Int32(1)
                    if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                        ri = ri + cutlass.Int32(1)
                    r6 = ri
                    if x6 - ri * ri > ri:
                        r6 = ri + cutlass.Int32(1)
                    aim = aim_base
                    if r6 > aim:
                        aim = r6
                    if cutlass.const_expr(self.r_const > 1):
                        if aim < amin:
                            aim = amin
                    scap_c = cutlass.Int32(SCPB)  # SCAP == SCPB for gvr_main (proven identity)
                    if aim > (scap_c >> cutlass.Int32(1)):
                        aim = scap_c >> cutlass.Int32(1)
                    if aim < k:
                        aim = k
                    n4w = n >> cutlass.Int32(2)
                    # pair-sample gate: (n > SCAP or small_dense) and n4 >= 4;
                    # small_dense = k > 1024 and not big and n <= SCAP and n > 2k
                    # (k/big folded into the launch-constant sd_en flag).
                    gate = cutlass.Int32(0)
                    if n > scap_c:
                        gate = cutlass.Int32(1)
                    if sd_en != cutlass.Int32(0):
                        if n <= scap_c:
                            if n > (k << cutlass.Int32(1)):
                                gate = cutlass.Int32(1)
                    if n4w < cutlass.Int32(4):
                        gate = cutlass.Int32(0)
                    if gate != cutlass.Int32(0):
                        # sel = sfac*n // aim via rcp (sfac*n <= 2^24: f32-exact
                        # to the last unit; quotient error < 1 => +-1 drift)
                        sel = cutlass.Int32(
                            cutlass.Float32(sfac * n) * cute.arch.rcp_approx(cutlass.Float32(aim))
                        )
                        if sel < cutlass.Int32(256):
                            sel = cutlass.Int32(256)
                        nh = n >> cutlass.Int32(1)
                        if sel > nh:
                            sel = nh
                        pairs = sel >> cutlass.Int32(3)
                        if pairs < cutlass.Int32(1):
                            pairs = cutlass.Int32(1)
                        half = n4w >> cutlass.Int32(1)
                        if half < cutlass.Int32(1):
                            half = cutlass.Int32(1)
                        if pairs > half:
                            pairs = half
                        SS2 = cutlass.Int32(
                            cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(pairs))
                        )
                        if SS2 < cutlass.Int32(1):
                            SS2 = cutlass.Int32(1)
                        SMP = cutlass.Int32(
                            cutlass.Float32(half) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                        )
                        # sample-window guard: the P1 gather indexes up to
                        # ~SMP*SS2*2 f32x4 lines; keep SMP*SS2 <= half so the
                        # window never walks past the row (approx error is
                        # bounded by +1, one decrement closes it)
                        if SMP * SS2 > half:
                            SMP = SMP - cutlass.Int32(1)
                        if SMP < cutlass.Int32(1):
                            SMP = cutlass.Int32(1)
                        # TGT/TGT2: i64 products // n -> f32 mul + one rcp(n).
                        # aim/SMP/k/n are all f32-exact here (<= 2^20); the
                        # quotients are <= 8*aim ~ 2^16, so the approx error
                        # stays far below 1 unit — +-1 at worst on the floor.
                        rn_ = cute.arch.rcp_approx(cutlass.Float32(n))
                        smp8f = cutlass.Float32(SMP) * cutlass.Float32(8.0)
                        TGT = cutlass.Int32(cutlass.Float32(aim) * smp8f * rn_)
                        if TGT < cutlass.Int32(1):
                            TGT = cutlass.Int32(1)
                        TGT2 = cutlass.Int32(cutlass.Float32(k) * smp8f * rn_)
                        if TGT2 < cutlass.Int32(1):
                            TGT2 = cutlass.Int32(1)
                if tidx == cutlass.Int32(0):
                    s_lad[0] = SMP
                    s_lad[1] = SS2
                    s_lad[2] = TGT
                    s_lad[3] = TGT2
            # Register-free L2 hints for the first U-batch of this CTA's own
            # P3 slice (clamped in-row): the data P3 touches first starts
            # flowing while warp0 walks the chain. Short rows clamp every
            # hint to the row's last line — harmless.
            plim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            for uu in cutlass.range_constexpr(U):
                # NOTE: names must not collide with the PRIME-LATE block's
                # i_/ic — the DSL kills inner-scope names at region exit and
                # a later same-name assignment inside a dynamic `if` trips
                # "is None prior to this if".
                pic = c0 + tidx + cutlass.Int32(uu * BLK)
                if pic >= c1:
                    pic = plim4
                C._prefetch_l2(x_addr + cutlass.Int64(pic) * cutlass.Int64(16))
            cute.arch.barrier()  # publish s_lad (also covers the smem inits)
            SMP = s_lad[0]
            SS2 = s_lad[1]
            TGT = s_lad[2]
            TGT2 = s_lad[3]

        # ============ P1: sample prefetch (hint gather LAZY) =================
        atom128 = C.g2r_atom_f32(128, invariant=True)
        fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
        fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
        shas = cutlass.Int32(0)
        if tidx < SMP:
            shas = cutlass.Int32(1)
        if shas != cutlass.Int32(0):
            p4 = tidx * SS2 * cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fsa)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

        # ============ P2: quantile rung from the sample ======================
        smn = cutlass.Float32(float("inf"))
        smx = cutlass.Float32(float("-inf"))
        if shas != cutlass.Int32(0):
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsa[t])
                smx = C.fmax_f32(smx, fsa[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fsb[t])
                smx = C.fmax_f32(smx, fsb[t])
        fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # strided-tail pair bufs
        fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
        j = tidx + cutlass.Int32(BLK)  # strided tail
        while j < SMP:
            p4 = j * SS2 * cutlass.Int32(2)
            C.ld_g_f32x4(atom128, x_addr, p4, fma_)
            C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fma_[t])
                smx = C.fmax_f32(smx, fma_[t])
            for t in cutlass.range_constexpr(4):
                smn = C.fmin_f32(smn, fmb_[t])
                smx = C.fmax_f32(smx, fmb_[t])
            j = j + cutlass.Int32(BLK)
        a0 = C.warp_min_u32(C.fkey(smn))
        c0m = C.warp_max_u32(C.fkey(smx))
        if lane == cutlass.Int32(0):
            s_wmn[tidx >> cutlass.Int32(5)] = a0
            s_wmx[tidx >> cutlass.Int32(5)] = c0m
        cute.arch.barrier()  # ---- barrier (sample redux publish) ----

        # PRIME-LATE prefetch block: strictly after the barrier.
        lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
        pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(PFD, 1))]
        if cutlass.const_expr(self.pf):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):  # prime, full slice
                for uu in cutlass.range_constexpr(PFD):
                    C.ld_g_f32x4(atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu])
            else:  # clamped prime
                for uu in cutlass.range_constexpr(PFD):
                    i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                    ic = i_
                    if ic >= c1:
                        ic = lim4
                    C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
            # asm prefetch site #1: gate (c1-c0)>=2*BLK*U && SMP>=160
            g1 = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(2 * BLK * U):
                if SMP >= cutlass.Int32(160):
                    g1 = cutlass.Int32(1)
            if g1 != cutlass.Int32(0):
                for uu in cutlass.range_constexpr(PFD, U):
                    C._prefetch_l2(
                        x_addr
                        + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                    )
        if cutlass.const_expr((not self.pf) and (not self.split)):
            fullsl = cutlass.Int32(0)
            if (c1 - c0) >= cutlass.Int32(BLK * U):
                fullsl = cutlass.Int32(1)
            if fullsl != cutlass.Int32(0):  # prefetch site #2
                for uu in cutlass.range_constexpr(U):
                    C._prefetch_l2(
                        x_addr
                        + cutlass.Int64(c0 + tidx + cutlass.Int32(uu * BLK)) * cutlass.Int64(16)
                    )
            else:
                if SMP > cutlass.Int32(0):  # prefetch site #3
                    for uu in cutlass.range_constexpr(U):
                        i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4
                        C._prefetch_l2(x_addr + cutlass.Int64(ic) * cutlass.Int64(16))

        # cross-warp sample reduce
        av = cutlass.Uint32(0xFFFFFFFF)
        cv = cutlass.Uint32(0)
        if lane < cutlass.Int32(NW):
            av = s_wmn[lane]
            cv = s_wmx[lane]
        SMIN = C.invkey(C.warp_min_u32(av))
        SMAX = C.invkey(C.warp_max_u32(cv))

        GMIN = cutlass.Float32(C.SENT_LO)  # sentinels
        GMAX = cutlass.Float32(C.SENT_HI)
        T = cutlass.Float32(_NEG_INF)
        HIC = cutlass.Float32(_NEG_INF)
        w = cutlass.Float32(0.0)
        sok = cutlass.Int32(0)
        if SMP > cutlass.Int32(0):
            if SMAX > SMIN:
                sok = cutlass.Int32(1)
        if sok != cutlass.Int32(0):  # sample histogram
            w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
            # rcp.approx.ftz.f32 = the CUDA arm's --use_fast_math 1.0f/w
            # (bare MUFU.RCP, no Newton refinement) — bitwise-aligned scale
            sc_s = cute.arch.rcp_approx(w)
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
            j = tidx + cutlass.Int32(BLK)  # tail re-loads
            while j < SMP:
                p4 = j * SS2 * cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                for t in cutlass.range_constexpr(4):
                    bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                    if bq > cutlass.Int32(NBS - 1):
                        bq = cutlass.Int32(NBS - 1)
                    C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                j = j + cutlass.Int32(BLK)
        cute.arch.barrier()  # ---- barrier (sample histogram) ----
        # triple-target ZERO scan: TGT / TGT2 / 2*TGT
        # (THREE = SHD || gated-SPLIT)
        C.scan_cross0(
            s_hist,
            TGT,
            tidx,
            s_res,
            TGT2,
            TGT * cutlass.Int32(2),
            s_hist,
            nb=NBS,
            zero=True,
            two=True,
            three=(self.shd or self.tshg),
        )
        cute.arch.barrier()  # ---- barrier (scan publish) ----

        tot0 = s_res[C.RES_TOT]
        b1v = s_res[C.RES_B]
        if sok != cutlass.Int32(0):
            if tot0 >= TGT:
                T = _fmaf(cutlass.Float32(b1v), w, SMIN)
        Trung = T  # snapshot
        needg = cutlass.Int32(1)  # degenerate sample
        if T > cutlass.Float32(_NEG_INF):
            needg = cutlass.Int32(0)
        if needg != cutlass.Int32(0):
            GMIN, GMAX = C.gather_hint(
                x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
            )  # 2 barriers inside
            T = GMIN
        if sok != cutlass.Int32(0):  # HIC tighten
            if tot0 >= TGT:
                b2v = s_res[C.RES_B2]
                if b2v >= cutlass.Int32(0):
                    Tk = _fmaf(cutlass.Float32(b2v), w, SMIN)
                    anch = T
                    if cutlass.const_expr(not self.split):
                        anch = C.fmin_f32(T, Trung)
                    d_ = C.fmax_f32(Tk - anch, cutlass.Float32(0.0))
                    HIC = C.fmax_f32(
                        _fmaf(cutlass.Float32(4.0), d_, T), _fmaf(cutlass.Float32(8.0), w, T)
                    )
        if cutlass.const_expr(self.shd or self.tshg):  # TSH floor (+gated SPLIT)
            if tidx == cutlass.Int32(0):
                t5 = cutlass.Float32(_NEG_INF)
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[C.RES_B3]
                        if b3v >= cutlass.Int32(0):
                            if T > GMIN:
                                T3 = _fmaf(cutlass.Float32(b3v), w, SMIN)
                                if T3 < T:
                                    t5 = T3
                s_tsh[0] = t5

        if cutlass.const_expr(self.tshg):
            # TSH-FLOOR STAGING: SPLIT has no retry ladder, so a rung
            # overshoot (count(>=T) < k) used to hand the LAST CTA a
            # single-CTA whole-row narrowing.  Stage at the sample's
            # rank-(2*TGT) floor instead: staged population ~aim -> ~2*aim,
            # and the merged histogram contains the k-crossing whenever
            # count(>=TSH) >= k.  TSH miss falls to GMIN/degen unchanged.
            cute.arch.barrier()
            t5s = s_tsh[0]
            # varlen: per-row runtime gate (tsh_run == 1 always in legacy
            # mode, so legacy codegen semantics are unchanged)
            if tsh_run != cutlass.Int32(0):
                if t5s > cutlass.Float32(_NEG_INF):
                    if t5s < T:
                        T = t5s

        # ============ attempt loop — MUST NOT unroll ============
        listN = cutlass.Int32(0)
        above = cutlass.Int32(0)
        m = cutlass.Int32(0)
        need = cutlass.Int32(0)
        B = cutlass.Int32(0)
        SC = cutlass.Float32(1.0)
        TF = T
        complete = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        fromg = cutlass.Int32(0)
        alive = cutlass.Int32(1)

        fr = [
            cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(max(U - PFD, 1))
        ]  # explicit batch
        att = cutlass.Int32(0)
        running = cutlass.Int32(1)
        while running != cutlass.Int32(0):
            if cutlass.const_expr(not self.split):  # SPLIT never retries (NATT=1)
                if att > cutlass.Int32(0):  # retry reset
                    if cutlass.const_expr(self.pf):
                        # exactness: re-prime pf[] (holds stale roll data)
                        fullsl = cutlass.Int32(0)
                        if (c1 - c0) >= cutlass.Int32(BLK * U):
                            fullsl = cutlass.Int32(1)
                        if fullsl != cutlass.Int32(0):
                            for uu in cutlass.range_constexpr(PFD):
                                C.ld_g_f32x4(
                                    atom128, x_addr, c0 + tidx + cutlass.Int32(uu * BLK), pf[uu]
                                )
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                i_ = c0 + tidx + cutlass.Int32(uu * BLK)
                                ic = i_
                                if ic >= c1:
                                    ic = lim4
                                C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (retry reset) ----

            TF = T  # window
            hi = C.fmax_f32(GMAX, T)
            if HIC > T:
                if HIC < hi:
                    hi = HIC
            WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
            wdok = cutlass.Int32(0)
            if WD > cutlass.Float32(0.0):
                wdok = cutlass.Int32(1)
            if wdok == cutlass.Int32(0):
                WD = cutlass.Float32(1e-30)
            # CUDA compiles its own `1.0f / WD` here to a bare MUFU.RCP
            # (approximate); div.rn's dependent rcp+Newton+CALL chain
            # serializes the attempt prologue. blk==512 ONLY: the
            # (256,8,4,·) family keeps the original div.rn spelling below.
            if cutlass.const_expr(self.blk == 512):
                SC = cute.arch.rcp_approx(WD)
            else:
                SC = cutlass.Float32(1.0) / WD

            # ---- P3 row pass ----
            span = c1 - c0
            step = cutlass.Int32(BLK * U)
            nFull = cutlass.Int32(0)
            rem = cutlass.Int32(0)
            if span > cutlass.Int32(0):  # peel
                nFull = span // step
                rem = span - nFull * step
            # _pin_i32: the isfull peel predicate reads nFull every tile iter;
            # unpinned, NVVM re-derives the whole ld.param+shr/sel div chain
            # at the loop head
            nFull = _pin_i32(nFull)
            nIt = nFull
            if rem > cutlass.Int32(0):
                nIt = nIt + cutlass.Int32(1)
            # _pin_i32: stop NVVM re-deriving the ceil-div bound (ld.param n +
            # shr/sel chain) inside the tile-loop condition region per iter
            nIt = _pin_i32(nIt)

            it = cutlass.Int32(0)
            while it < nIt:
                i0 = c0 + it * step + tidx
                M = cutlass.Int32(0)
                isfull = cutlass.Int32(0)
                if it < nFull:
                    isfull = cutlass.Int32(1)
                if isfull != cutlass.Int32(0):  # full body
                    for uu in cutlass.range_constexpr(PFD, U):
                        C.ld_g_f32x4(atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        for q in cutlass.range_constexpr(4):
                            M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                else:  # partial body
                    for uu in cutlass.range_constexpr(PFD, U):
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        ic = i_
                        if ic >= c1:
                            ic = lim4  # clamped address
                        C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                    for uu in cutlass.range_constexpr(U):
                        if cutlass.const_expr(uu < PFD):
                            vv = pf[uu]
                        else:
                            vv = fr[uu - PFD]
                        i_ = i0 + cutlass.Int32(uu * BLK)
                        okq = cutlass.Int32(0)
                        if i_ < c1:
                            okq = cutlass.Int32(1)
                        if okq != cutlass.Int32(0):  # ok-gated (+inf-pad escape)
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                # prefetch roll-forward BEFORE reservation/walk
                if cutlass.const_expr(self.pf):
                    hasnext = cutlass.Int32(0)
                    if it + cutlass.Int32(1) < nIt:
                        hasnext = cutlass.Int32(1)
                    if hasnext != cutlass.Int32(0):
                        j0 = i0 + step
                        infull = cutlass.Int32(0)  # warp-uniform peel
                        if it + cutlass.Int32(1) < nFull:
                            infull = cutlass.Int32(1)
                        if infull != cutlass.Int32(0):
                            for uu in cutlass.range_constexpr(PFD):
                                C.ld_g_f32x4(atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu])
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                j_ = j0 + cutlass.Int32(uu * BLK)
                                jc = j_
                                if jc >= c1:
                                    jc = lim4
                                C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                # warp-aggregated reservation
                cnt = cutlass.Int32(C.popc(M))
                inc = C.warp_incl_scan_add(cnt, lane)
                bpos = cutlass.Int32(0)
                if lane == cutlass.Int32(31):
                    if inc != cutlass.Int32(0):
                        bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                # survivor bit-walk, software-pipelined ONE deep;
                # reload X[idx] — do NOT hold the U float4s (spills)
                if M != cutlass.Int32(0):
                    bp = C.ffs_m1(M)
                    M = M & (M - cutlass.Int32(1))
                    idx = (
                        (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK)) << cutlass.Int32(2)
                    ) + (bp & cutlass.Int32(3))
                    if cutlass.const_expr(self.vstg and self.blk == 512):
                        xv = _ldg_f32_rs(x_addr, idx, x4_pin)
                    else:
                        xv = C.ldg_f32(x_addr, idx)
                    while M != cutlass.Int32(0):
                        bp2 = C.ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx2 = (
                            (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                            << cutlass.Int32(2)
                        ) + (bp2 & cutlass.Int32(3))
                        if cutlass.const_expr(self.vstg and self.blk == 512):
                            xv2 = _ldg_f32_rs(x_addr, idx2, x4_pin)
                        else:
                            xv2 = C.ldg_f32(x_addr, idx2)
                        pos = self._emitc(
                            xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                        )
                        idx = idx2
                        xv = xv2
                    pos = self._emitc(
                        xv, idx, pos, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                    )
                it = it + cutlass.Int32(1)
            # scalar tail, part 0 only
            i = tidx
            while i < tailn:
                x = C.ldg_f32(x_addr, tail0 + i)
                if x >= TF:
                    post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                    post = self._emitc(
                        x, tail0 + i, post, TF, SC, hb_pin, cb2_pin, s_hist, s_cbuf, s_cbuf2
                    )
                i = i + cutlass.Int32(BLK)
            cute.arch.barrier()  # ---- barrier (row pass) ----
            myn = s_scal[0]

            if cutlass.const_expr(self.split):
                # ---- SLAB HAND-OFF; exactly ONE attempt ----
                if tidx == cutlass.Int32(0):
                    pgo = cute.make_ptr(
                        cutlass.Int32,
                        goff_addr + row64 * cutlass.Int64(4),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    s_scal[3] = cutlass.Int32(cute.arch.atomic_add(pgo, myn))
                cute.arch.barrier()  # ---- barrier (slab offset) ----
                base = s_scal[3]
                if myn <= cutlass.Int32(SCPB):  # coalesced publish
                    i = tidx
                    while i < myn:
                        p = base + i
                        if p < cutlass.Int32(GCAP__main):
                            _st_g_u64(gbuf_row + cutlass.Int64(p) * cutlass.Int64(8), s_cbuf2[i])
                        i = i + cutlass.Int32(BLK)
                else:  # overflow re-sweep
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (overflow reset) ----
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i = lo2 + tidx
                    while i < hi2:
                        x = C.ldg_f32(x_addr, i)
                        if x >= TF:
                            pq = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP__main):
                                _st_g_u64(
                                    gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                    (cutlass.Uint64(cutlass.Uint32(i)) << cutlass.Uint64(32))
                                    | cutlass.Uint64(C.u32_of_f32(x)),
                                )
                        i = i + cutlass.Int32(BLK)
                    i = tidx  # true tail
                    while i < tailn:
                        x = C.ldg_f32(x_addr, tail0 + i)
                        if x >= TF:
                            pq = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                            p = base + pq
                            if p < cutlass.Int32(GCAP__main):
                                _st_g_u64(
                                    gbuf_row + cutlass.Int64(p) * cutlass.Int64(8),
                                    (
                                        cutlass.Uint64(cutlass.Uint32(tail0 + i))
                                        << cutlass.Uint64(32)
                                    )
                                    | cutlass.Uint64(C.u32_of_f32(x)),
                                )
                        i = i + cutlass.Int32(BLK)
                cute.arch.barrier()  # ---- barrier (slab publish) ----
                if tidx == cutlass.Int32(0):  # release + RMW
                    C.threadfence_gpu()
                    pdon = cute.make_ptr(
                        cutlass.Int64,
                        gdon_addr + row64 * cutlass.Int64(8),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    s_pk[0] = C.atomic_add_u64_gpu(
                        pdon, cutlass.Int64(1 << 32) + cutlass.Int64(myn)
                    )
                cute.arch.barrier()  # ---- barrier (arrival word) ----
                pk = s_pk[0]
                alive = cutlass.Int32(0)  # last-CTA test
                if cutlass.Int32(pk >> cutlass.Int64(32)) == R - cutlass.Int32(1):
                    alive = cutlass.Int32(1)
                if alive != cutlass.Int32(0):
                    C.threadfence_gpu()  # acquire
                    if tidx == cutlass.Int32(0):  # ZERO-RESTORE
                        _st_g_u32(goff_addr + row64 * cutlass.Int64(4), cutlass.Int32(0))
                        _st_g_u64(gdon_addr + row64 * cutlass.Int64(8), cutlass.Uint64(0))
                    total = cutlass.Int32(pk & cutlass.Int64(0xFFFFFFFF)) + myn
                    if total <= cutlass.Int32(GCAP__main):  # one-pass consume
                        listN = total
                        if total > cutlass.Int32(SCPB):
                            fromg = cutlass.Int32(1)
                        i = tidx
                        while i < listN:
                            gvx, gvy = C._ldcg_v2_i32(
                                gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                            )
                            if fromg == cutlass.Int32(0):
                                s_cbuf2[i] = (
                                    cutlass.Uint64(cutlass.Uint32(gvy)) << cutlass.Uint64(32)
                                ) | cutlass.Uint64(cutlass.Uint32(gvx))
                            bq = C.f2s_rz((C.f32_of_i32(gvx) - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            # resultless red off the pinned hist base
                            _red_shared_add1(hb_pin + (bq << cutlass.Int32(2)))
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # ---- barrier (slab histogram) ----
                        C.scan_cross0(
                            s_hist,
                            k,
                            tidx,
                            s_res,
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            s_hist,
                            nb=NBS,
                            zero=False,
                        )
                        cute.arch.barrier()  # ---- barrier (scan publish) ----
                        if s_res[C.RES_TOT] >= k:
                            valid = cutlass.Int32(1)
                            complete = cutlass.Int32(1)
                            above = s_res[C.RES_ABOVE]
                            m = s_res[C.RES_M]
                            need = k - above
                            B = s_res[C.RES_B]
                running = cutlass.Int32(0)  # break (NATT==1)
            else:
                # ---- non-split verify + rung ladder ----
                C.scan_cross0(
                    s_hist,
                    k,
                    tidx,
                    s_res,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    s_hist,
                    nb=NBS,
                    zero=False,
                )
                cute.arch.barrier()  # ---- barrier (verify scan) ----
                tot = s_res[C.RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):  # accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= cutlass.Int32(SCPB):
                        complete = cutlass.Int32(1)
                    listN = myn
                    above = s_res[C.RES_ABOVE]
                    m = s_res[C.RES_M]
                    need = k - above
                    B = s_res[C.RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(NATT - 1):  # ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        tshtaken = cutlass.Int32(0)  # TSH retry
                        if cutlass.const_expr(self.shd):
                            if att == cutlass.Int32(0):
                                T5 = s_tsh[0]
                                if T5 > cutlass.Float32(_NEG_INF):
                                    if T5 < TF:
                                        T = T5
                                        tshtaken = cutlass.Int32(1)
                        if tshtaken != cutlass.Int32(0):
                            cute.arch.barrier()  # ---- barrier (TSH retry) ----
                        else:
                            # LAZY GATHER (sentinel equality flag)
                            if GMIN == cutlass.Float32(C.SENT_LO):
                                GMIN, GMAX = C.gather_hint(
                                    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=KPT
                                )
                            floorhit = cutlass.Int32(1)
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN
                                cute.arch.barrier()  # ---- barrier (floor retry) ----
            att = att + cutlass.Int32(1)

        # ============ classification ============
        if alive != cutlass.Int32(0):
            whole = cutlass.Int32(0)
            if valid != cutlass.Int32(0):
                if need >= m:
                    whole = cutlass.Int32(1)
            lim1 = above
            if whole != cutlass.Int32(0):
                lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > cutlass.Int32(CMPB):
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m

            if degen == cutlass.Int32(0):
                # ---- P5 cursor emit ----
                if complete != cutlass.Int32(0):
                    i = tidx
                    while i < listN:
                        idv = cutlass.Int32(0)
                        bq = cutlass.Int32(0)
                        xv = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.vstg):
                            vx = cutlass.Int32(0)
                            vy = cutlass.Int32(0)
                            if cutlass.const_expr(self.split):
                                if fromg != cutlass.Int32(0):
                                    vx, vy = C._ldcg_v2_i32(
                                        gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                    )
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(
                                        cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            else:
                                pk64 = s_cbuf2[i]
                                vx = cutlass.Int32(
                                    cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                )
                                vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                            xv = C.f32_of_i32(vx)
                            idv = vy
                            bq = C.f2s_rz((xv - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                        else:
                            wpk = cutlass.Uint32(s_cbuf[i])
                            idv = cutlass.Int32(wpk & cutlass.Uint32(IDXM__main))
                            bq = cutlass.Int32(wpk >> cutlass.Uint32(IDXB__main))
                        if bq >= B:
                            p = C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                            if p < lim1:
                                out_row[p] = idv
                            else:
                                if whole == cutlass.Int32(0):
                                    q2 = p - above
                                    if q2 < cutlass.Int32(CMPB):
                                        if cutlass.const_expr(self.vstg):
                                            kk = C.fkey(xv)
                                        else:
                                            kk = C.fkey(C.ldg_f32(x_addr, idv))
                                        s_ck64[q2] = (
                                            cutlass.Uint64(kk) << cutlass.Uint64(32)
                                        ) | cutlass.Uint64(cutlass.Uint32(idv))
                        i = i + cutlass.Int32(BLK)
                else:
                    # collect overflow: scalar re-sweep, exact tail remap —
                    # zero extra live registers by design
                    lo2 = c0 << cutlass.Int32(2)
                    hi2 = c1 << cutlass.Int32(2)
                    i0_ = lo2 + tidx
                    while i0_ < hi2 + tailn:
                        i_ = i0_
                        if i0_ >= hi2:
                            i_ = tail0 + (i0_ - hi2)
                        x = C.ldg_f32(x_addr, i_)
                        if x >= TF:
                            bq = C.f2s_rz((x - TF) * SC)
                            if bq > cutlass.Int32(NBS - 1):
                                bq = cutlass.Int32(NBS - 1)
                            if bq >= B:
                                p = C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                                if p < lim1:
                                    out_row[p] = i_
                                else:
                                    if whole == cutlass.Int32(0):
                                        q2 = p - above
                                        if q2 < cutlass.Int32(CMPB):
                                            s_ck64[q2] = (
                                                cutlass.Uint64(C.fkey(x)) << cutlass.Uint64(32)
                                            ) | cutlass.Uint64(cutlass.Uint32(i_))
                        i0_ = i0_ + cutlass.Int32(BLK)

                # ---- P6 refine ----
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()  # ---- barrier (emit done) ----
                    if mc <= cutlass.Int32(QUADC_CLUS__main):  # O(mc^2) rank
                        mc2 = mc & cutlass.Int32(~1)
                        i = tidx
                        while i < mc:
                            # NOTE: values crossing a dynamic-while region are
                            # re-wrapped SIGNED by the DSL — every u64 compare
                            # must re-assert Uint64 at the USE site.
                            u64v = s_ck64[i]
                            r_ = cutlass.Int32(0)
                            jq = cutlass.Int32(0)
                            while jq < mc2:  # ulonglong2 16B reads
                                vlo, vhi = C._lds_v2_u64(ck_addr + jq * cutlass.Int32(8))
                                r_ = (
                                    r_
                                    + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                    + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                )
                                jq = jq + cutlass.Int32(2)
                            if mc2 < mc:  # odd tail
                                r_ = r_ + cutlass.Int32(
                                    cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                )
                            if r_ < need:
                                out_row[above + r_] = cutlass.Int32(
                                    cutlass.Uint32(
                                        cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                    )
                                )
                            i = i + cutlass.Int32(BLK)
                    else:
                        # key-space narrowing over ck64
                        if tidx == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        if tidx < cutlass.Int32(NBS):  # cleared ONCE
                            s_hist[tidx] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (narrowing init) ----
                        i = tidx
                        while i < mc:
                            kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                            C.atomic_min_cta(s_kmm.iterator + 0, kk)
                            C.atomic_max_cta(s_kmm.iterator + 1, kk)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # ---- barrier (key range) ----
                        rlo = s_kmm[0]
                        rhi = s_kmm[1]
                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        brk = cutlass.Int32(0)
                        lev = cutlass.Int32(0)
                        while brk == cutlass.Int32(0):  # <=6 levels
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                brk = cutlass.Int32(1)
                            elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            elif lev >= cutlass.Int32(6):
                                ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                brk = cutlass.Int32(1)
                            else:
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2_ = cutlass.Int32(32) - C.clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = b2_ - cutlass.Int32(self.lb)
                                if sh2 < cutlass.Int32(0):
                                    sh2 = cutlass.Int32(0)
                                sh2u = cutlass.Uint32(sh2)
                                i = tidx
                                while i < mc:  # re-bin
                                    uq = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                    if uq >= cutlass.Uint32(rlo):
                                        if uq <= cutlass.Uint32(rhi):
                                            du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                            if du > cutlass.Uint32(NBS - 1):
                                                du = cutlass.Uint32(NBS - 1)
                                            C.atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(du),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # ---- barrier (level hist) ----
                                C.scan_cross0(
                                    s_hist,
                                    needC,
                                    tidx,
                                    s_res,
                                    cutlass.Int32(0),
                                    cutlass.Int32(0),
                                    s_hist,
                                    nb=NBS,
                                    zero=True,
                                )
                                cute.arch.barrier()  # ---- barrier (level scan) ----
                                aboveC = aboveC + s_res[C.RES_ABOVE]
                                needC = needC - s_res[C.RES_ABOVE]
                                mm = s_res[C.RES_M]
                                sB = s_res[C.RES_B]
                                nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                if sB != cutlass.Int32(NBS - 1):
                                    rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        if tidx == cutlass.Int32(0):
                            s_scal[1] = cutlass.Int32(0)
                            s_scal[2] = cutlass.Int32(0)
                        cute.arch.barrier()  # ---- barrier (emit counters) ----
                        it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                        it = cutlass.Int32(0)
                        while it < it2:  # ballot emit
                            i = it * cutlass.Int32(BLK) + tidx
                            p1 = cutlass.Int32(0)
                            p2 = cutlass.Int32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                w64 = s_ck64[i]
                                iu = cutlass.Int64(cutlass.Uint32(w64 >> cutlass.Uint64(32)))
                                idv = cutlass.Int32(
                                    cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                )
                                if iu > ethr:
                                    p1 = cutlass.Int32(1)
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                            self._ballot_pair_emit(
                                p1,
                                p2,
                                idv,
                                above,
                                aboveC,
                                above + aboveC,
                                needC,
                                out_row,
                                s_scal,
                                lane,
                            )
                            it = it + cutlass.Int32(1)
            else:
                dga = cutlass.Int32(0)  # gate: valid && complete
                if valid != cutlass.Int32(0):
                    if complete != cutlass.Int32(0):
                        dga = cutlass.Int32(1)
                if dga != cutlass.Int32(0):
                    # ---- degen A: narrowing over STAGED candidates ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = listN
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (degen A init) ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx
                            while i < listN:
                                uq = cutlass.Uint32(0)
                                if cutlass.const_expr(self.vstg):
                                    vx = cutlass.Int32(0)
                                    vy = cutlass.Int32(0)
                                    if cutlass.const_expr(self.split):
                                        if fromg != cutlass.Int32(0):
                                            vx, vy = C._ldcg_v2_i32(
                                                gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                            )
                                        else:
                                            pk64 = s_cbuf2[i]
                                            vx = cutlass.Int32(
                                                cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                            )
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                    uq = C.fkey_bits(cutlass.Uint32(vx))
                                else:
                                    id0 = cutlass.Int32(
                                        cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                    )
                                    uq = C.fkey(C.ldg_f32(x_addr, id0))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            C.scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (listN + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        idv = cutlass.Int32(0)
                        if i < listN:
                            uq = cutlass.Uint32(0)
                            if cutlass.const_expr(self.vstg):
                                vx = cutlass.Int32(0)
                                vy = cutlass.Int32(0)
                                if cutlass.const_expr(self.split):
                                    if fromg != cutlass.Int32(0):
                                        vx, vy = C._ldcg_v2_i32(
                                            gbuf_row + cutlass.Int64(i) * cutlass.Int64(8)
                                        )
                                    else:
                                        pk64 = s_cbuf2[i]
                                        vx = cutlass.Int32(
                                            cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                        )
                                        vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                else:
                                    pk64 = s_cbuf2[i]
                                    vx = cutlass.Int32(
                                        cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    vy = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                                uq = C.fkey_bits(cutlass.Uint32(vx))
                                idv = vy
                            else:
                                idv = cutlass.Int32(
                                    cutlass.Uint32(s_cbuf[i]) & cutlass.Uint32(IDXM__main)
                                )
                                uq = C.fkey(C.ldg_f32(x_addr, idv))
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(
                            p1, p2, idv, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)
                else:
                    # ---- degen B: whole-row narrowing ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (degen B init) ----
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            i = tidx
                            while i < n:  # whole row
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            C.scan_cross0(
                                s_hist,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                s_hist,
                                nb=NBS,
                                zero=True,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        if i < n:
                            uq = C.fkey(C.ldg_f32(x_addr, i))
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(
                            p1, p2, i, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)

        # ---- varlen short-row epilogue (production heuristicTopKDecode
        # convention): every valid position is in the top-K — emit identity
        # indices and pad the tail with -1.  The body above ran as a
        # zero-work pass for these rows (n = 0, TGT = INT_MAX) so nothing
        # was written; only part 0 of a SPLIT row emits.
        if cutlass.const_expr(self.varlen):
            if short != cutlass.Int32(0):
                if part == cutlass.Int32(0):
                    i = tidx
                    while i < n_row:
                        out_row[i] = i
                        i = i + cutlass.Int32(BLK)
                    j = n_row + tidx
                    while j < k:
                        out_row[j] = cutlass.Int32(-1)
                        j = j + cutlass.Int32(BLK)

    # ------------------------------------------------------------------
    # host launcher (grid dim3(R, b); MINB wall via min_blocks_per_mp)
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        scap_dead: cutlass.Int32,
        cmp_dead: cutlass.Int32,
        R: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        kv_lens: cute.Tensor,
        aim_base: cutlass.Int32,
        sfac: cutlass.Int32,
        amin: cutlass.Int32,
        sd_en: cutlass.Int32,
        tsh_en: cutlass.Int32,
        stream,
    ):
        b = logits.shape[0]
        self.kern(
            logits,
            pre_idx,
            out,
            ws,
            n,
            npad,
            k,
            scap_dead,
            cmp_dead,
            R,
            SMP,
            TGT,
            Q,
            SS2,
            TGT2,
            kv_lens,
            aim_base,
            sfac,
            amin,
            sd_en,
            tsh_en,
        ).launch(grid=(R, b, 1), block=(self.blk, 1, 1), stream=stream, min_blocks_per_mp=self.minb)


# ---------------------------------------------------------------------------
# compile cache + torch-facing entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE = {}


def get_compiled(tpl, options_extra: str = ""):
    """Compile (or fetch) the gvr_main variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, KPT, SPLIT, TSHG)                — legacy, or
    tpl = (BLK, U, MINB, NBS, KPT, SPLIT, TSHG, NEXT_N, CR_SHIFT, R_CONST)
    — per-row varlen mode (TSHG slot is ignored: varlen compiles the TSH
    machinery in whenever SPLIT and gates it per row at runtime)."""
    key = (tuple(tpl), options_extra, C._compile_arch_token())
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    if len(tpl) == 7:
        blk, u, minb, nbs, kpt, split, tshg = tpl
        kern = GvrMainKernel(blk, u, minb, nbs, kpt, bool(split), bool(tshg))
    else:
        blk, u, minb, nbs, kpt, split, tshg, next_n, cr_shift, r_const = tpl
        kern = GvrMainKernel(
            blk,
            u,
            minb,
            nbs,
            kpt,
            bool(split),
            bool(tshg),
            varlen=True,
            next_n=next_n,
            cr_shift=cr_shift,
            r_const=r_const,
        )
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    w0 = cute.sym_int()
    v0 = cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16
    )
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16
    )
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16
    )
    ws_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (w0,), stride_order=(0,), assumed_align=16
    )
    kv_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (v0,), stride_order=(0,), assumed_align=4
    )
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)

    def _compile_fn():
        return cute.compile(
            kern,
            logits_fake,
            pre_fake,
            out_fake,
            ws_fake,
            *([cutlass.Int32(0)] * 11),
            kv_fake,
            *([cutlass.Int32(0)] * 5),
            stream=fake_stream,
            options=(_base_compile_opts() + " " + options_extra).strip(),
        )

    name = "main_" + "_".join(str(x) for x in tuple(tpl))
    if options_extra:
        name += "_opt_" + options_extra
    compiled = C._persist(name, _compile_fn)
    _COMPILE_CACHE[key] = compiled
    return compiled


def workspace_bytes() -> int:
    return WS_BYTES


def run(logits, pre_idx, n: int, out, ws):
    """torch-facing single-call entry: routes (b, n, k) through ct_dispatch,
    asserts the shape lands on gvr_main, launches the matching variant.
    ws: zero-initialised >=20,973,568-B CUDA buffer (reused across launches;
    the kernel restores the zeros it consumes)."""
    try:
        from . import gvr2_topk_host as ct_dispatch
    except ImportError:
        import gvr2_topk_host as ct_dispatch
    b, npad = logits.shape
    k = pre_idx.shape[1]
    r = ct_dispatch.route(b, int(n), npad, k)
    assert r["kernel"] == "main", f"shape routes to {r['kernel']}, not gvr_main"
    assert ws.numel() * ws.element_size() >= WS_BYTES
    rt = r["rt"]
    fn = get_compiled(tuple(r["tpl"]))
    fn(
        logits,
        pre_idx,
        out,
        ws,
        rt["n"],
        rt["npad"],
        rt["k"],
        rt["SCAP_"],
        rt["CMP_"],
        rt["R"],
        rt["SMP"],
        rt["TGT"],
        rt["Q"],
        rt["SS2"],
        rt["TGT2"],
        _legacy_dummy_kv(pre_idx),  # dummy kv_lens (dead in legacy mode)
        0,
        0,
        0,
        0,
        0,
    )
    return r


_LEGACY_DUMMY_KV = {}


def _legacy_dummy_kv(ref):
    """Cached 1-element int32 tensor per device for the dead kv_lens ABI slot
    (avoids a per-call allocator round-trip on the legacy hot path)."""
    d = ref.get_device()
    t = _LEGACY_DUMMY_KV.get(d)
    if t is None:
        t = ref.new_zeros(1)
        _LEGACY_DUMMY_KV[d] = t
    return t


# ===========================================================================
# ==== family: reg =============================================
# ===========================================================================
"""gvr_topk_reg — register-resident exact top-K, one CTA per row, histogram
bins in FLOAT space.

Template knobs (CUDA `gvr_topk_reg<BLK,VPT,MINB,KPT,CUR,DEG,IMGF,NBH>`):
ctor args of :class:`GvrTopkRegKernel`, read as `cutlass.const_expr(self.x)`
inside the kernel. Runtime args mirror the CUDA `(n, npad, k, CMP, IMGOFF,
QC)` — npad/k come from tensor shapes, IMGOFF is dropped (dispatch pins
IMGOFF == NBSEL == NBH at every site, asserted in the host wrapper).

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block is folded into the first 512 B so occupancy accounting
matches nvcc's static+dynamic sum):

    [0..5]        s_res    (shared slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt    (s_o1, s_oc)
    [8..9]        s_kmm    (s_kmin, s_kmax — Uint32)
    [10..11]      s_e12    (s_e1, s_e2)
    [16..16+NW)   ws       (scan_cross_w workspace)
    [48..48+NW)   wmn      (Uint32 warp min partials)
    [80..80+NW)   wmx      (Uint32 warp max partials)
    [128..128+NBH)          hist
    [128+NBH..128+NBH+CMP)  ck     (Uint32 crossing keys; CMP dynamic)
    [128+NBH+CMP..+2CMP)    ci     (Int32 crossing indices)
    img/bm alias ck at word 128+NBH (IMGOFF==NBH)

Launch smem = 512 + dispatch_smem_bytes (dynamic Int32).

TOOLCHAIN GOTCHA: dynamic launch smem with min_blocks_per_mp>1 crashes
cutlass_dsl._build_kernel_attrs (host ceil() on a dynamic value while
computing the PREFERRED_SHARED_MEMORY_CARVEOUT hint). `_no_carveout()`
scopes a monkeypatch around cute.compile dropping ONLY that hint (CUDA
__launch_bounds__ sets no carveout either); `.reqntid`/`.minnctapersm`
(the register wall) are unaffected.
"""


NB__reg = 1024  # NBH default
STATIC_WORDS = 128  # DSL smem prelude (static-__shared__ mirror)
STATIC_BYTES = STATIC_WORDS * 4
_NEG_INF__reg = float("-inf")
_POS_INF = float("inf")


# ---------------------------------------------------------------------------
# module-local FP spellings
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__reg(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32."""
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


@dsl_user_op
def _submul_asm(v, t, sc, *, loc=None, ip=None):
    """(v - t) * sc with two roundings, opaque to CSE/contraction.

    Used at the !BRL classify site so no sub-expression is shared with the
    emit's `fmaf(v - T, SC, OFF)` — the CUDA deliberately spells the two
    sites differently to stop the compiler holding all S q's live across
    the barrier.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip), t.ir_value(loc=loc, ip=ip), sc.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 rtmp;\n\tsub.rn.f32 rtmp, $1, $2;\n\tmul.rn.f32 $0, rtmp, $3;\n\t}",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _smem_addr_reg__reg(addr, *, loc=None, ip=None):
    """Pin a CTA-shared 32-bit byte address in ONE register.

    Identity `mov` behind an asm boundary: without it LLVM re-folds the
    `mov.b32 %r, __dynamic_shmem__0` symbol materialisation into EVERY use
    site, and ptxas then re-derives the CGA shared window (S2UR SR_CgaCtaId
    + UMOV + ULEA, 3 instructions) inside each divergent classify block.
    The asm result is not duplicable, so the window is materialised exactly
    once. Value-identical: a plain register copy.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "mov.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_add1__reg(addr, *, loc=None, ip=None):
    """CUDA classify `atomicAdd(&hist[bin], 1u)` with the result unused.

    `red` (not `atom`) is the result-less spelling — ptxas lowers it to the
    same ATOMS.POPC.INC.32 RZ the CUDA arm emits. Same ordering contract as
    atomic_add_cta: .relaxed scope .cta. Takes the final shared byte address
    as a plain Int32 so the address datapath stays ordinary IR (ptxas fuses
    the shl+add into one LEA against the pinned `_smem_addr_reg__reg` base).
    """
    llvm.inline_asm(
        res=None,
        operands_=[addr.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cta.shared.add.u32 [$0], 1;",
        constraints="r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _umin_u32(a, b):
    """unsigned min(a, b) — CUDA min() on the bin clamp (IMNMX)."""
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _fabsf(x):
    """|x| via sign-bit clear (exact, matches fabsf)."""
    return f32_of_u32(u32_of_f32(x) & cutlass.Uint32(0x7FFFFFFF))


def _f32_smem_atom():
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)


def _sts128_f32(atom, frag, base_addr, byte_off):
    p = cute.make_ptr(
        cutlass.Float32, base_addr + byte_off, cute.AddressSpace.smem, assumed_align=16
    )
    cute.copy(atom, frag, cute.make_tensor(p, cute.make_layout((4,))))


def _smem_view(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(
        dtype, sbase + cutlass.Int32(word_off * 4), cute.AddressSpace.smem, assumed_align=align
    )
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


@contextlib.contextmanager
def _no_carveout():
    """Scoped: drop the DSL's carveout hint (see module docstring)."""
    import cutlass.cutlass_dsl.cutlass as _cdsl

    orig = _cdsl._build_kernel_attrs
    _cdsl._build_kernel_attrs = lambda config: {}
    # DSL >= 4.6.1 also derives a carveout inside the compiled artifact from
    # the smem.max_smem_per_mp MLIR attribute (emitted when
    # min_blocks_per_mp > 1).  That calculation only sees the static smem, so
    # with dynamic launch smem it selects a 16 KiB shared-memory config and
    # pins the SM to one resident CTA.  Drop the attribute as well so the
    # driver default carveout stays in effect.
    base = getattr(_cdsl, "CutlassBaseDSL", None)
    orig_gen = getattr(base, "_generate_kernel_attrs", None)

    if orig_gen is not None:

        def _gen(self, config, _orig=orig_gen):
            ret = _orig(self, config)
            ret.pop("smem.max_smem_per_mp", None)
            return ret

        base._generate_kernel_attrs = _gen
    try:
        yield
    finally:
        _cdsl._build_kernel_attrs = orig
        if orig_gen is not None:
            base._generate_kernel_attrs = orig_gen


class GvrTopkRegKernel:
    """gvr_topk_reg<BLK, VPT, MINB, KPT, CUR, DEG, IMGF, NBH>."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        minb: int,
        kpt: int,
        cur: bool,
        deg: bool,
        img: bool,
        nbh: int = NB__reg,
        pdl: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
    ):
        assert blk in (256, 512, 1024) and vpt in (1, 2, 4)
        assert nbh in (256, 512, 1024, 2048)
        assert nbh % blk == 0 or blk % nbh == 0
        self.blk = blk
        self.vpt = vpt
        self.minb = minb
        self.kpt = kpt
        self.cur = bool(cur)
        self.deg = bool(deg)
        self.img = bool(img)
        self.nbh = nbh
        self.pdl = bool(pdl)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel/GvrRegClusKernel): n is re-derived PER
        # ROW in-kernel from a device kv_lens tensor; the scalar n launch arg
        # becomes the envelope clamp bound. next_n / cr_shift compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        # derived compile-time constants
        self.S = vpt * 4
        self.lnbh = {256: 8, 512: 9, 2048: 11}.get(nbh, 10)
        self.use_bm = (not deg) and (not img) and kpt >= 2 and vpt == 1
        self.use_img = img and vpt == 1
        self.brl = (minb * blk <= 1024) or (vpt == 1)

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
    ):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        KPT = cutlass.const_expr(self.kpt)
        NBH = cutlass.const_expr(self.nbh)  # noqa: F841
        S = cutlass.const_expr(self.S)
        LNBH = cutlass.const_expr(self.lnbh)
        NW = cutlass.const_expr(self.blk // 32)

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # knob default off

        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane = tid & cutlass.Int32(31)

        # ================= per-row varlen prologue (varlen mode only) =========
        # Production heuristicTopKDecode contract (GvrMainKernel /
        # GvrRegClusKernel discipline): row r serves request r // next_n with
        # n = (kv_lens[req] - next_n + r % next_n + 1) >> cr_shift, clamped to
        # the envelope launch arg n (the launcher admits this family only when
        # the envelope fits its capacity window, so per-row n never exceeds
        # capacity). One CTA per row, so the whole-body guard below is
        # trivially block-uniform. Short rows (n <= k) emit identity + (-1)
        # tail here (k can exceed BLK on this family -> strided loop, unlike
        # the reg_clus k <= BLK single predicate) and SKIP the body entirely:
        # a zero-work pass would reach the degenerate emitter and poison out.
        short = cutlass.Int32(0)
        prow = row
        if cutlass.const_expr(self.varlen):
            kq = cutlass.Int32(pre_idx.shape[1])
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            prow = req
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            if nv > n:
                nv = n
            if nv <= kq:
                short = cutlass.Int32(1)
            if short == cutlass.Int32(0):
                n = nv
            if short != cutlass.Int32(0):
                i = tid
                while i < kq:
                    ov = cutlass.Int32(-1)
                    if i < nv:
                        ov = i
                    out[row, i] = ov
                    i = i + cutlass.Int32(BLK)

        # ------------------------------------------------------------------
        # Predeclarations: the DSL AST transformer requires every scalar that
        # is (re)assigned under a dynamic if/while region to pre-exist with a
        # stable type at every enclosing region level. Constant inits are
        # sunk/dead-coded by LLVM, so this costs no registers.
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        r = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1e = cutlass.Int32(0)
        q2e = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        quad = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        m1 = cutlass.Int32(0)
        m2 = cutlass.Int32(0)
        t1 = cutlass.Int32(0)
        t2 = cutlass.Int32(0)
        c1 = cutlass.Int32(0)
        c2 = cutlass.Int32(0)
        s1 = cutlass.Int32(0)
        s2 = cutlass.Int32(0)
        p1 = cutlass.Int32(0)
        p2 = cutlass.Int32(0)
        wm = cutlass.Int32(0)
        sdyn = cutlass.Int32(0)
        nbw = cutlass.Int32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        uk = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        w = cutlass.Uint32(0)
        wt = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        u64 = cutlass.Int64(0)
        LOQ = cutlass.Float32(0.0)
        HIf = cutlass.Float32(0.0)
        LOf = cutlass.Float32(0.0)
        qt2 = cutlass.Float32(0.0)
        qt3 = cutlass.Float32(0.0)

        # wrap-scoping additions (varlen whole-body guard): names whose
        # first assignment moves under the dynamic `if short == 0` region
        # below and are reassigned deeper — same rule as the block above.
        a = cutlass.Uint32(0)
        c = cutlass.Uint32(0)
        lmin = cutlass.Uint32(0)
        lmax = cutlass.Uint32(0)
        esc = cutlass.Int32(0)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        tval = cutlass.Float32(0.0)
        wsel = cutlass.Float32(0.0)
        GMAX = cutlass.Float32(0.0)
        Tv = cutlass.Float32(0.0)
        lmn = cutlass.Float32(0.0)
        lmx = cutlass.Float32(0.0)

        if short == cutlass.Int32(0):
            npad = cutlass.Int32(logits.shape[1])  # noqa: F841
            k = cutlass.Int32(pre_idx.shape[1])
            out_row = out[row, None]
            x_addr = logits[row, None].iterator.toint()  # Int64 gmem byte base
            p_addr = pre_idx[prow, None].iterator.toint()  # request-level under varlen

            # ---- shared-memory window (map in module docstring) ----
            sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            sbase = sptr.toint()  # Int32 shared addr

            s_res = _smem_view(cutlass.Int32, sbase, 0, 6)
            s_cnt = _smem_view(cutlass.Int32, sbase, 6, 2)  # [0]=s_o1 [1]=s_oc
            s_kmm = _smem_view(cutlass.Uint32, sbase, 8, 2)  # [0]=s_kmin [1]=s_kmax
            s_e12 = _smem_view(cutlass.Int32, sbase, 10, 2)  # [0]=s_e1 [1]=s_e2
            s_ws = _smem_view(cutlass.Int32, sbase, 16, 32)
            s_wmn = _smem_view(cutlass.Uint32, sbase, 48, 32)
            s_wmx = _smem_view(cutlass.Uint32, sbase, 80, 32)
            s_hist = _smem_view(cutlass.Int32, sbase, STATIC_WORDS, self.nbh)
            ck_base = sbase + cutlass.Int32((STATIC_WORDS + self.nbh) * 4)
            ck = cute.make_tensor(
                cute.make_ptr(cutlass.Uint32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )  # typed view, no bound
            ci = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Int32,
                    ck_base + cmp_ * cutlass.Int32(4),
                    cute.AddressSpace.smem,
                    assumed_align=4,
                ),
                cute.make_layout((65536,)),
            )
            img_f = cute.make_tensor(  # aliases ck/ci
                cute.make_ptr(cutlass.Float32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )
            bm = cute.make_tensor(  # aliases ck
                cute.make_ptr(cutlass.Int32, ck_base, cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout((65536,)),
            )

            n4 = n >> cutlass.Int32(2)
            ntail = n - (n4 << cutlass.Int32(2))
            tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`

            # ---- hint prefetch: KPT coalesced pre_idx words BEFORE any
            # dependent gather; compiled out under DEG.
            pvs = []
            if cutlass.const_expr(not self.deg):
                for t in cutlass.range_constexpr(KPT):
                    pv = cutlass.Int32(-1)
                    j = tid + cutlass.Int32(t * self.blk)
                    if j < k:
                        pv = ld_g_i32(p_addr, j)
                    pvs.append(pv)

            # ---- row load: exact-fit peel + float4[VPT] register batch
            atom128 = g2r_atom_f32(128, invariant=True)
            frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
            if n4 >= cutlass.Int32(self.blk * self.vpt):  # block-uniform peel
                for u in cutlass.range_constexpr(VPT):
                    ld_g_f32x4(atom128, x_addr, tid + cutlass.Int32(u * self.blk), frags[u])
            else:  # predicated flat batch
                for u in cutlass.range_constexpr(VPT):
                    i = tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        ld_g_f32x4(atom128, x_addr, i, frags[u])
                for u in cutlass.range_constexpr(VPT):
                    i = tid + cutlass.Int32(u * self.blk)
                    if i >= n4:  # -INFINITY fill
                        for q in cutlass.range_constexpr(4):
                            frags[u][q] = cutlass.Float32(_NEG_INF__reg)

            tval = cutlass.Float32(_NEG_INF__reg)
            if tid < ntail:
                tval = ldg_f32(x_addr, tix)

            # ---- init
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            for z in cutlass.range_constexpr(self.nbh // self.blk):
                s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- bracket: 4 mutually exclusive compile-time arms
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.const_expr(self.use_img):
                fatom = _f32_smem_atom()
                for u in cutlass.range_constexpr(VPT):  # VPT == 1 here
                    i = tid + cutlass.Int32(u * self.blk)
                    if i < n4:
                        _sts128_f32(fatom, frags[u], ck_base, i * cutlass.Int32(16))
                if tid < ntail:
                    img_f[tix] = tval
                cute.arch.barrier()  # image staged
                for t in cutlass.range_constexpr(KPT):
                    p = pvs[t]
                    if cutlass.Uint32(p) < cutlass.Uint32(n):
                        uk = fkey(img_f[p])
                        if uk < lmin:
                            lmin = uk
                        if uk > lmax:
                            lmax = uk
                cute.arch.barrier()  # img dies
            elif cutlass.const_expr(self.use_bm):
                nbw = (n + cutlass.Int32(31)) >> cutlass.Int32(5)
                i = tid
                while i < nbw:  # bitmap clear
                    bm[i] = cutlass.Int32(0)
                    i = i + cutlass.Int32(BLK)
                cute.arch.barrier()  # bitmap cleared
                for t in cutlass.range_constexpr(KPT):
                    p = pvs[t]
                    if cutlass.Uint32(p) < cutlass.Uint32(n):
                        atomic_or_cta(
                            bm.iterator + (p >> cutlass.Int32(5)),
                            cutlass.Int32(1) << (p & cutlass.Int32(31)),
                        )
                cute.arch.barrier()  # bitmap set
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for u in cutlass.range_constexpr(VPT):
                    base = (tid + cutlass.Int32(u * self.blk)) << cutlass.Int32(2)
                    w = cutlass.Uint32(0)
                    if cutlass.Uint32(base) < cutlass.Uint32(n):
                        w = cutlass.Uint32(bm[base >> cutlass.Int32(5)]) >> cutlass.Uint32(
                            base & cutlass.Int32(31)
                        )
                    for cbit in cutlass.range_constexpr(4):
                        if (w & cutlass.Uint32(1 << cbit)) != cutlass.Uint32(0):
                            lmn = fmin_f32(lmn, _val(frags, 4 * u + cbit))
                            lmx = fmax_f32(lmx, _val(frags, 4 * u + cbit))
                if tid < ntail:
                    wt = cutlass.Uint32(bm[tix >> cutlass.Int32(5)]) >> cutlass.Uint32(
                        tix & cutlass.Int32(31)
                    )
                    if (wt & cutlass.Uint32(1)) != cutlass.Uint32(0):
                        lmn = fmin_f32(lmn, tval)
                        lmx = fmax_f32(lmx, tval)
                lmin = fkey(lmn)
                lmax = fkey(lmx)  # monotone
                cute.arch.barrier()  # bm dies
            elif cutlass.const_expr(self.deg):
                lmn = cutlass.Float32(_POS_INF)
                lmx = cutlass.Float32(_NEG_INF__reg)
                for s in cutlass.range_constexpr(S):
                    v = _val(frags, s)
                    if v > cutlass.Float32(_NEG_INF__reg):
                        lmn = fmin_f32(lmn, v)
                        lmx = fmax_f32(lmx, v)
                if tid < ntail:
                    lmn = fmin_f32(lmn, tval)
                    lmx = fmax_f32(lmx, tval)
                lmin = fkey(lmn)
                lmax = fkey(lmx)
            else:
                # default: KPT scattered fkey ldg gathers, batch-then-fold
                xs = []
                for t in cutlass.range_constexpr(KPT):
                    xv = cutlass.Float32(0.0)
                    if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                        xv = ldg_f32(x_addr, pvs[t])
                    xs.append(xv)
                for t in cutlass.range_constexpr(KPT):
                    if cutlass.Uint32(pvs[t]) < cutlass.Uint32(n):
                        uk = fkey(xs[t])
                        if uk < lmin:
                            lmin = uk
                        if uk > lmax:
                            lmax = uk

            # ---- block min/max in ONE barrier; publishes hist clear
            lmin = warp_min_u32(lmin)
            lmax = warp_max_u32(lmax)
            if lane == cutlass.Int32(0):
                s_wmn[tid >> cutlass.Int32(5)] = lmin
                s_wmx[tid >> cutlass.Int32(5)] = lmax
            cute.arch.barrier()  # warp partials published
            a = cutlass.Uint32(0xFFFFFFFF)
            c = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                a = cutlass.Uint32(s_wmn[lane])
                c = cutlass.Uint32(s_wmx[lane])
            lmin = warp_min_u32(a)
            lmax = warp_max_u32(c)
            Tv = invkey(lmin)
            GMAX = invkey(lmax)

            # ---- collapse guard, NaN-safe. An infinite bracket width
            # (GMAX=+inf from an in-window +inf, or Tv=-inf) makes SC=rcp(w)=0
            # and folds every value into bin 0, dropping the +inf; reject it
            # here (okc=0) and force the key-space escape below.
            # (upstream TensorRT-LLM PR #18625, ported verbatim.)
            okc = cutlass.Int32(0)
            if Tv < GMAX:
                w_ = GMAX - Tv
                # w_ < 3.5e38 (> FLT_MAX) rejects an infinite bracket width.
                if w_ > cutlass.Float32(1e-30) and w_ < cutlass.Float32(3.5e38):
                    okc = cutlass.Int32(1)
            if okc == cutlass.Int32(0):
                Tv = cutlass.Float32(SENT_LO)
                GMAX = cutlass.Float32(SENT_HI)

            # ---- bin transform constants
            BRL = cutlass.const_expr(self.brl)  # noqa: F841
            OFFf = cutlass.Float32(1.0 if self.brl else 0.0)
            recip = 1.0 / float(self.nbh - (2 if self.brl else 0))
            WD = (GMAX - Tv) * cutlass.Float32(recip)
            wsel = cutlass.Float32(1e-30)
            if WD > cutlass.Float32(0.0):
                wsel = WD
            # rcp.approx (single MUFU.RCP) — the CUDA arm's exact lowering of
            # `1.0f / wsel`; a plain `1.0 / wsel` would emit the IEEE div.rn
            # Newton triple + slowpath CALL on the barrier-bounded chain
            # feeding all S classify FMULs. Output exactness is SC-invariant
            # (any SC > 0 preserves the sign/monotonicity invariants) and the
            # WD > 0 arm is bit-identical to CUDA's MUFU.RCP.
            SC = cute.arch.rcp_approx(wsel)
            QCAPf = cutlass.Float32(float(self.nbh - 1))
            CQ0 = OFFf - Tv * SC
            CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf(CQ0) + cutlass.Float32(1.0))

            # ---- histogram
            if cutlass.const_expr(self.brl):
                # BRL classify arm: hist base pinned ONCE via the same
                # _smem_addr_reg__reg identity-mov used in the !BRL arm below,
                # and the result-discarded classify atomics spelled as
                # resultless red.shared (_red_shared_add1__reg).
                # Value-identical: same +1 to the same byte address
                # (hb + 4*bn == &s_hist[bn]), same .relaxed.cta ordering; the
                # q/bn computations are untouched so classify/emit
                # bit-identity (BRL requirement) is preserved. Emit-path hist
                # atomics (results used) are NOT touched.
                hb = _smem_addr_reg__reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
                for s in cutlass.range_constexpr(S):
                    q = _fmaf__reg(_val(frags, s), SC, CQ)
                    bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                    _red_shared_add1__reg(hb + (cutlass.Int32(bn) << cutlass.Int32(2)))
                qt = _fmaf__reg(tval, SC, CQ)  # unconditional
                bnt = _umin_u32(f2u_rz(qt), cutlass.Uint32(self.nbh - 1))
                _red_shared_add1__reg(hb + (cutlass.Int32(bnt) << cutlass.Int32(2)))
            else:
                # hist base pinned ONCE (byte addr, +STATIC_BYTES = word 128 map);
                # each site below is then LEA + ATOMS exactly like the CUDA arm
                # instead of re-deriving the shared window per divergent block.
                hb = _smem_addr_reg__reg(sbase + cutlass.Int32(STATIC_WORDS * 4))
                for s in cutlass.range_constexpr(S):
                    q = _submul_asm(_val(frags, s), Tv, SC)  # anti-CSE classify
                    if q >= cutlass.Float32(0.0):
                        _red_shared_add1__reg(hb + (f2s_rz(fmin_f32(q, QCAPf)) << cutlass.Int32(2)))
                qt = _submul_asm(tval, Tv, SC)
                if qt >= cutlass.Float32(0.0):
                    _red_shared_add1__reg(hb + (f2s_rz(fmin_f32(qt, QCAPf)) << cutlass.Int32(2)))
            cute.arch.barrier()  # histogram done

            # ---- crossing-bin find
            if cutlass.const_expr(self.cur or self.nbh > 1024):
                scan_cross_w(s_hist, s_ws, k, tid, s_res, blk=self.blk, nb=self.nbh)
            else:
                find_cross(s_hist, k, tid, s_res, nb=self.nbh)
            cute.arch.barrier()  # crossing published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            tot = s_res[RES_TOT]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)

            # ---- ESCAPE: radix descent over the key space
            # (upstream TensorRT-LLM PR #18501, ported verbatim.)
            # Count-crossing enforcement: when the hint-derived bracket's low
            # edge sits above the true k-th value, the classify arms count
            # fewer than k entries (q < 0 is never histogrammed) and the
            # crossing scan pins its bin-0 fallback as a fake crossing — the
            # whole-bin emit would then stop at the histogram total and leave
            # out_row[tot:k) unwritten. Exactness must come from the
            # count-crossing invariant, never from the bracket estimate: when
            # the histogram never reaches k (tot < k), take the escape — it
            # ranks the FULL row in key space, independent of the bracket.
            esc = cutlass.Int32(0)
            if whole == cutlass.Int32(0):
                if m > cmp_:
                    esc = cutlass.Int32(1)
            if tot < k:
                esc = cutlass.Int32(1)
            # Degenerate bracket (okc==0: infinite/collapsed width, e.g. an
            # in-window +inf) makes SC=0 and folds every value into bin 0, so
            # the histogram cannot rank the row; take the bracket-independent
            # key-space escape, where fkey(+inf) is the maximum key.
            # (upstream TensorRT-LLM PR #18625, ported verbatim.)
            if okc == cutlass.Int32(0):
                esc = cutlass.Int32(1)
            if esc == cutlass.Int32(1):
                if tid == cutlass.Int32(0):
                    s_cnt[0] = cutlass.Int32(0)
                    s_cnt[1] = cutlass.Int32(0)
                    # DEVIATION (race fix): the CUDA zeroes s_o1/s_oc again
                    # between the nA read and the emit with only ONE barrier
                    # pair around both — a read/write race. Emit instead
                    # through the path-exclusive s_e1/s_e2 slots, zeroed HERE
                    # under the existing barrier; the racy mid-emit rezero is
                    # dropped. Barrier count unchanged.
                    s_e12[0] = cutlass.Int32(0)
                    s_e12[1] = cutlass.Int32(0)
                cute.arch.barrier()  # escape init
                # Vector-lane bound: the register batch holds real data only
                # below 4*n4 — lanes in [4*n4, n) are the -inf FILL of the
                # last partial float4 (the real values there live in tval).
                # Bounding by n would count/emit each tail element twice
                # (once as a fill key, once via tval): benign while the tie
                # threshold is finite (a fill key loses every compare), but
                # it emits duplicate indices when the tie class is the -inf
                # key itself (in-window -inf entries are admissible).
                nvec = n4 << cutlass.Int32(2)
                # Narrow LNBH bits per level over the register batch — the
                # same descent the Phase-3 ck fallback below runs — instead of
                # one rescan per key bit: ceil(32/LNBH) levels at worst and
                # two in practice, since the first level's LNBH bits already
                # cut inside the exponent field, where the bisection took 32.
                # The exit arms carry the count crossing, so the old separate
                # above-count pass folds in.
                # Neither ethr nor aboveC is carried across the descent: ethr
                # is an Int64 derivable from rlo and the exit arm, and
                # aboveC + needC == k holds at every exit.
                rlo = cutlass.Uint32(0)
                rhi = cutlass.Uint32(0xFFFFFFFF)
                needC = k
                mm = n  # in-range count, carried from the previous level
                lev = cutlass.Int32(0)
                state = cutlass.Int32(0)  # 1 -> ethr = rlo, 2 -> ethr = rlo - 1
                while state == cutlass.Int32(0):
                    if needC == mm:
                        # whole in-range block is needed: threshold below it
                        needC = cutlass.Int32(0)
                        state = cutlass.Int32(2)
                    if state == cutlass.Int32(0):
                        if rlo >= rhi:  # single key left == the k-th key
                            state = cutlass.Int32(1)
                        if lev >= cutlass.Int32(34):  # non-binding: sh2 strictly shrinks
                            state = cutlass.Int32(1)
                    if state == cutlass.Int32(0):
                        d2 = rhi - rlo
                        b2w = cutlass.Int32(32) - clz_i32(cutlass.Int32(d2 | cutlass.Uint32(1)))
                        sh2 = cutlass.Int32(0)
                        if b2w > cutlass.Int32(LNBH):
                            sh2 = b2w - cutlass.Int32(LNBH)
                        for z in cutlass.range_constexpr(self.nbh // self.blk):
                            s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                        cute.arch.barrier()  # esc level clear
                        for s in cutlass.range_constexpr(S):
                            ix = (
                                (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                            ) + cutlass.Int32(s % 4)
                            if ix < nvec:
                                uev = fkey(_val(frags, s))
                                if uev >= rlo:
                                    if uev <= rhi:
                                        bne = _umin_u32(
                                            (uev - rlo) >> cutlass.Uint32(sh2),
                                            cutlass.Uint32(self.nbh - 1),
                                        )
                                        atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(bne),
                                            cutlass.Int32(1),
                                        )
                        if tid < ntail:
                            uev = fkey(tval)
                            if uev >= rlo:
                                if uev <= rhi:
                                    bne = _umin_u32(
                                        (uev - rlo) >> cutlass.Uint32(sh2),
                                        cutlass.Uint32(self.nbh - 1),
                                    )
                                    atomic_add_cta(
                                        s_hist.iterator + cutlass.Int32(bne), cutlass.Int32(1)
                                    )
                        cute.arch.barrier()  # esc level hist
                        if cutlass.const_expr(self.nbh > 1024):
                            scan_cross_w(s_hist, s_ws, needC, tid, s_res, blk=self.blk, nb=self.nbh)
                        else:
                            find_cross(s_hist, needC, tid, s_res, nb=self.nbh)
                        cute.arch.barrier()  # esc level scan
                        needC = needC - s_res[RES_ABOVE]
                        mm = s_res[RES_M]
                        b_lv = s_res[RES_B]
                        nlo = rlo + (cutlass.Uint32(b_lv) << cutlass.Uint32(sh2))
                        if b_lv != cutlass.Int32(self.nbh - 1):
                            rhi = nlo + (
                                (cutlass.Uint32(1) << cutlass.Uint32(sh2)) - cutlass.Uint32(1)
                            )
                        rlo = nlo
                        lev = lev + cutlass.Int32(1)
                nA = k - needC
                nT = needC
                ethr = cutlass.Int64(rlo)
                if state == cutlass.Int32(2):
                    ethr = ethr - cutlass.Int64(1)
                cute.arch.barrier()  # descent done
                lml = cutlass.Int32(cute.arch.lanemask_lt())
                for s in cutlass.range_constexpr(S):
                    ixv = (
                        (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                    ) + cutlass.Int32(s % 4)
                    u64 = cutlass.Int64(-1)
                    if ixv < nvec:
                        u64 = cutlass.Int64(fkey(_val(frags, s)))
                    q1e = cutlass.Int32(0)
                    q2e = cutlass.Int32(0)
                    if u64 > ethr:
                        q1e = cutlass.Int32(1)
                    if u64 == ethr:
                        q2e = cutlass.Int32(1)
                    n1 = ballot(q1e == cutlass.Int32(1))
                    n2 = ballot(q2e == cutlass.Int32(1))
                    b1 = cutlass.Int32(0)
                    b2 = cutlass.Int32(0)
                    if lane == cutlass.Int32(0):
                        if n1 != cutlass.Int32(0):
                            b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                        if n2 != cutlass.Int32(0):
                            b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                    b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                    b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                    p1e = b1 + popc(n1 & lml)
                    p2e = b2 + popc(n2 & lml)
                    if q1e == cutlass.Int32(1):
                        if p1e < nA:
                            out_row[p1e] = ixv
                    if q2e == cutlass.Int32(1):
                        if p2e < nT:
                            out_row[nA + p2e] = ixv
                # tail element
                u64 = cutlass.Int64(-1)
                if tid < ntail:
                    u64 = cutlass.Int64(fkey(tval))
                q1e = cutlass.Int32(0)
                q2e = cutlass.Int32(0)
                if u64 > ethr:
                    q1e = cutlass.Int32(1)
                if u64 == ethr:
                    q2e = cutlass.Int32(1)
                n1 = ballot(q1e == cutlass.Int32(1))
                n2 = ballot(q2e == cutlass.Int32(1))
                b1 = cutlass.Int32(0)
                b2 = cutlass.Int32(0)
                if lane == cutlass.Int32(0):
                    if n1 != cutlass.Int32(0):
                        b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                    if n2 != cutlass.Int32(0):
                        b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                p1e = b1 + popc(n1 & lml)
                p2e = b2 + popc(n2 & lml)
                if q1e == cutlass.Int32(1):
                    if p1e < nA:
                        out_row[p1e] = tix
                if q2e == cutlass.Int32(1):
                    if p2e < nT:
                        out_row[nA + p2e] = tix
                # (CUDA returns here — everything below is the else-arm)
            else:
                # ---- emit
                if cutlass.const_expr(self.cur):
                    LOQ = cutlass.Float32(Bv)  # int->float cvt
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = above + m
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(self.brl):
                            q = _fmaf__reg(_val(frags, s), SC, CQ)  # bit-identical to classify
                        else:
                            q = _fmaf__reg(_val(frags, s) - Tv, SC, OFFf)  # emit spelling
                        idx = (
                            (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        p = cutlass.Int32(0)
                        if q >= LOQ:
                            bn = _umin_u32(f2u_rz(q), cutlass.Uint32(self.nbh - 1))
                            p = atomic_add_cta(
                                s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1)
                            )
                            if p < lim1:
                                out_row[p] = idx
                            else:
                                if whole == cutlass.Int32(0):
                                    q2i = p - above
                                    if q2i < cmp_:  # escape-made-safe guard
                                        ck[q2i] = fkey(_val(frags, s))
                                        ci[q2i] = idx
                    # tail
                    if cutlass.const_expr(self.brl):
                        qt2 = _fmaf__reg(tval, SC, CQ)
                    else:
                        qt2 = _fmaf__reg(tval - Tv, SC, OFFf)
                    p = cutlass.Int32(0)
                    if qt2 >= LOQ:
                        bn = _umin_u32(f2u_rz(qt2), cutlass.Uint32(self.nbh - 1))
                        p = atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                        if p < lim1:
                            out_row[p] = tix
                        else:
                            if whole == cutlass.Int32(0):
                                q2i = p - above
                                if q2i < cmp_:
                                    ck[q2i] = fkey(tval)
                                    ci[q2i] = tix
                else:
                    # two-mask ballot emit
                    HIf = cutlass.Float32(_POS_INF)
                    LOf = cutlass.Float32(_POS_INF)
                    if whole == cutlass.Int32(1):
                        HIf = cutlass.Float32(Bv)
                    else:
                        if Bv < cutlass.Int32(self.nbh - 1):
                            HIf = cutlass.Float32(Bv + cutlass.Int32(1))
                        LOf = cutlass.Float32(Bv)
                    m1 = cutlass.Int32(0)
                    m2 = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(S):
                        if cutlass.const_expr(self.brl):
                            q = _fmaf__reg(_val(frags, s), SC, CQ)
                        else:
                            q = _fmaf__reg(_val(frags, s) - Tv, SC, OFFf)
                        if q >= HIf:
                            m1 = m1 | cutlass.Int32(1 << s)
                        else:
                            if q >= LOf:
                                m2 = m2 | cutlass.Int32(1 << s)
                    if cutlass.const_expr(self.brl):
                        qt3 = _fmaf__reg(tval, SC, CQ)
                    else:
                        qt3 = _fmaf__reg(tval - Tv, SC, OFFf)
                    t1 = cutlass.Int32(0)
                    t2 = cutlass.Int32(0)
                    if qt3 >= HIf:
                        t1 = cutlass.Int32(1)
                    else:
                        if qt3 >= LOf:
                            t2 = cutlass.Int32(1)
                    c1 = popc(m1) + t1
                    c2 = popc(m2) + t2
                    s1, s2 = warp_incl_scan_add2(c1, c2, lane)
                    b1 = cutlass.Int32(0)
                    b2 = cutlass.Int32(0)
                    if lane == cutlass.Int32(31):
                        b1 = atomic_add_cta(s_cnt.iterator, s1)
                        b2 = atomic_add_cta(s_cnt.iterator + 1, s2)
                    b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(31))
                    b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(31))
                    p1 = b1 + (s1 - c1)
                    p2 = b2 + (s2 - c2)
                    lim1 = above
                    if whole == cutlass.Int32(1):
                        lim1 = k
                    wm = m1  # sparse set-bit walk
                    while wm != cutlass.Int32(0):
                        sdyn = ffs_m1(wm)
                        idx = (
                            (tid + (sdyn >> cutlass.Int32(2)) * cutlass.Int32(self.blk))
                            << cutlass.Int32(2)
                        ) + (sdyn & cutlass.Int32(3))
                        if p1 < lim1:
                            out_row[p1] = idx
                        p1 = p1 + cutlass.Int32(1)
                        wm = wm & (wm - cutlass.Int32(1))
                    if t1 == cutlass.Int32(1):
                        if p1 < lim1:
                            out_row[p1] = tix
                        p1 = p1 + cutlass.Int32(1)
                    if m2 != cutlass.Int32(0):  # static-unrolled
                        for s in cutlass.range_constexpr(S):
                            if (m2 & cutlass.Int32(1 << s)) != cutlass.Int32(0):
                                idx = (
                                    (tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                                ) + cutlass.Int32(s % 4)
                                if p2 < cmp_:
                                    ck[p2] = fkey(_val(frags, s))
                                    ci[p2] = idx
                                p2 = p2 + cutlass.Int32(1)
                    if t2 == cutlass.Int32(1):
                        if p2 < cmp_:
                            ck[p2] = fkey(tval)
                            ci[p2] = tix
                        p2 = p2 + cutlass.Int32(1)

                # ---- refine (skipped when whole — CUDA returned inside emit)
                if whole == cutlass.Int32(0):
                    cute.arch.barrier()  # emit done
                    if cutlass.const_expr(self.cur):
                        mc = m
                        if mc > cmp_:
                            mc = cmp_
                    else:
                        mc = s_cnt[1]
                        if mc > cmp_:
                            mc = cmp_
                    quad = cutlass.Int32(0)
                    if mc >= m:
                        if mc <= qc:
                            quad = cutlass.Int32(1)
                    if quad == cutlass.Int32(1):
                        # O(mc^2) index-tie-broken rank
                        i = tid
                        while i < mc:
                            uq = cutlass.Uint32(ck[i])
                            r = cutlass.Int32(0)
                            j = cutlass.Int32(0)
                            while j < mc:
                                vq = cutlass.Uint32(ck[j])
                                tinc = cutlass.Int32(0)
                                if vq > uq:
                                    tinc = cutlass.Int32(1)
                                if vq == uq:
                                    if j < i:
                                        tinc = cutlass.Int32(1)
                                r = r + tinc
                                j = j + cutlass.Int32(1)
                            if r < need:
                                out_row[above + r] = ci[i]
                            i = i + cutlass.Int32(BLK)
                    else:
                        # ---- fallback: exact key-space narrowing
                        if tid == cutlass.Int32(0):
                            s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                            s_kmm[1] = cutlass.Uint32(0)
                        cute.arch.barrier()  # kmm init
                        i = tid
                        while i < mc:
                            kv = cutlass.Uint32(ck[i])
                            atomic_min_cta(s_kmm.iterator, kv)
                            atomic_max_cta(s_kmm.iterator + 1, kv)
                            i = i + cutlass.Int32(BLK)
                        cute.arch.barrier()  # key range published
                        rlo = cutlass.Uint32(s_kmm[0])
                        rhi = cutlass.Uint32(s_kmm[1])
                        ethr = cutlass.Int64(rlo)
                        aboveC = cutlass.Int32(0)
                        needC = need
                        mm = mc
                        lev = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(rlo) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if rlo >= rhi:
                                    ethr = cutlass.Int64(rlo)
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(rlo)
                                    done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = rhi - rlo
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNBH):
                                    sh2 = b2w - cutlass.Int32(LNBH)
                                for z in cutlass.range_constexpr(self.nbh // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < mc:
                                    unar = cutlass.Uint32(ck[i])
                                    if unar >= rlo:
                                        if unar <= rhi:
                                            bnn = (unar - rlo) >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32(bnn, cutlass.Uint32(self.nbh - 1))
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(bnn),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # level hist
                                if cutlass.const_expr(self.nbh > 1024):
                                    scan_cross_w(
                                        s_hist, s_ws, needC, tid, s_res, blk=self.blk, nb=self.nbh
                                    )
                                else:
                                    find_cross(s_hist, needC, tid, s_res, nb=self.nbh)
                                cute.arch.barrier()  # level scan
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = rlo + (cutlass.Uint32(b_lv) << cutlass.Uint32(sh2))
                                if b_lv != cutlass.Int32(self.nbh - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                        - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        # final two-predicate ballot emit
                        if tid == cutlass.Int32(0):
                            s_e12[0] = cutlass.Int32(0)
                            s_e12[1] = cutlass.Int32(0)
                        cute.arch.barrier()  # emit counters
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            idv = cutlass.Int32(0)
                            if i < mc:
                                uke = cutlass.Uint32(ck[i])
                                idv = ci[i]
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < mc:
                                if cutlass.Int64(uke) > ethr:
                                    q1f = cutlass.Int32(1)
                                if cutlass.Int64(uke) == ethr:
                                    q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_e12.iterator, popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_e12.iterator + 1, popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < aboveC:
                                    out_row[above + p1e] = idv
                            if q2f == cutlass.Int32(1):
                                if p2e < needC:
                                    out_row[above + aboveC + p2e] = idv
                            it = it + cutlass.Int32(1)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        cmp_: cutlass.Int32,
        qc: cutlass.Int32,
        smem_bytes: cutlass.Int32,
        stream,
    ):
        b = logits.shape[0]
        self.kern(logits, pre_idx, kv_lens, out, n, cmp_, qc, smem_bytes).launch(
            grid=(b, 1, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=self.minb,
            use_pdl=self.pdl,
        )


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__reg: dict = {}


def get_compiled__reg(tpl, dump_dir=None, pdl=False, varlen=False, next_n=1, cr_shift=0):
    """Compile (or fetch) the variant for constexpr tuple
    (BLK, VPT, MINB, KPT, CUR, DEG, IMG, NBH)."""
    key = (tuple(tpl), bool(pdl), bool(varlen), int(next_n), int(cr_shift), C._compile_arch_token())
    compiled = _COMPILE_CACHE__reg.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        blk, vpt, minb, kpt, cur, deg, img, nbh = tpl
        kernel = GvrTopkRegKernel(
            blk,
            vpt,
            minb,
            kpt,
            cur,
            deg,
            img,
            nbh,
            pdl=pdl,
            varlen=varlen,
            next_n=next_n,
            cr_shift=cr_shift,
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        lg_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
        )
        pi_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb2_, nc2_), stride_order=(1, 0), assumed_align=16
        )
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb3_, nc3_), stride_order=(1, 0), assumed_align=16
        )
        v0_ = cute.sym_int()
        kv_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (v0_,), stride_order=(0,), assumed_align=4
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = _base_compile_opts()
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"

        def _compile_fn():
            with _no_carveout():
                return cute.compile(
                    kernel,
                    lg_fake,
                    pi_fake,
                    kv_fake,
                    out_fake,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    stream=fake_stream,
                    options=opts,
                )

        if dump_dir:
            # debug dumps: compile in-process, do not persist
            compiled = _compile_fn()
        else:
            name = (
                "reg_" + "_".join(str(x) for x in tuple(tpl))
                + f"_pdl{int(bool(pdl))}_vl{int(bool(varlen))}_nn{int(next_n)}_cs{int(cr_shift)}"
            )
            compiled = C._persist(name, _compile_fn)
        _COMPILE_CACHE__reg[key] = compiled
    return compiled


def reg_topk(logits, pre_idx, n, out, rd=None):
    """torch-facing entry for the register family.

    logits [b, npad] f32, pre_idx [b, k] i32, out [b, >=k] i32, n = valid len.
    rd: optional pre-computed ct_dispatch.route() dict (must be reg/regimg).
    """
    if rd is None:
        try:
            from .gvr2_topk_host import route
        except ImportError:
            from gvr2_topk_host import route
        rd = route(logits.shape[0], int(n), logits.shape[1], pre_idx.shape[1])
    assert rd["kernel"] in ("reg", "regimg"), rd["kernel"]
    tpl = rd["tpl"]
    rt = rd["rt"]
    assert rt["IMGOFF"] == tpl[7], (rt["IMGOFF"], tpl[7])  # IMGOFF == NBH
    compiled = get_compiled__reg(tpl)
    smem = STATIC_BYTES + rd["smem"]
    try:
        from .gvr2_topk_host import _dummy_kv
    except ImportError:
        from gvr2_topk_host import _dummy_kv
    kv = _dummy_kv(logits.get_device(), logits.device)  # dead varlen ABI slot
    compiled(logits, pre_idx, kv, out, int(n), rt["CMP"], rt["QC"], smem)
    return out


# ===========================================================================
# ==== family: clus ============================================
# ===========================================================================
"""gvr_clus — clustered streaming self-sampling GVR; per-CTA stream mirrors
gvr_main.

Ctor knobs (compile-time, mirror of the CUDA template params):
    BLK = 1024, U ∈ {1,2,4,8}, MINB = 1, NBS = 256, CS ∈ {2,4,8}
    (+ scap/cmp smem-extent knobs: every reachable route has 8192/2048 —
     SCAP/CMP stay LIVE runtime args for all value logic, ABI parity).
Derived: HB=NBS, STEPC=BLK*U, PFD=min(U,4).

Signature (ABI parity with the CUDA form; Q is dead in-kernel):
    run__clus(logits[b,npad] f32, pre_idx[b,k] i32, out[b,k] i32) via
    kern(..., n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2)
Grid dim3(CS, b) native 2-D + cluster (CS,1,1); block 1024;
min_blocks_per_mp=1 (64-register wall); smem one SmemAllocator blob
mirroring the CUDA dynamic map hist|cbuf|ck64c|mrg, dyn-equivalent bytes ==
the host dispatch formula (asserted in run__clus()).

int2 staging convention (same as gvr_main): int2(value bits, index) is ONE
little-endian Uint64 = (idx << 32) | value_bits — single u64 smem ld/st.

Barrier / cluster-op placement mirrors the CUDA source one-for-one:
    sample redux publish, sample hist, scan publish,
    [degenerate sample: 2 inside gather_hint],
    retry preamble: clus.sync + __syncthreads,
    clus.sync (merge), __syncthreads (merge publish),
    [ladder gather: 2 inside gather_hint],
    clus.sync (EXIT RENDEZVOUS — the only one; rank!=0 falls through),
    narrowing / degen per-level barriers (+1 INSIDE scan_cross).
    NO loop-tail ladder barriers (gvr_clus has none — unlike gvr_main).
    All clus.sync = releasing aligned arrive+wait, never relaxed.
Cluster ops: merge = _merge_scan0_local, a LOCAL patched copy of
    merge_scan0 that rematerializes mapa per (q, r) like the CUDA
    (register-pressure fix); ONE packed u64 st.shared::cluster candidate
    push to rank-0 ck64c (never split into 4B stores); mapa of ck64c to
    rank 0.
Every rung/ladder decision is cluster-uniform by construction (identical
sample locations on every rank; merged tot; block-uniform gather) — the
conditional retry clus.sync cannot deadlock.
"""


QUADC_CLUS__clus = C.QUADC_CLUS

_NEG_INF__clus = float("-inf")


# ---------------------------------------------------------------------------
# single-rounding fma.rn.f32, used at the T / Tk / T3 / HIC sites.
# (x-TF)*SC classify shapes stay plain sub+mul (uncontractible).
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__clus(a, b, c, *, loc=None, ip=None):
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


class GvrClusKernel:
    """gvr_clus<BLK, U, MINB, NBS, CS> — clustered streaming GVR."""

    def __init__(
        self,
        blk: int,
        u: int,
        minb: int,
        nbs: int,
        cs: int,
        scap: int = 8192,
        cmp_: int = 2048,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
    ):
        assert blk == 1024, "gvr_clus is always BLK=1024"
        assert minb == 1, "gvr_clus is __launch_bounds__(BLK, 1)"
        assert nbs == 256, "SNB must stay 256"
        assert u in (1, 2, 4, 8) and cs in (2, 4, 8)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel / GvrRegClusKernel): n and the sampling-
        # ladder scalars are re-derived PER ROW in-kernel from a device
        # kv_lens tensor; the scalar launch args become the envelope clamp
        # bound (n) and dead slots (SMP/TGT/Q/SS2/TGT2).
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        self.lcs = cs.bit_length() - 1  # log2(CS) for the per-row Q shift
        self.blk = blk
        self.u = u
        self.minb = minb
        self.nbs = nbs
        self.cs = cs
        self.scap = scap  # smem extents only —
        self.cmp = cmp_  # value logic uses rt args
        self.hb = nbs
        self.stepc = blk * u
        self.pfd = u if u < 4 else 4  # PFD=min(U,4)
        self.lb = nbs.bit_length() - 1  # log2(NBS)=8
        # dynamic-region byte map: hist | cbuf(int2) | ck64c | mrg
        self.cbuf_bytes = (scap + 4) * 8
        assert self.cbuf_bytes % 16 == 0
        self.ck_off = self.cbuf_bytes  # inside the blob
        self.dyn_bytes = nbs * 4 + self.cbuf_bytes + cmp_ * 8 + nbs * 4
        # == host smc = SNB*8 + (SCAP+4)*8 + CMP*8

    # ------------------------------------------------------------------
    # GVR_EMITK: classify+stage one survivor.
    # bn via UNSIGNED saturating convert (f2u_rz); staging store is ONE
    # u64; branchless trash slot min(pos, SCAP) (runtime SCAP). Returns pos+1.
    # ------------------------------------------------------------------
    @cute.jit
    def _emitk(self, xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2):
        NBS = self.nbs
        bn_u = C.f2u_rz((xv - TF) * SC)
        if bn_u > cutlass.Uint32(NBS - 1):
            bn_u = cutlass.Uint32(NBS - 1)
        bn = cutlass.Int32(bn_u)
        C.atomic_add_cta(s_hist.iterator + bn, cutlass.Int32(1))
        ps = pos
        if ps > SCAP:
            ps = SCAP  # trash slot (IMNMX)
        s_cbuf2[ps] = (cutlass.Uint64(cutlass.Uint32(idx)) << cutlass.Uint64(32)) | cutlass.Uint64(
            C.u32_of_f32(xv)
        )
        return pos + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # P5 emit step: bn via SIGNED rz convert (__float2int_rz); bn>=B gate;
    # LOCAL mrg atomicAdd whose result is a CLUSTER-GLOBAL position
    # (prefix-biased cursors from merge_scan0); overflow -> ONE packed u64
    # DSMEM store to rank-0 ck64c (never split into 4B stores).
    # ------------------------------------------------------------------
    @cute.jit
    def _p5_emit(self, xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64):
        NBS = self.nbs
        bn = C.f2s_rz((xv - TF) * SC)
        if bn > cutlass.Int32(NBS - 1):
            bn = cutlass.Int32(NBS - 1)
        if bn >= B:
            p = C.atomic_add_cta(s_mrg.iterator + bn, cutlass.Int32(1))
            if p < lim1:
                out_row[p] = idv
            else:
                if whole == cutlass.Int32(0):
                    q2 = p - above
                    if q2 < CMP:
                        C._st_shared_cluster_u64(
                            rk64 + q2 * cutlass.Int32(8),
                            (cutlass.Uint64(C.fkey(xv)) << cutlass.Uint64(32))
                            | cutlass.Uint64(cutlass.Uint32(idv)),
                        )

    # ------------------------------------------------------------------
    # LOCAL patched copy of merge_scan0: rematerializes mapa per (q, r)
    # exactly like the CUDA instead of holding CS mapped base addresses
    # across the whole merge. The hoisted-array form costs CS extra
    # long-lived registers; with the U>=4 sixteen-register pf prime batch
    # it tips ptxas into spilling the batch across the rung phase.
    # Semantics, the DSMEM v4 load spelling, the register accumulation and
    # the prefix-biased STS.128 cursor write are IDENTICAL to merge_scan0.
    # NO barrier inside (caller pays the merge-publish barrier).
    # ------------------------------------------------------------------
    @cute.jit
    def _merge_scan0_local(self, s_hist, s_mrg, rank, target, tidx, s_res):
        NBS = self.nbs
        CS = self.cs
        BPT = NBS // 32
        NV = BPT // 4
        if tidx < cutlass.Int32(32):
            lane = tidx
            atom = C.smem_atom_i32_128()
            hbase = s_hist.iterator.toint()
            # pass 1: remote v4 accumulation of tot/pre per vector
            tot_r = []
            pre_r = []
            sm = cutlass.Int32(0)
            for q in cutlass.range_constexpr(NV):
                boff = (lane * cutlass.Int32(BPT) + cutlass.Int32(4 * q)) * cutlass.Int32(4)
                t = [cutlass.Int32(0)] * 4
                p = [cutlass.Int32(0)] * 4
                for r in cutlass.range_constexpr(CS):
                    mapped = C._mapa_shared_cluster_addr(
                        hbase + boff, cutlass.Int32(r)
                    )  # per-use mapa
                    v0, v1, v2, v3 = C._ld_shared_cluster_v4_u32(mapped)
                    t[0] = t[0] + v0
                    t[1] = t[1] + v1
                    t[2] = t[2] + v2
                    t[3] = t[3] + v3
                    if cutlass.Int32(r) < rank:  # predicated adds
                        p[0] = p[0] + v0
                        p[1] = p[1] + v1
                        p[2] = p[2] + v2
                        p[3] = p[3] + v3
                tot_r.append(t)
                pre_r.append(p)
                sm = sm + t[0] + t[1] + t[2] + t[3]
            # inclusive scan + totals
            w = C.warp_incl_scan_add(sm, lane)
            tt = cute.arch.shuffle_sync(w, cutlass.Int32(31))
            after = tt - w
            if lane == cutlass.Int32(0):
                s_res[C.RES_TOT] = tt
            base = lane * cutlass.Int32(BPT)
            # descending walk: crossing pin + prefix-biased cursors into mrg
            for q in cutlass.range_constexpr(NV - 1, -1, -1):
                o4 = cute.make_rmem_tensor((4,), cutlass.Int32)
                for j in cutlass.range_constexpr(3, -1, -1):
                    cq = tot_r[q][j]
                    o4[j] = after + pre_r[q][j]
                    gb = base + cutlass.Int32(4 * q + j)
                    cross = cutlass.Int32(0)
                    if after < target:
                        if (after + cq) >= target:
                            cross = cutlass.Int32(1)
                        if gb == cutlass.Int32(0):
                            cross = cutlass.Int32(1)
                    if cross != cutlass.Int32(0):
                        s_res[C.RES_B] = gb
                        s_res[C.RES_ABOVE] = after
                        s_res[C.RES_M] = cq
                    after = after + cq
                boff = (lane * cutlass.Int32(NV) + cutlass.Int32(q)) * cutlass.Int32(16)
                C.sts128_i32(atom, o4, s_mrg.iterator.toint(), boff)

    # ------------------------------------------------------------------
    # two-predicate warp-ballot emit step (narrowing and degen emits) —
    # same helper as the main family. s_scal[1]=s_o1, s_scal[2]=s_o2.
    # ------------------------------------------------------------------
    @cute.jit
    def _ballot_pair_emit(self, p1, p2, idv, base1, cap1, base2, cap2, out_row, s_scal, lane):
        n1 = C.ballot(p1 != cutlass.Int32(0))
        n2 = C.ballot(p2 != cutlass.Int32(0))
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        if lane == cutlass.Int32(0):
            if n1 != cutlass.Int32(0):
                b1 = C.atomic_add_cta(s_scal.iterator + 1, cutlass.Int32(C.popc(n1)))
            if n2 != cutlass.Int32(0):
                b2 = C.atomic_add_cta(s_scal.iterator + 2, cutlass.Int32(C.popc(n2)))
        b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
        b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
        lm = cutlass.Int32(cute.arch.lanemask_lt())
        if p1 != cutlass.Int32(0):
            p = b1 + cutlass.Int32(C.popc(n1 & lm))
            if p < cap1:
                out_row[base1 + p] = idv
        if p2 != cutlass.Int32(0):
            p = b2 + cutlass.Int32(C.popc(n2 & lm))
            if p < cap2:
                out_row[base2 + p] = idv

    # ------------------------------------------------------------------
    # kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        bigf: cutlass.Int32,
    ):
        BLK = self.blk
        U = self.u
        NBS = self.nbs
        CS = self.cs
        PFD = self.pfd
        STEPC = self.stepc
        NW = BLK // 32

        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()  # (rank, row)
        rank = bx
        row = by
        lane = tidx & cutlass.Int32(31)

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens; ladder scalars are
        # dead launch args, re-derived here by the route_dynamic() clus
        # formulas). Clus deviation: QUAD sample geometry runs for every
        # non-short row (host never launches clus at n <= SCAP, so SMP == 0 is
        # untested; sampling only steers the rung — exactness is
        # schedule-invariant). All values are pure functions of `row`, so the
        # whole-body guard and the cluster barriers stay cluster-uniform.
        # Short rows (n <= k): rank 0 emits identity + (-1); body is SKIPped.
        short = cutlass.Int32(0)
        prow = row
        if cutlass.const_expr(self.varlen):
            kq = cutlass.Int32(pre_idx.shape[1])
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            prow = req
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            if nv > n:
                nv = n
            if nv <= kq:
                short = cutlass.Int32(1)
            if short == cutlass.Int32(0):
                n = nv
                # ---- aim ladder (cheap mirror, all-thread) ----
                # Schedule quantities only (exactness is schedule-invariant,
                # same argument as the always-sample deviation above): divides
                # become MUFU.RCP multiplies and the isqrt fixup loops collapse
                # to single steps (f32 sqrt of an exactly-representable int
                # (6n <= 2^23) is within 1 of isqrt).  All-thread on purpose:
                # this family's mirror redundancy is small, and a
                # warp0+barrier hoist EXPOSES the chain's serial latency at
                # ~1 CTA/SM, while the redundant form hides it across warps.
                # Q (chunk ownership) keeps its exact shift form.
                x6 = cutlass.Int32(6) * nv
                ri = cutlass.Int32(cmath.sqrt(cutlass.Float32(x6)))
                if ri * ri > x6:
                    ri = ri - cutlass.Int32(1)
                if (ri + cutlass.Int32(1)) * (ri + cutlass.Int32(1)) <= x6:
                    ri = ri + cutlass.Int32(1)
                r6 = ri
                if x6 - ri * ri > ri:
                    r6 = ri + cutlass.Int32(1)
                # aim_base: R = CS > 1 always for this family; bigf is the
                # launch-computed occupancy flag (num_rows * CS <= 148).
                aim = k << cutlass.Int32(1)
                if bigf == cutlass.Int32(0):
                    if k >= cutlass.Int32(1024):
                        aim = (cutlass.Int32(11) * k) >> cutlass.Int32(3)
                    else:
                        aim = (cutlass.Int32(3) * k) >> cutlass.Int32(1)
                if r6 > aim:
                    aim = r6
                amin = cutlass.Int32(3) * k
                if cutlass.const_expr(self.cs != 2):
                    amin = (cutlass.Int32(7) * k) >> cutlass.Int32(1)
                if aim < amin:
                    aim = amin
                if aim > (SCAP >> cutlass.Int32(1)):
                    aim = SCAP >> cutlass.Int32(1)
                if aim < k:
                    aim = k
                # ---- QUAD sample geometry (route_dynamic clus override,
                # always-on per the deviation note above; k <= 1024 for this
                # family so sfac has no k > 1024 arm).
                n4v = nv >> cutlass.Int32(2)
                sel = cutlass.Int32(
                    cutlass.Float32(nv)
                    * cutlass.Float32(32.0 if self.cs == 2 else 16.0)
                    * cute.arch.rcp_approx(cutlass.Float32(aim))
                )
                if sel < cutlass.Int32(256):
                    sel = cutlass.Int32(256)
                nh = nv >> cutlass.Int32(1)
                if sel > nh:
                    sel = nh
                quads = sel >> cutlass.Int32(4)
                if quads < cutlass.Int32(1):
                    quads = cutlass.Int32(1)
                quarter = n4v >> cutlass.Int32(2)
                if quarter < cutlass.Int32(1):
                    quarter = cutlass.Int32(1)
                if quads > quarter:
                    quads = quarter
                SS2 = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(quads))
                )
                if SS2 < cutlass.Int32(1):
                    SS2 = cutlass.Int32(1)
                SMP = cutlass.Int32(
                    cutlass.Float32(quarter) * cute.arch.rcp_approx(cutlass.Float32(SS2))
                )
                # sample-window guard: P1 indexes up to ~SMP*SS2*4 lines; keep
                # SMP*SS2 <= quarter (approx error <= +1, one step closes it)
                if SMP * SS2 > quarter:
                    SMP = SMP - cutlass.Int32(1)
                if SMP < cutlass.Int32(1):
                    SMP = cutlass.Int32(1)
                rn_ = cute.arch.rcp_approx(cutlass.Float32(nv))
                smp16f = cutlass.Float32(SMP) * cutlass.Float32(16.0)
                TGT = cutlass.Int32(cutlass.Float32(aim) * smp16f * rn_)
                if TGT < cutlass.Int32(1):
                    TGT = cutlass.Int32(1)
                TGT2 = cutlass.Int32(cutlass.Float32(k) * smp16f * rn_)
                if TGT2 < cutlass.Int32(1):
                    TGT2 = cutlass.Int32(1)
                # NB: the Q launch slot is dead in this family (no in-kernel
                # consumer); chunk ownership derives from n4/STEPC below.
            if short != cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    if tidx < kq:
                        ov = cutlass.Int32(-1)
                        if tidx < nv:
                            ov = tidx
                        out[row, tidx] = ov

        # ---- shared memory (CUDA dynamic map order, then static allocs) ----
        smem = SmemAllocator()
        s_hist = smem.allocate_tensor(  # hist[NBS] @ blob start
            cutlass.Int32, cute.make_ordered_layout((self.hb,), order=(0,)), byte_alignment=128
        )
        blob = smem.allocate_tensor(  # cbuf(int2) | ck64c
            cutlass.Int8,
            cute.make_ordered_layout((self.cbuf_bytes + self.cmp * 8,), order=(0,)),
            byte_alignment=16,
        )
        s_mrg = smem.allocate_tensor(  # mrg[NBS]
            cutlass.Int32, cute.make_ordered_layout((self.nbs,), order=(0,)), byte_alignment=16
        )
        s_ws = smem.allocate_tensor(  # degen scan only
            cutlass.Int32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        # scan_cross predeclares its second-stage partial as Int32 and reads
        # s_ws inside a dynamic if — a Uint32 ws tensor trips the DSL type-
        # stability check (counts < 2^31 so Int32 is exact).
        s_wmn = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_wmx = smem.allocate_tensor(
            cutlass.Uint32, cute.make_ordered_layout((NW,), order=(0,)), byte_alignment=16
        )
        s_res = smem.allocate_tensor(  # shared slot map
            cutlass.Int32, cute.make_ordered_layout((8,), order=(0,)), byte_alignment=16
        )
        # scalar block: [0]=s_bufn [1]=s_o1 [2]=s_o2
        s_scal = smem.allocate_tensor(
            cutlass.Int32, cute.make_ordered_layout((4,), order=(0,)), byte_alignment=16
        )
        s_tsh = smem.allocate_tensor(
            cutlass.Float32, cute.make_ordered_layout((1,), order=(0,)), byte_alignment=4
        )
        s_kmm = smem.allocate_tensor(  # [0]=kmin [1]=kmax
            cutlass.Uint32, cute.make_ordered_layout((2,), order=(0,)), byte_alignment=8
        )
        sbase = blob.iterator.toint()
        s_cbuf2 = cute.make_tensor(  # int2 staged as u64
            cute.make_ptr(cutlass.Uint64, sbase, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.scap + 4,)),
        )
        ck_addr = sbase + cutlass.Int32(self.ck_off)
        s_ck64 = cute.make_tensor(
            cute.make_ptr(cutlass.Uint64, ck_addr, cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.cmp,)),
        )

        # ---- whole-body short-row guard (cluster-uniform: `short` is a pure
        # function of `row`, identical across all CS ranks and threads, so
        # every cluster barrier below stays aligned; short rows already
        # emitted identity + -1 tail in the prologue) ----
        if short == cutlass.Int32(0):
            # ---- row bases (pre_idx is request-level under varlen) ----
            row64 = cutlass.Int64(row)
            x_addr = logits.iterator.toint() + row64 * cutlass.Int64(npad) * cutlass.Int64(4)
            p_addr = pre_idx.iterator.toint() + cutlass.Int64(prow) * cutlass.Int64(
                k
            ) * cutlass.Int64(4)
            out_row = out[row, None]

            # ---- interleaved chunk ownership ----
            n4 = n >> cutlass.Int32(2)
            nCh = (n4 + cutlass.Int32(STEPC - 1)) // cutlass.Int32(STEPC)
            nFullG = n4 // cutlass.Int32(STEPC)
            tail0 = n4 << cutlass.Int32(2)
            tailn = cutlass.Int32(0)
            if rank == cutlass.Int32(0):
                tailn = n - tail0

            if tidx == cutlass.Int32(0):
                s_res[C.RES_B2] = cutlass.Int32(-1)
                s_res[C.RES_B3] = cutlass.Int32(-1)
                s_scal[0] = cutlass.Int32(0)  # s_bufn
            if tidx < cutlass.Int32(self.hb):  # HB<=BLK
                s_hist[tidx] = cutlass.Int32(0)

            # ============ P1: QUAD sample (hint gather LAZY) =====================
            # one 64B line = 4 float4 per location, TWO threads: tid takes the
            # lower pair at p4, tid+SMP the upper pair at p4+2.
            atom128 = C.g2r_atom_f32(128, invariant=True)
            fsa = cute.make_rmem_tensor((4,), cutlass.Float32)
            fsb = cute.make_rmem_tensor((4,), cutlass.Float32)
            smp2 = SMP * cutlass.Int32(2)
            shas = cutlass.Int32(0)
            if tidx < smp2:
                shas = cutlass.Int32(1)
            if shas != cutlass.Int32(0):
                p4 = tidx * SS2 * cutlass.Int32(4)
                if tidx >= SMP:
                    p4 = (tidx - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fsa)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fsb)

            # ============ P2: quantile rung, redundant per CTA ===================
            smn = cutlass.Float32(float("inf"))
            smx = cutlass.Float32(float("-inf"))
            if shas != cutlass.Int32(0):
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fsa[t])
                    smx = C.fmax_f32(smx, fsa[t])
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fsb[t])
                    smx = C.fmax_f32(smx, fsb[t])
            fma_ = cute.make_rmem_tensor((4,), cutlass.Float32)  # mop-up pair bufs
            fmb_ = cute.make_rmem_tensor((4,), cutlass.Float32)
            j = tidx + cutlass.Int32(BLK)  # mop-up
            while j < smp2:
                p4 = j * SS2 * cutlass.Int32(4)
                if j >= SMP:
                    p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fma_[t])
                    smx = C.fmax_f32(smx, fma_[t])
                for t in cutlass.range_constexpr(4):
                    smn = C.fmin_f32(smn, fmb_[t])
                    smx = C.fmax_f32(smx, fmb_[t])
                j = j + cutlass.Int32(BLK)
            a0 = C.warp_min_u32(C.fkey(smn))
            c0m = C.warp_max_u32(C.fkey(smx))
            if lane == cutlass.Int32(0):
                s_wmn[tidx >> cutlass.Int32(5)] = a0
                s_wmx[tidx >> cutlass.Int32(5)] = c0m
            cute.arch.barrier()  # ---- barrier (sample redux publish) ----

            # PRIME-LATE: every rank's sample has landed; prime NOW.
            lim4 = (npad >> cutlass.Int32(2)) - cutlass.Int32(1)
            pf = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(PFD)]
            for uu in cutlass.range_constexpr(PFD):  # clamped prime
                i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                ic = i_
                if ic >= n4:
                    ic = lim4
                C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
            # asm prefetch gate: DEEP rows only; empty for U<=PFD
            if cutlass.const_expr(U > PFD):
                gpp = cutlass.Int32(0)
                if n4 >= cutlass.Int32(32768):
                    if (rank + cutlass.Int32(1)) * cutlass.Int32(STEPC) <= n4:
                        gpp = cutlass.Int32(1)
                if gpp != cutlass.Int32(0):
                    for uu in cutlass.range_constexpr(PFD, U):
                        C._prefetch_l2(
                            x_addr
                            + cutlass.Int64(
                                rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                            )
                            * cutlass.Int64(16)
                        )

            # cross-warp sample reduce
            av = cutlass.Uint32(0xFFFFFFFF)
            cv = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                av = s_wmn[lane]
                cv = s_wmx[lane]
            SMIN = C.invkey(C.warp_min_u32(av))
            SMAX = C.invkey(C.warp_max_u32(cv))

            GMIN = cutlass.Float32(C.SENT_LO)  # sentinels
            GMAX = cutlass.Float32(C.SENT_HI)
            T = cutlass.Float32(_NEG_INF__clus)
            HIC = cutlass.Float32(_NEG_INF__clus)
            w = cutlass.Float32(0.0)
            sok = cutlass.Int32(0)
            if SMP > cutlass.Int32(0):
                if SMAX > SMIN:
                    sok = cutlass.Int32(1)
            if sok != cutlass.Int32(0):  # sample hist
                w = (SMAX - SMIN) * cutlass.Float32(1.0 / 256.0)
                # CUDA --use_fast_math lowers `1.0f / w` to a bare MUFU.RCP;
                # a plain `/` would emit the IEEE div.rn rcp+Newton+CALL
                # chain. w > 0 by the sok guard; bucketing is SC-invariant.
                sc_s = cute.arch.rcp_approx(w)
                if shas != cutlass.Int32(0):
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fsa[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fsb[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                j = tidx + cutlass.Int32(BLK)  # mop-up reloads
                while j < smp2:
                    p4 = j * SS2 * cutlass.Int32(4)
                    if j >= SMP:
                        p4 = (j - SMP) * SS2 * cutlass.Int32(4) + cutlass.Int32(2)
                    C.ld_g_f32x4(atom128, x_addr, p4, fma_)
                    C.ld_g_f32x4(atom128, x_addr, p4 + cutlass.Int32(1), fmb_)
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fma_[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    for t in cutlass.range_constexpr(4):
                        bq = C.f2s_rz((fmb_[t] - SMIN) * sc_s)
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                    j = j + cutlass.Int32(BLK)
            cute.arch.barrier()  # ---- barrier (sample histogram) ----
            # triple-target ZERO scan: TGT / TGT2 / 2*TGT
            C.scan_cross0(
                s_hist,
                TGT,
                tidx,
                s_res,
                TGT2,
                TGT * cutlass.Int32(2),
                s_hist,
                nb=NBS,
                zero=True,
                two=True,
                three=True,
            )
            cute.arch.barrier()  # ---- barrier (scan publish) ----

            tot0 = s_res[C.RES_TOT]
            b1v = s_res[C.RES_B]
            if sok != cutlass.Int32(0):
                if tot0 >= TGT:
                    T = _fmaf__clus(cutlass.Float32(b1v), w, SMIN)
            needg = cutlass.Int32(1)
            if T > cutlass.Float32(_NEG_INF__clus):
                needg = cutlass.Int32(0)
            if needg != cutlass.Int32(0):
                # degenerate sample: identical on every rank of the cluster
                GMIN, GMAX = C.gather_hint(
                    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                )  # 2 barriers
                T = GMIN
            if sok != cutlass.Int32(0):  # HIC tighten
                if tot0 >= TGT:
                    b2v = s_res[C.RES_B2]
                    if b2v >= cutlass.Int32(0):
                        Tk = _fmaf__clus(cutlass.Float32(b2v), w, SMIN)
                        up = C.fmax_f32(Tk - T, cutlass.Float32(0.0))
                        # heavy-tail cap by T - T3 (rank-TGT..rank-2TGT distance)
                        if tot0 >= TGT * cutlass.Int32(2):
                            b3v = s_res[C.RES_B3]
                            if b3v >= cutlass.Int32(0):
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T > T3:
                                    up = C.fmin_f32(up, cutlass.Float32(2.0) * (T - T3))
                        HIC = C.fmax_f32(
                            _fmaf__clus(cutlass.Float32(4.0), up, T),
                            _fmaf__clus(cutlass.Float32(8.0), w, T),
                        )
            # ladder floor kept in SHARED (64-register wall)
            if tidx == cutlass.Int32(0):
                t5 = cutlass.Float32(_NEG_INF__clus)
                if sok != cutlass.Int32(0):
                    if tot0 >= TGT * cutlass.Int32(2):
                        b3v = s_res[C.RES_B3]
                        if b3v >= cutlass.Int32(0):
                            if T > GMIN:
                                T3 = _fmaf__clus(cutlass.Float32(b3v), w, SMIN)
                                if T3 < T:
                                    t5 = T3
                s_tsh[0] = t5

            # ============ attempt loop — MUST NOT unroll ===========
            listN = cutlass.Int32(0)
            above = cutlass.Int32(0)
            m = cutlass.Int32(0)
            need = cutlass.Int32(0)
            B = cutlass.Int32(0)
            SC = cutlass.Float32(1.0)
            TF = T
            complete = cutlass.Int32(0)
            valid = cutlass.Int32(0)

            fr = [
                cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(U - PFD)
            ]  # explicit batch
            # (empty for U<=PFD — every row-pass float4 then comes from pf[])
            att = cutlass.Int32(0)
            running = cutlass.Int32(1)
            while running != cutlass.Int32(0):
                if att > cutlass.Int32(0):  # retry preamble
                    # exactness: re-prime pf[] (holds stale roll data)
                    if rank < nFullG:
                        for uu in cutlass.range_constexpr(PFD):
                            C.ld_g_f32x4(
                                atom128,
                                x_addr,
                                rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK),
                                pf[uu],
                            )
                    else:
                        for uu in cutlass.range_constexpr(PFD):
                            i_ = rank * cutlass.Int32(STEPC) + tidx + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= n4:
                                ic = lim4
                            C.ld_g_f32x4(atom128, x_addr, ic, pf[uu])
                    C._cluster_sync_aligned()  # ==== clus.sync (retry) ====
                    if tidx < cutlass.Int32(NBS):
                        s_hist[tidx] = cutlass.Int32(0)
                    if tidx == cutlass.Int32(0):
                        s_scal[0] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (retry reset) ----

                TF = T  # window
                hi = C.fmax_f32(GMAX, T)
                if HIC > T:
                    if HIC < hi:
                        hi = HIC
                WD = (hi - T) * cutlass.Float32(1.0 / 256.0)
                wdok = cutlass.Int32(0)
                if WD > cutlass.Float32(0.0):
                    wdok = cutlass.Int32(1)
                if wdok == cutlass.Int32(0):
                    WD = cutlass.Float32(1e-30)
                # MUFU.RCP spelling: WD >= 1e-30 finite by the wdok clamp;
                # classify bucketing is SC-invariant for any SC > 0.
                SC = cute.arch.rcp_approx(WD)

                # ---- P3 row pass over OWNED CHUNKS ----
                g = rank + cutlass.Int32(0)
                while g < nCh:
                    i0 = g * cutlass.Int32(STEPC) + tidx
                    M = cutlass.Int32(0)
                    isfull = cutlass.Int32(0)
                    if g < nFullG:
                        isfull = cutlass.Int32(1)
                    if isfull != cutlass.Int32(0):  # full body
                        for uu in cutlass.range_constexpr(PFD, U):
                            C.ld_g_f32x4(
                                atom128, x_addr, i0 + cutlass.Int32(uu * BLK), fr[uu - PFD]
                            )
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            for q in cutlass.range_constexpr(4):
                                M = M | (cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q))
                    else:  # partial body
                        for uu in cutlass.range_constexpr(PFD, U):
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            ic = i_
                            if ic >= n4:
                                ic = lim4  # clamp in [n, npad)
                            C.ld_g_f32x4(atom128, x_addr, ic, fr[uu - PFD])
                        for uu in cutlass.range_constexpr(U):
                            if cutlass.const_expr(uu < PFD):
                                vv = pf[uu]
                            else:
                                vv = fr[uu - PFD]
                            i_ = i0 + cutlass.Int32(uu * BLK)
                            okq = cutlass.Int32(0)
                            if i_ < n4:
                                okq = cutlass.Int32(1)
                            if okq != cutlass.Int32(0):  # +inf-pad escape, ok-gated
                                for q in cutlass.range_constexpr(4):
                                    M = M | (
                                        cutlass.Int32(vv[q] >= TF) << cutlass.Int32(uu * 4 + q)
                                    )
                    # ROLL THE PREFETCH FORWARD: next OWNED chunk, issued
                    # before the reservation and the survivor walk.
                    g2 = g + cutlass.Int32(CS)
                    if g2 < nCh:
                        j0 = g2 * cutlass.Int32(STEPC) + tidx
                        infull = cutlass.Int32(0)
                        if g2 < nFullG:
                            infull = cutlass.Int32(1)
                        if infull != cutlass.Int32(0):
                            for uu in cutlass.range_constexpr(PFD):
                                C.ld_g_f32x4(atom128, x_addr, j0 + cutlass.Int32(uu * BLK), pf[uu])
                        else:
                            for uu in cutlass.range_constexpr(PFD):
                                j_ = j0 + cutlass.Int32(uu * BLK)
                                jc = j_
                                if jc >= n4:
                                    jc = lim4
                                C.ld_g_f32x4(atom128, x_addr, jc, pf[uu])
                    # warp-aggregated slot reservation
                    cnt = cutlass.Int32(C.popc(M))
                    inc = C.warp_incl_scan_add(cnt, lane)
                    bpos = cutlass.Int32(0)
                    if lane == cutlass.Int32(31):
                        if inc != cutlass.Int32(0):
                            bpos = C.atomic_add_cta(s_scal.iterator + 0, inc)
                    pos = cute.arch.shuffle_sync(bpos, cutlass.Int32(31)) + (inc - cnt)
                    # survivor bit-walk, software-pipelined ONE deep;
                    # reload X[idx] — never hold the U float4s across the walk
                    if M != cutlass.Int32(0):
                        bp = C.ffs_m1(M)
                        M = M & (M - cutlass.Int32(1))
                        idx = (
                            (i0 + (bp >> cutlass.Int32(2)) * cutlass.Int32(BLK)) << cutlass.Int32(2)
                        ) + (bp & cutlass.Int32(3))
                        xv = C.ldg_f32(x_addr, idx)
                        while M != cutlass.Int32(0):
                            bp2 = C.ffs_m1(M)
                            M = M & (M - cutlass.Int32(1))
                            idx2 = (
                                (i0 + (bp2 >> cutlass.Int32(2)) * cutlass.Int32(BLK))
                                << cutlass.Int32(2)
                            ) + (bp2 & cutlass.Int32(3))
                            xv2 = C.ldg_f32(x_addr, idx2)
                            pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                            idx = idx2
                            xv = xv2
                        pos = self._emitk(xv, idx, pos, TF, SC, SCAP, s_hist, s_cbuf2)
                    g = g + cutlass.Int32(CS)
                # rank-0 scalar tail: per-thread atomics, bound-check
                i = tidx
                while i < tailn:
                    x = C.ldg_f32(x_addr, tail0 + i)
                    if x >= TF:
                        bq = C.f2s_rz((x - TF) * SC)  # signed form
                        if bq > cutlass.Int32(NBS - 1):
                            bq = cutlass.Int32(NBS - 1)
                        C.atomic_add_cta(s_hist.iterator + bq, cutlass.Int32(1))
                        post = C.atomic_add_cta(s_scal.iterator + 0, cutlass.Int32(1))
                        if post < SCAP:
                            s_cbuf2[post] = (
                                cutlass.Uint64(cutlass.Uint32(tail0 + i)) << cutlass.Uint64(32)
                            ) | cutlass.Uint64(C.u32_of_f32(x))
                    i = i + cutlass.Int32(BLK)

                # ---- cluster merge ----
                C._cluster_sync_aligned()  # ==== clus.sync (merge) ====
                myn = s_scal[0]
                self._merge_scan0_local(s_hist, s_mrg, rank, k, tidx, s_res)
                cute.arch.barrier()  # ---- barrier (merge publish) ----
                tot = s_res[C.RES_TOT]
                acc = cutlass.Int32(0)
                if tot >= k:
                    acc = cutlass.Int32(1)
                if acc != cutlass.Int32(0):  # accept
                    valid = cutlass.Int32(1)
                    complete = cutlass.Int32(0)
                    if myn <= SCAP:
                        complete = cutlass.Int32(1)
                    listN = myn
                    above = s_res[C.RES_ABOVE]
                    m = s_res[C.RES_M]
                    need = k - s_res[C.RES_ABOVE]
                    B = s_res[C.RES_B]
                    running = cutlass.Int32(0)
                else:
                    if att == cutlass.Int32(2):  # ladder exhausted
                        running = cutlass.Int32(0)
                    else:
                        # rung ladder — cluster-uniform on every arm
                        tshtaken = cutlass.Int32(0)
                        if att == cutlass.Int32(0):
                            T5 = s_tsh[0]
                            if T5 > cutlass.Float32(_NEG_INF__clus):
                                if T5 < TF:
                                    T = T5
                                    tshtaken = cutlass.Int32(1)
                        if tshtaken == cutlass.Int32(0):
                            # LAZY GATHER — every rank computes identical GMIN
                            if GMIN == cutlass.Float32(C.SENT_LO):
                                GMIN, GMAX = C.gather_hint(
                                    x_addr, p_addr, k, n, tidx, s_wmn, s_wmx, blk=BLK, kpt=1
                                )  # 2 barriers inside
                            floorhit = cutlass.Int32(1)
                            if T > GMIN:
                                floorhit = cutlass.Int32(0)
                            if floorhit != cutlass.Int32(0):
                                running = cutlass.Int32(0)
                            else:
                                T = GMIN
                att = att + cutlass.Int32(1)

            # ============ classification ============
            whole = cutlass.Int32(0)
            if valid != cutlass.Int32(0):
                if need >= m:
                    whole = cutlass.Int32(1)
            lim1 = above
            if whole != cutlass.Int32(0):
                lim1 = above + m
            degen = cutlass.Int32(0)
            if valid == cutlass.Int32(0):
                degen = cutlass.Int32(1)
            if m > CMP:
                degen = cutlass.Int32(1)
            mc = cutlass.Int32(0)
            if degen == cutlass.Int32(0):
                mc = m
            # crossing candidates land in RANK 0's ck64c via DSMEM
            rk64 = C._mapa_shared_cluster_addr(ck_addr, cutlass.Int32(0))

            if degen == cutlass.Int32(0):
                if complete != cutlass.Int32(0):
                    # ---- P5 emit from staged cbuf ----
                    i = tidx
                    while i < listN:
                        pk64 = s_cbuf2[i]
                        vx = cutlass.Int32(cutlass.Uint32(pk64 & cutlass.Uint64(0xFFFFFFFF)))
                        idv = cutlass.Int32(pk64 >> cutlass.Uint64(32))
                        xv = C.f32_of_i32(vx)
                        self._p5_emit(
                            xv, idv, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                        )
                        i = i + cutlass.Int32(BLK)
                else:
                    # ---- exactness re-sweep: OWNED CHUNKS + rank-0 true tail ----
                    g = rank + cutlass.Int32(0)
                    while g < nCh:
                        lo2 = (g * cutlass.Int32(STEPC)) << cutlass.Int32(2)
                        e4 = (g + cutlass.Int32(1)) * cutlass.Int32(STEPC)
                        if e4 > n4:
                            e4 = n4
                        hi2 = e4 << cutlass.Int32(2)
                        i = lo2 + tidx
                        while i < hi2:
                            x = C.ldg_f32(x_addr, i)
                            if x >= TF:
                                self._p5_emit(
                                    x, i, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                                )
                            i = i + cutlass.Int32(BLK)
                        g = g + cutlass.Int32(CS)
                    t2 = tidx
                    while t2 < tailn:
                        ii = tail0 + t2
                        x = C.ldg_f32(x_addr, ii)
                        if x >= TF:
                            self._p5_emit(
                                x, ii, TF, SC, B, above, lim1, whole, CMP, s_mrg, out_row, rk64
                            )
                        t2 = t2 + cutlass.Int32(BLK)

            # ============ EXIT RENDEZVOUS ============
            # all DSMEM traffic retired; the ONLY exit rendezvous. rank!=0
            # falls through to the kernel end (post-barrier asymmetric exit);
            # NO later cluster barrier.
            C._cluster_sync_aligned()  # ==== clus.sync (exit) ====

            if rank == cutlass.Int32(0):
                if degen == cutlass.Int32(0):
                    if whole == cutlass.Int32(0):
                        # ---- P6 rank-0 refine ----
                        if mc <= cutlass.Int32(QUADC_CLUS__clus):  # O(mc^2)
                            mc2 = mc & cutlass.Int32(~1)
                            i = tidx
                            while i < mc:
                                # re-assert Uint64 at every unsigned compare
                                # in/after dynamic loops.
                                u64v = s_ck64[i]
                                r_ = cutlass.Int32(0)
                                jq = cutlass.Int32(0)
                                while jq < mc2:  # ulonglong2 16B reads
                                    vlo, vhi = C._lds_v2_u64(ck_addr + jq * cutlass.Int32(8))
                                    r_ = (
                                        r_
                                        + cutlass.Int32(vlo > cutlass.Uint64(u64v))
                                        + cutlass.Int32(vhi > cutlass.Uint64(u64v))
                                    )
                                    jq = jq + cutlass.Int32(2)
                                if mc2 < mc:  # odd tail
                                    r_ = r_ + cutlass.Int32(
                                        cutlass.Uint64(s_ck64[mc2]) > cutlass.Uint64(u64v)
                                    )
                                if r_ < need:
                                    out_row[above + r_] = cutlass.Int32(
                                        cutlass.Uint32(
                                            cutlass.Uint64(u64v) & cutlass.Uint64(0xFFFFFFFF)
                                        )
                                    )
                                i = i + cutlass.Int32(BLK)
                        else:
                            # key-space narrowing over ck64c
                            if tidx == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            if tidx < cutlass.Int32(NBS):  # cleared ONCE
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (narrowing init) ----
                            i = tidx
                            while i < mc:
                                kk = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                C.atomic_min_cta(s_kmm.iterator + 0, kk)
                                C.atomic_max_cta(s_kmm.iterator + 1, kk)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (key range) ----
                            rlo = s_kmm[0]
                            rhi = s_kmm[1]
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            brk = cutlass.Int32(0)
                            lev = cutlass.Int32(0)
                            while brk == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    brk = cutlass.Int32(1)
                                elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                elif lev >= cutlass.Int32(6):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    brk = cutlass.Int32(1)
                                else:
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2_ = cutlass.Int32(32) - C.clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = b2_ - cutlass.Int32(self.lb)
                                    if sh2 < cutlass.Int32(0):
                                        sh2 = cutlass.Int32(0)
                                    sh2u = cutlass.Uint32(sh2)
                                    i = tidx
                                    while i < mc:  # re-bin
                                        uq = cutlass.Uint32(s_ck64[i] >> cutlass.Uint64(32))
                                        if uq >= cutlass.Uint32(rlo):
                                            if uq <= cutlass.Uint32(rhi):
                                                du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                                if du > cutlass.Uint32(NBS - 1):
                                                    du = cutlass.Uint32(NBS - 1)
                                                C.atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(du),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # ---- barrier (level hist) ----
                                    C.scan_cross0(
                                        s_hist,
                                        needC,
                                        tidx,
                                        s_res,
                                        cutlass.Int32(0),
                                        cutlass.Int32(0),
                                        s_hist,
                                        nb=NBS,
                                        zero=True,
                                    )
                                    cute.arch.barrier()  # ---- barrier (level scan) ----
                                    aboveC = aboveC + s_res[C.RES_ABOVE]
                                    needC = needC - s_res[C.RES_ABOVE]
                                    mm = s_res[C.RES_M]
                                    sB = s_res[C.RES_B]
                                    nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                                    if sB != cutlass.Int32(NBS - 1):
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1)
                                        )
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            if tidx == cutlass.Int32(0):
                                s_scal[1] = cutlass.Int32(0)
                                s_scal[2] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (emit counters) ----
                            it2 = (mc + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                            it = cutlass.Int32(0)
                            while it < it2:  # ballot emit
                                i = it * cutlass.Int32(BLK) + tidx
                                p1 = cutlass.Int32(0)
                                p2 = cutlass.Int32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:
                                    w64 = s_ck64[i]
                                    iu = cutlass.Int64(cutlass.Uint32(w64 >> cutlass.Uint64(32)))
                                    idv = cutlass.Int32(
                                        cutlass.Uint32(w64 & cutlass.Uint64(0xFFFFFFFF))
                                    )
                                    if iu > ethr:
                                        p1 = cutlass.Int32(1)
                                    if iu == ethr:
                                        p2 = cutlass.Int32(1)
                                self._ballot_pair_emit(
                                    p1,
                                    p2,
                                    idv,
                                    above,
                                    aboveC,
                                    above + aboveC,
                                    needC,
                                    out_row,
                                    s_scal,
                                    lane,
                                )
                                it = it + cutlass.Int32(1)
                else:
                    # ---- degen fallback: whole-row key-space narrowing
                    # (per-level clear + scan_cross w/ ws) ----
                    rlo = cutlass.Uint32(0)
                    rhi = cutlass.Uint32(0xFFFFFFFF)
                    above2 = cutlass.Int32(0)
                    need2 = k
                    m2 = n
                    ethr = cutlass.Int64(0)
                    tie_m = cutlass.Int32(1)
                    brk = cutlass.Int32(0)
                    lev = cutlass.Int32(0)
                    while brk == cutlass.Int32(0):  # <=8 levels
                        if need2 == m2:
                            ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                            above2 = above2 + m2
                            need2 = cutlass.Int32(0)
                            tie_m = cutlass.Int32(0)
                            brk = cutlass.Int32(1)
                        elif cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        elif lev >= cutlass.Int32(8):
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            brk = cutlass.Int32(1)
                        else:
                            d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                            b2_ = cutlass.Int32(32) - C.clz_i32(
                                cutlass.Int32(d2 | cutlass.Uint32(1))
                            )
                            sh2 = b2_ - cutlass.Int32(self.lb)
                            if sh2 < cutlass.Int32(0):
                                sh2 = cutlass.Int32(0)
                            sh2u = cutlass.Uint32(sh2)
                            if tidx < cutlass.Int32(NBS):  # per-level clear
                                s_hist[tidx] = cutlass.Int32(0)
                            cute.arch.barrier()  # ---- barrier (level clear) ----
                            i = tidx
                            while i < n:  # whole row
                                uq = C.fkey(C.ldg_f32(x_addr, i))
                                if uq >= cutlass.Uint32(rlo):
                                    if uq <= cutlass.Uint32(rhi):
                                        du = (uq - cutlass.Uint32(rlo)) >> sh2u
                                        if du > cutlass.Uint32(NBS - 1):
                                            du = cutlass.Uint32(NBS - 1)
                                        C.atomic_add_cta(
                                            s_hist.iterator + cutlass.Int32(du), cutlass.Int32(1)
                                        )
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # ---- barrier (level hist) ----
                            # block-parallel scan (ONE internal barrier; only
                            # use of ws in this kernel)
                            C.scan_cross(
                                s_hist,
                                s_ws,
                                need2,
                                tidx,
                                s_res,
                                cutlass.Int32(0),
                                blk=BLK,
                                nb=NBS,
                                two=False,
                            )
                            cute.arch.barrier()  # ---- barrier (level scan) ----
                            above2 = above2 + s_res[C.RES_ABOVE]
                            need2 = need2 - s_res[C.RES_ABOVE]
                            m2 = s_res[C.RES_M]
                            sB = s_res[C.RES_B]
                            nlo = cutlass.Uint32(rlo) + (cutlass.Uint32(sB) << sh2u)
                            if sB != cutlass.Int32(NBS - 1):
                                rhi = nlo + ((cutlass.Uint32(1) << sh2u) - cutlass.Uint32(1))
                            rlo = nlo
                            lev = lev + cutlass.Int32(1)
                    if tidx == cutlass.Int32(0):
                        s_scal[1] = cutlass.Int32(0)
                        s_scal[2] = cutlass.Int32(0)
                    cute.arch.barrier()  # ---- barrier (emit counters) ----
                    nA = k
                    nT = cutlass.Int32(0)
                    if tie_m != cutlass.Int32(0):
                        nA = above2
                        nT = need2
                    it2 = (n + cutlass.Int32(BLK - 1)) // cutlass.Int32(BLK)
                    it = cutlass.Int32(0)
                    while it < it2:
                        i = it * cutlass.Int32(BLK) + tidx
                        p1 = cutlass.Int32(0)
                        p2 = cutlass.Int32(0)
                        if i < n:
                            uq = C.fkey(C.ldg_f32(x_addr, i))
                            iu = cutlass.Int64(uq)
                            if iu > ethr:
                                p1 = cutlass.Int32(1)
                            if tie_m != cutlass.Int32(0):
                                if iu == ethr:
                                    p2 = cutlass.Int32(1)
                        self._ballot_pair_emit(
                            p1, p2, i, cutlass.Int32(0), nA, nA, nT, out_row, s_scal, lane
                        )
                        it = it + cutlass.Int32(1)

    # ------------------------------------------------------------------
    # host launcher: grid dim3(CS, b) + cluster (CS,1,1);
    # min_blocks_per_mp=1 == __launch_bounds__(1024, 1) 64-register wall.
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        npad: cutlass.Int32,
        k: cutlass.Int32,
        SCAP: cutlass.Int32,
        CMP: cutlass.Int32,
        SMP: cutlass.Int32,
        TGT: cutlass.Int32,
        Q: cutlass.Int32,
        SS2: cutlass.Int32,
        TGT2: cutlass.Int32,
        stream,
    ):
        # varlen grid rows = out.shape[0] (logits row count == out row count in
        # both modes); bigf = the route() `big` occupancy flag, a pure function
        # of (rows, CS) so it is launch-computed, not an ABI scalar.
        b = out.shape[0]
        bigf = cutlass.Int32(0)
        if b * cutlass.Int32(self.cs) <= cutlass.Int32(148):
            bigf = cutlass.Int32(1)
        self.kern(
            logits, pre_idx, kv_lens, out, n, npad, k, SCAP, CMP, SMP, TGT, Q, SS2, TGT2, bigf
        ).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.minb,
        )


# ---------------------------------------------------------------------------
# compile cache + torch-facing entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__clus = {}


def get_compiled__clus(
    tpl,
    scap: int = 8192,
    cmp_: int = 2048,
    options_extra: str = "",
    varlen: bool = False,
    next_n: int = 1,
    cr_shift: int = 0,
):
    """Compile (or fetch) the gvr_clus variant for constexpr tuple
    tpl = (BLK, U, MINB, NBS, CS); scap/cmp are smem-extent keys (every
    reachable route has 8192/2048 — asserted by run__clus())."""
    key = (
        tuple(tpl),
        scap,
        cmp_,
        options_extra,
        bool(varlen),
        int(next_n),
        int(cr_shift),
        C._compile_arch_token(),
    )
    hit = _COMPILE_CACHE__clus.get(key)
    if hit is not None:
        return hit
    blk, u, minb, nbs, cs = tpl
    kern = GvrClusKernel(
        blk, u, minb, nbs, cs, scap=scap, cmp_=cmp_, varlen=varlen, next_n=next_n, cr_shift=cr_shift
    )
    r0, c0 = cute.sym_int(), cute.sym_int()
    r1, c1 = cute.sym_int(), cute.sym_int()
    r2, c2 = cute.sym_int(), cute.sym_int()
    logits_fake = _crt.make_fake_compact_tensor(
        cutlass.Float32, (r0, c0), stride_order=(1, 0), assumed_align=16
    )
    pre_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r1, c1), stride_order=(1, 0), assumed_align=16
    )
    out_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (r2, c2), stride_order=(1, 0), assumed_align=16
    )
    v0 = cute.sym_int()
    kv_fake = _crt.make_fake_compact_tensor(
        cutlass.Int32, (v0,), stride_order=(0,), assumed_align=4
    )
    fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)

    def _compile_fn():
        return cute.compile(
            kern,
            logits_fake,
            pre_fake,
            kv_fake,
            out_fake,
            *([cutlass.Int32(0)] * 10),
            stream=fake_stream,
            options=(_base_compile_opts() + " " + options_extra).strip(),
        )

    name = (
        "clus_" + "_".join(str(x) for x in tuple(tpl))
        + f"_scap{int(scap)}_cmp{int(cmp_)}_vl{int(bool(varlen))}_nn{int(next_n)}_cs{int(cr_shift)}"
    )
    if options_extra:
        name += "_opt_" + options_extra
    compiled = C._persist(name, _compile_fn)
    _COMPILE_CACHE__clus[key] = compiled
    return compiled


def run__clus(logits, pre_idx, n: int, out):
    """torch-facing single-call entry: routes (b, n, k) through ct_dispatch,
    asserts the shape lands on gvr_clus, launches the matching variant.
    gvr_clus takes NO workspace."""
    import torch  # debug-entry only: module stays torch-free at import

    try:
        from . import gvr2_topk_host as ct_dispatch
    except ImportError:
        import gvr2_topk_host as ct_dispatch
    b, npad = logits.shape
    k = pre_idx.shape[1]
    r = ct_dispatch.route(b, int(n), npad, k)
    assert r["kernel"] == "clus", f"shape routes to {r['kernel']}, not gvr_clus"
    rt = r["rt"]
    kobj = GvrClusKernel(*r["tpl"], scap=rt["SCAP"], cmp_=rt["CMP"])
    assert r["smem"] == kobj.dyn_bytes, (r["smem"], kobj.dyn_bytes)
    fn = get_compiled__clus(tuple(r["tpl"]), scap=rt["SCAP"], cmp_=rt["CMP"])
    dkv = torch.zeros(1, dtype=torch.int32, device=logits.device)  # dead varlen slot
    fn(
        logits,
        pre_idx,
        dkv,
        out,
        rt["n"],
        rt["npad"],
        rt["k"],
        rt["SCAP"],
        rt["CMP"],
        rt["SMP"],
        rt["TGT"],
        rt["Q"],
        rt["SS2"],
        rt["TGT2"],
    )
    return r


def run_manual(logits, pre_idx, n: int, out, tpl, rt):
    """Manual-lattice entry for route()-unreachable (U, CS) members: launches
    tpl with caller-supplied runtime scalars (must be route()-consistent for
    the same CS; U only changes the chunk geometry)."""
    import torch  # debug-entry only: module stays torch-free at import

    fn = get_compiled__clus(tuple(tpl), scap=rt["SCAP"], cmp_=rt["CMP"])
    dkv = torch.zeros(1, dtype=torch.int32, device=logits.device)  # dead varlen slot
    fn(
        logits,
        pre_idx,
        dkv,
        out,
        rt["n"],
        rt["npad"],
        rt["k"],
        rt["SCAP"],
        rt["CMP"],
        rt["SMP"],
        rt["TGT"],
        rt["Q"],
        rt["SS2"],
        rt["TGT2"],
    )


# ===========================================================================
# ==== family: regclus =========================================
# ===========================================================================
"""gvr_reg_clus — CLUSTERED register-resident GVR: the register algorithm
(T = GMIN directly, one float-space histogram, one register sweep) run
across a cluster of CS CTAs; per-CTA instruction stream intentionally
identical to the single-CTA reg path plus two hardware cluster barriers and
CS DSMEM reads per bin. Signedness rule applied at every unsigned
compare/shift in/after dynamic loops.

Template knobs (CUDA `gvr_reg_clus<BLK,VPT,CS>`, all instantiations
BLK=BLKC=1024): ctor args of :class:`GvrRegClusKernel`. Runtime args mirror
the CUDA `(n, npad, k)` — npad/k come from tensor shapes, so only `n` crosses
the ABI. Launch: grid=(CS, b), cluster=(CS,1,1), block=1024, dynamic smem
45,056 B (+512 B static-mirror prelude), `__launch_bounds__(1024,1)` ==
min_blocks_per_mp=1 -> 64-register wall.

Shared-memory map (single dynamic window, word offsets; the CUDA static
__shared__ block folded into the first 512 B — byte-identical layout in every
CTA, a mapa/DSMEM requirement):

    [0..5]        s_res   (shared slot map RES_B/M/ABOVE/TOT/B2/B3)
    [6..7]        s_cnt   (s_o1, s_o2)
    [8..9]        s_kmm   (s_kmin, s_kmax — Uint32)
    [16..16+32)   ws      (scan_cross_w workspace)
    [48..48+32)   wmn     (Uint32 warp min partials)
    [80..80+32)   wmx     (Uint32 warp max partials)
    [128..1152)   hist    (this CTA's raw counts)
    [1152..2176)  mrg     (cluster totals -> per-CTA global write cursors)
    [2176..3200)  hoff    (this CTA's rank-exclusive bin offset)
    [3200..7296)  ck      (crossing keys, Uint32, CMPC=4096 slots)
    [7296..11392) ci      (crossing indices, Int32, CMPC slots)

Launch smem = 45,568 B (compile-time constant -> plain int at .launch();
MINB==1 so the _build_kernel_attrs carveout path is not taken and the reg
family's _no_carveout workaround is unnecessary here).
"""


# ---- constants --------------------------------------------------------------
NB__regclus = 1024  # histogram bins; == BLKC here
LNB = 10  # log2(NB__regclus) — reg_clus narrowing shift
QUADC__regclus = 96  # O(mc^2) rank gate
CMPC = 4096  # crossing slots PER CTA (pow2)
LCMPC = 12  # log2(CMPC)
BLKC = 1024  # CTA size

STATIC_WORDS__regclus = 128  # DSL smem prelude (static-__shared__ mirror)
STATIC_BYTES__regclus = STATIC_WORDS__regclus * 4
DYN_SMEM_BYTES = (3 * NB__regclus + 2 * CMPC) * 4  # 45,056
SMEM_BYTES = STATIC_BYTES__regclus + DYN_SMEM_BYTES  # 45,568

# word offsets into the shared window (module docstring)
W_HIST = STATIC_WORDS__regclus
W_MRG = STATIC_WORDS__regclus + NB__regclus
W_HOFF = STATIC_WORDS__regclus + 2 * NB__regclus
W_CK = STATIC_WORDS__regclus + 3 * NB__regclus
W_CI = STATIC_WORDS__regclus + 3 * NB__regclus + CMPC

_NEG_INF__regclus = float("-inf")
_POS_INF__regclus = float("inf")


# ---------------------------------------------------------------------------
# module-local FP/util spellings (same forms as the reg family)
# ---------------------------------------------------------------------------
@dsl_user_op
def _fmaf__regclus(a, b, c, *, loc=None, ip=None):
    """CUDA fmaf: single fma.rn.f32 (classify == emit bit-exact)."""
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


@cute.jit
def _umin_u32__regclus(a, b):
    """unsigned min(a, b) — CUDA min() on the bin clamp (IMNMX)."""
    r = a
    if b < a:
        r = b
    return r


@cute.jit
def _fabsf__regclus(x):
    """|x| via sign-bit clear (exact, matches fabsf)."""
    return f32_of_u32(u32_of_f32(x) & cutlass.Uint32(0x7FFFFFFF))


def _smem_view__regclus(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(
        dtype, sbase + cutlass.Int32(word_off * 4), cute.AddressSpace.smem, assumed_align=align
    )
    return cute.make_tensor(p, cute.make_layout((length,)))


def _val__regclus(frags, s: int):
    """val[s] accessor over the float4[VPT] register batch (constexpr s)."""
    return frags[s // 4][s % 4]


class GvrRegClusKernel:
    """gvr_reg_clus<BLK, VPT, CS>."""

    def __init__(
        self,
        blk: int,
        vpt: int,
        cs: int,
        pdl: bool = False,
        varlen: bool = False,
        next_n: int = 1,
        cr_shift: int = 0,
    ):
        assert blk == BLKC, "all instantiations BLK=BLKC=1024"
        assert vpt in (1, 2, 4) and cs in (2, 4, 8)
        self.blk = blk
        self.vpt = vpt
        self.cs = cs
        self.pdl = bool(pdl)
        # per-row varlen mode (production heuristicTopKDecode contract, same
        # semantics as GvrMainKernel): n is re-derived PER ROW in-kernel from
        # a device kv_lens tensor; the scalar n launch arg becomes the
        # envelope clamp bound. next_n / cr_shift are compile-time.
        self.varlen = bool(varlen)
        self.next_n = int(next_n)
        self.cr_shift = int(cr_shift)
        if self.varlen:
            assert self.next_n >= 1 and self.cr_shift in (0, 2)
        self.S = vpt * 4
        self.span = blk * vpt  # float4 per CTA

    # ------------------------------------------------------------------
    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
    ):
        BLK = cutlass.const_expr(self.blk)
        VPT = cutlass.const_expr(self.vpt)
        CS = cutlass.const_expr(self.cs)
        S = cutlass.const_expr(self.S)
        NW = cutlass.const_expr(self.blk // 32)

        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()  # knob default off

        tid, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()  # bx=rank
        lane = tid & cutlass.Int32(31)

        # ============ per-row varlen prologue — shared contract lives in ======
        # GvrMainKernel's prologue (per-row n from kv_lens, clamped to the
        # envelope arg; the launcher admits this family only when the envelope
        # fits its capacity window). Pure functions of `row` keep the body
        # guard and cluster barriers cluster-uniform. Short rows (n <= k):
        # rank 0 emits identity + (-1) and the body is SKIPped — a zero-work
        # pass would reach the crossing-overflow emitter and poison the output.
        short = cutlass.Int32(0)
        prow = row
        if cutlass.const_expr(self.varlen):
            kq = cutlass.Int32(pre_idx.shape[1])
            req = row // cutlass.Int32(self.next_n)
            rr = row % cutlass.Int32(self.next_n)
            prow = req
            kvl = kv_lens[req]
            nv = (kvl - cutlass.Int32(self.next_n) + rr + cutlass.Int32(1)) >> cutlass.Int32(
                self.cr_shift
            )
            if nv < cutlass.Int32(0):
                nv = cutlass.Int32(0)
            if nv > n:
                nv = n
            if nv <= kq:
                short = cutlass.Int32(1)
            if short == cutlass.Int32(0):
                n = nv
            if short != cutlass.Int32(0):
                if rank == cutlass.Int32(0):
                    if tid < kq:
                        ov = cutlass.Int32(-1)
                        if tid < nv:
                            ov = tid
                        out[row, tid] = ov

        # ------------------------------------------------------------------
        # Predeclarations (DSL AST rule: every scalar (re)assigned under a
        # dynamic if/while must pre-exist with a stable type; constant inits
        # are dead-coded).
        # ------------------------------------------------------------------
        i = cutlass.Int32(0)
        j = cutlass.Int32(0)
        rnk = cutlass.Int32(0)
        tinc = cutlass.Int32(0)
        mc = cutlass.Int32(0)
        p = cutlass.Int32(0)
        q2i = cutlass.Int32(0)
        idx = cutlass.Int32(0)
        lim1 = cutlass.Int32(0)
        aboveC = cutlass.Int32(0)
        needC = cutlass.Int32(0)
        mm = cutlass.Int32(0)
        lev = cutlass.Int32(0)
        done = cutlass.Int32(0)
        b2w = cutlass.Int32(0)
        sh2 = cutlass.Int32(0)
        b_lv = cutlass.Int32(0)
        it = cutlass.Int32(0)
        it2 = cutlass.Int32(0)
        idv = cutlass.Int32(0)
        q1f = cutlass.Int32(0)
        q2f = cutlass.Int32(0)
        n1 = cutlass.Int32(0)
        n2 = cutlass.Int32(0)
        b1 = cutlass.Int32(0)
        b2 = cutlass.Int32(0)
        p1e = cutlass.Int32(0)
        p2e = cutlass.Int32(0)
        lml = cutlass.Int32(0)
        nA = cutlass.Int32(0)
        nT = cutlass.Int32(0)
        tie_m = cutlass.Int32(0)
        pv0 = cutlass.Int32(-1)
        okc = cutlass.Int32(0)
        whole = cutlass.Int32(0)
        degen = cutlass.Int32(0)
        pre_a = cutlass.Int32(0)
        tot_a = cutlass.Int32(0)
        uk = cutlass.Uint32(0)
        uq = cutlass.Uint32(0)
        vq = cutlass.Uint32(0)
        kv = cutlass.Uint32(0)
        rlo = cutlass.Uint32(0)
        rhi = cutlass.Uint32(0)
        d2 = cutlass.Uint32(0)
        unar = cutlass.Uint32(0)
        bnn = cutlass.Uint32(0)
        nlo = cutlass.Uint32(0)
        uke = cutlass.Uint32(0)
        bn = cutlass.Uint32(0)
        ethr = cutlass.Int64(0)
        tval = cutlass.Float32(_NEG_INF__regclus)
        LOQ = cutlass.Float32(0.0)
        qv = cutlass.Float32(0.0)

        if short == cutlass.Int32(0):
            npad = cutlass.Int32(logits.shape[1])  # noqa: F841
            k = cutlass.Int32(pre_idx.shape[1])
            out_row = out[row, None]
            x_addr = logits[row, None].iterator.toint()  # Int64 gmem byte base
            p_addr = pre_idx[prow, None].iterator.toint()  # request-level under varlen

            # ---- shared-memory window (map in module docstring) ----
            sptr = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            sbase = sptr.toint()  # Int32 shared addr

            s_res = _smem_view__regclus(cutlass.Int32, sbase, 0, 6)
            s_cnt = _smem_view__regclus(cutlass.Int32, sbase, 6, 2)  # [0]=s_o1 [1]=s_o2
            s_kmm = _smem_view__regclus(cutlass.Uint32, sbase, 8, 2)  # [0]=s_kmin [1]=s_kmax
            s_ws = _smem_view__regclus(cutlass.Int32, sbase, 16, 32)
            s_wmn = _smem_view__regclus(cutlass.Uint32, sbase, 48, 32)
            s_wmx = _smem_view__regclus(cutlass.Uint32, sbase, 80, 32)
            s_hist = _smem_view__regclus(cutlass.Int32, sbase, W_HIST, NB__regclus)
            s_mrg = _smem_view__regclus(cutlass.Int32, sbase, W_MRG, NB__regclus)
            s_hoff = _smem_view__regclus(cutlass.Int32, sbase, W_HOFF, NB__regclus)
            s_ck = _smem_view__regclus(cutlass.Uint32, sbase, W_CK, CMPC)
            s_ci = _smem_view__regclus(cutlass.Int32, sbase, W_CI, CMPC, align=4)
            # raw byte bases for DSMEM (mapa) addressing
            hist_addr = sbase + cutlass.Int32(W_HIST * 4)
            ck_addr = sbase + cutlass.Int32(W_CK * 4)
            ci_addr = sbase + cutlass.Int32(W_CI * 4)

            n4 = n >> cutlass.Int32(2)
            ntail = n - (n4 << cutlass.Int32(2))
            base4 = rank * cutlass.Int32(self.span)
            tix = (n4 << cutlass.Int32(2)) + tid  # CUDA `tidx`

            # ---- P0: redundant hint gather, EVERY CTA (k<=BLK by dispatch
            # gate). One coalesced word per thread, NO cluster barrier —
            # GMIN/GMAX identical everywhere by construction.
            if tid < k:
                pv0 = ld_g_i32(p_addr, tid)

            # ---- P1: row load — predicated flat float4[VPT] batch (the CUDA
            # has NO exact-fit peel here, guard is per-load). Issue all loads
            # first, then -INFINITY-fill missed slots.
            atom128 = g2r_atom_f32(128, invariant=True)
            frags = [cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(VPT)]
            for u in cutlass.range_constexpr(VPT):
                i = base4 + tid + cutlass.Int32(u * self.blk)
                if i < n4:
                    ld_g_f32x4(atom128, x_addr, i, frags[u])
            for u in cutlass.range_constexpr(VPT):
                i = base4 + tid + cutlass.Int32(u * self.blk)
                if i >= n4:  # -INFINITY fill
                    for z in cutlass.range_constexpr(4):
                        frags[u][z] = cutlass.Float32(_NEG_INF__regclus)
            # tail element: rank 0 only
            if rank == cutlass.Int32(0):
                if tid < ntail:
                    tval = ldg_f32(x_addr, tix)

            # ---- P2: init. NB__regclus == BLK -> single-pass hist clear.
            if tid == cutlass.Int32(0):
                s_cnt[0] = cutlass.Int32(0)
                s_cnt[1] = cutlass.Int32(0)
            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)

            # ---- P3: GMIN/GMAX from the hint, ONE barrier fold.
            lmin = cutlass.Uint32(0xFFFFFFFF)
            lmax = cutlass.Uint32(0)
            if cutlass.Uint32(pv0) < cutlass.Uint32(n):
                uk = fkey(ldg_f32(x_addr, pv0))  # __ldg(X+pv0)
                lmin = uk
                lmax = uk
            lmin = warp_min_u32(lmin)
            lmax = warp_max_u32(lmax)
            if lane == cutlass.Int32(0):
                s_wmn[tid >> cutlass.Int32(5)] = lmin
                s_wmx[tid >> cutlass.Int32(5)] = lmax
            cute.arch.barrier()  # warp partials published
            a = cutlass.Uint32(0xFFFFFFFF)
            c = cutlass.Uint32(0)
            if lane < cutlass.Int32(NW):
                a = cutlass.Uint32(s_wmn[lane])
                c = cutlass.Uint32(s_wmx[lane])
            lmin = warp_min_u32(a)
            lmax = warp_max_u32(c)
            Tv = invkey(lmin)
            GMAX = invkey(lmax)

            # ---- collapse guard, NaN-safe. Reject infinite width as well:
            # otherwise SC becomes zero and can collapse +inf into a finite
            # histogram bin.
            okc = cutlass.Int32(0)
            if Tv < GMAX:
                w_ = GMAX - Tv
                if w_ > cutlass.Float32(1e-30) and w_ < cutlass.Float32(3.5e38):
                    okc = cutlass.Int32(1)
            if okc == cutlass.Int32(0):
                Tv = cutlass.Float32(SENT_LO)
                GMAX = cutlass.Float32(SENT_HI)

            # ---- bin transform constants: branchless trash bin.
            WD = (GMAX - Tv) * cutlass.Float32(1.0 / float(NB__regclus - 2))
            wsel = cutlass.Float32(1e-30)
            if WD > cutlass.Float32(0.0):
                wsel = WD
            # MUFU.RCP spelling (mirrors the reg family's site):
            # wsel >= 1e-30 finite; bucketing is SC-invariant for any SC > 0.
            SC = cute.arch.rcp_approx(wsel)
            CQ0 = cutlass.Float32(1.0) - Tv * SC
            CQ = CQ0 + cutlass.Float32(1e-6) * (_fabsf__regclus(CQ0) + cutlass.Float32(1.0))

            # ---- P4: histogram; tval add UNCONDITIONAL (trash bin swallows
            # -INFINITY via the saturating cvt).
            for s in cutlass.range_constexpr(S):
                qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)
                bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))
            qv = _fmaf__regclus(tval, SC, CQ)
            bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
            atomic_add_cta(s_hist.iterator + cutlass.Int32(bn), cutlass.Int32(1))

            # ---- P5: cluster merge
            _cluster_sync_aligned()  # histograms complete on every rank
            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                i = tid + cutlass.Int32(z * self.blk)
                # CS-unrolled remote u32 loads: batch-issue, then fold
                # (one mapa per (i, r) exactly like the CUDA)
                hvals = []
                for r in cutlass.range_constexpr(CS):
                    ma = _mapa_shared_cluster_addr(
                        hist_addr + (i << cutlass.Int32(2)), cutlass.Int32(r)
                    )
                    hvals.append(_ld_shared_cluster_i32(ma))
                tot_a = cutlass.Int32(0)
                pre_a = cutlass.Int32(0)
                for r in cutlass.range_constexpr(CS):
                    if cutlass.Int32(r) < rank:
                        pre_a = pre_a + hvals[r]  # rank-exclusive
                    tot_a = tot_a + hvals[r]
                s_mrg[i] = tot_a
                s_hoff[i] = pre_a

            # ---- P6: scan
            cute.arch.barrier()  # merge published
            scan_cross_w(s_mrg, s_ws, k, tid, s_res, blk=self.blk, nb=NB__regclus)
            cute.arch.barrier()  # scan published
            above = s_res[RES_ABOVE]
            m = s_res[RES_M]
            Bv = s_res[RES_B]
            need = k - above
            whole = cutlass.Int32(0)
            if need >= m:
                whole = cutlass.Int32(1)
            degen = cutlass.Int32(0)
            if m > cutlass.Int32(CS * CMPC):
                degen = cutlass.Int32(1)
            # The sentinel bracket also has infinite width. Bypass its
            # collapsed histogram and enter the exact whole-row key-space
            # fallback on rank 0, where fkey(+inf) is the maximum key.
            if okc == cutlass.Int32(0):
                whole = cutlass.Int32(0)
                degen = cutlass.Int32(1)
            for z in cutlass.range_constexpr(NB__regclus // self.blk):
                i = tid + cutlass.Int32(z * self.blk)
                s_mrg[i] = s_mrg[i] + s_hoff[i]  # global cursor
            cute.arch.barrier()  # cursors published

            # ---- P7: register sweep emit (!degen)
            if degen == cutlass.Int32(0):
                LOQ = cutlass.Float32(Bv)
                lim1 = above
                if whole == cutlass.Int32(1):
                    lim1 = above + m
                for s in cutlass.range_constexpr(S):
                    qv = _fmaf__regclus(_val__regclus(frags, s), SC, CQ)  # bit-identical
                    if qv >= LOQ:
                        bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                        p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                        idx = (
                            (base4 + tid + cutlass.Int32((s // 4) * self.blk)) << cutlass.Int32(2)
                        ) + cutlass.Int32(s % 4)
                        if p < lim1:
                            out_row[p] = idx
                        else:
                            if whole == cutlass.Int32(0):
                                # crossing overflow -> striped DSMEM slabs; TWO
                                # separate u32 remote stores (NOT packed)
                                q2i = p - above
                                rnk = q2i >> cutlass.Int32(LCMPC)
                                j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                                _st_shared_cluster_i32(
                                    _mapa_shared_cluster_addr(ck_addr + j, rnk),
                                    fkey(_val__regclus(frags, s)),
                                )
                                _st_shared_cluster_i32(
                                    _mapa_shared_cluster_addr(ci_addr + j, rnk), idx
                                )
                # tail element: tval == -INF fails q>=LOQ elsewhere
                qv = _fmaf__regclus(tval, SC, CQ)
                if qv >= LOQ:
                    bn = _umin_u32__regclus(f2u_rz(qv), cutlass.Uint32(NB__regclus - 1))
                    p = atomic_add_cta(s_mrg.iterator + cutlass.Int32(bn), cutlass.Int32(1))
                    if p < lim1:
                        out_row[p] = tix
                    else:
                        if whole == cutlass.Int32(0):
                            q2i = p - above
                            rnk = q2i >> cutlass.Int32(LCMPC)
                            j = (q2i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)
                            _st_shared_cluster_i32(
                                _mapa_shared_cluster_addr(ck_addr + j, rnk), fkey(tval)
                            )
                            _st_shared_cluster_i32(_mapa_shared_cluster_addr(ci_addr + j, rnk), tix)

            # ---- P8: release staging to rank 0
            cute.arch.barrier()
            _cluster_sync_aligned()

            # ---- P9: rank-0 selection
            if rank == cutlass.Int32(0):
                if whole == cutlass.Int32(0):
                    mc = m
                    if degen == cutlass.Int32(1):
                        mc = cutlass.Int32(0)
                    if degen == cutlass.Int32(0):
                        if mc <= cutlass.Int32(QUADC__regclus):
                            # (1) quad-96: all candidates LOCAL (96 < CMPC),
                            # O(mc^2) slot-order tie-broken rank
                            i = tid
                            while i < mc:
                                uq = cutlass.Uint32(s_ck[i])
                                rnk = cutlass.Int32(0)
                                j = cutlass.Int32(0)
                                while j < mc:
                                    vq = cutlass.Uint32(s_ck[j])
                                    tinc = cutlass.Int32(0)
                                    if vq > uq:
                                        tinc = cutlass.Int32(1)
                                    if vq == uq:
                                        if j < i:
                                            tinc = cutlass.Int32(1)
                                    rnk = rnk + tinc
                                    j = j + cutlass.Int32(1)
                                if rnk < need:
                                    out_row[above + rnk] = s_ci[i]
                                i = i + cutlass.Int32(BLK)
                        else:
                            # (2) key-space narrowing over striped DSMEM slabs:
                            # slot = i & (CMPC-1), rank = i >> LCMPC
                            if tid == cutlass.Int32(0):
                                s_kmm[0] = cutlass.Uint32(0xFFFFFFFF)
                                s_kmm[1] = cutlass.Uint32(0)
                            cute.arch.barrier()  # kmm init
                            i = tid
                            while i < mc:
                                kv = cutlass.Uint32(
                                    _ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ck_addr
                                            + ((i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)),
                                            i >> cutlass.Int32(LCMPC),
                                        )
                                    )
                                )
                                atomic_min_cta(s_kmm.iterator, kv)
                                atomic_max_cta(s_kmm.iterator + 1, kv)
                                i = i + cutlass.Int32(BLK)
                            cute.arch.barrier()  # key range published
                            rlo = cutlass.Uint32(s_kmm[0])
                            rhi = cutlass.Uint32(s_kmm[1])
                            ethr = cutlass.Int64(cutlass.Uint32(rlo))
                            aboveC = cutlass.Int32(0)
                            needC = need
                            mm = mc
                            lev = cutlass.Int32(0)
                            done = cutlass.Int32(0)
                            while done == cutlass.Int32(0):  # <=6 levels
                                if needC == mm:
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                    aboveC = aboveC + mm
                                    needC = cutlass.Int32(0)
                                    done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                    if lev >= cutlass.Int32(6):
                                        ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                        done = cutlass.Int32(1)
                                if done == cutlass.Int32(0):
                                    d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                    b2w = cutlass.Int32(32) - clz_i32(
                                        cutlass.Int32(d2 | cutlass.Uint32(1))
                                    )
                                    sh2 = cutlass.Int32(0)
                                    if b2w > cutlass.Int32(LNB):
                                        sh2 = b2w - cutlass.Int32(LNB)
                                    for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                        s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                    cute.arch.barrier()  # level clear
                                    i = tid
                                    while i < mc:
                                        unar = cutlass.Uint32(
                                            _ld_shared_cluster_i32(
                                                _mapa_shared_cluster_addr(
                                                    ck_addr
                                                    + (
                                                        (i & cutlass.Int32(CMPC - 1))
                                                        << cutlass.Int32(2)
                                                    ),
                                                    i >> cutlass.Int32(LCMPC),
                                                )
                                            )
                                        )
                                        if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                            if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                                bnn = (
                                                    cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                                ) >> cutlass.Uint32(sh2)
                                                bnn = _umin_u32__regclus(
                                                    bnn, cutlass.Uint32(NB__regclus - 1)
                                                )
                                                atomic_add_cta(
                                                    s_hist.iterator + cutlass.Int32(bnn),
                                                    cutlass.Int32(1),
                                                )
                                        i = i + cutlass.Int32(BLK)
                                    cute.arch.barrier()  # level hist
                                    find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                    cute.arch.barrier()  # level scan
                                    aboveC = aboveC + s_res[RES_ABOVE]
                                    needC = needC - s_res[RES_ABOVE]
                                    mm = s_res[RES_M]
                                    b_lv = s_res[RES_B]
                                    nlo = cutlass.Uint32(rlo) + (
                                        cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                    )
                                    if b_lv != cutlass.Int32(NB__regclus - 1):
                                        rhi = nlo + (
                                            (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                            - cutlass.Uint32(1)
                                        )
                                    rlo = nlo
                                    lev = lev + cutlass.Int32(1)
                            cute.arch.barrier()  # narrowing done
                            # two-predicate ballot emit over the striped slabs
                            lml = cutlass.Int32(cute.arch.lanemask_lt())
                            it2 = (mc + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                            it = cutlass.Int32(0)
                            while it < it2:
                                i = it * cutlass.Int32(BLK) + tid
                                uke = cutlass.Uint32(0)
                                idv = cutlass.Int32(0)
                                if i < mc:  # predicated remote
                                    uke = cutlass.Uint32(
                                        _ld_shared_cluster_i32(
                                            _mapa_shared_cluster_addr(
                                                ck_addr
                                                + (
                                                    (i & cutlass.Int32(CMPC - 1))
                                                    << cutlass.Int32(2)
                                                ),
                                                i >> cutlass.Int32(LCMPC),
                                            )
                                        )
                                    )
                                    idv = _ld_shared_cluster_i32(
                                        _mapa_shared_cluster_addr(
                                            ci_addr
                                            + ((i & cutlass.Int32(CMPC - 1)) << cutlass.Int32(2)),
                                            i >> cutlass.Int32(LCMPC),
                                        )
                                    )
                                q1f = cutlass.Int32(0)
                                q2f = cutlass.Int32(0)
                                if i < mc:
                                    if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                        q1f = cutlass.Int32(1)
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                                n1 = ballot(q1f == cutlass.Int32(1))
                                n2 = ballot(q2f == cutlass.Int32(1))
                                b1 = cutlass.Int32(0)
                                b2 = cutlass.Int32(0)
                                if lane == cutlass.Int32(0):
                                    if n1 != cutlass.Int32(0):
                                        b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                    if n2 != cutlass.Int32(0):
                                        b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                                b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                                b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                                p1e = b1 + popc(n1 & lml)
                                p2e = b2 + popc(n2 & lml)
                                if q1f == cutlass.Int32(1):
                                    if p1e < aboveC:
                                        out_row[above + p1e] = idv
                                if q2f == cutlass.Int32(1):
                                    if p2e < needC:
                                        out_row[above + aboveC + p2e] = idv
                                it = it + cutlass.Int32(1)
                    else:
                        # (3) degen safety net: crossing bin larger than the
                        # whole cluster buffer -> exact whole-row key-space
                        # narrowing by rank 0 alone, <=8 levels.
                        rlo = cutlass.Uint32(0)
                        rhi = cutlass.Uint32(0xFFFFFFFF)
                        aboveC = cutlass.Int32(0)  # above2
                        needC = k  # need2
                        mm = n  # m2
                        ethr = cutlass.Int64(0)
                        tie_m = cutlass.Int32(1)
                        lev = cutlass.Int32(0)
                        done = cutlass.Int32(0)
                        while done == cutlass.Int32(0):
                            if needC == mm:
                                ethr = cutlass.Int64(cutlass.Uint32(rlo)) - cutlass.Int64(1)
                                aboveC = aboveC + mm
                                needC = cutlass.Int32(0)
                                tie_m = cutlass.Int32(0)
                                done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                if cutlass.Uint32(rlo) >= cutlass.Uint32(rhi):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                                if lev >= cutlass.Int32(8):
                                    ethr = cutlass.Int64(cutlass.Uint32(rlo))
                                    done = cutlass.Int32(1)
                            if done == cutlass.Int32(0):
                                d2 = cutlass.Uint32(rhi) - cutlass.Uint32(rlo)
                                b2w = cutlass.Int32(32) - clz_i32(
                                    cutlass.Int32(d2 | cutlass.Uint32(1))
                                )
                                sh2 = cutlass.Int32(0)
                                if b2w > cutlass.Int32(LNB):
                                    sh2 = b2w - cutlass.Int32(LNB)
                                for z in cutlass.range_constexpr(NB__regclus // self.blk):
                                    s_hist[tid + cutlass.Int32(z * self.blk)] = cutlass.Int32(0)
                                cute.arch.barrier()  # level clear
                                i = tid
                                while i < n:  # whole-row bin
                                    unar = fkey(ldg_f32(x_addr, i))
                                    if cutlass.Uint32(unar) >= cutlass.Uint32(rlo):
                                        if cutlass.Uint32(unar) <= cutlass.Uint32(rhi):
                                            bnn = (
                                                cutlass.Uint32(unar) - cutlass.Uint32(rlo)
                                            ) >> cutlass.Uint32(sh2)
                                            bnn = _umin_u32__regclus(
                                                bnn, cutlass.Uint32(NB__regclus - 1)
                                            )
                                            atomic_add_cta(
                                                s_hist.iterator + cutlass.Int32(bnn),
                                                cutlass.Int32(1),
                                            )
                                    i = i + cutlass.Int32(BLK)
                                cute.arch.barrier()  # level hist
                                find_cross(s_hist, needC, tid, s_res, nb=NB__regclus)
                                cute.arch.barrier()  # level scan
                                aboveC = aboveC + s_res[RES_ABOVE]
                                needC = needC - s_res[RES_ABOVE]
                                mm = s_res[RES_M]
                                b_lv = s_res[RES_B]
                                nlo = cutlass.Uint32(rlo) + (
                                    cutlass.Uint32(b_lv) << cutlass.Uint32(sh2)
                                )
                                if b_lv != cutlass.Int32(NB__regclus - 1):
                                    rhi = nlo + (
                                        (cutlass.Uint32(1) << cutlass.Uint32(sh2))
                                        - cutlass.Uint32(1)
                                    )
                                rlo = nlo
                                lev = lev + cutlass.Int32(1)
                        cute.arch.barrier()  # narrowing done
                        nA = k  # tie_m ? above2 : k
                        if tie_m == cutlass.Int32(1):
                            nA = aboveC
                        nT = cutlass.Int32(0)
                        if tie_m == cutlass.Int32(1):
                            nT = needC
                        lml = cutlass.Int32(cute.arch.lanemask_lt())
                        it2 = (n + cutlass.Int32(self.blk - 1)) // cutlass.Int32(self.blk)
                        it = cutlass.Int32(0)
                        while it < it2:
                            i = it * cutlass.Int32(BLK) + tid
                            uke = cutlass.Uint32(0)
                            if i < n:
                                uke = fkey(ldg_f32(x_addr, i))
                            q1f = cutlass.Int32(0)
                            q2f = cutlass.Int32(0)
                            if i < n:
                                if cutlass.Int64(cutlass.Uint32(uke)) > ethr:
                                    q1f = cutlass.Int32(1)
                                if tie_m == cutlass.Int32(1):
                                    if cutlass.Int64(cutlass.Uint32(uke)) == ethr:
                                        q2f = cutlass.Int32(1)
                            n1 = ballot(q1f == cutlass.Int32(1))
                            n2 = ballot(q2f == cutlass.Int32(1))
                            b1 = cutlass.Int32(0)
                            b2 = cutlass.Int32(0)
                            if lane == cutlass.Int32(0):
                                if n1 != cutlass.Int32(0):
                                    b1 = atomic_add_cta(s_cnt.iterator, popc(n1))
                                if n2 != cutlass.Int32(0):
                                    b2 = atomic_add_cta(s_cnt.iterator + 1, popc(n2))
                            b1 = cute.arch.shuffle_sync(b1, cutlass.Int32(0))
                            b2 = cute.arch.shuffle_sync(b2, cutlass.Int32(0))
                            p1e = b1 + popc(n1 & lml)
                            p2e = b2 + popc(n2 & lml)
                            if q1f == cutlass.Int32(1):
                                if p1e < nA:
                                    out_row[p1e] = i
                            if q2f == cutlass.Int32(1):
                                if p2e < nT:
                                    out_row[nA + p2e] = i
                            it = it + cutlass.Int32(1)

            # ---- P10: FINAL cluster rendezvous — ALL ranks reach it;
            # keeps peers resident until rank 0 has read their ck/ci.
            _cluster_sync_aligned()

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        n: cutlass.Int32,
        stream,
    ):
        b = out.shape[0]
        self.kern(logits, pre_idx, kv_lens, out, n).launch(
            grid=(self.cs, b, 1),
            block=(self.blk, 1, 1),
            cluster=(self.cs, 1, 1),
            stream=stream,
            smem=SMEM_BYTES,
            min_blocks_per_mp=1,
            use_pdl=self.pdl,
        )


# ---------------------------------------------------------------------------
# host wrapper: compile cache + route()-driven entry
# ---------------------------------------------------------------------------
_COMPILE_CACHE__regclus: dict = {}


def get_compiled__regclus(tpl, dump_dir=None, pdl=False, varlen=False, next_n=1, cr_shift=0):
    """Compile (or fetch) the variant for constexpr tuple (BLK, VPT, CS)."""
    key = (tuple(tpl), bool(pdl), bool(varlen), int(next_n), int(cr_shift), C._compile_arch_token())
    compiled = _COMPILE_CACHE__regclus.get(key)
    if compiled is None:
        from cutlass.cute import runtime as _crt

        blk, vpt, cs = tpl
        kernel = GvrRegClusKernel(
            blk, vpt, cs, pdl=pdl, varlen=varlen, next_n=next_n, cr_shift=cr_shift
        )
        nb_, nc_ = cute.sym_int(), cute.sym_int()
        nb2_, nc2_ = cute.sym_int(), cute.sym_int()
        nb3_, nc3_ = cute.sym_int(), cute.sym_int()
        lg_fake = _crt.make_fake_compact_tensor(
            cutlass.Float32, (nb_, nc_), stride_order=(1, 0), assumed_align=16
        )
        pi_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb2_, nc2_), stride_order=(1, 0), assumed_align=16
        )
        out_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (nb3_, nc3_), stride_order=(1, 0), assumed_align=16
        )
        v0_ = cute.sym_int()
        kv_fake = _crt.make_fake_compact_tensor(
            cutlass.Int32, (v0_,), stride_order=(0,), assumed_align=4
        )
        fake_stream = _crt.make_fake_stream(use_tvm_ffi_env_stream=True)
        opts = _base_compile_opts()
        if dump_dir:
            opts += f" --keep-ptx --keep-cubin --dump-dir {dump_dir}"

        def _compile_fn():
            return cute.compile(
                kernel,
                lg_fake,
                pi_fake,
                kv_fake,
                out_fake,
                cutlass.Int32(0),
                stream=fake_stream,
                options=opts,
            )

        if dump_dir:
            # debug dumps: compile in-process, do not persist
            compiled = _compile_fn()
        else:
            name = (
                "regclus_" + "_".join(str(x) for x in tuple(tpl))
                + f"_pdl{int(bool(pdl))}_vl{int(bool(varlen))}_nn{int(next_n)}_cs{int(cr_shift)}"
            )
            compiled = C._persist(name, _compile_fn)
        _COMPILE_CACHE__regclus[key] = compiled
    return compiled


def regclus_topk(logits, pre_idx, n, out, rd=None):
    """torch-facing entry for the clustered register family.

    logits [b, npad] f32, pre_idx [b, k] i32, out [b, >=k] i32, n = valid len.
    rd: optional pre-computed ct_dispatch.route() dict (must be reg_clus).
    """
    if rd is None:
        try:
            from .gvr2_topk_host import route
        except ImportError:
            from gvr2_topk_host import route
        rd = route(logits.shape[0], int(n), logits.shape[1], pre_idx.shape[1])
    assert rd["kernel"] == "reg_clus", rd["kernel"]
    tpl = tuple(rd["tpl"])
    assert pre_idx.shape[1] <= tpl[0], "k <= BLK enforced by dispatch"
    assert rd["smem"] == DYN_SMEM_BYTES
    compiled = get_compiled__regclus(tpl)
    try:
        from .gvr2_topk_host import _dummy_kv
    except ImportError:
        from gvr2_topk_host import _dummy_kv
    # dead varlen ABI slot (upstream passed 4 args to the 5-arg ABI — fixed here)
    kv = _dummy_kv(logits.get_device(), logits.device)
    compiled(logits, pre_idx, kv, out, int(n))
    return out


__all__ = [
    "get_compiled",
    "get_compiled__clus",
    "get_compiled__reg",
    "get_compiled__regclus",
    "run",
    "run__clus",
]
