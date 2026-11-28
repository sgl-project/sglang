"""SM90 small-M BF16 ``x @ w.T`` (fp32 acc) in TileLang, for unquant's BF16 dispatch
(SGLANG_HOPPER_SMALL_M_GEMV) and the Muse Spark fp32 router."""

import contextlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import tilelang
import tilelang.language as T
import torch
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as _E,
)

_PTX = r"""
#include <tl_templates/cuda/intrin.h>
namespace tl { template <> __forceinline__ __device__ bool tl_shuffle_elect<1>() { return true; } }
#define DEVI __device__ __forceinline__
#define SA(p) (unsigned)__cvta_generic_to_shared(p)
DEVI unsigned bf2(float lo, float hi) { unsigned r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
DEVI void push4(const float* a, void* dst, int cta, void* bar) {
  unsigned d = SA(dst) + (threadIdx.x & 127) * 16, b = SA(bar);
  asm volatile("mapa.shared::cluster.u32 %0, %0, %2;\n mapa.shared::cluster.u32 %1, %1, %2;" : "+r"(d), "+r"(b) : "r"(cta));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(d), "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "r"(b) : "memory");
}
template <int S, bool F> DEVI void own4(const float* a, const float* r, int rs, void* o, int jb, int ld, int rem) {
  int t = threadIdx.x & 127, l = t & 31, tk = 8 * jb + (l & 7), ch = 2 * (t >> 5) + ((l >> 3) & 1);
  int e = 8 * (t >> 5) + (l >> 2), k = 8 * jb + 2 * (l & 3);
  float v[4] = {a[0], a[1], a[2], a[3]};
  r += 4 * t;
  for (int s = 0; s < S; ++s) { float4 p = *(const float4*)(r + s * rs); v[0] += p.x; v[1] += p.y; v[2] += p.z; v[3] += p.w; }
  if constexpr (F) {
    float* y = (float*)o;
    if (k < rem) y[k * ld + e] = v[0] + v[2];
    if (k + 1 < rem) y[k * ld + ld + e] = v[1] + v[3];
  } else {
    asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
                 :: "r"(SA(o) + tk * 128 + ((ch ^ (tk & 7)) << 4)), "r"(bf2(v[0], v[1])), "r"(bf2(v[2], v[3])) : "memory");
  }
}
#define WAIT(f, sc) DEVI void f(void* bar, int ph) { asm volatile("{\n .reg .pred p;\nW: mbarrier.try_wait.parity" sc \
  ".shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" :: "r"(SA(bar)), "r"(ph) : "memory"); }
WAIT(wait_cluster, ".acquire.cluster")
WAIT(wait_spin, "")
// recv live from the start: peers push into it
DEVI void live_from_here(const void*) {}
"""
_P = tilelang.PassConfigKey
_PC = {
    _P.TL_DISABLE_WARP_SPECIALIZED: True,
    _P.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    # Safe: global accesses are TMA (rows past M clipped by the tensor map) or own4's rem-guarded
    # store, and use() admits only table shapes (N % 64 == 0, K % (64 * NB) == 0).
    _P.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    _P.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


@tilelang.jit(pass_configs=_PC)
def _gw(TB, KC, NB, NS, PRE, F=False):
    M, N, K = T.dynamic("M"), T.dynamic("N"), T.dynamic("K")
    BK, NJ = 64 * NB, TB // 8
    NY, YT, WS, WT = (
        (T.dynamic("NY"), "float32", (K, N), (NS, BK, 64))
        if F
        else (N, "bfloat16", (N, K), (NS, 64, BK))
    )
    NJQ = -(-NJ // KC)

    def w_tile(i, r):
        k, n = slice(i * BK, (i + 1) * BK), slice(r, r + 64)
        return (k, n) if F else (n, k)

    @T.prim_func
    def main(
        x: T.Tensor((M, K), "bfloat16"),
        w: T.Tensor(WS, "bfloat16"),
        y: T.Tensor((M, NY), YT),
    ):
        with T.ClusterKernel(
            N // 64 * T.ceildiv(M, TB) * KC, threads=256, cluster_dims=KC, prelude=_PTX
        ) as bid:
            ws = T.alloc_shared(WT, "bfloat16")
            xs = T.alloc_shared((NS, TB, BK), "bfloat16")
            recv = T.alloc_shared((max(KC - 1, 1), NJQ, 128, 4), "float32")
            outs = T.alloc_shared((8 * NJQ, 64), "bfloat16")
            T.annotate_layout(
                {outs: tilelang.layout.make_full_bank_swizzled_layout(outs)}
            )
            full, empty, rbar = (
                T.alloc_barrier([1] * NS),
                T.alloc_barrier([128] * NS),
                T.alloc_barrier([1]),
            )
            acc = T.alloc_fragment((64, TB), "float32")
            tx, rank = T.get_thread_binding(), bid % KC
            row0 = T.alloc_var("int32", init=bid // KC // T.ceildiv(M, TB) * 64)
            t0 = T.alloc_var("int32", init=bid // KC % T.ceildiv(M, TB) * TB)
            i0 = T.alloc_var("int32", init=K // BK * rank // KC)
            nst = T.alloc_var("int32", init=K // BK * (rank + 1) // KC - i0)
            j0 = T.alloc_var("int32", init=NJ * rank // KC)
            j1 = T.alloc_var("int32", init=NJ * (rank + 1) // KC)
            if KC > 1:
                T.call_extern(
                    "handle", "live_from_here", T.address_of(recv[0, 0, 0, 0])
                )
                if tx == 0 and j1 > j0:
                    T.mbarrier_arrive_expect_tx(rbar[0], (KC - 1) * (j1 - j0) * 2048)
                T.cluster_arrive_relaxed()
            if tx < 1:
                for s in T.unroll(PRE):
                    if s < nst:
                        T.tma_copy(
                            w[w_tile(i0 + s, row0)],
                            ws[s, :, :],
                            barrier=full[s],
                            eviction_policy="evict_first",
                        )
                T.pdl_sync()
                for i in T.serial(nst):
                    if i >= NS:
                        T.call_extern(
                            "handle",
                            "wait_spin",
                            T.address_of(empty[i % NS]),
                            (i // NS + 1) % 2,
                        )
                    if i >= PRE:
                        T.tma_copy(
                            w[w_tile(i0 + i, row0)],
                            ws[i % NS, :, :],
                            barrier=full[i % NS],
                            eviction_policy="evict_first",
                        )
                    T.tma_copy(
                        x[t0 : t0 + TB, (i0 + i) * BK : (i0 + i + 1) * BK],
                        xs[i % NS, :, :],
                        barrier=full[i % NS],
                        eviction_policy="evict_last",
                    )
                    T.barrier_arrive(full[i % NS])
                T.pdl_trigger()
            elif tx >= 128:
                T.clear(acc)
                for r in T.serial(T.ceildiv(nst, NS)):
                    for s in T.unroll(NS):
                        if r * NS + s < nst:
                            T.call_extern(
                                "handle", "wait_spin", T.address_of(full[s]), r % 2
                            )
                            T.wgmma_gemm(
                                ws[s, :, :],
                                xs[s, :, :],
                                acc,
                                transpose_A=F,
                                transpose_B=True,
                            )
                            T.wait_wgmma(1)
                            if r * NS + s > 0:
                                T.barrier_arrive(empty[(s + NS - 1) % NS])
                T.wait_wgmma(0)
                T.sync_threads(1, 128)
                if KC > 1:
                    T.cluster_wait()
                    for i, j in T.Parallel(64, TB):
                        if i % 16 < 8 and j % 2 == 0:
                            q = ((j // 8 + 1) * KC + NJ - 1) // NJ - 1
                            if q != rank:
                                T.call_extern(
                                    "handle",
                                    "push4",
                                    T.address_of(acc[i, j]),
                                    T.address_of(
                                        recv[
                                            rank - T.min(T.max(rank - q, 0), 1),
                                            j // 8 - NJ * q // KC,
                                            0,
                                            0,
                                        ]
                                    ),
                                    q,
                                    T.address_of(rbar[0]),
                                )
                if j1 > j0:
                    if KC > 1:
                        T.call_extern(
                            "handle", "wait_cluster", T.address_of(rbar[0]), 0
                        )
                    for i, j in T.Parallel(64, TB):
                        if i % 16 < 8 and j % 2 == 0 and j // 8 >= j0 and j // 8 < j1:
                            T.call_extern(
                                "handle",
                                f"own4<{KC - 1}, {int(F)}>",
                                T.address_of(acc[i, j]),
                                T.address_of(recv[0, j // 8 - j0, 0, 0]),
                                NJQ * 512,
                                T.address_of(
                                    y[t0 + 8 * j0, row0 // 2] if F else outs[0, 0]
                                ),
                                j // 8 - j0,
                                NY,
                                M - t0 - 8 * j0,
                            )
                    T.fence_proxy_async()
                    T.sync_threads(1, 128)
                    LO = 8 * (NJ // KC) * (1 - F)
                    if LO > 0:
                        T.copy(
                            outs[0:LO, :],
                            y[t0 + 8 * j0 : t0 + 8 * j0 + LO, row0 : row0 + 64],
                        )
                    if NJQ * 8 > LO and (j1 - j0) * 8 > LO and not F:
                        T.copy(
                            outs[LO : NJQ * 8, :],
                            y[
                                t0 + 8 * j0 + LO : t0 + 8 * j0 + NJQ * 8,
                                row0 : row0 + 64,
                            ],
                        )

    return main


class _Building(threading.local):
    on = False


_BUILDING = _Building()


class _QuietBuild(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING or not _BUILDING.on


@contextlib.contextmanager
def _build_scope():
    """Only this module's build threads (_BUILDING.on) see these, and only while they run:
    TileLang 0.1.12 issues gcd(TB, 256)-wide wgmmas, they get one m64nTBk16;
    and TileLang's per-kernel INFO logs are dropped."""
    prefix = _E._initialize_wgmma_prefix

    def one_wgmma(self, n_dim=16):
        prefix(self, n_dim)
        if _BUILDING.on:
            self.wgmma_inst_n, self.wgmma_prefix = (
                self.warp_col_tiles,
                f"m64n{self.warp_col_tiles}k16",
            )

    handlers, quiet = logging.getLogger("tilelang").handlers, _QuietBuild()
    _E._initialize_wgmma_prefix = one_wgmma
    for h in handlers:
        h.addFilter(quiet)
    try:
        yield
    finally:
        _E._initialize_wgmma_prefix = prefix
        for h in handlers:
            h.removeFilter(quiet)


def _build(cfg):
    _BUILDING.on = True
    try:
        k = _gw(*cfg)
        if k.adapter.executable is None:
            k.adapter.executable = k.adapter.get_exportable_executable().jit()
    finally:
        _BUILDING.on = False
    return cfg, k


SMEM_MAX = 232448  # SM90 max dynamic shared memory per CTA
# (N, K) -> rows by M (m_hi, KC, token splits, k64 per stage, max stages, TB multiple, PRE = NS / 2);
# _GWC: bf16 out; _GWR / _GWS: fp32 router (decode / SP)
_GWC = {
    (2304, 8192): (
        (64, 3, 1, 2, 16, 8, 0),
        (112, 3, 1, 1, 12, 8, 0),
        (128, 3, 1, 2, 10, 64, 1),
        (256, 3, 1, 1, 8, 8, 1),
    ),
    (8192, 1024): (
        (16, 2, 1, 2, 4, 8, 0),
        (56, 1, 1, 2, 6, 8, 0),
        (64, 1, 1, 2, 11, 8, 0),
        (80, 1, 1, 2, 10, 8, 1),
        (256, 1, 1, 1, 12, 8, 1),
    ),
    (512, 8192): (
        (32, 6, 1, 4, 4, 8, 0),
        (128, 6, 1, 1, 8, 8, 1),
        (256, 4, 2, 1, 12, 8, 0),
    ),
    (1024, 4096): (
        (32, 4, 1, 4, 4, 8, 0),
        (64, 4, 1, 2, 8, 8, 0),
        (128, 6, 1, 2, 4, 8, 1),
        (256, 3, 2, 2, 9, 8, 1),
    ),
    (1280, 8192): (
        (64, 5, 1, 2, 12, 8, 0),
        (96, 5, 1, 1, 8, 8, 1),
        (128, 4, 1, 1, 11, 8, 1),
        (160, 5, 1, 1, 9, 8, 1),
        (224, 4, 1, 1, 5, 8, 0),
    ),
    (256, 8192): (
        (64, 8, 2, 2, 16, 8, 1),
        (160, 8, 2, 2, 16, 8, 0),
        (256, 6, 4, 2, 16, 8, 0),
    ),
    (1024, 40960): ((64, 6, 1, 1, 16, 8, 1),),
    (2048, 8192): (
        (64, 3, 1, 2, 16, 8, 0),
        (128, 3, 1, 1, 16, 8, 1),
        (192, 3, 1, 1, 6, 8, 1),
        (224, 2, 2, 2, 7, 8, 1),
    ),
    (4096, 8192): (
        (16, 2, 1, 2, 8, 8, 0),
        (64, 2, 1, 2, 16, 8, 0),
        (128, 2, 1, 2, 4, 8, 1),
        (256, 2, 1, 1, 16, 8, 1),
    ),
    (7168, 8192): ((32, 1, 1, 2, 12, 8, 0), (96, 1, 1, 2, 4, 8, 1)),
    (8192, 3584): (
        (16, 1, 1, 2, 12, 8, 0),
        (64, 1, 1, 2, 8, 8, 0),
        (96, 1, 1, 2, 6, 8, 1),
        (256, 1, 1, 1, 12, 8, 1),
    ),
    (8192, 4096): ((240, 1, 1, 1, 9, 8, 1),),
    (8192, 8192): ((112, 1, 1, 1, 12, 8, 1),),
}

# fp32 router: K split 6 ways keeps its error below cuBLAS split2.
_GWR = {(512, 8192): ((128, 6, 1, 2, 8, 8, 1), (208, 6, 2, 2, 8, 8, 1))}
_GWS = {(512, 8192): ((256, 6, 2, 2, 8, 8, 1),)}


def _config(m, n, k, tab=_GWC):
    for hi, kc, ts, nb, cap, al, half in tab.get((n, k), ()):
        if 0 < m <= hi:
            tb = (-(-m // ts) + al - 1) // al * al
            ns = min(
                cap,
                (SMEM_MAX - 2048 - 2048 * (kc - 1) * -(-(tb // 8) // kc))
                // (128 * nb * (64 + tb)),
            )
            return tb, kc, nb, ns, ns // 2 if half else ns


_K = {}


def use(m, n, k, tab=_GWC):
    return _config(m, n, k, tab) is not None


def gemv(x, w, out=None, tab=_GWR):
    """x [M, K], w [N, K] (fp32 out: [K, N]) contiguous, ``use(M, N, K)``."""
    if not x.dtype == w.dtype == torch.bfloat16:
        return torch.mm(x, w.t(), out=out)
    if out is None:
        out = x.new_empty((x.shape[0], w.shape[0]))
    if not _K:
        cfgs = sorted(
            {_config(m, n, k) for n, k in _GWC for m in range(1, 257)} - {None}
        )
        cfgs += sorted(
            {
                c + (True,)
                for t in (_GWR, _GWS)
                for m in range(1, 257)
                if (c := _config(m, 512, 8192, t))
            }
        )
        with (
            _build_scope(),
            ThreadPoolExecutor(
                16, initializer=torch.cuda.set_device, initargs=(x.device,)
            ) as ex,
        ):
            list(ex.map(_build, cfgs[x.device.index :: torch.cuda.device_count()]))
            _K.update(ex.map(_build, cfgs))
    f = out.dtype == torch.float32
    _K[
        _config(x.shape[0], *w.shape[::-1], tab) + (True,)
        if f
        else _config(x.shape[0], *w.shape)
    ](x, w, out)
    return out
