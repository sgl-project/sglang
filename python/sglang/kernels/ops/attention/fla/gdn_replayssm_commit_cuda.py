# SPDX-License-Identifier: Apache-2.0
"""The compact ReplaySSM commit of every GDN layer as one CUDA launch.

Same contract and math as ``commit_gdn_replayssm_circular``'s compact-commit kernel (capacity rows fold their
committed ring into the checkpoint; track rows snapshot the state at the requested accepted step from the
unfolded checkpoint), with one CTA per (layer, request, value head) instead of one Triton program per 64-column
tile. The CTA stages the checkpoint and its ring slice in shared memory with bulk async copies issued before the
cursor reads; 4 warps own 32 value rows each and run the fold ``K^T D`` as BF16 MMAs with FP32 accumulation and
the same four products (key high / low x update high / low): ``m16n8k8`` when the window has at most 8 entries,
``m16n8k16`` over all 16 otherwise. FP32 sums run in another order than Triton's (numeric, not bitwise).

Compiled with NVRTC (both track variants) on the first eager call; a call under CUDA-graph capture before that
returns False from ``try_commit`` and the Triton kernel runs instead. The finish kernel (cursor advance) stays
SGLang's: inside a CUDA graph it costs no measurable time.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_SRC = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
#define KD 128
#define VD 128
#define NW 4
#define FULL 0xffffffffu

struct alignas(16) U4 { u32 x, y, z, w; };

struct Params {
  u16* h0; long long h0_layer, h0_slot;          // [layers, slots, HV, V, K]
  const u16* d; long long d_layer, d_slot;       // [layers, slots, HV, L, V]
  const u16* k; long long k_layer, k_slot;       // [layers, slots, H, L, K]
  const float* g; long long g_layer, g_slot;     // [layers, slots, HV, L]
  const u16* dr; long long dr_layer, dr_slot;    // low parts (layout of d), or null
  const u16* kr; long long kr_layer, kr_slot;    // low parts (layout of k), or null
  const void* sidx; const void* ridx; const int* write_pos; const int* cache_base;
  const signed char* is_flush; const void* acc; const void* tidx; const void* tstep;
  int sidx64, ridx64, acc64, tidx64, tstep64;
  int B;
  long long null_block;
};

// shared memory: the checkpoint [V][K], then the ring's keys, key low parts, updates, update low parts
#define SH_STATE 0
#define SH_K (VD * KD)
#define SH_KR (SH_K + RING * KD)
#define SH_D (SH_KR + RING * KD)
#define SH_DR (SH_D + RING * VD)
#define SH_END (SH_DR + RING * VD)

__device__ __forceinline__ float bflo(u32 x) { return __int_as_float(x << 16); }
__device__ __forceinline__ float bfhi(u32 x) { return __int_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf2f(u16 x) { return __int_as_float(((u32)x) << 16); }
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688(float (&d)[4], u32 a0, u32 a1, u32 b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ long long ld_idx(const void* p, int i, int i64) {
  return i64 ? reinterpret_cast<const long long*>(p)[i] : (long long)reinterpret_cast<const int*>(p)[i];
}
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait0(u64* bar) {
  u32 ready = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(smem_u32(bar)) : "memory");
  } while (!ready);
}

// A fragments of bf16 hi / lo of (update[t][v] * decay[t]): rows v = vb + 16 m + r (+ 8), entries t = 2q, 2q + 1
// (a0 / a1) and 2q + 8, 2q + 9 (a2 / a3)
__device__ __forceinline__ void a_frags(const float (&u)[2][2][4], const float (&dec)[4], u32 (&AH)[2][4],
                                        u32 (&AL)[2][4]) {
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i & 1, jp = i >> 1;
      const float lo = u[m][h][2 * jp] * dec[2 * jp], hi = u[m][h][2 * jp + 1] * dec[2 * jp + 1];
      AH[m][i] = pack2(lo, hi);
      AL[m][i] = pack2(lo - bflo(AH[m][i]), hi - bfhi(AH[m][i]));
    }
}

// out = bf16(old e^total + K^T D) for this warp's rows, from the shared checkpoint and keys; B column n = r holds
// k = 32c + 8 (r >> 1) + 2e + (r & 1), so the C fragment of lane q is word e of its 16-byte chunk c
template <bool K16>
__device__ __forceinline__ void fold_store(const u16* sm, const int (&ph)[4], const u32 (&AH)[2][4],
                                           const u32 (&AL)[2][4], float etot, bool resid, u16* dst, int vb, int r,
                                           int q) {
  const u16* sk = sm + SH_K;
  const u16* skr = sm + SH_KR;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    U4 H[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        H[m][h] = *reinterpret_cast<const U4*>(sm + SH_STATE + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q);
    u32 out[2][2][4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int kk = 32 * c + 8 * (r >> 1) + 2 * e + (r & 1);
      const u32 b0 = (u32)sk[ph[0] * KD + kk] | ((u32)sk[ph[1] * KD + kk] << 16);
      const u32 b1 = K16 ? ((u32)sk[ph[2] * KD + kk] | ((u32)sk[ph[3] * KD + kk] << 16)) : 0u;
      u32 r0 = 0u, r1 = 0u;
      if (resid) {
        r0 = (u32)skr[ph[0] * KD + kk] | ((u32)skr[ph[1] * KD + kk] << 16);
        if (K16) r1 = (u32)skr[ph[2] * KD + kk] | ((u32)skr[ph[3] * KD + kk] << 16);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        if (K16) {
          mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], b0, b1);
          mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], b0, b1);
          if (resid) {
            mma16816(a, AH[m][0], AH[m][1], AH[m][2], AH[m][3], r0, r1);
            mma16816(a, AL[m][0], AL[m][1], AL[m][2], AL[m][3], r0, r1);
          }
        } else {  // entries 8-15 carry no weight: the k8 half of the same products
          mma1688(a, AH[m][0], AH[m][1], b0);
          mma1688(a, AL[m][0], AL[m][1], b0);
          if (resid) {
            mma1688(a, AH[m][0], AH[m][1], r0);
            mma1688(a, AL[m][0], AL[m][1], r0);
          }
        }
        const u32 w0 = word(H[m][0], e), w1 = word(H[m][1], e);
        out[m][0][e] = pack2(fmaf(bflo(w0), etot, a[0]), fmaf(bfhi(w0), etot, a[1]));
        out[m][1][e] = pack2(fmaf(bflo(w1), etot, a[2]), fmaf(bfhi(w1), etot, a[3]));
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        U4 v4;
        v4.x = out[m][h][0]; v4.y = out[m][h][1]; v4.z = out[m][h][2]; v4.w = out[m][h][3];
        *reinterpret_cast<U4*>(dst + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q) = v4;
      }
  }
}

// the window's decay: masked inclusive prefix of the log-decays (lane t holds entry t), e^{total - prefix_j}
__device__ __forceinline__ float window(float gt, int len, int lane, const int (&jj)[4], float (&dec)[4]) {
  float pf = lane < len ? gt : 0.f;
#pragma unroll
  for (int dd = 1; dd < 32; dd <<= 1) {
    const float o = __shfl_up_sync(FULL, pf, dd);
    if (lane >= dd) pf += o;
  }
  const float tot = __shfl_sync(FULL, pf, 31);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float pj = __shfl_sync(FULL, pf, jj[i]);
    dec[i] = jj[i] < len ? __expf(tot - pj) : 0.f;
  }
  return __expf(tot);
}

extern "C" __global__ void __launch_bounds__(NW * 32) gdn_replayssm_commit(const Params p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  __shared__ u64 bar;
  constexpr int GRP = HV / HK;
  const int cta = blockIdx.x;
  const int hv = cta % HV;
  const int n = (cta / HV) % p.B;
  const int layer = cta / (HV * p.B);
  const int hk = hv / GRP;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  const int vb = warp * 32;

  const long long slot = ld_idx(p.sidx, n, p.sidx64);
  const long long rid = ld_idx(p.ridx, n, p.ridx64);
  if (slot <= p.null_block) return;
  const bool resid = p.dr != nullptr && p.kr != nullptr;
  u16* sm = reinterpret_cast<u16*>(smem_raw);
  // the checkpoint and the whole ring slice need only the two indices: copies first, cursors meanwhile
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const u32 bytes = 2 * (VD * KD + (resid ? 4 : 2) * RING * KD);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)), "r"(bytes)
                 : "memory");
    bulk_g2s(sm + SH_STATE, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD, 2 * VD * KD,
             &bar);
    bulk_g2s(sm + SH_K, p.k + layer * p.k_layer + rid * p.k_slot + (long long)hk * RING * KD, 2 * RING * KD, &bar);
    bulk_g2s(sm + SH_D, p.d + layer * p.d_layer + rid * p.d_slot + (long long)hv * RING * VD, 2 * RING * VD, &bar);
    if (resid) {
      bulk_g2s(sm + SH_KR, p.kr + layer * p.kr_layer + rid * p.kr_slot + (long long)hk * RING * KD, 2 * RING * KD,
               &bar);
      bulk_g2s(sm + SH_DR, p.dr + layer * p.dr_layer + rid * p.dr_slot + (long long)hv * RING * VD, 2 * RING * VD,
               &bar);
    }
  }
  const int n_hist = p.write_pos[rid];
  const bool fold = p.is_flush[rid] != 0;
  const int base = p.cache_base[rid];
  long long tslot = p.null_block;
  int track_len = 0;
  if (HAS_TRACK) {
    tslot = ld_idx(p.tidx, n, p.tidx64);
    const long long ts = ld_idx(p.tstep, n, p.tstep64);
    if (!(tslot > p.null_block && ts >= 0)) tslot = p.null_block;
    track_len = n_hist - (int)ld_idx(p.acc, n, p.acc64) + (int)ts + 1;
  }
  const bool track = tslot > p.null_block;
  const float gt = lane < RING ? p.g[layer * p.g_layer + rid * p.g_slot + hv * RING + ((base + lane) & (RING - 1))] : 0.f;
  const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
  int ph[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ph[i] = (base + jj[i]) & (RING - 1);
  __syncthreads();  // the barrier's init before anyone waits on it
  mbar_wait0(&bar);  // also before an early exit: no copy may land after the CTA is gone
  if (!fold && !track) return;
  // windows of at most 8 entries (every fold of a step of up to 8 draft rows) take the k8 products
  const bool k16 = (fold && n_hist > 8) || (track && track_len > 8);

  float u[2][2][4];  // updates (high + low, FP32): [m][half][entry jj]
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int v = vb + 16 * m + 8 * h + r;
        float x = 0.f;
        if (i < 2 || k16) {
          x = bf2f(sm[SH_D + ph[i] * VD + v]);
          if (resid) x += bf2f(sm[SH_DR + ph[i] * VD + v]);
        }
        u[m][h][i] = x;
      }
  if (fold) {
    float dec[4];
    const float etot = window(gt, n_hist, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16) fold_store<true>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
    else fold_store<false>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + slot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
  }
  if (track) {
    float dec[4];
    const float etot = window(gt, track_len, lane, jj, dec);
    u32 AH[2][4], AL[2][4];
    a_frags(u, dec, AH, AL);
    if (k16) fold_store<true>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
    else fold_store<false>(sm, ph, AH, AL, etot, resid, p.h0 + layer * p.h0_layer + tslot * p.h0_slot + (long long)hv * VD * KD,
               vb, r, q);
  }
}
"""


class _Params(ctypes.Structure):
    _fields_ = [
        ("h0", ctypes.c_void_p), ("h0_layer", ctypes.c_int64), ("h0_slot", ctypes.c_int64),
        ("d", ctypes.c_void_p), ("d_layer", ctypes.c_int64), ("d_slot", ctypes.c_int64),
        ("k", ctypes.c_void_p), ("k_layer", ctypes.c_int64), ("k_slot", ctypes.c_int64),
        ("g", ctypes.c_void_p), ("g_layer", ctypes.c_int64), ("g_slot", ctypes.c_int64),
        ("dr", ctypes.c_void_p), ("dr_layer", ctypes.c_int64), ("dr_slot", ctypes.c_int64),
        ("kr", ctypes.c_void_p), ("kr_layer", ctypes.c_int64), ("kr_slot", ctypes.c_int64),
        ("sidx", ctypes.c_void_p), ("ridx", ctypes.c_void_p), ("write_pos", ctypes.c_void_p),
        ("cache_base", ctypes.c_void_p), ("is_flush", ctypes.c_void_p), ("acc", ctypes.c_void_p),
        ("tidx", ctypes.c_void_p), ("tstep", ctypes.c_void_p),
        ("sidx64", ctypes.c_int), ("ridx64", ctypes.c_int), ("acc64", ctypes.c_int),
        ("tidx64", ctypes.c_int), ("tstep64", ctypes.c_int),
        ("B", ctypes.c_int),
        ("null_block", ctypes.c_int64),
    ]


_LOCK = threading.Lock()
_KERNELS: dict = {}


def _check(res, what):
    err = res[0]
    if int(err) != 0:
        raise RuntimeError(f"gdn_replayssm_commit_cuda: {what}: {err!r}")
    return res[1] if len(res) == 2 else res[1:]


def _compile(device: int, hk: int, hv: int, ring: int, has_track: bool):
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    defines = dict(HK=hk, HV=hv, RING=ring, HAS_TRACK=int(has_track))
    src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _SRC
    prog = _check(nvrtc.nvrtcCreateProgram(src.encode(), b"gdn_replayssm_commit.cu", 0, [], []), "create")
    major, minor = torch.cuda.get_device_capability(device)
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-default-device"]
    if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        size = _check(nvrtc.nvrtcGetProgramLogSize(prog), "log size")
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("gdn_replayssm_commit_cuda: NVRTC failed:\n" + log.decode(errors="replace"))
    cubin = b" " * _check(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
    _check(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
    nvrtc.nvrtcDestroyProgram(prog)
    module = _check(cu.cuModuleLoadData(cubin), "module")
    fn = _check(cu.cuModuleGetFunction(module, b"gdn_replayssm_commit"), "function")
    _check(
        cu.cuFuncSetAttribute(
            fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem_bytes(ring)
        ),
        "smem attribute",
    )
    return module, fn


def _smem_bytes(ring: int) -> int:
    # the checkpoint [128][128] and four [ring][128] rings, BF16
    return 2 * (128 * 128 + 4 * ring * 128)


def _kernel(device: int, hk: int, hv: int, ring: int, has_track: bool):
    key = (device, hk, hv, ring, has_track)
    fn = _KERNELS.get(key)
    if fn is None:
        with _LOCK:
            fn = _KERNELS.get(key)
            if fn is None:
                fn = _KERNELS[key] = _compile(device, hk, hv, ring, has_track)[1]
    return fn


def supported(checkpoint_state: torch.Tensor, d_cache: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Shapes the CUDA commit handles: BF16 [layers, slots, HV, 128, 128] checkpoints, a BF16 ring of 16."""
    return (
        checkpoint_state.dtype == torch.bfloat16
        and tuple(checkpoint_state.shape[-2:]) == (128, 128)
        and checkpoint_state.stride()[2:] == (128 * 128, 128, 1)
        and d_cache.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and d_cache.shape[-2] == 16
        and d_cache.stride()[2:] == (16 * 128, 128, 1)
        and k_cache.stride()[2:] == (16 * 128, 128, 1)
        and checkpoint_state.shape[2] % k_cache.shape[2] == 0
    )


_LOGGED: set = set()


def try_commit(**kw) -> bool:
    """``commit(**kw)`` when the shapes fit and the kernel is compiled or compilable now; else False."""
    st, d, k = kw["checkpoint_state"], kw["d_cache"], kw["k_cache"]
    if not supported(st, d, k):
        _log_once("shape", "SGLANG_Q35MF_COMMIT_CUDA: unsupported ring / checkpoint shapes, Triton commit")
        return False
    device, hk, hv, ring = st.device.index, k.shape[2], st.shape[2], d.shape[-2]
    if kw["state_batch_indices"].shape[0] == 0:
        return False
    if any((device, hk, hv, ring, t) not in _KERNELS for t in (False, True)):
        if torch.cuda.is_current_stream_capturing():
            _log_once("capture", "SGLANG_Q35MF_COMMIT_CUDA: not compiled before graph capture, Triton commit")
            return False
        for t in (False, True):
            _kernel(device, hk, hv, ring, t)
    commit(**kw)
    _log_once(("on", st.shape[1] if st.dim() > 1 else 0), "SGLANG_Q35MF_COMMIT_CUDA: CUDA compact commit runs")
    return True


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg)


def _strides(t: Optional[torch.Tensor]):
    return (0, 0) if t is None else (t.stride(0), t.stride(1))


def commit(
    *,
    checkpoint_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    d_residual_cache: Optional[torch.Tensor],
    k_residual_cache: Optional[torch.Tensor],
    state_batch_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    accept_lens: torch.Tensor,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
    null_block_id: int = -1,
) -> None:
    """The compact-commit part of ``commit_gdn_replayssm_circular`` (not its cursor finish)."""
    from cuda.bindings import driver as cu

    num_layers, _, HV, V, K = checkpoint_state.shape
    H, ring = k_cache.shape[2], d_cache.shape[-2]
    B = state_batch_indices.shape[0]
    has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
    resid = d_residual_cache is not None and k_residual_cache is not None
    assert write_pos.dtype == torch.int32 and cache_base.dtype == torch.int32
    assert is_flush.dtype == torch.int8
    for t in (state_batch_indices, replay_indices, accept_lens, mamba_track_indices, mamba_steps_to_track):
        assert t is None or (t.is_contiguous() and t.dtype in (torch.int32, torch.int64))
    i64 = lambda t: int(t is not None and t.dtype == torch.int64)  # noqa: E731
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    p = _Params(
        checkpoint_state.data_ptr(), *_strides(checkpoint_state),
        d_cache.data_ptr(), *_strides(d_cache),
        k_cache.data_ptr(), *_strides(k_cache),
        g_cache.data_ptr(), *_strides(g_cache),
        ptr(d_residual_cache) if resid else 0, *(_strides(d_residual_cache) if resid else (0, 0)),
        ptr(k_residual_cache) if resid else 0, *(_strides(k_residual_cache) if resid else (0, 0)),
        state_batch_indices.data_ptr(), replay_indices.data_ptr(), write_pos.data_ptr(),
        cache_base.data_ptr(), is_flush.data_ptr(), accept_lens.data_ptr(),
        ptr(mamba_track_indices), ptr(mamba_steps_to_track),
        i64(state_batch_indices), i64(replay_indices), i64(accept_lens),
        i64(mamba_track_indices), i64(mamba_steps_to_track),
        B, null_block_id,
    )
    device = checkpoint_state.device.index
    fn = _kernel(device, H, HV, ring, has_track)
    cfg = cu.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = num_layers * B * HV, 1, 1
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 128, 1, 1
    cfg.sharedMemBytes = _smem_bytes(ring)
    cfg.hStream = cu.CUstream(torch.cuda.current_stream(checkpoint_state.device).cuda_stream)
    cfg.attrs, cfg.numAttrs = [], 0
    args = (ctypes.c_void_p * 1)(ctypes.addressof(p))
    _check(cu.cuLaunchKernelEx(cfg, fn, ctypes.addressof(args), 0), "launch")
