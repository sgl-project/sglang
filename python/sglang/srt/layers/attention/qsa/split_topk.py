"""Cluster-split exact top-k for the QSA decode indexer (v2).

``fast_topk`` runs one 1024-thread CTA per query row. At decode/verify row
counts (4-32) that occupies a handful of SMs, and each CTA's scan of 25k-65k
logits is bound by the latency of its own loads (two passes over the row).

Here every row is handled by a thread-block cluster of ``C`` CTAs (1-16, per shape). The
row's ``length`` (read on device) is split evenly across the cluster, so the
work per CTA follows the real length, not the logits width. Each CTA loads its
slice once (16-byte loads) into registers as order-preserving 32-bit keys, and
the cluster runs one exact MSD radix select over the keys with 11/11/10-bit
digits:

* each CTA builds a local 2048-bin histogram of the keys still matching the
  selected prefix, then adds its nonzero bins (and 64-bin group totals) into
  the leader CTA's shared memory (DSMEM ``red``);
* after one cluster barrier, one warp per CTA reads the 32 group totals and
  then the 64 bins of the threshold group from the leader (two DSMEM loads),
  finds the threshold digit with warp suffix scans, and the round ends.

A round stops early when the threshold bin is taken whole. After the last
round every key strictly above the threshold is written and exactly
``topk - #above`` keys equal to it, with slots handed out by one DSMEM atomic
per CTA. There is no candidate buffer, so nothing can be clipped, and there is
no global workspace or arrival counter: the kernel is CUDA-graph safe as is
(grid = rows * C, fixed for a given logits width).

Semantics match ``fast_topk(score, lengths, 512, row_starts=None)``: the
selected value multiset is the exact top-512, output order within a row is
unspecified, ties with the 512th value are broken arbitrarily, rows with
``length <= 512`` get ``0..length-1`` then -1 without reading scores, and a
row never reads past ``length`` (``length`` is clamped to the logits width).

The CUDA source ships in this module and is JIT-compiled for the running GPU.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

SPLIT_TOPK_CHUNK = 4096  # kept for API compatibility with v1
_SUPPORTED_K = (512,)
_THREADS = 1024
_MAX_ITERS = 8  # float4 loads per thread (32 keys in registers)
_MAX_CLUSTER = 16  # non-portable cluster size, allowed on sm_90+

_SOURCE = r"""
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {
namespace qsa_cluster_topk_detail {

constexpr int kTopK = 512;
constexpr int kBins = 2048;  // 11-bit digits
constexpr int kGroupBins = 64;
constexpr int kGroups = kBins / kGroupBins;  // 32 = one warp
constexpr int kRounds = 3;                   // key bits 31..21, 20..10, 9..0

struct Params {
  const float* __restrict__ input;      // [B, input_stride]
  const int32_t* __restrict__ lengths;  // [B]
  int32_t* __restrict__ indices;        // [B, kTopK]
  int64_t input_stride;
  int32_t width;
};

SGL_DEVICE uint32_t to_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

SGL_DEVICE uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
SGL_DEVICE void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
SGL_DEVICE void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
SGL_DEVICE uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
SGL_DEVICE uint32_t map_rank(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
SGL_DEVICE void remote_red_add(uint32_t addr, uint32_t v) {
  asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
SGL_DEVICE uint32_t remote_atom_add(uint32_t addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(addr), "r"(v) : "memory");
  return old;
}
SGL_DEVICE uint32_t remote_ld(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(addr) : "memory");
  return v;
}
SGL_DEVICE uint2 remote_ld2(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared::cluster.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr) : "memory");
  return v;
}

// Inclusive suffix sum over the lanes of a warp (lane l gets sum of lanes >= l).
SGL_DEVICE uint32_t warp_suffix_sum(uint32_t x, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_down_sync(0xffffffffu, x, o);
    if (lane + o < 32) x += t;
  }
  return x;
}

SGL_DEVICE uint32_t warp_prefix_sum(uint32_t x, int lane) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t t = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += t;
  }
  return x;
}

template <int kThreads, int kCluster, int kIters, bool kVec, bool kUsePDL>
__global__ __launch_bounds__(kThreads) void cluster_topk_kernel(const Params __grid_constant__ p) {
  constexpr int kItems = kIters * 4;
  constexpr int kBinsPerThread = kBins / kThreads;
  constexpr int kThreadsPerGroup = kGroupBins / kBinsPerThread;
  static_assert(kBins % kThreads == 0 && kThreadsPerGroup >= 1 && kThreadsPerGroup <= 32);
  static_assert(kThreads % 32 == 0);
  static_assert(kItems <= 64, "packed 16-bit per-warp counts");

  __shared__ alignas(16) uint32_t s_local[kBins];
  __shared__ alignas(16) uint32_t s_lead_hist[kRounds][kBins];  // read only in rank 0
  __shared__ alignas(16) uint32_t s_lead_group[kRounds][kGroups];
  __shared__ uint32_t s_lead_count;  // rank 0: packed (#above | #equal << 16)
  __shared__ uint32_t s_block_count;
  __shared__ uint32_t s_block_base;
  __shared__ uint32_t s_res[3];

  const int tx = threadIdx.x;
  const int lane = tx & 31;
  const uint32_t rank = cluster_rank();
  const int row = blockIdx.x / kCluster;

  // Shared-memory init does not depend on the producer kernel: run it before the PDL wait.
#pragma unroll
  for (int i = tx; i < kBins; i += kThreads) s_local[i] = 0;
  if (rank == 0) {
    for (int i = tx; i < kRounds * kBins; i += kThreads) (&s_lead_hist[0][0])[i] = 0;
    if (tx < kRounds * kGroups) (&s_lead_group[0][0])[tx] = 0;
    if (tx == 0) s_lead_count = 0;
  }
  if (tx == 0) s_block_count = 0;
  __syncthreads();
  cluster_arrive();  // released here, waited for right before the first DSMEM access

  device::PDLWaitPrimary<kUsePDL>();
  const int length = min(p.lengths[row], p.width);
  int32_t* out = p.indices + static_cast<int64_t>(row) * kTopK;
  if (length <= kTopK) {
    // fast_topk's naive path, decided per row so the whole cluster leaves together.
    if (rank == 0) {
      for (int i = tx; i < kTopK; i += kThreads) out[i] = (i < length) ? i : -1;
    }
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }

  // Even split of [0, length) over the cluster, 4-aligned slices.
  const int span = ((length + kCluster - 1) / kCluster + 3) & ~3;
  const int beg = static_cast<int>(rank) * span;
  const int n = max(0, min(length - beg, span));
  const float* src = p.input + static_cast<int64_t>(row) * p.input_stride + beg;

  uint32_t key[kItems];
#pragma unroll
  for (int j = 0; j < kIters; ++j) {
    const int off = (j * kThreads + tx) * 4;
    if (kVec && off + 4 <= n) {
      const float4 v = __ldg(reinterpret_cast<const float4*>(src + off));
      key[4 * j + 0] = to_key(v.x);
      key[4 * j + 1] = to_key(v.y);
      key[4 * j + 2] = to_key(v.z);
      key[4 * j + 3] = to_key(v.w);
    } else {
#pragma unroll
      for (int v = 0; v < 4; ++v) key[4 * j + v] = (off + v < n) ? to_key(__ldg(src + off + v)) : 0u;
    }
  }
  const auto valid = [&](int j, int v) { return (j * kThreads + tx) * 4 + v < n; };

  const uint32_t lead_hist = map_rank(smem_addr(&s_lead_hist[0][0]), 0);
  const uint32_t lead_group = map_rank(smem_addr(&s_lead_group[0][0]), 0);
  const uint32_t lead_count = map_rank(smem_addr(&s_lead_count), 0);

  uint32_t prefix = 0, pmask = 0;
  uint32_t rem = kTopK;  // still to select among keys matching the prefix
  uint32_t need = 0;     // keys equal (under pmask) to the prefix that are selected
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int shift = (r == 0) ? 21 : ((r == 1) ? 10 : 0);
    const uint32_t dmask = (r == 2) ? 0x3FFu : 0x7FFu;
#pragma unroll
    for (int j = 0; j < kIters; ++j) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const uint32_t k = key[4 * j + v];
        if (valid(j, v) && (k & pmask) == prefix) ::atomicAdd(&s_local[(k >> shift) & dmask], 1u);
      }
    }
    __syncthreads();
    if (r == 0) cluster_wait();  // every CTA started and the leader's buffers are zeroed

    uint32_t gsum = 0;
#pragma unroll
    for (int b = 0; b < kBinsPerThread; ++b) {
      const int bin = tx * kBinsPerThread + b;
      const uint32_t v = s_local[bin];
      if (v) {
        remote_red_add(lead_hist + static_cast<uint32_t>((r * kBins + bin) * 4), v);
        s_local[bin] = 0;
      }
      gsum += v;
    }
#pragma unroll
    for (int o = kThreadsPerGroup / 2; o > 0; o >>= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, o);
    if ((tx % kThreadsPerGroup) == 0 && gsum) {
      remote_red_add(lead_group + static_cast<uint32_t>((r * kGroups + tx / kThreadsPerGroup) * 4), gsum);
    }
    cluster_arrive();
    cluster_wait();

    if (tx < 32) {
      const uint32_t gv = remote_ld(lead_group + static_cast<uint32_t>((r * kGroups + lane) * 4));
      const uint32_t incl = warp_suffix_sum(gv, lane);
      const uint32_t excl = incl - gv;
      const uint32_t hit = __ballot_sync(0xffffffffu, incl >= rem && excl < rem);
      const int g = __ffs(hit) - 1;
      const uint32_t above_g = __shfl_sync(0xffffffffu, excl, g);
      const uint2 bv = remote_ld2(lead_hist + static_cast<uint32_t>((r * kBins + g * kGroupBins + 2 * lane) * 4));
      const uint32_t s = bv.x + bv.y;
      const uint32_t incl2 = warp_suffix_sum(s, lane);
      const uint32_t excl2 = incl2 - s;
      const uint32_t want = rem - above_g;
      if (excl2 < want && incl2 >= want) {
        if (excl2 + bv.y >= want) {
          s_res[0] = g * kGroupBins + 2 * lane + 1;
          s_res[1] = above_g + excl2;
          s_res[2] = bv.y;
        } else {
          s_res[0] = g * kGroupBins + 2 * lane;
          s_res[1] = above_g + excl2 + bv.y;
          s_res[2] = bv.x;
        }
      }
    }
    __syncthreads();
    const uint32_t bin = s_res[0];
    const uint32_t above = s_res[1];
    const uint32_t cnt = s_res[2];
    prefix |= bin << shift;
    pmask |= dmask << shift;
    if (above + cnt == rem) {  // the threshold bin is taken whole
      need = cnt;
      break;
    }
    rem -= above;
    need = rem;  // final when r == kRounds - 1: ties with the exact threshold key
  }

  // Selected: keys above the prefix (kTopK - need of them) and the first `need`
  // keys equal to it. Slots: one smem atomic per warp, one DSMEM atomic per CTA.
  uint32_t mine = 0;
#pragma unroll
  for (int j = 0; j < kIters; ++j) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t mk = key[4 * j + v] & pmask;
      if (valid(j, v)) mine += (mk > prefix) ? 1u : ((mk == prefix) ? 0x10000u : 0u);
    }
  }
  const uint32_t incl = warp_prefix_sum(mine, lane);
  uint32_t warp_base = 0;
  // Equal keys past the first kTopK of a warp/CTA can never get a slot: clamp
  // their contribution so the packed 16-bit counts cannot overflow.
  if (lane == 31) warp_base = ::atomicAdd(&s_block_count, (incl & 0xFFFFu) | (min(incl >> 16, static_cast<uint32_t>(kTopK)) << 16));
  warp_base = __shfl_sync(0xffffffffu, warp_base, 31);
  __syncthreads();
  if (tx == 0) {
    const uint32_t tot = s_block_count;
    const uint32_t eq = min(tot >> 16, static_cast<uint32_t>(kTopK));  // later equal slots are never used
    s_block_base = remote_atom_add(lead_count, (tot & 0xFFFFu) | (eq << 16));
  }
  __syncthreads();
  cluster_arrive();  // this CTA is done with the leader's shared memory
  const uint32_t base = s_block_base + warp_base + (incl - mine);
  uint32_t gpos = base & 0xFFFFu;
  uint32_t epos = base >> 16;
  const uint32_t gt_total = kTopK - need;
#pragma unroll
  for (int j = 0; j < kIters; ++j) {
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const uint32_t mk = key[4 * j + v] & pmask;
      if (valid(j, v)) {
        const int idx = beg + (j * kThreads + tx) * 4 + v;
        if (mk > prefix) {
          out[gpos++] = idx;
        } else if (mk == prefix) {
          if (epos < need) out[gt_total + epos] = idx;
          ++epos;
        }
      }
    }
  }
  if (rank == 0) cluster_wait();  // keep the leader's shared memory alive for its peers
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_cluster_topk_detail

template <int kThreads, int kCluster, int kIters, bool kVec, bool kUsePDL>
struct QsaClusterTopKKernel {
  static constexpr auto kernel = qsa_cluster_topk_detail::cluster_topk_kernel<kThreads, kCluster, kIters, kVec, kUsePDL>;

  static void run(const tvm::ffi::TensorView score, const tvm::ffi::TensorView lengths, const tvm::ffi::TensorView indices) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto L = SymbolicSize{"length"};
    auto S = SymbolicSize{"input_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({B, L}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(score);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(lengths);
    TensorMatcher({B, qsa_cluster_topk_detail::kTopK}).with_dtype<int32_t>().with_device(device).verify(indices);
    const int64_t width = L.unwrap();
    RuntimeCheck(width >= 1 && width <= static_cast<int64_t>(kCluster) * kThreads * 4 * kIters, "cluster top-k: width exceeds capacity");
    if constexpr (kVec) {
      RuntimeCheck(
          reinterpret_cast<uintptr_t>(score.data_ptr()) % 16 == 0 && S.unwrap() % 4 == 0,
          "cluster top-k: vector path needs 16-byte aligned rows");
    }
    const auto params = qsa_cluster_topk_detail::Params{
        .input = static_cast<const float*>(score.data_ptr()),
        .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .input_stride = S.unwrap(),
        .width = static_cast<int32_t>(width),
    };
    static bool configured = false;
    if (!configured) {
      if constexpr (kCluster > 8) {
        RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      }
      configured = true;
    }
    const auto grid = static_cast<uint32_t>(B.unwrap() * kCluster);
    LaunchKernel(grid, kThreads, device.unwrap(), 0).enable_pdl(kUsePDL).enable_cluster(dim3(kCluster, 1, 1))(kernel, params);
  }
};

}  // namespace sglang
"""


def _source_path() -> str:
    digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
    root = pathlib.Path(
        os.environ.get("SGLANG_QSA_SPLIT_TOPK_SRC_DIR")
        or os.path.join(os.path.expanduser("~"), ".cache", "sglang", "qsa_split_topk")
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"qsa_cluster_topk_{digest}.cuh"
    if not path.exists():
        tmp = root / f".{path.name}.{os.getpid()}"
        tmp.write_text(_SOURCE)
        os.replace(tmp, path)
    return str(path)


@cache_once
def _module(threads: int, cluster: int, iters: int, vec: bool):
    args = make_cpp_args(threads, cluster, iters, vec, is_arch_support_pdl())
    return load_jit(
        "qsa_cluster_topk",
        *args,
        cuda_files=[_source_path()],
        cuda_wrappers=[("cluster_topk", f"QsaClusterTopKKernel<{args}>::run")],
    )


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name)
    return int(v) if v else None


def _config(rows: int, width: int) -> tuple:
    """(threads, cluster, iters) for a [rows, width] score matrix (static per shape).

    Measured: ~128 CTAs of 1024 threads is best, i.e. a
    cluster of 16 for <= 8 rows, 8 for 16 rows, 4 for 32 rows; the cluster grows
    if needed so that a thread holds at most 32 keys."""
    threads = _env_int("SGLANG_QSA_TOPK_THREADS") or _THREADS
    cluster = _env_int("SGLANG_QSA_TOPK_CLUSTER")
    if cluster is None:
        cluster = 1
        while cluster < _MAX_CLUSTER and cluster * 2 * rows <= 128:
            cluster *= 2
        while cluster < _MAX_CLUSTER and cluster * threads * 4 * _MAX_ITERS < width:
            cluster *= 2
    per_thread4 = -(-width // (cluster * threads * 4))
    iters = 1
    while iters < per_thread4:
        iters *= 2
    return threads, cluster, iters


def reserve_split_topk(rows: int, width: int, device) -> None:
    """No workspace is needed any more; kept for API compatibility (call before capture)."""
    return None


def split_topk_supported(score: torch.Tensor, topk: int) -> bool:
    if not (score.is_cuda and score.dtype == torch.float32 and score.dim() == 2 and score.stride(1) == 1):
        return False
    if topk not in _SUPPORTED_K or score.shape[1] <= 0:
        return False
    threads, cluster, iters = _config(score.shape[0], score.shape[1])
    return iters <= _MAX_ITERS and score.shape[1] <= cluster * threads * 4 * iters


def split_fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int = 512,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in for ``fast_topk(score, lengths, topk, row_starts=None)``."""
    if row_starts is not None:
        raise ValueError("split top-k reads rows from column 0")
    if topk not in _SUPPORTED_K:
        raise RuntimeError(f"split top-k supports k in {_SUPPORTED_K}, got {topk}")
    rows, width = score.shape
    indices = score.new_empty((rows, topk), dtype=torch.int32)
    if rows == 0:
        return indices
    threads, cluster, iters = _config(rows, width)
    vec = score.data_ptr() % 16 == 0 and score.stride(0) % 4 == 0
    _module(threads, cluster, iters, vec).cluster_topk(score, lengths, indices)
    return indices


__all__ = ["SPLIT_TOPK_CHUNK", "reserve_split_topk", "split_fast_topk", "split_topk_supported"]
