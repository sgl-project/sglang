// Fused GroupNorm + SiLU CUDA kernels for SGLang diffusion VAEs (H200 / SM90).
// Standalone tvm-ffi build: direct TVM_FFI_DLL_EXPORT_TYPED_FUNC exports,
// tvm::ffi::TensorView arguments, output tensors passed last, launches on
// PyTorch's current stream. No sglang headers; the small device-helper layer
// below replaces the sgl_kernel include stack of the original implementation
// (see docs/candidate_lineage.md for the porting record).
//
// Contract: x and y are contiguous [B, C, spatial] views; one group is a
// contiguous block of group_size = (C/num_groups)*spatial; affine is
// per-channel: channel = g*channels_per_group + (i_within_group / spatial);
//   y = silu((x - mean)*rstd * weight[ch] + bias[ch]),  silu(z) = z*sigmoid(z)
// mean/var per (batch, group) in fp32 (biased; var clamped >= 0).
//
// Two execution paths, dispatched by group_size in solution/binding.py:
//   - small: one CTA per (batch, group), two passes in-kernel, no scratch.
//   - large: deterministic three-kernel pipeline (chunk stats -> per-row
//     finalize -> apply) over a persistent grid of (row, chunk) tasks; fp32
//     partial sums in caller-provided scratch, no atomics.

#include <ATen/cuda/CUDAContext.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using tvm::ffi::TensorView;

// ---------------------------------------------------------------------------
// Device helper layer (standalone replacements for the sgl_kernel utilities).
// ---------------------------------------------------------------------------

constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kWarpsPerBlock = kBlockThreads / kWarpThreads;  // 8
// Elements per CTA-task in the large path (256 thr * 4 vecs * 8 half = 8192).
// Must match the chunk constant in solution/binding.py.
constexpr int64_t kChunkElems = 8192;

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_f32<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <>
__device__ __forceinline__ float from_f32<float>(float v) {
  return v;
}

// 16-byte vector pack with element access, mirroring AlignedVector semantics.
// The streaming variants use last-use/streaming cache hints for the giant
// pipeline's read-once/write-once tensors (far beyond L2; avoids thrashing
// resident lines that the affine params and partials want).
template <typename T, int N>
struct alignas(16) Pack {
  T elems[N];
  static_assert(sizeof(T) * N == 16, "Pack must be exactly 16 bytes");
  __device__ __forceinline__ T& operator[](int i) { return elems[i]; }
  __device__ __forceinline__ const T& operator[](int i) const { return elems[i]; }
  __device__ __forceinline__ void load(const T* __restrict__ base, int64_t vec_idx) {
    *this = *(reinterpret_cast<const Pack*>(base) + vec_idx);
  }
  __device__ __forceinline__ void store(T* __restrict__ base, int64_t vec_idx) const {
    *(reinterpret_cast<Pack*>(base) + vec_idx) = *this;
  }
  __device__ __forceinline__ void load_streaming(const T* __restrict__ base, int64_t vec_idx) {
    const int4 v = __ldcs(reinterpret_cast<const int4*>(base) + vec_idx);
    *this = *reinterpret_cast<const Pack*>(&v);
  }
  __device__ __forceinline__ void store_streaming(T* __restrict__ base, int64_t vec_idx) const {
    __stcs(reinterpret_cast<int4*>(base) + vec_idx, *reinterpret_cast<const int4*>(this));
  }
};

// Butterfly sum over the low `Width` lanes (Width must be a power of two
// <= 32; xor offsets stay inside each Width-lane segment, so the full mask is
// safe for every resident lane).
template <int Width = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = Width / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// Block-wide reduction of two fp32 accumulators (sum, sumsq). Deterministic
// fixed-order tree, no atomics; result broadcast to all threads. `smem` must
// hold >= 2*kWarps + 2 floats. kWarps must be a power of two <= 32.
template <int kWarps = kWarpsPerBlock>
__device__ __forceinline__ void block_reduce2(float& a, float& b, float* smem) {
  a = warp_reduce_sum(a);
  b = warp_reduce_sum(b);
  const uint32_t lane = threadIdx.x & (kWarpThreads - 1);
  const uint32_t warp_id = threadIdx.x >> 5;
  if (lane == 0) {
    smem[warp_id] = a;
    smem[kWarps + warp_id] = b;
  }
  __syncthreads();
  if (warp_id == 0) {
    float ta = (lane < kWarps) ? smem[lane] : 0.0f;
    float tb = (lane < kWarps) ? smem[kWarps + lane] : 0.0f;
    ta = warp_reduce_sum<kWarps>(ta);
    tb = warp_reduce_sum<kWarps>(tb);
    if (lane == 0) {
      smem[2 * kWarps] = ta;
      smem[2 * kWarps + 1] = tb;
    }
  }
  __syncthreads();
  a = smem[2 * kWarps];
  b = smem[2 * kWarps + 1];
}

__device__ __forceinline__ float siluf(float z) {
  // accuracy-compatible sigmoid (no fast-math flag): z / (1 + exp(-z))
  return z / (1.0f + expf(-z));
}

// ---------------------------------------------------------------------------
// Kernel bodies (ported verbatim from the prior tuned implementation; only the
// helper spellings changed).
// ---------------------------------------------------------------------------

template <typename DType>
struct GnsParams {
  const void* __restrict__ x;
  const void* __restrict__ weight;
  const void* __restrict__ bias;
  void* __restrict__ y;
  int64_t channels;
  int64_t spatial;
  int64_t num_groups;
  int64_t channels_per_group;
  int64_t group_size;
  int64_t num_rows;
  float eps;
};

template <typename DType>
struct GnsLargeParams {
  const void* __restrict__ x;
  const void* __restrict__ weight;
  const void* __restrict__ bias;
  void* __restrict__ y;
  void* __restrict__ partial_sum;    // fp32 [num_rows * chunks_per_row]
  void* __restrict__ partial_sumsq;  // fp32 [num_rows * chunks_per_row]
  void* __restrict__ mean;           // fp32 [num_rows]
  void* __restrict__ rstd;           // fp32 [num_rows]
  void* __restrict__ row_counter;    // int32 [num_rows], zero between calls (giant path only)
  int64_t channels;
  int64_t spatial;
  int64_t num_groups;
  int64_t channels_per_group;
  int64_t group_size;
  int64_t num_rows;
  int64_t chunk_elems;
  int64_t chunks_per_row;
  float eps;
};

// Reduce x[0, nelem) into (lsum, lsumsq) in fp32. `vec_aligned` enables the
// 16-byte vector path; `kStream` selects streaming (last-use) loads for
// tensors with no reuse inside this kernel. Two independent accumulator
// pairs over a 2x-unrolled vector loop break the per-thread FADD dependency
// chain (the add latency otherwise caps throughput well below DRAM peak).
template <typename DType, int kVec, bool kStream = false>
__device__ __forceinline__ void accumulate_stats(const DType* __restrict__ x, int64_t nelem,
                                                 bool vec_aligned, float& lsum, float& lsumsq) {
  const int64_t nvec = vec_aligned ? nelem / kVec : 0;
  float s0 = 0.0f, q0 = 0.0f, s1 = 0.0f, q1 = 0.0f;
  int64_t vi = threadIdx.x;
  const int64_t stride = blockDim.x;
  for (; vi + stride < nvec; vi += 2 * stride) {
    Pack<DType, kVec> a;
    Pack<DType, kVec> b;
    if constexpr (kStream) {
      a.load_streaming(x, vi);
      b.load_streaming(x, vi + stride);
    } else {
      a.load(x, vi);
      b.load(x, vi + stride);
    }
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xa = to_f32<DType>(a[j]);
      const float xb = to_f32<DType>(b[j]);
      s0 += xa;
      q0 += xa * xa;
      s1 += xb;
      q1 += xb * xb;
    }
  }
  for (; vi < nvec; vi += stride) {
    Pack<DType, kVec> v;
    if constexpr (kStream) {
      v.load_streaming(x, vi);
    } else {
      v.load(x, vi);
    }
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xf = to_f32<DType>(v[j]);
      s0 += xf;
      q0 += xf * xf;
    }
  }
  lsum += (s0 + s1);
  lsumsq += (q0 + q1);
  for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
    const float xf = to_f32<DType>(x[i]);
    lsum += xf;
    lsumsq += xf * xf;
  }
}

// Normalize + per-channel affine + SiLU over x[0, nelem) -> y, where
// `group_off` is the start offset within the group (for channel indexing).
template <typename DType, int kVec>
__device__ __forceinline__ void apply_affine_silu(const DType* __restrict__ x, DType* __restrict__ y,
                                                  int64_t nelem, int64_t group_off, int64_t spatial,
                                                  const DType* __restrict__ weight,
                                                  const DType* __restrict__ bias, int64_t ch_base,
                                                  float mean, float rstd, bool vec_aligned) {
  const int64_t nvec = vec_aligned ? nelem / kVec : 0;
  // Fast path: the whole tile lies in ONE channel. Common for the large path
  // (a chunk << spatial); loads the affine once and drops the per-vector
  // int64 channel division (the dominant compute cost found by the prior
  // round's profiling).
  if (group_off / spatial == (group_off + nelem - 1) / spatial) {
    const int64_t c = group_off / spatial;
    const float w = to_f32<DType>(weight[ch_base + c]);
    const float bb = to_f32<DType>(bias[ch_base + c]);
    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
      Pack<DType, kVec> v;
      v.load(x, vi);
      Pack<DType, kVec> o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o[j] = from_f32<DType>(siluf((to_f32<DType>(v[j]) - mean) * rstd * w + bb));
      }
      o.store(y, vi);
    }
    for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
      y[i] = from_f32<DType>(siluf((to_f32<DType>(x[i]) - mean) * rstd * w + bb));
    }
    return;
  }
  for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
    const int64_t i0 = group_off + vi * kVec;
    Pack<DType, kVec> v;
    v.load(x, vi);
    Pack<DType, kVec> o;
    const int64_t c0 = i0 / spatial;
    const int64_t c1 = (i0 + kVec - 1) / spatial;
    if (c0 == c1) {  // whole vector within one channel -> scalar affine
      const float w = to_f32<DType>(weight[ch_base + c0]);
      const float bb = to_f32<DType>(bias[ch_base + c0]);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o[j] = from_f32<DType>(siluf((to_f32<DType>(v[j]) - mean) * rstd * w + bb));
      }
    } else {  // straddles a channel boundary -> per-lane affine
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int64_t c = (i0 + j) / spatial;
        const float w = to_f32<DType>(weight[ch_base + c]);
        const float bb = to_f32<DType>(bias[ch_base + c]);
        o[j] = from_f32<DType>(siluf((to_f32<DType>(v[j]) - mean) * rstd * w + bb));
      }
    }
    o.store(y, vi);
  }
  for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
    const int64_t c = (group_off + i) / spatial;
    const float w = to_f32<DType>(weight[ch_base + c]);
    const float bb = to_f32<DType>(bias[ch_base + c]);
    y[i] = from_f32<DType>(siluf((to_f32<DType>(x[i]) - mean) * rstd * w + bb));
  }
}

// ---------------- small path: one CTA per (batch, group) ----------------
// kThreads is selectable: 256 for tiny groups (launch-bound), 1024 for the
// crossover band just under the chunked threshold, where one CTA per group
// otherwise starves per-SM memory parallelism (more resident loads per SM).
template <typename DType, int kThreads = kBlockThreads>
__global__ void gns_one_pass_kernel(const GnsParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));  // fp16/bf16 -> 8, fp32 -> 4
  constexpr int kWarps = kThreads / kWarpThreads;
  __shared__ float smem[2 * kWarps + 2];

  const int64_t row = blockIdx.x;  // grid == num_rows
  const int64_t b = row / p.num_groups;
  const int64_t g = row - b * p.num_groups;
  const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
  const int64_t ch_base = g * p.channels_per_group;
  const int64_t n = p.group_size;

  const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base;
  DType* __restrict__ y = static_cast<DType*>(p.y) + group_base;
  const DType* __restrict__ weight = static_cast<const DType*>(p.weight);
  const DType* __restrict__ bias = static_cast<const DType*>(p.bias);

  const bool vec_ok = (n % kVec == 0) && ((group_base % kVec) == 0);

  float lsum = 0.0f, lsumsq = 0.0f;
  accumulate_stats<DType, kVec>(x, n, vec_ok, lsum, lsumsq);
  block_reduce2<kWarps>(lsum, lsumsq, smem);

  const float inv_n = 1.0f / static_cast<float>(n);
  const float mean = lsum * inv_n;
  const float var = fmaxf(lsumsq * inv_n - mean * mean, 0.0f);
  const float rstd = rsqrtf(var + p.eps);

  apply_affine_silu<DType, kVec>(x, y, n, 0, p.spatial, weight, bias, ch_base, mean, rstd, vec_ok);
}

// ---------------- large path: three-kernel pipeline ----------------
template <typename DType>
__global__ void gns_stats_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  __shared__ float smem[2 * kWarpsPerBlock + 2];
  const int64_t total_tasks = p.num_rows * p.chunks_per_row;
  float* __restrict__ psum = static_cast<float*>(p.partial_sum);
  float* __restrict__ psumsq = static_cast<float*>(p.partial_sumsq);

  for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
    const int64_t row = task / p.chunks_per_row;
    const int64_t chunk = task - row * p.chunks_per_row;
    const int64_t b = row / p.num_groups;
    const int64_t g = row - b * p.num_groups;
    const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
    const int64_t chunk_start = chunk * p.chunk_elems;
    const int64_t chunk_end =
        (chunk_start + p.chunk_elems < p.group_size) ? (chunk_start + p.chunk_elems) : p.group_size;
    const int64_t nelem = chunk_end - chunk_start;
    const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base + chunk_start;
    const bool vec_ok = ((group_base + chunk_start) % kVec) == 0;

    float lsum = 0.0f, lsumsq = 0.0f;
    accumulate_stats<DType, kVec>(x, nelem, vec_ok, lsum, lsumsq);
    block_reduce2(lsum, lsumsq, smem);
    if (threadIdx.x == 0) {
      psum[task] = lsum;
      psumsq[task] = lsumsq;
    }
    __syncthreads();  // smem reuse barrier before next task
  }
}

template <typename DType>
__global__ void gns_finalize_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  __shared__ float smem[2 * kWarpsPerBlock + 2];
  const int64_t row = blockIdx.x;  // grid == num_rows
  const int64_t base = row * p.chunks_per_row;
  const float* __restrict__ psum = static_cast<const float*>(p.partial_sum);
  const float* __restrict__ psumsq = static_cast<const float*>(p.partial_sumsq);

  float lsum = 0.0f, lsumsq = 0.0f;
  for (int64_t c = threadIdx.x; c < p.chunks_per_row; c += blockDim.x) {
    lsum += psum[base + c];
    lsumsq += psumsq[base + c];
  }
  block_reduce2(lsum, lsumsq, smem);
  if (threadIdx.x == 0) {
    const float inv_n = 1.0f / static_cast<float>(p.group_size);
    const float mean = lsum * inv_n;
    const float var = fmaxf(lsumsq * inv_n - mean * mean, 0.0f);
    static_cast<float*>(p.mean)[row] = mean;
    static_cast<float*>(p.rstd)[row] = rsqrtf(var + p.eps);
  }
}

template <typename DType>
__global__ void gns_apply_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  const int64_t total_tasks = p.num_rows * p.chunks_per_row;
  const float* __restrict__ mean_arr = static_cast<const float*>(p.mean);
  const float* __restrict__ rstd_arr = static_cast<const float*>(p.rstd);

  for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
    const int64_t row = task / p.chunks_per_row;
    const int64_t chunk = task - row * p.chunks_per_row;
    const int64_t b = row / p.num_groups;
    const int64_t g = row - b * p.num_groups;
    const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
    const int64_t ch_base = g * p.channels_per_group;
    const float mean = mean_arr[row];
    const float rstd = rstd_arr[row];
    const int64_t chunk_start = chunk * p.chunk_elems;
    const int64_t chunk_end =
        (chunk_start + p.chunk_elems < p.group_size) ? (chunk_start + p.chunk_elems) : p.group_size;
    const int64_t nelem = chunk_end - chunk_start;
    const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base + chunk_start;
    DType* __restrict__ y = static_cast<DType*>(p.y) + group_base + chunk_start;
    const DType* __restrict__ weight = static_cast<const DType*>(p.weight);
    const DType* __restrict__ bias = static_cast<const DType*>(p.bias);
    const bool vec_ok = ((group_base + chunk_start) % kVec) == 0;

    apply_affine_silu<DType, kVec>(x, y, nelem, chunk_start, p.spatial, weight, bias, ch_base, mean,
                                   rstd, vec_ok);
  }
}

// ---------------- giant path: register-lean exact-grid pipeline ----------------
// Profiling of the chunked path on the production giant shapes showed the
// generic apply kernel at 52 regs/thread -> 4 blocks/SM -> ~44% occupancy and
// ~41% DRAM, while the stats kernel (32 regs, 100% theoretical occupancy) ran
// ~68% DRAM. These variants trade the persistent grid-stride loop for one
// task per CTA and cap the register budget at the H200 full-occupancy
// boundary (64K regs/SM / 2048 threads = 32 regs/thread), with a single
// hoisted-affine fast loop for tiles that stay inside one channel (the giant
// regime: chunk << spatial) and a compact per-element loop for the rare
// channel-straddling tile.

template <typename DType>
__global__ void __launch_bounds__(kBlockThreads, 8)
    gns_giant_stats_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  __shared__ float smem[2 * kWarpsPerBlock + 2];
  const int64_t task = blockIdx.x;  // exact grid: one task per CTA
  float* __restrict__ psum = static_cast<float*>(p.partial_sum);
  float* __restrict__ psumsq = static_cast<float*>(p.partial_sumsq);
  const int64_t row = task / p.chunks_per_row;
  const int64_t chunk = task - row * p.chunks_per_row;
  const int64_t b = row / p.num_groups;
  const int64_t g = row - b * p.num_groups;
  const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
  const int64_t chunk_start = chunk * p.chunk_elems;
  const int64_t chunk_end =
      (chunk_start + p.chunk_elems < p.group_size) ? (chunk_start + p.chunk_elems) : p.group_size;
  const int64_t nelem = chunk_end - chunk_start;
  const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base + chunk_start;
  const bool vec_ok = ((group_base + chunk_start) % kVec) == 0;

  float lsum = 0.0f, lsumsq = 0.0f;
  // x has no reuse inside this kernel (the apply kernel re-reads it from
  // DRAM at these sizes anyway): streaming loads keep L2 for the partials.
  accumulate_stats<DType, kVec, /*kStream=*/true>(x, nelem, vec_ok, lsum, lsumsq);
  block_reduce2(lsum, lsumsq, smem);

  // Last-arriving CTA of each row folds the per-row finalize in here (saves
  // the separate finalize launch). Publication order: partials first, fence,
  // then the arrival counter; the last arrival therefore observes every
  // partial of its row. The counter self-cleans to zero for the next call.
  if (threadIdx.x == 0) {
    psum[task] = lsum;
    psumsq[task] = lsumsq;
    __threadfence();
    const int prev = atomicAdd(reinterpret_cast<int*>(p.row_counter) + row, 1);
    smem[0] = (prev == p.chunks_per_row - 1) ? 1.0f : 0.0f;
  }
  __syncthreads();
  const bool last_of_row = smem[0] != 0.0f;
  if (last_of_row) {
    const int64_t base = row * p.chunks_per_row;
    float s = 0.0f, q = 0.0f;
    for (int64_t c = threadIdx.x; c < p.chunks_per_row; c += blockDim.x) {
      s += psum[base + c];
      q += psumsq[base + c];
    }
    block_reduce2(s, q, smem);  // fixed-order tree: deterministic
    if (threadIdx.x == 0) {
      const float inv_n = 1.0f / static_cast<float>(p.group_size);
      const float mu = s * inv_n;
      const float var = fmaxf(q * inv_n - mu * mu, 0.0f);
      static_cast<float*>(p.mean)[row] = mu;
      static_cast<float*>(p.rstd)[row] = rsqrtf(var + p.eps);
      reinterpret_cast<int*>(p.row_counter)[row] = 0;
    }
  }
}

template <typename DType>
__global__ void __launch_bounds__(kBlockThreads, 8)
    gns_giant_apply_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  const int64_t task = blockIdx.x;  // exact grid: one task per CTA
  const DType* __restrict__ weight = static_cast<const DType*>(p.weight);
  const DType* __restrict__ bias = static_cast<const DType*>(p.bias);
  const int64_t row = task / p.chunks_per_row;
  const int64_t chunk = task - row * p.chunks_per_row;
  const int64_t b = row / p.num_groups;
  const int64_t g = row - b * p.num_groups;
  const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
  const int64_t ch_base = g * p.channels_per_group;
  const float mean = static_cast<const float*>(p.mean)[row];
  const float rstd = static_cast<const float*>(p.rstd)[row];
  const int64_t chunk_start = chunk * p.chunk_elems;
  const int64_t chunk_end =
      (chunk_start + p.chunk_elems < p.group_size) ? (chunk_start + p.chunk_elems) : p.group_size;
  const int64_t nelem = chunk_end - chunk_start;
  const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base + chunk_start;
  DType* __restrict__ y = static_cast<DType*>(p.y) + group_base + chunk_start;

  // A tile spans at most two channels in the giant regime (chunk_elems <=
  // spatial whenever group_size >= the giant threshold), so process the tile
  // as one or two single-channel segments, each with hoisted affine and a
  // vector stream. Segment boundaries are uniform across the CTA (no
  // divergence); segment starts inherit channel-boundary alignment whenever
  // spatial is a multiple of the vector width.
  int64_t seg_start = 0;
  while (seg_start < nelem) {
    const int64_t c = (chunk_start + seg_start) / p.spatial;
    const int64_t channel_end = (c + 1) * p.spatial - chunk_start;
    const int64_t seg_end = channel_end < nelem ? channel_end : nelem;
    const int64_t seg_len = seg_end - seg_start;
    const float w = to_f32<DType>(weight[ch_base + c]);
    const float bb = to_f32<DType>(bias[ch_base + c]);
    const float scale = rstd * w;
    const float shift = bb - mean * scale;
    const DType* __restrict__ xs = x + seg_start;
    DType* __restrict__ ys = y + seg_start;
    const bool seg_vec =
        (((group_base + chunk_start + seg_start) % kVec) == 0) && ((seg_len % kVec) == 0);
    if (seg_vec) {
      // x is a last-use read here and y is written once and not re-read:
      // streaming hints on both sides of the stream.
      const int64_t nvec = seg_len / kVec;
      for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
        Pack<DType, kVec> v;
        v.load_streaming(xs, vi);
        Pack<DType, kVec> o;
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          o[j] = from_f32<DType>(siluf(to_f32<DType>(v[j]) * scale + shift));
        }
        o.store_streaming(ys, vi);
      }
    } else {
      for (int64_t i = threadIdx.x; i < seg_len; i += blockDim.x) {
        ys[i] = from_f32<DType>(siluf(to_f32<DType>(xs[i]) * scale + shift));
      }
    }
    seg_start = seg_end;
  }
}

// ---------------- clean-giant path: channel-aligned tiles ----------------
// Specialization for giant groups whose per-channel spatial extent is an
// exact multiple of the tile size (the host gates on spatial % tile == 0 and
// group_size % tile == 0). Every CTA tile then lies inside one channel by
// construction: the apply kernel hoists the affine once and runs a single
// branch-free vector stream (this mirrors the structural edge the copied
// baseline's hoisted-affine apply has on exactly this class), and the stats
// kernel drops the tail/alignment handling. Same fused deterministic
// last-block finalize as the generic giant path.

template <typename DType>
__global__ void __launch_bounds__(kBlockThreads, 8)
    gns_clean_stats_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  __shared__ float smem[2 * kWarpsPerBlock + 2];
  const int64_t task = blockIdx.x;  // exact grid: one tile per CTA
  float* __restrict__ psum = static_cast<float*>(p.partial_sum);
  float* __restrict__ psumsq = static_cast<float*>(p.partial_sumsq);
  const int64_t row = task / p.chunks_per_row;
  const int64_t chunk = task - row * p.chunks_per_row;
  const int64_t b = row / p.num_groups;
  const int64_t g = row - b * p.num_groups;
  const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
  const DType* __restrict__ x =
      static_cast<const DType*>(p.x) + group_base + chunk * p.chunk_elems;

  float lsum = 0.0f, lsumsq = 0.0f;
  // Tiles are whole and 16B-aligned by construction; streaming loads (no
  // reuse inside this kernel). Two independent accumulator pairs break the
  // FADD dependency chain.
  const int64_t nvec = p.chunk_elems / kVec;
  float s0 = 0.0f, q0 = 0.0f, s1 = 0.0f, q1 = 0.0f;
  int64_t vi = threadIdx.x;
  for (; vi + blockDim.x < nvec; vi += 2 * blockDim.x) {
    Pack<DType, kVec> a;
    Pack<DType, kVec> c;
    a.load_streaming(x, vi);
    c.load_streaming(x, vi + blockDim.x);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xa = to_f32<DType>(a[j]);
      const float xc = to_f32<DType>(c[j]);
      s0 += xa;
      q0 += xa * xa;
      s1 += xc;
      q1 += xc * xc;
    }
  }
  for (; vi < nvec; vi += blockDim.x) {
    Pack<DType, kVec> v;
    v.load_streaming(x, vi);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xf = to_f32<DType>(v[j]);
      s0 += xf;
      q0 += xf * xf;
    }
  }
  lsum = s0 + s1;
  lsumsq = q0 + q1;
  block_reduce2(lsum, lsumsq, smem);

  if (threadIdx.x == 0) {
    psum[task] = lsum;
    psumsq[task] = lsumsq;
    __threadfence();
    const int prev = atomicAdd(reinterpret_cast<int*>(p.row_counter) + row, 1);
    smem[0] = (prev == p.chunks_per_row - 1) ? 1.0f : 0.0f;
  }
  __syncthreads();
  if (smem[0] != 0.0f) {
    const int64_t base = row * p.chunks_per_row;
    float s = 0.0f, q = 0.0f;
    for (int64_t c = threadIdx.x; c < p.chunks_per_row; c += blockDim.x) {
      s += psum[base + c];
      q += psumsq[base + c];
    }
    block_reduce2(s, q, smem);  // fixed-order tree: deterministic
    if (threadIdx.x == 0) {
      const float inv_n = 1.0f / static_cast<float>(p.group_size);
      const float mu = s * inv_n;
      const float var = fmaxf(q * inv_n - mu * mu, 0.0f);
      static_cast<float*>(p.mean)[row] = mu;
      static_cast<float*>(p.rstd)[row] = rsqrtf(var + p.eps);
      reinterpret_cast<int*>(p.row_counter)[row] = 0;
    }
  }
}

template <typename DType>
__global__ void __launch_bounds__(kBlockThreads, 8)
    gns_clean_apply_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  const int64_t task = blockIdx.x;  // exact grid: one tile per CTA
  const int64_t row = task / p.chunks_per_row;
  const int64_t chunk = task - row * p.chunks_per_row;
  const int64_t b = row / p.num_groups;
  const int64_t g = row - b * p.num_groups;
  const int64_t group_base = b * p.channels * p.spatial + g * p.group_size;
  const int64_t chunk_start = chunk * p.chunk_elems;
  const DType* __restrict__ x = static_cast<const DType*>(p.x) + group_base + chunk_start;
  DType* __restrict__ y = static_cast<DType*>(p.y) + group_base + chunk_start;

  // Whole tile inside one channel by construction: hoist the affine once.
  const int64_t c = g * p.channels_per_group + chunk_start / p.spatial;
  const float mean = static_cast<const float*>(p.mean)[row];
  const float rstd = static_cast<const float*>(p.rstd)[row];
  const float w = to_f32<DType>(static_cast<const DType*>(p.weight)[c]);
  const float bb = to_f32<DType>(static_cast<const DType*>(p.bias)[c]);
  const float scale = rstd * w;
  const float shift = bb - mean * scale;

  const int64_t nvec = p.chunk_elems / kVec;
  for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
    Pack<DType, kVec> v;
    v.load_streaming(x, vi);
    Pack<DType, kVec> o;
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      o[j] = from_f32<DType>(siluf(to_f32<DType>(v[j]) * scale + shift));
    }
    o.store_streaming(y, vi);
  }
}

// ---------------------------------------------------------------------------
// Host-side validation, dtype dispatch, launches.
// ---------------------------------------------------------------------------

void check(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error("group_norm_silu_candidate: " + msg);
}

bool same_dtype(const DLDataType& a, const DLDataType& b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

struct Shape3D {
  int64_t batch;
  int64_t channels;
  int64_t spatial;
};

Shape3D validate_common(const TensorView& x, const TensorView& weight, const TensorView& bias,
                        const TensorView& y, int64_t num_groups) {
  check(x.ndim() == 3, "x must be a 3-D [B, C, spatial] view");
  check(y.ndim() == 3, "y must be a 3-D [B, C, spatial] view");
  check(weight.ndim() == 1 && bias.ndim() == 1, "weight/bias must be 1-D");
  check(x.device().device_type == kDLCUDA && y.device().device_type == kDLCUDA,
        "x/y must be CUDA tensors");
  const DLDataType dt = x.dtype();
  check(same_dtype(dt, y.dtype()) && same_dtype(dt, weight.dtype()) && same_dtype(dt, bias.dtype()),
        "x/weight/bias/y dtypes must match");
  const int64_t batch = x.size(0);
  const int64_t channels = x.size(1);
  const int64_t spatial = x.size(2);
  for (int i = 0; i < 3; ++i) check(x.size(i) == y.size(i), "x/y shapes must match");
  check(weight.size(0) == channels && bias.size(0) == channels,
        "weight/bias must have C elements");
  check(num_groups > 0 && channels % num_groups == 0, "channels must be divisible by num_groups");
  // Compact row-major layout required (binding normalizes inputs beforehand;
  // torch-exported DLPack tensors always carry strides).
  check(x.stride(2) == 1 && x.stride(1) == spatial && x.stride(0) == channels * spatial,
        "x must be contiguous");
  check(y.stride(2) == 1 && y.stride(1) == spatial && y.stride(0) == channels * spatial,
        "y must be contiguous");
  return Shape3D{batch, channels, spatial};
}

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

void launch_check(const char* what) {
  const cudaError_t err = cudaGetLastError();
  check(err == cudaSuccess, std::string(what) + " launch failed: " + cudaGetErrorString(err));
}

// Group-size boundary where the one-CTA-per-group kernel switches to wide
// (1024-thread) blocks: with few resident CTAs, per-SM memory parallelism is
// the limiter, and wider blocks quadruple the outstanding loads per SM.
constexpr int64_t kSmallWideThreshold = 32768;

template <typename DType>
void run_small_typed(const TensorView& x, const TensorView& weight, const TensorView& bias,
                     const TensorView& y, int64_t num_groups, double eps, const Shape3D& s) {
  const int64_t channels_per_group = s.channels / num_groups;
  const GnsParams<DType> params{
      x.data_ptr(),
      weight.data_ptr(),
      bias.data_ptr(),
      y.data_ptr(),
      s.channels,
      s.spatial,
      num_groups,
      channels_per_group,
      channels_per_group * s.spatial,
      s.batch * num_groups,
      static_cast<float>(eps),
  };
  if (params.group_size >= kSmallWideThreshold) {
    gns_one_pass_kernel<DType, 1024>
        <<<static_cast<uint32_t>(params.num_rows), 1024, 0, current_stream()>>>(params);
  } else {
    gns_one_pass_kernel<DType, kBlockThreads>
        <<<static_cast<uint32_t>(params.num_rows), kBlockThreads, 0, current_stream()>>>(params);
  }
  launch_check("gns_one_pass_kernel");
}

int blocks_per_sm(const void* kernel) {
  int num_blocks = 0;
  const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, kernel, kBlockThreads, 0);
  check(err == cudaSuccess, "occupancy query failed");
  return num_blocks > 0 ? num_blocks : 1;
}

int sm_count() {
  int device = 0;
  check(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
  int count = 0;
  check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) == cudaSuccess,
        "SM count query failed");
  return count;
}

template <typename DType>
void run_large_typed(const TensorView& x, const TensorView& weight, const TensorView& bias,
                     const TensorView& y, const TensorView& partial_sum,
                     const TensorView& partial_sumsq, const TensorView& mean,
                     const TensorView& rstd, int64_t num_groups, double eps, const Shape3D& s) {
  const int64_t channels_per_group = s.channels / num_groups;
  const int64_t group_size = channels_per_group * s.spatial;
  const int64_t num_rows = s.batch * num_groups;
  const int64_t chunks_per_row = (group_size + kChunkElems - 1) / kChunkElems;
  const int64_t total_tasks = num_rows * chunks_per_row;

  const DLDataType f32{kDLFloat, 32, 1};
  check(same_dtype(partial_sum.dtype(), f32) && same_dtype(partial_sumsq.dtype(), f32) &&
            same_dtype(mean.dtype(), f32) && same_dtype(rstd.dtype(), f32),
        "scratch tensors must be fp32");
  check(partial_sum.ndim() == 1 && partial_sum.size(0) >= total_tasks,
        "partial_sum scratch too small");
  check(partial_sumsq.ndim() == 1 && partial_sumsq.size(0) >= total_tasks,
        "partial_sumsq scratch too small");
  check(mean.ndim() == 1 && mean.size(0) >= num_rows, "mean scratch too small");
  check(rstd.ndim() == 1 && rstd.size(0) >= num_rows, "rstd scratch too small");

  const GnsLargeParams<DType> params{
      x.data_ptr(),
      weight.data_ptr(),
      bias.data_ptr(),
      y.data_ptr(),
      partial_sum.data_ptr(),
      partial_sumsq.data_ptr(),
      mean.data_ptr(),
      rstd.data_ptr(),
      nullptr,  // row_counter: giant path only
      s.channels,
      s.spatial,
      num_groups,
      channels_per_group,
      group_size,
      num_rows,
      kChunkElems,
      chunks_per_row,
      static_cast<float>(eps),
  };
  cudaStream_t stream = current_stream();
  const int sms = sm_count();

  auto* stats_k = gns_stats_kernel<DType>;
  const int64_t grid_s = std::min<int64_t>(
      total_tasks, static_cast<int64_t>(blocks_per_sm(reinterpret_cast<const void*>(stats_k))) * sms);
  stats_k<<<static_cast<uint32_t>(grid_s), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_stats_kernel");

  gns_finalize_kernel<DType>
      <<<static_cast<uint32_t>(num_rows), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_finalize_kernel");

  auto* apply_k = gns_apply_kernel<DType>;
  const int64_t grid_a = std::min<int64_t>(
      total_tasks, static_cast<int64_t>(blocks_per_sm(reinterpret_cast<const void*>(apply_k))) * sms);
  apply_k<<<static_cast<uint32_t>(grid_a), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_apply_kernel");
}

template <typename DType>
void run_giant_typed(const TensorView& x, const TensorView& weight, const TensorView& bias,
                     const TensorView& y, const TensorView& partial_sum,
                     const TensorView& partial_sumsq, const TensorView& mean,
                     const TensorView& rstd, const TensorView& row_counter, int64_t num_groups,
                     double eps, int64_t stats_chunk_elems, int64_t apply_chunk_elems,
                     const Shape3D& s) {
  const int64_t channels_per_group = s.channels / num_groups;
  const int64_t group_size = channels_per_group * s.spatial;
  const int64_t num_rows = s.batch * num_groups;
  check(stats_chunk_elems > 0 && stats_chunk_elems % 8 == 0,
        "stats_chunk_elems must be a positive multiple of 8");
  check(apply_chunk_elems > 0 && apply_chunk_elems % 8 == 0,
        "apply_chunk_elems must be a positive multiple of 8");
  // The two pipeline stages are independently tiled: partial sums and the
  // fused finalize are keyed by the stats tiling only (the apply kernel reads
  // just mean/rstd), so each kernel gets the tile size that suits it.
  const int64_t stats_chunks = (group_size + stats_chunk_elems - 1) / stats_chunk_elems;
  const int64_t stats_tasks = num_rows * stats_chunks;
  const int64_t apply_chunks = (group_size + apply_chunk_elems - 1) / apply_chunk_elems;
  const int64_t apply_tasks = num_rows * apply_chunks;
  check(stats_tasks <= 0x7fffffff && apply_tasks <= 0x7fffffff, "grid too large");

  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType i32{kDLInt, 32, 1};
  check(same_dtype(partial_sum.dtype(), f32) && same_dtype(partial_sumsq.dtype(), f32) &&
            same_dtype(mean.dtype(), f32) && same_dtype(rstd.dtype(), f32),
        "scratch tensors must be fp32");
  check(same_dtype(row_counter.dtype(), i32), "row_counter must be int32");
  check(partial_sum.ndim() == 1 && partial_sum.size(0) >= stats_tasks,
        "partial_sum scratch too small");
  check(partial_sumsq.ndim() == 1 && partial_sumsq.size(0) >= stats_tasks,
        "partial_sumsq scratch too small");
  check(mean.ndim() == 1 && mean.size(0) >= num_rows, "mean scratch too small");
  check(rstd.ndim() == 1 && rstd.size(0) >= num_rows, "rstd scratch too small");
  check(row_counter.ndim() == 1 && row_counter.size(0) >= num_rows, "row_counter too small");

  GnsLargeParams<DType> params{
      x.data_ptr(),
      weight.data_ptr(),
      bias.data_ptr(),
      y.data_ptr(),
      partial_sum.data_ptr(),
      partial_sumsq.data_ptr(),
      mean.data_ptr(),
      rstd.data_ptr(),
      row_counter.data_ptr(),
      s.channels,
      s.spatial,
      num_groups,
      channels_per_group,
      group_size,
      num_rows,
      stats_chunk_elems,
      stats_chunks,
      static_cast<float>(eps),
  };
  cudaStream_t stream = current_stream();

  // Two launches: stats folds the per-row finalize into its last-arriving CTA.
  // Exact one-task-per-CTA grids measured faster than stride loops here (the
  // loop state pushed the kernels over the 32-reg full-occupancy boundary).
  gns_giant_stats_kernel<DType>
      <<<static_cast<uint32_t>(stats_tasks), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_giant_stats_kernel");
  params.chunk_elems = apply_chunk_elems;
  params.chunks_per_row = apply_chunks;
  gns_giant_apply_kernel<DType>
      <<<static_cast<uint32_t>(apply_tasks), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_giant_apply_kernel");
}

template <typename DType>
void run_clean_giant_typed(const TensorView& x, const TensorView& weight, const TensorView& bias,
                           const TensorView& y, const TensorView& partial_sum,
                           const TensorView& partial_sumsq, const TensorView& mean,
                           const TensorView& rstd, const TensorView& row_counter,
                           int64_t num_groups, double eps, int64_t chunk_elems,
                           const Shape3D& s) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  const int64_t channels_per_group = s.channels / num_groups;
  const int64_t group_size = channels_per_group * s.spatial;
  const int64_t num_rows = s.batch * num_groups;
  check(chunk_elems > 0 && chunk_elems % kVec == 0,
        "clean chunk_elems must be a positive multiple of the vector width");
  check(s.spatial % chunk_elems == 0,
        "clean-giant route requires spatial to be a multiple of chunk_elems");
  const int64_t chunks_per_row = group_size / chunk_elems;  // exact by the checks above
  const int64_t total_tasks = num_rows * chunks_per_row;
  check(total_tasks <= 0x7fffffff, "grid too large");

  const DLDataType f32{kDLFloat, 32, 1};
  const DLDataType i32{kDLInt, 32, 1};
  check(same_dtype(partial_sum.dtype(), f32) && same_dtype(partial_sumsq.dtype(), f32) &&
            same_dtype(mean.dtype(), f32) && same_dtype(rstd.dtype(), f32),
        "scratch tensors must be fp32");
  check(same_dtype(row_counter.dtype(), i32), "row_counter must be int32");
  check(partial_sum.ndim() == 1 && partial_sum.size(0) >= total_tasks,
        "partial_sum scratch too small");
  check(partial_sumsq.ndim() == 1 && partial_sumsq.size(0) >= total_tasks,
        "partial_sumsq scratch too small");
  check(mean.ndim() == 1 && mean.size(0) >= num_rows, "mean scratch too small");
  check(rstd.ndim() == 1 && rstd.size(0) >= num_rows, "rstd scratch too small");
  check(row_counter.ndim() == 1 && row_counter.size(0) >= num_rows, "row_counter too small");

  const GnsLargeParams<DType> params{
      x.data_ptr(),
      weight.data_ptr(),
      bias.data_ptr(),
      y.data_ptr(),
      partial_sum.data_ptr(),
      partial_sumsq.data_ptr(),
      mean.data_ptr(),
      rstd.data_ptr(),
      row_counter.data_ptr(),
      s.channels,
      s.spatial,
      num_groups,
      channels_per_group,
      group_size,
      num_rows,
      chunk_elems,
      chunks_per_row,
      static_cast<float>(eps),
  };
  cudaStream_t stream = current_stream();
  gns_clean_stats_kernel<DType>
      <<<static_cast<uint32_t>(total_tasks), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_clean_stats_kernel");
  gns_clean_apply_kernel<DType>
      <<<static_cast<uint32_t>(total_tasks), kBlockThreads, 0, stream>>>(params);
  launch_check("gns_clean_apply_kernel");
}

enum class Kind { kF16, kBF16, kF32 };

Kind dtype_kind(const DLDataType& dt) {
  if (dt.lanes == 1 && dt.code == kDLFloat && dt.bits == 16) return Kind::kF16;
  if (dt.lanes == 1 && dt.code == kDLBfloat && dt.bits == 16) return Kind::kBF16;
  if (dt.lanes == 1 && dt.code == kDLFloat && dt.bits == 32) return Kind::kF32;
  throw std::runtime_error("group_norm_silu_candidate: unsupported dtype (fp16/bf16/fp32 only)");
}

}  // namespace

// Exported entry points. solution/binding.py unifies them behind
// `group_norm_silu_candidate` and owns the small/large dispatch threshold.
void gns_candidate_small(TensorView x, TensorView weight, TensorView bias, int64_t num_groups,
                         double eps, TensorView y) {
  const Shape3D s = validate_common(x, weight, bias, y, num_groups);
  switch (dtype_kind(x.dtype())) {
    case Kind::kF16:
      run_small_typed<__half>(x, weight, bias, y, num_groups, eps, s);
      break;
    case Kind::kBF16:
      run_small_typed<__nv_bfloat16>(x, weight, bias, y, num_groups, eps, s);
      break;
    case Kind::kF32:
      run_small_typed<float>(x, weight, bias, y, num_groups, eps, s);
      break;
  }
}

void gns_candidate_large(TensorView x, TensorView weight, TensorView bias, TensorView partial_sum,
                         TensorView partial_sumsq, TensorView mean, TensorView rstd,
                         int64_t num_groups, double eps, TensorView y) {
  const Shape3D s = validate_common(x, weight, bias, y, num_groups);
  switch (dtype_kind(x.dtype())) {
    case Kind::kF16:
      run_large_typed<__half>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                              num_groups, eps, s);
      break;
    case Kind::kBF16:
      run_large_typed<__nv_bfloat16>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                                     num_groups, eps, s);
      break;
    case Kind::kF32:
      run_large_typed<float>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd, num_groups,
                             eps, s);
      break;
  }
}

void gns_candidate_giant(TensorView x, TensorView weight, TensorView bias, TensorView partial_sum,
                         TensorView partial_sumsq, TensorView mean, TensorView rstd,
                         TensorView row_counter, int64_t num_groups, double eps,
                         int64_t stats_chunk_elems, int64_t apply_chunk_elems, TensorView y) {
  const Shape3D s = validate_common(x, weight, bias, y, num_groups);
  switch (dtype_kind(x.dtype())) {
    case Kind::kF16:
      run_giant_typed<__half>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                              row_counter, num_groups, eps, stats_chunk_elems, apply_chunk_elems,
                              s);
      break;
    case Kind::kBF16:
      run_giant_typed<__nv_bfloat16>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                                     row_counter, num_groups, eps, stats_chunk_elems,
                                     apply_chunk_elems, s);
      break;
    case Kind::kF32:
      run_giant_typed<float>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                             row_counter, num_groups, eps, stats_chunk_elems, apply_chunk_elems,
                             s);
      break;
  }
}

void gns_candidate_clean_giant(TensorView x, TensorView weight, TensorView bias,
                               TensorView partial_sum, TensorView partial_sumsq, TensorView mean,
                               TensorView rstd, TensorView row_counter, int64_t num_groups,
                               double eps, int64_t chunk_elems, TensorView y) {
  const Shape3D s = validate_common(x, weight, bias, y, num_groups);
  switch (dtype_kind(x.dtype())) {
    case Kind::kF16:
      run_clean_giant_typed<__half>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                                    row_counter, num_groups, eps, chunk_elems, s);
      break;
    case Kind::kBF16:
      run_clean_giant_typed<__nv_bfloat16>(x, weight, bias, y, partial_sum, partial_sumsq, mean,
                                           rstd, row_counter, num_groups, eps, chunk_elems, s);
      break;
    case Kind::kF32:
      run_clean_giant_typed<float>(x, weight, bias, y, partial_sum, partial_sumsq, mean, rstd,
                                   row_counter, num_groups, eps, chunk_elems, s);
      break;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gns_candidate_small, gns_candidate_small);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gns_candidate_large, gns_candidate_large);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gns_candidate_giant, gns_candidate_giant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gns_candidate_clean_giant, gns_candidate_clean_giant);
