// Fused GroupNorm + SiLU candidate kernels for NVIDIA B200 (sm_100).
//
// Semantics (matches the copied upstream Triton baseline):
//   y = silu(group_norm(x, num_groups, weight, bias, eps))
//   mean/var per (batch, group) over (C/G) * prod(spatial) elements, computed
//   in fp32 with var = E[x^2] - E[x]^2 (clamped >= 0); per-channel affine;
//   silu(t) = t * sigmoid(t). Output is written CONTIGUOUS (NC...) regardless
//   of input layout — identical to the baseline, which materializes
//   x.contiguous() and returns a contiguous tensor.
//
// Regimes (host-side selection, thresholds env-tunable for crossover sweeps):
//   - generic     : strided two-pass, one CTA per (batch, group). Correct for
//                   every supported dtype/layout; safety net + fp32 path.
//   - cont_small  : contiguous 16-bit rows, one CTA per group, 16B-vectorized
//                   two-pass (small groups, launch/latency bound).
//   - cont_split  : contiguous 16-bit rows, split-group stats kernel with
//                   last-CTA finalize + vectorized apply kernel (fills the
//                   machine for medium/large groups; the upstream one-pass
//                   path only launches B*G = 32 CTAs at B=1).
//   - nchw_last   : channels-last(-3d) 16-bit rows read NATIVELY (position-
//                   major); per-CTA tiles accumulate all-group partial sums,
//                   last-CTA finalize, then an apply kernel stages normalized
//                   tiles in shared memory and writes the contiguous NC...
//                   output with coalesced channel-major runs. Skips the
//                   baseline's x.contiguous() materialization entirely.
//
// Inputs may be contiguous or arbitrarily strided; strided inputs are never
// materialized. No fast-math compile flags (contract). The fp32 generic path
// uses IEEE expf; the 16-bit production regimes use the SFU exp class
// (__expf) matching what the upstream Triton baseline's tl.sigmoid lowers to
// (see silu_fast below).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxDims = 5;
constexpr int kVecHalves = 8;  // 16 bytes of 16-bit elements

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}

__device__ __forceinline__ float silu(float t) { return t / (1.0f + expf(-t)); }

// The 16-bit production regimes use the same exp accuracy class as the
// upstream Triton baseline, whose tl.sigmoid lowers to the SFU exp2 path
// (ex2.approx) — NCU showed the IEEE expf sequence made the apply kernels
// instruction-throughput-bound (SM ~83% busy, DRAM ~15%). This is a per-call
// intrinsic choice, NOT a fast-math compile flag; the fp32 generic path keeps
// the IEEE form for the strict fp32 oracle gate.
__device__ __forceinline__ float silu_fast(float t) {
  return t / (1.0f + __expf(-t));
}

template <typename A>
__device__ __forceinline__ A warp_reduce_sum(A v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Block reduction of (sum, sumsq) pairs; result valid in thread 0, then
// broadcast through smem stats slots by the caller.
template <typename A>
struct BlockSums {
  A sum;
  A sumsq;
};

template <typename A>
__device__ __forceinline__ BlockSums<A> block_reduce_pair(
    A sum, A sumsq, A* smem /* >= 2*warps accumulators */) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  sum = warp_reduce_sum(sum);
  sumsq = warp_reduce_sum(sumsq);
  if (lane == 0) {
    smem[warp] = sum;
    smem[warps + warp] = sumsq;
  }
  __syncthreads();
  BlockSums<A> out{A(0), A(0)};
  if (warp == 0) {
    A s = (lane < warps) ? smem[lane] : A(0);
    A q = (lane < warps) ? smem[warps + lane] : A(0);
    s = warp_reduce_sum(s);
    q = warp_reduce_sum(q);
    out.sum = s;
    out.sumsq = q;
  }
  return out;
}

// Accumulator selection: fp32 inputs accumulate in double so the
// E[x^2]-E[x]^2 form stays accurate on adversarial inputs (offset / tiny
// variance); 16-bit inputs accumulate in fp32 — same numerics class as the
// upstream baseline. fp32 rows are correctness-grid only, never production.
template <typename T>
struct AccOf {
  using type = float;
};
template <>
struct AccOf<float> {
  using type = double;
};

// ---------------------------------------------------------------------------
// generic regime (any dtype/layout/rank) — one CTA per (batch, group)
// ---------------------------------------------------------------------------

struct StridedLayout {
  int64_t sizes[kMaxDims];    // [cpg, spatial dims...]
  int64_t strides[kMaxDims];  // input strides for those dims
  int ndim;
  int64_t batch_stride;
  int64_t channel_stride;
};

__device__ __forceinline__ int64_t strided_offset(
    int64_t idx, const StridedLayout& lay) {
  int64_t off = 0;
  int64_t rem = idx;
#pragma unroll
  for (int d = kMaxDims - 1; d >= 1; --d) {
    if (d < lay.ndim) {
      int64_t sz = lay.sizes[d];
      int64_t q = rem / sz;
      int64_t r = rem - q * sz;
      off += r * lay.strides[d];
      rem = q;
    }
  }
  off += rem * lay.strides[0];
  return off;
}

template <typename T>
__global__ void gns_generic_two_pass_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    StridedLayout lay,
    int64_t num_groups,
    int64_t channels_per_group,
    int64_t spatial,
    int64_t group_size,
    float eps) {
  using A = typename AccOf<T>::type;
  const int64_t group = blockIdx.x % num_groups;
  const int64_t batch = blockIdx.x / num_groups;
  const T* xg =
      x + batch * lay.batch_stride + group * channels_per_group * lay.channel_stride;
  T* og = out + (batch * num_groups + group) * group_size;

  __shared__ A smem[2 * (kBlockThreads / 32)];
  __shared__ float stats[2];

  A sum = A(0), sumsq = A(0);
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const A v = static_cast<A>(to_float<T>(xg[strided_offset(i, lay)]));
    sum += v;
    sumsq += v * v;
  }
  const BlockSums<A> tot = block_reduce_pair(sum, sumsq, smem);
  if (threadIdx.x == 0) {
    const A inv = A(1) / static_cast<A>(group_size);
    const A mean = tot.sum * inv;
    A var = tot.sumsq * inv - mean * mean;
    var = var < A(0) ? A(0) : var;
    stats[0] = static_cast<float>(mean);
    stats[1] = static_cast<float>(A(1) / sqrt(var + static_cast<A>(eps)));
  }
  __syncthreads();
  const float mean = stats[0];
  const float rstd = stats[1];

  const int64_t weight_base = group * channels_per_group;
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float v = to_float<T>(xg[strided_offset(i, lay)]);
    const int64_t ch = i / spatial;
    const float w = to_float<T>(weight[weight_base + ch]);
    const float b = to_float<T>(bias[weight_base + ch]);
    og[i] = from_float<T>(silu((v - mean) * rstd * w + b));
  }
}

// ---------------------------------------------------------------------------
// cont_small regime — contiguous 16-bit, one CTA per group, 16B vectors
// Preconditions (host-checked): x fully contiguous, 16-bit dtype,
// group_size % 8 == 0, spatial % 8 == 0, x/out group bases 16B-aligned.
// ---------------------------------------------------------------------------

template <typename T>
union Vec8 {
  uint4 raw;
  T elems[kVecHalves];
};

template <typename T>
__global__ void gns_cont_small_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int64_t num_groups,
    int64_t channels_per_group,
    int64_t spatial,
    int64_t group_size,
    float eps) {
  const int64_t group = blockIdx.x % num_groups;
  const int64_t batch = blockIdx.x / num_groups;
  const int64_t base = (batch * num_groups + group) * group_size;
  const uint4* xv = reinterpret_cast<const uint4*>(x + base);
  uint4* ov = reinterpret_cast<uint4*>(out + base);
  const int64_t nvec = group_size / kVecHalves;

  __shared__ float smem[2 * (kBlockThreads / 32) + 2];
  float* stats = smem + 2 * (kBlockThreads / 32);

  float sum = 0.0f, sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
    Vec8<T> v;
    v.raw = xv[i];
#pragma unroll
    for (int k = 0; k < kVecHalves; ++k) {
      const float f = to_float<T>(v.elems[k]);
      sum += f;
      sumsq += f * f;
    }
  }
  const BlockSums<float> tot = block_reduce_pair(sum, sumsq, smem);
  if (threadIdx.x == 0) {
    const float inv = 1.0f / static_cast<float>(group_size);
    const float mean = tot.sum * inv;
    float var = tot.sumsq * inv - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    stats[0] = mean;
    stats[1] = rsqrtf(var + eps);
  }
  __syncthreads();
  const float mean = stats[0];
  const float rstd = stats[1];

  const int64_t weight_base = group * channels_per_group;
  const int64_t spatial_vec = spatial / kVecHalves;  // spatial % 8 == 0
  for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
    Vec8<T> v;
    v.raw = xv[i];
    // spatial % 8 == 0 ⇒ one 8-wide vector never crosses a channel boundary.
    const int64_t ch = i / spatial_vec;
    const float w = to_float<T>(weight[weight_base + ch]);
    const float b = to_float<T>(bias[weight_base + ch]);
    Vec8<T> o;
#pragma unroll
    for (int k = 0; k < kVecHalves; ++k) {
      o.elems[k] = from_float<T>(silu_fast((to_float<T>(v.elems[k]) - mean) * rstd * w + b));
    }
    ov[i] = o.raw;
  }
}

// ---------------------------------------------------------------------------
// cont_split regime — contiguous 16-bit, medium/large groups.
// Stage 1: per-(row, chunk) CTAs reduce 16B-vectorized partial sums; the LAST
//          finishing CTA of each row reduces that row's partials (fixed order,
//          deterministic) into stats[row] = {mean, rstd}.
// Stage 2: per-(row, chunk) CTAs normalize + affine + silu and write out.
// Scratch: fp32 partials [rows*chunks*2], fp32 stats [rows*2],
//          u32 counters [rows]; allocated stream-ordered by the host.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void gns_split_stats_kernel(
    const T* __restrict__ x,
    float* __restrict__ partials,  // [rows][chunks][2]
    float* __restrict__ stats,     // [rows][2]
    unsigned int* __restrict__ counters,  // [rows]
    int64_t chunks,
    int64_t chunk_vecs,
    int64_t group_size,
    float eps) {
  const int64_t row = blockIdx.x;
  const int64_t chunk = blockIdx.y;
  const int64_t nvec = group_size / kVecHalves;
  const int64_t v0 = chunk * chunk_vecs;
  const int64_t v1 = (v0 + chunk_vecs < nvec) ? (v0 + chunk_vecs) : nvec;
  const uint4* xv = reinterpret_cast<const uint4*>(x + row * group_size);

  __shared__ float smem[2 * (kBlockThreads / 32) + 1];

  float sum = 0.0f, sumsq = 0.0f;
  for (int64_t i = v0 + threadIdx.x; i < v1; i += blockDim.x) {
    Vec8<T> v;
    v.raw = xv[i];
#pragma unroll
    for (int k = 0; k < kVecHalves; ++k) {
      const float f = to_float<T>(v.elems[k]);
      sum += f;
      sumsq += f * f;
    }
  }
  const BlockSums<float> tot = block_reduce_pair(sum, sumsq, smem);
  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    partials[(row * chunks + chunk) * 2 + 0] = tot.sum;
    partials[(row * chunks + chunk) * 2 + 1] = tot.sumsq;
    __threadfence();
    // Generation counting: the counter is never reset between same-layout
    // calls; each call advances it by exactly `chunks`, so the last finisher
    // of THIS call sees old % chunks == chunks - 1.
    const unsigned int done = atomicAdd(&counters[row], 1u);
    is_last = (done % static_cast<unsigned int>(chunks)) ==
              static_cast<unsigned int>(chunks - 1);
  }
  __syncthreads();
  if (!is_last) return;

  // Last CTA for this row: deterministic ordered reduction of the partials.
  float s = 0.0f, q = 0.0f;
  for (int64_t c = threadIdx.x; c < chunks; c += blockDim.x) {
    s += partials[(row * chunks + c) * 2 + 0];
    q += partials[(row * chunks + c) * 2 + 1];
  }
  __syncthreads();  // smem reuse barrier before the second reduction
  const BlockSums<float> fin = block_reduce_pair(s, q, smem);
  if (threadIdx.x == 0) {
    const float inv = 1.0f / static_cast<float>(group_size);
    const float mean = fin.sum * inv;
    float var = fin.sumsq * inv - mean * mean;
    var = var < 0.0f ? 0.0f : var;
    stats[row * 2 + 0] = mean;
    stats[row * 2 + 1] = rsqrtf(var + eps);
  }
}

template <typename T>
__global__ void gns_split_apply_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    const float* __restrict__ stats,  // [rows][2]
    int64_t num_groups,
    int64_t channels_per_group,
    int64_t spatial,
    int64_t chunk_vecs,
    int64_t group_size) {
  const int64_t row = blockIdx.x;
  const int64_t chunk = blockIdx.y;
  const int64_t group = row % num_groups;
  const int64_t nvec = group_size / kVecHalves;
  const int64_t v0 = chunk * chunk_vecs;
  const int64_t v1 = (v0 + chunk_vecs < nvec) ? (v0 + chunk_vecs) : nvec;
  const uint4* xv = reinterpret_cast<const uint4*>(x + row * group_size);
  uint4* ov = reinterpret_cast<uint4*>(out + row * group_size);

  const float mean = stats[row * 2 + 0];
  const float rstd = stats[row * 2 + 1];
  const int64_t weight_base = group * channels_per_group;
  const int64_t spatial_vec = spatial / kVecHalves;

  // NCU (profile/r1_losers): the per-vector int64 division made this kernel
  // instruction-throughput-bound (SM 84% busy, DRAM 14%) while the upstream
  // scalar-affine variant runs division-free. Most chunks lie entirely inside
  // one channel (spatial >> chunk for the large rows), so hoist the affine
  // load to a chunk constant and keep the dividing loop only for the rare
  // channel-crossing chunks.
  const int64_t ch_first = v0 / spatial_vec;
  const int64_t ch_last = (v1 - 1) / spatial_vec;
  if (ch_first == ch_last) {
    const float w = to_float<T>(weight[weight_base + ch_first]);
    const float b = to_float<T>(bias[weight_base + ch_first]);
    for (int64_t i = v0 + threadIdx.x; i < v1; i += blockDim.x) {
      Vec8<T> v;
      v.raw = xv[i];
      Vec8<T> o;
#pragma unroll
      for (int k = 0; k < kVecHalves; ++k) {
        o.elems[k] = from_float<T>(silu_fast((to_float<T>(v.elems[k]) - mean) * rstd * w + b));
      }
      ov[i] = o.raw;
    }
    return;
  }
  for (int64_t i = v0 + threadIdx.x; i < v1; i += blockDim.x) {
    Vec8<T> v;
    v.raw = xv[i];
    const int64_t ch = i / spatial_vec;
    const float w = to_float<T>(weight[weight_base + ch]);
    const float b = to_float<T>(bias[weight_base + ch]);
    Vec8<T> o;
#pragma unroll
    for (int k = 0; k < kVecHalves; ++k) {
      o.elems[k] = from_float<T>(silu_fast((to_float<T>(v.elems[k]) - mean) * rstd * w + b));
    }
    ov[i] = o.raw;
  }
}

// ---------------------------------------------------------------------------
// nchw_last regime — channels-last(-3d) 16-bit rows, batch 1 per row set.
// Memory is position-major: S spatial positions x C contiguous channels.
// Preconditions (host-checked): stride(1)==1; the spatial dims are contiguous
// position-major with stride C; C % 8 == 0; spatial % 8 == 0 not required
// here, only the position tiling below; 16B alignment of x and out.
//
// Stage 1 (stats): CTAs grid-stride over position tiles; each thread owns a
// fixed 8-channel lane (so a fixed group, or two groups when C/G == 4) and
// accumulates fp32 partial sums locally; per-tile partials are combined in
// shared memory and added to global per-(batch, group) partials; the last
// finishing CTA computes mean/rstd for all groups of that batch.
// Stage 2 (apply): CTAs re-read tiles, normalize with per-lane cached
// weight/bias, stage results in shared memory as [C][P], then flush
// channel-major runs of P contiguous elements to the NC... output.
// ---------------------------------------------------------------------------

constexpr int kNcSmemBytes = 32 * 1024;  // staging budget per CTA

template <typename T>
__global__ void gns_nc_stats_kernel(
    const T* __restrict__ x,
    float* __restrict__ partials,   // [tiles][G][2] accumulated atomically
    float* __restrict__ stats,      // [B][G][2]
    unsigned int* __restrict__ counters,  // [B]
    int64_t batch_stride,
    int64_t channels,
    int64_t num_groups,
    int64_t spatial,
    int64_t tile_positions,
    int64_t tiles_per_batch,
    int64_t group_size,
    float eps) {
  const int64_t batch = blockIdx.z;
  const int64_t tile = blockIdx.x;
  const int64_t p0 = tile * tile_positions;
  const int64_t p1 = (p0 + tile_positions < spatial) ? (p0 + tile_positions) : spatial;
  const int64_t lanes = channels / kVecHalves;       // 8-channel vector lanes
  const int64_t lane = threadIdx.x % lanes;
  const int64_t prow = threadIdx.x / lanes;          // position row within CTA pass
  const int64_t prows = blockDim.x / lanes;
  const int64_t cpg = channels / num_groups;
  // Lane's first channel and its group(s). When cpg >= 8 the whole 8-channel
  // vector is in one group; when cpg == 4 it spans two consecutive groups.
  const int64_t ch0 = lane * kVecHalves;
  const int64_t g_lo = ch0 / cpg;
  const int64_t g_hi = (ch0 + kVecHalves - 1) / cpg;

  const uint4* xv = reinterpret_cast<const uint4*>(x + batch * batch_stride);

  float sum_lo = 0.0f, sq_lo = 0.0f, sum_hi = 0.0f, sq_hi = 0.0f;
  if (threadIdx.x < lanes * prows) {
    for (int64_t p = p0 + prow; p < p1; p += prows) {
      Vec8<T> v;
      v.raw = xv[p * lanes + lane];
#pragma unroll
      for (int k = 0; k < kVecHalves; ++k) {
        const float f = to_float<T>(v.elems[k]);
        if (k < 4) {
          sum_lo += f;
          sq_lo += f * f;
        } else {
          sum_hi += f;
          sq_hi += f * f;
        }
      }
    }
  }

  // Shared accumulation across the CTA for all groups.
  __shared__ float g_sum[64];  // [G][2] with G <= 32
  for (int i = threadIdx.x; i < 2 * num_groups; i += blockDim.x) g_sum[i] = 0.0f;
  __syncthreads();
  if (threadIdx.x < lanes * prows) {
    if (g_lo == g_hi) {
      atomicAdd(&g_sum[g_lo * 2 + 0], sum_lo + sum_hi);
      atomicAdd(&g_sum[g_lo * 2 + 1], sq_lo + sq_hi);
    } else {
      atomicAdd(&g_sum[g_lo * 2 + 0], sum_lo);
      atomicAdd(&g_sum[g_lo * 2 + 1], sq_lo);
      atomicAdd(&g_sum[g_hi * 2 + 0], sum_hi);
      atomicAdd(&g_sum[g_hi * 2 + 1], sq_hi);
    }
  }
  __syncthreads();

  float* batch_partials = partials + (batch * tiles_per_batch + tile) * num_groups * 2;
  for (int i = threadIdx.x; i < 2 * num_groups; i += blockDim.x) {
    batch_partials[i] = g_sum[i];
  }

  __shared__ bool is_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    // Generation counting; see gns_split_stats_kernel.
    const unsigned int done = atomicAdd(&counters[batch], 1u);
    is_last = (done % static_cast<unsigned int>(tiles_per_batch)) ==
              static_cast<unsigned int>(tiles_per_batch - 1);
  }
  __syncthreads();
  if (!is_last) return;

  // Cross-tile reduction for every group of this batch. NCU
  // (profile/r1_losers): the previous one-thread-per-group serial loop over
  // all tiles put a long scalar tail on the kernel's critical path (SM busy
  // 6%). Use eight threads per group with a strided deterministic
  // accumulation order, then a segmented shuffle combine.
  constexpr int kSub = 8;  // threads per group; segments stay inside a warp
  const int g = threadIdx.x / kSub;
  const int sub = threadIdx.x % kSub;
  if (g < num_groups) {
    float s = 0.0f, q = 0.0f;
    const float* base = partials + batch * tiles_per_batch * num_groups * 2;
    for (int64_t t = sub; t < tiles_per_batch; t += kSub) {
      s += base[(t * num_groups + g) * 2 + 0];
      q += base[(t * num_groups + g) * 2 + 1];
    }
#pragma unroll
    for (int off = kSub / 2; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xffffffffu, s, off);
      q += __shfl_down_sync(0xffffffffu, q, off);
    }
    if (sub == 0) {
      const float inv = 1.0f / static_cast<float>(group_size);
      const float mean = s * inv;
      float var = q * inv - mean * mean;
      var = var < 0.0f ? 0.0f : var;
      stats[(batch * num_groups + g) * 2 + 0] = mean;
      stats[(batch * num_groups + g) * 2 + 1] = rsqrtf(var + eps);
    }
  }
}

// Staging pad: position-major rows of (C + kStagePad) 16-bit elements. The
// +4 keeps 8-byte alignment for paired-element stores while making the row
// stride a non-multiple of 32 banks. NCU (profile/r1_losers): the previous
// channel-major [C][P] layout serialized on ~7.8M shared-store bank
// conflicts (row stride was a multiple of the bank count).
constexpr int kStagePad = 4;

template <typename T>
__global__ void gns_nc_apply_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    const float* __restrict__ stats,  // [B][G][2]
    int64_t batch_stride,
    int64_t channels,
    int64_t num_groups,
    int64_t spatial,
    int64_t tile_positions) {
  extern __shared__ unsigned char smem_raw[];
  T* stage = reinterpret_cast<T*>(smem_raw);  // [tile_positions][C + kStagePad]
  const int64_t stage_stride = channels + kStagePad;

  const int64_t batch = blockIdx.z;
  const int64_t tile = blockIdx.x;
  const int64_t p0 = tile * tile_positions;
  const int64_t pcount =
      (p0 + tile_positions < spatial) ? tile_positions : (spatial - p0);
  const int64_t lanes = channels / kVecHalves;
  const int64_t lane = threadIdx.x % lanes;
  const int64_t prow = threadIdx.x / lanes;
  const int64_t prows = blockDim.x / lanes;
  const int64_t cpg = channels / num_groups;
  const int64_t ch0 = lane * kVecHalves;

  const uint4* xv = reinterpret_cast<const uint4*>(x + batch * batch_stride);

  // Per-lane cached stats and affine parameters for its 8 channels.
  float w[kVecHalves], b[kVecHalves], mean[kVecHalves], rstd[kVecHalves];
  if (threadIdx.x < lanes * prows) {
#pragma unroll
    for (int k = 0; k < kVecHalves; ++k) {
      const int64_t ch = ch0 + k;
      const int64_t g = ch / cpg;
      w[k] = to_float<T>(weight[ch]);
      b[k] = to_float<T>(bias[ch]);
      mean[k] = stats[(batch * num_groups + g) * 2 + 0];
      rstd[k] = stats[(batch * num_groups + g) * 2 + 1];
    }
    for (int64_t p = prow; p < pcount; p += prows) {
      Vec8<T> v;
      v.raw = xv[(p0 + p) * lanes + lane];
      Vec8<T> o;
#pragma unroll
      for (int k = 0; k < kVecHalves; ++k) {
        const float f = to_float<T>(v.elems[k]);
        o.elems[k] = from_float<T>(silu_fast((f - mean[k]) * rstd[k] * w[k] + b[k]));
      }
      // Two 8-byte stores into the position-major stage row (8B-aligned:
      // ch0 % 8 == 0 and the row stride is a multiple of 4 elements).
      uint2* dst = reinterpret_cast<uint2*>(stage + p * stage_stride + ch0);
      const uint2* src = reinterpret_cast<const uint2*>(o.elems);
      dst[0] = src[0];
      dst[1] = src[1];
    }
  }
  __syncthreads();

  // Flush: gather strided stage reads per channel (conflict-light: the row
  // stride in words is not a multiple of the bank count) and write the
  // contiguous NC... output with one 16-byte store per 8 positions.
  T* ob = out + batch * channels * spatial;
  if (pcount == tile_positions && (tile_positions % kVecHalves) == 0) {
    const int64_t vec_per_ch = tile_positions / kVecHalves;
    for (int64_t i = threadIdx.x; i < channels * vec_per_ch; i += blockDim.x) {
      const int64_t c = i / vec_per_ch;
      const int64_t vp = i % vec_per_ch;
      Vec8<T> o;
#pragma unroll
      for (int k = 0; k < kVecHalves; ++k) {
        o.elems[k] = stage[(vp * kVecHalves + k) * stage_stride + c];
      }
      reinterpret_cast<uint4*>(ob + c * spatial + p0)[vp] = o.raw;
    }
  } else {
    const int64_t total = channels * pcount;
    for (int64_t i = threadIdx.x; i < total; i += blockDim.x) {
      const int64_t c = i / pcount;
      const int64_t p = i % pcount;
      ob[c * spatial + p0 + p] = stage[p * stage_stride + c];
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side dispatch
// ---------------------------------------------------------------------------

void check(bool cond, const char* msg) {
  if (!cond) {
    TVM_FFI_THROW(RuntimeError) << msg;
  }
}

int64_t env_int(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::atoll(v);
}

// The split and channels-last regimes need small per-call scratch (fp32
// partial sums, per-row stats, completion counters). Two measured hazards
// drove this design:
//   1. cudaMallocAsync with the default pool (release threshold 0) trims the
//      pool at every stream sync — the benchmark's CUDA-event waits — so
//      allocations periodically pay a REAL cudaMalloc (bimodal multi-hundred-
//      microsecond swings between identical rows).
//   2. Even pooled alloc/free + a counter memset cost a few microseconds per
//      call, visible on ~100 us rows.
// Use a process-lifetime grow-only scratch buffer instead. Completion
// counters are NEVER reset between same-layout calls: the last-CTA test uses
// modular arithmetic on a monotonically growing counter ("generation"
// counting), so the counter region only needs zeroing when the scratch
// layout changes (tracked by a signature) or the buffer is (re)grown.
struct ScratchArena {
  void* buf = nullptr;
  size_t capacity = 0;
  uint64_t signature = 0;  // layout signature of the most recent user

  // Returns a buffer of at least `bytes`; zeroes the whole buffer (on the
  // stream) when grown or when the layout signature changes.
  void* acquire(size_t bytes, uint64_t sig, cudaStream_t stream) {
    if (capacity < bytes) {
      if (buf != nullptr) {
        // Growth is rare (signature-stable steady state never grows); a full
        // device sync makes freeing the in-flight buffer safe.
        C10_CUDA_CHECK(cudaDeviceSynchronize());
        C10_CUDA_CHECK(cudaFree(buf));
        buf = nullptr;
        capacity = 0;
      }
      C10_CUDA_CHECK(cudaMalloc(&buf, bytes));
      capacity = bytes;
      signature = 0;  // force re-zero below
    }
    if (signature != sig) {
      C10_CUDA_CHECK(cudaMemsetAsync(buf, 0, capacity, stream));
      signature = sig;
    }
    return buf;
  }
};

// Scratch is keyed per (device, stream): concurrent streams (or alternating
// devices) must never share partials/stats/counters while kernels are in
// flight — reuse within one key is safe because it is stream-ordered. The
// global mutex covers the whole acquire (steady state is a compare + return;
// growth, which synchronizes, is rare and cold).
void* acquire_scratch(size_t bytes, uint64_t sig, cudaStream_t stream) {
  static std::mutex arenas_mutex;
  static std::map<std::pair<int, cudaStream_t>, ScratchArena> arenas;
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(arenas_mutex);
  return arenas[{device, stream}].acquire(bytes, sig, stream);
}

struct Geometry {
  int ndim;
  int64_t batch;
  int64_t channels;
  int64_t spatial;
  int64_t channels_per_group;
  int64_t group_size;
  bool x_contiguous;
  bool x_channels_last;  // position-major: stride(1)==1, spatial dims = C-major
  bool aligned16;        // x and out base pointers 16B aligned
};

Geometry analyze(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& out,
    int64_t num_groups) {
  Geometry g;
  g.ndim = static_cast<int>(x.ndim());
  g.batch = x.size(0);
  g.channels = x.size(1);
  g.channels_per_group = g.channels / num_groups;
  g.spatial = 1;
  for (int d = 2; d < g.ndim; ++d) g.spatial *= x.size(d);
  g.group_size = g.channels_per_group * g.spatial;

  // Fully contiguous?
  g.x_contiguous = true;
  {
    int64_t expect = 1;
    for (int d = g.ndim - 1; d >= 0; --d) {
      if (x.size(d) != 1 && x.stride(d) != expect) g.x_contiguous = false;
      expect *= x.size(d);
    }
  }
  // Channels-last (position-major): channel stride 1 and the spatial dims
  // form a contiguous position-major block of stride C; batch stride C*S.
  g.x_channels_last = false;
  if (g.ndim >= 3 && x.stride(1) == 1) {
    bool ok = true;
    int64_t expect = g.channels;
    for (int d = g.ndim - 1; d >= 2; --d) {
      if (x.size(d) != 1 && x.stride(d) != expect) ok = false;
      expect *= x.size(d);
    }
    if (x.size(0) != 1 && x.stride(0) != g.channels * g.spatial) ok = false;
    g.x_channels_last = ok;
  }
  const auto xp = reinterpret_cast<uintptr_t>(x.data_ptr());
  const auto op = reinterpret_cast<uintptr_t>(out.data_ptr());
  g.aligned16 = ((xp % 16) == 0) && ((op % 16) == 0);
  return g;
}

// Regime selection codes (also exported for reporting):
// 0 generic, 1 cont_small, 2 cont_split, 3 nchw_last
int64_t select_regime_impl(const Geometry& g, const DLDataType& dt) {
  const bool is16 = (dt.bits == 16) && (dt.code == kDLFloat || dt.code == kDLBfloat);
  const int64_t small_max = env_int("GNS_SMALL_MAX", 65536);
  // nchw_last accumulates each 8-channel vector as two FIXED 4-channel
  // halves, so every group boundary must land at offset 0 or 4 of an
  // 8-aligned channel window. That holds exactly when cpg % 4 == 0 (cpg 4:
  // halves are whole groups; cpg 8k: vectors never cross; cpg 12, 20, ...:
  // boundaries are multiples of 4). cpg values like 5/6/7/10 would split a
  // group mid-half and corrupt the statistics — they route to the generic
  // kernel instead. Also at most 32 groups (the per-CTA group accumulator
  // and the finalize segmentation are sized for 32).
  if (is16 && g.x_channels_last && !g.x_contiguous && g.aligned16 &&
      (g.channels % kVecHalves) == 0 && (g.channels / kVecHalves) <= kBlockThreads &&
      (g.spatial % kVecHalves) == 0 && g.channels <= 1024 &&
      (g.channels_per_group % 4) == 0 &&
      (g.channels / g.channels_per_group) <= 32) {
    return 3;
  }
  if (is16 && g.x_contiguous && g.aligned16 && (g.group_size % kVecHalves) == 0 &&
      (g.spatial % kVecHalves) == 0) {
    return (g.group_size <= small_max) ? 1 : 2;
  }
  return 0;
}

template <typename T>
void launch_generic(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& bias,
    const tvm::ffi::TensorView& out,
    const Geometry& g,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {
  StridedLayout lay;
  lay.batch_stride = x.stride(0);
  lay.channel_stride = x.stride(1);
  lay.sizes[0] = g.channels_per_group;
  lay.strides[0] = x.stride(1);
  lay.ndim = 1;
  for (int d = 2; d < g.ndim; ++d) {
    lay.sizes[lay.ndim] = x.size(d);
    lay.strides[lay.ndim] = x.stride(d);
    lay.ndim += 1;
  }
  for (int d = lay.ndim; d < kMaxDims; ++d) {
    lay.sizes[d] = 1;
    lay.strides[d] = 0;
  }
  const dim3 grid(static_cast<unsigned>(g.batch * num_groups));
  gns_generic_two_pass_kernel<T><<<grid, kBlockThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      static_cast<const T*>(weight.data_ptr()),
      static_cast<const T*>(bias.data_ptr()),
      static_cast<T*>(out.data_ptr()),
      lay, num_groups, g.channels_per_group, g.spatial, g.group_size,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_cont_small(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& bias,
    const tvm::ffi::TensorView& out,
    const Geometry& g,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(g.batch * num_groups));
  gns_cont_small_kernel<T><<<grid, kBlockThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      static_cast<const T*>(weight.data_ptr()),
      static_cast<const T*>(bias.data_ptr()),
      static_cast<T*>(out.data_ptr()),
      num_groups, g.channels_per_group, g.spatial, g.group_size,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_cont_split(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& bias,
    const tvm::ffi::TensorView& out,
    const Geometry& g,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {
  const int64_t rows = g.batch * num_groups;
  const int64_t chunk_elems = env_int("GNS_CHUNK", 16384);
  const int64_t chunk_vecs = chunk_elems / kVecHalves;
  const int64_t nvec = g.group_size / kVecHalves;
  const int64_t chunks = (nvec + chunk_vecs - 1) / chunk_vecs;

  const size_t partial_bytes = static_cast<size_t>(rows * chunks * 2) * sizeof(float);
  const size_t stats_bytes = static_cast<size_t>(rows * 2) * sizeof(float);
  const size_t counter_bytes = static_cast<size_t>(rows) * sizeof(unsigned int);
  const uint64_t sig = (0x2ULL << 60) ^ (static_cast<uint64_t>(rows) << 40) ^
                       (static_cast<uint64_t>(chunks) << 8);
  float* scratch = static_cast<float*>(acquire_scratch(
      partial_bytes + stats_bytes + counter_bytes, sig, stream));
  float* partials = scratch;
  float* stats = scratch + rows * chunks * 2;
  unsigned int* counters = reinterpret_cast<unsigned int*>(stats + rows * 2);

  const dim3 grid1(static_cast<unsigned>(rows), static_cast<unsigned>(chunks));
  gns_split_stats_kernel<T><<<grid1, kBlockThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      partials, stats, counters, chunks, chunk_vecs, g.group_size,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gns_split_apply_kernel<T><<<grid1, kBlockThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      static_cast<const T*>(weight.data_ptr()),
      static_cast<const T*>(bias.data_ptr()),
      static_cast<T*>(out.data_ptr()),
      stats, num_groups, g.channels_per_group, g.spatial, chunk_vecs,
      g.group_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_nc(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& bias,
    const tvm::ffi::TensorView& out,
    const Geometry& g,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {
  // Tile positions sized so the padded position-major staging buffer
  // [P][C + kStagePad] fits the smem budget.
  int64_t tile_positions =
      kNcSmemBytes / ((g.channels + kStagePad) * static_cast<int64_t>(sizeof(T)));
  tile_positions = (tile_positions / kVecHalves) * kVecHalves;
  if (tile_positions > g.spatial) {
    tile_positions = ((g.spatial + kVecHalves - 1) / kVecHalves) * kVecHalves;
  }
  check(tile_positions >= kVecHalves, "channels too large for nc tile staging");
  const int64_t tiles = (g.spatial + tile_positions - 1) / tile_positions;
  const int64_t batch_stride = x.stride(0);

  const size_t partial_bytes =
      static_cast<size_t>(g.batch * tiles * num_groups * 2) * sizeof(float);
  const size_t stats_bytes =
      static_cast<size_t>(g.batch * num_groups * 2) * sizeof(float);
  const size_t counter_bytes = static_cast<size_t>(g.batch) * sizeof(unsigned int);
  const uint64_t sig = (0x3ULL << 60) ^ (static_cast<uint64_t>(g.batch) << 40) ^
                       (static_cast<uint64_t>(tiles) << 8) ^
                       static_cast<uint64_t>(num_groups);
  float* scratch = static_cast<float*>(acquire_scratch(
      partial_bytes + stats_bytes + counter_bytes, sig, stream));
  float* partials = scratch;
  float* stats = scratch + g.batch * tiles * num_groups * 2;
  unsigned int* counters = reinterpret_cast<unsigned int*>(stats + g.batch * num_groups * 2);

  const dim3 grid(static_cast<unsigned>(tiles), 1u, static_cast<unsigned>(g.batch));
  gns_nc_stats_kernel<T><<<grid, kBlockThreads, 0, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      partials, stats, counters, batch_stride, g.channels, num_groups,
      g.spatial, tile_positions, tiles, g.group_size, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t stage_bytes =
      static_cast<size_t>((g.channels + kStagePad) * tile_positions) * sizeof(T);
  gns_nc_apply_kernel<T><<<grid, kBlockThreads, stage_bytes, stream>>>(
      static_cast<const T*>(x.data_ptr()),
      static_cast<const T*>(weight.data_ptr()),
      static_cast<const T*>(bias.data_ptr()),
      static_cast<T*>(out.data_ptr()),
      stats, batch_stride, g.channels, num_groups, g.spatial, tile_positions);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_typed(
    const tvm::ffi::TensorView& x,
    const tvm::ffi::TensorView& weight,
    const tvm::ffi::TensorView& bias,
    const tvm::ffi::TensorView& out,
    const Geometry& g,
    int64_t regime,
    int64_t num_groups,
    double eps,
    cudaStream_t stream) {
  switch (regime) {
    case 1:
      launch_cont_small<T>(x, weight, bias, out, g, num_groups, eps, stream);
      break;
    case 2:
      launch_cont_split<T>(x, weight, bias, out, g, num_groups, eps, stream);
      break;
    case 3:
      launch_nc<T>(x, weight, bias, out, g, num_groups, eps, stream);
      break;
    default:
      launch_generic<T>(x, weight, bias, out, g, num_groups, eps, stream);
  }
}

void group_norm_silu(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    int64_t num_groups,
    double eps,
    tvm::ffi::TensorView out) {
  const int ndim = static_cast<int>(x.ndim());
  check(ndim >= 2 && ndim <= kMaxDims, "x must have 2..5 dims");
  check(num_groups > 0 && x.size(1) % num_groups == 0,
        "channels must divide num_groups");
  check(weight.ndim() == 1 && bias.ndim() == 1, "weight/bias must be 1-D");
  check(weight.size(0) == x.size(1) && bias.size(0) == x.size(1),
        "weight/bias must have C elements");
  check(out.ndim() == ndim, "out must match x rank");
  int64_t numel = 1;
  for (int d = 0; d < ndim; ++d) {
    check(out.size(d) == x.size(d), "out must match x shape");
    numel *= x.size(d);
  }
  // Contiguous output by contract (mirrors the upstream baseline's return).
  int64_t expect = 1;
  for (int d = ndim - 1; d >= 0; --d) {
    check(out.stride(d) == expect || out.size(d) == 1, "out must be contiguous");
    expect *= out.size(d);
  }
  check(x.dtype() == weight.dtype() && x.dtype() == bias.dtype() &&
            x.dtype() == out.dtype(),
        "dtype mismatch");
  const DLDevice xdev = x.device();
  check(xdev.device_type == kDLCUDA, "x must be a CUDA tensor");
  check(weight.device().device_type == kDLCUDA &&
            weight.device().device_id == xdev.device_id &&
            bias.device().device_type == kDLCUDA &&
            bias.device().device_id == xdev.device_id &&
            out.device().device_type == kDLCUDA &&
            out.device().device_id == xdev.device_id,
        "x/weight/bias/out must be on the same CUDA device");
  if (numel == 0) {
    return;
  }

  // Launch on the stream of x's device (parity with the upstream baseline's
  // `with torch.cuda.device(x.device)`): if the caller's current device
  // differs from x's, the current-device stream would be the wrong one and
  // the launches would touch pointers from another device.
  const auto device_index = static_cast<c10::DeviceIndex>(xdev.device_id);
  c10::cuda::CUDAGuard device_guard(device_index);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index);
  const Geometry g = analyze(x, out, num_groups);
  const DLDataType dt = x.dtype();
  const int64_t regime = select_regime_impl(g, dt);

  if (dt.code == kDLFloat && dt.bits == 16) {
    dispatch_typed<__half>(x, weight, bias, out, g, regime, num_groups, eps, stream);
  } else if (dt.code == kDLBfloat && dt.bits == 16) {
    dispatch_typed<__nv_bfloat16>(x, weight, bias, out, g, regime, num_groups, eps, stream);
  } else if (dt.code == kDLFloat && dt.bits == 32) {
    dispatch_typed<float>(x, weight, bias, out, g, regime, num_groups, eps, stream);
  } else {
    TVM_FFI_THROW(RuntimeError) << "unsupported dtype (fp16/bf16/fp32 only)";
  }
}

// Reporting helper (untimed): which regime would the dispatcher pick?
int64_t group_norm_silu_regime(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    int64_t num_groups,
    tvm::ffi::TensorView out) {
  const Geometry g = analyze(x, out, num_groups);
  return select_regime_impl(g, x.dtype());
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_norm_silu, group_norm_silu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(group_norm_silu_regime, group_norm_silu_regime);
