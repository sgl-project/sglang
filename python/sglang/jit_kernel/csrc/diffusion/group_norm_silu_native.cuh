// Fused GroupNorm + SiLU CUDA kernel for SGLang diffusion VAEs (H200 / SM90).
// Built + exported via SGLang jit_kernel / tvm-ffi (load_jit), mirroring the
// conventions of python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh.
//
// Contract: x is contiguous [B, C, spatial]; one group is a contiguous block of
// group_size = (C/num_groups)*spatial; affine is per-channel:
//   channel = g*channels_per_group + (i_within_group / spatial).
//   y = silu((x - mean)*rstd * weight[ch] + bias[ch]); silu(z) = z*sigmoid(z).
// mean/var are computed per (batch, group) in fp32 (biased; var clamped >= 0).
//
// Two paths, dispatched in Python by group_size:
//   - SMALL: one CTA per (batch, group), two passes in-kernel (no scratch). Best
//     for small/tiny groups where launch overhead dominates.
//   - LARGE: deterministic 3-stage (stats -> finalize -> apply) over a persistent
//     grid-stride of (row, chunk) tasks. Many CTAs per group -> high occupancy on
//     the big video shapes. Partial sums use fp32 scratch, NO atomics (deterministic).

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>

namespace {

using namespace device;

constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kWarpsPerBlock = kBlockThreads / kWarpThreads;  // 8
// Elements processed per CTA-task in the LARGE path (256 thr * 4 vecs * 8 half = 8192).
// Must match _CHUNK_ELEMS in src/register.py.
constexpr int64_t kChunkElems = 8192;

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

// Block-wide reduction of two fp32 accumulators (sum, sumsq). Deterministic
// (fixed-order tree reduce, no atomics). Result broadcast to all threads.
// `smem` must have >= 2*kWarpsPerBlock + 2 floats.
SGL_DEVICE void block_reduce2(float& a, float& b, float* smem) {
  a = warp::reduce_sum(a);
  b = warp::reduce_sum(b);
  const uint32_t lane = threadIdx.x & (kWarpThreads - 1);
  const uint32_t warp_id = threadIdx.x >> 5;
  if (lane == 0) {
    smem[warp_id] = a;
    smem[kWarpsPerBlock + warp_id] = b;
  }
  __syncthreads();
  if (warp_id == 0) {
    float ta = (lane < kWarpsPerBlock) ? smem[lane] : 0.0f;
    float tb = (lane < kWarpsPerBlock) ? smem[kWarpsPerBlock + lane] : 0.0f;
    ta = warp::reduce_sum<kWarpsPerBlock>(ta);
    tb = warp::reduce_sum<kWarpsPerBlock>(tb);
    if (lane == 0) {
      smem[2 * kWarpsPerBlock] = ta;
      smem[2 * kWarpsPerBlock + 1] = tb;
    }
  }
  __syncthreads();
  a = smem[2 * kWarpsPerBlock];
  b = smem[2 * kWarpsPerBlock + 1];
}

SGL_DEVICE float siluf(float z) {
  // accuracy-compatible sigmoid (no fast-math): z / (1 + exp(-z))
  return z / (1.0f + math::exp<fp32_t>(-z));
}

// Reduce x[off, off+nelem) into (lsum, lsumsq) in fp32. `vec_aligned` enables half8.
template <typename DType, int kVec>
SGL_DEVICE void accumulate_stats(const DType* __restrict__ x, int64_t nelem, bool vec_aligned,
                                 float& lsum, float& lsumsq) {
  const int64_t nvec = vec_aligned ? nelem / kVec : 0;
  for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
    AlignedVector<DType, kVec> v;
    v.load(x, vi);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float xf = cast<fp32_t>(v[j]);
      lsum += xf;
      lsumsq += xf * xf;
    }
  }
  for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
    const float xf = cast<fp32_t>(x[i]);
    lsum += xf;
    lsumsq += xf * xf;
  }
}

// Normalize + per-channel affine + SiLU over x[off, off+nelem) -> y, where
// `group_off` is the start offset within the group (for channel indexing).
template <typename DType, int kVec>
SGL_DEVICE void apply_affine_silu(const DType* __restrict__ x, DType* __restrict__ y, int64_t nelem,
                                  int64_t group_off, int64_t spatial, const DType* __restrict__ weight,
                                  const DType* __restrict__ bias, int64_t ch_base, float mean, float rstd,
                                  bool vec_aligned) {
  const int64_t nvec = vec_aligned ? nelem / kVec : 0;
  // Fast path: the whole tile lies in ONE channel. This is the common case for the LARGE
  // path (a chunk of kChunkElems << spatial), so load the affine ONCE and drop the
  // per-vector int64 channel division entirely (the dominant compute cost found by NCU).
  if (group_off / spatial == (group_off + nelem - 1) / spatial) {
    const int64_t c = group_off / spatial;
    const float w = cast<fp32_t>(weight[ch_base + c]);
    const float bb = cast<fp32_t>(bias[ch_base + c]);
    for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
      AlignedVector<DType, kVec> v;
      v.load(x, vi);
      AlignedVector<DType, kVec> o;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o[j] = cast<DType>(siluf((cast<fp32_t>(v[j]) - mean) * rstd * w + bb));
      }
      o.store(y, vi);
    }
    for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
      y[i] = cast<DType>(siluf((cast<fp32_t>(x[i]) - mean) * rstd * w + bb));
    }
    return;
  }
  for (int64_t vi = threadIdx.x; vi < nvec; vi += blockDim.x) {
    const int64_t i0 = group_off + vi * kVec;
    AlignedVector<DType, kVec> v;
    v.load(x, vi);
    AlignedVector<DType, kVec> o;
    const int64_t c0 = i0 / spatial;
    const int64_t c1 = (i0 + kVec - 1) / spatial;
    if (c0 == c1) {  // whole vector within one channel -> scalar affine
      const float w = cast<fp32_t>(weight[ch_base + c0]);
      const float bb = cast<fp32_t>(bias[ch_base + c0]);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        o[j] = cast<DType>(siluf((cast<fp32_t>(v[j]) - mean) * rstd * w + bb));
      }
    } else {  // straddles a channel boundary -> per-lane affine
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        const int64_t c = (i0 + j) / spatial;
        const float w = cast<fp32_t>(weight[ch_base + c]);
        const float bb = cast<fp32_t>(bias[ch_base + c]);
        o[j] = cast<DType>(siluf((cast<fp32_t>(v[j]) - mean) * rstd * w + bb));
      }
    }
    o.store(y, vi);
  }
  for (int64_t i = nvec * kVec + threadIdx.x; i < nelem; i += blockDim.x) {
    const int64_t c = (group_off + i) / spatial;
    const float w = cast<fp32_t>(weight[ch_base + c]);
    const float bb = cast<fp32_t>(bias[ch_base + c]);
    y[i] = cast<DType>(siluf((cast<fp32_t>(x[i]) - mean) * rstd * w + bb));
  }
}

// ---------------- SMALL path: one CTA per (batch, group) ----------------
template <typename DType, bool kUsePDL>
__global__ void gns_one_pass_kernel(const GnsParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));  // fp16/bf16 -> 8, fp32 -> 4
  __shared__ float smem[2 * kWarpsPerBlock + 2];

  PDLWaitPrimary<kUsePDL>();

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
  block_reduce2(lsum, lsumsq, smem);

  const float inv_n = 1.0f / static_cast<float>(n);
  const float mean = lsum * inv_n;
  const float var = math::max<fp32_t>(lsumsq * inv_n - mean * mean, 0.0f);
  const float rstd = math::rsqrt<fp32_t>(var + p.eps);

  apply_affine_silu<DType, kVec>(x, y, n, 0, p.spatial, weight, bias, ch_base, mean, rstd, vec_ok);

  PDLTriggerSecondary<kUsePDL>();
}

// ---------------- LARGE path: 3-stage multi-CTA ----------------
template <typename DType, bool kUsePDL>
__global__ void gns_stats_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  __shared__ float smem[2 * kWarpsPerBlock + 2];
  const int64_t total_tasks = p.num_rows * p.chunks_per_row;
  float* __restrict__ psum = static_cast<float*>(p.partial_sum);
  float* __restrict__ psumsq = static_cast<float*>(p.partial_sumsq);

  PDLWaitPrimary<kUsePDL>();
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
  PDLTriggerSecondary<kUsePDL>();
}

template <typename DType, bool kUsePDL>
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
    const float var = math::max<fp32_t>(lsumsq * inv_n - mean * mean, 0.0f);
    static_cast<float*>(p.mean)[row] = mean;
    static_cast<float*>(p.rstd)[row] = math::rsqrt<fp32_t>(var + p.eps);
  }
}

template <typename DType, bool kUsePDL>
__global__ void gns_apply_kernel(const GnsLargeParams<DType> __grid_constant__ p) {
  constexpr int kVec = 16 / static_cast<int>(sizeof(DType));
  const int64_t total_tasks = p.num_rows * p.chunks_per_row;
  const float* __restrict__ mean_arr = static_cast<const float*>(p.mean);
  const float* __restrict__ rstd_arr = static_cast<const float*>(p.rstd);

  PDLWaitPrimary<kUsePDL>();
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

    apply_affine_silu<DType, kVec>(x, y, nelem, chunk_start, p.spatial, weight, bias, ch_base, mean, rstd, vec_ok);
  }
  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace

template <typename DType, bool kUsePDL>
struct GroupNormSiluKernel {
  // SMALL path: x, y contiguous [B, C, spatial]; weight, bias [C].
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView y,
      int64_t num_groups,
      double eps) {
    using namespace host;
    auto B = SymbolicSize{"B"};
    auto C = SymbolicSize{"C"};
    auto S = SymbolicSize{"spatial"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({B, C, S}).with_dtype<DType>().with_device(dev).verify(x).verify(y);
    TensorMatcher({C}).with_dtype<DType>().with_device(dev).verify(weight).verify(bias);

    const int64_t channels = C.unwrap();
    const int64_t spatial = S.unwrap();
    RuntimeCheck(num_groups > 0 && channels % num_groups == 0, "channels must be divisible by num_groups");
    const int64_t channels_per_group = channels / num_groups;

    const auto params = GnsParams<DType>{
        .x = x.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.data_ptr(),
        .y = y.data_ptr(),
        .channels = channels,
        .spatial = spatial,
        .num_groups = num_groups,
        .channels_per_group = channels_per_group,
        .group_size = channels_per_group * spatial,
        .num_rows = B.unwrap() * num_groups,
        .eps = static_cast<float>(eps),
    };
    const auto device = dev.unwrap();
    LaunchKernel(static_cast<uint32_t>(params.num_rows), kBlockThreads, device)
        .enable_pdl(kUsePDL)(gns_one_pass_kernel<DType, kUsePDL>, params);
  }

  // LARGE path: scratch (partial_sum/sumsq [num_rows*chunks_per_row], mean/rstd [num_rows], fp32)
  // is pre-allocated in Python. chunks_per_row = ceil(group_size / kChunkElems).
  static void
  run_large(tvm::ffi::TensorView x,
            tvm::ffi::TensorView weight,
            tvm::ffi::TensorView bias,
            tvm::ffi::TensorView y,
            tvm::ffi::TensorView partial_sum,
            tvm::ffi::TensorView partial_sumsq,
            tvm::ffi::TensorView mean,
            tvm::ffi::TensorView rstd,
            int64_t num_groups,
            double eps) {
    using namespace host;
    auto B = SymbolicSize{"B"};
    auto C = SymbolicSize{"C"};
    auto S = SymbolicSize{"spatial"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({B, C, S}).with_dtype<DType>().with_device(dev).verify(x).verify(y);
    TensorMatcher({C}).with_dtype<DType>().with_device(dev).verify(weight).verify(bias);

    const int64_t channels = C.unwrap();
    const int64_t spatial = S.unwrap();
    RuntimeCheck(num_groups > 0 && channels % num_groups == 0, "channels must be divisible by num_groups");
    const int64_t channels_per_group = channels / num_groups;
    const int64_t group_size = channels_per_group * spatial;
    const int64_t num_rows = B.unwrap() * num_groups;
    const int64_t chunks_per_row = (group_size + kChunkElems - 1) / kChunkElems;
    const int64_t total_tasks = num_rows * chunks_per_row;

    auto TT = SymbolicSize{"total_tasks"};
    TT.set_value(total_tasks);
    auto NR = SymbolicSize{"num_rows"};
    NR.set_value(num_rows);
    TensorMatcher({TT}).with_dtype<fp32_t>().with_device(dev).verify(partial_sum).verify(partial_sumsq);
    TensorMatcher({NR}).with_dtype<fp32_t>().with_device(dev).verify(mean).verify(rstd);

    const auto params = GnsLargeParams<DType>{
        .x = x.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.data_ptr(),
        .y = y.data_ptr(),
        .partial_sum = partial_sum.data_ptr(),
        .partial_sumsq = partial_sumsq.data_ptr(),
        .mean = mean.data_ptr(),
        .rstd = rstd.data_ptr(),
        .channels = channels,
        .spatial = spatial,
        .num_groups = num_groups,
        .channels_per_group = channels_per_group,
        .group_size = group_size,
        .num_rows = num_rows,
        .chunk_elems = kChunkElems,
        .chunks_per_row = chunks_per_row,
        .eps = static_cast<float>(eps),
    };
    const auto device = dev.unwrap();
    const uint32_t sm_count = runtime::get_sm_count(device.device_id);

    // Dependent stages run serially on the current stream; no PDL between them.
    auto stats_k = gns_stats_kernel<DType, kUsePDL>;
    const uint32_t bps_s = runtime::get_blocks_per_sm(stats_k, kBlockThreads);
    const uint32_t grid_s = static_cast<uint32_t>(std::min<int64_t>(total_tasks, static_cast<int64_t>(bps_s) * sm_count));
    LaunchKernel(grid_s, kBlockThreads, device)(stats_k, params);

    auto fin_k = gns_finalize_kernel<DType, kUsePDL>;
    LaunchKernel(static_cast<uint32_t>(num_rows), kBlockThreads, device)(fin_k, params);

    auto apply_k = gns_apply_kernel<DType, kUsePDL>;
    const uint32_t bps_a = runtime::get_blocks_per_sm(apply_k, kBlockThreads);
    const uint32_t grid_a = static_cast<uint32_t>(std::min<int64_t>(total_tasks, static_cast<int64_t>(bps_a) * sm_count));
    LaunchKernel(grid_a, kBlockThreads, device)(apply_k, params);
  }
};
