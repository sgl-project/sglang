#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cstdint>

namespace {

constexpr uint32_t kFusedBlockSize = 256;

SGL_DEVICE fp8_e4m3_t quant_one(float x, float inv_scale) {
  using namespace device;
  const float scaled = x * inv_scale;
  const float clipped = math::min(math::max(scaled, -math::FP8_E4M3_MAX), math::FP8_E4M3_MAX);
  return static_cast<fp8_e4m3_t>(clipped);
}

template <typename T, int kVec>
__global__ void q_amax_kernel(
    const T* __restrict__ q_in, int q_row, int64_t q_in_row_stride, int num_tokens, float* __restrict__ amax_out) {
  using namespace device;
  using in_vec_t = AlignedVector<T, kVec>;
  const int64_t q_row_vecs = q_row / kVec;
  const int64_t total = static_cast<int64_t>(num_tokens) * q_row_vecs;
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t gstride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  float thread_max = 0.0f;
  for (int64_t idx = gtid; idx < total; idx += gstride) {
    const int64_t t = idx / q_row_vecs;
    const int64_t jv = idx % q_row_vecs;
    in_vec_t v;
    v.load(q_in + t * q_in_row_stride, jv);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      thread_max = math::max(thread_max, math::abs(static_cast<float>(v[i])));
    }
  }
  __shared__ float smem[kFusedBlockSize / 32];
  cta::reduce_max(thread_max, smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomic::max(amax_out, smem[0]);
  }
}

template <typename T, int kVec>
__global__ void apply_kernel(
    const T* __restrict__ q_in,
    fp8_e4m3_t* __restrict__ q_out,
    int q_row,
    int64_t q_in_row_stride,
    float* __restrict__ amax_io,
    int* __restrict__ done_counter,
    float* __restrict__ bmm1_out,
    float bmm1_extra,
    const T* __restrict__ k_in,
    const T* __restrict__ v_in,
    fp8_e4m3_t* __restrict__ k_cache,
    fp8_e4m3_t* __restrict__ v_cache,
    const int64_t* __restrict__ cache_loc,
    float inv_k_scale,
    float inv_v_scale,
    int num_tokens,
    int kv_row,
    int64_t k_in_row_stride,
    int64_t v_in_row_stride,
    int64_t cache_row_stride) {
  using namespace device;
  using in_vec_t = AlignedVector<T, kVec>;
  using out_vec_t = AlignedVector<fp8_e4m3_t, kVec>;

  const float amax = *amax_io;
  const float inv_q = math::FP8_E4M3_MAX / amax;
  const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t gstride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  if (gtid == 0) {
    *bmm1_out = (amax / math::FP8_E4M3_MAX) * bmm1_extra;
  }

  const int64_t q_row_vecs = q_row / kVec;
  const int64_t q_total = static_cast<int64_t>(num_tokens) * q_row_vecs;
  for (int64_t idx = gtid; idx < q_total; idx += gstride) {
    const int64_t t = idx / q_row_vecs;
    const int64_t jv = idx % q_row_vecs;
    in_vec_t v;
    v.load(q_in + t * q_in_row_stride, jv);
    out_vec_t o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      o[i] = quant_one(static_cast<float>(v[i]), inv_q);
    }
    o.store(q_out + t * q_row, jv);
  }

  const int64_t row_vecs = kv_row / kVec;
  const int64_t kv_total = static_cast<int64_t>(num_tokens) * row_vecs;
  for (int64_t idx = gtid; idx < kv_total; idx += gstride) {
    const int64_t t = idx / row_vecs;
    const int64_t jv = idx % row_vecs;
    const int64_t slot = cache_loc[t];
    in_vec_t k_vec;
    k_vec.load(k_in + t * k_in_row_stride, jv);
    out_vec_t k_o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      k_o[i] = quant_one(static_cast<float>(k_vec[i]), inv_k_scale);
    }
    k_o.store(k_cache + slot * cache_row_stride, jv);
    in_vec_t v_vec;
    v_vec.load(v_in + t * v_in_row_stride, jv);
    out_vec_t v_o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      v_o[i] = quant_one(static_cast<float>(v_vec[i]), inv_v_scale);
    }
    v_o.store(v_cache + slot * cache_row_stride, jv);
  }

  // Re-zero the persistent amax buffer for the next call. The last block to
  // finish (atomicAdd == gridDim-1) is guaranteed every other block already
  // read amax, so the reset is race-free without a separate fill kernel.
  __shared__ bool s_last;
  if (threadIdx.x == 0) {
    s_last = (atomicAdd(done_counter, 1) == gridDim.x - 1);
  }
  __syncthreads();
  if (s_last && threadIdx.x == 0) {
    *amax_io = 0.0f;
    *done_counter = 0;
  }
}

template <typename Kernel>
uint32_t plan_grid(Kernel kernel, int64_t work_items, const DLDevice& device) {
  using namespace host;
  static const uint32_t per_sm = runtime::get_blocks_per_sm(kernel, kFusedBlockSize);
  static const uint32_t num_sm = runtime::get_sm_count(device.device_id);
  const uint32_t max_grid = per_sm * num_sm;
  const int64_t want = div_ceil(work_items, static_cast<int64_t>(kFusedBlockSize));
  if (want < 1) return 1;
  return static_cast<uint32_t>(want > max_grid ? max_grid : want);
}

template <typename T>
void fused_q_quant_kv_write(
    tvm::ffi::TensorView q_in,
    tvm::ffi::TensorView q_out,
    tvm::ffi::TensorView amax_buf,
    tvm::ffi::TensorView done_counter,
    tvm::ffi::TensorView bmm1_out,
    tvm::ffi::TensorView k_in,
    tvm::ffi::TensorView v_in,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView cache_loc,
    double inv_k_scale,
    double inv_v_scale,
    double bmm1_extra,
    int64_t num_tokens,
    int64_t q_row,
    int64_t kv_row,
    int64_t q_in_row_stride,
    int64_t k_in_row_stride,
    int64_t v_in_row_stride,
    int64_t cache_row_stride) {
  using namespace host;

  auto N = SymbolicSize{"n_q"};
  auto one = SymbolicSize{"one"};
  auto nt = SymbolicSize{"num_tokens"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({N}).with_dtype<fp8_e4m3_t>().with_device(device_).verify(q_out);
  TensorMatcher({one}).with_dtype<float>().with_device(device_).verify(amax_buf).verify(bmm1_out);
  TensorMatcher({nt}).with_dtype<int64_t>().with_device(device_).verify(cache_loc);

  const DLDevice device = device_.unwrap();
  RuntimeCheck(
      static_cast<int64_t>(nt.unwrap()) == num_tokens,
      "fused_q_quant_kv_write: cache_loc length ",
      nt.unwrap(),
      " != num_tokens ",
      num_tokens);

  constexpr int kVec = 16 / sizeof(T);
  RuntimeCheck(q_row % kVec == 0, "fused_q_quant_kv_write: q_row ", q_row, " must be a multiple of ", kVec);
  RuntimeCheck(kv_row % kVec == 0, "fused_q_quant_kv_write: kv_row ", kv_row, " must be a multiple of ", kVec);

  auto* kA = q_amax_kernel<T, kVec>;
  auto* kB = apply_kernel<T, kVec>;

  const int64_t q_vecs = num_tokens * (q_row / kVec);
  const int64_t kv_vecs = num_tokens * (kv_row / kVec);
  const int64_t apply_work = q_vecs > kv_vecs ? q_vecs : kv_vecs;

  LaunchKernel(plan_grid(kA, q_vecs, device), kFusedBlockSize, device)(
      kA,
      static_cast<const T*>(q_in.data_ptr()),
      static_cast<int>(q_row),
      q_in_row_stride,
      static_cast<int>(num_tokens),
      static_cast<float*>(amax_buf.data_ptr()));

  LaunchKernel(plan_grid(kB, apply_work, device), kFusedBlockSize, device)(
      kB,
      static_cast<const T*>(q_in.data_ptr()),
      static_cast<fp8_e4m3_t*>(q_out.data_ptr()),
      static_cast<int>(q_row),
      q_in_row_stride,
      static_cast<float*>(amax_buf.data_ptr()),
      static_cast<int*>(done_counter.data_ptr()),
      static_cast<float*>(bmm1_out.data_ptr()),
      static_cast<float>(bmm1_extra),
      static_cast<const T*>(k_in.data_ptr()),
      static_cast<const T*>(v_in.data_ptr()),
      static_cast<fp8_e4m3_t*>(k_cache.data_ptr()),
      static_cast<fp8_e4m3_t*>(v_cache.data_ptr()),
      static_cast<const int64_t*>(cache_loc.data_ptr()),
      static_cast<float>(inv_k_scale),
      static_cast<float>(inv_v_scale),
      static_cast<int>(num_tokens),
      static_cast<int>(kv_row),
      k_in_row_stride,
      v_in_row_stride,
      cache_row_stride);
}

}  // namespace
