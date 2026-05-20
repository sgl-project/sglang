#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDType, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/runtime.cuh>  // For runtime::get_blocks_per_sm, get_sm_count
#include <sgl_kernel/type.cuh>     // For dtype_trait, fp16_t, bf16_t, fp32_t, packed_t, cast
#include <sgl_kernel/utils.cuh>    // For SGL_DEVICE, LaunchKernel, PDLWait/Trigger
#include <sgl_kernel/vec.cuh>      // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "fast_hadamard_transform_common.h"  // For hadamard_mult_thread, hadamard_mult_warp, cilog2
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

struct FusedRopeHadamardParams {
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;  // NOTE: this k is pre-offset in host code to reduce computation in kernel
  const void* __restrict__ cos_sin_cache_ptr;
  const void* __restrict__ positions;
  int64_t q_stride_bytes;
  int64_t k_stride_bytes;
  int64_t head_stride_bytes;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float had_scale;
};

constexpr uint32_t kBlockSize = 128;

template <bool kIsNeox, int kHeadDim, int kRopeDim, bool kUsePDL, typename DType, typename IdType>
__global__ __launch_bounds__(kBlockSize)  //
    void fused_rope_hadamard_kernel(const __grid_constant__ FusedRopeHadamardParams params) {
  using namespace device;

  static_assert(sizeof(DType) == 2, "fused_rope_hadamard: only fp16/bf16 supported");

  constexpr int kNElts = 16 / sizeof(DType);
  constexpr int kWorkThreads = kHeadDim / kNElts;
  constexpr int kLogNElts = cilog2(kNElts);
  constexpr int kLogWorkThreads = cilog2(kWorkThreads);
  constexpr int kNChunks = 1;
  constexpr int kVecSize = kNElts / 2;
  constexpr int kRopeHalf = kRopeDim / 2;
  constexpr int kRopeXThreads = kRopeHalf / kNElts;
  constexpr int kRopeYThreads = kRopeDim / kNElts;
  constexpr int kWorkersPerBlock = kBlockSize / kWorkThreads;
  constexpr int64_t kCosSinStrideBytes = kRopeDim * sizeof(float);

  static_assert(kHeadDim > 0 && (kHeadDim & (kHeadDim - 1)) == 0, "kHeadDim must be a power of 2");
  static_assert(kRopeDim > 0 && kRopeDim <= kHeadDim, "kRopeDim must be in (0, kHeadDim]");
  static_assert(kRopeDim % (2 * kNElts) == 0);
  static_assert(kWorkThreads >= 4 && kWorkThreads <= 32);
  static_assert(kBlockSize % kWorkThreads == 0);
  static_assert((1 << kLogNElts) == kNElts);
  static_assert((1 << kLogWorkThreads) == kWorkThreads);
  static_assert(kVecSize >= 1);

  using DType2 = packed_t<DType>;
  using InputStorage = AlignedVector<DType2, kVecSize>;

  const uint32_t lane_id = threadIdx.x % kWorkThreads;
  const uint32_t worker_in_block = threadIdx.x / kWorkThreads;
  const uint32_t start_worker_id = blockIdx.x * kWorkersPerBlock + worker_in_block;
  const uint32_t total_workers = (params.num_qo_heads + params.num_kv_heads) * params.num_tokens;
  const uint32_t worker_stride = gridDim.x * kWorkersPerBlock;

  const auto cos_cache_ptr = params.cos_sin_cache_ptr;
  const auto sin_cache_ptr = pointer::offset(cos_cache_ptr, kCosSinStrideBytes / 2);

  PDLWaitPrimary<kUsePDL>();

  for (uint32_t work_id = start_worker_id; work_id < total_workers; work_id += worker_stride) {
    const uint32_t num_q_and_k_heads = params.num_qo_heads + params.num_kv_heads;
    const uint32_t token_id = work_id / num_q_and_k_heads;
    const uint32_t head_id = work_id % num_q_and_k_heads;
    const bool load_q = head_id < params.num_qo_heads;
    const auto pos = static_cast<const IdType*>(params.positions)[token_id];

    const auto input = pointer::offset(
        load_q ? params.q_ptr : params.k_ptr,
        token_id * (load_q ? params.q_stride_bytes : params.k_stride_bytes),
        head_id * params.head_stride_bytes);
    const auto cos_ptr = pointer::offset(cos_cache_ptr, pos * kCosSinStrideBytes);
    const auto sin_ptr = pointer::offset(sin_cache_ptr, pos * kCosSinStrideBytes);

    auto input_vec = load_as<InputStorage>(input, lane_id);
    float x_vals[kNChunks][kNElts];
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const auto [a, b] = cast<fp32x2_t>(input_vec[j]);
      x_vals[0][2 * j] = a;
      x_vals[0][2 * j + 1] = b;
    }

    if constexpr (kIsNeox) {
      using CacheStorage = AlignedVector<fp32x2_t, kVecSize>;
      const uint32_t lane_in_rope = lane_id & (kRopeXThreads - 1);
      const auto cos_pair = load_as<CacheStorage>(cos_ptr, lane_in_rope);
      const auto sin_pair = load_as<CacheStorage>(sin_ptr, lane_in_rope);
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        const auto [cos_0, cos_1] = cos_pair[j];
        const auto [sin_0, sin_1] = sin_pair[j];
        for (int sub = 0; sub < 2; ++sub) {
          const float cos_v = (sub == 0) ? cos_0 : cos_1;
          const float sin_v = (sub == 0) ? sin_0 : sin_1;
          const int idx = 2 * j + sub;
          const float my_val = x_vals[0][idx];
          const float pair_val = __shfl_xor_sync(0xFFFFFFFFu, my_val, kRopeXThreads, kWorkThreads);
          if (lane_id < kRopeXThreads) {
            x_vals[0][idx] = my_val * cos_v - pair_val * sin_v;
          } else if (lane_id < kRopeYThreads) {
            x_vals[0][idx] = pair_val * sin_v + my_val * cos_v;
          }
        }
      }
    } else {
      using CacheStorage = AlignedVector<float, kVecSize>;
      if (lane_id < kRopeYThreads) {
        const auto cos_vec = load_as<CacheStorage>(cos_ptr, lane_id);
        const auto sin_vec = load_as<CacheStorage>(sin_ptr, lane_id);
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          const float x = x_vals[0][2 * j];
          const float y = x_vals[0][2 * j + 1];
          const float cos = cos_vec[j];
          const float sin = sin_vec[j];
          x_vals[0][2 * j] = x * cos - y * sin;
          x_vals[0][2 * j + 1] = x * sin + y * cos;
        }
      }
    }

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWorkThreads, 0, kNChunks, kNElts>(x_vals);

    InputStorage out_vec;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      out_vec[j] =
          cast<DType2, fp32x2_t>({x_vals[0][2 * j] * params.had_scale, x_vals[0][2 * j + 1] * params.had_scale});
    }
    store_as<InputStorage>(input, out_vec, lane_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kIsNeox, int kHeadDim, int kRopeDim, bool kUsePDL, typename DType>
struct FusedRopeHadamardKernel {
  static constexpr int kNElts = 16 / sizeof(DType);
  static constexpr int kWorkThreads = kHeadDim / kNElts;

  template <typename IdType>
  static constexpr auto _kernel = fused_rope_hadamard_kernel<kIsNeox, kHeadDim, kRopeDim, kUsePDL, DType, IdType>;

  static auto get_num_sm(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto Q = SymbolicSize{"num_qo_heads"};
    auto K = SymbolicSize{"num_kv_heads"};
    auto Hd = SymbolicSize{"head_dim"};
    auto Rd = SymbolicSize{"rope_dim"};
    auto Dq = SymbolicSize{"q_stride"};
    auto Dk = SymbolicSize{"k_stride"};
    auto Dh = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    Hd.set_value(kHeadDim);
    Rd.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, Q, Hd})  // q input
        .with_strides({Dq, Dh, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, K, Hd})  // k input
        .with_strides({Dk, Dh, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({-1, Rd})  // cos_sin_cache
        .with_dtype<float>()
        .with_device(device)
        .verify(cos_sin_cache);
    TensorMatcher({N})  // positions
        .with_dtype<int32_t, int64_t>(id_type)
        .with_device(device)
        .verify(positions);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto num_qo_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_kv_heads = static_cast<uint32_t>(K.unwrap());
    const auto q_stride_bytes = static_cast<int64_t>(Dq.unwrap()) * sizeof(DType);
    const auto k_stride_bytes = static_cast<int64_t>(Dk.unwrap()) * sizeof(DType);
    const auto head_stride_bytes = static_cast<int64_t>(Dh.unwrap()) * sizeof(DType);

    // NOTE: we offset the k here to reduce computation cost in the kernel
    const int64_t k_offset = static_cast<int64_t>(num_qo_heads) * head_stride_bytes;

    FusedRopeHadamardParams params;
    std::memset(&params, 0, sizeof(params));
    params.q_ptr = const_cast<void*>(q.data_ptr());
    params.k_ptr = pointer::offset(const_cast<void*>(k.data_ptr()), -k_offset);
    params.cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    params.positions = positions.data_ptr();
    params.q_stride_bytes = q_stride_bytes;
    params.k_stride_bytes = k_stride_bytes;
    params.head_stride_bytes = head_stride_bytes;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.num_tokens = num_tokens;
    params.had_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    const DLDevice dev = device.unwrap();
    const auto is_int32 = id_type.is_type<int32_t>();
    const auto kernel = is_int32 ? _kernel<int32_t> : _kernel<int64_t>;

    static const uint32_t kOccupancy[2] = {
        runtime::get_blocks_per_sm(_kernel<int32_t>, kBlockSize),
        runtime::get_blocks_per_sm(_kernel<int64_t>, kBlockSize),
    };
    const uint32_t num_sm = get_num_sm(dev);
    const uint32_t max_blocks = num_sm * kOccupancy[is_int32 ? 0 : 1];
    const uint32_t total_workers = (num_qo_heads + num_kv_heads) * num_tokens;
    constexpr uint32_t kWorkersPerBlock = kBlockSize / kWorkThreads;
    const uint32_t needed_blocks = div_ceil(total_workers, kWorkersPerBlock);
    const uint32_t num_blocks = std::min(max_blocks, needed_blocks);

    LaunchKernel(num_blocks, kBlockSize, dev)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
