#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

template <typename DType, int kVecSize>
__global__ void per_token_quant_fp8_kernel(const DType* __restrict__ input,
                                           fp8_e4m3_t* __restrict__ output_q,
                                           float* __restrict__ output_s,
                                           const int64_t hidden_dim) {
  using namespace device;
  using packed_dt = packed_t<DType>;
  constexpr int kPackedVecSize = kVecSize / 2;

  const int64_t token_idx = blockIdx.x;

  const auto* token_input = reinterpret_cast<const packed_dt*>(input + token_idx * hidden_dim);
  fp8_e4m3_t* token_output = output_q + token_idx * hidden_dim;

  const int32_t num_vec_elems = hidden_dim / kVecSize;
  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  float max_value = 0.0f;

  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    AlignedVector<packed_dt, kPackedVecSize> vec;
    vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kPackedVecSize; ++j) {
      fp32x2_t val = cast<fp32x2_t>(vec[j]);
      max_value = math::max(max_value, math::max(math::abs(val.x), math::abs(val.y)));
    }
  }

  __shared__ float smem[kWarpThreads];
  cta::reduce_max(max_value, smem);
  __syncthreads();

  const float absmax = smem[0];
  const float scale = absmax / math::FP8_E4M3_MAX;
  if (tid == 0) {
    output_s[token_idx] = scale;
  }
  const float scale_inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;

  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    AlignedVector<packed_dt, kPackedVecSize> in_vec;
    in_vec.load(token_input, i);

    AlignedVector<fp8_e4m3_t, kVecSize> out_vec;
#pragma unroll
    for (int j = 0; j < kPackedVecSize; ++j) {
      fp32x2_t val = cast<fp32x2_t>(in_vec[j]);
      val.x = math::max(math::min(val.x * scale_inv, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      val.y = math::max(math::min(val.y * scale_inv, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      out_vec[2 * j] = static_cast<fp8_e4m3_t>(val.x);
      out_vec[2 * j + 1] = static_cast<fp8_e4m3_t>(val.y);
    }
    out_vec.store(token_output, i);
  }
}

// ---------------------------------------------------------------------------
// Warp-local kernel: 8 tokens per CTA (one warp per token).
// Uses shared memory to cache input when hidden_dim is small, avoiding
// a second global memory read in Pass 2.
// ---------------------------------------------------------------------------
template <typename DType, int kVecSize, bool kUseSmem>
__global__ void per_token_quant_fp8_warp_kernel(const DType* __restrict__ input,
                                                fp8_e4m3_t* __restrict__ output_q,
                                                float* __restrict__ output_s,
                                                const int64_t hidden_dim,
                                                const int64_t num_tokens) {
  using namespace device;
  using packed_dt = packed_t<DType>;
  constexpr int kPackedVecSize = kVecSize / 2;
  constexpr int kTokensPerCTA = 8;

  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane_id = threadIdx.x & (kWarpThreads - 1);
  const int64_t token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  const auto* token_input = reinterpret_cast<const packed_dt*>(input + token_id * hidden_dim);
  fp8_e4m3_t* token_output = output_q + token_id * hidden_dim;

  // Shared memory: each warp gets its own padded slice
  extern __shared__ char smem_buffer[];
  constexpr int kSmemPadding = 32;
  const int warp_smem_stride = (hidden_dim * sizeof(DType) + kSmemPadding - 1) / kSmemPadding * kSmemPadding;
  auto* shared_input = reinterpret_cast<DType*>(smem_buffer + warp_id * warp_smem_stride);

  const int32_t num_vec_elems = hidden_dim / kVecSize;
  float max_value = 0.0f;

  // Pass 1: Load from global, optionally cache to smem, compute absmax
  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpThreads) {
    AlignedVector<packed_dt, kPackedVecSize> vec;
    vec.load(token_input, i);

    if constexpr (kUseSmem) {
#pragma unroll
      for (int j = 0; j < kPackedVecSize; ++j) {
        reinterpret_cast<packed_dt*>(shared_input)[i * kPackedVecSize + j] = vec[j];
      }
    }

#pragma unroll
    for (int j = 0; j < kPackedVecSize; ++j) {
      fp32x2_t val = cast<fp32x2_t>(vec[j]);
      max_value = math::max(max_value, math::max(math::abs(val.x), math::abs(val.y)));
    }
  }

  if constexpr (kUseSmem) {
    __syncwarp();
  }

  max_value = warp::reduce_max(max_value);

  const float scale = max_value / math::FP8_E4M3_MAX;
  if (lane_id == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;

  // Pass 2: Quantize (read from smem or re-read global)
  for (int32_t i = lane_id; i < num_vec_elems; i += kWarpThreads) {
    AlignedVector<packed_dt, kPackedVecSize> in_vec;

    if constexpr (kUseSmem) {
#pragma unroll
      for (int j = 0; j < kPackedVecSize; ++j) {
        in_vec[j] = reinterpret_cast<packed_dt*>(shared_input)[i * kPackedVecSize + j];
      }
    } else {
      in_vec.load(token_input, i);
    }

    AlignedVector<fp8_e4m3_t, kVecSize> out_vec;
#pragma unroll
    for (int j = 0; j < kPackedVecSize; ++j) {
      fp32x2_t val = cast<fp32x2_t>(in_vec[j]);
      val.x = math::max(math::min(val.x * scale_inv, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      val.y = math::max(math::min(val.y * scale_inv, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
      out_vec[2 * j] = static_cast<fp8_e4m3_t>(val.x);
      out_vec[2 * j + 1] = static_cast<fp8_e4m3_t>(val.y);
    }
    out_vec.store(token_output, i);
  }
}

template <typename DType>
void per_token_quant_fp8(tvm::ffi::TensorView input,
                         tvm::ffi::TensorView output_q,
                         tvm::ffi::TensorView output_s) {
  using namespace host;

  auto M = SymbolicSize{"num_tokens"};
  auto D = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, D})  //
      .with_strides({D, 1})
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({M, D})  //
      .with_strides({D, 1})
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(output_q);
  TensorMatcher({M})  //
      .with_dtype<fp32_t>()
      .with_device(device)
      .verify(output_s);

  const auto num_tokens = static_cast<int64_t>(M.unwrap());
  const auto hidden_dim = static_cast<int64_t>(D.unwrap());

  RuntimeCheck(hidden_dim % 4 == 0, "per_token_quant_fp8: hidden_dim must be divisible by 4, got ", hidden_dim);

  constexpr int kMaxVecSize = device::kMaxVecBytes / sizeof(DType);
  constexpr int kTokensPerCTA = 8;
  constexpr int kWarpSize = 32;
  constexpr int kWarpBlockSize = kTokensPerCTA * kWarpSize;  // 256
  constexpr int kSmallBatchBlockSize = 256;
  constexpr std::size_t kSmemThreshold = 48 * 1024;  // 48 KB

  const auto* input_ptr = static_cast<const DType*>(input.data_ptr());
  auto* output_q_ptr = static_cast<fp8_e4m3_t*>(output_q.data_ptr());
  auto* output_s_ptr = static_cast<float*>(output_s.data_ptr());

  const auto dl_device = device.unwrap();
  const auto sm_count = runtime::get_sm_count(dl_device.device_id);
  const bool use_warp_kernel = (num_tokens >= static_cast<int64_t>(sm_count) * 2 * kTokensPerCTA);

  // Shared memory sizing for warp kernel
  constexpr int kSmemPadding = 32;
  const int warp_smem_stride = (hidden_dim * sizeof(DType) + kSmemPadding - 1) / kSmemPadding * kSmemPadding;
  const std::size_t dynamic_smem_bytes = static_cast<std::size_t>(warp_smem_stride) * kTokensPerCTA;
  const bool use_smem = (hidden_dim < 2048) && (dynamic_smem_bytes < kSmemThreshold);

  auto launch_warp = [&](auto kernel, std::size_t smem) {
    const dim3 grid((num_tokens + kTokensPerCTA - 1) / kTokensPerCTA);
    LaunchKernel(grid, kWarpBlockSize, dl_device, smem)(
        kernel, input_ptr, output_q_ptr, output_s_ptr, hidden_dim, num_tokens);
  };

  auto launch_small_batch = [&](auto kernel) {
    LaunchKernel(num_tokens, kSmallBatchBlockSize, dl_device)(
        kernel, input_ptr, output_q_ptr, output_s_ptr, hidden_dim);
  };

  #define DISPATCH_WARP_KERNEL(VECSIZE)                                                        \
    if (use_smem) {                                                                            \
      launch_warp(per_token_quant_fp8_warp_kernel<DType, VECSIZE, true>, dynamic_smem_bytes);  \
    } else {                                                                                   \
      launch_warp(per_token_quant_fp8_warp_kernel<DType, VECSIZE, false>, 0);                  \
    }

  if (hidden_dim % kMaxVecSize == 0) {
    if (use_warp_kernel) {
      DISPATCH_WARP_KERNEL(kMaxVecSize);
    } else {
      launch_small_batch(per_token_quant_fp8_kernel<DType, kMaxVecSize>);
    }
  } else if constexpr (kMaxVecSize > 8) {
    if (hidden_dim % 8 == 0) {
      if (use_warp_kernel) {
        DISPATCH_WARP_KERNEL(8);
      } else {
        launch_small_batch(per_token_quant_fp8_kernel<DType, 8>);
      }
    } else {
      if (use_warp_kernel) {
        DISPATCH_WARP_KERNEL(4);
      } else {
        launch_small_batch(per_token_quant_fp8_kernel<DType, 4>);
      }
    }
  } else {
    if (use_warp_kernel) {
      DISPATCH_WARP_KERNEL(4);
    } else {
      launch_small_batch(per_token_quant_fp8_kernel<DType, 4>);
    }
  }
  #undef DISPATCH_WARP_KERNEL
}

}  // namespace
