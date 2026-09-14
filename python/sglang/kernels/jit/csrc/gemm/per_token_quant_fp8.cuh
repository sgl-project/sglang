#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>

namespace sglang {

constexpr uint32_t kPerTokenQuantWarpSize = 32;
constexpr uint32_t kPerTokenQuantTokensPerCTA = 8;

/** \brief Quantize one token per warp for large token batches. */
template <typename T, int kVecSize>
__global__ void per_token_quant_fp8_warp_kernel(
    const T* __restrict__ input,
    fp8_e4m3_t* __restrict__ output_q,
    float* __restrict__ output_s,
    uint32_t hidden_dim,
    uint32_t num_tokens) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t warp_id = threadIdx.x / kPerTokenQuantWarpSize;
  const uint32_t lane_id = threadIdx.x % kPerTokenQuantWarpSize;
  const uint32_t token_id = blockIdx.x * kPerTokenQuantTokensPerCTA + warp_id;
  if (token_id >= num_tokens) {
    return;
  }

  const T* token_input = input + token_id * hidden_dim;
  fp8_e4m3_t* token_output = output_q + token_id * hidden_dim;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t i = lane_id; i < num_vecs; i += kPerTokenQuantWarpSize) {
    input_vec_t input_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  const float scale = warp::reduce_max(max_value) / math::FP8_E4M3_MAX;
  if (lane_id == 0) {
    output_s[token_id] = scale;
  }
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  for (uint32_t i = lane_id; i < num_vecs; i += kPerTokenQuantWarpSize) {
    input_vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(token_output, i);
  }
}

/** \brief Quantize one token per CTA for small token batches. */
template <typename T, int kVecSize>
__global__ void per_token_quant_fp8_cta_kernel(
    const T* __restrict__ input, fp8_e4m3_t* __restrict__ output_q, float* __restrict__ output_s, uint32_t hidden_dim) {
  using namespace device;
  using input_vec_t = AlignedVector<T, kVecSize>;
  using output_vec_t = AlignedVector<fp8_e4m3_t, kVecSize>;

  const uint32_t token_id = blockIdx.x;
  const T* token_input = input + token_id * hidden_dim;
  fp8_e4m3_t* token_output = output_q + token_id * hidden_dim;
  const uint32_t num_vecs = hidden_dim / kVecSize;

  float max_value = 0.0f;
  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      max_value = math::max(max_value, math::abs(static_cast<float>(input_vec[j])));
    }
  }

  __shared__ float reduction_smem[kPerTokenQuantWarpSize];
  __shared__ float scale_smem;
  cta::reduce_max(max_value, reduction_smem);
  __syncthreads();
  if (threadIdx.x == 0) {
    scale_smem = reduction_smem[0] / math::FP8_E4M3_MAX;
    output_s[token_id] = scale_smem;
  }
  __syncthreads();
  const float scale_inv = 1.0f / scale_smem;

  for (uint32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    input_vec_t input_vec;
    output_vec_t output_vec;
    input_vec.load(token_input, i);
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      const float value = static_cast<float>(input_vec[j]) * scale_inv;
      output_vec[j] = static_cast<fp8_e4m3_t>(math::max(math::min(value, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX));
    }
    output_vec.store(token_output, i);
  }
}

template <typename T, int kVecSize>
void launch_per_token_quant_fp8(
    DLDevice device, const T* input, fp8_e4m3_t* output_q, float* output_s, uint32_t hidden_dim, uint32_t num_tokens) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t sm_count = host::runtime::get_sm_count(device.device_id);
  const bool use_warp_kernel = num_tokens >= sm_count * 2 * kPerTokenQuantTokensPerCTA;
  if (use_warp_kernel) {
    const uint32_t grid = host::div_ceil(num_tokens, kPerTokenQuantTokensPerCTA);
    host::LaunchKernel(grid, kBlockSize, device)(
        per_token_quant_fp8_warp_kernel<T, kVecSize>, input, output_q, output_s, hidden_dim, num_tokens);
  } else {
    host::LaunchKernel(num_tokens, kBlockSize, device)(
        per_token_quant_fp8_cta_kernel<T, kVecSize>, input, output_q, output_s, hidden_dim);
  }
}

/** \brief Validate and launch dynamic per-token FP8 E4M3 quantization. */
template <typename T>
void per_token_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;
  auto M = SymbolicSize{"num_tokens"};
  auto MOutput = SymbolicSize{"output_num_tokens"};
  auto K = SymbolicSize{"hidden_dim"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({M, K}).with_dtype<T>().with_device(device).verify(input);
  TensorMatcher({MOutput, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output_q);
  TensorMatcher({MOutput, 1}).with_dtype<float>().with_device(device).verify(output_s);

  CHECK_HOST(M.unwrap() > 0) << "per_token_quant_fp8: num_tokens must be positive";
  CHECK_HOST(MOutput.unwrap() >= M.unwrap())
      << "per_token_quant_fp8: output buffers must have at least " << M.unwrap() << " rows, got " << MOutput.unwrap();
  CHECK_HOST(K.unwrap() > 0 && K.unwrap() % 4 == 0)
      << "per_token_quant_fp8: hidden_dim must be positive and divisible by 4, got " << K.unwrap();
  CHECK_HOST(M.unwrap() <= UINT32_MAX && K.unwrap() <= UINT32_MAX)
      << "per_token_quant_fp8: dimensions exceed uint32 indexing";

  const uint32_t num_tokens = static_cast<uint32_t>(M.unwrap());
  const uint32_t hidden_dim = static_cast<uint32_t>(K.unwrap());
  constexpr uint32_t kMaxVecSize = 16 / sizeof(T);
  if (hidden_dim % kMaxVecSize == 0) {
    launch_per_token_quant_fp8<T, kMaxVecSize>(
        device.unwrap(),
        static_cast<const T*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  } else {
    launch_per_token_quant_fp8<T, 4>(
        device.unwrap(),
        static_cast<const T*>(input.data_ptr()),
        static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        hidden_dim,
        num_tokens);
  }
}

}  // namespace sglang
