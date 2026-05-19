#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil

#include <sgl_kernel/atomic.cuh>  // For atomic::max
#include <sgl_kernel/cta.cuh>     // For cta::reduce_max
#include <sgl_kernel/math.cuh>    // For math::max, math::abs, math::FP8_E4M3_MAX
#include <sgl_kernel/tile.cuh>    // For tile::Memory
#include <sgl_kernel/utils.cuh>   // For LaunchKernel, kWarpThreads, fp16_t, bf16_t, fp32_t
#include <sgl_kernel/vec.cuh>     // For AlignedVector
#include <sgl_kernel/warp.cuh>    // For warp reduction (used by cta::reduce_max)

#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t kBlockSize = 256;

// Identical to per_tensor_absmax_kernel in per_tensor_quant_fp8.cuh,
// but exposed as a standalone entry point so callers that only need
// the scale (without the quantization pass) can avoid the extra write.
template <typename T>
__global__ void
per_tensor_absmax_fp8_kernel(const T* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
  using namespace device;
  constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  const int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  float max_value = 0.0f;
  if (gid * VEC_SIZE + VEC_SIZE <= num_elements) {
    using vec_t = AlignedVector<T, VEC_SIZE>;
    const auto gmem_in = tile::Memory<vec_t>::thread();
    const auto input_vec = gmem_in.load(input, gid);
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      const float value = static_cast<float>(input_vec[i]);
      max_value = math::max(max_value, math::abs(value));
    }
  } else if (gid * VEC_SIZE < num_elements) {
    [[unlikely]];
    const auto remainder = num_elements - gid * VEC_SIZE;
    for (uint32_t i = 0; i < remainder; ++i) {
      const float value = static_cast<float>(input[gid * VEC_SIZE + i]);
      max_value = math::max(max_value, math::abs(value));
    }
  }

  __shared__ float smem[kWarpThreads];
  cta::reduce_max(max_value, smem);
  if (threadIdx.x == 0) {
    const auto max_value = smem[0];
    atomic::max(output_s, max_value / math::FP8_E4M3_MAX);
  }
}

template <typename DType>
void per_tensor_absmax_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_s) {
  using namespace host;

  auto N = SymbolicSize{"num_elements"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({1})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(output_s);

  const auto num_elements = N.unwrap();

  constexpr size_t kElementsPerBlock = kBlockSize * (16 / sizeof(DType));
  const uint32_t num_blocks = div_ceil(num_elements, kElementsPerBlock);

  LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
      per_tensor_absmax_fp8_kernel<DType>,
      static_cast<const DType*>(input.data_ptr()),
      static_cast<float*>(output_s.data_ptr()),
      static_cast<int64_t>(num_elements));
}

}  // namespace
