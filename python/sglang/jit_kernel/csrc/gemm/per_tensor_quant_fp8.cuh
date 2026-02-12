#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/atomic.cuh>
#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t kBlockSize = 256;

// each warp will handle 512B data
template <typename T>
__global__ void
per_tensor_absmax_kernel(const T* __restrict__ input, float* __restrict__ output_s, const int64_t num_elements) {
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
    [[unlikely]];  // poorly aligned case, do not optimize
    const auto remainder = num_elements - gid * VEC_SIZE;
    for (uint32_t i = 0; i < remainder; ++i) {
      const float value = static_cast<float>(input[gid * VEC_SIZE + i]);
      max_value = math::max(max_value, math::abs(value));
    }
  }

  // reduce within block and then atomic reduce between blocks
  __shared__ float smem[kWarpThreads];
  cta::reduce_max(max_value, smem);
  if (threadIdx.x == 0) {
    const auto max_value = smem[0];
    atomic::max(output_s, max_value / math::FP8_E4M3_MAX);
  }
}

[[maybe_unused]]
SGL_DEVICE float fp8_e4m3_clip(float val) {
  namespace math = device::math;
  return math::max(math::min(val, math::FP8_E4M3_MAX), -math::FP8_E4M3_MAX);
}

template <typename T, typename DST_DTYPE>
__global__ void per_tensor_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output,
    const float* __restrict__ scale,
    const int64_t num_elements) {
  using namespace device;
  constexpr uint32_t VEC_SIZE = 16 / sizeof(T);

  const int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const float scale_val = 1.0f / (*scale);

  if (gid * VEC_SIZE + VEC_SIZE <= num_elements) {
    using input_vec_t = AlignedVector<T, VEC_SIZE>;
    using output_vec_t = AlignedVector<DST_DTYPE, VEC_SIZE>;
    const auto gmem_in = tile::Memory<input_vec_t>::thread();
    const auto gmem_out = tile::Memory<output_vec_t>::thread();
    const auto input_vec = gmem_in.load(input, gid);
    output_vec_t output_vec;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      const float value = fp8_e4m3_clip(static_cast<float>(input_vec[i]) * scale_val);
      output_vec[i] = static_cast<DST_DTYPE>(value);
    }
    gmem_out.store(output, output_vec, gid);
  } else if (gid * VEC_SIZE < num_elements) {
    [[unlikely]];  // poorly aligned case, do not optimize
    const auto remainder = num_elements - gid * VEC_SIZE;
    for (uint32_t i = 0; i < remainder; ++i) {
      const float value = fp8_e4m3_clip(static_cast<float>(input[gid * VEC_SIZE + i]) * scale_val);
      output[gid * VEC_SIZE + i] = static_cast<DST_DTYPE>(value);
    }
  }
}

template <bool kIsStatic, typename DType>
void per_tensor_quant_fp8(tvm::ffi::TensorView input, tvm::ffi::TensorView output_q, tvm::ffi::TensorView output_s) {
  using namespace host;

  auto device = SymbolicDevice{};
  auto N = SymbolicSize{"num_elements"};
  device.set_options<kDLCUDA>();

  TensorMatcher({N})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);
  TensorMatcher({N})  //
      .with_dtype<fp8_e4m3_t>()
      .with_device(device)
      .verify(output_q);
  TensorMatcher({1})  //
      .with_dtype<float>()
      .with_device(device)
      .verify(output_s);

  const auto num_elements = N.unwrap();

  constexpr size_t kElementsPerBlock = kBlockSize * (16 / sizeof(DType));
  const uint32_t num_blocks = div_ceil(num_elements, kElementsPerBlock);

  if constexpr (!kIsStatic) {
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
        per_tensor_absmax_kernel<DType>,
        static_cast<const DType*>(input.data_ptr()),
        static_cast<float*>(output_s.data_ptr()),
        static_cast<int64_t>(num_elements));
  }

  LaunchKernel(num_blocks, kBlockSize, device.unwrap())(
      per_tensor_quant_fp8_kernel<DType, fp8_e4m3_t>,
      static_cast<const DType*>(input.data_ptr()),
      static_cast<fp8_e4m3_t*>(output_q.data_ptr()),
      static_cast<const float*>(output_s.data_ptr()),
      static_cast<int64_t>(num_elements));
}

}  // namespace
