#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Elementwise float -> fp8 e4m3 through the same `pack_fp8` every fp8 store in the
// DSv4 tree goes through. It is here so that cast can be pinned against torch on its
// own: a wrong rounding or saturation boundary in there does not show up as a failure
// in the fused kernels -- it just looks like fp8 quantizing worse than it should.

namespace sglang {

constexpr size_t kCvtBlockSize = 256;

__global__ void cvt_fp8_e4m3_kernel(uint8_t* dst, const float* src, size_t num_pairs) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs) return;
  reinterpret_cast<fp8x2_e4m3_t*>(dst)[idx] = deepseek_v4::fp8::pack_fp8(src[2 * idx], src[2 * idx + 1]);
}

void cvt_fp8_e4m3(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  auto N = SymbolicSize{"num_elements"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLGPU>();

  TensorMatcher({N}).with_strides({1}).with_dtype<float>().with_device(device_).verify(src);
  TensorMatcher({N}).with_strides({1}).with_dtype<uint8_t>().with_device(device_).verify(dst);

  const size_t num_elements = N.unwrap();
  // pack_fp8 converts two values at a time
  RuntimeCheck(num_elements > 0 && num_elements % 2 == 0, "num_elements must be even and non-zero, got ", num_elements);

  const size_t num_pairs = num_elements / 2;
  LaunchKernel(div_ceil(num_pairs, kCvtBlockSize), kCvtBlockSize, device_.unwrap())(
      cvt_fp8_e4m3_kernel, static_cast<uint8_t*>(dst.data_ptr()), static_cast<const float*>(src.data_ptr()), num_pairs);
}

}  // namespace sglang
