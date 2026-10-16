#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace sglang {

/// \brief Evaluate NewGELU or QuickGELU in FP32.
template <bool kQuick>
SGL_DEVICE float gelu_f32(float x) {
  if constexpr (kQuick) {
    return x / (1.0f + expf(-1.702f * x));
  } else {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
  }
}

/// \brief Vectorized GELU with a scalar tail; kVecSize=1 handles unaligned pointers.
template <typename T, bool kQuick, bool kUsePDL, uint32_t kVecSize>
__global__ void gelu_kernel(T* __restrict__ out, const T* __restrict__ input, uint32_t n) {
  using vec_t = device::AlignedVector<T, kVecSize>;
  const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t offset = tid * kVecSize;
  if (offset >= n) return;
  device::PDLWaitPrimary<kUsePDL>();
  if (n - offset >= kVecSize) {
    auto values = device::load_as<vec_t>(input, tid);
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      values[i] = device::cast<T>(gelu_f32<kQuick>(device::cast<fp32_t>(values[i])));
    }
    device::store_as<vec_t>(out, values, tid);
  } else {
    for (uint64_t i = offset; i < n; ++i) {
      out[i] = device::cast<T>(gelu_f32<kQuick>(device::cast<fp32_t>(input[i])));
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/// \brief Validate and launch GELU at a fixed vector width.
template <typename T, bool kQuick, bool kUsePDL, uint32_t kWidth>
void launch_gelu(tvm::ffi::TensorView input, tvm::ffi::TensorView out) {
  using namespace host;
  auto N = SymbolicSize{"num_elements"};
  auto device_ = SymbolicDevice{};
  // TensorMatcher stores a span into the temporary shape list. Keep every
  // verification in this full expression so that the list stays alive.
  TensorMatcher({N})
      .template with_dtype<T>()
      .template with_device<kDLCUDA>(device_)
      .ensure_alignment(kWidth * sizeof(T))
      .verify(input)
      .verify(out);
  const auto n = N.unwrap();
  CHECK_HOST(n <= std::numeric_limits<uint32_t>::max()) << "too many GELU elements for 32-bit indexing";
  if (n == 0) return;

  constexpr uint32_t kBlockSize = 256;
  const auto blocks = div_ceil(div_ceil(n, static_cast<int64_t>(kWidth)), static_cast<int64_t>(kBlockSize));
  LaunchKernel(blocks, kBlockSize, device_.unwrap())
      .enable_pdl(kUsePDL)(
          gelu_kernel<T, kQuick, kUsePDL, kWidth>,
          static_cast<T*>(out.data_ptr()),
          static_cast<const T*>(input.data_ptr()),
          static_cast<uint32_t>(n));
}

/// \brief Select scalar GELU for small inputs or unaligned pointers.
template <typename T, bool kQuick, bool kUsePDL>
void gelu(tvm::ffi::TensorView input, tvm::ffi::TensorView out) {
  constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(T);
  const auto aligned = (reinterpret_cast<uintptr_t>(input.data_ptr()) % device::kMaxVecBytes == 0) &&
                       (reinterpret_cast<uintptr_t>(out.data_ptr()) % device::kMaxVecBytes == 0);
  // Scalar accesses provide more parallel work for small/medium inputs.
  // Benchmarks favor scalar through 64K elements and vectors at 1M elements.
  // Keep the vectorized path for larger inputs to reduce memory instructions.
  if (aligned && input.numel() > 65536) {
    launch_gelu<T, kQuick, kUsePDL, kVecSize>(input, out);
  } else {
    launch_gelu<T, kQuick, kUsePDL, 1>(input, out);
  }
}

}  // namespace sglang
