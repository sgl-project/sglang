#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sglang {

template <typename T, int kVecN>
__global__ void gelu_tanh_cat_kernel(
    const T* __restrict__ attn,
    const T* __restrict__ mlp,
    T* __restrict__ output,
    uint32_t num_vecs,
    uint32_t attn_vecs,
    uint32_t mlp_vecs) {
  using vec_t = device::AlignedVector<T, kVecN>;
  const uint32_t row_vecs = attn_vecs + mlp_vecs;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vecs; i += stride) {
    const uint32_t row = i / row_vecs;
    const uint32_t col = i % row_vecs;
    vec_t value;
    if (col < attn_vecs) {
      value.load(attn, row * attn_vecs + col);
    } else {
      value.load(mlp, row * mlp_vecs + col - attn_vecs);
#pragma unroll
      for (int j = 0; j < kVecN; ++j) {
        const float x = device::cast<fp32_t>(value[j]);
        // Match aten's tanh GELU operation order, including the BF16 store
        // that precedes the eager concatenation.
        constexpr float kBeta = 0.7978845608028654f;
        constexpr float kKappa = 0.044715f;
        const float cube = x * x * x;
        const float inner = kBeta * (x + kKappa * cube);
        value[j] = device::cast<T>(0.5f * x * (1.0f + tanhf(inner)));
      }
    }
    value.store(output, i);
  }
}

template <typename T>
void gelu_tanh_cat(tvm::ffi::TensorView attn, tvm::ffi::TensorView mlp, tvm::ffi::TensorView output) {
  using namespace host;
  auto rows = SymbolicSize{"rows"};
  auto attn_width = SymbolicSize{"attn_width"};
  auto mlp_width = SymbolicSize{"mlp_width"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  TensorMatcher({rows, attn_width}).with_dtype<T>().with_device(device_).verify(attn);
  TensorMatcher({rows, mlp_width}).with_dtype<T>().with_device(device_).verify(mlp);
  const int64_t a = attn_width.unwrap();
  const int64_t m = mlp_width.unwrap();
  TensorMatcher({rows, a + m}).with_dtype<T>().with_device(device_).verify(output);
  constexpr int kVecN = 16 / sizeof(T);
  CHECK_HOST(a > 0 && m > 0 && a % kVecN == 0 && m % kVecN == 0)
      << "gelu_tanh_cat: both widths must be positive multiples of " << kVecN;
  CHECK_HOST(rows.unwrap() > 0) << "gelu_tanh_cat: rows must be positive";
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(attn.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(mlp.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0)
      << "gelu_tanh_cat: tensors must be 16-byte aligned";
  const int64_t n = rows.unwrap() * ((a + m) / kVecN);
  CHECK_HOST(n <= std::numeric_limits<int32_t>::max()) << "gelu_tanh_cat: tensor is too large";
  constexpr int kBlockSize = 256;
  const auto kernel = gelu_tanh_cat_kernel<T, kVecN>;
  const int64_t occupancy = runtime::get_blocks_per_sm(kernel, kBlockSize);
  const int64_t sms = runtime::get_sm_count(device_.unwrap().device_id);
  const int64_t grid = std::min(sms * occupancy, div_ceil(n, int64_t{kBlockSize}));
  LaunchKernel(grid, kBlockSize, device_.unwrap())(
      kernel,
      static_cast<const T*>(attn.data_ptr()),
      static_cast<const T*>(mlp.data_ptr()),
      static_cast<T*>(output.data_ptr()),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(a / kVecN),
      static_cast<uint32_t>(m / kVecN));
}

}  // namespace sglang
