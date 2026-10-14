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
  CHECK_HOST(attn.ndim() >= 2) << "gelu_tanh_cat: expected at least two dimensions";
  auto dtype = SymbolicDType{};
  dtype.set_options<T>();
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();
  for (const auto tensor : {attn, mlp, output}) {
    dtype.verify(tensor.dtype());
    device_.verify(tensor.device());
    CHECK_HOST(tensor.ndim() == attn.ndim() && tensor.is_contiguous())
        << "gelu_tanh_cat: expected contiguous tensors of equal rank";
    for (int64_t dim = 0; dim < attn.ndim() - 1; ++dim) {
      CHECK_HOST(tensor.size(dim) == attn.size(dim)) << "gelu_tanh_cat: leading dimensions must match";
    }
  }
  const int64_t a = attn.size(-1);
  const int64_t m = mlp.size(-1);
  CHECK_HOST(output.size(-1) == a + m) << "gelu_tanh_cat: incorrect output width";
  constexpr int kVecN = 16 / sizeof(T);
  CHECK_HOST(a > 0 && m > 0 && a % kVecN == 0 && m % kVecN == 0)
      << "gelu_tanh_cat: both widths must be positive multiples of " << kVecN;
  const int64_t rows = attn.numel() / a;
  CHECK_HOST(rows > 0) << "gelu_tanh_cat: rows must be positive";
  CHECK_HOST(
      reinterpret_cast<uintptr_t>(attn.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(mlp.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0)
      << "gelu_tanh_cat: tensors must be 16-byte aligned";
  const int64_t n = rows * ((a + m) / kVecN);
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
