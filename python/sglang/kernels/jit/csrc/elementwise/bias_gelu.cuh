#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sglang {

SGL_DEVICE float gelu_tanh(float x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  const float cdf = 0.5f * (1.0f + tanhf(kBeta * (x + kAlpha * x * x * x)));
  return x * cdf;
}

/**
 * \brief Add a row-wise bias and apply approximate GELU.
 *
 * The intermediate bias result is rounded to the input dtype before GELU to
 * preserve the eager add-then-GELU numerical boundary.
 */
template <typename T, int kVecN, bool kUsePDL>
__global__ void bias_gelu_tanh_kernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t num_vecs,
    int64_t row_vecs) {
  using vec_t = device::AlignedVector<T, kVecN>;

  device::PDLWaitPrimary<kUsePDL>();
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vec_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; vec_id < num_vecs;
       vec_id += stride) {
    vec_t x;
    vec_t b;
    x.load(input, vec_id);
    b.load(bias, vec_id % row_vecs);

    vec_t result;
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      const float x_f32 = device::cast<fp32_t>(x[i]);
      const float bias_f32 = device::cast<fp32_t>(b[i]);
      const T biased = device::cast<T>(x_f32 + bias_f32);
      result[i] = device::cast<T>(gelu_tanh(device::cast<fp32_t>(biased)));
    }
    result.store(output, vec_id);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Validate and launch row-wise bias plus approximate GELU.
 *
 * \param input Contiguous two-dimensional input tensor.
 * \param bias Contiguous bias matching the final input dimension.
 * \param output Contiguous two-dimensional output tensor.
 */
template <typename T, bool kUsePDL>
void bias_gelu_tanh(tvm::ffi::TensorView input, tvm::ffi::TensorView bias, tvm::ffi::TensorView output) {
  using namespace host;

  auto num_rows = SymbolicSize{"num_rows"};
  auto hidden_dim = SymbolicSize{"hidden_dim"};
  auto device_ = SymbolicDevice{};
  device_.set_options<kDLCUDA>();

  TensorMatcher({num_rows, hidden_dim}).with_dtype<T>().with_device(device_).verify(input);
  TensorMatcher({hidden_dim}).with_dtype<T>().with_device(device_).verify(bias);
  TensorMatcher({num_rows, hidden_dim}).with_dtype<T>().with_device(device_).verify(output);

  constexpr int kVecN = device::kMaxVecBytes / sizeof(T);
  const int64_t rows = num_rows.unwrap();
  const int64_t width = hidden_dim.unwrap();
  CHECK_HOST(rows > 0) << "bias_gelu_tanh: num_rows must be positive";
  CHECK_HOST(width > 0 && width % kVecN == 0)
      << "bias_gelu_tanh: hidden_dim must be positive and divisible by " << kVecN;

  const int64_t row_vecs = width / kVecN;
  const int64_t num_vecs = rows * row_vecs;
  constexpr int64_t kBlockSize = 256;
  const auto kernel = bias_gelu_tanh_kernel<T, kVecN, kUsePDL>;
  const int64_t occupancy = runtime::get_blocks_per_sm(kernel, kBlockSize);
  const int64_t num_sms = runtime::get_sm_count(device_.unwrap().device_id);
  const int64_t grid = std::min(num_sms * occupancy, div_ceil(num_vecs, kBlockSize));
  LaunchKernel(grid, kBlockSize, device_.unwrap())
      .enable_pdl(kUsePDL)(
          kernel,
          static_cast<const T*>(input.data_ptr()),
          static_cast<const T*>(bias.data_ptr()),
          static_cast<T*>(output.data_ptr()),
          num_vecs,
          row_vecs);
}

}  // namespace sglang
