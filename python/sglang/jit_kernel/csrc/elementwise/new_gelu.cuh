#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, fp16_t, bf16_t, fp32_t
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template <typename T, int kVecN>
__global__ void new_gelu_kernel(T* __restrict__ dst,
                                const T* __restrict__ src,
                                uint32_t n_total) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;  // sqrt(2/pi)

  using vec_t = device::AlignedVector<T, kVecN>;
  const uint32_t n_vecs = n_total / kVecN;

  // vectorized body
  const uint32_t vec_stride = blockDim.x * gridDim.x;
  for (uint32_t vi = blockIdx.x * blockDim.x + threadIdx.x;
       vi < n_vecs;
       vi += vec_stride) {
    vec_t v;
    v.load(src, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      float val = static_cast<float>(v[i]);
      float cdf = 0.5f * (1.0f + tanhf(kBeta * (val + kAlpha * val * val * val)));
      v[i] = static_cast<T>(val * cdf);
    }
    v.store(dst, vi);
  }

  // scalar tail
  const uint32_t base = n_vecs * kVecN;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       base + i < n_total;
       i += vec_stride) {
    float val = static_cast<float>(src[base + i]);
    float cdf = 0.5f * (1.0f + tanhf(kBeta * (val + kAlpha * val * val * val)));
    dst[base + i] = static_cast<T>(val * cdf);
  }
}

template <typename T>
void new_gelu(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace host;

  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA, kDLROCM>();

  TensorMatcher({N})  //
      .with_dtype<T>()
      .with_device(device_)
      .verify(dst)
      .verify(src);

  const uint32_t n = static_cast<uint32_t>(N.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(n > 0, "new_gelu: num_elements must be > 0, got ", n);

  constexpr int kVecN = 16 / sizeof(T);
  const uint32_t n_work_items = div_ceil(n, static_cast<uint32_t>(kVecN));

  constexpr uint32_t kBlockSize = 256;
  const uint32_t grid = div_ceil(n_work_items, kBlockSize);

  LaunchKernel(grid, kBlockSize, device)(
      new_gelu_kernel<T, kVecN>,
      static_cast<T*>(dst.data_ptr()),
      static_cast<const T*>(src.data_ptr()),
      n);
}

}  // namespace
