#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "SYCLHelpers.h"

template <typename T_out, typename T_scale, int GroupK = 16, int GroupN = 16, int SgSize = 16>
struct AWQDequantizeKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  AWQDequantizeKernelFunctor(
      const int* qweight,     // Quantized weights (4-bit packed in uint8_t) [N, K/2]
      const T_scale* scales,  // Scaling factors [N, K / group_size]
      const int* qzeros,      // Packed and offsetted zero points (4-bit packed in uint8_t) [N, (K/group_size)/2]
      T_out* output,          // Output dequantized weights [N, K]
      const size_t N,         // Output dimension
      const size_t K,
      const size_t group_size)
      :  // Input dimension
        qweight(qweight),
        scales(scales),
        qzeros(qzeros),
        output(output),
        N(N),
        K(K),
        group_size(group_size) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(SgSize)]] void operator()(sycl::nd_item<2> item) const {
    int col = item.get_global_id(1);
    int row = item.get_global_id(0);

    int group_idx = row / group_size;
    int scale_offset = elements * col + group_idx * N * elements;
    const T_scale* loaded_scale = (scales + scale_offset);
    int loaded_zeros = *(qzeros + col + group_idx * N);
    int loaded_weights = *(qweight + col + row * N);
    T_out* optr = output + elements * col + elements * row * N;
    for (int iter = 0; iter < elements; ++iter) {
      auto tmp_w = static_cast<T_scale>((loaded_weights >> (4 * orders[iter])) & 0x0f);
      auto tmp_z = static_cast<T_scale>((loaded_zeros >> (4 * orders[iter])) & 0x0f);
      optr[iter] = static_cast<T_out>((tmp_w - tmp_z) * loaded_scale[iter]);
    }
    return;
  }

 private:
  const int* qweight;     // Quantized weights (4-bit packed in uint8_t) [N, K/2]
  const T_scale* scales;  // Scaling factors [N, K / group_size]
  const int* qzeros;      // Packed and offsetted zero points (4-bit packed in uint8_t) [N, (K/group_size)/2]
  T_out* output;          // Output dequantized weights [N, K]
  const size_t N;         // Output dimension
  const size_t K;         // Input dimension
  const size_t group_size;
  static constexpr int orders[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  static constexpr int bits = 4;
  static constexpr int elements = 8;
};

// Host-side wrapper function
template <typename T_out = float, typename T_scale = float>  // Default to float
void dequantize_awq_sycl(
    sycl::queue& q,
    const int* d_qweight,     // Device pointer to quantized weights
    const T_scale* d_scales,  // Device pointer to scales
    const int* d_qzeros,      // Device pointer to quantized zero points
    T_out* d_output,          // Device pointer to output buffer
    const size_t N,
    const size_t K,
    const size_t group_size) {
  const size_t total_elements = N * K;
  if (total_elements == 0) return;  // Nothing to do

  size_t constexpr SgSize = 16;
  size_t constexpr GroupK = 16;
  size_t constexpr GroupN = 16;
  assert(group_size % GroupK == 0);
  assert(K % GroupK == 0 && N % GroupN == 0);
  sycl::range<2> global_range{K, N};
  sycl::range<2> local_range{GroupN, GroupK};

  // Launch the kernel
  auto nd_range = sycl::nd_range<2>(global_range, local_range);
  auto kfn = AWQDequantizeKernelFunctor<T_out, T_scale, GroupK, GroupN, SgSize>(
      d_qweight, d_scales, d_qzeros, d_output, N, K, group_size);
  sycl_kernel_submit(global_range, local_range, q, kfn);
  return;
}

at::Tensor awq_dequantize(at::Tensor qweight, at::Tensor scales, at::Tensor qzeros) {
  size_t K = qweight.size(0);
  size_t N = qweight.size(1);
  size_t group_size = K / scales.size(0);

  auto output_tensor_options = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
  at::Tensor output = torch::empty({static_cast<long>(K), static_cast<long>(N) * 8}, output_tensor_options);

  auto _qweight = reinterpret_cast<int*>(qweight.data_ptr<int>());
  auto _zeros = reinterpret_cast<int*>(qzeros.data_ptr<int>());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (scales.scalar_type() == at::ScalarType::Half) {
    auto _scales = reinterpret_cast<sycl::half*>(scales.data_ptr<at::Half>());
    auto _output = reinterpret_cast<sycl::half*>(output.data_ptr<at::Half>());
    dequantize_awq_sycl<sycl::half, sycl::half>(queue, _qweight, _scales, _zeros, _output, N, K, group_size);
  } else {
    auto _scales = reinterpret_cast<sycl::ext::oneapi::bfloat16*>(scales.data_ptr<at::BFloat16>());
    auto _output = reinterpret_cast<sycl::ext::oneapi::bfloat16*>(output.data_ptr<at::BFloat16>());
    dequantize_awq_sycl<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(
        queue, _qweight, _scales, _zeros, _output, N, K, group_size);
  }

  return output;
}
