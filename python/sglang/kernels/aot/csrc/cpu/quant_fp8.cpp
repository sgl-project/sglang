#include "quant_fp8.h"

#include "common.h"
#include "vec.h"

namespace {

template <CpuIsa isa>
void per_token_quant_fp8_kernel_impl(
    at::Float8_e4m3fn* __restrict__ output,
    float* __restrict__ scales,
    const at::BFloat16* __restrict__ input,
    int64_t M,
    int64_t K,
    int64_t stride) {
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_fp8<isa>(output + m * K, scales[m], input + m * stride, K);
    }
  });
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> per_token_quant_fp8_cpu(at::Tensor& input) {
  CHECK_CPU(input);
  CHECK_DIM(2, input);
  CHECK_LAST_DIM_CONTIGUOUS(input);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "per_token_quant_fp8: expect bfloat16 input.");
  const int64_t M = input.size(0);
  const int64_t K = input.size(1);
  TORCH_CHECK(K > 0 && K % 32 == 0, "per_token_quant_fp8: K must be a positive multiple of 32.");

  auto output = at::empty({M, K}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({M}, input.options().dtype(at::kFloat));
  CPU_DISPATCH_ISA([&] {
    per_token_quant_fp8_kernel_impl<isa>(
        output.data_ptr<at::Float8_e4m3fn>(),
        scales.data_ptr<float>(),
        input.data_ptr<at::BFloat16>(),
        M,
        K,
        input.stride(0));
  });
  return std::make_tuple(output, scales);
}
