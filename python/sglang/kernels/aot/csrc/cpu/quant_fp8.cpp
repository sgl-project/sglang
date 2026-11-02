#include "common.h"
#include "vec.h"

// activation : dynamic, per-token, symmetric
//
// A  : [M, K] bfloat16, K a multiple of 32
// Aq : [M, K] float8_e4m3fn
// As : [M] float32
//
std::tuple<at::Tensor, at::Tensor> per_token_quant_fp8_cpu(at::Tensor& A) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(A);
  CHECK_DIM(2, A);
  TORCH_CHECK(A.scalar_type() == at::kBFloat16, "per_token_quant_fp8: expect A to be bfloat16.");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t lda = A.stride(0);
  TORCH_CHECK(K % 32 == 0, "per_token_quant_fp8: expect K to be a multiple of 32, got ", K);

  auto Aq = at::empty({M, K}, A.options().dtype(at::kFloat8_e4m3fn));
  auto As = at::empty({M}, A.options().dtype(at::kFloat));

  at::Float8_e4m3fn* __restrict__ Aq_data = Aq.data_ptr<at::Float8_e4m3fn>();
  float* __restrict__ As_data = As.data_ptr<float>();
  const at::BFloat16* __restrict__ A_data = A.data_ptr<at::BFloat16>();
  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      quantize_row_fp8e4m3(Aq_data + m * K, As_data[m], A_data + m * lda, K);
    }
  });
  return std::make_tuple(Aq, As);
}

bool cpu_has_avx10_2() {
  return avx10_2_available();
}
