#include "../common.h"
#include "op.h"

namespace {

// out = mat1 @ mat2 + bias
template <typename scalar_t>
void int8_scaled_mm_impl(
    scalar_t* __restrict__ out,         // [M, N], row major
    const int8_t* __restrict__ mat1,    // [M, K], row major
    const int8_t* __restrict__ mat2,    // [K, N], column major
    const float* __restrict__ scales1,  // [M, 1], mat1 scales
    const float* __restrict__ scales2,  // [1, N], mat2 scales
    const float* __restrict__ bias,     // [1, N]
    int64_t M,
    int64_t N,
    int64_t K) {
  TORCH_CHECK(false, "not supported yet");
}

template <>
void int8_scaled_mm_impl<at::BFloat16>(
    at::BFloat16* __restrict__ out,
    const int8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  const int slice_size = (M * K * sizeof(int8_t)) > kL2Size ? 64 : 8;
  const int num_slices = (N + slice_size - 1) / slice_size;

  auto mm = [mat1, mat2, out, M, N, K, scales1, scales2, bias, slice_size](int64_t begin, int64_t end) {
    for (int64_t slice_idx = begin; slice_idx < end; ++slice_idx) {
      const int64_t n_start = slice_idx * slice_size;
      const int64_t n_end = std::min(n_start + slice_size, N);
      const int slice_width = static_cast<int>(n_end - n_start);

      const int8_t* a_ptr = mat1;
      const int8_t* b_ptr = mat2 + n_start * K;
      bfloat16_t* c_ptr = reinterpret_cast<bfloat16_t*>(out) + n_start;

      op::i8mm_matmul(a_ptr, b_ptr, c_ptr, M, K, N, slice_width, scales1, scales2 + n_start);

      // NOTE: matmul reduces matrix values to BF16, may influence precision
      if (bias) {
        op::add_bias(c_ptr, bias + n_start, M, N, slice_width);
      }
    }
  };

  at::parallel_for(0, num_slices, 0, mm);
}

}  // anonymous namespace

// weight     :  static, per-channel, symmetric
// activation : dynamic,   per-token, symmetric
//
// mat1    : [M, K]
// mat2    : [N, K]
// scales1 : [M]
// scales2 : [N]
// bias    : [N]
// out     : [M, N]
//
// fused activation quantization and matmul
at::Tensor int8_scaled_mm_with_quant(
    at::Tensor& mat1,
    at::Tensor& mat2,
    at::Tensor& scales2,
    const std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype,
    bool /*is_vnni*/) {
  RECORD_FUNCTION("sgl-kernel::int8_scaled_mm_with_quant", std::vector<c10::IValue>({mat1, mat2, scales2, bias}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat1.size(1);
  int64_t lda = mat1.stride(0);

  CHECK_EQ(mat2.size(1), K);
  CHECK_EQ(scales2.numel(), N);

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16, "int8_scaled_mm_with_quant: expect A to be bfloat16.");
  TORCH_CHECK(st == out_dtype, "int8_scaled_mm_with_quant: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kChar, "int8_scaled_mm_with_quant: expect mat2 to be int8.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat, "int8_scaled_mm_with_quant: expect scales to be float32.");

  const int64_t buffer_size = M * K + M * sizeof(float);
  auto buffer = at::empty({buffer_size}, mat1.options().dtype(at::kChar));
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "int8_scaled_mm_with_quant_kernel_impl", [&] {
    int8_t* __restrict__ Aq_data = buffer.data_ptr<int8_t>();
    float* __restrict__ As_data = (float*)((void*)(Aq_data + M * K));
    const scalar_t* __restrict__ A_data = mat1.data_ptr<scalar_t>();

    const int64_t grain = kL1Size / (K * sizeof(scalar_t));
    at::parallel_for(0, M, grain, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        op::quantize_row_int8(Aq_data + m * K, As_data + m, A_data + m * lda, K);
      }
    });

    int8_scaled_mm_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        Aq_data,
        mat2.data_ptr<int8_t>(),
        As_data,
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K);
  });
  return out;
}
