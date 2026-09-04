#include "common.h"
#include "vec.h"

namespace {

template <typename index_t>
void validate_indices_in_range(
    const index_t* __restrict__ indices, int64_t rows, int64_t table_rows) {
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t idx = static_cast<int64_t>(indices[row]);
    TORCH_CHECK(
        idx >= 0 && idx < table_rows,
        "indexed_scale_shift_bf16_: indices[",
        row,
        "] = ",
        idx,
        " out of range for table rows ",
        table_rows);
  }
}

template <typename index_t>
void indexed_scale_shift_bf16_kernel_impl(
    at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ shift,
    const at::BFloat16* __restrict__ scale,
    const index_t* __restrict__ indices,
    int64_t rows,
    int64_t hidden_size,
    int64_t x_stride_row,
    int64_t shift_stride_row,
    int64_t scale_stride_row) {
  using bVec = at::vec::Vectorized<at::BFloat16>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t kVecSize = bVec::size();

  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
    const fVec one_vec(1.0f);
    for (int64_t row = begin; row < end; ++row) {
      const int64_t idx = static_cast<int64_t>(indices[row]);
      at::BFloat16* x_ptr = x + row * x_stride_row;
      const at::BFloat16* shift_ptr = shift + idx * shift_stride_row;
      const at::BFloat16* scale_ptr = scale + idx * scale_stride_row;

      int64_t d = 0;
#pragma GCC unroll 4
      for (; d <= hidden_size - kVecSize; d += kVecSize) {
        auto [x_fvec0, x_fvec1] = load_float_vec2(x_ptr + d);
        auto [shift_fvec0, shift_fvec1] = load_float_vec2(shift_ptr + d);
        auto [scale_fvec0, scale_fvec1] = load_float_vec2(scale_ptr + d);
        auto one_plus_scale_bf16 = convert_from_float_ext<at::BFloat16>(
            scale_fvec0 + one_vec, scale_fvec1 + one_vec);
        fVec one_plus_scale_fvec0, one_plus_scale_fvec1;
        std::tie(one_plus_scale_fvec0, one_plus_scale_fvec1) =
            at::vec::convert_to_float(one_plus_scale_bf16);
        auto scaled_bf16 = convert_from_float_ext<at::BFloat16>(
            x_fvec0 * one_plus_scale_fvec0, x_fvec1 * one_plus_scale_fvec1);
        fVec scaled_fvec0, scaled_fvec1;
        std::tie(scaled_fvec0, scaled_fvec1) = at::vec::convert_to_float(scaled_bf16);
        auto output_bf16 = convert_from_float_ext<at::BFloat16>(
            scaled_fvec0 + shift_fvec0, scaled_fvec1 + shift_fvec1);
        output_bf16.store(x_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden_size; ++d) {
        const float scale_rounded = static_cast<float>(static_cast<at::BFloat16>(
            1.0f + static_cast<float>(scale_ptr[d])));
        const float scaled = static_cast<float>(static_cast<at::BFloat16>(
            static_cast<float>(x_ptr[d]) * scale_rounded));
        x_ptr[d] = static_cast<at::BFloat16>(scaled + static_cast<float>(shift_ptr[d]));
      }
    }
  });
}

}  // namespace

at::Tensor indexed_scale_shift_bf16_cpu_impl(
    at::Tensor& x,
    const at::Tensor& shift,
    const at::Tensor& scale,
    const at::Tensor& indices) {
  CHECK_INPUT(x);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(shift);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(scale);
  CHECK_INPUT(indices);
  CHECK_DIM(2, x);
  CHECK_DIM(2, shift);
  CHECK_DIM(2, scale);
  CHECK_DIM(1, indices);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(shift.scalar_type() == at::kBFloat16, "shift must be bfloat16");
  TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be bfloat16");
  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "indices must be int32 or int64");
  CHECK_EQ(x.size(0), indices.size(0));
  CHECK_EQ(shift.size(0), scale.size(0));
  CHECK_EQ(shift.size(1), scale.size(1));
  CHECK_EQ(x.size(1), shift.size(1));

  const int64_t rows = x.size(0);
  if (rows == 0) {
    return x;
  }
  const int64_t table_rows = shift.size(0);
  const int64_t hidden_size = x.size(1);
  if (indices.scalar_type() == at::kInt) {
    const auto* indices_ptr = indices.data_ptr<int32_t>();
    validate_indices_in_range(indices_ptr, rows, table_rows);
    indexed_scale_shift_bf16_kernel_impl(
        x.data_ptr<at::BFloat16>(), shift.data_ptr<at::BFloat16>(),
        scale.data_ptr<at::BFloat16>(), indices_ptr, rows, hidden_size,
        x.stride(0), shift.stride(0), scale.stride(0));
  } else {
    const auto* indices_ptr = indices.data_ptr<int64_t>();
    validate_indices_in_range(indices_ptr, rows, table_rows);
    indexed_scale_shift_bf16_kernel_impl(
        x.data_ptr<at::BFloat16>(), shift.data_ptr<at::BFloat16>(),
        scale.data_ptr<at::BFloat16>(), indices_ptr, rows, hidden_size,
        x.stride(0), shift.stride(0), scale.stride(0));
  }
  return x;
}