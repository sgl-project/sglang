#include "common.h"
#include "vec.h"

namespace {

template <typename index_t>
void indexed_gate_bf16_kernel_impl(
    at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ gate,
    const at::BFloat16* __restrict__ other,
    const index_t* __restrict__ indices,
    int64_t rows,
    int64_t hidden,
    int64_t gate_rows) {
  using bVec = at::vec::Vectorized<at::BFloat16>;
  constexpr int64_t kVecSize = bVec::size();

  const auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t index = static_cast<int64_t>(indices[row]);
      TORCH_CHECK(
          index >= 0 && index < gate_rows,
          "indices[",
          row,
          "] out of range: ",
          index,
          " vs rows=",
          gate_rows);

      at::BFloat16* __restrict__ x_ptr = x + row * hidden;
      const at::BFloat16* __restrict__ gate_ptr = gate + index * hidden;
      const at::BFloat16* __restrict__ other_ptr = other + row * hidden;

      int64_t d = 0;
#pragma GCC unroll 4
      for (; d <= hidden - kVecSize; d += kVecSize) {
        auto [x0, x1] = load_float_vec2(x_ptr + d);
        auto [gate0, gate1] = load_float_vec2(gate_ptr + d);
        auto [other0, other1] = load_float_vec2(other_ptr + d);
        x0 = x0 + gate0 * other0;
        x1 = x1 + gate1 * other1;
        convert_from_float_ext<at::BFloat16>(x0, x1).store(x_ptr + d);
      }
#pragma GCC unroll 4
      for (; d < hidden; ++d) {
        const float x_val = static_cast<float>(x_ptr[d]);
        const float gate_val = static_cast<float>(gate_ptr[d]);
        const float other_val = static_cast<float>(other_ptr[d]);
        x_ptr[d] = static_cast<at::BFloat16>(x_val + gate_val * other_val);
      }
    }
  };

  if (at::get_num_threads() == 1 || rows < GRAIN_SIZE) {
    loop(0, rows);
    return;
  }

  at::parallel_for(0, rows, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    loop(begin, end);
  });
}

}  // namespace

at::Tensor indexed_gate_bf16_(
    at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& other,
    const at::Tensor& indices) {
  CHECK_INPUT(x);
  CHECK_INPUT(gate);
  CHECK_INPUT(other);
  CHECK_INPUT(indices);
  CHECK_DIM(2, x);
  CHECK_DIM(2, gate);
  CHECK_DIM(2, other);
  CHECK_DIM(1, indices);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(gate.scalar_type() == at::kBFloat16, "gate must be bfloat16");
  TORCH_CHECK(other.scalar_type() == at::kBFloat16, "other must be bfloat16");
  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "indices must be int32 or int64");
  CHECK_EQ(x.sizes(), other.sizes());
  CHECK_EQ(indices.size(0), x.size(0));
  CHECK_EQ(x.size(1), gate.size(1));

  const int64_t rows = x.size(0);
  if (rows == 0) {
    return x;
  }
  const int64_t hidden = x.size(1);
  const int64_t gate_rows = gate.size(0);

  AT_DISPATCH_INTEGRAL_TYPES(indices.scalar_type(), "indexed_gate_bf16_", [&] {
    indexed_gate_bf16_kernel_impl(
        x.data_ptr<at::BFloat16>(),
        gate.data_ptr<at::BFloat16>(),
        other.data_ptr<at::BFloat16>(),
        indices.data_ptr<scalar_t>(),
        rows,
        hidden,
        gate_rows);
  });
  return x;
}