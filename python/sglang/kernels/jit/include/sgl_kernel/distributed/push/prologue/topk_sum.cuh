#pragma once

#include <sgl_kernel/math.cuh>

#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/push/prologue/common.cuh>

namespace sglang::device::distributed::push {

using tvm::ffi::TensorView;

/// The beam prologue: `routed_scale * sum_k partial[row, k]` in fp32, rounded to bf16.
template <uint32_t kHiddenSize, uint32_t kTopK>
struct TopKSum {
 public:
  using Vec = VecStorage<bf16_t>;

  struct Params {
    const bf16_t* partial;  // [num_tokens, kTopK, kHiddenSize]
    float routed_scale;
  };

  SGL_DEVICE TopKSum(const Params& p, uint32_t row, uint32_t col, uint32_t /*vid*/)
      : m_params(p), m_row(row), m_col(col) {}

  SGL_DEVICE Vec load() const {
    Vec in[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      in[k].load(assume_global(m_params.partial) + (m_row * kTopK + k) * kHiddenSize, m_col);
    }
    fp32x2_t acc[kVecPairs<bf16_t>]{};
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
#pragma unroll
      for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
        const auto [x, y] = cast<fp32x2_t>(in[k][j]);
        acc[j].x += x;
        acc[j].y += y;
      }
    }
    Vec out;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto x = acc[j].x * m_params.routed_scale;
      const auto y = acc[j].y * m_params.routed_scale;
      out[j] = cast<bf16x2_t>(fp32x2_t{x, y});
    }
    return out;
  }

  static Params make_params(CallShape& shape, TensorView partial, float routed_scale) {
    host::TensorMatcher({shape.m_tokens, kTopK, kHiddenSize})  //
        .with_dtype<bf16_t>()
        .with_device(shape.m_device)
        .ensure_alignment(16)
        .verify(partial);
    return {.partial = get_pointer<const bf16_t>(partial), .routed_scale = routed_scale};
  }

 private:
  const Params& m_params;
  uint32_t m_row;
  uint32_t m_col;
};

// Beam 1.0
static_assert(Prologue<TopKSum<5120, 8>>);

}  // namespace sglang::device::distributed::push
