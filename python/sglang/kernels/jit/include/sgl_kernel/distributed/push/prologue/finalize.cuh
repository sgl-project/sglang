#pragma once

#include <sgl_kernel/math.cuh>

#include <sgl_kernel/distributed/fusion.cuh>
#include <sgl_kernel/distributed/push/prologue/common.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang::device::distributed::push {

namespace ptx {

using namespace sglang::device::ptx;

/// Loads 16 bytes from `ptr` if `valid`, else zeroes `vec`.
template <typename V>
SGL_DEVICE void ld_global_16B_or_zero(V& vec, const void* ptr, bool valid) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto reg = reinterpret_cast<uint32_t*>(&vec);
  const auto pred = static_cast<uint32_t>(valid);
  // no memory clobber, so that the compiler may reorder it with other memory accesses
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %5, 0;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\t"
      "mov.b32 %2, 0;\n\tmov.b32 %3, 0;\n\t@p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n\t}"
      : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
      : "l"(ptr), "r"(pred));
}

}  // namespace ptx

using tvm::ffi::TensorView, tvm::ffi::Optional;

/// GATED rounds the gated shared branch and the routed sum each to bf16, then adds them in bf16.
enum class SharedExpertMode : uint32_t {
  NONE = 0,
  ADD = 1,    // bf16(bf16(routed) + shared)
  GATED = 2,  // bf16(gate * shared) + bf16(routed)
};

/// The MoE finalize prologue: `sum_k weights[row, k] * gemm2_out[idx[row * kTopK + k]]` (idx -1 skipped) in fp32,
/// rounded to bf16, then the shared expert as `kMode` says.
template <uint32_t kHiddenSize, uint32_t kTopK, typename WeightT, SharedExpertMode kMode>
struct FinalizeSum {
 public:
  using Vec = VecStorage<bf16_t>;
  static constexpr uint32_t kVecSize = sizeof(Vec) / sizeof(bf16_t);

  struct Params {
    const bf16_t* gemm2_out;      // [num_permuted_rows, kHiddenSize]
    const int32_t* idx;           // [num_tokens * kTopK], -1 = dropped slot
    const WeightT* weights;       // [num_tokens, kTopK]
    const bf16_t* shared_output;  // [num_tokens, kHiddenSize] (kMode != NONE)
    const float* shared_gate;     // [num_tokens] (kMode == GATED)
  };

  SGL_DEVICE FinalizeSum(const Params& p, uint32_t row, uint32_t col, uint32_t vid)
      : m_params(p), m_row(row), m_col(col), m_vid(vid) {}

  SGL_DEVICE Vec load() const {
    using enum SharedExpertMode;
    const auto idx_row = assume_global(m_params.idx) + m_row * static_cast<int64_t>(kTopK);
    const auto weight_row = assume_global(m_params.weights) + m_row * static_cast<int64_t>(kTopK);
    int32_t idx[kTopK];
    WeightT weight[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      idx[k] = idx_row[k];
      weight[k] = weight_row[k];
    }
    [[maybe_unused]] Vec shared;
    [[maybe_unused]] float gate;
    if constexpr (kMode != NONE) {
      shared.load(assume_global(m_params.shared_output), m_vid);
    }
    if constexpr (kMode == GATED) {
      gate = assume_global(m_params.shared_gate)[m_row];
    }

    Vec in[kTopK];
#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const auto src =
          assume_global(m_params.gemm2_out) + idx[k] * static_cast<int64_t>(kHiddenSize) + m_col * kVecSize;
      ptx::ld_global_16B_or_zero(in[k], src, idx[k] >= 0);
      // a dropped slot adds 0 * -0, which leaves every acc unchanged (-0 included), whatever its weight holds
      weight[k] = idx[k] >= 0 ? weight[k] : static_cast<WeightT>(-0.0f);
    }

    fp32x2_t acc[kVecPairs<bf16_t>];
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      acc[j] = fp32x2_t{0.0f, 0.0f};
    }

#pragma unroll
    for (uint32_t k = 0; k < kTopK; ++k) {
      const auto w = weight[k];
#pragma unroll
      for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
        if constexpr (std::is_same_v<WeightT, bf16_t>) {
          acc[j].x = math::fma_f32_bf16(in[k][j].x, w, acc[j].x);
          acc[j].y = math::fma_f32_bf16(in[k][j].y, w, acc[j].y);
        } else {
          acc[j] = math::fma_f32x2(cast<fp32x2_t>(in[k][j]), {w, w}, acc[j]);
        }
      }
    }

    Vec out;
#pragma unroll
    for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
      const auto routed = cast<bf16x2_t>(acc[j]);
      if constexpr (kMode == NONE) {
        out[j] = routed;
      } else if constexpr (kMode == GATED) {
        const auto s = cast<fp32x2_t>(shared[j]);
        const auto gated = cast<bf16x2_t>(fp32x2_t{gate * s.x, gate * s.y});
        out[j] = __hadd2(gated, routed);
      } else {
        out[j] = __hadd2(routed, shared[j]);
      }
    }

    return out;
  }

  static Params make_params(
      CallShape& shape,  //
      TensorView gemm2_out,
      TensorView expanded_idx_to_permuted_idx,
      TensorView expert_weights,
      Optional<TensorView> shared_output,
      Optional<TensorView> shared_gate) {
    using enum SharedExpertMode;
    CHECK_HOST(shared_output.has_value() == (kMode != NONE) && shared_gate.has_value() == (kMode == GATED));
    using host::TensorMatcher;
    auto& T = shape.m_tokens;
    auto& device = shape.m_device;
    TensorMatcher({-1, kHiddenSize})  // gemm2_out
        .with_dtype<bf16_t>()
        .with_device(device)
        .ensure_alignment(16)
        .verify(gemm2_out);
    TensorMatcher({T, kTopK})  // expert_weights
        .with_dtype<WeightT>()
        .with_device(device)
        .verify(expert_weights);
    TensorMatcher({-1})  // expanded_idx_to_permuted_idx
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(expanded_idx_to_permuted_idx);
    const auto num_slots = int64_t{shape.get_num_tokens()} * kTopK;
    CHECK_HOST(expanded_idx_to_permuted_idx.size(0) == num_slots)
        << "expanded_idx_to_permuted_idx holds " << expanded_idx_to_permuted_idx.size(0)
        << " slots, not num_tokens * top_k";
    if constexpr (kMode != NONE) shape.verify_tensor(*shared_output, kHiddenSize);
    if constexpr (kMode == GATED) {
      TensorMatcher({T})  // shared_gate
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(*shared_gate);
    }
    return {
        .gemm2_out = get_pointer<const bf16_t>(gemm2_out),
        .idx = get_pointer<const int32_t>(expanded_idx_to_permuted_idx),
        .weights = get_pointer<const WeightT>(expert_weights),
        .shared_output = get_pointer<const bf16_t>(shared_output),
        .shared_gate = get_pointer<const float>(shared_gate),
    };
  }

 private:
  const Params& m_params;
  uint32_t m_row;
  uint32_t m_col;
  uint32_t m_vid;
};

static_assert(Prologue<FinalizeSum<5120, 6, bf16_t, SharedExpertMode::ADD>>);

}  // namespace sglang::device::distributed::push
