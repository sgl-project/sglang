#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/communicator.cuh>

#include <cstdint>

namespace sglang::host::distributed {

struct CallShape {
 public:
  explicit CallShape() : m_tokens("num_tokens") {
    m_device.set_options<kDLCUDA>();
  }

  void verify_tensor(TensorView tensor, int64_t hidden_size, DebugInfo info = {}) {
    TensorMatcher({m_tokens, hidden_size})  //
        .with_dtype<bf16_t>()
        .with_device(m_device)
        .ensure_alignment(16)
        .verify(tensor, info);
  }

  void verify_weight(TensorView weight, int64_t hidden_size, DebugInfo info = {}) {
    TensorMatcher({hidden_size})  //
        .with_dtype<bf16_t>()
        .with_device(m_device)
        .ensure_alignment(16)
        .verify(weight, info);
  }

  uint32_t get_num_tokens(DebugInfo info = {}) const {
    return m_tokens.unwrap(info);
  }

  DLDevice get_device(DebugInfo info = {}) const {
    return m_device.unwrap(info);
  }

  SymbolicSize m_tokens;
  SymbolicDevice m_device;
};

template <typename T>
inline T* get_pointer(TensorView view) {
  return static_cast<T*>(view.data_ptr());
}

template <typename T>
inline T* get_pointer(Optional<TensorView> view) {
  return view.has_value() ? static_cast<T*>((*view).data_ptr()) : nullptr;
}

}  // namespace sglang::host::distributed

namespace sglang::device::distributed {

/// Introduce the host helpers
using host::distributed::CallShape;
using host::distributed::get_pointer;

/// The packed pairs (`packed_t<T>`) in one thread's 16-byte vector.
template <typename T>
inline constexpr uint32_t kVecPairs = 16 / sizeof(T) / 2;

/// One thread's 16-byte vector of T, as packed pairs.
template <typename T>
using VecStorage = AlignedVector<packed_t<T>, kVecPairs<T>>;

/// The empty-slot marker of a 16-byte vector of T, checked per 4-byte word.
template <typename T>
using VecLamport = LamportTrait<T, 16 / sizeof(T), /*kAtom=*/4>;

/// fp8 e4m3 codes of a pair, saturating: only the lower bound is clamped, which maps NaN to -448 as a two-sided clamp
/// does (`fmaxf` returns the non-NaN operand); the conversion saturates the upper side (`.satfinite`).
SGL_DEVICE fp8x2_e4m3_t to_fp8_e4m3(fp32x2_t v) {
  return cast<fp8x2_e4m3_t>(fp32x2_t{fmaxf(v.x, -448.0f), fmaxf(v.y, -448.0f)});
}

/// Adds the squares of a bf16 vector into two fp32 accumulators (even / odd lanes of the pairs), each a bf16 x bf16
/// FMA rounded once (one `FHFMA.BF16` on SM100, no unpack).
SGL_DEVICE void bf16_square_sum(const VecStorage<bf16_t>& x, float& acc0, float& acc1) {
#pragma unroll
  for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
    acc0 = math::fma_f32_bf16(x[j].x, x[j].x, acc0);
    acc1 = math::fma_f32_bf16(x[j].y, x[j].y, acc1);
  }
}

/// The RMSNorm factor `rsqrt(sum / hidden + eps)` from a row's sum of squares, with `inv_hidden` = 1 / hidden.
SGL_DEVICE float rms_factor(float square_sum, float inv_hidden, float eps) {
  return math::rsqrt(square_sum * inv_hidden + eps);
}

/// `bf16(x * w * r)`: `x * w` in fp32 (one `FHFMA.BF16` with a -0 addend on SM100, which keeps the sign of zero),
/// times `r`, rounded once.
SGL_DEVICE VecStorage<bf16_t> bf16_norm_scale(const VecStorage<bf16_t>& x, const VecStorage<bf16_t>& w, float r) {
  VecStorage<bf16_t> out;
#pragma unroll
  for (uint32_t j = 0; j < kVecPairs<bf16_t>; ++j) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
    const auto a = math::fma_f32_bf16(x[j].x, w[j].x, -0.0f);
    const auto b = math::fma_f32_bf16(x[j].y, w[j].y, -0.0f);
    out[j] = cast<bf16x2_t>(math::mul_f32x2(fp32x2_t{a, b}, fp32x2_t{r, r}));
#else
    const auto xf = cast<fp32x2_t>(x[j]);
    const auto wf = cast<fp32x2_t>(w[j]);
    out[j] = cast<bf16x2_t>(math::mul_f32x2(math::mul_f32x2(xf, wf), fp32x2_t{r, r}));
#endif
  }
  return out;
}

}  // namespace sglang::device::distributed
