/// \file type.cuh
/// \brief Dtype trait system for CUDA scalar/packed types.
///
/// `dtype_trait<T>` provides per-type metadata: packed type alias,
/// conversion functions (`from`), and unary/binary math operations.
/// Use `device::cast<To>(from_value)` for type conversion on device.
///
/// Registered types:
/// | Scalar    | Packed (x2)  | Notes                         |
/// |-----------|-------------|-------------------------------|
/// | `fp32_t`  | `fp32x2_t`  | Full math ops (abs,sqrt,...) |
/// | `fp16_t`  | `fp16x2_t`  | Conversion only             |
/// | `bf16_t`  | `bf16x2_t`  | Conversion only             |
/// | `fp32x2_t`| `fp32x4_t`  | Packed float2 <-> half2/bf162 |

#pragma once
#include <sgl_kernel/utils.cuh>

template <typename T>
struct dtype_trait {};

#define SGL_REGISTER_DTYPE_TRAIT(TYPE, PACK2, ...)  \
  template <>                                       \
  struct dtype_trait<TYPE> {                        \
    using self_t = TYPE;                            \
    using packed_t = PACK2;                         \
    template <typename S>                           \
    SGL_DEVICE static self_t from(const S& value) { \
      return static_cast<TYPE>(value);              \
    }                                               \
    __VA_ARGS__                                     \
  }

#define SGL_REGISTER_TYPE_END static_assert(true)

#define SGL_REGISTER_FROM_FUNCTION(FROM, FN)     \
  SGL_DEVICE static self_t from(const FROM& x) { \
    return FN(x);                                \
  }                                              \
  static_assert(true)

#define SGL_REGISTER_UNARY_FUNCTION(NAME, FN)      \
  SGL_DEVICE static self_t NAME(const self_t& x) { \
    return FN(x);                                  \
  }                                                \
  static_assert(true)

#define SGL_REGISTER_BINARY_FUNCTION(NAME, FN)                      \
  SGL_DEVICE static self_t NAME(const self_t& x, const self_t& y) { \
    return FN(x, y);                                                \
  }                                                                 \
  static_assert(true)

SGL_REGISTER_DTYPE_TRAIT(
    fp32_t, fp32x2_t, SGL_REGISTER_TYPE_END;  //
    SGL_REGISTER_FROM_FUNCTION(fp16_t, __half2float);
    SGL_REGISTER_FROM_FUNCTION(bf16_t, __bfloat162float);
    SGL_REGISTER_UNARY_FUNCTION(abs, fabsf);
    SGL_REGISTER_UNARY_FUNCTION(sqrt, sqrtf);
    SGL_REGISTER_UNARY_FUNCTION(rsqrt, rsqrtf);
    SGL_REGISTER_UNARY_FUNCTION(exp, expf);
    SGL_REGISTER_UNARY_FUNCTION(sin, sinf);
    SGL_REGISTER_UNARY_FUNCTION(cos, cosf);
    SGL_REGISTER_BINARY_FUNCTION(max, fmaxf);
    SGL_REGISTER_BINARY_FUNCTION(min, fminf););
SGL_REGISTER_DTYPE_TRAIT(fp16_t, fp16x2_t);
SGL_REGISTER_DTYPE_TRAIT(bf16_t, bf16x2_t);

/// TODO: Add ROCM implementation
SGL_REGISTER_DTYPE_TRAIT(
    fp32x2_t, fp32x4_t, SGL_REGISTER_TYPE_END; SGL_REGISTER_FROM_FUNCTION(fp16x2_t, __half22float2);
    SGL_REGISTER_FROM_FUNCTION(bf16x2_t, __bfloat1622float2););

SGL_REGISTER_DTYPE_TRAIT(
    fp16x2_t, void, SGL_REGISTER_TYPE_END; SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22half2_rn););

SGL_REGISTER_DTYPE_TRAIT(
    bf16x2_t, void, SGL_REGISTER_TYPE_END; SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22bfloat162_rn););

#undef SGL_REGISTER_DTYPE_TRAIT
#undef SGL_REGISTER_FROM_FUNCTION

/// \brief Alias: the packed (x2) type for `T`.
template <typename T>
using packed_t = typename dtype_trait<T>::packed_t;

namespace device {

/**
 * \brief Cast a value from type `From` to type `To` on device.
 *
 * Dispatches through `dtype_trait<To>::from()`, which uses the appropriate
 * CUDA intrinsic (e.g. `__half2float`, `__float22half2_rn`).
 */
template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return dtype_trait<To>::from(value);
}

}  // namespace device
