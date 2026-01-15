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

SGL_REGISTER_DTYPE_TRAIT(fp32_t, fp32x2_t, SGL_REGISTER_TYPE_END;  //
                         SGL_REGISTER_UNARY_FUNCTION(abs, fabsf);
                         SGL_REGISTER_UNARY_FUNCTION(sqrt, sqrtf);
                         SGL_REGISTER_UNARY_FUNCTION(rsqrt, rsqrtf);
                         SGL_REGISTER_BINARY_FUNCTION(max, fmaxf);
                         SGL_REGISTER_BINARY_FUNCTION(min, fminf););
SGL_REGISTER_DTYPE_TRAIT(fp16_t, fp16x2_t);
SGL_REGISTER_DTYPE_TRAIT(bf16_t, bf16x2_t);

/// TODO: Add ROCM implementation
SGL_REGISTER_DTYPE_TRAIT(fp32x2_t, fp32x4_t, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp16x2_t, __half22float2);
                         SGL_REGISTER_FROM_FUNCTION(bf16x2_t, __bfloat1622float2););

SGL_REGISTER_DTYPE_TRAIT(fp16x2_t, void, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22half2_rn););

SGL_REGISTER_DTYPE_TRAIT(bf16x2_t, void, SGL_REGISTER_TYPE_END;
                         SGL_REGISTER_FROM_FUNCTION(fp32x2_t, __float22bfloat162_rn););

#undef SGL_REGISTER_DTYPE_TRAIT
#undef SGL_REGISTER_FROM_FUNCTION

template <typename T>
using packed_t = typename dtype_trait<T>::packed_t;

namespace device {

template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
  return dtype_trait<To>::from(value);
}

}  // namespace device
