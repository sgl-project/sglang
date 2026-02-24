#pragma once
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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

#ifndef USE_ROCM
SGL_REGISTER_DTYPE_TRAIT(
    fp8_e4m3_t, fp8x2_e4m3_t, SGL_REGISTER_TYPE_END; SGL_DEVICE static self_t from(const bf16_t& x) {
      self_t r;
      r.__x = __nv_cvt_bfloat16raw_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
      return r;
    } SGL_DEVICE static self_t from(const fp16_t& x) {
      self_t r;
      r.__x = __nv_cvt_halfraw_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
      return r;
    });
#endif

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

// ---------------------------------------------------------------------------
// FP8 max clamp value — platform-dependent
//   CUDA (e4m3fn):      448.0f
//   AMD FNUZ (e4m3fnuz): 224.0f
//   AMD E4M3 (e4m3fn):  448.0f
// ---------------------------------------------------------------------------
#ifndef USE_ROCM
constexpr float kFP8E4M3Max = 448.0f;
#else  // USE_ROCM
#if HIP_FP8_TYPE_FNUZ
constexpr float kFP8E4M3Max = 224.0f;
#else   // HIP_FP8_TYPE_E4M3
constexpr float kFP8E4M3Max = 448.0f;
#endif  // HIP_FP8_TYPE_FNUZ
#endif  // USE_ROCM
