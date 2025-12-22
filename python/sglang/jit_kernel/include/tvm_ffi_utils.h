#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t float64_code = encode_dlpack_dtype(dl_float64);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t int32_code = encode_dlpack_dtype(dl_int32);

#define _DISPATCH_CASE_F16(c_type, ...) \
  case float16_code: {                  \
    using c_type = nv_half;             \
    __VA_ARGS__();                      \
    return true;                        \
  }
#define _DISPATCH_CASE_F32(c_type, ...) \
  case float32_code: {                  \
    using c_type = float;               \
    __VA_ARGS__();                      \
    return true;                        \
  }
#define _DISPATCH_CASE_F64(c_type, ...) \
  case float64_code: {                  \
    using c_type = double;              \
    __VA_ARGS__();                      \
    return true;                        \
  }
#define _DISPATCH_CASE_BF16(c_type, ...) \
  case bfloat16_code: {                  \
    using c_type = nv_bfloat16;          \
    __VA_ARGS__();                       \
    return true;                         \
  }

#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FLOAT(dlpack_dtype, c_type, ...)                                              \
  [&]() -> bool {                                                                                                    \
    switch (encode_dlpack_dtype(dlpack_dtype)) {                                                                     \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                                        \
      _DISPATCH_CASE_F32(c_type, __VA_ARGS__)                                                                        \
      _DISPATCH_CASE_F64(c_type, __VA_ARGS__)                                                                        \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                                       \
      default:                                                                                                       \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type " << (int)(dlpack_dtype).code \
                              << " " << (int)(dlpack_dtype).bits;                                                    \
        return false;                                                                                                \
    }                                                                                                                \
  }()
#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FLOAT16(dlpack_dtype, c_type, ...)                                            \
  [&]() -> bool {                                                                                                    \
    switch (encode_dlpack_dtype(dlpack_dtype)) {                                                                     \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                                        \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                                       \
      default:                                                                                                       \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type " << (int)(dlpack_dtype).code \
                              << " " << (int)(dlpack_dtype).bits;                                                    \
        return false;                                                                                                \
    }                                                                                                                \
  }()

#define TVM_FFI_GET_CUDA_STREAM(data) \
  static_cast<cudaStream_t>(TVMFFIEnvGetStream(data.device().device_type, data.device().device_id))

#define CHECK_CUDA_SUCCESS(err)                                                        \
  do {                                                                                 \
    TVM_FFI_ICHECK(err == cudaSuccess) << "CUDA Failure: " << cudaGetErrorString(err); \
  } while (0)

#define _TVM_FFI_UTILS_INLINE __attribute__((host)) __attribute__((device)) __inline__ __attribute__((always_inline))

// ATen implement math op with {float, double, int, int64_t}
// clang-format off
#define _TVM_FFI_UTILS_ARITHMETIC_WITH_(dtype, other) \
    _TVM_FFI_UTILS_INLINE dtype operator+(const dtype &lh, const other &rh) { return lh + dtype(rh); }  \
    _TVM_FFI_UTILS_INLINE dtype operator-(const dtype &lh, const other &rh) { return lh - dtype(rh); }  \
    _TVM_FFI_UTILS_INLINE dtype operator*(const dtype &lh, const other &rh) { return lh * dtype(rh); }  \
    _TVM_FFI_UTILS_INLINE dtype operator/(const dtype &lh, const other &rh) { return lh / dtype(rh); }  \
    _TVM_FFI_UTILS_INLINE dtype operator+(const other &lh, const dtype &rh) { return dtype(lh) + rh; }  \
    _TVM_FFI_UTILS_INLINE dtype operator-(const other &lh, const dtype &rh) { return dtype(lh) - rh; }  \
    _TVM_FFI_UTILS_INLINE dtype operator*(const other &lh, const dtype &rh) { return dtype(lh) * rh; }  \
    _TVM_FFI_UTILS_INLINE dtype operator/(const other &lh, const dtype &rh) { return dtype(lh) / rh; }
// clang-format on

// clang-format off
#define _TVM_FFI_UTILS_COMPARE_WITH_(dtype, other) \
    _TVM_FFI_UTILS_INLINE bool operator<=(const dtype &lh, const other &rh) { return lh <= dtype(rh); } \
    _TVM_FFI_UTILS_INLINE bool operator<(const dtype &lh, const other &rh) { return lh < dtype(rh); }   \
    _TVM_FFI_UTILS_INLINE bool operator>=(const dtype &lh, const other &rh) { return lh >= dtype(rh); } \
    _TVM_FFI_UTILS_INLINE bool operator>(const dtype &lh, const other &rh) { return lh > dtype(rh); }
// clang-format on

#define _TVM_FFI_UTILS_HABS(dtype) \
  _TVM_FFI_UTILS_INLINE dtype abs(dtype v) { return __habs(v); }

// ATen
// 1. rely on implicitly conversion to float for comparisons
//   c10::{Float8_e4m3fn,Float8_e4m3fnuz,Float8_e5m2,Float8_e5m2fnuz,Float8_e8m0fnu,Half}
// 2. impl operator{<,>} for c10::BFloat16
// Ref: https://github.com/pytorch/pytorch/commit/5e2ef2a465f79957faf5c56fe2a66d7a9b18e1a2
// For nv types, we'll met conflict between `arithmetic >= arithmetic` and `nv_half >= nv_half`
// TODO: ATen implement conversions with {float, double, int, int64_t}

#define _TVM_FFI_UTILS_NV_HALF                    \
  _TVM_FFI_UTILS_ARITHMETIC_WITH_(nv_half, float) \
  _TVM_FFI_UTILS_COMPARE_WITH_(nv_half, float)    \
  _TVM_FFI_UTILS_HABS(nv_half)

#define _TVM_FFI_UTILS_NV_BFLOAT16                    \
  _TVM_FFI_UTILS_ARITHMETIC_WITH_(nv_bfloat16, float) \
  _TVM_FFI_UTILS_COMPARE_WITH_(nv_bfloat16, float)    \
  _TVM_FFI_UTILS_HABS(nv_bfloat16)
