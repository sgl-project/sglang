#pragma once

#if defined(CPU_CAPABILITY_RVV)

#include <c10/util/Half.h>
#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "vector_math.h"

namespace rvv_constants {
// BLOCK_N = VLEN/4: weight packing tile size; each tile spans 2 vector iterations (m4).
inline constexpr int BLOCK_N = __riscv_v_fixed_vlen / 4;
// MAX_VL_ELEMENTS_M{4,8}: compile-time max VL for LMUL=4 and LMUL=8
inline constexpr size_t MAX_VL_ELEMENTS_M4 = __riscv_v_fixed_vlen / 8;
inline constexpr size_t MAX_VL_ELEMENTS_M8 = __riscv_v_fixed_vlen / 4;
// L1 cache size for K-tiling. Set by -DRVV_L1_CACHE_KB=N at cmake time (default 32KB).
// K-tile KB is derived as: L1_CACHE_BYTES * multiplier / VLEN
#ifndef RVV_L1_CACHE_KB
#define RVV_L1_CACHE_KB 32
#endif
inline constexpr int64_t L1_CACHE_BYTES = static_cast<int64_t>(RVV_L1_CACHE_KB) * 1024;
}  // namespace rvv_constants

// Fixed-width vector type: width follows __riscv_v_fixed_vlen (set at compile time via -mrvv-vector-bits=N).
// Enables stack arrays of vector type (e.g., vf32m1_t arr[N]) in kernels that need them.
typedef vfloat32m1_t vf32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

#ifndef MAX_HEAD_SIZE
#define MAX_HEAD_SIZE 256
#endif

#define AT_DISPATCH_RVV_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                \
    at::ScalarType _st = TYPE;                                         \
    switch (_st) {                                                     \
      case at::ScalarType::Float: {                                    \
        using scalar_t = float;                                        \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::ScalarType::Half: {                                     \
        using scalar_t = at::Half;                                     \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::ScalarType::BFloat16: {                                 \
        using scalar_t = at::BFloat16;                                 \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'"); \
    }                                                                  \
  }()

// Dispatch macro for packed types
#define CPU_DISPATCH_PACKED_TYPES_RVV(TYPE, ...)                 \
  [&] {                                                          \
    switch (TYPE) {                                              \
      case at::ScalarType::BFloat16: {                           \
        using packed_t = at::BFloat16;                           \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Half: {                               \
        using packed_t = at::Half;                               \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Char: {                               \
        using packed_t = int8_t;                                 \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false, "Unsupported floating data type.\n"); \
    }                                                            \
  }()

// Type Conversion Helpers

// BF16 to FP32
// Shift-widening: BF16 and FP32 share exponent field, left-shift by 16 bits

inline vfloat32m1_t bf16_to_f32m1(const uint16_t* ptr, size_t vl) {
  vuint16mf2_t v_bf16 = __riscv_vle16_v_u16mf2(ptr, vl);
  vuint32m1_t v_u32 = __riscv_vzext_vf2_u32m1(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m1(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_u32);
}

inline vfloat32m4_t bf16_to_f32m4(const uint16_t* ptr, size_t vl) {
  vuint16m2_t v_bf16 = __riscv_vle16_v_u16m2(ptr, vl);
  vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m4(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m4_f32m4(v_u32);
}

inline vfloat32m8_t bf16_to_f32m8(const uint16_t* ptr, size_t vl) {
  vuint16m4_t v_bf16 = __riscv_vle16_v_u16m4(ptr, vl);
  vuint32m8_t v_u32 = __riscv_vzext_vf2_u32m8(v_bf16, vl);
  v_u32 = __riscv_vsll_vx_u32m8(v_u32, 16, vl);
  return __riscv_vreinterpret_v_u32m8_f32m8(v_u32);
}

// FP32 to BF16
// Extract upper 16 bits (sign + exp + upper 7 mantissa)

#if defined(__riscv_zvfh)
inline vfloat16m2_t f32m4_to_f16(vfloat32m4_t v, size_t vl) {
  return __riscv_vfncvt_f_f_w_f16m2(v, vl);
}
#endif

inline vuint16m2_t f32m4_to_bf16(vfloat32m4_t v_f32, size_t vl) {
  vuint32m4_t v_u32 = __riscv_vreinterpret_v_f32m4_u32m4(v_f32);
  return __riscv_vnsrl_wx_u16m2(v_u32, 16, vl);
}

inline vuint16m4_t f32m8_to_bf16(vfloat32m8_t v_f32, size_t vl) {
  vuint32m8_t v_u32 = __riscv_vreinterpret_v_f32m8_u32m8(v_f32);
  return __riscv_vnsrl_wx_u16m4(v_u32, 16, vl);
}

// Load / Store Helpers
// scratch: caller-owned buffer (MIN MAX_VL_ELEMENTS_M4 floats, 64-byte aligned) used as a temporary store in scalar
// fallback paths when Zvfh is absent.

template <typename scalar_t>
inline vfloat32m1_t load_as_float_m1(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m1(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16mf2_t v_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m1(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m1(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return bf16_to_f32m1(reinterpret_cast<const uint16_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m1(scratch, vl);
  }
}

template <typename scalar_t>
inline vfloat32m4_t load_as_float_m4(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m4(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m2_t v_f16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m4(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return bf16_to_f32m4(reinterpret_cast<const uint16_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

template <typename scalar_t>
inline vfloat32m8_t load_as_float_m8(const scalar_t* ptr, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vle32_v_f32m8(ptr, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m4_t v_f16 = __riscv_vle16_v_f16m4(reinterpret_cast<const _Float16*>(ptr), vl);
    return __riscv_vfwcvt_f_f_v_f32m8(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m8(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    return bf16_to_f32m8(reinterpret_cast<const uint16_t*>(ptr), vl);
  } else {
    for (size_t i = 0; i < vl; ++i)
      scratch[i] = static_cast<float>(ptr[i]);
    return __riscv_vle32_v_f32m8(scratch, vl);
  }
}

template <typename scalar_t>
inline vfloat32m4_t load_strided_as_float_m4(const scalar_t* ptr, ptrdiff_t stride_byte, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return __riscv_vlse32_v_f32m4(ptr, stride_byte, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m2_t v_f16 = __riscv_vlse16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), stride_byte, vl);
    return __riscv_vfwcvt_f_f_v_f32m4(v_f16, vl);
#else
    for (size_t i = 0; i < vl; ++i) {
      const scalar_t* elem_ptr =
          reinterpret_cast<const scalar_t*>(reinterpret_cast<const char*>(ptr) + i * stride_byte);
      scratch[i] = static_cast<float>(*elem_ptr);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    // Manual strided load + convert for BF16
    vuint16m2_t v_bf16 = __riscv_vlse16_v_u16m2(reinterpret_cast<const uint16_t*>(ptr), stride_byte, vl);
    vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_bf16, vl);
    v_u32 = __riscv_vsll_vx_u32m4(v_u32, 16, vl);
    return __riscv_vreinterpret_v_u32m4_f32m4(v_u32);
  } else {
    for (size_t i = 0; i < vl; ++i) {
      const scalar_t* elem_ptr =
          reinterpret_cast<const scalar_t*>(reinterpret_cast<const char*>(ptr) + i * stride_byte);
      scratch[i] = static_cast<float>(*elem_ptr);
    }
    return __riscv_vle32_v_f32m4(scratch, vl);
  }
}

template <typename scalar_t>
inline void
store_strided_from_float_m4(scalar_t* ptr, ptrdiff_t stride_byte, vfloat32m4_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vsse32_v_f32m4(ptr, stride_byte, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v, vl);
    __riscv_vsse16_v_f16m2(reinterpret_cast<_Float16*>(ptr), stride_byte, v_f16, vl);
#else
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      scalar_t* elem = reinterpret_cast<scalar_t*>(reinterpret_cast<char*>(ptr) + i * stride_byte);
      *elem = static_cast<scalar_t>(scratch[i]);
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
    __riscv_vsse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), stride_byte, v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i) {
      scalar_t* elem = reinterpret_cast<scalar_t*>(reinterpret_cast<char*>(ptr) + i * stride_byte);
      *elem = static_cast<scalar_t>(scratch[i]);
    }
  }
}

template <typename scalar_t>
inline void store_from_float_m1(scalar_t* ptr, vfloat32m1_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m1(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16mf2_t v_f16 = __riscv_vfncvt_f_f_w_f16mf2(v, vl);
    __riscv_vse16_v_f16mf2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint32m1_t v_u32 = __riscv_vreinterpret_v_f32m1_u32m1(v);
    vuint16mf2_t v_bf16 = __riscv_vnsrl_wx_u16mf2(v_u32, 16, vl);
    __riscv_vse16_v_u16mf2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m1(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

template <typename scalar_t>
inline void store_from_float_m2(scalar_t* ptr, vfloat32m2_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m2(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m1_t v_f16 = __riscv_vfncvt_f_f_w_f16m1(v, vl);
    __riscv_vse16_v_f16m1(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m2(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint32m2_t v_u32 = __riscv_vreinterpret_v_f32m2_u32m2(v);
    vuint16m1_t v_bf16 = __riscv_vnsrl_wx_u16m1(v_u32, 16, vl);
    __riscv_vse16_v_u16m1(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m2(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

template <typename scalar_t>
inline void store_from_float_m4(scalar_t* ptr, vfloat32m4_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m4(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v, vl);
    __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m4(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

template <typename scalar_t>
inline void store_from_float_m8(scalar_t* ptr, vfloat32m8_t v, size_t vl, float* scratch) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    __riscv_vse32_v_f32m8(ptr, v, vl);
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    vfloat16m4_t v_f16 = __riscv_vfncvt_f_f_w_f16m4(v, vl);
    __riscv_vse16_v_f16m4(reinterpret_cast<_Float16*>(ptr), v_f16, vl);
#else
    __riscv_vse32_v_f32m8(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    vuint16m4_t v_bf16 = f32m8_to_bf16(v, vl);
    __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(ptr), v_bf16, vl);
  } else {
    __riscv_vse32_v_f32m8(scratch, v, vl);
    for (size_t i = 0; i < vl; ++i)
      ptr[i] = static_cast<scalar_t>(scratch[i]);
  }
}

// Memory Operations (Copy, Fill, Transpose)

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int64_t size) {
  size_t vl = 0;

  if constexpr (std::is_same_v<scalar_t, float>) {
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vfloat32m4_t v_val = __riscv_vfmv_v_f_f32m4(val, vl);
      __riscv_vse32_v_f32m4(out + d, v_val, vl);
    }
  } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh)
    // FP16: use hardware narrowing convert
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m2(size - d);
      vfloat32m4_t v_f32 = __riscv_vfmv_v_f_f32m4(val, vl);
      vfloat16m2_t v_f16 = __riscv_vfncvt_f_f_w_f16m2(v_f32, vl);
      __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(out + d), v_f16, vl);
    }
#else
    const scalar_t hval = static_cast<scalar_t>(val);
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m2(size - d);
      for (size_t i = 0; i < vl; ++i)
        out[d + i] = hval;
    }
#endif
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    // BF16: use bit-shift method (BF16 is upper 16 bits of FP32)
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m2(size - d);
      vfloat32m4_t v_f32 = __riscv_vfmv_v_f_f32m4(val, vl);
      vuint16m2_t v_bf16 = f32m4_to_bf16(v_f32, vl);
      __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(out + d), v_bf16, vl);
    }
  } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
    // int32: used by MoE for sorted_ids, expert_ids, total_cnts
    int32_t ival = static_cast<int32_t>(val);
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vint32m4_t v_val = __riscv_vmv_v_x_i32m4(ival, vl);
      __riscv_vse32_v_i32m4(out + d, v_val, vl);
    }
  }
}

// Copy Stub Functions - Multiple Overloads

// 1. Same-type copy: scalar_t -> scalar_t
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  size_t vl = 0;  // Initialized to silence -Wuninitialized; always set before use in loop body.
  if constexpr (std::is_same_v<scalar_t, float>) {
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e32m4(size - d);
      vfloat32m4_t v_data = __riscv_vle32_v_f32m4(src + d, vl);
      __riscv_vse32_v_f32m4(out + d, v_data, vl);
    }
  } else {
    // 16-bit element copy (FP16, BF16): bitwise, no conversion needed.
    for (int64_t d = 0; d < size; d += vl) {
      vl = __riscv_vsetvl_e16m4(size - d);
      vuint16m4_t v_data = __riscv_vle16_v_u16m4(reinterpret_cast<const uint16_t*>(src + d), vl);
      __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(out + d), v_data, vl);
    }
  }
}

// 2. Type conversion with scale: (float * scale) -> scalar_t
template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ acc, float s, int64_t size) {
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  for (int64_t d = 0; d < size; d += vl) {
    vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_acc = __riscv_vle32_v_f32m4(acc + d, vl);
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_acc, s, vl);
    store_from_float_m4(out + d, v_scaled, vl, scratch);
  }
}

// 3. float -> scalar_t (no scale) — disabled when scalar_t=float to avoid ambiguity with overload 1
template <typename scalar_t, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  int64_t d = 0;
  // Process in chunks of vl_max
  for (; d + static_cast<int64_t>(vl_max) <= size; d += vl_max) {
    vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + d, vl_max);
    store_from_float_m4(out + d, v_data, vl_max, scratch);
  }

  // Handle remaining elements
  if (d < size) {
    size_t tail_vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + d, tail_vl);
    store_from_float_m4(out + d, v_data, tail_vl, scratch);
  }
}

// 4. scalar_t -> float — disabled when scalar_t=float to avoid ambiguity with overload 1
template <typename scalar_t, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void copy_stub(float* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  int64_t d = 0;
  // Process in chunks of vl_max
  for (; d + static_cast<int64_t>(vl_max) <= size; d += vl_max) {
    vfloat32m4_t v_f32 = load_as_float_m4(input + d, vl_max, scratch);
    __riscv_vse32_v_f32m4(out + d, v_f32, vl_max);
  }

  // Handle remaining elements
  if (d < size) {
    size_t tail_vl = __riscv_vsetvl_e32m4(size - d);
    vfloat32m4_t v_f32 = load_as_float_m4(input + d, tail_vl, scratch);
    __riscv_vse32_v_f32m4(out + d, v_f32, tail_vl);
  }
}

// Quantization Operations

template <typename scalar_t>
inline void quantize_row_uint8_asymmetric_with_scale(
    uint8_t* __restrict__ Aq, const scalar_t* __restrict__ A, int64_t K, float scale) {
  const float safe_scale = (scale > 1e-9f) ? scale : 1e-9f;
  const float inv_scale = 1.0f / safe_scale;
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);

    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);
    vfloat32m4_t v_scaled = __riscv_vfmul_vf_f32m4(v_val, inv_scale, vl);
    vint32m4_t v_i32 = __riscv_vfcvt_x_f_v_i32m4(v_scaled, vl);
    v_i32 = __riscv_vadd_vx_i32m4(v_i32, 128, vl);
    v_i32 = __riscv_vmax_vx_i32m4(v_i32, 0, vl);
    v_i32 = __riscv_vmin_vx_i32m4(v_i32, 255, vl);
    vuint32m4_t v_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_i32);
    vuint16m2_t v_u16 = __riscv_vnclipu_wx_u16m2(v_u32, 0, __RISCV_VXRM_RNU, vl);
    vuint8m1_t v_u8 = __riscv_vnclipu_wx_u8m1(v_u16, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_u8m1(Aq + k, v_u8, vl);
  }
}

template <typename scalar_t>
inline void quantize_row_uint8_asymmetric_infer_scale(
    uint8_t* __restrict__ Aq, float& scale_out, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7f) {
  float max_val = 0.f;
  size_t vl;
  alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M4];
  vfloat32m4_t v_max_acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

  for (int64_t k = 0; k < K; k += vl) {
    vl = __riscv_vsetvl_e32m4(K - k);
    vfloat32m4_t v_val = load_as_float_m4(A + k, vl, scratch);
    vfloat32m4_t v_abs = __riscv_vfsgnjx_vv_f32m4(v_val, v_val, vl);
    v_max_acc = __riscv_vfmax_vv_f32m4_tu(v_max_acc, v_max_acc, v_abs, vl);
  }

  vfloat32m1_t v_max_scalar =
      __riscv_vfredmax_vs_f32m4_f32m1(v_max_acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m4());
  max_val = __riscv_vfmv_f_s_f32m1_f32(v_max_scalar);
  max_val = std::max(max_val, eps);
  const float scale = max_val / 127.0f;
  quantize_row_uint8_asymmetric_with_scale(Aq, A, K, scale);
  scale_out = scale;
}

inline void pretransform_u8_to_centered_i8(int8_t* __restrict__ dst, const uint8_t* __restrict__ src, int64_t size) {
  size_t vl;
  for (int64_t k = 0; k < size; k += vl) {
    vl = __riscv_vsetvl_e8m1(size - k);
    vuint8m1_t v_u8 = __riscv_vle8_v_u8m1(src + k, vl);
    // Map uint8 [0, 255] into centered int8 [-128, 127] by flipping the sign bit.
    v_u8 = __riscv_vxor_vx_u8m1(v_u8, 0x80, vl);
    vint8m1_t v_i8 = __riscv_vreinterpret_v_u8m1_i8m1(v_u8);
    __riscv_vse8_v_i8m1(dst + k, v_i8, vl);
  }
}

inline void pack_weight_int8_with_comp(
    int8_t* __restrict__ packed_w, const int8_t* __restrict__ orig_w, int64_t N, int64_t K, int64_t block_n) {
  const int64_t packed_data_size = K * block_n;
  int32_t* packed_comp = reinterpret_cast<int32_t*>(packed_w + packed_data_size);
  size_t vl;

  // Zero-initialize compensation buffer once, then accumulate while packing.
  for (int64_t j = 0; j < block_n; j += vl) {
    vl = __riscv_vsetvl_e32m4(block_n - j);
    vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
    __riscv_vse32_v_i32m4(packed_comp + j, v_zero, vl);
  }

  for (int64_t k = 0; k < K; ++k) {
    int8_t* dst = packed_w + k * block_n;
    int64_t j = 0;
    while (j < N) {
      vl = __riscv_vsetvl_e8m1(N - j);
      vint8m1_t v_data = __riscv_vlse8_v_i8m1(orig_w + (j * K + k), K, vl);
      __riscv_vse8_v_i8m1(dst + j, v_data, vl);

      vint16m2_t v_i16 = __riscv_vsext_vf2_i16m2(v_data, vl);
      vint32m4_t v_i32 = __riscv_vsext_vf2_i32m4(v_i16, vl);
      vint32m4_t v_acc = __riscv_vle32_v_i32m4(packed_comp + j, vl);
      v_acc = __riscv_vadd_vv_i32m4(v_acc, v_i32, vl);
      __riscv_vse32_v_i32m4(packed_comp + j, v_acc, vl);

      j += vl;
    }
    while (j < block_n) {
      vl = __riscv_vsetvl_e8m1(block_n - j);
      vint8m1_t v_zero = __riscv_vmv_v_x_i8m1(0, vl);
      __riscv_vse8_v_i8m1(dst + j, v_zero, vl);
      j += vl;
    }
  }

  // Finalize shared-CPU-compatible compensation: 128 * sum(weight_col).
  for (int64_t j = 0; j < block_n; j += vl) {
    vl = __riscv_vsetvl_e32m4(block_n - j);
    vint32m4_t v_comp = __riscv_vle32_v_i32m4(packed_comp + j, vl);
    v_comp = __riscv_vmul_vx_i32m4(v_comp, 128, vl);
    __riscv_vse32_v_i32m4(packed_comp + j, v_comp, vl);
  }
}

struct AlignedArena {
  char* ptr;

  explicit AlignedArena(void* base, size_t align = 64) {
    auto addr = reinterpret_cast<uintptr_t>(base);
    ptr = reinterpret_cast<char*>((addr + align - 1) & ~(align - 1));
  }

  template <typename T>
  T* alloc(size_t count, size_t align = 64) {
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    ptr = reinterpret_cast<char*>((addr + align - 1) & ~(align - 1));
    T* result = reinterpret_cast<T*>(ptr);
    ptr += count * sizeof(T);
    return result;
  }
};

// Softmax helper: in-place exp and sum (no normalization)
// scores[i] = exp(scores[i] - max), returns sum = Σ scores[i]
// NOTE: Does NOT normalize scores. Callers use unnormalized exp values for the online softmax.
inline float exp_and_sum(float* __restrict__ scores, int n_size, float m_i) {
  if (n_size <= 0) return 0.0f;

  size_t vl_max = __riscv_vsetvlmax_e32m4();
  float total_sum = 0.0f;

  vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  for (int j = 0; j < n_size; j += vl_max) {
    size_t vl = __riscv_vsetvl_e32m4(n_size - j);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(scores + j, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, m_i, vl);
    vfloat32m4_t vex = vfexp_f32m4(vx, vl);
    __riscv_vse32_v_f32m4(scores + j, vex, vl);

    vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(vex, vzero, vl);
    total_sum += __riscv_vfmv_f_s_f32m1_f32(vsum);
  }

  return total_sum;
}

inline float rvv_reduce_max_f32(const float* data, int64_t len) {
  if (len <= 0) return -std::numeric_limits<float>::infinity();

  float max_val = -std::numeric_limits<float>::infinity();
  int64_t remaining = len;

  while (remaining > 0) {
    size_t vl = __riscv_vsetvl_e32m8(remaining);
    vfloat32m8_t vdata = __riscv_vle32_v_f32m8(data + (len - remaining), vl);

    vfloat32m1_t vcurrent_max = __riscv_vfmv_s_f_f32m1(max_val, 1);
    vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vdata, vcurrent_max, vl);
    max_val = __riscv_vfmv_f_s_f32m1_f32(vmax);

    remaining -= vl;
  }

  return max_val;
}

#endif  // CPU_CAPABILITY_RVV
