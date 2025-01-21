// Adapted from https://github.com/vllm-project/vllm/blob/main/csrc/quantization/vectorization.cuh
#pragma once
/**
 * __device__ datatypes vectorized by 4
 */

// Include both AMD and NVIDIA fp8 types to avoid circular import
// TODO(luka/varun) use FP8_TYPE instead after refactoring
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

// 8-element vector for half and bfloat16
template <typename scalar_t>
struct __align__(16) vec8_t {
  scalar_t x1;
  scalar_t x2;
  scalar_t x3;
  scalar_t x4;
  scalar_t x5;
  scalar_t x6;
  scalar_t x7;
  scalar_t x8;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> || std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};
