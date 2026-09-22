/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 */
#pragma once

#include <torch/extension.h>

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define INTEGER_TYPE_SWITCH(type, INDEX_TYPE, ...)   \
  [&] {                                              \
    if (type == torch::kLong) {                      \
      using INDEX_TYPE = int64_t;                    \
      return __VA_ARGS__();                          \
    } else if(type == torch::kInt32) {               \
      using INDEX_TYPE = int32_t;                    \
      return __VA_ARGS__();                          \
    } else {                                         \
      TORCH_CHECK(                                   \
        false, "Unsupported integer dtype: ", type);                 \
    }                                                \
  }()

#define FLOATING_TYPE_SWITCH(type, FP_TYPE, ...)        \
  [&] {                                                 \
    if (type == at::ScalarType::Float) {                \
      using FP_TYPE = float;                            \
      return __VA_ARGS__();                             \
    } else if (type == at::ScalarType::BFloat16) {      \
      using FP_TYPE = nv_bfloat16;                      \
      return __VA_ARGS__();                             \
    } else {                                            \
      TORCH_CHECK(                                      \
        false, "Unsupported floating point dtype: ", type);                    \
    }                                                   \
  }()

#define CLUSTER_SIZE_SWITCH(cluster_size, ...)                   \
  [&] {                                                          \
    if (cluster_size == 1) {                                     \
      constexpr static int CLUSTER_SIZE = 1;                     \
      return __VA_ARGS__();                                      \
    } else if (cluster_size == 2) {                              \
      constexpr static int CLUSTER_SIZE = 2;                     \
      return __VA_ARGS__();                                      \
    } else if (cluster_size == 4) {                              \
      constexpr static int CLUSTER_SIZE = 4;                     \
      return __VA_ARGS__();                                      \
    } else if (cluster_size == 8) {                              \
      constexpr static int CLUSTER_SIZE = 8;                     \
      return __VA_ARGS__();                                      \
    } else {                                                     \
      TORCH_CHECK(                                               \
        false, "Unsupported cluster_size");                      \
    }                                                            \
  }()
