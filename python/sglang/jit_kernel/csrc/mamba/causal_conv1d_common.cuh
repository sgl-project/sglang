/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
// clang-format off
// Common helpers for the causal_conv1d JIT kernels.
// Adapted from sgl-kernel/csrc/mamba/causal_conv1d.h.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace {

// Shared parameter struct used by both the forward and update kernels.
struct ConvParamsBase {
  using index_t = uint32_t;

  int batch, dim, seqlen, width;
  int64_t pad_slot_id;
  bool silu_activation;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;
  index_t weight_c_stride;
  index_t weight_width_stride;
  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  int conv_state_len;
  index_t conv_state_batch_stride;
  index_t conv_state_c_stride;
  index_t conv_state_l_stride;

  void* __restrict__ x_ptr;
  void* __restrict__ weight_ptr;
  void* __restrict__ bias_ptr;
  void* __restrict__ out_ptr;

  void* __restrict__ conv_state_ptr;
  void* __restrict__ query_start_loc_ptr;
  void* __restrict__ has_initial_state_ptr;
  void* __restrict__ cache_indices_ptr;
  int32_t* __restrict__ cache_seqlens;
  int32_t* __restrict__ conv_state_indices_ptr;

  void* __restrict__ seq_idx_ptr;

  void* initial_states_ptr;
  index_t initial_states_batch_stride;
  index_t initial_states_l_stride;
  index_t initial_states_c_stride;

  void* final_states_ptr;
  index_t final_states_batch_stride;
  index_t final_states_l_stride;
  index_t final_states_c_stride;

  void* conv_states_ptr;
  index_t conv_states_batch_stride;
  index_t conv_states_l_stride;
  index_t conv_states_c_stride;
};

// Constexpr max over an initializer_list (libc++ doesn't make std::max constexpr on ROCm).
constexpr inline size_t conv_custom_max(std::initializer_list<size_t> ilist) {
#ifndef USE_ROCM
  return std::max(ilist);
#else
  return *std::max_element(ilist.begin(), ilist.end());
#endif
}

// Map a byte count to a corresponding integer-vector type for vectorized loads/stores.
template <int BYTES>
struct ConvBytesToType {};

template <>
struct ConvBytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct ConvBytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct ConvBytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct ConvBytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

// Compile-time bool-to-template-arg switch (constexpr-if helper inside lambdas).
#define CONV_BOOL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    if (COND) {                                 \
      static constexpr bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      static constexpr bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

}  // namespace
