/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "marlin_template.h"

namespace device::marlin_moe {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {};

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
template <int moe_block_size>
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    const int32_t* __restrict__ sorted_token_ids_ptr,
    const int32_t* __restrict__ expert_ids_ptr,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int size_m,
    int size_k,
    int top_k) {
  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  int num_moe_blocks = div_ceil(num_tokens_past_padded, moe_block_size);
  int32_t block_sorted_ids[moe_block_size];
  int block_num_valid_tokens = 0;
  int64_t old_expert_id = 0;
  int64_t expert_id = 0;
  int row_stride = size_k * sizeof(half) / 16;

  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;
    int4* tmp_block_sorted_ids = reinterpret_cast<int4*>(block_sorted_ids);
    for (int i = 0; i < moe_block_size / 4; i++) {
      tmp_block_sorted_ids[i] = ((int4*)sorted_token_ids_ptr)[block_id * moe_block_size / 4 + i];
    }
    for (int i = 0; i < moe_block_size; i++) {
      if (block_sorted_ids[i] >= size_m * top_k) {
        block_num_valid_tokens = i;
        break;
      };
    }
  };

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int in_offset = (row / top_k) * row_stride;
    int out_offset = row * row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + in_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + out_offset);

    int base_k = 0;

    for (int i = 0; i < iters; i++) {
      auto cur_k = base_k + threadIdx.x;
      int src_pos = perm_int_ptr[cur_k];

      out_half[cur_k] = a_row_half[src_pos];

      base_k += default_threads;
    }

    if (rest) {
      if (threadIdx.x < rest) {
        auto cur_k = base_k + threadIdx.x;
        int src_pos = perm_int_ptr[cur_k];

        out_half[cur_k] = a_row_half[src_pos];
      }
    }
  };

  for (int index = blockIdx.x; index < num_moe_blocks; index += gridDim.x) {
    old_expert_id = expert_id;
    int tmp_expert_id = expert_ids_ptr[index];
    if (tmp_expert_id == -1) continue;
    expert_id = tmp_expert_id;
    perm_int_ptr += (expert_id - old_expert_id) * size_k;
    read_moe_block_data(index);

    for (int i = 0; i < block_num_valid_tokens; i++)
      permute_row(block_sorted_ids[i]);
  }
}

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(
    thread_config_t const& th_config,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * pipe_stages * 2;  // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);             // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * pipe_stages;
  }
}

int get_kernel_cache_size(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 4;
  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);
  int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size + sh_block_meta_size;

  return total_size;
}

bool is_valid_config(
    thread_config_t const& th_config,
    bool m_block_size_8,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float,
    int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config,
      m_block_size_8,
      thread_m_blocks,
      prob_m,
      prob_n,
      prob_k,
      num_bits,
      group_size,
      has_act_order,
      is_k_full,
      has_zp,
      is_zp_float);
  return cache_size + 512 <= max_shared_mem;
}

#define _GET_IF(                                                                                                       \
    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
  else if (                                                                                                            \
      q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                  \
      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&        \
      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                      \
    constexpr auto S_TYPE = W_TYPE == host::kFE2M1f                                                                    \
                                ? (GROUP_BLOCKS == 1 ? host::kFE4M3fn : host::kFE8M0fnu)                               \
                                : (std::is_same<scalar_t, half>::value ? host::kFloat16 : host::kBFloat16);            \
    kernel = Marlin<                                                                                                   \
        scalar_t,                                                                                                      \
        W_TYPE.id(),                                                                                                   \
        S_TYPE.id(),                                                                                                   \
        NUM_THREADS,                                                                                                   \
        THREAD_M_BLOCKS,                                                                                               \
        THREAD_N_BLOCKS,                                                                                               \
        THREAD_K_BLOCKS,                                                                                               \
        M_BLOCK_SIZE_8,                                                                                                \
        pipe_stages,                                                                                                   \
        GROUP_BLOCKS,                                                                                                  \
        IS_ZP_FLOAT>;                                                                                                  \
  }

// COMMON: cases for (group_blocks in [-1, 2, 4, 8] and is_zp_float == false)
//         this is the most common cases
// BIGGROUP: cases for big group size (group_blocks in [-1, 8])
// FZP: cases for float-zero-point (is_zp_float = true)
// ACT: cases for act order case (group_blocks == 0)
// NVFP4: cases for nvfp4(e2m1) (group_blocks == 1)
// MXFP4: cases for mxfp4(e2m1) (group_blocks == 2)
#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                        \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                        \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF(W_TYPE)            \
  COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)

#define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)   \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF(W_TYPE)            \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)

#define NVFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define NVFP4_GET_IF(W_TYPE)            \
  NVFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  NVFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  NVFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  NVFP4_GET_IF_M234(W_TYPE, 8, 4, 128)

#define MXFP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)

#define MXFP4_GET_IF(W_TYPE)            \
  MXFP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  MXFP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  MXFP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  MXFP4_GET_IF_M234(W_TYPE, 8, 4, 128)

// We currently have 4-bit models only with group_blocks == 4
#define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)      \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF(W_TYPE)            \
  FZP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  FZP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FZP_GET_IF_M234(W_TYPE, 8, 4, 128)

// We currently have 4-bit models only with group_blocks == 4
#define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)        \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF(W_TYPE)            \
  ACT_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  ACT_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \
  ACT_GET_IF_M234(W_TYPE, 8, 4, 128)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const host::ScalarType q_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    bool has_act_order,
    bool has_zp,
    int group_blocks,
    int num_threads,
    bool is_zp_float) {
  int num_bits = q_type.size_bits();
  auto kernel = MarlinDefault;
  if (false) {
  }

  COMMON_GET_IF(host::kU4)
  COMMON_GET_IF(host::kU4B8)
  COMMON_GET_IF(host::kU8B128)

  NVFP4_GET_IF(host::kFE2M1f)

  BIGGROUP_GET_IF(host::kFE4M3fn)

  ACT_GET_IF(host::kU4B8)
  ACT_GET_IF(host::kU8B128)
  if (std::is_same<scalar_t, nv_bfloat16>::value) {
    if (false) {
    }
    MXFP4_GET_IF(host::kFE2M1f)
  }

  return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(
    const host::ScalarType& q_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config,
            m_block_size_8,
            thread_m_blocks,
            prob_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            has_act_order,
            is_k_full,
            has_zp,
            is_zp_float,
            max_shared_mem)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config,
        m_block_size_8,
        thread_m_blocks,
        prob_m,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
        has_zp,
        is_zp_float);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : (group_size / 16);
    }

    auto kernel = get_marlin_kernel<scalar_t>(
        q_type,
        thread_m_blocks,
        th_config.thread_n / 16,
        th_config.thread_k / 16,
        m_block_size_8,
        has_act_order,
        has_zp,
        group_blocks,
        th_config.num_threads,
        is_zp_float);

    if (kernel == MarlinDefault) continue;

    if (thread_m_blocks > 1) {
      exec_cfg = {1, th_config};
      break;
    } else {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, kernel);
      int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
      int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1024));
      allow_count = max(min(allow_count, 4), 1);
      if (allow_count > count) {
        count = allow_count;
        exec_cfg = {count, th_config};
      };
    }
  }

  return exec_cfg;
}

template <typename scalar_t>
void marlin_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* b_bias,
    void* s,
    void* s2,
    void* zp,
    void* g_idx,
    void* perm,
    void* a_tmp,
    void* sorted_token_ids,
    void* expert_ids,
    void* num_tokens_past_padded,
    void* topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    bool is_ep,
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,
    host::ScalarType const& q_type,
    bool has_bias,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;

  if (has_zp) {
    host::RuntimeCheck(
        q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
  } else {
    host::RuntimeCheck(
        q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,
        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
        "has_zp = False. Got = ",
        q_type.str());
  }

  host::RuntimeCheck(
      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      host::RuntimeCheck(group_size != -1);
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    } else {
      host::RuntimeCheck(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      host::RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const int4* s_ptr = (const int4*)s;
  const uint16_t* s2_ptr = (const uint16_t*)s2;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    auto perm_kernel = permute_cols_kernel<8>;
    if (moe_block_size == 8) {
    } else if (moe_block_size == 16)
      perm_kernel = permute_cols_kernel<16>;
    else if (moe_block_size == 32)
      perm_kernel = permute_cols_kernel<32>;
    else if (moe_block_size == 48)
      perm_kernel = permute_cols_kernel<48>;
    else if (moe_block_size == 64)
      perm_kernel = permute_cols_kernel<64>;
    else
      host::Panic("unsupported moe_block_size ", moe_block_size);

    // clang-format off
    perm_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  host::RuntimeCheck(max_shared_mem > 0);

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
    exec_cfg = exec_config_t{1, thread_tfg};
    host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
    host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
  } else {
    // Auto config
    exec_cfg = determine_exec_config<scalar_t>(
        q_type,
        prob_m,
        prob_n,
        prob_k,
        thread_m_blocks,
        m_block_size_8,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
        has_zp,
        is_zp_float,
        max_shared_mem);
    thread_tfg = exec_cfg.tb_cfg;
  }

  int num_threads = thread_tfg.num_threads;
  thread_k = thread_tfg.thread_k;
  thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  host::RuntimeCheck(
      is_valid_config(
          thread_tfg,
          m_block_size_8,
          thread_m_blocks,
          prob_m,
          prob_n,
          prob_k,
          num_bits,
          group_size,
          has_act_order,
          is_k_full,
          has_zp,
          is_zp_float,
          max_shared_mem),
      "Invalid thread config: thread_m_blocks = ",
      thread_m_blocks,
      ", thread_k = ",
      thread_tfg.thread_k,
      ", thread_n = ",
      thread_tfg.thread_n,
      ", num_threads = ",
      thread_tfg.num_threads,
      " for MKN = [",
      prob_m,
      ", ",
      prob_k,
      ", ",
      prob_n,
      "] and num_bits = ",
      num_bits,
      ", group_size = ",
      group_size,
      ", has_act_order = ",
      has_act_order,
      ", is_k_full = ",
      is_k_full,
      ", has_zp = ",
      has_zp,
      ", is_zp_float = ",
      is_zp_float,
      ", max_shared_mem = ",
      max_shared_mem);

  auto kernel = get_marlin_kernel<scalar_t>(
      q_type,
      thread_m_blocks,
      thread_n_blocks,
      thread_k_blocks,
      m_block_size_8,
      has_act_order,
      has_zp,
      group_blocks,
      num_threads,
      is_zp_float);

  if (kernel == MarlinDefault) {
    host::Panic(
        "Unsupported shapes: MNK = [",
        prob_m,
        ", ",
        prob_n,
        ", ",
        prob_k,
        "]",
        ", has_act_order = ",
        has_act_order,
        ", num_groups = ",
        num_groups,
        ", group_size = ",
        group_size,
        ", thread_m_blocks = ",
        thread_m_blocks,
        ", thread_n_blocks = ",
        thread_n_blocks,
        ", thread_k_blocks = ",
        thread_k_blocks,
        ", num_bits = ",
        num_bits);
  }

  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,
      topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,
      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem);
  // clang-format on
}

#endif

}  // namespace device::marlin_moe

template <typename scalar_t>
void moe_wna16_marlin_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView global_scale,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView g_idx,
    tvm::ffi::TensorView perm,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView a_tmp,
    tvm::ffi::TensorView c_tmp,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool has_act_order,
    bool has_bias,
    bool is_k_full,
    bool has_zp,
    int64_t num_groups,
    int64_t group_size,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  int pack_factor = 32 / b_q_type.size_bits();

  if (moe_block_size != 8) {
    RuntimeCheck(moe_block_size % 16 == 0, "unsupported moe_block_size=", moe_block_size);
    RuntimeCheck(moe_block_size >= 16 && moe_block_size <= 64, "unsupported moe_block_size=", moe_block_size);
  }

  // Verify A
  RuntimeCheck(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0), ", size_m = ", size_m);
  RuntimeCheck(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1), ", size_k = ", size_k);

  // Verify B
  RuntimeCheck(
      size_k % device::marlin::tile_size == 0,
      "size_k = ",
      size_k,
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  RuntimeCheck(
      (size_k / device::marlin::tile_size) == b_q_weight.size(1),
      "Shape mismatch: b_q_weight.size(1) = ",
      b_q_weight.size(1),
      ", size_k = ",
      size_k,
      ", tile_size = ",
      device::marlin::tile_size);
  RuntimeCheck(
      b_q_weight.size(2) % device::marlin::tile_size == 0,
      "b_q_weight.size(2) = ",
      b_q_weight.size(2),
      " is not divisible by tile_size = ",
      device::marlin::tile_size);
  int64_t actual_size_n = (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor;
  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);

  device.verify(b_q_weight.device());
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  device.verify(b_scales.device());
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");

  // thread_k, thread_n, sms
  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  // Verify c (allocation done in Python)
  device.verify(c.device());
  RuntimeCheck(c.is_contiguous(), "c is not contiguous");
  RuntimeCheck(
      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);
  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);

  // Alloc c_tmp: SKIP, done in Python

  // Detect groupsize: b_scales rank and dims
  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");
  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);
  RuntimeCheck(
      b_scales.size(1) == num_groups, "b_scales dim 1 = ", b_scales.size(1), " is not num_groups = ", num_groups);

  // Validate g_idx, perm (Optional unwrap done in Python; empty tensors when absent)
  if (g_idx.size(g_idx.dim() - 1) > 0 && perm.size(perm.dim() - 1) > 0) {
    device.verify(g_idx.device());
    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");
    device.verify(perm.device());
    RuntimeCheck(perm.is_contiguous(), "perm is not contiguous");

    int64_t g_idx_last = g_idx.size(g_idx.dim() - 1);
    int64_t perm_last = perm.size(perm.dim() - 1);
    RuntimeCheck(
        (g_idx_last == 0 && perm_last == 0) || (g_idx_last == size_k && perm_last == size_k),
        "Unexpected g_idx.size(-1) = ",
        g_idx_last,
        " and perm.size(-1) = ",
        perm_last,
        ", where size_k = ",
        size_k);
  }
  // has_act_order derivation: SKIP (passed as param)

  // Verify group_size consistency
  if (has_act_order) {
    // SKIP: a_tmp allocation done in Python
    if (is_k_full) {
      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
    }
  } else {
    if (num_groups > 1) {
      RuntimeCheck(
          size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);
    }
  }

  // Verify global_scale (Optional unwrap done in Python)
  int64_t global_scale_size = global_scale.size(0);
  if (global_scale_size > 0) {
    RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scale can only be used for nvfp4 format.");
  } else {
    RuntimeCheck(
        !(b_q_type == kFE2M1f && group_size == 16), "the global_scale parameter must be passed for nvfp4 format.");
  }

  // Verify b_bias (Optional unwrap done in Python)
  if (has_bias) {
    device.verify(b_bias.device());
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(0) != size_n");
    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
  }

  // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)

  // Verify b_q_type vs has_zp
  if (has_zp) {
    device.verify(b_zeros.device());
    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    RuntimeCheck(
        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
  } else {
    RuntimeCheck(
        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "
        "float4_e2m1f when "
        "has_zp = False. Got = ",
        b_q_type.str());
  }

  if (has_zp && is_zp_float) {
    RuntimeCheck(
        std::is_same<scalar_t, fp16_t>::value,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");
    if (is_zp_float) {
      RuntimeCheck(b_zeros.size(2) == size_n, "b_zeros dim 2 = ", b_zeros.size(2), " is not size_n = ", size_n);
      RuntimeCheck(
          num_groups == b_zeros.size(1), "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(num_groups != -1, "num_groups must be != -1");
    } else {
      RuntimeCheck(
          b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
      RuntimeCheck(
          b_zeros.size(2) == size_n / pack_factor,
          "b_zeros dim 2 = ",
          b_zeros.size(2),
          " is not size_n / pack_factor = ",
          size_n / pack_factor);
    }
  }

  // Verify workspace size
  RuntimeCheck(
      size_n % device::marlin::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      device::marlin::min_thread_n);

  int64_t max_n_tiles = size_n / device::marlin::min_thread_n;
  int64_t min_workspace_size =
      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);
  RuntimeCheck(
      workspace.size(0) >= min_workspace_size,
      "workspace.numel = ",
      workspace.size(0),
      " is below min_workspace_size = ",
      min_workspace_size);

  // Early return for zero-size M (moved after all validation)
  if (size_m == 0) return;

  device::marlin_moe::marlin_mm<scalar_t>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      b_scales.data_ptr(),
      global_scale.data_ptr(),
      b_zeros.data_ptr(),
      g_idx.data_ptr(),
      perm.data_ptr(),
      a_tmp.data_ptr(),
      sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(),
      num_tokens_post_padded.data_ptr(),
      topk_weights.data_ptr(),
      static_cast<int>(moe_block_size),
      static_cast<int>(top_k),
      mul_topk_weights,
      is_ep,
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      workspace.data_ptr(),
      b_q_type,
      has_bias,
      has_act_order,
      is_k_full,
      has_zp,
      static_cast<int>(num_groups),
      static_cast<int>(group_size),
      dev,
      stream,
      thread_k,
      thread_n,
      sms,
      use_atomic_add,
      use_fp32_reduce,
      is_zp_float);
}
