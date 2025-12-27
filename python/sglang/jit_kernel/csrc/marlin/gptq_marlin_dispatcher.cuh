// Marlin Dispatcher
#pragma once

#include <sgl_kernel/marlin/marlin.cuh>
#include <sgl_kernel/marlin/scalar_type.hpp>

#include <cuda_fp16.h>     // half
#include <cuda_runtime.h>  // int4, __global__ 等（通常够用）

#include <type_traits>

#include "gptq_marlin_kernel.cuh"

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#define MARLIN_KERNEL_PARAMS                                                                                         \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
      const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, \
      const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks,        \
      bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME {
__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

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
  int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
  int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8);
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

  int total_size = max(sh_b_size, sh_red_size) + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(
    thread_config_t const& th_config,
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
  return cache_size <= max_shared_mem;
}

#define _GET_IF(                                                                                                       \
    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \
  else if (                                                                                                            \
      q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&                  \
      thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS &&        \
      num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) {                                                      \
    kernel = Marlin<                                                                                                   \
        scalar_t,                                                                                                      \
        W_TYPE.id(),                                                                                                   \
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
// FP4: cases for nvfp4(e2m1) (group_blocks == 1)
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
  COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
  COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)

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
  BIGGROUP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  BIGGROUP_GET_IF_M234(W_TYPE, 4, 8, 128)

#define FP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)        \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
  _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
  _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF(W_TYPE)            \
  FP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
  FP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
  FP4_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  FP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FP4_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  FP4_GET_IF_M234(W_TYPE, 4, 8, 128)

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
  FZP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \
  FZP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  FZP_GET_IF_M234(W_TYPE, 4, 8, 128)

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
  ACT_GET_IF_M1(W_TYPE, 4, 8, 128)    \
  ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \
  ACT_GET_IF_M234(W_TYPE, 8, 4, 128)  \
  ACT_GET_IF_M234(W_TYPE, 4, 8, 128)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const sglang::ScalarType q_type,
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

  COMMON_GET_IF(sglang::kU4)
  COMMON_GET_IF(sglang::kU4B8)
  COMMON_GET_IF(sglang::kU8B128)

  FP4_GET_IF(sglang::kFE2M1f)

  BIGGROUP_GET_IF(sglang::kFE4M3fn)

  ACT_GET_IF(sglang::kU4B8)
  ACT_GET_IF(sglang::kU8B128)

  if (std::is_same<scalar_t, half>::value) {
    if (false) {
    }
    FZP_GET_IF(sglang::kU4)
  }

  return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(
    const sglang::ScalarType& q_type,
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
    int max_shared_mem,
    int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config,
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
      group_blocks = group_size == -1 ? -1 : group_size / 16;
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

    // int m_tiles = div_ceil(prob_m, thread_m_blocks * 16);
    // int n_tiles = prob_n / th_config.thread_n;
    // int k_tiles = prob_k / th_config.thread_k;

    return {1, th_config};
  }

  return exec_cfg;
}
}  // namespace MARLIN_NAMESPACE_NAME
