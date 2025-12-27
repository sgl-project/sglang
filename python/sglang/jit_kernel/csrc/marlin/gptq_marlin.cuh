/*
 * Modified by SGLang Team
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

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/utils.h>    // For div_ceil, RuntimeCheck

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "gptq_marlin_dispatcher.cuh"
#include "gptq_marlin_kernel.cuh"

namespace MARLIN_NAMESPACE_NAME {

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
__global__ void permute_cols_kernel(
    int4 const* __restrict__ a_int4_ptr,
    int const* __restrict__ perm_int_ptr,
    int4* __restrict__ out_int4_ptr,
    int size_m,
    int size_k,
    int lda,
    int block_rows) {
  auto start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_block_rows = finish_row - start_row;

  int input_row_stride = lda * sizeof(half) / 16;
  int output_row_stride = size_k * sizeof(half) / 16;

  auto permute_row = [&](int row) {
    int iters = size_k / default_threads;
    int rest = size_k % default_threads;

    int input_offset = row * input_row_stride;
    int output_offset = row * output_row_stride;

    half const* a_row_half = reinterpret_cast<half const*>(a_int4_ptr + input_offset);
    half* out_half = reinterpret_cast<half*>(out_int4_ptr + output_offset);

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

  for (int i = 0; i < cur_block_rows; i++) {
    int cur_row = start_row + i;
    if (cur_row < size_m) {
      permute_row(cur_row);
    }
  }
}

template <typename scalar_t>
void marlin_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* s,
    void* s2,
    void* zp,
    void* g_idx,
    void* perm,
    void* a_tmp,
    int prob_m,
    int prob_n,
    int prob_k,
    int lda,
    void* workspace,
    sglang::ScalarType const& q_type,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k_init,
    int thread_n_init,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  const DLDevice device = DLDevice{kDLCUDA, dev};
  if (has_zp) {
    RuntimeCheck(
        q_type == sglang::kU4 || q_type == sglang::kU8,
        "q_type must be u4 or u8 when has_zp = True. Got = ",
        q_type.str());
  } else {
    RuntimeCheck(
        q_type == sglang::kU4B8 || q_type == sglang::kU8B128 || q_type == sglang::kFE4M3fn || q_type == sglang::kFE2M1f,
        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
        "has_zp = False. Got = ",
        q_type.str());
  }

  RuntimeCheck(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      RuntimeCheck(group_size != -1);
      group_blocks = group_size / 16;
      RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    } else {
      RuntimeCheck(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      RuntimeCheck(
          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = q_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* s_ptr = (const int4*)s;
  const uint16_t* s2_ptr = (const uint16_t*)s2;
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;

  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, sms);
    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    LaunchKernel(sms, default_threads, stream)(
      permute_cols_kernel,
      A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows
    );
    // clang-format on
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  RuntimeCheck(max_shared_mem > 0);

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
      RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config<scalar_t>(
          q_type,
          prob_m_split,
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
          max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1) max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

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

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);

    bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    LaunchKernel(blocks, num_threads, device, static_cast<size_t>(max_shared_mem_new))(
        kernel,
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    A_ptr += prob_m_split * (lda / 8);
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace MARLIN_NAMESPACE_NAME

void gptq_marlin_gemm(
    tvm::ffi::TensorView a,
    tvm::ffi::Optional<tvm::ffi::TensorView> a_tmp,
    tvm::ffi::TensorView c,
    tvm::ffi::Optional<tvm::ffi::TensorView> c_tmp,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::Optional<tvm::ffi::TensorView> global_scale,
    tvm::ffi::Optional<tvm::ffi::TensorView> b_zeros,
    tvm::ffi::Optional<tvm::ffi::TensorView> g_idx,
    tvm::ffi::Optional<tvm::ffi::TensorView> perm,
    tvm::ffi::TensorView workspace,
    sglang::ScalarTypeId const& b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
  using namespace host;

  const sglang::ScalarType b_q_type = sglang::ScalarType::from_id(static_cast<sglang::ScalarTypeId>(b_q_type_id));
  const int pack_factor = 32 / b_q_type.size_bits();
  const int actual_size_n = (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  const int dev_id = a.device().device_id;
  const bool has_zp = b_zeros.has_value() && b_zeros.value().size(-1) > 0;

  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev_id);

  if (size_m == 0) return;

  // Detect groupsize and act_order
  int num_groups = -1;

  num_groups = b_scales.size(0);
  bool has_act_order =
      (g_idx.has_value() && g_idx.value().size(-1) > 0) && (perm.has_value() && perm.value().size(-1) > 0);

  int group_size = -1;
  if (has_act_order) {
    if (is_k_full) {
      group_size = static_cast<int>(size_k / num_groups);
    } else {
      group_size = 0;
    }
  } else {
    if (num_groups > 1) {
      group_size = static_cast<int>(size_k / num_groups);
    } else {
      group_size = -1;
    }
  }

  int min_workspace_size = sms;
  int dev = a.device().device_id;

  DLDataType a_dtype = a.dtype();
  if (a_dtype.code == kDLFloat && a_dtype.bits == 16) {
    void* scales_ptr;
    if (b_q_type == sglang::kFE2M1f) {
      scales_ptr = static_cast<__nv_fp8_e4m3*>(b_scales.data_ptr());
    } else {
      scales_ptr = static_cast<__half*>(b_scales.data_ptr());
    }
    marlin::marlin_mm<half>(
        static_cast<__half*>(a.data_ptr()),
        b_q_weight.data_ptr(),
        c.data_ptr(),
        c_tmp.has_value() ? c_tmp.value().data_ptr() : nullptr,
        scales_ptr,
        global_scale.has_value() ? global_scale.value().data_ptr() : nullptr,
        b_zeros.has_value() ? b_zeros.value().data_ptr() : nullptr,
        g_idx.has_value() ? g_idx.value().data_ptr() : nullptr,
        perm.has_value() ? perm.value().data_ptr() : nullptr,
        a_tmp.has_value() ? static_cast<__half*>(a_tmp.value().data_ptr()) : nullptr,
        static_cast<int>(size_m),
        static_cast<int>(size_n),
        static_cast<int>(size_k),
        a.stride(0),
        workspace.data_ptr(),
        b_q_type,
        has_act_order,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        dev,
        stream,
        thread_k,
        thread_n,
        sms,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float);
  } else {
    void* scales_ptr;
    if (b_q_type == sglang::kFE2M1f) {
      scales_ptr = static_cast<__nv_fp8_e4m3*>(b_scales.data_ptr());
    } else {
      scales_ptr = static_cast<__nv_bfloat16*>(b_scales.data_ptr());
    }

    marlin::marlin_mm<nv_bfloat16>(
        static_cast<nv_bfloat16*>(a.data_ptr()),
        b_q_weight.data_ptr(),
        c.data_ptr(),
        c_tmp.has_value() ? c_tmp.value().data_ptr() : nullptr,
        scales_ptr,
        global_scale.has_value() ? global_scale.value().data_ptr() : nullptr,
        b_zeros.has_value() ? b_zeros.value().data_ptr() : nullptr,
        g_idx.has_value() ? g_idx.value().data_ptr() : nullptr,
        perm.has_value() ? perm.value().data_ptr() : nullptr,
        a_tmp.has_value() ? static_cast<nv_bfloat16*>(a_tmp.value().data_ptr()) : nullptr,
        static_cast<int>(size_m),
        static_cast<int>(size_n),
        static_cast<int>(size_k),
        a.stride(0),
        workspace.data_ptr(),
        b_q_type,
        has_act_order,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        dev,
        stream,
        thread_k,
        thread_n,
        sms,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float);
  }
}
