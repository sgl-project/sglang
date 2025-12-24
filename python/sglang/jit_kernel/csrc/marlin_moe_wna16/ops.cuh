
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/warp.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm_ffi_utils.h>

#include <optional>

#include "marlin.cuh"

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

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

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

template <typename scalar_t, bool has_zp>
exec_config_t determine_exec_config(
    const DLDataType q_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool is_zp_float,
    int max_shared_mem) {
  return exec_config_t{1, thread_config_t{-1, -1, -1}};
}

template <typename scalar_t, int moe_block_size, bool has_zp, bool is_zp_float>
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
    int top_k,
    bool mul_topk_weights,
    bool is_ep,
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,
    DLDataType q_type,
    bool has_bias,
    bool has_act_order,
    bool is_k_full,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k,
    int thread_n,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce) {
  int thread_m_blocks = div_ceil(moe_block_size, 16);
  bool m_block_size_8 = moe_block_size == 8;
  if (has_zp) {
    TVM_FFI_ICHECK(q_type == dl_uint8 || q_type == dl_uint8)
        << "q_type must be u4 or u8 when has_zp = True. Got = " << q_type;
  } else {
    TVM_FFI_ICHECK(q_type == dl_uint8 || q_type == dl_uint8 || q_type == dl_fp8_e4m3fn || q_type == dl_fp4_e2m1fn)
        << "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when has_zp = False. Got = " << q_type;
  }

  TVM_FFI_ICHECK(prob_m > 0 && prob_n > 0 && prob_k > 0)
      << "Invalid MNK = [" << prob_m << ", " << prob_n << ", " << prob_k << "]";

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TVM_FFI_ICHECK(group_size != -1);
      group_blocks = group_size / 16;
      TVM_FFI_ICHECK(prob_k % group_blocks == 0)
          << "prob_k = " << prob_k << " is not divisible by group_blocks = " << group_blocks;
    } else {
      TVM_FFI_ICHECK(group_size == 0) << "group_size must be 0";
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TVM_FFI_ICHECK(prob_k % group_blocks == 0)
          << "prob_k = " << prob_k << " is not divisible by group_blocks = " << group_blocks;
    }
  }

  int num_bits = q_type.bits;
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
    TVM_FFI_ICHECK(
        moe_block_size == 8 || moe_block_size == 16 || moe_block_size == 32 || moe_block_size == 48 ||
        moe_block_size == 64)
        << "unsupported moe_block_size";

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    permute_cols_kernel<moe_block_size><<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,
        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);
    // clang-format on
    A_ptr = a_tmp_ptr;
    prob_m = prob_m * top_k;
    top_k = 1;

    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  TVM_FFI_ICHECK(max_shared_mem > 0) << "max_shared_mem must be > 0";

  // Set thread config
  exec_config_t exec_cfg;
  thread_config_t thread_tfg;
  if (thread_k != -1 && thread_n != -1) {
    thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
    exec_cfg = exec_config_t{1, thread_tfg};
    TVM_FFI_ICHECK(prob_n % thread_n == 0) << "prob_n = " << prob_n << " is not divisible by thread_n = " << thread_n;
    TVM_FFI_ICHECK(prob_k % thread_k == 0) << "prob_k = " << prob_k << " is not divisible by thread_k = " << thread_k;
  } else {
    // Auto config
    exec_cfg = determine_exec_config<scalar_t, has_zp>(
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

  TVM_FFI_ICHECK(is_valid_config(
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
      max_shared_mem))
      << "Invalid thread config: thread_m_blocks = " << thread_m_blocks << ", thread_k = " << thread_k
      << ", thread_n = " << thread_n << ", num_threads = " << num_threads << " for MKN = [" << prob_m << ", " << prob_k
      << ", " << prob_n << "] and num_bits = " << num_bits << ", group_size = " << group_size
      << ", has_act_order = " << has_act_order << ", is_k_full = " << is_k_full << ", has_zp = " << has_zp
      << ", is_zp_float = " << is_zp_float << ", max_shared_mem = " << max_shared_mem;
}

template <int moe_block_size, bool has_zp, bool is_zp_float>
void moe_wna16_marlin_gemm(
    const tvm::ffi::TensorView& a,
    const tvm::ffi::TensorView& c,
    const tvm::ffi::TensorView& b_q_weight,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& b_bias_or_none,
    const tvm::ffi::TensorView& b_scales,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& global_scale_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& b_zeros_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& g_idx_or_none,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& perm_or_none,
    const tvm::ffi::TensorView& workspace,
    const tvm::ffi::TensorView& sorted_token_ids,
    const tvm::ffi::TensorView& expert_ids,
    const tvm::ffi::TensorView& num_tokens_past_padded,
    const tvm::ffi::TensorView& topk_weights,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce) {
  const DLDataType b_q_type = b_q_weight.dtype();
  int pack_factor = 32 / b_q_type.bits;

  if constexpr (moe_block_size != 8) {
    static_assert(moe_block_size % 16 == 0, "unsupported moe_block_size");
    static_assert(moe_block_size >= 16 && moe_block_size <= 64, "unsupported moe_block_size");
  }

  // Verify A
  TVM_FFI_ICHECK(a.size(0) == size_m) << "Shape mismatch: a.size(0) = " << a.size(0) << ", size_m = " << size_m;
  TVM_FFI_ICHECK(a.size(1) == size_k) << "Shape mismatch: a.size(1) = " << a.size(1) << ", size_k = " << size_k;

  // Verify B
  TVM_FFI_ICHECK(size_k % MARLIN_NAMESPACE_NAME::tile_size == 0)
      << "size_k = " << size_k << " is not divisible by tile_size = " << MARLIN_NAMESPACE_NAME::tile_size;
  TVM_FFI_ICHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(1))
      << "Shape mismatch: b_q_weight.size(1) = " << b_q_weight.size(1) << ", size_k = " << size_k
      << ", tile_size = " << MARLIN_NAMESPACE_NAME::tile_size;
  TVM_FFI_ICHECK(b_q_weight.size(2) % MARLIN_NAMESPACE_NAME::tile_size == 0)
      << "b_q_weight.size(2) = " << b_q_weight.size(2)
      << " is not divisible by tile_size = " << MARLIN_NAMESPACE_NAME::tile_size;
  int actual_size_n = (b_q_weight.size(2) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  TVM_FFI_ICHECK(size_n == actual_size_n) << "size_n = " << size_n << ", actual_size_n = " << actual_size_n;

  // Verify device and strides
  TVM_FFI_ICHECK(a.device().device_type == kDLCUDA) << "A is not on GPU";
  TVM_FFI_ICHECK(a.is_contiguous()) << "A is not contiguous";

  TVM_FFI_ICHECK(b_q_weight.device().device_type == kDLCUDA) << "b_q_weight is not on GPU";
  TVM_FFI_ICHECK(b_q_weight.is_contiguous()) << "b_q_weight is not contiguous";

  TVM_FFI_ICHECK(b_scales.device().device_type == kDLCUDA) << "b_scales is not on GPU";
  TVM_FFI_ICHECK(b_scales.is_contiguous()) << "b_scales is not contiguous";

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.device().device_id));

  // Verify C
  TVM_FFI_ICHECK(c.device().device_type == kDLCUDA) << "c is not on GPU";
  TVM_FFI_ICHECK(c.is_contiguous()) << "c is not contiguous";
  TVM_FFI_ICHECK(c.size(0) == size_m * top_k)
      << "Shape mismatch: c.size(0) = " << c.size(0) << ", size_m * topk = " << size_m * top_k;
  TVM_FFI_ICHECK(c.size(1) == size_n) << "Shape mismatch: c.size(1) = " << c.size(1) << ", size_n = " << size_n;

  const auto empty_tensor = tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {0}, dl_float32, c.device());

  // Alloc C tmp buffer that is going to be used for the global reduce
  tvm::ffi::TensorView c_tmp = empty_tensor;
  if (use_fp32_reduce && !use_atomic_add) {
    // max num of threadblocks is sms * 4
    long max_c_tmp_size = min(
        (long)size_n * sorted_token_ids.size(0), (long)sms * 4 * moe_block_size * MARLIN_NAMESPACE_NAME::max_thread_n);
    if (moe_block_size == 8) max_c_tmp_size *= 2;
    const auto c_tmp_tensor =
        tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {max_c_tmp_size}, dl_float32, c.device());
    c_tmp = c_tmp_tensor;
  }

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.ndim();
  TVM_FFI_ICHECK(rank == 3) << "b_scales rank = " << rank << " is not 3";
  TVM_FFI_ICHECK(b_scales.size(2) == size_n)
      << "b_scales dim 2 = " << b_scales.size(2) << " is not size_n = " << size_n;
  num_groups = b_scales.size(1);

  tvm::ffi::TensorView g_idx = empty_tensor, perm = empty_tensor, a_tmp = empty_tensor;

  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();

    TVM_FFI_ICHECK(g_idx.device().device_type == kDLCUDA) << "g_idx is not on GPU";
    TVM_FFI_ICHECK(g_idx.is_contiguous()) << "g_idx is not contiguous";
    TVM_FFI_ICHECK(perm.device().device_type == kDLCUDA) << "perm is not on GPU";
    TVM_FFI_ICHECK(perm.is_contiguous()) << "perm is not contiguous";

    TVM_FFI_ICHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) || (g_idx.size(-1) == size_k && perm.size(-1) == size_k))
        << "Unexpected g_idx.size(-1) = " << g_idx.size(-1) << " and perm.size(-1) = " << perm.size(-1)
        << ", where size_k = " << size_k;
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    const auto a_tensor =
        tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {size_m * top_k, size_k}, a.dtype(), a.device());
    a_tmp = a_tensor;
    if (is_k_full) {
      TVM_FFI_ICHECK(num_groups > 1) << "For act_order, num_groups must be > 1";
      TVM_FFI_ICHECK(size_k % num_groups == 0)
          << "size_k = " << size_k << " is not divisible by num_groups = " << num_groups;
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }
  } else {
    if (num_groups > 1) {
      TVM_FFI_ICHECK(size_k % num_groups == 0)
          << "size_k = " << size_k << " is not divisible by num_groups = " << num_groups;
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  tvm::ffi::TensorView global_scale = empty_tensor;
  if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    TVM_FFI_ICHECK(b_q_type.code == kDLFloat4_e2m1fn && group_size == 16)
        << "global_scale can only be used for nvfp4 format.";
  } else {
    TVM_FFI_ICHECK(!(b_q_type.code == kDLFloat4_e2m1fn && group_size == 16))
        << "the global_scale parameter must be passed for nvfp4 format.";
  }

  bool has_bias = b_bias_or_none.has_value();
  tvm::ffi::TensorView b_bias = empty_tensor;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    TVM_FFI_ICHECK(b_bias.device().device_type == kDLCUDA) << "b_bias is not on GPU";
    TVM_FFI_ICHECK(b_bias.is_contiguous()) << "b_bias is not contiguous";
    TVM_FFI_ICHECK(b_bias.size(1) == size_n) << "b_bias.size(1) != size_n";
    TVM_FFI_ICHECK(b_bias.stride(1) == 1) << "b_bias.stride(1) != 1";
  }

  tvm::ffi::TensorView b_zeros = empty_tensor;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TVM_FFI_ICHECK(b_zeros.device().device_type == kDLCUDA) << "b_zeros is not on GPU";
    TVM_FFI_ICHECK(b_zeros.is_contiguous()) << "b_zeros is not contiguous";
  }
  TVM_FFI_ICHECK(has_zp == b_zeros.size(-1) > 0)
      << "has_zp = " << has_zp << " is not equal to b_zeros.size(-1) = " << b_zeros.size(-1);
  if (has_zp) {
    TVM_FFI_ICHECK(b_q_type.code == kDLUInt && (b_q_type.bits == 4 || b_q_type.bits == 8))
        << "b_q_type must be u4 or u8 when has_zp = True. Got = " << b_q_type;
  } else {
    TVM_FFI_ICHECK(
        (b_q_type.code == kDLUInt && (b_q_type.bits == 4 || b_q_type.bits == 8)) || b_q_type.code == kDLFloat4_e2m1fn)
        << "b_q_type must be uint4b8, uint8b128 or float4_e2m1f when has_zp = False. Got = " << b_q_type;
  }

  if (has_zp && is_zp_float) {
    TVM_FFI_ICHECK(a.dtype().code == kDLFloat && a.dtype().bits == 16)
        << "Computation type must be float16 (half) when using float zero points.";
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.ndim();
    TVM_FFI_ICHECK(rank == 3) << "b_zeros rank = " << rank << " is not 3";
    if (is_zp_float) {
      TVM_FFI_ICHECK(b_zeros.size(2) == size_n)
          << "b_zeros dim 2 = " << b_zeros.size(2) << " is not size_n = " << size_n;
      TVM_FFI_ICHECK(num_groups == b_zeros.size(1))
          << "b_zeros dim 1 = " << b_zeros.size(1) << " is not num_groups = " << num_groups;
      TVM_FFI_ICHECK(num_groups != -1) << "num_groups must be != -1";
    } else {
      TVM_FFI_ICHECK(b_zeros.size(1) == num_groups)
          << "b_zeros dim 1 = " << b_zeros.size(1) << " is not num_groups = " << num_groups;
      TVM_FFI_ICHECK(b_zeros.size(2) == size_n / pack_factor)
          << "b_zeros dim 2 = " << b_zeros.size(2) << " is not size_n / pack_factor = " << size_n / pack_factor;
    }
  }

  // Verify workspace size
  TVM_FFI_ICHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0)
      << "size_n = " << size_n << " is not divisible by min_thread_n = " << MARLIN_NAMESPACE_NAME::min_thread_n;

  int max_n_tiles = size_n / MARLIN_NAMESPACE_NAME::min_thread_n;
  int min_workspace_size = min(max_n_tiles * (int)(sorted_token_ids.size(0) / moe_block_size), sms * 4);
  TVM_FFI_ICHECK(workspace.numel() >= min_workspace_size)
      << "workspace.numel = " << workspace.numel() << " is below min_workspace_size = " << min_workspace_size;

  int dev = a.device().device_id;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FLOAT16(a.dtype(), scalar_t, [&]() {
    if (b_q_type.code == kDLFloat4_e2m1fn) {
      TVM_FFI_ICHECK(group_size == 16 || group_size == 32) << "group_size must be 16 or 32";
    }
    void* scales_ptr = b_scales.data_ptr();

    marlin_mm<scalar_t, moe_block_size, has_zp, is_zp_float>(
        a.data_ptr(),
        b_q_weight.data_ptr(),
        c.data_ptr(),
        c_tmp.data_ptr(),
        b_bias.data_ptr(),
        scales_ptr,
        global_scale.data_ptr(),
        b_zeros.data_ptr(),
        g_idx.data_ptr(),
        perm.data_ptr(),
        a_tmp.data_ptr(),
        sorted_token_ids.data_ptr(),
        expert_ids.data_ptr(),
        num_tokens_past_padded.data_ptr(),
        topk_weights.data_ptr(),
        top_k,
        mul_topk_weights,
        is_ep,
        size_m,
        size_n,
        size_k,
        workspace.data_ptr(),
        b_q_type,
        has_bias,
        has_act_order,
        is_k_full,
        num_groups,
        group_size,
        dev,
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(a.device().device_type, dev)),
        thread_k,
        thread_n,
        sms,
        use_atomic_add,
        use_fp32_reduce);
  });
}

}  // namespace MARLIN_NAMESPACE_NAME
