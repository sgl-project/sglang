#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/runtime.cuh>  // For host::runtime::get_sm_count
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace {

struct PlanWriteExpectedTokensParams {
  // Inputs.
  const int64_t* __restrict__ req_pool_indices;           // [bs_padded] int64
  const int64_t* __restrict__ prefix_lens;                // [bs_padded] int64
  const int64_t* __restrict__ write_offsets;              // [bs_padded + 1] int64 cumulative prefix sum
  const int32_t* __restrict__ expected_token_pool;        // [max_reqs, max_context_len] int32, may be nullptr
  const int32_t* __restrict__ expected_token_valid_lens;  // [max_reqs] int32, may be nullptr
  // Outputs.
  int64_t* __restrict__ out_expected_input_tokens;  // [write_entry_capacity] int64
  // Sizes / strides.
  int32_t bs_padded;
  int64_t write_entry_capacity;
  int64_t pool_stride0;  // expected_token_pool row stride (in int32 elements)
  int64_t pool_max_context_len;
  int32_t slot_token_offset;  // 0 for target pools; 1 for EAGLE draft pools
};

// Binary search for the largest req_id such that write_offsets[req_id] <= tid. Mirror of the
// find_req_id in canary_plan_entries.cuh. Pre-condition: tid is strictly less than
// write_offsets[bs_padded] = total_write; bs_padded >= 1; write_offsets[0] = 0.
SGL_DEVICE int32_t find_req_id_for_write(const int64_t* __restrict__ write_offsets, int32_t bs_padded, int64_t tid) {
  int32_t lo = 0;
  int32_t hi = bs_padded;  // exclusive upper bound; write_offsets[hi] > tid
  while (hi - lo > 1) {
    const int32_t mid = (lo + hi) >> 1;
    if (write_offsets[mid] <= tid) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Persistent grid; one thread = one write entry (with stride). Template parameter HAS_TOKEN_POOL
// switches the pool-gather path off entirely when the validator is disabled.
template <bool HAS_TOKEN_POOL>
__global__ void
plan_write_expected_tokens_persistent_kernel(const PlanWriteExpectedTokensParams __grid_constant__ params) {
  const int64_t total_write = params.write_offsets[params.bs_padded];
  if (total_write <= 0) {
    return;
  }

  // Contract: caller ensures write_entry_capacity >= total_write. Trap on contract break.
  if (total_write > params.write_entry_capacity) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf(
          "kv-canary plan_write_expected_tokens: total_write=%lld exceeds "
          "write_entry_capacity=%lld (capacity contract broken)\n",
          static_cast<long long>(total_write),
          static_cast<long long>(params.write_entry_capacity));
    }
    __trap();
  }

  const int64_t tid_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  const int32_t slot_token_offset = params.slot_token_offset;

  for (int64_t tid = tid_start; tid < total_write; tid += stride) {
    int64_t expected_token = -1;

    if constexpr (HAS_TOKEN_POOL) {
      // 1) Find the owning req via binary search over write_offsets.
      const int32_t req_id = find_req_id_for_write(params.write_offsets, params.bs_padded, tid);
      const int64_t req_start = params.write_offsets[req_id];
      const int64_t entry_idx = tid - req_start;

      // 2) write_pos covers [prefix_lens[r], prefix_lens[r] + write_count). Apply slot_token_offset
      // to land on the source-of-truth row.
      const int64_t rp = params.req_pool_indices[req_id];
      const int64_t prefix_len = params.prefix_lens[req_id];
      const int64_t write_pos = prefix_len + entry_idx;
      const int64_t sot_pos = write_pos + static_cast<int64_t>(slot_token_offset);

      // 3) Gather from pool if in range; else fall through to the -1 sentinel which makes the WRITE
      // kernel skip the token-mismatch check.
      const int64_t valid_len = static_cast<int64_t>(params.expected_token_valid_lens[rp]);
      if (sot_pos >= 0 && sot_pos < valid_len && sot_pos < params.pool_max_context_len) {
        const int32_t token = params.expected_token_pool[rp * params.pool_stride0 + sot_pos];
        expected_token = static_cast<int64_t>(token);
      }
    }

    // 4) Scatter. Every live tid writes a value (pool token or sentinel).
    params.out_expected_input_tokens[tid] = expected_token;
  }
}

struct PlanWriteExpectedTokensKernel {
  static constexpr int kBlockSize = 128;
  static constexpr int kBlocksPerSm = 8;

  static auto get_num_sms(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView prefix_lens,
      const tvm::ffi::TensorView write_offsets,
      const tvm::ffi::Optional<tvm::ffi::TensorView> expected_token_pool,
      const tvm::ffi::Optional<tvm::ffi::TensorView> expected_token_valid_lens,
      const tvm::ffi::TensorView out_expected_input_tokens,
      int32_t slot_token_offset) {
    using namespace host;

    SymbolicSize Nbs = {"bs_padded"};
    SymbolicSize Nwrite_offsets = {"write_offsets_len"};
    SymbolicSize Ncap = {"write_entry_capacity"};
    SymbolicSize Nmax_reqs = {"pool_max_reqs"};
    SymbolicSize Nmax_context_len = {"pool_max_context_len"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    TensorMatcher({Nbs})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(req_pool_indices)
        .verify(prefix_lens);
    TensorMatcher({Nwrite_offsets})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(write_offsets);
    TensorMatcher({Ncap})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(out_expected_input_tokens);

    const bool has_token_pool = expected_token_pool.has_value() && expected_token_valid_lens.has_value();
    RuntimeCheck(
        expected_token_pool.has_value() == expected_token_valid_lens.has_value(),
        "plan_write_expected_tokens: expected_token_pool and expected_token_valid_lens must "
        "both be present or both absent");
    if (has_token_pool) {
      TensorMatcher({Nmax_reqs, Nmax_context_len})  //
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device_)
          .verify(expected_token_pool.value());
      TensorMatcher({Nmax_reqs})  //
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device_)
          .verify(expected_token_valid_lens.value());
    }
    RuntimeCheck(
        Nwrite_offsets.unwrap() >= Nbs.unwrap() + 1,
        "plan_write_expected_tokens: write_offsets length must be >= bs_padded + 1");

    const int64_t bs_padded = Nbs.unwrap();
    if (bs_padded <= 0) {
      return;
    }

    const int32_t* pool_ptr =
        has_token_pool ? static_cast<const int32_t*>(expected_token_pool.value().data_ptr()) : nullptr;
    const int32_t* valid_lens_ptr =
        has_token_pool ? static_cast<const int32_t*>(expected_token_valid_lens.value().data_ptr()) : nullptr;
    const int64_t pool_stride0 = has_token_pool ? static_cast<int64_t>(Nmax_context_len.unwrap()) : 0;
    const int64_t pool_max_context_len = pool_stride0;

    const PlanWriteExpectedTokensParams params = PlanWriteExpectedTokensParams{
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr()),
        .write_offsets = static_cast<const int64_t*>(write_offsets.data_ptr()),
        .expected_token_pool = pool_ptr,
        .expected_token_valid_lens = valid_lens_ptr,
        .out_expected_input_tokens = static_cast<int64_t*>(out_expected_input_tokens.data_ptr()),
        .bs_padded = static_cast<int32_t>(bs_padded),
        .write_entry_capacity = static_cast<int64_t>(Ncap.unwrap()),
        .pool_stride0 = pool_stride0,
        .pool_max_context_len = pool_max_context_len,
        .slot_token_offset = slot_token_offset,
    };

    const DLDevice device = device_.unwrap();
    const int num_sms = get_num_sms(device);
    const int num_blocks = num_sms * kBlocksPerSm;

    const dim3 grid(num_blocks);
    const dim3 block(kBlockSize);

    if (has_token_pool) {
      LaunchKernel(grid, block, device)(plan_write_expected_tokens_persistent_kernel<true>, params);
    } else {
      LaunchKernel(grid, block, device)(plan_write_expected_tokens_persistent_kernel<false>, params);
    }
  }
};

}  // namespace
