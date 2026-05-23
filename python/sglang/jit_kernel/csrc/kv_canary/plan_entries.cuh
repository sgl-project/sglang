// Persistent CUDA kernel for the kv_canary plan-entries step.
//
//   - 1 thread = 1 verify entry (embarrassingly parallel; no atomics / sync / shmem).
//   - 1-D grid sized to num_sms * blocks_per_sm; each thread strides over total_verify entries via a
//     persistent loop. ``total_verify`` is read on-device from verify_offsets_scratch[bs_padded] so the
//     grid is static and the kernel is cuda-graph friendly.
//   - Per-thread: binary-search verify_offsets_scratch (len bs_padded+1 <= 4097) to find req_id, then
//     a few global loads + 3 scatter stores.
//
// Byte-equality contract: the (slot, position, prev_slot) triples this kernel writes must match the
// python reference in ``kv_canary/plan_ref.py`` row-for-row.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/runtime.cuh>  // For host::runtime::get_sm_count
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_runtime.h>

namespace canary_plan_entries {

// Per-launch device-side params struct. Passed via __grid_constant__.
struct PlanEntriesParams {
  // Inputs.
  const int64_t* __restrict__ req_pool_indices;        // [bs_padded] int64
  const int64_t* __restrict__ prefix_lens;             // [bs_padded] int64
  const int32_t* __restrict__ req_to_token;            // [max_reqs, max_seq_len] int32
  const int64_t* __restrict__ full_to_swa_lut;         // [lut_len] int64, may be nullptr when !HAS_SWA_LUT
  const int64_t* __restrict__ verify_offsets_scratch;  // [bs_padded + 1] int64 (cumulative prefix sum)
  // Outputs.
  int64_t* __restrict__ out_verify_slot_indices;       // [verify_capacity] int64
  int64_t* __restrict__ out_verify_positions;          // [verify_capacity] int64
  int64_t* __restrict__ out_verify_prev_slot_indices;  // [verify_capacity] int64
  // Sizes / strides.
  int32_t bs_padded;
  int64_t req_to_token_stride0;
  int32_t swa_window_size;
};

// Binary search for the largest req_id such that verify_offsets[req_id] <= tid. Pre-condition: tid is
// strictly less than verify_offsets[bs_padded] = total_verify; bs_padded >= 1; verify_offsets[0] = 0.
SGL_DEVICE int32_t
find_req_id(const int64_t* __restrict__ verify_offsets, int32_t bs_padded, int64_t tid) {
  int32_t lo = 0;
  int32_t hi = bs_padded;  // exclusive upper bound; verify_offsets[hi] > tid
  while (hi - lo > 1) {
    const int32_t mid = (lo + hi) >> 1;
    if (verify_offsets[mid] <= tid) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Translate raw slot value via the SWA LUT. Sentinel passthrough (-1 stays -1). Clamp slot to
// ``lut_len - 1`` defensively; in practice the caller never produces out-of-range slots.
SGL_DEVICE int64_t swa_translate(const int64_t* __restrict__ lut, int64_t lut_len, int64_t raw_slot) {
  if (raw_slot < 0) {
    return raw_slot;
  }
  int64_t safe = raw_slot;
  if (lut_len > 0 && safe >= lut_len) {
    safe = lut_len - 1;
  }
  return lut[safe];
}

// Persistent grid; one thread = one verify entry (with stride). Template parameter HAS_SWA_LUT switches
// the SWA-translate path off entirely in the FULL pool variant.
template <bool HAS_SWA_LUT>
__global__ void plan_entries_persistent_kernel(
    const PlanEntriesParams __grid_constant__ params,
    int64_t lut_len  // only meaningful when HAS_SWA_LUT
) {
  const int64_t total_verify = params.verify_offsets_scratch[params.bs_padded];
  if (total_verify <= 0) {
    return;
  }

  const int64_t tid_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  const int32_t swa_window = params.swa_window_size;
  const int64_t req_to_token_stride0 = params.req_to_token_stride0;

  for (int64_t tid = tid_start; tid < total_verify; tid += stride) {
    // 1) Find the owning req via binary search over verify_offsets.
    const int32_t req_id = find_req_id(params.verify_offsets_scratch, params.bs_padded, tid);
    const int64_t req_start = params.verify_offsets_scratch[req_id];
    const int64_t entry_idx = tid - req_start;

    // 2) Load per-req metadata. Padding rows have verify_count=0 in the offsets prefix-sum, so a tid in
    // the live range can never land on a padding req; no explicit padding-row check needed.
    const int64_t rp = params.req_pool_indices[req_id];
    const int64_t prefix_len = params.prefix_lens[req_id];
    const int64_t window_start = (swa_window > 0) ? (prefix_len - swa_window > 0 ? prefix_len - swa_window : 0) : 0;
    const int64_t position = window_start + entry_idx;

    // 3) Gather slot + prev_slot via req_to_token.
    const int64_t row_base = rp * req_to_token_stride0;
    const int32_t slot_raw = params.req_to_token[row_base + position];
    int64_t slot;
    if constexpr (HAS_SWA_LUT) {
      slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(slot_raw));
    } else {
      slot = static_cast<int64_t>(slot_raw);
    }

    int64_t prev_slot;
    if (position > 0) {
      const int32_t prev_raw = params.req_to_token[row_base + position - 1];
      if constexpr (HAS_SWA_LUT) {
        prev_slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(prev_raw));
      } else {
        prev_slot = static_cast<int64_t>(prev_raw);
      }
    } else {
      prev_slot = -1;
    }

    // 4) Scatter. out_idx == tid since verify_offsets[req_id] + entry_idx == tid by construction.
    params.out_verify_slot_indices[tid] = slot;
    params.out_verify_positions[tid] = position;
    params.out_verify_prev_slot_indices[tid] = prev_slot;
  }
}

// JIT-callable host launcher. Selects the templated kernel via the HAS_SWA_LUT bool. The persistent grid
// is sized to ``num_sms * kBlocksPerSm`` blocks of ``kBlockSize`` threads. For the H200 we expect 132
// SMs which yields 132 * 8 = 1056 blocks of 128 threads = 135,168 persistent threads.
struct PlanEntriesKernel {
  static constexpr int kBlockSize = 128;
  static constexpr int kBlocksPerSm = 8;

  static auto get_num_sms(DLDevice device) {
    static const auto kNumSM = host::runtime::get_sm_count(device.device_id);
    return kNumSM;
  }

  static void
  run(const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView prefix_lens,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::Optional<tvm::ffi::TensorView> full_to_swa_index_mapping,
      const tvm::ffi::TensorView verify_offsets_scratch,
      const tvm::ffi::TensorView out_verify_slot_indices,
      const tvm::ffi::TensorView out_verify_positions,
      const tvm::ffi::TensorView out_verify_prev_slot_indices,
      int64_t req_to_token_stride0,
      int64_t bs_padded,
      int32_t swa_window_size) {
    using namespace host;

    const bool has_swa_lut = full_to_swa_index_mapping.has_value();
    const int64_t* lut_ptr =
        has_swa_lut ? static_cast<const int64_t*>(full_to_swa_index_mapping.value().data_ptr()) : nullptr;
    const int64_t lut_len = has_swa_lut ? static_cast<int64_t>(full_to_swa_index_mapping.value().shape()[0]) : 0;

    const PlanEntriesParams params = PlanEntriesParams{
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa_lut = lut_ptr,
        .verify_offsets_scratch = static_cast<const int64_t*>(verify_offsets_scratch.data_ptr()),
        .out_verify_slot_indices = static_cast<int64_t*>(out_verify_slot_indices.data_ptr()),
        .out_verify_positions = static_cast<int64_t*>(out_verify_positions.data_ptr()),
        .out_verify_prev_slot_indices = static_cast<int64_t*>(out_verify_prev_slot_indices.data_ptr()),
        .bs_padded = static_cast<int32_t>(bs_padded),
        .req_to_token_stride0 = req_to_token_stride0,
        .swa_window_size = static_cast<int32_t>(swa_window_size),
    };

    if (bs_padded <= 0) {
      return;
    }

    const DLDevice device = req_pool_indices.device();
    const int num_sms = get_num_sms(device);
    const int num_blocks = num_sms * kBlocksPerSm;

    const dim3 grid(num_blocks);
    const dim3 block(kBlockSize);

    if (has_swa_lut) {
      LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<true>, params, lut_len);
    } else {
      LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<false>, params, lut_len);
    }
  }
};

}  // namespace canary_plan_entries
