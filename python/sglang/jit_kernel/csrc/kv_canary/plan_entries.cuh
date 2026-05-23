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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

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
__device__ __forceinline__ int32_t
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
__device__ __forceinline__ int64_t swa_translate(const int64_t* __restrict__ lut, int64_t lut_len, int64_t raw_slot) {
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

// Type-checking ptr extractor (mirrors the elementwise/fused_metadata_copy.cuh helpers).
template <typename T>
inline const T* unwrap_data_ptr(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have the expected dtype");
  }
  return static_cast<const T*>(tensor.data_ptr());
}

template <typename T>
inline T* unwrap_data_ptr_mut(const tvm::ffi::TensorView& tensor, const char* name) {
  using namespace host;
  if (tensor.data_ptr()) {
    RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have the expected dtype");
  }
  return static_cast<T*>(tensor.data_ptr());
}

template <typename T>
inline const T*
unwrap_optional_data_ptr(const tvm::ffi::Optional<tvm::ffi::TensorView>& optional_tensor, const char* name) {
  using namespace host;
  if (!optional_tensor.has_value()) {
    return nullptr;
  }
  const auto& tensor = optional_tensor.value();
  RuntimeCheck(is_type<T>(tensor.dtype()), "Tensor ", name, " must have the expected dtype");
  return static_cast<const T*>(tensor.data_ptr());
}

// JIT-callable host launcher. Selects the templated kernel via the HAS_SWA_LUT bool. The persistent grid
// is sized to ``num_sms * BLOCKS_PER_SM`` blocks of BLOCK_SIZE threads; we read ``num_sms`` once per
// launch from cudaGetDeviceProperties through TVM FFI's device context, but for cuda-graph safety we
// avoid querying the device every call -- ``num_sms`` is cached in a function-local static and only
// looked up once per (device_id) the launcher sees. For the H200 we expect 132 SMs which yields
// 132 * 8 = 1056 blocks of 128 threads = 135,168 persistent threads.
//
// Tuning rationale (per the plan note):
//   - BLOCK_SIZE = 128: small enough to keep occupancy headroom on SM89/90, large enough that the binary
//     search fits in warp-uniform branches (within a warp, contiguous tids usually share a req_id).
//   - BLOCKS_PER_SM = 8: target occupancy of 8 ctas per SM with 128 threads each = 1024 active threads
//     per SM (well within the H200 / Hopper 2048-thread-per-SM limit). The actual runtime occupancy may
//     differ; this is the "embarrassingly parallel cap" rather than a hard tuning point.
struct PlanEntriesKernel {
  static constexpr int kBlockSize = 128;
  static constexpr int kBlocksPerSm = 8;

  static int get_num_sms(DLDevice device) {
    // Lazy + cached per call site; cudaGetDeviceProperties is fast enough but we still avoid hitting it
    // on every launch by storing the (device_id, multiProcessorCount) result.
    static int cached_device_id = -1;
    static int cached_num_sms = 0;
    if (device.device_id != cached_device_id) {
      int num_sms = 0;
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.device_id);
      if (num_sms <= 0) {
        num_sms = 80;  // sane fallback (V100 / A30 baseline)
      }
      cached_device_id = device.device_id;
      cached_num_sms = num_sms;
    }
    return cached_num_sms;
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

    const int64_t* lut_ptr = unwrap_optional_data_ptr<int64_t>(full_to_swa_index_mapping, "full_to_swa_lut");
    const bool has_swa_lut = (lut_ptr != nullptr);
    int64_t lut_len = 0;
    if (has_swa_lut) {
      lut_len = static_cast<int64_t>(full_to_swa_index_mapping.value().shape()[0]);
    }

    const PlanEntriesParams params = PlanEntriesParams{
        .req_pool_indices = unwrap_data_ptr<int64_t>(req_pool_indices, "req_pool_indices"),
        .prefix_lens = unwrap_data_ptr<int64_t>(prefix_lens, "prefix_lens"),
        .req_to_token = unwrap_data_ptr<int32_t>(req_to_token, "req_to_token"),
        .full_to_swa_lut = lut_ptr,
        .verify_offsets_scratch = unwrap_data_ptr<int64_t>(verify_offsets_scratch, "verify_offsets_scratch"),
        .out_verify_slot_indices = unwrap_data_ptr_mut<int64_t>(out_verify_slot_indices, "out_verify_slot_indices"),
        .out_verify_positions = unwrap_data_ptr_mut<int64_t>(out_verify_positions, "out_verify_positions"),
        .out_verify_prev_slot_indices =
            unwrap_data_ptr_mut<int64_t>(out_verify_prev_slot_indices, "out_verify_prev_slot_indices"),
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
