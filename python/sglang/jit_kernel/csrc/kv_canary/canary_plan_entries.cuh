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

struct PlanEntriesParams {
  // Inputs.
  const int64_t* __restrict__ req_pool_indices;        // [bs_padded] int64
  const int64_t* __restrict__ prefix_lens;             // [bs_padded] int64
  const int32_t* __restrict__ req_to_token;            // [max_reqs, max_seq_len] int32
  const int64_t* __restrict__ full_to_swa_lut;         // [lut_len] int64, may be nullptr when !HAS_SWA_LUT
  const int64_t* __restrict__ verify_offsets_scratch;  // [bs_padded + 1] int64 (cumulative prefix sum)
  const int32_t* __restrict__ verify_enable;           // [1] int32 — 0 ⇒ skip scatter entirely
  // Source-of-truth token pool for verify-time SOT check. May be nullptr when !HAS_TOKEN_POOL.
  const int32_t* __restrict__ expected_token_pool;     // [pool_rows, pool_cols] int32
  const int32_t* __restrict__ expected_token_valid_lens;  // [pool_rows] int32
  // Outputs.
  int64_t* __restrict__ out_verify_slot_indices;       // [verify_capacity] int64
  int64_t* __restrict__ out_verify_positions;          // [verify_capacity] int64
  int64_t* __restrict__ out_verify_prev_slot_indices;  // [verify_capacity] int64
  int64_t* __restrict__ out_verify_expected_tokens;    // [verify_capacity] int64
  // Sizes / strides.
  int32_t bs_padded;
  int64_t verify_capacity;  // out_verify_*[verify_capacity]; scatter is clamped to this length.
  int64_t req_to_token_stride0;
  int64_t expected_token_pool_stride0;  // only meaningful when HAS_TOKEN_POOL
  int32_t swa_window_size;
  int32_t slot_token_offset;            // logical-position offset (0 target, 1 EAGLE draft)
};

// Binary search for the largest req_id such that verify_offsets[req_id] <= tid. Pre-condition: tid is
// strictly less than verify_offsets[bs_padded] = total_verify; bs_padded >= 1; verify_offsets[0] = 0.
SGL_DEVICE int32_t find_req_id(const int64_t* __restrict__ verify_offsets, int32_t bs_padded, int64_t tid) {
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

// Persistent grid; one thread = one verify entry (with stride). Template parameters switch the SWA-translate
// path off entirely in the FULL pool variant and the SOT-token gather off when no req-truth pool was wired.
template <bool HAS_SWA_LUT, bool HAS_TOKEN_POOL>
__global__ void plan_entries_persistent_kernel(
    const PlanEntriesParams __grid_constant__ params,
    int64_t lut_len  // only meaningful when HAS_SWA_LUT
) {
  const int64_t total_verify = params.verify_offsets_scratch[params.bs_padded];
  if (total_verify <= 0) {
    return;
  }

  if (*params.verify_enable == 0) {
    return;
  }

  // Contract: offsets kernel clears verify_enable iff total_verify > verify_capacity, so reaching
  // here implies total_verify <= verify_capacity. Trap on contract break (cassert is NDEBUG-gated).
  if (total_verify > params.verify_capacity) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf(
          "kv-canary plan_entries: total_verify=%lld exceeds verify_capacity=%lld with "
          "verify_enable=1 (offsets/entries contract broken)\n",
          static_cast<long long>(total_verify),
          static_cast<long long>(params.verify_capacity));
    }
    __trap();
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
    const int64_t out_position = window_start + entry_idx;

    // 3) Gather slot + prev_slot via req_to_token.
    const int64_t row_base = rp * req_to_token_stride0;
    const int32_t slot_raw = params.req_to_token[row_base + out_position];
    int64_t out_slot;
    if constexpr (HAS_SWA_LUT) {
      out_slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(slot_raw));
    } else {
      out_slot = static_cast<int64_t>(slot_raw);
    }

    int64_t out_prev_slot;
    if (out_position > 0) {
      const int32_t prev_raw = params.req_to_token[row_base + out_position - 1];
      if constexpr (HAS_SWA_LUT) {
        out_prev_slot = swa_translate(params.full_to_swa_lut, lut_len, static_cast<int64_t>(prev_raw));
      } else {
        out_prev_slot = static_cast<int64_t>(prev_raw);
      }
    } else {
      out_prev_slot = -1;
    }

    // 4) Gather the source-of-truth token at position+slot_token_offset for the verify-time check.
    // ``-1`` sentinel means "skip" — covers speculative draft decode slots and EAGLE draft rotation
    // bonus tail (both fall past the committed valid_lens).
    int64_t out_expected_token = -1;
    if constexpr (HAS_TOKEN_POOL) {
      const int64_t sot_position = out_position + static_cast<int64_t>(params.slot_token_offset);
      if (sot_position >= 0) {
        const int32_t valid_len = params.expected_token_valid_lens[rp];
        if (sot_position < static_cast<int64_t>(valid_len)) {
          const int64_t pool_idx = rp * params.expected_token_pool_stride0 + sot_position;
          out_expected_token = static_cast<int64_t>(params.expected_token_pool[pool_idx]);
        }
      }
    }

    // 5) Scatter. out_idx == tid since verify_offsets[req_id] + entry_idx == tid by construction.
    params.out_verify_slot_indices[tid] = out_slot;
    params.out_verify_positions[tid] = out_position;
    params.out_verify_prev_slot_indices[tid] = out_prev_slot;
    params.out_verify_expected_tokens[tid] = out_expected_token;
  }
}

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
      const tvm::ffi::TensorView verify_enable,
      const tvm::ffi::Optional<tvm::ffi::TensorView> expected_token_pool,
      const tvm::ffi::Optional<tvm::ffi::TensorView> expected_token_valid_lens,
      const tvm::ffi::TensorView out_verify_slot_indices,
      const tvm::ffi::TensorView out_verify_positions,
      const tvm::ffi::TensorView out_verify_prev_slot_indices,
      const tvm::ffi::TensorView out_verify_expected_tokens,
      int32_t swa_window_size,
      int32_t slot_token_offset) {
    using namespace host;

    SymbolicSize Nbs = {"bs_padded"};
    SymbolicSize Nscratch = {"verify_offsets_scratch_len"};
    SymbolicSize Ncap = {"verify_capacity"};
    SymbolicSize Nmax_reqs = {"max_reqs"};
    SymbolicSize Nmax_seq_len = {"max_seq_len"};
    SymbolicSize Nlut = {"lut_len"};
    SymbolicSize Npool_rows = {"expected_token_pool_rows"};
    SymbolicSize Npool_cols = {"expected_token_pool_cols"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    TensorMatcher({Nbs})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(req_pool_indices)
        .verify(prefix_lens);
    TensorMatcher({Nscratch})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(verify_offsets_scratch);
    TensorMatcher({1})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(verify_enable);
    TensorMatcher({Ncap})  //
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(out_verify_slot_indices)
        .verify(out_verify_positions)
        .verify(out_verify_prev_slot_indices)
        .verify(out_verify_expected_tokens);
    TensorMatcher({Nmax_reqs, Nmax_seq_len})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(req_to_token);
    const bool has_swa_lut = full_to_swa_index_mapping.has_value();
    if (has_swa_lut) {
      TensorMatcher({Nlut})  //
          .with_dtype<int64_t>()
          .with_device<kDLCUDA>(device_)
          .verify(full_to_swa_index_mapping.value());
    }
    const bool has_token_pool = expected_token_pool.has_value();
    RuntimeCheck(
        has_token_pool == expected_token_valid_lens.has_value(),
        "plan_entries: expected_token_pool and expected_token_valid_lens must be both set or both unset");
    if (has_token_pool) {
      TensorMatcher({Npool_rows, Npool_cols})
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device_)
          .verify(expected_token_pool.value());
      TensorMatcher({Npool_rows})
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device_)
          .verify(expected_token_valid_lens.value());
    }
    RuntimeCheck(Nscratch.unwrap() >= Nbs.unwrap() + 1, "verify_offsets_scratch length must be >= bs_padded + 1");

    const int64_t bs_padded = Nbs.unwrap();
    if (bs_padded <= 0) {
      return;
    }

    const int64_t* lut_ptr =
        has_swa_lut ? static_cast<const int64_t*>(full_to_swa_index_mapping.value().data_ptr()) : nullptr;
    const int64_t lut_len = has_swa_lut ? static_cast<int64_t>(Nlut.unwrap()) : 0;

    const int32_t* pool_ptr =
        has_token_pool ? static_cast<const int32_t*>(expected_token_pool.value().data_ptr()) : nullptr;
    const int32_t* valid_lens_ptr =
        has_token_pool ? static_cast<const int32_t*>(expected_token_valid_lens.value().data_ptr()) : nullptr;
    const int64_t pool_stride0 = has_token_pool ? static_cast<int64_t>(Npool_cols.unwrap()) : 0;

    const PlanEntriesParams params = PlanEntriesParams{
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa_lut = lut_ptr,
        .verify_offsets_scratch = static_cast<const int64_t*>(verify_offsets_scratch.data_ptr()),
        .verify_enable = static_cast<const int32_t*>(verify_enable.data_ptr()),
        .expected_token_pool = pool_ptr,
        .expected_token_valid_lens = valid_lens_ptr,
        .out_verify_slot_indices = static_cast<int64_t*>(out_verify_slot_indices.data_ptr()),
        .out_verify_positions = static_cast<int64_t*>(out_verify_positions.data_ptr()),
        .out_verify_prev_slot_indices = static_cast<int64_t*>(out_verify_prev_slot_indices.data_ptr()),
        .out_verify_expected_tokens = static_cast<int64_t*>(out_verify_expected_tokens.data_ptr()),
        .bs_padded = static_cast<int32_t>(bs_padded),
        .verify_capacity = static_cast<int64_t>(Ncap.unwrap()),
        .req_to_token_stride0 = static_cast<int64_t>(Nmax_seq_len.unwrap()),
        .expected_token_pool_stride0 = pool_stride0,
        .swa_window_size = swa_window_size,
        .slot_token_offset = slot_token_offset,
    };

    const DLDevice device = device_.unwrap();
    const int num_sms = get_num_sms(device);
    const int num_blocks = num_sms * kBlocksPerSm;

    const dim3 grid(num_blocks);
    const dim3 block(kBlockSize);

    if (has_swa_lut) {
      if (has_token_pool) {
        LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<true, true>, params, lut_len);
      } else {
        LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<true, false>, params, lut_len);
      }
    } else {
      if (has_token_pool) {
        LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<false, true>, params, lut_len);
      } else {
        LaunchKernel(grid, block, device)(plan_entries_persistent_kernel<false, false>, params, lut_len);
      }
    }
  }
};

}  // namespace
