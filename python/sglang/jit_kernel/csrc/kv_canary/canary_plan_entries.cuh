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
  const int64_t* __restrict__ req_pool_indices;               // [bs_padded] int64
  const int64_t* __restrict__ prefix_lens;                    // [bs_padded] int64
  const int32_t* __restrict__ req_to_token;                   // [max_reqs, max_seq_len] int32
  const int64_t* __restrict__ full_to_swa_lut;                // [lut_len] int64, may be nullptr when !HAS_SWA_LUT
  const int64_t* __restrict__ verify_offsets_scratch;         // [bs_padded + 1] int64 (cumulative prefix sum)
  const int32_t* __restrict__ verify_enable;                  // [1] int32 — 0 ⇒ skip scatter entirely
  const int32_t* __restrict__ req_to_verify_expected_tokens;  // [max_reqs, max_context_len] int32, may be nullptr
  const int64_t* __restrict__ req_to_verify_expected_tokens_valid_lens;  // [bs_padded] int64 per-req snapshot length;
                                                                         // nullptr iff pool is null
  // Outputs.
  int64_t* __restrict__ out_verify_slot_indices;        // [verify_capacity] int64
  int64_t* __restrict__ out_verify_expected_tokens;     // [verify_capacity] int64
  int64_t* __restrict__ out_verify_expected_positions;  // [verify_capacity] int64
  int64_t* __restrict__ out_verify_prev_slot_indices;   // [verify_capacity] int64
  // Sizes / strides.
  int32_t bs_padded;
  int64_t verify_capacity;  // out_verify_*[verify_capacity]; scatter is clamped to this length.
  int64_t req_to_token_stride0;
  int64_t req_to_verify_expected_tokens_stride0;
  int32_t kv_token_id_vs_position_offset;  // 0 for target pools; +1 for EAGLE draft.
  int32_t swa_window_size;
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

// Persistent grid; one thread = one verify entry (with stride). Template parameter HAS_SWA_LUT switches
// the SWA-translate path off entirely in the FULL pool variant.
// HAS_VERIFY_EXPECTED_TOKEN_POOL toggles the source-of-truth token gather; when off, every active entry
// writes the ``-1`` sentinel so the verify kernel skips the token-mismatch check.
template <bool HAS_SWA_LUT, bool HAS_VERIFY_EXPECTED_TOKEN_POOL>
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

    int64_t out_expected_input_id = -1;
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      const int64_t sot_pos = out_position + static_cast<int64_t>(params.kv_token_id_vs_position_offset);
      const int64_t valid_len = params.req_to_verify_expected_tokens_valid_lens[req_id];
      if (sot_pos >= 0 && sot_pos < valid_len) {
        const int32_t token =
            params.req_to_verify_expected_tokens[rp * params.req_to_verify_expected_tokens_stride0 + sot_pos];
        out_expected_input_id = static_cast<int64_t>(token);
      }
    }

    // 4) Scatter. out_idx == tid since verify_offsets[req_id] + entry_idx == tid by construction.
    params.out_verify_slot_indices[tid] = out_slot;
    params.out_verify_expected_tokens[tid] = out_expected_input_id;
    params.out_verify_expected_positions[tid] = out_position;
    params.out_verify_prev_slot_indices[tid] = out_prev_slot;
  }
}

template <bool HAS_SWA_LUT, bool HAS_VERIFY_EXPECTED_TOKEN_POOL>
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
      const tvm::ffi::Optional<tvm::ffi::TensorView> req_to_verify_expected_tokens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> req_to_verify_expected_tokens_valid_lens,
      const tvm::ffi::TensorView out_verify_slot_indices,
      const tvm::ffi::TensorView out_verify_expected_tokens,
      const tvm::ffi::TensorView out_verify_expected_positions,
      const tvm::ffi::TensorView out_verify_prev_slot_indices,
      int32_t kv_token_id_vs_position_offset,
      int32_t swa_window_size) {
    using namespace host;

    SymbolicSize Nbs = {"bs_padded"};
    SymbolicSize Nscratch = {"verify_offsets_scratch_len"};
    SymbolicSize Ncap = {"verify_capacity"};
    SymbolicSize Nmax_reqs = {"max_reqs"};
    SymbolicSize Nmax_seq_len = {"max_seq_len"};
    SymbolicSize Nlut = {"lut_len"};
    SymbolicSize Npool_rows = {"req_to_verify_expected_tokens_rows"};
    SymbolicSize Npool_cols = {"req_to_verify_expected_tokens_cols"};
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
        .verify(out_verify_expected_tokens)
        .verify(out_verify_expected_positions)
        .verify(out_verify_prev_slot_indices);
    TensorMatcher({Nmax_reqs, Nmax_seq_len})  //
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(req_to_token);
    RuntimeCheck(
        full_to_swa_index_mapping.has_value() == HAS_SWA_LUT,
        "full_to_swa_index_mapping presence does not match HAS_SWA_LUT specialization");
    RuntimeCheck(
        req_to_verify_expected_tokens.has_value() == HAS_VERIFY_EXPECTED_TOKEN_POOL,
        "req_to_verify_expected_tokens presence does not match HAS_VERIFY_EXPECTED_TOKEN_POOL specialization");
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      RuntimeCheck(
          req_to_verify_expected_tokens_valid_lens.has_value(),
          "req_to_verify_expected_tokens_valid_lens must be set when req_to_verify_expected_tokens is set");
    }
    if constexpr (HAS_SWA_LUT) {
      TensorMatcher({Nlut})  //
          .with_dtype<int64_t>()
          .with_device<kDLCUDA>(device_)
          .verify(full_to_swa_index_mapping.value());
    }
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      TensorMatcher({Npool_rows, Npool_cols})  //
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>(device_)
          .verify(req_to_verify_expected_tokens.value());
      TensorMatcher({Nbs})  //
          .with_dtype<int64_t>()
          .with_device<kDLCUDA>(device_)
          .verify(req_to_verify_expected_tokens_valid_lens.value());
    }
    RuntimeCheck(Nscratch.unwrap() >= Nbs.unwrap() + 1, "verify_offsets_scratch length must be >= bs_padded + 1");

    const int64_t bs_padded = Nbs.unwrap();
    if (bs_padded <= 0) {
      return;
    }

    const int64_t* lut_ptr = nullptr;
    int64_t lut_len = 0;
    if constexpr (HAS_SWA_LUT) {
      lut_ptr = static_cast<const int64_t*>(full_to_swa_index_mapping.value().data_ptr());
      lut_len = static_cast<int64_t>(Nlut.unwrap());
    }

    const int32_t* expected_token_ids_ptr = nullptr;
    int64_t expected_token_ids_stride0 = 0;
    const int64_t* req_to_verify_expected_tokens_valid_lens_ptr = nullptr;
    if constexpr (HAS_VERIFY_EXPECTED_TOKEN_POOL) {
      expected_token_ids_ptr = static_cast<const int32_t*>(req_to_verify_expected_tokens.value().data_ptr());
      expected_token_ids_stride0 = static_cast<int64_t>(Npool_cols.unwrap());
      req_to_verify_expected_tokens_valid_lens_ptr =
          static_cast<const int64_t*>(req_to_verify_expected_tokens_valid_lens.value().data_ptr());
    }

    const PlanEntriesParams params = PlanEntriesParams{
        .req_pool_indices = static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        .prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .full_to_swa_lut = lut_ptr,
        .verify_offsets_scratch = static_cast<const int64_t*>(verify_offsets_scratch.data_ptr()),
        .verify_enable = static_cast<const int32_t*>(verify_enable.data_ptr()),
        .req_to_verify_expected_tokens = expected_token_ids_ptr,
        .req_to_verify_expected_tokens_valid_lens = req_to_verify_expected_tokens_valid_lens_ptr,
        .out_verify_slot_indices = static_cast<int64_t*>(out_verify_slot_indices.data_ptr()),
        .out_verify_expected_tokens = static_cast<int64_t*>(out_verify_expected_tokens.data_ptr()),
        .out_verify_expected_positions = static_cast<int64_t*>(out_verify_expected_positions.data_ptr()),
        .out_verify_prev_slot_indices = static_cast<int64_t*>(out_verify_prev_slot_indices.data_ptr()),
        .bs_padded = static_cast<int32_t>(bs_padded),
        .verify_capacity = static_cast<int64_t>(Ncap.unwrap()),
        .req_to_token_stride0 = static_cast<int64_t>(Nmax_seq_len.unwrap()),
        .req_to_verify_expected_tokens_stride0 = expected_token_ids_stride0,
        .kv_token_id_vs_position_offset = kv_token_id_vs_position_offset,
        .swa_window_size = swa_window_size,
    };

    const DLDevice device = device_.unwrap();
    const int num_sms = get_num_sms(device);
    const int num_blocks = num_sms * kBlocksPerSm;

    const dim3 grid(num_blocks);
    const dim3 block(kBlockSize);

    LaunchKernel(grid, block, device)(
        plan_entries_persistent_kernel<HAS_SWA_LUT, HAS_VERIFY_EXPECTED_TOKEN_POOL>, params, lut_len);
  }
};

}  // namespace
