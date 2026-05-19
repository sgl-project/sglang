// CUDA write kernel and tvm-ffi entry for the KV cache canary.
//
// Algorithm: kernels.md §2.6 (canary_write_step / Implementation). One CUDA block per active write req,
// single thread per block (chain is intrinsically serial). The host wrapper kv_cache_canary_write.py pins
// the byte-equal contract; the torch reference is kv_cache_canary_write_ref.py.

#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "canary_common.cuh"
#include <cstdint>

namespace canary {

namespace {

// Single thread per block — chain advance is inherently serial.
constexpr uint32_t kWriteBlockSize = 1;

// Sentinel value for full_to_swa_index_mapping_present. When 0, the kernel skips SWA translation and uses
// fb_out_cache_loc[i] directly. Mirrors the host wrapper's "lut is None" branch.
constexpr int32_t kSwaMappingAbsent = 0;
constexpr int32_t kSwaMappingPresent = 1;

struct WriteKernelParams {
  uint8_t* canary_buf;
  int64_t slot_stride_bytes;
  int64_t canary_num_slots;

  // Plan tensors.
  const int32_t* write_offsets;
  const int32_t* write_seed_slot_indices;
  const int32_t* write_num_valid_reqs;
  int32_t write_req_capacity;

  // ForwardBatch passthroughs.
  const int32_t* fb_input_ids;
  const int32_t* fb_positions;
  const int32_t* fb_out_cache_loc;

  // SWA LUT (always passed; presence flag toggles whether the kernel applies it).
  const int32_t* full_to_swa_index_mapping;
  int32_t swa_mapping_present;
  int32_t swa_lut_len;

  // Pseudo-mode oracle inputs.
  CanaryPseudoMode pseudo_mode;
  const int32_t* pseudo_expected_tokens;
  const int32_t* pseudo_expected_positions;

  // Violation sink.
  int64_t* violation_ring;
  int32_t* violation_write_index;
  int32_t ring_capacity;
  int32_t kernel_kind;

  // Health counters.
  int64_t* slot_run_counter;
  int64_t* kernel_run_counter;

  // Real-KV sources.
  RealKvSourceHandle sources[kMaxRealKvSources];
  int32_t num_sources;
  RealKvHashMode real_kv_hash_mode;
};

// SWA translation matching _swa_translate in kv_cache_canary_write_ref.py. Negative slot_full passes
// through unchanged; out-of-bounds indices clamp to lut_len - 1 (the trailing sentinel row).
SGL_DEVICE inline int64_t
swa_translate_one(int64_t slot_full, const int32_t* lut, int32_t lut_len, int32_t mapping_present) {
  if (mapping_present == kSwaMappingAbsent) {
    return slot_full;
  }
  if (slot_full < 0) {
    return slot_full;
  }
  int64_t safe_idx = slot_full;
  if (lut_len > 0 && safe_idx >= static_cast<int64_t>(lut_len)) {
    safe_idx = static_cast<int64_t>(lut_len - 1);
  }
  return static_cast<int64_t>(lut[safe_idx]);
}

__global__ void canary_write_kernel(const WriteKernelParams __grid_constant__ p) {
  const uint32_t r = blockIdx.x;

  // Unconditional kernel_run_counter bump (block 0 is always present).
  if (r == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p.kernel_run_counter), 1ULL);
  }

  const int32_t active = *p.write_num_valid_reqs;
  if (r >= static_cast<uint32_t>(active)) {
    return;
  }

  const int32_t entry_start = p.write_offsets[r];
  const int32_t entry_end = p.write_offsets[r + 1];
  const int32_t entry_count = entry_end - entry_start;
  if (entry_count <= 0) {
    return;
  }

  const int32_t seed_slot_idx = p.write_seed_slot_indices[r];

  // Initialize running_prev_hash. When seed >= 0, apply the same advance step that produced seed's
  // successor — keeping slot[0]'s stored prev_hash consistent with §6.1's chain link.
  uint64_t running_prev_hash;
  if (seed_slot_idx >= 0) {
    const int64_t seed_idx64 = static_cast<int64_t>(seed_slot_idx);
    const int64_t seed_token = canary_load_field(p.canary_buf, seed_idx64, p.slot_stride_bytes, kCanaryFieldToken);
    const int64_t seed_position =
        canary_load_field(p.canary_buf, seed_idx64, p.slot_stride_bytes, kCanaryFieldPosition);
    const int64_t seed_prev_hash =
        canary_load_field(p.canary_buf, seed_idx64, p.slot_stride_bytes, kCanaryFieldPrevHash);
    const int64_t seed_real_kv_hash =
        canary_load_field(p.canary_buf, seed_idx64, p.slot_stride_bytes, kCanaryFieldRealKvHash);
    running_prev_hash = splitmix64_mix4(
        static_cast<uint64_t>(seed_prev_hash),
        static_cast<uint64_t>(seed_token),
        static_cast<uint64_t>(seed_position),
        static_cast<uint64_t>(seed_real_kv_hash));
  } else {
    running_prev_hash = splitmix64(kCanaryChainAnchor);
  }

  for (int32_t j = 0; j < entry_count; ++j) {
    const int32_t i = entry_start + j;
    const int64_t slot_full = static_cast<int64_t>(p.fb_out_cache_loc[i]);
    const int64_t slot =
        swa_translate_one(slot_full, p.full_to_swa_index_mapping, p.swa_lut_len, p.swa_mapping_present);
    const int64_t token = static_cast<int64_t>(p.fb_input_ids[i]);
    const int64_t position = static_cast<int64_t>(p.fb_positions[i]);

    const uint64_t real_kv_hash_u64 = fold_real_kv_sources(p.sources, p.num_sources, slot, p.real_kv_hash_mode);
    const int64_t real_kv_hash = static_cast<int64_t>(real_kv_hash_u64);

    // Pseudo-mode comparison. Mismatch records a single violation row carrying both bits OR'd together;
    // chain still advances on the actual (token, position) below so a downstream verify won't cascade.
    if (p.pseudo_mode == CanaryPseudoMode::kOn) {
      const int64_t expected_token = static_cast<int64_t>(p.pseudo_expected_tokens[i]);
      const int64_t expected_position = static_cast<int64_t>(p.pseudo_expected_positions[i]);
      int64_t mismatch_bits = 0;
      if (token != expected_token) {
        mismatch_bits |= kFailReasonWriteTokenMismatch;
      }
      if (position != expected_position) {
        mismatch_bits |= kFailReasonWritePositionMismatch;
      }
      if (mismatch_bits != 0) {
        record_violation(
            p.violation_ring,
            p.violation_write_index,
            p.ring_capacity,
            p.kernel_kind,
            slot,
            position,
            /* stored_token = */ token,
            /* expected_token = */ expected_token,
            /* stored_chain_hash (running running_prev_hash about to be written) = */
            static_cast<int64_t>(running_prev_hash),
            /* expected_aux = expected_position */ expected_position,
            mismatch_bits);
      }
    }

    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldToken, token);
    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldPosition, position);
    canary_store_field(
        p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldPrevHash, static_cast<int64_t>(running_prev_hash));
    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldRealKvHash, real_kv_hash);

    running_prev_hash = splitmix64_mix4(
        running_prev_hash,
        static_cast<uint64_t>(token),
        static_cast<uint64_t>(position),
        static_cast<uint64_t>(real_kv_hash));
  }

  // Each block contributes its entry_count to slot_run_counter once at exit.
  atomicAdd(reinterpret_cast<unsigned long long*>(p.slot_run_counter), static_cast<unsigned long long>(entry_count));
}

}  // namespace

// API source of truth: docstring of canary_write_step in python/sglang/jit_kernel/kv_cache_canary_write.py.
//
// ABI notes (same as verify):
// - real_kv_buf_0 .. real_kv_buf_3 are 4 fixed uint8 tensor slots.
// - real_kv_source_params is a CPU int32 [kMaxRealKvSources, 3] table of (page_size, num_bytes_per_token,
//   read_bytes) triplets.
// - full_to_swa_index_mapping is passed as a tensor with swa_mapping_present == kSwaMappingPresent if and
//   only if the SWA LUT is in effect. When absent the wrapper passes a 1-element dummy and sets
//   swa_mapping_present = 0.
inline void canary_write_step_cuda(
    tvm::ffi::TensorView canary_buf,
    tvm::ffi::TensorView write_offsets,
    tvm::ffi::TensorView write_seed_slot_indices,
    tvm::ffi::TensorView write_num_valid_reqs,
    tvm::ffi::TensorView fb_input_ids,
    tvm::ffi::TensorView fb_positions,
    tvm::ffi::TensorView fb_out_cache_loc,
    tvm::ffi::TensorView full_to_swa_index_mapping,
    int64_t swa_mapping_present,
    int64_t kernel_kind,
    int64_t pseudo_mode,
    tvm::ffi::TensorView pseudo_expected_tokens,
    tvm::ffi::TensorView pseudo_expected_positions,
    tvm::ffi::TensorView violation_ring,
    tvm::ffi::TensorView violation_write_index,
    tvm::ffi::TensorView slot_run_counter,
    tvm::ffi::TensorView kernel_run_counter,
    tvm::ffi::TensorView real_kv_buf_0,
    tvm::ffi::TensorView real_kv_buf_1,
    tvm::ffi::TensorView real_kv_buf_2,
    tvm::ffi::TensorView real_kv_buf_3,
    tvm::ffi::TensorView real_kv_source_params,
    int64_t num_sources,
    int64_t real_kv_hash_mode) {
  using namespace host;

  SymbolicSize N_slots = {"num_canary_slots"};
  SymbolicSize N_stride = {"slot_stride_bytes"};
  SymbolicSize N_write_reqs = {"write_req_capacity"};
  SymbolicSize N_tokens = {"num_tokens_padded"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();

  TensorMatcher({N_slots, N_stride}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(canary_buf);

  // write_offsets has shape [write_req_capacity + 1].
  TensorMatcher({N_write_reqs}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(write_seed_slot_indices);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(write_num_valid_reqs);

  TensorMatcher({N_tokens})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(fb_input_ids)
      .verify(fb_positions)
      .verify(fb_out_cache_loc)
      .verify(pseudo_expected_tokens)
      .verify(pseudo_expected_positions);

  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(violation_write_index);
  SymbolicSize N_ring = {"ring_capacity"};
  TensorMatcher({N_ring, static_cast<int64_t>(kViolationFields)})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(violation_ring);
  TensorMatcher({1})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(slot_run_counter)
      .verify(kernel_run_counter);

  TensorMatcher({SymbolicSize{"real_kv_rows_0"}, SymbolicSize{"real_kv_cols_0"}})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_0);
  TensorMatcher({SymbolicSize{"real_kv_rows_1"}, SymbolicSize{"real_kv_cols_1"}})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_1);
  TensorMatcher({SymbolicSize{"real_kv_rows_2"}, SymbolicSize{"real_kv_cols_2"}})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_2);
  TensorMatcher({SymbolicSize{"real_kv_rows_3"}, SymbolicSize{"real_kv_cols_3"}})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_3);
  TensorMatcher({static_cast<int64_t>(kMaxRealKvSources), static_cast<int64_t>(kRealKvSourceFieldsPerEntry)})
      .with_dtype<int32_t>()
      .with_device<kDLCPU>()
      .verify(real_kv_source_params);

  const int64_t slot_stride_bytes = N_stride.unwrap();
  const int64_t num_slots = N_slots.unwrap();
  const int32_t write_req_capacity = static_cast<int32_t>(N_write_reqs.unwrap());
  const int32_t ring_capacity = static_cast<int32_t>(N_ring.unwrap());
  const DLDevice device = device_.unwrap();

  RuntimeCheck(
      write_offsets.size(0) == static_cast<int64_t>(write_req_capacity) + 1,
      "canary_write: write_offsets.size(0) must equal write_req_capacity + 1 (",
      static_cast<int64_t>(write_req_capacity) + 1,
      "), got ",
      write_offsets.size(0));
  RuntimeCheck(
      slot_stride_bytes >= static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      "canary_write: slot_stride_bytes must hold at least ",
      static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
      " bytes per slot, got ",
      slot_stride_bytes);
  RuntimeCheck(
      num_sources >= 0 && num_sources <= static_cast<int64_t>(kMaxRealKvSources),
      "canary_write: num_sources must be in [0, ",
      static_cast<int64_t>(kMaxRealKvSources),
      "], got ",
      num_sources);

  WriteKernelParams p{};
  p.canary_buf = static_cast<uint8_t*>(canary_buf.data_ptr());
  p.slot_stride_bytes = slot_stride_bytes;
  p.canary_num_slots = num_slots;
  p.write_offsets = static_cast<const int32_t*>(write_offsets.data_ptr());
  p.write_seed_slot_indices = static_cast<const int32_t*>(write_seed_slot_indices.data_ptr());
  p.write_num_valid_reqs = static_cast<const int32_t*>(write_num_valid_reqs.data_ptr());
  p.write_req_capacity = write_req_capacity;
  p.fb_input_ids = static_cast<const int32_t*>(fb_input_ids.data_ptr());
  p.fb_positions = static_cast<const int32_t*>(fb_positions.data_ptr());
  p.fb_out_cache_loc = static_cast<const int32_t*>(fb_out_cache_loc.data_ptr());
  p.full_to_swa_index_mapping = static_cast<const int32_t*>(full_to_swa_index_mapping.data_ptr());
  p.swa_mapping_present = static_cast<int32_t>(swa_mapping_present);
  p.swa_lut_len = static_cast<int32_t>(full_to_swa_index_mapping.size(0));
  p.pseudo_mode = static_cast<CanaryPseudoMode>(pseudo_mode);
  p.pseudo_expected_tokens = static_cast<const int32_t*>(pseudo_expected_tokens.data_ptr());
  p.pseudo_expected_positions = static_cast<const int32_t*>(pseudo_expected_positions.data_ptr());
  p.violation_ring = static_cast<int64_t*>(violation_ring.data_ptr());
  p.violation_write_index = static_cast<int32_t*>(violation_write_index.data_ptr());
  p.ring_capacity = ring_capacity;
  p.kernel_kind = static_cast<int32_t>(kernel_kind);
  p.slot_run_counter = static_cast<int64_t*>(slot_run_counter.data_ptr());
  p.kernel_run_counter = static_cast<int64_t*>(kernel_run_counter.data_ptr());

  const int32_t* params = static_cast<const int32_t*>(real_kv_source_params.data_ptr());
  tvm::ffi::TensorView source_bufs[kMaxRealKvSources] = {real_kv_buf_0, real_kv_buf_1, real_kv_buf_2, real_kv_buf_3};
  for (int s = 0; s < kMaxRealKvSources; ++s) {
    p.sources[s].tensor = static_cast<const uint8_t*>(source_bufs[s].data_ptr());
    p.sources[s].row_stride_bytes = static_cast<int32_t>(source_bufs[s].size(1));
    p.sources[s].page_size = params[s * kRealKvSourceFieldsPerEntry + kRealKvSourceFieldPageSize];
    p.sources[s].num_bytes_per_token = params[s * kRealKvSourceFieldsPerEntry + kRealKvSourceFieldNumBytesPerToken];
    p.sources[s].read_bytes = params[s * kRealKvSourceFieldsPerEntry + kRealKvSourceFieldReadBytes];
  }
  p.num_sources = static_cast<int32_t>(num_sources);
  p.real_kv_hash_mode = static_cast<RealKvHashMode>(real_kv_hash_mode);

  // Grid: one block per write req capacity slot; the kernel early-exits on r >= write_num_valid_reqs[0].
  // Always launch at least one block so the unconditional kernel_run_counter bump runs even when capacity
  // == 0.
  const uint32_t grid = write_req_capacity == 0 ? 1u : static_cast<uint32_t>(write_req_capacity);
  LaunchKernel(grid, kWriteBlockSize, device)(canary_write_kernel, p);
}

}  // namespace canary
