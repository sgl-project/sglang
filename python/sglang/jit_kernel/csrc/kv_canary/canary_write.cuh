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

struct WriteKernelParams {
  uint8_t* canary_buf;
  int64_t slot_stride_bytes;

  // Plan tensors.
  const int64_t* write_offsets;
  const int64_t* write_seed_slot_indices;
  const int32_t* write_num_valid_reqs;
  int32_t write_req_capacity;

  // ForwardBatch passthroughs. out_cache_loc is caller-pre-translated for SWA groups; the kernel
  // treats it opaquely and skips entries with slot < 0.
  const int64_t* input_ids;
  const int64_t* positions;
  const int64_t* out_cache_loc;

  // Pseudo-mode oracle inputs.
  bool enable_write_input_assert;
  const int64_t* expected_input_tokens;
  const int64_t* expected_input_positions;

  // Violation sink (ring + write_index + capacity + kernel_kind bundled in canary_common.cuh).
  ViolationSink violation_sink;

  // Health counters.
  int64_t* slot_run_counter;
  int64_t* kernel_run_counter;

  // Gates the chain-step position assert below. Default-on (1); CanaryManager zeros during the
  // warmup window and flips back in mark_init_finished().
  const int32_t* enable_chain_position_assert;

  // Real-KV sources.
  RealKvSourceHandle sources[kMaxRealKvSources];
  int32_t num_sources;
  RealKvHashMode real_kv_hash_mode;
};

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

  const int64_t entry_start = p.write_offsets[r];
  const int64_t entry_end = p.write_offsets[r + 1];
  const int64_t entry_count = entry_end - entry_start;
  if (entry_count <= 0) {
    return;
  }

  const int64_t seed_slot_idx = p.write_seed_slot_indices[r];

  // Initialize running_prev_hash by advancing the chain from the seed slot
  uint64_t running_prev_hash =
      compute_slot_hash(p.canary_buf, p.slot_stride_bytes, static_cast<int64_t>(seed_slot_idx));

  // Assumes eagle topk=1 (linear chain). Under topk>1 target_verify would be a tree and
  // sibling positions share parent.pos+1, breaking this invariant.
  const bool do_chain_position_assert = (seed_slot_idx >= 0) && (*p.enable_chain_position_assert != 0);
  int64_t running_prev_position = 0;
  if (do_chain_position_assert) {
    running_prev_position = canary_load_field(p.canary_buf, seed_slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
  }

  int64_t entries_written = 0;
  for (int64_t entry_offset = 0; entry_offset < entry_count; ++entry_offset) {
    const int64_t entry_idx = entry_start + entry_offset;
    const int64_t slot = p.out_cache_loc[entry_idx];

    if (slot < 0) {
      continue;
    }
    ++entries_written;

    const int64_t token = p.input_ids[entry_idx];
    const int64_t position = p.positions[entry_idx];

    const uint64_t real_kv_hash_u64 = real_kv_fold_sources(p.sources, p.num_sources, slot, p.real_kv_hash_mode);
    const int64_t real_kv_hash = static_cast<int64_t>(real_kv_hash_u64);

    if (p.enable_write_input_assert) {
      const int64_t expected_token = p.expected_input_tokens[entry_idx];
      const int64_t expected_position = p.expected_input_positions[entry_idx];
      FailReason mismatch_bits{};
      if (token != expected_token) {
        mismatch_bits |= FailReason::kWriteTokenMismatch;
      }
      if (position != expected_position) {
        mismatch_bits |= FailReason::kWritePositionMismatch;
      }
      if (mismatch_bits != FailReason{}) {
        record_violation(
            p.violation_sink,
            ViolationRow{
                /* slot_idx = */ slot,
                /* position = */ position,
                /* stored_token = */ token,
                /* expected_token = */ expected_token,
                /* stored_chain_hash = */ static_cast<int64_t>(running_prev_hash),
                /* expected_aux = expected_position */ expected_position,
                /* fail_reason_bits = */ static_cast<int64_t>(mismatch_bits),
            });
      }
    }

    if (do_chain_position_assert) {
      const int64_t expected_position_chain = running_prev_position + 1;
      if (position != expected_position_chain) {
        record_violation(
            p.violation_sink,
            ViolationRow{
                /* slot_idx = */ slot,
                /* position = */ position,
                /* stored_token = */ token,
                /* expected_token = */ token,
                /* stored_chain_hash = */ static_cast<int64_t>(running_prev_hash),
                /* expected_aux = expected_position */ expected_position_chain,
                /* fail_reason_bits = */ static_cast<int64_t>(FailReason::kWritePositionMismatch),
            });
      }
      running_prev_position = position;
    }

    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldToken, token);
    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldPosition, position);
    canary_store_field(
        p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldPrevHash, static_cast<int64_t>(running_prev_hash));
    canary_store_field(p.canary_buf, slot, p.slot_stride_bytes, kCanaryFieldRealKvHash, real_kv_hash);

    running_prev_hash =
        splitmix64_mix3(running_prev_hash, static_cast<uint64_t>(token), static_cast<uint64_t>(position));
  }

  // Each block contributes its non-skipped entry count to slot_run_counter once at exit.
  atomicAdd(
      reinterpret_cast<unsigned long long*>(p.slot_run_counter), static_cast<unsigned long long>(entries_written));
}

}  // namespace

// API source of truth: docstring of canary_write_step in python/sglang/jit_kernel/kv_canary/write.py.
//
// ABI notes (same as verify):
// - real_kv_buf_0 .. real_kv_buf_3 are 4 fixed uint8 tensor slots.
// - real_kv_source_params is a CPU int32 [kMaxRealKvSources, 3] table of (page_size, num_bytes_per_token,
//   read_bytes) triplets.
// - out_cache_loc is caller-pre-translated for SWA groups; -1 entries mark skip. The kernel does not
//   consult any LUT.
inline void canary_write_step_cuda(
    tvm::ffi::TensorView canary_buf,
    tvm::ffi::TensorView write_offsets,
    tvm::ffi::TensorView write_seed_slot_indices,
    tvm::ffi::TensorView write_num_valid_reqs,
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView out_cache_loc,
    int64_t kernel_kind,
    int64_t enable_write_input_assert,
    const tvm::ffi::Optional<tvm::ffi::TensorView> expected_input_tokens,
    const tvm::ffi::Optional<tvm::ffi::TensorView> expected_input_positions,
    tvm::ffi::TensorView violation_ring,
    tvm::ffi::TensorView violation_write_index,
    tvm::ffi::TensorView slot_run_counter,
    tvm::ffi::TensorView kernel_run_counter,
    tvm::ffi::TensorView enable_chain_position_assert,
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

  // write_offsets has shape [write_req_capacity + 1]; the length relationship is pinned by the
  // RuntimeCheck below, this matcher pins dtype + device.
  SymbolicSize N_write_offsets = {"write_offsets_len"};
  TensorMatcher({N_write_offsets}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(write_offsets);
  TensorMatcher({N_write_reqs}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(write_seed_slot_indices);
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(write_num_valid_reqs);

  TensorMatcher({N_tokens})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device_)
      .verify(input_ids)
      .verify(positions)
      .verify(out_cache_loc);
  const bool enable_write_input_assert_bool = (enable_write_input_assert != 0);
  RuntimeCheck(
      enable_write_input_assert_bool == expected_input_tokens.has_value(),
      "canary_write: expected_input_tokens presence must match enable_write_input_assert");
  RuntimeCheck(
      enable_write_input_assert_bool == expected_input_positions.has_value(),
      "canary_write: expected_input_positions presence must match enable_write_input_assert");
  if (enable_write_input_assert_bool) {
    TensorMatcher({N_tokens})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(expected_input_tokens.value())
        .verify(expected_input_positions.value());
  }

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
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(enable_chain_position_assert);

  SymbolicSize N_real_kv_rows_0 = {"real_kv_rows_0"};
  SymbolicSize N_real_kv_cols_0 = {"real_kv_cols_0"};
  TensorMatcher({N_real_kv_rows_0, N_real_kv_cols_0})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_0);
  SymbolicSize N_real_kv_rows_1 = {"real_kv_rows_1"};
  SymbolicSize N_real_kv_cols_1 = {"real_kv_cols_1"};
  TensorMatcher({N_real_kv_rows_1, N_real_kv_cols_1})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_1);
  SymbolicSize N_real_kv_rows_2 = {"real_kv_rows_2"};
  SymbolicSize N_real_kv_cols_2 = {"real_kv_cols_2"};
  TensorMatcher({N_real_kv_rows_2, N_real_kv_cols_2})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_2);
  SymbolicSize N_real_kv_rows_3 = {"real_kv_rows_3"};
  SymbolicSize N_real_kv_cols_3 = {"real_kv_cols_3"};
  TensorMatcher({N_real_kv_rows_3, N_real_kv_cols_3})
      .with_dtype<uint8_t>()
      .with_device<kDLCUDA>(device_)
      .verify(real_kv_buf_3);
  TensorMatcher({static_cast<int64_t>(kMaxRealKvSources), static_cast<int64_t>(kRealKvSourceFieldsPerEntry)})
      .with_dtype<int32_t>()
      .with_device<kDLCPU>()
      .verify(real_kv_source_params);

  const int64_t slot_stride_bytes = N_stride.unwrap();
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
  p.write_offsets = static_cast<const int64_t*>(write_offsets.data_ptr());
  p.write_seed_slot_indices = static_cast<const int64_t*>(write_seed_slot_indices.data_ptr());
  p.write_num_valid_reqs = static_cast<const int32_t*>(write_num_valid_reqs.data_ptr());
  p.write_req_capacity = write_req_capacity;
  p.input_ids = static_cast<const int64_t*>(input_ids.data_ptr());
  p.positions = static_cast<const int64_t*>(positions.data_ptr());
  p.out_cache_loc = static_cast<const int64_t*>(out_cache_loc.data_ptr());
  p.enable_write_input_assert = enable_write_input_assert_bool;
  p.expected_input_tokens =
      enable_write_input_assert_bool ? static_cast<const int64_t*>(expected_input_tokens.value().data_ptr()) : nullptr;
  p.expected_input_positions = enable_write_input_assert_bool
                                   ? static_cast<const int64_t*>(expected_input_positions.value().data_ptr())
                                   : nullptr;
  p.violation_sink.ring = static_cast<int64_t*>(violation_ring.data_ptr());
  p.violation_sink.write_index = static_cast<int32_t*>(violation_write_index.data_ptr());
  p.violation_sink.ring_capacity = ring_capacity;
  p.violation_sink.kernel_kind = static_cast<int32_t>(kernel_kind);
  p.slot_run_counter = static_cast<int64_t*>(slot_run_counter.data_ptr());
  p.kernel_run_counter = static_cast<int64_t*>(kernel_run_counter.data_ptr());
  p.enable_chain_position_assert = static_cast<const int32_t*>(enable_chain_position_assert.data_ptr());

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
