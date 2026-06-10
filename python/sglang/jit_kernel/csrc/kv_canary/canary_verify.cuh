#pragma once

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For div_ceil, RuntimeCheck

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "canary_common.cuh"
#include <cstdint>

namespace canary {

namespace {

constexpr uint32_t kVerifyBlockSize = 512;
constexpr uint32_t kPersistentBlocks = 64;

struct VerifyKernelParams {
  // Canary buffer this launch verifies. Read-only.
  const uint8_t* canary_buf;
  int64_t slot_stride_bytes;

  // Plan tensors.
  const int64_t* verify_slot_indices;
  const int64_t* verify_expected_tokens;
  const int64_t* verify_expected_positions;
  const int64_t* verify_prev_slot_indices;
  const int32_t* verify_num_valid;
  const int32_t* verify_enable;
  int32_t verify_capacity;

  // Violation sink (ring + write_index + capacity + kernel_kind bundled in canary_common.cuh).
  ViolationSink violation_sink;

  // Health counters.
  int64_t* slot_run_counter;
  int64_t* kernel_run_counter;

  // Real-KV sources (fixed-size ABI; padding slots have read_bytes = 0).
  RealKvSourceHandle sources[kMaxRealKvSources];
  int32_t num_sources;
  RealKvHashMode real_kv_hash_mode;
};

template <bool CHECK_VERIFY_EXPECTED_TOKEN>
__global__ void canary_verify_kernel(const VerifyKernelParams __grid_constant__ p) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;

  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p.kernel_run_counter), 1ULL);
  }

  if (*p.verify_enable == 0) {
    return;
  }

  const int32_t active = min(*p.verify_num_valid, p.verify_capacity);

  uint32_t local_active_count = 0;
  for (uint32_t entry_idx = tid; entry_idx < static_cast<uint32_t>(active); entry_idx += stride) {
    ++local_active_count;

    const int64_t slot_idx = p.verify_slot_indices[entry_idx];
    const int64_t expected_position = p.verify_expected_positions[entry_idx];
    const int64_t prev_slot_idx = p.verify_prev_slot_indices[entry_idx];
    int64_t expected_input_id = -1;
    if constexpr (CHECK_VERIFY_EXPECTED_TOKEN) {
      expected_input_id = p.verify_expected_tokens[entry_idx];
    }

    if (slot_idx == kTokenToKvSlotPadding) {
      continue;
    }

    const int64_t stored_token = canary_load_field(p.canary_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldToken);
    const int64_t stored_position =
        canary_load_field(p.canary_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPosition);
    const int64_t stored_chain_hash =
        canary_load_field(p.canary_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldPrevHash);
    const int64_t stored_real_kv_hash =
        canary_load_field(p.canary_buf, slot_idx, p.slot_stride_bytes, kCanaryFieldRealKvHash);

    const bool prev_reachable = (prev_slot_idx != kTokenToKvSlotPadding);
    const int64_t expected_chain_hash =
        prev_reachable ? static_cast<int64_t>(compute_slot_hash(p.canary_buf, p.slot_stride_bytes, prev_slot_idx))
                       : stored_chain_hash;

    const uint64_t expected_real_kv_hash_u64 =
        real_kv_fold_sources(p.sources, p.num_sources, slot_idx, p.real_kv_hash_mode);
    const int64_t expected_real_kv_hash = static_cast<int64_t>(expected_real_kv_hash_u64);

    FailReason fail_reason_bits{};
    if (prev_reachable && stored_chain_hash != expected_chain_hash) {
      fail_reason_bits |= FailReason::kVerifyChainHashMismatch;
    }
    if constexpr (CHECK_VERIFY_EXPECTED_TOKEN) {
      if (expected_input_id != -1 && stored_token != expected_input_id) {
        fail_reason_bits |= FailReason::kVerifyTokenMismatch;
      }
    }
    if (stored_position != expected_position) {
      fail_reason_bits |= FailReason::kVerifyPositionMismatch;
    }
    if (stored_real_kv_hash != expected_real_kv_hash) {
      fail_reason_bits |= FailReason::kVerifyRealKvHashMismatch;
    }

    if (fail_reason_bits != FailReason{}) {
      record_violation(
          p.violation_sink,
          ViolationRow{
              /* slot_idx = */ slot_idx,
              /* position = */ stored_position,
              /* stored_token = */ stored_token,
              /* expected_token = */ expected_input_id,
              /* stored_chain_hash = */ stored_chain_hash,
              /* expected_aux = */ expected_chain_hash,
              /* fail_reason_bits = */ static_cast<int64_t>(fail_reason_bits),
          });
    }
  }

  uint32_t warp_active_count = local_active_count;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_active_count += __shfl_down_sync(0xFFFFFFFFu, warp_active_count, offset);
  }
  if ((threadIdx.x & 31u) == 0u && warp_active_count != 0u) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(p.slot_run_counter), static_cast<unsigned long long>(warp_active_count));
  }
}

}  // namespace

// API source of truth: docstring of canary_verify_step in python/sglang/jit_kernel/kv_canary/verify.py.
//
// ABI notes:
// - real_kv_buf_0 .. real_kv_buf_3 are 4 fixed uint8 tensor slots. Unused slots are dummy 1-byte tensors;
//   the host wrapper sets real_kv_source_params[s, 2] (read_bytes) = 0 for them.
// - real_kv_source_params is a CPU int32 tensor of shape [kMaxRealKvSources, 3]: per-source (page_size,
//   num_bytes_per_token, read_bytes). Lives on CPU because the host wrapper materializes it from a tuple
//   of Python dataclasses.
// - num_sources is passed as int64 (tvm-ffi scalar convention) but always in [0, kMaxRealKvSources].
template <bool CHECK_VERIFY_EXPECTED_TOKEN>
struct CanaryVerifyKernel {
  static void
  run(tvm::ffi::TensorView canary_buf,
      tvm::ffi::TensorView verify_slot_indices,
      tvm::ffi::TensorView verify_expected_tokens,
      tvm::ffi::TensorView verify_expected_positions,
      tvm::ffi::TensorView verify_prev_slot_indices,
      tvm::ffi::TensorView verify_num_valid,
      tvm::ffi::TensorView verify_enable,
      int64_t kernel_kind,
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
    SymbolicSize N_verify = {"verify_capacity"};
    SymbolicDevice device_;
    device_.set_options<kDLCUDA>();

    TensorMatcher({N_slots, N_stride}).with_dtype<uint8_t>().with_device<kDLCUDA>(device_).verify(canary_buf);

    TensorMatcher({N_verify})
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>(device_)
        .verify(verify_slot_indices)
        .verify(verify_expected_tokens)
        .verify(verify_expected_positions)
        .verify(verify_prev_slot_indices);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(verify_num_valid);
    TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(verify_enable);

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

    // Real-KV source buffers are 2-D uint8 (any shape); dim 1 carries the row-stride in bytes. They live on
    // CUDA. real_kv_source_params is a small CPU int32 table of length kMaxRealKvSources * 3.
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
    const int32_t verify_capacity = static_cast<int32_t>(N_verify.unwrap());
    const int32_t ring_capacity = static_cast<int32_t>(N_ring.unwrap());
    const DLDevice device = device_.unwrap();

    RuntimeCheck(
        slot_stride_bytes >= static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
        "canary_verify: slot_stride_bytes must hold at least ",
        static_cast<int64_t>(kCanaryFieldsPerSlot * sizeof(int64_t)),
        " bytes per slot, got ",
        slot_stride_bytes);
    RuntimeCheck(
        num_sources >= 0 && num_sources <= static_cast<int64_t>(kMaxRealKvSources),
        "canary_verify: num_sources must be in [0, ",
        static_cast<int64_t>(kMaxRealKvSources),
        "], got ",
        num_sources);

    VerifyKernelParams p{};
    p.canary_buf = static_cast<const uint8_t*>(canary_buf.data_ptr());
    p.slot_stride_bytes = slot_stride_bytes;
    p.verify_slot_indices = static_cast<const int64_t*>(verify_slot_indices.data_ptr());
    p.verify_expected_tokens = static_cast<const int64_t*>(verify_expected_tokens.data_ptr());
    p.verify_expected_positions = static_cast<const int64_t*>(verify_expected_positions.data_ptr());
    p.verify_prev_slot_indices = static_cast<const int64_t*>(verify_prev_slot_indices.data_ptr());
    p.verify_num_valid = static_cast<const int32_t*>(verify_num_valid.data_ptr());
    p.verify_enable = static_cast<const int32_t*>(verify_enable.data_ptr());
    p.verify_capacity = verify_capacity;
    p.violation_sink.ring = static_cast<int64_t*>(violation_ring.data_ptr());
    p.violation_sink.write_index = static_cast<int32_t*>(violation_write_index.data_ptr());
    p.violation_sink.ring_capacity = ring_capacity;
    p.violation_sink.kernel_kind = static_cast<int32_t>(kernel_kind);
    p.slot_run_counter = static_cast<int64_t*>(slot_run_counter.data_ptr());
    p.kernel_run_counter = static_cast<int64_t*>(kernel_run_counter.data_ptr());

    // Materialize the source handle array on the host. The CPU param tensor carries the per-source ints.
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

    const uint32_t grid = kPersistentBlocks;
    LaunchKernel(grid, kVerifyBlockSize, device)(canary_verify_kernel<CHECK_VERIFY_EXPECTED_TOKEN>, p);
  }
};

}  // namespace canary
