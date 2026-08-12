#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace device::compress {

/// \brief Per-batch decode plan. Layout: 16 bytes.
struct alignas(16) DecodePlan {
  uint32_t seq_len;
  int32_t write_loc;
  int32_t read_page_0;
  int32_t read_page_1;
};

/// \brief Per-token compress plan (used by c4/c128 prefill). Layout: 16 bytes.
struct alignas(16) CompressPlan {
  uint32_t seq_len;
  uint16_t ragged_id;
  uint16_t buffer_len;
  int32_t read_page_0;
  /// \brief Stage 0 (CPU): batch_id (used to look up page table).
  /// \brief Stage 1 (GPU): final state-pool write location.
  int32_t read_page_1;

  static SGL_DEVICE __host__ CompressPlan invalid() {
    return CompressPlan{-1u, 0, 0, -1, -1};
  }

  SGL_DEVICE __host__ bool is_invalid() const {
    return seq_len == -1u;
  }
};

/// \brief Per-token write plan (used by c4/c128 prefill). Layout: 8 bytes.
struct alignas(8) WritePlan {
  /// \brief Stage 0 (CPU): packed `(batch_id << 16) | ragged_id`.
  /// \brief Stage 1 (GPU): just `ragged_id`.
  uint32_t ragged_id;
  /// \brief Stage 0 (CPU): position + 1 (used to look up state slot).
  /// \brief Stage 1 (GPU): final state-pool write location.
  int32_t write_loc;

  static SGL_DEVICE __host__ WritePlan invalid() {
    return WritePlan{-1u, -1};
  }

  SGL_DEVICE __host__ bool is_invalid() const {
    return ragged_id == -1u;
  }
};

}  // namespace device::compress

namespace host::compress {

using device::compress::CompressPlan;
using device::compress::DecodePlan;
using device::compress::WritePlan;

static_assert(alignof(DecodePlan) == sizeof(DecodePlan));
static_assert(sizeof(DecodePlan) == 16);
static_assert(alignof(CompressPlan) == sizeof(CompressPlan));
static_assert(sizeof(CompressPlan) == 16);
static_assert(alignof(WritePlan) == sizeof(WritePlan));
static_assert(sizeof(WritePlan) == 8);

inline auto verify_plan_d(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const DecodePlan* {
  TensorMatcher({N, sizeof(DecodePlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const DecodePlan*>(t.data_ptr());
}

inline auto verify_plan_c(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const CompressPlan* {
  TensorMatcher({N, sizeof(CompressPlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const CompressPlan*>(t.data_ptr());
}

inline auto verify_plan_w(tvm::ffi::TensorView t, SymbolicSize& N, SymbolicDevice& device) -> const WritePlan* {
  TensorMatcher({N, sizeof(WritePlan)})  //
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(t);
  return static_cast<const WritePlan*>(t.data_ptr());
}

}  // namespace host::compress
