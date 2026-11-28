// gfx950 DCP8 all-to-all of 128-wide attention partials and LSE in one launch: each rank stages its rows in its IPC
// scratch, then reads its heads from every rank's.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include

#ifndef USE_ROCM
#error "dcp_ipc_a2a_hip.cuh is a ROCm kernel"
#endif

namespace sglang {
namespace dcp_ipc_a2a {

inline constexpr int kN = 8;
inline constexpr int kD = 128;
inline constexpr int kMaxRows = 1024;
inline constexpr int kVecPerHead = kD * 2 / 16;
inline constexpr int kBlocks = aiter::kMaxBlocks;
inline constexpr int kRowsPerPass = 4;

template <int kHr>
struct Shape {
  static constexpr int kSlotVecs = kHr * kVecPerHead;
  static constexpr int kRowVecs = kN * kSlotVecs;
  static constexpr int kThreads = kRowsPerPass * kSlotVecs;
  static constexpr int kStageVecs = kRowVecs / kThreads;
  // the scratch holds kMaxRows rows of partials, then their LSE
  static constexpr int64_t kDataBytes = int64_t{kMaxRows} * kRowVecs * 16;
  static_assert(kHr >= 1 && kThreads <= 1024, "heads per rank");
};

// Not HIP's union-based uint4: register arrays of it go through scratch memory.
using Vec = uint32_t __attribute__((ext_vector_type(4)));

struct Params {
  const Vec* __restrict__ out;
  const float* __restrict__ lse;
  Vec* __restrict__ recv_out;
  float* __restrict__ recv_lse;
  const char* scratch[kN];
  char* self_scratch;
  aiter::RankSignals sg;
  aiter::Signal* self_sg;
  int32_t rank;
  int32_t rows;
};

// On gfx9 a wait for a load also waits for older stores, so peers are only read, each pass issues its loads before
// its stores, and scratch[] is indexed with constants only (a runtime-indexed kernarg read is a vector load).
template <int kHr>
__global__ __launch_bounds__(Shape<kHr>::kThreads, 1) void exchange_kernel(const __grid_constant__ Params p) {
  using S = Shape<kHr>;
  const int tid = threadIdx.x;
  // Rows may be overwritten only after every rank has read them in the previous call.
  aiter::start_sync<kN>(p.sg, p.self_sg, p.rank);
  Vec* own = reinterpret_cast<Vec*>(p.self_scratch);
  float* own_lse = reinterpret_cast<float*>(p.self_scratch + S::kDataBytes);
  for (int k = 0; blockIdx.x + k * kBlocks < p.rows; k += kRowsPerPass) {
    Vec x[kRowsPerPass][S::kStageVecs];
    float l[kRowsPerPass];
#pragma unroll
    for (int r = 0; r < kRowsPerPass; ++r) {
      const int row = blockIdx.x + (k + r) * kBlocks;
      if (row < p.rows) {
#pragma unroll
        for (int i = 0; i < S::kStageVecs; ++i)
          x[r][i] = p.out[static_cast<int64_t>(row) * S::kRowVecs + tid + i * S::kThreads];
        if (tid < kN * kHr) l[r] = p.lse[row * kN * kHr + tid];
      }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerPass; ++r) {
      const int row = blockIdx.x + (k + r) * kBlocks;
      if (row < p.rows) {
#pragma unroll
        for (int i = 0; i < S::kStageVecs; ++i)
          own[static_cast<int64_t>(row) * S::kRowVecs + tid + i * S::kThreads] = x[r][i];
        if (tid < kN * kHr) own_lse[row * kN * kHr + tid] = l[r];
      }
    }
  }
  // AITER's barriers are per block: block b reads only the rows block b staged on every rank.
  aiter::end_sync<kN>(p.sg, p.self_sg, p.rank);
  const int vec = tid % S::kSlotVecs;
  for (int k = 0; blockIdx.x + k * kBlocks < p.rows; k += kRowsPerPass) {
    const int row = blockIdx.x + (k + tid / S::kSlotVecs) * kBlocks;
    if (row < p.rows) {
      const int64_t at = static_cast<int64_t>(row) * S::kRowVecs + p.rank * S::kSlotVecs + vec;
      const int64_t at_lse = static_cast<int64_t>(row) * kN * kHr + p.rank * kHr + vec;
      Vec x[kN];
      float l[kN];
#pragma unroll
      for (int s = 0; s < kN; ++s)
        x[s] = reinterpret_cast<const Vec*>(p.scratch[s])[at];
      if (vec < kHr) {
#pragma unroll
        for (int s = 0; s < kN; ++s)
          l[s] = reinterpret_cast<const float*>(p.scratch[s] + S::kDataBytes)[at_lse];
      }
#pragma unroll
      for (int s = 0; s < kN; ++s)
        p.recv_out[(static_cast<int64_t>(s) * p.rows + row) * S::kSlotVecs + vec] = x[s];
      if (vec < kHr) {
#pragma unroll
        for (int s = 0; s < kN; ++s)
          p.recv_lse[(static_cast<int64_t>(s) * p.rows + row) * kHr + vec] = l[s];
      }
    }
  }
}

template <int kHr>
struct DcpIpcA2aHip {
  static void exchange(
      int64_t comm,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView lse,
      const tvm::ffi::TensorView recv_out,
      const tvm::ffi::TensorView recv_lse) {
    using namespace host;
    auto B = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLROCM>();
    TensorMatcher({B, kN * kHr, kD}).with_dtype<bf16_t>().with_device(device).verify(out);
    TensorMatcher({B, kN * kHr}).with_dtype<float>().with_device(device).verify(lse);
    TensorMatcher({kN, B, kHr, kD}).with_dtype<bf16_t>().with_device(device).verify(recv_out);
    TensorMatcher({kN, B, kHr}).with_dtype<float>().with_device(device).verify(recv_lse);
    RuntimeCheck(
        out.is_contiguous() && lse.is_contiguous() && recv_out.is_contiguous() && recv_lse.is_contiguous(),
        "buffers must be contiguous");
    const int64_t rows = B.unwrap();
    RuntimeCheck(rows <= kMaxRows, "the DCP IPC all-to-all takes up to ", kMaxRows, " rows");
    if (rows == 0) return;
    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kN, "the DCP IPC all-to-all requires DCP", kN);
    Params params{
        .out = static_cast<const Vec*>(out.data_ptr()),
        .lse = static_cast<const float*>(lse.data_ptr()),
        .recv_out = static_cast<Vec*>(recv_out.data_ptr()),
        .recv_lse = static_cast<float*>(recv_lse.data_ptr()),
        .scratch = {},
        .self_scratch = nullptr,
        .sg = communicator->sg_,
        .self_sg = communicator->self_sg_,
        .rank = communicator->rank_,
        .rows = static_cast<int32_t>(rows),
    };
    for (int i = 0; i < kN; ++i)
      params.scratch[i] = reinterpret_cast<const char*>(communicator->sg_.signals[i]) + sizeof(aiter::Signal);
    params.self_scratch = reinterpret_cast<char*>(communicator->self_sg_) + sizeof(aiter::Signal);
    LaunchKernel(kBlocks, Shape<kHr>::kThreads, device.unwrap())(exchange_kernel<kHr>, params);
  }
};

}  // namespace dcp_ipc_a2a
}  // namespace sglang
