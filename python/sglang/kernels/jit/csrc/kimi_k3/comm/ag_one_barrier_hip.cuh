// K3 ROCm (TP8): AITER's allgather_lastdim_add with one cross-GPU barrier.
//
// AITER's kernel (csrc/include/custom_all_reduce.cuh) is start_sync, remote
// reads of every peer's registered input, end_sync. The trailing end_sync only
// keeps a rank from overwriting its input while a slower peer still reads it.
// Here the caller instead keeps the input alive (the graph allocator cannot
// reuse it) until after the next collective, whose start barrier already
// implies every peer finished this kernel (kernels on a stream are ordered).
// The per-element math is AITER's verbatim, so the output is bit-identical:
//   out[row, s*ld + x] = bf16(bf16(y_s[row, x] + b) [+ c]).
// AITER's per-block _flag counters stay consistent across ranks: every rank
// launches the same kernel sequence with the same grids.
//
// Graph capture only: eager calls stage the input through AITER's shared
// input pool, which the next collective overwrites before its barrier.
#pragma once

#ifndef USE_ROCM
#error "ag_one_barrier_hip.cuh is ROCm only"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <custom_all_reduce.cuh>  // AITER: CustomAllreduce, RankData, start_sync, packed_assign_add

namespace sglang {
namespace k3_ag_one_barrier {

constexpr int kNG = 8;
constexpr int kPack = 8;
using T16 = opus::bf16_t;
using P = opus::vector_t<T16, kPack>;
using A = opus::vector_t<opus::fp32_t, kPack>;

template <bool kHasC>
__global__ void __launch_bounds__(512, 1) ag_add_one_barrier_kernel(
    aiter::RankData* _dp,
    aiter::RankSignals sg,
    aiter::Signal* self_sg,
    T16* __restrict__ result,
    int rank,
    int size,  // packs in one rank's input
    int last_dim_size,
    const T16* __restrict__ add_b,
    const T16* __restrict__ add_c) {
  constexpr int tnum_gpu = 512 / kNG;
  const int warp_id = threadIdx.x / tnum_gpu;
  const int lane_id = threadIdx.x % tnum_gpu;
  const int tid = blockIdx.x * tnum_gpu + lane_id;
  const int stride = gridDim.x * tnum_gpu;
  last_dim_size /= kPack;
  const P* ptrs[kNG];
#pragma unroll
  for (int i = 0; i < kNG; ++i)
    ptrs[i] = reinterpret_cast<const P*>(_dp->ptrs[i]);
  const P* bp = reinterpret_cast<const P*>(add_b);
  const P* cp = reinterpret_cast<const P*>(add_c);
  aiter::start_sync<kNG>(sg, self_sg, rank);
  for (int idx = tid; idx < size; idx += stride) {
    const int y = idx / last_dim_size;
    const int x = idx % last_dim_size;
    const int write_idx = (kNG * y + warp_id) * last_dim_size + x;
    A acc = aiter::upcast(ptrs[warp_id][idx]);
    aiter::packed_assign_add<opus::fp32_t, kPack>(acc, aiter::upcast(bp[write_idx]));
    P o = aiter::downcast<P>(acc);
    if constexpr (kHasC) {
      A acc2 = aiter::upcast(o);
      aiter::packed_assign_add<opus::fp32_t, kPack>(acc2, aiter::upcast(cp[write_idx]));
      o = aiter::downcast<P>(acc2);
    }
    reinterpret_cast<P*>(result)[write_idx] = o;
  }
  // no end_sync: see the file comment
}

struct AgOneBarrier {
  // `comm` is AITER's CustomAllreduce*; y must be a graph-capture tensor (its
  // peer addresses are registered when AITER's capture() scope exits)
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView add_b,
      const tvm::ffi::TensorView add_c) {  // numel 0 -> no c
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto L = SymbolicSize{"ld"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R, L}).with_dtype<bf16_t>().with_device(device).verify(y);
    const int64_t rows = R.unwrap(), ld = L.unwrap();
    TensorMatcher({rows, ld * kNG}).with_dtype<bf16_t>().with_device(device).verify(output).verify(add_b);
    const bool has_c = add_c.numel() > 0;
    if (has_c) TensorMatcher({rows, ld * kNG}).with_dtype<bf16_t>().with_device(device).verify(add_c);
    RuntimeCheck(ld % kPack == 0, "all-gather rows must be 16-byte aligned");
    auto* ca = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(ca->world_size_ == kNG, "one-barrier all-gather is TP8 only");
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    hipStreamCaptureStatus status;
    RuntimeDeviceCheck(hipStreamIsCapturing(stream, &status));
    RuntimeCheck(status == hipStreamCaptureStatusActive, "one-barrier all-gather is graph-capture only");
    aiter::RankData* rd = ca->get_buffer_RD(stream, y.data_ptr());
    const int size = int(rows * ld / kPack);
    const int blocks = std::min((size + 63) / 64, 80);  // AITER's grid
    const T16* c = has_c ? static_cast<const T16*>(add_c.data_ptr()) : nullptr;
    auto* out = static_cast<T16*>(output.data_ptr());
    const auto* b = static_cast<const T16*>(add_b.data_ptr());
    if (has_c)
      hipLaunchKernelGGL(
          (ag_add_one_barrier_kernel<true>),
          dim3(blocks),
          dim3(512),
          0,
          stream,
          rd,
          ca->sg_,
          ca->self_sg_,
          out,
          ca->rank_,
          size,
          int(ld),
          b,
          c);
    else
      hipLaunchKernelGGL(
          (ag_add_one_barrier_kernel<false>),
          dim3(blocks),
          dim3(512),
          0,
          stream,
          rd,
          ca->sg_,
          ca->self_sg_,
          out,
          ca->rank_,
          size,
          int(ld),
          b,
          c);
    RuntimeDeviceCheck(hipGetLastError());
  }
};

}  // namespace k3_ag_one_barrier
}  // namespace sglang
