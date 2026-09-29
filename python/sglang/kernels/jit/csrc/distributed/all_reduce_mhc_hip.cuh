// Built on AITER's custom all-reduce (csrc/include/custom_all_reduce.cuh, a ROCm port of vLLM's):
// a TP4 all-reduce of DeepSeek-V4.1's 1-8 row decode hidden states fused with hc_post, split
// over H, on the peer buffers and signals of AITER's CustomAllreduce (SGLang's ROCm custom
// all-reduce). Compiled with -ffp-contract=off: the unfused all-reduce + hc_post rounds every
// multiply and add.
#pragma once

#ifndef USE_ROCM
#error "all_reduce_mhc_hip.cuh runs on AITER's ROCm custom all-reduce"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <custom_all_reduce.cuh>  // AITER csrc/include: CustomAllreduce, RankData, start_sync, packed_reduce

namespace sglang {
namespace all_reduce_mhc_hip {

constexpr int kWorldSize = 4;
constexpr int kHc = 4;
constexpr int kHidden = 5120;
constexpr int kMaxRows = 8;
constexpr int kThreads = 64;
constexpr int kPack = 8;  // bf16 per 16-byte packet
constexpr int kPacks = kHidden / kPack;
constexpr int kBlocksPerRow = kPacks / kThreads;
static_assert(kBlocksPerRow * kThreads == kPacks, "a row's packets split evenly over the blocks");
static_assert(kMaxRows * kBlocksPerRow <= aiter::kMaxBlocks, "launch exceeds AITER signal slots");

using P = opus::vector_t<opus::bf16_t, kPack>;
using A = opus::vector_t<opus::fp32_t, kPack>;

__global__ void all_reduce_mhc_post_kernel(
    aiter::RankData* peers,
    aiter::RankSignals signals,
    aiter::Signal* local_signals,
    int rank,
    opus::bf16_t* output,
    const opus::bf16_t* residual,
    const float* post,
    const float* comb) {
  const int row = blockIdx.x / kBlocksPerRow;
  const int pack = (blockIdx.x % kBlocksPerRow) * kThreads + threadIdx.x;
  const P* inputs[kWorldSize];
#pragma unroll
  for (int r = 0; r < kWorldSize; ++r)
    inputs[r] = reinterpret_cast<const P*>(peers->ptrs[r]);
  aiter::start_sync<kWorldSize>(signals, local_signals, rank);
  const P reduced = aiter::packed_reduce<P, kWorldSize, A>(inputs, row * kPacks + pack);
#pragma unroll
  for (int out_h = 0; out_h < kHc; ++out_h) {
    A mixed;
#pragma unroll
    for (int v = 0; v < kPack; ++v)
      mixed[v] = aiter::upcast_s(reduced[v]) * post[row * kHc + out_h];
#pragma unroll
    for (int in_h = 0; in_h < kHc; ++in_h) {
      const P values = reinterpret_cast<const P*>(residual)[(row * kHc + in_h) * kPacks + pack];
      const float coefficient = comb[(row * kHc + in_h) * kHc + out_h];
#pragma unroll
      for (int v = 0; v < kPack; ++v)
        mixed[v] += aiter::upcast_s(values[v]) * coefficient;
    }
    reinterpret_cast<P*>(output)[(row * kHc + out_h) * kPacks + pack] = aiter::downcast<P>(mixed);
  }
  aiter::end_sync<kWorldSize, true>(signals, local_signals, rank);
}

struct AllReduceMhcPostKernel {
  // `comm` is AITER's CustomAllreduce*; `registered` is its registered input pool, into which the
  // input is staged outside graph capture (0 while capturing: capture registers the input itself)
  static void
  run(int64_t comm,
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView post,
      const tvm::ffi::TensorView comb,
      int64_t registered,
      int64_t registered_bytes) {
    using namespace host;
    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHidden}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({M, kHc, kHidden}).with_dtype<bf16_t>().with_device(device).verify(residual).verify(output);
    TensorMatcher({M, kHc}).with_dtype<float>().with_device(device).verify(post);
    TensorMatcher({M, kHc, kHc}).with_dtype<float>().with_device(device).verify(comb);
    const int64_t rows = M.unwrap();
    RuntimeCheck(1 <= rows && rows <= kMaxRows, "all_reduce_mhc_post takes 1-", kMaxRows, " rows, got ", rows);

    auto* communicator = reinterpret_cast<aiter::CustomAllreduce*>(comm);
    RuntimeCheck(communicator->world_size_ == kWorldSize, "all_reduce_mhc_post requires TP", kWorldSize);
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    void* values = input.data_ptr();
    if (registered != 0) {
      const int64_t bytes = rows * kHidden * static_cast<int64_t>(sizeof(bf16_t));
      RuntimeCheck(bytes <= registered_bytes, "all-reduce input pool is too small");
      values = reinterpret_cast<void*>(registered);
      RuntimeDeviceCheck(hipMemcpyAsync(values, input.data_ptr(), bytes, hipMemcpyDeviceToDevice, stream));
    }
    aiter::RankData* peers = communicator->get_buffer_RD(stream, values);
    LaunchKernel(static_cast<uint32_t>(rows * kBlocksPerRow), kThreads, stream)(
        all_reduce_mhc_post_kernel,
        peers,
        communicator->sg_,
        communicator->self_sg_,
        communicator->rank_,
        static_cast<opus::bf16_t*>(output.data_ptr()),
        static_cast<const opus::bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(post.data_ptr()),
        static_cast<const float*>(comb.data_ptr()));
  }
};

}  // namespace all_reduce_mhc_hip
}  // namespace sglang
