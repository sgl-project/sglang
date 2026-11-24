// SPDX-License-Identifier: Apache-2.0
// RoutedExchange combine: grid barrier with every peer, then each token's
// top-k rows summed in top-k order (FP32), each row read from its owner rank.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "../inkling/inkling_ar_barrier.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_exchange {

constexpr int threads(int hidden) {
  return (hidden / 16 + 31) / 32 * 32;
}

template <int H, int TOPK, int WORLD, bool USE_PDL>
__global__ void __launch_bounds__(threads(H)) combine_kernel(
    int64_t const* __restrict__ buffer_ptrs,
    int64_t const* __restrict__ flag_ptrs,
    uint32_t* state,
    int const* __restrict__ src2dst,
    int const* __restrict__ sorted_ids,
    __nv_bfloat16* out,
    int64_t* calls,
    int tokens,
    int64_t slot_size,
    int rank,
    int num_local) {
  __shared__ __nv_bfloat16 const* rows[TOPK];
  device::PDLWaitPrimary<USE_PDL>();
  device::PDLTriggerSecondary<USE_PDL>();
  int64_t const call = calls[0];
  int64_t const slot = (call & 1) * slot_size;
  inkling_ar::grid_system_barrier<WORLD>(state, reinterpret_cast<void* const*>(flag_ptrs), rank, 0, false);
  bool const active = threadIdx.x < H / 16;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    if (threadIdx.x < TOPK) {
      int dst = src2dst[token * TOPK + threadIdx.x];
      int owner = sorted_ids[dst] / num_local;
      rows[threadIdx.x] = reinterpret_cast<__nv_bfloat16 const*>(buffer_ptrs[owner]) + slot + int64_t(dst) * H;
    }
    __syncthreads();
    if (active) {
      uint4 raw[TOPK][2];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        uint4 const* src = reinterpret_cast<uint4 const*>(rows[k] + threadIdx.x * 16);
        raw[k][0] = src[0];
        raw[k][1] = src[1];
      }
      float total[16] = {};
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(raw[k]);
#pragma unroll
        for (int i = 0; i < 16; ++i)
          total[i] = __fadd_rn(total[i], __bfloat162float(h[i]));
      }
      uint4 packed[2];
      __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(packed);
#pragma unroll
      for (int i = 0; i < 16; ++i)
        p[i] = __float2bfloat16_rn(total[i]);
      uint4* dst = reinterpret_cast<uint4*>(out + int64_t(token) * H + threadIdx.x * 16);
      dst[0] = packed[0];
      dst[1] = packed[1];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && atomicAdd(reinterpret_cast<unsigned long long*>(calls + 1), 1ull) == gridDim.x - 1) {
    calls[1] = 0;
    calls[0] = call + 1;
  }
}

template <int H, int TOPK, int WORLD, bool USE_PDL>
void combine(
    tvm::ffi::TensorView buffer_ptrs,
    tvm::ffi::TensorView flag_ptrs,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView src2dst,
    tvm::ffi::TensorView sorted_ids,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView calls,
    int64_t blocks,
    int64_t slot_size,
    int64_t rank,
    int64_t num_local) {
  static_assert(H % 16 == 0 && H / 16 <= 1024);
  int tokens = out.size(0);
  host::RuntimeCheck(out.size(1) == H && src2dst.size(0) == tokens * TOPK, "Unsupported routed exchange shape");
  host::LaunchKernel(blocks, threads(H), out.device())
      .enable_pdl(USE_PDL)(
          combine_kernel<H, TOPK, WORLD, USE_PDL>,
          static_cast<int64_t const*>(buffer_ptrs.data_ptr()),
          static_cast<int64_t const*>(flag_ptrs.data_ptr()),
          static_cast<uint32_t*>(state.data_ptr()),
          static_cast<int const*>(src2dst.data_ptr()),
          static_cast<int const*>(sorted_ids.data_ptr()),
          static_cast<__nv_bfloat16*>(out.data_ptr()),
          static_cast<int64_t*>(calls.data_ptr()),
          tokens,
          slot_size,
          int(rank),
          int(num_local));
}

}  // namespace sglang::inkling2_exchange
