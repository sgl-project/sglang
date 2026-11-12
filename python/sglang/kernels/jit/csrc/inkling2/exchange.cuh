// SPDX-License-Identifier: Apache-2.0
// RoutedExchange combine: per-CTA flag barrier with every peer, then each token's
// top-k rows summed in top-k order (FP32), each row read from its owner rank.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace sglang::inkling2_exchange {

constexpr int threads(int hidden) {
  return (hidden / 16 + 31) / 32 * 32;
}

// Grid-wide barrier with every peer: the last CTA to arrive exchanges epoch
// flags with the peers (one thread per peer), then releases the other CTAs.
// state: [arrivals, releases, epoch]; flags: one word per peer.
template <int WORLD>
__device__ __forceinline__ void grid_barrier(uint32_t* state, int64_t const* flag_ptrs, int rank) {
  __shared__ uint32_t s_released, s_epoch;
  __shared__ bool s_leader;
  if (threadIdx.x == 0) {
    s_released = *reinterpret_cast<uint32_t volatile*>(state + 1);
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(state), "r"(gridDim.x - 1)
                 : "memory");
    s_leader = old == gridDim.x - 1;
    if (s_leader) {
      s_epoch = state[2] + 1;
      state[2] = s_epoch;
      __threadfence_system();
    }
  }
  __syncthreads();
  if (s_leader) {
    if (threadIdx.x < WORLD) {
      int const peer = threadIdx.x;
      uint32_t* flag = reinterpret_cast<uint32_t*>(flag_ptrs[peer]) + rank;
      asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(flag), "r"(s_epoch) : "memory");
      uint32_t const* own = reinterpret_cast<uint32_t const*>(flag_ptrs[rank]) + peer;
      uint32_t seen;
      do {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(own) : "memory");
      } while (static_cast<int32_t>(seen - s_epoch) < 0);
    }
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(state + 1), "r"(1u) : "memory");
  } else if (threadIdx.x == 0) {
    uint32_t seen;
    do {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(state + 1) : "memory");
    } while (seen == s_released);
  }
  __syncthreads();
}

template <int H, int TOPK, int WORLD>
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
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  int const block = blockIdx.x;
  int64_t const call = calls[0];
  int64_t const slot = (call & 1) * slot_size;
  grid_barrier<WORLD>(state, flag_ptrs, rank);
  bool const active = threadIdx.x < H / 16;
  for (int token = block; token < tokens; token += gridDim.x) {
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
      float total[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        total[i] = 0.0f;
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

template <int H, int TOPK, int WORLD>
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
      .enable_pdl(true)(
          combine_kernel<H, TOPK, WORLD>,
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
