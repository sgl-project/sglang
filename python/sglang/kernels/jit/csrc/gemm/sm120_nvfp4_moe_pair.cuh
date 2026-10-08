/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "sm120_nvfp4_moe_mtp_mma.cuh"

namespace sglang {
namespace nvfp4_pair_mma {
using namespace nvfp4_mtp_mma;
// Store each original pair of FP4 weight bits across contiguous output channels.
// One 16-byte fetch then corresponds to exactly two activation values, enabling
// zero-filled asynchronous loads without discarding any nonzero multiplication.
template <unsigned N, unsigned K>
__device__ __forceinline__ void
pair_prefetch(uint8_t* shared, const uint8_t* weights, unsigned tile, unsigned step, unsigned lane, const uint32_t* z) {
#pragma unroll
  for (unsigned j = 0; j < 2; ++j) {
    unsigned pair = lane + 32 * j, source_pair = step * 64 + pair;
    unsigned bytes = source_pair < K / 2 ? 16 : 0;
    if (bytes && (reinterpret_cast<const uint8_t*>(z)[source_pair] & 0x77u) == 0) bytes = 0;
    const uint8_t* src = weights + (bytes ? source_pair : 0) * N + tile;
    unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared + pair * 16));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void
pair_load_a(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3, const uint8_t* s, unsigned lane) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(s + lane * 16));
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(addr)
               : "memory");
}
template <uint32_t N, uint32_t K, uint32_t TopK, bool PDL, bool ZeroLoad = false>
__global__ __launch_bounds__(128) void fc2(
    const uint32_t* __restrict__ z,
    const uint8_t* __restrict__ zs,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sf,
    const float* __restrict__ alpha,
    const int32_t* __restrict__ ids,
    const float* __restrict__ routing,
    bf16_t* __restrict__ partial) {
  __shared__ __align__(16) uint8_t ws[4][2][1024];
  uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint32_t token = blockIdx.z, slot = blockIdx.y, route = token * TopK + slot;
  PDLTriggerSecondary<PDL>();
  PDLWaitPrimary<PDL>();
  uint32_t expert = ids[route], tile = (blockIdx.x * 4 + warp) * 16;
  uint32_t row = tile + lane / 4, col = (lane % 4) * 4;
  const uint8_t* weights = w + static_cast<size_t>(expert) * N * K / 2;
  const uint8_t* scales = sf + static_cast<size_t>(expert) * N * (K / 16);
  z += route * (K / 8);
  zs += route * (K / 16);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
  for (uint32_t stage = 0; stage < 2; ++stage)
    pair_prefetch<N, K>(ws[warp][stage], weights, tile, stage, lane, z);
#pragma unroll 1
  for (uint32_t base = 0; base < K; base += 128) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (uint32_t inner = 0; inner < 128; inner += 64) {
      if (base + inner >= K) break;
      uint32_t k = base + inner;
      uint32_t a0, a1, a2, a3;
      pair_load_a(a0, a1, a2, a3, ws[warp][(base / 128) % 2] + (inner / 2) * 16, lane);
      uint32_t sa =
          *reinterpret_cast<const uint32_t*>(scales + sf_index<K / 16>(tile + lane / 4 + (lane % 2) * 8, k / 16));
      uint32_t sb = *reinterpret_cast<const uint32_t*>(zs + k / 16);
      mma(d0, d1, d2, d3, a0, a1, a2, a3, z[k / 8 + lane % 4], z[k / 8 + lane % 4 + 4], sa, sb);
    }
    __syncwarp();
    pair_prefetch<N, K>(ws[warp][(base / 128) % 2], weights, tile, base / 128 + 2, lane, z);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 == 0) {
    partial[route * N + row] = cast<bf16_t>((d0 * alpha[expert]) * routing[route]);
    partial[route * N + row + 8] = cast<bf16_t>((d2 * alpha[expert]) * routing[route]);
  }
}
}  // namespace nvfp4_pair_mma

template <uint32_t N1, uint32_t K1, uint32_t N2, uint32_t K2, uint32_t TopK, bool PDL, bool ZeroLoad = false>
struct Sm120Nvfp4MoePair {
  static_assert(N1 == 2688 && K1 == 1024 && N2 == 1024 && K2 == 2688 && TopK == 22);
  /// Run 1 to 4 token rows with BF16 epilogues and deterministic reduction.
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView w1,
      tvm::ffi::TensorView s1,
      tvm::ffi::TensorView w2,
      tvm::ffi::TensorView s2,
      tvm::ffi::TensorView a1,
      tvm::ffi::TensorView a2,
      tvm::ffi::TensorView g1,
      tvm::ffi::TensorView g2,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView routing,
      tvm::ffi::TensorView z,
      tvm::ffi::TensorView zs,
      tvm::ffi::TensorView partial,
      tvm::ffi::TensorView out) {
    using namespace host;
    using namespace device;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"token_rows"};
    TensorMatcher({rows, K1}).with_dtype<bf16_t>().with_device(dev).verify(x);
    CHECK_HOST(rows.unwrap() >= 1 && rows.unwrap() <= 4) << "Expected 1 <= M <= 4";
    const uint32_t M = static_cast<uint32_t>(rows.unwrap());
    TensorMatcher({512 * N1 * K1 / 2}).with_dtype<uint8_t>().with_device(dev).verify(w1);
    TensorMatcher({512 * N2 * K2 / 2}).with_dtype<uint8_t>().with_device(dev).verify(w2);
    TensorMatcher({512 * N1 * K1 / 16}).with_dtype<uint8_t>().with_device(dev).verify(s1);
    TensorMatcher({512 * N2 * K2 / 16}).with_dtype<uint8_t>().with_device(dev).verify(s2);
    TensorMatcher({512}).with_dtype<float>().with_device(dev).verify(a1).verify(a2);
    TensorMatcher({1}).with_dtype<float>().with_device(dev).verify(g1).verify(g2);
    TensorMatcher({M * TopK}).with_dtype<int32_t>().with_device(dev).verify(ids);
    TensorMatcher({M * TopK}).with_dtype<float>().with_device(dev).verify(routing);
    TensorMatcher({M * TopK * N1 / 8}).with_dtype<int32_t>().with_device(dev).verify(z);
    TensorMatcher({M * TopK * N1 / 16}).with_dtype<uint8_t>().with_device(dev).verify(zs);
    TensorMatcher({M * TopK * N2}).with_dtype<bf16_t>().with_device(dev).verify(partial);
    TensorMatcher({M, N2}).with_dtype<bf16_t>().with_device(dev).verify(out);
    LaunchKernel({N1 / 64, TopK, M}, 128, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_mtp_mma::fc1<N1, K1, TopK, PDL>,
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const uint8_t*>(w1.data_ptr()),
            static_cast<const uint8_t*>(s1.data_ptr()),
            static_cast<const float*>(a1.data_ptr()),
            static_cast<const float*>(g1.data_ptr()),
            static_cast<const float*>(g2.data_ptr()),
            static_cast<const int32_t*>(ids.data_ptr()),
            static_cast<uint32_t*>(z.data_ptr()),
            static_cast<uint8_t*>(zs.data_ptr()));
    LaunchKernel({N2 / 64, TopK, M}, 128, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_pair_mma::fc2<N2, K2, TopK, PDL, ZeroLoad>,
            static_cast<const uint32_t*>(z.data_ptr()),
            static_cast<const uint8_t*>(zs.data_ptr()),
            static_cast<const uint8_t*>(w2.data_ptr()),
            static_cast<const uint8_t*>(s2.data_ptr()),
            static_cast<const float*>(a2.data_ptr()),
            static_cast<const int32_t*>(ids.data_ptr()),
            static_cast<const float*>(routing.data_ptr()),
            static_cast<bf16_t*>(partial.data_ptr()));
    LaunchKernel({(M * N2 + 255) / 256}, 256, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_mtp_mma::reduce<N2, TopK, PDL>,
            static_cast<const bf16_t*>(partial.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            M);
  }
};
}  // namespace sglang
