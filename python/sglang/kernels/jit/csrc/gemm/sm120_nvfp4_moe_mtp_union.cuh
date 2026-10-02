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

// Reuse each selected expert weight across up to 16 token rows. Every
// activation and selected expert contribution is computed at its original precision.
#include "sm120_nvfp4_moe_mtp_mma.cuh"

namespace sglang {
namespace nvfp4_mtp_union {
using namespace nvfp4_mtp_mma;

template <uint32_t K>
__global__ void quantize_input(
    const bf16_t* __restrict__ x,
    const float* __restrict__ global,
    uint32_t* __restrict__ q,
    uint8_t* __restrict__ sf,
    uint32_t rows) {
  uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group < rows * K / 16) {
    float v[16];
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i)
      v[i] = static_cast<float>(x[group * 16 + i]);
    quantize16(v, q[2 * group], q[2 * group + 1], sf[group], *global);
  }
}

// Each expert's first route owns its joint computation. Its other routes are
// written by that owner, with their own activations, scales and routing weights.
template <uint32_t TopK, uint32_t Slots>
__device__ __forceinline__ bool match_routes(const int32_t* ids, int32_t* routes, uint32_t route, uint32_t rows) {
  uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int32_t expert = ids[route];
#pragma unroll
  for (uint32_t t = warp; t < Slots; t += 4) {
    bool match = t < rows && lane < TopK && ids[t * TopK + lane] == expert;
    uint32_t mask = __ballot_sync(0xffffffff, match);
    if (lane == 0) routes[t] = mask ? static_cast<int32_t>(t * TopK + __ffs(mask) - 1) : -1;
  }
  __syncthreads();
  int32_t first = Slots * TopK;
#pragma unroll
  for (uint32_t t = 0; t < Slots; ++t)
    if (routes[t] >= 0) first = min(first, routes[t]);
  return first == route;
}

template <uint32_t N, uint32_t K, uint32_t TopK, uint32_t Slots, bool PDL>
__global__ __launch_bounds__(128) void fc1(
    const uint32_t* __restrict__ xq,
    const uint8_t* __restrict__ xs,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sf,
    const float* __restrict__ alpha,
    const float* __restrict__ global2,
    const int32_t* __restrict__ ids,
    uint32_t* __restrict__ z,
    uint8_t* __restrict__ zsf,
    uint32_t rows) {
  __shared__ float act[4][Slots][16];
  __shared__ int32_t routes[Slots];
  __shared__ __align__(16) uint8_t ws[4][2][2304];
  uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint32_t token = blockIdx.z, route = token * TopK + blockIdx.y;
  PDLTriggerSecondary<PDL>();
  PDLWaitPrimary<PDL>();
  if (!match_routes<TopK, Slots>(ids, routes, route, rows)) return;
  uint32_t expert = ids[route], tile = (blockIdx.x * 4 + warp) * 16;
  const uint8_t* weights = w + static_cast<size_t>(expert) * N * K / 2;
  const uint8_t* scales = sf + static_cast<size_t>(expert) * N * (K / 16);
  float d[Slots / 8][4] = {};
#pragma unroll
  for (uint32_t stage = 0; stage < 2; ++stage)
    prefetch<K>(ws[warp][stage], weights, tile, stage, lane);
#pragma unroll 1
  for (uint32_t base = 0; base < K; base += 256) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (uint32_t inner = 0; inner < 256; inner += 64) {
      if (base + inner >= K) break;
      uint32_t k = base + inner, a0, a1, a2, a3;
      load_a(a0, a1, a2, a3, ws[warp][(base / 256) % 2] + inner / 2, lane);
      uint32_t sa =
          *reinterpret_cast<const uint32_t*>(scales + sf_index<K / 16>(tile + lane / 4 + (lane % 2) * 8, k / 16));
#pragma unroll
      for (uint32_t g = 0; g < Slots / 8; ++g) {
        uint32_t bt = 8 * g + lane / 4;
        if (bt >= rows) bt = token;
        uint32_t sb = *reinterpret_cast<const uint32_t*>(xs + bt * (K / 16) + k / 16);
        mma(d[g][0],
            d[g][1],
            d[g][2],
            d[g][3],
            a0,
            a1,
            a2,
            a3,
            xq[bt * (K / 8) + k / 8 + lane % 4],
            xq[bt * (K / 8) + k / 8 + lane % 4 + 4],
            sa,
            sb);
      }
    }
    __syncwarp();
    prefetch<K>(ws[warp][(base / 256) % 2], weights, tile, base / 256 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
  for (uint32_t g = 0; g < Slots / 8; ++g) {
    uint32_t ct = 8 * g + 2 * (lane % 4);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      float y = static_cast<float>(cast<bf16_t>(d[g][i] * alpha[expert]));
      y = fmaxf(y, 0.f);
      act[warp][ct + i % 2][lane / 4 + (i / 2) * 8] = static_cast<float>(cast<bf16_t>(y * y));
    }
  }
  __syncwarp();
  if (lane < Slots && routes[lane] >= 0) {
    size_t qoff = routes[lane] * (N / 8) + tile / 8;
    size_t soff = routes[lane] * (N / 16) + tile / 16;
    quantize16(act[warp][lane], z[qoff], z[qoff + 1], zsf[soff], *global2);
  }
}

template <uint32_t N, uint32_t K, uint32_t TopK, uint32_t Slots, bool PDL>
__global__ __launch_bounds__(128) void fc2(
    const uint32_t* __restrict__ z,
    const uint8_t* __restrict__ zs,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sf,
    const float* __restrict__ alpha,
    const int32_t* __restrict__ ids,
    const float* __restrict__ routing,
    bf16_t* __restrict__ partial,
    uint32_t rows) {
  __shared__ __align__(16) uint8_t ws[4][2][2304];
  __shared__ int32_t routes[Slots];
  uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint32_t route = blockIdx.z * TopK + blockIdx.y;
  PDLTriggerSecondary<PDL>();
  PDLWaitPrimary<PDL>();
  if (!match_routes<TopK, Slots>(ids, routes, route, rows)) return;
  uint32_t expert = ids[route], tile = (blockIdx.x * 4 + warp) * 16;
  const uint8_t* weights = w + static_cast<size_t>(expert) * N * K / 2;
  const uint8_t* scales = sf + static_cast<size_t>(expert) * N * (K / 16);
  float d[Slots / 8][4] = {};
#pragma unroll
  for (uint32_t stage = 0; stage < 2; ++stage)
    prefetch<K>(ws[warp][stage], weights, tile, stage, lane);
#pragma unroll 1
  for (uint32_t base = 0; base < K; base += 256) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (uint32_t inner = 0; inner < 256; inner += 64) {
      if (base + inner >= K) break;
      uint32_t k = base + inner, a0, a1, a2, a3;
      load_a(a0, a1, a2, a3, ws[warp][(base / 256) % 2] + inner / 2, lane);
      uint32_t sa =
          *reinterpret_cast<const uint32_t*>(scales + sf_index<K / 16>(tile + lane / 4 + (lane % 2) * 8, k / 16));
#pragma unroll
      for (uint32_t g = 0; g < Slots / 8; ++g) {
        uint32_t bt = 8 * g + lane / 4;
        uint32_t broute = routes[bt] >= 0 ? routes[bt] : route;
        uint32_t sb = *reinterpret_cast<const uint32_t*>(zs + broute * (K / 16) + k / 16);
        mma(d[g][0],
            d[g][1],
            d[g][2],
            d[g][3],
            a0,
            a1,
            a2,
            a3,
            z[broute * (K / 8) + k / 8 + lane % 4],
            z[broute * (K / 8) + k / 8 + lane % 4 + 4],
            sa,
            sb);
      }
    }
    __syncwarp();
    prefetch<K>(ws[warp][(base / 256) % 2], weights, tile, base / 256 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
  for (uint32_t g = 0; g < Slots / 8; ++g) {
    uint32_t ct = 8 * g + 2 * (lane % 4);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      int32_t cr = routes[ct + i % 2];
      if (cr >= 0)
        partial[cr * N + tile + lane / 4 + (i / 2) * 8] = cast<bf16_t>((d[g][i] * alpha[expert]) * routing[cr]);
    }
  }
}
}  // namespace nvfp4_mtp_union
template <uint32_t N1, uint32_t K1, uint32_t N2, uint32_t K2, uint32_t TopK, uint32_t Slots, bool PDL>
struct Sm120Nvfp4MoeMtpUnion {
  static_assert(Slots == 8 || Slots == 16);
  static_assert(N1 == 2688 && K1 == 1024 && N2 == 1024 && K2 == 2688 && TopK == 22);
  /// Run 1 to 16 token rows with BF16 epilogues and deterministic reduction.
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
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView xq,
      tvm::ffi::TensorView xs) {
    using namespace host;
    using namespace device;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"token_rows"};
    TensorMatcher({rows, K1}).with_dtype<bf16_t>().with_device(dev).verify(x);
    CHECK_HOST(rows.unwrap() >= 1 && rows.unwrap() <= Slots) << "Expected 1 <= M <= 16";
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
    TensorMatcher({M * K1 / 8}).with_dtype<int32_t>().with_device(dev).verify(xq);
    TensorMatcher({M * K1 / 16}).with_dtype<uint8_t>().with_device(dev).verify(xs);
    LaunchKernel({(M * K1 / 16 + 127) / 128}, 128, dev.unwrap())(
        nvfp4_mtp_union::quantize_input<K1>,
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const float*>(g1.data_ptr()),
        static_cast<uint32_t*>(xq.data_ptr()),
        static_cast<uint8_t*>(xs.data_ptr()),
        M);
    LaunchKernel({N1 / 64, TopK, M}, 128, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_mtp_union::fc1<N1, K1, TopK, Slots, PDL>,
            static_cast<const uint32_t*>(xq.data_ptr()),
            static_cast<const uint8_t*>(xs.data_ptr()),
            static_cast<const uint8_t*>(w1.data_ptr()),
            static_cast<const uint8_t*>(s1.data_ptr()),
            static_cast<const float*>(a1.data_ptr()),
            static_cast<const float*>(g2.data_ptr()),
            static_cast<const int32_t*>(ids.data_ptr()),
            static_cast<uint32_t*>(z.data_ptr()),
            static_cast<uint8_t*>(zs.data_ptr()),
            M);
    LaunchKernel({N2 / 64, TopK, M}, 128, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_mtp_union::fc2<N2, K2, TopK, Slots, PDL>,
            static_cast<const uint32_t*>(z.data_ptr()),
            static_cast<const uint8_t*>(zs.data_ptr()),
            static_cast<const uint8_t*>(w2.data_ptr()),
            static_cast<const uint8_t*>(s2.data_ptr()),
            static_cast<const float*>(a2.data_ptr()),
            static_cast<const int32_t*>(ids.data_ptr()),
            static_cast<const float*>(routing.data_ptr()),
            static_cast<bf16_t*>(partial.data_ptr()),
            M);
    LaunchKernel({(M * N2 + 255) / 256}, 256, dev.unwrap())
        .enable_pdl(PDL)(
            nvfp4_mtp_mma::reduce<N2, TopK, PDL>,
            static_cast<const bf16_t*>(partial.data_ptr()),
            static_cast<bf16_t*>(out.data_ptr()),
            M);
  }
};
}  // namespace sglang
