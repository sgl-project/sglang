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

// Native SM120 NVFP4 MMA operand layout follows NVIDIA CUTLASS mma_sm120.hpp
// and mma_traits_sm120.hpp, distributed under BSD-3-Clause. Public reference
// copies at CUTLASS v4.8.0:
// https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cute/arch/mma_sm120.hpp
// https://github.com/NVIDIA/cutlass/blob/098de2a652cf8f00fd70b2df54051c7eccbb855a/include/cute/atom/mma_traits_sm120.hpp
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {
namespace nvfp4_mtp_mma {
using namespace device;

__device__ __forceinline__ float rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t fp4x2(float hi, float lo) {
  uint32_t d;
  asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ void quantize16(const float* v, uint32_t& q0, uint32_t& q1, uint8_t& sf, float global) {
  float amax = 0.f;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  __nv_fp8_e4m3 f(global * (amax * rcp(6.f)));
  sf = f.__x;
  float scale = amax != 0.f ? rcp(static_cast<float>(f) * rcp(global)) : 0.f;
  q0 = 0;
  q1 = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    q0 |= fp4x2(v[2 * i + 1] * scale, v[2 * i] * scale) << (8 * i);
    q1 |= fp4x2(v[8 + 2 * i + 1] * scale, v[8 + 2 * i] * scale) << (8 * i);
  }
}
template <uint32_t G>
__device__ __forceinline__ uint32_t sf_index(uint32_t row, uint32_t group) {
  return (row / 128) * (128 * G) + (row % 32) * 16 + ((row % 128) / 32) * 4 + (group / 4) * 512 + group % 4;
}
__device__ __forceinline__ void
mma(float& d0,
    float& d1,
    float& d2,
    float& d3,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t b0,
    uint32_t b1,
    uint32_t sfa,
    uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// Stage 256 K elements in contiguous 128B row segments. Padding each shared
// row by 16B avoids the eight-way ldmatrix bank conflict at a 128B stride.
template <uint32_t K>
__device__ __forceinline__ void
prefetch(uint8_t* shared, const uint8_t* weights, uint32_t tile, uint32_t step, uint32_t lane) {
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    uint32_t row = lane / 8 + 4 * j, col = (lane % 8) * 16;
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(shared + row * 144 + col));
    uint32_t source_col = step * 128 + col;
    uint32_t bytes = source_col < K / 2 ? 16 : 0;
    const uint8_t* src = weights + (tile + row) * (K / 2) + (bytes ? source_col : 0);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void
load_a(uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, const uint8_t* shared, uint32_t lane) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared + (lane % 16) * 144 + (lane / 16) * 16));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
               : "r"(addr)
               : "memory");
}

template <uint32_t N, uint32_t K, uint32_t TopK, bool PDL>
__global__ __launch_bounds__(128) void fc1(
    const bf16_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sf,
    const float* __restrict__ alpha,
    const float* __restrict__ global1,
    const float* __restrict__ global2,
    const int32_t* __restrict__ ids,
    uint32_t* __restrict__ z,
    uint8_t* __restrict__ zsf) {
  __shared__ uint32_t xq[K / 8];
  __shared__ uint8_t xs[K / 16];
  __shared__ float act[4][16];
  __shared__ __align__(16) uint8_t ws[4][2][2304];
  uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t token = blockIdx.z, slot = blockIdx.y;
  PDLTriggerSecondary<PDL>();
  PDLWaitPrimary<PDL>();
  if (tid < K / 16) {
    float v[16];
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i)
      v[i] = static_cast<float>(x[token * K + tid * 16 + i]);
    quantize16(v, xq[2 * tid], xq[2 * tid + 1], xs[tid], *global1);
  }
  __syncthreads();
  uint32_t expert = ids[token * TopK + slot];
  uint32_t tile = (blockIdx.x * 4 + warp) * 16;
  uint32_t row = tile + lane / 4, col = (lane % 4) * 4;
  const uint8_t* weights = w + static_cast<size_t>(expert) * N * K / 2;
  const uint8_t* scales = sf + static_cast<size_t>(expert) * N * (K / 16);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
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
      uint32_t k = base + inner;
      uint32_t a0, a1, a2, a3;
      load_a(a0, a1, a2, a3, ws[warp][(base / 256) % 2] + inner / 2, lane);
      uint32_t sa =
          *reinterpret_cast<const uint32_t*>(scales + sf_index<K / 16>(tile + lane / 4 + (lane % 2) * 8, k / 16));
      uint32_t sb = *reinterpret_cast<const uint32_t*>(xs + k / 16);
      mma(d0, d1, d2, d3, a0, a1, a2, a3, xq[k / 8 + lane % 4], xq[k / 8 + lane % 4 + 4], sa, sb);
    }
    __syncwarp();
    prefetch<K>(ws[warp][(base / 256) % 2], weights, tile, base / 256 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 == 0) {
    float y0 = static_cast<float>(cast<bf16_t>(d0 * alpha[expert]));
    float y1 = static_cast<float>(cast<bf16_t>(d2 * alpha[expert]));
    y0 = fmaxf(y0, 0.f);
    y1 = fmaxf(y1, 0.f);
    act[warp][lane / 4] = static_cast<float>(cast<bf16_t>(y0 * y0));
    act[warp][lane / 4 + 8] = static_cast<float>(cast<bf16_t>(y1 * y1));
  }
  __syncwarp();
  if (lane == 0) {
    size_t qoff = (token * TopK + slot) * (N / 8) + tile / 8;
    size_t soff = (token * TopK + slot) * (N / 16) + tile / 16;
    quantize16(act[warp], z[qoff], z[qoff + 1], zsf[soff], *global2);
  }
}

template <uint32_t N, uint32_t K, uint32_t TopK, bool PDL>
__global__ __launch_bounds__(128) void fc2(
    const uint32_t* __restrict__ z,
    const uint8_t* __restrict__ zs,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sf,
    const float* __restrict__ alpha,
    const int32_t* __restrict__ ids,
    const float* __restrict__ routing,
    bf16_t* __restrict__ partial) {
  __shared__ __align__(16) uint8_t ws[4][2][2304];
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
    prefetch<K>(ws[warp][stage], weights, tile, stage, lane);
#pragma unroll 1
  for (uint32_t base = 0; base < K; base += 256) {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (uint32_t inner = 0; inner < 256; inner += 64) {
      if (base + inner >= K) break;
      uint32_t k = base + inner;
      uint32_t a0, a1, a2, a3;
      load_a(a0, a1, a2, a3, ws[warp][(base / 256) % 2] + inner / 2, lane);
      uint32_t sa =
          *reinterpret_cast<const uint32_t*>(scales + sf_index<K / 16>(tile + lane / 4 + (lane % 2) * 8, k / 16));
      uint32_t sb = *reinterpret_cast<const uint32_t*>(zs + k / 16);
      mma(d0, d1, d2, d3, a0, a1, a2, a3, z[k / 8 + lane % 4], z[k / 8 + lane % 4 + 4], sa, sb);
    }
    __syncwarp();
    prefetch<K>(ws[warp][(base / 256) % 2], weights, tile, base / 256 + 2, lane);
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  if (lane % 4 == 0) {
    partial[route * N + row] = cast<bf16_t>((d0 * alpha[expert]) * routing[route]);
    partial[route * N + row + 8] = cast<bf16_t>((d2 * alpha[expert]) * routing[route]);
  }
}
template <uint32_t N, uint32_t TopK, bool PDL>
__global__ void reduce(const bf16_t* __restrict__ partial, bf16_t* __restrict__ out, uint32_t rows) {
  PDLTriggerSecondary<PDL>();
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  PDLWaitPrimary<PDL>();
  if (i < rows * N) {
    uint32_t token = i / N, col = i % N;
    float value = 0.f;
#pragma unroll
    for (uint32_t slot = 0; slot < TopK; ++slot)
      value += static_cast<float>(partial[(token * TopK + slot) * N + col]);
    out[i] = cast<bf16_t>(value);
  }
}
}  // namespace nvfp4_mtp_mma

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
template <uint32_t N, uint32_t K, uint32_t TopK, bool PDL>
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

template <uint32_t N1, uint32_t K1, uint32_t N2, uint32_t K2, uint32_t TopK, bool PDL, bool PairLayout>
struct Sm120Nvfp4MoeMtpMma {
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
    TensorMatcher({512, N1, K1 / 2}).with_dtype<uint8_t>().with_device(dev).verify(w1);
    if constexpr (PairLayout)
      TensorMatcher({512, K2 / 2, N2}).with_dtype<uint8_t>().with_device(dev).verify(w2);
    else
      TensorMatcher({512, N2, K2 / 2}).with_dtype<uint8_t>().with_device(dev).verify(w2);
    TensorMatcher({512 * N1 * K1 / 16}).with_dtype<uint8_t>().with_device(dev).verify(s1);
    TensorMatcher({512 * N2 * K2 / 16}).with_dtype<uint8_t>().with_device(dev).verify(s2);
    TensorMatcher({512}).with_dtype<float>().with_device(dev).verify(a1).verify(a2);
    TensorMatcher({1}).with_dtype<float>().with_device(dev).verify(g1).verify(g2);
    TensorMatcher({M, TopK}).with_dtype<int32_t>().with_device(dev).verify(ids);
    TensorMatcher({M, TopK}).with_dtype<float>().with_device(dev).verify(routing);
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
            PairLayout ? nvfp4_pair_mma::fc2<N2, K2, TopK, PDL> : nvfp4_mtp_mma::fc2<N2, K2, TopK, PDL>,
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
