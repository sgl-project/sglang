// SPDX-License-Identifier: Apache-2.0
// NVFP4 / MXFP8 quantization of this rank's routed MoE rows for the grouped
// GEMMs: one CTA per local row (rows offsets[0]..offsets[E]); NVFP4 splits each
// 16-value block over split() lanes, MXFP8 gives each lane 16 values. Other rows
// are left unset.

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sglang::inkling2_routed_quant {

constexpr int kMaxExperts = 1024;
constexpr float kFltMax = 3.4028234663852886e38f;

__device__ __forceinline__ float e4m3_round(float v, uint8_t& bits) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
  bits = pair & 0xFF;
  uint32_t halves;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(pair));
  return __half2float(__ushort_as_half(halves & 0xFFFF));
}

__device__ __forceinline__ uint32_t e2m1_pack(float low, float high, float& rlow, float& rhigh) {
  uint32_t packed, halves;
  asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; "
      "cvt.u32.u8 %0, b; cvt.rn.f16x2.e2m1x2 %1, b; }"
      : "=r"(packed), "=r"(halves)
      : "f"(low), "f"(high));
  rlow = __half2float(__ushort_as_half(halves & 0xFFFF));
  rhigh = __half2float(__ushort_as_half(halves >> 16));
  return packed;
}

// Quantize one 16-value block with the scale (6 or 4) of smaller error. The block
// is split over 16 / E consecutive lanes holding E values each; the error sums
// keep the pairwise order of the whole block (cross-lane levels by xor shuffles).
template <int E>
__device__ __forceinline__ void
quantize_part(float const (&x)[E], float amax, float block_max, uint32_t& packed, uint8_t& scale) {
  float encoder = fminf(__fdiv_rn(1536.0f, amax), kFltMax);
  float decoder = __fdiv_rn(1.0f, encoder);
  float outer = __fmul_rn(amax, float(1.0 / 1536.0));
  uint8_t bits6, bits4;
  float scale6 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, float(1.0 / 6.0))), 448.0f), bits6);
  float scale4 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, 0.25f)), 448.0f), bits4);
  float inverse6 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale6, decoder)), kFltMax);
  float inverse4 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale4, decoder)), kFltMax);
  uint32_t p6 = 0, p4 = 0;
  float e6[E], e4[E];
#pragma unroll
  for (int i = 0; i < E / 2; ++i) {
    float r0, r1;
    p6 |= e2m1_pack(__fmul_rn(x[2 * i], inverse6), __fmul_rn(x[2 * i + 1], inverse6), r0, r1) << (8 * i);
    e6[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale6), outer), x[2 * i]));
    e6[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale6), outer), x[2 * i + 1]));
    p4 |= e2m1_pack(__fmul_rn(x[2 * i], inverse4), __fmul_rn(x[2 * i + 1], inverse4), r0, r1) << (8 * i);
    e4[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale4), outer), x[2 * i]));
    e4[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale4), outer), x[2 * i + 1]));
  }
#pragma unroll
  for (int width = E / 2; width >= 1; width /= 2) {
#pragma unroll
    for (int i = 0; i < width; ++i) {
      e6[i] = __fadd_rn(e6[2 * i], e6[2 * i + 1]);
      e4[i] = __fadd_rn(e4[2 * i], e4[2 * i + 1]);
    }
  }
#pragma unroll
  for (int s = 1; s < 16 / E; s *= 2) {
    e6[0] = __fadd_rn(e6[0], __shfl_xor_sync(0xFFFFFFFF, e6[0], s));
    e4[0] = __fadd_rn(e4[0], __shfl_xor_sync(0xFFFFFFFF, e4[0], s));
  }
  bool choose4 = e4[0] < e6[0];
  packed = choose4 ? p4 : p6;
  scale = choose4 ? bits4 : bits6;
}

template <int E>
__device__ __forceinline__ float part_max(float const (&x)[E]) {
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < E; ++i)
    m = fmaxf(m, fabsf(x[i]));
#pragma unroll
  for (int s = 1; s < 16 / E; s *= 2)
    m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, s));
  return m;
}

// CUTLASS Sm1xxBlkScaledConfig SFA layout: 128x4 atoms, K-atom fastest; each
// expert's rows start on a 128-row boundary.
__device__ __forceinline__ int64_t scale_base(int const* offsets, int expert, int scale_k) {
  return int64_t((offsets[expert] - offsets[0] + expert * 127) / 128 * 128) * scale_k;
}

__device__ __forceinline__ int scale_offset(int local, int col, int scale_k) {
  return ((local / 128) * (scale_k / 4) + col / 4) * 512 + (local % 32) * 16 + ((local / 32) % 4) * 4 + col % 4;
}

__device__ __forceinline__ int row_expert(int const* offsets, int experts, int row) {
  int lo = 0, hi = experts;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

constexpr int threads(int width) {
  return (width / 16 + 31) / 32 * 32;
}

// Lanes per 16-value NVFP4 block: as many as fit one CTA per row.
constexpr int split(int width) {
  return width / 16 * 4 <= 1024 ? 4 : width / 16 * 2 <= 1024 ? 2 : 1;
}
constexpr int split_threads(int width) {
  return (width / 16 * split(width) + 31) / 32 * 32;
}

template <int kThreads>
__device__ __forceinline__ float block_max_reduce(float v, float* red) {
#pragma unroll
  for (int s = 16; s >= 1; s /= 2)
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, s));
  constexpr int kWarps = kThreads / 32;
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
  __syncthreads();
  v = red[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w)
    v = fmaxf(v, red[w]);
  return v;
}

// Zero the scales of each expert's padding rows: one warp per 128x4 atom of the
// expert's last row tile, each lane writing rows lane + {0, 32, 64, 96}.
__device__ __forceinline__ void zero_tails(int const* offsets, int experts, uint8_t* x_scales, int scale_k) {
  int const atoms = scale_k / 4;
  int const lane = threadIdx.x % 32;
  int const warps = blockDim.x / 32;
  for (int item = blockIdx.x * warps + threadIdx.x / 32; item < experts * atoms; item += gridDim.x * warps) {
    int e = item / atoms;
    int count = offsets[e + 1] - offsets[e];
    int used = count % 128;
    if (used == 0) continue;
    uint32_t* atom = reinterpret_cast<uint32_t*>(
        x_scales + scale_base(offsets, e, scale_k) + (int64_t(count / 128) * atoms + item % atoms) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i >= used) atom[i] = 0;
  }
}

__device__ __forceinline__ void load_offsets(int const* offsets, int experts, int* s_offsets) {
  for (int i = threadIdx.x; i <= experts; i += blockDim.x)
    s_offsets[i] = offsets[i];
  __syncthreads();
}

template <int E>
__device__ __forceinline__ void store_part(
    uint8_t* values,
    uint8_t* x_scales,
    int const* s_offsets,
    int expert,
    int row,
    int block,
    int part,
    int k,
    uint32_t packed,
    uint8_t scale) {
  uint8_t* dst = values + int64_t(row) * (k / 2) + block * 8 + part * (E / 2);
  if constexpr (E == 8)
    *reinterpret_cast<uint32_t*>(dst) = packed;
  else
    *reinterpret_cast<uint16_t*>(dst) = uint16_t(packed);
  if (part == 0) {
    int scale_k = (k + 63) / 64 * 4;
    x_scales[scale_base(s_offsets, expert, scale_k) + scale_offset(row - s_offsets[expert], block, scale_k)] = scale;
  }
}

template <int K>
__global__ void __launch_bounds__(split_threads(K)) quantize_routed(
    __nv_bfloat16 const* __restrict__ x,
    int const* __restrict__ dst2src,
    uint8_t* values,
    uint8_t* x_scales,
    float* amax_out,
    int const* __restrict__ offsets,
    int top_k,
    int experts) {
  constexpr int kThreads = split_threads(K), kSplit = split(K), E = 16 / kSplit;
  static_assert(K % 64 == 0 && kSplit > 1);
  __shared__ int s_offsets[kMaxExperts + 1];
  __shared__ float red[2][kThreads / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  load_offsets(offsets, experts, s_offsets);
  int const block = threadIdx.x / kSplit, part = threadIdx.x % kSplit;
  bool const active = block < K / 16;
  int parity = 0;
  for (int row = s_offsets[0] + blockIdx.x; row < s_offsets[experts]; row += gridDim.x, parity ^= 1) {
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i)
      v[i] = 0.0f;
    if (active) {
      int64_t source = dst2src[row] / top_k;
      __nv_bfloat16 const* src = x + source * K + block * 16 + part * E;
      if constexpr (E == 8) {
        uint4 raw = *reinterpret_cast<uint4 const*>(src);
        __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
        for (int i = 0; i < E; ++i)
          v[i] = __bfloat162float(h[i]);
      } else {
        uint2 raw = *reinterpret_cast<uint2 const*>(src);
        __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
        for (int i = 0; i < E; ++i)
          v[i] = __bfloat162float(h[i]);
      }
    }
    float block_max = part_max<E>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale;
    quantize_part<E>(v, amax, block_max, packed, scale);
    if (active) {
      if (threadIdx.x == 0) amax_out[row] = amax;
      store_part<E>(
          values, x_scales, s_offsets, row_expert(s_offsets, experts, row), row, block, part, K, packed, scale);
    }
  }
  zero_tails(s_offsets, experts, x_scales, (K + 63) / 64 * 4);
}

__device__ __forceinline__ float capped_swiglu(float gate, float up, float gamma, float gate_cap, float up_cap) {
  float sigmoid;
  asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
      "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
      "rcp.approx.ftz.f32 %0, t; }"
      : "=f"(sigmoid)
      : "f"(gate));
  if (gate_cap > 0) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(__fdiv_rn(gate, gate_cap)));
    gate = __fmul_rn(gate_cap, t);
  }
  if (up_cap > 0) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(__fdiv_rn(up, up_cap)));
    up = __fmul_rn(up_cap, t);
  }
  return __fmul_rn(__fmul_rn(__fmul_rn(gate, sigmoid), up), gamma);
}

template <int N>
__global__ void __launch_bounds__(split_threads(N)) swiglu_quantize_routed(
    float const* __restrict__ acc,
    float const* __restrict__ acc_amax,
    float const* __restrict__ w_amax,
    float const* __restrict__ w_denominator,
    float const* __restrict__ gammas,
    uint8_t* values,
    uint8_t* x_scales,
    float* amax_out,
    int const* __restrict__ offsets,
    int experts,
    float gate_cap,
    float up_cap) {
  constexpr int kThreads = split_threads(N), kSplit = split(N), E = 16 / kSplit;
  static_assert(N % 64 == 0 && kSplit > 1);
  __shared__ int s_offsets[kMaxExperts + 1];
  __shared__ float red[2][kThreads / 32];
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  load_offsets(offsets, experts, s_offsets);
  int const block = threadIdx.x / kSplit, part = threadIdx.x % kSplit;
  bool const active = block < N / 16;
  float const rcp = __frcp_rn(__fmul_rn(1536.0f, *w_denominator));
  int parity = 0;
  for (int row = s_offsets[0] + blockIdx.x; row < s_offsets[experts]; row += gridDim.x, parity ^= 1) {
    int expert = row_expert(s_offsets, experts, row);
    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i)
      v[i] = 0.0f;
    if (active) {
      float4 const* src = reinterpret_cast<float4 const*>(acc + int64_t(row) * 2 * N + block * 32 + part * 2 * E);
      float4 raw[E / 2];
#pragma unroll
      for (int i = 0; i < E / 2; ++i)
        raw[i] = src[i];
      float const* a = reinterpret_cast<float const*>(raw);
      float scale = __fmul_rn(__fmul_rn(acc_amax[row], w_amax[expert]), rcp);
      float gamma = gammas[row];
#pragma unroll
      for (int i = 0; i < E; ++i) {
        float gate = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a[2 * i], scale)));
        float up = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a[2 * i + 1], scale)));
        v[i] = __bfloat162float(__float2bfloat16_rn(capped_swiglu(gate, up, gamma, gate_cap, up_cap)));
      }
    }
    float block_max = part_max<E>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale_bits;
    quantize_part<E>(v, amax, block_max, packed, scale_bits);
    if (active) {
      if (threadIdx.x == 0) amax_out[row] = amax;
      store_part<E>(values, x_scales, s_offsets, expert, row, block, part, N, packed, scale_bits);
    }
  }
  zero_tails(s_offsets, experts, x_scales, (N + 63) / 64 * 4);
}

// _mxfp8_block in kernels/moe.py for one 32-value block split over a lane pair.
__device__ __forceinline__ void mxfp8_half_block(float const (&x)[16], uint4& values, uint8_t& exponent_out) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1));
  float const min_scale = __int_as_float(0x00400000);
  float raw = __fmul_rn(amax, float(1.0 / 448.0));
  int exponent = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
  exponent = raw <= min_scale ? 0 : min(exponent, 254);
  if (raw != raw) exponent = 255;
  float scale = fmaxf(__int_as_float(exponent << 23), min_scale);
  float inverse = exponent == 0 ? __int_as_float(0x7F000000) : __frcp_rn(scale);
  if (exponent == 255) inverse = 0.0f;
  uint16_t pairs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float lo = __fmul_rn(x[2 * i], inverse), hi = __fmul_rn(x[2 * i + 1], inverse);
    lo = fminf(fmaxf(lo, -448.0f), 448.0f);
    hi = fminf(fmaxf(hi, -448.0f), 448.0f);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pairs[i]) : "f"(hi), "f"(lo));
  }
  values = *reinterpret_cast<uint4*>(pairs);
  exponent_out = uint8_t(exponent);
}

template <int N>
__global__ void __launch_bounds__(threads(N)) mxfp8_swiglu_quantize_routed(
    __nv_bfloat16 const* __restrict__ x,
    float const* __restrict__ gammas,
    uint8_t* values,
    uint8_t* exponents,
    int const* __restrict__ offsets,
    int experts,
    float gate_cap,
    float up_cap) {
  static_assert(N % 32 == 0 && N / 16 <= 1024);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  int const block = threadIdx.x;
  bool const active = block < N / 16;
  int const lo = offsets[0], hi = offsets[experts];
  for (int row = lo + blockIdx.x; row < hi; row += gridDim.x) {
    float v[16];
    if (active) {
      uint4 const* src = reinterpret_cast<uint4 const*>(x + int64_t(row) * 2 * N + block * 32);
      uint4 raw[4] = {src[0], src[1], src[2], src[3]};
      __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(raw);
      float gamma = gammas[row];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = __bfloat162float(__float2bfloat16_rn(
            capped_swiglu(__bfloat162float(h[2 * i]), __bfloat162float(h[2 * i + 1]), gamma, gate_cap, up_cap)));
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i)
        v[i] = 0.0f;
    }
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    if (active) {
      *reinterpret_cast<uint4*>(values + int64_t(row) * N + block * 16) = packed;
      if (block % 2 == 0) exponents[int64_t(row) * (N / 32) + block / 2] = exponent;
    }
  }
}

// _mxfp8_quantize in kernels/moe.py over rows of N bf16 values, each row
// split over kRowParts CTAs.
constexpr int kRowParts = 3;

template <int N>
__global__ void __launch_bounds__(N / 16 / kRowParts)
    mxfp8_quantize_rows(__nv_bfloat16 const* __restrict__ x, uint8_t* values, uint8_t* exponents, int rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
  int const block = (blockIdx.x % kRowParts) * blockDim.x + threadIdx.x;
  for (int row = blockIdx.x / kRowParts; row < rows; row += gridDim.x / kRowParts) {
    float v[16];
    uint4 const* src = reinterpret_cast<uint4 const*>(x + int64_t(row) * N + block * 16);
    uint4 raw[2] = {src[0], src[1]};
    __nv_bfloat16 const* h = reinterpret_cast<__nv_bfloat16 const*>(raw);
#pragma unroll
    for (int i = 0; i < 16; ++i)
      v[i] = __bfloat162float(h[i]);
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    *reinterpret_cast<uint4*>(values + int64_t(row) * N + block * 16) = packed;
    if (block % 2 == 0) exponents[int64_t(row) * (N / 32) + block / 2] = exponent;
  }
}

inline int grid_size(int rows, int sm_count) {
  return std::max(1, std::min(rows, 4 * sm_count));
}

template <int K>
void quantize(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView dst2src,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView amax,
    tvm::ffi::TensorView offsets,
    int64_t top_k,
    int64_t sm_count) {
  using host::RuntimeCheck;
  int experts = offsets.size(0) - 1;
  RuntimeCheck(x.size(1) == K && experts <= kMaxExperts, "Unsupported routed NVFP4 quantize shape");
  host::LaunchKernel(grid_size(dst2src.size(0), sm_count), split_threads(K), x.device())
      .enable_pdl(true)(
          quantize_routed<K>,
          static_cast<__nv_bfloat16 const*>(x.data_ptr()),
          static_cast<int const*>(dst2src.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(amax.data_ptr()),
          static_cast<int const*>(offsets.data_ptr()),
          int(top_k),
          experts);
}

template <int N>
void swiglu_quantize(
    tvm::ffi::TensorView acc,
    tvm::ffi::TensorView acc_amax,
    tvm::ffi::TensorView w_amax,
    tvm::ffi::TensorView w_denominator,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView x_scales,
    tvm::ffi::TensorView amax,
    tvm::ffi::TensorView offsets,
    double gate_cap,
    double up_cap,
    int64_t sm_count) {
  using host::RuntimeCheck;
  int experts = offsets.size(0) - 1;
  RuntimeCheck(acc.size(1) == 2 * N && experts <= kMaxExperts, "Unsupported routed NVFP4 swiglu quantize shape");
  host::LaunchKernel(grid_size(acc.size(0), sm_count), split_threads(N), acc.device())
      .enable_pdl(true)(
          swiglu_quantize_routed<N>,
          static_cast<float const*>(acc.data_ptr()),
          static_cast<float const*>(acc_amax.data_ptr()),
          static_cast<float const*>(w_amax.data_ptr()),
          static_cast<float const*>(w_denominator.data_ptr()),
          static_cast<float const*>(gammas.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(x_scales.data_ptr()),
          static_cast<float*>(amax.data_ptr()),
          static_cast<int const*>(offsets.data_ptr()),
          experts,
          float(gate_cap),
          float(up_cap));
}

template <int N>
void mxfp8_swiglu_quantize(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView gammas,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView exponents,
    tvm::ffi::TensorView offsets,
    double gate_cap,
    double up_cap,
    int64_t sm_count) {
  host::RuntimeCheck(x.size(1) == 2 * N, "Unsupported MXFP8 swiglu shape");
  host::LaunchKernel(grid_size(x.size(0), sm_count), threads(N), x.device())
      .enable_pdl(true)(
          mxfp8_swiglu_quantize_routed<N>,
          static_cast<__nv_bfloat16 const*>(x.data_ptr()),
          static_cast<float const*>(gammas.data_ptr()),
          static_cast<uint8_t*>(values.data_ptr()),
          static_cast<uint8_t*>(exponents.data_ptr()),
          static_cast<int const*>(offsets.data_ptr()),
          int(offsets.size(0) - 1),
          float(gate_cap),
          float(up_cap));
}

template <int N>
void mxfp8_quantize(
    tvm::ffi::TensorView x, tvm::ffi::TensorView values, tvm::ffi::TensorView exponents, int64_t sm_count) {
  if constexpr (N / 16 % (32 * kRowParts) == 0) {
    int const rows = x.numel() / N;
    host::RuntimeCheck(x.size(-1) == N, "Unsupported MXFP8 quantize shape");
    host::LaunchKernel(grid_size(rows, sm_count) * kRowParts, N / 16 / kRowParts, x.device())
        .enable_pdl(true)(
            mxfp8_quantize_rows<N>,
            static_cast<__nv_bfloat16 const*>(x.data_ptr()),
            static_cast<uint8_t*>(values.data_ptr()),
            static_cast<uint8_t*>(exponents.data_ptr()),
            rows);
  } else {
    host::RuntimeCheck(false, "MXFP8 quantize rows must split into whole warps");
  }
}

}  // namespace sglang::inkling2_routed_quant
