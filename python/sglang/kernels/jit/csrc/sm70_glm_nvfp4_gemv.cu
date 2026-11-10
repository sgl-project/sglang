// SPDX-License-Identifier: Apache-2.0
// Batch-1..4 NVFP4 GEMV for SM70 Marlin weights whose N is a multiple of 256.
//
// Marlin stores four 64-wide tiles interleaved at that width. The 64-wide
// formula is a different layout and must not be used here. Dots accumulate
// in FP32; the FP4 values and block scales stay in FP16.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace sm70_glm_nvfp4_gemv {

constexpr int kThreads = 128;
constexpr float kMarlinScaleCompensation = 1.0f;
__device__ __constant__ int kScaleLogicalToStored[8] = {0, 2, 1, 3, 4, 6, 5, 7};

__device__ __forceinline__ void dequant_fp4x8(uint32_t packed, __half2* values) {
  const uint32_t even =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  packed >>= 4;
  const uint32_t odd =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  uint32_t result[4] = {
      __byte_perm(even, 0, 0x2404), __byte_perm(odd, 0, 0x2404),
      __byte_perm(even, 0, 0x3414), __byte_perm(odd, 0, 0x3414)};
#pragma unroll
  for (int p = 0; p < 4; ++p)
    values[p] = *reinterpret_cast<__half2*>(&result[p]);
}

__device__ __forceinline__ __half2 load_scale_pair(const uint8_t* encoded,
                                                   int logical0,
                                                   int logical1) {
  return __halves2half2(
      __ushort_as_half(static_cast<uint16_t>(encoded[logical0]) << 7),
      __ushort_as_half(static_cast<uint16_t>(encoded[logical1]) << 7));
}

template <bool VectorScales>
__device__ __forceinline__ void load_scales8(const uint8_t* encoded,
                                            __half2* out) {
  if constexpr (VectorScales) {
    const uint2 raw = *reinterpret_cast<const uint2*>(encoded);
    const uint32_t bits[4] = {
        __byte_perm(raw.x, 0, 0x4240) << 7, __byte_perm(raw.x, 0, 0x4341) << 7,
        __byte_perm(raw.y, 0, 0x4240) << 7, __byte_perm(raw.y, 0, 0x4341) << 7};
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = *reinterpret_cast<const __half2*>(&bits[p]);
  } else {
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = load_scale_pair(encoded, kScaleLogicalToStored[2 * p],
                              kScaleLogicalToStored[2 * p + 1]);
  }
}

// Both halves of scale2 hold the scale of logical column sub of the qword at
// encoded: load_scales8<true>'s half sub, read as its one stored byte. Indexing
// load_scales8's output by a runtime sub spills it to the stack instead.
__device__ __forceinline__ int scale_byte_offset(int sub) {
  return (sub & 4) | ((sub & 1) << 1) | ((sub >> 1) & 1);
}
__device__ __forceinline__ uint32_t scale2_from_byte(uint32_t byte) {
  return (byte << 7) | (byte << 23);
}
__device__ __forceinline__ uint32_t lane_scale2(const uint8_t* encoded, int sub) {
  return scale2_from_byte(encoded[scale_byte_offset(sub)]);
}

template <bool VectorScales>
__global__ void __launch_bounds__(kThreads, 2)
gemv_partial_kernel(const __half* __restrict__ input,
                    const uint32_t* __restrict__ weight,
                    const uint8_t* __restrict__ scales,
                    int M,
                    int N,
                    int K,
                    int split_k,
                    float* __restrict__ partials) {
  const int qwords = N >> 3;
  const int groups = K >> 4;
  const int groups_per_split = groups / split_k;
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int total = M * split_k * qwords;
  if (work >= total) {
    return;
  }
  const int qword = work % qwords;
  const int split_token = work / qwords;
  const int split = split_token % split_k;
  const int token = split_token / split_k;
  const int n_base = qword * 8;
  const int group_n_tile = qword >> 5;
  const int subtile = (qword >> 3) & 3;
  const int qword_in_tile = qword & 7;
  const int group_begin = split * groups_per_split;

  float accum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const __half* row = input + static_cast<int64_t>(token) * K;
  for (int group_it = 0; group_it < groups_per_split; ++group_it) {
    const int group = group_begin + group_it;
    __half2 scale[4];
    load_scales8<VectorScales>(scales + static_cast<int64_t>(group) * N + n_base,
                              scale);
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const int k = group * 16 + r;
      const float x = __half2float(row[k]);
      const int64_t offset = static_cast<int64_t>(group) * (static_cast<int64_t>(N) * 2) +
                             static_cast<int64_t>(group_n_tile) * 512 +
                             qword_in_tile * 4 + subtile + r * 32;
      __half2 value[4];
      dequant_fp4x8(weight[offset], value);
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const __half2 term = __hmul2(scale[p], value[p]);
        accum[2 * p] = fmaf(x, __half2float(term.x), accum[2 * p]);
        accum[2 * p + 1] = fmaf(x, __half2float(term.y), accum[2 * p + 1]);
      }
    }
  }
  float* out = partials + (static_cast<int64_t>(split) * M + token) * N + n_base;
  const float multiplier = kMarlinScaleCompensation;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    out[p] = accum[p] * multiplier;
  }
}

__global__ void __launch_bounds__(kThreads, 2)
gemv_reduce_kernel(const float* __restrict__ partials,
                   const float* __restrict__ global_scale,
                   int M,
                   int N,
                   int split_k,
                   __half* __restrict__ output) {
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int total = M * N;
  if (work >= total) {
    return;
  }
  const int token = work / N;
  const int n = work - token * N;
  float sum = 0.f;
  for (int split = 0; split < split_k; ++split) {
    sum += partials[(static_cast<int64_t>(split) * M + token) * N + n];
  }
  sum *= global_scale[0];
  output[work] = __float2half_rn(sum);
}

void gemv(torch::Tensor input,
          torch::Tensor weight,
          torch::Tensor scales,
          torch::Tensor global_scale,
          torch::Tensor partials,
          torch::Tensor output,
          int64_t split_k) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous CUDA FP16 [M, K]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kInt &&
                  weight.dim() == 2 && weight.is_contiguous(),
              "weight must be contiguous CUDA int32 Marlin tiles");
  TORCH_CHECK(scales.is_cuda() && scales.element_size() == 1 &&
                  scales.dim() == 2 && scales.is_contiguous(),
              "scales must be contiguous byte metadata [K/16, N]");
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == at::kFloat &&
                  global_scale.numel() >= 1,
              "global_scale must be a CUDA FP32 scalar");
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(scales.size(1));
  TORCH_CHECK(M >= 1 && M <= 4, "GEMV batch must be 1..4");
  TORCH_CHECK(N % 256 == 0 && K % 16 == 0, "N%256 and K%16 are required");
  TORCH_CHECK(weight.size(0) == K / 16 && weight.size(1) == N * 2,
              "weight shape must be [K/16, N*2]");
  TORCH_CHECK(scales.size(0) == K / 16, "scales shape must be [K/16, N]");
  TORCH_CHECK(split_k >= 1 && (K / 16) % split_k == 0, "split_k must divide K/16");
  TORCH_CHECK(partials.is_cuda() && partials.scalar_type() == at::kFloat &&
                  partials.numel() == static_cast<int64_t>(split_k) * M * N,
              "partials must be FP32 [split, M, N]");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf &&
                  output.numel() == static_cast<int64_t>(M) * N,
              "output must be FP16 [M, N]");

  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool vector_scales =
      (reinterpret_cast<uintptr_t>(scales.data_ptr()) % 8u) == 0;
  const int qwords = N >> 3;
  const int total = M * static_cast<int>(split_k) * qwords;
  const int blocks = (total + kThreads - 1) / kThreads;
  auto partial = vector_scales ? gemv_partial_kernel<true> : gemv_partial_kernel<false>;
  partial<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(weight.data_ptr<int>()),
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      M,
      N,
      K,
      static_cast<int>(split_k),
      partials.data_ptr<float>());
  const int out_blocks = (M * N + kThreads - 1) / kThreads;
  gemv_reduce_kernel<<<out_blocks, kThreads, 0, stream>>>(
      partials.data_ptr<float>(),
      global_scale.data_ptr<float>(),
      M,
      N,
      static_cast<int>(split_k),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define HMMA_M8N8K4(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// Same fp16 bits dequant_fp4x8 produces for one E2M1 nibble.
__device__ __forceinline__ __half nibble_to_half(unsigned nib) {
  const unsigned bits = ((nib & 8u) << 12) | ((nib & 7u) << 9);
  return __ushort_as_half(static_cast<unsigned short>(bits));
}

__device__ __forceinline__ int marlin_nibble_pos(int sub) {
  return (sub & 1) ? (4 + (sub >> 1)) : (sub >> 1);
}

// Gather each lane's 16 K-group nibbles into one uint64 so the GEMV loads a
// contiguous uint2 instead of 16 strided Marlin words. Layout is
// [N/32][K/16][lane].
__global__ void repack_kernel(const uint32_t* __restrict__ weight,
                              int N,
                              int groups,
                              int64_t* __restrict__ packed) {
  const int lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int n = tile * 32 + lane_col;
  const int qword = n >> 3;
  const int sub = n & 7;
  const int pos = marlin_nibble_pos(sub);
  const int group_n_tile = qword >> 5;
  const int subtile = (qword >> 3) & 3;
  const int qword_in_tile = qword & 7;
  const int64_t base = static_cast<int64_t>(group) * (static_cast<int64_t>(N) * 2) +
                       static_cast<int64_t>(group_n_tile) * 512 + qword_in_tile * 4 +
                       subtile;
  uint64_t bits = 0;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const uint32_t word = weight[base + static_cast<int64_t>(r) * 32];
    const uint64_t nib = (word >> (pos * 4)) & 0xFu;
    bits |= nib << (4 * r);
  }
  packed[(static_cast<int64_t>(tile) * groups + group) * 32 + lane] =
      static_cast<int64_t>(bits);
}

// Inverse of repack_kernel. The destination must be zero; each lane ORs its
// nibble into a distinct 4-bit slot of the Marlin word.
__global__ void unpack_kernel(const int64_t* __restrict__ packed,
                              int N,
                              int groups,
                              uint32_t* __restrict__ weight) {
  const int lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x);
  const int group = static_cast<int>(blockIdx.y);
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int n = tile * 32 + lane_col;
  const int qword = n >> 3;
  const int sub = n & 7;
  const int pos = marlin_nibble_pos(sub);
  const int group_n_tile = qword >> 5;
  const int subtile = (qword >> 3) & 3;
  const int qword_in_tile = qword & 7;
  const int64_t base = static_cast<int64_t>(group) * (static_cast<int64_t>(N) * 2) +
                       static_cast<int64_t>(group_n_tile) * 512 + qword_in_tile * 4 +
                       subtile;
  const uint64_t bits = static_cast<uint64_t>(
      packed[(static_cast<int64_t>(tile) * groups + group) * 32 + lane]);
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    const uint32_t nib = static_cast<uint32_t>((bits >> (4 * r)) & 0xFu);
    atomicOr(weight + base + static_cast<int64_t>(r) * 32, nib << (pos * 4));
  }
}

// Inverse of repack_kernel for a stack of experts in one launch. One thread
// per (expert, K group, qword) writes that qword's 16 Marlin words whole;
// thread order follows the Marlin word order so the stores coalesce.
__global__ void unpack_experts_kernel(const int64_t* __restrict__ packed,
                                      int N,
                                      int groups,
                                      int64_t total,
                                      uint32_t* __restrict__ weight) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int qwords = N >> 3;
  const int t = static_cast<int>(idx % qwords);
  const int64_t expert_group = idx / qwords;
  const int group = static_cast<int>(expert_group % groups);
  const int64_t expert = expert_group / groups;
  const int group_n_tile = t >> 5;
  const int qword_in_tile = (t >> 2) & 7;
  const int subtile = t & 3;
  const int qword = group_n_tile * 32 + subtile * 8 + qword_in_tile;
  const int64_t* src =
      packed + ((expert * (N >> 5) + (qword >> 2)) * groups + group) * 32;
  uint64_t bits[8];
#pragma unroll
  for (int sub = 0; sub < 8; ++sub) {
    const int lane = (sub & 3) | ((qword & 3) << 2) | ((sub >> 2) << 4);
    bits[sub] = static_cast<uint64_t>(src[lane]);
  }
  uint32_t* dst = weight + (expert * groups + group) * (static_cast<int64_t>(N) * 2) +
                  group_n_tile * 512 + qword_in_tile * 4 + subtile;
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    uint32_t word = 0;
#pragma unroll
    for (int sub = 0; sub < 8; ++sub)
      word |= static_cast<uint32_t>((bits[sub] >> (4 * r)) & 0xFu) << (marlin_nibble_pos(sub) * 4);
    dst[r * 32] = word;
  }
}

__device__ __forceinline__ uint32_t nibbles_to_half2(uint32_t pair) {
  return ((pair & 0x00080008u) << 12) | ((pair & 0x00070007u) << 9);
}

__device__ __forceinline__ void unpack8(uint32_t x, uint32_t& h0, uint32_t& h1, uint32_t& h2,
                                       uint32_t& h3) {
  const uint32_t ev = x & 0x0F0F0F0Fu;
  const uint32_t od = (x >> 4) & 0x0F0F0F0Fu;
  // Byte i of ev into byte 0, byte i of od into byte 2; nibbles_to_half2 reads
  // only the low nibble of those two bytes, so bytes 1 and 3 are don't-care.
  h0 = nibbles_to_half2(__byte_perm(ev, od, 0x0400));
  h1 = nibbles_to_half2(__byte_perm(ev, od, 0x0501));
  h2 = nibbles_to_half2(__byte_perm(ev, od, 0x0602));
  h3 = nibbles_to_half2(__byte_perm(ev, od, 0x0703));
}

__device__ __forceinline__ uint32_t scale_half2(uint32_t h, uint32_t scale2) {
  uint32_t out;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h), "r"(scale2));
  return out;
}

__global__ void __launch_bounds__(32, 8)
hmma_packed_kernel(const __half* __restrict__ input,
                   const int64_t* __restrict__ packed,
                   const uint8_t* __restrict__ scales,
                   const float* __restrict__ global_scale,
                   int M,
                   int N,
                   int K,
                   __half* __restrict__ output) {
  const int lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x);
  const int n0 = tile * 32;
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int n = n0 + lane_col;
  const int qword = n >> 3;
  const int sub = n & 7;
  const int groups = K >> 4;
  const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);
  const uint2* row = reinterpret_cast<const uint2*>(
      packed + (static_cast<int64_t>(tile) * groups) * 32 + lane);
  float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int group = 0; group < groups; ++group) {
    const uint2 packed_k = __ldcs(row + static_cast<int64_t>(group) * 32);
    const uint32_t scale2 = lane_scale2(scales + static_cast<int64_t>(group) * N + (qword << 3), sub);
    uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
    unpack8(packed_k.x, b0, b1, b2, b3);
    unpack8(packed_k.y, b4, b5, b6, b7);
    b0 = scale_half2(b0, scale2);
    b1 = scale_half2(b1, scale2);
    b2 = scale_half2(b2, scale2);
    b3 = scale_half2(b3, scale2);
    b4 = scale_half2(b4, scale2);
    b5 = scale_half2(b5, scale2);
    b6 = scale_half2(b6, scale2);
    b7 = scale_half2(b7, scale2);
    uint4 a01 = make_uint4(0, 0, 0, 0);
    uint4 a23 = make_uint4(0, 0, 0, 0);
    if (arow < M) {
      const __half* a =
          input + (static_cast<int64_t>(arow) * K + static_cast<int64_t>(group) * 16);
      a01 = *reinterpret_cast<const uint4*>(a);
      a23 = *reinterpret_cast<const uint4*>(a + 8);
    }
    const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
    const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
    HMMA_M8N8K4(c, A0[0], A0[1], b0, b1);
    HMMA_M8N8K4(c, A0[2], A0[3], b2, b3);
    HMMA_M8N8K4(c, A1[0], A1[1], b4, b5);
    HMMA_M8N8K4(c, A1[2], A1[3], b6, b7);
  }

  const float gscale = global_scale[0];
  const int qp = (lane >> 2) & 3;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int orow = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    col = n0 + qp * 8 + col;
    if (orow < M) {
      output[static_cast<int64_t>(orow) * N + col] = __float2half_rn(c[i] * gscale);
    }
  }
}

#undef HMMA_M8N8K4

torch::Tensor repack(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kInt && weight.dim() == 2 &&
                  weight.is_contiguous(),
              "weight must be contiguous CUDA int32 [K/16, N*2]");
  const int groups = static_cast<int>(weight.size(0));
  const int N = static_cast<int>(weight.size(1) / 2);
  TORCH_CHECK(N % 32 == 0 && groups > 0, "N%32 and K/16 > 0 are required");
  auto packed = torch::empty({N / 32, groups, 32}, weight.options().dtype(torch::kLong));
  const c10::cuda::CUDAGuard guard(weight.device());
  repack_kernel<<<dim3(N / 32, groups), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(weight.data_ptr<int>()),
      N,
      groups,
      packed.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

void unpack_into(torch::Tensor packed, torch::Tensor weight) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong && packed.dim() == 3 &&
                  packed.is_contiguous() && packed.size(2) == 32,
              "packed must be contiguous CUDA int64 [N/32, K/16, 32]");
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kInt && weight.dim() == 2 &&
                  weight.is_contiguous(),
              "weight must be contiguous CUDA int32 [K/16, N*2]");
  const int tiles = static_cast<int>(packed.size(0));
  const int groups = static_cast<int>(packed.size(1));
  const int N = tiles * 32;
  TORCH_CHECK(weight.size(0) == groups && weight.size(1) == N * 2,
              "weight shape does not match packed");
  const c10::cuda::CUDAGuard guard(weight.device());
  weight.zero_();
  unpack_kernel<<<dim3(tiles, groups), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      packed.data_ptr<int64_t>(),
      N,
      groups,
      reinterpret_cast<uint32_t*>(weight.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack_experts_into(torch::Tensor packed, torch::Tensor weight) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong && packed.dim() == 4 &&
                  packed.is_contiguous() && packed.size(3) == 32,
              "packed must be contiguous CUDA int64 [E, N/32, K/16, 32]");
  const int experts = static_cast<int>(packed.size(0));
  const int N = static_cast<int>(packed.size(1)) * 32;
  const int groups = static_cast<int>(packed.size(2));
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kInt && weight.dim() == 3 &&
                  weight.is_contiguous() && weight.size(0) == experts &&
                  weight.size(1) == groups && weight.size(2) == N * 2,
              "weight must be contiguous CUDA int32 [E, K/16, N*2]");
  TORCH_CHECK(N % 256 == 0, "N must be a multiple of 256");
  const int64_t total = static_cast<int64_t>(experts) * groups * (N >> 3);
  const int threads = 256;
  const c10::cuda::CUDAGuard guard(weight.device());
  unpack_experts_kernel<<<static_cast<unsigned>((total + threads - 1) / threads), threads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      packed.data_ptr<int64_t>(), N, groups, total,
      reinterpret_cast<uint32_t*>(weight.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemv_hmma(torch::Tensor input,
               torch::Tensor packed,
               torch::Tensor scales,
               torch::Tensor global_scale,
               torch::Tensor output) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
                  input.is_contiguous(),
              "input must be contiguous CUDA FP16 [M, K]");
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(scales.size(1));
  const int groups = K >> 4;
  TORCH_CHECK(M >= 1 && M <= 4, "HMMA GEMV batch must be 1..4");
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0, "N%32 and K%16 are required");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong &&
                  packed.size(0) == N / 32 && packed.size(1) == groups && packed.size(2) == 32,
              "packed must be int64 [N/32, K/16, 32]");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf && output.size(0) == M &&
                  output.size(1) == N,
              "output must be FP16 [M, N]");
  const c10::cuda::CUDAGuard guard(input.device());
  hmma_packed_kernel<<<N / 32, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      packed.data_ptr<int64_t>(),
      reinterpret_cast<const uint8_t*>(scales.data_ptr()),
      global_scale.data_ptr<float>(),
      M,
      N,
      K,
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define HMMA_M8N8K4(C, A0, A1, B0, B1)                                      \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// One warp owns 32 columns of every route. Invalid experts store zeros so the
// SiLU that follows can read the buffer. Down multiplies (accum * route) * global
// in fp32, matching Marlin's epilogue, then rounds once.
template <bool kPerRouteInput, bool kApplyRoute>
__global__ void moe_hmma_kernel(const __half* __restrict__ input,
                                const int64_t* __restrict__ packed,
                                const uint8_t* __restrict__ scales,
                                const float* __restrict__ global_scale,
                                const int* __restrict__ topk_ids,
                                const float* __restrict__ topk_weights,
                                int routes,
                                int topk,
                                int num_experts,
                                int N,
                                int K,
                                __half* __restrict__ output) {
  const int lane = threadIdx.x & 31;
  const int tile = static_cast<int>(blockIdx.x);
  const int n0 = tile * 32;
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int qword = (n0 + lane_col) >> 3;
  const int sub = lane_col & 7;
  const int groups = K >> 4;
  const int tiles = N >> 5;
  const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int qp = (lane >> 2) & 3;

  for (int route = 0; route < routes; ++route) {
    const int expert = topk_ids[route];
    const bool valid = expert >= 0 && expert < num_experts;
    float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (valid) {
      const int token = kPerRouteInput ? route : route / topk;
      const __half* in_row = input + static_cast<int64_t>(token) * K;
      const int64_t* expert_packed =
          packed + (static_cast<int64_t>(expert) * tiles * groups) * 32;
      const uint2* row = reinterpret_cast<const uint2*>(
          expert_packed + (static_cast<int64_t>(tile) * groups) * 32 + lane);
      const uint8_t* expert_scales =
          scales + (static_cast<int64_t>(expert) * groups) * N;
      for (int group = 0; group < groups; ++group) {
        const uint2 packed_k = __ldcs(row + static_cast<int64_t>(group) * 32);
        const uint32_t scale2 = lane_scale2(expert_scales + static_cast<int64_t>(group) * N + (qword << 3), sub);
        uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
        unpack8(packed_k.x, b0, b1, b2, b3);
        unpack8(packed_k.y, b4, b5, b6, b7);
        b0 = scale_half2(b0, scale2);
        b1 = scale_half2(b1, scale2);
        b2 = scale_half2(b2, scale2);
        b3 = scale_half2(b3, scale2);
        b4 = scale_half2(b4, scale2);
        b5 = scale_half2(b5, scale2);
        b6 = scale_half2(b6, scale2);
        b7 = scale_half2(b7, scale2);
        uint4 a01 = make_uint4(0, 0, 0, 0);
        uint4 a23 = make_uint4(0, 0, 0, 0);
        if (arow == 0) {
          const __half* a = in_row + static_cast<int64_t>(group) * 16;
          a01 = *reinterpret_cast<const uint4*>(a);
          a23 = *reinterpret_cast<const uint4*>(a + 8);
        }
        const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
        const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
        HMMA_M8N8K4(c, A0[0], A0[1], b0, b1);
        HMMA_M8N8K4(c, A0[2], A0[3], b2, b3);
        HMMA_M8N8K4(c, A1[0], A1[1], b4, b5);
        HMMA_M8N8K4(c, A1[2], A1[3], b6, b7);
      }
    }
    const float gscale = valid ? global_scale[expert] : 0.f;
    const float route_w = (kApplyRoute && valid) ? topk_weights[route] : 1.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int orow = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      if (orow != 0) {
        continue;
      }
      int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      col = n0 + qp * 8 + col;
      float value = c[i];
      if constexpr (kApplyRoute) {
        value = value * route_w * gscale;
      } else {
        value = value * gscale;
      }
      output[static_cast<int64_t>(route) * N + col] = __float2half_rn(value);
    }
  }
}

__global__ void moe_reduce_kernel(const __half* __restrict__ route_out,
                                  const int* __restrict__ topk_ids,
                                  int batch,
                                  int N,
                                  int topk,
                                  int num_experts,
                                  float routed_scale,
                                  __half* __restrict__ output) {
  const int work = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (work >= batch * N) {
    return;
  }
  const int token = work / N;
  const int n = work - token * N;
  float sum = 0.f;
  for (int t = 0; t < topk; ++t) {
    const int route = token * topk + t;
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    sum += __half2float(route_out[static_cast<int64_t>(route) * N + n]);
  }
  output[work] = __float2half_rn(sum * routed_scale);
}

// One K group (16 rows) of one warp's 32-column tile. !has_a feeds zeros.
__device__ __forceinline__ void hmma_group_step(uint2 packed_k,
                                                uint32_t scale2,
                                                bool has_a,
                                                const __half* a,
                                                float (&c)[8]) {
  uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
  unpack8(packed_k.x, b0, b1, b2, b3);
  unpack8(packed_k.y, b4, b5, b6, b7);
  b0 = scale_half2(b0, scale2);
  b1 = scale_half2(b1, scale2);
  b2 = scale_half2(b2, scale2);
  b3 = scale_half2(b3, scale2);
  b4 = scale_half2(b4, scale2);
  b5 = scale_half2(b5, scale2);
  b6 = scale_half2(b6, scale2);
  b7 = scale_half2(b7, scale2);
  uint4 a01 = make_uint4(0, 0, 0, 0);
  uint4 a23 = make_uint4(0, 0, 0, 0);
  if (has_a) {
    a01 = *reinterpret_cast<const uint4*>(a);
    a23 = *reinterpret_cast<const uint4*>(a + 8);
  }
  const unsigned* A0 = reinterpret_cast<const unsigned*>(&a01);
  const unsigned* A1 = reinterpret_cast<const unsigned*>(&a23);
  HMMA_M8N8K4(c, A0[0], A0[1], b0, b1);
  HMMA_M8N8K4(c, A0[2], A0[3], b2, b3);
  HMMA_M8N8K4(c, A1[0], A1[1], b4, b5);
  HMMA_M8N8K4(c, A1[2], A1[3], b6, b7);
}

// K groups [g_begin, g_end) of one warp's tile, in order. row is the lane's
// packed column, scale_qword its qword's scales at group 0, a_row its A row
// (nullptr feeds zeros); pointers step one group at a time.
template <int kUnroll>
__device__ __forceinline__ void hmma_group_run(const uint2* row,
                                               const uint8_t* scale_qword,
                                               int sub,
                                               const __half* a_row,
                                               int N,
                                               int g_begin,
                                               int g_end,
                                               float (&c)[8]) {
  const bool has_a = a_row != nullptr;
  const uint2* wp = row + static_cast<int64_t>(g_begin) * 32;
  const uint8_t* sp = scale_qword + static_cast<int64_t>(g_begin) * N + scale_byte_offset(sub);
  // Without an A row the pointer only advances; it is never read.
  const __half* ap = has_a ? a_row + g_begin * 16 : reinterpret_cast<const __half*>(row);
#pragma unroll(kUnroll)
  for (int group = g_begin; group < g_end; ++group) {
    hmma_group_step(__ldcs(wp), scale2_from_byte(*sp), has_a, ap, c);
    wp += 32;
    sp += N;
    ap += 16;
  }
}

// Column (0..31) inside the tile and output row of accumulator i.
__device__ __forceinline__ int hmma_acc_col(int lane, int i) {
  return ((lane >> 2) & 3) * 8 + ((i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2));
}
__device__ __forceinline__ int hmma_acc_row(int lane, int i) {
  return (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
}

// Split-K variants: kSplit warps share one 32-column tile, each owns a
// contiguous run of K groups, and warp partials are summed in warp order, so
// results are deterministic. kSplit == 1 reproduces the single-warp kernels.
template <int kSplit>
__global__ void __launch_bounds__(32 * kSplit)
hmma_splitk_kernel(const __half* __restrict__ input,
                   const int64_t* __restrict__ packed,
                   const uint8_t* __restrict__ scales,
                   const float* __restrict__ global_scale,
                   int M,
                   int N,
                   int K,
                   __half* __restrict__ output) {
  __shared__ float partial[kSplit][8][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = static_cast<int>(blockIdx.x);
  const int n0 = tile * 32;
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int qword = (n0 + lane_col) >> 3;
  const int sub = lane_col & 7;
  const int groups = K >> 4;
  const int per_warp = (groups + kSplit - 1) / kSplit;
  const int g_begin = warp * per_warp;
  const int g_end = min(groups, g_begin + per_warp);
  const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);
  const __half* a_row = arow < M ? input + static_cast<int64_t>(arow) * K : nullptr;
  const uint2* row = reinterpret_cast<const uint2*>(
      packed + (static_cast<int64_t>(tile) * groups) * 32 + lane);
  float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  hmma_group_run<4>(row, scales + (qword << 3), sub, a_row, N, g_begin, g_end, c);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    partial[warp][hmma_acc_row(lane, i)][hmma_acc_col(lane, i)] = c[i];
  __syncthreads();
  const float gscale = global_scale[0];
  for (int idx = threadIdx.x; idx < M * 32; idx += blockDim.x) {
    const int r = idx >> 5;
    const int col = idx & 31;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kSplit; ++w) sum += partial[w][r][col];
    output[static_cast<int64_t>(r) * N + n0 + col] = __float2half_rn(sum * gscale);
  }
}

// Grid (N/32, routes). Invalid routes store zeros, as moe_hmma_kernel does.
template <int kSplit, bool kPerRouteInput, bool kApplyRoute>
__global__ void __launch_bounds__(32 * kSplit)
moe_hmma_splitk_kernel(const __half* __restrict__ input,
                       const int64_t* __restrict__ packed,
                       const uint8_t* __restrict__ scales,
                       const float* __restrict__ global_scale,
                       const int* __restrict__ topk_ids,
                       const float* __restrict__ topk_weights,
                       int topk,
                       int num_experts,
                       int N,
                       int K,
                       __half* __restrict__ output,
                       int routed_routes,
                       const int64_t* __restrict__ shared_packed,
                       const uint8_t* __restrict__ shared_scales,
                       const float* __restrict__ shared_global_scale) {
  __shared__ float partial[kSplit][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = static_cast<int>(blockIdx.x);
  const int route = static_cast<int>(blockIdx.y);
  const int n0 = tile * 32;
  __half* out = output + static_cast<int64_t>(route) * N + n0;
  // Routes past routed_routes are one shared-expert row per token, weight 1.
  const bool shared = route >= routed_routes;
  const int expert = shared ? 0 : topk_ids[route];
  if (!shared && (expert < 0 || expert >= num_experts)) {
    if (threadIdx.x < 32) out[threadIdx.x] = __float2half_rn(0.f);
    return;
  }
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int qword = (n0 + lane_col) >> 3;
  const int sub = lane_col & 7;
  const int groups = K >> 4;
  const int tiles = N >> 5;
  const int per_warp = (groups + kSplit - 1) / kSplit;
  const int g_begin = warp * per_warp;
  const int g_end = min(groups, g_begin + per_warp);
  const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int token = kPerRouteInput ? route : (shared ? route - routed_routes : route / topk);
  const __half* a_row = arow == 0 ? input + static_cast<int64_t>(token) * K : nullptr;
  const uint2* row = reinterpret_cast<const uint2*>(
      shared ? shared_packed + (static_cast<int64_t>(tile) * groups) * 32 + lane
             : packed + ((static_cast<int64_t>(expert) * tiles + tile) * groups) * 32 + lane);
  const uint8_t* expert_scales =
      shared ? shared_scales : scales + (static_cast<int64_t>(expert) * groups) * N;
  float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  hmma_group_run<4>(row, expert_scales + (qword << 3), sub, a_row, N, g_begin, g_end, c);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (hmma_acc_row(lane, i) == 0) partial[warp][hmma_acc_col(lane, i)] = c[i];
  __syncthreads();
  if (threadIdx.x < 32) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kSplit; ++w) sum += partial[w][threadIdx.x];
    const float gscale = shared ? shared_global_scale[0] : global_scale[expert];
    if constexpr (kApplyRoute) {
      sum = sum * (shared ? 1.0f : topk_weights[route]) * gscale;
    } else {
      sum = sum * gscale;
    }
    out[threadIdx.x] = __float2half_rn(sum);
  }
}

// moe_hmma_splitk_kernel with the routes that share an expert stacked into
// the HMMA's A rows, so each expert is streamed once. The block of a group's
// first route does the work; the others return. Rows do not mix in the HMMA,
// so each output row is bitwise what the one-route kernel gives.
// A token's top-k experts are distinct, so up to four tokens fill at most four
// rows; the smaller partials buffer lets two blocks share an SM.
inline constexpr int kGroupRows = 4;

// Two blocks per SM and these unroll depths measured fastest at four tokens
// (TP8 w13 and w2 slices); they change scheduling only, not the sum order.
template <int kSplit, bool kPerRouteInput, bool kApplyRoute>
__global__ void __launch_bounds__(32 * kSplit, 2)
moe_hmma_grouped_kernel(const __half* __restrict__ input,
                        const int64_t* __restrict__ packed,
                        const uint8_t* __restrict__ scales,
                        const float* __restrict__ global_scale,
                        const int* __restrict__ topk_ids,
                        const float* __restrict__ topk_weights,
                        int topk,
                        int num_experts,
                        int N,
                        int K,
                        __half* __restrict__ output,
                        int routed_routes,
                        int shared_rows,
                        const int64_t* __restrict__ shared_packed,
                        const uint8_t* __restrict__ shared_scales,
                        const float* __restrict__ shared_global_scale) {
  __shared__ float partial[kSplit][kGroupRows][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tile = static_cast<int>(blockIdx.x);
  const int route = static_cast<int>(blockIdx.y);
  const int n0 = tile * 32;
  const bool shared = route >= routed_routes;
  // Every warp reads all route ids (at most 32), so grouping costs no extra round trip.
  const int lane_expert = lane < routed_routes ? topk_ids[lane] : -1;
  const int expert = shared ? 0 : __shfl_sync(0xffffffffu, lane_expert, route);
  if (!shared && (expert < 0 || expert >= num_experts)) {
    if (threadIdx.x < 32) {
      output[static_cast<int64_t>(route) * N + n0 + threadIdx.x] = __float2half_rn(0.f);
    }
    return;
  }
  // Bit r of `mine` is set for the routes r this block computes, at most kGroupRows;
  // groups are runs of kGroupRows in route order and shared rows form their own groups.
  unsigned mine;
  if (shared) {
    const int j = route - routed_routes;
    if (j % kGroupRows != 0) return;
    mine = ((1u << min(kGroupRows, shared_rows - j)) - 1u) << j;
  } else {
    const unsigned match = __ballot_sync(0xffffffffu, lane_expert == expert);
    const unsigned below = (1u << route) - 1u;
    if (__popc(match & below) % kGroupRows != 0) return;
    mine = match & ~below;
    for (int extra = __popc(mine) - kGroupRows; extra > 0; --extra) mine &= ~(1u << (31 - __clz(mine)));
  }
  const int count = __popc(mine);
  // Route of row i: the i-th set bit of `mine` (shared rows are offset by routed_routes).
  auto row_route = [&](int i) {
    unsigned m = mine;
    for (int k = 0; k < i; ++k) m &= m - 1u;
    return (__ffs(m) - 1) + (shared ? routed_routes : 0);
  };
  const int lane_col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
  const int qword = (n0 + lane_col) >> 3;
  const int sub = lane_col & 7;
  const int groups = K >> 4;
  const int tiles = N >> 5;
  const int per_warp = (groups + kSplit - 1) / kSplit;
  const int g_begin = warp * per_warp;
  const int g_end = min(groups, g_begin + per_warp);
  const int arow = (lane & 3) + ((lane & 16) ? 4 : 0);
  const __half* a_row = nullptr;
  if (arow < count) {
    const int r = row_route(arow);
    const int token = kPerRouteInput ? r : (shared ? r - routed_routes : r / topk);
    a_row = input + static_cast<int64_t>(token) * K;
  }
  const uint2* row = reinterpret_cast<const uint2*>(
      shared ? shared_packed + (static_cast<int64_t>(tile) * groups) * 32 + lane
             : packed + ((static_cast<int64_t>(expert) * tiles + tile) * groups) * 32 + lane);
  const uint8_t* expert_scales =
      shared ? shared_scales : scales + (static_cast<int64_t>(expert) * groups) * N;
  float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  hmma_group_run<(kSplit >= 8 ? 4 : 2)>(row, expert_scales + (qword << 3), sub, a_row, N, g_begin, g_end, c);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (hmma_acc_row(lane, i) < kGroupRows) partial[warp][hmma_acc_row(lane, i)][hmma_acc_col(lane, i)] = c[i];
  __syncthreads();
  const float gscale = shared ? shared_global_scale[0] : global_scale[expert];
  for (int idx = threadIdx.x; idx < count * 32; idx += blockDim.x) {
    const int i = idx >> 5;
    const int col = idx & 31;
    const int r = row_route(i);
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kSplit; ++w) sum += partial[w][i][col];
    if constexpr (kApplyRoute) {
      sum = sum * (shared ? 1.0f : topk_weights[r]) * gscale;
    } else {
      sum = sum * gscale;
    }
    output[static_cast<int64_t>(r) * N + n0 + col] = __float2half_rn(sum);
  }
}

#undef HMMA_M8N8K4

// Warps per 32-column tile: at least 8 K groups each, at most 16 warps.
int hmma_split_for(int groups, int64_t split) {
  if (split > 0) return static_cast<int>(split);
  int s = 16;
  while (s > 1 && groups / s < 8) s >>= 1;
  return s;
}

template <typename Launch>
void dispatch_split(int split, Launch&& launch) {
  switch (split) {
    case 1: launch(std::integral_constant<int, 1>{}); break;
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    default: TORCH_CHECK(false, "split must be 1, 2, 4, 8 or 16");
  }
}

void gemv_hmma_splitk(torch::Tensor input,
                      torch::Tensor packed,
                      torch::Tensor scales,
                      torch::Tensor global_scale,
                      torch::Tensor output,
                      int64_t split) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf && input.dim() == 2 &&
                  input.is_contiguous(),
              "input must be contiguous CUDA FP16 [M, K]");
  const int M = static_cast<int>(input.size(0));
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(scales.size(1));
  const int groups = K >> 4;
  TORCH_CHECK(M >= 1 && M <= 4, "HMMA GEMV batch must be 1..4");
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0, "N%32 and K%16 are required");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong &&
                  packed.size(0) == N / 32 && packed.size(1) == groups && packed.size(2) == 32,
              "packed must be int64 [N/32, K/16, 32]");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf && output.size(0) == M &&
                  output.size(1) == N && output.is_contiguous(),
              "output must be contiguous FP16 [M, N]");
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_split(hmma_split_for(groups, split), [&](auto s) {
    constexpr int S = decltype(s)::value;
    hmma_splitk_kernel<S><<<N / 32, 32 * S, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        packed.data_ptr<int64_t>(),
        reinterpret_cast<const uint8_t*>(scales.data_ptr()),
        global_scale.data_ptr<float>(),
        M,
        N,
        K,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_moe_hmma_splitk(torch::Tensor input,
                            torch::Tensor packed,
                            torch::Tensor scales,
                            torch::Tensor global_scale,
                            torch::Tensor topk_ids,
                            torch::Tensor topk_weights,
                            torch::Tensor output,
                            bool per_route_input,
                            bool apply_route_weight,
                            int64_t split,
                            int shared_rows,
                            const int64_t* shared_packed,
                            const uint8_t* shared_scales,
                            const float* shared_global_scale,
                            bool group_experts) {
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(output.size(1));
  const int routes = static_cast<int>(output.size(0)) - shared_rows;
  const int num_experts = static_cast<int>(packed.size(0));
  const int groups = K >> 4;
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf && input.is_contiguous());
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong && packed.dim() == 4 &&
                  packed.is_contiguous() && packed.size(1) == N / 32 &&
                  packed.size(2) == groups && packed.size(3) == 32);
  TORCH_CHECK(scales.is_cuda() && scales.element_size() == 1 && scales.is_contiguous() &&
                  scales.numel() == static_cast<int64_t>(num_experts) * groups * N);
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == at::kFloat &&
                  global_scale.numel() >= num_experts);
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == routes);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf && output.is_contiguous());
  const int tokens = per_route_input ? static_cast<int>(input.size(0)) - shared_rows
                                     : static_cast<int>(input.size(0));
  const int topk = per_route_input ? 1 : static_cast<int>(routes / tokens);
  TORCH_CHECK(per_route_input ? tokens == routes : tokens * topk == routes);
  TORCH_CHECK(shared_rows == 0 || per_route_input || shared_rows == tokens,
              "one shared-expert row per token");
  TORCH_CHECK(per_route_input == apply_route_weight, "unsupported moe HMMA mode");
  if (apply_route_weight) {
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.scalar_type() == at::kFloat &&
                    topk_weights.numel() == routes);
  }
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const __half* in = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
  const int64_t* pk = packed.data_ptr<int64_t>();
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  const float* gs = global_scale.data_ptr<float>();
  const int* ids = topk_ids.data_ptr<int>();
  const float* tw = apply_route_weight ? topk_weights.data_ptr<float>() : nullptr;
  __half* y = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
  TORCH_CHECK(!group_experts || (routes <= 32 && shared_rows <= 32),
              "expert grouping supports at most 32 routes");
  const dim3 grid(N / 32, routes + shared_rows);
  dispatch_split(hmma_split_for(groups, split), [&](auto s) {
    constexpr int S = decltype(s)::value;
    if (group_experts && apply_route_weight) {
      moe_hmma_grouped_kernel<S, true, true><<<grid, 32 * S, 0, stream>>>(
          in, pk, sc, gs, ids, tw, topk, num_experts, N, K, y, routes, shared_rows, shared_packed,
          shared_scales, shared_global_scale);
    } else if (group_experts) {
      moe_hmma_grouped_kernel<S, false, false><<<grid, 32 * S, 0, stream>>>(
          in, pk, sc, gs, ids, tw, topk, num_experts, N, K, y, routes, shared_rows, shared_packed,
          shared_scales, shared_global_scale);
    } else if (apply_route_weight) {
      moe_hmma_splitk_kernel<S, true, true><<<grid, 32 * S, 0, stream>>>(
          in, pk, sc, gs, ids, tw, topk, num_experts, N, K, y, routes, shared_packed, shared_scales,
          shared_global_scale);
    } else {
      moe_hmma_splitk_kernel<S, false, false><<<grid, 32 * S, 0, stream>>>(
          in, pk, sc, gs, ids, tw, topk, num_experts, N, K, y, routes, shared_packed, shared_scales,
          shared_global_scale);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_hmma_splitk(torch::Tensor input,
                     torch::Tensor packed,
                     torch::Tensor scales,
                     torch::Tensor global_scale,
                     torch::Tensor topk_ids,
                     torch::Tensor topk_weights,
                     torch::Tensor output,
                     bool per_route_input,
                     bool apply_route_weight,
                     int64_t split,
                     bool group_experts) {
  launch_moe_hmma_splitk(input, packed, scales, global_scale, topk_ids, topk_weights, output,
                         per_route_input, apply_route_weight, split, 0, nullptr, nullptr, nullptr,
                         group_experts);
}

// Routed experts plus one dense shared expert: output rows [routes..routes+tokens)
// are the shared expert, bitwise what gemv_hmma_splitk gives for that weight.
void moe_hmma_splitk_shared(torch::Tensor input,
                            torch::Tensor packed,
                            torch::Tensor scales,
                            torch::Tensor global_scale,
                            torch::Tensor topk_ids,
                            torch::Tensor topk_weights,
                            torch::Tensor output,
                            bool per_route_input,
                            int64_t split,
                            torch::Tensor shared_packed,
                            torch::Tensor shared_scales,
                            torch::Tensor shared_global_scale,
                            int64_t shared_rows,
                            bool group_experts) {
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(output.size(1));
  const int groups = K >> 4;
  TORCH_CHECK(shared_rows >= 1 && shared_rows <= 4);
  TORCH_CHECK(shared_packed.is_cuda() && shared_packed.scalar_type() == at::kLong &&
                  shared_packed.is_contiguous() && shared_packed.dim() == 3 &&
                  shared_packed.size(0) == N / 32 && shared_packed.size(1) == groups &&
                  shared_packed.size(2) == 32,
              "shared_packed must be int64 [N/32, K/16, 32]");
  TORCH_CHECK(shared_scales.is_cuda() && shared_scales.element_size() == 1 &&
                  shared_scales.is_contiguous() &&
                  shared_scales.numel() == static_cast<int64_t>(groups) * N,
              "shared_scales must be [K/16, N] bytes");
  TORCH_CHECK(shared_global_scale.is_cuda() && shared_global_scale.scalar_type() == at::kFloat &&
              shared_global_scale.numel() >= 1);
  TORCH_CHECK(shared_packed.device() == input.device() && shared_scales.device() == input.device() &&
              shared_global_scale.device() == input.device());
  launch_moe_hmma_splitk(input, packed, scales, global_scale, topk_ids, topk_weights, output,
                         per_route_input, per_route_input, split, static_cast<int>(shared_rows),
                         shared_packed.data_ptr<int64_t>(),
                         reinterpret_cast<const uint8_t*>(shared_scales.data_ptr()),
                         shared_global_scale.data_ptr<float>(), group_experts);
}

void moe_hmma(torch::Tensor input,
              torch::Tensor packed,
              torch::Tensor scales,
              torch::Tensor global_scale,
              torch::Tensor topk_ids,
              torch::Tensor topk_weights,
              torch::Tensor output,
              bool per_route_input,
              bool apply_route_weight) {
  const int K = static_cast<int>(input.size(1));
  const int N = static_cast<int>(output.size(1));
  const int routes = static_cast<int>(output.size(0));
  const int num_experts = static_cast<int>(packed.size(0));
  const int groups = K >> 4;
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf && input.is_contiguous());
  TORCH_CHECK(N % 32 == 0 && K % 16 == 0);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kLong && packed.dim() == 4 &&
                  packed.size(1) == N / 32 && packed.size(2) == groups && packed.size(3) == 32);
  TORCH_CHECK(scales.is_cuda() && scales.element_size() == 1 && scales.is_contiguous() &&
                  scales.numel() == static_cast<int64_t>(num_experts) * groups * N);
  TORCH_CHECK(global_scale.is_cuda() && global_scale.scalar_type() == at::kFloat &&
                  global_scale.numel() >= num_experts);
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == routes);
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf && output.is_contiguous());
  const int topk = per_route_input ? 1 : static_cast<int>(routes / input.size(0));
  TORCH_CHECK(per_route_input || input.size(0) * topk == routes);
  if (apply_route_weight) {
    TORCH_CHECK(topk_weights.is_cuda() && topk_weights.scalar_type() == at::kFloat &&
                    topk_weights.numel() == routes);
  }
  const c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const __half* in = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
  const int64_t* pk = packed.data_ptr<int64_t>();
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  const float* gs = global_scale.data_ptr<float>();
  const int* ids = topk_ids.data_ptr<int>();
  const float* tw = apply_route_weight ? topk_weights.data_ptr<float>() : nullptr;
  __half* y = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
  if (!per_route_input && !apply_route_weight) {
    moe_hmma_kernel<false, false><<<N / 32, 32, 0, stream>>>(
        in, pk, sc, gs, ids, tw, routes, topk, num_experts, N, K, y);
  } else if (per_route_input && apply_route_weight) {
    moe_hmma_kernel<true, true><<<N / 32, 32, 0, stream>>>(
        in, pk, sc, gs, ids, tw, routes, topk, num_experts, N, K, y);
  } else {
    TORCH_CHECK(false, "unsupported moe HMMA mode");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_reduce(torch::Tensor route_out,
                torch::Tensor topk_ids,
                torch::Tensor output,
                int64_t num_experts,
                double routed_scale) {
  const int batch = static_cast<int>(output.size(0));
  const int N = static_cast<int>(output.size(1));
  const int topk = static_cast<int>(route_out.size(0) / batch);
  TORCH_CHECK(route_out.is_cuda() && route_out.scalar_type() == at::kHalf &&
                  route_out.is_contiguous());
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kHalf && output.size(1) == N);
  TORCH_CHECK(topk_ids.numel() == static_cast<int64_t>(batch) * topk);
  const c10::cuda::CUDAGuard guard(output.device());
  const int threads = 128;
  const int blocks = (batch * N + threads - 1) / threads;
  moe_reduce_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(route_out.data_ptr<at::Half>()),
      topk_ids.data_ptr<int>(),
      batch,
      N,
      topk,
      static_cast<int>(num_experts),
      static_cast<float>(routed_scale),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sm70_glm_nvfp4_gemv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &sm70_glm_nvfp4_gemv::gemv, "SM70 NVFP4 GEMV (N multiple of 256)");
  m.def("repack", &sm70_glm_nvfp4_gemv::repack, "Repack Marlin NVFP4 for the HMMA GEMV");
  m.def("unpack_into", &sm70_glm_nvfp4_gemv::unpack_into, "Restore Marlin NVFP4 from the HMMA pack");
  m.def("unpack_experts_into", &sm70_glm_nvfp4_gemv::unpack_experts_into,
        "Restore a stack of Marlin NVFP4 experts from the HMMA pack");
  m.def("gemv_hmma", &sm70_glm_nvfp4_gemv::gemv_hmma, "SM70 NVFP4 HMMA GEMV");
  m.def("moe_hmma", &sm70_glm_nvfp4_gemv::moe_hmma, "SM70 NVFP4 routed HMMA GEMV");
  m.def("gemv_hmma_splitk", &sm70_glm_nvfp4_gemv::gemv_hmma_splitk,
        "SM70 NVFP4 HMMA GEMV, split-K across warps (split 0 = auto)");
  m.def("moe_hmma_splitk", &sm70_glm_nvfp4_gemv::moe_hmma_splitk,
        "SM70 NVFP4 routed HMMA GEMV, split-K across warps (split 0 = auto); "
        "group_experts streams each expert once for all its routes");
  m.def("moe_hmma_splitk_shared", &sm70_glm_nvfp4_gemv::moe_hmma_splitk_shared,
        "SM70 NVFP4 split-K routed HMMA GEMV with one shared-expert row per token");
  m.def("moe_reduce", &sm70_glm_nvfp4_gemv::moe_reduce, "Sum routed HMMA outputs");
}
