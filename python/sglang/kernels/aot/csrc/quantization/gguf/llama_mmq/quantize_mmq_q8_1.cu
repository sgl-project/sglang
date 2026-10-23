// Activation quantizer for the vendored llama.cpp MMQ kernels: x [M, K] in fp32, fp16 or bf16 to the
// block_q8_1_mmq layout of ggml/src/ggml-cuda/mmq.cuh.
//
// Derived from quantize_mmq_q8_1 / quantize_mmq_q8_1_cuda in llama.cpp's ggml/src/ggml-cuda/quantize.cu
// (commit in ggml/LLAMA_CPP_COMMIT; MIT License, Copyright (c) 2023-2026 The ggml authors, see ggml/LICENSE).
// Changes from upstream:
//   * the input is read as fp32, fp16 or bf16 (upstream reads fp32 only), so half-precision activations
//     need no conversion pass; scalar loads are used when the rows are not 4-element aligned;
//   * only the dense 2D case (no ids, no channels / samples, no PDL);
//   * the block sums (DS4 and D2S6 layouts) are d * sum(q), the sum of the values the kernel actually
//     multiplies, instead of the sum of the unquantized inputs. The kernels use the sum to apply a per-block
//     weight minimum or offset (Q4_1, Q5_1, Q4_K, Q5_K, Q2_K, IQ1_S, ...); with the quantized sum that term
//     cancels exactly and the result keeps only the q8_1 rounding of the products (SGLang's own q8_1
//     quantizer does the same, see gguf_kernel.cu);
//   * an all-zero block gets d = 0 and q = 0 explicitly.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "mmq.cuh"
#include "sgl_llama_mmq.h"

namespace sgl_llama_mmq {

static constexpr int kQuantizeBlockSize = 128;  // CUDA_QUANTIZE_BLOCK_SIZE_MMQ in quantize.cuh
static_assert(MATRIX_ROW_PADDING % (4 * kQuantizeBlockSize) == 0, "Risk of out-of-bounds access.");

template <typename T>
struct load4;

template <>
struct load4<float> {
  static __device__ __forceinline__ float4 vec(const float* p) {
    return *(const float4*)p;
  }
};

template <>
struct load4<half> {
  static __device__ __forceinline__ float4 vec(const half* p) {
    const uint2 u = *(const uint2*)p;
    const float2 a = __half22float2(*(const half2*)&u.x);
    const float2 b = __half22float2(*(const half2*)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
  }
};

template <>
struct load4<nv_bfloat16> {
  static __device__ __forceinline__ float4 vec(const nv_bfloat16* p) {
    const uint2 u = *(const uint2*)p;
    // bf16 -> fp32 is a 16-bit shift.
    return make_float4(
        __uint_as_float(u.x << 16),
        __uint_as_float(u.x & 0xFFFF0000u),
        __uint_as_float(u.y << 16),
        __uint_as_float(u.y & 0xFFFF0000u));
  }
};

template <typename T>
static __device__ __forceinline__ float to_float(const T v) {
  return float(v);
}
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(const nv_bfloat16 v) {
  return __bfloat162float(v);
}

// Grid: (M, K_padded / (4 * kQuantizeBlockSize)); each thread quantizes 4 consecutive values.
template <typename T, mmq_q8_1_ds_layout ds_layout, bool vec_load>
static __global__ void quantize_mmq_q8_1_kernel(
    const T* __restrict__ x, void* __restrict__ vy, const int64_t K, const int64_t ldx, const int64_t K_padded) {
  constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
  if (i0 >= K_padded) {
    return;
  }
  const int64_t row = blockIdx.x;
  const T* xr = x + row * ldx;

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (i0 < K) {  // K % 4 == 0 (K is a multiple of the weight block size)
    if constexpr (vec_load) {
      xi = load4<T>::vec(xr + i0);
    } else {
      xi = make_float4(to_float(xr[i0 + 0]), to_float(xr[i0 + 1]), to_float(xr[i0 + 2]), to_float(xr[i0 + 3]));
    }
  }

  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }

  const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
  const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);

  float sum = 0.0f;
  if constexpr (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    int sumq = (int)q.x + (int)q.y + (int)q.z + (int)q.w;
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
      sumq += __shfl_xor_sync(0xFFFFFFFF, sumq, offset, WARP_SIZE);
    }
    sum = d * (float)sumq;
  }

  const int64_t k_block = i0 / QK8_1_MMQ;  // block of 128 values along K
  const int64_t iqs = i0 % QK8_1_MMQ;      // value index in the block
  block_q8_1_mmq* y = (block_q8_1_mmq*)vy + k_block * gridDim.x + row;

  ((char4*)y->qs)[iqs / 4] = q;

  if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
    if (iqs % 16 == 0 && iqs < 96) {
      y->d2s6[2 + iqs / 16] = sum;
      if (iqs % 64 == 0) {
        y->d2s6[iqs / 64] = d;
      }
    }
  } else if (iqs % 32 == 0) {
    if constexpr (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
      y->ds4[iqs / 32] = make_half2(d, sum);
    } else {
      y->d4[iqs / 32] = d;
    }
  }
}

template <typename T, mmq_q8_1_ds_layout ds_layout>
static void launch_quantize(
    const T* x, void* yq, int64_t M, int64_t K, int64_t ldx, int64_t K_padded, bool vec_load, cudaStream_t stream) {
  const dim3 num_blocks(M, K_padded / (4 * kQuantizeBlockSize), 1);
  const dim3 block_size(kQuantizeBlockSize, 1, 1);
  if (vec_load) {
    quantize_mmq_q8_1_kernel<T, ds_layout, true><<<num_blocks, block_size, 0, stream>>>(x, yq, K, ldx, K_padded);
  } else {
    quantize_mmq_q8_1_kernel<T, ds_layout, false><<<num_blocks, block_size, 0, stream>>>(x, yq, K, ldx, K_padded);
  }
}

template <typename T>
static void
quantize_typed(const T* x, void* yq, int64_t M, int64_t K, int64_t ldx, ggml_type type, cudaStream_t stream) {
  const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
  constexpr int64_t vec_bytes = 4 * sizeof(T);
  const bool vec_load = (uintptr_t)x % vec_bytes == 0 && ldx % 4 == 0;
  switch (mmq_get_q8_1_ds_layout(type)) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_D4>(x, yq, M, K, ldx, K_padded, vec_load, stream);
      break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_DS4>(x, yq, M, K, ldx, K_padded, vec_load, stream);
      break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
      launch_quantize<T, MMQ_Q8_1_DS_LAYOUT_D2S6>(x, yq, M, K, ldx, K_padded, vec_load, stream);
      break;
    default:
      GGML_ABORT("unexpected q8_1 layout");
  }
}

void quantize_q8_1_mmq(
    const void* x, XType xtype, int64_t M, int64_t K, int64_t ldx, int64_t type, void* yq, cudaStream_t stream) {
  GGML_ASSERT(K % 4 == 0);
  GGML_ASSERT(M <= INT_MAX);
  if (M == 0) {
    return;
  }
  switch (xtype) {
    case XType::F32:
      quantize_typed((const float*)x, yq, M, K, ldx, (ggml_type)type, stream);
      break;
    case XType::F16:
      quantize_typed((const half*)x, yq, M, K, ldx, (ggml_type)type, stream);
      break;
    case XType::BF16:
      quantize_typed((const nv_bfloat16*)x, yq, M, K, ldx, (ggml_type)type, stream);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace sgl_llama_mmq
