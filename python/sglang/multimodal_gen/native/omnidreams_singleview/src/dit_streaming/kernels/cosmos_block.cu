// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// cosmos_block.cu — streaming transformer block orchestrator for Cosmos DiT.
//
// One call per layer replaces the ~14 ATen ops the previous bridge issued
// (3x adaln-LoRA × 4 ops + 3x LN + 7x matmul + 3x reshape + 3x gate add).
// Internally the orchestrator drives:
//
//   - bf16 CUTLASS RRR GEMMs   (Q/K/V/out projections, FFN GEMM1+GELU, FFN GEMM2)
//   - cosmos_layernorm_modulate (no-affine LN + (1+scale)+shift)
//   - cosmos_residual_gate     (x + gate * y)
//   - cosmos_rmsnorm_per_head  (Q/K RMSNorm before RoPE)
//   - cosmos_rope_pack         (BMHK pack + rotate-half RoPE in one kernel)
//   - cudaMemcpyAsync          (append K_rot, V into KV cache)
//   - run_cudnn_fmha_packed_qkv (cuDNN SDPA, already optimal)
//
// The CA branch reuses the same primitives but skips QKV (the K/V come
// from FlashDreams' pre-cached encoder buffers; only Q is projected here).

#include "cosmos_block.cuh"
#include "sgl_gemm_shim.cuh"
#include "attention.cuh"            // run_cudnn_fmha_packed_qkv
#include "cosmos_fp8_two_gemm.cuh"
#include "dtype_utils.cuh"
#include "ops.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>

namespace omnidreams_singleview {

namespace {

// [DIAGNOSTIC — revert after P4a debug] env-gated per-sublayer failure trace.
static inline bool cosmos_block_debug_enabled() {
  const char* v = std::getenv("OMNIDREAMS_DIT_BLOCK_DEBUG");
  return v && v[0] && v[0] != '0' && v[0] != 'f' && v[0] != 'F'
         && v[0] != 'n' && v[0] != 'N';
}
#define COSMOS_DBG_FAIL() do { \
  if (::omnidreams_singleview::cosmos_block_debug_enabled()) \
    std::fprintf(stderr, "[cosmos_block] FAIL %s:%d err=%d (%s)\n", \
                 __FILE__, __LINE__, int(err), cudaGetErrorString(err)); \
} while (0)
#define COSMOS_DBG_SYNC(tag) do { \
  if (::omnidreams_singleview::cosmos_block_debug_enabled()) { \
    cudaError_t _se = cudaDeviceSynchronize(); \
    std::fprintf(stderr, "[cosmos_block] SYNC line=%d tag=%s err=%d (%s)\n", \
                 __LINE__, tag, int(_se), cudaGetErrorString(_se)); \
  } \
} while (0)

static int cosmos_qkv_rope_threads_for_shape(int D) {
  const char* env = std::getenv("OMNIDREAMS_DIT_QKV_ROPE_THREADS");
  if (env && env[0]) {
    int requested = std::atoi(env);
    if (requested == 64 || requested == 128 || requested == 256) {
      return requested;
    }
  }
  return (D <= 128) ? 64 : 256;
}

__global__ void cosmos_bf16_to_fp8_kernel(
    const cutlass::bfloat16_t* __restrict__ src,
    cutlass::float_e4m3_t* __restrict__ dst,
    int64_t n)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  dst[idx] = to_fp8(omnidreams_singleview::to_float(src[idx]));
}

__global__ void cosmos_half_to_bf16_kernel(
    const cutlass::half_t* __restrict__ src,
    cutlass::bfloat16_t* __restrict__ dst,
    int64_t n)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = omnidreams_singleview::from_float<cutlass::bfloat16_t>(omnidreams_singleview::to_float(src[idx]));
}

__global__ void cosmos_half_to_fp8_kernel(
    const cutlass::half_t* __restrict__ src,
    cutlass::float_e4m3_t* __restrict__ dst,
    int64_t n)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  dst[idx] = to_fp8(omnidreams_singleview::to_float(src[idx]));
}

__global__ void cosmos_fp8_to_half_bf16_kernel(
    const cutlass::float_e4m3_t* __restrict__ src,
    cutlass::half_t* __restrict__ dst_half,
    cutlass::bfloat16_t* __restrict__ dst_bf16,
    int64_t n)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  cutlass::NumericConverter<float, cutlass::float_e4m3_t> to_f32;
  float value = to_f32(src[idx]);
  if (dst_half) {
    cutlass::NumericConverter<cutlass::half_t, float> to_half;
    dst_half[idx] = to_half(value);
  }
  if (dst_bf16) {
    dst_bf16[idx] = omnidreams_singleview::from_float<cutlass::bfloat16_t>(value);
  }
}

__global__ void cosmos_fill_float_kernel(float* __restrict__ dst, int64_t n, float value)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = value;
  }
}

__global__ void cosmos_fp8_bmhd_to_bhmd_kernel(
    const cutlass::float_e4m3_t* __restrict__ src,
    cutlass::float_e4m3_t* __restrict__ dst,
    int B,
    int M,
    int H,
    int D)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = int64_t(B) * M * H * D;
  if (idx >= total) return;

  const int d = int(idx % D);
  int64_t t = idx / D;
  const int h = int(t % H);
  t /= H;
  const int m = int(t % M);
  const int b = int(t / M);
  const int64_t dst_idx =
      ((int64_t(b) * H + h) * M + m) * D + d;
  dst[dst_idx] = src[idx];
}

__global__ void cosmos_col_scale_bias_to_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    const cutlass::half_t* __restrict__ bias,
    cutlass::bfloat16_t* __restrict__ output,
    float scale_mul,
    int out_features,
    int64_t total_elems,
    bool gelu)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = int(idx % out_features);
  float x = omnidreams_singleview::to_float(data[idx]) * (omnidreams_singleview::to_float(scale[col]) * scale_mul);
  if (bias) {
    x += omnidreams_singleview::to_float(bias[col]);
  }
  if (gelu) {
    constexpr float c = 0.044715f;
    float y = 0.7978845608028654f * (x + c * x * x * x);
    x = 0.5f * x * (1.0f + tanhf(y));
  }
  output[idx] = omnidreams_singleview::from_float<cutlass::bfloat16_t>(x);
}

__global__ void cosmos_col_scale_residual_gate_bf16_kernel(
    const cutlass::half_t* __restrict__ data,
    const cutlass::half_t* __restrict__ scale,
    cutlass::bfloat16_t* __restrict__ residual_inout,
    const cutlass::bfloat16_t* __restrict__ gate,
    float scale_mul,
    int out_features,
    int rows_per_b,
    int64_t total_elems)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int col = int(idx % out_features);
  int row = int(idx / out_features);
  int b_row = (rows_per_b > 0) ? (row / rows_per_b) : 0;
  float projected = omnidreams_singleview::to_float(data[idx]) * (omnidreams_singleview::to_float(scale[col]) * scale_mul);
  float g = omnidreams_singleview::to_float(gate[size_t(b_row) * out_features + col]);
  float x = omnidreams_singleview::to_float(residual_inout[idx]);
  residual_inout[idx] = omnidreams_singleview::from_float<cutlass::bfloat16_t>(x + g * projected);
}

__global__ void cosmos_scale_gate_half_kernel(
    const cutlass::half_t* __restrict__ scale,
    const cutlass::bfloat16_t* __restrict__ gate,
    cutlass::half_t* __restrict__ alpha,
    int n,
    float alpha_mul)
{
  int idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = omnidreams_singleview::to_float(scale[idx]) * omnidreams_singleview::to_float(gate[idx]) * alpha_mul;
  alpha[idx] = omnidreams_singleview::from_float<cutlass::half_t>(v);
}

__device__ inline void cosmos_atomic_max_float_nonneg(float* addr, float val) {
  val = fmaxf(val, 0.0f);
  unsigned int* addr_as_ui = reinterpret_cast<unsigned int*>(addr);
  atomicMax(addr_as_ui, __float_as_uint(val));
}

__global__ void cosmos_bf16_absmax_kernel(
    const cutlass::bfloat16_t* __restrict__ src,
    float* __restrict__ out,
    int64_t n)
{
  float local_max = 0.0f;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n;
       idx += int64_t(blockDim.x) * gridDim.x) {
    local_max = fmaxf(local_max, fabsf(omnidreams_singleview::to_float(src[idx])));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
  }

  __shared__ float warp_max[8];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = local_max;
  __syncthreads();

  float block_max = 0.0f;
  if (warp == 0) {
    block_max = (threadIdx.x < (blockDim.x >> 5)) ? warp_max[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      block_max = fmaxf(block_max, __shfl_down_sync(0xffffffffu, block_max, offset));
    }
    if (lane == 0) {
      cosmos_atomic_max_float_nonneg(out, block_max);
    }
  }
}

__global__ void cosmos_split_qkv_row_kernel(
    const cutlass::bfloat16_t* __restrict__ qkv,
    cutlass::bfloat16_t* __restrict__ q,
    cutlass::bfloat16_t* __restrict__ k,
    cutlass::bfloat16_t* __restrict__ v,
    int rows,
    int features)
{
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = int64_t(rows) * features;
  if (idx >= total) return;
  int col = int(idx % features);
  int row = int(idx / features);
  const cutlass::bfloat16_t* src_row = qkv + size_t(row) * (3 * features);
  q[idx] = src_row[col];
  k[idx] = src_row[features + col];
  v[idx] = src_row[2 * features + col];
}

__device__ __forceinline__ float cosmos_warp_sum(float x) {
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, off);
  }
  return x;
}

template <bool Enabled>
__device__ __forceinline__ float cosmos_bf16_raw_or_zero(
    const cutlass::bfloat16_t* __restrict__ src,
    size_t idx)
{
  if constexpr (Enabled) {
    return omnidreams_singleview::to_float(src[idx]);
  } else {
    return 0.0f;
  }
}

template <bool Enabled>
__device__ __forceinline__ void cosmos_store_warp_sum_if(
    float* __restrict__ sums,
    float value,
    int lane,
    int warp)
{
  if constexpr (Enabled) {
    if (lane == 0) {
      sums[warp] = value;
    }
  }
}

template <bool Enabled>
__device__ __forceinline__ float cosmos_reduce_warp_sums_if(
    float* __restrict__ sums,
    int tid,
    int warp,
    int warp_count)
{
  if constexpr (Enabled) {
    float total = (tid < warp_count) ? sums[tid] : 0.0f;
    return (warp == 0) ? cosmos_warp_sum(total) : 0.0f;
  } else {
    return 0.0f;
  }
}

template <bool Enabled>
__device__ __forceinline__ void cosmos_store_block_sum_if(
    float* __restrict__ sums,
    float value,
    int tid)
{
  if constexpr (Enabled) {
    if (tid == 0) {
      sums[0] = value;
    }
  }
}

template <bool Enabled>
__device__ __forceinline__ float cosmos_rms_from_block_sum_if(
    const float* __restrict__ sums,
    int dim,
    float eps)
{
  if constexpr (Enabled) {
    return rsqrtf(sums[0] / float(dim) + eps);
  } else {
    return 0.0f;
  }
}

static cudaError_t cosmos_bf16_to_fp8(
    const cutlass::bfloat16_t* src,
    cutlass::float_e4m3_t* dst,
    int64_t n,
    cudaStream_t stream)
{
  if (n <= 0) return cudaSuccess;
  if (!src || !dst) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_bf16_to_fp8_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(src, dst, n);
  return cudaGetLastError();
}

static cudaError_t cosmos_half_to_bf16(
    const cutlass::half_t* src,
    cutlass::bfloat16_t* dst,
    int64_t n,
    cudaStream_t stream)
{
  if (n <= 0) return cudaSuccess;
  if (!src || !dst) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_half_to_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(src, dst, n);
  return cudaGetLastError();
}

static cudaError_t cosmos_half_to_fp8(
    const cutlass::half_t* src,
    cutlass::float_e4m3_t* dst,
    int64_t n,
    cudaStream_t stream)
{
  if (n <= 0) return cudaSuccess;
  if (!src || !dst) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_half_to_fp8_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(src, dst, n);
  return cudaGetLastError();
}

static cudaError_t cosmos_fp8_to_half_bf16(
    const cutlass::float_e4m3_t* src,
    cutlass::half_t* dst_half,
    cutlass::bfloat16_t* dst_bf16,
    int64_t n,
    cudaStream_t stream)
{
  if (n <= 0) return cudaSuccess;
  if (!src || (!dst_half && !dst_bf16)) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_fp8_to_half_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst_half, dst_bf16, n);
  return cudaGetLastError();
}

static cudaError_t cosmos_fill_float(float* dst, int64_t n, float value, cudaStream_t stream)
{
  if (n <= 0) return cudaSuccess;
  if (!dst) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_fill_float_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(dst, n, value);
  return cudaGetLastError();
}

static cudaError_t cosmos_fp8_bmhd_to_bhmd(
    const cutlass::float_e4m3_t* src,
    cutlass::float_e4m3_t* dst,
    int B,
    int M,
    int H,
    int D,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  if (!src || !dst) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t total = int64_t(B) * M * H * D;
  int64_t blocks = (total + threads - 1) / threads;
  cosmos_fp8_bmhd_to_bhmd_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst, B, M, H, D);
  return cudaGetLastError();
}

static cudaError_t cosmos_col_scale_bias_to_bf16(
    const cutlass::half_t* data,
    const cutlass::half_t* scale,
    const cutlass::half_t* bias,
    cutlass::bfloat16_t* output,
    int N,
    int out_features,
    bool gelu,
    cudaStream_t stream,
    float scale_mul)
{
  if (!data || !scale || !output) return cudaErrorInvalidValue;
  int64_t total = int64_t(N) * out_features;
  if (total <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  cosmos_col_scale_bias_to_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      data, scale, bias, output, scale_mul, out_features, total, gelu);
  return cudaGetLastError();
}

static cudaError_t cosmos_col_scale_residual_gate_bf16(
    const cutlass::half_t* data,
    const cutlass::half_t* scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    int N,
    int out_features,
    int B,
    cudaStream_t stream,
    float scale_mul)
{
  if (!data || !scale || !residual_inout || !gate) return cudaErrorInvalidValue;
  int64_t total = int64_t(N) * out_features;
  if (total <= 0) return cudaSuccess;
  int rows_per_b = (B > 1) ? (N / B) : N;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  cosmos_col_scale_residual_gate_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      data, scale, residual_inout, gate, scale_mul, out_features, rows_per_b, total);
  return cudaGetLastError();
}

static cudaError_t cosmos_scale_gate_half(
    const cutlass::half_t* scale,
    const cutlass::bfloat16_t* gate,
    cutlass::half_t* alpha,
    int n,
    cudaStream_t stream,
    float alpha_mul = 1.0f)
{
  if (n <= 0) return cudaSuccess;
  if (!scale || !gate || !alpha) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  cosmos_scale_gate_half_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      scale, gate, alpha, n, alpha_mul);
  return cudaGetLastError();
}

static cudaError_t cosmos_record_bf16_absmax(
    const cutlass::bfloat16_t* src,
    float* dst,
    int64_t n,
    cudaStream_t stream)
{
  if (!dst) return cudaSuccess;
  if (!src || n <= 0) return cudaErrorInvalidValue;
  cudaError_t err = cudaMemsetAsync(dst, 0, sizeof(float), stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  blocks = blocks > 1024 ? 1024 : blocks;
  cosmos_bf16_absmax_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(src, dst, n);
  return cudaGetLastError();
}

static bool cosmos_linear_backend_allows_fp8(const CosmosBlockParams& p)
{
  return p.linear_backend == CosmosLinearBackend::FP8 ||
         p.linear_backend == CosmosLinearBackend::MIXED;
}

static bool cosmos_weight_is_fp8(const CosmosBlockParams& p, const cutlass::half_t* weight_scale)
{
  return cosmos_linear_backend_allows_fp8(p) && weight_scale != nullptr;
}

static float cosmos_fp8_activation_scale_or_one(const CosmosBlockParams& p, int site)
{
  if (!p.fp8_activation_scales || site < 0 || site >= kCosmosFp8ActivationScaleSites) {
    return 1.0f;
  }
  float scale = p.fp8_activation_scales[site];
  return scale > 0.0f ? scale : 1.0f;
}

static cudaError_t cosmos_trace_copy(
    cutlass::bfloat16_t* dst,
    const cutlass::bfloat16_t* src,
    int64_t elems,
    cudaStream_t stream)
{
  if (!dst) return cudaSuccess;
  if (!src || elems <= 0) return cudaErrorInvalidValue;
  return cudaMemcpyAsync(dst, src, size_t(elems) * sizeof(cutlass::bfloat16_t),
                         cudaMemcpyDeviceToDevice, stream);
}

static cudaError_t cosmos_split_qkv_row(
    const cutlass::bfloat16_t* qkv,
    cutlass::bfloat16_t* q,
    cutlass::bfloat16_t* k,
    cutlass::bfloat16_t* v,
    int rows,
    int features,
    cudaStream_t stream)
{
  if (rows <= 0 || features <= 0) return cudaSuccess;
  if (!qkv || !q || !k || !v) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t total = int64_t(rows) * features;
  int64_t blocks = (total + threads - 1) / threads;
  cosmos_split_qkv_row_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      qkv, q, k, v, rows, features);
  return cudaGetLastError();
}

template <typename ElementT>
__global__ void cosmos_pack_rope_to_fp8_kernel(
    const ElementT* __restrict__ src,
    const ElementT* __restrict__ cos_tab,
    const ElementT* __restrict__ sin_tab,
    cutlass::float_e4m3_t* __restrict__ dst,
    cutlass::float_e4m3_t* __restrict__ dst_bhmd,
    int B, int M, int H, int D)
{
  int row_m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  const int total_m = B * M;
  if (row_m >= total_m || h >= H) return;

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  int b = row_m / M;
  int m = row_m - b * M;
  int d_half = D / 2;
  size_t base_src = size_t(row_m) * (H * D) + size_t(h) * D;
  size_t base_dst = (size_t(row_m) * H + size_t(h)) * D;
  size_t base_dst_bhmd = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D;
  size_t base_rt  = size_t(row_m) * D;

  for (int d = tid; d < D; d += blockDim.x) {
    float x_d  = omnidreams_singleview::to_float(src[base_src + d]);
    int   d_op = (d < d_half) ? (d + d_half) : (d - d_half);
    float x_op = omnidreams_singleview::to_float(src[base_src + d_op]);
    float c    = omnidreams_singleview::to_float(cos_tab[base_rt + d]);
    float s    = omnidreams_singleview::to_float(sin_tab[base_rt + d]);
    float rot  = (d < d_half) ? (-x_op) : x_op;
    cutlass::float_e4m3_t out = to_fp8(x_d * c + rot * s);
    dst[base_dst + d] = out;
    if (dst_bhmd) {
      dst_bhmd[base_dst_bhmd + d] = out;
    }
  }
}

template <typename ElementT>
static cudaError_t cosmos_pack_rope_to_fp8(
    const ElementT* src,
    const ElementT* cos_tab,
    const ElementT* sin_tab,
    cutlass::float_e4m3_t* dst,
    cutlass::float_e4m3_t* dst_bhmd,
    int B, int M, int H, int D,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  if (!src || !cos_tab || !sin_tab || !dst) return cudaErrorInvalidValue;
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, B * M);
  cosmos_pack_rope_to_fp8_kernel<ElementT><<<grid, threads, 0, stream>>>(
      src, cos_tab, sin_tab, dst, dst_bhmd, B, M, H, D);
  return cudaGetLastError();
}

template <typename ElementT>
__global__ void cosmos_bf16_to_fp8_cache_write_kernel(
    const ElementT* __restrict__ src,       // [M, H, D]
    cutlass::float_e4m3_t* __restrict__ dst, // [B, cap, H, D]
    int M, int H, int D,
    int write_start, int cache_cap, int b_idx)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  size_t src_off = (size_t(m) * H + size_t(h)) * D;
  size_t dst_off = ((size_t(b_idx) * cache_cap + size_t(write_start) + size_t(m)) * H + size_t(h)) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    dst[dst_off + d] = to_fp8(omnidreams_singleview::to_float(src[src_off + d]));
  }
}

template <typename ElementT>
static cudaError_t cosmos_write_kv_cache_fp8(
    const ElementT* k_src,
    const ElementT* v_src,
    cutlass::float_e4m3_t* k_cache,
    cutlass::float_e4m3_t* v_cache,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0) return cudaSuccess;
  if (!k_src || !v_src || !k_cache || !v_cache) return cudaErrorInvalidValue;
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, M);
  for (int b = 0; b < B; ++b) {
    cosmos_bf16_to_fp8_cache_write_kernel<ElementT><<<grid, threads, 0, stream>>>(
        k_src + size_t(b) * M * H * D, k_cache, M, H, D, write_start, cache_cap, b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    cosmos_bf16_to_fp8_cache_write_kernel<ElementT><<<grid, threads, 0, stream>>>(
        v_src + size_t(b) * M * H * D, v_cache, M, H, D, write_start, cache_cap, b);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

static cudaError_t cosmos_linear_prequantized_fp8(
    const CosmosBlockParams& p,
    const cutlass::float_e4m3_t* input_fp8,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* output,
    int N,
    int in_features,
    int out_features,
    bool gelu,
    cudaStream_t stream,
    float input_scale = 1.0f)
{
  if (p.linear_backend != CosmosLinearBackend::FP8) {
    if (p.linear_backend != CosmosLinearBackend::MIXED) {
      return cudaErrorNotSupported;
    }
  }
  if (!cosmos_weight_is_fp8(p, weight_scale)) {
    return cudaErrorNotSupported;
  }
  if (!input_fp8 || !p.buf.linear_half_scratch || !weight_scale) {
    return cudaErrorInvalidValue;
  }

  if (!gelu && input_scale == 1.0f) {
    // sgl-kernel fp8_scaled_mm: colscale[N] == scales_b (per-output col).
    {
      at::Tensor out_sgl = omnidreams_singleview::sgl_linear_rcr_fp8_colscale_bf16(
          input_fp8, weight, weight_scale,
          N, in_features, out_features, stream);
      size_t bytes = out_sgl.numel() * out_sgl.element_size();
      cudaMemcpyAsync(output, out_sgl.data_ptr(), bytes,
                      cudaMemcpyDeviceToDevice, stream);
      return cudaSuccess;
    }
  }

  // Fallthrough (gelu=true or input_scale != 1.0): sgl bare FP8 GEMM + post-op.
  {
    at::Tensor out_sgl = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        input_fp8, weight, N, in_features, out_features);
    size_t bytes = out_sgl.numel() * out_sgl.element_size();
    cudaMemcpyAsync(p.buf.linear_half_scratch, out_sgl.data_ptr(), bytes,
                    cudaMemcpyDeviceToDevice, stream);
  }
  return cosmos_col_scale_bias_to_bf16(
      p.buf.linear_half_scratch, weight_scale, nullptr, output,
      N, out_features, gelu, stream, input_scale);

static cudaError_t cosmos_linear_bf16_or_fp8(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* input,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* output,
    int N,
    int in_features,
    int out_features,
    bool gelu,
    cudaStream_t stream)
{
  const bool use_fp8 = cosmos_weight_is_fp8(p, weight_scale);
  if (!use_fp8) {
    if (p.linear_backend == CosmosLinearBackend::FP8) {
      return cudaErrorInvalidValue;
    }
    auto* w_bf16 = reinterpret_cast<const cutlass::bfloat16_t*>(weight);
    return gelu
        ? cutlass_linear_layer_rrr_gelu_bf16(input, w_bf16, nullptr, output,
                                             N, in_features, out_features, stream)
        : cutlass_linear_layer_rrr_bf16(input, w_bf16, nullptr, output,
                                        N, in_features, out_features, stream);
  }

  if (!cosmos_linear_backend_allows_fp8(p)) {
    return cudaErrorNotSupported;
  }
  if (!p.buf.linear_fp8_scratch) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cosmos_bf16_to_fp8(
      input, p.buf.linear_fp8_scratch, int64_t(N) * in_features, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  return cosmos_linear_prequantized_fp8(
      p, p.buf.linear_fp8_scratch, weight, weight_scale, output,
      N, in_features, out_features, gelu, stream);
}

static cudaError_t cosmos_linear_bf16_or_fp8_prepared(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* input,
    const void* weight,
    const cutlass::bfloat16_t* weight_prepared,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* output,
    int N,
    int in_features,
    int out_features,
    bool gelu,
    cudaStream_t stream)
{
  const bool use_fp8 = cosmos_weight_is_fp8(p, weight_scale);
  if (!use_fp8 && weight_prepared) {
    if (p.linear_backend == CosmosLinearBackend::FP8) {
      return cudaErrorInvalidValue;
    }
    return gelu
        ? cutlass_linear_layer_rrr_gelu_bf16_prepared(
            input, weight_prepared, nullptr, output,
            N, in_features, out_features, stream)
        : cutlass_linear_layer_rrr_bf16_prepared(
            input, weight_prepared, nullptr, output,
            N, in_features, out_features, stream);
  }
  return cosmos_linear_bf16_or_fp8(
      p, input, weight, weight_scale, output,
      N, in_features, out_features, gelu, stream);
}

static cudaError_t cosmos_linear_prequantized_fp8_output(
    const CosmosBlockParams& p,
    const cutlass::float_e4m3_t* input_fp8,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::float_e4m3_t* output_fp8,
    int N,
    int in_features,
    int out_features,
    bool gelu,
    cudaStream_t stream,
    float output_scale = 1.0f)
{
  if (!cosmos_weight_is_fp8(p, weight_scale)) {
    return cudaErrorNotSupported;
  }
  if (!input_fp8 || !p.buf.linear_half_scratch || !output_fp8 || !weight_scale) {
    return cudaErrorInvalidValue;
  }

  // sgl-kernel FP8 GEMM -> bf16 scratch -> post-op (colscale+GELU+FP8 cast).
  {
    at::Tensor out_sgl = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        input_fp8, weight, N, in_features, out_features);
    size_t bytes = out_sgl.numel() * out_sgl.element_size();
    cudaMemcpyAsync(p.buf.linear_half_scratch, out_sgl.data_ptr(), bytes,
                    cudaMemcpyDeviceToDevice, stream);
  }
  return gelu
      ? apply_col_scale_bias_gelu_to_fp8(
            p.buf.linear_half_scratch, output_fp8, weight_scale, nullptr,
            N, out_features, stream, 1.0f, output_scale)
      : apply_col_scale_bias_to_fp8(
            p.buf.linear_half_scratch, output_fp8, weight_scale, nullptr,
            N, out_features, stream, 1.0f, output_scale);
}

static cudaError_t cosmos_linear_prequantized_fp8_residual(
    const CosmosBlockParams& p,
    const cutlass::float_e4m3_t* input_fp8,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream,
    float input_scale = 1.0f)
{
  if (!cosmos_weight_is_fp8(p, weight_scale)) {
    return cudaErrorNotSupported;
  }
  if (!input_fp8 || !p.buf.linear_half_scratch || !weight_scale || !residual_inout || !gate) {
    return cudaErrorInvalidValue;
  }

  // sgl-kernel FP8 GEMM -> bf16 scratch -> col_scale_residual_gate post-op.
  {
    at::Tensor out_sgl = omnidreams_singleview::sgl_linear_rcr_fp8_bare(
        input_fp8, weight, N, in_features, out_features);
    size_t bytes = out_sgl.numel() * out_sgl.element_size();
    cudaMemcpyAsync(p.buf.linear_half_scratch, out_sgl.data_ptr(), bytes,
                    cudaMemcpyDeviceToDevice, stream);
  }
  return cosmos_col_scale_residual_gate_bf16(
      p.buf.linear_half_scratch, weight_scale, residual_inout, gate,
      N, out_features, p.B, stream, input_scale);
}


// Fused: FP8 GEMM + (residual into x) + LN/modulate -> FP8 input for the next
// FP8 GEMM, all in one launch sequence (cuBLASLt FP8 GEMM + one combined
// post-op kernel). Mirrors `cosmos_linear_prequantized_fp8_residual` for the
// B==1 fast path; falls back to the unfused two-step path otherwise. Used at
// the SA-out / CA-out call sites where the next op is an FP8-only LN/modulate.
static cudaError_t cosmos_linear_prequantized_fp8_residual_layernorm_modulate_to_fp8_only(
    const CosmosBlockParams& p,
    const cutlass::float_e4m3_t* input_fp8,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    const cutlass::bfloat16_t* ln_shift,
    const cutlass::bfloat16_t* ln_scale,
    cutlass::float_e4m3_t* fp8_out,
    int N,
    int in_features,
    int out_features,
    int B,
    float eps,
    cudaStream_t stream,
    float input_scale = 1.0f)
{
  if (!cosmos_weight_is_fp8(p, weight_scale)) {
    return cudaErrorNotSupported;
  }
  if (!input_fp8 || !p.buf.linear_half_scratch || !weight_scale ||
      !residual_inout || !gate || !ln_shift || !ln_scale || !fp8_out) {
    return cudaErrorInvalidValue;
  }

  // Delegate to #3 (sgl GEMM) + LN/modulate post-op.
  cudaError_t err = cosmos_linear_prequantized_fp8_residual(
      p, input_fp8, weight, weight_scale, residual_inout, gate,
      N, in_features, out_features, stream, input_scale);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  return cosmos_layernorm_modulate_to_fp8_only<cutlass::bfloat16_t>(
      residual_inout, ln_shift, ln_scale, fp8_out,
      N, out_features, B, eps, stream);
}

static cudaError_t cosmos_linear_bf16_or_fp8_residual_prepared(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* input,
    const void* weight,
    const cutlass::bfloat16_t* weight_prepared,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
  const bool use_fp8 = cosmos_weight_is_fp8(p, weight_scale);
  if (!use_fp8 && weight_prepared) {
    if (p.linear_backend == CosmosLinearBackend::FP8) {
      return cudaErrorInvalidValue;
    }
    return cutlass_linear_layer_rrr_bf16_prepared_gated_residual(
        input, weight_prepared, gate, residual_inout,
        N, in_features, out_features, stream);
  }

  cutlass::bfloat16_t* tmp = p.buf.normed;
  cudaError_t err = cosmos_linear_bf16_or_fp8(
      p, input, weight, weight_scale, tmp,
      N, in_features, out_features, /*gelu=*/false, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  return cosmos_residual_gate<cutlass::bfloat16_t>(
      residual_inout, residual_inout, tmp, gate, N, out_features, p.B, stream);
}

static cudaError_t cosmos_linear_bf16_input_fp8_residual(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* input,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
  if (!p.buf.linear_fp8_scratch || !input) {
    return cudaErrorInvalidValue;
  }
  cudaError_t err = cosmos_bf16_to_fp8(
      input, p.buf.linear_fp8_scratch, int64_t(N) * in_features, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  return cosmos_linear_prequantized_fp8_residual(
      p, p.buf.linear_fp8_scratch, weight, weight_scale, residual_inout, gate,
      N, in_features, out_features, stream);
}

static cudaError_t cosmos_linear_half_input_fp8_residual(
    const CosmosBlockParams& p,
    const cutlass::half_t* input,
    const void* weight,
    const cutlass::half_t* weight_scale,
    cutlass::bfloat16_t* residual_inout,
    const cutlass::bfloat16_t* gate,
    int N,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
  if (!cosmos_weight_is_fp8(p, weight_scale)) {
    return cudaErrorNotSupported;
  }
  if (!p.buf.linear_fp8_scratch || !input) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cosmos_half_to_fp8(
      input, p.buf.linear_fp8_scratch, int64_t(N) * in_features, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  return cosmos_linear_prequantized_fp8_residual(
      p, p.buf.linear_fp8_scratch, weight, weight_scale, residual_inout, gate,
      N, in_features, out_features, stream);
}

static cudaError_t cosmos_attention_fp8_cudnn(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* q,
    const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v,
    const cutlass::float_e4m3_t* q_fp8,
    const cutlass::float_e4m3_t* q_fp8_bhmd,
    int q_fp8_bhmd_tokens,
    const cutlass::float_e4m3_t* k_fp8,
    const cutlass::float_e4m3_t* v_fp8,
    const cutlass::float_e4m3_t* k_fp8_bhmd,
    const cutlass::float_e4m3_t* v_fp8_bhmd,
    int k_fp8_bhmd_tokens,
    int v_fp8_bhmd_tokens,
    const cutlass::float_e4m3_t* v_fp8_bhdm,
    cutlass::bfloat16_t* out,
    cutlass::float_e4m3_t* out_fp8,
    int Mk,
    bool write_bf16_output,
    bool write_fp8_output,
    cudaStream_t stream)
{
  if (!p.buf.attn_q_fp8 || !p.buf.attn_k_fp8 || !p.buf.attn_v_fp8) {
    return cudaErrorInvalidValue;
  }
  const int64_t q_elems = int64_t(p.B) * p.M * p.H * p.D;
  const int64_t kv_elems = int64_t(p.B) * Mk * p.H * p.D;
  cudaError_t err = cudaSuccess;

  const cutlass::float_e4m3_t* q_attn = q_fp8;
  if (!q_attn) {
    if (!q) return cudaErrorInvalidValue;
    err = cosmos_bf16_to_fp8(q, p.buf.attn_q_fp8, q_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    q_attn = p.buf.attn_q_fp8;
  }
  const cutlass::float_e4m3_t* k_attn = k_fp8;
  const cutlass::float_e4m3_t* v_attn = v_fp8;
  if (!k_attn || !v_attn) {
    if (!k || !v) return cudaErrorInvalidValue;
    err = cosmos_bf16_to_fp8(k, p.buf.attn_k_fp8, kv_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_bf16_to_fp8(v, p.buf.attn_v_fp8, kv_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    k_attn = p.buf.attn_k_fp8;
    v_attn = p.buf.attn_v_fp8;
  }

  if (write_fp8_output && !out_fp8) {
    return cudaErrorInvalidValue;
  }
  constexpr int kScaleSlot = 0;
  constexpr int kAmaxSSlot = 4;
  constexpr int kAmaxOSlot = 8;
  if (!p.buf.attn_tc_scale || p.buf.attn_tc_scale_elems <= kAmaxOSlot) {
    return cudaErrorInvalidValue;
  }
  if (!p.buf.attn_tc_scale_is_ones) {
    err = cosmos_fill_float(p.buf.attn_tc_scale + kScaleSlot, 1, 1.0f, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }

  cutlass::float_e4m3_t* fp8_out = out_fp8;
  if (!fp8_out) {
    if (!p.buf.linear_fp8_scratch) return cudaErrorInvalidValue;
    fp8_out = p.buf.linear_fp8_scratch;
  }
  const auto sdpa_selection = select_cosmos_fp8_sdpa(p.B, p.M, Mk, p.H, p.D);
  const bool input_bhmd = sdpa_selection.layout == "bhmd";
  const cutlass::float_e4m3_t* q_cudnn = q_attn;
  const cutlass::float_e4m3_t* k_cudnn = k_attn;
  const cutlass::float_e4m3_t* v_cudnn = v_attn;
  if (input_bhmd) {
    if (!p.buf.attn_q_bhmd_fp8 || !p.buf.attn_k_bhmd_fp8 || !p.buf.attn_v_bhmd_fp8) {
      return cudaErrorInvalidValue;
    }
    const bool q_bhmd_packed = q_fp8_bhmd && q_fp8_bhmd_tokens == p.M;
    const bool k_bhmd_packed = k_fp8_bhmd && k_fp8_bhmd_tokens == Mk;
    const bool v_bhmd_packed = v_fp8_bhmd && v_fp8_bhmd_tokens == Mk;
    q_cudnn = q_bhmd_packed ? q_fp8_bhmd : p.buf.attn_q_bhmd_fp8;
    k_cudnn = k_bhmd_packed ? k_fp8_bhmd : p.buf.attn_k_bhmd_fp8;
    v_cudnn = v_bhmd_packed ? v_fp8_bhmd : p.buf.attn_v_bhmd_fp8;
    if (!q_bhmd_packed) {
      err = cosmos_fp8_bmhd_to_bhmd(q_attn, p.buf.attn_q_bhmd_fp8, p.B, p.M, p.H, p.D, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    if (!k_bhmd_packed) {
      err = cosmos_fp8_bmhd_to_bhmd(k_attn, p.buf.attn_k_bhmd_fp8, p.B, Mk, p.H, p.D, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    if (!v_bhmd_packed && v_attn != p.buf.attn_v_bhmd_fp8) {
      err = cosmos_fp8_bmhd_to_bhmd(v_attn, p.buf.attn_v_bhmd_fp8, p.B, Mk, p.H, p.D, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
  }
  (void)v_fp8_bhdm;
  err = run_cudnn_fmha_packed_qkv_fp8(
      q_cudnn, k_cudnn, v_cudnn, fp8_out,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kScaleSlot,
      p.buf.attn_tc_scale + kAmaxSSlot,
      p.buf.attn_tc_scale + kAmaxOSlot,
      p.B, p.M, Mk, p.H, p.D, /*causal=*/false, 0.0f, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  if (write_fp8_output) {
    if (!write_bf16_output) {
      return cudaSuccess;
    }
    if (!out) return cudaErrorInvalidValue;
    return cosmos_fp8_to_half_bf16(out_fp8, nullptr, out, q_elems, stream);
  }
  if (!write_bf16_output) {
    return cudaSuccess;
  }
  if (!out) return cudaErrorInvalidValue;
  return cosmos_fp8_to_half_bf16(fp8_out, nullptr, out, q_elems, stream);
}

static cudaError_t cosmos_attention_bf16_or_fp8(
    const CosmosBlockParams& p,
    const cutlass::bfloat16_t* q,
    const cutlass::bfloat16_t* k,
    const cutlass::bfloat16_t* v,
    const cutlass::float_e4m3_t* q_fp8,
    const cutlass::float_e4m3_t* q_fp8_bhmd,
    int q_fp8_bhmd_tokens,
    const cutlass::float_e4m3_t* k_fp8,
    const cutlass::float_e4m3_t* v_fp8,
    const cutlass::float_e4m3_t* k_fp8_bhmd,
    const cutlass::float_e4m3_t* v_fp8_bhmd,
    int k_fp8_bhmd_tokens,
    int v_fp8_bhmd_tokens,
    const cutlass::float_e4m3_t* v_fp8_bhdm,
    const uint8_t* q_sage3_fp4,
    const cutlass::float_e4m3_t* q_sage3_sf,
    const uint8_t* k_sage3_fp4,
    const uint8_t* v_sage3_fp4,
    const cutlass::float_e4m3_t* k_sage3_sf,
    const cutlass::float_e4m3_t* v_sage3_sf,
    int q_sage3_padded,
    int k_sage3_padded,
    cutlass::bfloat16_t* out,
    cutlass::float_e4m3_t* out_fp8,
    int Mk,
    bool write_bf16_output,
    bool write_fp8_output,
    cudaStream_t stream)

  if (p.attention_backend == CosmosAttentionBackend::SAGE3) {
    if (!write_bf16_output || write_fp8_output || !q || !k || !v || !out) {
      return cudaErrorInvalidValue;
    }
    return run_sage3_fmha_packed_qkv(
        q, k, v, out,
        p.B, p.M, Mk, p.H, p.D, /*causal=*/false, /*scale=*/0.f, stream);
  }

  if (p.attention_backend == CosmosAttentionBackend::SAGE3_FP8) {
    if (!write_bf16_output || write_fp8_output || !out) {
      return cudaErrorInvalidValue;
    }
    if (q_sage3_fp4 && q_sage3_sf && k_sage3_fp4 && v_sage3_fp4 &&
        k_sage3_sf && v_sage3_sf) {
      return run_sage3_fmha_packed_qkv_fp4(
          q_sage3_fp4, k_sage3_fp4, v_sage3_fp4,
          q_sage3_sf, k_sage3_sf, v_sage3_sf, out,
          p.B, p.M, Mk, p.H, p.D, /*causal=*/false, /*scale=*/0.f,
          q_sage3_padded, k_sage3_padded, stream);
    }
    if (q_sage3_fp4 && q_sage3_sf && k_fp8 && v_fp8) {
      return run_sage3_fmha_packed_qfp4_kvfp8(
          q_sage3_fp4, q_sage3_sf, k_fp8, v_fp8, out,
          p.B, p.M, Mk, p.H, p.D, /*causal=*/false, /*scale=*/0.f,
          q_sage3_padded, stream);
    }
    if (!q_fp8 || !k_fp8 || !v_fp8) {
      return cudaErrorInvalidValue;
    }
    return run_sage3_fmha_packed_qkv_fp8(
        q_fp8, k_fp8, v_fp8, out,
        p.B, p.M, Mk, p.H, p.D, /*causal=*/false, /*scale=*/0.f, stream);
  }

  if (p.attention_backend == CosmosAttentionBackend::CUDNN_BF16) {
    if (!out) return cudaErrorInvalidValue;
    return run_cudnn_fmha_packed_qkv(
        q, k, v, out,
        p.B, p.M, Mk, p.H, p.D, /*causal=*/false, /*scale=*/0.f, stream);
  }

  const bool use_fp8_cudnn = p.attention_backend == CosmosAttentionBackend::FP8_CUDNN;
  const bool use_fp8_dense_ref =
      p.attention_backend == CosmosAttentionBackend::FP8_DENSE_REF;
  if (!use_fp8_cudnn && !use_fp8_dense_ref) {
    return cudaErrorNotSupported;
  }
  if (!p.buf.attn_q_fp8 || !p.buf.attn_k_fp8 || !p.buf.attn_v_fp8) {
    return cudaErrorInvalidValue;
  }

  const int64_t q_elems = int64_t(p.B) * p.M * p.H * p.D;
  const int64_t kv_elems = int64_t(p.B) * Mk * p.H * p.D;
  cudaError_t err = cudaSuccess;
  const cutlass::float_e4m3_t* q_attn = q_fp8;
  if (!q_attn) {
    err = cosmos_bf16_to_fp8(q, p.buf.attn_q_fp8, q_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    q_attn = p.buf.attn_q_fp8;
  }
  const cutlass::float_e4m3_t* k_attn = k_fp8;
  const cutlass::float_e4m3_t* v_attn = v_fp8;
  if (!k_attn || !v_attn) {
    err = cosmos_bf16_to_fp8(k, p.buf.attn_k_fp8, kv_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_bf16_to_fp8(v, p.buf.attn_v_fp8, kv_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    k_attn = p.buf.attn_k_fp8;
    v_attn = p.buf.attn_v_fp8;
  }

  if (write_fp8_output && !out_fp8) {
    return cudaErrorInvalidValue;
  }
  if (!p.buf.attn_o_half && !(use_fp8_cudnn && write_fp8_output)) {
    return cudaErrorInvalidValue;
  }
  if (use_fp8_dense_ref) {
    if (!p.buf.attn_scores_half || !p.buf.attn_probs_fp8) {
      return cudaErrorInvalidValue;
    }
    err = run_cosmos_fp8_dense_ref(
        q_attn, k_attn, v_attn,
        p.buf.attn_q_bhmd_fp8,
        p.buf.attn_k_bhmd_fp8,
        p.buf.attn_v_bhmd_fp8,
        p.buf.attn_scores_half,
        p.buf.attn_probs_fp8,
        p.buf.attn_o_bhmd_half,
        p.buf.attn_o_half,
        p.B, p.M, Mk, p.H, p.D, /*causal=*/false, stream);
  } else if (use_fp8_cudnn) {
    // cuDNN frontend scalar tensors may be loaded with vectorized alignment
    // assumptions. Keep every scalar pointer 16-byte aligned by using float
    // slots spaced four floats apart.
    constexpr int kScaleSlot = 0;
    constexpr int kAmaxSSlot = 4;
    constexpr int kAmaxOSlot = 8;
    if (!p.buf.attn_tc_scale || p.buf.attn_tc_scale_elems <= kAmaxOSlot) {
      return cudaErrorInvalidValue;
    }
    if (!p.buf.attn_tc_scale_is_ones) {
      err = cosmos_fill_float(p.buf.attn_tc_scale + kScaleSlot, 1, 1.0f, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    cutlass::float_e4m3_t* fp8_out = out_fp8;
    if (!fp8_out) {
      if (!p.buf.linear_fp8_scratch) return cudaErrorInvalidValue;
      fp8_out = p.buf.linear_fp8_scratch;
    }
    const auto sdpa_selection = select_cosmos_fp8_sdpa(p.B, p.M, Mk, p.H, p.D);
    const bool input_bhmd = sdpa_selection.layout == "bhmd";
    const cutlass::float_e4m3_t* q_cudnn = q_attn;
    const cutlass::float_e4m3_t* k_cudnn = k_attn;
    const cutlass::float_e4m3_t* v_cudnn = v_attn;
    if (input_bhmd) {
      if (!p.buf.attn_q_bhmd_fp8 || !p.buf.attn_k_bhmd_fp8 || !p.buf.attn_v_bhmd_fp8) {
        return cudaErrorInvalidValue;
      }
      const bool q_bhmd_packed = q_fp8_bhmd && q_fp8_bhmd_tokens == p.M;
      const bool k_bhmd_packed = k_fp8_bhmd && k_fp8_bhmd_tokens == Mk;
      const bool v_bhmd_packed = v_fp8_bhmd && v_fp8_bhmd_tokens == Mk;
      q_cudnn = q_bhmd_packed ? q_fp8_bhmd : p.buf.attn_q_bhmd_fp8;
      k_cudnn = k_bhmd_packed ? k_fp8_bhmd : p.buf.attn_k_bhmd_fp8;
      v_cudnn = v_bhmd_packed ? v_fp8_bhmd : p.buf.attn_v_bhmd_fp8;
      if (!q_bhmd_packed) {
        err = cosmos_fp8_bmhd_to_bhmd(q_attn, p.buf.attn_q_bhmd_fp8, p.B, p.M, p.H, p.D, stream);
        if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      }
      if (!k_bhmd_packed) {
        err = cosmos_fp8_bmhd_to_bhmd(k_attn, p.buf.attn_k_bhmd_fp8, p.B, Mk, p.H, p.D, stream);
        if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      }
      if (!v_bhmd_packed && v_attn != p.buf.attn_v_bhmd_fp8) {
        err = cosmos_fp8_bmhd_to_bhmd(v_attn, p.buf.attn_v_bhmd_fp8, p.B, Mk, p.H, p.D, stream);
        if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      }
    }
    (void)v_fp8_bhdm;
    err = run_cudnn_fmha_packed_qkv_fp8(
        q_cudnn, k_cudnn, v_cudnn, fp8_out,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kScaleSlot,
        p.buf.attn_tc_scale + kAmaxSSlot,
        p.buf.attn_tc_scale + kAmaxOSlot,
        p.B, p.M, Mk, p.H, p.D, /*causal=*/false, 0.0f, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    if (!out_fp8) {
      err = cosmos_fp8_to_half_bf16(
          fp8_out,
          p.buf.attn_o_half,
          write_bf16_output ? out : nullptr,
          q_elems,
          stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  if (write_fp8_output && !use_fp8_cudnn) {
    err = cosmos_half_to_fp8(p.buf.attn_o_half, out_fp8, q_elems, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }

  if (use_fp8_cudnn && write_fp8_output) {
    if (!write_bf16_output) {
      return cudaSuccess;
    }
    if (!out) return cudaErrorInvalidValue;
    return cosmos_fp8_to_half_bf16(out_fp8, nullptr, out, q_elems, stream);
  }

  if (!write_bf16_output) {
    return cudaSuccess;
  }
  if (!out) return cudaErrorInvalidValue;
  return cosmos_half_to_bf16(p.buf.attn_o_half, out, q_elems, stream);
}

}  // namespace

static bool cosmos_block_profile_enabled() {
  const char* v = std::getenv("OMNIDREAMS_DIT_PROFILE");
  return v && v[0] && v[0] != '0';
}

// ---------------------------------------------------------------------------
// Pack [M, K] row-major activations into [M, H, D] BMHK layout AND apply
// rotate-half RoPE in one fused kernel.
//
// Layout note: with K = H * D row-major, [M, K] is already byte-equivalent
// to [M, H, D] row-major -- no transposition needed. So this kernel reduces
// to: read element, multiply with cos/sin, write back. The "pack" is just
// a memcpy when no RoPE is needed; with RoPE we apply rotate-half in-place.
//
// rotate-half convention (matches Cosmos's apply_cosmos_rope in the bridge):
//   For each [M, H, D] row, with d_half = D / 2:
//     out[..., :d_half]  = x[..., :d_half] * cos[..., :d_half]
//                        - x[..., d_half:] * sin[..., :d_half]
//     out[..., d_half:]  = x[..., d_half:] * cos[..., d_half:]
//                        + x[..., :d_half] * sin[..., d_half:]
//
// (`cos`/`sin` are already broadcast across H -- only [M, D] -- since the
// bridge precomputes them as `[M, 1, 1, D] -> [1, M, 1, D]`.)
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_pack_rope_kernel(
    const ElementT* __restrict__ src,      // [M, K] row-major (K = H * D)
    const ElementT* __restrict__ cos_tab,  // [M, D]
    const ElementT* __restrict__ sin_tab,  // [M, D]
    ElementT* __restrict__ dst,            // [M, H, D] row-major
    int M, int H, int D)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;

  int d_half = D / 2;
  size_t base_src = size_t(m) * (H * D) + size_t(h) * D;
  size_t base_dst = (size_t(m) * H + size_t(h)) * D;
  size_t base_rt  = size_t(m) * D;

  for (int d = tid; d < D; d += blockDim.x) {
    float x_d  = omnidreams_singleview::to_float(src[base_src + d]);
    int   d_op = (d < d_half) ? (d + d_half) : (d - d_half);
    float x_op = omnidreams_singleview::to_float(src[base_src + d_op]);
    // For first half (d < d_half): rotate_half pairs (x[d], x[d_op]) where d_op = d + d_half.
    //   pair: (-x_op, x_d) under rotate_half (negate second half, swap halves).
    //   So: out[d] = x_d * cos[d] + (-x_op) * sin[d].
    // For second half (d >= d_half):
    //   pair element after rotate_half is x_op (which is x[d_op]=x[d-d_half], from the first half).
    //   out[d] = x_d * cos[d] + x_op * sin[d].
    float c    = omnidreams_singleview::to_float(cos_tab[base_rt + d]);
    float s    = omnidreams_singleview::to_float(sin_tab[base_rt + d]);
    float rot  = (d < d_half) ? (-x_op) : x_op;
    dst[base_dst + d] = omnidreams_singleview::from_float<ElementT>(x_d * c + rot * s);
  }
}

template <typename ElementT>
static cudaError_t cosmos_pack_rope(
    const ElementT* src,
    const ElementT* cos_tab,
    const ElementT* sin_tab,
    ElementT* dst,
    int M, int H, int D,
    cudaStream_t stream)
{
  if (M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, M);
  cosmos_pack_rope_kernel<ElementT><<<grid, threads, 0, stream>>>(
      src, cos_tab, sin_tab, dst, M, H, D);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Split fused [M, 3K] QKV projection output into contiguous [M, K] buffers.
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_split_qkv_kernel(
    const ElementT* __restrict__ qkv,
    ElementT* __restrict__ q,
    ElementT* __restrict__ k,
    ElementT* __restrict__ v,
    int M, int K)
{
  int m = blockIdx.y;
  int which = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || which >= 3) return;

  const ElementT* src = qkv + size_t(m) * 3 * K + size_t(which) * K;
  ElementT* dst = (which == 0) ? q : ((which == 1) ? k : v);
  dst += size_t(m) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    dst[j] = src[j];
  }
}

template <typename ElementT>
cudaError_t cosmos_split_qkv(
    const ElementT* qkv,
    ElementT* q,
    ElementT* k,
    ElementT* v,
    int M, int K,
    cudaStream_t stream)
{
  if (M <= 0 || K <= 0) return cudaSuccess;
  dim3 grid(3, M);
  cosmos_split_qkv_kernel<ElementT><<<grid, 128, 0, stream>>>(qkv, q, k, v, M, K);
  return cudaGetLastError();
}

template <typename ElementT>
__global__ void cosmos_qkv_postprocess_cache_kernel(
    const ElementT* __restrict__ qkv,       // [B*M, 3K]
    const ElementT* __restrict__ q_gamma,   // [D]
    const ElementT* __restrict__ k_gamma,   // [D]
    const ElementT* __restrict__ cos_tab,   // [M, D]
    const ElementT* __restrict__ sin_tab,   // [M, D]
    ElementT* __restrict__ q_out,           // [B*M, H, D]
    ElementT* __restrict__ k_cache,         // [B, cap, H, D]
    ElementT* __restrict__ v_cache,         // [B, cap, H, D]
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    float eps)
{
  int mh = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;
  int tid = threadIdx.x;
  if (mh >= H || b >= B || m >= M) return;

  int K = H * D;
  int global_m = b * M + m;
  size_t base = size_t(global_m) * 3 * K + size_t(mh) * D;
  size_t q_base = base;
  size_t k_base = base + K;
  size_t v_base = base + 2 * K;

  extern __shared__ float smem[];
  float* q_smem = smem;
  float* k_smem = smem + blockDim.x;
  float q_acc = 0.f;
  float k_acc = 0.f;
  for (int d = tid; d < D; d += blockDim.x) {
    float qv = omnidreams_singleview::to_float(qkv[q_base + d]);
    float kv = omnidreams_singleview::to_float(qkv[k_base + d]);
    q_acc += qv * qv;
    k_acc += kv * kv;
  }
  q_smem[tid] = q_acc;
  k_smem[tid] = k_acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      q_smem[tid] += q_smem[tid + off];
      k_smem[tid] += k_smem[tid + off];
    }
    __syncthreads();
  }
  float q_rms = rsqrtf(q_smem[0] / float(D) + eps);
  float k_rms = rsqrtf(k_smem[0] / float(D) + eps);
  int d_half = D / 2;
  size_t q_out_base = (size_t(global_m) * H + mh) * D;
  size_t cache_base = ((size_t(b) * cache_cap + size_t(write_start + m)) * H + mh) * D;
  size_t rope_base = size_t(m) * D;

  for (int d = tid; d < D; d += blockDim.x) {
    int d_op = (d < d_half) ? (d + d_half) : (d - d_half);
    float c = omnidreams_singleview::to_float(cos_tab[rope_base + d]);
    float s = omnidreams_singleview::to_float(sin_tab[rope_base + d]);

    float qv = omnidreams_singleview::to_float(qkv[q_base + d]) * q_rms * omnidreams_singleview::to_float(q_gamma[d]);
    float qop = omnidreams_singleview::to_float(qkv[q_base + d_op]) * q_rms * omnidreams_singleview::to_float(q_gamma[d_op]);
    float qrot = (d < d_half) ? (-qop) : qop;
    q_out[q_out_base + d] = omnidreams_singleview::from_float<ElementT>(qv * c + qrot * s);

    float kv = omnidreams_singleview::to_float(qkv[k_base + d]) * k_rms * omnidreams_singleview::to_float(k_gamma[d]);
    float kop = omnidreams_singleview::to_float(qkv[k_base + d_op]) * k_rms * omnidreams_singleview::to_float(k_gamma[d_op]);
    float krot = (d < d_half) ? (-kop) : kop;
    k_cache[cache_base + d] = omnidreams_singleview::from_float<ElementT>(kv * c + krot * s);

    v_cache[cache_base + d] = qkv[v_base + d];
  }
}

template <typename ElementT>
static cudaError_t cosmos_qkv_postprocess_cache(
    const ElementT* qkv,
    const ElementT* q_gamma,
    const ElementT* k_gamma,
    const ElementT* cos_tab,
    const ElementT* sin_tab,
    ElementT* q_out,
    ElementT* k_cache,
    ElementT* v_cache,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  int threads = (D <= 64) ? 64 : (D <= 128 ? 128 : 256);
  dim3 grid(H, B, M);
  size_t smem = 2 * threads * sizeof(float);
  cosmos_qkv_postprocess_cache_kernel<ElementT><<<grid, threads, smem, stream>>>(
      qkv, q_gamma, k_gamma, cos_tab, sin_tab, q_out, k_cache, v_cache,
      B, M, H, D, write_start, cache_cap, 1e-6f);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Pack [M, K] row-major into [M, H, D] without RoPE (V branch).
// (Identity copy; included for clarity. Could be skipped entirely since
// [M, K] and [M, H, D] share the same byte layout when K = H * D.)
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_pack_identity_kernel(
    const ElementT* __restrict__ src,
    ElementT* __restrict__ dst,
    int M, int H, int D)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;
  size_t base_src = size_t(m) * (H * D) + size_t(h) * D;
  size_t base_dst = (size_t(m) * H + size_t(h)) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    dst[base_dst + d] = src[base_src + d];
  }
}

// ---------------------------------------------------------------------------
// Unpack [M, H, D] BMHK layout back to [M, K] row-major (FMHA output -> out proj).
// Same byte layout when K = H * D, so this is an identity copy.
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_unpack_kernel(
    const ElementT* __restrict__ src,
    ElementT* __restrict__ dst,
    int M, int H, int D)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;
  size_t base_src = (size_t(m) * H + size_t(h)) * D;
  size_t base_dst = size_t(m) * (H * D) + size_t(h) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    dst[base_dst + d] = src[base_src + d];
  }
}

// ---------------------------------------------------------------------------
// Append packed [M, H, D] BMHK K and V buffers into a [B, cap, H, D] cache
// at slot [write_start : write_start + M).
//
// Layout reminder:
//   cache:   [B, cap, H, D]   -> stride is (cap * H * D, H * D, D, 1)
//   src:     [M, H, D]        -> contiguous
//
// We do this as `cudaMemcpy2DAsync` would, but with a small kernel since
// the BMHK rows are contiguous within each batch slice and we only write
// for one batch (B=1 in the streaming path).
// ---------------------------------------------------------------------------
template <typename ElementT>
__global__ void cosmos_write_kv_cache_kernel(
    const ElementT* __restrict__ src,      // [M, H, D]
    ElementT* __restrict__ cache,          // [B, cap, H, D]
    int M, int H, int D,
    int write_start, int cache_cap, int b_idx)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;
  size_t src_off   = (size_t(m) * H + size_t(h)) * D;
  size_t dst_off   = ((size_t(b_idx) * cache_cap + size_t(write_start) + size_t(m)) * H + size_t(h)) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    cache[dst_off + d] = src[src_off + d];
  }
}

template <typename ElementT>
static cudaError_t cosmos_write_kv_cache(
    const ElementT* k_src,
    const ElementT* v_src,
    ElementT* k_cache,
    ElementT* v_cache,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0) return cudaSuccess;
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, M);
  for (int b = 0; b < B; ++b) {
    cosmos_write_kv_cache_kernel<ElementT><<<grid, threads, 0, stream>>>(
        k_src + size_t(b) * M * H * D, k_cache, M, H, D, write_start, cache_cap, b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    cosmos_write_kv_cache_kernel<ElementT><<<grid, threads, 0, stream>>>(
        v_src + size_t(b) * M * H * D, v_cache, M, H, D, write_start, cache_cap, b);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

template <typename ElementT>
__global__ void cosmos_write_kv_cache_bf16_and_fp8_kernel(
    const ElementT* __restrict__ k_src,       // [M, H, D]
    const ElementT* __restrict__ v_src,       // [M, H, D]
    ElementT* __restrict__ k_cache,           // [B, cap, H, D]
    ElementT* __restrict__ v_cache,           // [B, cap, H, D]
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8,
    int M, int H, int D,
    int write_start, int cache_cap, int b_idx)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  size_t src_off = (size_t(m) * H + size_t(h)) * D;
  size_t dst_off = ((size_t(b_idx) * cache_cap + size_t(write_start) + size_t(m)) * H + size_t(h)) * D;
  for (int d = tid; d < D; d += blockDim.x) {
    ElementT k_val = k_src[src_off + d];
    ElementT v_val = v_src[src_off + d];
    k_cache[dst_off + d] = k_val;
    v_cache[dst_off + d] = v_val;
    k_cache_fp8[dst_off + d] = to_fp8(omnidreams_singleview::to_float(k_val));
    v_cache_fp8[dst_off + d] = to_fp8(omnidreams_singleview::to_float(v_val));
  }
}

template <typename ElementT>
static cudaError_t cosmos_write_kv_cache_bf16_and_fp8(
    const ElementT* k_src,
    const ElementT* v_src,
    ElementT* k_cache,
    ElementT* v_cache,
    cutlass::float_e4m3_t* k_cache_fp8,
    cutlass::float_e4m3_t* v_cache_fp8,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0) return cudaSuccess;
  if (!k_src || !v_src || !k_cache || !v_cache || !k_cache_fp8 || !v_cache_fp8) {
    return cudaErrorInvalidValue;
  }
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, M);
  for (int b = 0; b < B; ++b) {
    cosmos_write_kv_cache_bf16_and_fp8_kernel<ElementT><<<grid, threads, 0, stream>>>(
        k_src + size_t(b) * M * H * D,
        v_src + size_t(b) * M * H * D,
        k_cache,
        v_cache,
        k_cache_fp8,
        v_cache_fp8,
        M, H, D, write_start, cache_cap, b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

template <typename ElementT>
__global__ void cosmos_write_self_kv_cache_rope_bf16_and_fp8_kernel(
    const ElementT* __restrict__ k_src,       // [M, H, D] row-major, pre-RMSNorm
    const ElementT* __restrict__ v_src,       // [M, H, D] row-major
    const ElementT* __restrict__ cos_tab,     // [M, D]
    const ElementT* __restrict__ sin_tab,     // [M, D]
    ElementT* __restrict__ k_cache,           // [B, cap, H, D]
    ElementT* __restrict__ v_cache,           // [B, cap, H, D]
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhdm,
    int M, int H, int D,
    int write_start, int cache_cap, int b_idx)
{
  int m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M || h >= H) return;

  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
  int d_half = D / 2;
  size_t src_off = (size_t(m) * H + size_t(h)) * D;
  size_t rt_off = size_t(m) * D;
  size_t cache_m = size_t(write_start) + size_t(m);
  size_t dst_off = ((size_t(b_idx) * cache_cap + cache_m) * H + size_t(h)) * D;
  size_t k_tc_off = ((size_t(b_idx) * H + size_t(h)) * cache_cap + cache_m) * D;
  size_t v_tc_off = ((size_t(b_idx) * H + size_t(h)) * D + size_t(0)) * cache_cap + cache_m;
  for (int d = tid; d < D; d += blockDim.x) {
    float k_d = omnidreams_singleview::to_float(k_src[src_off + d]);
    int d_op = (d < d_half) ? (d + d_half) : (d - d_half);
    float k_op = omnidreams_singleview::to_float(k_src[src_off + d_op]);
    float c = omnidreams_singleview::to_float(cos_tab[rt_off + d]);
    float s = omnidreams_singleview::to_float(sin_tab[rt_off + d]);
    float rot = (d < d_half) ? (-k_op) : k_op;
    float k_out_f = k_d * c + rot * s;
    ElementT k_out = omnidreams_singleview::from_float<ElementT>(k_out_f);
    ElementT v_out = v_src[src_off + d];
    if (k_cache) {
      k_cache[dst_off + d] = k_out;
    }
    if (v_cache) {
      v_cache[dst_off + d] = v_out;
    }
    cutlass::float_e4m3_t k_fp8 = to_fp8(k_out_f);
    cutlass::float_e4m3_t v_fp8 = to_fp8(omnidreams_singleview::to_float(v_out));
    k_cache_fp8[dst_off + d] = k_fp8;
    v_cache_fp8[dst_off + d] = v_fp8;
    if (k_cache_fp8_bhmd) {
      k_cache_fp8_bhmd[k_tc_off + d] = k_fp8;
    }
    if (v_cache_fp8_bhmd) {
      v_cache_fp8_bhmd[k_tc_off + d] = v_fp8;
    }
    if (v_cache_fp8_bhdm) {
      v_cache_fp8_bhdm[v_tc_off + size_t(d) * cache_cap] = v_fp8;
    }
  }
}

template <typename ElementT>
static cudaError_t cosmos_write_self_kv_cache_rope_bf16_and_fp8(
    const ElementT* k_src,
    const ElementT* v_src,
    const ElementT* cos_tab,
    const ElementT* sin_tab,
    ElementT* k_cache,
    ElementT* v_cache,
    cutlass::float_e4m3_t* k_cache_fp8,
    cutlass::float_e4m3_t* v_cache_fp8,
    cutlass::float_e4m3_t* k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhdm,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0) return cudaSuccess;
  if (!k_src || !v_src || !cos_tab || !sin_tab || !k_cache_fp8 || !v_cache_fp8) {
    return cudaErrorInvalidValue;
  }
  int threads = (D <= 128) ? 64 : 128;
  dim3 grid(H, M);
  for (int b = 0; b < B; ++b) {
    cosmos_write_self_kv_cache_rope_bf16_and_fp8_kernel<ElementT><<<grid, threads, 0, stream>>>(
        k_src + size_t(b) * M * H * D,
        v_src + size_t(b) * M * H * D,
        cos_tab,
        sin_tab,
        k_cache,
        v_cache,
        k_cache_fp8,
        v_cache_fp8,
        k_cache_fp8_bhmd,
        v_cache_fp8_bhmd,
        v_cache_fp8_bhdm,
        M, H, D, write_start, cache_cap, b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}

template <bool WriteQFp8>
__global__ void cosmos_fused_qkv_rope_cache_fp8_kernel(
    const cutlass::bfloat16_t* __restrict__ qkv_row,
    const cutlass::bfloat16_t* __restrict__ q_gamma,
    const cutlass::bfloat16_t* __restrict__ k_gamma,
    const cutlass::bfloat16_t* __restrict__ cos_tab,
    const cutlass::bfloat16_t* __restrict__ sin_tab,
    cutlass::float_e4m3_t* __restrict__ q_fp8,
    cutlass::float_e4m3_t* __restrict__ q_fp8_bhmd,
    cutlass::bfloat16_t* __restrict__ k_cache,
    cutlass::bfloat16_t* __restrict__ v_cache,
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhdm,
    int B, int M, int H, int D,
    int write_start, int cache_cap)
{
  int row_m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  const int total_m = B * M;
  if (row_m >= total_m || h >= H) return;

  __shared__ float q_warp_sums[WriteQFp8 ? 8 : 1];
  __shared__ float k_warp_sums[8];

  const int b = row_m / M;
  const int m = row_m - b * M;
  const int K = H * D;
  const int d_half = D / 2;
  const size_t row_base = size_t(row_m) * (3 * K);
  const size_t q_base = row_base + size_t(h) * D;
  const size_t k_base = row_base + size_t(K) + size_t(h) * D;
  const size_t v_base = row_base + size_t(2 * K) + size_t(h) * D;
  const size_t rt_base = size_t(row_m) * D;
  const size_t bmhk_base = (size_t(row_m) * H + size_t(h)) * D;
  const size_t cache_m = size_t(write_start) + size_t(m);
  const size_t cache_base = ((size_t(b) * cache_cap + cache_m) * H + size_t(h)) * D;
  const size_t k_tc_base = ((size_t(b) * H + size_t(h)) * cache_cap + cache_m) * D;
  const size_t v_tc_base = ((size_t(b) * H + size_t(h)) * D) * cache_cap + cache_m;

  float q_acc = 0.0f;
  float k_acc = 0.0f;
  for (int d = tid; d < D; d += blockDim.x) {
    if constexpr (WriteQFp8) {
      float q = cosmos_bf16_raw_or_zero<WriteQFp8>(qkv_row, q_base + d);
      q_acc += q * q;
    }
    float k = omnidreams_singleview::to_float(qkv_row[k_base + d]);
    k_acc += k * k;
  }
  if constexpr (WriteQFp8) {
    q_acc = cosmos_warp_sum(q_acc);
  }
  k_acc = cosmos_warp_sum(k_acc);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_count = (blockDim.x + 31) >> 5;
  if (lane == 0) {
    k_warp_sums[warp] = k_acc;
  }
  cosmos_store_warp_sum_if<WriteQFp8>(q_warp_sums, q_acc, lane, warp);
  __syncthreads();
  float q_total = cosmos_reduce_warp_sums_if<WriteQFp8>(
      q_warp_sums, tid, warp, warp_count);
  float k_total = (tid < warp_count) ? k_warp_sums[tid] : 0.0f;
  k_total = (warp == 0) ? cosmos_warp_sum(k_total) : 0.0f;
  if (tid == 0) {
    k_warp_sums[0] = k_total;
  }
  cosmos_store_block_sum_if<WriteQFp8>(q_warp_sums, q_total, tid);
  __syncthreads();
  const float q_rms = cosmos_rms_from_block_sum_if<WriteQFp8>(
      q_warp_sums, D, 1e-6f);
  const float k_rms = rsqrtf(k_warp_sums[0] / float(D) + 1e-6f);
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  for (int d = tid; d < D; d += blockDim.x) {
    int d_op = (d < d_half) ? (d + d_half) : (d - d_half);

    float c = omnidreams_singleview::to_float(cos_tab[rt_base + d]);
    float s = omnidreams_singleview::to_float(sin_tab[rt_base + d]);
    if constexpr (WriteQFp8) {
      float q_d = omnidreams_singleview::to_float(qkv_row[q_base + d]) *
                  q_rms * omnidreams_singleview::to_float(q_gamma[d]);
      float q_op = omnidreams_singleview::to_float(qkv_row[q_base + d_op]) *
                   q_rms * omnidreams_singleview::to_float(q_gamma[d_op]);
      cutlass::bfloat16_t q_d_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(q_d);
      cutlass::bfloat16_t q_op_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(q_op);
      float q_norm = omnidreams_singleview::to_float(q_d_bf16);
      float q_norm_op = omnidreams_singleview::to_float(q_op_bf16);
      float q_rot = q_norm * c + ((d < d_half) ? -q_norm_op : q_norm_op) * s;
      cutlass::float_e4m3_t q_out = to_fp8(q_rot);
      q_fp8[bmhk_base + d] = q_out;
      if (q_fp8_bhmd) {
        size_t q_tc_idx = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D + d;
        q_fp8_bhmd[q_tc_idx] = q_out;
      }
    }

    float k_d = omnidreams_singleview::to_float(qkv_row[k_base + d]) *
                k_rms * omnidreams_singleview::to_float(k_gamma[d]);
    float k_op = omnidreams_singleview::to_float(qkv_row[k_base + d_op]) *
                 k_rms * omnidreams_singleview::to_float(k_gamma[d_op]);
    cutlass::bfloat16_t k_d_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k_d);
    cutlass::bfloat16_t k_op_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k_op);
    float k_norm = omnidreams_singleview::to_float(k_d_bf16);
    float k_norm_op = omnidreams_singleview::to_float(k_op_bf16);
    float k_rot = k_norm * c + ((d < d_half) ? -k_norm_op : k_norm_op) * s;
    cutlass::bfloat16_t k_out_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k_rot);
    cutlass::bfloat16_t v_out_bf16 = qkv_row[v_base + d];
    if (k_cache) {
      k_cache[cache_base + d] = k_out_bf16;
    }
    if (v_cache) {
      v_cache[cache_base + d] = v_out_bf16;
    }
    cutlass::float_e4m3_t k_out_fp8 = to_fp8(k_rot);
    cutlass::float_e4m3_t v_out_fp8 = to_fp8(omnidreams_singleview::to_float(v_out_bf16));
    if (k_cache_fp8) {
      k_cache_fp8[cache_base + d] = k_out_fp8;
    }
    if (v_cache_fp8) {
      v_cache_fp8[cache_base + d] = v_out_fp8;
    }
    if (k_cache_fp8_bhmd) {
      k_cache_fp8_bhmd[k_tc_base + d] = k_out_fp8;
    }
    if (v_cache_fp8_bhmd) {
      v_cache_fp8_bhmd[k_tc_base + d] = v_out_fp8;
    }
    if (v_cache_fp8_bhdm) {
      v_cache_fp8_bhdm[v_tc_base + size_t(d) * cache_cap] = v_out_fp8;
    }
  }
}

template <bool WriteQFp8>
__global__ void cosmos_fused_qkv_rope_cache_fp8_d128_kernel(
    const cutlass::bfloat16_t* __restrict__ qkv_row,
    const cutlass::bfloat16_t* __restrict__ q_gamma,
    const cutlass::bfloat16_t* __restrict__ k_gamma,
    const cutlass::bfloat16_t* __restrict__ cos_tab,
    const cutlass::bfloat16_t* __restrict__ sin_tab,
    cutlass::float_e4m3_t* __restrict__ q_fp8,
    cutlass::float_e4m3_t* __restrict__ q_fp8_bhmd,
    cutlass::bfloat16_t* __restrict__ k_cache,
    cutlass::bfloat16_t* __restrict__ v_cache,
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8,
    cutlass::float_e4m3_t* __restrict__ k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_cache_fp8_bhdm,
    int B, int M, int H,
    int write_start, int cache_cap)
{
  constexpr int D = 128;
  constexpr int d_half = D / 2;
  int row_m = blockIdx.y;
  int h = blockIdx.x;
  int tid = threadIdx.x;
  const int total_m = B * M;
  if (row_m >= total_m || h >= H || tid >= d_half) return;

  __shared__ float q_warp_sums[WriteQFp8 ? 2 : 1];
  __shared__ float k_warp_sums[2];

  const int b = row_m / M;
  const int m = row_m - b * M;
  const int K = H * D;
  const size_t row_base = size_t(row_m) * (3 * K);
  const size_t q_base = row_base + size_t(h) * D;
  const size_t k_base = row_base + size_t(K) + size_t(h) * D;
  const size_t v_base = row_base + size_t(2 * K) + size_t(h) * D;
  const size_t rt_base = size_t(row_m) * D;
  const size_t bmhk_base = (size_t(row_m) * H + size_t(h)) * D;
  const size_t cache_m = size_t(write_start) + size_t(m);
  const size_t cache_base = ((size_t(b) * cache_cap + cache_m) * H + size_t(h)) * D;
  const size_t k_tc_base = ((size_t(b) * H + size_t(h)) * cache_cap + cache_m) * D;
  const size_t v_tc_base = ((size_t(b) * H + size_t(h)) * D) * cache_cap + cache_m;

  const int j0 = tid;
  const int j1 = tid + d_half;
  float q0_raw = 0.0f;
  float q1_raw = 0.0f;
  float q_acc = 0.0f;
  if constexpr (WriteQFp8) {
    q0_raw = omnidreams_singleview::to_float(qkv_row[q_base + j0]);
    q1_raw = omnidreams_singleview::to_float(qkv_row[q_base + j1]);
    q_acc = q0_raw * q0_raw + q1_raw * q1_raw;
    q_acc = cosmos_warp_sum(q_acc);
  }
  float k0_raw = omnidreams_singleview::to_float(qkv_row[k_base + j0]);
  float k1_raw = omnidreams_singleview::to_float(qkv_row[k_base + j1]);
  float k_acc = k0_raw * k0_raw + k1_raw * k1_raw;

  k_acc = cosmos_warp_sum(k_acc);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    k_warp_sums[warp] = k_acc;
  }
  cosmos_store_warp_sum_if<WriteQFp8>(q_warp_sums, q_acc, lane, warp);
  __syncthreads();
  float q_total = cosmos_reduce_warp_sums_if<WriteQFp8>(
      q_warp_sums, tid, warp, 2);
  float k_total = (tid < 2) ? k_warp_sums[tid] : 0.0f;
  k_total = (warp == 0) ? cosmos_warp_sum(k_total) : 0.0f;
  if (tid == 0) {
    k_warp_sums[0] = k_total;
  }
  cosmos_store_block_sum_if<WriteQFp8>(q_warp_sums, q_total, tid);
  __syncthreads();
  const float q_rms = cosmos_rms_from_block_sum_if<WriteQFp8>(
      q_warp_sums, D, 1e-6f);
  const float k_rms = rsqrtf(k_warp_sums[0] / float(D) + 1e-6f);
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  float c0 = omnidreams_singleview::to_float(cos_tab[rt_base + j0]);
  float s0 = omnidreams_singleview::to_float(sin_tab[rt_base + j0]);
  float c1 = omnidreams_singleview::to_float(cos_tab[rt_base + j1]);
  float s1 = omnidreams_singleview::to_float(sin_tab[rt_base + j1]);

  if constexpr (WriteQFp8) {
    float q0 = q0_raw * q_rms * omnidreams_singleview::to_float(q_gamma[j0]);
    float q1 = q1_raw * q_rms * omnidreams_singleview::to_float(q_gamma[j1]);
    cutlass::bfloat16_t q0_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(q0);
    cutlass::bfloat16_t q1_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(q1);
    float qn0 = omnidreams_singleview::to_float(q0_bf16);
    float qn1 = omnidreams_singleview::to_float(q1_bf16);
    cutlass::float_e4m3_t q_out0 = to_fp8(qn0 * c0 - qn1 * s0);
    cutlass::float_e4m3_t q_out1 = to_fp8(qn1 * c1 + qn0 * s1);
    q_fp8[bmhk_base + j0] = q_out0;
    q_fp8[bmhk_base + j1] = q_out1;
    if (q_fp8_bhmd) {
      size_t q_tc_base = ((size_t(b) * H + size_t(h)) * M + size_t(m)) * D;
      q_fp8_bhmd[q_tc_base + j0] = q_out0;
      q_fp8_bhmd[q_tc_base + j1] = q_out1;
    }
  }

  float k0 = k0_raw * k_rms * omnidreams_singleview::to_float(k_gamma[j0]);
  float k1 = k1_raw * k_rms * omnidreams_singleview::to_float(k_gamma[j1]);
  cutlass::bfloat16_t k0_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k0);
  cutlass::bfloat16_t k1_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k1);
  float kn0 = omnidreams_singleview::to_float(k0_bf16);
  float kn1 = omnidreams_singleview::to_float(k1_bf16);
  float k_rot0 = kn0 * c0 - kn1 * s0;
  float k_rot1 = kn1 * c1 + kn0 * s1;
  cutlass::bfloat16_t k_out0_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k_rot0);
  cutlass::bfloat16_t k_out1_bf16 = omnidreams_singleview::from_float<cutlass::bfloat16_t>(k_rot1);
  cutlass::bfloat16_t v_out0_bf16 = qkv_row[v_base + j0];
  cutlass::bfloat16_t v_out1_bf16 = qkv_row[v_base + j1];
  if (k_cache) {
    k_cache[cache_base + j0] = k_out0_bf16;
    k_cache[cache_base + j1] = k_out1_bf16;
  }
  if (v_cache) {
    v_cache[cache_base + j0] = v_out0_bf16;
    v_cache[cache_base + j1] = v_out1_bf16;
  }
  cutlass::float_e4m3_t k_out0_fp8 = to_fp8(k_rot0);
  cutlass::float_e4m3_t k_out1_fp8 = to_fp8(k_rot1);
  cutlass::float_e4m3_t v_out0_fp8 = to_fp8(omnidreams_singleview::to_float(v_out0_bf16));
  cutlass::float_e4m3_t v_out1_fp8 = to_fp8(omnidreams_singleview::to_float(v_out1_bf16));
  if (k_cache_fp8) {
    k_cache_fp8[cache_base + j0] = k_out0_fp8;
    k_cache_fp8[cache_base + j1] = k_out1_fp8;
  }
  if (v_cache_fp8) {
    v_cache_fp8[cache_base + j0] = v_out0_fp8;
    v_cache_fp8[cache_base + j1] = v_out1_fp8;
  }
  if (k_cache_fp8_bhmd) {
    k_cache_fp8_bhmd[k_tc_base + j0] = k_out0_fp8;
    k_cache_fp8_bhmd[k_tc_base + j1] = k_out1_fp8;
  }
  if (v_cache_fp8_bhmd) {
    v_cache_fp8_bhmd[k_tc_base + j0] = v_out0_fp8;
    v_cache_fp8_bhmd[k_tc_base + j1] = v_out1_fp8;
  }
  if (v_cache_fp8_bhdm) {
    v_cache_fp8_bhdm[v_tc_base + size_t(j0) * cache_cap] = v_out0_fp8;
    v_cache_fp8_bhdm[v_tc_base + size_t(j1) * cache_cap] = v_out1_fp8;
  }
}

template <bool WriteQFp8>
static cudaError_t cosmos_launch_fused_qkv_rope_cache_fp8_kernel(
    dim3 grid,
    int threads,
    cudaStream_t stream,
    const cutlass::bfloat16_t* qkv_row,
    const cutlass::bfloat16_t* q_gamma,
    const cutlass::bfloat16_t* k_gamma,
    const cutlass::bfloat16_t* cos_tab,
    const cutlass::bfloat16_t* sin_tab,
    cutlass::float_e4m3_t* q_fp8,
    cutlass::float_e4m3_t* q_fp8_bhmd,
    cutlass::bfloat16_t* k_cache,
    cutlass::bfloat16_t* v_cache,
    cutlass::float_e4m3_t* k_cache_fp8,
    cutlass::float_e4m3_t* v_cache_fp8,
    cutlass::float_e4m3_t* k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhdm,
    int B, int M, int H, int D,
    int write_start, int cache_cap)
{
  cosmos_fused_qkv_rope_cache_fp8_kernel<WriteQFp8><<<grid, threads, 0, stream>>>(
      qkv_row, q_gamma, k_gamma, cos_tab, sin_tab,
      q_fp8, q_fp8_bhmd,
      k_cache, v_cache,
      k_cache_fp8, v_cache_fp8,
      k_cache_fp8_bhmd, v_cache_fp8_bhmd, v_cache_fp8_bhdm,
      B, M, H, D, write_start, cache_cap);
  return cudaGetLastError();
}

static cudaError_t cosmos_fused_qkv_rope_cache_fp8(
    const cutlass::bfloat16_t* qkv_row,
    const cutlass::bfloat16_t* q_gamma,
    const cutlass::bfloat16_t* k_gamma,
    const cutlass::bfloat16_t* cos_tab,
    const cutlass::bfloat16_t* sin_tab,
    cutlass::float_e4m3_t* q_fp8,
    cutlass::float_e4m3_t* q_fp8_bhmd,
    cutlass::bfloat16_t* k_cache,
    cutlass::bfloat16_t* v_cache,
    cutlass::float_e4m3_t* k_cache_fp8,
    cutlass::float_e4m3_t* v_cache_fp8,
    cutlass::float_e4m3_t* k_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhmd,
    cutlass::float_e4m3_t* v_cache_fp8_bhdm,
    int B, int M, int H, int D,
    int write_start, int cache_cap,
    cudaStream_t stream)
{
  if (B <= 0 || M <= 0 || H <= 0 || D <= 0) return cudaSuccess;
  if (!qkv_row || !q_gamma || !k_gamma || !cos_tab || !sin_tab) {
    return cudaErrorInvalidValue;
  }
  int threads = cosmos_qkv_rope_threads_for_shape(D);
  dim3 grid(H, B * M);
  if (D == 128 && threads == 64) {
    if (q_fp8) {
      cosmos_fused_qkv_rope_cache_fp8_d128_kernel<true><<<grid, 64, 0, stream>>>(
          qkv_row, q_gamma, k_gamma, cos_tab, sin_tab,
          q_fp8, q_fp8_bhmd,
          k_cache, v_cache,
          k_cache_fp8, v_cache_fp8,
          k_cache_fp8_bhmd, v_cache_fp8_bhmd, v_cache_fp8_bhdm,
          B, M, H, write_start, cache_cap);
    } else {
      cosmos_fused_qkv_rope_cache_fp8_d128_kernel<false><<<grid, 64, 0, stream>>>(
          qkv_row, q_gamma, k_gamma, cos_tab, sin_tab,
          q_fp8, q_fp8_bhmd,
          k_cache, v_cache,
          k_cache_fp8, v_cache_fp8,
          k_cache_fp8_bhmd, v_cache_fp8_bhmd, v_cache_fp8_bhdm,
          B, M, H, write_start, cache_cap);
    }
    return cudaGetLastError();
  }
  auto launch = q_fp8
      ? &cosmos_launch_fused_qkv_rope_cache_fp8_kernel<true>
      : &cosmos_launch_fused_qkv_rope_cache_fp8_kernel<false>;
  return launch(
      grid, threads, stream,
      qkv_row, q_gamma, k_gamma, cos_tab, sin_tab,
      q_fp8, q_fp8_bhmd,
      k_cache, v_cache,
      k_cache_fp8, v_cache_fp8,
      k_cache_fp8_bhmd, v_cache_fp8_bhmd, v_cache_fp8_bhdm,
      B, M, H, D, write_start, cache_cap);
}

// ---------------------------------------------------------------------------
// Streaming transformer block orchestrator.
// ---------------------------------------------------------------------------
cudaError_t cosmos_run_transformer_block_streaming(
    const CosmosBlockParams& p,
    cudaStream_t stream)
{
  using bf16 = cutlass::bfloat16_t;
  cudaError_t err;

  const int M    = p.M;
  const int K    = p.K;
  const int H    = p.H;
  const int D    = p.D;
  const int FF   = p.FF;
  const int B    = p.B;
  const int Mk_c = p.Mk_cross;
  const int read_end = p.self_attn_write_start + M;
  bool prof = cosmos_block_profile_enabled();
  enum {
    EV_START = 0,
    EV_AFTER_LORA,
    EV_AFTER_SA_LN,
    EV_AFTER_SA_QKV,
    EV_AFTER_SA_POST_QKV,
    EV_AFTER_SA_FMHA,
    EV_AFTER_SA_OUT,
    EV_AFTER_CA_LN_Q,
    EV_AFTER_CA_FMHA,
    EV_AFTER_CA_OUT,
    EV_AFTER_MLP_LN,
    EV_AFTER_FFN1,
    EV_AFTER_FFN2,
    EV_AFTER_DONE,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (prof) cudaEventRecord(ev[idx], stream);
  };
  if (prof) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }

  // ===========================================================================
  // 0) adaln-LoRA: produce three (shift, scale, gate) bundles per block.
  //
  // The bridge passes a SiLU-applied t_emb buffer (shape [B, K]) so each
  // sub-layer's helper just runs two GEMMs + an add. Fixed scheduler-step
  // replay may precompute the final [B, 3K] bundles and pass them directly.
  // ===========================================================================
  const bool has_any_precomputed_mods =
      p.precomputed_mods_sa || p.precomputed_mods_ca || p.precomputed_mods_mlp;
  const bool has_all_precomputed_mods =
      p.precomputed_mods_sa && p.precomputed_mods_ca && p.precomputed_mods_mlp;
  if (has_any_precomputed_mods && !has_all_precomputed_mods) {
    return cudaErrorInvalidValue;
  }
  // Skip the per-block AdaLN GEMMs when either mechanism says mods are
  // precomputed: main's external `precomputed_mods_*` pointers, or this
  // branch's `adaln_precomputed` flag (bridge has already populated
  // `p.buf.mods_*` via the pre-stacked + batched up-GEMM path).
  if (!p.adaln_precomputed && !has_all_precomputed_mods) {
    err = cosmos_adaln_lora_split<bf16>(
        p.t_emb, p.w.adaln_sa_down, p.w.adaln_sa_up, p.adaln_lora_3D,
        p.buf.lora_hidden_sa, p.buf.mods_sa, B, K, p.lora_dim, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_adaln_lora_split<bf16>(
        p.t_emb, p.w.adaln_ca_down, p.w.adaln_ca_up, p.adaln_lora_3D,
        p.buf.lora_hidden_ca, p.buf.mods_ca, B, K, p.lora_dim, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_adaln_lora_split<bf16>(
        p.t_emb, p.w.adaln_mlp_down, p.w.adaln_mlp_up, p.adaln_lora_3D,
        p.buf.lora_hidden_mlp, p.buf.mods_mlp, B, K, p.lora_dim, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }
  rec(EV_AFTER_LORA);
  COSMOS_DBG_SYNC("after_lora");

  // (shift, scale, gate) views into the [B, 3K] mods buffer.
  const bf16* mods_sa = has_all_precomputed_mods ? p.precomputed_mods_sa : p.buf.mods_sa;
  const bf16* mods_ca = has_all_precomputed_mods ? p.precomputed_mods_ca : p.buf.mods_ca;
  const bf16* mods_ml = has_all_precomputed_mods ? p.precomputed_mods_mlp : p.buf.mods_mlp;
  const bf16* shift_sa = mods_sa;
  const bf16* scale_sa = mods_sa + size_t(K);
  const bf16* gate_sa  = mods_sa + size_t(2 * K);
  const bf16* shift_ca = mods_ca;
  const bf16* scale_ca = mods_ca + size_t(K);
  const bf16* gate_ca  = mods_ca + size_t(2 * K);
  const bf16* shift_ml = mods_ml;
  const bf16* scale_ml = mods_ml + size_t(K);
  const bf16* gate_ml  = mods_ml + size_t(2 * K);

  // The mods buffer is laid out [B, 3K] so the three vectors for batch b
  // live at offsets (3K*b + 0K, 3K*b + K, 3K*b + 2K). For B==1 this is
  // exactly what cosmos_layernorm_modulate consumes (stride = K, b_row=0).
  // For B>1 we need to pass stride = 3K so the sub-vector picker advances
  // 3K bytes per batch row -- but cosmos_layernorm_modulate currently
  // assumes scale/shift are tightly packed [B, K]. So we restrict to B==1
  // (the only case the streaming path actually uses).
  //
  // FlashDreams' generate() runs uncond and cond as separate forwards
  // (not CFG-batched into B=2), so this is the correct steady-state
  // assumption. If a caller ever passes B>1 we fall back to a slow path
  // (TODO Phase 2).

  // ===========================================================================
  // 1) Self-attention residual
  //    a) ln+modulate  -> normed
  // ===========================================================================
  bf16* normed = p.buf.normed;
  const bool sage3_fp8_attention =
      p.attention_backend == CosmosAttentionBackend::SAGE3_FP8;
  const int sage3_q_padded = ((M + 127) / 128) * 128;
  const bool quantized_attention =
      p.attention_backend == CosmosAttentionBackend::FP8_CUDNN ||
      p.attention_backend == CosmosAttentionBackend::FP8_DENSE_REF ||
      sage3_fp8_attention;
  const bool fp8_cudnn_attention =
      p.attention_backend == CosmosAttentionBackend::FP8_CUDNN;
  const bool self_fp8_cudnn_bhmd_input =
      fp8_cudnn_attention &&
      select_cosmos_fp8_sdpa(B, M, read_end, H, D).layout == "bhmd";
  const bool sa_q_fp8 = cosmos_weight_is_fp8(p, p.w.sa_w_q_scale);
  const bool sa_k_fp8 = cosmos_weight_is_fp8(p, p.w.sa_w_k_scale);
  const bool sa_v_fp8 = cosmos_weight_is_fp8(p, p.w.sa_w_v_scale);
  const bool sa_qkv_fp8 = cosmos_weight_is_fp8(p, p.w.sa_w_qkv_scale);
  const bool sa_out_fp8 = cosmos_weight_is_fp8(p, p.w.sa_w_out_scale);
  const bool ca_q_fp8 = cosmos_weight_is_fp8(p, p.w.ca_w_q_scale);
  const bool ca_out_fp8 = cosmos_weight_is_fp8(p, p.w.ca_w_out_scale);
  const bool ffn1_fp8 = cosmos_weight_is_fp8(p, p.w.ffn_w1_scale);
  const bool ffn2_fp8 = cosmos_weight_is_fp8(p, p.w.ffn_w2_scale);
  const bool sa_normed_needs_fp8 = sa_qkv_fp8 || sa_q_fp8 || sa_k_fp8 || sa_v_fp8;
  const bool ca_normed_needs_fp8 = ca_q_fp8;
  const bool ffn_normed_needs_fp8 = ffn1_fp8;
  const bool use_fused_fp8_qkv =
      sa_qkv_fp8 && p.w.sa_w_qkv && p.w.sa_w_qkv_scale && p.buf.qkv_row;
  const bool sa_normed_fp8_only =
      use_fused_fp8_qkv || (sa_q_fp8 && sa_k_fp8 && sa_v_fp8);
  if (sa_normed_needs_fp8) {
    err = sa_normed_fp8_only
        ? cosmos_layernorm_modulate_to_fp8_only<bf16>(
            p.x, shift_sa, scale_sa, p.buf.linear_fp8_scratch, M * B, K, B, 1e-6f, stream)
        : cosmos_layernorm_modulate_to_fp8<bf16>(
            p.x, shift_sa, scale_sa, normed, p.buf.linear_fp8_scratch, M * B, K, B, 1e-6f, stream);
  } else {
    err = cosmos_layernorm_modulate<bf16>(
        p.x, shift_sa, scale_sa, normed, M * B, K, B, 1e-6f, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_SA_LN);
  COSMOS_DBG_SYNC("after_sa_ln");

  // 1b) Q/K/V projections (three separate [K, K] GEMMs).
  //
  //     The FP8 path can consume an offline pre-fused [3K, K] QKV tensor to
  //     turn three small GEMMs into one larger tensor-core GEMM. When callers
  //     do not provide it, the split Q/K/V tensors remain the fallback.
  bf16* q_row = p.buf.q_row;
  bf16* k_row = p.buf.k_row;
  bf16* v_row = p.buf.v_row;
  const bool use_fused_bf16_qkv =
      !use_fused_fp8_qkv && !sa_normed_needs_fp8 && p.w.sa_w_qkv_prepared &&
      p.buf.qkv_row && !quantized_attention && !p.fp8_kv_cache_enabled;
  if (use_fused_fp8_qkv) {
    err = cosmos_linear_prequantized_fp8(
        p, p.buf.linear_fp8_scratch, p.w.sa_w_qkv, p.w.sa_w_qkv_scale, p.buf.qkv_row,
        M * B, K, 3 * K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    if (!quantized_attention) {
      err = cosmos_split_qkv_row(p.buf.qkv_row, q_row, k_row, v_row, M * B, K, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
  } else if (use_fused_bf16_qkv) {
    err = cutlass_linear_layer_rrr_bf16_prepared(
        normed, p.w.sa_w_qkv_prepared, /*bias=*/nullptr, p.buf.qkv_row,
        M * B, K, 3 * K, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else if (sa_normed_needs_fp8) {
    if (!p.buf.linear_fp8_scratch) return cudaErrorInvalidValue;
    err = sa_q_fp8
        ? cosmos_linear_prequantized_fp8(p, p.buf.linear_fp8_scratch, p.w.sa_w_q, p.w.sa_w_q_scale, q_row,
                                         M * B, K, K, /*gelu=*/false, stream)
        : cosmos_linear_bf16_or_fp8(p, normed, p.w.sa_w_q, p.w.sa_w_q_scale, q_row,
                                    M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = sa_k_fp8
        ? cosmos_linear_prequantized_fp8(p, p.buf.linear_fp8_scratch, p.w.sa_w_k, p.w.sa_w_k_scale, k_row,
                                         M * B, K, K, /*gelu=*/false, stream)
        : cosmos_linear_bf16_or_fp8(p, normed, p.w.sa_w_k, p.w.sa_w_k_scale, k_row,
                                    M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = sa_v_fp8
        ? cosmos_linear_prequantized_fp8(p, p.buf.linear_fp8_scratch, p.w.sa_w_v, p.w.sa_w_v_scale, v_row,
                                         M * B, K, K, /*gelu=*/false, stream)
        : cosmos_linear_bf16_or_fp8(p, normed, p.w.sa_w_v, p.w.sa_w_v_scale, v_row,
                                    M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else {
    err = cosmos_linear_bf16_or_fp8_prepared(
        p, normed, p.w.sa_w_q, nullptr, p.w.sa_w_q_scale, q_row,
        M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_linear_bf16_or_fp8_prepared(
        p, normed, p.w.sa_w_k, nullptr, p.w.sa_w_k_scale, k_row,
        M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    err = cosmos_linear_bf16_or_fp8_prepared(
        p, normed, p.w.sa_w_v, nullptr, p.w.sa_w_v_scale, v_row,
        M * B, K, K, /*gelu=*/false, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }
  rec(EV_AFTER_SA_QKV);
  COSMOS_DBG_SYNC("after_sa_qkv");

  // 1c-1e) Per-head RMSNorm, RoPE, Q FP8 production, and KV cache write.
  bf16* q_bmhk = p.buf.q_bmhk;
  bf16* k_bmhk = p.buf.k_bmhk;
  bf16* v_bmhk = p.buf.v_bmhk;
  const cutlass::float_e4m3_t* q_attn_fp8 = nullptr;
  const cutlass::float_e4m3_t* q_attn_fp8_bhmd = nullptr;
  const uint8_t* q_attn_sage3_fp4 = nullptr;
  const cutlass::float_e4m3_t* q_attn_sage3_sf = nullptr;
  if (use_fused_bf16_qkv) {
    err = cosmos_qkv_postprocess_cache<bf16>(
        p.buf.qkv_row,
        p.w.sa_q_norm,
        p.w.sa_k_norm,
        p.rope_cos,
        p.rope_sin,
        q_bmhk,
        p.k_self_cache,
        p.v_self_cache,
        B, M, H, D, p.self_attn_write_start, p.self_attn_cache_cap, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else if (use_fused_fp8_qkv && quantized_attention) {
    if (sage3_fp8_attention) {
      if (!p.buf.attn_q_sage3_fp4 || !p.buf.attn_q_sage3_sf) {
        return cudaErrorInvalidValue;
      }
      err = sage3_quantize_q_bf16(
          p.buf.qkv_row, p.w.sa_q_norm, p.rope_cos, p.rope_sin,
          p.buf.attn_q_sage3_fp4, p.buf.attn_q_sage3_sf,
          B, M, H, D, 3 * K, /*input_head_offset=*/0,
          /*apply_rope=*/true, sage3_q_padded, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      q_attn_sage3_fp4 = p.buf.attn_q_sage3_fp4;
      q_attn_sage3_sf = p.buf.attn_q_sage3_sf;
    }
    cutlass::float_e4m3_t* q_bhmd_out =
        (!sage3_fp8_attention && self_fp8_cudnn_bhmd_input)
            ? p.buf.attn_q_bhmd_fp8
            : nullptr;
    err = cosmos_fused_qkv_rope_cache_fp8(
        p.buf.qkv_row,
        p.w.sa_q_norm,
        p.w.sa_k_norm,
        p.rope_cos,
        p.rope_sin,
        sage3_fp8_attention ? nullptr : p.buf.attn_q_fp8,
        q_bhmd_out,
        p.write_bf16_self_kv_cache ? p.k_self_cache : nullptr,
        p.write_bf16_self_kv_cache ? p.v_self_cache : nullptr,
        p.fp8_kv_cache_enabled ? p.k_self_cache_fp8 : nullptr,
        p.fp8_kv_cache_enabled ? p.v_self_cache_fp8 : nullptr,
        p.k_self_cache_fp8_bhmd,
        p.v_self_cache_fp8_bhmd,
        p.v_self_cache_fp8_bhdm,
        B, M, H, D, p.self_attn_write_start, p.self_attn_cache_cap, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    q_attn_fp8 = sage3_fp8_attention ? nullptr : p.buf.attn_q_fp8;
    q_attn_fp8_bhmd = q_bhmd_out;
  } else {
    // Per-head RMSNorm on Q and K. Operates in-place on the [M, K] row
    // buffers viewed as [B*M, H, D].
    if (!sage3_fp8_attention) {
      err = cosmos_rmsnorm_per_head<bf16>(q_row, p.w.sa_q_norm, B, M, H, D, 1e-6f, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    err = cosmos_rmsnorm_per_head<bf16>(k_row, p.w.sa_k_norm, B, M, H, D, 1e-6f, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

    if (quantized_attention) {
      if (sage3_fp8_attention) {
        if (!p.buf.attn_q_sage3_fp4 || !p.buf.attn_q_sage3_sf) {
          return cudaErrorInvalidValue;
        }
        err = sage3_quantize_q_bf16(
            q_row, p.w.sa_q_norm, p.rope_cos, p.rope_sin,
            p.buf.attn_q_sage3_fp4, p.buf.attn_q_sage3_sf,
            B, M, H, D, H * D, /*input_head_offset=*/0,
            /*apply_rope=*/true, sage3_q_padded, stream);
        if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
        q_attn_sage3_fp4 = p.buf.attn_q_sage3_fp4;
        q_attn_sage3_sf = p.buf.attn_q_sage3_sf;
      } else {
        cutlass::float_e4m3_t* q_bhmd_out =
            self_fp8_cudnn_bhmd_input ? p.buf.attn_q_bhmd_fp8 : nullptr;
        err = cosmos_pack_rope_to_fp8<bf16>(
            q_row, p.rope_cos, p.rope_sin, p.buf.attn_q_fp8, q_bhmd_out, B, M, H, D, stream);
        if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
        q_attn_fp8 = p.buf.attn_q_fp8;
        q_attn_fp8_bhmd = q_bhmd_out;
      }
    } else {
      err = cosmos_pack_rope<bf16>(q_row, p.rope_cos, p.rope_sin, q_bmhk, B * M, H, D, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    if (!p.fp8_kv_cache_enabled) {
      err = cosmos_pack_rope<bf16>(k_row, p.rope_cos, p.rope_sin, k_bmhk, B * M, H, D, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
    // V: identity copy (V_row -> V_bmhk). Since K = H*D, the layouts are
    // byte-equivalent and we can avoid the kernel; just alias the pointer.
    v_bmhk = v_row;

    // Append K_bmhk, V_bmhk into the self-attn KV cache at the configured
    // write cursor. After this, the cache holds the valid prefix
    // [0 : write_start + M).
    if (p.fp8_kv_cache_enabled) {
      err = cosmos_write_self_kv_cache_rope_bf16_and_fp8<bf16>(
          k_row, v_bmhk, p.rope_cos, p.rope_sin,
          p.write_bf16_self_kv_cache ? p.k_self_cache : nullptr,
          p.write_bf16_self_kv_cache ? p.v_self_cache : nullptr,
          p.k_self_cache_fp8, p.v_self_cache_fp8,
          p.k_self_cache_fp8_bhmd, p.v_self_cache_fp8_bhmd, p.v_self_cache_fp8_bhdm,
          B, M, H, D, p.self_attn_write_start, p.self_attn_cache_cap, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    } else {
      err = cosmos_write_kv_cache<bf16>(
          k_bmhk, v_bmhk, p.k_self_cache, p.v_self_cache,
          B, M, H, D, p.self_attn_write_start, p.self_attn_cache_cap, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
  }
  rec(EV_AFTER_SA_POST_QKV);
  COSMOS_DBG_SYNC("after_sa_post_qkv");

  // 1f) cuDNN FMHA against the historical cache + just-written tokens.
  //     Q is [B*M, H, D]; K, V are [B*read_end, H, D] (the valid prefix of
  //     the cache for this batch).
  bf16* o_bmhk = p.buf.o_bmhk;
  const bool sa_out_proj_from_quantized_attention =
      quantized_attention && !sage3_fp8_attention && sa_out_fp8;
  const bool sa_attn_writes_fp8 =
      sa_out_proj_from_quantized_attention &&
      p.attention_backend == CosmosAttentionBackend::FP8_CUDNN;
  const bool fuse_sa_fp8_residual_postprocess =
      sa_out_fp8 && quantized_attention;
  const bool self_tc_cache_contiguous =
      p.fp8_kv_cache_enabled && p.k_self_cache_fp8_bhmd &&
      p.v_self_cache_fp8_bhmd &&
      read_end == p.self_attn_cache_cap;
  err = cosmos_attention_bf16_or_fp8(
      p,
      quantized_attention ? nullptr : reinterpret_cast<const cutlass::bfloat16_t*>(q_bmhk),
      reinterpret_cast<const cutlass::bfloat16_t*>(p.k_self_cache),
      reinterpret_cast<const cutlass::bfloat16_t*>(p.v_self_cache),
      q_attn_fp8,
      q_attn_fp8_bhmd,
      q_attn_fp8_bhmd ? p.M : 0,
      p.fp8_kv_cache_enabled ? p.k_self_cache_fp8 : nullptr,
      p.fp8_kv_cache_enabled ? p.v_self_cache_fp8 : nullptr,
      self_tc_cache_contiguous ? p.k_self_cache_fp8_bhmd : nullptr,
      self_tc_cache_contiguous ? p.v_self_cache_fp8_bhmd : nullptr,
      self_tc_cache_contiguous ? read_end : 0,
      self_tc_cache_contiguous ? read_end : 0,
      self_tc_cache_contiguous ? p.v_self_cache_fp8_bhdm : nullptr,
      q_attn_sage3_fp4,
      q_attn_sage3_sf,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      sage3_q_padded,
      0,
      reinterpret_cast<cutlass::bfloat16_t*>(o_bmhk),
      sa_attn_writes_fp8 ? p.buf.linear_fp8_scratch : nullptr,
      read_end, !sa_out_proj_from_quantized_attention, sa_attn_writes_fp8,
      stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_SA_FMHA);

  // 1g) Unpack [M, H, D] -> [M, K]. Same byte layout when K = H*D, so
  //     we can alias the pointer; we use the existing q_row buffer as the
  //     output slot for the SA out projection.
  bf16* attn_out_row = p.buf.attn_out_row;
  // Identity unpack: src pointer can be reused directly because BMHK packed
  // [M, H, D] and row-major [M, K] have identical byte layout for K = H*D.
  attn_out_row = o_bmhk;

  // 1h) SA output GEMM and gated residual.
  //     Cosmos has no bias on output_proj (Linear with bias=False).
  // Fast path: when the next op is the CA LN/modulate-to-FP8, fuse the
  // (col_scale + residual) post-op with the LN/modulate-to-FP8 in one kernel
  // to skip the BF16 re-read of x.
  const bool fuse_sa_out_with_ca_ln =
      fuse_sa_fp8_residual_postprocess && sa_attn_writes_fp8 && ca_normed_needs_fp8;
  bool ca_ln_already_done = false;
  if (fuse_sa_fp8_residual_postprocess) {
    if (fuse_sa_out_with_ca_ln) {
      err = cosmos_linear_prequantized_fp8_residual_layernorm_modulate_to_fp8_only(
          p, p.buf.linear_fp8_scratch, p.w.sa_w_out, p.w.sa_w_out_scale,
          p.x, gate_sa,
          shift_ca, scale_ca, p.buf.linear_fp8_scratch,
          M * B, K, K, B, 1e-6f, stream);
      ca_ln_already_done = (err == cudaSuccess);
    } else if (sa_attn_writes_fp8) {
      err = cosmos_linear_prequantized_fp8_residual(
          p, p.buf.linear_fp8_scratch, p.w.sa_w_out, p.w.sa_w_out_scale,
          p.x, gate_sa, M * B, K, K, stream);
    } else if (sa_out_proj_from_quantized_attention) {
      err = cosmos_linear_half_input_fp8_residual(
          p, p.buf.attn_o_half, p.w.sa_w_out, p.w.sa_w_out_scale,
          p.x, gate_sa, M * B, K, K, stream);
    } else {
      err = cosmos_linear_bf16_input_fp8_residual(
          p, attn_out_row, p.w.sa_w_out, p.w.sa_w_out_scale,
          p.x, gate_sa, M * B, K, K, stream);
    }
  } else {
    err = cosmos_linear_bf16_or_fp8_residual_prepared(
        p, attn_out_row, p.w.sa_w_out, p.w.sa_w_out_prepared, p.w.sa_w_out_scale,
        p.x, gate_sa, M * B, K, K, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  err = cosmos_trace_copy(p.trace_sa_out, p.x, p.trace_elems, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_SA_OUT);

  // ===========================================================================
  // 2) Cross-attention residual (Q only -- K/V are pre-cached)
  // ===========================================================================
  if (ca_ln_already_done) {
    // CA LN/modulate -> FP8 was fused into the SA-out residual kernel above.
    err = cudaSuccess;
  } else if (ca_normed_needs_fp8) {
    err = cosmos_layernorm_modulate_to_fp8_only<bf16>(
        p.x, shift_ca, scale_ca, p.buf.linear_fp8_scratch, M * B, K, B, 1e-6f, stream);
  } else {
    err = cosmos_layernorm_modulate<bf16>(
        p.x, shift_ca, scale_ca, normed, M * B, K, B, 1e-6f, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  // 2b) CA Q projection (no K/V projection; encoder K/V are pre-cached).
  if (ca_q_fp8) {
    err = cosmos_linear_prequantized_fp8(p, p.buf.linear_fp8_scratch, p.w.ca_w_q, p.w.ca_w_q_scale, q_row,
                                         M * B, K, K, /*gelu=*/false, stream);
  } else {
    err = cosmos_linear_bf16_or_fp8_prepared(
        p, normed, p.w.ca_w_q, p.w.ca_w_q_prepared, p.w.ca_w_q_scale, q_row,
        M * B, K, K, /*gelu=*/false, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }

  // 2c) Per-head Q RMSNorm. No RoPE on cross-attn. For quantized attention,
  // emit the post-RMSNorm Q directly as FP8 here so the attention launcher
  // does not need a separate BF16-to-FP8 conversion for cross-Q.
  const cutlass::float_e4m3_t* ca_q_attn_fp8 = nullptr;
  const cutlass::float_e4m3_t* ca_q_attn_fp8_bhmd = nullptr;
  const uint8_t* ca_q_attn_sage3_fp4 = nullptr;
  const cutlass::float_e4m3_t* ca_q_attn_sage3_sf = nullptr;
  const bool cross_quantized_attention = quantized_attention;
  const bool cross_fp8_cudnn_bhmd_input =
      fp8_cudnn_attention &&
      select_cosmos_fp8_sdpa(B, M, Mk_c, H, D).layout == "bhmd";
  if (cross_quantized_attention) {
    if (sage3_fp8_attention) {
      if (!p.buf.attn_q_sage3_fp4 || !p.buf.attn_q_sage3_sf) {
        return cudaErrorInvalidValue;
      }
      err = sage3_quantize_q_bf16(
          q_row, p.w.ca_q_norm, nullptr, nullptr,
          p.buf.attn_q_sage3_fp4, p.buf.attn_q_sage3_sf,
          B, M, H, D, H * D, /*input_head_offset=*/0,
          /*apply_rope=*/false, sage3_q_padded, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      ca_q_attn_sage3_fp4 = p.buf.attn_q_sage3_fp4;
      ca_q_attn_sage3_sf = p.buf.attn_q_sage3_sf;
    } else {
      cutlass::float_e4m3_t* ca_q_bhmd_out =
          cross_fp8_cudnn_bhmd_input ? p.buf.attn_q_bhmd_fp8 : nullptr;
      err = cosmos_rmsnorm_per_head_to_fp8<bf16>(
          q_row, p.w.ca_q_norm, p.buf.attn_q_fp8, ca_q_bhmd_out, B, M, H, D, 1e-6f, stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
      ca_q_attn_fp8 = p.buf.attn_q_fp8;
      ca_q_attn_fp8_bhmd = ca_q_bhmd_out;
    }
  } else {
    err = cosmos_rmsnorm_per_head<bf16>(q_row, p.w.ca_q_norm, B, M, H, D, 1e-6f, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }
  rec(EV_AFTER_CA_LN_Q);

  // 2d) FMHA against pre-cached encoder K/V.
  //     Q  [B*M, H, D]
  //     K  [B*Mk_c, H, D]   (FlashDreams writes [B*V, Mk_c, H, D] -- with V==1 this is [B, Mk_c, H, D])
  //     V  [B*Mk_c, H, D]
  const bool ca_out_proj_from_quantized_attention =
      cross_quantized_attention && !sage3_fp8_attention && ca_out_fp8;
  const bool ca_attn_writes_fp8 =
      ca_out_proj_from_quantized_attention &&
      (p.attention_backend == CosmosAttentionBackend::FP8_CUDNN);
  const bool fuse_ca_fp8_residual_postprocess =
      ca_out_fp8 && cross_quantized_attention;
  err = cosmos_attention_bf16_or_fp8(
      p,
      cross_quantized_attention ? nullptr : reinterpret_cast<const cutlass::bfloat16_t*>(q_row),
      reinterpret_cast<const cutlass::bfloat16_t*>(p.k_cross),
      reinterpret_cast<const cutlass::bfloat16_t*>(p.v_cross),
      ca_q_attn_fp8,
      ca_q_attn_fp8_bhmd,
      ca_q_attn_fp8_bhmd ? p.M : 0,
      p.fp8_kv_cache_enabled ? p.k_cross_fp8 : nullptr,
      p.fp8_kv_cache_enabled ? p.v_cross_fp8 : nullptr,
      p.fp8_kv_cache_enabled ? p.k_cross_fp8_bhmd : nullptr,
      p.fp8_kv_cache_enabled ? p.v_cross_fp8_bhmd : nullptr,
      p.fp8_kv_cache_enabled ? p.k_cross_fp8_bhmd_tokens : 0,
      p.fp8_kv_cache_enabled ? p.v_cross_fp8_bhmd_tokens : 0,
      p.fp8_kv_cache_enabled ? p.v_cross_fp8_bhdm : nullptr,
      ca_q_attn_sage3_fp4,
      ca_q_attn_sage3_sf,
      p.k_cross_sage3_fp4,
      p.v_cross_sage3_fp4,
      p.k_cross_sage3_sf,
      p.v_cross_sage3_sf,
      sage3_q_padded,
      p.Mk_cross_sage3_padded,
      reinterpret_cast<cutlass::bfloat16_t*>(o_bmhk),
      ca_attn_writes_fp8 ? p.buf.linear_fp8_scratch : nullptr,
      Mk_c, !ca_out_proj_from_quantized_attention, ca_attn_writes_fp8,
      stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_CA_FMHA);

  // 2e) CA out projection (and possibly fused MLP LN/modulate -> FP8).
  // Same fusion pattern as SA-out: when CA-out residual is the FP8 fast path
  // and the next op is MLP LN/modulate-to-FP8, fold both into one launch.
  const bool fuse_ca_out_with_mlp_ln =
      fuse_ca_fp8_residual_postprocess && ca_attn_writes_fp8 && ffn_normed_needs_fp8;
  bool mlp_ln_already_done = false;
  if (fuse_ca_fp8_residual_postprocess) {
    if (fuse_ca_out_with_mlp_ln) {
      err = cosmos_linear_prequantized_fp8_residual_layernorm_modulate_to_fp8_only(
          p, p.buf.linear_fp8_scratch, p.w.ca_w_out, p.w.ca_w_out_scale,
          p.x, gate_ca,
          shift_ml, scale_ml, p.buf.linear_fp8_scratch,
          M * B, K, K, B, 1e-6f, stream);
      mlp_ln_already_done = (err == cudaSuccess);
    } else if (ca_attn_writes_fp8) {
      err = cosmos_linear_prequantized_fp8_residual(
          p, p.buf.linear_fp8_scratch, p.w.ca_w_out, p.w.ca_w_out_scale,
          p.x, gate_ca, M * B, K, K, stream);
    } else if (ca_out_proj_from_quantized_attention) {
      err = cosmos_linear_half_input_fp8_residual(
          p, p.buf.attn_o_half, p.w.ca_w_out, p.w.ca_w_out_scale,
          p.x, gate_ca, M * B, K, K, stream);
    } else {
      err = cosmos_linear_bf16_input_fp8_residual(
          p, o_bmhk, p.w.ca_w_out, p.w.ca_w_out_scale,
          p.x, gate_ca, M * B, K, K, stream);
    }
  } else {
    err = cosmos_linear_bf16_or_fp8_residual_prepared(
        p, o_bmhk, p.w.ca_w_out, p.w.ca_w_out_prepared, p.w.ca_w_out_scale,
        p.x, gate_ca, M * B, K, K, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  err = cosmos_trace_copy(p.trace_ca_out, p.x, p.trace_elems, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_CA_OUT);

  // ===========================================================================
  // 3) FFN residual
  // ===========================================================================
  if (mlp_ln_already_done) {
    // MLP LN/modulate -> FP8 was fused into the CA-out residual kernel above.
    err = cudaSuccess;
  } else if (ffn_normed_needs_fp8) {
    err = cosmos_layernorm_modulate_to_fp8_only<bf16>(
        p.x, shift_ml, scale_ml, p.buf.linear_fp8_scratch, M * B, K, B, 1e-6f, stream);
  } else {
    err = cosmos_layernorm_modulate<bf16>(
        p.x, shift_ml, scale_ml, normed, M * B, K, B, 1e-6f, stream);
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_MLP_LN);

  // 3b) FFN GEMM1 + GELU: [M, K] x [K, FF] -> [M, FF]
  // 3c) FFN GEMM2: [M, FF] x [FF, K] -> [M, K]
  const float ffn1_gelu_fp8_scale =
      cosmos_fp8_activation_scale_or_one(p, kCosmosFp8ActivationScaleFfn1Gelu);
  if (ffn1_fp8 && ffn2_fp8) {
    err = cosmos_linear_prequantized_fp8_output(
        p, p.buf.linear_fp8_scratch, p.w.ffn_w1, p.w.ffn_w1_scale, p.buf.linear_fp8_scratch,
        M * B, K, FF, /*gelu=*/true, stream, ffn1_gelu_fp8_scale);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else {
    err = ffn1_fp8
        ? cosmos_linear_prequantized_fp8(
            p, p.buf.linear_fp8_scratch, p.w.ffn_w1, p.w.ffn_w1_scale, p.buf.ffn_intermediate,
            M * B, K, FF, /*gelu=*/true, stream)
        : cosmos_linear_bf16_or_fp8_prepared(
            p, normed, p.w.ffn_w1, p.w.ffn_w1_prepared, p.w.ffn_w1_scale, p.buf.ffn_intermediate,
            M * B, K, FF, /*gelu=*/true, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    if (p.fp8_activation_amax_out) {
      err = cosmos_record_bf16_absmax(
          p.buf.ffn_intermediate,
          p.fp8_activation_amax_out + kCosmosFp8ActivationScaleFfn1Gelu,
          int64_t(M) * B * FF,
          stream);
      if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
    }
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_FFN1);

  if (ffn1_fp8 && ffn2_fp8) {
    err = cosmos_linear_prequantized_fp8_residual(
        p, p.buf.linear_fp8_scratch, p.w.ffn_w2, p.w.ffn_w2_scale,
        p.x, gate_ml, M * B, FF, K, stream, ffn1_gelu_fp8_scale);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else if (ffn2_fp8) {
    err = cosmos_linear_bf16_input_fp8_residual(
        p, p.buf.ffn_intermediate, p.w.ffn_w2, p.w.ffn_w2_scale,
        p.x, gate_ml, M * B, FF, K, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  } else {
    err = cosmos_linear_bf16_or_fp8_residual_prepared(
        p, p.buf.ffn_intermediate, p.w.ffn_w2, p.w.ffn_w2_prepared, p.w.ffn_w2_scale,
        p.x, gate_ml, M * B, FF, K, stream);
    if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  }
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_FFN2);
  err = cosmos_trace_copy(p.trace_ffn_out, p.x, p.trace_elems, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  err = cosmos_trace_copy(p.trace_block_out, p.x, p.trace_elems, stream);
  if (err != cudaSuccess) { COSMOS_DBG_FAIL(); return err; }
  rec(EV_AFTER_DONE);

  if (prof) {
    cudaEventSynchronize(ev[EV_AFTER_DONE]);
    auto ms = [&](int a, int b) -> float {
      float out = 0.f;
      cudaEventElapsedTime(&out, ev[a], ev[b]);
      return out;
    };
    std::printf(
        "[cosmos_block] lora=%.3f sa_ln=%.3f sa_qkv=%.3f sa_post=%.3f sa_fmha=%.3f sa_out=%.3f "
        "ca_ln_q=%.3f ca_fmha=%.3f ca_out=%.3f mlp_ln=%.3f ffn1=%.3f ffn2=%.3f mlp_gate=%.3f total=%.3f ms\n",
        ms(EV_START, EV_AFTER_LORA),
        ms(EV_AFTER_LORA, EV_AFTER_SA_LN),
        ms(EV_AFTER_SA_LN, EV_AFTER_SA_QKV),
        ms(EV_AFTER_SA_QKV, EV_AFTER_SA_POST_QKV),
        ms(EV_AFTER_SA_POST_QKV, EV_AFTER_SA_FMHA),
        ms(EV_AFTER_SA_FMHA, EV_AFTER_SA_OUT),
        ms(EV_AFTER_SA_OUT, EV_AFTER_CA_LN_Q),
        ms(EV_AFTER_CA_LN_Q, EV_AFTER_CA_FMHA),
        ms(EV_AFTER_CA_FMHA, EV_AFTER_CA_OUT),
        ms(EV_AFTER_CA_OUT, EV_AFTER_MLP_LN),
        ms(EV_AFTER_MLP_LN, EV_AFTER_FFN1),
        ms(EV_AFTER_FFN1, EV_AFTER_FFN2),
        ms(EV_AFTER_FFN2, EV_AFTER_DONE),
        ms(EV_START, EV_AFTER_DONE));
    for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
  }

  return cudaSuccess;
}

template cudaError_t cosmos_split_qkv<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, cutlass::bfloat16_t*, cutlass::bfloat16_t*,
    cutlass::bfloat16_t*, int, int, cudaStream_t);
template cudaError_t cosmos_split_qkv<cutlass::half_t>(
    const cutlass::half_t*, cutlass::half_t*, cutlass::half_t*,
    cutlass::half_t*, int, int, cudaStream_t);

} // namespace omnidreams_singleview
