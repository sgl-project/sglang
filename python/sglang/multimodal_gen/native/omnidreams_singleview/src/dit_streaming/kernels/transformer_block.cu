// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "workspace.cuh"
#include "attention.cuh"
#include "ops.cuh"
#include "linear_utils.cuh"
#include "dtype_utils.cuh"
#include "helper.h"
#include "common/profile_config.h"
#include "common/profiler.h"
#include "linear_utils.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/numeric_types.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cstdio>
#include <vector>
#include <type_traits>
#include <cstdint>

// Debug helper: check for NaN/Inf in a half buffer on device
// Uncomment `#define FP8_NAN_DEBUG 1` to enable
// #define FP8_NAN_DEBUG 1
#ifdef FP8_NAN_DEBUG
static __global__ void nan_check_kernel(const half* data, int64_t n, int* found_nan, int* found_inf) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = __half2float(data[idx]);
  if (isnan(v)) atomicAdd(found_nan, 1);
  if (isinf(v)) atomicAdd(found_inf, 1);
}
static bool check_nan_inf(const cutlass::half_t* buf, int64_t count, const char* label, cudaStream_t stream) {
  int h_nan = 0, h_inf = 0;
  int *d_nan, *d_inf;
  cudaMallocAsync(&d_nan, sizeof(int), stream);
  cudaMallocAsync(&d_inf, sizeof(int), stream);
  cudaMemsetAsync(d_nan, 0, sizeof(int), stream);
  cudaMemsetAsync(d_inf, 0, sizeof(int), stream);
  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);
  nan_check_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(buf), count, d_nan, d_inf);
  cudaMemcpyAsync(&h_nan, d_nan, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_inf, d_inf, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFreeAsync(d_nan, stream);
  cudaFreeAsync(d_inf, stream);
  if (h_nan > 0 || h_inf > 0) {
    printf("[NAN_DEBUG] %s: %d NaN, %d Inf out of %lld elements\n", label, h_nan, h_inf, (long long)count);
    return true;
  }
  return false;
}
#define NAN_CHECK(buf, count, label) check_nan_inf(buf, count, label, stream)
#else
#define NAN_CHECK(buf, count, label) false
#endif

namespace omnidreams_singleview {

// =============================================================================
// Templated Kernels for dual fp16/bf16 support
// =============================================================================

// Templated LayerNorm + modulation kernel
// temb_row_stride: stride between rows in temb. 6*K when expanded [M,6,K], 0 when broadcast [1,6,K].
template <typename ElementT>
__global__ void layernorm_modulate_row_kernel(
    const ElementT* __restrict__ X_row, int M, int K, float eps,
    const ElementT* __restrict__ scale_shift_table,
    const ElementT* __restrict__ temb, int temb_row_stride,
    int scale_idx, int shift_idx,
    ElementT* __restrict__ Y_row) {
  int row = blockIdx.x; if (row >= M) return;
  extern __shared__ float s[];
  float* ssum = s; float* ssum2 = s + blockDim.x;
  int tid = threadIdx.x;

  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(X_row[size_t(row) * K + j]);
    acc1 += v; acc2 += v * v;
  }
  ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) { ssum[tid] += ssum[tid + off]; ssum2[tid] += ssum2[tid + off]; }
    __syncthreads();
  }
  float mean = ssum[0] / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);

  for (int j = tid; j < K; j += blockDim.x) {
    float v = (omnidreams_singleview::to_float(X_row[size_t(row) * K + j]) - mean) * inv;
    float base_scale = omnidreams_singleview::to_float(scale_shift_table[scale_idx * K + j]);
    float base_shift = omnidreams_singleview::to_float(scale_shift_table[shift_idx * K + j]);
    size_t temb_offset = size_t(row) * temb_row_stride;
    float add_scale = omnidreams_singleview::to_float(temb[temb_offset + scale_idx * K + j]);
    float add_shift = omnidreams_singleview::to_float(temb[temb_offset + shift_idx * K + j]);
    v = v * (1.f + base_scale + add_scale) + (base_shift + add_shift);
    Y_row[size_t(row) * K + j] = omnidreams_singleview::from_float<ElementT>(v);
  }
}

// Templated affine LayerNorm kernel
template <typename ElementT>
__global__ void layernorm_affine_row_kernel(
    const ElementT* __restrict__ X_row, int M, int K, float eps,
    const ElementT* __restrict__ gamma,
    const ElementT* __restrict__ beta,
    ElementT* __restrict__ Y_row) {
  int row = blockIdx.x; if (row >= M) return;
  extern __shared__ float smem[];
  float* ssum = smem; float* ssum2 = smem + blockDim.x;
  int tid = threadIdx.x;

  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = omnidreams_singleview::to_float(X_row[size_t(row) * K + j]);
    acc1 += v; acc2 += v * v;
  }
  ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) { ssum[tid]+=ssum[tid+off]; ssum2[tid]+=ssum2[tid+off]; }
    __syncthreads();
  }
  float mean = ssum[0]/float(K);
  float var  = ssum2[0]/float(K) - mean*mean;
  float inv = rsqrtf(var + eps);

  for (int j = tid; j < K; j += blockDim.x) {
    float v = (omnidreams_singleview::to_float(X_row[size_t(row) * K + j]) - mean) * inv;
    float g = gamma ? omnidreams_singleview::to_float(gamma[j]) : 1.f;
    float b = beta  ? omnidreams_singleview::to_float(beta[j])  : 0.f;
    v = v * g + b;
    Y_row[size_t(row) * K + j] = omnidreams_singleview::from_float<ElementT>(v);
  }
}

// Templated residual + gate kernel (in-place)
// temb_row_stride: stride between rows in temb. 6*K when expanded, 0 when broadcast.
template <typename ElementT>
__global__ void residual_gate_add_kernel_rr(
    ElementT* __restrict__ D_row, int ld_row, int M, int K,
    const ElementT* __restrict__ S_row, int ld_src,
    const ElementT* __restrict__ scale_shift_table,
    const ElementT* __restrict__ temb, int temb_row_stride,
    int gate_idx) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < K) {
    size_t pos_dest = size_t(i) * ld_row + j;
    size_t pos_src  = size_t(i) * ld_src + j;
    float d_val = omnidreams_singleview::to_float(D_row[pos_dest]);
    float s_val = omnidreams_singleview::to_float(S_row[pos_src]);
    float base_gate = omnidreams_singleview::to_float(scale_shift_table[gate_idx * K + j]);
    size_t temb_offset = size_t(i) * temb_row_stride;
    float add_gate = omnidreams_singleview::to_float(temb[temb_offset + gate_idx * K + j]);
    float g_val = base_gate + add_gate;
    D_row[pos_dest] = omnidreams_singleview::from_float<ElementT>(d_val + s_val * g_val);
  }
}

// Templated residual + gate kernel (separate src/dest)
// temb_row_stride: stride between rows in temb. 6*K when expanded, 0 when broadcast.
template <typename ElementT>
__global__ void residual_gate_add_kernel_sr(
    ElementT* __restrict__ D_dest, int ld_dest, int M, int K,
    const ElementT* __restrict__ S_row, int ld_src,
    const ElementT* __restrict__ scale_shift_table,
    const ElementT* __restrict__ temb, int temb_row_stride,
    int gate_idx,
    const ElementT* __restrict__ D_src, int ld_srcD) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < K) {
    size_t pos_dest = size_t(i) * ld_dest + j;
    size_t pos_srcD = size_t(i) * ld_srcD + j;
    size_t pos_srcS = size_t(i) * ld_src + j;
    float d_val = omnidreams_singleview::to_float(D_src[pos_srcD]);
    float s_val = omnidreams_singleview::to_float(S_row[pos_srcS]);
    float base_gate = omnidreams_singleview::to_float(scale_shift_table[gate_idx * K + j]);
    size_t temb_offset = size_t(i) * temb_row_stride;
    float add_gate = omnidreams_singleview::to_float(temb[temb_offset + gate_idx * K + j]);
    float g_val = base_gate + add_gate;
    D_dest[pos_dest] = omnidreams_singleview::from_float<ElementT>(d_val + s_val * g_val);
  }
}

// Note: Templated kernels are implicitly instantiated when called via dispatch functions.
// CUDA __global__ functions don't support explicit template instantiation syntax.

// =============================================================================
// Legacy fp16-only kernels (kept for backward compatibility)
// =============================================================================

// Row-major Affine LayerNorm + on-the-fly modulation, row-major in/out
// This matches TurboDiffusion: norm(x) * (1 + scale) + shift
// where norm(x) = (x - mean) * rsqrt(var) * gamma + beta
// temb_row_stride: stride between rows in temb. 6*K when expanded [M,6,K], 0 when broadcast [1,6,K].
__global__ void layernorm_affine_modulate_row_rm_to_row_half_kernel(
    const cutlass::half_t* __restrict__ X_row, int M, int K, float eps,
    const cutlass::half_t* __restrict__ gamma,  // [K] affine weight (can be nullptr for identity)
    const cutlass::half_t* __restrict__ beta,   // [K] affine bias (can be nullptr for zero)
    const cutlass::half_t* __restrict__ scale_shift_table,
    const cutlass::half_t* __restrict__ temb, int temb_row_stride,
    int scale_idx, int shift_idx,
    cutlass::half_t* __restrict__ Y_row) {
  int row = blockIdx.x; if (row >= M) return;
  extern __shared__ float s[];
  float* ssum = s; float* ssum2 = s + blockDim.x;
  int tid = threadIdx.x;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = to_f32(X_row[size_t(row) * K + j]);
    acc1 += v; acc2 += v * v;
  }
  ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) { ssum[tid] += ssum[tid + off]; ssum2[tid] += ssum2[tid + off]; }
    __syncthreads();
  }
  float mean = ssum[0] / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = (to_f32(X_row[size_t(row) * K + j]) - mean) * inv;
    float g = gamma ? to_f32(gamma[j]) : 1.f;
    float b = beta  ? to_f32(beta[j])  : 0.f;
    v = v * g + b;
    float base_scale = to_f32(scale_shift_table[scale_idx * K + j]);
    float base_shift = to_f32(scale_shift_table[shift_idx * K + j]);
    size_t temb_offset = size_t(row) * temb_row_stride;
    float add_scale = to_f32(temb[temb_offset + scale_idx * K + j]);
    float add_shift = to_f32(temb[temb_offset + shift_idx * K + j]);
    v = v * (1.f + base_scale + add_scale) + (base_shift + add_shift);
    Y_row[size_t(row) * K + j] = to_f16(v);
  }
}

// Legacy: Row-major LayerNorm without affine + on-the-fly modulation (for backwards compatibility)
// temb_row_stride: stride between rows in temb. 6*K when expanded [M,6,K], 0 when broadcast [1,6,K].
__global__ void layernorm_modulate_row_rm_to_row_half_kernel(const cutlass::half_t* __restrict__ X_row, int M, int K,
                                                            float eps,
                                                            const cutlass::half_t* __restrict__ scale_shift_table,
                                                            const cutlass::half_t* __restrict__ temb,
                                                            int temb_row_stride,
                                                            int scale_idx, int shift_idx,
                                                            cutlass::half_t* __restrict__ Y_row) {
  int row = blockIdx.x; if (row >= M) return;
  extern __shared__ float s[];
  float* ssum = s; float* ssum2 = s + blockDim.x;
  int tid = threadIdx.x;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = to_f32(X_row[size_t(row) * K + j]);
    acc1 += v; acc2 += v * v;
  }
  ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) { ssum[tid] += ssum[tid + off]; ssum2[tid] += ssum2[tid + off]; }
    __syncthreads();
  }
  float mean = ssum[0] / float(K);
  float var  = ssum2[0] / float(K) - mean * mean;
  float inv  = rsqrtf(var + eps);
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = (to_f32(X_row[size_t(row) * K + j]) - mean) * inv;
    float base_scale = to_f32(scale_shift_table[scale_idx * K + j]);
    float base_shift = to_f32(scale_shift_table[shift_idx * K + j]);
    size_t temb_offset = size_t(row) * temb_row_stride;
    float add_scale = to_f32(temb[temb_offset + scale_idx * K + j]);
    float add_shift = to_f32(temb[temb_offset + shift_idx * K + j]);
    v = v * (1.f + base_scale + add_scale) + (base_shift + add_shift);
    Y_row[size_t(row) * K + j] = to_f16(v);
  }
}

// Residual add with gate: row-major destination and source with on-the-fly gate calculation
// temb_row_stride: stride between rows in temb. 6*K when expanded, 0 when broadcast.
__global__ void residual_gate_add_kernel_half_rr(cutlass::half_t* __restrict__ D_row, int ld_row, int M, int K,
                                                const cutlass::half_t* __restrict__ S_row, int ld_src,
                                                const cutlass::half_t* __restrict__ scale_shift_table,
                                                const cutlass::half_t* __restrict__ temb, int temb_row_stride,
                                                int gate_idx) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < K) {
    size_t pos_dest = size_t(i) * ld_row + j;
    size_t pos_src  = size_t(i) * ld_src + j;
    cutlass::NumericConverter<float, cutlass::half_t> to_f32;
    cutlass::NumericConverter<cutlass::half_t, float> to_f16;
    float d_val = to_f32(D_row[pos_dest]);
    float s_val = to_f32(S_row[pos_src]);
    float base_gate = to_f32(scale_shift_table[gate_idx * K + j]);
    size_t temb_offset = size_t(i) * temb_row_stride;
    float add_gate = to_f32(temb[temb_offset + gate_idx * K + j]);
    float g_val = base_gate + add_gate;
    D_row[pos_dest] = to_f16(d_val + s_val * g_val);
  }
}

// Simple row-major residual add: D += S
__global__ void add_inplace_half_rr(cutlass::half_t* __restrict__ D_row, int ld_row, int M, int K,
                                    const cutlass::half_t* __restrict__ S_row, int ld_src) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < K) {
    size_t pos_dest = size_t(i) * ld_row + j;
    size_t pos_src  = size_t(i) * ld_src + j;
    cutlass::NumericConverter<float, cutlass::half_t> to_f32;
    cutlass::NumericConverter<cutlass::half_t, float> to_f16;
    float d = to_f32(D_row[pos_dest]);
    float s = to_f32(S_row[pos_src]);
    D_row[pos_dest] = to_f16(d + s);
  }
}

// Residual add with gate: dest = src + S * gate (separate src/dest)
// temb_row_stride: stride between rows in temb. 6*K when expanded, 0 when broadcast.
__global__ void residual_gate_add_kernel_half_sr(cutlass::half_t* __restrict__ D_dest, int ld_dest, int M, int K,
                                                const cutlass::half_t* __restrict__ S_row, int ld_src,
                                                const cutlass::half_t* __restrict__ scale_shift_table,
                                                const cutlass::half_t* __restrict__ temb, int temb_row_stride,
                                                int gate_idx,
                                                const cutlass::half_t* __restrict__ D_src, int ld_srcD) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < K) {
    size_t pos_dest = size_t(i) * ld_dest + j;
    size_t pos_srcD = size_t(i) * ld_srcD + j;
    size_t pos_srcS = size_t(i) * ld_src + j;
    cutlass::NumericConverter<float, cutlass::half_t> to_f32;
    cutlass::NumericConverter<cutlass::half_t, float> to_f16;
    float d_val = to_f32(D_src[pos_srcD]);
    float s_val = to_f32(S_row[pos_srcS]);
    float base_gate = to_f32(scale_shift_table[gate_idx * K + j]);
    size_t temb_offset = size_t(i) * temb_row_stride;
    float add_gate = to_f32(temb[temb_offset + gate_idx * K + j]);
    float g_val = base_gate + add_gate;
    D_dest[pos_dest] = to_f16(d_val + s_val * g_val);
  }
}

// Row-major affine LayerNorm: X_row -> Y_row applying gamma/beta
__global__ void layernorm_affine_row_to_row_half_kernel(const cutlass::half_t* __restrict__ X_row, int M, int K,
                                                       float eps,
                                                       const cutlass::half_t* __restrict__ gamma,
                                                       const cutlass::half_t* __restrict__ beta,
                                                       cutlass::half_t* __restrict__ Y_row) {
  int row = blockIdx.x; if (row >= M) return;
  extern __shared__ float smem[];
  float* ssum = smem; float* ssum2 = smem + blockDim.x;
  int tid = threadIdx.x;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float acc1 = 0.f, acc2 = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = to_f32(X_row[size_t(row) * K + j]);
    acc1 += v; acc2 += v * v;
  }
  ssum[tid] = acc1; ssum2[tid] = acc2; __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) { ssum[tid]+=ssum[tid+off]; ssum2[tid]+=ssum2[tid+off]; } __syncthreads(); }
  float mean = ssum[0]/float(K);
  float var  = ssum2[0]/float(K) - mean*mean;
  float inv = rsqrtf(var + eps);
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = (to_f32(X_row[size_t(row) * K + j]) - mean) * inv;
    float g = gamma ? to_f32(gamma[j]) : 1.f;
    float b = beta  ? to_f32(beta[j])  : 0.f;
    v = v * g + b;
    Y_row[size_t(row) * K + j] = to_f16(v);
  }
}

template <typename WeightT, typename Arch>
cudaError_t run_transformer_block(const TransformerBlockParamsT<WeightT>& p, cudaStream_t stream) {
  int B = p.B;
  int Mq = p.Mq;
  int K = p.K, H = p.H, D = p.D;
  int M = B * Mq;
  int Mk_text = p.Mk ? p.Mk : Mq;
  int Mk_img = p.Mk_img;
  auto backend = get_attention_backend();

  // Optional fine-grained profiling (level >= 2): per-block breakdown
  int prof_lvl = g_wan_profile_level.load(std::memory_order_relaxed);
  int print_every = g_wan_profile_print_every.load(std::memory_order_relaxed);
  if (print_every < 1) print_every = 1;
  long long call_idx = 0;
  bool profile_block = (prof_lvl >= 2);
  if (profile_block) {
    call_idx = g_wan_profile_call_idx.fetch_add(1, std::memory_order_relaxed);
    profile_block = (call_idx % print_every) == 0;
  }
  enum {
    EV_START = 0,
    EV_AFTER_LN_SA,
    EV_AFTER_SA,        // after self-attn output projection, before residual gate add
    EV_AFTER_SA_RES,
    EV_AFTER_LN_CA,
    EV_AFTER_CA,
    EV_AFTER_LN_FFN,    // after FFN pre-norm+mod
    EV_AFTER_FFN1,
    EV_AFTER_FFN2_GEMM, // after FFN GEMM2, before residual gate add
    EV_AFTER_FFN2,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (profile_block) cudaEventRecord(ev[idx], stream);
  };
  if (profile_block) {
    for (int i = 0; i < EV_COUNT; ++i) { cudaEventCreate(&ev[i]); }
    rec(EV_START);
  }

  // Use output buffer as destination for updated hidden state; read from p.hidden_states directly
  cutlass::half_t* dHiddenR = p.out;
  cutlass::half_t* scratch_mk_a = p.workspace->scratch_mk_a;
  cutlass::half_t* scratch_mk_b = p.workspace->scratch_mk_b;

  // 1) Self-attention pre-norm + mod -> row-major
  //    Arch::kSelfAttnNorm selects RMSNorm (no affine) vs LayerNorm (gamma+beta).
  cutlass::half_t* dPreSAhRM = scratch_mk_a;

  if constexpr (Arch::kSelfAttnNorm == NormType::RMSNorm) {
    layernorm_modulate_row_rm_to_row_half_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
      p.hidden_states, M, K, 1e-6f, p.temb_scale_shift_table, p.temb, p.temb_row_stride, 1, 0, dPreSAhRM);
  } else {
    layernorm_affine_modulate_row_rm_to_row_half_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
      p.hidden_states, M, K, 1e-6f, p.ln1_gamma, nullptr,
      p.temb_scale_shift_table, p.temb, p.temb_row_stride, 1, 0, dPreSAhRM);
  }
  { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return e; } }
  NAN_CHECK(dPreSAhRM, int64_t(M)*K, "after_LN_SA");
  rec(EV_AFTER_LN_SA);

  // 1b) Run self attention (row-major path)
  cutlass::half_t* dSAOut = scratch_mk_b;
  AttentionDeviceParamsT<WeightT> psa{}; psa.B=B; psa.Mq=Mq; psa.K=K; psa.H=H; psa.D=D; psa.hidden_states=dPreSAhRM;
  psa.w_qkv=p.sa_w_qkv; psa.w_qkv_scale=p.sa_w_qkv_scale; psa.b_qkv=p.sa_b_qkv; psa.norm_q_gamma=p.sa_norm_q_gamma; psa.norm_k_gamma=p.sa_norm_k_gamma;
  psa.w_out=p.sa_w_out; psa.w_out_scale=p.sa_w_out_scale; psa.b_out=p.sa_b_out; psa.rotary_cos=p.rotary_cos; psa.rotary_sin=p.rotary_sin;
  // Per-block INT8 quantization scales
  psa.w_qkv_block_scale=p.sa_w_qkv_block_scale; psa.w_out_block_scale=p.sa_w_out_block_scale;
  psa.use_perblock_quant=p.use_perblock_quant;
  psa.workspace = p.workspace;

  // INT8 on Blackwell: fuse gate+residual into output GEMM epilogue
  // INT8 on Ada: use separate output + residual (fused epilogue crashes in this TU on SM89)
  if constexpr (std::is_same_v<WeightT, int8_t>) {
    static int s_gpu_major_sa = -1;
    if (s_gpu_major_sa < 0) {
      int dev; cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&s_gpu_major_sa, cudaDevAttrComputeCapabilityMajor, dev);
    }
    if (s_gpu_major_sa >= 10) {
      // Blackwell: fused gated residual in GEMM epilogue
      psa.out_after_linear = dHiddenR;
      psa.gate_sst = p.temb_scale_shift_table;
      psa.gate_temb = p.temb;
      psa.gate_temb_row_stride = p.temb_row_stride;
      psa.gate_idx = 2;
      psa.residual_src = p.hidden_states;
    } else {
      // Ada: separate output projection, residual added below
      psa.out_after_linear = dSAOut;
    }
  } else {
    psa.out_after_linear = dSAOut;  // non-INT8: separate residual kernel
  }
  cudaError_t err = run_self_attention<WeightT>(psa, stream);
  if (err != cudaSuccess) return err;
  NAN_CHECK(psa.out_after_linear, int64_t(M)*K, "after_SA_out");
  rec(EV_AFTER_SA);

  // 1c) Residual with gate into row-major destination
  // For INT8 on Blackwell, this was fused into the GEMM above. For all others, do it here.
  {
    bool need_separate_residual = true;
    if constexpr (std::is_same_v<WeightT, int8_t>) {
      static int s_gpu_major_res = -1;
      if (s_gpu_major_res < 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_gpu_major_res, cudaDevAttrComputeCapabilityMajor, dev);
      }
      need_separate_residual = (s_gpu_major_res < 10);  // Ada needs separate residual
    }
    if (need_separate_residual) {
      residual_gate_add_kernel_half_sr<<<dim3((K+31)/32,(M+31)/32), dim3(32,32), 0, stream>>>(
        dHiddenR, K, M, K, dSAOut, K, p.temb_scale_shift_table, p.temb, p.temb_row_stride, 2, p.hidden_states, K);
      { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return e; } }
    }
  }
  NAN_CHECK(dHiddenR, int64_t(M)*K, "after_SA_residual");
  rec(EV_AFTER_SA_RES);

  // 2) Cross-attention pre-norm -> row-major (use current hidden state as source)
  cutlass::half_t* dPreCAhRM = scratch_mk_a;  // Reuse scratch_mk_a
  layernorm_affine_row_to_row_half_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
    dHiddenR, M, K, 1e-6f, p.ln2_gamma, p.ln2_bias, dPreCAhRM);
  { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { return e; } }
  NAN_CHECK(dPreCAhRM, int64_t(M)*K, "after_LN_CA");
  rec(EV_AFTER_LN_CA);

  // 2b) Cross attention (I2V: text + optional image branch).
  //    Arch::kHasImageCrossAttn controls whether the fused I2V path is compiled in.
  bool has_img_branch = false;
  if constexpr (Arch::kHasImageCrossAttn) {
    has_img_branch = (Mk_img > 0) && (p.encoder_hidden_states_img != nullptr) &&
                     (p.ca_w_add_k != nullptr) && (p.ca_w_add_v != nullptr) &&
                     (p.ca_norm_added_k_gamma != nullptr);
  }

  // If image branch is present, prefer the fused I2V path (shared Q, concatenated KV, single attention).
  // This is supported for:
  // - CUTLASS backend (half weights)
  // - cuDNN backend (half and FP8 weights)
  bool used_fused_i2v = false;
  if (has_img_branch) {
    // Only use the fused I2V path when supported by the selected backend:
    // - CUTLASS fused I2V currently supports B=1 only
    // - cuDNN fused I2V supports B>=1 (including CFG-batched B=2)
    if (backend == AttnBackend::CUDNN || B == 1) {
    CrossAttentionI2VParamsT<WeightT> pi2v{};
    pi2v.B = B;
    pi2v.M = Mq; pi2v.K = K; pi2v.H = H; pi2v.D = D;
    pi2v.Mk_text = Mk_text; pi2v.Mk_img = Mk_img; pi2v.added_kv_proj_dim = p.added_kv_proj_dim;
    pi2v.hidden_states = dPreCAhRM;
    pi2v.encoder_hidden_states_text = p.encoder_hidden_states;
    pi2v.encoder_hidden_states_img  = p.encoder_hidden_states_img;
    pi2v.w_q = p.ca_w_q; pi2v.b_q = p.ca_b_q; pi2v.w_q_scale = p.ca_w_q_scale;
    pi2v.w_kv = p.ca_w_kv; pi2v.b_kv = p.ca_b_kv; pi2v.w_kv_scale = p.ca_w_kv_scale;
    pi2v.w_add_k = p.ca_w_add_k; pi2v.b_add_k = p.ca_b_add_k; pi2v.w_add_k_scale = p.ca_w_add_k_scale;
    pi2v.w_add_v = p.ca_w_add_v; pi2v.b_add_v = p.ca_b_add_v; pi2v.w_add_v_scale = p.ca_w_add_v_scale;
    pi2v.norm_q_gamma = p.ca_norm_q_gamma;
    pi2v.norm_k_gamma = p.ca_norm_k_gamma;
    pi2v.norm_added_k_gamma = p.ca_norm_added_k_gamma;
    pi2v.w_out = p.ca_w_out; pi2v.b_out = p.ca_b_out; pi2v.w_out_scale = p.ca_w_out_scale;  // bias applied once
    pi2v.out_after_linear = dHiddenR;                   // residual inout when fuse_residual=true
    pi2v.workspace = p.workspace;
    err = run_cross_attention_i2v<WeightT>(pi2v, /*fuse_residual=*/true, stream);
    if (err == cudaSuccess) {
      used_fused_i2v = true;
    } else if (err != cudaErrorNotSupported) {
      return err;
    }
    }
  }
  // For cuDNN, the non-fused fallback does not support the I2V added-KV (image) branch.
  if (backend == AttnBackend::CUDNN && has_img_branch && !used_fused_i2v) {
    return cudaErrorNotSupported;
  }
  if (!used_fused_i2v) {
    // Text branch (fuse residual into output projection)
    AttentionDeviceParamsT<WeightT> pca_text{};
    pca_text.B = B; pca_text.Mq = Mq; pca_text.K = K; pca_text.H = H; pca_text.D = D;
    pca_text.Mk = Mk_text; pca_text.Mk_img = 0; pca_text.added_kv_proj_dim = p.added_kv_proj_dim;
    pca_text.encoder_batch_size = p.encoder_batch_size;
    pca_text.hidden_states = dPreCAhRM;
    pca_text.encoder_hidden_states = p.encoder_hidden_states;
    pca_text.w_q = p.ca_w_q; pca_text.w_q_scale = p.ca_w_q_scale; pca_text.b_q = p.ca_b_q;
    pca_text.w_kv = p.ca_w_kv; pca_text.w_kv_scale = p.ca_w_kv_scale; pca_text.b_kv = p.ca_b_kv;
    pca_text.norm_q_gamma = p.ca_norm_q_gamma; pca_text.norm_k_gamma = p.ca_norm_k_gamma;
    pca_text.w_out = p.ca_w_out; pca_text.w_out_scale = p.ca_w_out_scale; pca_text.b_out = p.ca_b_out;
    // Per-block INT8 quantization scales
    pca_text.w_q_block_scale=p.ca_w_q_block_scale; pca_text.w_kv_block_scale=p.ca_w_kv_block_scale;
    pca_text.w_out_block_scale=p.ca_w_out_block_scale; pca_text.use_perblock_quant=p.use_perblock_quant;
    pca_text.out_after_linear = dHiddenR;  // residual inout
    pca_text.workspace = p.workspace;
    err = run_cross_attention<WeightT>(pca_text, /*fuse_residual=*/true, stream);
    if (err != cudaSuccess) return err;

    // Optional image branch (bias-less) fused into residual
    if (has_img_branch) {
      AttentionDeviceParamsT<WeightT> pca_img{};
      pca_img.B = B; pca_img.Mq = Mq; pca_img.K = K; pca_img.H = H; pca_img.D = D;
      pca_img.Mk = 0; pca_img.Mk_img = Mk_img; pca_img.added_kv_proj_dim = p.added_kv_proj_dim;
      pca_img.encoder_batch_size = p.encoder_batch_size;
      pca_img.hidden_states = dPreCAhRM;
      pca_img.encoder_hidden_states = p.encoder_hidden_states_img;
      pca_img.w_q = p.ca_w_q; pca_img.w_q_scale = p.ca_w_q_scale; pca_img.b_q = p.ca_b_q;
      pca_img.w_kv = nullptr; pca_img.b_kv = nullptr;
      pca_img.w_kv_scale = nullptr;
      pca_img.w_add_k = p.ca_w_add_k; pca_img.b_add_k = p.ca_b_add_k;
      pca_img.w_add_v = p.ca_w_add_v; pca_img.b_add_v = p.ca_b_add_v;
      pca_img.w_add_k_scale = p.ca_w_add_k_scale;
      pca_img.w_add_v_scale = p.ca_w_add_v_scale;
      pca_img.norm_q_gamma = p.ca_norm_q_gamma; pca_img.norm_k_gamma = nullptr;
      pca_img.norm_added_k_gamma = p.ca_norm_added_k_gamma;
      pca_img.w_out = p.ca_w_out; pca_img.w_out_scale = p.ca_w_out_scale; pca_img.b_out = nullptr;  // bias only once (text)
      // Per-block INT8 quantization scales (image branch shares same out projection)
      pca_img.w_q_block_scale=p.ca_w_q_block_scale; pca_img.w_out_block_scale=p.ca_w_out_block_scale;
      pca_img.use_perblock_quant=p.use_perblock_quant;
      pca_img.out_after_linear = dHiddenR;                  // residual inout
      pca_img.workspace = p.workspace;
      err = run_cross_attention<WeightT>(pca_img, /*fuse_residual=*/true, stream);
      if (err != cudaSuccess) return err;
    }
  }

  NAN_CHECK(dHiddenR, int64_t(M)*K, "after_CA_residual");
  rec(EV_AFTER_CA);

  // 3) FFN pre-norm + mod (row-major)
  cutlass::half_t* dPreFFh = scratch_mk_a;  // Reuse scratch_mk_a
  layernorm_modulate_row_rm_to_row_half_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
    dHiddenR, M, K, 1e-6f, p.temb_scale_shift_table, p.temb, p.temb_row_stride, 4, 3, dPreFFh);
  { cudaError_t e3 = cudaGetLastError(); if (e3 != cudaSuccess) { return e3; } }
  rec(EV_AFTER_LN_FFN);

  // 3b) FFN GEMM1: [M,K] x [FF,K]^T -> [M,FF] with GELU activation
  int FF = p.FF;
  {
    cutlass::half_t* dFF2_row = scratch_mk_b;  // Reuse scratch_mk_b

    if constexpr (std::is_same_v<WeightT, int8_t>) {
      // Detect GPU arch: SM < 10 = Ada/Ampere, SM >= 10 = Blackwell+
      static int s_gpu_major = -1;
      if (s_gpu_major < 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_gpu_major, cudaDevAttrComputeCapabilityMajor, dev);
      }
      bool use_fused_ffn = (s_gpu_major >= 10);  // Blackwell: fused kernels work

      if (use_fused_ffn) {
        // Blackwell path: fused GEMM+GELU+requant and GEMM+gated_residual
        // Step 1: Quantize FFN input [M,K] FP16 -> INT8
        cudaError_t q_err = quantize_per_block_128(
          reinterpret_cast<const half*>(dPreFFh),
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          M, K, stream, false);
        if (q_err != cudaSuccess) return q_err;

        // Step 2: FFN up GEMM + GELU with fused requantize epilogue
        int8_t* ffn_int8_out = reinterpret_cast<int8_t*>(p.workspace->ffn_intermediate);
        float* ffn_int8_out_scales = reinterpret_cast<float*>(
          reinterpret_cast<char*>(p.workspace->ffn_intermediate) + (size_t)M * FF);

        cudaError_t err1 = int8_gemm_gelu_requant(
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          reinterpret_cast<const int8_t*>(p.ffn_w1),
          p.ffn_w1_block_scale,
          ffn_int8_out,
          ffn_int8_out_scales,
          reinterpret_cast<const half*>(p.ffn_b1),
          M, FF, K, stream,
          reinterpret_cast<half*>(dFF2_row));  // fp16_temp for 2-stage fallback
        if (err1 != cudaSuccess) return err1;

        rec(EV_AFTER_FFN1);

        // Step 3: FFN down GEMM + gated residual (fused)
        cudaError_t err2 = int8_gemm_gated_residual(
          ffn_int8_out,
          ffn_int8_out_scales,
          reinterpret_cast<const int8_t*>(p.ffn_w2),
          p.ffn_w2_block_scale,
          reinterpret_cast<half*>(dHiddenR),
          reinterpret_cast<const half*>(p.ffn_b2),
          reinterpret_cast<const half*>(p.temb_scale_shift_table),
          reinterpret_cast<const half*>(p.temb),
          p.temb_row_stride,
          5,  // gate_idx for FFN
          M, K, FF, stream);
        if (err2 != cudaSuccess) return err2;

        rec(EV_AFTER_FFN2_GEMM);

      } else {
        // Ada/SM89 path: unfused kernels
        cutlass::half_t* dFF1h = p.workspace->ffn_intermediate;

        cudaError_t q_err = quantize_per_block_128(
          reinterpret_cast<const half*>(dPreFFh),
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          M, K, stream, false);
        if (q_err != cudaSuccess) return q_err;

        cudaError_t err1 = int8_gemm_gelu(
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          reinterpret_cast<const int8_t*>(p.ffn_w1),
          p.ffn_w1_block_scale,
          reinterpret_cast<half*>(dFF1h),
          reinterpret_cast<const half*>(p.ffn_b1),
          M, FF, K, stream);
        if (err1 != cudaSuccess) return err1;

        rec(EV_AFTER_FFN1);

        cudaError_t rq_err = quantize_per_block_128(
          reinterpret_cast<const half*>(dFF1h),
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          M, FF, stream, false);
        if (rq_err != cudaSuccess) return rq_err;

        cudaError_t err2 = int8_gemm(
          p.workspace->int8_scratch,
          p.workspace->int8_act_block_scales,
          reinterpret_cast<const int8_t*>(p.ffn_w2),
          p.ffn_w2_block_scale,
          reinterpret_cast<half*>(dFF2_row),
          reinterpret_cast<const half*>(p.ffn_b2),
          M, K, FF, stream);
        if (err2 != cudaSuccess) return err2;

        rec(EV_AFTER_FFN2_GEMM);

        residual_gate_add_kernel_half_rr<<<dim3((K+31)/32,(M+31)/32), dim3(32,32), 0, stream>>>(
          dHiddenR, K, M, K, dFF2_row, K, p.temb_scale_shift_table, p.temb, p.temb_row_stride, 5);
        { cudaError_t e4 = cudaGetLastError(); if (e4 != cudaSuccess) { return e4; } }
      }

    } else {
      // FP16/FP8 path: separate GEMM1 + quantize + GEMM2
      cutlass::half_t* dFF1h = p.workspace->ffn_intermediate;
      // For FP8 with per-channel scales: fuse GELU output -> FP8 conversion for GEMM2 input,
      // eliminating intermediate FP16 write/read between FFN up and FFN down.
      cutlass::half_t* ffn2_fp8_scratch = p.workspace->cudnn.sa_qkv_row;
      constexpr bool is_fp8 = std::is_same<WeightT, cutlass::float_e4m3_t>::value;
      cutlass::float_e4m3_t* fp8_next = nullptr;
      if constexpr (is_fp8) {
        if (p.ffn_w1_scale != nullptr) {
          fp8_next = reinterpret_cast<cutlass::float_e4m3_t*>(ffn2_fp8_scratch);
        }
      }
      cudaError_t err1 = apply_linear_row_gelu<WeightT>(
        dPreFFh,
        p.ffn_w1,
        p.ffn_b1,
        dFF1h,
        M, K, FF, stream,
        scratch_mk_b,  // fp8_scratch
        p.ffn_w1_scale,  // weight_scale (for FP8 with per-channel scales)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales (for per-block INT8)
        p.ffn_w1_block_scale,  // weight_block_scale (for per-block INT8)
        false,  // use_swizzled_weights
        fp8_next);  // fused GELU->FP8 output (nullptr for non-FP8 or no weight_scale)
      if (err1 != cudaSuccess) return err1;

      NAN_CHECK(dFF1h, int64_t(M)*FF, "after_FFN1_gelu");
      rec(EV_AFTER_FFN1);

      // 3c) FFN GEMM2: [M,FF] x [K,FF]^T -> [M,K]
      // When fp8_next was set, the fused kernel already wrote FP8 data to ffn2_fp8_scratch,
      // so skip the input quantization step in apply_linear_row.
      bool skip_quant = (fp8_next != nullptr);
      cudaError_t err2 = apply_linear_row<WeightT>(
        dFF1h,
        p.ffn_w2,
        p.ffn_b2,
        dFF2_row,
        M, FF, K, stream,
        ffn2_fp8_scratch,  // fp8_scratch (has FP8 data if fused, else used as scratch)
        p.ffn_w2_scale,  // weight_scale (for FP8)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales
        p.ffn_w2_block_scale,  // weight_block_scale
        false,  // use_swizzled_weights
        skip_quant);  // fp8_input_preconverted
      if (err2 != cudaSuccess) return err2;

      NAN_CHECK(dFF2_row, int64_t(M)*K, "after_FFN2_gemm");
      // 3d) Residual with gate (FP16/FP8 path only -- INT8 path fuses this into GEMM)
      //    Arch::kFFNGateStyle: 1 = gated residual from scale_shift_table, 0 = plain residual add
      rec(EV_AFTER_FFN2_GEMM);
      if constexpr (Arch::kFFNGateStyle == 1) {
        residual_gate_add_kernel_half_rr<<<dim3((K+31)/32,(M+31)/32), dim3(32,32), 0, stream>>>(
          dHiddenR, K, M, K, dFF2_row, K, p.temb_scale_shift_table, p.temb, p.temb_row_stride, 5);
      } else {
        add_inplace_half_rr<<<dim3((K+31)/32,(M+31)/32), dim3(32,32), 0, stream>>>(
          dHiddenR, K, M, K, dFF2_row, K);
      }
      { cudaError_t e4 = cudaGetLastError(); if (e4 != cudaSuccess) { return e4; } }
    }
  }

  NAN_CHECK(dHiddenR, int64_t(M)*K, "after_FFN_residual");
  rec(EV_AFTER_FFN2);

  if (profile_block) {
    cudaEventSynchronize(ev[EV_AFTER_FFN2]);
    auto ms = [&](int a, int b) -> float {
      float out = 0.f;
      cudaEventElapsedTime(&out, ev[a], ev[b]);
      return out;
    };
    float t_ln1   = ms(EV_START, EV_AFTER_LN_SA);
    float t_self  = ms(EV_AFTER_LN_SA, EV_AFTER_SA_RES);
    float t_ln2   = ms(EV_AFTER_SA_RES, EV_AFTER_LN_CA);
    float t_cross = ms(EV_AFTER_LN_CA, EV_AFTER_CA);
    float t_ffn1  = ms(EV_AFTER_CA, EV_AFTER_FFN1);
    float t_ffn2  = ms(EV_AFTER_FFN1, EV_AFTER_FFN2);
    float t_total = ms(EV_START, EV_AFTER_FFN2);

    std::printf(
      "[wan_block][L=%d][call=%lld] M=%d K=%d H=%d D=%d Mk_text=%d Mk_img=%d | ln1=%.3f self=%.3f ln2=%.3f cross=%.3f ffn1=%.3f ffn2=%.3f total=%.3f ms\n",
      p.layer_idx, call_idx, M, K, H, D, Mk_text, Mk_img,
      t_ln1, t_self, t_ln2, t_cross, t_ffn1, t_ffn2, t_total
    );

#ifdef OMNIDREAMS_SINGLEVIEW_PROFILE
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "ln1", t_ln1);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "self_attn", t_self);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "ln2", t_ln2);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "cross_attn", t_cross);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "ffn1", t_ffn1);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "ffn2", t_ffn2);
    OMNIDREAMS_SINGLEVIEW_PROF_RECORD("block_detail", "total", t_total);
#endif

    if (prof_lvl >= 3) {
      float t_sa_attn = ms(EV_AFTER_LN_SA, EV_AFTER_SA);
      float t_sa_res  = ms(EV_AFTER_SA, EV_AFTER_SA_RES);
      float t_ffn_ln  = ms(EV_AFTER_CA, EV_AFTER_LN_FFN);
      float t_ffn_g1  = ms(EV_AFTER_LN_FFN, EV_AFTER_FFN1);
      float t_ffn_g2  = ms(EV_AFTER_FFN1, EV_AFTER_FFN2_GEMM);
      float t_ffn_res = ms(EV_AFTER_FFN2_GEMM, EV_AFTER_FFN2);
      std::printf(
        "[wan_block_detail][L=%d][call=%lld] sa_attn=%.3f sa_res=%.3f | ffn_ln=%.3f ffn_g1=%.3f ffn_g2=%.3f ffn_res=%.3f ms\n",
        p.layer_idx, call_idx,
        t_sa_attn, t_sa_res,
        t_ffn_ln, t_ffn_g1, t_ffn_g2, t_ffn_res
      );
    }

    for (int i = 0; i < EV_COUNT; ++i) { cudaEventDestroy(ev[i]); }
  }

  return cudaSuccess;
}

// Explicit template instantiations (WeightT x Arch combinations).
// Currently only WanArchTraits is instantiated; future models add their own lines.
template cudaError_t run_transformer_block<cutlass::half_t, WanArchTraits>(
  const TransformerBlockParamsT<cutlass::half_t>& p,
  cudaStream_t stream
);
template cudaError_t run_transformer_block<cutlass::float_e4m3_t, WanArchTraits>(
  const TransformerBlockParamsT<cutlass::float_e4m3_t>& p,
  cudaStream_t stream
);
template cudaError_t run_transformer_block<int8_t, WanArchTraits>(
  const TransformerBlockParamsT<int8_t>& p,
  cudaStream_t stream
);

} // namespace omnidreams_singleview
