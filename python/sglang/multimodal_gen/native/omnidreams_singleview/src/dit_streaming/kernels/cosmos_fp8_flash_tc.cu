// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cosmos_fp8_flash_tc.cuh"

#include "cosmos_fp8_tc_probe.cuh"

#include "cutlass/numeric_conversion.h"
#include "cutlass/util/device_memory.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace omnidreams_singleview {
namespace {

constexpr int kTcTileMultiple = 128;

static bool is_tc_tile_multiple(int value) {
  return value > 0 && (value % kTcTileMultiple) == 0;
}

__global__ void fill_float_kernel(float* __restrict__ dst, int64_t n, float value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = value;
  }
}

__global__ void pack_qkv_bmhk_to_tc_layouts_kernel(
    const cutlass::float_e4m3_t* __restrict__ q_src,
    const cutlass::float_e4m3_t* __restrict__ k_src,
    const cutlass::float_e4m3_t* __restrict__ v_src,
    cutlass::float_e4m3_t* __restrict__ q_dst_bhmd,
    cutlass::float_e4m3_t* __restrict__ k_dst_bhmd,
    cutlass::float_e4m3_t* __restrict__ v_dst_bhdm,
    int B,
    int Mq,
    int Mk,
    int H,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t q_total = static_cast<int64_t>(B) * Mq * H * D;
  const int64_t kv_total = static_cast<int64_t>(B) * Mk * H * D;
  const int64_t total = q_total + 2 * kv_total;
  if (idx >= total) return;

  if (idx < q_total) {
    int d = static_cast<int>(idx % D);
    int h = static_cast<int>((idx / D) % H);
    int m = static_cast<int>((idx / (static_cast<int64_t>(D) * H)) % Mq);
    int b = static_cast<int>(idx / (static_cast<int64_t>(Mq) * H * D));
    size_t dst_idx = ((static_cast<size_t>(b) * H + h) * Mq + m) * D + d;
    q_dst_bhmd[dst_idx] = q_src[idx];
    return;
  }

  idx -= q_total;
  if (idx < kv_total) {
    int d = static_cast<int>(idx % D);
    int h = static_cast<int>((idx / D) % H);
    int m = static_cast<int>((idx / (static_cast<int64_t>(D) * H)) % Mk);
    int b = static_cast<int>(idx / (static_cast<int64_t>(Mk) * H * D));
    size_t dst_idx = ((static_cast<size_t>(b) * H + h) * Mk + m) * D + d;
    k_dst_bhmd[dst_idx] = k_src[idx];
    return;
  }

  idx -= kv_total;
  int d = static_cast<int>(idx % D);
  int h = static_cast<int>((idx / D) % H);
  int m = static_cast<int>((idx / (static_cast<int64_t>(D) * H)) % Mk);
  int b = static_cast<int>(idx / (static_cast<int64_t>(Mk) * H * D));
  size_t dst_idx = ((static_cast<size_t>(b) * H + h) * D + d) * Mk + m;
  v_dst_bhdm[dst_idx] = v_src[idx];
}

__global__ void pack_q_bmhk_to_bhmd_kernel(
    const cutlass::float_e4m3_t* __restrict__ q_src,
    cutlass::float_e4m3_t* __restrict__ q_dst_bhmd,
    int B,
    int Mq,
    int H,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(B) * Mq * H * D;
  if (idx >= total) return;

  int d = static_cast<int>(idx % D);
  int h = static_cast<int>((idx / D) % H);
  int m = static_cast<int>((idx / (static_cast<int64_t>(D) * H)) % Mq);
  int b = static_cast<int>(idx / (static_cast<int64_t>(Mq) * H * D));
  size_t dst_idx = ((static_cast<size_t>(b) * H + h) * Mq + m) * D + d;
  q_dst_bhmd[dst_idx] = q_src[idx];
}

__global__ void softmax_bf16_scores_to_fp8_kernel(
    const cutlass::bfloat16_t* __restrict__ scores,
    cutlass::float_e4m3_t* __restrict__ probs,
    int groups,
    int rows,
    int cols,
    bool causal,
    float softmax_scale) {
  int row = blockIdx.x;
  int group = blockIdx.y;
  if (group >= groups || row >= rows) return;

  int tid = threadIdx.x;
  extern __shared__ float smem[];
  cutlass::NumericConverter<float, cutlass::bfloat16_t> to_f32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  const size_t group_offset = static_cast<size_t>(group) * rows * cols;
  const cutlass::bfloat16_t* score_row = scores + group_offset + static_cast<size_t>(row) * cols;
  cutlass::float_e4m3_t* prob_row = probs + group_offset + static_cast<size_t>(row) * cols;

  float max_val = -FLT_MAX;
  for (int col = tid; col < cols; col += blockDim.x) {
    bool valid = !causal || col <= row;
    if (valid) {
      float value = to_f32(score_row[col]) * softmax_scale;
      max_val = fmaxf(max_val, value);
    }
  }
  smem[tid] = max_val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }
  max_val = smem[0];

  float sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    bool valid = !causal || col <= row;
    if (valid) {
      float value = to_f32(score_row[col]) * softmax_scale;
      sum += expf(value - max_val);
    }
  }
  smem[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }
  sum = smem[0];

  for (int col = tid; col < cols; col += blockDim.x) {
    bool valid = !causal || col <= row;
    float p = 0.0f;
    if (valid && sum > 0.0f) {
      float value = to_f32(score_row[col]) * softmax_scale;
      p = expf(value - max_val) / sum;
    }
    prob_row[col] = to_fp8(p);
  }
}

__global__ void unpack_bhmd_bf16_to_bmhk_half_fp8_kernel(
    const cutlass::bfloat16_t* __restrict__ src,
    cutlass::half_t* __restrict__ dst_half,
    cutlass::float_e4m3_t* __restrict__ dst_fp8,
    int B,
    int M,
    int H,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(B) * H * M * D;
  if (idx >= total) return;

  int d = static_cast<int>(idx % D);
  int m = static_cast<int>((idx / D) % M);
  int h = static_cast<int>((idx / (static_cast<int64_t>(D) * M)) % H);
  int b = static_cast<int>(idx / (static_cast<int64_t>(H) * M * D));
  size_t dst_idx = ((static_cast<size_t>(b) * M + m) * H + h) * D + d;

  cutlass::NumericConverter<float, cutlass::bfloat16_t> to_f32;
  float value = to_f32(src[idx]);
  if (dst_half) {
    cutlass::NumericConverter<cutlass::half_t, float> to_half;
    dst_half[dst_idx] = to_half(value);
  }
  if (dst_fp8) {
    cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;
    dst_fp8[dst_idx] = to_fp8(value);
  }
}

static cudaError_t pack_qkv_bmhk_to_tc_layouts(
    const cutlass::float_e4m3_t* q_src,
    const cutlass::float_e4m3_t* k_src,
    const cutlass::float_e4m3_t* v_src,
    cutlass::float_e4m3_t* q_dst_bhmd,
    cutlass::float_e4m3_t* k_dst_bhmd,
    cutlass::float_e4m3_t* v_dst_bhdm,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    cudaStream_t stream) {
  int64_t q_total = static_cast<int64_t>(B) * Mq * H * D;
  int64_t kv_total = static_cast<int64_t>(B) * Mk * H * D;
  int64_t total = q_total + 2 * kv_total;
  if (total <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  pack_qkv_bmhk_to_tc_layouts_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      q_src, k_src, v_src, q_dst_bhmd, k_dst_bhmd, v_dst_bhdm, B, Mq, Mk, H, D);
  return cudaGetLastError();
}

static cudaError_t pack_q_bmhk_to_bhmd(
    const cutlass::float_e4m3_t* q_src,
    cutlass::float_e4m3_t* q_dst_bhmd,
    int B,
    int Mq,
    int H,
    int D,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(B) * Mq * H * D;
  if (total <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  pack_q_bmhk_to_bhmd_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      q_src, q_dst_bhmd, B, Mq, H, D);
  return cudaGetLastError();
}

static cudaError_t unpack_bhmd_bf16_to_bmhk_half_fp8(
    const cutlass::bfloat16_t* src,
    cutlass::half_t* dst_half,
    cutlass::float_e4m3_t* dst_fp8,
    int B,
    int M,
    int H,
    int D,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(B) * M * H * D;
  if (total <= 0) return cudaSuccess;
  if (!src || (!dst_half && !dst_fp8)) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  unpack_bhmd_bf16_to_bmhk_half_fp8_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst_half, dst_fp8, B, M, H, D);
  return cudaGetLastError();
}

static cudaError_t fill_float(float* dst, int64_t n, float value, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  fill_float_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(dst, n, value);
  return cudaGetLastError();
}

}  // namespace

cudaError_t run_cosmos_fp8_flash_tc(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    cudaStream_t stream) {
  if (!q || !k || !v || !o || B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) {
    return cudaErrorInvalidValue;
  }
  if (D != 128) {
    return cudaErrorInvalidValue;
  }
  if (!is_tc_tile_multiple(Mq) || !is_tc_tile_multiple(Mk)) {
    return cudaErrorNotSupported;
  }

  const int groups = B * H;
  const size_t q_group_elems = static_cast<size_t>(Mq) * D;
  const size_t k_group_elems = static_cast<size_t>(Mk) * D;
  const size_t score_group_elems = static_cast<size_t>(Mq) * Mk;
  const size_t out_group_elems = static_cast<size_t>(Mq) * D;
  const int64_t q_total = static_cast<int64_t>(groups) * q_group_elems;
  const int64_t k_total = static_cast<int64_t>(groups) * k_group_elems;
  const int64_t v_t_total = static_cast<int64_t>(groups) * static_cast<int64_t>(D) * Mk;
  const int64_t score_total = static_cast<int64_t>(groups) * score_group_elems;
  const int64_t out_total = static_cast<int64_t>(groups) * out_group_elems;
  const int64_t scale_elems_per_group = std::max<int64_t>(
      std::max<int64_t>(q_group_elems, k_group_elems),
      score_group_elems);
  const int64_t scale_elems = static_cast<int64_t>(groups) * scale_elems_per_group;

  cutlass::device_memory::allocation<cutlass::float_e4m3_t> q_bhmd(q_total);
  cutlass::device_memory::allocation<cutlass::float_e4m3_t> k_bhmd(k_total);
  cutlass::device_memory::allocation<cutlass::float_e4m3_t> v_bhdm(v_t_total);
  cutlass::device_memory::allocation<cutlass::bfloat16_t> scores(score_total);
  cutlass::device_memory::allocation<cutlass::bfloat16_t> score_c_scratch(score_total);
  cutlass::device_memory::allocation<cutlass::float_e4m3_t> probs(score_total);
  cutlass::device_memory::allocation<cutlass::bfloat16_t> out_bhmd(out_total);
  cutlass::device_memory::allocation<cutlass::bfloat16_t> out_c_scratch(out_total);
  cutlass::device_memory::allocation<float> tc_scale(scale_elems);

  return run_cosmos_fp8_flash_tc_workspace(
      q,
      k,
      v,
      q_bhmd.get(),
      k_bhmd.get(),
      v_bhdm.get(),
      scores.get(),
      score_c_scratch.get(),
      probs.get(),
      out_bhmd.get(),
      out_c_scratch.get(),
      o,
      nullptr,
      tc_scale.get(),
      scale_elems,
      B, Mq, Mk, H, D, causal, /*tc_scale_is_ones=*/false, stream);
}

cudaError_t run_cosmos_fp8_flash_tc_workspace(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::float_e4m3_t* k_bhmd,
    cutlass::float_e4m3_t* v_bhdm,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream) {
  if (!q || !k || !v || !q_bhmd || !k_bhmd || !v_bhdm || !scores || !score_c_scratch ||
      !probs || !out_bhmd || !out_c_scratch || (!o && !o_fp8) || !tc_scale ||
      B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) {
    return cudaErrorInvalidValue;
  }
  if (D != 128) {
    return cudaErrorInvalidValue;
  }
  if (!is_tc_tile_multiple(Mq) || !is_tc_tile_multiple(Mk)) {
    return cudaErrorNotSupported;
  }

  const int groups = B * H;
  const size_t q_group_elems = static_cast<size_t>(Mq) * D;
  const size_t k_group_elems = static_cast<size_t>(Mk) * D;
  const size_t score_group_elems = static_cast<size_t>(Mq) * Mk;
  const int64_t scale_elems_per_group = std::max<int64_t>(
      std::max<int64_t>(q_group_elems, k_group_elems),
      score_group_elems);
  const int64_t required_scale_elems = static_cast<int64_t>(groups) * scale_elems_per_group;
  if (tc_scale_elems < required_scale_elems) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaSuccess;
  if (!tc_scale_is_ones) {
    err = fill_float(tc_scale, required_scale_elems, 1.0f, stream);
    if (err != cudaSuccess) return err;
  }
  err = pack_qkv_bmhk_to_tc_layouts(q, k, v, q_bhmd, k_bhmd, v_bhdm, B, Mq, Mk, H, D, stream);
  if (err != cudaSuccess) return err;

  err = run_cosmos_fp8_tc_probe_qk_batched(
      q_bhmd,
      k_bhmd,
      tc_scale,
      tc_scale,
      score_c_scratch,
      scores,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  constexpr int softmax_threads = 256;
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  softmax_bf16_scores_to_fp8_kernel<<<
      dim3(Mq, groups),
      softmax_threads,
      softmax_threads * sizeof(float),
      stream>>>(scores, probs, groups, Mq, Mk, causal, softmax_scale);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = run_cosmos_fp8_tc_probe_pv_batched(
      probs,
      v_bhdm,
      tc_scale,
      tc_scale,
      out_c_scratch,
      out_bhmd,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  return unpack_bhmd_bf16_to_bmhk_half_fp8(out_bhmd, o, o_fp8, B, Mq, H, D, stream);
}

cudaError_t run_cosmos_fp8_flash_tc_workspace_prepacked_kv(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k_bhmd,
    const cutlass::float_e4m3_t* v_bhdm,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream) {
  if (!q || !k_bhmd || !v_bhdm || !q_bhmd || !scores || !score_c_scratch ||
      !probs || !out_bhmd || !out_c_scratch || (!o && !o_fp8) || !tc_scale ||
      B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) {
    return cudaErrorInvalidValue;
  }
  if (D != 128) {
    return cudaErrorInvalidValue;
  }
  if (!is_tc_tile_multiple(Mq) || !is_tc_tile_multiple(Mk)) {
    return cudaErrorNotSupported;
  }

  const int groups = B * H;
  const size_t q_group_elems = static_cast<size_t>(Mq) * D;
  const size_t k_group_elems = static_cast<size_t>(Mk) * D;
  const size_t score_group_elems = static_cast<size_t>(Mq) * Mk;
  const int64_t scale_elems_per_group = std::max<int64_t>(
      std::max<int64_t>(q_group_elems, k_group_elems),
      score_group_elems);
  const int64_t required_scale_elems = static_cast<int64_t>(groups) * scale_elems_per_group;
  if (tc_scale_elems < required_scale_elems) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaSuccess;
  if (!tc_scale_is_ones) {
    err = fill_float(tc_scale, required_scale_elems, 1.0f, stream);
    if (err != cudaSuccess) return err;
  }
  err = pack_q_bmhk_to_bhmd(q, q_bhmd, B, Mq, H, D, stream);
  if (err != cudaSuccess) return err;

  err = run_cosmos_fp8_tc_probe_qk_batched(
      q_bhmd,
      k_bhmd,
      tc_scale,
      tc_scale,
      score_c_scratch,
      scores,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  constexpr int softmax_threads = 256;
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  softmax_bf16_scores_to_fp8_kernel<<<
      dim3(Mq, groups),
      softmax_threads,
      softmax_threads * sizeof(float),
      stream>>>(scores, probs, groups, Mq, Mk, causal, softmax_scale);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = run_cosmos_fp8_tc_probe_pv_batched(
      probs,
      v_bhdm,
      tc_scale,
      tc_scale,
      out_c_scratch,
      out_bhmd,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  return unpack_bhmd_bf16_to_bmhk_half_fp8(out_bhmd, o, o_fp8, B, Mq, H, D, stream);
}

cudaError_t run_cosmos_fp8_flash_tc_workspace_prepacked_qkv(
    const cutlass::float_e4m3_t* q_bhmd,
    const cutlass::float_e4m3_t* k_bhmd,
    const cutlass::float_e4m3_t* v_bhdm,
    cutlass::bfloat16_t* scores,
    cutlass::bfloat16_t* score_c_scratch,
    cutlass::float_e4m3_t* probs,
    cutlass::bfloat16_t* out_bhmd,
    cutlass::bfloat16_t* out_c_scratch,
    cutlass::half_t* o,
    cutlass::float_e4m3_t* o_fp8,
    float* tc_scale,
    int64_t tc_scale_elems,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    bool tc_scale_is_ones,
    cudaStream_t stream) {
  if (!q_bhmd || !k_bhmd || !v_bhdm || !scores || !score_c_scratch ||
      !probs || !out_bhmd || !out_c_scratch || (!o && !o_fp8) || !tc_scale ||
      B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) {
    return cudaErrorInvalidValue;
  }
  if (D != 128) {
    return cudaErrorInvalidValue;
  }
  if (!is_tc_tile_multiple(Mq) || !is_tc_tile_multiple(Mk)) {
    return cudaErrorNotSupported;
  }

  const int groups = B * H;
  const size_t q_group_elems = static_cast<size_t>(Mq) * D;
  const size_t k_group_elems = static_cast<size_t>(Mk) * D;
  const size_t score_group_elems = static_cast<size_t>(Mq) * Mk;
  const int64_t scale_elems_per_group = std::max<int64_t>(
      std::max<int64_t>(q_group_elems, k_group_elems),
      score_group_elems);
  const int64_t required_scale_elems = static_cast<int64_t>(groups) * scale_elems_per_group;
  if (tc_scale_elems < required_scale_elems) {
    return cudaErrorInvalidValue;
  }

  cudaError_t err = cudaSuccess;
  if (!tc_scale_is_ones) {
    err = fill_float(tc_scale, required_scale_elems, 1.0f, stream);
    if (err != cudaSuccess) return err;
  }

  err = run_cosmos_fp8_tc_probe_qk_batched(
      q_bhmd,
      k_bhmd,
      tc_scale,
      tc_scale,
      score_c_scratch,
      scores,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  constexpr int softmax_threads = 256;
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  softmax_bf16_scores_to_fp8_kernel<<<
      dim3(Mq, groups),
      softmax_threads,
      softmax_threads * sizeof(float),
      stream>>>(scores, probs, groups, Mq, Mk, causal, softmax_scale);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = run_cosmos_fp8_tc_probe_pv_batched(
      probs,
      v_bhdm,
      tc_scale,
      tc_scale,
      out_c_scratch,
      out_bhmd,
      Mq, Mk, D, groups, stream);
  if (err != cudaSuccess) return err;

  return unpack_bhmd_bf16_to_bmhk_half_fp8(out_bhmd, o, o_fp8, B, Mq, H, D, stream);
}

}  // namespace omnidreams_singleview
