// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cosmos_fp8_two_gemm.cuh"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/numeric_conversion.h"

#include <cfloat>
#include <cmath>

namespace omnidreams_singleview {
namespace {

using CosmosFp8BatchedRcrGemm = cutlass::gemm::device::GemmBatched<
    cutlass::float_e4m3_t,
    cutlass::layout::RowMajor,
    cutlass::float_e4m3_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    3>;

static cudaError_t fp8_batched_rcr_gemm(
    const cutlass::float_e4m3_t* input_row,
    const cutlass::float_e4m3_t* weight_col,
    cutlass::half_t* output_row,
    int batch_count,
    int m,
    int k,
    int n,
    int64_t batch_stride_a,
    int64_t batch_stride_b,
    int64_t batch_stride_c,
    float alpha,
    cudaStream_t stream) {
  CosmosFp8BatchedRcrGemm gemm_op;
  CosmosFp8BatchedRcrGemm::Arguments args(
      {m, n, k},
      {input_row, k},
      batch_stride_a,
      {weight_col, k},
      batch_stride_b,
      {output_row, n},
      batch_stride_c,
      {output_row, n},
      batch_stride_c,
      {alpha, 0.0f},
      batch_count);

  cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaSuccess;
}

__global__ void fp8_dense_ref_softmax_kernel(
    const cutlass::half_t* __restrict__ scores,
    cutlass::float_e4m3_t* __restrict__ probs,
    int groups,
    int rows,
    int cols,
    bool causal) {
  int row = blockIdx.x;
  int group = blockIdx.y;
  if (group >= groups || row >= rows) return;

  int tid = threadIdx.x;
  extern __shared__ float smem[];
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::float_e4m3_t, float> to_fp8;

  const size_t group_offset = static_cast<size_t>(group) * rows * cols;
  const cutlass::half_t* score_row = scores + group_offset + static_cast<size_t>(row) * cols;
  cutlass::float_e4m3_t* prob_row = probs + group_offset + static_cast<size_t>(row) * cols;

  float max_val = -FLT_MAX;
  for (int col = tid; col < cols; col += blockDim.x) {
    bool valid = !causal || col <= row;
    if (valid) {
      max_val = fmaxf(max_val, to_f32(score_row[col]));
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
      sum += expf(to_f32(score_row[col]) - max_val);
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
      p = expf(to_f32(score_row[col]) - max_val) / sum;
    }
    prob_row[col] = to_fp8(p);
  }
}

template <typename ElementT>
__global__ void pack_bmhk_to_bhmd_kernel(
    const ElementT* __restrict__ src,
    ElementT* __restrict__ dst,
    int B,
    int M,
    int H,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(B) * M * H * D;
  if (idx >= total) return;
  int d = static_cast<int>(idx % D);
  int h = static_cast<int>((idx / D) % H);
  int m = static_cast<int>((idx / (D * H)) % M);
  int b = static_cast<int>(idx / (static_cast<int64_t>(M) * H * D));
  size_t dst_idx = ((static_cast<size_t>(b) * H + h) * M + m) * D + d;
  dst[dst_idx] = src[idx];
}

template <typename ElementT>
__global__ void pack_bhmd_to_bmhk_kernel(
    const ElementT* __restrict__ src,
    ElementT* __restrict__ dst,
    int B,
    int M,
    int H,
    int D) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(B) * H * M * D;
  if (idx >= total) return;
  int d = static_cast<int>(idx % D);
  int m = static_cast<int>((idx / D) % M);
  int h = static_cast<int>((idx / (D * M)) % H);
  int b = static_cast<int>(idx / (static_cast<int64_t>(H) * M * D));
  size_t dst_idx = ((static_cast<size_t>(b) * M + m) * H + h) * D + d;
  dst[dst_idx] = src[idx];
}

template <typename ElementT>
static cudaError_t pack_bmhk_to_bhmd(
    const ElementT* src,
    ElementT* dst,
    int B,
    int M,
    int H,
    int D,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(B) * M * H * D;
  if (total <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  pack_bmhk_to_bhmd_kernel<ElementT><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst, B, M, H, D);
  return cudaGetLastError();
}

template <typename ElementT>
static cudaError_t pack_bhmd_to_bmhk(
    const ElementT* src,
    ElementT* dst,
    int B,
    int M,
    int H,
    int D,
    cudaStream_t stream) {
  int64_t total = static_cast<int64_t>(B) * M * H * D;
  if (total <= 0) return cudaSuccess;
  constexpr int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  pack_bhmd_to_bmhk_kernel<ElementT><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst, B, M, H, D);
  return cudaGetLastError();
}

}  // namespace

cudaError_t run_cosmos_fp8_dense_ref(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::float_e4m3_t* k_bhmd,
    cutlass::float_e4m3_t* v_bhmd,
    cutlass::half_t* scores,
    cutlass::float_e4m3_t* probs,
    cutlass::half_t* o_bhmd,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    cudaStream_t stream) {
  if (!q || !k || !v || !q_bhmd || !k_bhmd || !v_bhmd || !scores || !probs || !o_bhmd || !o) {
    return cudaErrorInvalidValue;
  }
  if (B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) return cudaErrorInvalidValue;
  if (!(D == 32 || D == 64 || D == 128)) return cudaErrorInvalidValue;

  cudaError_t err = pack_bmhk_to_bhmd(q, q_bhmd, B, Mq, H, D, stream);
  if (err != cudaSuccess) return err;
  err = pack_bmhk_to_bhmd(k, k_bhmd, B, Mk, H, D, stream);
  if (err != cudaSuccess) return err;
  err = pack_bmhk_to_bhmd(v, v_bhmd, B, Mk, H, D, stream);
  if (err != cudaSuccess) return err;

  const int groups = B * H;
  const size_t q_group_elems = static_cast<size_t>(Mq) * D;
  const size_t kv_group_elems = static_cast<size_t>(Mk) * D;
  const size_t score_group_elems = static_cast<size_t>(Mq) * Mk;
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));

  err = fp8_batched_rcr_gemm(
      q_bhmd,
      k_bhmd,
      scores,
      groups,
      Mq, D, Mk,
      static_cast<int64_t>(q_group_elems),
      static_cast<int64_t>(kv_group_elems),
      static_cast<int64_t>(score_group_elems),
      attn_scale,
      stream);
  if (err != cudaSuccess) return err;

  fp8_dense_ref_softmax_kernel<<<dim3(Mq, groups), 256, 256 * sizeof(float), stream>>>(
      scores, probs, groups, Mq, Mk, causal);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = fp8_batched_rcr_gemm(
      probs,
      v_bhmd,
      o_bhmd,
      groups,
      Mq, Mk, D,
      static_cast<int64_t>(score_group_elems),
      static_cast<int64_t>(kv_group_elems),
      static_cast<int64_t>(q_group_elems),
      1.0f,
      stream);
  if (err != cudaSuccess) return err;

  return pack_bhmd_to_bmhk(o_bhmd, o, B, Mq, H, D, stream);
}

}  // namespace omnidreams_singleview
