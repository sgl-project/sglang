/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_scan.cuh>
#include <limits>
#include <optional>
#include <tuple>

namespace {
constexpr int kTable = 16384;
constexpr int kThreads = 256;
constexpr int kPerThread = kTable / kThreads;
constexpr int kShift = 26;
constexpr int kSlotMask = (1 << kShift) - 1;

// A group contains at most 5 * 2048 entries, so the hash table cannot fill.
// Bits 26..30 record token ownership; FlashMLA masks each query's softmax
// support using these bits. Top-k slots must be unique within each input row.
__global__ void __launch_bounds__(kThreads) prepare_kernel(
    const uint4* q,
    const uint4* q_rope,
    int64_t q_row_stride,
    int64_t q_head_stride,
    int64_t rope_row_stride,
    int64_t rope_head_stride,
    uint4* packed_q,
    const int* rows,
    int64_t row_stride,
    int tokens,
    int heads,
    int topk,
    int group,
    int out_stride,
    int* out,
    int* lengths) {
  extern __shared__ int table[];
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  const int first_token = blockIdx.x * group;
  int* dst = out + int64_t(blockIdx.x) * out_stride;
  for (int i = threadIdx.x; i < out_stride; i += kThreads)
    dst[i] = -1;

  // Concatenate split Q directly into the packed allocation, zeroing only the
  // final group's missing tokens. An already packed view skips this copy.
  if (packed_q != nullptr) {
    for (int i = threadIdx.x; i < group * heads * 72; i += kThreads) {
      const int token = first_token + i / (heads * 72);
      const int head = (i / 72) % heads;
      const int vec = i % 72;
      uint4 value = make_uint4(0, 0, 0, 0);
      if (token < tokens) {
        value = q_rope != nullptr && vec >= 64
                    ? q_rope[int64_t(token) * rope_row_stride + head * rope_head_stride + vec - 64]
                    : q[int64_t(token) * q_row_stride + head * q_head_stride + vec];
      }
      packed_q[int64_t(blockIdx.x) * group * heads * 72 + i] = value;
    }
  }
  for (int i = threadIdx.x; i < kTable; i += kThreads)
    table[i] = -1;
  __syncthreads();
  for (int e = threadIdx.x; e < group * topk; e += kThreads) {
    const int token = first_token + e / topk;
    if (token >= tokens) continue;
    const int slot = rows[int64_t(token) * row_stride + e % topk];
    if (slot == -1) continue;
    CUDA_KERNEL_ASSERT(slot >= 0 && slot <= kSlotMask);
    const int bit = 1 << (kShift + e / topk);
    unsigned h = (unsigned(slot) * 2654435761u) & (kTable - 1);
    while (true) {
      const int prev = atomicCAS(&table[h], -1, slot | bit);
      if (prev == -1) break;
      if ((prev & kSlotMask) == slot) {
        atomicOr(&table[h], bit);
        break;
      }
      h = (h + 1) & (kTable - 1);
    }
  }
  __syncthreads();
  int values[kPerThread], count = 0;
  for (int j = 0; j < kPerThread; ++j) {
    // Consecutive lanes read different banks. A blocked per-thread layout
    // creates a 32-way conflict here; union output order is not significant.
    values[j] = table[j * kThreads + threadIdx.x];
    count += values[j] != -1;
  }
  int offset, total;
  Scan(scan_tmp).ExclusiveSum(count, offset, total);
  for (int j = 0; j < kPerThread; ++j) {
    if (values[j] != -1) dst[offset++] = values[j];
  }
  if (threadIdx.x == 0) lengths[blockIdx.x] = total;
}

void check_q(const at::Tensor& q, int64_t dim, const at::Tensor& ref) {
  TORCH_CHECK(q.is_cuda() && q.device() == ref.device(), "Q tensors must be on the same CUDA device");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3, "Q must be a rank-3 BF16 tensor");
  TORCH_CHECK(q.size(0) == ref.size(0) && q.size(1) == ref.size(1) && q.size(2) == dim, "invalid Q shape");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0, "Q strides must permit 16-byte loads");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "Q must be 16-byte aligned");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
flashmla_token_pack_prepare(const at::Tensor& q, const std::optional<at::Tensor>& q_rope, const at::Tensor& indices) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3, "Q must be a rank-3 CUDA tensor");
  check_q(q, q_rope.has_value() ? 512 : 576, q);
  if (q_rope.has_value()) check_q(*q_rope, 64, q);
  TORCH_CHECK(indices.is_cuda() && indices.device() == q.device(), "indices must be on Q's CUDA device");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 2, "indices must be a rank-2 int32 tensor");
  TORCH_CHECK(indices.size(0) >= q.size(0) && indices.stride(1) == 1, "invalid indices rows or inner stride");
  TORCH_CHECK(indices.size(1) > 0 && indices.size(1) <= 2048, "top-k must be in [1, 2048]");
  TORCH_CHECK(q.size(1) > 0 && q.size(1) <= 32, "head count must be in [1, 32]");
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int>::max() - 5, "too many tokens");
  const c10::cuda::CUDAGuard device_guard(q.device());
  const int tokens = q.size(0), heads = q.size(1), topk = indices.size(1);
  const int group = std::min(64 / heads, 5), groups = (tokens + group - 1) / group;
  const int width = ((group * topk + 63) / 64) * 64;
  const bool use_view = !q_rope.has_value() && q.is_contiguous() && tokens % group == 0;
  auto packed_q =
      use_view ? q.view({groups, 1, group * heads, 576}) : at::empty({groups, 1, group * heads, 576}, q.options());
  auto packed_indices = at::empty({groups, width}, indices.options());
  auto lengths = at::empty({groups}, indices.options());
  if (tokens != 0) {
    // Set this on the guarded device, not through a process-global static flag.
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kTable * sizeof(int)));
    prepare_kernel<<<groups, kThreads, kTable * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(q.data_ptr()),
        q_rope.has_value() ? reinterpret_cast<const uint4*>(q_rope->data_ptr()) : nullptr,
        q.stride(0) / 8,
        q.stride(1) / 8,
        q_rope.has_value() ? q_rope->stride(0) / 8 : 0,
        q_rope.has_value() ? q_rope->stride(1) / 8 : 0,
        use_view ? nullptr : reinterpret_cast<uint4*>(packed_q.data_ptr()),
        indices.data_ptr<int>(),
        indices.stride(0),
        tokens,
        heads,
        topk,
        group,
        width,
        packed_indices.data_ptr<int>(),
        lengths.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed_q, packed_indices, lengths};
}
