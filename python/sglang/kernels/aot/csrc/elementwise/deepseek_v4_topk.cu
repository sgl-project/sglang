/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sgl_kernel/dsa/legacy_radix_topk.cuh>

namespace {

constexpr uint32_t kMaxTopK = 1024;
#ifdef USE_ROCM
// CDNA3/CDNA4: this kernel is one block per row and is latency-bound on its
// O(c4_len) histogram and emit passes. A full 1024-thread block (16 wavefronts
// of 64 lanes) instead of 512 doubles the per-block scan parallelism, which is
// ~1.6x faster at 128k context (c4_len = 32768) and never slower at short
// context. The selected index set is unchanged. CUDA keeps 512.
constexpr uint32_t kBlockSize = 1024;
#else
constexpr uint32_t kBlockSize = 512;
#endif

#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSMEM = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSMEM = 48 * 1024;  // bytes
#endif
static_assert(kSMEM % (2 * sizeof(int32_t)) == 0, "kSMEM must be a multiple of 8 bytes.");

struct TopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
  uint32_t topk;
  int64_t output_stride;
};

__device__ __forceinline__ int32_t
page_to_slot(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | static_cast<int32_t>(i & mask);
}

__device__ void naive_paged_transform(
    int32_t length,
    uint32_t topk,
    uint32_t page_bits,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices_out,
    int32_t* __restrict__ raw_indices_out) {
  for (uint32_t i = threadIdx.x; i < topk; i += kBlockSize) {
    if (i < static_cast<uint32_t>(length)) {
      page_indices_out[i] = page_to_slot(page_table, i, page_bits);
      if (raw_indices_out != nullptr) {
        raw_indices_out[i] = static_cast<int32_t>(i);
      }
    } else {
      page_indices_out[i] = -1;
      if (raw_indices_out != nullptr) {
        raw_indices_out[i] = -1;
      }
    }
  }
}

__device__ void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, uint32_t length, uint32_t topk) {
  ::sglang::device::legacy_radix_topk::
      select<static_cast<int>(kBlockSize), static_cast<int>(kMaxTopK), static_cast<int>(kSMEM / (2 * sizeof(int32_t)))>(
          input, output, 0, static_cast<int>(length), static_cast<int>(topk));
  __syncthreads();
}

__global__ __launch_bounds__(kBlockSize) void deepseek_v4_topk_transform_kernel(const TopKParams params) {
  const auto bid = blockIdx.x;
  const auto raw_seq_len = params.seq_lens[bid];
  const auto seq_len = raw_seq_len < 0 ? 0 : raw_seq_len;
  const auto topk = params.topk;
  const auto score_ptr = params.scores + bid * params.score_stride;
  const auto page_ptr = params.page_table + bid * params.page_table_stride;
  const auto indices_ptr = params.page_indices + bid * params.output_stride;
  const auto raw_indices_ptr =
      params.raw_indices != nullptr ? params.raw_indices + bid * params.output_stride : nullptr;

  if (seq_len <= static_cast<int32_t>(topk)) {
    naive_paged_transform(seq_len, topk, params.page_bits, page_ptr, indices_ptr, raw_indices_ptr);
    return;
  }

  __shared__ int32_t s_topk_indices[kMaxTopK];
  radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len), topk);

  for (uint32_t i = threadIdx.x; i < topk; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    indices_ptr[i] = raw < 0 ? -1 : page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
    if (raw_indices_ptr != nullptr) {
      raw_indices_ptr[i] = raw;
    }
  }
}

template <auto* f, size_t kMaxDynamicSMEM>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
#else
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSMEM);
#endif
  }();
  TORCH_CHECK(
      result == cudaSuccess, "deepseek_v4_topk_transform: cudaFuncSetAttribute failed: ", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void deepseek_v4_topk_transform_512(
    const at::Tensor& scores,
    const at::Tensor& seq_lens,
    const at::Tensor& page_table,
    at::Tensor& page_indices,
    int64_t page_size,
    std::optional<at::Tensor> raw_indices_opt) {
  CHECK_CUDA(scores);
  CHECK_CUDA(seq_lens);
  CHECK_CUDA(page_table);
  CHECK_CUDA(page_indices);
  if (raw_indices_opt.has_value()) {
    CHECK_CUDA(raw_indices_opt.value());
  }

  TORCH_CHECK(
      scores.dim() == 2 && scores.scalar_type() == at::kFloat, "scores must be float32 with shape [B, max_seq_len]");
  TORCH_CHECK(scores.stride(1) == 1, "scores must be contiguous along the last dim");

  TORCH_CHECK(
      seq_lens.dim() == 1 && seq_lens.is_contiguous() && seq_lens.scalar_type() == at::kInt,
      "seq_lens must be int32 with shape [B], contiguous");

  TORCH_CHECK(
      page_table.dim() == 2 && page_table.scalar_type() == at::kInt,
      "page_table must be int32 with shape [B, num_pages]");
  TORCH_CHECK(page_table.stride(1) == 1, "page_table must be contiguous along the last dim");

  const auto topk = page_indices.size(1);
  TORCH_CHECK(
      page_indices.dim() == 2 && page_indices.is_contiguous() && page_indices.scalar_type() == at::kInt,
      "page_indices must be int32 with shape [B, topk], contiguous");
  TORCH_CHECK(
      topk > 0 && topk <= static_cast<int64_t>(kMaxTopK),
      "page_indices last dim must be in [1, ",
      kMaxTopK,
      "], got ",
      topk);

  const auto B = scores.size(0);
  TORCH_CHECK(
      seq_lens.size(0) == B && page_table.size(0) == B && page_indices.size(0) == B,
      "batch sizes must match across scores, seq_lens, page_table, page_indices");

  TORCH_CHECK(
      page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a positive power of 2, got ", page_size);
  const auto page_bits = static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(page_size)));

  int32_t* raw_ptr = nullptr;
  if (raw_indices_opt.has_value()) {
    auto& raw = raw_indices_opt.value();
    TORCH_CHECK(
        raw.dim() == 2 && raw.is_contiguous() && raw.scalar_type() == at::kInt,
        "raw_indices must be int32 with shape [B, topk], contiguous");
    TORCH_CHECK(raw.size(0) == B && raw.size(1) == topk, "raw_indices shape must match page_indices [B, ", topk, "]");
    raw_ptr = raw.data_ptr<int32_t>();
  }

  const TopKParams params{
      .scores = scores.data_ptr<float>(),
      .seq_lens = seq_lens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = page_indices.data_ptr<int32_t>(),
      .raw_indices = raw_ptr,
      .score_stride = scores.stride(0),
      .page_table_stride = page_table.stride(0),
      .page_bits = page_bits,
      .topk = static_cast<uint32_t>(topk),
      .output_stride = topk,
  };

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<uint32_t>(B));
  const dim3 block(kBlockSize);

  setup_kernel_smem_once<deepseek_v4_topk_transform_kernel, kSMEM>();
  deepseek_v4_topk_transform_kernel<<<grid, block, kSMEM, stream>>>(params);

  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "deepseek_v4_topk_transform kernel launch failed: ", ::cudaGetErrorString(err));
}
