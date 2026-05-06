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

namespace {

constexpr uint32_t kTopK = 512;
constexpr uint32_t kBlockSize = 512;
static_assert(kTopK <= kBlockSize, "kTopK must be <= kBlockSize for the final scatter loop.");

#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSMEM = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSMEM = 48 * 1024;  // bytes
#endif
static_assert(kSMEM % (2 * sizeof(int32_t)) == 0, "kSMEM must be a multiple of 8 bytes.");

struct TopK512Params {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t page_bits;
};

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int32_t
page_to_slot(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | static_cast<int32_t>(i & mask);
}

__device__ void naive_paged_transform(
    int32_t length,
    uint32_t page_bits,
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices_out,
    int32_t* __restrict__ raw_indices_out) {
  const uint32_t tx = threadIdx.x;
  if (tx < static_cast<uint32_t>(length)) {
    page_indices_out[tx] = page_to_slot(page_table, tx, page_bits);
    if (raw_indices_out != nullptr) {
      raw_indices_out[tx] = static_cast<int32_t>(tx);
    }
  } else if (tx < kTopK) {
    page_indices_out[tx] = -1;
    if (raw_indices_out != nullptr) {
      raw_indices_out[tx] = -1;
    }
  }
}

__device__ void radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length) {
  constexpr uint32_t RADIX = 256;
  constexpr uint32_t BLOCK_SIZE = kBlockSize;
  constexpr uint32_t SMEM_INPUT_SIZE = kSMEM / (2 * sizeof(int32_t));

  alignas(128) __shared__ uint32_t _s_histogram_buf[2][RADIX + 32];
  alignas(128) __shared__ uint32_t s_counter;
  alignas(128) __shared__ uint32_t s_threshold_bin_id;
  alignas(128) __shared__ uint32_t s_num_input[2];
  alignas(128) __shared__ int32_t s_last_remain;

  extern __shared__ uint32_t s_input_idx[][SMEM_INPUT_SIZE];

  const uint32_t tx = threadIdx.x;
  uint32_t remain_topk = kTopK;
  auto& s_histogram = _s_histogram_buf[0];

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = _s_histogram_buf[k][tx];
        if (tx + j < RADIX) {
          value += _s_histogram_buf[k][tx + j];
        }
        _s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();
  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  {
    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];
    if (remain_topk == 0) {
      for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
        const uint32_t bin = convert_to_uint8(input[idx]);
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        }
      }
      __syncthreads();
      return;
    }
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw_input = input[idx];
      const uint32_t bin = convert_to_uint8(raw_input);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = static_cast<int32_t>(idx);
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin32 = convert_to_uint32(raw_input);
          const auto sub_bin = (bin32 >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const auto r_idx = round % 2;

    const auto raw_num_input = s_num_input[r_idx];
    const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = static_cast<int32_t>(remain_topk - s_histogram[tx + 1]);
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[threshold_bin + 1];

    if (remain_topk == 0) {
      for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          output[pos] = static_cast<int32_t>(idx);
        }
      }
      __syncthreads();
      break;
    }
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < num_input; i += BLOCK_SIZE) {
      const auto idx = s_input_idx[r_idx][i];
      const auto raw_input = input[idx];
      const auto offset = 24 - round * 8;
      const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        output[pos] = static_cast<int32_t>(idx);
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const auto pos = ::atomicAdd(&s_last_remain, -1);
          if (pos > 0) {
            output[kTopK - pos] = static_cast<int32_t>(idx);
          }
        } else {
          const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
          if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
            s_input_idx[r_idx ^ 1][pos] = idx;
            const auto bin32 = convert_to_uint32(raw_input);
            const auto sub_bin = (bin32 >> (offset - 8)) & 0xFF;
            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kBlockSize) void deepseek_v4_topk_transform_512_kernel(const TopK512Params params) {
  const auto bid = blockIdx.x;
  const auto seq_len = params.seq_lens[bid];
  const auto score_ptr = params.scores + bid * params.score_stride;
  const auto page_ptr = params.page_table + bid * params.page_table_stride;
  const auto indices_ptr = params.page_indices + bid * kTopK;
  const auto raw_indices_ptr = params.raw_indices != nullptr ? params.raw_indices + bid * kTopK : nullptr;

  if (seq_len <= static_cast<int32_t>(kTopK)) {
    naive_paged_transform(seq_len, params.page_bits, page_ptr, indices_ptr, raw_indices_ptr);
    return;
  }

  __shared__ int32_t s_topk_indices[kTopK];
  radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len));

  __syncthreads();
  const auto tx = threadIdx.x;
  if (tx < kTopK) {
    const auto raw = s_topk_indices[tx];
    indices_ptr[tx] = page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
    if (raw_indices_ptr != nullptr) {
      raw_indices_ptr[tx] = raw;
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
      result == cudaSuccess,
      "deepseek_v4_topk_transform_512: cudaFuncSetAttribute failed: ",
      ::cudaGetErrorString(result));
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

  TORCH_CHECK(
      page_indices.dim() == 2 && page_indices.is_contiguous() && page_indices.scalar_type() == at::kInt,
      "page_indices must be int32 with shape [B, ",
      kTopK,
      "], contiguous");
  TORCH_CHECK(
      page_indices.size(1) == static_cast<int64_t>(kTopK),
      "page_indices last dim must be ",
      kTopK,
      ", got ",
      page_indices.size(1));

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
        "raw_indices must be int32 with shape [B, ",
        kTopK,
        "], contiguous");
    TORCH_CHECK(
        raw.size(0) == B && raw.size(1) == static_cast<int64_t>(kTopK), "raw_indices shape must be [B, ", kTopK, "]");
    raw_ptr = raw.data_ptr<int32_t>();
  }

  const TopK512Params params{
      .scores = scores.data_ptr<float>(),
      .seq_lens = seq_lens.data_ptr<int32_t>(),
      .page_table = page_table.data_ptr<int32_t>(),
      .page_indices = page_indices.data_ptr<int32_t>(),
      .raw_indices = raw_ptr,
      .score_stride = scores.stride(0),
      .page_table_stride = page_table.stride(0),
      .page_bits = page_bits,
  };

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<uint32_t>(B));
  const dim3 block(kBlockSize);

  setup_kernel_smem_once<deepseek_v4_topk_transform_512_kernel, kSMEM>();
  deepseek_v4_topk_transform_512_kernel<<<grid, block, kSMEM, stream>>>(params);

  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "deepseek_v4_topk_transform_512 kernel launch failed: ", ::cudaGetErrorString(err));
}
