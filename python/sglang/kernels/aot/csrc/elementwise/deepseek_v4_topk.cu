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

// seq_lens[b] must not exceed scores.size(1) or page_table.size(1) << page_bits: a row reads up to its length
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
  // Emit each row's picks in ascending order (see bitonic_sort_u32).
  bool sort_output;
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
  uint32_t remain_topk = topk;
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
            output[topk - pos] = static_cast<int32_t>(idx);
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

// Bitonic sort of n (a power of two, 64 <= n <= kMaxTopK) 32-bit keys, one per thread: strides
// below the wavefront width exchange through lane shuffles, the wider ones through LDS.

// lane ^ J's value in registers: DPP for J <= 8, gfx950 permlane swaps for J = 16, 32; __shfl_xor
// is an LDS round trip on every stage's dependent chain.
template <uint32_t J>
__device__ __forceinline__ uint32_t lane_xor(uint32_t v) {
#if defined(__HIP_PLATFORM_AMD__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  if constexpr (J == 1) {  // quad_perm [1, 0, 3, 2]
    return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0xB1, 0xF, 0xF, true));
  }
  if constexpr (J == 2) {  // quad_perm [2, 3, 0, 1]
    return static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0x4E, 0xF, 0xF, true));
  }
  if constexpr (J == 4 || J == 8) {
    // within a 16-lane row: bit clear reads J lanes up (row_shl), bit set J lanes down (row_shr)
    const uint32_t shl =
        static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0x100 | J, 0xF, 0xF, true));
    const uint32_t shr =
        static_cast<uint32_t>(__builtin_amdgcn_update_dpp(0, static_cast<int>(v), 0x110 | J, 0xF, 0xF, true));
    return (__lane_id() & J) ? shr : shl;
  }
#endif
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx950__)
  if constexpr (J == 16) {
    // rows 1 and 3 of the first operand swap with rows 0 and 2 of the second
    const auto pair = __builtin_amdgcn_permlane16_swap(v, v, false, false);
    return (__lane_id() & 16) ? pair[0] : pair[1];
  }
  if constexpr (J == 32) {
    const auto pair = __builtin_amdgcn_permlane32_swap(v, v, false, false);
    return (__lane_id() & 32) ? pair[0] : pair[1];
  }
#endif
  return static_cast<uint32_t>(__shfl_xor(static_cast<int>(v), static_cast<int>(J), 64));
}

// One bitonic stage (merge size K, stride J); templated so the network unrolls with no stride switch.
template <uint32_t K, uint32_t J>
__device__ __forceinline__ uint32_t bitonic_stage(uint32_t v, uint32_t* __restrict__ s_vals, uint32_t tx, uint32_t n) {
  const bool up = (tx & K) == 0;
  const bool lower = (tx & J) == 0;
  uint32_t w;
  if constexpr (J >= 64) {
    __syncthreads();
    if (tx < n) s_vals[tx] = v;
    __syncthreads();
    w = tx < n ? s_vals[tx ^ J] : ~0u;
  } else {
    w = lane_xor<J>(v);
  }
  return (lower == up) ? min(v, w) : max(v, w);
}

template <uint32_t N, uint32_t K, uint32_t J>
__device__ __forceinline__ uint32_t bitonic_network(uint32_t v, uint32_t* __restrict__ s_vals, uint32_t tx) {
  v = bitonic_stage<K, J>(v, s_vals, tx, N);
  if constexpr (J > 1) {
    return bitonic_network<N, K, J / 2>(v, s_vals, tx);
  } else if constexpr (K < N) {
    return bitonic_network<N, K * 2, K>(v, s_vals, tx);
  } else {
    return v;
  }
}

// Sort the N (a power of two, 64 <= N <= kBlockSize) values in s_vals ascending, one per thread.
template <uint32_t N>
__device__ void bitonic_sort_fixed(uint32_t* __restrict__ s_vals) {
  static_assert(N >= 64 && N <= kBlockSize && (N & (N - 1)) == 0, "one value per thread");
  const uint32_t tx = threadIdx.x;
  uint32_t v = tx < N ? s_vals[tx] : ~0u;
  v = bitonic_network<N, 2, 1>(v, s_vals, tx);
  __syncthreads();
  if (tx < N) s_vals[tx] = v;
  __syncthreads();
}

__device__ void bitonic_sort_u32(uint32_t* __restrict__ s_vals, uint32_t n) {
  const uint32_t tx = threadIdx.x;
  if (n <= kBlockSize) {
    switch (n) {
      case 64:
        bitonic_sort_fixed<64>(s_vals);
        return;
      case 128:
        bitonic_sort_fixed<128>(s_vals);
        return;
      case 256:
        bitonic_sort_fixed<256>(s_vals);
        return;
      case 512:
        bitonic_sort_fixed<512>(s_vals);
        return;
      default:
        if constexpr (kBlockSize >= 1024) {
          bitonic_sort_fixed<1024>(s_vals);
          return;
        }
        break;
    }
  }
  // more values than threads (a smaller block than kMaxTopK): every stage through LDS
  for (uint32_t k = 2; k <= n; k <<= 1) {
    for (uint32_t j = k >> 1; j > 0; j >>= 1) {
      for (uint32_t i = tx; i < n; i += kBlockSize) {
        const uint32_t partner = i ^ j;
        if (partner > i) {
          const uint32_t a = s_vals[i];
          const uint32_t b = s_vals[partner];
          const bool up = (i & k) == 0;
          if ((a > b) == up) {
            s_vals[i] = b;
            s_vals[partner] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ uint32_t next_pow2_at_least_64(uint32_t x) {
  uint32_t n = 64;
  while (n < x)
    n <<= 1;
  return n;
}

__global__ __launch_bounds__(kBlockSize) void deepseek_v4_topk_transform_kernel(const TopKParams params) {
  const auto bid = blockIdx.x;
  const auto seq_len = params.seq_lens[bid];
  const auto topk = params.topk;
  const auto score_ptr = params.scores + bid * params.score_stride;
  const auto page_ptr = params.page_table + bid * params.page_table_stride;
  const auto indices_ptr = params.page_indices + bid * params.output_stride;
  const auto raw_indices_ptr =
      params.raw_indices != nullptr ? params.raw_indices + bid * params.output_stride : nullptr;

  __shared__ int32_t s_topk_indices[kMaxTopK];
  __shared__ uint32_t s_sort_vals[kMaxTopK];

  // key: the position when the row has raw indices, else the slot (sort_selection_rows' order)
  const bool key_is_position = raw_indices_ptr != nullptr;
  uint32_t count = topk;
  if (seq_len <= static_cast<int32_t>(topk)) {
    if (!params.sort_output || key_is_position) {
      // ascending positions with the -1 padding last: already the sorted row
      naive_paged_transform(seq_len, topk, params.page_bits, page_ptr, indices_ptr, raw_indices_ptr);
      return;
    }
    // every position is a pick; the slot order still has to be established
    count = static_cast<uint32_t>(seq_len);
    for (uint32_t i = threadIdx.x; i < count; i += kBlockSize) {
      s_topk_indices[i] = static_cast<int32_t>(i);
    }
  } else {
    radix_topk(score_ptr, s_topk_indices, static_cast<uint32_t>(seq_len), topk);
  }
  __syncthreads();

  if (params.sort_output) {
    const uint32_t n = next_pow2_at_least_64(count);
    for (uint32_t i = threadIdx.x; i < n; i += kBlockSize) {
      uint32_t key = ~0u;
      if (i < count) {
        const int32_t raw = s_topk_indices[i];
        key = static_cast<uint32_t>(
            key_is_position ? raw : page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits));
      }
      s_sort_vals[i] = key;
    }
    __syncthreads();
    bitonic_sort_u32(s_sort_vals, n);
    for (uint32_t i = threadIdx.x; i < topk; i += kBlockSize) {
      int32_t slot = -1;
      int32_t raw = -1;
      if (i < count) {
        const int32_t key = static_cast<int32_t>(s_sort_vals[i]);
        if (key_is_position) {
          raw = key;
          slot = page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
        } else {
          slot = key;
        }
      }
      indices_ptr[i] = slot;
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = raw;
      }
    }
    return;
  }

  for (uint32_t i = threadIdx.x; i < topk; i += kBlockSize) {
    const auto raw = s_topk_indices[i];
    indices_ptr[i] = page_to_slot(page_ptr, static_cast<uint32_t>(raw), params.page_bits);
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
    std::optional<at::Tensor> raw_indices_opt,
    bool sort_output) {
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
      .sort_output = sort_output,
  };

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<uint32_t>(B));
  const dim3 block(kBlockSize);

  setup_kernel_smem_once<deepseek_v4_topk_transform_kernel, kSMEM>();
  deepseek_v4_topk_transform_kernel<<<grid, block, kSMEM, stream>>>(params);

  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "deepseek_v4_topk_transform kernel launch failed: ", ::cudaGetErrorString(err));
}
