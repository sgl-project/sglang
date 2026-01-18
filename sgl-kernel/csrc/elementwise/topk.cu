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

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 512;

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

__device__ __forceinline__ auto pack_float_uint32(float x, int y) -> uint64_t {
  uint32_t bits = __float_as_uint(x);
  bits = (bits >= 0x80000000u) ? (bits - 1) : (bits ^ 0x7fffffffu);
  return uint64_t(bits) << 32 | y;
}

template <typename T, int kThreadsPerBlock, int kWarpSize>
struct bitonic_sort {
  T (&buffer)[TopK];
  static_assert(
      kThreadsPerBlock * kThreadsPerBlock >= kWarpSize * TopK,
      "assert failed: kThreadsPerBlock * kThreadsPerBlock >= kWarpSize * TopK");
  static constexpr int kValueSize = TopK / kThreadsPerBlock;
  static constexpr int kSwizzleSize = kWarpSize * kValueSize;
  static constexpr int kSwizzleNum = kThreadsPerBlock / kSwizzleSize;
  const int swizzleIdx = threadIdx.x % kSwizzleNum * kSwizzleSize * kValueSize | threadIdx.x / kSwizzleNum;
  const int originIdx = threadIdx.x * kValueSize;

  __device__ __forceinline__ void swizzle(T (&value)[kValueSize]) {
#pragma unroll
    for (int i = 0; i < kValueSize; ++i)
      buffer[originIdx | i] = value[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kValueSize; ++i)
      value[i] = buffer[swizzleIdx | kSwizzleSize * i];
  }

  __device__ __forceinline__ void restore(T (&value)[kValueSize]) {
#pragma unroll
    for (int i = 0; i < kValueSize; ++i)
      buffer[swizzleIdx | i * kSwizzleSize] = value[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kValueSize; ++i)
      value[i] = buffer[originIdx | i];
  }

  __device__ __forceinline__ static void sortpair(T& x, T& y, bool descend) {
    auto tmp = x;
    const bool swap = descend ^ (tmp > y);
    x = swap ? y : tmp;
    y = swap ? tmp : y;
  }

  __device__ __forceinline__ static void merge(T (&value)[kValueSize], int step, const bool descend) {
#pragma unroll
    for (int j = step; j > 0; j >>= 1) {
      const bool rev = descend ^ bool(threadIdx.x & j);
#pragma unroll
      for (int k = 0; k < kValueSize; ++k) {
        auto x = __shfl_xor_sync(0xFFFFFFFF, value[k], j);
        value[k] = (value[k] <= x) ^ rev ? value[k] : x;
      }
    }

#pragma unroll
    for (int j = kValueSize >> 1; j > 0; j >>= 1)
#pragma unroll
      for (int k = 0; k < kValueSize >> 1; ++k)
        sortpair(value[k + (k & -j)], value[k + (k & -j) | j], descend);
  }

  __device__ __forceinline__ void merge2048(T (&value)[kValueSize], const bool descend) {
    swizzle(value);
    merge(value, kSwizzleNum >> 1, descend);
    restore(value);
    merge(value, kWarpSize >> 1, descend);
  }

  __device__ __forceinline__ void sort2048(T (&value)[kValueSize], const bool descend) {
#pragma unroll
    for (int i = 1; i < kValueSize >> 1; i <<= 1)
#pragma unroll
      for (int j = i; j > 0; j >>= 1)
#pragma unroll
        for (int k = 0; k < kValueSize >> 1; ++k)
          sortpair(value[k + (k & -j)], value[k + (k & -j) | j], i & k);
#pragma unroll
    for (int i = 1; i <= kWarpSize; i <<= 1)
      merge(value, i >> 1, bool(threadIdx.x & i));
#pragma unroll
    for (int i = 1; i < kValueSize >> 1; i <<= 1) {
      swizzle(value);
#pragma unroll
      for (int j = i; j > 0; j >>= 1)
#pragma unroll
        for (int k = 0; k < kValueSize >> 1; ++k)
          sortpair(value[k + (k & -j)], value[k + (k & -j) | j], i & k);
      restore(value);
      merge(value, kWarpSize >> 1, bool(threadIdx.x & i * kWarpSize * 2));
    }
#pragma unroll
    for (int i = 1; i < kSwizzleNum; i <<= 1) {
      swizzle(value);
      merge(value, i >> 1, bool(threadIdx.x & i));
      restore(value);
      merge(value, kWarpSize >> 1, bool(threadIdx.x & i * kSwizzleSize));
    }
    merge2048(value, descend);
  }
};

using bitonic_sort_t = bitonic_sort<uint64_t, kThreadsPerBlock, 32>;

__device__ void fast_topk_cuda_tl(
    const float* __restrict__ input, int (&index)[bitonic_sort_t::kValueSize], int row_start, int length) {
  __shared__ uint64_t buffer[TopK];

  bitonic_sort_t sorter = {buffer};
  uint64_t result[bitonic_sort_t::kValueSize];
#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i)
    result[i] = uint64_t(-1);

  for (int k = 0; k < (length + TopK - 1) / TopK; ++k) {
    int idx = k * TopK + threadIdx.x * bitonic_sort_t::kValueSize;
    uint64_t value[bitonic_sort_t::kValueSize];
#pragma unroll
    for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
      int j = idx + i;
      value[i] = j < length ? pack_float_uint32(input[j + row_start], j) : uint64_t(-1);
    }
    sorter.sort2048(value, true);
#pragma unroll
    for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
      result[i] = min(result[i], value[i]);
    }
    sorter.merge2048(result, false);
  }

#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
    index[i] = int(result[i]);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  int index[bitonic_sort_t::kValueSize];
  fast_topk_cuda_tl(score, index, row_start, length);
#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
    indice[threadIdx.x * bitonic_sort_t::kValueSize + i] = index[i];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  int index[bitonic_sort_t::kValueSize];
  fast_topk_cuda_tl(score, index, row_start, length);
#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
    dst_page_entry[threadIdx.x * bitonic_sort_t::kValueSize + i] = index[i] != -1 ? src_page_entry[index[i]] : -1;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  int index[bitonic_sort_t::kValueSize];
  fast_topk_cuda_tl(score, index, row_start, length);
#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
    dst_page_entry[threadIdx.x * bitonic_sort_t::kValueSize + i] = index[i] != -1 ? src_page_entry[index[i]] : -1;
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  int index[bitonic_sort_t::kValueSize];
  fast_topk_cuda_tl(score, index, row_start, length);
#pragma unroll
  for (int i = 0; i < bitonic_sort_t::kValueSize; ++i) {
    dst_indices_entry[threadIdx.x * bitonic_sort_t::kValueSize + i] = index[i] != -1 ? index[i] + offset : -1;
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  topk_kernel<<<grid, block, 0, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    topk_transform_decode_kernel<<<grid, block, 0, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    topk_transform_prefill_kernel<<<grid, block, 0, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  topk_transform_prefill_ragged_kernel<<<grid, block, 0, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
