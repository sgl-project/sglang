/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
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

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  ::sglang::device::legacy_radix_topk::select<kThreadsPerBlock, TopK, static_cast<int>(kSmem / (2 * sizeof(int)))>(
      input, index, row_start, length, TopK);
  __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
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
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = pos_0 < 0 ? -1 : src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = pos_1 < 0 ? -1 : src_page_entry[pos_1];
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

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = pos_0 < 0 ? -1 : src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = pos_1 < 0 ? -1 : src_page_entry[pos_1];
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

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 < 0 ? -1 : pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 < 0 ? -1 : pos_1 + offset;
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

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
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
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
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
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
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

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
