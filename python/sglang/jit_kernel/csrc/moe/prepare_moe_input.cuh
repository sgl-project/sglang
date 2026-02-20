// Adapt from sgl-kernel/csrc/moe/prepare_moe_input.cu
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>

using tvm::ffi::TensorView;

namespace {

constexpr uint64_t kThreadsPerExpert = 512;

// ---------------------------------------------------------------------------
// prepare_moe_input kernels — compute problem sizes and sorted permutations
// ---------------------------------------------------------------------------

__global__ void compute_problem_sizes_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t* __restrict__ atomic_buffer,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {
  int expert_id = blockIdx.x;
  int occurrences = 0;
  for (int64_t i = threadIdx.x; i < topk_length; i += kThreadsPerExpert) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = static_cast<int32_t>(2 * n);
    problem_sizes1[expert_id * 3 + 2] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3 + 2] = static_cast<int32_t>(n);
  }
}

__global__ void compute_expert_offsets_kernel(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets_kernel(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ blockscale_offsets,
    int32_t* __restrict__ atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  int32_t tot_rounded_offset = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int64_t i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    int num_tokens = problem_sizes1[i * 3];
    int rounded_num_tokens = (num_tokens + 127) / 128 * 128;
    tot_offset += num_tokens;
    tot_rounded_offset += rounded_num_tokens;
    expert_offsets[i + 1] = tot_offset;
    blockscale_offsets[i + 1] = tot_rounded_offset;
  }
}

__global__ void compute_arg_sorts_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    int32_t* __restrict__ atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  int expert_id = blockIdx.x;
  for (int64_t i = threadIdx.x; i < topk_length; i += kThreadsPerExpert) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

// ---------------------------------------------------------------------------
// shuffle_rows kernel — dtype-agnostic 128-bit vectorized row copy
// ---------------------------------------------------------------------------

__global__ void shuffle_rows_kernel(
    const void* __restrict__ input,
    const int32_t* __restrict__ dst2src_map,
    void* __restrict__ output,
    int64_t num_dst_rows,
    int64_t row_bytes) {
  int64_t dst_row = static_cast<int64_t>(blockIdx.x);
  if (dst_row >= num_dst_rows) return;

  int64_t src_row = dst2src_map[dst_row];
  const char* src_ptr = static_cast<const char*>(input) + src_row * row_bytes;
  char* dst_ptr = static_cast<char*>(output) + dst_row * row_bytes;

  // 128-bit (16-byte) vectorized copy
  int64_t num_vecs = row_bytes / 16;
  const uint4* src_vec = reinterpret_cast<const uint4*>(src_ptr);
  uint4* dst_vec = reinterpret_cast<uint4*>(dst_ptr);
  for (int64_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
  // scalar remainder
  int64_t rem_start = num_vecs * 16;
  for (int64_t b = rem_start + threadIdx.x; b < row_bytes; b += blockDim.x) {
    dst_ptr[b] = src_ptr[b];
  }
}

// ---------------------------------------------------------------------------
// apply_shuffle_mul_sum kernel — gather, scale, and reduce over topk
// Accumulates in float32 regardless of input dtype.
// ---------------------------------------------------------------------------

template <typename scalar_t>
struct FloatVec {
  static constexpr uint32_t N = 16u / sizeof(scalar_t);
  float data[N];

  __device__ void fill(float v) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) data[i] = v;
  }

  __device__ void cast_load(const scalar_t* ptr) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      if constexpr (std::is_same_v<scalar_t, fp16_t>)
        data[i] = __half2float(ptr[i]);
      else if constexpr (std::is_same_v<scalar_t, bf16_t>)
        data[i] = __bfloat162float(ptr[i]);
      else
        data[i] = static_cast<float>(ptr[i]);
    }
  }

  __device__ void cast_store(scalar_t* ptr) const {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) {
      if constexpr (std::is_same_v<scalar_t, fp16_t>)
        ptr[i] = __float2half(data[i]);
      else if constexpr (std::is_same_v<scalar_t, bf16_t>)
        ptr[i] = __float2bfloat16(data[i]);
      else
        ptr[i] = static_cast<scalar_t>(data[i]);
    }
  }

  __device__ float& operator[](uint32_t i) { return data[i]; }
  __device__ const float& operator[](uint32_t i) const { return data[i]; }
};

template <typename scalar_t>
__global__ void apply_shuffle_mul_sum_kernel(
    const scalar_t* __restrict__ input,   // [m * topk, row_stride]
    scalar_t* __restrict__ output,        // [m, row_stride]
    const int32_t* __restrict__ permutation,  // [m * topk]
    int m,
    int topk,
    int row_stride,
    const scalar_t* __restrict__ factors)  // [m * topk], or nullptr
{
  int i = blockIdx.x;
  if (i >= m) return;

  using vec_t = FloatVec<scalar_t>;
  constexpr uint32_t vec_size = vec_t::N;

  int thread_idx = threadIdx.x;
  int stride = blockDim.x;

  // Vectorized part
  for (int d_vec = thread_idx; d_vec < row_stride / (int)vec_size; d_vec += stride) {
    int d = d_vec * (int)vec_size;
    vec_t sum_vec;
    sum_vec.fill(0.0f);

    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];

      vec_t val_vec;
      val_vec.cast_load(input + src_row * row_stride + d);

      float factor = (factors != nullptr) ? static_cast<float>(factors[token_major_idx]) : 1.0f;
#pragma unroll
      for (uint32_t k = 0; k < vec_size; ++k) {
        sum_vec[k] += factor * val_vec[k];
      }
    }
    sum_vec.cast_store(output + i * row_stride + d);
  }

  // Scalar remainder
  int rem_start = (row_stride / (int)vec_size) * (int)vec_size;
  for (int d = rem_start + thread_idx; d < row_stride; d += stride) {
    float sum_val = 0.0f;
    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];
      float val = static_cast<float>(input[src_row * row_stride + d]);
      float factor = (factors != nullptr) ? static_cast<float>(factors[token_major_idx]) : 1.0f;
      sum_val += factor * val;
    }
    if constexpr (std::is_same_v<scalar_t, fp16_t>)
      output[i * row_stride + d] = __float2half(sum_val);
    else if constexpr (std::is_same_v<scalar_t, bf16_t>)
      output[i * row_stride + d] = __float2bfloat16(sum_val);
    else
      output[i * row_stride + d] = static_cast<scalar_t>(sum_val);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launchers (tvm-ffi interface)
// ---------------------------------------------------------------------------

void prepare_moe_input(
    TensorView topk_ids,
    TensorView expert_offsets,
    tvm::ffi::Optional<TensorView> blockscale_offsets,
    TensorView problem_sizes1,
    TensorView problem_sizes2,
    TensorView input_permutation,
    TensorView output_permutation,
    TensorView atomic_buffer,
    int64_t num_experts,
    int64_t n,
    int64_t k) {
  using namespace host;

  RuntimeCheck(topk_ids.dtype().code == kDLInt && topk_ids.dtype().bits == 32, "topk_ids must be int32");
  RuntimeCheck(num_experts > 0, "num_experts must be positive");

  const int64_t topk_length = [&] {
    int64_t total = 1;
    for (int d = 0; d < topk_ids.dim(); ++d) total *= topk_ids.size(d);
    return total;
  }();

  const int64_t topk = (topk_ids.dim() >= 2) ? topk_ids.size(1) : 1;

  const int32_t* topk_ids_ptr = static_cast<const int32_t*>(topk_ids.data_ptr());
  int32_t* expert_offsets_ptr = static_cast<int32_t*>(expert_offsets.data_ptr());
  int32_t* problem_sizes1_ptr = static_cast<int32_t*>(problem_sizes1.data_ptr());
  int32_t* problem_sizes2_ptr = static_cast<int32_t*>(problem_sizes2.data_ptr());
  int32_t* input_perm_ptr = static_cast<int32_t*>(input_permutation.data_ptr());
  int32_t* output_perm_ptr = static_cast<int32_t*>(output_permutation.data_ptr());
  int32_t* atomic_ptr = static_cast<int32_t*>(atomic_buffer.data_ptr());

  cudaStream_t stream = LaunchKernel::resolve_device(topk_ids.device());

  uint32_t num_threads = static_cast<uint32_t>(std::min((int64_t)kThreadsPerExpert, topk_length));
  uint32_t num_blocks = static_cast<uint32_t>(num_experts);

  compute_problem_sizes_kernel<<<num_blocks, num_threads, 0, stream>>>(
      topk_ids_ptr, problem_sizes1_ptr, problem_sizes2_ptr, atomic_ptr, topk_length, n, k);

  if (blockscale_offsets.has_value()) {
    int32_t* bs_offsets_ptr = static_cast<int32_t*>(blockscale_offsets.value().data_ptr());
    compute_expert_blockscale_offsets_kernel<<<1, 1, 0, stream>>>(
        problem_sizes1_ptr, expert_offsets_ptr, bs_offsets_ptr, atomic_ptr, num_experts);
  } else {
    compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(
        problem_sizes1_ptr, expert_offsets_ptr, atomic_ptr, num_experts);
  }

  compute_arg_sorts_kernel<<<num_blocks, num_threads, 0, stream>>>(
      topk_ids_ptr, input_perm_ptr, output_perm_ptr, atomic_ptr, topk_length, topk);
}

void shuffle_rows(TensorView input, TensorView dst2src_map, TensorView output) {
  using namespace host;

  RuntimeCheck(input.dim() == 2, "input must be 2-D [num_src_rows, num_cols]");
  RuntimeCheck(output.dim() == 2, "output must be 2-D [num_dst_rows, num_cols]");
  RuntimeCheck(dst2src_map.dim() == 1, "dst2src_map must be 1-D");
  RuntimeCheck(input.size(1) == output.size(1), "num_cols mismatch between input and output");
  RuntimeCheck(output.size(0) == dst2src_map.size(0), "num_dst_rows mismatch");

  const int64_t num_dst_rows = output.size(0);
  const int64_t num_cols = input.size(1);

  // Compute element byte size from dtype
  const int64_t elem_bytes = static_cast<int64_t>(input.dtype().bits) / 8;
  const int64_t row_bytes = num_cols * elem_bytes;

  RuntimeCheck(row_bytes % 16 == 0, "row_bytes must be a multiple of 16 for vectorized copy");

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  dim3 grid(static_cast<unsigned>(num_dst_rows));
  dim3 block(256);

  shuffle_rows_kernel<<<grid, block, 0, stream>>>(
      input.data_ptr(),
      static_cast<const int32_t*>(dst2src_map.data_ptr()),
      output.data_ptr(),
      num_dst_rows,
      row_bytes);
}

template <typename scalar_t>
static void launch_apply_shuffle_mul_sum(
    TensorView input,
    TensorView output,
    TensorView permutation,
    tvm::ffi::Optional<TensorView> factors) {
  using namespace host;
  const int m = static_cast<int>(output.size(0));
  const int topk_times_m = static_cast<int>(input.size(0));
  const int topk = (m > 0) ? topk_times_m / m : 1;
  const int row_stride = static_cast<int>(output.size(1));

  constexpr uint32_t vec_size = 16u / sizeof(scalar_t);
  const uint32_t block_threads = static_cast<uint32_t>(
      std::min(static_cast<int>(row_stride / (int)vec_size), 1024));

  const scalar_t* factors_ptr = factors.has_value()
      ? static_cast<const scalar_t*>(factors.value().data_ptr())
      : nullptr;

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  dim3 grid(static_cast<unsigned>(m));
  dim3 block(block_threads > 0 ? block_threads : 1u);

  apply_shuffle_mul_sum_kernel<scalar_t><<<grid, block, 0, stream>>>(
      static_cast<const scalar_t*>(input.data_ptr()),
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<const int32_t*>(permutation.data_ptr()),
      m,
      topk,
      row_stride,
      factors_ptr);
}

void apply_shuffle_mul_sum(
    TensorView input,
    TensorView output,
    TensorView permutation,
    tvm::ffi::Optional<TensorView> factors) {
  using namespace host;

  RuntimeCheck(input.dim() == 2, "input must be 2-D [m * topk, row_stride]");
  RuntimeCheck(output.dim() == 2, "output must be 2-D [m, row_stride]");
  RuntimeCheck(permutation.dim() == 1, "permutation must be 1-D [m * topk]");
  RuntimeCheck(input.size(1) == output.size(1), "row_stride mismatch");

  auto dtype = output.dtype();
  if (host::is_type<fp16_t>(dtype))
    launch_apply_shuffle_mul_sum<fp16_t>(input, output, permutation, factors);
  else if (host::is_type<bf16_t>(dtype))
    launch_apply_shuffle_mul_sum<bf16_t>(input, output, permutation, factors);
  else if (host::is_type<fp32_t>(dtype))
    launch_apply_shuffle_mul_sum<fp32_t>(input, output, permutation, factors);
  else
    RuntimeCheck(false, "apply_shuffle_mul_sum: unsupported dtype (expected fp16, bf16, or fp32)");
}
