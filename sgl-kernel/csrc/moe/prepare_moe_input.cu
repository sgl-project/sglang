#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <torch/all.h>

#include <iostream>

#include "cutlass/array.h"

constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(
    const int* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
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

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* blockscale_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  int32_t tot_rounded_offset = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    int num_tokens = problem_sizes1[i * 3];
    int rounded_num_tokens = (num_tokens + (128 - 1)) / 128 * 128;
    tot_offset += num_tokens;
    tot_rounded_offset += rounded_num_tokens;
    expert_offsets[i + 1] = tot_offset;
    blockscale_offsets[i + 1] = tot_rounded_offset;
  }
}

__global__ void compute_arg_sorts(
    const int32_t* __restrict__ topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  int expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

void get_moe_prepare_input_caller(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  auto stream = at::cuda::getCurrentCUDAStream(topk_ids.device().index());
  auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor atomic_buffer = torch::zeros(num_experts, options_int32);

  uint32_t num_threads = static_cast<uint32_t>(min(THREADS_PER_EXPERT, topk_ids.numel()));
  uint32_t num_blocks = static_cast<uint32_t>(num_experts);

  compute_problem_sizes<<<num_blocks, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(problem_sizes1.data_ptr()),
      static_cast<int32_t*>(problem_sizes2.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      topk_ids.numel(),
      n,
      k);
  if (blockscale_offsets.has_value()) {
    compute_expert_blockscale_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(blockscale_offsets.value().data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()),
        num_experts);
  } else {
    compute_expert_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(problem_sizes1.data_ptr()),
        static_cast<int32_t*>(expert_offsets.data_ptr()),
        static_cast<int32_t*>(atomic_buffer.data_ptr()),
        num_experts);
  }
  compute_arg_sorts<<<num_blocks, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(input_permutation.data_ptr()),
      static_cast<int32_t*>(output_permutation.data_ptr()),
      static_cast<int32_t*>(atomic_buffer.data_ptr()),
      topk_ids.numel(),
      topk_ids.size(1));
}

void prepare_moe_input(
    const torch::Tensor& topk_ids,
    torch::Tensor& expert_offsets,
    const std::optional<torch::Tensor>& blockscale_offsets,
    torch::Tensor& problem_sizes1,
    torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation,
    torch::Tensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
  get_moe_prepare_input_caller(
      topk_ids,
      expert_offsets,
      blockscale_offsets,
      problem_sizes1,
      problem_sizes2,
      input_permutation,
      output_permutation,
      num_experts,
      n,
      k);
  return;
}

template <typename T>
__global__ void shuffleRowsKernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    // Load 128-bits per thread
    constexpr uint64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    auto const start_offset = threadIdx.x;
    auto const stride = blockDim.x;
    auto const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (auto elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

#define DECLARE_SHUFFLE_ROWS(T)      \
  __global__ void shuffleRowsKernel( \
      const T* input,                \
      const int32_t* dst2src_map,    \
      T* output,                     \
      int64_t num_src_rows,          \
      int64_t num_dest_rows,         \
      int64_t num_cols);

DECLARE_SHUFFLE_ROWS(float);
DECLARE_SHUFFLE_ROWS(half);
DECLARE_SHUFFLE_ROWS(__nv_bfloat16);
DECLARE_SHUFFLE_ROWS(__nv_fp8_e4m3);
DECLARE_SHUFFLE_ROWS(uint8_t);

#define SHUFFLE_ROWS(T)                                    \
  shuffleRowsKernel<T><<<blocks, threads, 0, stream>>>(    \
      reinterpret_cast<const T*>(input),                   \
      static_cast<const int32_t*>(dst2src_map.data_ptr()), \
      reinterpret_cast<T*>(output),                        \
      num_src_rows,                                        \
      num_dst_rows,                                        \
      num_cols)

#define DTYPE_DISPATCH_CASE(T, CUDA_T) \
  case T:                              \
    SHUFFLE_ROWS(CUDA_T);              \
    break;

void shuffle_rows_caller(
    const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  TORCH_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  uint32_t blocks = static_cast<uint32_t>(output_tensor.size(0));
  uint32_t threads = 256;
  int64_t num_dst_rows = output_tensor.size(0);
  int64_t num_src_rows = input_tensor.size(0);
  int64_t num_cols = input_tensor.size(1);
  const void* input = input_tensor.data_ptr();
  void* output = output_tensor.data_ptr();
  switch (input_tensor.scalar_type()) {
    DTYPE_DISPATCH_CASE(torch::kFloat16, half);
    DTYPE_DISPATCH_CASE(torch::kBFloat16, __nv_bfloat16);
    DTYPE_DISPATCH_CASE(torch::kFloat32, float);
    DTYPE_DISPATCH_CASE(torch::kFloat8_e4m3fn, __nv_fp8_e4m3);
    DTYPE_DISPATCH_CASE(torch::kUInt8, uint8_t);
    default:
      TORCH_CHECK(false, "[moe replicate input] data type dispatch fail!");
  }
  return;
}

void shuffle_rows(const torch::Tensor& input_tensor, const torch::Tensor& dst2src_map, torch::Tensor& output_tensor) {
  shuffle_rows_caller(input_tensor, dst2src_map, output_tensor);
  return;
}

template <typename scalar_t>
__global__ void apply_shuffle_mul_sum_kernel(
    const scalar_t* __restrict__ input_tensor,  // [m * topk, row_stride]
    scalar_t* __restrict__ output_tensor,       // [m, row_stride]
    const int32_t* __restrict__ permutation,    // [m * topk]
    int m,
    int topk,
    int row_stride,
    const scalar_t* __restrict__ factors)  // [m * topk] or nullptr
{
  int i = blockIdx.x;   // [0, m * topk)
  int d = threadIdx.x;  // [0, row_stride)

  if (i >= m || d >= row_stride) return;

  scalar_t sum_val = 0.0;

  for (int j = 0; j < topk; ++j) {
    int index_2d = i * topk + j;
    int src_row = permutation[index_2d];
    if (src_row >= m) continue;

    scalar_t val = input_tensor[src_row * row_stride + d];

    scalar_t factor = 1.0;
    if (factors != nullptr) {
      factor = factors[index_2d];
    }

    sum_val += factor * val;
  }

  output_tensor[i * row_stride + d] = sum_val;
}

void get_apply_shuffle_mul_sum_caller(
    const torch::Tensor& input_tensor,                // [m * topk, row_stride], bf16/f16
    torch::Tensor& output_tensor,                     // [m, row_stride], bf16/f16
    const torch::Tensor& permutation,                 // [m * topk], int32
    const std::optional<torch::Tensor>& factors_opt)  // optional [m * topk], bf16/f16
{
  TORCH_CHECK(input_tensor.dim() == 2, "input_tensor must be 2D [m * topk, row_stride]");
  TORCH_CHECK(output_tensor.dim() == 2, "output_tensor must be 2D [m, row_stride]");
  TORCH_CHECK(permutation.dim() == 1, "permutation must be 1D [m * topk]");

  int m = output_tensor.size(0);
  int topk = int(permutation.size(0) / m);
  int row_stride = output_tensor.size(1);

  TORCH_CHECK(permutation.size(0) == m * topk, "permutation size must match m * topk");

  dim3 block(std::min(256, row_stride));
  dim3 grid(m);  // blockIdx.x = j, blockIdx.y = i
  auto stream = at::cuda::getCurrentCUDAStream(input_tensor.device().index());

  const int32_t* perm_ptr = permutation.data_ptr<int32_t>();

  void* factors_ptr = nullptr;
  if (factors_opt.has_value()) {
    TORCH_CHECK(factors_opt->dtype() == output_tensor.dtype(), "Factors must match output dtype");
    TORCH_CHECK(factors_opt->numel() == m * topk, "Factors must have shape [m * topk]");
    factors_ptr = factors_opt->data_ptr();
  }

  if (output_tensor.scalar_type() == at::ScalarType::Half) {
    const at::Half* factor_data = static_cast<const at::Half*>(factors_ptr);
    apply_shuffle_mul_sum_kernel<at::Half><<<grid, block, 0, stream>>>(
        input_tensor.data_ptr<at::Half>(),
        output_tensor.data_ptr<at::Half>(),
        perm_ptr,
        m,
        topk,
        row_stride,
        static_cast<const at::Half*>(factors_ptr));
  } else if (output_tensor.scalar_type() == at::ScalarType::BFloat16) {
    const c10::BFloat16* factor_data = static_cast<const c10::BFloat16*>(factors_ptr);
    apply_shuffle_mul_sum_kernel<c10::BFloat16><<<grid, block, 0, stream>>>(
        input_tensor.data_ptr<c10::BFloat16>(),
        output_tensor.data_ptr<c10::BFloat16>(),
        perm_ptr,
        m,
        topk,
        row_stride,
        static_cast<const c10::BFloat16*>(factors_ptr));
  } else {
    TORCH_CHECK(false, "Unsupported output dtype for cast+mul kernel: ", output_tensor.scalar_type());
  }
}

/**
 * @brief Applies a permutation-based shuffle, element-wise multiplication, and reduction over the second dimension.
 *
 * This function performs the equivalent of the following PyTorch expression:
 *
 *     (c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)
 *
 * Specifically:
 * - `input` is shuffled using the `permutation` tensor.
 * - The shuffled tensor is reshaped and multiplied element-wise with `factors` (e.g., top-k weights).
 * - The result is summed along dimension 1 (the top-k dimension), and stored in `output`.
 *
 * @param input        Input tensor of shape (m * topk, k), representing c2.
 * @param output       Output tensor of shape (m, k), where the final reduced results are stored.
 * @param permutation  Index tensor (e.g., c_map) that maps positions in `input` to shuffled layout.
 * @param factors      Optional scaling factors (e.g., top-k weights), shape (m * topk) or (m, topk).
 */
void apply_shuffle_mul_sum(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& permutation,
    const std::optional<torch::Tensor>& factors) {
  get_apply_shuffle_mul_sum_caller(input, output, permutation, factors);
}
