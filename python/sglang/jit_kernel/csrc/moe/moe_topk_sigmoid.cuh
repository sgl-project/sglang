// Adapt from https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/moe/topk_softmax_kernels.cu
// which is originally adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/cub.cuh>
#include <cub/util_type.cuh>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

// CUDA 12.9+ deprecated cub::Max/Min in favour of cuda::maximum/minimum
#if CUDA_VERSION >= 12090
#include <cuda/functional>
using MaxReduceOp = cuda::maximum<>;
using MinReduceOp = cuda::minimum<>;
#else
using MaxReduceOp = cub::Max;
using MinReduceOp = cub::Min;
#endif

#include <cfloat>
#include <cstdint>
#include <type_traits>

using tvm::ffi::TensorView;

#ifndef MOE_TOPK_SIGMOID_WARP_SIZE
#define MOE_TOPK_SIGMOID_WARP_SIZE 32
#endif

namespace {

static constexpr int WARP_SIZE = MOE_TOPK_SIGMOID_WARP_SIZE;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---------------------------------------------------------------------------
// Aligned array — avoids CUTLASS dependency; identical semantics.
// ---------------------------------------------------------------------------
template <typename T, int N, int Alignment = static_cast<int>(sizeof(T) * N)>
class alignas(Alignment) AlignedArray {
  T data[N];
};

// ---------------------------------------------------------------------------
// Type conversion helper
// ---------------------------------------------------------------------------
template <typename T>
__device__ float convert_to_float(T x) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(x);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(x);
  } else {
    return static_cast<float>(x);
  }
}

// ---------------------------------------------------------------------------
// moeSigmoid — fallback sigmoid kernel (used for non-power-of-2 experts)
// ---------------------------------------------------------------------------
template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeSigmoid(
    const T* input, const bool* finished, float* output, const int num_cols, const float* correction_bias) {
  const int thread_row_offset = blockIdx.x * num_cols;

  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    float val = convert_to_float<T>(input[idx]);
    val = 1.0f / (1.0f + expf(-val));
    if (correction_bias != nullptr) {
      val = val + correction_bias[ii];
    }
    output[idx] = val;
  }
}

// ---------------------------------------------------------------------------
// moeTopK — fallback top-k kernel (used for non-power-of-2 experts)
// ---------------------------------------------------------------------------
template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_sigmoid,
    const bool* finished,
    float* output,
    int* indices,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* correction_bias) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x;
  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_sigmoid[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];
        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int expert = result_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * block_row + k_idx;
      float val = result_kvp.value;
      if (correction_bias != nullptr) {
        val -= correction_bias[expert];
      }
      output[idx] = val;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      row_sum_for_renormalize += val;
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

// ---------------------------------------------------------------------------
// topkGatingSigmoid — optimised kernel for power-of-2 expert counts
// ---------------------------------------------------------------------------
template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSigmoid(
    const T* input,
    const bool* finished,
    float* output,
    const int num_rows,
    int* indices,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* correction_bias) {
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(VPT % ELTS_PER_LDG == 0, "");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "");
  static_assert(THREADS_PER_ROW <= WARP_SIZE, "");

  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "");

  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;

  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  using AccessType = AlignedArray<T, ELTS_PER_LDG>;

  T row_chunk_temp[VPT];
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_temp);
  const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  float row_chunk[VPT];
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    float val = convert_to_float<T>(row_chunk_temp[ii]);
    val = 1.0f / (1.0f + expf(-val));
    if (correction_bias != nullptr) {
      const int group_id = ii / ELTS_PER_LDG;
      const int local_id = ii % ELTS_PER_LDG;
      const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
      val = val + correction_bias[expert_idx];
    }
    row_chunk[ii] = val;
  }

  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  float row_sum_for_renormalize = 0;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max = __shfl_xor_sync(0xffffffff, max_val, mask, THREADS_PER_ROW);
      int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, THREADS_PER_ROW);
      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    if (thread_group_idx == 0) {
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * thread_row + k_idx;
      if (correction_bias != nullptr) {
        max_val -= correction_bias[expert];
      }
      output[idx] = max_val;
      indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
      row_sum_for_renormalize += max_val;
    }

    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
      }
    }
  }

  if (renormalize && thread_group_idx == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * thread_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

// ---------------------------------------------------------------------------
// Compile-time constants helper
// ---------------------------------------------------------------------------
namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
  static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

// ---------------------------------------------------------------------------
// Per-expert-count launcher helper
// ---------------------------------------------------------------------------
template <typename T, int EXPERTS, int WARPS_PER_TB>
void topkGatingSigmoidLauncherHelper(
    const T* input,
    const bool* finished,
    float* output,
    int* indices,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float* correction_bias,
    cudaStream_t stream) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
  using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topkGatingSigmoid<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
      input, finished, output, num_rows, indices, k, start_expert, end_expert, renormalize, correction_bias);
}

// ---------------------------------------------------------------------------
// Dispatch macro — used inside topkGatingSigmoidKernelLauncher
// ---------------------------------------------------------------------------
#define LAUNCH_SIGMOID(TYPE, NUM_EXPERTS, WARPS_PER_TB)             \
  topkGatingSigmoidLauncherHelper<TYPE, NUM_EXPERTS, WARPS_PER_TB>( \
      gating_output,                                                \
      nullptr,                                                      \
      topk_weights,                                                 \
      topk_indices,                                                 \
      num_tokens,                                                   \
      topk,                                                         \
      0,                                                            \
      num_experts,                                                  \
      renormalize,                                                  \
      correction_bias,                                              \
      stream)

// ---------------------------------------------------------------------------
// Main launcher: dispatches on num_experts
// ---------------------------------------------------------------------------
template <typename T>
void topkGatingSigmoidKernelLauncher(
    const T* gating_output,
    float* topk_weights,
    int* topk_indices,
    float* sigmoid_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    const float* correction_bias,
    cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  switch (num_experts) {
    case 1:
      LAUNCH_SIGMOID(T, 1, WARPS_PER_TB);
      break;
    case 2:
      LAUNCH_SIGMOID(T, 2, WARPS_PER_TB);
      break;
    case 4:
      LAUNCH_SIGMOID(T, 4, WARPS_PER_TB);
      break;
    case 8:
      LAUNCH_SIGMOID(T, 8, WARPS_PER_TB);
      break;
    case 16:
      LAUNCH_SIGMOID(T, 16, WARPS_PER_TB);
      break;
    case 32:
      LAUNCH_SIGMOID(T, 32, WARPS_PER_TB);
      break;
    case 64:
      LAUNCH_SIGMOID(T, 64, WARPS_PER_TB);
      break;
    case 128:
      LAUNCH_SIGMOID(T, 128, WARPS_PER_TB);
      break;
    case 256:
      LAUNCH_SIGMOID(T, 256, WARPS_PER_TB);
      break;
    default: {
      // Fallback: non-power-of-2 or >256 experts
      using namespace host;
      RuntimeCheck(
          sigmoid_workspace != nullptr, "sigmoid_workspace must be provided for num_experts that are not a power of 2");
      static constexpr int TPB = 256;
      moeSigmoid<T, TPB>
          <<<num_tokens, TPB, 0, stream>>>(gating_output, nullptr, sigmoid_workspace, num_experts, correction_bias);
      moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
          sigmoid_workspace,
          nullptr,
          topk_weights,
          topk_indices,
          num_experts,
          topk,
          0,
          num_experts,
          renormalize,
          correction_bias);
    }
  }
}

#undef LAUNCH_SIGMOID

}  // namespace

// ---------------------------------------------------------------------------
// Host launcher (tvm-ffi interface)
// ---------------------------------------------------------------------------
template <typename T>
void topk_sigmoid(
    TensorView gating_output,
    TensorView topk_weights,
    TensorView topk_ids,
    TensorView workspace,
    bool renormalize,
    tvm::ffi::Optional<TensorView> correction_bias) {
  using namespace host;

  // --- Input validation ---
  RuntimeCheck(gating_output.dim() == 2, "gating_output must be 2-D");
  RuntimeCheck(topk_weights.dim() == 2, "topk_weights must be 2-D");
  RuntimeCheck(topk_ids.dim() == 2, "topk_ids must be 2-D");

  const int64_t num_tokens = gating_output.shape()[0];
  const int64_t num_experts = gating_output.shape()[1];
  const int64_t topk = topk_weights.shape()[1];

  RuntimeCheck(
      topk_weights.shape()[0] == num_tokens && topk_ids.shape()[0] == num_tokens,
      "topk_weights and topk_ids must have num_tokens rows");
  RuntimeCheck(topk_ids.shape()[1] == topk, "topk_ids second dim must match topk_weights");
  RuntimeCheck(topk <= num_experts, "topk must be <= num_experts");

  // correction_bias validation
  if (correction_bias.has_value()) {
    const auto& bias = correction_bias.value();
    RuntimeCheck(bias.dim() == 1, "correction_bias must be 1-D");
    RuntimeCheck(bias.shape()[0] == num_experts, "correction_bias size must equal num_experts");
    RuntimeCheck(
        bias.dtype().code == DLDataTypeCode::kDLFloat && bias.dtype().bits == 32, "correction_bias must be float32");
  }

  const T* gating_ptr = static_cast<const T*>(gating_output.data_ptr());
  float* weights_ptr = static_cast<float*>(topk_weights.data_ptr());
  int* indices_ptr = static_cast<int*>(topk_ids.data_ptr());
  float* workspace_ptr = static_cast<float*>(workspace.data_ptr());
  const float* bias_ptr =
      correction_bias.has_value() ? static_cast<const float*>(correction_bias.value().data_ptr()) : nullptr;

  cudaStream_t stream = LaunchKernel::resolve_device(gating_output.device());

  topkGatingSigmoidKernelLauncher<T>(
      gating_ptr,
      weights_ptr,
      indices_ptr,
      workspace_ptr,
      static_cast<int>(num_tokens),
      static_cast<int>(num_experts),
      static_cast<int>(topk),
      renormalize,
      bias_ptr,
      stream);
}
