#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace {

// ---------------------------------------------------------------------------
// Local AlignedArray — replaces cutlass::AlignedArray<T, N>.
// alignas(16) matches CUDA's maximum vectorized-load alignment (128-bit).
// ---------------------------------------------------------------------------
template <typename T, int N>
struct alignas(16) AlignedArray {
  T data[N];
  __device__ __host__ T& operator[](int i) {
    return data[i];
  }
  __device__ __host__ const T& operator[](int i) const {
    return data[i];
  }
};

// ---------------------------------------------------------------------------
// Fixed constants (same as AOT version)
// ---------------------------------------------------------------------------
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;  // maximum VPT: num_experts / num_expert_group

template <typename T, int N>
using Array = AlignedArray<T, N>;

// QQ NOTE: this must be sized >= params.VPT; MAX_VPT satisfies that
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

// ---------------------------------------------------------------------------
// Comparison helpers
// fp16_t (__half) has an ambiguous operator> — cast through float to resolve.
// bf16_t (__nv_bfloat16) and fp32_t (float) have unambiguous operator>.
// ---------------------------------------------------------------------------
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, fp16_t>) {
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    return a > b;
  }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, fp16_t>) {
    return static_cast<float>(a) == static_cast<float>(b);
  } else {
    return a == b;
  }
}

// ---------------------------------------------------------------------------
// Core device implementation — algorithm identical to the AOT version.
// Params may be KernelParams (compile-time constexpr fields) or
// KernelParamsDynamic (runtime fields); both expose the same field names.
// ---------------------------------------------------------------------------
template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    return;
  }

  int64_t topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Local scratch: row values (sigmoid output) and bias-added values
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> row_chunk;
  AccessType<T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(thread_read_ptr);

  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> bias_chunk;
  AccessType<T> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(bias_thread_read_ptr);

  // QQ NOTE: loop-based load avoids misaligned-address issues when params.VPT < 8
  // (vectorised single-load would misalign since local array is MAX_VPT-sized)
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = vec_thread_read_ptr[0][ii];
    bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
  }

  __syncthreads();

  ////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
  }
  __syncthreads();

  ////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

  ////////////////////// Exclude Groups //////////////////////
  // Iteratively zero out the (THREADS_PER_ROW - topk_group) lowest-scoring groups.
  // Each iteration finds the group with the minimum top-2 sum and clears it.
  // Higher expert index wins ties (warp-reduce argmin with higher-index tie-break).
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
    int expert = first_elt_read_by_thread;

    // Local top-2 in this thread's group
    T max_val = static_cast<T>(-FLT_MAX);
    T max_val_second = static_cast<T>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      T val = bias_chunk[ii];
      if (cmp_gt(val, max_val)) {
        max_val_second = max_val;
        max_val = val;
      } else if (cmp_gt(val, max_val_second)) {
        max_val_second = val;
      }
    }

    // Group weight = sum of top-2 sigmoid values
    T max_sum = max_val + max_val_second;

    // Warp-level argmin: find the group with the smallest max_sum.
    // Tie-breaking: higher expert index wins.
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max_sum =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_sum), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // higher indices win ties
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // The thread owning the lowest-scoring group clears its values to +FLT_MAX
    // (sentinel: excluded from topk selection below)
    int const thread_to_clear_in_group = expert / params.VPT;
    if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
      for (int ii = 0; ii < params.VPT; ++ii) {
        bias_chunk[ii] = static_cast<T>(FLT_MAX);
      }
    }
  }

  __syncthreads();

  ////////////////////// TopK Selection //////////////////////
  // Iteratively pick the highest-scoring expert across all non-excluded groups.
  // Tie-breaking: lower expert index wins.
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    // Local argmax within this thread's group (skip excluded slots = FLT_MAX)
    T max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!cmp_eq(max_val, static_cast<T>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        T val = bias_chunk[ii];
        if (cmp_gt(val, max_val)) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = static_cast<T>(-FLT_MAX);
    }

    // Warp-level argmax: lower expert index wins ties
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group) {
      int expert_to_clear_in_thread = expert % params.VPT;

      // Remove selected expert so it is not picked again
      bias_chunk[expert_to_clear_in_thread] = static_cast<T>(-FLT_MAX);

      // Write sigmoid weight (pre-bias) and expert index to output
      output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    // Thread 0 in each row accumulates output sum for normalisation
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    __syncthreads();
  }

  ////////////////////// Fused Shared Experts //////////////////////
  // Append virtual "shared expert" entries at the end of the topk slots.
  // Their weight = output_sum / routed_scaling_factor.
  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;
    for (int i = 0; i < num_fused_shared_experts; ++i) {
      indices_ptr[last_idx + i] = static_cast<int32_t>(params.NUM_EXPERTS + i);
      output_ptr[last_idx + i] = output_sum / routed_scaling_factor;
    }
  }
  __syncthreads();

  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum;
      if (apply_routed_scaling_factor_on_output) {
        output_ptr[idx] *= routed_scaling_factor;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Static kernel: all layout params are compile-time constants.
// Used for known configurations (256/8, 256/16, 128/4, 128/8).
// ---------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// Macro: compute compile-time constants and launch the static kernel.
// T is the dtype template parameter inherited from the enclosing host function.
// stream is a cudaStream_t resolved in the host launcher before this macro.
#define LAUNCH_MOE_GATE_CONFIG(EXPERTS, EXPERT_GROUP)                                                         \
  do {                                                                                                        \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                           \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;           \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                               \
    host::LaunchKernel(                                                                                       \
        {static_cast<uint32_t>(num_blocks), 1u, 1u},                                                          \
        {static_cast<uint32_t>(WARP_SIZE), static_cast<uint32_t>(WARPS_PER_CTA), 1u},                         \
        stream)(                                                                                              \
        moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA>, \
        input.data_ptr(),                                                                                     \
        bias.data_ptr(),                                                                                      \
        static_cast<float*>(output.data_ptr()),                                                               \
        static_cast<int32_t*>(indices.data_ptr()),                                                            \
        num_rows,                                                                                             \
        topk_group,                                                                                           \
        topk,                                                                                                 \
        num_fused_shared_experts,                                                                             \
        routed_scaling_factor,                                                                                \
        apply_routed_scaling_factor_on_output);                                                               \
    dispatched = true;                                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Dynamic kernel: layout params are resolved at kernel-launch time.
// Fallback for configurations not covered by the static switch below.
// ---------------------------------------------------------------------------
struct KernelParamsDynamic {
  int VPT;
  int NUM_EXPERTS;
  int THREADS_PER_ROW;
  int ROWS_PER_WARP;
  int ROWS_PER_CTA;
  int WARPS_PER_CTA;
};

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    void* input,
    void* bias,
    float* output_ptr,
    int32_t* indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = static_cast<int>(num_experts);
  params.VPT = static_cast<int>(num_experts / num_expert_group);
  params.THREADS_PER_ROW = static_cast<int>(num_expert_group);
  params.WARPS_PER_CTA = WARPS_PER_CTA;
  params.ROWS_PER_WARP = static_cast<int>(max((int64_t)1, (int64_t)(WARP_SIZE / num_expert_group)));
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl<T>(
      input,
      bias,
      output_ptr,
      indices_ptr,
      num_rows,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output,
      params);
}

// ---------------------------------------------------------------------------
// Host launcher — tvm-ffi style, templated on dtype T.
// T is one of: fp32_t, fp16_t, bf16_t (set by the Python JIT wrapper).
// Outputs are destination-passing: caller pre-allocates output and indices.
// ---------------------------------------------------------------------------
template <typename T>
void moe_fused_gate(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView indices,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  using namespace host;

  // --- Input validation ---
  RuntimeCheck(
      input.dtype().code == bias.dtype().code && input.dtype().bits == bias.dtype().bits,
      "input and bias must have the same dtype");
  RuntimeCheck(input.dim() == 2, "input must be 2-D [num_rows, num_experts]");
  RuntimeCheck(bias.dim() == 1, "bias must be 1-D [num_experts]");

  const int64_t num_rows = input.shape()[0];
  const int64_t num_experts = input.shape()[1];

  RuntimeCheck(bias.shape()[0] == num_experts, "bias size must match num_experts");
  RuntimeCheck(
      output.dim() == 2 && output.shape()[0] == num_rows && output.shape()[1] == topk,
      "output must be [num_rows, topk]");
  RuntimeCheck(
      indices.dim() == 2 && indices.shape()[0] == num_rows && indices.shape()[1] == topk,
      "indices must be [num_rows, topk]");

  RuntimeCheck((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, got ", num_experts);
  RuntimeCheck(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, got ",
      num_experts,
      " / ",
      num_expert_group);

  const int64_t computed_vpt = num_experts / num_expert_group;
  RuntimeCheck(
      computed_vpt <= MAX_VPT, "num_experts / num_expert_group = ", computed_vpt, " exceeds MAX_VPT=", MAX_VPT);

  // --- Grid dimensions (same formula as AOT version) ---
  const int64_t rows_per_warp = max((int64_t)1, (int64_t)(WARP_SIZE / num_expert_group));
  const int64_t num_warps = div_ceil(num_rows, rows_per_warp);
  const int64_t num_blocks = div_ceil(num_warps, (int64_t)WARPS_PER_CTA);

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  // --- Static dispatch for known (num_experts, num_expert_group) configurations ---
  // The dtype is already baked into T at JIT compile time; no dtype switch needed here.
  // Supported static configs:
  //   (256, 8)  — DeepSeek V3: VPT=32, ROWS_PER_WARP=4, ROWS_PER_CTA=24
  //   (256, 16) —              VPT=16, ROWS_PER_WARP=2, ROWS_PER_CTA=12
  //   (128, 4)  —              VPT=32, ROWS_PER_WARP=8, ROWS_PER_CTA=48
  //   (128, 8)  —              VPT=16, ROWS_PER_WARP=4, ROWS_PER_CTA=24
  bool dispatched = false;
  switch (num_experts) {
    case 256:
      if (num_expert_group == 8) {
        LAUNCH_MOE_GATE_CONFIG(256, 8);
      } else if (num_expert_group == 16) {
        LAUNCH_MOE_GATE_CONFIG(256, 16);
      }
      break;
    case 128:
      if (num_expert_group == 4) {
        LAUNCH_MOE_GATE_CONFIG(128, 4);
      } else if (num_expert_group == 8) {
        LAUNCH_MOE_GATE_CONFIG(128, 8);
      }
      break;
    default:
      break;
  }

  if (!dispatched) {
    // Dynamic fallback: handles any valid (num_experts, num_expert_group) where VPT <= MAX_VPT.
    // #pragma unroll has no effect here since loop bounds are runtime values.
    host::LaunchKernel(
        {static_cast<uint32_t>(num_blocks), 1u, 1u},
        {static_cast<uint32_t>(WARP_SIZE), static_cast<uint32_t>(WARPS_PER_CTA), 1u},
        stream)(
        moe_fused_gate_kernel_dynamic<T>,
        input.data_ptr(),
        bias.data_ptr(),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t*>(indices.data_ptr()),
        num_rows,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  }
}

#undef LAUNCH_MOE_GATE_CONFIG

}  // namespace
