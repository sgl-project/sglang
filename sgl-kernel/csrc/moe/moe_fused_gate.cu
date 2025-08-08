#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>
template <typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;
using bfloat16_t = cutlass::bfloat16_t;
using float16_t = cutlass::half_t;
using float32_t = float;

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">" matches these operands: built-in operator
// "arithmetic > arithmetic" function "operator>(const __half &, const __half &)"
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast to float.
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    // For types like float, at::BFloat16, or cutlass::half_t / cutlass::bfloat16_t, assume operator> works as expected.
    return a > b;
  }
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    return static_cast<float>(a) == static_cast<float>(b);
  } else {
    return a == b;
  }
}

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 512;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

template <typename T>
__device__ inline T recalculate_sigmoid(int expert_idx, T* input_ptr) {
  T val = input_ptr[expert_idx];
  float fval = static_cast<float>(val);
  // Clamp the value to avoid overflow in expf
  fval = fminf(fmaxf(fval, -20.0f), 20.0f);
  float sigmoid;
  if (fval >= 0) {
    sigmoid = 1.0f / (1.0f + expf(-fval));
  } else {
    float exp_val = expf(fval);
    sigmoid = exp_val / (1.0f + exp_val);
  }
  return static_cast<T>(sigmoid);
}

// Helper function to handle shared experts and output rescaling
template <typename T, typename Params>
__device__ void process_shared_experts_and_rescale(
    int thread_group_idx,
    int64_t thread_row,
    int64_t topk,
    int64_t topk_excluding_share_expert_fusion,
    int64_t num_fused_shared_experts,
    float output_sum,
    double routed_scaling_factor,
    float* output_ptr,
    int32_t* indices_ptr,
    Params params,
    int64_t num_rows) {
  if (thread_row >= num_rows) {
    return;
  }
  // Process shared experts
  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;

    if (last_idx < topk * num_rows) {
      int64_t expert_offset = 0;
      indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
      output_ptr[last_idx] = output_sum / routed_scaling_factor;

      if (num_fused_shared_experts > 1) {
        for (int i = 1; i < num_fused_shared_experts; ++i) {
          ++last_idx;
          if (last_idx >= topk * num_rows) break;
          ++expert_offset;
          indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
          output_ptr[last_idx] = output_sum / routed_scaling_factor;
        }
      }
    }
  }
  __syncthreads();

  // Rescale outputs
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      if (idx < topk * num_rows) {
        output_ptr[idx] = output_ptr[idx] / output_sum;
      }
    }
  }
}

// Unified implementation that handles both small and large VPT cases
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
    Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    // 确保所有无效线程在任何情况下都退出
    return;
  }

  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  // Cast pointers to type T
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Common variables for both small and large VPT cases
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  AccessType<T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(thread_read_ptr);
  AccessType<T> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(bias_thread_read_ptr);

  // Variables for large VPT processing
  __shared__ T shared_sigmoid[WARP_SIZE * (32 + 1)];
  __shared__ T shared_bias[WARP_SIZE * (32 + 1)];
  __shared__ int current_tile_idx;
  int thread_shared_offset = tidx % WARP_SIZE;

  // Variables to track global maximum values (used in large VPT case)
  T global_max_val = static_cast<T>(-FLT_MAX);
  T global_max_val_second = static_cast<T>(-FLT_MAX);
  int global_max_idx = -1;
  int global_max_second_idx = -1;

  Array<T, 32> row_chunk;   // Used for both small and large VPT cases
  Array<T, 32> bias_chunk;  // Used for both small and large VPT cases

  //-------------------------------------------------------------------------
  // Data Loading & Sigmoid Calculation
  //-------------------------------------------------------------------------
  if (params.VPT <= 32) {
// Small VPT case: Load all data at once
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      row_chunk[ii] = vec_thread_read_ptr[0][ii];
      bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
    }

    __syncthreads();

// Sigmoid calculation
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      // Add input value clipping to ensure numerical stability
      float tmp = static_cast<float>(row_chunk[ii]);
      tmp = fminf(fmaxf(tmp, -20.0f), 20.0f);

      // Add different handling for positive and negative values to improve stability
      float sigmoid_val;
      if (tmp >= 0) {
        sigmoid_val = 1.0f / (1.0f + expf(-tmp));
      } else {
        float exp_val = expf(tmp);
        sigmoid_val = exp_val / (1.0f + exp_val);
      }

      row_chunk[ii] = static_cast<T>(sigmoid_val);
    }
    __syncthreads();

// Add bias
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
    }
  } else {
// Large VPT case: Process data in tiles
#pragma unroll
    for (int tile = 0; tile < (params.VPT + 31) / 32; ++tile) {
      // Set current tile index
      if (tidx == 0 && threadIdx.y == 0) {
        current_tile_idx = tile;
      }
      __syncthreads();

      int tile_offset = tile * 32;
      int tile_size = min(32, params.VPT - tile_offset);
      if (tile_size <= 0) break;

      // Prefetch next tile data
      if (tile + 1 < (params.VPT + 31) / 32) {
        int next_offset = (tile + 1) * 32;
        int prefetch_size = min(32, params.VPT - next_offset);
        if (prefetch_size > 0) {
#pragma unroll
          for (int i = 0; i < prefetch_size; i += 8) {
            if (std::is_same<T, float32_t>::value) {
              volatile float dummy = __ldg(reinterpret_cast<const float*>(&thread_read_ptr[next_offset + i]));
              volatile float dummy2 = __ldg(reinterpret_cast<const float*>(&bias_thread_read_ptr[next_offset + i]));
            } else {
              volatile T dummy = thread_read_ptr[next_offset + i];
              volatile T dummy2 = bias_thread_read_ptr[next_offset + i];
            }
          }
        }
      }

// Load current tile data
#pragma unroll
      for (int ii = 0; ii < tile_size; ++ii) {
        int global_idx = tile_offset + ii;
        row_chunk[ii] = thread_read_ptr[global_idx];
        bias_chunk[ii] = bias_thread_read_ptr[global_idx];
      }

// Process current tile
#pragma unroll
      for (int ii = 0; ii < tile_size; ++ii) {
        int global_idx = tile_offset + ii;

        // Calculate sigmoid
        float input_val = static_cast<float>(row_chunk[ii]);
        input_val = fminf(fmaxf(input_val, -20.0f), 20.0f);
        float sigmoid_val;
        // sigmoid
        if (input_val >= 0) {
          sigmoid_val = 1.0f / (1.0f + expf(-input_val));
        } else {
          float exp_val = expf(input_val);
          sigmoid_val = exp_val / (1.0f + exp_val);
        }
        T sigmoid_t = static_cast<T>(sigmoid_val);
        // add bias
        T val_with_bias = sigmoid_t + bias_chunk[ii];

        // Store the results using linear indexing
        int shared_idx = thread_shared_offset + ii * WARP_SIZE;
        shared_sigmoid[shared_idx] = sigmoid_t;
        shared_bias[shared_idx] = val_with_bias;

        // Update global maximum tracking
        if (cmp_gt(val_with_bias, global_max_val)) {
          global_max_val_second = global_max_val;
          global_max_second_idx = global_max_idx;
          global_max_val = val_with_bias;
          global_max_idx = global_idx;
        } else if (cmp_gt(val_with_bias, global_max_val_second)) {
          global_max_val_second = val_with_bias;
          global_max_second_idx = global_idx;
        }
      }
      __syncthreads();
    }
  }

//-------------------------------------------------------------------------
// Exclude Groups
//-------------------------------------------------------------------------
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
    int expert = first_elt_read_by_thread;
    T max_val, max_val_second, max_sum;

    if (params.VPT <= 32) {
      // Small VPT: Find local maximums from bias_chunk
      max_val = static_cast<T>(-FLT_MAX);
      max_val_second = static_cast<T>(-FLT_MAX);

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
    } else {
      // Large VPT: Use pre-computed global maximums
      max_val = global_max_val;
      max_val_second = global_max_val_second;
    }

    // Sum top2 sigmoid weight values as the group weight
    max_sum = max_val + max_val_second;

// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max_sum =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_sum), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (cmp_gt(max_sum, other_max_sum) || (cmp_eq(other_max_sum, max_sum) && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // Clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
        if (params.VPT <= 32) {
// Small VPT: Clear all values in bias_chunk
#pragma unroll
          for (int ii = 0; ii < params.VPT; ++ii) {
            bias_chunk[ii] = static_cast<T>(FLT_MAX);
          }
        } else {
          // Large VPT: Clear specific values in shared memory or recalculate
          int expert_mod = expert % params.VPT;
          if (expert_mod == global_max_idx || expert_mod == global_max_second_idx) {
            int tile_idx = expert_mod / 32;
            int local_idx = expert_mod % 32;

            // Clear in shared memory if in current tile
            if (tile_idx == current_tile_idx) {
              int shared_idx = thread_shared_offset + local_idx * (WARP_SIZE + 1);
              shared_bias[shared_idx] = static_cast<T>(FLT_MAX);
            }

            // Reset and recalculate global maximums
            global_max_val = static_cast<T>(-FLT_MAX);
            global_max_val_second = static_cast<T>(-FLT_MAX);
            global_max_idx = -1;
            global_max_second_idx = -1;

#pragma unroll
            for (int i = 0; i < params.VPT; ++i) {
              int tile_idx = i / 32;
              int local_idx = i % 32;

              T val;
              if (tile_idx == current_tile_idx) {
                int shared_idx = thread_shared_offset + local_idx * (WARP_SIZE + 1);
                val = shared_bias[shared_idx];
              } else {
                val = recalculate_sigmoid(i, thread_read_ptr) + bias_thread_read_ptr[i];
              }

              if (cmp_gt(val, global_max_val) && !cmp_eq(val, static_cast<T>(FLT_MAX))) {
                global_max_val_second = global_max_val;
                global_max_second_idx = global_max_idx;
                global_max_val = val;
                global_max_idx = i;
              } else if (cmp_gt(val, global_max_val_second) && !cmp_eq(val, static_cast<T>(FLT_MAX))) {
                global_max_val_second = val;
                global_max_second_idx = i;
              }
            }
          }
        }
      }
    }
  }

  __syncthreads();

  //-------------------------------------------------------------------------
  // Topk Selection
  //-------------------------------------------------------------------------
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    T max_val;
    int expert = first_elt_read_by_thread;

    if (params.VPT <= 32) {
      // Small VPT: Find local max directly from bias_chunk
      max_val = bias_chunk[0];

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
    } else {
      // Large VPT: Scan through all tiles to find maximum
      T local_max_val = static_cast<T>(-FLT_MAX);
      int local_max_idx = -1;

#pragma unroll
      for (int i = 0; i < params.VPT; ++i) {
        int tile_idx = i / 32;
        int local_idx = i % 32;

        T val;
        if (tile_idx == current_tile_idx) {
          int shared_idx = thread_shared_offset + local_idx * WARP_SIZE;
          val = shared_bias[shared_idx];
        } else {
          int global_idx = tile_idx * 32 + local_idx;
          float input_val = static_cast<float>(thread_read_ptr[global_idx]);
          float sigmoid_val;
          if (input_val >= 0) {
            sigmoid_val = 1.0f / (1.0f + expf(-input_val));
          } else {
            float exp_val = expf(input_val);
            sigmoid_val = exp_val / (1.0f + exp_val);
          }
          val = static_cast<T>(sigmoid_val) + bias_thread_read_ptr[global_idx];
        }

        if (cmp_gt(val, local_max_val) && !cmp_eq(val, static_cast<T>(FLT_MAX)) &&
            !cmp_eq(val, static_cast<T>(-FLT_MAX))) {
          local_max_val = val;
          local_max_idx = i;
        }
      }

      if (local_max_idx == -1) {
        max_val = static_cast<T>(-FLT_MAX);
      } else {
        max_val = local_max_val;
        expert = first_elt_read_by_thread + local_max_idx;
      }
    }

// argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (cmp_gt(other_max, max_val) || (cmp_eq(other_max, max_val) && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group && idx < topk * num_rows) {
      int expert_to_clear_in_thread = expert % params.VPT;
      indices_ptr[idx] = static_cast<int32_t>(expert);

      if (params.VPT <= 32) {
        // Small VPT: Clear and store directly
        bias_chunk[expert_to_clear_in_thread] = static_cast<T>(-FLT_MAX);
        output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
      } else {
        // Large VPT: Handle tiled case
        int tile_idx = expert_to_clear_in_thread / 32;
        int local_idx = expert_to_clear_in_thread % 32;

        if (tile_idx == current_tile_idx) {
          int shared_idx = thread_shared_offset + local_idx * WARP_SIZE;
          shared_bias[shared_idx] = static_cast<T>(-FLT_MAX);
          output_ptr[idx] = static_cast<float>(shared_sigmoid[shared_idx]);
        } else {
          int global_idx = expert_to_clear_in_thread;
          float input_val = static_cast<float>(thread_read_ptr[global_idx]);
          float sigmoid_val;
          if (input_val >= 0) {
            sigmoid_val = 1.0f / (1.0f + expf(-input_val));
          } else {
            float exp_val = expf(input_val);
            sigmoid_val = exp_val / (1.0f + exp_val);
          }
          output_ptr[idx] = sigmoid_val;
        }
      }

      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    // accumulate sum for all elements
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    __syncthreads();
  }

  // Process shared experts and rescale outputs
  process_shared_experts_and_rescale<T>(
      thread_group_idx,
      thread_row,
      topk,
      topk_excluding_share_expert_fusion,
      num_fused_shared_experts,
      output_sum,
      routed_scaling_factor,
      output_ptr,
      indices_ptr,
      params,
      num_rows);
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
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
    double routed_scaling_factor) {
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
      params);
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                 \
  do {                                                                                                   \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                      \
    /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                       \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;      \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                          \
    /* Calculate shared memory size */                                                                   \
    size_t shared_mem_size = (VPT <= 32) ? 0 : (2 * WARP_SIZE * 32 * sizeof(T) + sizeof(int));           \
    moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
        <<<static_cast<unsigned int>(num_blocks), block_dim, shared_mem_size, stream>>>(                 \
            input.data_ptr(),                                                                            \
            bias.data_ptr(),                                                                             \
            output.data_ptr<float>(),                                                                    \
            indices.data_ptr<int32_t>(),                                                                 \
            num_rows,                                                                                    \
            topk_group,                                                                                  \
            topk,                                                                                        \
            num_fused_shared_experts,                                                                    \
            routed_scaling_factor);                                                                      \
    dispatched = true;                                                                                   \
  } while (0)

//------------------------------------------------------------------------------
// Dynamic Kernel Version (parameters computed at runtime)
//------------------------------------------------------------------------------
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
    double routed_scaling_factor) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3, this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
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
      params);
}

//------------------------------------------------------------------------------
// Host Launcher Function
//------------------------------------------------------------------------------
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  int64_t num_rows = input.size(0);
  int64_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  if (num_expert_group == 1 && num_experts > 128) {
    auto gate = (1.0f / (1.0f + (-input).exp())) + bias;
    
    int64_t topk_excluding_shared = topk - num_fused_shared_experts;
    auto topk_result = gate.topk(topk_excluding_shared, 1, true, true);
    
    output.narrow(1, 0, topk_excluding_shared).copy_(std::get<0>(topk_result));
    indices.narrow(1, 0, topk_excluding_shared).copy_(std::get<1>(topk_result));
    
    if (num_fused_shared_experts > 0) {
        auto output_sum = std::get<0>(topk_result).sum(1);
        for (int64_t i = 0; i < num_fused_shared_experts; ++i) {
            indices.select(1, topk_excluding_shared + i).fill_(num_experts + i);
            output.select(1, topk_excluding_shared + i).copy_(
                output_sum / static_cast<float>(routed_scaling_factor));
        }
    }
    
    auto row_sums = output.sum(1, true);
    output.div_(row_sums);
    
    return {output, indices};
  }
  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Check 1: Ensure that num_experts is a power of 2, or 384 for Kimi K2.
  TORCH_CHECK(
      (num_experts & (num_experts - 1)) == 0 || num_experts == 384,
      "num_experts must be a power of 2 or 384, but got ",
      num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int64_t computed_vpt = num_experts / num_expert_group;
  // Check 3: Ensure that num_experts/num_expert_group does not exceed MAX_VPT=512. Maximum VPT indicate max value per
  // threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 384 experts, with 1 group.
  //   Case 2: 256 experts, with 8 or 16 groups.
  //   Case 3: 128 experts, with 4 or 8 groups.
  //   Case 4: 64 experts, with 1 groups.
  //   Case 5: other cases, require 8 <= num_experts / num_expert_group <= 64.
  bool dispatched = false;
  switch (num_experts) {
    case 384:
      if (num_expert_group == 1)
        // Kimi K2 config: VPT = 384/1 = 384, ROWS_PER_WARP = 32/1 = 32
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 384, 1);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 384, 1);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 384, 1);
        }
      break;
    case 256:
      if (num_expert_group == 8) {
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 8);
        }
      } else if (num_expert_group == 16) {
        // Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2 = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 16);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 16);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 16);
        }
      }
      break;
    case 128:
      if (num_expert_group == 4) {
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/4 = 8, ROWS_PER_CTA = 6 * 8 = 48.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 4);
        }
      } else if (num_expert_group == 8) {
        // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 8);
        }
      }
      break;
    case 64:
      if (num_expert_group == 1)
        // VPT = 64/1 = 64, ROWS_PER_WARP = 32/1 = 32, ROWS_PER_CTA = 6 * 32 = 192.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 64, 1);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 64, 1);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 64, 1);
        }
      break;
    default:
      break;
  }
  if (!dispatched) {
    // Fallback to the dynamic kernel if none of the supported combinations match.
    // currently only support num_experts / num_expert_group <= 512 for dynamic kernels
    if (input.scalar_type() == at::kBFloat16) {
      // QQ NOTE: for bfloat16, we use cutlass::bfloat16_t
      // QQ NOTE: shared memory size is 2 * WARP_SIZE * 32 * sizeof(bfloat16_t) + sizeof(int)
      size_t shared_mem_size_bf16 = (computed_vpt <= 32) ? 0 : (2 * WARP_SIZE * 32 * sizeof(bfloat16_t) + sizeof(int));
      moe_fused_gate_kernel_dynamic<bfloat16_t>
          <<<static_cast<unsigned int>(num_blocks), block_dim, shared_mem_size_bf16, stream>>>(
              input.data_ptr(),
              bias.data_ptr(),
              output.data_ptr<float>(),
              indices.data_ptr<int32_t>(),
              num_rows,
              num_experts,
              num_expert_group,
              topk_group,
              topk,
              num_fused_shared_experts,
              routed_scaling_factor);
    } else if (input.scalar_type() == at::kHalf) {
      size_t shared_mem_size_f16 = (computed_vpt <= 32) ? 0 : (2 * WARP_SIZE * 32 * sizeof(float16_t) + sizeof(int));
      moe_fused_gate_kernel_dynamic<float16_t>
          <<<static_cast<unsigned int>(num_blocks), block_dim, shared_mem_size_f16, stream>>>(
              input.data_ptr(),
              bias.data_ptr(),
              output.data_ptr<float>(),
              indices.data_ptr<int32_t>(),
              num_rows,
              num_experts,
              num_expert_group,
              topk_group,
              topk,
              num_fused_shared_experts,
              routed_scaling_factor);
    } else if (input.scalar_type() == at::kFloat) {
      size_t shared_mem_size_f32 = (computed_vpt <= 32) ? 0 : (2 * WARP_SIZE * 32 * sizeof(float32_t) + sizeof(int));
      moe_fused_gate_kernel_dynamic<float32_t>
          <<<static_cast<unsigned int>(num_blocks), block_dim, shared_mem_size_f32, stream>>>(
              input.data_ptr(),
              bias.data_ptr(),
              output.data_ptr<float>(),
              indices.data_ptr<int32_t>(),
              num_rows,
              num_experts,
              num_expert_group,
              topk_group,
              topk,
              num_fused_shared_experts,
              routed_scaling_factor);
    } else {
      TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
    }
  }
  return {output, indices};
}
