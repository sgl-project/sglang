#include <musa_runtime.h>
#include <mutlass/array.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>
#include <stdio.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
template <typename T, int N>
using AlignedArray = mutlass::AlignedArray<T, N>;
using bfloat16_t = mutlass::bfloat16_t;
using float16_t = mutlass::half_t;
using float32_t = float;

constexpr float log2ef = 1.4426950408889634074f;

static __device__ __forceinline__ float fast_expf(float a) {
  return __musa_exp2_f(a * log2ef);
}

static __device__ __forceinline__ float fast_rcpf(float x) {
  float y = __frcp_rn(x);
  y = y * (2.f - x * y);
  return y;
}

// QQ NOTE: to handle the case for at::Half, error: more than one operator ">"
// matches these operands: built-in operator "arithmetic > arithmetic" function
// "operator>(const __half &, const __half &)"
template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  if constexpr (std::is_same<T, at::Half>::value) {
    // at::Half (or float16_t in our native case) causes ambiguity, so we cast
    // to float.
    return static_cast<float>(a) > static_cast<float>(b);
  } else {
    // For types like float, at::BFloat16, or mutlass::half_t /
    // mutlass::bfloat16_t, assume operator> works as expected.
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

template <typename T>
__device__ inline bool cmp_ge(const T& a, const T& b, const int& x, const int& y) {
  return (x > y && a == b) || a < b;
}

// Fixed constants common to both dynamic and static template versions:
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 16;
static constexpr int MAX_VPT = 32;  // maximum VPT we support, > params.VPT = num_expert / num_expert_group

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

template <typename T, typename Params>
__device__ void moe_fused_gate_impl_dynamic(
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
  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret
  // the address of row_chunk as a pointer to AccessType.
  T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> row_chunk;
  AccessType<T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(thread_read_ptr);

  T* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> bias_chunk;
  AccessType<T> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<T> const*>(bias_thread_read_ptr);

  // QQ NOTE: doing the follow will be slower than loop assign and more
  // importantly have misaligned address issue when params.VPT < 8 and mismatch
  // with MAX_VPT AccessType<T>* row_chunk_vec_ptr =
  // reinterpret_cast<AccessType<T>*>(&row_chunk); row_chunk_vec_ptr[0] =
  // vec_thread_read_ptr[0];
  if (thread_row < num_rows) {
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      row_chunk[ii] = vec_thread_read_ptr[0][ii];
      bias_chunk[ii] = vec_bias_thread_read_ptr[0][ii];
    }
  }

  ////////////////////// Sigmoid //////////////////////
  if (thread_row < num_rows) {
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      row_chunk[ii] = static_cast<T>(fast_rcpf(1.0f + fast_expf(-float(row_chunk[ii]))));
    }
  }

  ////////////////////// Add Bias //////////////////////
  if (thread_row < num_rows) {
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
    }
  }

  ////////////////////// Exclude Groups //////////////////////
  if (thread_row < num_rows) {
#pragma unroll
    for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
         ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
      int expert = first_elt_read_by_thread;
      // local argmax
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

      // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each
      // expert group and sum them as the group weight to select expert groups
      T max_sum = max_val + max_val_second;

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

      // clear the max value in the thread
      if (k_idx < params.THREADS_PER_ROW - topk_group) {
        int const thread_to_clear_in_group = expert / params.VPT;

        if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
          for (int ii = 0; ii < params.VPT; ++ii) {
            bias_chunk[ii] = static_cast<T>(FLT_MAX);
          }
        }
      }
    }
  }

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    if (thread_row < num_rows) {
      // local argmax
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

      if (thread_group_idx == thread_to_clear_in_group) {
        int expert_to_clear_in_thread = expert % params.VPT;

#pragma unroll
        for (int v = 0; v < MAX_VPT; v++) {
          if (v < params.VPT && expert_to_clear_in_thread == v) {
            // clear the max value in the thread
            bias_chunk[v] = static_cast<T>(-FLT_MAX);
            // store output
            output_ptr[idx] = static_cast<float>(row_chunk[v]);
          }
        }
        indices_ptr[idx] = static_cast<int32_t>(expert);
      }

      __threadfence_block();
      // accumulate sum for all elements
      if (thread_group_idx == 0) {
        output_sum += output_ptr[idx];
      }
    }
  }

  if (thread_row < num_rows) {
    if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
      int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;
      int64_t expert_offset = 0;
      indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);

      // Set the weight to the sum of all weights divided by
      // routed_scaling_factor
      output_ptr[last_idx] = output_sum / routed_scaling_factor;

      if (num_fused_shared_experts > 1) {
        for (int i = 1; i < num_fused_shared_experts; ++i) {
          ++last_idx;
          ++expert_offset;
          indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
          // Set the weight to the sum of all weights divided by
          // routed_scaling_factor
          output_ptr[last_idx] = output_sum / routed_scaling_factor;
        }
      }
    }
  }
  __threadfence_block();

  ////////////////////// Rescale Output //////////////////////
  if (thread_row < num_rows) {
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
}

template <typename T, typename Params, int Vlen>
__device__ void moe_fused_gate_impl_static(
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
    float last_val,
    Params params) {
  using ArrayVal = AlignedArray<T, Vlen>;
  using ArrayIndex = AlignedArray<int, Vlen>;

  int tidx = threadIdx.x % (params.NUM_EXPERTS / Vlen) * Vlen;
  int tidy = threadIdx.x / (params.NUM_EXPERTS / Vlen);
  int64_t thread_row = blockIdx.x * params.ROWS_PER_CTA + tidy;

  constexpr int NR_EXPERTS = params.NUM_EXPERTS;
  constexpr int NR_ROWS_PER_CTA = params.ROWS_PER_CTA;
  constexpr int NR_EXPERT_GRPS = params.NUM_EXPERTS / params.VPT;
  constexpr int NR_EXPERT_PER_GRP = params.VPT;
  constexpr int NR_THREADS_PER_GRP = NR_EXPERT_PER_GRP / Vlen;
  __shared__ int smem_grp_flag[NR_ROWS_PER_CTA * NR_EXPERT_GRPS];
  __shared__ float smem_grp_max_sum[NR_ROWS_PER_CTA * NR_EXPERT_GRPS];
  __shared__ T smem_score[NR_ROWS_PER_CTA * NR_EXPERTS];
  __shared__ int smem_idx[NR_ROWS_PER_CTA * NR_EXPERTS];
  __shared__ T smem_bias[NR_EXPERTS];

  static_assert(Vlen <= NR_EXPERT_PER_GRP);

  // Calculate topk_excluding_share_expert_fusion from topk
  int topk_excluding_share_expert_fusion = topk - num_fused_shared_experts;

  // Cast pointers to type T:
  auto* input_ptr = reinterpret_cast<T*>(input);
  auto* bias_ptr = reinterpret_cast<T*>(bias);
  auto* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int grp_idx = tidx / NR_EXPERT_PER_GRP;
  int exp_idx_in_grp = tidx % NR_EXPERT_PER_GRP;

  ArrayVal row_chunk;
  ArrayVal bias_chunk;
  ArrayIndex idx_chunk;
  if (thread_row < num_rows) {
    row_chunk = *(ArrayVal*)(thread_row_ptr + tidx);
    bias_chunk = *(ArrayVal*)(bias_ptr + tidx);
  }

#pragma unroll
  for (int v = 0; v < Vlen; v++) {
    ////////////////////// Sigmoid //////////////////////
    row_chunk[v] = static_cast<T>(fast_rcpf(1.0f + fast_expf(-float(row_chunk[v]))));
    if (tidy == 0) {
      smem_bias[tidx + v] = bias_chunk[v];
    }
    bias_chunk[v] = row_chunk[v] + bias_chunk[v];
    idx_chunk[v] = tidx + v;
  }

  int max_idx = exp_idx_in_grp;
  T max_val = bias_chunk[0];
  float max_sum = 0.f;

  ////////////////////// top 1 //////////////////////
#pragma unroll
  for (int v = 1; v < Vlen; v++) {
    // per-thread max
    if (bias_chunk[v] > max_val) {
      max_val = bias_chunk[v];
      max_idx = exp_idx_in_grp + v;
    }
  }
#pragma unroll
  for (int mask = NR_THREADS_PER_GRP / 2; mask > 0; mask /= 2) {
    T peer_max_val = static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, NR_THREADS_PER_GRP));
    int peer_idx = __shfl_xor_sync(0xFFFFFFFF, max_idx, mask, NR_THREADS_PER_GRP);
    if (cmp_gt(peer_max_val, max_val)) {
      max_val = peer_max_val;
      max_idx = peer_idx;
    }
  }
  int top1_max_idx = __shfl_sync(0xFFFFFFFF, static_cast<float>(max_idx), 0, NR_THREADS_PER_GRP);
  max_sum += max_val;

  ////////////////////// top 2 //////////////////////
  max_val = static_cast<T>(-FLT_MAX);
  for (int v = 0; v < Vlen; v++) {
    // per-thread reset
    if (bias_chunk[v] > max_val && exp_idx_in_grp + v != top1_max_idx) {
      max_val = bias_chunk[v];
    }
  }
#pragma unroll
  for (int mask = NR_THREADS_PER_GRP / 2; mask > 0; mask /= 2) {
    T peer_max_val = static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, NR_THREADS_PER_GRP));
    if (cmp_gt(peer_max_val, max_val)) {
      max_val = peer_max_val;
    }
  }
  max_sum += max_val;

  ////////////////////// sort groups by max_sum //////////////////////
  if (exp_idx_in_grp == 0) {
    smem_grp_max_sum[tidy * NR_EXPERT_GRPS + grp_idx] = max_sum;
    smem_grp_flag[tidy * NR_EXPERT_GRPS + grp_idx] = grp_idx;
  }
  __syncthreads_lm();
  int cur_grp_rank = 0;
  if (exp_idx_in_grp == 0) {
    float cur_grp_max = max_sum;
#pragma unroll
    for (int i = 0; i < NR_EXPERT_GRPS; i++) {
      float other_grp_max = smem_grp_max_sum[tidy * NR_EXPERT_GRPS + i];
      int other_grp_idx = smem_grp_flag[tidy * NR_EXPERT_GRPS + i];
      if (cmp_ge(cur_grp_max, other_grp_max, grp_idx, other_grp_idx)) {
        cur_grp_rank++;
      }
    }
  }
  __syncthreads_lm();
  if (exp_idx_in_grp == 0) {
    smem_grp_flag[tidy * NR_EXPERT_GRPS + grp_idx] = cur_grp_rank;
  }
  __syncthreads_lm();

  ////////////////////// TopK experts //////////////////////
  cur_grp_rank = smem_grp_flag[tidy * NR_EXPERT_GRPS + grp_idx];

#pragma unroll
  for (int v = 0; v < Vlen; v++) {
    if (cur_grp_rank >= topk_group) {
      bias_chunk[v] = static_cast<T>(-FLT_MAX);
    }
  }

  float output_sum = 0.f;
  for (int i = 0; i < topk_excluding_share_expert_fusion; i++) {
    T thread_max_val = static_cast<T>(-FLT_MAX);
    int thread_max_idx = idx_chunk[0];
#pragma unroll
    for (int v = 0; v < Vlen; v++) {
      if (bias_chunk[v] > thread_max_val) {
        thread_max_val = bias_chunk[v];
        thread_max_idx = idx_chunk[v];
      }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
      T peer_max_val = static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(thread_max_val), mask, WARP_SIZE));
      int peer_idx = __shfl_xor_sync(0xFFFFFFFF, thread_max_idx, mask, WARP_SIZE);
      if (cmp_ge(thread_max_val, peer_max_val, thread_max_idx, peer_idx)) {
        thread_max_val = peer_max_val;
        thread_max_idx = peer_idx;
      }
    }
    int warp_max_idx = __shfl_sync(0xFFFFFFFF, thread_max_idx, 0, WARP_SIZE);

    if (tidx == 0) {
      // restore row_chunk
      float restored_val = (float)thread_max_val - (float)smem_bias[thread_max_idx];
      output_sum += restored_val;
      smem_score[tidy * NR_EXPERTS + i] = (T)restored_val;
      smem_idx[tidy * NR_EXPERTS + i] = thread_max_idx;
    }

#pragma unroll
    for (int v = 0; v < Vlen; v++) {
      if (warp_max_idx == idx_chunk[v]) {
        bias_chunk[v] = static_cast<T>(-FLT_MAX);
      }
    }
  }

  __syncthreads_lm();
  output_sum = __shfl_sync(0xFFFFFFFF, output_sum, 0, WARP_SIZE);

  ////////////////////// store output //////////////////////
  int64_t out_idx = thread_row * topk;
  int tid_st_x = threadIdx.x % WARP_SIZE;
  if (thread_row < num_rows) {
    for (int i = tid_st_x; i < topk_excluding_share_expert_fusion; i += WARP_SIZE) {
      float output_val = smem_score[tidy * NR_EXPERTS + i] * fast_rcpf(output_sum);
      if (apply_routed_scaling_factor_on_output) {
        output_val *= routed_scaling_factor;
      }
      output_ptr[out_idx + i] = output_val;
      indices_ptr[out_idx + i] = smem_idx[tidy * NR_EXPERTS + i];
    }
  }

  ////////////////////// handle shared experts //////////////////////
  if (thread_row < num_rows && tidx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = thread_row * topk + topk_excluding_share_expert_fusion;
    int64_t expert_offset = 0;
    // Set the weight to the sum of all weights divided by routed_scaling_factor
    indices_ptr[last_idx] = static_cast<int32_t>(NR_EXPERTS + expert_offset);
    output_ptr[last_idx] = last_val;

    if (num_fused_shared_experts > 1) {
      for (int i = 1; i < num_fused_shared_experts; ++i) {
        ++last_idx;
        ++expert_offset;
        indices_ptr[last_idx] = static_cast<int32_t>(NR_EXPERTS + expert_offset);
        output_ptr[last_idx] = last_val;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Templated Kernel Version (using compile-time constants)
//------------------------------------------------------------------------------
template <int VPT_, int NUM_EXPERTS_, int ROWS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
};

template <typename T, int VPT, int NUM_EXPERTS, int ROWS_PER_CTA, int Vlen>
__global__ void moe_fused_gate_kernel_static(
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
    float last_val) {
  KernelParams<VPT, NUM_EXPERTS, ROWS_PER_CTA> params;
  moe_fused_gate_impl_static<T, KernelParams<VPT, NUM_EXPERTS, ROWS_PER_CTA>, Vlen>(
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
      last_val,
      params);
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_MOE_GATE_CONFIG(T, EXPERTS, EXPERT_GROUP)                                                       \
  do {                                                                                                         \
    constexpr int vlen = EXPERTS / WARP_SIZE;                                                                  \
    int block_x = num_experts / vlen;                                                                          \
    int block_y = block_size / block_x;                                                                        \
    int64_t num_blocks = (num_rows + block_y - 1) / block_y;                                                   \
    dim3 block_dim(block_size, 1, 1);                                                                          \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                            \
    constexpr int ROWS_PER_CTA = block_size / (EXPERTS / vlen);                                                \
    moe_fused_gate_kernel_static<T, VPT, (EXPERTS), ROWS_PER_CTA, vlen><<<num_blocks, block_dim, 0, stream>>>( \
        input.data_ptr(),                                                                                      \
        bias.data_ptr(),                                                                                       \
        output.data_ptr<float>(),                                                                              \
        indices.data_ptr<int32_t>(),                                                                           \
        num_rows,                                                                                              \
        topk_group,                                                                                            \
        topk,                                                                                                  \
        num_fused_shared_experts,                                                                              \
        routed_scaling_factor,                                                                                 \
        apply_routed_scaling_factor_on_output,                                                                 \
        last_val);                                                                                             \
    dispatched = true;                                                                                         \
  } while (0);

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
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  KernelParamsDynamic params;
  params.NUM_EXPERTS = num_experts;             // e.g, for deepseek v3, this is 256
  params.VPT = num_experts / num_expert_group;  // e.g., for deepseek v3, this is 256 / 8 = 32
  params.THREADS_PER_ROW = num_expert_group;    // fixed as num_expert_group, e.g., for deepseek v3,
                                                // this is 8
  params.WARPS_PER_CTA = WARPS_PER_CTA;         // fixed as 6
  params.ROWS_PER_WARP = std::max<int64_t>(1, WARP_SIZE / num_expert_group);  // WARP_SIZE is fixed as 32
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  moe_fused_gate_impl_dynamic<T>(
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

void dispatch_moe_fuse_gate_dynamic(
    at::Tensor& output,
    at::Tensor& indices,
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  // Compute grid dimensions based on runtime value for num_expert_group.
  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
  const musaStream_t stream = at::musa::getCurrentMUSAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Fallback to the dynamic kernel if none of the supported combinations match.
  // currently only support num_experts / num_expert_group <= 32 for dynamic
  // kernels
  if (input.scalar_type() == at::kBFloat16) {
    moe_fused_gate_kernel_dynamic<bfloat16_t><<<num_blocks, block_dim, 0, stream>>>(
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
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kHalf) {
    moe_fused_gate_kernel_dynamic<float16_t><<<num_blocks, block_dim, 0, stream>>>(
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
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else if (input.scalar_type() == at::kFloat) {
    moe_fused_gate_kernel_dynamic<float32_t><<<num_blocks, block_dim, 0, stream>>>(
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
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  } else {
    TORCH_CHECK(false, "Unsupported data type for moe_fused_gate");
  }
}

bool dispatch_moe_fuse_gate_static(
    at::Tensor& output,
    at::Tensor& indices,
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  const musaStream_t stream = at::musa::getCurrentMUSAStream();
  bool dispatched = false;
  float last_val = apply_routed_scaling_factor_on_output ? 1.f : 1.f / routed_scaling_factor;
  // Dispatch to templated kernel for known compile-time configurations.
  // We currently only support for:
  //   Case 1: 256 experts, with 8 or 16 groups.
  //   Case 2: 128 experts, with 4 or 8 groups.
  //   Case 3: other cases, require 8 <= num_experts / num_expert_group <= 32
  constexpr int block_size = 256;
  switch (num_experts) {
    case 256:
      if (num_expert_group == 8) {
        // This is deepseek v3 case. Here VPT = 256/8 = 32, ROWS_PER_WARP = 32/8
        // = 4, ROWS_PER_CTA = 6 * 4 = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 256, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 256, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 256, 8);
        }
      } else if (num_expert_group == 16) {
        //   Here VPT = 256/16 = 16, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA
        //   = 6 * 2 = 12.
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
        // VPT = 128/4 = 32, ROWS_PER_WARP = 32/16 = 2, ROWS_PER_CTA = 6 * 2
        // = 12.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 4);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 4);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 4);
        }
      } else if (num_expert_group == 8) {
        // VPT = 128/8 = 16, ROWS_PER_WARP = 32/8 = 4, ROWS_PER_CTA = 6 * 4
        //   = 24.
        if (input.scalar_type() == at::kBFloat16) {
          LAUNCH_MOE_GATE_CONFIG(bfloat16_t, 128, 8);
        } else if (input.scalar_type() == at::kHalf) {
          LAUNCH_MOE_GATE_CONFIG(float16_t, 128, 8);
        } else if (input.scalar_type() == at::kFloat) {
          LAUNCH_MOE_GATE_CONFIG(float32_t, 128, 8);
        }
      }
      break;
    default:
      break;
  }

  return dispatched;
}

#undef LAUNCH_MOE_GATE_CONFIG

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
    double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(input.dtype() == bias.dtype(), "input and bias should have the same dtype");
  int64_t num_rows = input.size(0);
  int32_t num_experts = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  // Check 1: Ensure that num_experts is a power of 2.
  TORCH_CHECK((num_experts & (num_experts - 1)) == 0, "num_experts must be a power of 2, but got ", num_experts);

  // Check 2: Ensure that num_experts is divisible by num_expert_group. (this
  // also means num_expert_group is power of 2)
  TORCH_CHECK(
      num_experts % num_expert_group == 0,
      "num_experts must be divisible by num_expert_group, but got ",
      num_experts,
      " / ",
      num_expert_group);

  int computed_vpt = num_experts / num_expert_group;
  // Check 3: Ensure that num_experts/num_expert_group does not exceed
  // MAX_VPT=32. Maximum VPT indicate max value per threads we can process.
  TORCH_CHECK(
      computed_vpt <= MAX_VPT,
      "Per group experts: num_experts / num_expert_group = (",
      computed_vpt,
      ") exceeds the maximum supported (",
      MAX_VPT,
      ")");

  bool static_dispatched = dispatch_moe_fuse_gate_static(
      output,
      indices,
      input,
      bias,
      num_rows,
      num_experts,
      num_expert_group,
      topk_group,
      topk,
      num_fused_shared_experts,
      routed_scaling_factor,
      apply_routed_scaling_factor_on_output);

  if (!static_dispatched) {
    dispatch_moe_fuse_gate_dynamic(
        output,
        indices,
        input,
        bias,
        num_rows,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
  }

  return {output, indices};
}
