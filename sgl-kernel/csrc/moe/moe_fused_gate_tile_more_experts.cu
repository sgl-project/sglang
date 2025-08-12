// add support the number of group expert from the original 32 to 128/512 implemented with tiling

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <torch/all.h>

#include <cfloat>
#include <type_traits>
#include <algorithm>

// Reuse aliases compatible with moe_fused_gate.cu
template <typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;
using bfloat16_t = cutlass::bfloat16_t;
using float16_t = cutlass::half_t;
using float32_t = float;

static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;

// Maximum experts per thread (VPT) we target via tiling
// You can raise this to 512 if needed; performance and register pressure should be reassessed.
static constexpr int MAX_TILED_VPT = 512;
static constexpr int TILE_VPT = 32; // tile size processed per thread per pass

template <typename T>
__device__ inline float to_float(T x) {
  if constexpr (std::is_same<T, float16_t>::value || std::is_same<T, at::Half>::value) {
    return static_cast<float>(x);
  } else if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, at::BFloat16>::value) {
    return static_cast<float>(x);
  } else {
    return static_cast<float>(x);
  }
}

template <typename T>
__device__ inline bool cmp_gt(const T& a, const T& b) {
  // Cast to float to avoid half comparison ambiguity
  return static_cast<float>(a) > static_cast<float>(b);
}

template <typename T>
__device__ inline bool cmp_eq(const T& a, const T& b) {
  return static_cast<float>(a) == static_cast<float>(b);
}

__device__ inline float sigmoidf_approx(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

struct KernelParamsDynamicTiled {
  int VPT;                // experts per thread (per group)
  int NUM_EXPERTS;        // total experts
  int THREADS_PER_ROW;    // num_expert_group
  int ROWS_PER_WARP;      // max(1, 32 / num_expert_group)
  int ROWS_PER_CTA;       // WARPS_PER_CTA * ROWS_PER_WARP
  int WARPS_PER_CTA;      // fixed as 6
};

// Argmin reduce across THREADS_PER_ROW lanes within a warp-subgroup
__device__ inline void warp_argmin_pair(float &val, int &idx, int width) {
  // width must be power of two and <= 32
  unsigned mask = 0xFFFFFFFFu;
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    float other_val = __shfl_xor_sync(mask, val, offset, width);
    int other_idx = __shfl_xor_sync(mask, idx, offset, width);
    // Keep the larger (worse) index on tie to follow original behavior
    if (val > other_val || (val == other_val && other_idx > idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// Argmax reduce across THREADS_PER_ROW lanes within a warp-subgroup
__device__ inline void warp_argmax_pair(float &val, int &idx, int width) {
  unsigned mask = 0xFFFFFFFFu;
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    float other_val = __shfl_xor_sync(mask, val, offset, width);
    int other_idx = __shfl_xor_sync(mask, idx, offset, width);
    // Keep the smaller index on tie to follow original behavior
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename T, int TILE>
__global__ void moe_fused_gate_kernel_tiled(
    const void* __restrict__ input,
    const void* __restrict__ bias,
    float* __restrict__ output_ptr,
    int32_t* __restrict__ indices_ptr,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  KernelParamsDynamicTiled params;
  params.NUM_EXPERTS = static_cast<int>(num_experts);
  params.THREADS_PER_ROW = static_cast<int>(num_expert_group);
  params.VPT = static_cast<int>(num_experts / num_expert_group);
  params.WARPS_PER_CTA = WARPS_PER_CTA;
  params.ROWS_PER_WARP = max(1, WARP_SIZE / params.THREADS_PER_ROW);
  params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;

  int lane = threadIdx.x;                // 0..31
  int warp_id = threadIdx.y;             // 0..WARPS_PER_CTA-1
  int thread_group_idx = lane % params.THREADS_PER_ROW;   // lane within group-subwarp
  int row_in_warp = lane / params.THREADS_PER_ROW;        // which row this lane processes inside the warp

  int64_t thread_row = static_cast<int64_t>(blockIdx.x) * params.ROWS_PER_CTA +
                       static_cast<int64_t>(warp_id) * params.ROWS_PER_WARP +
                       static_cast<int64_t>(row_in_warp);
  if (thread_row >= num_rows) return;

  const T* __restrict__ input_ptr = reinterpret_cast<const T*>(input);
  const T* __restrict__ bias_ptr = reinterpret_cast<const T*>(bias);

  int VPT = params.VPT;
  int first_elt_read_by_thread = thread_group_idx * VPT; // base expert index within the row handled by this lane

  // Phase 1: exclude worst groups iteratively until only topk_group groups remain
  bool group_active = true;
  int groups_to_exclude = params.THREADS_PER_ROW - static_cast<int>(topk_group);
  for (int iter = 0; iter < groups_to_exclude; ++iter) {
    float local_best = -FLT_MAX;
    float local_second = -FLT_MAX;

    if (group_active) {
      // Scan tiles to compute top-2 within this group's experts
      for (int base = 0; base < VPT; base += TILE) {
        int tile_elems = min(TILE, VPT - base);
        int64_t row_offset = thread_row * params.NUM_EXPERTS + first_elt_read_by_thread + base;
#pragma unroll
        for (int ii = 0; ii < tile_elems; ++ii) {
          float s = sigmoidf_approx(to_float(input_ptr[row_offset + ii]));
          float v = s + to_float(bias_ptr[first_elt_read_by_thread + base + ii]);
          if (v > local_best) {
            local_second = local_best;
            local_best = v;
          } else if (v > local_second) {
            local_second = v;
          }
        }
      }
    }

    float group_score = group_active ? (local_best + local_second) : FLT_MAX; // argmin
    int marker = first_elt_read_by_thread; // used to recover the group index via marker / VPT

    // argmin across lanes in this row-subwarp
    warp_argmin_pair(group_score, marker, params.THREADS_PER_ROW);

    int thread_to_exclude = marker / VPT;
    if (thread_group_idx == thread_to_exclude) {
      group_active = false;
    }
    __syncwarp();
  }

  // Phase 2: pick topk_excluding_shared experts from remaining groups
  int topk_excl_shared = static_cast<int>(topk - num_fused_shared_experts);
  float output_sum = 0.0f;

  // Track per-thread selected local indices to avoid picking them again
  // Upper bound: one thread may win many times theoretically, but in practice topk is small.
  const int MAX_LOCAL_SELECT = 64; // safe upper bound for routing top-k
  int selected_count = 0;
  // Implement local selected indices container
  int selected_local_indices[MAX_LOCAL_SELECT];

  for (int k_idx = 0; k_idx < topk_excl_shared; ++k_idx) {
    float local_max_val = -FLT_MAX;
    int local_best_global = -1;

    if (group_active) {
      // Scan tiles and find local best candidate skipping previously selected indices
      for (int base = 0; base < VPT; base += TILE) {
        int tile_elems = min(TILE, VPT - base);
        int64_t row_offset = thread_row * params.NUM_EXPERTS + first_elt_read_by_thread + base;
#pragma unroll
        for (int ii = 0; ii < tile_elems; ++ii) {
          int local_idx = base + ii;
          bool skip = false;
          for (int t = 0; t < selected_count; ++t) {
            if (selected_local_indices[t] == local_idx) { skip = true; break; }
          }
          if (skip) continue;

          float s = sigmoidf_approx(to_float(input_ptr[row_offset + ii]));
          float v = s + to_float(bias_ptr[first_elt_read_by_thread + local_idx]);
          if (v > local_max_val) {
            local_max_val = v;
            local_best_global = first_elt_read_by_thread + local_idx;
          }
        }
      }
    }

    // Perform argmax across lanes (within group-subwarp)
    int winner_global = local_best_global;
    float winner_val = local_max_val;
    warp_argmax_pair(winner_val, winner_global, params.THREADS_PER_ROW);

    int winner_thread = (winner_global >= 0) ? (winner_global / VPT) : -1;
    int64_t out_idx = topk * thread_row + k_idx;
    if (winner_thread >= 0 && thread_group_idx == winner_thread) {
      int expert_local = winner_global % VPT;
      // Record local selection to avoid picking it again later
      if (selected_count < MAX_LOCAL_SELECT) {
        selected_local_indices[selected_count++] = expert_local;
      }

      // Write outputs (weight uses sigmoid only, index uses absolute expert id)
      int64_t row_elem_base = thread_row * params.NUM_EXPERTS + first_elt_read_by_thread + expert_local;
      float weight = sigmoidf_approx(to_float(input_ptr[row_elem_base]));
      output_ptr[out_idx] = weight;
      indices_ptr[out_idx] = static_cast<int32_t>(winner_global);
    }

    // Accumulate output sum once per row-subwarp
    if (thread_group_idx == 0) {
      // Note: winner thread writes before this read in same warp; safe within warp
      output_sum += output_ptr[out_idx];
    }
    __syncwarp();
  }

  // Append fused shared experts if requested
  if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
    int64_t last_idx = topk * thread_row + topk_excl_shared;
    int64_t expert_offset = 0;
    indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
    output_ptr[last_idx] = output_sum / static_cast<float>(routed_scaling_factor);

    for (int i = 1; i < num_fused_shared_experts; ++i) {
      ++last_idx;
      ++expert_offset;
      indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);
      output_ptr[last_idx] = output_sum / static_cast<float>(routed_scaling_factor);
    }
  }
  __syncwarp();

  // Renormalize by the sum of real experts
  if (thread_group_idx == 0) {
    float denom = output_sum;
    for (int i = 0; i < topk; ++i) {
      int64_t idx = topk * thread_row + i;
      output_ptr[idx] = output_ptr[idx] / denom;
    }
  }
}

// ----------------------------------------------------------------------------
// Static tiled kernel template (compile-time params). Currently supports
// THREADS_PER_ROW == 1 path (no warp reductions). Can be extended later.
// ----------------------------------------------------------------------------
template <typename T,
          int NUM_EXPERTS,
          int THREADS_PER_ROW,
          int ROWS_PER_WARP,
          int ROWS_PER_CTA,
          int WARPS_PER_CTA_,
          int TILE>
__global__ void moe_fused_gate_kernel_tiled_static(
    const void* __restrict__ input,
    const void* __restrict__ bias,
    float* __restrict__ output_ptr,
    int32_t* __restrict__ indices_ptr,
    int64_t num_rows,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  static_assert(THREADS_PER_ROW == 1, "static tiled kernel currently supports THREADS_PER_ROW==1 only");
  (void)WARPS_PER_CTA_;

  int lane = threadIdx.x;     // 0..31
  int warp_id = threadIdx.y;  // 0..WARPS_PER_CTA-1
  int row_in_warp = lane;     // THREADS_PER_ROW=1 => row_in_warp = lane

  int64_t thread_row = static_cast<int64_t>(blockIdx.x) * ROWS_PER_CTA +
                       static_cast<int64_t>(warp_id) * ROWS_PER_WARP +
                       static_cast<int64_t>(row_in_warp);
  if (thread_row >= num_rows) return;

  const T* __restrict__ input_ptr = reinterpret_cast<const T*>(input);
  const T* __restrict__ bias_ptr = reinterpret_cast<const T*>(bias);

  int64_t row_base = thread_row * NUM_EXPERTS;
  int topk_excl_shared = static_cast<int>(topk - num_fused_shared_experts);
  topk_excl_shared = max(0, topk_excl_shared);
  const int MAX_TOPK = 32;
  topk_excl_shared = min(topk_excl_shared, MAX_TOPK);

  float best_choice[MAX_TOPK];
  float best_weight[MAX_TOPK];
  int   best_index[MAX_TOPK];
#pragma unroll
  for (int i = 0; i < MAX_TOPK; ++i) {
    best_choice[i] = -FLT_MAX;
    best_weight[i] = 0.0f;
    best_index[i] = -1;
  }

  for (int e = 0; e < NUM_EXPERTS; ++e) {
    float s = sigmoidf_approx(to_float(input_ptr[row_base + e]));
    float choice = s + to_float(bias_ptr[e]);
    if (choice > best_choice[topk_excl_shared - 1]) {
      int j = topk_excl_shared - 1;
      while (j > 0 && choice > best_choice[j - 1]) {
        best_choice[j] = best_choice[j - 1];
        best_weight[j] = best_weight[j - 1];
        best_index[j] = best_index[j - 1];
        --j;
      }
      best_choice[j] = choice;
      best_weight[j] = s;
      best_index[j] = e;
    }
  }

  float real_sum = 0.0f;
  for (int k = 0; k < topk_excl_shared; ++k) {
    int64_t out_idx = topk * thread_row + k;
    output_ptr[out_idx] = best_weight[k];
    indices_ptr[out_idx] = static_cast<int32_t>(best_index[k]);
    real_sum += best_weight[k];
  }

  if (num_fused_shared_experts > 0) {
    float shared_w = (real_sum == 0.0f) ? 0.0f : (real_sum / static_cast<float>(routed_scaling_factor));
    int64_t base = topk * thread_row + topk_excl_shared;
    for (int i = 0; i < num_fused_shared_experts; ++i) {
      indices_ptr[base + i] = static_cast<int32_t>(NUM_EXPERTS + i);
      output_ptr[base + i] = shared_w;
    }
  }

  if (real_sum > 0.0f) {
    for (int i = 0; i < topk; ++i) {
      int64_t out_idx = topk * thread_row + i;
      output_ptr[out_idx] = output_ptr[out_idx] / real_sum;
    }
  }
}

// Public dispatcher for static tiled instantiations
std::vector<at::Tensor> moe_fused_gate_tiled_static(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
  int64_t num_rows = input.size(0);
  int64_t num_experts = input.size(1);

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Currently: specialize THREADS_PER_ROW=1 for selected NUM_EXPERTS with TILE=32
  if (num_experts == 384 && num_expert_group == 1) {
    constexpr int THREADS_PER_ROW = 1;
    constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    int64_t rows_per_warp = ROWS_PER_WARP;
    int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
    int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

    if (input.scalar_type() == at::kBFloat16) {
      moe_fused_gate_kernel_tiled_static<bfloat16_t, 384, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else if (input.scalar_type() == at::kHalf) {
      moe_fused_gate_kernel_tiled_static<float16_t, 384, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else if (input.scalar_type() == at::kFloat) {
      moe_fused_gate_kernel_tiled_static<float32_t, 384, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else {
      TORCH_CHECK(false, "Unsupported dtype for moe_fused_gate_tiled_static");
    }
  } else if (num_experts == 64 && num_expert_group == 1) {
    constexpr int THREADS_PER_ROW = 1;
    constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    int64_t rows_per_warp = ROWS_PER_WARP;
    int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
    int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

    if (input.scalar_type() == at::kBFloat16) {
      moe_fused_gate_kernel_tiled_static<bfloat16_t, 64, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else if (input.scalar_type() == at::kHalf) {
      moe_fused_gate_kernel_tiled_static<float16_t, 64, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else if (input.scalar_type() == at::kFloat) {
      moe_fused_gate_kernel_tiled_static<float32_t, 64, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA, TILE_VPT>
          <<<num_blocks, block_dim, 0, stream>>>(
              input.data_ptr(), bias.data_ptr(), output.data_ptr<float>(), indices.data_ptr<int32_t>(),
              num_rows, topk, num_fused_shared_experts, routed_scaling_factor);
    } else {
      TORCH_CHECK(false, "Unsupported dtype for moe_fused_gate_tiled_static");
    }
  } else {
    TORCH_CHECK(false, "moe_fused_gate_tiled_static: unsupported combination");
  }

  return {output, indices};
}

// Host launcher for tiled kernel. This does not auto-wire into the existing `moe_fused_gate` API.
// Call this from the dispatcher when VPT > 32 and <= MAX_TILED_VPT.
std::vector<at::Tensor> moe_fused_gate_tiled(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
  int64_t num_rows = input.size(0);
  int64_t num_experts = input.size(1);

  TORCH_CHECK(
      ((num_experts & (num_experts - 1)) == 0) || (num_experts == 384),
      "num_experts must be power of 2 or 384 (Kimi K2)");
  TORCH_CHECK(num_experts % num_expert_group == 0, "num_experts must be divisible by num_expert_group");
  int64_t VPT = num_experts / num_expert_group;
  TORCH_CHECK(VPT <= MAX_TILED_VPT, "VPT exceeds MAX_TILED_VPT=", MAX_TILED_VPT);
  TORCH_CHECK(num_expert_group <= WARP_SIZE, "num_expert_group must be <= 32");

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto output = torch::empty({num_rows, topk}, options);
  auto indices = torch::empty({num_rows, topk}, options.dtype(torch::kInt32));

  int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
  int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
  int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

  // Launch tiled dynamic kernel by dtype
  if (input.scalar_type() == at::kBFloat16) {
    moe_fused_gate_kernel_tiled<bfloat16_t, TILE_VPT><<<num_blocks, block_dim, 0, stream>>>(
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
    moe_fused_gate_kernel_tiled<float16_t, TILE_VPT><<<num_blocks, block_dim, 0, stream>>>(
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
    moe_fused_gate_kernel_tiled<float32_t, TILE_VPT><<<num_blocks, block_dim, 0, stream>>>(
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
    TORCH_CHECK(false, "Unsupported data type for moe_fused_gate_tiled");
  }

  return {output, indices};
}

