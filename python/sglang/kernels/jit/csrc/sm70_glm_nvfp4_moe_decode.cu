// SPDX-License-Identifier: Apache-2.0
// Small-batch NVFP4 MoE decode for GLM-5.3-Flash on Volta (SM70).
//
// Same Marlin-weight streaming as the Qwen3.8 kernel, retargeted to this
// model's decode shape: top-8, hidden 4096, intermediate 2048.  Split-K is
// chosen so each split divides the group count (256 and 128).  Dot products
// use native half2 FMA; FP32 reductions fuse SwiGLU and the route weights.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>

namespace sm70_nvfp4_moe {

// Only the unused Qwen top-k kernels below still mention this. The GEMM
// bounds-checks against the weight tensor's expert count.
constexpr int kExperts = 288;
constexpr int kTopK = 8;
constexpr int kHidden = 4096;
constexpr int kIntermediate = 2048;
constexpr int kGateUp = 2 * kIntermediate;
constexpr int kGroupSize = 16;
// 4096/16 = 256 groups, 2048/16 = 128 groups. Split wide enough that one
// local expert still fills the SMs; the route loop is inside the thread
// so masked experts do not launch empty blocks.
constexpr int kSplitK = 64;
constexpr int kDownSplitK = 64;
constexpr int kThreads = 128;
static_assert((kHidden / kGroupSize) % kSplitK == 0, "gate split must tile K");
static_assert((kIntermediate / kGroupSize) % kDownSplitK == 0,
              "down split must tile K");
constexpr float kMarlinScaleCompensation = 1.0f;

__device__ __forceinline__ void warp_argmax(float& value, int& index) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_value > value ||
        (other_value == value && other_index < index)) {
      value = other_value;
      index = other_index;
    }
  }
}

template <typename T>
__device__ __forceinline__ float load_logit(const T* logits, int index) {
  return static_cast<float>(logits[index]);
}

template <>
__device__ __forceinline__ float load_logit<__half>(const __half* logits,
                                                     int index) {
  return __half2float(logits[index]);
}

template <typename T>
__global__ __launch_bounds__(256, 1)
void topk10_softmax_kernel(const T* __restrict__ logits,
                           float* __restrict__ topk_weights,
                           int* __restrict__ topk_ids) {
  // Eight warps independently retain the best ten of 64 logits. Warp zero
  // then retains the best ten of those 80 candidates. Softmax over the final
  // ten is exactly softmax(all logits) followed by top-k renormalization.
  __shared__ float candidate_values[80];
  __shared__ int candidate_ids[80];
  __shared__ float selected_values[kTopK];
  const int row = static_cast<int>(blockIdx.x);
  logits += row * kExperts;
  topk_weights += row * kTopK;
  topk_ids += row * kTopK;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int first_id = warp * 64 + lane;
  const int second_id = first_id + 32;
  float first = load_logit(logits, first_id);
  float second = load_logit(logits, second_id);

#pragma unroll
  for (int pick = 0; pick < kTopK; ++pick) {
    float best = first;
    int best_id = first_id;
    if (second > best || (second == best && second_id < best_id)) {
      best = second;
      best_id = second_id;
    }
    warp_argmax(best, best_id);
    const int winner = __shfl_sync(0xffffffffu, best_id, 0);
    if (lane == 0) {
      candidate_values[warp * kTopK + pick] = best;
      candidate_ids[warp * kTopK + pick] = best_id;
    }
    if (first_id == winner) {
      first = -FLT_MAX;
    }
    if (second_id == winner) {
      second = -FLT_MAX;
    }
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }
  float values[3];
  int ids[3];
#pragma unroll
  for (int item = 0; item < 3; ++item) {
    const int candidate = lane + item * 32;
    values[item] =
        candidate < 80 ? candidate_values[candidate] : -FLT_MAX;
    ids[item] = candidate < 80 ? candidate_ids[candidate] : INT_MAX;
  }
#pragma unroll
  for (int pick = 0; pick < kTopK; ++pick) {
    float best = values[0];
    int best_id = ids[0];
#pragma unroll
    for (int item = 1; item < 3; ++item) {
      if (values[item] > best ||
          (values[item] == best && ids[item] < best_id)) {
        best = values[item];
        best_id = ids[item];
      }
    }
    warp_argmax(best, best_id);
    const int winner = __shfl_sync(0xffffffffu, best_id, 0);
    if (lane == 0) {
      selected_values[pick] = best;
      topk_ids[pick] = best_id;
    }
#pragma unroll
    for (int item = 0; item < 3; ++item) {
      if (ids[item] == winner) {
        values[item] = -FLT_MAX;
      }
    }
  }
  __syncwarp();
  float weight = lane < kTopK
                     ? __expf(selected_values[lane] - selected_values[0])
                     : 0.0f;
  float sum = weight;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  sum = __shfl_sync(0xffffffffu, sum, 0);
  if (lane < kTopK) {
    topk_weights[lane] = weight / sum;
  }
}

__global__ void topk10_radix_half(const half* __restrict__ logits,
                                  float* __restrict__ weights,
                                  int* __restrict__ ids) {
  // CUB's stable sort preserves lower-ID ties when the initial blocked order
  // is expert-ID order. Carry the ID as a 16-bit value instead of adding nine
  // sort-key bits: three six-bit passes suffice for all FP16 logits.
  using Sort = cub::BlockRadixSort<uint16_t, 128, 512 / 128, uint16_t, 6>;
  __shared__ typename Sort::TempStorage temp;
  logits += blockIdx.x * 512;
  weights += blockIdx.x * 10;
  ids += blockIdx.x * 10;
  uint16_t keys[512 / 128], values[512 / 128];
#pragma unroll
  for (int j = 0; j < 512 / 128; ++j) {
    int id = threadIdx.x * (512 / 128) + j;
    uint32_t b = __half_as_ushort(logits[id]);
    if ((b & 0x7fff) == 0) b = 0;
    uint32_t ordered = (b & 0x8000) ? (~b & 0xffff) : (b ^ 0x8000);
    keys[j] = ordered;
    values[j] = id;
  }
  Sort(temp).SortDescendingBlockedToStriped(keys, values, 0, 16);
  if (threadIdx.x < 32) {
    int id = values[0];
    float v = __half2float(logits[id]);
    float mx = __shfl_sync(0xffffffff, v, 0);
    float w = threadIdx.x < 10 ? __expf(v - mx) : 0.f;
    float sum = w;
#pragma unroll
    for (int s = 16; s > 0; s /= 2) sum += __shfl_down_sync(0xffffffff, sum, s);
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (threadIdx.x < 10) {
      ids[threadIdx.x] = id;
      weights[threadIdx.x] = w / sum;
    }
  }
}

__device__ __constant__ int kScaleLogicalToStored[8] = {0, 2, 1, 3,
                                                        4, 6, 5, 7};

__device__ __forceinline__ void dequant_fp4x8(uint32_t packed,
                                              __half2* values) {
  // Expand even/odd nibbles once, then interleave zero bytes into four half2
  // values with native PRMT instructions. This preserves the original packed
  // column order and Marlin's scaled FP4 representation bit for bit.
  const uint32_t even =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  packed >>= 4;
  const uint32_t odd =
      ((packed << 1) & 0x0e0e0e0eu) | ((packed << 4) & 0x80808080u);
  uint32_t result[4] = {
      __byte_perm(even, 0, 0x2404), __byte_perm(odd, 0, 0x2404),
      __byte_perm(even, 0, 0x3414), __byte_perm(odd, 0, 0x3414)};
#pragma unroll
  for (int p = 0; p < 4; ++p)
    values[p] = *reinterpret_cast<__half2*>(&result[p]);
}

// SGLang's SM70 Marlin preprocessing encodes a non-negative scale s as a
// byte whose bits become the FP16 representation of s*128 after << 7.
__device__ __forceinline__ __half2 load_scale_pair(
    const uint8_t* encoded, int logical0, int logical1) {
  return __halves2half2(
      __ushort_as_half(static_cast<uint16_t>(encoded[logical0]) << 7),
      __ushort_as_half(static_cast<uint16_t>(encoded[logical1]) << 7));
}

// Packed scales have eight adjacent bytes per qword. Load the whole group
// once and interleave bytes directly into the original FP16 scale pairs.
// The scalar specialization retains support for unaligned metadata views.
template <bool VectorScales>
__device__ __forceinline__ void load_scales8(const uint8_t* encoded, half2* out) {
  if constexpr (VectorScales) {
    const uint2 raw = *reinterpret_cast<const uint2*>(encoded);
    const uint32_t bits[4] = {
        __byte_perm(raw.x, 0, 0x4240) << 7,
        __byte_perm(raw.x, 0, 0x4341) << 7,
        __byte_perm(raw.y, 0, 0x4240) << 7,
        __byte_perm(raw.y, 0, 0x4341) << 7};
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = *reinterpret_cast<const half2*>(&bits[p]);
  } else {
#pragma unroll
    for (int p = 0; p < 4; ++p)
      out[p] = load_scale_pair(encoded, kScaleLogicalToStored[2 * p],
                              kScaleLogicalToStored[2 * p + 1]);
  }
}

template <bool VectorScales>
__global__ void __launch_bounds__(kThreads, 2)
gate_up_partial_kernel(const __half* __restrict__ input,
                       const uint32_t* __restrict__ weight,
                       const uint8_t* __restrict__ scales,
                       const int* __restrict__ topk_ids,
                       int num_routes,
                       int num_experts,
                       float* __restrict__ partials) {
  constexpr int kQwords = kGateUp / 8;
  constexpr int kGroups = kHidden / kGroupSize;
  constexpr int kGroupsPerSplit = kGroups / kSplitK;
  constexpr int kExpertWords = (kHidden / 16) * (kGateUp * 2);

  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = kSplitK * kQwords;
  if (work >= kTotal) {
    return;
  }
  const int qword = work % kQwords;
  const int split = work / kQwords;
  const int n_base = qword * 8;
  const int group_n_tile = qword >> 5;
  const int subtile = (qword >> 3) & 3;
  const int qword_in_tile = qword & 7;
  const int group_begin = split * kGroupsPerSplit;

  for (int route = 0; route < num_routes; ++route) {
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    const int token = route / kTopK;
    const uint32_t* expert_weight =
        weight + static_cast<int64_t>(expert) * kExpertWords;
    const uint8_t* expert_scales =
        scales + static_cast<int64_t>(expert) * kGroups * kGateUp;
    // FP32 accumulation. FP16 FMAs over K=4096 move a close argmax after
    // the 42 MoE layers; the extra registers do not change the weight traffic.
    float accum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int group_it = 0; group_it < kGroupsPerSplit; ++group_it) {
      const int group = group_begin + group_it;
      __half2 scale[4];
      load_scales8<VectorScales>(expert_scales + group * kGateUp + n_base, scale);
#pragma unroll
      for (int r = 0; r < kGroupSize; ++r) {
        const int k = group * kGroupSize + r;
        const float x = __half2float(input[token * kHidden + k]);
        // N is a multiple of 256, so Marlin stores four 64-wide tiles
        // interleaved. The 64-wide formula is only for N that is not.
        const int offset = group * (kGateUp * 2) + group_n_tile * 512 +
                           qword_in_tile * 4 + subtile + r * 32;
        const uint32_t packed = expert_weight[offset];
        __half2 value[4];
        dequant_fp4x8(packed, value);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const __half2 term = __hmul2(scale[p], value[p]);
          accum[2 * p] = fmaf(x, __half2float(term.x), accum[2 * p]);
          accum[2 * p + 1] = fmaf(x, __half2float(term.y), accum[2 * p + 1]);
        }
      }
    }
    float* out = partials +
                 ((split * num_routes + route) * kGateUp + n_base);
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      out[p] = accum[p];
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2)
gate_up_reduce_silu_kernel(const float* __restrict__ partials,
                           const float* __restrict__ global_scales,
                           const int* __restrict__ topk_ids,
                           int num_routes,
                           int num_experts,
                           float swiglu_limit,
                           __half* __restrict__ activated) {
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = num_routes * kIntermediate;
  if (work >= kTotal) {
    return;
  }
  const int route = work / kIntermediate;
  const int n = work - route * kIntermediate;
  const int expert = topk_ids[route];
  if (expert < 0 || expert >= num_experts) {
    activated[work] = __float2half(0.0f);
    return;
  }
  float gate = 0.0f;
  float up = 0.0f;
#pragma unroll
  for (int split = 0; split < kSplitK; ++split) {
    const float* part = partials + (split * num_routes + route) * kGateUp;
    gate += part[n];
    up += part[n + kIntermediate];
  }
  const float global =
      global_scales[expert] * kMarlinScaleCompensation;
  gate *= global;
  up *= global;
  // GLM-5.3 sets swiglu_limit=10. Clamp in FP16, then SiLU, matching
  // silu_and_mul_clamp: gate = min(gate, limit), up = clamp(up, -limit, limit).
  if (swiglu_limit > 0.0f) {
    const __half limit_h = __float2half_rn(swiglu_limit);
    const __half neg_h = __float2half_rn(-swiglu_limit);
    __half gate_h = __float2half_rn(gate);
    __half up_h = __float2half_rn(up);
    gate_h = __hlt(limit_h, gate_h) ? limit_h : gate_h;
    up_h = __hlt(up_h, neg_h) ? neg_h : up_h;
    up_h = __hlt(limit_h, up_h) ? limit_h : up_h;
    gate = __half2float(gate_h);
    up = __half2float(up_h);
  }
  activated[work] = __float2half_rn((gate / (1.0f + __expf(-gate))) * up);
}

template <bool VectorScales>
__global__ void __launch_bounds__(kThreads, 2)
down_partial_kernel(const __half* __restrict__ activated,
                    const uint32_t* __restrict__ weight,
                    const uint8_t* __restrict__ scales,
                    const float* __restrict__ global_scales,
                    const int* __restrict__ topk_ids,
                    const float* __restrict__ topk_weights,
                    int num_routes,
                    int num_experts,
                    float* __restrict__ partials) {
  constexpr int kQwords = kHidden / 8;
  constexpr int kGroups = kIntermediate / kGroupSize;
  constexpr int kGroupsPerSplit = kGroups / kDownSplitK;
  constexpr int kExpertWords = (kIntermediate / 16) * (kHidden * 2);

  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  const int kTotal = kDownSplitK * kQwords;
  if (work >= kTotal) {
    return;
  }
  const int qword = work % kQwords;
  const int split = work / kQwords;
  const int n_base = qword * 8;
  const int group_n_tile = qword >> 5;
  const int subtile = (qword >> 3) & 3;
  const int qword_in_tile = qword & 7;
  const int group_begin = split * kGroupsPerSplit;

  for (int route = 0; route < num_routes; ++route) {
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    float* out = partials +
                 (route * kDownSplitK + split) * kHidden + n_base;
    const uint32_t* expert_weight =
        weight + static_cast<int64_t>(expert) * kExpertWords;
    const uint8_t* expert_scales =
        scales + static_cast<int64_t>(expert) * kGroups * kHidden;
    const __half* route_input = activated + route * kIntermediate;
    float accum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int group_it = 0; group_it < kGroupsPerSplit; ++group_it) {
      const int group = group_begin + group_it;
      __half2 scale[4];
      load_scales8<VectorScales>(expert_scales + group * kHidden + n_base, scale);
#pragma unroll
      for (int r = 0; r < kGroupSize; ++r) {
        const int k = group * kGroupSize + r;
        const float x = __half2float(route_input[k]);
        const int offset = group * (kHidden * 2) + group_n_tile * 512 +
                           qword_in_tile * 4 + subtile + r * 32;
        const uint32_t packed = expert_weight[offset];
        __half2 value[4];
        dequant_fp4x8(packed, value);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const __half2 term = __hmul2(scale[p], value[p]);
          accum[2 * p] = fmaf(x, __half2float(term.x), accum[2 * p]);
          accum[2 * p + 1] = fmaf(x, __half2float(term.y), accum[2 * p + 1]);
        }
      }
    }
    const float multiplier = global_scales[expert] *
                             kMarlinScaleCompensation * topk_weights[route];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      out[p] = accum[p] * multiplier;
    }
  }
}

__global__ void __launch_bounds__(kThreads, 2)
down_reduce_kernel(const float* __restrict__ partials,
                   const int* __restrict__ topk_ids,
                   int num_experts,
                   int batch_size,
                   __half* __restrict__ output) {
  const int work = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (work >= batch_size * kHidden) {
    return;
  }
  const int token = work / kHidden;
  const int n = work - token * kHidden;
  float sum = 0.0f;
#pragma unroll
  for (int token_route = 0; token_route < kTopK; ++token_route) {
    const int route = token * kTopK + token_route;
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
#pragma unroll
    for (int split = 0; split < kDownSplitK; ++split) {
      sum += partials[(route * kDownSplitK + split) * kHidden + n];
    }
  }
  output[work] = __float2half_rn(sum);
}

void decode(torch::Tensor input, torch::Tensor w13, torch::Tensor w2,
            torch::Tensor w13_scales, torch::Tensor w2_scales,
            torch::Tensor w13_global, torch::Tensor w2_global,
            torch::Tensor topk_ids, torch::Tensor topk_weights,
            torch::Tensor gate_up_partials, torch::Tensor activated,
            torch::Tensor down_partials, torch::Tensor output,
            double swiglu_limit) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                  input.dim() == 2 && input.size(0) >= 1 &&
                  input.size(0) <= 4 && input.size(1) == kHidden,
              "input must be CUDA FP16 [M, 4096], 1 <= M <= 4");
  const int batch_size = static_cast<int>(input.size(0));
  const int num_routes = batch_size * kTopK;
  const int num_experts = static_cast<int>(w13.size(0));
  TORCH_CHECK(w13.is_cuda() && w13.scalar_type() == at::kInt,
              "w13 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w2.is_cuda() && w2.scalar_type() == at::kInt,
              "w2 must be CUDA int32 Marlin weights");
  TORCH_CHECK(w13_scales.is_cuda() && w13_scales.element_size() == 1,
              "w13_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w2_scales.is_cuda() && w2_scales.element_size() == 1,
              "w2_scales must be byte-sized CUDA metadata");
  TORCH_CHECK(w13_global.scalar_type() == at::kFloat &&
                  w2_global.scalar_type() == at::kFloat,
              "global scales must be FP32");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == num_routes,
              "topk_ids must be int32 [M, 8]");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.numel() == num_routes,
              "topk_weights must be FP32 [M, 8]");
  TORCH_CHECK(gate_up_partials.scalar_type() == at::kFloat &&
                  gate_up_partials.numel() == kSplitK * num_routes * kGateUp,
              "gate_up_partials has the wrong shape");
  TORCH_CHECK(activated.scalar_type() == at::kHalf &&
                  activated.numel() == num_routes * kIntermediate,
              "activated has the wrong shape");
  TORCH_CHECK(down_partials.scalar_type() == at::kFloat &&
                  down_partials.numel() == num_routes * kDownSplitK * kHidden,
              "down_partials has the wrong shape");
  TORCH_CHECK(output.scalar_type() == at::kHalf &&
                  output.numel() == batch_size * kHidden,
              "output has the wrong shape");

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int gate_up_work = kSplitK * (kGateUp / 8);
  const auto gate_kernel = (reinterpret_cast<uintptr_t>(w13_scales.data_ptr()) % 8 == 0)
                               ? gate_up_partial_kernel<true>
                               : gate_up_partial_kernel<false>;
  gate_kernel<<<(gate_up_work + kThreads - 1) / kThreads, kThreads,
                           0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w13.data_ptr<int>()),
      reinterpret_cast<const uint8_t*>(w13_scales.data_ptr()),
      topk_ids.data_ptr<int>(), num_routes, num_experts,
      gate_up_partials.data_ptr<float>());
  const int activated_work = num_routes * kIntermediate;
  gate_up_reduce_silu_kernel<<<
      (activated_work + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      gate_up_partials.data_ptr<float>(), w13_global.data_ptr<float>(),
      topk_ids.data_ptr<int>(), num_routes, num_experts,
      static_cast<float>(swiglu_limit),
      reinterpret_cast<__half*>(activated.data_ptr<at::Half>()));
  const int down_work = kDownSplitK * (kHidden / 8);
  const auto down_kernel = (reinterpret_cast<uintptr_t>(w2_scales.data_ptr()) % 8 == 0)
                               ? down_partial_kernel<true>
                               : down_partial_kernel<false>;
  down_kernel<<<(down_work + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const uint32_t*>(w2.data_ptr<int>()),
      reinterpret_cast<const uint8_t*>(w2_scales.data_ptr()),
      w2_global.data_ptr<float>(), topk_ids.data_ptr<int>(),
      topk_weights.data_ptr<float>(), num_routes, num_experts,
      down_partials.data_ptr<float>());
  const int output_work = batch_size * kHidden;
  down_reduce_kernel<<<(output_work + kThreads - 1) / kThreads, kThreads, 0,
                       stream>>>(down_partials.data_ptr<float>(),
                                topk_ids.data_ptr<int>(), num_experts, batch_size,
                                reinterpret_cast<__half*>(
                                    output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void topk10_softmax(torch::Tensor logits, torch::Tensor topk_weights,
                    torch::Tensor topk_ids) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim() == 2 &&
                  logits.size(0) >= 1 && logits.size(0) <= 4 &&
                  logits.size(1) == kExperts,
              "logits must be contiguous CUDA [M, 512], 1 <= M <= 4");
  const int batch_size = static_cast<int>(logits.size(0));
  TORCH_CHECK(logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kFloat,
              "logits must be FP16 or FP32");
  TORCH_CHECK(topk_weights.is_cuda() &&
                  topk_weights.scalar_type() == at::kFloat &&
                  topk_weights.numel() == batch_size * kTopK,
              "topk_weights must be CUDA FP32 [M, 10]");
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == at::kInt &&
                  topk_ids.numel() == batch_size * kTopK,
              "topk_ids must be CUDA int32 [M, 10]");
  const at::cuda::OptionalCUDAGuard guard(device_of(logits));
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(logits.get_device());
  if (logits.scalar_type() == at::kHalf) {
    topk10_radix_half<<<batch_size, 128, 0, stream>>>(
        reinterpret_cast<const __half*>(logits.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>());
  } else {
    topk10_softmax_kernel<<<batch_size, 256, 0, stream>>>(
        logits.data_ptr<float>(), topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sm70_nvfp4_moe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &sm70_nvfp4_moe::decode,
        "GLM-5.3 small-batch SM70 NVFP4 MoE decode");
}
