#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "flashmla_utils.h"
#include "kerutils/kerutils.cuh"
#include "params.h"
#include <math_constants.h>

using namespace cute;

namespace smxx::decode {

template <typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    flash_fwd_mla_combine_kernel(__grid_constant__ const CombineParams params) {
  // grid_shape: [batch_size*s_q, 1, h_q/BLOCK_SIZE_M]
  // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
  static_assert(NUM_THREADS / 32 == BLOCK_SIZE_M);  // The number of warps == block_size_m
  const int batch_s_q_idx = blockIdx.x;
  const int batch_idx = batch_s_q_idx / params.s_q;
  const int s_q_idx = batch_s_q_idx - batch_idx * params.s_q;
  const int h_block_idx = blockIdx.z;
  const int warp_idx = threadIdx.x / 32;
  const int lane_idx = threadIdx.x % 32;

  int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M * h_block_idx);
  if (warp_idx >= num_valid_heads) {
    return;
  }

  const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
  const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
  const int my_num_splits = end_split_idx - start_split_idx;
  if (my_num_splits == 1) {
    return;
  }

  FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);

  Tensor gLseAccum = make_tensor(
      make_gmem_ptr(
          (float*)params.lse_accum + start_split_idx * params.stride_lse_accum_split +
          s_q_idx * params.stride_lse_accum_s_q + h_block_idx * BLOCK_SIZE_M),
      Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
      make_stride(params.stride_lse_accum_split, _1{}));
  Tensor gLse = make_tensor(
      make_gmem_ptr(
          (float*)params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q +
          h_block_idx * BLOCK_SIZE_M),
      Shape<Int<BLOCK_SIZE_M>>{},
      Stride<_1>{});

  __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];

  // Wait for the previous kernel (the MLA kernel) to finish
  cudaGridDependencySynchronize();

  // Prefetch
  static_assert(HEAD_DIM_V % (32 * 4) == 0);
  constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32 * 4);
  float* oaccum_ptr = params.o_accum + start_split_idx * params.stride_o_accum_split +
                      s_q_idx * params.stride_o_accum_s_q +
                      (h_block_idx * BLOCK_SIZE_M + warp_idx) * params.stride_o_accum_h_q;
  float4 datas[ELEMS_PER_THREAD];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    datas[i] = *(float4*)(oaccum_ptr + lane_idx * 4 +
                          i * 128);  // NOTE We don't use __ldg here since it is incompatible with PDL
  }

  // Warp #i gathers LseAccum for seq #i
  {
    constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);
    float local_lse[NUM_LSE_PER_THREAD];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      local_lse[i] = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
    }

    float max_lse = -INFINITY;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
      max_lse = max(max_lse, local_lse[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2)
      max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

    float sum_lse = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
      sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset >= 1; offset /= 2)
      sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

    float global_lse = (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
    if (lane_idx == 0) gLse(warp_idx) = global_lse / (float)M_LOG2E;

    if (params.attn_sink != nullptr) {
      int q_head_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
      float attn_sink = __ldg(params.attn_sink + q_head_idx);
      if (global_lse != INFINITY) {
        // If attn_sink is +inf, global_lse will be +inf and scale factors will be exp2f(local_lse - inf) = 0 (since
        // local_lse never becomes +inf) If attn_sink is -inf, this has no effect on global_lse
        global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
      } else {
        // We have no tokens to attend, so global lse should be attn_sink*CUDART_L2E_F (+inf if it's -inf or +inf)
        global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);
    }
  }

  __syncwarp();

  // Warp #i accumulates activation for seq #i
  {
    float4 result[ELEMS_PER_THREAD];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
      result[i] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int split = 0; split < my_num_splits; ++split) {
      float lse_scale = smem_buf[warp_idx][split];
      // if (lse_scale != 0.f) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        result[i].x += lse_scale * datas[i].x;
        result[i].y += lse_scale * datas[i].y;
        result[i].z += lse_scale * datas[i].z;
        result[i].w += lse_scale * datas[i].w;
        if (split != my_num_splits - 1) {
          datas[i] = *(float4*)(oaccum_ptr + (split + 1) * params.stride_o_accum_split + lane_idx * 4 + i * 128);
        }
      }
      // }
    }

    const int h_q_idx = h_block_idx * BLOCK_SIZE_M + warp_idx;
    ElementT* o_ptr = (ElementT*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                      h_q_idx * params.stride_o_h_q;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      float4 data = result[i];
      ElementT data_converted[4];
      data_converted[0] = (ElementT)(data.x);
      data_converted[1] = (ElementT)(data.y);
      data_converted[2] = (ElementT)(data.z);
      data_converted[3] = (ElementT)(data.w);
      static_assert(sizeof(ElementT) == 2);
      *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(uint64_t*)data_converted;
    }
  }
}

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...) \
  [&] {                                              \
    if (NUM_SPLITS <= 32) {                          \
      constexpr static int NAME = 32;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 64) {                   \
      constexpr static int NAME = 64;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 96) {                   \
      constexpr static int NAME = 96;                \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 128) {                  \
      constexpr static int NAME = 128;               \
      return __VA_ARGS__();                          \
    } else if (NUM_SPLITS <= 160) {                  \
      constexpr static int NAME = 160;               \
      return __VA_ARGS__();                          \
    } else {                                         \
      FLASH_ASSERT(false);                           \
    }                                                \
  }()

template <typename ElementT>
void run_flash_mla_combine_kernel(CombineParams& params) {
  static constexpr int HEAD_DIM_V = 512;  // Since only this head dimension is supported by Flash MLA
  FLASH_ASSERT(params.d_v == HEAD_DIM_V);
  MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
    constexpr int BLOCK_SIZE_M = 8;
    constexpr int NUM_THREADS = BLOCK_SIZE_M * 32;
    constexpr size_t smem_size = BLOCK_SIZE_M * (NUM_SPLITS + 1) * sizeof(float);
    auto combine_kernel = &flash_fwd_mla_combine_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS>;
    CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t combine_kernel_config = {
        dim3(params.b * params.s_q, 1, ku::ceil_div(params.h_q, BLOCK_SIZE_M)),
        dim3(NUM_THREADS, 1, 1),
        0,
        params.stream,
        attribute,
        1};
    CHECK_CUDA(cudaLaunchKernelEx(&combine_kernel_config, combine_kernel, params));
  });
  CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_flash_mla_combine_kernel<cutlass::bfloat16_t>(CombineParams& params);

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_mla_combine_kernel<cutlass::half_t>(CombineParams& params);
#endif

}  // namespace smxx::decode
