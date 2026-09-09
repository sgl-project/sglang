// Adapted from the pinned FlashMLA combine kernel; see ../../../LICENSE.
// This private variant includes the sink in both output normalization and LSE.
#include <cutlass/bfloat16.h>
#include <math_constants.h>

#include <kerutils/kerutils.cuh>

#include "attention_math.h"
#include "combine.h"
#include "flashmla_utils.h"

namespace kvbit::dsv4 {

template <int MAX_SPLITS>
__global__ void __launch_bounds__(256) combine_int4_kernel(__grid_constant__ const CombineParams params) {
  const int batch = blockIdx.x / params.s_q;
  const int query = blockIdx.x % params.s_q;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int head = blockIdx.z * 8 + warp;
  const int start = __ldg(params.num_splits_ptr + batch);
  const int count = __ldg(params.num_splits_ptr + batch + 1) - start;
  if (count == 1) return;
  FLASH_DEVICE_ASSERT(count > 1 && count <= MAX_SPLITS);
  cudaGridDependencySynchronize();

  constexpr int ELEMS_PER_THREAD = 4;
  float* values = params.o_accum + start * params.stride_o_accum_split + query * params.stride_o_accum_s_q +
                  head * params.stride_o_accum_h_q;
  float4 data[ELEMS_PER_THREAD];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    data[i] = *(float4*)(values + lane * 4 + i * 128);

  constexpr int LSE_PER_LANE = (MAX_SPLITS + 31) / 32;
  float partial_lse[LSE_PER_LANE];
  float maximum = -INFINITY;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < LSE_PER_LANE; ++i) {
    const int split = i * 32 + lane;
    partial_lse[i] =
        split < count
            ? params.lse_accum
                  [(start + split) * params.stride_lse_accum_split + query * params.stride_lse_accum_s_q + head]
            : -INFINITY;
    maximum = fmaxf(maximum, partial_lse[i]);
  }
  CUTLASS_PRAGMA_UNROLL
  for (int offset = 16; offset >= 1; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  maximum = maximum == -INFINITY ? 0.0f : maximum;
  float sum = 0.0f;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < LSE_PER_LANE; ++i)
    sum += exp2f(partial_lse[i] - maximum);
  CUTLASS_PRAGMA_UNROLL
  for (int offset = 16; offset >= 1; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);

  const float sink = params.attn_sink ? __ldg(params.attn_sink + head) * CUDART_L2E_F : -INFINITY;
  const float total_lse = lse_with_sink_log2(sum, maximum, sink);
  if (lane == 0)
    params.lse[batch * params.stride_lse_b + query * params.stride_lse_s_q + head] = total_lse * CUDART_LN2_F;

  __shared__ float weights[8][MAX_SPLITS];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < LSE_PER_LANE; ++i) {
    const int split = i * 32 + lane;
    if (split < MAX_SPLITS) weights[warp][split] = exp2f(partial_lse[i] - total_lse);
  }
  __syncwarp();

  float4 result[ELEMS_PER_THREAD];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    result[i] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
  for (int split = 0; split < count; ++split) {
    const float weight = weights[warp][split];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      result[i].x += weight * data[i].x;
      result[i].y += weight * data[i].y;
      result[i].z += weight * data[i].z;
      result[i].w += weight * data[i].w;
      if (split + 1 != count)
        data[i] = *(float4*)(values + (split + 1) * params.stride_o_accum_split + lane * 4 + i * 128);
    }
  }
  using bf16 = cutlass::bfloat16_t;
  bf16* output = static_cast<bf16*>(params.out) + batch * params.stride_o_b + query * params.stride_o_s_q +
                 head * params.stride_o_h_q;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
    bf16 converted[4] = {bf16(result[i].x), bf16(result[i].y), bf16(result[i].z), bf16(result[i].w)};
    *(uint64_t*)(output + lane * 4 + i * 128) = *reinterpret_cast<uint64_t*>(converted);
  }
}

template <int MAX_SPLITS>
void launch_combine(CombineParams& params) {
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {
      dim3(params.b * params.s_q, 1, params.h_q / 8), dim3(256, 1, 1), 0, params.stream, attribute, 1};
  CHECK_CUDA(cudaLaunchKernelEx(&config, &combine_int4_kernel<MAX_SPLITS>, params));
}

void run_int4_combine(CombineParams& params) {
  FLASH_ASSERT(params.d_v == 512 && params.h_q == 64);
  if (params.num_sm_parts <= 32)
    launch_combine<32>(params);
  else if (params.num_sm_parts <= 64)
    launch_combine<64>(params);
  else if (params.num_sm_parts <= 96)
    launch_combine<96>(params);
  else if (params.num_sm_parts <= 128)
    launch_combine<128>(params);
  else if (params.num_sm_parts <= 160)
    launch_combine<160>(params);
  else if (params.num_sm_parts <= 192)
    launch_combine<192>(params);
  else if (params.num_sm_parts <= 224)
    launch_combine<224>(params);
  else if (params.num_sm_parts <= 256)
    launch_combine<256>(params);
  else
    FLASH_ASSERT(false);
  CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace kvbit::dsv4
