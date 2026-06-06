// Fused SwiGLU+LoRA activation -> NVFP4 per-token quant for the FP4 MoE LoRA path.
//
// Modeled on tensorrt_llm::kernels::nvfp4QuantAndPerTokenScaleKernel (flashinfer
// quantization.cuh): one block per expanded row, per-token amax via cub::BlockReduce,
// cvt_warp_fp16_to_fp4 for the e4m3 block scale + e2m1 packing + swizzled SF layout.
// The ONLY change vs that kernel is the pass-1 input: instead of reading activated_bf16
// from gmem, it reads the interleaved gate/up GEMM1 output + the LoRA delta and computes
// silu(up)*gate on the fly, rounds to bf16 (matching the standalone activation kernel's
// bf16 output exactly), caches in smem, and also writes activation_lora_input. The down
// GEMM input (activated_bf16) is therefore never materialized to HBM.
//
// Because the activated value is rounded to bf16 before quantization (same as the separate
// activation kernel) and silu matches, the fp4 / SF / per_token_sf / activation_lora_input
// outputs are BITWISE-identical to the unfused activation -> quant#2 chain.
#pragma once

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <optional>

namespace flashinfer {
namespace sgl_fused_act_quant {

namespace tk = tensorrt_llm::kernels;

// Same silu as moe::dev::activation (trtllm_fused_moe_dev_kernel.cu:55): x / (1 + exp(-x)).
inline __device__ float fused_silu(float x) {
  return x / (1.0f + expf(-x));
}

// One block per expanded row. gateUp is the column-interleaved GEMM1 output (g0,u0,g1,u1,...)
// indexed by permutedIdx; loraDelta is the contiguous [gate|up] delta indexed by expandedIdx.
template <uint32_t BLOCK_SIZE, tensorrt_llm::QuantizationSFLayout SF_LAYOUT, bool DISABLE_FP4_FAST_MATH>
__global__ void fusedActivationQuantKernel(
    int m,                                        // numTokens * topK (number of expanded rows)
    int innerHalf,                                // inter == n (output width per row); must be a multiple of 16
    int innerDim,                                 // gate_up_n == 2 * innerHalf
    __nv_bfloat16 const* __restrict__ gateUp,     // interleaved gate/up, [.., innerDim] by permutedIdx
    __nv_bfloat16 const* __restrict__ loraDelta,  // [.., innerDim] by expandedIdx, may be null
    __nv_bfloat16* __restrict__ loraInputOut,     // [.., innerHalf] by expandedIdx, may be null
    int32_t const* __restrict__ expandedIdxToPermutedIdx,
    float globalScaleInv,
    uint8_t* __restrict__ weightOutput,  // fp4 [.., innerHalf/2] by permutedIdx
    uint8_t* __restrict__ scaleOutput,   // swizzled e4m3 SF
    float* __restrict__ perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  using InType = tk::PackedVec<__nv_bfloat16, SF_VEC_SIZE>;  // 16 bf16 == 8 __nv_bfloat162
  using PackedFp4Type = uint64_t;                            // SF_VEC_SIZE == 16

  int const expandedIdx = blockIdx.x;
  if (expandedIdx >= m) return;
  int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
  int const num_vecs_per_row = innerHalf / SF_VEC_SIZE;
  int64_t const liBaseRow = (int64_t)expandedIdx * innerHalf;

  // Padding row: the separate activation kernel writes 0 to activation_lora_input and skips
  // the quant outputs. Mirror that, then return.
  if (permutedIdx < 0) {
    if (loraInputOut != nullptr) {
      InType z;
#pragma unroll
      for (int i = 0; i < SF_VEC_SIZE / 2; ++i)
        z.elts[i] = __float2bfloat162_rn(0.0f);
      for (int vecIdx = threadIdx.x; vecIdx < num_vecs_per_row; vecIdx += BLOCK_SIZE) {
        *reinterpret_cast<InType*>(&loraInputOut[liBaseRow + (int64_t)vecIdx * SF_VEC_SIZE]) = z;
      }
    }
    return;
  }

  int64_t const permBase = (int64_t)permutedIdx * innerDim;  // gate_up row (interleaved)
  int64_t const expBase = (int64_t)expandedIdx * innerDim;   // delta row (contiguous gate|up)
  (void)DISABLE_FP4_FAST_MATH;

  // 1 SF block (16 outputs) per thread, held in registers across the amax barrier (no smem cache):
  // requires num_vecs_per_row <= BLOCK_SIZE (inter=2048 -> 128 == BLOCK_SIZE). With
  // CVT_ELTS_PER_THREAD == SF_VEC_SIZE the cvt needs no cross-thread shuffle, so masking is safe.
  int const vecIdx = threadIdx.x;
  bool const active = vecIdx < num_vecs_per_row;

  InType vec;
  float localAmax = 0.f;
  if (active) {
    int const h0 = vecIdx * SF_VEC_SIZE;
    __nv_bfloat16 const* g = gateUp + permBase + (int64_t)2 * h0;     // 32 interleaved bf16
    __nv_bfloat16 const* dlo = loraDelta + expBase + h0;              // silu-arg delta (lower half)
    __nv_bfloat16 const* dhi = loraDelta + expBase + innerHalf + h0;  // multiplier delta (upper half)
    __nv_bfloat162 amax2 = __float2bfloat162_rn(0.0f);
    union {
      int4 v[4];
      __nv_bfloat16 b[32];
    } gu;
    union {
      int4 v[2];
      __nv_bfloat16 b[16];
    } dl, dh;
    int4 const* gp = reinterpret_cast<int4 const*>(g);
#pragma unroll
    for (int k = 0; k < 4; ++k)
      gu.v[k] = gp[k];
    if (loraDelta != nullptr) {
      int4 const* dlp = reinterpret_cast<int4 const*>(dlo);
      int4 const* dhp = reinterpret_cast<int4 const*>(dhi);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        dl.v[k] = dlp[k];
        dh.v[k] = dhp[k];
      }
    }
#pragma unroll
    for (int i = 0; i < SF_VEC_SIZE / 2; ++i) {  // 8 bf162 = 16 output elements
      int const j0 = 2 * i, j1 = 2 * i + 1;
      float even0 = (float)gu.b[2 * j0], odd0 = (float)gu.b[2 * j0 + 1];
      float even1 = (float)gu.b[2 * j1], odd1 = (float)gu.b[2 * j1 + 1];
      float a0 = odd0, b0 = even0, a1 = odd1, b1 = even1;
      if (loraDelta != nullptr) {
        a0 += (float)dl.b[j0];
        b0 += (float)dh.b[j0];
        a1 += (float)dl.b[j1];
        b1 += (float)dh.b[j1];
      }
      float act0 = fused_silu(a0) * b0;
      float act1 = fused_silu(a1) * b1;
      __nv_bfloat162 e = __float22bfloat162_rn(make_float2(act0, act1));
      vec.elts[i] = e;
      amax2 = __hmax2(amax2, __habs2(e));
    }
    localAmax = (float)__hmax(amax2.x, amax2.y);
    if (loraInputOut != nullptr) {
      *reinterpret_cast<InType*>(&loraInputOut[liBaseRow + h0]) = vec;
    }
  }

  // ---- per-token scale: blockReduce amax, broadcast via smem (no gmem round-trip) ----
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float sScale;
  float const globalAmax = BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});
  if (threadIdx.x == 0) {
    float const pts = globalAmax * globalScaleInv;
    perTokenScaleOutput[permutedIdx] = pts;
    sScale = pts;
  }
  __syncthreads();
  float const globalEncodeScale = tk::reciprocal_approximate_ftz(sScale);

  // ---- quantize from registers (cvt computes the per-16 e4m3 block scale internally) ----
  if (active) {
    uint8_t fp8Scale;
    // 5 template args on this flashinfer build: Type, SF_VEC_SIZE, CVT_ELTS_PER_THREAD,
    // UE8M0_SF=false, TE_EXACT_NVFP4=false (the default nvfp4 quant path).
    auto fp4Vals = tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, SF_VEC_SIZE, SF_VEC_SIZE, false, false>(
        vec, globalEncodeScale, &fp8Scale);
    int64_t const vecOffset = (int64_t)permutedIdx * num_vecs_per_row + vecIdx;
    reinterpret_cast<PackedFp4Type*>(weightOutput)[vecOffset] = fp4Vals;

    // Match nvfp4QuantAndPerTokenScaleKernel exactly (it passes the kernel's `m` as numRows).
    int64_t sfOffset;
    if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      sfOffset = vecOffset;
    } else if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      sfOffset = tk::get_sf_out_offset_128x4(std::nullopt, permutedIdx, vecIdx, m, num_vecs_per_row);
    } else {
      sfOffset = tk::get_sf_out_offset_8x4(std::nullopt, permutedIdx, vecIdx, m, num_vecs_per_row);
    }
    scaleOutput[sfOffset] = fp8Scale;
  }
}

// Host launch: globalScaleInv = 1/448/6. BLOCK_SIZE must be >= innerHalf/16 (one SF block/thread).
inline void launchFusedActivationQuant(
    int m,
    int innerHalf,
    int innerDim,
    __nv_bfloat16 const* gateUp,
    __nv_bfloat16 const* loraDelta,
    __nv_bfloat16* loraInputOut,
    int32_t const* expandedIdxToPermutedIdx,
    float globalScaleInv,
    uint8_t* weightOutput,
    uint8_t* scaleOutput,
    float* perTokenScaleOutput,
    tensorrt_llm::QuantizationSFLayout sfLayout,
    bool disableFp4FastMath,
    cudaStream_t stream) {
  constexpr uint32_t BLOCK_SIZE = 128;  // == innerHalf/16 for inter=2048 (one SF block per thread)
  dim3 const grid(m), block(BLOCK_SIZE);

  auto launch = [&](auto layoutTag, auto fastMathTag) {
    fusedActivationQuantKernel<BLOCK_SIZE, decltype(layoutTag)::value, decltype(fastMathTag)::value>
        <<<grid, block, 0, stream>>>(
            m,
            innerHalf,
            innerDim,
            gateUp,
            loraDelta,
            loraInputOut,
            expandedIdxToPermutedIdx,
            globalScaleInv,
            weightOutput,
            scaleOutput,
            perTokenScaleOutput);
  };
  auto withFastMath = [&](auto layoutTag) {
    if (disableFp4FastMath) {
      launch(layoutTag, std::integral_constant<bool, true>{});
    } else {
      launch(layoutTag, std::integral_constant<bool, false>{});
    }
  };
  if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    withFastMath(
        std::integral_constant<
            tensorrt_llm::QuantizationSFLayout,
            tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4>{});
  } else if (sfLayout == tensorrt_llm::QuantizationSFLayout::LINEAR) {
    withFastMath(
        std::integral_constant<tensorrt_llm::QuantizationSFLayout, tensorrt_llm::QuantizationSFLayout::LINEAR>{});
  } else {
    withFastMath(
        std::integral_constant<tensorrt_llm::QuantizationSFLayout, tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4>{});
  }
}

}  // namespace sgl_fused_act_quant
}  // namespace flashinfer
