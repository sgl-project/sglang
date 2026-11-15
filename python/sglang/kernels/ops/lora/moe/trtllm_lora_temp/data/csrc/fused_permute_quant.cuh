// Fused permute + NvFP4-per-token-quant for the FP4 MoE-LoRA gate_up path.
//
// Background (decode bs64, EP8): the plain path runs `permuteKernel` (gather
// bf16 hidden into the padded [max_padded, hidden] permuted buffer) then
// `nvfp4QuantAndPerTokenScaleKernel` over ALL max_padded rows. At decode only
// num_tokens*top_k of those rows are real (the rest are padding), so both
// kernels waste ~6x of their work on padding, and the bf16 permuted buffer is a
// full HBM round-trip (written by permute, read back by quant).
//
// This kernel fuses the two: it reads the UN-permuted hidden, NvFP4-quantizes
// each (token,expert) pair's row, and scatter-writes fp4 + swizzled block-sf +
// per-token-sf directly to that pair's permuted position. It iterates only the
// num_tokens*top_k real pairs (skipping pad), and never materializes the bf16
// permuted buffer.
//
// It mirrors `nvfp4QuantAndPerTokenScaleKernel` (quantization.cuh) — same amax,
// same per-token-scale recipe, same `cvt_warp_fp16_to_fp4`, same swizzled-sf
// offset (`get_sf_out_offset_8x4`) — with the single `rowIdx` split into a READ
// row (the unpermuted source token) and a WRITE row (the permuted destination).
// For the valid rows the result is BITWISE-identical to the plain
// permute->quant chain (the chain's quant reads permuted_hidden[writeRow],
// filled by permute from hidden[readRow]; we read hidden[readRow] directly),
// verified by the bench's fused-vs-old guard.
//
// PER-TOKEN-SCALE BRANCH: uses TE_EXACT (globalEncodeScale =
// __fdiv_rn(globalScale, globalAmax), stored scale = 1/globalEncodeScale, cvt
// TE_EXACT_NVFP4=true). This matches the installed flashinfer 0.6.11.post1,
// whose DISPATCH macro hard-codes the bf16 kernel to TE_EXACT_NVFP4=true
// (quantization.cu DISPATCH_NVP4_QUANT_AND_PER_TOKEN_SCALE_KERNEL). NOTE: if a
// future flashinfer reverts the bf16 path to the fast-math branch
// (reciprocal_approximate / TE_EXACT=false), or if FLASHINFER_NVFP4_4OVER6 is
// enabled, this fused kernel would diverge — re-validate the bench's
// fused-vs-old bitwise guard against the deployed flashinfer before trusting it
// there.
//
// Two variants (both kept, selectable, for cross-scenario perf comparison):
//   - no-dedup: grid over the num_tokens*top_k pairs; each block
//   re-reads+re-quantizes its source
//     token and writes 1 destination (more blocks -> better occupancy at tiny
//     decode sizes).
//   - dedup:    grid over num_tokens; each block reads+quantizes its token once
//   and scatter-writes
//     to all of that token's (valid) permuted destinations (no redundant quant,
//     fewer blocks).
//
// Helpers are pulled from quantization_utils.cuh (cvt_warp_fp16_to_fp4 /
// get_sf_out_offset_* / PackedVec / reciprocal_approximate_ftz) rather than
// quantization.cuh, because the latter pulls in
// nv_internal/.../common/cudaUtils.h, which ODR-conflicts with the
// flashinfer/trtllm/common twin already in
// trtllm_fused_moe_kernel_launcher.cu's TU. loadPackedVec lives in
// quantization.cuh, so we do a direct aligned PackedVec load instead.
#pragma once

#include <cub/cub.cuh>

#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <optional>
#include <type_traits>

namespace sgl_fused_permute_quant {

namespace tk = tensorrt_llm::kernels;

// Quantize source row `readRow` of `input` (unpermuted) and write fp4 +
// block-sf + per-token-sf to destination row `writeRow` of the permuted
// outputs. `numRowsSf` is the SF buffer's row count
// (= max_padded), matching the plain quant's `m` arg to get_sf_out_offset_*.
template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__device__ __forceinline__ void
fused_quant_one_row(uint32_t n, T const *input, int readRow, int writeRow,
                    int numRowsSf, float globalScaleInv, uint8_t *weightOutput,
                    uint8_t *scaleOutput, float *perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int ELTS_PER_THREAD = 16;
  using InType = tk::PackedVec<T, ELTS_PER_THREAD>;
  using PackedFp4Type =
      std::conditional_t<ELTS_PER_THREAD == 16, uint64_t, uint32_t>;
  uint32_t const num_vecs_per_row = (n + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  uint32_t const num_sf_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;
  InType const *inBase = reinterpret_cast<InType const *>(input);

  // ---- pass 1: per-row amax over the (unpermuted) source row ----
  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx];
    std::remove_reference_t<decltype(vec_in.elts[0])> a(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD / 2; ++i) {
      a = __hmax2(a, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(a.x, a.y)));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax =
      BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});

  // ---- per-token scale (TE_EXACT branch — production instantiates
  // TE_EXACT_NVFP4=true for bf16,
  //      quantization.cu:247): globalEncodeScale = globalScale/globalAmax
  //      (exact __fdiv_rn), stored per-token scale = 1/globalEncodeScale.
  //      __shared__ scalar replaces the gmem round-trip (bit-identical: an fp32
  //      store->load doesn't change the value). ----
  __shared__ float sEncodeScale;
  if (threadIdx.x == 0) {
    float const globalScale = __fdiv_rn(1.0f, globalScaleInv);
    float const rowEncodeScale =
        globalAmax != 0.0f ? fminf(__fdiv_rn(globalScale, globalAmax), FLT_MAX)
                           : FLT_MAX;
    sEncodeScale = rowEncodeScale != 0.0f ? rowEncodeScale : 1.0f;
  }
  __syncthreads();
  float const globalEncodeScale = sEncodeScale;
  float const perTokenScale = __fdiv_rn(1.0f, globalEncodeScale);
  if (threadIdx.x == 0)
    perTokenScaleOutput[writeRow] = perTokenScale;

  // ---- pass 2: quantize + scatter-write to the permuted destination ----
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(readRow) * num_vecs_per_row + vecIdx];
    uint8_t fp8Scale;
    auto fp4Vals = tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, ELTS_PER_THREAD,
                                            /*UE8M0_SF=*/false,
                                            /*TE_EXACT_NVFP4=*/true>(
        vec_in, globalEncodeScale, &fp8Scale);
    reinterpret_cast<PackedFp4Type *>(
        weightOutput)[static_cast<int64_t>(writeRow) * num_vecs_per_row +
                      vecIdx] = fp4Vals;

    int64_t sfOffset;
    if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
      sfOffset = static_cast<int64_t>(writeRow) * num_sf_vecs_per_row + vecIdx;
    } else if constexpr (SF_LAYOUT ==
                         tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
      sfOffset = tk::get_sf_out_offset_128x4(/*batchIdx=*/0, writeRow, vecIdx,
                                             numRowsSf, num_sf_vecs_per_row);
    } else {
      sfOffset = tk::get_sf_out_offset_8x4(/*batchIdx=*/0, writeRow, vecIdx,
                                           numRowsSf, num_sf_vecs_per_row);
    }
    scaleOutput[sfOffset] = fp8Scale;
  }
}

// no-dedup: grid.x = num_tokens*top_k (one block per (token,expert) pair).
template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantKernel(
    uint32_t numPairs, uint32_t n, uint32_t topK, int numRowsSf, T const *input,
    float globalScaleInv, int32_t const *expandedIdxToPermutedIdx,
    uint8_t *weightOutput, uint8_t *scaleOutput, float *perTokenScaleOutput) {
  uint32_t const expandedIdx = blockIdx.x;
  if (expandedIdx >= numPairs)
    return;
  int const writeRow = expandedIdxToPermutedIdx[expandedIdx];
  if (writeRow < 0)
    return;
  int const readRow = static_cast<int>(expandedIdx / topK);
  fused_quant_one_row<T, BLOCK_SIZE, SF_LAYOUT>(
      n, input, readRow, writeRow, numRowsSf, globalScaleInv, weightOutput,
      scaleOutput, perTokenScaleOutput);
}

// dedup: grid.x = num_tokens (one block per source token, scatter to its top_k
// destinations).
template <typename T, uint32_t BLOCK_SIZE,
          tensorrt_llm::QuantizationSFLayout SF_LAYOUT>
__global__ void fusedPermuteNvfp4QuantDedupKernel(
    uint32_t numTokens, uint32_t n, uint32_t topK, int numRowsSf,
    T const *input, float globalScaleInv,
    int32_t const *expandedIdxToPermutedIdx, uint8_t *weightOutput,
    uint8_t *scaleOutput, float *perTokenScaleOutput) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int ELTS_PER_THREAD = 16;
  using InType = tk::PackedVec<T, ELTS_PER_THREAD>;
  using PackedFp4Type =
      std::conditional_t<ELTS_PER_THREAD == 16, uint64_t, uint32_t>;
  uint32_t const token = blockIdx.x;
  if (token >= numTokens)
    return;
  uint32_t const num_vecs_per_row = (n + ELTS_PER_THREAD - 1) / ELTS_PER_THREAD;
  uint32_t const num_sf_vecs_per_row = (n + SF_VEC_SIZE - 1) / SF_VEC_SIZE;
  InType const *inBase = reinterpret_cast<InType const *>(input);

  // pass 1: amax over the source token row (read once).
  float localAmax = 0.f;
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(token) * num_vecs_per_row + vecIdx];
    std::remove_reference_t<decltype(vec_in.elts[0])> a(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD / 2; ++i) {
      a = __hmax2(a, __habs2(vec_in.elts[i]));
    }
    localAmax = fmaxf(localAmax, static_cast<float>(__hmax(a.x, a.y)));
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  float const globalAmax =
      BlockReduce(tempStorage).Reduce(localAmax, cuda::maximum<>{});

  // TE_EXACT per-token scale (matches production; see fused_quant_one_row).
  __shared__ float sEncodeScale;
  if (threadIdx.x == 0) {
    float const globalScale = __fdiv_rn(1.0f, globalScaleInv);
    float const rowEncodeScale =
        globalAmax != 0.0f ? fminf(__fdiv_rn(globalScale, globalAmax), FLT_MAX)
                           : FLT_MAX;
    sEncodeScale = rowEncodeScale != 0.0f ? rowEncodeScale : 1.0f;
  }
  __syncthreads();
  float const globalEncodeScale = sEncodeScale;
  float const perTokenScale = __fdiv_rn(1.0f, globalEncodeScale);

  // per-token scale -> each (valid) destination (top_k small; first top_k
  // threads write).
  if (threadIdx.x < topK) {
    int const writeRow = expandedIdxToPermutedIdx[token * topK + threadIdx.x];
    if (writeRow >= 0)
      perTokenScaleOutput[writeRow] = perTokenScale;
  }

  // pass 2: quantize each vec once, scatter to all valid destinations.
  for (uint32_t vecIdx = threadIdx.x; vecIdx < num_vecs_per_row;
       vecIdx += BLOCK_SIZE) {
    InType vec_in =
        inBase[static_cast<int64_t>(token) * num_vecs_per_row + vecIdx];
    uint8_t fp8Scale;
    auto fp4Vals = tk::cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, ELTS_PER_THREAD,
                                            /*UE8M0_SF=*/false,
                                            /*TE_EXACT_NVFP4=*/true>(
        vec_in, globalEncodeScale, &fp8Scale);
#pragma unroll 1
    for (uint32_t k = 0; k < topK; ++k) {
      int const writeRow = expandedIdxToPermutedIdx[token * topK + k];
      if (writeRow < 0)
        continue;
      reinterpret_cast<PackedFp4Type *>(
          weightOutput)[static_cast<int64_t>(writeRow) * num_vecs_per_row +
                        vecIdx] = fp4Vals;
      int64_t sfOffset;
      if constexpr (SF_LAYOUT == tensorrt_llm::QuantizationSFLayout::LINEAR) {
        sfOffset =
            static_cast<int64_t>(writeRow) * num_sf_vecs_per_row + vecIdx;
      } else if constexpr (SF_LAYOUT ==
                           tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
        sfOffset = tk::get_sf_out_offset_128x4(/*batchIdx=*/0, writeRow, vecIdx,
                                               numRowsSf, num_sf_vecs_per_row);
      } else {
        sfOffset = tk::get_sf_out_offset_8x4(/*batchIdx=*/0, writeRow, vecIdx,
                                             numRowsSf, num_sf_vecs_per_row);
      }
      scaleOutput[sfOffset] = fp8Scale;
    }
  }
}

// Launcher. `dedup` picks the variant. `n` (= hidden) must be a multiple of 16.
// `numRowsSf` is the SF buffer's row count (max_padded).
template <typename T>
void invokeFusedPermuteNvfp4Quant(uint32_t numTokens, uint32_t topK, uint32_t n,
                                  int numRowsSf, T const *input,
                                  float globalScaleInv,
                                  int32_t const *expandedIdxToPermutedIdx,
                                  uint8_t *weightOutput, uint8_t *scaleOutput,
                                  float *perTokenScaleOutput,
                                  tensorrt_llm::QuantizationSFLayout sfLayout,
                                  bool dedup, cudaStream_t stream) {
  // [opt] Occupancy tuning (ncu: kernel is occupancy-bound, not DRAM-bound —
  // DRAM <1%, achieved occupancy was 19.8% no-dedup / 5.5% dedup at
  // BLOCK_SIZE=128). The dedup variant launches only num_tokens CTAs (=64 at
  // decode bs64), so it is the most CTA-starved; widening the block raises
  // threads/CTA and hides the per-row amax-reduction + scatter latency. Decode
  // bs64 dedup sweep: 128 -> 5.52us, 256 -> 4.09us, 512 -> 3.71us. 512 is the
  // chosen default (the prod path uses dedup). (7168/16 = 448 vecs/row, so >448
  // threads idle on the tail, but the win dominates.)
  constexpr uint32_t BLOCK_SIZE = 512;
  dim3 const block(BLOCK_SIZE);

  auto dispatch = [&](auto layoutTag) {
    constexpr tensorrt_llm::QuantizationSFLayout LAYOUT =
        decltype(layoutTag)::value;
    if (dedup) {
      dim3 const grid(numTokens);
      fusedPermuteNvfp4QuantDedupKernel<T, BLOCK_SIZE, LAYOUT>
          <<<grid, block, 0, stream>>>(numTokens, n, topK, numRowsSf, input,
                                       globalScaleInv, expandedIdxToPermutedIdx,
                                       weightOutput, scaleOutput,
                                       perTokenScaleOutput);
    } else {
      dim3 const grid(numTokens * topK);
      fusedPermuteNvfp4QuantKernel<T, BLOCK_SIZE, LAYOUT>
          <<<grid, block, 0, stream>>>(numTokens * topK, n, topK, numRowsSf,
                                       input, globalScaleInv,
                                       expandedIdxToPermutedIdx, weightOutput,
                                       scaleOutput, perTokenScaleOutput);
    }
  };

  if (sfLayout == tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4) {
    dispatch(std::integral_constant<
             tensorrt_llm::QuantizationSFLayout,
             tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4>{});
  } else {
    dispatch(std::integral_constant<
             tensorrt_llm::QuantizationSFLayout,
             tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4>{});
  }
}

} // namespace sgl_fused_permute_quant
