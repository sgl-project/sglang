/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 Rong Shuo.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// k_ext (selective residue) NVFP4 activation quantization.
//
// Ported from ResInfer csrc/nvfp4/src/fp4/nvfp4_residue_pack8.cu and
// nvfp4_residue_pack16.cu, rehosted on the sglang JIT tvm-ffi conventions.
// Ratio-specialized no-swap kernels only; the swap-residue experiment was
// deliberately not ported.
//
// Contract (mirrors the reference torch op):
//   input          [M, K] fp16/bf16, K % 16 == 0
//   input_sf       float32 scalar (global scale reciprocal)
//   channel_masks  uint8 [K/8], bit b of byte i marks channel 8*i+b salient;
//                  every byte carries exactly residue_per_8 set bits (the
//                  exporter guarantees per-block uniform selection)
//   output         uint8, >= M * n_ext / 2 bytes; each row is
//                  [base K/2 bytes | residue (n_ext-K)/2 bytes]
//   output_sf      uint8 (fp8-e4m3), swizzled 128x4 tiled layout over
//                  (M, n_ext); residue SFs live at the extended-K positions
//   n_ext          K + num_salient
//   residue_per_8  1 (ratio 1/8), 2 (2/8), or 4 (4/8)
//
// pack16 gives each thread one full 16-element SF vector (no cross-thread
// main amax) and loads it with one 256-bit ld on Blackwell; output is
// bit-identical to pack8. elts_mode: 0=auto (B200-measured policy on cc
// 10.x, pack8 elsewhere), 8, 16.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include "residue_nvfp4_utils.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {
namespace residue_nvfp4 {

// ─────────────────────────────── pack8 ─────────────────────────────────────

__device__ __forceinline__ float
pick_pack8_residue_value(float2 const (&residue)[CVT_FP4_ELTS_PER_THREAD / 2], int idx) {
  int pairIdx = idx >> 1;
  return (idx & 1) == 0 ? residue[pairIdx].x : residue[pairIdx].y;
}

__device__ __forceinline__ uint32_t pack_pack8_residue_group_r0125(float residue0, int laneId) {
  // Mirrors generic RESIDUE_PER_THREAD=1: each writer lane gathers one
  // 8-value residue chunk from source lanes writerGroup*8..writerGroup*8+7.
  int writerGroup = laneId < 4 ? laneId : 0;
  int sourceFirstThread = writerGroup * 8;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int sourceThread0 = sourceFirstThread + i * 2;
    int sourceThread1 = sourceThread0 + 1;
    float val0 = __shfl_sync(uint32_t(-1), residue0, sourceThread0);
    float val1 = __shfl_sync(uint32_t(-1), residue0, sourceThread1);
    residueChunk[i] = make_float2(val0, val1);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

__device__ __forceinline__ uint32_t pack_pack8_residue_group_r025(float residue0, float residue1, int laneId) {
  // ratio=0.25: each thread contributes two residue values; four adjacent
  // lanes form one uint32 FP4 residue chunk. Constant full shuffle mask is
  // safe because the launcher rounds block size to full warps.
  int writerGroup = laneId < 8 ? laneId : 0;
  int sourceFirstThread = writerGroup * 4;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int sourceThread = sourceFirstThread + i;
    float val0 = __shfl_sync(uint32_t(-1), residue0, sourceThread);
    float val1 = __shfl_sync(uint32_t(-1), residue1, sourceThread);
    residueChunk[i] = make_float2(val0, val1);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

__device__ __forceinline__ uint32_t
pack_pack8_residue_group_r050(float residue0, float residue1, float residue2, float residue3, int laneId) {
  // Mirrors generic RESIDUE_PER_THREAD=4: each writer lane gathers one
  // 8-value residue chunk from two adjacent source lanes.
  int writerGroup = laneId < 16 ? laneId : 0;
  int sourceFirstThread = writerGroup * 2;

  float2 residueChunk[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float vals[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      int residueIdx = i * 2 + j;
      int sourceThread = sourceFirstThread + (residueIdx >> 2);
      int sourceOffset = residueIdx & 3;
      float sourceValue;
      if (sourceOffset == 0) {
        sourceValue = residue0;
      } else if (sourceOffset == 1) {
        sourceValue = residue1;
      } else if (sourceOffset == 2) {
        sourceValue = residue2;
      } else {
        sourceValue = residue3;
      }
      vals[j] = __shfl_sync(uint32_t(-1), sourceValue, sourceThread);
    }
    residueChunk[i] = make_float2(vals[0], vals[1]);
  }
  return fp32_vec_to_e2m1(residueChunk);
}

// Shared main-quant + residue derivation for the pack8 ratio helpers: on
// return `residues` holds original - dequant for all 8 positions and the
// packed main data is returned.
template <class Type, bool UE8M0_SF>
__device__ __forceinline__ uint32_t pack8_main_quant(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout, float2 (&residues)[CVT_FP4_ELTS_PER_THREAD / 2]) {
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 fp2ValsOriginal[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      residues[i] = __half22float2(vec.elts[i]);
    } else {
      residues[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2ValsOriginal[i] = residues[i];
    residues[i].x *= outputScale;
    residues[i].y *= outputScale;
  }

  uint32_t e2m1Vec = fp32x8_to_e2m1_with_dequant(residues);

  float inverseOutputScale = outputScale != 0 ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    residues[i].x *= inverseOutputScale;
    residues[i].y *= inverseOutputScale;
    residues[i].x = fp2ValsOriginal[i].x - residues[i].x;
    residues[i].y = fp2ValsOriginal[i].y - residues[i].y;
  }
  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r0125(
    PackedVec<Type>& vec,
    float SFScaleVal,
    uint8_t* SFout,
    uint8_t mask,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  uint32_t e2m1Vec = pack8_main_quant<Type, UE8M0_SF>(vec, SFScaleVal, SFout, fp2Vals);

  int i0 = __ffs(mask) - 1;
  float residue0 = pick_pack8_residue_value(fp2Vals, i0);

  float localMaxResidue = fabsf(residue0);
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<16>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue = pack_pack8_residue_group_r0125(residue0 * residueOutputScale, laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }
  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r025(
    PackedVec<Type>& vec,
    float SFScaleVal,
    uint8_t* SFout,
    uint8_t mask,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  uint32_t e2m1Vec = pack8_main_quant<Type, UE8M0_SF>(vec, SFScaleVal, SFout, fp2Vals);

  int i0 = __ffs(mask) - 1;
  int i1 = __ffs(mask & (mask - 1)) - 1;
  float residue0 = pick_pack8_residue_value(fp2Vals, i0);
  float residue1 = pick_pack8_residue_value(fp2Vals, i1);

  float localMaxResidue = fmaxf(fabsf(residue0), fabsf(residue1));
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<8>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue =
      pack_pack8_residue_group_r025(residue0 * residueOutputScale, residue1 * residueOutputScale, laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }
  return e2m1Vec;
}

template <class Type, bool UE8M0_SF = false>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4_pack8_r050(
    PackedVec<Type>& vec,
    float SFScaleVal,
    uint8_t* SFout,
    uint8_t mask,
    uint32_t* residueDataOut,
    uint8_t* residueSFOut) {
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
  uint32_t e2m1Vec = pack8_main_quant<Type, UE8M0_SF>(vec, SFScaleVal, SFout, fp2Vals);

  int i0 = __ffs(mask) - 1;
  uint8_t remaining = mask & (mask - 1);
  int i1 = __ffs(remaining) - 1;
  remaining = remaining & (remaining - 1);
  int i2 = __ffs(remaining) - 1;
  remaining = remaining & (remaining - 1);
  int i3 = __ffs(remaining) - 1;

  float residue0 = pick_pack8_residue_value(fp2Vals, i0);
  float residue1 = pick_pack8_residue_value(fp2Vals, i1);
  float residue2 = pick_pack8_residue_value(fp2Vals, i2);
  float residue3 = pick_pack8_residue_value(fp2Vals, i3);

  float localMaxResidue = fmaxf(fmaxf(fabsf(residue0), fabsf(residue1)), fmaxf(fabsf(residue2), fabsf(residue3)));
  int laneId = threadIdx.x % 32;
  float sfGroupMaxResidue = warp_group_max<4>(localMaxResidue, laneId);
  float residueSFValue = SFScaleVal * (sfGroupMaxResidue * reciprocal_approximate_ftz(6.0f));
  residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
  float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

  uint32_t packedResidue = pack_pack8_residue_group_r050(
      residue0 * residueOutputScale,
      residue1 * residueOutputScale,
      residue2 * residueOutputScale,
      residue3 * residueOutputScale,
      laneId);
  if (residueDataOut) {
    *residueDataOut = packedResidue;
  }
  return e2m1Vec;
}

template <class Type, bool UE8M0_SF, int RESIDUE_PER_8_ELTS>
__global__ void __launch_bounds__(512, RESIDUE_NVFP4_BLOCKS_PER_SM(512)) cvt_fp16_to_fp4_residue_pack8_no_swap(
    int32_t numRows,
    int32_t numTotalCols,
    int32_t numMainCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint8_t const* channelIndices) {
  using PackedVecT = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF = (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / CVT_FP4_ELTS_PER_THREAD) {
    return;
  }

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVecT inVec{};
    if (validRow) {
      int64_t inOffset = static_cast<int64_t>(rowIdx) * (numMainCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      inVec = reinterpret_cast<PackedVecT const*>(in)[inOffset];
    }

    auto sfOut =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(rowIdx, colIdx, numTotalCols, SFout);

    uint32_t* outPos = nullptr;
    uint32_t* residueDataOut = nullptr;
    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      outPos = out + outOffset;
      residueDataOut =
          cvt_residue_data_get_offset<uint32_t, RESIDUE_PER_8_ELTS>(rowIdx, colIdx, numMainCols, numTotalCols, out);
    }

    auto residueSFOut =
        cvt_residue_sf_get_offset<uint8_t, RESIDUE_PER_8_ELTS>(rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    uint32_t packed;
    if constexpr (RESIDUE_PER_8_ELTS == 1) {
      packed = cvt_warp_fp16_to_fp4_pack8_r0125<Type, UE8M0_SF>(
          inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    } else if constexpr (RESIDUE_PER_8_ELTS == 2) {
      packed = cvt_warp_fp16_to_fp4_pack8_r025<Type, UE8M0_SF>(
          inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    } else {
      static_assert(RESIDUE_PER_8_ELTS == 4);
      packed = cvt_warp_fp16_to_fp4_pack8_r050<Type, UE8M0_SF>(
          inVec, SFScaleVal, sfOut, channelIndices[colIdx], residueDataOut, residueSFOut);
    }
    if (outPos) {
      *outPos = packed;
    }
  }
}

// ─────────────────────────────── pack16 ────────────────────────────────────

namespace pack16 {

__device__ __forceinline__ float pick_residue_value(float2 const (&residue)[4], int idx) {
  int pairIdx = idx >> 1;
  return (idx & 1) == 0 ? residue[pairIdx].x : residue[pairIdx].y;
}

// Residue data chunk address for the elt16 mapping. Chunks are the same
// element-ordered consecutive uint32s the pack8 helpers emit; an elt16 warp
// spans two pack8 warp sections, i.e. CHUNKS_PER_WARP = 32 * R2 / 8 chunks
// starting at warpId * CHUNKS_PER_WARP. R2 = residues per thread.
template <int R2>
__device__ __forceinline__ uint32_t*
residue_data_offset(int rowIdx, int colIdx, int numMainCols, int numTotalCols, uint32_t* out) {
  constexpr int CHUNKS_PER_WARP = 32 * R2 / 8;
  int64_t u32PerRow = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t baseU32 = numMainCols / CVT_FP4_ELTS_PER_UINT32;
  int64_t rowBase = static_cast<int64_t>(rowIdx) * u32PerRow + baseU32;
  if constexpr (R2 == 8) {
    return out + rowBase + colIdx;
  } else {
    int warpId = colIdx / 32;
    int laneId = colIdx % 32;
    if (laneId >= CHUNKS_PER_WARP) return nullptr;
    return out + rowBase + warpId * CHUNKS_PER_WARP + laneId;
  }
}

// Residue SF address for the elt16 mapping: identical placement to
// cvt_residue_sf_get_offset, re-derived for 16-element threads.
template <class SFType, int R2>
__device__ __forceinline__ SFType*
residue_sf_offset(int rowIdx, int colIdx, int numTotalCols, int numMainCols, SFType* base) {
  constexpr int THREADS_PER_SF_GROUP = 16 / R2;
  int warpId = colIdx / 32;
  int laneId = colIdx % 32;
  if (laneId % THREADS_PER_SF_GROUP != 0) return nullptr;
  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int residueElementIdx = numMainCols + warpId * 32 * R2 + sfGroupId * 16;
  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int64_t SFOffset = compute_tiled_sf_offset<SFType>(rowIdx, kIdx, numTotalCols);
  return base + SFOffset;
}

// Main quantization for one 16-element SF vector (thread-local, no
// shuffles). Math follows the pack8 helpers line-for-line on each 8-element
// half so the output stays bit-identical. On return lo/hi hold the unscaled
// residues (original - dequant) for all 16 positions.
template <class Type, bool UE8M0_SF>
__device__ __forceinline__ u32x2 main_quant16(
    PackedVec<Type>& loVec,
    PackedVec<Type>& hiVec,
    float SFScaleVal,
    uint8_t* SFout,
    float2 (&loResidue)[4],
    float2 (&hiResidue)[4]) {
  auto localMax = __habs2(loVec.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(loVec.elts[i]));
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(hiVec.elts[i]));
  }
  float vecMax = float(__hmax(localMax.x, localMax.y));

  float SFValue = compute_sf_from_max(vecMax, SFScaleVal);
  SFValue = encode_sf_to_fp8<UE8M0_SF>(SFValue, SFout);
  float outputScale = compute_output_scale_precise(SFValue, SFScaleVal);

  float2 loOriginal[4];
  float2 hiOriginal[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<Type, half>) {
      loOriginal[i] = __half22float2(loVec.elts[i]);
      hiOriginal[i] = __half22float2(hiVec.elts[i]);
    } else {
      loOriginal[i] = __bfloat1622float2(loVec.elts[i]);
      hiOriginal[i] = __bfloat1622float2(hiVec.elts[i]);
    }
    loResidue[i].x = loOriginal[i].x * outputScale;
    loResidue[i].y = loOriginal[i].y * outputScale;
    hiResidue[i].x = hiOriginal[i].x * outputScale;
    hiResidue[i].y = hiOriginal[i].y * outputScale;
  }

  u32x2 packed;
  packed.lo = fp32x8_to_e2m1_with_dequant(loResidue);
  packed.hi = fp32x8_to_e2m1_with_dequant(hiResidue);

  float inverseOutputScale = outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    loResidue[i].x = loOriginal[i].x - loResidue[i].x * inverseOutputScale;
    loResidue[i].y = loOriginal[i].y - loResidue[i].y * inverseOutputScale;
    hiResidue[i].x = hiOriginal[i].x - hiResidue[i].x * inverseOutputScale;
    hiResidue[i].y = hiOriginal[i].y - hiResidue[i].y * inverseOutputScale;
  }
  return packed;
}

}  // namespace pack16

// r0125: 2 residues/thread, 8-thread SF groups, 4 source lanes/chunk
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2) cvt_fp16_to_fp4_residue_pack16_r0125_no_swap(
    int32_t numRows,
    int32_t numTotalCols,
    int32_t numMainCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 2>(rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask1) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(hiResidue, i1);

    float localMaxResidue = fmaxf(fabsf(rv0), fabsf(rv1));
    float sfGroupMaxResidue = warp_group_max<8>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;

    // Writer lane L packs residues from source lanes 4L..4L+3 (2 each).
    float2 residueChunk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int sourceThread = (laneId < 8 ? laneId : 0) * 4 + i;
      residueChunk[i] =
          make_float2(__shfl_sync(uint32_t(-1), rv0, sourceThread), __shfl_sync(uint32_t(-1), rv1, sourceThread));
    }

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<2>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

// r025: 4 residues/thread, 4-thread SF groups, 2 source lanes/chunk
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2) cvt_fp16_to_fp4_residue_pack16_r025_no_swap(
    int32_t numRows,
    int32_t numTotalCols,
    int32_t numMainCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 4>(rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    int i1 = __ffs(mask0 & (mask0 - 1)) - 1;
    int i2 = __ffs(mask1) - 1;
    int i3 = __ffs(mask1 & (mask1 - 1)) - 1;
    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(hiResidue, i2);
    float rv3 = pack16::pick_residue_value(hiResidue, i3);

    float localMaxResidue = fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3)));
    float sfGroupMaxResidue = warp_group_max<4>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);
    rv0 *= residueOutputScale;
    rv1 *= residueOutputScale;
    rv2 *= residueOutputScale;
    rv3 *= residueOutputScale;

    // Writer lane L packs residues from source lanes 2L, 2L+1 (4 each).
    int writerGroup = laneId < 16 ? laneId : 0;
    float2 residueChunk[4];
    residueChunk[0] =
        make_float2(__shfl_sync(uint32_t(-1), rv0, writerGroup * 2), __shfl_sync(uint32_t(-1), rv1, writerGroup * 2));
    residueChunk[1] =
        make_float2(__shfl_sync(uint32_t(-1), rv2, writerGroup * 2), __shfl_sync(uint32_t(-1), rv3, writerGroup * 2));
    residueChunk[2] = make_float2(
        __shfl_sync(uint32_t(-1), rv0, writerGroup * 2 + 1), __shfl_sync(uint32_t(-1), rv1, writerGroup * 2 + 1));
    residueChunk[3] = make_float2(
        __shfl_sync(uint32_t(-1), rv2, writerGroup * 2 + 1), __shfl_sync(uint32_t(-1), rv3, writerGroup * 2 + 1));

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<4>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      if (residueDataOut) {
        *residueDataOut = fp32_vec_to_e2m1(residueChunk);
      }
    }
  }
}

// r050: 8 residues/thread = exactly one chunk, 2-thread SF groups, no gather
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, 2) cvt_fp16_to_fp4_residue_pack16_r050_no_swap(
    int32_t numRows,
    int32_t numTotalCols,
    int32_t numMainCols,
    Type const* in,
    float const* SFScale,
    uint32_t* out,
    uint32_t* SFout,
    uint8_t const* channelIndices) {
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  uint8_t* extraSFStart = reinterpret_cast<uint8_t*>(SFout);
  int32_t effectiveRows = computeEffectiveRows(numRows);

  int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
  if (colIdx >= numMainCols / 16) {
    return;
  }
  uint8_t mask0 = channelIndices[colIdx * 2];
  uint8_t mask1 = channelIndices[colIdx * 2 + 1];
  int laneId = threadIdx.x % 32;

  for (int rowIdx = blockIdx.x; rowIdx < effectiveRows; rowIdx += gridDim.x) {
    bool validRow = rowIdx < numRows;
    PackedVec<Type> loVec{};
    PackedVec<Type> hiVec{};
    if (validRow) {
      load_packed_vec16(in + static_cast<int64_t>(rowIdx) * numMainCols + colIdx * 16, loVec, hiVec);
    }

    auto sfOut = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 1>(rowIdx, colIdx, numTotalCols, SFout);
    auto residueSFOut = pack16::residue_sf_offset<uint8_t, 8>(rowIdx, colIdx, numTotalCols, numMainCols, extraSFStart);

    float2 loResidue[4];
    float2 hiResidue[4];
    u32x2 packed = pack16::main_quant16<Type, UE8M0_SF>(loVec, hiVec, SFScaleVal, sfOut, loResidue, hiResidue);

    int i0 = __ffs(mask0) - 1;
    uint8_t rem = mask0 & (mask0 - 1);
    int i1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int i3 = __ffs(rem) - 1;
    int j0 = __ffs(mask1) - 1;
    rem = mask1 & (mask1 - 1);
    int j1 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j2 = __ffs(rem) - 1;
    rem = rem & (rem - 1);
    int j3 = __ffs(rem) - 1;

    float rv0 = pack16::pick_residue_value(loResidue, i0);
    float rv1 = pack16::pick_residue_value(loResidue, i1);
    float rv2 = pack16::pick_residue_value(loResidue, i2);
    float rv3 = pack16::pick_residue_value(loResidue, i3);
    float rv4 = pack16::pick_residue_value(hiResidue, j0);
    float rv5 = pack16::pick_residue_value(hiResidue, j1);
    float rv6 = pack16::pick_residue_value(hiResidue, j2);
    float rv7 = pack16::pick_residue_value(hiResidue, j3);

    float localMaxResidue = fmaxf(
        fmaxf(fmaxf(fabsf(rv0), fabsf(rv1)), fmaxf(fabsf(rv2), fabsf(rv3))),
        fmaxf(fmaxf(fabsf(rv4), fabsf(rv5)), fmaxf(fabsf(rv6), fabsf(rv7))));
    float sfGroupMaxResidue = warp_group_max<2>(localMaxResidue, laneId);
    float residueSFValue = compute_sf_from_max(sfGroupMaxResidue, SFScaleVal);
    residueSFValue = encode_sf_to_fp8<UE8M0_SF>(residueSFValue, residueSFOut);
    float residueOutputScale = compute_output_scale_precise(residueSFValue, SFScaleVal);

    float2 residueChunk[4];
    residueChunk[0] = make_float2(rv0 * residueOutputScale, rv1 * residueOutputScale);
    residueChunk[1] = make_float2(rv2 * residueOutputScale, rv3 * residueOutputScale);
    residueChunk[2] = make_float2(rv4 * residueOutputScale, rv5 * residueOutputScale);
    residueChunk[3] = make_float2(rv6 * residueOutputScale, rv7 * residueOutputScale);

    if (validRow) {
      int64_t outOffset = static_cast<int64_t>(rowIdx) * (numTotalCols / CVT_FP4_ELTS_PER_UINT32) + colIdx * 2;
      *reinterpret_cast<uint2*>(out + outOffset) = make_uint2(packed.lo, packed.hi);
      uint32_t* residueDataOut = pack16::residue_data_offset<8>(rowIdx, colIdx, numMainCols, numTotalCols, out);
      *residueDataOut = fp32_vec_to_e2m1(residueChunk);
    }
  }
}

// ───────────────────────────── launchers ───────────────────────────────────

inline void pack8_launch_config(int m, int n, int multiProcessorCount, dim3& grid, dim3& block) {
  int numThreads = std::max(n / CVT_FP4_ELTS_PER_THREAD, 32);
  // Round block size to full warps: the residue gathers use a constant full
  // shuffle mask; extra threads early-return via the colIdx bound check.
  int blockSize = std::min(((std::min(numThreads, 512) + 31) / 32) * 32, 512);
  block = dim3(blockSize);
  int const numBlocksPerSM = runtime_blocks_per_sm(blockSize);
  grid = dim3(
      std::min(computeEffectiveRows(m), multiProcessorCount * numBlocksPerSM),
      (numThreads + blockSize - 1) / blockSize);
}

inline void pack16_launch_config(int m, int n, int multiProcessorCount, dim3& grid, dim3& block) {
  int numThreads = std::max(n / 16, 32);
  int blockSize = std::min(((std::min(numThreads, 512) + 31) / 32) * 32, 512);
  block = dim3(blockSize);
  int const numBlocksPerSM = runtime_blocks_per_sm(blockSize);
  grid = dim3(
      std::min(computeEffectiveRows(m), multiProcessorCount * numBlocksPerSM),
      (numThreads + blockSize - 1) / blockSize);
}

// pack16 needs an even uint32 row stride for its 64-bit main-data stores.
inline bool use_kext_pack16(int elts_mode, int cc_major, int m, int n, int n_ext, int residue_per_8) {
  if (elts_mode == 8) return false;
  if (n % 16 != 0 || n_ext % 16 != 0) return false;
  if (elts_mode == 16) return true;
  if (cc_major != 10) return false;
  // B200 measurements: pack16 wins 1.2-1.7x at prefill for every ratio; at
  // decode it wins for r0125 but loses for r025/r050, whose doubled
  // per-thread residue state still costs at small M.
  if (residue_per_8 == 1) return true;
  return m >= 1024;
}

template <typename T>
void invokeFP4QuantizationKExt(
    int m,
    int n_ext,
    int n,
    T const* input,
    float const* SFScale,
    uint32_t* output,
    uint32_t* SFOuput,
    bool useUE8M0,
    uint8_t const* channelIndices,
    int residue_per_8,
    int elts_mode,
    int multiProcessorCount,
    int ccMajor,
    cudaStream_t stream) {
  bool pack16v = use_kext_pack16(elts_mode, ccMajor, m, n, n_ext, residue_per_8);

  dim3 grid, block;
  if (pack16v) {
    pack16_launch_config(m, n, multiProcessorCount, grid, block);
  } else {
    pack8_launch_config(m, n, multiProcessorCount, grid, block);
  }

  auto launch = [&](auto kernel) {
    kernel<<<grid, block, 0, stream>>>(m, n_ext, n, input, SFScale, output, SFOuput, channelIndices);
    cudaError_t status = cudaGetLastError();
    host::RuntimeCheck(
        status == cudaSuccess, "residue nvfp4 k_ext quant kernel launch failed: ", cudaGetErrorString(status));
  };

  // UE8M0 scales are not used by this integration; keep the template plumbed
  // through but always instantiate the E4M3 variant.
  (void)useUE8M0;
  if (pack16v) {
    switch (residue_per_8) {
      case 1:
        return launch(cvt_fp16_to_fp4_residue_pack16_r0125_no_swap<T, false>);
      case 2:
        return launch(cvt_fp16_to_fp4_residue_pack16_r025_no_swap<T, false>);
      case 4:
        return launch(cvt_fp16_to_fp4_residue_pack16_r050_no_swap<T, false>);
      default:
        break;
    }
  } else {
    switch (residue_per_8) {
      case 1:
        return launch(cvt_fp16_to_fp4_residue_pack8_no_swap<T, false, 1>);
      case 2:
        return launch(cvt_fp16_to_fp4_residue_pack8_no_swap<T, false, 2>);
      case 4:
        return launch(cvt_fp16_to_fp4_residue_pack8_no_swap<T, false, 4>);
      default:
        break;
    }
  }
  host::Panic("residue nvfp4 k_ext quant: unsupported residue_per_8 (expected 1, 2, or 4)");
}

}  // namespace residue_nvfp4

template <typename DType>
void scaled_fp4_quant_with_mask(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView output_sf,
    tvm::ffi::TensorView input_sf,
    tvm::ffi::TensorView channel_masks,
    int64_t n_ext,
    int64_t residue_per_8,
    int64_t elts_mode) {
  using namespace host;
  namespace rn = residue_nvfp4;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto M = SymbolicSize{"m"};
  auto K = SymbolicSize{"k"};

  TensorMatcher({M, K})  //
      .with_dtype<DType>()
      .with_device(device)
      .verify(input);

  const auto m = static_cast<int32_t>(M.unwrap());
  const auto k = static_cast<int32_t>(K.unwrap());
  RuntimeCheck(k % 16 == 0, "residue nvfp4 k_ext quant: K must be a multiple of 16, got ", k);
  RuntimeCheck(
      residue_per_8 == 1 || residue_per_8 == 2 || residue_per_8 == 4,
      "residue nvfp4 k_ext quant: residue_per_8 must be 1, 2, or 4 (ratio "
      "1.0 is served by the M-extension path), got ",
      residue_per_8);
  RuntimeCheck(
      n_ext == k + (k / 8) * residue_per_8, "residue nvfp4 k_ext quant: n_ext must equal K + K/8*residue_per_8");

  auto OutLen = SymbolicSize{"out_len"};
  auto SfLen = SymbolicSize{"sf_len"};
  auto MaskLen = SymbolicSize{"mask_len"};
  TensorMatcher({OutLen}).with_dtype<uint8_t>().with_device(device).verify(output);
  TensorMatcher({SfLen}).with_dtype<uint8_t>().with_device(device).verify(output_sf);
  TensorMatcher({MaskLen}).with_dtype<uint8_t>().with_device(device).verify(channel_masks);
  RuntimeCheck(
      output.numel() >= static_cast<int64_t>(m) * n_ext / 2, "residue nvfp4 k_ext quant: output tensor too small");
  RuntimeCheck(channel_masks.numel() >= k / 8, "residue nvfp4 k_ext quant: channel_masks must have K/8 bytes");

  const int64_t num_m_tiles = (m + 127) / 128;
  const int64_t num_k_tiles = (n_ext + 63) / 64;
  RuntimeCheck(
      output_sf.numel() >= num_m_tiles * num_k_tiles * 512,
      "residue nvfp4 k_ext quant: output_sf tensor too small for the swizzled layout");

  auto InSfLen = SymbolicSize{"in_sf_len"};
  TensorMatcher({InSfLen}).with_dtype<float>().with_device(device).verify(input_sf);
  RuntimeCheck(input_sf.numel() >= 1, "residue nvfp4 k_ext quant: input_sf must have 1 element");

  const auto dl_device = device.unwrap();
  const auto info = rn::query_device_info(dl_device.device_id);
  const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dl_device.device_type, dl_device.device_id));

  rn::invokeFP4QuantizationKExt(
      m,
      static_cast<int>(n_ext),
      k,
      static_cast<DType const*>(input.data_ptr()),
      static_cast<float const*>(input_sf.data_ptr()),
      reinterpret_cast<uint32_t*>(output.data_ptr()),
      reinterpret_cast<uint32_t*>(output_sf.data_ptr()),
      /*useUE8M0=*/false,
      static_cast<uint8_t const*>(channel_masks.data_ptr()),
      static_cast<int>(residue_per_8),
      static_cast<int>(elts_mode),
      info.multi_processor_count,
      info.cc_major,
      stream);
}

}  // namespace sglang
