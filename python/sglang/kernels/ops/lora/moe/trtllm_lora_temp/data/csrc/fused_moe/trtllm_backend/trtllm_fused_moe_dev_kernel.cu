/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/utils.cuh"
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const &input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element,
                                 U::kElements>
      converter;
  return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x) { return x / (1.0f + expf(-x)); }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on
  // primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens;
       tokenIdx += gridDim.z) {
    // Look over experts per token
    for (int k = blockIdx.y; k < params.topK; k += gridDim.y) {
      int const expandedIdx = tokenIdx * params.topK + k;
      int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

      // Loop over hidden dim
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
           hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
        if (permutedIdx == -1) {
          if (params.activationLoraInputOutPtr != nullptr) {
            int64_t const activationIdx =
                (int64_t)expandedIdx * (params.innerDim / 2) + hiddenIdx;
            params.activationLoraInputOutPtr[activationIdx] =
                cutlass::bfloat16_t(0.0f);
          }
          continue;
        }

        // Use int64_t to avoid overflow when permutedIdx * innerDim > INT32_MAX
        int64_t const permBase = (int64_t)permutedIdx * params.innerDim;

        // Contiguous input: gate = col hiddenIdx, up = col innerDim/2 +
        // hiddenIdx. Interleaved input (fused de-interleave): gate = col
        // 2*hiddenIdx, up = col 2*hiddenIdx+1.
        float x1, x2;
        if (params.interleavedGateUpInput) {
          x1 = (float)params.inPtr[permBase + 2 * hiddenIdx];
          x2 = (float)params.inPtr[permBase + 2 * hiddenIdx + 1];
        } else {
          x1 = (float)params.inPtr[permBase + hiddenIdx];
          x2 = (float)params.inPtr[permBase + hiddenIdx + params.innerDim / 2];
        }
        if (params.gateUpLoraDeltaPtr != nullptr) {
          int64_t const loraBaseIdx =
              (int64_t)expandedIdx * params.innerDim + hiddenIdx;
          x1 += static_cast<float>(
              params.gateUpLoraDeltaPtr[loraBaseIdx + params.innerDim / 2]);
          x2 += static_cast<float>(params.gateUpLoraDeltaPtr[loraBaseIdx]);
        }

        float act = silu(x2);
        Type out = (Type)(act * x1);
        if (params.activationLoraInputOutPtr != nullptr) {
          int64_t const activationIdx =
              (int64_t)expandedIdx * (params.innerDim / 2) + hiddenIdx;
          params.activationLoraInputOutPtr[activationIdx] =
              static_cast<cutlass::bfloat16_t>(act * x1);
        }

        int64_t const outIdx =
            (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
        params.outPtr[outIdx] = out;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Vectorized bf16 SwiGLU+LoRA activation for the FP4-LoRA path (interleaved
// gate/up input). Each thread processes 4 consecutive hidden pairs per step:
// one 128-bit load for the interleaved gate/up, two 64-bit loads for the LoRA
// delta, and 64-bit stores. This raises memory-level parallelism toward the HBM
// bandwidth roofline (vs the scalar activationKernel, which does 1
// element/thread with per-element scalar loads). Numerically identical to
// activationKernel (same per-element float math + silu), so the testbed asserts
// bitwise equality. Requires interleaved bf16 input and (innerDim/2) % 4 == 0
// (run() guards this).
__global__ void activationKernelOpt(
    cutlass::bfloat16_t const
        *__restrict__ inPtr, // interleaved gate/up GEMM1 output
    cutlass::bfloat16_t *__restrict__ outPtr, // activated [.., innerDim/2]
    cutlass::bfloat16_t const *__restrict__ gateUpLoraDeltaPtr,  // may be null
    cutlass::bfloat16_t *__restrict__ activationLoraInputOutPtr, // may be null
    int const *__restrict__ expandedIdxToPermutedIdx, int innerDim,
    int numTokens, int topK) {
  int const innerHalf = innerDim / 2;
  for (int tokenIdx = blockIdx.z; tokenIdx < numTokens; tokenIdx += gridDim.z) {
    for (int k = blockIdx.y; k < topK; k += gridDim.y) {
      int const expandedIdx = tokenIdx * topK + k;
      int const permutedIdx = expandedIdxToPermutedIdx[expandedIdx];
      for (int h = (threadIdx.x + blockDim.x * blockIdx.x) * 4; h < innerHalf;
           h += blockDim.x * gridDim.x * 4) {
        int64_t const liBase = (int64_t)expandedIdx * innerHalf + h;
        if (permutedIdx == -1) {
          if (activationLoraInputOutPtr != nullptr) {
            *reinterpret_cast<int2 *>(&activationLoraInputOutPtr[liBase]) =
                make_int2(0, 0);
          }
          continue;
        }

        // gate/up: 8 interleaved bf16 (g0,u0,...,g3,u3) via one 128-bit load.
        int64_t const permBase = (int64_t)permutedIdx * innerDim + 2 * h;
        int4 const rawGU = *reinterpret_cast<int4 const *>(&inPtr[permBase]);
        cutlass::bfloat16_t const *gu =
            reinterpret_cast<cutlass::bfloat16_t const *>(&rawGU);

        float gate[4], up[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          gate[j] = (float)gu[2 * j];   // even col 2k   -> x1 (gate)
          up[j] = (float)gu[2 * j + 1]; // odd col  2k+1 -> x2 (up)
        }

        if (gateUpLoraDeltaPtr != nullptr) {
          // delta is contiguous [gate | up] per expandedIdx row: up +=
          // delta[h], gate += delta[h+innerHalf].
          int64_t const dBase = (int64_t)expandedIdx * innerDim + h;
          int2 const rawUp =
              *reinterpret_cast<int2 const *>(&gateUpLoraDeltaPtr[dBase]);
          int2 const rawGate = *reinterpret_cast<int2 const *>(
              &gateUpLoraDeltaPtr[dBase + innerHalf]);
          cutlass::bfloat16_t const *dUp =
              reinterpret_cast<cutlass::bfloat16_t const *>(&rawUp);
          cutlass::bfloat16_t const *dGate =
              reinterpret_cast<cutlass::bfloat16_t const *>(&rawGate);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            up[j] += (float)dUp[j];
            gate[j] += (float)dGate[j];
          }
        }

        __align__(8) cutlass::bfloat16_t res[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          res[j] = (cutlass::bfloat16_t)(silu(up[j]) * gate[j]);
        }
        int2 const packed = *reinterpret_cast<int2 const *>(res);

        int64_t const outBase = (int64_t)permutedIdx * innerHalf + h;
        *reinterpret_cast<int2 *>(&outPtr[outBase]) = packed;
        if (activationLoraInputOutPtr != nullptr) {
          *reinterpret_cast<int2 *>(&activationLoraInputOutPtr[liBase]) =
              packed;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4Max {
  __device__ __forceinline__ float4 operator()(float4 const &a,
                                               float4 const &b) const {
    float4 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    result.z = fmaxf(a.z, b.z);
    result.w = fmaxf(a.w, b.w);
    return result;
  }
};

struct Float2Max {
  __device__ __forceinline__ float2 operator()(float2 const &a,
                                               float2 const &b) const {
    float2 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    return result;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, int size>
__device__ __forceinline__ VecType packedTypeFromArray(float data[size]) {
  return {};
}

template <>
__device__ __forceinline__ float4
packedTypeFromArray<float4, 4>(float data[4]) {
  float4 result;
  result.x = data[0];
  result.y = data[1];
  result.z = data[2];
  result.w = data[3];
  return result;
}

template <>
__device__ __forceinline__ float2
packedTypeFromArray<float2, 2>(float data[2]) {
  float2 result;
  result.x = data[0];
  result.y = data[1];
  return result;
}

template <>
__device__ __forceinline__ float packedTypeFromArray<float, 1>(float data[1]) {
  return data[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PackedType, int size>
__device__ __forceinline__ cutlass::Array<float, size>
arrayFromPackedType(PackedType data) {
  return cutlass::Array<float, size>{};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 4>
arrayFromPackedType<float4, 4>(float4 data) {
  return cutlass::Array<float, 4>{data.x, data.y, data.z, data.w};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 2>
arrayFromPackedType<float2, 2>(float2 data) {
  return cutlass::Array<float, 2>{data.x, data.y};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 1>
arrayFromPackedType<float, 1>(float data) {
  return cutlass::Array<float, 1>{data};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_TOKENS_PER_CTA> struct KernelTraits;

template <> struct KernelTraits<4> {
  using MaxOp = Float4Max;
  using PackedType = float4;
};

template <> struct KernelTraits<2> {
  using MaxOp = Float2Max;
  using PackedType = float2;
};

template <> struct KernelTraits<1> {
  using MaxOp = cuda::maximum<>;
  using PackedType = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA = 128;

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  int32_t constexpr NumTokensPerCta = KernelParams::NumTokensPerCta;
  using KernelTraits = KernelTraits<NumTokensPerCta>;
  using MaxOp = typename KernelTraits::MaxOp;
  using PackedType = typename KernelTraits::PackedType;
  using BlockReduce =
      cub::BlockReduce<PackedType, DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA>;

  __shared__ float s_scaleOutArr[NumTokensPerCta];
  __shared__ typename BlockReduce::TempStorage tempStorage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on
  // primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // The largest (finite) value that can be represented using E4m3.
  float constexpr E4m3MaxVal{448.f};

  int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
  // Loop over tokens
  float scale1Arr[NumTokensPerCta];
  float scale2Arr[NumTokensPerCta];
  float dataX1Arr[NumTokensPerCta];
  float dataX2Arr[NumTokensPerCta];
  float outArr[NumTokensPerCta];
  float absOutArr[NumTokensPerCta];
  int permutedIdxArr[NumTokensPerCta];

  // Loop over tokens
  for (int k = blockIdx.z; k < params.topK; k += gridDim.z) {
    for (int tokenCtaIdx = blockIdx.y * NumTokensPerCta;
         tokenCtaIdx < params.numTokens;
         tokenCtaIdx += gridDim.y * NumTokensPerCta) {
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
           hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta;
             tokenInCtaIdx++) {
          scale1Arr[tokenInCtaIdx] = 0.0f;
          scale2Arr[tokenInCtaIdx] = 0.0f;
          dataX1Arr[tokenInCtaIdx] = 0.0f;
          dataX2Arr[tokenInCtaIdx] = 0.0f;
          outArr[tokenInCtaIdx] = 0.0f;
          absOutArr[tokenInCtaIdx] = 0.0f;
        }
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta;
             tokenInCtaIdx++) {
          int const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (tokenIdx >= params.numTokens) {
            break;
          }

          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
          permutedIdxArr[tokenInCtaIdx] = permutedIdx;
          if (permutedIdx == -1) {
            continue;
          }

          // Process blocks for this CTA
          // Use int64_t to avoid overflow when permutedIdx * innerDim >
          // INT32_MAX
          int64_t const baseIdx =
              (int64_t)permutedIdx * params.innerDim + hiddenIdx;

          int64_t const scale1Idx =
              (int64_t)permutedIdx +
              (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
          int64_t const scale2Idx =
              (int64_t)permutedIdx +
              (int64_t)totalNumPaddedTokens *
                  ((hiddenIdx / 128) + (params.innerDim / 2 / 128));

          scale1Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale1Idx];
          scale2Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale2Idx];
          dataX1Arr[tokenInCtaIdx] = static_cast<float>(params.inPtr[baseIdx]);
          dataX2Arr[tokenInCtaIdx] =
              static_cast<float>(params.inPtr[baseIdx + params.innerDim / 2]);
        }

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta;
             tokenInCtaIdx++) {
          float x1 = scale1Arr[tokenInCtaIdx] * dataX1Arr[tokenInCtaIdx];
          float x2 = scale2Arr[tokenInCtaIdx] * dataX2Arr[tokenInCtaIdx];
          auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (params.gateUpLoraDeltaPtr != nullptr &&
              tokenIdx < params.numTokens) {
            int const expandedIdx = tokenIdx * params.topK + k;
            int64_t const loraBaseIdx =
                (int64_t)expandedIdx * params.innerDim + hiddenIdx;
            x1 += static_cast<float>(
                params.gateUpLoraDeltaPtr[loraBaseIdx + params.innerDim / 2]);
            x2 += static_cast<float>(params.gateUpLoraDeltaPtr[loraBaseIdx]);
          }
          float act = silu(x2);
          float out = act * x1;
          outArr[tokenInCtaIdx] = out;
          absOutArr[tokenInCtaIdx] = fabsf(out);
        }

        auto absOutPacked =
            packedTypeFromArray<PackedType, NumTokensPerCta>(absOutArr);
        auto aMaxPacked =
            BlockReduce(tempStorage).Reduce(absOutPacked, MaxOp{});
        auto aMaxArr =
            arrayFromPackedType<PackedType, NumTokensPerCta>(aMaxPacked);

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta;
             tokenInCtaIdx++) {
          if (threadIdx.x == 0) {
            auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
            if (tokenIdx >= params.numTokens) {
              break;
            }
            int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
            if (permutedIdx == -1) {
              continue;
            }
            // Make sure the scale is strictly positive to avoid division by
            // zero in case the maximum is zero.
            float scaleOut = fmaxf(aMaxArr[tokenInCtaIdx] / E4m3MaxVal,
                                   std::numeric_limits<float>::min());
            s_scaleOutArr[tokenInCtaIdx] = scaleOut;
            int64_t const scaleOut_idx =
                (int64_t)permutedIdxArr[tokenInCtaIdx] +
                (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
            params.outDqSfsPtr[scaleOut_idx] = scaleOut;
          }
        }
        __syncthreads();

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta;
             tokenInCtaIdx++) {
          auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (tokenIdx >= params.numTokens) {
            break;
          }
          int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
          if (permutedIdx == -1) {
            if (params.activationLoraInputOutPtr != nullptr) {
              int const expandedIdx = tokenIdx * params.topK + k;
              int64_t const activationIdx =
                  (int64_t)expandedIdx * (params.innerDim / 2) + hiddenIdx;
              params.activationLoraInputOutPtr[activationIdx] =
                  cutlass::bfloat16_t(0.0f);
            }
            continue;
          }
          float const scaleOut = s_scaleOutArr[tokenInCtaIdx];
          int64_t const outIdx =
              (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
          params.outPtr[outIdx] =
              static_cast<Type>(outArr[tokenInCtaIdx] / scaleOut);
          if (params.activationLoraInputOutPtr != nullptr) {
            int const expandedIdx = tokenIdx * params.topK + k;
            int64_t const activationIdx =
                (int64_t)expandedIdx * (params.innerDim / 2) + hiddenIdx;
            params.activationLoraInputOutPtr[activationIdx] =
                static_cast<cutlass::bfloat16_t>(outArr[tokenInCtaIdx]);
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const &data, void *stream) {
  if (data.mDtypeElt == tg::Dtype::E2m1) {
    // Note: this should be unreachable because the options are checked
    // beforehand. E2m1 requires using higher-precision intermediate data
    // (bf16).
    FLASHINFER_CHECK(false, "Activation with E2m1_t isn't supported.");
    return;
  }

  if (data.mUseDeepSeekFp8) {
    constexpr int NUM_ELTS_PER_LOAD = 1;
    constexpr int NUM_ELTS_PER_SF = 128;

    int device{-1};
    cudaGetDevice(&device);
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

    // Output dimension is innerDim / 2, and each scale block is 128 elements
    int const outputDim = data.innerDim / 2;
    int const numScaleBlocks =
        (outputDim + NUM_ELTS_PER_SF - 1) / NUM_ELTS_PER_SF;
    int const gridSizeX =
        (numScaleBlocks + NUM_ELTS_PER_LOAD - 1) / NUM_ELTS_PER_LOAD;

    auto numCtas = gridSizeX * data.numTokens * data.topK;
    // FIXME: This is heruistic based on very short benchmark.
    int numTokensPerCta = 1;
    if (numCtas > numSms * 32) {
      numTokensPerCta = 4;
    } else if (numCtas > numSms * 4) {
      numTokensPerCta = 2;
    } else {
      numTokensPerCta = 1;
    }

    int const gridSizeY = std::min(
        8192, (data.numTokens + numTokensPerCta - 1) / numTokensPerCta);

    const dim3 grid(gridSizeX, gridSizeY, data.topK);

    LAUNCH_ACTIVATION(data, activationDeepSeekKernel, numTokensPerCta, grid,
                      DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA, 0, stream);
  } else if (data.actOptMode == 1 && data.mDtypeElt == tg::Dtype::Bfloat16 &&
             data.interleavedGateUpInput && (data.innerDim / 2) % 4 == 0) {
    int const numThreads = 256;
    int const innerHalf = data.innerDim / 2;
    int const defaultGx = (innerHalf / 4 + numThreads - 1) / numThreads;
    int const gridX =
        data.actGridXOverride > 0 ? data.actGridXOverride : defaultGx;
    const dim3 grid(gridX, data.topK, std::min(8192, data.numTokens));

    activationKernelOpt<<<grid, numThreads, 0, (cudaStream_t)stream>>>(
        static_cast<cutlass::bfloat16_t const *>(data.inPtr),
        static_cast<cutlass::bfloat16_t *>(data.outPtr),
        data.gateUpLoraDeltaPtr, data.activationLoraInputOutPtr,
        data.expandedIdxToPermutedIdx, data.innerDim, data.numTokens,
        data.topK);
  } else {
    int const numThreads = 256;
    int const gridX = data.actGridXOverride > 0 ? data.actGridXOverride
                                                : (data.innerDim / 128);
    const dim3 grid(gridX, data.topK, std::min(8192, data.numTokens));

    LAUNCH_ACTIVATION(data, activationKernel, 1, grid, numThreads, 0, stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace convertsf {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

namespace dev {
// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF
// tensor where dataRowIdx and dataBlkColIdx are the respective indices of the
// row and the block of 16 elts from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  // The number of rows of SF per block.
  static int32_t constexpr NumRowsPerSfBlock = 128;
  // The number of cols of SF per block.
  static int32_t constexpr NumColsPerSfBlock = 4;
  // The size of each SF block.
  static int32_t constexpr NumBytesPerSfBlock =
      NumRowsPerSfBlock * NumColsPerSfBlock;

  // The number of rows of data per SF block.
  static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
  // The number of cols of blocks of data per SF block.
  static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

  // The row of the SF block in the SF tensor.
  int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
  // The col of the SF block in the SF tensor.
  int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
  // The blocks are stored row-major in the tensor of scaling factors.
  int sfBlkIdx =
      sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

  // Find the row in the SF block.
  int sfRowIdx =
      (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
  // Find the col in the SF block.
  int sfColIdx = (dataBlkColIdx % 4);

  // Compute the offset in bytes.
  return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock +
         sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the
// corresponding scaling factor in the SF tensor. Optionally, a startTokenIndex
// can be provided if the first token is not the start token in the SF tensor.
// This is useful when inflight batching is enabled in TRT-LLM, where the
// context and generation output are stored as one output tensor. In this case,
// the generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  // The number of elements per sf.
  int32_t constexpr NumEltsPerSf = 16;
  // The GMEM offset of the output element.
  int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
  // The row/col indices of the corresponding SF element.
  int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
  int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
  // Compute the SF offset.
  return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer
// subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(void const *gmemOutPtr,
                                      void const *gmemBasePtr,
                                      int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  return getSfOffset<NumBitsPerElt>(
      reinterpret_cast<char const *>(gmemOutPtr) -
          reinterpret_cast<char const *>(gmemBasePtr),
      hiddenDim, startTokenIdx);
}

} // namespace dev

// TODO: it would be nice to move some of that logic to Fp4Utils.h
template <tg::SfLayout Layout>
inline __device__ int32_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  if constexpr (Layout == tg::SfLayout::Linear) {
    return numDataBlksPerRow * dataRowIdx + dataBlkColIdx;
  } else if constexpr (Layout == tg::SfLayout::R128c4) {
    return static_cast<int32_t>(
        dev::getSfOffset(dataRowIdx, dataBlkColIdx, numDataBlksPerRow));
  } else if constexpr (Layout == tg::SfLayout::R8c4 ||
                       Layout == tg::SfLayout::R8c16) {
    static int32_t constexpr NumRowsPerSfBlock = 8;
    static int32_t constexpr NumColsPerSfBlock =
        (Layout == tg::SfLayout::R8c4) ? 4 : 16;
    static int32_t constexpr NumBytesPerSfBlock =
        NumRowsPerSfBlock * NumColsPerSfBlock;
    int sfBlkRowIdx = dataRowIdx / NumRowsPerSfBlock;
    int sfBlkColIdx = dataBlkColIdx / NumColsPerSfBlock;
    int sfBlkIdx =
        sfBlkRowIdx * numDataBlksPerRow / NumColsPerSfBlock + sfBlkColIdx;
    int sfRowIdx = dataRowIdx % NumRowsPerSfBlock;
    int sfColIdx = dataBlkColIdx % NumColsPerSfBlock;
    return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock +
           sfColIdx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <tg::SfLayout LayoutSrc, tg::SfLayout LayoutDst, typename KernelParams>
__device__ void convertSfCommon(KernelParams params) {
  // Note: it's assumed that the number of scaling factors per row is a multiple
  // of 4.
  constexpr int VecSize = 4;
  using VecType = uint32_t;
  static_assert(sizeof(VecType) == VecSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Immediately trigger the secondary kernel when using PDL, then wait on
  // primary.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // TODO: consider optimizing if used in production.
  // This is a naive kernel. It's not doing coalesced loads.

  int const numSfPerRow = params.hiddenDimSf;

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens;
       tokenIdx += gridDim.y) {
    for (int hiddenSfVecIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenSfVecIdx < numSfPerRow / VecSize;
         hiddenSfVecIdx += blockDim.x * gridDim.x) {
      // Index of the first SF in the vector.
      int const hiddenSfIdx = VecSize * hiddenSfVecIdx;

      // Load scale factors.
      int sfIdxIn = getSfOffset<LayoutSrc>(tokenIdx, hiddenSfIdx, numSfPerRow);
      const VecType sfVec =
          reinterpret_cast<VecType const *>(params.inSfPtr)[sfIdxIn / VecSize];

      // Store scale factors.
      int const sfIdxOut =
          getSfOffset<LayoutDst>(tokenIdx, hiddenSfIdx, numSfPerRow);
      reinterpret_cast<VecType *>(params.outSfPtr)[sfIdxOut / VecSize] = sfVec;
    }
  }
}

#define CONVERT_FP4_SF_KERNEL(LayoutSrc, LayoutDst)                            \
  template <typename KernelParams>                                             \
  __global__ void convertSf##LayoutSrc##To##LayoutDst##Kernel(                 \
      KernelParams params) {                                                   \
    convertSfCommon<tg::SfLayout::LayoutSrc, tg::SfLayout::LayoutDst>(params); \
  }
// We only need a conversion to the linear layout.
CONVERT_FP4_SF_KERNEL(R128c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c16, Linear);
#undef CONVERT_FP4_SF_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const &data, void *stream) {
  constexpr int VecSize = 4;
  int const numThreads = 128;
  int const numBlocksX =
      (data.hiddenDimSf / VecSize - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);
#define CONVERT_FP4_SF_LAUNCH(LayoutSrc, LayoutDst)                            \
  if (data.sfLayoutSrc == tg::SfLayout::LayoutSrc &&                           \
      data.sfLayoutDst == tg::SfLayout::LayoutDst) {                           \
    LAUNCH_PDL(data, false, cutlass::float_e4m3_t,                             \
               convertSf##LayoutSrc##To##LayoutDst##Kernel, numBlocks,         \
               numThreads, 0, stream);                                         \
    return;                                                                    \
  }
  CONVERT_FP4_SF_LAUNCH(R128c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c16, Linear);
#undef CONVERT_FP4_SF_LAUNCH
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace convertsf

namespace permute {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on
  // primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens;
       tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenIdx < params.hiddenDim; hiddenIdx += blockDim.x * gridDim.x) {
      // Load chunk of token into registers
      const Type data = params.inPtr[tokenIdx * params.hiddenDim + hiddenIdx];

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        // Skip EP-unrouted (token, k) slots: under expert parallelism the
        // routing emits permutedIdx == -1 for slots whose expert is not on this
        // rank. The write was previously unguarded, so at prefill scale the
        // negative index OOBs the output buffer (illegal memory access).
        // moe::dev::finalize already skips permutedIdx == -1.
        if (permutedIdx < 0) {
          continue;
        }
        params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
      }
    }
    if (params.useDeepSeekFp8) {
      for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x;
           scaleIdx < params.hiddenDim / 128;
           scaleIdx += blockDim.x * gridDim.x) {
        for (int k = 0; k < params.topK; k++) {
          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

          int const idx_in = tokenIdx + params.numTokens * scaleIdx;
          int const idx_out =
              permutedIdx + params.totalNumPaddedTokens[0] * scaleIdx;

          params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const &data, void *stream) {
  int const numThreads = 256;
  int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);

  LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens;
       tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenIdx < params.hiddenDim; hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float data = 0.0F;

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

        if (permutedIdx == -1) {
          continue;
        }

        if (params.expertWeightsPtr != nullptr) {
          TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
          data += float{scale} *
                  float{params.inPtr[permutedIdx * params.hiddenDimPadded +
                                     hiddenIdx]};
        } else {
          data += float{
              params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        }
      }

      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] =
          static_cast<Type>(data);
    }
  }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

__device__ float4 vectorizedLoadPtx(float4 const *ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and
// performs the final skip connection.
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int MaxTopK = 64;

typedef struct __CUDA_ALIGN__(4) {
  cutlass::bfloat16_t array[2];
} bfloat16_2;

typedef struct __CUDA_ALIGN__(8) {
  cutlass::bfloat16_t array[4];
} bfloat16_4;

typedef struct __CUDA_ALIGN__(8) {
  half array[4];
} half_4;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_> struct ScaleTraitsStruct;

template <> struct ScaleTraitsStruct<1, cutlass::bfloat16_t> {
  using PackedType = cutlass::bfloat16_t;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 1>;
};

template <> struct ScaleTraitsStruct<2, cutlass::bfloat16_t> {
  using PackedType = bfloat16_2;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 2>;
};

template <> struct ScaleTraitsStruct<4, cutlass::bfloat16_t> {
  using PackedType = bfloat16_4;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 4>;
};

template <> struct ScaleTraitsStruct<1, float> {
  using PackedType = float;
  using ArrayType = cutlass::Array<float, 1>;
};

template <> struct ScaleTraitsStruct<2, float> {
  using PackedType = float2;
  using ArrayType = cutlass::Array<float, 2>;
};

template <> struct ScaleTraitsStruct<4, float> {
  using PackedType = float4;
  using ArrayType = cutlass::Array<float, 4>;
};

template <> struct ScaleTraitsStruct<1, half> {
  using PackedType = half;
  using ArrayType = cutlass::Array<half, 1>;
};

template <> struct ScaleTraitsStruct<2, half> {
  using PackedType = half2;
  using ArrayType = cutlass::Array<half, 2>;
};

template <> struct ScaleTraitsStruct<4, half> {
  using PackedType = half_4;
  using ArrayType = cutlass::Array<half, 4>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_> struct FinalizeTraits;

template <typename TypeExpW_> struct FinalizeTraits<1, TypeExpW_> {
  using IdxPackedType = int;
  using IdxArrayType = cutlass::Array<int, 1>;
  using ScaleTraits = ScaleTraitsStruct<1, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_> struct FinalizeTraits<2, TypeExpW_> {
  using IdxPackedType = int2;
  using IdxArrayType = cutlass::Array<int, 2>;
  using ScaleTraits = ScaleTraitsStruct<2, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_> struct FinalizeTraits<4, TypeExpW_> {
  using IdxPackedType = int4;
  using IdxArrayType = cutlass::Array<int, 4>;
  using ScaleTraits = ScaleTraitsStruct<4, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;
  int constexpr TopKUnrollFactor = KernelParams::TopKUnrollFactor;

  static_assert(TopKUnrollFactor == 1 || TopKUnrollFactor == 2 ||
                    TopKUnrollFactor == 4,
                "TopKUnrollFactor must be 1, 2, or 4");
  using FinalizeTraits = FinalizeTraits<TopKUnrollFactor, TypeExpW>;
  using IdxPackedType = typename FinalizeTraits::IdxPackedType;
  using IdxArrayType = typename FinalizeTraits::IdxArrayType;
  using ScalePackedType = typename FinalizeTraits::ScalePackedType;
  using ScaleArrayType = typename FinalizeTraits::ScaleArrayType;

  int const hiddenDimPaddedBits =
      params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
  int const hiddenDimBits =
      params.hiddenDim * cutlass::sizeof_bits<Type>::value;
  assert(hiddenDimPaddedBits % 128 == 0);
  assert(hiddenDimBits % 128 == 0);

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / cutlass::sizeof_bits<Type>::value;
  using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol =
      params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;
  bool const useScale = params.expertWeightsPtr != nullptr;

  __shared__ ScalePackedType scaleArrSmem[MaxTopK / TopKUnrollFactor];
  __shared__ IdxPackedType permutedIdxArrSmem[MaxTopK / TopKUnrollFactor];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < params.topK / TopKUnrollFactor;
       kChunkIdx += blockDim.x) {
    int const expandedIdx =
        tokenIdx * params.topK + kChunkIdx * TopKUnrollFactor;
    auto permutedIdxPacked = reinterpret_cast<IdxPackedType const *>(
        params.expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    auto scalePacked =
        useScale ? reinterpret_cast<ScalePackedType const *>(
                       params.expertWeightsPtr)[expandedIdx / TopKUnrollFactor]
                 : ScalePackedType{TypeExpW(1.f)};

    scaleArrSmem[kChunkIdx] = scalePacked;
    permutedIdxArrSmem[kChunkIdx] = permutedIdxPacked;
  }

  auto const offset = tokenIdx * params.hiddenDim;
  Type *outputPtr = params.outPtr + offset;
  auto *outElemPtr = reinterpret_cast<OutputElem *>(outputPtr);
  auto const *inElemPtr = reinterpret_cast<InputElem const *>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif
  __syncthreads();

  for (int elemIndex = startOffset; elemIndex < numElemsInCol;
       elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0);
    for (int kChunkIdx = 0; kChunkIdx < params.topK / TopKUnrollFactor;
         kChunkIdx++) {
      auto permutedIdxArr = *reinterpret_cast<IdxArrayType const *>(
          &permutedIdxArrSmem[kChunkIdx]);
      InputElem inputElemArr[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }

        auto const *inputPermutedPtr =
            inElemPtr + permutedIdx * numElemsInPaddedCol;

        float4 input = vectorizedLoadPtx(
            reinterpret_cast<float4 const *>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const *>(&input);
      }
      auto scaleArr =
          *reinterpret_cast<ScaleArrayType const *>(&scaleArrSmem[kChunkIdx]);
      auto const scaleFloatArr =
          arrayConvert<ScaleArrayType, cutlass::Array<float, TopKUnrollFactor>>(
              scaleArr);

#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto scale = useScale ? scaleFloatArr[ki] : 1.0f;
        ComputeElem expertResult =
            arrayConvert<InputElem, ComputeElem>(inputElemArr[ki]);
        threadOutput = threadOutput + scale * expertResult;
      }
    }
    OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
    outElemPtr[elemIndex] = outputElem;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using BlockReduce = cub::BlockReduce<float, 128>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens;
       tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenIdx < params.hiddenDim; hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float acc = 0.0f;

      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
        int const scaleIdx =
            permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
        float const blockScale =
            params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

        float const expertProb =
            (float)params.expertWeightsPtr[tokenIdx * params.topK + k];

        float const scale = expertProb * blockScale;
        acc +=
            scale *
            static_cast<float>(
                params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]);
      }

      // The largest (finite) value that can be represented using E4m3.
      float constexpr E4m3MaxVal{448.f};

      // Compute the absolute max
      float aMax =
          BlockReduce(temp_storage).Reduce(fabsf(acc), cuda::maximum<>{});

      if (threadIdx.x == 0) {
        if (params.outDqSfsPtr) {
          s_scaleOut = aMax / E4m3MaxVal;
          int const scaleOut_idx =
              tokenIdx + hiddenIdx / 128 * params.numTokens;
          params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
        } else {
          s_scaleOut = 1.0f;
        }
      }
      __syncthreads();
      float const scaleOut = s_scaleOut;
      __syncthreads();
      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] =
          (Type)(acc / scaleOut);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void run(Data const &data, void *stream) {
  if (data.mUseDeepSeekFp8) {
    int const numThreads = 128;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535
    // specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH_TOPK_EXPW(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0,
                     stream);
  } else {
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535
    // specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);

    if (numBlocksX * numBlocksY < 1184) {
      // The number 1184 comes from 148 * 8, where 148 is the number of SMs
      // (Streaming Multiprocessors) in the Blackwell architecture, and the
      // value 8 means that each Streaming Multiprocessor (SM) can hold up to 8
      // blocks for this kernel. This limitation is intended to ensure that when
      // the number of waves is greater than 1, we choose to use the kernel with
      // vectorized loading.
      dim3 numBlocks(numBlocksX, numBlocksY);
      LAUNCH_TOPK_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
    } else {
      FLASHINFER_CHECK(data.topK <= MaxTopK,
                       "Finalize kernel with vectorized loading is not "
                       "supported for this TopK value: %d",
                       data.topK);
      LAUNCH_TOPK_EXPW(data, finalizeKernelVecLoad,
                       /*numBlocks=*/data.numTokens,
                       /*numThreads=*/FINALIZE_THREADS_PER_BLOCK, 0, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
