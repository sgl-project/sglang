/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <iostream>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tensorrt_llm/kernels/quantization.h"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {

namespace btg = batchedGemm::trtllm::gen;

namespace Routing {
namespace {
inline int32_t computeLog2(int32_t val, std::string const& name = "") {
  int32_t n = val;
  int32_t out = 0;
  while (n >>= 1) {
    ++out;
  }
  if ((1 << out) != val) {
    out = -1;
  }
  return out;
}
}  // namespace

Runner::Runner() {}

Runner::Runner(int32_t tileTokensDim) : mTileTokensDim(tileTokensDim) {}

void Runner::run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts,
                 int32_t topK, int32_t nGroup, int32_t topkGroup, int32_t localExpertOffset,
                 int32_t localNumExperts, float routedScalingFactor, int32_t* routingExpertIndexes,
                 int32_t* expertCountHistogram, int32_t* permutedIdxSize,
                 int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx,
                 int32_t* permutedIdxToTokenIdx, void* expertWeights, int32_t* numTokensPerExpert,
                 int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit,
                 int32_t* numNonExitingCtas, btg::Dtype dtypeElt, btg::Dtype dtypeBias,
                 bool useRoutingScalesOnInput, bool useDeepSeekFp8,
                 RoutingMethodType routingMethodType, cudaStream_t stream, btg::Dtype dtypeLogits,
                 bool normTopkProb, int16_t* routing_replay_out) {
  if (routingMethodType == RoutingMethodType::DeepSeekV3 && nGroup <= 1) {
    // DeepSeek no-groups case: use routingCustom with SigmoidBias preprocess
    // and ScaledSumNormalize postprocess. This is more efficient than the full DeepSeek
    // kernel because it uses the warp-level routingTopKExperts flow.
    moe::dev::routing::routingCustom::Data routingData;

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;
    routingData.mUsePdl = true;
    routingData.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    routingData.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    routingData.mPtrRoutingBias = routingBias;
    routingData.mDtypeBias = dtypeBias;
    routingData.mRouteScale = routedScalingFactor;

    routingData.mPtrScores = routingLogits;
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    moe::dev::routing::routingCustom::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::MiniMax2) {
    // MiniMaxM2: sigmoid(logit) + bias → topK → renormalize un-biased sigmoid scores.
    // Similar to DeepSeek no-groups but with routeScale = 1.0 and epsilon = 1e-20
    // to match the Python reference: weight / (sum + 1e-20).
    moe::dev::routing::routingCustom::Data routingData;

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;
    routingData.mUsePdl = true;
    routingData.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    routingData.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    routingData.mPtrRoutingBias = routingBias;
    routingData.mDtypeBias = dtypeBias;
    routingData.mRouteScale = 1.0f;
    routingData.mSumEpsilon = 1e-20f;

    routingData.mPtrScores = routingLogits;
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    moe::dev::routing::routingCustom::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::DeepSeekV3) {
    FLASHINFER_CHECK(topK <= 22, "For DeepSeek routing method, must have topK <= 22");
    FLASHINFER_CHECK(topkGroup <= 4, "For DeepSeek routing method, must have topkGroup <= 4");
    moe::dev::routing::routingDeepSeek::Data routingData;
    routingData.mDtypeOutput =
        btg::Dtype::Bfloat16;               // for DeepSeek, the expW is currently always bfloat16
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mDtypeBias = dtypeBias;     // for DeepSeek, the bias can be bfloat16 or fp32
    routingData.mUsePdl = true;

    // output:
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    // input:
    routingData.mPtrRoutingBias = routingBias;
    routingData.mPtrScores = routingLogits;  // type-erased; InputT selected by forceFloatInput
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mNumExpertGroups = nGroup;
    routingData.mNumLimitedGroups = topkGroup;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mRouteScale = routedScalingFactor;
    routingData.mUseRoutingSoftmax = false;
    routingData.mPtrRoutingReplayOut = routing_replay_out;
    moe::dev::routing::routingDeepSeek::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::Llama4) {
    FLASHINFER_CHECK(topK == 1, "For Llama routing method, must have topK == 1");
    if (nGroup > 0 || topkGroup > 0) {
      FLASHINFER_WARN("For Llama routing method, nGroup/topkGroup is ignored, got ", nGroup, "/",
                      topkGroup);
    }
    moe::dev::routing::routingLlama4::Data routingData;
    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mUsePdl = true;

    // output:
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    // input:
    routingData.mPtrScores = routingLogits;
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;
    moe::dev::routing::routingLlama4::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::Default        /* Softmax -> TopK */
             || routingMethodType == RoutingMethodType::Renormalize /* TopK -> Softmax */
             || routingMethodType ==
                    RoutingMethodType::RenormalizeNaive      /* Softmax -> TopK -> Renormalize */
             || routingMethodType == RoutingMethodType::TopK /* TopK only (no softmax) */
             || routingMethodType ==
                    RoutingMethodType::SigmoidRenorm /* Sigmoid -> TopK -> Renormalize */
             || routingMethodType == RoutingMethodType::Sigmoid /* Sigmoid -> TopK */) {
    using namespace moe::dev::routing;
    routingCustom::Data routingData;

    //
    // Config
    //

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mUsePdl = true;

    // Map routing method types to policy-based routing:
    // Note: RenormalizeNaive (Softmax → TopK → SumNormalize) is mathematically equivalent
    // to Renormalize (TopK → Softmax), because taking softmax over all experts, selecting
    // top-K, and dividing by their sum produces the same result as applying softmax only
    // over the top-K values. We therefore use the same Renormalize implementation for both.
    if (routingMethodType == RoutingMethodType::Default) {
      // Softmax -> TopK (softmax on all scores, then select top-K)
      routingData.mPreprocessType = RoutingPreprocessType::Softmax;
      routingData.mPostprocessType = RoutingPostprocessType::None;
    } else if (routingMethodType == RoutingMethodType::SigmoidRenorm) {
      // Sigmoid -> TopK -> SumNormalize (renormalize)
      routingData.mPreprocessType = RoutingPreprocessType::Sigmoid;
      routingData.mPostprocessType = RoutingPostprocessType::SumNormalize;
      routingData.mNormTopkProb = normTopkProb;
    } else if (routingMethodType == RoutingMethodType::Sigmoid) {
      // Sigmoid -> TopK (no renormalization)
      routingData.mPreprocessType = RoutingPreprocessType::Sigmoid;
      routingData.mPostprocessType = RoutingPostprocessType::SumNormalize;
      routingData.mNormTopkProb = false;
    } else if (routingMethodType == RoutingMethodType::Renormalize ||
               routingMethodType == RoutingMethodType::RenormalizeNaive) {
      // TopK -> Softmax (also used for RenormalizeNaive, see comment above)
      routingData.mPreprocessType = RoutingPreprocessType::None;
      routingData.mPostprocessType = RoutingPostprocessType::Softmax;
    } else {
      // TopK only (no softmax or renormalize)
      routingData.mPreprocessType = RoutingPreprocessType::None;
      routingData.mPostprocessType = RoutingPostprocessType::None;
    }

    routingData.mPtrScores = routingLogits;

    //
    // Outputs
    //
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    //
    // Grouped Gemm Launch Config Buffers
    //
    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    //
    // Inputs
    //
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    routingCustom::run(routingData, stream);
  } else {
    FLASHINFER_CHECK(false, "Unimplemented routing method ",
                     serializeMoeRoutingMethodType(routingMethodType), " of enum ",
                     (int)routingMethodType);
  }
}
}  // namespace Routing

namespace PermuteGemm1 {

using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::isGatedActivation;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::serializeActivationType;

static inline ActType activationTypeToGatedActType(ActivationType actType) {
  switch (actType) {
    case ActivationType::Swiglu:
      return ActType::SwiGlu;
    case ActivationType::Geglu:
      return ActType::GeGlu;
    default:
      FLASHINFER_CHECK(false, "Unsupported gated activation type ",
                       serializeActivationType(actType), " of enum ",
                       static_cast<int64_t>(actType));
  }
  return ActType::SwiGlu;
}

static inline EltwiseActType activationTypeToEltwiseActType(ActivationType actType) {
  switch (actType) {
    case ActivationType::Relu2:
      return EltwiseActType::Relu2;
    case ActivationType::Identity:
      return EltwiseActType::None;
    default:
      FLASHINFER_CHECK(false, "Unsupported eltwise activation type ",
                       serializeActivationType(actType), " of enum ",
                       static_cast<int64_t>(actType));
  }
  return EltwiseActType::None;
}

tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput, int32_t tileTokensDim,
    bool useDeepSeekFp8, ActivationType activationType, bool useShuffledMatrix,
    batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
    bool usePerChannelScaling) {
  int64_t actTypeInt = static_cast<int64_t>(activationType);
  FLASHINFER_CHECK(
      0 <= actTypeInt && actTypeInt < static_cast<int64_t>(ActivationType::InvalidType),
      "Unknown activation type", serializeActivationType(activationType), "of enum", actTypeInt);
  bool isGatedAct = isGatedActivation(activationType);
  if (isGatedAct) {
    ActType actType = activationTypeToGatedActType(activationType);
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
        // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
        .dtypeA = dtypeWeights,
        .dtypeB = dtypeAct,
        .dtypeC = dtypeOutput,
        .actType = actType,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = !useDeepSeekFp8,
        .routeAct = true,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = useDeepSeekFp8 ? 64 : 128,
        .useShuffledMatrix = useShuffledMatrix,
        .weightLayout = weightLayout,
        .usePerTokenScaling = usePerTokenScaling,
        .usePerChannelScaling = usePerChannelScaling,
    };
    return options;
  } else {
    EltwiseActType actType = activationTypeToEltwiseActType(activationType);
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
        // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
        .dtypeA = dtypeWeights,
        .dtypeB = dtypeAct,
        .dtypeC = dtypeOutput,
        .eltwiseActType = actType,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = false,
        .routeAct = true,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = 128,
        .useShuffledMatrix = useShuffledMatrix,
        .weightLayout = weightLayout,
        .usePerTokenScaling = usePerTokenScaling,
        .usePerChannelScaling = usePerChannelScaling};
    return options;
  }
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
               bool useDeepSeekFp8, int tileTokensDim, ActivationType activationType,
               bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
               bool usePerTokenScaling, bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOutput(dtypeOutput),
      mTileTokensDim(tileTokensDim),
      mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(getOptions(
          mDtypeAct, mDtypeWeights, mDtypeOutput, mTileTokensDim, useDeepSeekFp8, activationType,
          useShuffledMatrix, weightLayout, usePerTokenScaling, usePerChannelScaling))),
      mActType(activationType) {}

void Runner::run(void* hiddenState, void* hiddenStateScale, void* weights, void* weightsScale,
                 void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
                 float* outputScalesGateScalar, float* ptrBias, float* ptrAlpha, float* ptrBeta,
                 float* ptrClampLimit, void* output, void* outputScale, int32_t topK,
                 int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
                 int32_t numTokens, int32_t* permutedIdxToTokenIdx, int32_t* ptrNumNonExitingCtas,
                 int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
                 int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace, bool useRoutingScalesOnInput,
                 int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  mRunner.run(numTokens, intermediateSizeFactor * intermediateSize, hiddenSize, {}, numTokens,
              numExperts, maxNumCtasInBatchDim, hiddenState, hiddenStateScale, weights,
              weightsScale, perTokenScales, perChannelScales, outputScalesScalar,
              outputScalesGateScalar, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, output,
              outputScale, permutedIdxToTokenIdx, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
              ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, bmm1Workspace, stream, device,
              configIndex, enable_pdl);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                       int32_t numExperts, int32_t numTokens,
                                       int32_t configIndex) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  return mRunner.getWorkspaceSizeInBytes(numTokens, intermediateSizeFactor * intermediateSize,
                                         hiddenSize, {}, numTokens, numExperts,
                                         maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numExperts,
                                           int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  return mRunner.getDefaultValidConfigIndex(numTokens, intermediateSizeFactor * intermediateSize,
                                            hiddenSize, {}, numTokens, numExperts,
                                            maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                int32_t intermediateSize, int32_t numExperts,
                                int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  auto const isValid =
      mRunner.isValidConfigIndex(configIndex, numTokens, intermediateSizeFactor * intermediateSize,
                                 hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);

  return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const {
  return mRunner.getPassingConfigIndices();
}
}  // namespace PermuteGemm1

namespace Gemm2 {
tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, int32_t tileTokensDim,
    bool useDeepSeekFp8, bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
    bool usePerTokenScaling, bool usePerChannelScaling) {
  tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
      // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
      .dtypeA = dtypeWeights,
      .dtypeB = dtypeAct,
      .dtypeC = dtypeOut,
      .eltwiseActType = EltwiseActType::None,
      .deepSeekFp8 = useDeepSeekFp8,
      .fusedAct = false,
      .routeAct = false,
      .staticBatch = false,
      .transposeMmaOutput = true,
      .tileSize = tileTokensDim,
      .epilogueTileM = useDeepSeekFp8 ? 64 : 128,
      .useShuffledMatrix = useShuffledMatrix,
      .weightLayout = weightLayout,
      .usePerTokenScaling = usePerTokenScaling,
      .usePerChannelScaling = usePerChannelScaling};
  return options;
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut,
               bool useDeepSeekFp8, int tileTokensDim, bool useShuffledMatrix,
               batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
               bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOut(dtypeOut),
      mTileTokensDim(tileTokensDim),
      mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(dtypeAct, dtypeWeights, dtypeOut, tileTokensDim, useDeepSeekFp8,
                     useShuffledMatrix, weightLayout, usePerTokenScaling, usePerChannelScaling))) {}

void Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weights,
                 void* weightsScale, void* perTokenScales, void* perChannelScales,
                 float* outputScalesScalar, float* ptrBias, void* output, void* outputScale,
                 int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
                 int32_t numTokens, int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
                 int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit, void* bmm2Workspace,
                 int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  mRunner.run(
      numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
      permutedHiddenState, permutedHiddenStateScale, weights, weightsScale,
      /* perTokensSfA */ perTokenScales,
      /* perTokensSfB */ perChannelScales, outputScalesScalar, /* outputScalesGateScalar */ nullptr,
      ptrBias,
      /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* clampLimit */ nullptr, output, outputScale,
      /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
      ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, bmm2Workspace, stream, device, configIndex,
      enable_pdl);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                       int32_t numExperts, int32_t numTokens,
                                       int32_t configIndex) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  return mRunner.getWorkspaceSizeInBytes(numTokens, hiddenSize, intermediateSize, {}, numTokens,
                                         numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numExperts,
                                           int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  return mRunner.getDefaultValidConfigIndex(numTokens, hiddenSize, intermediateSize, {}, numTokens,
                                            numExperts, maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                int32_t intermediateSize, int32_t numExperts,
                                int32_t numTokens) const {
  auto const maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

  auto const isValid =
      mRunner.isValidConfigIndex(configIndex, numTokens, hiddenSize, intermediateSize, {},
                                 numTokens, numExperts, maxNumCtasInBatchDim);

  return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const {
  return mRunner.getPassingConfigIndices();
}
}  // namespace Gemm2

namespace MoE {
Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, bool useDeepSeekFp8,
               int32_t tileTokensDim, ActivationType activationType, bool useShuffledMatrix,
               batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScalingGemm1,
               bool usePerTokenScalingGemm2, bool usePerChannelScalingGemm1,
               bool usePerChannelScalingGemm2)
    : mUsePerTokenScalingGemm1(usePerTokenScalingGemm1),
      mUsePerTokenScalingGemm2(usePerTokenScalingGemm2),
      mUsePerChannelScalingGemm1(usePerChannelScalingGemm1),
      mUsePerChannelScalingGemm2(usePerChannelScalingGemm2),
      mPermuteGemm1(PermuteGemm1::Runner(
          dtypeAct, dtypeWeights, usePerTokenScalingGemm2 ? btg::Dtype::Bfloat16 : dtypeAct,
          useDeepSeekFp8, tileTokensDim, activationType, useShuffledMatrix, weightLayout,
          usePerTokenScalingGemm1, usePerChannelScalingGemm1)),
      mGemm2(Gemm2::Runner(dtypeAct, dtypeWeights, btg::Dtype::Bfloat16, useDeepSeekFp8,
                           tileTokensDim, useShuffledMatrix, weightLayout, usePerTokenScalingGemm2,
                           usePerChannelScalingGemm2)) {
  auto const& gemm1PassingIndices = mPermuteGemm1.getPassingConfigIndices();
  auto const& gemm2PassingIndices = mGemm2.getPassingConfigIndices();

  auto const totalPassingIndices = gemm1PassingIndices.size() * gemm2PassingIndices.size();
  mPassingConfigs.reserve(totalPassingIndices);

  for (auto const& indexGemm1 : gemm1PassingIndices) {
    for (auto const& indexGemm2 : gemm2PassingIndices) {
      mPassingConfigs.push_back(MoEConfig{indexGemm1, indexGemm2});
    }
  }
  FLASHINFER_CHECK(!mPassingConfigs.empty(),
                   "No compatible configs found for the fp8 block scale MoE runner.");
}

Runner::Runner(btg::Dtype dtypeElt, bool useDeepSeekFp8, int32_t tileTokensDim,
               bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
               bool usePerTokenScalingGemm1, bool usePerTokenScalingGemm2,
               bool usePerChannelScalingGemm1, bool usePerChannelScalingGemm2)
    : Runner(dtypeElt, dtypeElt, useDeepSeekFp8, tileTokensDim, ActivationType::Swiglu,
             useShuffledMatrix, weightLayout, usePerTokenScalingGemm1, usePerTokenScalingGemm2,
             usePerChannelScalingGemm1, usePerChannelScalingGemm2) {}

void Runner::setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
                        moe::dev::convertsf::Data& convertSfData,
                        moe::dev::activation::Data& activationData,
                        moe::dev::finalize::Data& finalizeData) {
  // Setup sf conversion data if needed
  convertSfData.inSfPtr = args.hidden_states_scale;
  convertSfData.outSfPtr = workspace.hidden_states_scale_linear;
  convertSfData.hiddenDimSf = args.hidden_size / 16;
  convertSfData.numTokens = args.num_tokens;
  convertSfData.sfLayoutSrc = btg::SfLayout::R128c4;
  convertSfData.sfLayoutDst = btg::SfLayout::Linear;
  convertSfData.mUsePdl = true;

  // Setup activation data
  activationData.mDtypeElt = args.mDtypeElt;
  activationData.mUsePdl = true;
  activationData.mUseDeepSeekFp8 = true;
  activationData.inPtr = workspace.gemm1_output;
  activationData.outPtr = workspace.activation_output;
  activationData.inDqSfsPtr = workspace.gemm1_output_scale;
  activationData.outDqSfsPtr = workspace.activation_output_scale;
  activationData.gateUpLoraDeltaPtr =
      reinterpret_cast<cutlass::bfloat16_t const*>(args.gate_up_lora_delta);
  activationData.activationLoraInputOutPtr =
      reinterpret_cast<cutlass::bfloat16_t*>(args.activation_lora_input);
  activationData.innerDim =
      args.intermediate_size * (isGatedActivation(args.activation_type) ? 2 : 1);
  activationData.topK = args.top_k;
  activationData.numTokens = args.num_tokens;
  activationData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;

  activationData.totalNumPaddedTokens = workspace.total_num_padded_tokens;

  // Setup finalize data
  if (args.do_finalize) {
    // Setup finalize data
    finalizeData.mDtypeElt = args.mDtypeOut;
    finalizeData.mDtypeExpW = args.mDtypeExpW;
    finalizeData.mUsePdl = true;
    finalizeData.mUseDeepSeekFp8 = false;
    finalizeData.inPtr = workspace.gemm2_output;
    finalizeData.outPtr = args.output;
    finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
    finalizeData.outDqSfsPtr = args.output_scale;
    if (args.mUseRoutingScalesOnInput) {
      finalizeData.expertWeightsPtr = nullptr;
    } else {
      finalizeData.expertWeightsPtr = workspace.expert_weights;
    }
    finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
    finalizeData.numTokens = args.num_tokens;
    finalizeData.numExperts = args.num_experts;
    finalizeData.topK = args.top_k;
    // We want to fuse unpadding into the finalize kernel, so we need to use the output hidden size.
    finalizeData.hiddenDim = args.hidden_size_output.value_or(args.hidden_size);
    finalizeData.hiddenDimPadded = args.hidden_size;
    finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
  }
}

std::tuple<int32_t, int32_t> Runner::getWorkspaceSizeInBytes(MoERunnerArgs const& args,
                                                             int64_t configIndex) const {
  FLASHINFER_CHECK(configIndex >= 0 && configIndex < static_cast<int64_t>(mPassingConfigs.size()),
                   "Invalid MoE config index ", configIndex, ", valid range is [0, ",
                   static_cast<int64_t>(mPassingConfigs.size()) - 1, "].");
  auto const& config = mPassingConfigs[configIndex];

  auto workspace_size_fc1 = static_cast<int32_t>(mPermuteGemm1.getWorkspaceSizeInBytes(
      args.top_k, args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens,
      config.gemm1Config));
  auto workspace_size_fc2 = static_cast<int32_t>(
      mGemm2.getWorkspaceSizeInBytes(args.top_k, args.hidden_size, args.intermediate_size,
                                     args.local_num_experts, args.num_tokens, config.gemm2Config));
  return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

std::vector<int64_t> Runner::getValidConfigIndices(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize,
                                                   int32_t numLocalExperts,
                                                   int32_t numTokens) const {
  std::vector<int64_t> validIndices;

  for (int i = 0; i < mPassingConfigs.size(); ++i) {
    auto const& config = mPassingConfigs[i];

    if (mPermuteGemm1.isValidConfigIndex(config.gemm1Config, topK, hiddenSize, intermediateSize,
                                         numLocalExperts, numTokens) &&
        mGemm2.isValidConfigIndex(config.gemm2Config, topK, hiddenSize, intermediateSize,
                                  numLocalExperts, numTokens)) {
      validIndices.push_back(i);
    }
  }

  return validIndices;
}

int64_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numLocalExperts,
                                           int32_t numTokens) const {
  int32_t indexGemm1 = mPermuteGemm1.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize,
                                                                numLocalExperts, numTokens);
  int32_t indexGemm2 = mGemm2.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize,
                                                         numLocalExperts, numTokens);

  auto it = std::find_if(mPassingConfigs.begin(), mPassingConfigs.end(),
                         [indexGemm1, indexGemm2](MoEConfig cfg) {
                           return (cfg.gemm1Config == indexGemm1 && cfg.gemm2Config == indexGemm2);
                         });
  FLASHINFER_CHECK(it != mPassingConfigs.end(),
                   "No compatible configs found for the block scale MoE runner.");
  return std::distance(mPassingConfigs.begin(), it);
}

void Runner::run(MoERunnerArgs const& args, MoEWorkspace const& workspace, int device,
                 cudaStream_t stream, int64_t configIndex, bool enable_pdl) {
  FLASHINFER_CHECK(configIndex >= 0 && configIndex < static_cast<int64_t>(mPassingConfigs.size()),
                   "Invalid MoE config index ", configIndex, ", valid range is [0, ",
                   static_cast<int64_t>(mPassingConfigs.size()) - 1, "].");
  FLASHINFER_CHECK(!mUsePerChannelScalingGemm1 && !mUsePerChannelScalingGemm2,
                   "Per-channel scaling is currently not supported.");
  // Setup all operation data
  moe::dev::activation::Data activationData;
  moe::dev::finalize::Data finalizeData;
  moe::dev::convertsf::Data convertSfData;
  sync_check_cuda_error(stream);
  setOpsData(args, workspace, convertSfData, activationData, finalizeData);

  void* hidden_states_scale_linear{args.hidden_states_scale};

  auto const& config = mPassingConfigs[configIndex];

  mPermuteGemm1.run(
      args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
      workspace.token_scales, /*perChannelScales*/ nullptr, args.output1_scales_scalar,
      args.output1_scales_gate_scalar, args.gemm1_bias, args.gemm1_alpha, args.gemm1_beta,
      args.gemm1_clamp_limit, workspace.gemm1_output, workspace.gemm1_output_scale, args.top_k,
      args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens,
      workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas,
      workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx,
      workspace.cta_idx_xy_to_mn_limit, workspace.bmm1_workspace, args.mUseRoutingScalesOnInput,
      device, stream, config.gemm1Config, enable_pdl);

  // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
  void* gemm2_input = workspace.gemm1_output;
  void* gemm2_input_scale = workspace.gemm1_output_scale;
  // We do activation only for DeepSeek FP8, as cubins do not have fused activation.
  if (args.mDtypeElt == btg::Dtype::E4m3 && args.mUseDeepSeekFp8) {
    // GEMM1-LoRA overlap: activation consumes gate_up_lora_delta, so wait on the LoRA
    // side-stream event HERE (not before the op) -- permute+GEMM1 above overlapped the
    // side-stream LoRA shrink/expand. nullptr event = no wait (serial / non-overlap path).
    if (args.lora_ready_event != nullptr) {
      cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(args.lora_ready_event), 0);
    }
    // Run activation
    moe::dev::activation::run(activationData, stream);
    gemm2_input = workspace.activation_output;
    gemm2_input_scale = workspace.activation_output_scale;
  } else if (mUsePerTokenScalingGemm2) {
    // TODO(siyuan): currently only support per-token nvfp4 quantization
    FLASHINFER_CHECK(
        mPermuteGemm1.mDtypeOutput == btg::Dtype::Bfloat16,
        "When using explicit quantization, PermuteGemm1 output dtype must be Bfloat16.");
    FLASHINFER_CHECK(mGemm2.mDtypeAct == btg::Dtype::E2m1,
                     "Currently only support NvFP4 when using explicit quantization.");
    FLASHINFER_CHECK(
        workspace.token_scales_fc2 != nullptr,
        "workspace.token_scales_fc2 must be provided When using explicit quantization.");
    // FIXME(siyuan): Detect from the kernel config. Currently only tile size >= 128 will use R128c4
    auto sfLayout = mGemm2.mTileTokensDim >= 128 ? QuantizationSFLayout::SWIZZLED_128x4
                                                 : QuantizationSFLayout::SWIZZLED_8x4;

    // TODO(siyuan): should this value be exposed?
    float globalScaleInv = 1.f / 448.f / 6.f;
    invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
        args.num_tokens * args.top_k, args.intermediate_size,
        reinterpret_cast<__nv_bfloat16 const*>(workspace.gemm1_output), globalScaleInv,
        workspace.expanded_idx_to_permuted_idx,
        reinterpret_cast<uint8_t*>(workspace.activation_output),
        reinterpret_cast<uint8_t*>(workspace.activation_output_scale),
        reinterpret_cast<float*>(workspace.token_scales_fc2), sfLayout, stream);

    gemm2_input = workspace.activation_output;
    gemm2_input_scale = workspace.activation_output_scale;
  }

  // Run gemm2
  mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale,
             workspace.token_scales_fc2, /*perChannelScales*/ nullptr, args.output2_scales_scalar,
             args.gemm2_bias, workspace.gemm2_output, workspace.gemm2_output_scale, args.top_k,
             args.hidden_size, args.intermediate_size, args.local_num_experts, args.num_tokens,
             workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
             workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
             workspace.bmm2_workspace, device, stream, config.gemm2Config, enable_pdl);

  // Run finalize
  if (args.do_finalize) {
    // Run finalize
    moe::dev::finalize::run(finalizeData, stream);
    sync_check_cuda_error(stream);
  }
}
}  // namespace MoE

}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
