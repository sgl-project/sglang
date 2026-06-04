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

#pragma once

#include <string>

#include "DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
// #include "flashinfer/trtllm/common/cudaDriverWrapper.h"
#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/common/cudaUtils.h"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {

namespace MoE {
class Runner;
}  // namespace MoE

namespace Routing {

// The type of method in top-K routing, for use in torch custom op
// Please keep this in sync with the counterpart defined in
// flashinfer/fused_moe/core.py
enum class RoutingMethodType : int64_t {
  // Default: Softmax -> TopK
  Default = 0,
  // Renormalize: TopK -> Softmax
  Renormalize = 1,
  // DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the
  // Top4 groups
  DeepSeekV3 = 2,
  // Llama4: Top1 -> Sigmoid
  Llama4 = 3,
  // RenormalizeNaive: Softmax -> TopK -> Renormalize
  RenormalizeNaive = 4,
  // TopK only (no softmax)
  TopK = 5,
  // SigmoidRenorm: Sigmoid -> TopK -> Renormalize (divide by sum of top-K weights)
  SigmoidRenorm = 6,
  // MiniMax2: Sigmoid + Bias -> TopK -> ScaledSumNormalize (routeScale=1.0, epsilon=1e-20)
  MiniMax2 = 7,
  // Sigmoid: Sigmoid -> TopK (no renormalization)
  Sigmoid = 8,
  // Unspecified
  Unspecified = 9,
};

inline int32_t maybeGetMinTokenCount(int32_t numPaddedTokens, int32_t hiddenSize,
                                     int32_t dtypeSizeBits) {
  // Pad so total size exceeds 128KiB for performance reasons
  int32_t minNumTokensRequired = common::divUp(128 * 1024 * 8, hiddenSize * dtypeSizeBits);
  return std::max(numPaddedTokens, minNumTokensRequired);
}

inline std::string serializeMoeRoutingMethodType(RoutingMethodType routingMethodType) {
  switch (routingMethodType) {
    case RoutingMethodType::Default:
      return "Default";
    case RoutingMethodType::Renormalize:
      return "Renormalize";
    case RoutingMethodType::DeepSeekV3:
      return "DeepSeekV3";
    case RoutingMethodType::Llama4:
      return "Llama4";
    case RoutingMethodType::RenormalizeNaive:
      return "RenormalizeNaive";
    case RoutingMethodType::TopK:
      return "TopK";
    case RoutingMethodType::SigmoidRenorm:
      return "SigmoidRenorm";
    case RoutingMethodType::MiniMax2:
      return "MiniMax2";
    case RoutingMethodType::Sigmoid:
      return "Sigmoid";
    default:
      return "InvalidRountingMethod";  // TODO throw error
  };
}

inline int32_t getMaxNumCtasInBatchDim(int32_t numTokens, int32_t topK, int32_t numExperts,
                                       int32_t tileTokensDim) {
  // For MoE, mNumTokens != 0 and the number of CTAs is known only at runtime.
  // We launch maximally possible number of CTAs and use ptrNumNonExitingCtas to determine
  // the actual number of CTAs to run.

  // Initialize number of tokens with the number of expanded tokens after routing.
  int32_t numRemainingTokens = numTokens * topK;
  int32_t maxNumCtasInBatchDim = 0;
  // First, distribute one token each expert until token depletion to maximize CTA tile count.
  int32_t numExpertsFilled = std::min(numExperts, numRemainingTokens);
  maxNumCtasInBatchDim += numExpertsFilled;
  numRemainingTokens -= numExpertsFilled;
  // Next, greedily pour all remaining tokens to one expert to maximize CTA tile count.
  // E.g., at this point tokens over 4 experts are [1, 1, 1, 1], and we have 4 tokens left.
  // If each CTA handles 4 tokens/expert, the greedy strategy is to pour all remaining tokens
  // to any one expert to get to the 5th CTA tile. Otherwise, we can only get 4 tiles in total.
  //
  // Another way to reason about this is to pour the remaining tokens into buckets of some fixed
  // capacity. These buckets, if full, can then be attributed to any expert; it does not have to
  // belong to the same expert every time.
  if (numRemainingTokens > 0) {
    // For every tileTokenDim tokens, we add an extra CTA tile in the token dimension.
    // The number of CTA tiles is given by divDown(numRemainingTokens, tokenTileDim).
    maxNumCtasInBatchDim += (numRemainingTokens / tileTokensDim);
  }
  return maxNumCtasInBatchDim;
}

inline int32_t getMaxPermutedPaddedCount(int32_t numTokens, int32_t expertsPerToken,
                                         int32_t numExperts, int32_t padding) {
  int32_t maxCtas = getMaxNumCtasInBatchDim(numTokens, expertsPerToken, numExperts, padding);
  return maxCtas * padding;
}

class Runner {
 public:
  explicit Runner();

  explicit Runner(int32_t tileTokensDim);

  void run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts,
           int32_t topK, int32_t nGroups, int32_t topkGroups, int32_t localExpertOffset,
           int32_t localNumExperts, float routedScalingFactor, int32_t* routingExpertIndexes,
           int32_t* expertCountHistogram, int32_t* permutedIdxSize,
           int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx,
           int32_t* permutedIdxToTokenIdx, void* expertWeights, int32_t* numTokensPerExpert,
           int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas,
           batchedGemm::trtllm::gen::Dtype dtypeElt, batchedGemm::trtllm::gen::Dtype dtypeBias,
           bool useRoutingScalesOnInput, bool useDeepSeekFp8, RoutingMethodType routingMethodType,
           cudaStream_t stream, batchedGemm::trtllm::gen::Dtype dtypeLogits,
           bool normTopkProb = true, int16_t* routing_replay_out = nullptr);

 private:
  friend class MoE::Runner;
  int32_t mTileTokensDim{8};
};
}  // namespace Routing

namespace MoE {
// The type of activation function
// Please keep this in sync with the counterpart defined in flashinfer/flashinfer/fused_moe/core.py
enum class ActivationType : int64_t {
  Gelu = 0,
  Relu = 1,
  Silu = 2,
  Swiglu = 3,
  Geglu = 4,
  SwigluBias = 5,
  Relu2 = 6,
  Identity = 7,
  InvalidType = 8,  // Must be last
};

inline std::string serializeActivationType(ActivationType activationType) {
  switch (activationType) {
    case ActivationType::Gelu:
      return "Gelu";
    case ActivationType::Relu:
      return "Relu";
    case ActivationType::Silu:
      return "Silu";
    case ActivationType::Swiglu:
      return "Swiglu";
    case ActivationType::Geglu:
      return "Geglu";
    case ActivationType::SwigluBias:
      return "SwigluBias";
    case ActivationType::Relu2:
      return "Relu2";
    case ActivationType::Identity:
      return "Identity";
    default:
      return "InvalidActivationType";  // TODO throw error
  };
}

inline bool isGatedActivation(ActivationType activationType) {
  return activationType == ActivationType::Swiglu || activationType == ActivationType::Geglu ||
         activationType == ActivationType::SwigluBias;
}

}  // namespace MoE

namespace PermuteGemm1 {
class Runner {
 public:
  explicit Runner(batchedGemm::trtllm::gen::Dtype dtypeAct,
                  batchedGemm::trtllm::gen::Dtype dtypeWeights,
                  batchedGemm::trtllm::gen::Dtype dtypeOutput, bool useDeepSeekFp8,
                  int tileTokensDim, MoE::ActivationType activationType, bool useShuffledMatrix,
                  batchedGemm::gemm::MatrixLayout weight_layout, bool usePerTokenScaling,
                  bool usePerChannelScaling, bool forceUnfusedAct = false);

  size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                 int32_t numExperts, int32_t numTokens, int32_t configIndex) const;

  [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize, int32_t numExperts,
                                                   int32_t numTokens) const;

  [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                        int32_t intermediateSize, int32_t numExperts,
                                        int32_t numTokens) const;

  [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

  void run(void* hiddenState, void* hiddenStateScale, void* weight, void* weightScale,
           void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
           float* outputScalesGateScalar, float* ptrBias, float* ptrGatedActAlpha,
           float* ptrGatedActBeta, float* ptrClampLimit, void* output, void* outputScale,
           int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
           int32_t numTokens, int32_t* permutedIdxToTokenIdx, int32_t* ptrNumNonExitingCtas,
           int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
           int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace, bool useRoutingScalesOnInput,
           int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl);

 private:
  friend class MoE::Runner;
  batchedGemm::trtllm::gen::Dtype mDtypeAct;
  batchedGemm::trtllm::gen::Dtype mDtypeWeights;
  batchedGemm::trtllm::gen::Dtype mDtypeOutput;
  int32_t mTileTokensDim;
  tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner mRunner;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType mActType;
};
}  // namespace PermuteGemm1

namespace Gemm2 {
class Runner {
 public:
  explicit Runner(batchedGemm::trtllm::gen::Dtype dtypeAct,
                  batchedGemm::trtllm::gen::Dtype dtypeWeights,
                  batchedGemm::trtllm::gen::Dtype outputDtype, bool useDeepSeekFp8,
                  int tileTokensDim, bool useShuffledMatrix,
                  batchedGemm::gemm::MatrixLayout weight_layout, bool usePerTokenScaling,
                  bool usePerChannelScaling);

  size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                 int32_t numExperts, int32_t numTokens, int32_t configIndex) const;

  [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize, int32_t numExperts,
                                                   int32_t numTokens) const;

  [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                        int32_t intermediateSize, int32_t numExperts,
                                        int32_t numTokens) const;

  [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

  void run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weight,
           void* weightScale, void* perTokenScales, void* perChannelScales,
           float* outputScalesScalar, float* ptrBias, void* output, void* outputScale, int32_t topK,
           int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens,
           int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
           int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit, void* bmm2Workspace,
           int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl);

 private:
  friend class MoE::Runner;
  batchedGemm::trtllm::gen::Dtype mDtypeAct;
  batchedGemm::trtllm::gen::Dtype mDtypeWeights;
  batchedGemm::trtllm::gen::Dtype mDtypeOut;
  int32_t mTileTokensDim;
  tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner mRunner;
};
}  // namespace Gemm2

namespace MoE {
namespace btg = batchedGemm::trtllm::gen;

struct MoERunnerArgs {
  void* routing_logits = nullptr;  // [num_tokens, num_experts] in float, generated after
                                   // gemm(hidden_state, routing_weights)
  void* routing_bias = nullptr;    // [num_experts] in bfloat16 for now = mDtypeExpW
  void* hidden_states = nullptr;   // [num_tokens, hidden_size] in fp8 = mDtypeElt
  // [hidden_size/128, num_tokens] in float for e4m3 DS recipe
  // and [num_tokens, hidden_size/16] in float for e2m1
  void* hidden_states_scale = nullptr;

  // Gemm input:
  void* gemm1_weights = nullptr;
  void* gemm1_weights_scale = nullptr;
  void* gemm2_weights = nullptr;
  void* gemm2_weights_scale = nullptr;

  float* gemm1_bias = nullptr;
  float* gemm1_alpha = nullptr;
  float* gemm1_beta = nullptr;
  float* gemm1_clamp_limit = nullptr;
  float* gemm2_bias = nullptr;

  ActivationType activation_type = ActivationType::Swiglu;

  int32_t num_tokens{0};
  int32_t num_experts{0};
  // Hidden dimension input of MoE block. It might be padded.
  int32_t hidden_size{0};
  // Hidden dimension output of MoE block. It is not padded.
  // If not provided it is the same as hidden_size.
  std::optional<int32_t> hidden_size_output;
  // TODO: only compiled routing kernel supports top_k = 8
  int32_t top_k{0};
  int32_t n_group{0};
  // TODO: only compiled routing kernel supports topk_group = 4
  int32_t topk_group{0};
  float routed_scaling_factor{0.0f};
  int32_t intermediate_size{0};
  int32_t local_expert_offset{0};
  int32_t local_num_experts{0};
  // TODO: support other types
  btg::Dtype mDtypeElt{btg::Dtype::Void};
  btg::Dtype mDtypeExpW{btg::Dtype::Bfloat16};
  btg::Dtype mDtypeOut{btg::Dtype::Bfloat16};

  // Apply routing scale factors to input activations
  bool mUseRoutingScalesOnInput{false};
  bool mUseDeepSeekFp8{false};
  float* output1_scales_scalar = nullptr;
  float* output1_scales_gate_scalar = nullptr;
  float* output2_scales_scalar = nullptr;

  // Optional LoRA bridge buffers used by the copied SGLang TRTLLM FP8 path.
  // gate_up_lora_delta: [num_tokens * top_k, 2 * intermediate_size], bf16,
  // in FlashInfer gate/up order (up first, gate second).
  // activation_lora_input: [num_tokens * top_k, intermediate_size], bf16,
  // populated with the post-activation intermediate for down-proj LoRA.
  void* gate_up_lora_delta = nullptr;
  void* activation_lora_input = nullptr;

  // Optional CUDA event (cudaEvent_t) recorded on the LoRA side stream. When set, the
  // runner waits on it right before the activation kernel (which consumes
  // gate_up_lora_delta), so permute+GEMM1 overlap the side-stream LoRA shrink/expand
  // instead of joining before the whole MoE op. nullptr = no wait (serial behavior).
  void* lora_ready_event = nullptr;

  // Output:
  void* output = nullptr;
  float* output_scale = nullptr;

  // finalize
  bool do_finalize{true};
};

struct MoEWorkspace {
  // Routing intermediate outputs:
  int32_t* routing_expert_indexes = nullptr;
  int32_t* permuted_idx_size = nullptr;
  int32_t* total_num_padded_tokens = nullptr;  // TODO: duplicate of permuted_idx_size
  int32_t total_max_padded_tokens{0};

  int32_t* expanded_idx_to_permuted_idx = nullptr;
  int32_t* permuted_idx_to_expanded_idx = nullptr;
  int32_t* permuted_idx_to_token_idx = nullptr;

  // consumed by finalize kernel
  void* expert_weights = nullptr;  // [num_tokens, top_k] in bfloat16 = mDtypeExpW
  // consumed by permuteGemm1 kernel
  void* token_scales = nullptr;
  // consumed by Gemm2 kernel
  void* token_scales_fc2 = nullptr;

  int32_t* cta_idx_xy_to_batch_idx = nullptr;
  int32_t* cta_idx_xy_to_mn_limit = nullptr;
  int32_t* num_non_exiting_ctas = nullptr;

  void* hidden_states_scale_linear = nullptr;

  // Permute intermediate outputs:
  void* permuted_hidden_states = nullptr;
  float* permuted_hidden_states_scale = nullptr;

  // Gemm1 intermediate outputs:
  int32_t ProjUpTileN{0};
  void* gemm1_output = nullptr;
  float* gemm1_output_scale = nullptr;

  // Activation intermediate outputs:
  void* activation_output = nullptr;
  float* activation_output_scale = nullptr;
  // Unfused FP4 LoRA: bf16 [max_padded_tokens, intermediate_size] activation output written by the
  // standalone activation kernel (gate_up LoRA added pre-SwiGLU), then NvFP4-quantized for GEMM2.
  void* activated_lora_bf16 = nullptr;

  // Gemm2 intermediate outputs:
  void* gemm2_output = nullptr;
  float* gemm2_output_scale = nullptr;

  // Finalize intermediate outputs (placeholder not used)
  void* finalize_output = nullptr;
  float* finalize_output_scale = nullptr;

  // FC1 workspace:
  void* bmm1_workspace = nullptr;

  // FC2 workspace:
  void* bmm2_workspace = nullptr;
};

// Config indices to be used with Batched GEMM runners
struct MoEConfig {
  int64_t gemm1Config;
  int64_t gemm2Config;
};

class Runner {
 public:
  // FIXME: tileTokensDim is hardcoded for now
  Runner(batchedGemm::trtllm::gen::Dtype dtypeAct, batchedGemm::trtllm::gen::Dtype dtypeWeights,
         bool useDeepSeekFp8, int tileTokensDim = 8,
         ActivationType activationType = ActivationType::Swiglu, bool useShuffledMatrix = false,
         batchedGemm::gemm::MatrixLayout weight_layout = batchedGemm::gemm::MatrixLayout::MajorK,
         bool usePerTokenScalingGemm1 = false, bool usePerTokenScalingGemm2 = false,
         bool usePerChannelScalingGemm1 = false, bool usePerChannelScalingGemm2 = false,
         bool unfuseActForLora = false);
  Runner(batchedGemm::trtllm::gen::Dtype dtypeElt, bool useDeepSeekFp8, int tileTokensDim = 8,
         bool useShuffledMatrix = false,
         batchedGemm::gemm::MatrixLayout weight_layout = batchedGemm::gemm::MatrixLayout::MajorK,
         bool usePerTokenScalingGemm1 = false, bool usePerTokenScalingGemm2 = false,
         bool usePerChannelScalingGemm1 = false, bool usePerChannelScalingGemm2 = false);

  void run(MoERunnerArgs const& args, MoEWorkspace const& workspace, int device,
           cudaStream_t stream, int64_t configIndex, bool enable_pdl);

  [[nodiscard]] std::tuple<int32_t, int32_t> getWorkspaceSizeInBytes(MoERunnerArgs const& args,
                                                                     int64_t configIndex) const;

  [[nodiscard]] std::vector<int64_t> getValidConfigIndices(int32_t topK, int32_t hiddenSize,
                                                           int32_t intermediateSize,
                                                           int32_t numLocalExperts,
                                                           int32_t numTokens) const;

  [[nodiscard]] int64_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize,
                                                   int32_t numLocalExperts,
                                                   int32_t numTokens) const;

 private:
  void setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
                  moe::dev::convertsf::Data& convertSfData,
                  moe::dev::activation::Data& activationData,
                  moe::dev::finalize::Data& finalizeData);

 private:
  bool mUsePerTokenScalingGemm1;
  bool mUsePerTokenScalingGemm2;
  bool mUsePerChannelScalingGemm1;
  bool mUsePerChannelScalingGemm2;
  // When true (FP4 LoRA path), GEMM1 emits the raw gate_up projection (fusedAct=false) so the
  // standalone activation kernel can inject the gate_up LoRA delta pre-SwiGLU and capture the
  // post-activation input for the down LoRA — mirroring the DeepSeek-FP8 unfused activation.
  bool mUnfuseActForLora;
  PermuteGemm1::Runner mPermuteGemm1;
  Gemm2::Runner mGemm2;

  // This will be the cartesian product of the passing configs for gemm1 and gemm2
  // This allows us to autotune the MoE as one operation instead of tuning gemm1 and gemm2
  // separately
  std::vector<MoEConfig> mPassingConfigs;
};
}  // namespace MoE

}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
