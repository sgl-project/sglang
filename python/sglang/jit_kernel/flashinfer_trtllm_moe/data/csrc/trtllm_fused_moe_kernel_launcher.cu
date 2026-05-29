/*
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
#include <cuda_runtime.h>
#include <flashinfer/exception.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/GemmGatedActOptions.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "nv_internal/tensorrt_llm/kernels/quantization.h"
#include "nv_internal/tensorrt_llm/thop/utils.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::Routing::RoutingMethodType;
using tvm::ffi::Array;
using tvm::ffi::Optional;

// Validate routing_replay_out tensor properties.
// NOTE: dim0 >= num_tokens is intentionally NOT checked — with CUDA graphs the buffer
// is pre-allocated at maximum batch size and reused across steps with varying num_tokens.
static void validate_routing_replay_out(TensorView const& replay, TensorView const& hidden_states,
                                        int64_t top_k) {
  TVM_FFI_ICHECK(replay.device().device_type == kDLCUDA)
      << "routing_replay_out must be a CUDA tensor";
  TVM_FFI_ICHECK(replay.device().device_id == hidden_states.device().device_id)
      << "routing_replay_out must be on the same device as hidden_states";
  TVM_FFI_ICHECK(replay.ndim() == 2) << "routing_replay_out must be 2D [num_tokens, top_k]";
  TVM_FFI_ICHECK(replay.size(1) == top_k) << "routing_replay_out dim1 must equal top_k";
  TVM_FFI_ICHECK((replay.dtype() == DLDataType{kDLInt, 16, 1}))
      << "routing_replay_out must be int16 dtype";
  TVM_FFI_ICHECK(replay.IsContiguous())
      << "routing_replay_out must be contiguous (packed row-major)";
}

enum class Fp8QuantizationType {
  NoneFp8,
  DeepSeekFp8,
  MxFp8,
  PerTensorFp8,
};

inline std::string fp8QuantizationTypeToString(Fp8QuantizationType quantization_type) {
  switch (quantization_type) {
    default:
    case Fp8QuantizationType::NoneFp8:
      return "NoneFp8";
    case Fp8QuantizationType::DeepSeekFp8:
      return "DeepSeekFp8";
    case Fp8QuantizationType::MxFp8:
      return "MxFp8";
    case Fp8QuantizationType::PerTensorFp8:
      return "PerTensorFp8";
  }
}

inline ActivationType validateAndCastActivationType(int64_t act_type) {
  TVM_FFI_ICHECK(act_type >= 0 && act_type < static_cast<int64_t>(ActivationType::InvalidType))
      << "Invalid activation type: " << act_type;
  return static_cast<ActivationType>(act_type);
}

// Utility function to compute the next power of two
inline int32_t nextPowerOfTwo(float value) {
  int32_t n = static_cast<int32_t>(std::ceil(value));
  if (n <= 1) return 1;

  // If n is already a power of 2, return it
  if ((n & (n - 1)) == 0) return n;

  // Find the next power of 2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}

std::set<int32_t> computeSelectedTileN(std::vector<int32_t> const& supported_tile_nums,
                                       int64_t const num_tokens, int64_t const top_k,
                                       int64_t const num_local_experts) {
  TVM_FFI_ICHECK(!supported_tile_nums.empty()) << "supported_tile_nums must not be empty.";
  float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
  // NOTE: This differs from Python AutoTuner bucketing:
  // - AutoTuner maps raw num_tokens with last_positive_power_of_2 (round-down).
  // - Here we map derived avg_tokens_per_expert and use nextPowerOfTwo (round-up).
  // Because they round different quantities in different directions, cache bucket and runtime
  // tile candidates can diverge; launcher-side tactic resolution handles that mismatch.
  // assume supported_tile_nums is sorted
  int32_t tile_tokens_dim = std::clamp(nextPowerOfTwo(avg_tokens_per_expert),
                                       supported_tile_nums.front(), supported_tile_nums.back());
  auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);
  FLASHINFER_CHECK(
      it != supported_tile_nums.end(), "computeSelectedTileN expected exact tile ", tile_tokens_dim,
      " in supported_tile_nums (size=", supported_tile_nums.size(),
      "). Please keep supported_tile_nums as a dense power-of-2 ladder for this launcher.");

  // Candidate tile set centered on the heuristic tile.
  // This function returns nearby candidates (not a single final tile):
  //   center, +1, +2, and -1 neighbors when available.
  // Final tile choice is made later (autotuner-provided tile if valid, otherwise fallback policy).
  std::set<int32_t> selected_tile_nums;
  selected_tile_nums.insert(tile_tokens_dim);
  if (std::next(it) != supported_tile_nums.end()) {
    selected_tile_nums.insert(*std::next(it));
    if (std::next(std::next(it)) != supported_tile_nums.end()) {
      selected_tile_nums.insert(*std::next(std::next(it)));
    }
  }
  if (it != supported_tile_nums.begin()) {
    selected_tile_nums.insert(*std::prev(it));
  }

  return selected_tile_nums;
}

int64_t selectDefaultTileN(std::vector<int32_t> const& supported_tile_nums,
                           int64_t const num_tokens, int64_t const top_k,
                           int64_t const num_local_experts) {
  auto selected = computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
  TVM_FFI_ICHECK(!selected.empty()) << "No selected tile_N candidates for current MoE input.";
  return *selected.begin();
}

// Resolve the (tile_N, config) pair passed from Python side, applying fallback logic
// when tile_N is -1.
std::pair<int64_t, int64_t> resolveMoeTileAndConfig(Array<int64_t> const& config_index,
                                                    std::vector<int32_t> const& supported_tile_nums,
                                                    int64_t const num_tokens, int64_t const top_k,
                                                    int64_t const num_local_experts) {
  // Python side convention: tactic is [tile_N, config]
  TVM_FFI_ICHECK(config_index.size() == 2)
      << "Invalid tactic, expected to be [tile_N, config], but got array of size "
      << config_index.size();
  const int64_t tile_N = config_index[0];
  const int64_t config = config_index[1];

  if (tile_N == -1 || config == -1) {
    // Use fallback tactic
    auto const default_tile_N =
        selectDefaultTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
    return {default_tile_N, -1};
  }

  return {tile_N, config};
}

class FusedMoeLauncher {
 protected:
  Optional<TensorView> routing_logits;
  Optional<TensorView> routing_bias;
  TensorView hidden_states;
  TensorView gemm1_weights;
  Optional<TensorView> output1_scales_scalar;
  Optional<TensorView> output1_scales_gate_scalar;
  TensorView gemm2_weights;
  Optional<TensorView> output2_scales_scalar;
  Optional<TensorView> per_token_scales;
  Tensor per_token_scales_fc2;

  int64_t tile_tokens_dim{};
  int64_t routing_method_type{};
  bool use_shuffled_weight{};
  batchedGemm::gemm::MatrixLayout weight_layout{batchedGemm::gemm::MatrixLayout::MajorK};

  std::tuple<int, int> device_version;
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs> args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  btg::Dtype mDtypeAct{btg::Dtype::Bfloat16};
  btg::Dtype mDtypeWeights{btg::Dtype::Bfloat16};
  btg::Dtype mRoutingBiasDtype{
      btg::Dtype::Bfloat16};  // Dtype for expert weights in routing, based on routing bias
  btg::Dtype mRoutingLogitsDtype{btg::Dtype::Bfloat16};
  bool norm_topk_prob{true};
  ActivationType activation_type{ActivationType::Swiglu};
  btg::Dtype mDtypeScore{btg::Dtype::Bfloat16};

  // Optional routing replay output: [num_tokens, top_k] int16 tensor
  Optional<TensorView> routing_replay_out;

  int64_t intermediate_size_factor{2};

 public:
  // Constructor that initializes all TensorView members
  FusedMoeLauncher(const Optional<TensorView>& routing_logits,
                   const Optional<TensorView>& routing_bias, const TensorView& hidden_states,
                   const TensorView& gemm1_weights,
                   const Optional<TensorView>& output1_scales_scalar,
                   const Optional<TensorView>& output1_scales_gate_scalar,
                   const TensorView& gemm2_weights,
                   const Optional<TensorView>& output2_scales_scalar,
                   const Optional<TensorView>& per_token_scales)
      : routing_logits(routing_logits),
        routing_bias(routing_bias),
        hidden_states(hidden_states),
        gemm1_weights(gemm1_weights),
        output1_scales_scalar(output1_scales_scalar),
        output1_scales_gate_scalar(output1_scales_gate_scalar),
        gemm2_weights(gemm2_weights),
        output2_scales_scalar(output2_scales_scalar),
        per_token_scales(per_token_scales),
        tile_tokens_dim{},
        routing_method_type{},
        use_shuffled_weight{},
        weight_layout{batchedGemm::gemm::MatrixLayout::MajorK},
        mDtypeAct{btg::Dtype::Bfloat16},
        mDtypeWeights{btg::Dtype::Bfloat16},
        activation_type{ActivationType::Swiglu},
        intermediate_size_factor{2} {}

 public:
  void set_routing_replay_out(const Optional<TensorView>& replay_out) {
    routing_replay_out = replay_out;
  }

 protected:
  // Initialize common data necessary for later.
  // May throw exception from TVM_FFI_ICHECK.
  void init_common(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
                   int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
                   int64_t weight_layout, ActivationType activation_type,
                   bool norm_topk_prob = true);

  // Routing logits [num_tokens, num_experts]
  void check_routing_logits() const {
    if (routing_logits.has_value()) {
      // Check shape
      TVM_FFI_ICHECK_EQ(routing_logits.value().ndim(), 2) << "routing_logits must be 2D.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(0), hidden_states.size(0))
          << "routing_logits and hidden_states must have the same number of tokens.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(1), args->num_experts)
          << "routing_logits dim1 must match num_experts.";

      // Check dtype
      TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                     routing_logits.value().dtype() == dl_bfloat16)
          << "routing_logits must be float or bfloat16.";
    }
  }

  // Routing bias [num_experts]
  void check_routing_bias_shape() const {
    if (routing_bias.has_value()) {
      TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), args->num_experts)
          << "routing_bias has incorrect shape.";
    }
  }

  // Hidden states [num_tokens, hidden_size]
  void check_hidden_states_shape() const {
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    TVM_FFI_ICHECK_EQ(hidden_states.size(1), args->intermediate_size)
        << "hidden_states has incorrect shape.";
  }

  // GEMM1 or GEMM2 weights [num_experts, M, K] or [num_experts, K/block_k, M, block_k]
  void check_weights_shape(std::string which_weights) const {
    TensorView weights = (which_weights == "gemm1") ? gemm1_weights : gemm2_weights;
    if (which_weights != "gemm1" && which_weights != "gemm2") {
      TVM_FFI_LOG_AND_THROW(InternalError) << "Internal error: which_weights = " << which_weights;
    }

    int64_t Mn = 0, K = 0;
    if (weight_layout == batchedGemm::gemm::MatrixLayout::MajorK) {
      // MajorK [num_experts, M, K]
      Mn = weights.size(1);
      K = weights.size(2);
    } else if (weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      Mn = weights.size(2);
      int64_t block_k = weights.size(3);
      K = weights.size(1) * block_k;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Unsupported weight_layout: " << (int)weight_layout;
    }
    if (which_weights == "gemm1") {
      // Gated MoE activations (e.g. Swiglu/Geglu) pack gate+up projections in GEMM1,
      // so Mn = 2 * intermediate_size and must be even.
      if (intermediate_size_factor == 2) {
        TVM_FFI_ICHECK_EQ(Mn % 2, 0) << which_weights << " weights Mn dimension must be even.";
      }
      // Non-gated activations (e.g. Relu2) use a single projection in GEMM1,
      // so Mn = intermediate_size. This check covers both gated and non-gated cases.
      TVM_FFI_ICHECK_EQ(args->intermediate_size * intermediate_size_factor, Mn)
          << "intermediate_size has incorrect shape.";
      TVM_FFI_ICHECK_EQ(K, hidden_states.size(1))
          << which_weights << " weights K dimension must be equal to hidden_size.";
    } else if (which_weights == "gemm2") {
      // GEMM2 always consumes the post-activation hidden of size intermediate_size.
      TVM_FFI_ICHECK_EQ(K, args->intermediate_size)
          << which_weights << " weights K dimension must be equal to intermediate_size.";
    }
  }

  void check_routing_common() const {
    TVM_FFI_ICHECK(args->top_k > 0 && args->top_k <= args->num_experts)
        << "top_k must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_num_experts > 0 && args->local_num_experts <= args->num_experts)
        << "local_num_experts must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_expert_offset >= 0 &&
                   args->local_expert_offset + args->local_num_experts <= args->num_experts)
        << "expert offset and count must be within valid range";

    check_routing_logits();

    if (routing_bias.has_value()) {
      check_routing_bias_shape();
    }
  }

  // Routing phase workspace tensors (allocated in prepare_routing() or prepare_routing_common())
  Tensor num_tokens_per_expert;
  Tensor total_num_padded_tokens;
  Tensor expanded_idx_to_permuted_idx;
  Tensor permuted_idx_to_token_idx;
  Tensor expert_weights;
  Tensor expert_indexes;
  Tensor expert_count_histogram;
  Tensor cta_idx_xy_to_batch_idx;
  Tensor cta_idx_xy_to_mn_limit;
  Tensor num_non_exiting_ctas;

  void prepare_routing_common() {
    // Allocate routing phase workspace tensors
    num_tokens_per_expert = alloc_tensor({args->num_experts}, dl_int32, hidden_states.device());
    int32_t max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());

    expanded_idx_to_permuted_idx =
        alloc_tensor({args->num_tokens * args->top_k}, dl_int32, hidden_states.device());

    permuted_idx_to_token_idx =
        alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());

    expert_indexes =
        alloc_tensor({args->num_tokens, args->top_k}, dl_int32, hidden_states.device());

    // expert_weights allocation should be done by derived class since data type could vary

    int64_t const size_of_expert_count_histogram = std::max(args->num_experts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor({size_of_expert_count_histogram},
                                          dl_int32,  // 256 is the max number of threads per block
                                                     // and max number of experts
                                          hidden_states.device());

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states.device());

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = static_cast<int*>(expert_indexes.data_ptr());
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.expanded_idx_to_permuted_idx =
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx.data_ptr());
    // workspace.expert_weights will be set by derived class after expert_weights allocation
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas.data_ptr());

    // Set dtype of score based on actual routing_logits dtype
    if (routing_logits.has_value()) {
      if (routing_logits.value().dtype() == dl_float32) {
        mDtypeScore = btg::Dtype::Fp32;
      } else {
        mDtypeScore = btg::Dtype::Bfloat16;
      }
    }
  }

  void check_moe_common() const {
    // Hidden states [num_tokens, hidden_size]
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
  }

  // MoE computation phase workspace tensors (allocated in prepare_moe() or prepare_moe_common())
  Tensor gemm1_output;
  Tensor activation_output;
  Tensor gemm2_output;
  Tensor workspace_fc1;
  Tensor workspace_fc2;
  Tensor output;
  int64_t moe_tactic{-1};
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;

  void prepare_moe_common(int64_t& moe_tactic) {
    using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;
    bool usePerTokenScalingGemm1 =
        per_token_scales.has_value() ||
        static_cast<RoutingMethodType>(this->routing_method_type) == RoutingMethodType::Llama4;
    bool usePerTokenScalingGemm2 =
        per_token_scales.has_value() && this->mDtypeAct != btg::Dtype::Bfloat16;
    // For FP8 block-scale (E4m3 activations, E4m3 weights) with DeepSeek FP8, use the
    // weights-only Runner constructor to match the original kernel path and numerics.
    if (this->mDtypeAct == btg::Dtype::E4m3 && this->mDtypeWeights == btg::Dtype::E4m3 &&
        args->mUseDeepSeekFp8) {
      moe_runner = std::make_unique<RunnerType>(this->mDtypeWeights, args->mUseDeepSeekFp8,
                                                (int32_t)tile_tokens_dim, this->use_shuffled_weight,
                                                this->weight_layout, usePerTokenScalingGemm1,
                                                usePerTokenScalingGemm2, false, false);
    } else {
      moe_runner = std::make_unique<RunnerType>(
          this->mDtypeAct, this->mDtypeWeights, args->mUseDeepSeekFp8, (int32_t)tile_tokens_dim,
          this->activation_type, this->use_shuffled_weight, this->weight_layout,
          usePerTokenScalingGemm1, usePerTokenScalingGemm2);
    }

    if (moe_tactic == -1) {
      moe_tactic = moe_runner->getDefaultValidConfigIndex(
          args->top_k, args->hidden_size, args->intermediate_size, args->local_num_experts,
          args->num_tokens);
    }
    auto valid_cfgs =
        moe_runner->getValidConfigIndices(args->top_k, args->hidden_size, args->intermediate_size,
                                          args->local_num_experts, args->num_tokens);
    auto valid_it = std::find(valid_cfgs.begin(), valid_cfgs.end(), moe_tactic);
    FLASHINFER_CHECK(valid_it != valid_cfgs.end(), "Invalid MoE tactic ", moe_tactic,
                     " for tile_N=", tile_tokens_dim, ". Number of valid tactics for this tile is ",
                     valid_cfgs.size(),
                     ". This often indicates a stale or mismatched autotuner cache entry.");
    this->moe_tactic = moe_tactic;

    auto workspace_sizes = moe_runner->getWorkspaceSizeInBytes(*args, moe_tactic);
    workspace_fc1 = alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states.device());
    workspace_fc2 = alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states.device());
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }

 public:
  virtual void check_routing() const = 0;
  virtual void prepare_routing() = 0;
  virtual void check_moe() const = 0;
  virtual void prepare_moe(int64_t& moe_tactic) = 0;

  // Main entry point for all the executions.
  // Do initializations prior to calling this as the initializations are different for bf16, fp8 and
  // fp4. The executions are non-blocking by default.
  virtual Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,
                            bool use_routing_scales_on_input = false,
                            bool use_deep_seek_fp8 = false) {
    check_routing();
    prepare_routing();

    // Execute routing
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(hidden_states.device());

    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(
        args->routing_logits, args->routing_bias, args->num_tokens, args->num_experts, args->top_k,
        args->n_group, args->topk_group, args->local_expert_offset, args->local_num_experts,
        args->routed_scaling_factor, workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), workspace.expert_weights,
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }
};

void FusedMoeLauncher::init_common(
    std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
    int64_t weight_layout, ActivationType activation_type, bool norm_topk_prob) {
  // Check devicearchitecture: Blackwell (SM 10.x) required
  auto device = hidden_states.device().device_id;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TVM_FFI_ICHECK(major == 10 || major == 12)
      << "MoE kernel requires SM 10.x or SM 12.x architecture. Current device has SM " << major
      << minor;
  this->device_version = std::make_tuple(major, minor);

  args->routing_logits = routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr;
  args->routing_bias = routing_bias.has_value() ? routing_bias.value().data_ptr() : nullptr;
  args->hidden_states = hidden_states.data_ptr();
  args->gemm1_weights = gemm1_weights.data_ptr();
  args->gemm2_weights = gemm2_weights.data_ptr();

  this->args = std::move(args);
  this->tile_tokens_dim = tile_tokens_dim;
  this->routing_method_type = routing_method_type;
  this->use_shuffled_weight = use_shuffled_weight;
  TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2)
      << "the value of weight_layout is not recognized";
  this->weight_layout = static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout);
  this->activation_type = activation_type;
  this->intermediate_size_factor = isGatedActivation(activation_type) ? 2 : 1;
  this->norm_topk_prob = norm_topk_prob;
}

class Bf16MoeLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  Bf16MoeLauncher(Optional<TensorView> const& routing_logits,
                  Optional<TensorView> const& routing_bias, TensorView const& expert_indices,
                  TensorView const& expert_weights, TensorView const& hidden_states,
                  TensorView const& gemm1_weights, TensorView const& gemm2_weights)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                         Optional<TensorView>(), Optional<TensorView>(), gemm2_weights,
                         Optional<TensorView>(), Optional<TensorView>()),
        expert_indices(expert_indices),
        expert_weights(expert_weights) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, bool norm_topk_prob = true) {
    // Do base class init and perform common checks
    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
    if (expert_indices.ndim() == 2 && expert_indices.size(0) > 0) {
      // Pre-computed routing: expert_indices is a packed tensor
      // Format: (expert_id << 16) | (weight_bf16.view(int16))
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }

    // TODO n_group, topk_group validation?
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    bool has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    if (has_precomputed_indices) {
      // Use expert_indices directly
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    }
    bool has_precomputed_weights = expert_weights.ndim() == 2 && expert_weights.size(0) > 0;
    if (has_precomputed_weights) {
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    } else {
      auto ew_dtype = mDtypeScore == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, ew_dtype, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK)
        << "BF16 Moe: weight_layout must be BlockMajorK";
    check_weights_shape("gemm1");
    check_weights_shape("gemm2");

    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
    gemm1_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16,
                                hidden_states.device());
    activation_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16,
                                     hidden_states.device());
    gemm2_output = alloc_tensor({max_num_padded_tokens, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = activation_output.data_ptr();
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;
  }

  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               bool use_shuffled_weight, int64_t weight_layout) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16,  // dtype_act
          btg::Dtype::Bfloat16,  // dtype_weights
          false,                 // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type), use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }

 private:
  TensorView expert_weights;
  TensorView expert_indices;
};

class Fp8PerTensorLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  // Constructor that passes TensorView parameters to base constructor
  Fp8PerTensorLauncher(TensorView const& routing_logits, Optional<TensorView> const& routing_bias,
                       TensorView const& hidden_states, TensorView const& gemm1_weights,
                       TensorView const& output1_scales_scalar,
                       TensorView const& output1_scales_gate_scalar,
                       TensorView const& gemm2_weights, TensorView const& output2_scales_scalar)
      : FusedMoeLauncher(Optional<TensorView>(routing_logits), routing_bias, hidden_states,
                         gemm1_weights, Optional<TensorView>(output1_scales_scalar),
                         Optional<TensorView>(output1_scales_gate_scalar), gemm2_weights,
                         Optional<TensorView>(output2_scales_scalar), Optional<TensorView>()),
        use_routing_scales_on_input(false) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, bool use_routing_scales_on_input_param,
            ActivationType activation_type, bool norm_topk_prob = true) {
    this->use_routing_scales_on_input = use_routing_scales_on_input_param;

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      mDtypeAct = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      mDtypeAct = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      mDtypeAct = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for FP8 MoE.";
    }
    mDtypeWeights = btg::Dtype::E4m3;

    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  norm_topk_prob);
  }

  void check_routing() const override { FusedMoeLauncher::check_routing_common(); }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    args->mDtypeOut = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    auto expert_weights_dtype = mRoutingLogitsDtype == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
    expert_weights =
        alloc_tensor({args->num_tokens, args->top_k}, expert_weights_dtype, hidden_states.device());

    workspace.expert_weights = expert_weights.data_ptr();
    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      workspace.token_scales = expert_weights.data_ptr();  // Consumed by permuteGemm1 kernel
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(output1_scales_scalar.has_value())
        << "output1_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().dtype(), dl_float32)
        << "output1_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().ndim(), 1)
        << "output1_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
        << "output1_scales_gate_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().dtype(), dl_float32)
        << "output1_scales_gate_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().ndim(), 1)
        << "output1_scales_gate_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_gate_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output2_scales_scalar.has_value())
        << "output2_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().dtype(), dl_float32)
        << "output2_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().ndim(), 1)
        << "output2_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().size(0), args->local_num_experts)
        << "output2_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(hidden_states.dtype() == dl_float8_e4m3fn ||
                   hidden_states.dtype() == dl_float16 || hidden_states.dtype() == dl_bfloat16)
        << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens_gemm1 = workspace.total_max_padded_tokens + args->num_experts;
    int32_t max_num_padded_tokens_gemm2 = workspace.total_max_padded_tokens;

    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, 2 * args->intermediate_size},
                                dl_uint8, hidden_states.device());
    gemm1_output_scale =
        alloc_tensor({2 * args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32,
                     hidden_states.device());

    activation_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size},
                                     dl_uint8, hidden_states.device());
    activation_output_scale =
        alloc_tensor({args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32,
                     hidden_states.device());

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = activation_output.data_ptr();
    workspace.activation_output_scale = static_cast<float*>(activation_output_scale.data_ptr());
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    // Set scale pointers
    TVM_FFI_ICHECK(output1_scales_scalar.has_value());
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value());
    TVM_FFI_ICHECK(output2_scales_scalar.has_value());

    args->output1_scales_scalar = static_cast<float*>(output1_scales_scalar.value().data_ptr());
    args->output1_scales_gate_scalar =
        static_cast<float*>(output1_scales_gate_scalar.value().data_ptr());
    args->output2_scales_scalar = static_cast<float*>(output2_scales_scalar.value().data_ptr());
  }

 private:
  bool use_routing_scales_on_input;
  Tensor gemm1_output_scale;
  Tensor activation_output_scale;

 public:
  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               bool use_shuffled_weight, int64_t weight_layout,
                                               btg::Dtype dtype_act, btg::Dtype dtype_weights) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act, dtype_weights,
          false,  // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type), use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout),
          true,  // usePerTokenScalingGemm1. always true for per-tensor fp8 due to llama4 routing
          false, false, false);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class Fp8BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mBaseSupportedTileNums = {8, 16, 32, 64, 128};

  static std::vector<int32_t> getSupportedTileNums(Fp8QuantizationType quantization_type) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (quantization_type == Fp8QuantizationType::MxFp8) {
      tiles.push_back(256);
    }
    return tiles;
  }

  Fp8BlockScaleLauncher(Optional<TensorView> const& routing_logits,
                        Optional<TensorView> const& routing_bias, TensorView const& hidden_states,
                        TensorView const& hidden_states_scale, TensorView const& gemm1_weights,
                        TensorView const& gemm1_weights_scale, TensorView const& gemm2_weights,
                        TensorView const& gemm2_weights_scale, TensorView const& expert_indices,
                        TensorView const& expert_weights, Fp8QuantizationType quantization_type,
                        Optional<TensorView> const& gate_up_lora_delta = Optional<TensorView>(),
                        Optional<TensorView> const& activation_lora_input = Optional<TensorView>())
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                         Optional<TensorView>(), Optional<TensorView>(), gemm2_weights,
                         Optional<TensorView>(), Optional<TensorView>()),
        hidden_states_scale(hidden_states_scale),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm2_weights_scale(gemm2_weights_scale),
        expert_indices(expert_indices),
        expert_weights(expert_weights),
        gate_up_lora_delta(gate_up_lora_delta),
        activation_lora_input(activation_lora_input),
        quantization_type(quantization_type) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, bool norm_topk_prob = true) {
    if (quantization_type == Fp8QuantizationType::MxFp8) {
      mDtypeAct = btg::Dtype::MxE4m3;
      mDtypeWeights = btg::Dtype::MxE4m3;
    } else {
      mDtypeAct = btg::Dtype::E4m3;
      mDtypeWeights = btg::Dtype::E4m3;
    }

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    // Output is always bfloat16 for FP8 block scale
    args->mDtypeOut = btg::Dtype::Bfloat16;

    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  norm_topk_prob);
  }

  void check_routing() const override {
    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    if (expert_indices.ndim() == 2 && expert_indices.size(0) > 0) {
      // Pre-computed routing: expert_indices is a packed tensor
      // Format: (expert_id << 16) | (weight_bf16.view(int16))
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }

    FusedMoeLauncher::check_routing_common();

    if (static_cast<RoutingMethodType>(routing_method_type) != RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group <= 1)
          << "Current routing kernel (no groups) only supports n_group <= 1";
      TVM_FFI_ICHECK(args->topk_group <= 1)
          << "Current routing kernel (no groups) only supports topk_group <= 1";
    }

    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group != 0) << "n_group should not be zero for DeepSeekV3 routing";
      TVM_FFI_ICHECK(args->topk_group != 0) << "if n_group is given, topk_group must be given";
      TVM_FFI_ICHECK_EQ(args->num_experts % args->n_group, 0)
          << "num_experts must be divisible by n_group";
      // DeepSeekV3 routing supports top_k up to:
      // - 8  when num_experts <= 384 (NumKimiK2Experts)
      // - 22 when num_experts > 384 (NumNemotronExperts path)
      // Keep this in sync with LAUNCH_ROUTING_DEEPSEEK in trtllm_fused_moe_routing_deepseek.cu.
      constexpr int32_t kNumKimiK2Experts = 384;  // same as in trtllm_fused_moe_routing_deepseek.cu
      int32_t max_supported_top_k = args->num_experts <= kNumKimiK2Experts ? 8 : 22;
      TVM_FFI_ICHECK(args->top_k <= max_supported_top_k && args->top_k > 0)
          << "Current routing kernel (with groups) only supports top_k<=" << max_supported_top_k
          << " && top_k>0 for num_experts=" << args->num_experts << ".";
      TVM_FFI_ICHECK(args->topk_group <= 4 && args->topk_group > 0)
          << "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.";
      TVM_FFI_ICHECK_LE(args->topk_group, args->n_group)
          << "n_group must not be smaller than topk_group.";
      TVM_FFI_ICHECK_LT(args->top_k, (args->topk_group * args->num_experts / args->n_group))
          << "top_k must be less than total number of experts in selected groups";
    } else if (static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::Renormalize ||
               static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::RenormalizeNaive ||
               static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::SigmoidRenorm ||
               static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Sigmoid) {
      TVM_FFI_ICHECK(args->top_k <= 32 && args->top_k > 0)
          << "Current routing kernel (no groups) only supports top_k<=32 && top_k>0.";
    } else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      TVM_FFI_ICHECK_EQ(args->top_k, 1)
          << "Current routing kernel (no groups, Llama4) only supports top_k=1.";
    }

    TVM_FFI_ICHECK_EQ(args->num_experts % 4, 0)
        << "Routing kernel expects that num_experts must be divisible by 4";
    TVM_FFI_ICHECK_GT(args->num_experts, args->top_k) << "num_experts must be greater than top_k";
    TVM_FFI_ICHECK_LE(args->local_num_experts + args->local_expert_offset, args->num_experts)
        << "num_experts must be greater or equal to local_num_experts + local_expert_offset";
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    args->mUseDeepSeekFp8 = quantization_type == Fp8QuantizationType::DeepSeekFp8;
    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    bool has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    if (has_precomputed_indices) {
      // Use expert_indices directly
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    } else {
      // Use routing_logits directly
      args->routing_logits = static_cast<float*>(routing_logits.value().data_ptr());
    }
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    bool has_precomputed_weights = expert_weights.ndim() == 2 && expert_weights.size(0) > 0;
    if (!has_precomputed_weights) {
      auto ew_dtype = mDtypeScore == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, ew_dtype, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    } else {
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_float8_e4m3fn) << "hidden_states must be fp8.";
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_float32)
          << "hidden_states_scale must be float.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2) << "hidden_states_scale must be 2D.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), hidden_states.size(1) / 128)
          << "hidden_states_scale dim0 must match hidden_states dim1 / 128.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), args->num_tokens)
          << "hidden_states_scale dim1 must match num_tokens.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_CHECK(weight_layout == batchedGemm::gemm::MatrixLayout::MajorK,
                    "weight_layout must be MajorK for MxFp8.");
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_uint8);
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "gemm1_weights must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "gemm2_weights must be fp8.";

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float32)
          << "gemm1_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3) << "gemm1_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), args->local_num_experts)
          << "gemm1_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
          << "intermediate_size must be a multiple of 128.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1),
                        intermediate_size_factor * args->intermediate_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), args->hidden_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_uint8)
          << "gemm1_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float32)
          << "gemm2_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3) << "gemm2_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), args->local_num_experts)
          << "gemm2_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), args->hidden_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), args->intermediate_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_uint8)
          << "gemm2_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    check_weights_shape("gemm1");
    check_weights_shape("gemm2");

    if (gate_up_lora_delta.has_value()) {
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().dtype(), dl_bfloat16)
          << "gate_up_lora_delta must be bf16.";
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().ndim(), 3)
          << "gate_up_lora_delta must be [num_tokens, top_k, 2 * intermediate_size].";
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(0), args->num_tokens);
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(1), args->top_k);
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(2),
                        args->intermediate_size * intermediate_size_factor);
    }
    if (activation_lora_input.has_value()) {
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().dtype(), dl_bfloat16)
          << "activation_lora_input must be bf16.";
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().ndim(), 3)
          << "activation_lora_input must be [num_tokens, top_k, intermediate_size].";
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(0), args->num_tokens);
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(1), args->top_k);
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(2), args->intermediate_size);
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
          << "intermediate_size must be a multiple of 128.";
    }
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    // Calculate max_num_padded_tokens for gemm1 and gemm2 using maybeGetMinTokenCount
    int32_t max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(args->mDtypeElt));
    int32_t max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(args->mDtypeOut));

    gemm1_output = alloc_tensor(
        {max_num_padded_tokens_gemm1, intermediate_size_factor * args->intermediate_size}, dl_uint8,
        hidden_states.device());

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      gemm1_output_scale = alloc_tensor({intermediate_size_factor * args->intermediate_size / 128,
                                         workspace.total_max_padded_tokens},
                                        dl_float32, hidden_states.device());
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      // MxFP8 fuses the activation so no need for intermediate_size_factor
      int64_t sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1,
                                                                  args->intermediate_size / 32);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      activation_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size},
                                       dl_uint8, hidden_states.device());
      activation_output_scale =
          alloc_tensor({args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32,
                       hidden_states.device());
    }

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      workspace.activation_output = activation_output.data_ptr();
      workspace.activation_output_scale = static_cast<float*>(activation_output_scale.data_ptr());
    }
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    args->hidden_states_scale = static_cast<float*>(hidden_states_scale.data_ptr());
    args->gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale.data_ptr());
    args->gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale.data_ptr());
    args->gate_up_lora_delta =
        gate_up_lora_delta.has_value() ? gate_up_lora_delta.value().data_ptr() : nullptr;
    args->activation_lora_input =
        activation_lora_input.has_value() ? activation_lora_input.value().data_ptr() : nullptr;
  }

 private:
  TensorView hidden_states_scale;
  TensorView gemm1_weights_scale;
  TensorView gemm2_weights_scale;
  Tensor gemm1_output_scale;
  Tensor activation_output_scale;
  TensorView expert_indices;
  TensorView expert_weights;
  Optional<TensorView> gate_up_lora_delta;
  Optional<TensorView> activation_lora_input;
  Fp8QuantizationType quantization_type;

 public:
  // Override to handle pre-computed routing
  Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,
                    bool use_routing_scales_on_input = false,
                    bool use_deep_seek_fp8 = false) override {
    check_routing();
    prepare_routing();

    cudaStream_t routing_stream = get_stream(hidden_states.device());
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);

    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    bool use_precomputed = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    // When using pre-computed routing, pass nullptr as routing_logits to tell the
    // routing runner to use the pre-computed expert indices from workspace.routing_expert_indexes
    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(
        use_precomputed ? nullptr : args->routing_logits, args->routing_bias, args->num_tokens,
        args->num_experts, args->top_k, args->n_group, args->topk_group, args->local_expert_offset,
        args->local_num_experts, args->routed_scaling_factor, workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), workspace.expert_weights,
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }

  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k, int64_t hidden_size, int64_t intermediate_size, int64_t num_local_experts,
      int64_t num_tokens, bool use_shuffled_weight, int64_t weight_layout, btg::Dtype dtype_act,
      btg::Dtype dtype_weights, Fp8QuantizationType quantization_type, int64_t act_type) {
    Array<Array<int64_t>> valid_configs;
    auto activation_type = validateAndCastActivationType(act_type);

    auto supported_tile_nums = getSupportedTileNums(quantization_type);
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;
      // Keep getValidConfigs constructor path aligned with runtime prepare_moe_common().
      // This branch is for DeepSeek FP8 (E4m3 activations + E4m3 weights).
      if (quantization_type == Fp8QuantizationType::DeepSeekFp8 && dtype_act == btg::Dtype::E4m3 &&
          dtype_weights == btg::Dtype::E4m3) {
        TVM_FFI_ICHECK(static_cast<int>(activation_type) ==
                       static_cast<int>(ActivationType::Swiglu))
            << "DeepSeekFp8 only supports ActivationType::Swiglu, got "
            << static_cast<int>(activation_type) << ".";
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_weights, true /* useDeepSeekFp8 */, tile_N, use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));
      } else {
        // Under current trtllm_get_valid_moe_configs() dispatch rules, this else-path is
        // reached only by FP8 block-scale MXFP8 (dtype_act=dtype_weights=MxE4m3).
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_act,                                              // dtypeAct
            dtype_weights,                                          // dtypeWeights
            quantization_type == Fp8QuantizationType::DeepSeekFp8,  // useDeepSeekFp8
            tile_N, activation_type, use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));
      }

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class MxInt4BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  MxInt4BlockScaleLauncher(TensorView const& routing_logits,
                           Optional<TensorView> const& routing_bias,
                           TensorView const& hidden_states, TensorView const& gemm1_weights,
                           TensorView const& gemm1_weights_scale,
                           Optional<TensorView> const& gemm1_alpha,
                           Optional<TensorView> const& gemm1_beta,
                           Optional<TensorView> const& gemm1_clamp_limit,
                           TensorView const& gemm2_weights, TensorView const& gemm2_weights_scale)
      : FusedMoeLauncher(Optional<TensorView>(routing_logits), routing_bias, hidden_states,
                         gemm1_weights, Optional<TensorView>(), Optional<TensorView>(),
                         gemm2_weights, Optional<TensorView>(), Optional<TensorView>()),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm2_weights_scale(gemm2_weights_scale) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool norm_topk_prob = true) {
    // currently only support mxint4 x bf16
    auto dtype = hidden_states.dtype();
    if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }
    args->mDtypeOut = btg::Dtype::Bfloat16;

    mDtypeAct = btg::Dtype::Bfloat16;
    mDtypeWeights = btg::Dtype::MxInt4;

    FusedMoeLauncher::init_common(
        std::move(args), tile_tokens_dim, routing_method_type,
        /*use_shuffled_weight=*/true,
        static_cast<int64_t>(batchedGemm::gemm::MatrixLayout::BlockMajorK), ActivationType::Swiglu,
        norm_topk_prob);
  }

  void check_routing() const override { FusedMoeLauncher::check_routing_common(); }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = mDtypeAct;
    args->mUseDeepSeekFp8 = false;
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    auto expert_weights_dtype = mRoutingLogitsDtype == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
    expert_weights =
        alloc_tensor({args->num_tokens, args->top_k}, expert_weights_dtype, hidden_states.device());

    workspace.expert_weights = expert_weights.data_ptr();
  }

  void check_moe() const override {
    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::Bfloat16)
        << "Only Bfloat16 is supported by MxInt4 block scale MoE";

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_bfloat16)
        << "gemm1_weights_scale must be bf16.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_bfloat16)
        << "gemm2_weights_scale must be bf16.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale = nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->output1_scales_scalar = nullptr;
    args->output1_scales_gate_scalar = nullptr;
    args->output2_scales_scalar = nullptr;

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden = args->intermediate_size;
    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_bfloat16,
                                hidden_states.device());

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    // Setup workspace pointers
    workspace.hidden_states_scale_linear = nullptr;  // MxInt4 doesn't use linear scale
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    // Note: activation_output and activation_output_scale are set by the base class
    // prepare_moe_common() when gated activation is used
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
  }

 private:
  TensorView gemm1_weights_scale;
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
  TensorView gemm2_weights_scale;
  int32_t max_num_padded_tokens_gemm1{};
  int32_t max_num_padded_tokens_gemm2{};

 public:
  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16, btg::Dtype::MxInt4,
          false,  // useDeepSeekFp8
          tile_N, ActivationType::Swiglu,
          /*useShuffledMatrix*/ true, batchedGemm::gemm::MatrixLayout::BlockMajorK);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class FP4BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 4> mBaseSupportedTileNums = {8, 16, 32, 64};

  static std::vector<int32_t> getSupportedTileNums(btg::Dtype dtype_act) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (dtype_act != btg::Dtype::Bfloat16) {
      tiles.push_back(128);
      tiles.push_back(256);
    }
    return tiles;
  }

  FP4BlockScaleLauncher(
      Optional<TensorView> const& routing_logits, Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states, Optional<TensorView> const& hidden_states_scale,
      TensorView const& gemm1_weights, TensorView const& gemm1_weights_scale,
      Optional<TensorView> const& gemm1_bias, Optional<TensorView> const& gemm1_alpha,
      Optional<TensorView> const& gemm1_beta, Optional<TensorView> const& gemm1_clamp_limit,
      TensorView const& gemm2_weights, TensorView const& gemm2_weights_scale,
      Optional<TensorView> const& gemm2_bias, Optional<TensorView> const& output1_scales_scalar,
      Optional<TensorView> const& output1_scales_gate_scalar,
      Optional<TensorView> const& output2_scales_scalar,
      Optional<TensorView> const& per_token_scales, TensorView const& expert_indices,
      TensorView const& expert_weights)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                         output1_scales_scalar, output1_scales_gate_scalar, gemm2_weights,
                         output2_scales_scalar, per_token_scales),
        hidden_states_scale(hidden_states_scale),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm1_bias(gemm1_bias),
        gemm1_alpha(gemm1_alpha),
        gemm1_beta(gemm1_beta),
        gemm1_clamp_limit(gemm1_clamp_limit),
        gemm2_weights_scale(gemm2_weights_scale),
        gemm2_bias(gemm2_bias),
        expert_indices(expert_indices),
        expert_weights(expert_weights) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, btg::Dtype dtype_act,
            btg::Dtype dtype_weights, bool norm_topk_prob = true) {
    // Set data types
    args->mDtypeElt = dtype_act;
    args->mDtypeOut = btg::Dtype::Bfloat16;  // Output is always BF16 for FP4
    args->mUseDeepSeekFp8 = false;           // FP4 doesn't use DeepSeek FP8

    mDtypeAct = dtype_act;
    mDtypeWeights = dtype_weights;

    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  norm_topk_prob);
  }

  void check_routing() const override {
    // First call base class common routing checks
    FusedMoeLauncher::check_routing_common();
  }

  void prepare_routing() override {
    num_tokens_per_expert = alloc_tensor({args->num_experts}, dl_int32, hidden_states.device());
    int32_t max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());
    expanded_idx_to_permuted_idx =
        alloc_tensor({args->num_tokens * args->top_k}, dl_int32, hidden_states.device());
    permuted_idx_to_token_idx =
        alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());

    int64_t const size_of_expert_count_histogram = std::max(args->num_experts * 2, 256 * 2);
    expert_count_histogram =
        alloc_tensor({size_of_expert_count_histogram}, dl_int32, hidden_states.device());

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);
    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());
    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());
    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states.device());

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes =
        static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.expanded_idx_to_permuted_idx =
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx.data_ptr());
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas.data_ptr());

    args->mDtypeElt = mDtypeAct;
    auto routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  }

  void check_moe() const override {
    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::Bfloat16 ||
                   mDtypeAct == btg::Dtype::E4m3 || mDtypeAct == btg::Dtype::MxE4m3)
        << "Only E2m1, Bfloat16, MxE4m3 and E4m3 are supported by Fp4 block scale MoE";

    if (mDtypeAct == btg::Dtype::E2m1) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::E2m1)
          << "Only E2m1 and MxE2m1 are supported by block scale MoE with E2m1 activation";
      TVM_FFI_ICHECK(hidden_states_scale.has_value())
          << "hidden_states_scale is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_scalar.has_value())
          << "output1_scales_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value())
          << "output2_scales_scalar is required for E2m1 activation";
    } else if (mDtypeAct == btg::Dtype::Bfloat16 || mDtypeAct == btg::Dtype::E4m3 ||
               mDtypeAct == btg::Dtype::MxE4m3) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::MxE2m1)
          << "Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or "
             "MxE4m3 activation";
    }

    if (mDtypeAct == btg::Dtype::E4m3) {
      TVM_FFI_ICHECK(output1_scales_scalar.has_value())
          << "output1_scales_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value())
          << "output2_scales_scalar is required for E4m3 activation";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float8_e4m3fn)
        << "gemm1_weights_scale must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float8_e4m3fn)
        << "gemm2_weights_scale must be fp8.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale =
        hidden_states_scale.has_value() ? hidden_states_scale.value().data_ptr() : nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_bias =
        gemm1_bias.has_value() ? static_cast<float*>(gemm1_bias.value().data_ptr()) : nullptr;
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->gemm2_bias =
        gemm2_bias.has_value() ? static_cast<float*>(gemm2_bias.value().data_ptr()) : nullptr;
    args->output1_scales_scalar =
        output1_scales_scalar.has_value()
            ? static_cast<float*>(output1_scales_scalar.value().data_ptr())
            : nullptr;
    args->output1_scales_gate_scalar =
        output1_scales_gate_scalar.has_value()
            ? static_cast<float*>(output1_scales_gate_scalar.value().data_ptr())
            : nullptr;
    args->output2_scales_scalar =
        output2_scales_scalar.has_value()
            ? static_cast<float*>(output2_scales_scalar.value().data_ptr())
            : nullptr;

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    auto const sf_vec_size = mDtypeWeights == btg::Dtype::MxE2m1 ? 32 : 16;

    max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden =
        mDtypeAct == btg::Dtype::E2m1 ? args->intermediate_size / 2 : args->intermediate_size;
    if (mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::MxE4m3) {
      int64_t sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(
          max_num_padded_tokens_gemm1, args->intermediate_size / sf_vec_size);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }
    if (!per_token_scales.has_value()) {
      gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden},
                                  mDtypeAct == btg::Dtype::Bfloat16 ? dl_bfloat16 : dl_uint8,
                                  hidden_states.device());
    } else {  // FC1 output is Bfloat16
      TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1)
          << "NvFP4 MoE: currently only support NvFP4 x NvFP4 when using per-token scaling.";
      // When per-token scales are used, the FC1 output is always BF16 and will be quantized
      gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size},
                                  dl_bfloat16, hidden_states.device());
      activation_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_uint8,
                                       hidden_states.device());
      per_token_scales_fc2 =
          alloc_tensor({max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());
    }

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    // Setup workspace pointers
    workspace.hidden_states_scale_linear = nullptr;  // FP4 doesn't use linear scale
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = gemm1_output_scale.has_value()
                                       ? static_cast<float*>(gemm1_output_scale.value().data_ptr())
                                       : nullptr;
    if (per_token_scales.has_value()) {
      workspace.token_scales = per_token_scales.value().data_ptr();
      workspace.activation_output = activation_output.data_ptr();
      workspace.activation_output_scale = workspace.gemm1_output_scale;
      workspace.token_scales_fc2 = per_token_scales_fc2.data_ptr();
    }
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
  }

 private:
  Optional<TensorView> hidden_states_scale;
  TensorView gemm1_weights_scale;
  Optional<TensorView> gemm1_bias;
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
  TensorView gemm2_weights_scale;
  Optional<TensorView> gemm2_bias;
  int32_t max_num_padded_tokens_gemm1{};
  int32_t max_num_padded_tokens_gemm2{};
  Optional<Tensor> gemm1_output_scale;
  TensorView expert_indices;
  TensorView expert_weights;

 public:
  Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true,
                    bool use_routing_scales_on_input = false,
                    bool use_deep_seek_fp8 = false) override {
    check_routing();
    prepare_routing();

    // Execute routing
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(hidden_states.device());

    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(
        args->routing_logits, args->routing_bias, args->num_tokens, args->num_experts, args->top_k,
        args->n_group, args->topk_group, args->local_expert_offset, args->local_num_experts,
        args->routed_scaling_factor, static_cast<int*>(expert_indices.data_ptr()),
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), expert_weights.data_ptr(),
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    // Match original FP4 behavior for return values
    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }

  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               btg::Dtype dtype_act, btg::Dtype dtype_weights,
                                               bool use_per_token_scaling) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes = getSupportedTileNums(dtype_act);
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act, dtype_weights,
          false,  // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type),
          /*useShuffledMatrix*/ true,
          /*weight_layout*/ batchedGemm::gemm::MatrixLayout::MajorK,
          // NOTE(siyuan): currently FP4 MoE always apply per-token scaling to both FC1 and FC2.
          /*usePerTokenScalingGemm1*/ use_per_token_scaling,
          /*usePerTokenScalingGemm2*/ use_per_token_scaling, false, false);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

Array<Tensor> trtllm_bf16_moe(Optional<TensorView> const& routing_logits,
                              Optional<TensorView> const& routing_bias,
                              TensorView const& expert_indices, TensorView const& expert_weights,
                              TensorView const& hidden_states, TensorView const& gemm1_weights,
                              TensorView const& gemm2_weights, TensorView output,
                              int64_t num_experts, int64_t top_k, Optional<int64_t> n_group,
                              Optional<int64_t> topk_group, int64_t intermediate_size,
                              int64_t local_expert_offset, int64_t local_num_experts,
                              Optional<double> routed_scaling_factor, int64_t routing_method_type,
                              bool use_shuffled_weight, int64_t weight_layout, bool do_finalize,
                              bool enable_pdl, Array<int64_t> moe_tactic, int64_t activation_type,
                              bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  // Just some basic type validation first and leave more checks to the launcher
  if (routing_logits.has_value()) {
    TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                   routing_logits.value().dtype() == dl_bfloat16)
        << "BF16 MoE: routing_logits must be bfloat16 or float.";
  }
  TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_bfloat16)
      << "BF16 MoE: hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_bfloat16)
      << "BF16 MoE: gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_bfloat16)
      << "BF16 MoE: gemm2_weights must be bfloat16.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  auto const activation = validateAndCastActivationType(activation_type);

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(Bf16MoeLauncher::mSupportedTileNums.begin(),
                                       Bf16MoeLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles (not just the computeSelectedTileN subset)
  // so that autotuner-cached tactics always find their tile_N in the map.
  // Launcher creation is cheap (no GPU allocation until run()), so this is safe.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Bf16MoeLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Bf16MoeLauncher>(routing_logits, routing_bias, expert_indices,
                                                      expert_weights, hidden_states, gemm1_weights,
                                                      gemm2_weights);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, activation, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(moe_tactic, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing BF16 MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

Array<Tensor> trtllm_fp8_per_tensor_scale_moe(
    TensorView routing_logits, Optional<TensorView> routing_bias, TensorView hidden_states,
    TensorView gemm1_weights, TensorView output1_scales_scalar,
    TensorView output1_scales_gate_scalar, TensorView gemm2_weights,
    TensorView output2_scales_scalar, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    bool use_routing_scales_on_input, int64_t routing_method_type, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, int64_t activation_type, bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  // Basic type validation
  auto dtype = hidden_states.dtype();
  auto activation = validateAndCastActivationType(activation_type);

  TVM_FFI_ICHECK(dtype == dl_float8_e4m3fn || dtype == dl_float16 || dtype == dl_bfloat16)
      << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_gate_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output2_scales_scalar must be float32.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);

  // Use default values that match the original function behavior
  bool use_shuffled_weight = true;  // Original uses /*useShuffledMatrix*/ true
  int64_t weight_layout = 0;        // Default to MajorK

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(Fp8PerTensorLauncher::mSupportedTileNums.begin(),
                                       Fp8PerTensorLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Fp8PerTensorLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Fp8PerTensorLauncher>(
        routing_logits, routing_bias, hidden_states, gemm1_weights, output1_scales_scalar,
        output1_scales_gate_scalar, gemm2_weights, output2_scales_scalar);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, use_routing_scales_on_input, activation, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP8 per-tensor MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl, use_routing_scales_on_input);
}

Array<Tensor> trtllm_fp8_block_scale_moe_impl(
    Optional<TensorView> routing_logits, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states, TensorView hidden_states_scale,
    TensorView gemm1_weights, TensorView gemm1_weights_scale, TensorView gemm2_weights,
    TensorView gemm2_weights_scale, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_shuffled_weight, int64_t weight_layout, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, Fp8QuantizationType quantization_type,
    int64_t act_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out,
    Optional<TensorView> gate_up_lora_delta, Optional<TensorView> activation_lora_input) {
  auto activation_type = validateAndCastActivationType(act_type);
  // DeepSeekFp8 currently uses a TRTLLM runner that hardwires Swiglu activation semantics.
  // Fail for any other activation to avoid silently running incorrect activation behavior.
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8 &&
      activation_type != ActivationType::Swiglu) {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "DeepSeekFp8 only supports ActivationType::Swiglu in this runner path. "
        << "Received activation_type=" << static_cast<int>(activation_type);
  }

  // Basic type validation
  auto dtype = hidden_states.dtype();

  // Either routing_logits or expert_indices must be provided
  // expert_indices is a packed tensor: (expert_id << 16) | (weight_bf16.view(int16))
  bool use_routing_logits = routing_logits.has_value();
  // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
  bool use_precomputed_routing = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;

  TVM_FFI_ICHECK(use_routing_logits || use_precomputed_routing)
      << "Either routing_logits or expert_indices must be provided.";

  (void)use_routing_logits;
  TVM_FFI_ICHECK(dtype == dl_float16 || dtype == dl_bfloat16 || dtype == dl_float8_e4m3fn)
      << "FP8 block scale MoE: hidden_states must be fp16, bf16, or fp8.";
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: hidden_states_scale must be float32.";
  } else if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: hidden_states_scale must be uint8.";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
  }
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 block scale MoE: gemm1_weights must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 block scale MoE: gemm2_weights must be fp8.";
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: gemm1_weights_scale must be float32.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: gemm2_weights_scale must be float32.";
  } else if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: gemm1_weights_scale must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: gemm2_weights_scale must be uint8.";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
  }

  if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK(use_shuffled_weight) << "use_shuffled_weight must be true for MxFp8.";
    TVM_FFI_ICHECK(weight_layout == 0) << "weight_layout must be 0 for MxFp8.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);

  auto supported_tile_nums = Fp8BlockScaleLauncher::getSupportedTileNums(quantization_type);
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Fp8BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : supported_tile_nums) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Fp8BlockScaleLauncher>(
        routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, expert_indices, expert_weights,
        quantization_type, gate_up_lora_delta, activation_lora_input);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, activation_type, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] = resolveMoeTileAndConfig(config_index, supported_tile_nums,
                                                        num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP8 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher with DeepSeek FP8 enabled - it will create its own runner internally
  return selected_launcher->run(
      config, enable_pdl, false /* use_routing_scales_on_input */,
      quantization_type == Fp8QuantizationType::DeepSeekFp8 /* use_deep_seek_fp8 */);
}

Array<Tensor> trtllm_fp8_block_scale_moe(
    Optional<TensorView> routing_logits, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states, TensorView hidden_states_scale,
    TensorView gemm1_weights, TensorView gemm1_weights_scale, TensorView gemm2_weights,
    TensorView gemm2_weights_scale, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_shuffled_weight, int64_t weight_layout, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, Fp8QuantizationType quantization_type,
    int64_t act_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  return trtllm_fp8_block_scale_moe_impl(
      routing_logits, expert_indices, expert_weights, routing_bias, hidden_states,
      hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
      output, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
      local_num_experts, routed_scaling_factor, routing_method_type, use_shuffled_weight,
      weight_layout, do_finalize, enable_pdl, config_index, quantization_type, act_type,
      norm_topk_prob, routing_replay_out, Optional<TensorView>(), Optional<TensorView>());
}

Array<Tensor> sgl_trtllm_fp8_block_scale_moe_lora(
    Optional<TensorView> routing_logits, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states, TensorView hidden_states_scale,
    TensorView gemm1_weights, TensorView gemm1_weights_scale, TensorView gemm2_weights,
    TensorView gemm2_weights_scale, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_shuffled_weight, int64_t weight_layout, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, Fp8QuantizationType quantization_type,
    int64_t act_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out,
    TensorView gate_up_lora_delta, TensorView activation_lora_input) {
  if (quantization_type != Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "sgl_trtllm_fp8_block_scale_moe_lora currently supports DeepSeekFp8 only.";
  }
  return trtllm_fp8_block_scale_moe_impl(
      routing_logits, expert_indices, expert_weights, routing_bias, hidden_states,
      hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
      output, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
      local_num_experts, routed_scaling_factor, routing_method_type, use_shuffled_weight,
      weight_layout, do_finalize, enable_pdl, config_index, quantization_type, act_type,
      norm_topk_prob, routing_replay_out, Optional<TensorView>(gate_up_lora_delta),
      Optional<TensorView>(activation_lora_input));
}

__global__ void sgl_trtllm_fp8_block_scale_moe_lora_finalize_kernel(
    cutlass::bfloat16_t const* __restrict__ gemm2_output,
    cutlass::bfloat16_t const* __restrict__ expert_weights,
    int32_t const* __restrict__ expanded_idx_to_permuted_idx,
    cutlass::bfloat16_t const* __restrict__ down_lora_delta,
    cutlass::bfloat16_t* __restrict__ output, int64_t num_tokens, int64_t top_k,
    int64_t hidden_size, int64_t hidden_size_padded, float routed_scaling_factor) {
  for (int64_t token_idx = blockIdx.y; token_idx < num_tokens; token_idx += gridDim.y) {
    for (int64_t hidden_idx = threadIdx.x + blockDim.x * blockIdx.x; hidden_idx < hidden_size;
         hidden_idx += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      float lora_acc = 0.0f;
      for (int64_t k = 0; k < top_k; ++k) {
        int64_t const expanded_idx = token_idx * top_k + k;
        int32_t const permuted_idx = expanded_idx_to_permuted_idx[expanded_idx];
        if (permuted_idx != -1) {
          float const expert_prob =
              static_cast<float>(expert_weights[token_idx * top_k + k]);
          acc += expert_prob *
                 static_cast<float>(gemm2_output[permuted_idx * hidden_size_padded + hidden_idx]);
        }
        lora_acc +=
            static_cast<float>(down_lora_delta[expanded_idx * hidden_size + hidden_idx]);
      }
      output[token_idx * hidden_size + hidden_idx] =
          static_cast<cutlass::bfloat16_t>(acc + routed_scaling_factor * lora_acc);
    }
  }
}

void sgl_trtllm_fp8_block_scale_moe_lora_finalize(
    TensorView gemm2_output, TensorView expert_weights, TensorView expanded_idx_to_permuted_idx,
    TensorView down_lora_delta, TensorView output, Optional<double> routed_scaling_factor) {
  TVM_FFI_ICHECK_EQ(gemm2_output.dtype(), dl_bfloat16)
      << "gemm2_output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(expert_weights.dtype(), dl_bfloat16)
      << "expert_weights must be bfloat16.";
  TVM_FFI_ICHECK((expanded_idx_to_permuted_idx.dtype() == DLDataType{kDLInt, 32, 1}))
      << "expanded_idx_to_permuted_idx must be int32.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.dtype(), dl_bfloat16)
      << "down_lora_delta must be bfloat16.";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16) << "output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_output.ndim(), 2) << "gemm2_output must be 2D.";
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.ndim(), 1)
      << "expanded_idx_to_permuted_idx must be 1D.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.ndim(), 3) << "down_lora_delta must be 3D.";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D.";
  TVM_FFI_ICHECK(gemm2_output.IsContiguous()) << "gemm2_output must be contiguous.";
  TVM_FFI_ICHECK(expert_weights.IsContiguous()) << "expert_weights must be contiguous.";
  TVM_FFI_ICHECK(expanded_idx_to_permuted_idx.IsContiguous())
      << "expanded_idx_to_permuted_idx must be contiguous.";
  TVM_FFI_ICHECK(down_lora_delta.IsContiguous()) << "down_lora_delta must be contiguous.";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous.";

  int64_t const num_tokens = output.size(0);
  int64_t const hidden_size = output.size(1);
  int64_t const top_k = down_lora_delta.size(1);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens)
      << "expert_weights dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(expert_weights.size(1), top_k) << "expert_weights dim1 must equal top_k.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(0), num_tokens)
      << "down_lora_delta dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(2), hidden_size)
      << "down_lora_delta dim2 must equal hidden_size.";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.size(0), num_tokens * top_k)
      << "expanded_idx_to_permuted_idx size must equal num_tokens * top_k.";
  TVM_FFI_ICHECK(gemm2_output.size(1) >= hidden_size)
      << "gemm2_output hidden dimension is smaller than output hidden dimension.";

  int const num_threads = 128;
  int const num_blocks_x = (hidden_size + num_threads - 1) / num_threads;
  int const num_blocks_y = std::min<int64_t>(8192, num_tokens);
  dim3 grid(num_blocks_x, num_blocks_y);
  cudaStream_t stream = get_stream(output.device());
  sgl_trtllm_fp8_block_scale_moe_lora_finalize_kernel<<<grid, num_threads, 0, stream>>>(
      static_cast<cutlass::bfloat16_t const*>(gemm2_output.data_ptr()),
      static_cast<cutlass::bfloat16_t const*>(expert_weights.data_ptr()),
      static_cast<int32_t const*>(expanded_idx_to_permuted_idx.data_ptr()),
      static_cast<cutlass::bfloat16_t const*>(down_lora_delta.data_ptr()),
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()), num_tokens, top_k, hidden_size,
      gemm2_output.size(1), static_cast<float>(routed_scaling_factor.value_or(1.0)));
  auto err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

Array<Tensor> trtllm_fp4_block_scale_moe(
    Optional<TensorView> routing_logits, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states,
    Optional<TensorView> hidden_states_scale, TensorView gemm1_weights,
    TensorView gemm1_weights_scale, Optional<TensorView> gemm1_bias,
    Optional<TensorView> gemm1_alpha, Optional<TensorView> gemm1_beta,
    Optional<TensorView> gemm1_clamp_limit, TensorView gemm2_weights,
    TensorView gemm2_weights_scale, Optional<TensorView> gemm2_bias,
    Optional<TensorView> output1_scales_scalar, Optional<TensorView> output1_scales_gate_scalar,
    Optional<TensorView> output2_scales_scalar, Optional<TensorView> per_token_scales,
    int64_t num_experts, int64_t top_k, Optional<int64_t> n_group, Optional<int64_t> topk_group,
    int64_t intermediate_size, int64_t local_expert_offset, int64_t local_num_experts,
    Optional<double> routed_scaling_factor, int64_t routing_method_type, bool do_finalize,
    bool enable_pdl, int64_t act_type, TensorView output, Array<int64_t> config_index,
    bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  if (hidden_states.dtype() == dl_uint8) hidden_size *= 2;

  int64_t hidden_states_scale_vec_size = -1;
  if (hidden_states_scale.has_value()) {
    hidden_states_scale_vec_size =
        (static_cast<int64_t>(num_tokens) * hidden_size) / hidden_states_scale.value().numel();
  }
  int64_t intermediate_size_factor =
      isGatedActivation(static_cast<ActivationType>(act_type)) ? 2 : 1;
  int64_t logical_scale_count = static_cast<int64_t>(local_num_experts) * intermediate_size *
                                intermediate_size_factor * hidden_size;
  int64_t weight_scale_vec_size_raw = logical_scale_count / gemm1_weights_scale.numel();

  // Snap to nearest valid sf_vec_size (16 or 32).
  // The raw value may be slightly smaller than the true vec_size because
  // block_scale_interleave pads scale columns to a multiple of 4, inflating numel().
  int64_t weight_scale_vec_size = weight_scale_vec_size_raw > 16 ? 32 : 16;

  // Round-trip validation: the unpadded scale count must not exceed actual numel
  // (padding only adds elements, never removes them).
  int64_t expected_unpadded = logical_scale_count / weight_scale_vec_size;
  TVM_FFI_ICHECK(gemm1_weights_scale.numel() >= expected_unpadded)
      << "weight scale tensor too small: numel=" << gemm1_weights_scale.numel()
      << " but expected at least " << expected_unpadded
      << " for sf_vec_size=" << weight_scale_vec_size;

  auto mDtypeWeights = weight_scale_vec_size == 16 ? btg::Dtype::E2m1 : btg::Dtype::MxE2m1;

  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 ||
                   routing_bias.value().dtype() == dl_float32)
        << "routing_bias must be bfloat16 or float.";

    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be fp4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_uint8 || hidden_states.dtype() == dl_bfloat16 ||
                 hidden_states.dtype() == dl_float8_e4m3fn)
      << "hidden_states must be bf16, fp8 or uint8 (packed fp4).";

  auto mDtypeAct = btg::Dtype::Bfloat16;
  if (hidden_states.dtype() == dl_uint8) {
    TVM_FFI_ICHECK(hidden_states_scale.has_value() &&
                   hidden_states_scale.value().dtype() == dl_float8_e4m3fn)
        << "hidden_states_scale must be provided for fp4 activation.";
    if (hidden_states_scale_vec_size == 16) {
      mDtypeAct = btg::Dtype::E2m1;
    } else if (hidden_states_scale_vec_size == 32) {
      mDtypeAct = btg::Dtype::MxE2m1;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state scale shape.";
    }
  } else if (hidden_states.dtype() == dl_float8_e4m3fn) {
    if (hidden_states_scale.has_value()) {
      if (hidden_states_scale_vec_size == 32) {
        mDtypeAct = btg::Dtype::MxE4m3;
      } else {
        TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state scale shape.";
      }
    } else {
      mDtypeAct = btg::Dtype::E4m3;
    }
  }

  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN = FP4BlockScaleLauncher::getSupportedTileNums(mDtypeAct);
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<FP4BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    // For E2m1, hidden_size is already multiplied by 2 above, so use it directly
    args->hidden_size = hidden_size;
    args->hidden_size_output = output.size(1);
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<FP4BlockScaleLauncher>(
        routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
        gemm2_weights_scale, gemm2_bias, output1_scales_scalar, output1_scales_gate_scalar,
        output2_scales_scalar, per_token_scales, expert_indices, expert_weights);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, /*use_shuffled_weight=*/true,
                   /*weight_layout=*/0, static_cast<ActivationType>(act_type), mDtypeAct,
                   mDtypeWeights, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP4 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

Array<Tensor> trtllm_mxint4_block_scale_moe(
    TensorView routing_logits, Optional<TensorView> routing_bias, TensorView hidden_states,
    TensorView gemm1_weights, TensorView gemm1_weights_scale, Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta, Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights, TensorView gemm2_weights_scale, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool do_finalize, bool enable_pdl, TensorView output,
    Array<int64_t> config_index, bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  // Just some basic type validation first and leave more checks to the launcher

  int weight_scale_vec_size =
      (local_num_experts * intermediate_size * 2 * hidden_size) / gemm1_weights_scale.numel();

  TVM_FFI_ICHECK(weight_scale_vec_size == 32) << "unsupported weight_scale_vec_size.";

  TVM_FFI_ICHECK(routing_logits.dtype() == dl_float32 || routing_logits.dtype() == dl_bfloat16)
      << "routing_logits must be float or bfloat16.";
  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2) << "routing_logits must be 2D.";
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), num_experts) << "routing_logits has incorrect shape.";
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16) << "routing_bias must be bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be int4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_bfloat16) << "hidden_states must be bf16.";

  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN(MxInt4BlockScaleLauncher::mSupportedTileNums.begin(),
                                       MxInt4BlockScaleLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<MxInt4BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    // For E2m1, hidden_size is already multiplied by 2 above, so use it directly
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<MxInt4BlockScaleLauncher>(
        routing_logits, routing_bias, hidden_states, gemm1_weights, gemm1_weights_scale,
        gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing MXINT4 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

Array<Array<int64_t>> trtllm_get_valid_moe_configs(
    int64_t const dtype_act_, int64_t const dtype_weights_,
    Fp8QuantizationType fp8_quantization_type, int64_t const top_k, int64_t const hidden_size,
    int64_t const intermediate_size, int64_t const num_local_experts, int64_t const act_type,
    bool const use_shuffled_weight, int64_t const weight_layout, bool const use_per_token_scaling,
    int64_t const num_tokens) {
  auto activation_type = validateAndCastActivationType(act_type);
  auto dtype_act = static_cast<btg::Dtype>(dtype_act_);
  auto dtype_weights = static_cast<btg::Dtype>(dtype_weights_);

  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::MxInt4) {
    // MxInt4 MoE
    return MxInt4BlockScaleLauncher::getValidConfigs(top_k, hidden_size, intermediate_size,
                                                     num_local_experts, num_tokens);
  }
  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::Bfloat16) {
    // BF16 MoE
    return Bf16MoeLauncher::getValidConfigs(top_k, hidden_size, intermediate_size,
                                            num_local_experts, num_tokens, act_type,
                                            use_shuffled_weight, weight_layout);

  } else if (fp8_quantization_type == Fp8QuantizationType::DeepSeekFp8 &&
             dtype_act == btg::Dtype::E4m3 && dtype_weights == btg::Dtype::E4m3) {
    if (activation_type != ActivationType::Swiglu) {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "DeepSeekFp8 only supports ActivationType::Swiglu, "
          << "got act_type=" << act_type << ".";
    }
    // FP8 block scale (DeepSeek)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, use_shuffled_weight,
        weight_layout, dtype_act, dtype_weights, fp8_quantization_type, act_type);
  } else if (fp8_quantization_type == Fp8QuantizationType::MxFp8 &&
             dtype_act == btg::Dtype::MxE4m3 && dtype_weights == btg::Dtype::MxE4m3) {
    // FP8 block scale (MxFp8)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, use_shuffled_weight,
        weight_layout, dtype_act, dtype_weights, fp8_quantization_type, act_type);
  } else if ((fp8_quantization_type == Fp8QuantizationType::PerTensorFp8 ||
              fp8_quantization_type == Fp8QuantizationType::NoneFp8) &&
             dtype_weights == btg::Dtype::E4m3) {
    return Fp8PerTensorLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type,
        use_shuffled_weight, weight_layout, dtype_act, dtype_weights);
  } else if (dtype_weights == btg::Dtype::E2m1 || dtype_weights == btg::Dtype::MxE2m1) {
    // FP4 block scale
    return FP4BlockScaleLauncher::getValidConfigs(top_k, hidden_size, intermediate_size,
                                                  num_local_experts, num_tokens, act_type,
                                                  dtype_act, dtype_weights, use_per_token_scaling);
  }

  TVM_FFI_LOG_AND_THROW(NotImplementedError)
      << "Unsupported data type combination for getValidConfigs: "
      << "dtype_act=" << static_cast<int>(dtype_act)
      << ", dtype_weights=" << static_cast<int>(dtype_weights)
      << ", fp8_quantization_type=" << fp8QuantizationTypeToString(fp8_quantization_type);

  // Unreachable code - added to suppress compiler warning
  return Array<Array<int64_t>>();
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_bf16_moe, trtllm_bf16_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_moe, trtllm_fp8_per_tensor_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_block_scale_moe, trtllm_fp8_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp8_block_scale_moe_lora,
                              sgl_trtllm_fp8_block_scale_moe_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp8_block_scale_moe_lora_finalize,
                              sgl_trtllm_fp8_block_scale_moe_lora_finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxint4_block_scale_moe, trtllm_mxint4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_configs, trtllm_get_valid_moe_configs);

}  // namespace flashinfer
