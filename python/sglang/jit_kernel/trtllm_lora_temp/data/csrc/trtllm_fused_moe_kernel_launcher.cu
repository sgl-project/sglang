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
#include <flashinfer/exception.h>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/GemmGatedActOptions.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "fused_activation_quant.cuh"
#include "fused_permute_quant.cuh"  // fused permute+nvfp4-quant (gate_up de-pad), used by bench_fused_permute_quant
#include "nv_internal/tensorrt_llm/kernels/quantization.h"
#include "nv_internal/tensorrt_llm/thop/utils.h"
#include "tvm_ffi_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <nvrtc.h>
#include <set>
#include <unordered_map>
#include <vector>

namespace flashinfer {

namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::Routing::RoutingMethodType;
using tvm::ffi::Array;
using tvm::ffi::Optional;

// Validate routing_replay_out tensor properties.
// NOTE: dim0 >= num_tokens is intentionally NOT checked — with CUDA graphs the buffer
// is pre-allocated at maximum batch size and reused across steps with varying num_tokens.
static void validate_routing_replay_out(TensorView const& replay, TensorView const& hidden_states, int64_t top_k) {
  TVM_FFI_ICHECK(replay.device().device_type == kDLCUDA) << "routing_replay_out must be a CUDA tensor";
  TVM_FFI_ICHECK(replay.device().device_id == hidden_states.device().device_id)
      << "routing_replay_out must be on the same device as hidden_states";
  TVM_FFI_ICHECK(replay.ndim() == 2) << "routing_replay_out must be 2D [num_tokens, top_k]";
  TVM_FFI_ICHECK(replay.size(1) == top_k) << "routing_replay_out dim1 must equal top_k";
  TVM_FFI_ICHECK((replay.dtype() == DLDataType{kDLInt, 16, 1})) << "routing_replay_out must be int16 dtype";
  TVM_FFI_ICHECK(replay.IsContiguous()) << "routing_replay_out must be contiguous (packed row-major)";
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

std::set<int32_t> computeSelectedTileN(
    std::vector<int32_t> const& supported_tile_nums,
    int64_t const num_tokens,
    int64_t const top_k,
    int64_t const num_local_experts) {
  TVM_FFI_ICHECK(!supported_tile_nums.empty()) << "supported_tile_nums must not be empty.";
  float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
  // NOTE: This differs from Python AutoTuner bucketing:
  // - AutoTuner maps raw num_tokens with last_positive_power_of_2 (round-down).
  // - Here we map derived avg_tokens_per_expert and use nextPowerOfTwo (round-up).
  // Because they round different quantities in different directions, cache bucket and runtime
  // tile candidates can diverge; launcher-side tactic resolution handles that mismatch.
  // assume supported_tile_nums is sorted
  int32_t tile_tokens_dim =
      std::clamp(nextPowerOfTwo(avg_tokens_per_expert), supported_tile_nums.front(), supported_tile_nums.back());
  auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);
  FLASHINFER_CHECK(
      it != supported_tile_nums.end(),
      "computeSelectedTileN expected exact tile ",
      tile_tokens_dim,
      " in supported_tile_nums (size=",
      supported_tile_nums.size(),
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

int64_t selectDefaultTileN(
    std::vector<int32_t> const& supported_tile_nums,
    int64_t const num_tokens,
    int64_t const top_k,
    int64_t const num_local_experts) {
  auto selected = computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
  TVM_FFI_ICHECK(!selected.empty()) << "No selected tile_N candidates for current MoE input.";
  return *selected.begin();
}

// Resolve the (tile_N, config) pair passed from Python side, applying fallback logic
// when tile_N is -1.
std::pair<int64_t, int64_t> resolveMoeTileAndConfig(
    Array<int64_t> const& config_index,
    std::vector<int32_t> const& supported_tile_nums,
    int64_t const num_tokens,
    int64_t const top_k,
    int64_t const num_local_experts) {
  // Python side convention: tactic is [tile_N, config]
  TVM_FFI_ICHECK(config_index.size() == 2)
      << "Invalid tactic, expected to be [tile_N, config], but got array of size " << config_index.size();
  const int64_t tile_N = config_index[0];
  const int64_t config = config_index[1];

  if (tile_N == -1 || config == -1) {
    // Use fallback tactic
    auto const default_tile_N = selectDefaultTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
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
  btg::Dtype mRoutingBiasDtype{btg::Dtype::Bfloat16};  // Dtype for expert weights in routing, based on routing bias
  btg::Dtype mRoutingLogitsDtype{btg::Dtype::Bfloat16};
  bool norm_topk_prob{true};
  ActivationType activation_type{ActivationType::Swiglu};
  btg::Dtype mDtypeScore{btg::Dtype::Bfloat16};

  // Optional routing replay output: [num_tokens, top_k] int16 tensor
  Optional<TensorView> routing_replay_out;

  int64_t intermediate_size_factor{2};

 public:
  // Constructor that initializes all TensorView members
  FusedMoeLauncher(
      const Optional<TensorView>& routing_logits,
      const Optional<TensorView>& routing_bias,
      const TensorView& hidden_states,
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
  void init_common(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      ActivationType activation_type,
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
      TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 || routing_logits.value().dtype() == dl_bfloat16)
          << "routing_logits must be float or bfloat16.";
    }
  }

  // Routing bias [num_experts]
  void check_routing_bias_shape() const {
    if (routing_bias.has_value()) {
      TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), args->num_experts) << "routing_bias has incorrect shape.";
    }
  }

  // Hidden states [num_tokens, hidden_size]
  void check_hidden_states_shape() const {
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    TVM_FFI_ICHECK_EQ(hidden_states.size(1), args->intermediate_size) << "hidden_states has incorrect shape.";
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
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported weight_layout: " << (int)weight_layout;
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
    TVM_FFI_ICHECK(args->top_k > 0 && args->top_k <= args->num_experts) << "top_k must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_num_experts > 0 && args->local_num_experts <= args->num_experts)
        << "local_num_experts must be between 1 and num_experts";
    TVM_FFI_ICHECK(
        args->local_expert_offset >= 0 && args->local_expert_offset + args->local_num_experts <= args->num_experts)
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
    int32_t max_num_padded_tokens = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());

    expanded_idx_to_permuted_idx = alloc_tensor({args->num_tokens * args->top_k}, dl_int32, hidden_states.device());

    permuted_idx_to_token_idx = alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());

    expert_indexes = alloc_tensor({args->num_tokens, args->top_k}, dl_int32, hidden_states.device());

    // expert_weights allocation should be done by derived class since data type could vary

    int64_t const size_of_expert_count_histogram = std::max(args->num_experts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor(
        {size_of_expert_count_histogram},
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
    workspace.expanded_idx_to_permuted_idx = static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
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
    bool usePerTokenScalingGemm2 = per_token_scales.has_value() && this->mDtypeAct != btg::Dtype::Bfloat16;
    // For FP8 block-scale (E4m3 activations, E4m3 weights) with DeepSeek FP8, use the
    // weights-only Runner constructor to match the original kernel path and numerics.
    if (this->mDtypeAct == btg::Dtype::E4m3 && this->mDtypeWeights == btg::Dtype::E4m3 && args->mUseDeepSeekFp8) {
      moe_runner = std::make_unique<RunnerType>(
          this->mDtypeWeights,
          args->mUseDeepSeekFp8,
          (int32_t)tile_tokens_dim,
          this->use_shuffled_weight,
          this->weight_layout,
          usePerTokenScalingGemm1,
          usePerTokenScalingGemm2,
          false,
          false);
    } else {
      moe_runner = std::make_unique<RunnerType>(
          this->mDtypeAct,
          this->mDtypeWeights,
          args->mUseDeepSeekFp8,
          (int32_t)tile_tokens_dim,
          this->activation_type,
          this->use_shuffled_weight,
          this->weight_layout,
          usePerTokenScalingGemm1,
          usePerTokenScalingGemm2);
    }

    if (moe_tactic == -1) {
      moe_tactic = moe_runner->getDefaultValidConfigIndex(
          args->top_k, args->hidden_size, args->intermediate_size, args->local_num_experts, args->num_tokens);
    }
    auto valid_cfgs = moe_runner->getValidConfigIndices(
        args->top_k, args->hidden_size, args->intermediate_size, args->local_num_experts, args->num_tokens);
    auto valid_it = std::find(valid_cfgs.begin(), valid_cfgs.end(), moe_tactic);
    FLASHINFER_CHECK(
        valid_it != valid_cfgs.end(),
        "Invalid MoE tactic ",
        moe_tactic,
        " for tile_N=",
        tile_tokens_dim,
        ". Number of valid tactics for this tile is ",
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
  virtual Array<Tensor>
  run(int64_t moe_tactic,
      bool enable_pdl = true,
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
        args->routing_logits,
        args->routing_bias,
        args->num_tokens,
        args->num_experts,
        args->top_k,
        args->n_group,
        args->topk_group,
        args->local_expert_offset,
        args->local_num_experts,
        args->routed_scaling_factor,
        workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()),
        workspace.expert_weights,
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()),
        args->mDtypeElt,
        mRoutingBiasDtype,
        use_routing_scales_on_input,
        use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type),
        routing_stream,
        mRoutingLogitsDtype,
        norm_topk_prob,
        replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic, enable_pdl);

    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }
};

void FusedMoeLauncher::init_common(
    std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
    int64_t tile_tokens_dim,
    int64_t routing_method_type,
    bool use_shuffled_weight,
    int64_t weight_layout,
    ActivationType activation_type,
    bool norm_topk_prob) {
  // Check devicearchitecture: Blackwell (SM 10.x) required
  auto device = hidden_states.device().device_id;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TVM_FFI_ICHECK(major == 10 || major == 12)
      << "MoE kernel requires SM 10.x or SM 12.x architecture. Current device has SM " << major << minor;
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
  TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2) << "the value of weight_layout is not recognized";
  this->weight_layout = static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout);
  this->activation_type = activation_type;
  this->intermediate_size_factor = isGatedActivation(activation_type) ? 2 : 1;
  this->norm_topk_prob = norm_topk_prob;
}

class Bf16MoeLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  Bf16MoeLauncher(
      Optional<TensorView> const& routing_logits,
      Optional<TensorView> const& routing_bias,
      TensorView const& expert_indices,
      TensorView const& expert_weights,
      TensorView const& hidden_states,
      TensorView const& gemm1_weights,
      TensorView const& gemm2_weights)
      : FusedMoeLauncher(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            Optional<TensorView>(),
            Optional<TensorView>(),
            gemm2_weights,
            Optional<TensorView>(),
            Optional<TensorView>()),
        expert_indices(expert_indices),
        expert_weights(expert_weights) {}

  void init(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      ActivationType activation_type,
      bool norm_topk_prob = true) {
    // Do base class init and perform common checks
    FusedMoeLauncher::init_common(
        std::move(args),
        tile_tokens_dim,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation_type,
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
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k) << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }

    // TODO n_group, topk_group validation?
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype = routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype = routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
    bool has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    if (has_precomputed_indices) {
      // Use expert_indices directly
      workspace.routing_expert_indexes = static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
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

    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0) << "the second dimension of weights must be a multiple of 128.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
    gemm1_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16, hidden_states.device());
    activation_output =
        alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16, hidden_states.device());
    gemm2_output = alloc_tensor({max_num_padded_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = activation_output.data_ptr();
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->output == nullptr) {
      output = alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;
  }

  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t num_local_experts,
      int64_t num_tokens,
      int64_t act_type,
      bool use_shuffled_weight,
      int64_t weight_layout) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16,  // dtype_act
          btg::Dtype::Bfloat16,  // dtype_weights
          false,                 // useDeepSeekFp8
          tile_N,
          static_cast<ActivationType>(act_type),
          use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));

      auto cfgs =
          moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);

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
  Fp8PerTensorLauncher(
      TensorView const& routing_logits,
      Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states,
      TensorView const& gemm1_weights,
      TensorView const& output1_scales_scalar,
      TensorView const& output1_scales_gate_scalar,
      TensorView const& gemm2_weights,
      TensorView const& output2_scales_scalar)
      : FusedMoeLauncher(
            Optional<TensorView>(routing_logits),
            routing_bias,
            hidden_states,
            gemm1_weights,
            Optional<TensorView>(output1_scales_scalar),
            Optional<TensorView>(output1_scales_gate_scalar),
            gemm2_weights,
            Optional<TensorView>(output2_scales_scalar),
            Optional<TensorView>()),
        use_routing_scales_on_input(false) {}

  void init(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      bool use_routing_scales_on_input_param,
      ActivationType activation_type,
      bool norm_topk_prob = true) {
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

    FusedMoeLauncher::init_common(
        std::move(args),
        tile_tokens_dim,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation_type,
        norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
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

    args->mDtypeOut = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype = routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype = routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    auto expert_weights_dtype = mRoutingLogitsDtype == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
    expert_weights = alloc_tensor({args->num_tokens, args->top_k}, expert_weights_dtype, hidden_states.device());

    workspace.expert_weights = expert_weights.data_ptr();
    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      workspace.token_scales = expert_weights.data_ptr();  // Consumed by permuteGemm1 kernel
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(output1_scales_scalar.has_value()) << "output1_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().dtype(), dl_float32) << "output1_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().ndim(), 1) << "output1_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value()) << "output1_scales_gate_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().dtype(), dl_float32)
        << "output1_scales_gate_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().ndim(), 1) << "output1_scales_gate_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_gate_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output2_scales_scalar.has_value()) << "output2_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().dtype(), dl_float32) << "output2_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().ndim(), 1) << "output2_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().size(0), args->local_num_experts)
        << "output2_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(
        hidden_states.dtype() == dl_float8_e4m3fn || hidden_states.dtype() == dl_float16 ||
        hidden_states.dtype() == dl_bfloat16)
        << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens_gemm1 = workspace.total_max_padded_tokens + args->num_experts;
    int32_t max_num_padded_tokens_gemm2 = workspace.total_max_padded_tokens;

    gemm1_output =
        alloc_tensor({max_num_padded_tokens_gemm1, 2 * args->intermediate_size}, dl_uint8, hidden_states.device());
    gemm1_output_scale = alloc_tensor(
        {2 * args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());

    activation_output =
        alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size}, dl_uint8, hidden_states.device());
    activation_output_scale =
        alloc_tensor({args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16, hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = activation_output.data_ptr();
    workspace.activation_output_scale = static_cast<float*>(activation_output_scale.data_ptr());
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->output == nullptr) {
      output = alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    // Set scale pointers
    TVM_FFI_ICHECK(output1_scales_scalar.has_value());
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value());
    TVM_FFI_ICHECK(output2_scales_scalar.has_value());

    args->output1_scales_scalar = static_cast<float*>(output1_scales_scalar.value().data_ptr());
    args->output1_scales_gate_scalar = static_cast<float*>(output1_scales_gate_scalar.value().data_ptr());
    args->output2_scales_scalar = static_cast<float*>(output2_scales_scalar.value().data_ptr());
  }

 private:
  bool use_routing_scales_on_input;
  Tensor gemm1_output_scale;
  Tensor activation_output_scale;

 public:
  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t num_local_experts,
      int64_t num_tokens,
      int64_t act_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      btg::Dtype dtype_act,
      btg::Dtype dtype_weights) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act,
          dtype_weights,
          false,  // useDeepSeekFp8
          tile_N,
          static_cast<ActivationType>(act_type),
          use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout),
          true,  // usePerTokenScalingGemm1. always true for per-tensor fp8 due to llama4 routing
          false,
          false,
          false);

      auto cfgs =
          moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);

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

  Fp8BlockScaleLauncher(
      Optional<TensorView> const& routing_logits,
      Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states,
      TensorView const& hidden_states_scale,
      TensorView const& gemm1_weights,
      TensorView const& gemm1_weights_scale,
      TensorView const& gemm2_weights,
      TensorView const& gemm2_weights_scale,
      TensorView const& expert_indices,
      TensorView const& expert_weights,
      Fp8QuantizationType quantization_type,
      Optional<TensorView> const& gate_up_lora_delta = Optional<TensorView>(),
      Optional<TensorView> const& activation_lora_input = Optional<TensorView>())
      : FusedMoeLauncher(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            Optional<TensorView>(),
            Optional<TensorView>(),
            gemm2_weights,
            Optional<TensorView>(),
            Optional<TensorView>()),
        hidden_states_scale(hidden_states_scale),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm2_weights_scale(gemm2_weights_scale),
        expert_indices(expert_indices),
        expert_weights(expert_weights),
        gate_up_lora_delta(gate_up_lora_delta),
        activation_lora_input(activation_lora_input),
        quantization_type(quantization_type) {}

  void init(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      ActivationType activation_type,
      bool norm_topk_prob = true) {
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

    FusedMoeLauncher::init_common(
        std::move(args),
        tile_tokens_dim,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation_type,
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
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k) << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }

    FusedMoeLauncher::check_routing_common();

    if (static_cast<RoutingMethodType>(routing_method_type) != RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group <= 1) << "Current routing kernel (no groups) only supports n_group <= 1";
      TVM_FFI_ICHECK(args->topk_group <= 1) << "Current routing kernel (no groups) only supports topk_group <= 1";
    }

    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group != 0) << "n_group should not be zero for DeepSeekV3 routing";
      TVM_FFI_ICHECK(args->topk_group != 0) << "if n_group is given, topk_group must be given";
      TVM_FFI_ICHECK_EQ(args->num_experts % args->n_group, 0) << "num_experts must be divisible by n_group";
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
      TVM_FFI_ICHECK_LE(args->topk_group, args->n_group) << "n_group must not be smaller than topk_group.";
      TVM_FFI_ICHECK_LT(args->top_k, (args->topk_group * args->num_experts / args->n_group))
          << "top_k must be less than total number of experts in selected groups";
    } else if (
        static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Renormalize ||
        static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::RenormalizeNaive ||
        static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::SigmoidRenorm ||
        static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Sigmoid) {
      TVM_FFI_ICHECK(args->top_k <= 32 && args->top_k > 0)
          << "Current routing kernel (no groups) only supports top_k<=32 && top_k>0.";
    } else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      TVM_FFI_ICHECK_EQ(args->top_k, 1) << "Current routing kernel (no groups, Llama4) only supports top_k=1.";
    }

    TVM_FFI_ICHECK_EQ(args->num_experts % 4, 0) << "Routing kernel expects that num_experts must be divisible by 4";
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
      workspace.routing_expert_indexes = static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    } else {
      // Use routing_logits directly
      args->routing_logits = static_cast<float*>(routing_logits.value().data_ptr());
    }
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype = routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype = routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

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
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_float32) << "hidden_states_scale must be float.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2) << "hidden_states_scale must be 2D.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), hidden_states.size(1) / 128)
          << "hidden_states_scale dim0 must match hidden_states dim1 / 128.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), args->num_tokens)
          << "hidden_states_scale dim1 must match num_tokens.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_CHECK(
          weight_layout == batchedGemm::gemm::MatrixLayout::MajorK, "weight_layout must be MajorK for MxFp8.");
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_uint8);
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "gemm1_weights must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "gemm2_weights must be fp8.";

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float32) << "gemm1_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3) << "gemm1_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), args->local_num_experts)
          << "gemm1_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0) << "intermediate_size must be a multiple of 128.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1), intermediate_size_factor * args->intermediate_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), args->hidden_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_uint8) << "gemm1_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float32) << "gemm2_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3) << "gemm2_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), args->local_num_experts)
          << "gemm2_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), args->hidden_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), args->intermediate_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_uint8) << "gemm2_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    check_weights_shape("gemm1");
    check_weights_shape("gemm2");

    if (gate_up_lora_delta.has_value()) {
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().dtype(), dl_bfloat16) << "gate_up_lora_delta must be bf16.";
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().ndim(), 3)
          << "gate_up_lora_delta must be [num_tokens, top_k, 2 * intermediate_size].";
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(0), args->num_tokens);
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(1), args->top_k);
      TVM_FFI_ICHECK_EQ(gate_up_lora_delta.value().size(2), args->intermediate_size * intermediate_size_factor);
    }
    if (activation_lora_input.has_value()) {
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().dtype(), dl_bfloat16) << "activation_lora_input must be bf16.";
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().ndim(), 3)
          << "activation_lora_input must be [num_tokens, top_k, intermediate_size].";
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(0), args->num_tokens);
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(1), args->top_k);
      TVM_FFI_ICHECK_EQ(activation_lora_input.value().size(2), args->intermediate_size);
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0) << "intermediate_size must be a multiple of 128.";
    }
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    // Calculate max_num_padded_tokens for gemm1 and gemm2 using maybeGetMinTokenCount
    int32_t max_num_padded_tokens_gemm1 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens, args->intermediate_size, btg::dtypeGetNumBits(args->mDtypeElt));
    int32_t max_num_padded_tokens_gemm2 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens, args->hidden_size, btg::dtypeGetNumBits(args->mDtypeOut));

    gemm1_output = alloc_tensor(
        {max_num_padded_tokens_gemm1, intermediate_size_factor * args->intermediate_size},
        dl_uint8,
        hidden_states.device());

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      gemm1_output_scale = alloc_tensor(
          {intermediate_size_factor * args->intermediate_size / 128, workspace.total_max_padded_tokens},
          dl_float32,
          hidden_states.device());
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      // MxFP8 fuses the activation so no need for intermediate_size_factor
      int64_t sf_size =
          tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1, args->intermediate_size / 32);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      activation_output =
          alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size}, dl_uint8, hidden_states.device());
      activation_output_scale = alloc_tensor(
          {args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());
    }

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16, hidden_states.device());

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
      output = alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    args->hidden_states_scale = static_cast<float*>(hidden_states_scale.data_ptr());
    args->gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale.data_ptr());
    args->gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale.data_ptr());
    args->gate_up_lora_delta = gate_up_lora_delta.has_value() ? gate_up_lora_delta.value().data_ptr() : nullptr;
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
  Array<Tensor>
  run(int64_t moe_tactic,
      bool enable_pdl = true,
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
        use_precomputed ? nullptr : args->routing_logits,
        args->routing_bias,
        args->num_tokens,
        args->num_experts,
        args->top_k,
        args->n_group,
        args->topk_group,
        args->local_expert_offset,
        args->local_num_experts,
        args->routed_scaling_factor,
        workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()),
        workspace.expert_weights,
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()),
        args->mDtypeElt,
        mRoutingBiasDtype,
        use_routing_scales_on_input,
        use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type),
        routing_stream,
        mRoutingLogitsDtype,
        norm_topk_prob,
        replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic, enable_pdl);

    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }

  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t num_local_experts,
      int64_t num_tokens,
      bool use_shuffled_weight,
      int64_t weight_layout,
      btg::Dtype dtype_act,
      btg::Dtype dtype_weights,
      Fp8QuantizationType quantization_type,
      int64_t act_type) {
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
        TVM_FFI_ICHECK(static_cast<int>(activation_type) == static_cast<int>(ActivationType::Swiglu))
            << "DeepSeekFp8 only supports ActivationType::Swiglu, got " << static_cast<int>(activation_type) << ".";
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_weights,
            true /* useDeepSeekFp8 */,
            tile_N,
            use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));
      } else {
        // Under current trtllm_get_valid_moe_configs() dispatch rules, this else-path is
        // reached only by FP8 block-scale MXFP8 (dtype_act=dtype_weights=MxE4m3).
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_act,                                              // dtypeAct
            dtype_weights,                                          // dtypeWeights
            quantization_type == Fp8QuantizationType::DeepSeekFp8,  // useDeepSeekFp8
            tile_N,
            activation_type,
            use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));
      }

      auto cfgs =
          moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);

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

  MxInt4BlockScaleLauncher(
      TensorView const& routing_logits,
      Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states,
      TensorView const& gemm1_weights,
      TensorView const& gemm1_weights_scale,
      Optional<TensorView> const& gemm1_alpha,
      Optional<TensorView> const& gemm1_beta,
      Optional<TensorView> const& gemm1_clamp_limit,
      TensorView const& gemm2_weights,
      TensorView const& gemm2_weights_scale)
      : FusedMoeLauncher(
            Optional<TensorView>(routing_logits),
            routing_bias,
            hidden_states,
            gemm1_weights,
            Optional<TensorView>(),
            Optional<TensorView>(),
            gemm2_weights,
            Optional<TensorView>(),
            Optional<TensorView>()),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm2_weights_scale(gemm2_weights_scale) {}

  void init(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool norm_topk_prob = true) {
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
        std::move(args),
        tile_tokens_dim,
        routing_method_type,
        /*use_shuffled_weight=*/true,
        static_cast<int64_t>(batchedGemm::gemm::MatrixLayout::BlockMajorK),
        ActivationType::Swiglu,
        norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = mDtypeAct;
    args->mUseDeepSeekFp8 = false;
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype = routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype = routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    auto expert_weights_dtype = mRoutingLogitsDtype == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
    expert_weights = alloc_tensor({args->num_tokens, args->top_k}, expert_weights_dtype, hidden_states.device());

    workspace.expert_weights = expert_weights.data_ptr();
  }

  void check_moe() const override {
    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::Bfloat16) << "Only Bfloat16 is supported by MxInt4 block scale MoE";

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_bfloat16) << "gemm1_weights_scale must be bf16.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_bfloat16) << "gemm2_weights_scale must be bf16.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale = nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_alpha = gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta = gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit =
        gemm1_clamp_limit.has_value() ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr()) : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->output1_scales_scalar = nullptr;
    args->output1_scales_gate_scalar = nullptr;
    args->output2_scales_scalar = nullptr;

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    max_num_padded_tokens_gemm1 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens, args->intermediate_size, btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens,
        args->hidden_size,
        btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden = args->intermediate_size;
    gemm1_output =
        alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_bfloat16, hidden_states.device());

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16, hidden_states.device());

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
  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k, int64_t hidden_size, int64_t intermediate_size, int64_t num_local_experts, int64_t num_tokens) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums = computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16,
          btg::Dtype::MxInt4,
          false,  // useDeepSeekFp8
          tile_N,
          ActivationType::Swiglu,
          /*useShuffledMatrix*/ true,
          batchedGemm::gemm::MatrixLayout::BlockMajorK);

      auto cfgs =
          moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);

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
      Optional<TensorView> const& routing_logits,
      Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states,
      Optional<TensorView> const& hidden_states_scale,
      TensorView const& gemm1_weights,
      TensorView const& gemm1_weights_scale,
      Optional<TensorView> const& gemm1_bias,
      Optional<TensorView> const& gemm1_alpha,
      Optional<TensorView> const& gemm1_beta,
      Optional<TensorView> const& gemm1_clamp_limit,
      TensorView const& gemm2_weights,
      TensorView const& gemm2_weights_scale,
      Optional<TensorView> const& gemm2_bias,
      Optional<TensorView> const& output1_scales_scalar,
      Optional<TensorView> const& output1_scales_gate_scalar,
      Optional<TensorView> const& output2_scales_scalar,
      Optional<TensorView> const& per_token_scales,
      TensorView const& expert_indices,
      TensorView const& expert_weights)
      : FusedMoeLauncher(
            routing_logits,
            routing_bias,
            hidden_states,
            gemm1_weights,
            output1_scales_scalar,
            output1_scales_gate_scalar,
            gemm2_weights,
            output2_scales_scalar,
            per_token_scales),
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

  void init(
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
      int64_t tile_tokens_dim,
      int64_t routing_method_type,
      bool use_shuffled_weight,
      int64_t weight_layout,
      ActivationType activation_type,
      btg::Dtype dtype_act,
      btg::Dtype dtype_weights,
      bool norm_topk_prob = true) {
    // Set data types
    args->mDtypeElt = dtype_act;
    args->mDtypeOut = btg::Dtype::Bfloat16;  // Output is always BF16 for FP4
    args->mUseDeepSeekFp8 = false;           // FP4 doesn't use DeepSeek FP8

    mDtypeAct = dtype_act;
    mDtypeWeights = dtype_weights;

    FusedMoeLauncher::init_common(
        std::move(args),
        tile_tokens_dim,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation_type,
        norm_topk_prob);
  }

  void check_routing() const override {
    // First call base class common routing checks
    FusedMoeLauncher::check_routing_common();
  }

  void prepare_routing() override {
    num_tokens_per_expert = alloc_tensor({args->num_experts}, dl_int32, hidden_states.device());
    int32_t max_num_padded_tokens = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());
    expanded_idx_to_permuted_idx = alloc_tensor({args->num_tokens * args->top_k}, dl_int32, hidden_states.device());
    permuted_idx_to_token_idx = alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());

    int64_t const size_of_expert_count_histogram = std::max(args->num_experts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor({size_of_expert_count_histogram}, dl_int32, hidden_states.device());

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);
    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());
    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());
    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states.device());

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.expanded_idx_to_permuted_idx = static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx.data_ptr());
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas.data_ptr());

    args->mDtypeElt = mDtypeAct;
    auto routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype = routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype = routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  }

  void check_moe() const override {
    TVM_FFI_ICHECK(
        mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::Bfloat16 || mDtypeAct == btg::Dtype::E4m3 ||
        mDtypeAct == btg::Dtype::MxE4m3)
        << "Only E2m1, Bfloat16, MxE4m3 and E4m3 are supported by Fp4 block scale MoE";

    if (mDtypeAct == btg::Dtype::E2m1) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::E2m1)
          << "Only E2m1 and MxE2m1 are supported by block scale MoE with E2m1 activation";
      TVM_FFI_ICHECK(hidden_states_scale.has_value()) << "hidden_states_scale is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_scalar.has_value()) << "output1_scales_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value()) << "output2_scales_scalar is required for E2m1 activation";
    } else if (mDtypeAct == btg::Dtype::Bfloat16 || mDtypeAct == btg::Dtype::E4m3 || mDtypeAct == btg::Dtype::MxE4m3) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::MxE2m1)
          << "Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or "
             "MxE4m3 activation";
    }

    if (mDtypeAct == btg::Dtype::E4m3) {
      TVM_FFI_ICHECK(output1_scales_scalar.has_value()) << "output1_scales_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value()) << "output2_scales_scalar is required for E4m3 activation";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float8_e4m3fn) << "gemm1_weights_scale must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float8_e4m3fn) << "gemm2_weights_scale must be fp8.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale = hidden_states_scale.has_value() ? hidden_states_scale.value().data_ptr() : nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_bias = gemm1_bias.has_value() ? static_cast<float*>(gemm1_bias.value().data_ptr()) : nullptr;
    args->gemm1_alpha = gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta = gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit =
        gemm1_clamp_limit.has_value() ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr()) : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->gemm2_bias = gemm2_bias.has_value() ? static_cast<float*>(gemm2_bias.value().data_ptr()) : nullptr;
    args->output1_scales_scalar =
        output1_scales_scalar.has_value() ? static_cast<float*>(output1_scales_scalar.value().data_ptr()) : nullptr;
    args->output1_scales_gate_scalar = output1_scales_gate_scalar.has_value()
                                           ? static_cast<float*>(output1_scales_gate_scalar.value().data_ptr())
                                           : nullptr;
    args->output2_scales_scalar =
        output2_scales_scalar.has_value() ? static_cast<float*>(output2_scales_scalar.value().data_ptr()) : nullptr;

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    auto const sf_vec_size = mDtypeWeights == btg::Dtype::MxE2m1 ? 32 : 16;

    max_num_padded_tokens_gemm1 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens, args->intermediate_size, btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 = tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
        workspace.total_max_padded_tokens,
        args->hidden_size,
        btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden =
        mDtypeAct == btg::Dtype::E2m1 ? args->intermediate_size / 2 : args->intermediate_size;
    if (mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::MxE4m3) {
      int64_t sf_size =
          tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1, args->intermediate_size / sf_vec_size);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }
    if (!per_token_scales.has_value()) {
      gemm1_output = alloc_tensor(
          {max_num_padded_tokens_gemm1, gemm1_output_hidden},
          mDtypeAct == btg::Dtype::Bfloat16 ? dl_bfloat16 : dl_uint8,
          hidden_states.device());
    } else {  // FC1 output is Bfloat16
      TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1)
          << "NvFP4 MoE: currently only support NvFP4 x NvFP4 when using per-token scaling.";
      // When per-token scales are used, the FC1 output is always BF16 and will be quantized
      gemm1_output =
          alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size}, dl_bfloat16, hidden_states.device());
      activation_output =
          alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_uint8, hidden_states.device());
      per_token_scales_fc2 = alloc_tensor({max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());
    }

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16, hidden_states.device());

    // Setup workspace pointers
    workspace.hidden_states_scale_linear = nullptr;  // FP4 doesn't use linear scale
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale =
        gemm1_output_scale.has_value() ? static_cast<float*>(gemm1_output_scale.value().data_ptr()) : nullptr;
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
  Array<Tensor>
  run(int64_t moe_tactic,
      bool enable_pdl = true,
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
        args->routing_logits,
        args->routing_bias,
        args->num_tokens,
        args->num_experts,
        args->top_k,
        args->n_group,
        args->topk_group,
        args->local_expert_offset,
        args->local_num_experts,
        args->routed_scaling_factor,
        static_cast<int*>(expert_indices.data_ptr()),
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        nullptr /*permuted_idx_to_expanded_idx.data_ptr()*/,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()),
        expert_weights.data_ptr(),
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()),
        args->mDtypeElt,
        mRoutingBiasDtype,
        use_routing_scales_on_input,
        use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type),
        routing_stream,
        mRoutingLogitsDtype,
        norm_topk_prob,
        replay_ptr);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic, enable_pdl);

    // Match original FP4 behavior for return values
    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, FusedMoeLauncher::expert_weights, expanded_idx_to_permuted_idx};
  }

  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t num_local_experts,
      int64_t num_tokens,
      int64_t act_type,
      btg::Dtype dtype_act,
      btg::Dtype dtype_weights,
      bool use_per_token_scaling) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes = getSupportedTileNums(dtype_act);
    std::set<int32_t> selected_tile_nums = computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act,
          dtype_weights,
          false,  // useDeepSeekFp8
          tile_N,
          static_cast<ActivationType>(act_type),
          /*useShuffledMatrix*/ true,
          /*weight_layout*/ batchedGemm::gemm::MatrixLayout::MajorK,
          // NOTE(siyuan): currently FP4 MoE always apply per-token scaling to both FC1 and FC2.
          /*usePerTokenScalingGemm1*/ use_per_token_scaling,
          /*usePerTokenScalingGemm2*/ use_per_token_scaling,
          false,
          false);

      auto cfgs =
          moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

Array<Tensor> trtllm_bf16_moe(
    Optional<TensorView> const& routing_logits,
    Optional<TensorView> const& routing_bias,
    TensorView const& expert_indices,
    TensorView const& expert_weights,
    TensorView const& hidden_states,
    TensorView const& gemm1_weights,
    TensorView const& gemm2_weights,
    TensorView output,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool use_shuffled_weight,
    int64_t weight_layout,
    bool do_finalize,
    bool enable_pdl,
    Array<int64_t> moe_tactic,
    int64_t activation_type,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  // Just some basic type validation first and leave more checks to the launcher
  if (routing_logits.has_value()) {
    TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 || routing_logits.value().dtype() == dl_bfloat16)
        << "BF16 MoE: routing_logits must be bfloat16 or float.";
  }
  TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_bfloat16) << "BF16 MoE: hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_bfloat16) << "BF16 MoE: gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_bfloat16) << "BF16 MoE: gemm2_weights must be bfloat16.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  auto const activation = validateAndCastActivationType(activation_type);

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(
      Bf16MoeLauncher::mSupportedTileNums.begin(), Bf16MoeLauncher::mSupportedTileNums.end());
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
    auto launcher = std::make_unique<Bf16MoeLauncher>(
        routing_logits, routing_bias, expert_indices, expert_weights, hidden_states, gemm1_weights, gemm2_weights);
    launcher->init(
        std::move(args),
        curr_tile_N,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation,
        norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(moe_tactic, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(), "Internal error: missing BF16 MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

Array<Tensor> trtllm_fp8_per_tensor_scale_moe(
    TensorView routing_logits,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    TensorView gemm1_weights,
    TensorView output1_scales_scalar,
    TensorView output1_scales_gate_scalar,
    TensorView gemm2_weights,
    TensorView output2_scales_scalar,
    TensorView output,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    bool use_routing_scales_on_input,
    int64_t routing_method_type,
    bool do_finalize,
    bool enable_pdl,
    Array<int64_t> config_index,
    int64_t activation_type,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  // Basic type validation
  auto dtype = hidden_states.dtype();
  auto activation = validateAndCastActivationType(activation_type);

  TVM_FFI_ICHECK(dtype == dl_float8_e4m3fn || dtype == dl_float16 || dtype == dl_bfloat16)
      << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar.dtype(), dl_float32) << "FP8 MoE: output1_scales_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_gate_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar.dtype(), dl_float32) << "FP8 MoE: output2_scales_scalar must be float32.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);

  // Use default values that match the original function behavior
  bool use_shuffled_weight = true;  // Original uses /*useShuffledMatrix*/ true
  int64_t weight_layout = 0;        // Default to MajorK

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(
      Fp8PerTensorLauncher::mSupportedTileNums.begin(), Fp8PerTensorLauncher::mSupportedTileNums.end());
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
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar);
    launcher->init(
        std::move(args),
        curr_tile_N,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        use_routing_scales_on_input,
        activation,
        norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(
      launcher_it != launchers_map.end(), "Internal error: missing FP8 per-tensor MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl, use_routing_scales_on_input);
}

Array<Tensor> trtllm_fp8_block_scale_moe_impl(
    Optional<TensorView> routing_logits,
    TensorView expert_indices,
    TensorView expert_weights,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    TensorView hidden_states_scale,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    TensorView output,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool use_shuffled_weight,
    int64_t weight_layout,
    bool do_finalize,
    bool enable_pdl,
    Array<int64_t> config_index,
    Fp8QuantizationType quantization_type,
    int64_t act_type,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out,
    Optional<TensorView> gate_up_lora_delta,
    Optional<TensorView> activation_lora_input,
    int64_t lora_ready_event = 0,
    int64_t gemm2_done_event = 0) {
  auto activation_type = validateAndCastActivationType(act_type);
  // DeepSeekFp8 currently uses a TRTLLM runner that hardwires Swiglu activation semantics.
  // Fail for any other activation to avoid silently running incorrect activation behavior.
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8 && activation_type != ActivationType::Swiglu) {
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
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
  }
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "FP8 block scale MoE: gemm1_weights must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "FP8 block scale MoE: gemm2_weights must be fp8.";
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
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
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
    // GEMM1-LoRA overlap: cudaEvent_t handle (recorded on the LoRA side stream) the runner
    // waits on right before activation; 0 = no wait (serial path).
    args->lora_ready_event = reinterpret_cast<void*>(lora_ready_event);
    // Down-LoRA/finalize overlap: cudaEvent_t handle the runner records right after GEMM2
    // (before finalize) so the LoRA side stream can overlap the down-proj LoRA with
    // finalize; 0 = no record (serial path).
    args->gemm2_done_event = reinterpret_cast<void*>(gemm2_done_event);

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Fp8BlockScaleLauncher>(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        expert_indices,
        expert_weights,
        quantization_type,
        gate_up_lora_delta,
        activation_lora_input);
    launcher->init(
        std::move(args),
        curr_tile_N,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        activation_type,
        norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, supported_tile_nums, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(
      launcher_it != launchers_map.end(), "Internal error: missing FP8 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher with DeepSeek FP8 enabled - it will create its own runner internally
  return selected_launcher->run(
      config,
      enable_pdl,
      false /* use_routing_scales_on_input */,
      quantization_type == Fp8QuantizationType::DeepSeekFp8 /* use_deep_seek_fp8 */);
}

Array<Tensor> trtllm_fp8_block_scale_moe(
    Optional<TensorView> routing_logits,
    TensorView expert_indices,
    TensorView expert_weights,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    TensorView hidden_states_scale,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    TensorView output,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool use_shuffled_weight,
    int64_t weight_layout,
    bool do_finalize,
    bool enable_pdl,
    Array<int64_t> config_index,
    Fp8QuantizationType quantization_type,
    int64_t act_type,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  return trtllm_fp8_block_scale_moe_impl(
      routing_logits,
      expert_indices,
      expert_weights,
      routing_bias,
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      gemm2_weights,
      gemm2_weights_scale,
      output,
      num_experts,
      top_k,
      n_group,
      topk_group,
      intermediate_size,
      local_expert_offset,
      local_num_experts,
      routed_scaling_factor,
      routing_method_type,
      use_shuffled_weight,
      weight_layout,
      do_finalize,
      enable_pdl,
      config_index,
      quantization_type,
      act_type,
      norm_topk_prob,
      routing_replay_out,
      Optional<TensorView>(),
      Optional<TensorView>());
}

Array<Tensor> sgl_trtllm_fp8_block_scale_moe_lora(
    Optional<TensorView> routing_logits,
    TensorView expert_indices,
    TensorView expert_weights,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    TensorView hidden_states_scale,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    TensorView output,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool use_shuffled_weight,
    int64_t weight_layout,
    bool do_finalize,
    bool enable_pdl,
    Array<int64_t> config_index,
    Fp8QuantizationType quantization_type,
    int64_t act_type,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out,
    TensorView gate_up_lora_delta,
    TensorView activation_lora_input,
    int64_t lora_ready_event,
    int64_t gemm2_done_event) {
  if (quantization_type != Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "sgl_trtllm_fp8_block_scale_moe_lora currently supports DeepSeekFp8 only.";
  }
  return trtllm_fp8_block_scale_moe_impl(
      routing_logits,
      expert_indices,
      expert_weights,
      routing_bias,
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      gemm2_weights,
      gemm2_weights_scale,
      output,
      num_experts,
      top_k,
      n_group,
      topk_group,
      intermediate_size,
      local_expert_offset,
      local_num_experts,
      routed_scaling_factor,
      routing_method_type,
      use_shuffled_weight,
      weight_layout,
      do_finalize,
      enable_pdl,
      config_index,
      quantization_type,
      act_type,
      norm_topk_prob,
      routing_replay_out,
      Optional<TensorView>(gate_up_lora_delta),
      Optional<TensorView>(activation_lora_input),
      lora_ready_event,
      gemm2_done_event);
}

__global__ void sgl_trtllm_fp8_block_scale_moe_lora_finalize_kernel(
    cutlass::bfloat16_t const* __restrict__ gemm2_output,
    cutlass::bfloat16_t const* __restrict__ expert_weights,
    int32_t const* __restrict__ expanded_idx_to_permuted_idx,
    cutlass::bfloat16_t const* __restrict__ down_lora_delta,
    cutlass::bfloat16_t* __restrict__ output,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_size,
    int64_t hidden_size_padded,
    float routed_scaling_factor) {
  for (int64_t token_idx = blockIdx.y; token_idx < num_tokens; token_idx += gridDim.y) {
    for (int64_t hidden_idx = threadIdx.x + blockDim.x * blockIdx.x; hidden_idx < hidden_size;
         hidden_idx += blockDim.x * gridDim.x) {
      float acc = 0.0f;
      float lora_acc = 0.0f;
      for (int64_t k = 0; k < top_k; ++k) {
        int64_t const expanded_idx = token_idx * top_k + k;
        int32_t const permuted_idx = expanded_idx_to_permuted_idx[expanded_idx];
        if (permuted_idx != -1) {
          float const expert_prob = static_cast<float>(expert_weights[token_idx * top_k + k]);
          acc += expert_prob * static_cast<float>(gemm2_output[permuted_idx * hidden_size_padded + hidden_idx]);
        }
        lora_acc += static_cast<float>(down_lora_delta[expanded_idx * hidden_size + hidden_idx]);
      }
      output[token_idx * hidden_size + hidden_idx] =
          static_cast<cutlass::bfloat16_t>(acc + routed_scaling_factor * lora_acc);
    }
  }
}

void sgl_trtllm_fp8_block_scale_moe_lora_finalize(
    TensorView gemm2_output,
    TensorView expert_weights,
    TensorView expanded_idx_to_permuted_idx,
    TensorView down_lora_delta,
    TensorView output,
    Optional<double> routed_scaling_factor) {
  TVM_FFI_ICHECK_EQ(gemm2_output.dtype(), dl_bfloat16) << "gemm2_output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(expert_weights.dtype(), dl_bfloat16) << "expert_weights must be bfloat16.";
  TVM_FFI_ICHECK((expanded_idx_to_permuted_idx.dtype() == DLDataType{kDLInt, 32, 1}))
      << "expanded_idx_to_permuted_idx must be int32.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.dtype(), dl_bfloat16) << "down_lora_delta must be bfloat16.";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16) << "output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_output.ndim(), 2) << "gemm2_output must be 2D.";
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.ndim(), 1) << "expanded_idx_to_permuted_idx must be 1D.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.ndim(), 3) << "down_lora_delta must be 3D.";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D.";
  TVM_FFI_ICHECK(gemm2_output.IsContiguous()) << "gemm2_output must be contiguous.";
  TVM_FFI_ICHECK(expert_weights.IsContiguous()) << "expert_weights must be contiguous.";
  TVM_FFI_ICHECK(expanded_idx_to_permuted_idx.IsContiguous()) << "expanded_idx_to_permuted_idx must be contiguous.";
  TVM_FFI_ICHECK(down_lora_delta.IsContiguous()) << "down_lora_delta must be contiguous.";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous.";

  int64_t const num_tokens = output.size(0);
  int64_t const hidden_size = output.size(1);
  int64_t const top_k = down_lora_delta.size(1);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens) << "expert_weights dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(expert_weights.size(1), top_k) << "expert_weights dim1 must equal top_k.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(0), num_tokens) << "down_lora_delta dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(2), hidden_size) << "down_lora_delta dim2 must equal hidden_size.";
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
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      num_tokens,
      top_k,
      hidden_size,
      gemm2_output.size(1),
      static_cast<float>(routed_scaling_factor.value_or(1.0)));
  auto err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

Array<Tensor> trtllm_fp4_block_scale_moe(
    Optional<TensorView> routing_logits,
    TensorView expert_indices,
    TensorView expert_weights,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    Optional<TensorView> hidden_states_scale,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    Optional<TensorView> gemm1_bias,
    Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta,
    Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    Optional<TensorView> gemm2_bias,
    Optional<TensorView> output1_scales_scalar,
    Optional<TensorView> output1_scales_gate_scalar,
    Optional<TensorView> output2_scales_scalar,
    Optional<TensorView> per_token_scales,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool do_finalize,
    bool enable_pdl,
    int64_t act_type,
    TensorView output,
    Array<int64_t> config_index,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  if (hidden_states.dtype() == dl_uint8) hidden_size *= 2;

  int64_t hidden_states_scale_vec_size = -1;
  if (hidden_states_scale.has_value()) {
    hidden_states_scale_vec_size =
        (static_cast<int64_t>(num_tokens) * hidden_size) / hidden_states_scale.value().numel();
  }
  int64_t intermediate_size_factor = isGatedActivation(static_cast<ActivationType>(act_type)) ? 2 : 1;
  int64_t logical_scale_count =
      static_cast<int64_t>(local_num_experts) * intermediate_size * intermediate_size_factor * hidden_size;
  int64_t weight_scale_vec_size_raw = logical_scale_count / gemm1_weights_scale.numel();

  // Snap to nearest valid sf_vec_size (16 or 32).
  // The raw value may be slightly smaller than the true vec_size because
  // block_scale_interleave pads scale columns to a multiple of 4, inflating numel().
  int64_t weight_scale_vec_size = weight_scale_vec_size_raw > 16 ? 32 : 16;

  // Round-trip validation: the unpadded scale count must not exceed actual numel
  // (padding only adds elements, never removes them).
  int64_t expected_unpadded = logical_scale_count / weight_scale_vec_size;
  TVM_FFI_ICHECK(gemm1_weights_scale.numel() >= expected_unpadded)
      << "weight scale tensor too small: numel=" << gemm1_weights_scale.numel() << " but expected at least "
      << expected_unpadded << " for sf_vec_size=" << weight_scale_vec_size;

  auto mDtypeWeights = weight_scale_vec_size == 16 ? btg::Dtype::E2m1 : btg::Dtype::MxE2m1;

  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 || routing_bias.value().dtype() == dl_float32)
        << "routing_bias must be bfloat16 or float.";

    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts) << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be fp4 packed in uint8.";
  TVM_FFI_ICHECK(
      hidden_states.dtype() == dl_uint8 || hidden_states.dtype() == dl_bfloat16 ||
      hidden_states.dtype() == dl_float8_e4m3fn)
      << "hidden_states must be bf16, fp8 or uint8 (packed fp4).";

  auto mDtypeAct = btg::Dtype::Bfloat16;
  if (hidden_states.dtype() == dl_uint8) {
    TVM_FFI_ICHECK(hidden_states_scale.has_value() && hidden_states_scale.value().dtype() == dl_float8_e4m3fn)
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
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        output2_scales_scalar,
        per_token_scales,
        expert_indices,
        expert_weights);
    launcher->init(
        std::move(args),
        curr_tile_N,
        routing_method_type,
        /*use_shuffled_weight=*/true,
        /*weight_layout=*/0,
        static_cast<ActivationType>(act_type),
        mDtypeAct,
        mDtypeWeights,
        norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(
      launcher_it != launchers_map.end(), "Internal error: missing FP4 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

// ===========================================================================
// NVFP4 MoE LoRA (decomposed / unfused-activation) — FP4 sibling of the FP8
// trtllm-lora op. The standard NVFP4 path fuses SwiGLU into GEMM1, which leaves
// no seam to inject the gate_up LoRA delta pre-activation. We therefore run the
// MoE as a hand-wired pipeline that mirrors what MoE::Runner::run does for the
// DeepSeek-FP8 + per-token-NvFP4 path, but with the gate_up projection executed
// as a raw (no-activation) grouped GEMM via Gemm2::Runner so the standalone,
// LoRA-aware activation kernel can run between the two GEMMs:
//
//   gather (permute bf16) -> NvFP4 quant -> gate_up GEMM (K=hidden, N=2*inter,
//   raw bf16 out) -> activation (adds gate_up_lora_delta pre-SwiGLU, writes
//   activation_lora_input) -> NvFP4 quant -> down GEMM (K=inter, N=hidden) ->
//   finalize.
//
// The hidden states are supplied as bf16 (path 3: the dispatch feeds bf16 and this op permutes
// then NvFP4-quantizes internally, with globalScaleInv = 1/448/6 + per-token scaling, matching
// SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION). No fp4-input dequant round-trip.
// ===========================================================================

// Decomposed NvFP4 MoE-LoRA launcher. Reuses FusedMoeLauncher's routing-phase
// workspace allocation/bookkeeping (via prepare_routing-style setup) but owns
// the MoE compute pipeline.
class FP4BlockScaleLoraLauncher {
 public:
  // Match the plain FP4 E2m1 path's tile ladder (FP4BlockScaleLauncher::getSupportedTileNums for
  // non-bf16 act). Large prefills (high avg tokens/expert) need 128/256; capping at 64 makes
  // selectDefaultTileN pick a tile too small for the Gemm2 cubin to have a valid config at that
  // token count -> "Failed to initialize the TMA descriptor / illegal memory access".
  static constexpr std::array<int32_t, 6> mBaseSupportedTileNums = {8, 16, 32, 64, 128, 256};

  static std::vector<int32_t> getSupportedTileNums() {
    return std::vector<int32_t>(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
  }

  FP4BlockScaleLoraLauncher(
      TensorView const& expert_indices,
      TensorView const& expert_weights,
      Optional<TensorView> const& routing_bias,
      TensorView const& hidden_states,
      Optional<TensorView> const& hidden_states_scale,
      TensorView const& gemm1_weights,
      TensorView const& gemm1_weights_scale,
      TensorView const& gemm2_weights,
      TensorView const& gemm2_weights_scale,
      Optional<TensorView> const& output1_scales_scalar,
      Optional<TensorView> const& output1_scales_gate_scalar,
      Optional<TensorView> const& output2_scales_scalar,
      TensorView const& gate_up_lora_delta,
      TensorView const& activation_lora_input,
      TensorView const& output,
      int64_t lora_ready_event,
      int64_t gemm2_done_event)
      : expert_indices_(expert_indices),
        expert_weights_(expert_weights),
        routing_bias_(routing_bias),
        hidden_states_(hidden_states),
        hidden_states_scale_(hidden_states_scale),
        gemm1_weights_(gemm1_weights),
        gemm1_weights_scale_(gemm1_weights_scale),
        gemm2_weights_(gemm2_weights),
        gemm2_weights_scale_(gemm2_weights_scale),
        output1_scales_scalar_(output1_scales_scalar),
        output1_scales_gate_scalar_(output1_scales_gate_scalar),
        output2_scales_scalar_(output2_scales_scalar),
        gate_up_lora_delta_(gate_up_lora_delta),
        activation_lora_input_(activation_lora_input),
        output_(output),
        lora_ready_event_(lora_ready_event),
        gemm2_done_event_(gemm2_done_event) {}

  // Returns {output} when do_finalize, else {gemm2_output, expert_weights,
  // expanded_idx_to_permuted_idx} for a downstream finalize kernel.
  Array<Tensor>
  run(int64_t num_experts,
      int64_t top_k,
      int64_t intermediate_size,
      int64_t local_expert_offset,
      int64_t local_num_experts,
      double routed_scaling_factor,
      int64_t routing_method_type,
      int64_t tile_tokens_dim,
      bool norm_topk_prob,
      bool do_finalize,
      bool enable_pdl,
      bool use_fused_permute_quant) {
    namespace moe_ns = tensorrt_llm::kernels::trtllmgen_moe;
    auto device = hidden_states_.device();
    int dev_id = device.device_id;
    cudaStream_t stream = get_stream(device);

    int64_t const num_tokens = hidden_states_.size(0);
    int64_t const hidden_size =
        hidden_states_.dtype() == dl_uint8 ? hidden_states_.size(1) * 2 : hidden_states_.size(1);
    int64_t const inter = intermediate_size;
    int64_t const gate_up_n = 2 * inter;  // gated SwiGLU

    // ---- 1) routing (precomputed packed topk) ----
    Tensor num_tokens_per_expert = alloc_tensor({num_experts}, dl_int32, device);
    int32_t max_num_padded_tokens =
        moe_ns::Routing::getMaxPermutedPaddedCount(num_tokens, top_k, num_experts, tile_tokens_dim);
    Tensor total_num_padded_tokens = alloc_tensor({1}, dl_int32, device);
    Tensor expanded_idx_to_permuted_idx = alloc_tensor({num_tokens * top_k}, dl_int32, device);
    Tensor permuted_idx_to_token_idx = alloc_tensor({max_num_padded_tokens}, dl_int32, device);
    int64_t const hist_size = std::max<int64_t>(num_experts * 2, 256 * 2);
    Tensor expert_count_histogram = alloc_tensor({hist_size}, dl_int32, device);
    int32_t max_num_ctas = moe_ns::Routing::getMaxNumCtasInBatchDim(num_tokens, top_k, num_experts, tile_tokens_dim);
    Tensor cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, device);
    Tensor cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, device);
    Tensor num_non_exiting_ctas = alloc_tensor({1}, dl_int32, device);

    auto routing_bias_dtype = routing_bias_.has_value() ? routing_bias_.value().dtype() : dl_bfloat16;
    btg::Dtype mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    // The wrapper passes an empty placeholder for expert_weights; the routing runner
    // writes the unpacked per-(token,k) weights here. Allocate it ourselves (mirrors
    // Fp8BlockScaleLauncher::prepare_routing when has_precomputed_weights is false).
    // If the caller did pass a real expert_weights tensor, copy it into the allocation
    // afterwards is unnecessary; we just compute into our own buffer for a clean Tensor
    // return type. The bf16 routing-weight values are identical either way.
    auto ew_dtype = mRoutingBiasDtype == btg::Dtype::Fp32 ? dl_float32 : dl_bfloat16;
    Tensor expert_weights_alloc = alloc_tensor({num_tokens, top_k}, ew_dtype, device);
    void* expert_weights_ptr = expert_weights_alloc.data_ptr();

    moe_ns::Routing::Runner routing_runner(tile_tokens_dim);
    routing_runner.run(
        /*routing_logits=*/nullptr,
        routing_bias_.has_value() ? routing_bias_.value().data_ptr() : nullptr,
        num_tokens,
        num_experts,
        top_k,
        /*n_group=*/0,
        /*topk_group=*/0,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        static_cast<int*>(const_cast<void*>(expert_indices_.data_ptr())),
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        /*permuted_idx_to_expanded_idx=*/nullptr,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()),
        expert_weights_ptr,
        static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()),
        btg::Dtype::Bfloat16,
        mRoutingBiasDtype,
        /*useRoutingScalesOnInput=*/false,
        /*useDeepSeekFp8=*/false,
        static_cast<RoutingMethodType>(routing_method_type),
        stream,
        btg::Dtype::Bfloat16,
        norm_topk_prob,
        /*routing_replay_out=*/nullptr);

    // ---- 2) hidden as bf16 (path 3: dispatch feeds bf16; the op quantizes internally) ----
    TVM_FFI_ICHECK(hidden_states_.dtype() == dl_bfloat16)
        << "fp4 LoRA (path 3) requires bf16 hidden_states; the dispatch feeds bf16 and the op "
           "permutes+NvFP4-quantizes internally (no python pre-quant / dequant round-trip).";
    void* hidden_bf16_ptr = hidden_states_.data_ptr();

    int64_t const tile = tile_tokens_dim;
    // gate_up GEMM act operand (permuted fp4 hidden + scales). Declared in run() scope because the
    // gate_up GEMM (step 5) consumes them; the LARGE [max_padded, hidden] bf16 gather buffer
    // (permuted_hidden_bf16, ~4 GB at a 32K-token prefill) lives only inside the block below, so it
    // frees right after the quant -- before the equally-large [max_padded, hidden] gemm2_output is
    // allocated (step 8), letting the caching allocator reuse its block (halves the op's peak).
    auto gu_sfLayout = tile >= 128 ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                                   : tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4;
    int64_t const hidden_sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens, hidden_size / 16);
    Tensor hidden_fp4 = alloc_tensor({max_num_padded_tokens, hidden_size / 2}, dl_uint8, device);
    Tensor hidden_fp4_sf = alloc_tensor({hidden_sf_size}, dl_uint8, device);
    Tensor hidden_per_token_sf = alloc_tensor({max_num_padded_tokens}, dl_float32, device);
    if (use_fused_permute_quant && tile < 128) {
      // Invariants the fused kernel relies on (review hardening): hidden must be a multiple of the
      // 16-wide PackedVec load, and top_k must fit the dedup per-token-scale write (threadIdx<topK
      // over BLOCK_SIZE=512 threads). Both always hold for the supported models; check loudly.
      TVM_FFI_ICHECK(hidden_size % 16 == 0)
          << "fused permute+quant requires hidden_size % 16 == 0, got " << hidden_size;
      TVM_FFI_ICHECK(top_k <= 512) << "fused permute+quant dedup requires top_k <= BLOCK_SIZE(512), got " << top_k;
      // ---- 3+4 FUSED ---- read UN-permuted hidden and scatter-write fp4 + swizzled block-sf +
      // per-token-sf to the permuted positions in one kernel: de-pads (only num_tokens*top_k rows)
      // and drops the bf16 permuted round-trip. Bitwise-identical to the plain permute->quant chain
      // (else branch) for the valid rows. Gated to tile<128 (SWIZZLED_8x4) — the validated decode
      // path; prefill (tile>=128, 128x4) keeps the plain chain.
      // Padding rows of hidden_fp4/_sf are left UNwritten (fused touches only valid rows). Safe by
      // the down-quant precedent: step-7 quant#2 likewise writes only valid rows (m=num_tokens*top_k
      // + the map) and step-8 Gemm2 consumes it fine — both GEMMs bound work by num_non_exiting_ctas
      // / total_num_padded_tokens / cta_idx_xy and never read padding rows.
      // dedup=true: the per-token-grid variant (quantize each token once, scatter) — fewer reads and
      // (with BLOCK_SIZE=512) faster than no-dedup at decode (3.71us vs 6.25us, bench).
      float const gu_globalScaleInv = 1.f / 448.f / 6.f;
      sgl_fused_permute_quant::invokeFusedPermuteNvfp4Quant<__nv_bfloat16>(
          num_tokens,
          top_k,
          hidden_size,
          max_num_padded_tokens,
          reinterpret_cast<__nv_bfloat16 const*>(hidden_bf16_ptr),
          gu_globalScaleInv,
          static_cast<int32_t const*>(expanded_idx_to_permuted_idx.data_ptr()),
          reinterpret_cast<uint8_t*>(hidden_fp4.data_ptr()),
          reinterpret_cast<uint8_t*>(hidden_fp4_sf.data_ptr()),
          reinterpret_cast<float*>(hidden_per_token_sf.data_ptr()),
          gu_sfLayout,
          /*dedup=*/true,
          stream);
    } else {
      // ---- 3) permute (gather) bf16 hidden -> [max_padded, hidden] (transient, freed at block end)
      Tensor permuted_hidden_bf16 = alloc_tensor({max_num_padded_tokens, hidden_size}, dl_bfloat16, device);
      // Padded (unused) rows of permuted_hidden_bf16 are intentionally left UNINITIALIZED: the prior
      // DEFENSIVE zero-init memset is always skipped now (env gate removed — proven safe). Padding rows
      // never reach a valid output (moe::dev::finalize gathers only real tokens via the index map and
      // skips permutedIdx==-1; the NvFP4 per-token scale is strictly per-row; the grouped GEMMs are
      // per-row), so skipping it reclaims the ~max_padded*hidden*2B memset (a prefill/TTFT win).
      {
        moe::dev::permute::Data permData;
        permData.mDtypeElt = btg::Dtype::Bfloat16;
        permData.mUsePdl = false;
        permData.mUseDeepSeekFp8 = false;
        permData.inPtr = hidden_bf16_ptr;
        permData.outPtr = permuted_hidden_bf16.data_ptr();
        permData.inDqSfsPtr = nullptr;
        permData.outDqSfsPtr = nullptr;
        permData.expandedIdxToPermutedIdx = static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
        permData.hiddenDim = hidden_size;
        permData.numTokens = num_tokens;
        permData.topK = top_k;
        permData.totalNumPaddedTokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
        moe::dev::permute::run(permData, stream);
      }

      // ---- 4) NvFP4 quant of permuted bf16 hidden -> permuted fp4 + swizzled sf + per-token sf ----
      float const gu_globalScaleInv = 1.f / 448.f / 6.f;
      // input is already permuted, so map=nullptr and m=max_padded (process all rows; padding=0).
      tensorrt_llm::kernels::invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
          max_num_padded_tokens,
          hidden_size,
          reinterpret_cast<__nv_bfloat16 const*>(permuted_hidden_bf16.data_ptr()),
          gu_globalScaleInv,
          /*expanded_idx_to_permuted_idx=*/nullptr,
          reinterpret_cast<uint8_t*>(hidden_fp4.data_ptr()),
          reinterpret_cast<uint8_t*>(hidden_fp4_sf.data_ptr()),
          reinterpret_cast<float*>(hidden_per_token_sf.data_ptr()),
          gu_sfLayout,
          stream);
    }  // permuted_hidden_bf16 frees here -> its ~4 GB block is reused by gemm2_output (step 8).

    // ---- 5) gate_up GEMM: raw Gemm2::Runner(E2m1,E2m1,bf16, K=hidden, N=2*inter). ----
    // Per-token (hidden) scale on the act operand + output1_scales_gate_scalar (g1_alphas) on
    // the result reconstructs the TRUE gate_up projection (no NvFP4 unfused-act GEMM1 cubin
    // exists, so we cannot reuse the plain PermuteGemm1 path here). In per-token mode g1_alphas
    // applied to BOTH halves == the plain fused path's gate(g1_alphas)/up(g1_scale_c=g1_alphas).
    // Output columns are INTERLEAVED (the w13 is in the gated layout); de-interleaved in 5b.
    Tensor gate_up_bf16 = alloc_tensor({max_num_padded_tokens, gate_up_n}, dl_bfloat16, device);
    {
      moe_ns::Gemm2::Runner gemm_gate_up(
          btg::Dtype::E2m1,
          btg::Dtype::E2m1,
          btg::Dtype::Bfloat16,
          /*useDeepSeekFp8=*/false,
          (int)tile,
          /*useShuffledMatrix=*/true,
          batchedGemm::gemm::MatrixLayout::MajorK,
          /*usePerTokenScaling=*/true,
          /*usePerChannelScaling=*/false);
      int64_t cfg =
          gemm_gate_up.getDefaultValidConfigIndex(top_k, hidden_size, gate_up_n, local_num_experts, num_tokens);
      size_t ws =
          gemm_gate_up.getWorkspaceSizeInBytes(top_k, hidden_size, gate_up_n, local_num_experts, num_tokens, cfg);
      Tensor gemm_ws = alloc_tensor({(int64_t)ws}, dl_int8, device);
      // Gemm2::Runner semantics: hiddenSize is the OUTPUT N dim, intermediateSize is the K dim.
      gemm_gate_up.run(
          hidden_fp4.data_ptr(),
          hidden_fp4_sf.data_ptr(),
          gemm1_weights_.data_ptr(),
          gemm1_weights_scale_.data_ptr(),
          /*perTokenScales=*/hidden_per_token_sf.data_ptr(),
          /*perChannelScales=*/nullptr,
          output1_scales_gate_scalar_.has_value() ? static_cast<float*>(output1_scales_gate_scalar_.value().data_ptr())
                                                  : nullptr,
          /*ptrBias=*/nullptr,
          gate_up_bf16.data_ptr(),
          /*outputScale=*/nullptr,
          top_k,
          /*hiddenSize(N)=*/gate_up_n,
          /*intermediateSize(K)=*/hidden_size,
          local_num_experts,
          num_tokens,
          static_cast<int*>(num_non_exiting_ctas.data_ptr()),
          static_cast<int*>(total_num_padded_tokens.data_ptr()),
          static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
          static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
          gemm_ws.data_ptr(),
          dev_id,
          stream,
          (int)cfg,
          enable_pdl);
    }

    // The gated w13 weight makes GEMM1 emit gate_up columns pairwise-interleaved as
    // (g0,u0,g1,u1,...) instead of the contiguous [gate | up] layout. Rather than run a
    // standalone de-interleave kernel into a [max_num_padded_tokens, gate_up_n] scratch
    // buffer, the activation kernel below de-interleaves on read (interleavedGateUpInput),
    // fusing away that kernel + its HBM round-trip + the scratch allocation.

    // GEMM1-LoRA overlap: wait the side-stream LoRA event that produced the
    // gate_up_lora_delta consumed by the activation below, so the gate_up GEMM1
    // above overlaps the side-stream LoRA shrink/expand. No-op (0)
    // on the single-stream path.
    if (lora_ready_event_ != 0) {
      cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(lora_ready_event_), 0);
    }

    // ---- 6+7) activation (SwiGLU + LoRA) then NvFP4 per-token quant of the result ----
    // Three modes, selected by env (read once per process; the JIT kernel has no Python->C++
    // config channel). FUSE has priority over VEC:
    //   SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1 -> single fused kernel, activated_bf16 never
    //                                                   materialized to HBM (~1.5x over the pair).
    //   else SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=1   -> vectorized activationKernelOpt + quant#2.
    //   else (default)                                -> scalar activationKernel + quant#2.
    auto sfLayout = tile >= 128 ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                                : tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4;
    float const globalScaleInv = 1.f / 448.f / 6.f;
    int64_t const act_sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens, inter / 16);
    Tensor act_fp4 = alloc_tensor({max_num_padded_tokens, inter / 2}, dl_uint8, device);
    Tensor act_fp4_sf = alloc_tensor({act_sf_size}, dl_uint8, device);
    Tensor act_per_token_sf = alloc_tensor({max_num_padded_tokens}, dl_float32, device);

    auto envFlag = [](char const* name) {
      char const* e = std::getenv(name);
      return e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
    };
    static int const fuseActQuant = envFlag("SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE") ? 1 : 0;
    static int const actOptMode = envFlag("SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC") ? 1 : 0;

    if (fuseActQuant) {
      // Fused: gate_up (interleaved) + lora_delta -> act_fp4/sf/per_token + activation_lora_input,
      // without materializing activated_bf16. inter must be a multiple of 16 (always true here).
      flashinfer::sgl_fused_act_quant::launchFusedActivationQuant(
          num_tokens * top_k,
          inter,
          gate_up_n,
          reinterpret_cast<__nv_bfloat16 const*>(gate_up_bf16.data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(gate_up_lora_delta_.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(activation_lora_input_.data_ptr()),
          static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
          globalScaleInv,
          reinterpret_cast<uint8_t*>(act_fp4.data_ptr()),
          reinterpret_cast<uint8_t*>(act_fp4_sf.data_ptr()),
          reinterpret_cast<float*>(act_per_token_sf.data_ptr()),
          sfLayout,
          /*disableFp4FastMath=*/false,
          stream);
    } else {
      Tensor activated_bf16 = alloc_tensor({max_num_padded_tokens, inter}, dl_bfloat16, device);
      {
        moe::dev::activation::Data actData;
        actData.mDtypeElt = btg::Dtype::Bfloat16;
        actData.mUsePdl = false;
        actData.mUseDeepSeekFp8 = false;
        actData.inPtr = gate_up_bf16.data_ptr();
        actData.interleavedGateUpInput = true;
        actData.outPtr = activated_bf16.data_ptr();
        actData.inDqSfsPtr = nullptr;
        actData.outDqSfsPtr = nullptr;
        actData.gateUpLoraDeltaPtr = static_cast<cutlass::bfloat16_t const*>(gate_up_lora_delta_.data_ptr());
        actData.activationLoraInputOutPtr = static_cast<cutlass::bfloat16_t*>(activation_lora_input_.data_ptr());
        actData.innerDim = gate_up_n;
        actData.numTokens = num_tokens;
        actData.topK = top_k;
        actData.expandedIdxToPermutedIdx = static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
        actData.totalNumPaddedTokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
        actData.actOptMode = actOptMode;
        moe::dev::activation::run(actData, stream);
      }
      // quant#2: m = num_tokens*top_k + the expanded->permuted map so only valid (non-padding)
      // permuted rows are quantized (padding rows of activated_bf16 are left uninitialized).
      tensorrt_llm::kernels::invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
          num_tokens * top_k,
          inter,
          reinterpret_cast<__nv_bfloat16 const*>(activated_bf16.data_ptr()),
          globalScaleInv,
          static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
          reinterpret_cast<uint8_t*>(act_fp4.data_ptr()),
          reinterpret_cast<uint8_t*>(act_fp4_sf.data_ptr()),
          reinterpret_cast<float*>(act_per_token_sf.data_ptr()),
          sfLayout,
          stream);
    }

    // ---- 8) down GEMM: Gemm2::Runner(E2m1,E2m1,bf16, K=inter, N=hidden) ----
    // We run in per-token-activation mode (SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION), where
    // w13/w2 input_scale == 1 so g1_scale_c == g1_alphas and g2_alphas == w2_weight_scale_2.
    // The gate_up GEMM (step 5) applied g1_alphas to BOTH gate and up halves, which — with
    // input_scale==1 — equals the plain fused path's separate gate(g1_alphas)/up(g1_scale_c)
    // scaling. The standalone SwiGLU therefore produces the TRUE silu(gate)*up (same space as
    // the proven DeepSeek-FP8 unfused path), so the down GEMM applies output2_scales_scalar
    // (g2_alphas) DIRECTLY — exactly like the plain path's GEMM2 (runner.cu mGemm2.run).
    Tensor gemm2_output = alloc_tensor({max_num_padded_tokens, hidden_size}, dl_bfloat16, device);
    {
      moe_ns::Gemm2::Runner gemm_down(
          btg::Dtype::E2m1,
          btg::Dtype::E2m1,
          btg::Dtype::Bfloat16,
          /*useDeepSeekFp8=*/false,
          (int)tile,
          /*useShuffledMatrix=*/true,
          batchedGemm::gemm::MatrixLayout::MajorK,
          /*usePerTokenScaling=*/true,
          /*usePerChannelScaling=*/false);
      int64_t cfg = gemm_down.getDefaultValidConfigIndex(top_k, hidden_size, inter, local_num_experts, num_tokens);
      size_t ws = gemm_down.getWorkspaceSizeInBytes(top_k, hidden_size, inter, local_num_experts, num_tokens, cfg);
      Tensor gemm_ws = alloc_tensor({(int64_t)ws}, dl_int8, device);
      float* down_scale_ptr =
          output2_scales_scalar_.has_value() ? static_cast<float*>(output2_scales_scalar_.value().data_ptr()) : nullptr;
      gemm_down.run(
          act_fp4.data_ptr(),
          act_fp4_sf.data_ptr(),
          gemm2_weights_.data_ptr(),
          gemm2_weights_scale_.data_ptr(),
          /*perTokenScales=*/act_per_token_sf.data_ptr(),
          /*perChannelScales=*/nullptr,
          down_scale_ptr,
          /*ptrBias=*/nullptr,
          gemm2_output.data_ptr(),
          /*outputScale=*/nullptr,
          top_k,
          /*hiddenSize(N)=*/hidden_size,
          /*intermediateSize(K)=*/inter,
          local_num_experts,
          num_tokens,
          static_cast<int*>(num_non_exiting_ctas.data_ptr()),
          static_cast<int*>(total_num_padded_tokens.data_ptr()),
          static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
          static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
          gemm_ws.data_ptr(),
          dev_id,
          stream,
          (int)cfg,
          enable_pdl);
    }

    // Down-LoRA/finalize overlap: signal "base down GEMM done" so the LoRA side stream can
    // start the down-proj LoRA shrink/expand concurrent with the finalize below. 0 = no-op.
    if (gemm2_done_event_ != 0) {
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(gemm2_done_event_), stream);
    }

    if (!do_finalize) {
      return {gemm2_output, expert_weights_alloc, expanded_idx_to_permuted_idx};
    }

    // ---- 9) finalize (combine by expert weight) -> output [num_tokens, hidden] ----
    {
      moe::dev::finalize::Data finData;
      finData.mDtypeElt = btg::Dtype::Bfloat16;
      finData.mDtypeExpW = mRoutingBiasDtype;
      finData.mUsePdl = false;
      finData.mUseDeepSeekFp8 = false;
      finData.inPtr = gemm2_output.data_ptr();
      finData.outPtr = output_.data_ptr();
      finData.inDqSfsPtr = nullptr;
      finData.outDqSfsPtr = nullptr;
      finData.expertWeightsPtr = expert_weights_ptr;
      finData.expandedIdxToPermutedIdx = static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
      finData.numTokens = num_tokens;
      finData.numExperts = num_experts;
      finData.topK = top_k;
      finData.hiddenDim = hidden_size;
      finData.hiddenDimPadded = hidden_size;
      finData.totalNumPaddedTokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
      moe::dev::finalize::run(finData, stream);
    }
    sync_check_cuda_error(stream);
    // do_finalize=True: result written into output_ (the wrapper returns output_
    // directly and ignores this value).
    return Array<Tensor>();
  }

 private:
  TensorView expert_indices_;
  TensorView expert_weights_;
  Optional<TensorView> routing_bias_;
  TensorView hidden_states_;
  Optional<TensorView> hidden_states_scale_;
  TensorView gemm1_weights_;
  TensorView gemm1_weights_scale_;
  TensorView gemm2_weights_;
  TensorView gemm2_weights_scale_;
  Optional<TensorView> output1_scales_scalar_;
  Optional<TensorView> output1_scales_gate_scalar_;
  Optional<TensorView> output2_scales_scalar_;
  TensorView gate_up_lora_delta_;
  TensorView activation_lora_input_;
  TensorView output_;
  int64_t lora_ready_event_ = 0;
  int64_t gemm2_done_event_ = 0;
};

Array<Tensor> sgl_trtllm_fp4_block_scale_moe_lora(
    Optional<TensorView> routing_logits,
    TensorView expert_indices,
    TensorView expert_weights,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    Optional<TensorView> hidden_states_scale,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    Optional<TensorView> gemm1_bias,
    Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta,
    Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    Optional<TensorView> gemm2_bias,
    Optional<TensorView> output1_scales_scalar,
    Optional<TensorView> output1_scales_gate_scalar,
    Optional<TensorView> output2_scales_scalar,
    Optional<TensorView> per_token_scales,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool do_finalize,
    bool enable_pdl,
    int64_t act_type,
    TensorView output,
    Array<int64_t> config_index,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out,
    TensorView gate_up_lora_delta,
    TensorView activation_lora_input,
    int64_t lora_ready_event,
    bool use_fused_permute_quant,
    int64_t gemm2_done_event) {
  auto activation_type = validateAndCastActivationType(act_type);
  TVM_FFI_ICHECK(isGatedActivation(activation_type))
      << "sgl_trtllm_fp4_block_scale_moe_lora currently supports gated (SwiGLU) activation only.";
  (void)routing_logits;
  (void)gemm1_bias;
  (void)gemm1_alpha;
  (void)gemm1_beta;
  (void)gemm1_clamp_limit;
  (void)gemm2_bias;
  (void)per_token_scales;
  (void)n_group;
  (void)topk_group;
  (void)routing_replay_out;

  // Precomputed routing is required (packed topk_ids).
  TVM_FFI_ICHECK(expert_indices.ndim() == 2 && expert_indices.size(0) == hidden_states.size(0))
      << "fp4 LoRA requires precomputed packed expert_indices [num_tokens, top_k].";
  TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "fp4 LoRA: weights must be packed fp4 uint8.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float8_e4m3fn) << "fp4 LoRA: gemm1_weights_scale must be fp8 e4m3.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float8_e4m3fn) << "fp4 LoRA: gemm2_weights_scale must be fp8 e4m3.";
  TVM_FFI_ICHECK_EQ(gate_up_lora_delta.dtype(), dl_bfloat16) << "gate_up_lora_delta must be bf16.";
  TVM_FFI_ICHECK_EQ(gate_up_lora_delta.ndim(), 3)
      << "gate_up_lora_delta must be [num_tokens, top_k, 2*intermediate_size].";
  TVM_FFI_ICHECK_EQ(gate_up_lora_delta.size(2), 2 * intermediate_size);
  TVM_FFI_ICHECK_EQ(activation_lora_input.dtype(), dl_bfloat16) << "activation_lora_input must be bf16.";
  TVM_FFI_ICHECK_EQ(activation_lora_input.ndim(), 3)
      << "activation_lora_input must be [num_tokens, top_k, intermediate_size].";
  TVM_FFI_ICHECK_EQ(activation_lora_input.size(2), intermediate_size);
  TVM_FFI_ICHECK(gate_up_lora_delta.IsContiguous() && activation_lora_input.IsContiguous())
      << "lora bridge buffers must be contiguous.";

  int64_t const num_tokens = hidden_states.size(0);
  int64_t const tile_tokens_dim =
      selectDefaultTileN(FP4BlockScaleLoraLauncher::getSupportedTileNums(), num_tokens, top_k, local_num_experts);

  FP4BlockScaleLoraLauncher launcher(
      expert_indices,
      expert_weights,
      routing_bias,
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      gemm2_weights,
      gemm2_weights_scale,
      output1_scales_scalar,
      output1_scales_gate_scalar,
      output2_scales_scalar,
      gate_up_lora_delta,
      activation_lora_input,
      output,
      lora_ready_event,
      gemm2_done_event);
  return launcher.run(
      num_experts,
      top_k,
      intermediate_size,
      local_expert_offset,
      local_num_experts,
      routed_scaling_factor.value_or(1.0),
      routing_method_type,
      tile_tokens_dim,
      norm_topk_prob,
      do_finalize,
      enable_pdl,
      use_fused_permute_quant);
}

// bf16 combine + down-lora-delta merge — NvFP4 analog of the FP8 finalize op.
// Logic is dtype-agnostic bf16, so this mirrors
// sgl_trtllm_fp8_block_scale_moe_lora_finalize verbatim.
void sgl_trtllm_fp4_block_scale_moe_lora_finalize(
    TensorView gemm2_output,
    TensorView expert_weights,
    TensorView expanded_idx_to_permuted_idx,
    TensorView down_lora_delta,
    TensorView output,
    Optional<double> routed_scaling_factor) {
  TVM_FFI_ICHECK_EQ(gemm2_output.dtype(), dl_bfloat16) << "gemm2_output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(expert_weights.dtype(), dl_bfloat16) << "expert_weights must be bfloat16.";
  TVM_FFI_ICHECK((expanded_idx_to_permuted_idx.dtype() == DLDataType{kDLInt, 32, 1}))
      << "expanded_idx_to_permuted_idx must be int32.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.dtype(), dl_bfloat16) << "down_lora_delta must be bfloat16.";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16) << "output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_output.ndim(), 2) << "gemm2_output must be 2D.";
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.ndim(), 1) << "expanded_idx_to_permuted_idx must be 1D.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.ndim(), 3) << "down_lora_delta must be 3D.";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D.";
  TVM_FFI_ICHECK(gemm2_output.IsContiguous()) << "gemm2_output must be contiguous.";
  TVM_FFI_ICHECK(expert_weights.IsContiguous()) << "expert_weights must be contiguous.";
  TVM_FFI_ICHECK(expanded_idx_to_permuted_idx.IsContiguous()) << "expanded_idx_to_permuted_idx must be contiguous.";
  TVM_FFI_ICHECK(down_lora_delta.IsContiguous()) << "down_lora_delta must be contiguous.";
  TVM_FFI_ICHECK(output.IsContiguous()) << "output must be contiguous.";

  int64_t const num_tokens = output.size(0);
  int64_t const hidden_size = output.size(1);
  int64_t const top_k = down_lora_delta.size(1);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens) << "expert_weights dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(expert_weights.size(1), top_k) << "expert_weights dim1 must equal top_k.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(0), num_tokens) << "down_lora_delta dim0 must equal num_tokens.";
  TVM_FFI_ICHECK_EQ(down_lora_delta.size(2), hidden_size) << "down_lora_delta dim2 must equal hidden_size.";
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
      static_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      num_tokens,
      top_k,
      hidden_size,
      gemm2_output.size(1),
      static_cast<float>(routed_scaling_factor.value_or(1.0)));
  auto err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

Array<Tensor> trtllm_mxint4_block_scale_moe(
    TensorView routing_logits,
    Optional<TensorView> routing_bias,
    TensorView hidden_states,
    TensorView gemm1_weights,
    TensorView gemm1_weights_scale,
    Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta,
    Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights,
    TensorView gemm2_weights_scale,
    int64_t num_experts,
    int64_t top_k,
    Optional<int64_t> n_group,
    Optional<int64_t> topk_group,
    int64_t intermediate_size,
    int64_t local_expert_offset,
    int64_t local_num_experts,
    Optional<double> routed_scaling_factor,
    int64_t routing_method_type,
    bool do_finalize,
    bool enable_pdl,
    TensorView output,
    Array<int64_t> config_index,
    bool norm_topk_prob,
    Optional<TensorView> routing_replay_out) {
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  // Just some basic type validation first and leave more checks to the launcher

  int weight_scale_vec_size = (local_num_experts * intermediate_size * 2 * hidden_size) / gemm1_weights_scale.numel();

  TVM_FFI_ICHECK(weight_scale_vec_size == 32) << "unsupported weight_scale_vec_size.";

  TVM_FFI_ICHECK(routing_logits.dtype() == dl_float32 || routing_logits.dtype() == dl_bfloat16)
      << "routing_logits must be float or bfloat16.";
  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2) << "routing_logits must be 2D.";
  TVM_FFI_ICHECK_EQ(routing_logits.size(1), num_experts) << "routing_logits has incorrect shape.";
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16) << "routing_bias must be bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts) << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be int4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_bfloat16) << "hidden_states must be bf16.";

  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN(
      MxInt4BlockScaleLauncher::mSupportedTileNums.begin(), MxInt4BlockScaleLauncher::mSupportedTileNums.end());
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
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(
      launcher_it != launchers_map.end(),
      "Internal error: missing MXINT4 block-scale MoE launcher for tile_N=",
      tile_N);
  auto& selected_launcher = launcher_it->second;

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl);
}

Array<Array<int64_t>> trtllm_get_valid_moe_configs(
    int64_t const dtype_act_,
    int64_t const dtype_weights_,
    Fp8QuantizationType fp8_quantization_type,
    int64_t const top_k,
    int64_t const hidden_size,
    int64_t const intermediate_size,
    int64_t const num_local_experts,
    int64_t const act_type,
    bool const use_shuffled_weight,
    int64_t const weight_layout,
    bool const use_per_token_scaling,
    int64_t const num_tokens) {
  auto activation_type = validateAndCastActivationType(act_type);
  auto dtype_act = static_cast<btg::Dtype>(dtype_act_);
  auto dtype_weights = static_cast<btg::Dtype>(dtype_weights_);

  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::MxInt4) {
    // MxInt4 MoE
    return MxInt4BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);
  }
  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::Bfloat16) {
    // BF16 MoE
    return Bf16MoeLauncher::getValidConfigs(
        top_k,
        hidden_size,
        intermediate_size,
        num_local_experts,
        num_tokens,
        act_type,
        use_shuffled_weight,
        weight_layout);

  } else if (
      fp8_quantization_type == Fp8QuantizationType::DeepSeekFp8 && dtype_act == btg::Dtype::E4m3 &&
      dtype_weights == btg::Dtype::E4m3) {
    if (activation_type != ActivationType::Swiglu) {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "DeepSeekFp8 only supports ActivationType::Swiglu, "
                                                 << "got act_type=" << act_type << ".";
    }
    // FP8 block scale (DeepSeek)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k,
        hidden_size,
        intermediate_size,
        num_local_experts,
        num_tokens,
        use_shuffled_weight,
        weight_layout,
        dtype_act,
        dtype_weights,
        fp8_quantization_type,
        act_type);
  } else if (
      fp8_quantization_type == Fp8QuantizationType::MxFp8 && dtype_act == btg::Dtype::MxE4m3 &&
      dtype_weights == btg::Dtype::MxE4m3) {
    // FP8 block scale (MxFp8)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k,
        hidden_size,
        intermediate_size,
        num_local_experts,
        num_tokens,
        use_shuffled_weight,
        weight_layout,
        dtype_act,
        dtype_weights,
        fp8_quantization_type,
        act_type);
  } else if (
      (fp8_quantization_type == Fp8QuantizationType::PerTensorFp8 ||
       fp8_quantization_type == Fp8QuantizationType::NoneFp8) &&
      dtype_weights == btg::Dtype::E4m3) {
    return Fp8PerTensorLauncher::getValidConfigs(
        top_k,
        hidden_size,
        intermediate_size,
        num_local_experts,
        num_tokens,
        act_type,
        use_shuffled_weight,
        weight_layout,
        dtype_act,
        dtype_weights);
  } else if (dtype_weights == btg::Dtype::E2m1 || dtype_weights == btg::Dtype::MxE2m1) {
    // FP4 block scale
    return FP4BlockScaleLauncher::getValidConfigs(
        top_k,
        hidden_size,
        intermediate_size,
        num_local_experts,
        num_tokens,
        act_type,
        dtype_act,
        dtype_weights,
        use_per_token_scaling);
  }

  TVM_FFI_LOG_AND_THROW(NotImplementedError)
      << "Unsupported data type combination for getValidConfigs: "
      << "dtype_act=" << static_cast<int>(dtype_act) << ", dtype_weights=" << static_cast<int>(dtype_weights)
      << ", fp8_quantization_type=" << fp8QuantizationTypeToString(fp8_quantization_type);

  // Unreachable code - added to suppress compiler warning
  return Array<Array<int64_t>>();
}

// ---------------------------------------------------------------------------
// FP4 LoRA cubin probe (de-risk gate). The FP8-like LoRA design needs the FP4 GEMM1
// to run with fusedAct=false so the standalone activation kernel can inject the
// gate_up LoRA delta pre-SwiGLU (DeepSeek-FP8 already unfuses; NvFP4 normally fuses).
// This builds an E2m1/E2m1 NvFP4 MoE::Runner both fused and unfused and reports
// getValidConfigIndices().size() for each. A non-negative unfused value means the
// trtllm-gen cubin set includes a non-fused gated GEMM1 for NvFP4 (path is viable);
// -1 means construction threw (no such cubin / unsupported combo).
Array<int64_t> sgl_trtllm_fp4_probe_unfused(
    int64_t top_k,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t num_local_experts,
    int64_t num_tokens,
    int64_t tile_n) {
  using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;
  using ActivationType = tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
  // Probe variants:
  //   [0] Swiglu fused   (normal NvFP4 path, fusedAct=true)              -> sanity, expect >0
  //   [1] Swiglu unfused (fusedAct=false gated GEMM1)                    -> known dead (-1)
  //   [2] Identity non-gated (plain route-act GEMM1, raw 2*inter output) -> candidate for unfuse
  //   [3] Relu2   non-gated                                              -> alt candidate
  auto probe = [&](ActivationType act, bool unfuse) -> int64_t {
    try {
      RunnerType r(
          btg::Dtype::E2m1,
          btg::Dtype::E2m1,
          /*useDeepSeekFp8=*/false,
          static_cast<int>(tile_n),
          act,
          /*useShuffledMatrix=*/true,
          batchedGemm::gemm::MatrixLayout::MajorK,
          /*usePerTokenScalingGemm1=*/true,
          /*usePerTokenScalingGemm2=*/true,
          /*usePerChannelScalingGemm1=*/false,
          /*usePerChannelScalingGemm2=*/false,
          /*unfuseActForLora=*/unfuse);
      auto cfgs = r.getValidConfigIndices(
          static_cast<int32_t>(top_k),
          static_cast<int32_t>(hidden_size),
          static_cast<int32_t>(intermediate_size),
          static_cast<int32_t>(num_local_experts),
          static_cast<int32_t>(num_tokens));
      return static_cast<int64_t>(cfgs.size());
    } catch (...) {
      return -1;
    }
  };
  return {
      probe(ActivationType::Swiglu, false),
      probe(ActivationType::Swiglu, true),
      probe(ActivationType::Identity, false),
      probe(ActivationType::Relu2, false)};
}

// Evaluate Solution 1: use the non-gated routeAct=false GEMM2-style batched GEMM as the gate_up
// (GEMM1) projection. Returns [num_passing_cubins, num_valid_for_dims] for an E2m1->bf16 Gemm2
// runner at the requested (output=out_dim, K=k_dim) shape. A positive valid count means the
// existing GEMM2 cubin can compute the raw (un-activated) gate_up projection on permuted tokens.
Array<int64_t> sgl_trtllm_fp4_probe_gemm2(
    int64_t out_dim, int64_t k_dim, int64_t num_local_experts, int64_t num_tokens, int64_t top_k, int64_t tile_n) {
  using G2 = tensorrt_llm::kernels::trtllmgen_moe::Gemm2::Runner;
  auto probe = [&](bool per_tok) -> Array<int64_t> {
    try {
      G2 r(
          btg::Dtype::E2m1,
          btg::Dtype::E2m1,
          btg::Dtype::Bfloat16,
          /*useDeepSeekFp8=*/false,
          static_cast<int>(tile_n),
          /*useShuffledMatrix=*/true,
          batchedGemm::gemm::MatrixLayout::MajorK,
          /*usePerTokenScaling=*/per_tok,
          /*usePerChannelScaling=*/false);
      auto passing = r.getPassingConfigIndices();
      int64_t valid = 0;
      for (auto idx : passing) {
        if (r.isValidConfigIndex(
                static_cast<int32_t>(idx),
                static_cast<int32_t>(top_k),
                static_cast<int32_t>(out_dim),
                static_cast<int32_t>(k_dim),
                static_cast<int32_t>(num_local_experts),
                static_cast<int32_t>(num_tokens))) {
          valid++;
        }
      }
      return {static_cast<int64_t>(passing.size()), valid};
    } catch (...) {
      return {-1, -1};
    }
  };
  auto pt = probe(true);
  auto npt = probe(false);
  return {pt[0], pt[1], npt[0], npt[1]};
}

// ===== Standalone single-kernel runners (for the lora_moe_triton_prep testbeds) =====
// Expose the in-op permute / NvFP4-quant / activation kernels (which otherwise only run inside
// FP4BlockScaleLoraLauncher::run) so a self-contained bench/correctness script can drive them.
// Each takes PRE-ALLOCATED in/out tensors and only builds the Data + launches the kernel (no
// device alloc -> CUDA-graph-capture-safe timing). Data setup mirrors FP4BlockScaleLoraLauncher::run.
int64_t bench_permute(
    TensorView hidden_in,
    TensorView idx_map,
    TensorView total_pad,
    TensorView permuted_out,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_size) {
  cudaStream_t stream = get_stream(hidden_in.device());
  moe::dev::permute::Data d;
  d.mDtypeElt = btg::Dtype::Bfloat16;
  d.mUsePdl = false;
  d.mUseDeepSeekFp8 = false;
  d.inPtr = hidden_in.data_ptr();
  d.outPtr = permuted_out.data_ptr();
  d.inDqSfsPtr = nullptr;
  d.outDqSfsPtr = nullptr;
  d.expandedIdxToPermutedIdx = static_cast<int*>(idx_map.data_ptr());
  d.hiddenDim = hidden_size;
  d.numTokens = num_tokens;
  d.topK = top_k;
  d.totalNumPaddedTokens = static_cast<int*>(total_pad.data_ptr());
  moe::dev::permute::run(d, stream);
  return 0;
}

int64_t bench_nvfp4_quant(
    TensorView in_bf16,
    Optional<TensorView> idx_map,
    TensorView out_fp4,
    TensorView out_sf,
    TensorView out_ptsf,
    int64_t m,
    int64_t n,
    int64_t tile) {
  cudaStream_t stream = get_stream(in_bf16.device());
  auto sfLayout = tile >= 128 ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                              : tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4;
  float const gsi = 1.f / 448.f / 6.f;
  int* map = idx_map.has_value() ? static_cast<int*>(idx_map.value().data_ptr()) : nullptr;
  tensorrt_llm::kernels::invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
      m,
      n,
      reinterpret_cast<__nv_bfloat16 const*>(in_bf16.data_ptr()),
      gsi,
      map,
      reinterpret_cast<uint8_t*>(out_fp4.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sf.data_ptr()),
      reinterpret_cast<float*>(out_ptsf.data_ptr()),
      sfLayout,
      stream);
  return 0;
}

int64_t bench_activation(
    TensorView gate_up,
    TensorView lora_delta,
    TensorView idx_map,
    TensorView total_pad,
    TensorView activated_out,
    TensorView lora_input_out,
    int64_t inner_dim,
    int64_t num_tokens,
    int64_t top_k,
    int64_t grid_x_override,
    int64_t opt_mode) {
  cudaStream_t stream = get_stream(gate_up.device());
  moe::dev::activation::Data d;
  d.mDtypeElt = btg::Dtype::Bfloat16;
  d.mUsePdl = false;
  d.mUseDeepSeekFp8 = false;
  d.inPtr = gate_up.data_ptr();
  d.interleavedGateUpInput = true;
  d.outPtr = activated_out.data_ptr();
  d.inDqSfsPtr = nullptr;
  d.outDqSfsPtr = nullptr;
  d.gateUpLoraDeltaPtr = static_cast<cutlass::bfloat16_t const*>(lora_delta.data_ptr());
  d.activationLoraInputOutPtr = static_cast<cutlass::bfloat16_t*>(lora_input_out.data_ptr());
  d.innerDim = inner_dim;
  d.numTokens = num_tokens;
  d.topK = top_k;
  d.expandedIdxToPermutedIdx = static_cast<int*>(idx_map.data_ptr());
  d.totalNumPaddedTokens = static_cast<int*>(total_pad.data_ptr());
  d.actGridXOverride = static_cast<int32_t>(grid_x_override);
  d.actOptMode = static_cast<int32_t>(opt_mode);
  moe::dev::activation::run(d, stream);
  return 0;
}

// Fused permute + NvFP4-per-token-quant: reads UN-permuted bf16 hidden ([num_tokens, hidden]) and
// scatter-writes fp4 + swizzled block-sf + per-token-sf to the permuted positions, replacing the
// plain permuteKernel + nvfp4QuantAndPerTokenScale #1 pair. dedup!=0 picks the per-token-grid
// (quantize once, scatter) variant; dedup==0 picks the per-pair-grid (re-quantize per pair) variant.
int64_t bench_fused_permute_quant(
    TensorView hidden_in,
    TensorView idx_map,
    TensorView out_fp4,
    TensorView out_sf,
    TensorView out_ptsf,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden_size,
    int64_t maxpad,
    int64_t tile,
    int64_t dedup) {
  cudaStream_t stream = get_stream(hidden_in.device());
  auto sfLayout = tile >= 128 ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                              : tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4;
  float const gsi = 1.f / 448.f / 6.f;
  sgl_fused_permute_quant::invokeFusedPermuteNvfp4Quant<__nv_bfloat16>(
      static_cast<uint32_t>(num_tokens),
      static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(hidden_size),
      static_cast<int>(maxpad),
      reinterpret_cast<__nv_bfloat16 const*>(hidden_in.data_ptr()),
      gsi,
      static_cast<int32_t const*>(idx_map.data_ptr()),
      reinterpret_cast<uint8_t*>(out_fp4.data_ptr()),
      reinterpret_cast<uint8_t*>(out_sf.data_ptr()),
      reinterpret_cast<float*>(out_ptsf.data_ptr()),
      sfLayout,
      dedup != 0,
      stream);
  return 0;
}

// Standalone runner for the fused SwiGLU+LoRA activation -> NVFP4 per-token quant kernel
// (FP4 MoE LoRA aggressive fusion). Mirrors step 6 (activation) + step 7 (quant#2) of
// FP4BlockScaleLoraLauncher::run but skips materializing activated_bf16. Output is
// bitwise-identical to bench_activation -> bench_nvfp4_quant(#2). disableFastMath is fixed to
// the default (false) here, matching the testbed (no SGLANG fp4 fast-math env set).
int64_t bench_fused_act_quant(
    TensorView gate_up,         // interleaved gate/up [.., inner_dim] bf16, by permutedIdx
    TensorView lora_delta,      // [num_tokens, top_k, inner_dim] bf16, by expandedIdx
    TensorView idx_map,         // [num_tokens*top_k] int32 (expanded -> permuted, -1 = padding)
    TensorView fp4_out,         // [.., inner_half/2] uint8
    TensorView sf_out,          // swizzled e4m3 SF, uint8
    TensorView ptsf_out,        // [..] float32 (per-token scale, by permutedIdx)
    TensorView lora_input_out,  // [num_tokens, top_k, inner_half] bf16, by expandedIdx
    int64_t inner_half,
    int64_t inner_dim,
    int64_t num_tokens,
    int64_t top_k,
    int64_t tile) {
  cudaStream_t stream = get_stream(gate_up.device());
  auto sfLayout = tile >= 128 ? tensorrt_llm::QuantizationSFLayout::SWIZZLED_128x4
                              : tensorrt_llm::QuantizationSFLayout::SWIZZLED_8x4;
  float const globalScaleInv = 1.f / 448.f / 6.f;
  flashinfer::sgl_fused_act_quant::launchFusedActivationQuant(
      static_cast<int>(num_tokens * top_k),
      static_cast<int>(inner_half),
      static_cast<int>(inner_dim),
      reinterpret_cast<__nv_bfloat16 const*>(gate_up.data_ptr()),
      reinterpret_cast<__nv_bfloat16 const*>(lora_delta.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(lora_input_out.data_ptr()),
      static_cast<int32_t const*>(idx_map.data_ptr()),
      globalScaleInv,
      reinterpret_cast<uint8_t*>(fp4_out.data_ptr()),
      reinterpret_cast<uint8_t*>(sf_out.data_ptr()),
      reinterpret_cast<float*>(ptsf_out.data_ptr()),
      sfLayout,
      /*disableFp4FastMath=*/false,
      stream);
  return 0;
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_bf16_moe, trtllm_bf16_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_moe, trtllm_fp8_per_tensor_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_block_scale_moe, trtllm_fp8_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp8_block_scale_moe_lora, sgl_trtllm_fp8_block_scale_moe_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_trtllm_fp8_block_scale_moe_lora_finalize, sgl_trtllm_fp8_block_scale_moe_lora_finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp4_block_scale_moe_lora, sgl_trtllm_fp4_block_scale_moe_lora);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_trtllm_fp4_block_scale_moe_lora_finalize, sgl_trtllm_fp4_block_scale_moe_lora_finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp4_probe_unfused, sgl_trtllm_fp4_probe_unfused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trtllm_fp4_probe_gemm2, sgl_trtllm_fp4_probe_gemm2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bench_permute, bench_permute);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bench_nvfp4_quant, bench_nvfp4_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bench_activation, bench_activation);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bench_fused_permute_quant, bench_fused_permute_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bench_fused_act_quant, bench_fused_act_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxint4_block_scale_moe, trtllm_mxint4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_configs, trtllm_get_valid_moe_configs);

}  // namespace flashinfer
