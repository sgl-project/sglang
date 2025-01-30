#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"
#include "utils.h"

using namespace tensorrt_llm::kernels;

namespace {

template<typename T>
CutlassMoeFCRunner<T, T, T> createMoeRunner() {
    CutlassMoeFCRunner<T, T, T> runner;
    // Use default configs
    return runner;
}

tensorrt_llm::ActivationType parseActivationType(const std::string& act_type) {
    if (act_type == "relu") {
        return tensorrt_llm::ActivationType::Relu;
    } else if (act_type == "gelu") {
        return tensorrt_llm::ActivationType::Gelu;
    } else if (act_type == "swiglu") {
        return tensorrt_llm::ActivationType::Swiglu;
    } else if (act_type == "geglu") {
        return tensorrt_llm::ActivationType::Geglu;
    } else {
        throw std::runtime_error("Unsupported activation type: " + act_type);
    }
}

template<typename DataType, typename WeightType, typename OutputType>
void runMoeFCKernel(DataType* input,                   // [num_tokens, hidden_size] 
                    float* gating_output,        // [num_tokens, num_experts]
                    WeightType* fc1_expert_weights,       // [num_experts, hidden_size, inter_size]
                    tensorrt_llm::ActivationType act_type,
                    WeightType* fc2_expert_weights,       // [num_experts, inter_size, hidden_size]
                    int64_t num_tokens,
                    int64_t hidden_size, 
                    int64_t inter_size,
                    int64_t num_experts,
                    int64_t top_k,
                    OutputType* output,                   // [num_tokens, hidden_size]
                    void* workspace,
                    WeightType* w1_scale,                 // Optional scaling factors
                    WeightType* w2_scale,
                    cudaStream_t stream) {
    
    CutlassMoeFCRunner<DataType, WeightType, OutputType> runner;
    
    // Allocate intermediate buffers from workspace
    char* ws_ptr = static_cast<char*>(workspace);
    float* scale_probs = reinterpret_cast<float*>(ws_ptr);
    ws_ptr += sizeof(float) * num_tokens * top_k;
    
    int* source_map = reinterpret_cast<int*>(ws_ptr);
    ws_ptr += sizeof(int) * num_tokens * top_k;
    
    int* selected_experts = reinterpret_cast<int*>(ws_ptr);
    ws_ptr += sizeof(int) * num_tokens * top_k;

    // No parallelism config for now
    MOEParallelismConfig parallel_config;
    parallel_config.ep_size = 1;
    parallel_config.ep_rank = 0;
    parallel_config.tp_size = 1;
    parallel_config.tp_rank = 0;

    // No bias tensors
    WeightType* fc1_expert_bias = nullptr;
    WeightType* fc2_expert_bias = nullptr;

    // Setup quantization params if scales are provided
    QuantParams quant_params;
    if (w1_scale != nullptr && w2_scale != nullptr) {
        quant_params = QuantParams::FP8(
            reinterpret_cast<float const*>(w1_scale),
            reinterpret_cast<float const*>(w2_scale),
            nullptr);
    }

    // No LoRA params
    LoraParams lora_params;

    // Run MoE kernel
    runner.runMoe(input,
                  gating_output, 
                  fc1_expert_weights,
                  fc1_expert_bias,
                  act_type,
                  fc2_expert_weights, 
                  fc2_expert_bias,
                  quant_params,
                  num_tokens,
                  hidden_size,
                  inter_size,
                  num_experts,
                  top_k,
                  ws_ptr,
                  output,
                  nullptr,  // finished mask
                  num_tokens, // active rows
                  scale_probs,
                  source_map,
                  selected_experts,
                  0.0f, // epsilon for sparse mixer
                  parallel_config,
                  MOEExpertScaleNormalizationMode::NONE,
                  false, // use_lora
                  lora_params,
                  stream);
}

} // namespace

torch::Tensor trt_fused_moe(torch::Tensor input_activations,
                        torch::Tensor gating_output,
                        torch::Tensor fc1_expert_weights,
                        std::string fc1_activation_type_str,
                        torch::Tensor fc2_expert_weights,
                        const c10::optional<torch::Tensor> w1_scale,
                        const c10::optional<torch::Tensor> w2_scale,
                        int64_t top_k) {
    
    CHECK_INPUT(input_activations);
    CHECK_INPUT(gating_output); 
    CHECK_INPUT(fc1_expert_weights);
    CHECK_INPUT(fc2_expert_weights);

    // Get dimensions
    int64_t num_tokens = input_activations.size(0);
    int64_t hidden_size = input_activations.size(1);
    int64_t num_experts = gating_output.size(1);
    int64_t inter_size = fc1_expert_weights.size(2);

    // Validate dimensions
    CHECK_EQ(gating_output.size(0), num_tokens);
    CHECK_EQ(fc1_expert_weights.size(0), num_experts);
    CHECK_EQ(fc1_expert_weights.size(1), hidden_size);
    CHECK_EQ(fc2_expert_weights.size(0), num_experts);
    CHECK_EQ(fc2_expert_weights.size(1), inter_size);
    CHECK_EQ(fc2_expert_weights.size(2), hidden_size);

    // Create output tensor
    auto output = torch::empty_like(input_activations);

    // Get activation type
    auto act_type = parseActivationType(fc1_activation_type_str);

    // Calculate workspace size
    size_t ws_size = sizeof(float) * num_tokens * top_k + // scale_probs
                     sizeof(int) * num_tokens * top_k * 2; // source_map + selected_experts

    auto runner = createMoeRunner<float>();
    ws_size += runner.getWorkspaceSize(num_tokens,
                                     hidden_size,
                                     inter_size,
                                     num_experts,
                                     top_k,
                                     act_type,
                                     MOEExpertScaleNormalizationMode::NONE,
                                     MOEParallelismConfig{},
                                     false);

    // Allocate workspace
    auto options = torch::TensorOptions()
                      .dtype(torch::kUInt8)
                      .device(input_activations.device());
    auto workspace = torch::empty(ws_size, options);

    // Get current stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto input_type = input_activations.scalar_type();
    const auto weight_type = fc1_expert_weights.scalar_type();

    bool dispatched = false;

    // FP32
    if (input_type == torch::kFloat32) {
        if (weight_type == torch::kFloat32) {
            runMoeFCKernel<float, float, float>(
                static_cast<float*>(input_activations.data_ptr()),
                static_cast<float*>(gating_output.data_ptr()),
                static_cast<float*>(fc1_expert_weights.data_ptr()),
                act_type,
                static_cast<float*>(fc2_expert_weights.data_ptr()),
                num_tokens,
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                static_cast<float*>(output.data_ptr()),
                workspace.data_ptr(),
                w1_scale.has_value() ? static_cast<float*>(w1_scale.value().data_ptr()) : nullptr,
                w2_scale.has_value() ? static_cast<float*>(w2_scale.value().data_ptr()) : nullptr,
                stream);
            dispatched = true;
        }
    }
    // FP16
    else if (input_type == torch::kFloat16) {
        if (weight_type == torch::kFloat16) {
            runMoeFCKernel<half, half, half>(
                static_cast<half*>(input_activations.data_ptr()),
                static_cast<float*>(gating_output.data_ptr()),
                static_cast<half*>(fc1_expert_weights.data_ptr()),
                act_type,
                static_cast<half*>(fc2_expert_weights.data_ptr()),
                num_tokens,
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                static_cast<half*>(output.data_ptr()),
                workspace.data_ptr(),
                w1_scale.has_value() ? static_cast<half*>(w1_scale.value().data_ptr()) : nullptr,
                w2_scale.has_value() ? static_cast<half*>(w2_scale.value().data_ptr()) : nullptr,
                stream);
            dispatched = true;
        }
        else if (weight_type == torch::kInt8) {
            runMoeFCKernel<half, uint8_t, half>(
                static_cast<half*>(input_activations.data_ptr()),
                static_cast<float*>(gating_output.data_ptr()),
                static_cast<uint8_t*>(fc1_expert_weights.data_ptr()),
                act_type,
                static_cast<uint8_t*>(fc2_expert_weights.data_ptr()),
                num_tokens,
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                static_cast<half*>(output.data_ptr()),
                workspace.data_ptr(),
                w1_scale.has_value() ? static_cast<uint8_t*>(w1_scale.value().data_ptr()) : nullptr,
                w2_scale.has_value() ? static_cast<uint8_t*>(w2_scale.value().data_ptr()) : nullptr,
                stream);
            dispatched = true;
        }
        else if (weight_type == torch::kUInt4) {
            runMoeFCKernel<half, cutlass::uint4b_t, half>(
                static_cast<half*>(input_activations.data_ptr()),
                static_cast<float*>(gating_output.data_ptr()),
                static_cast<cutlass::uint4b_t*>(fc1_expert_weights.data_ptr()),
                act_type,
                static_cast<cutlass::uint4b_t*>(fc2_expert_weights.data_ptr()),
                num_tokens,
                hidden_size,
                inter_size,
                num_experts,
                top_k,
                static_cast<half*>(output.data_ptr()),
                workspace.data_ptr(),
                w1_scale.has_value() ? static_cast<cutlass::uint4b_t*>(w1_scale.value().data_ptr()) : nullptr,
                w2_scale.has_value() ? static_cast<cutlass::uint4b_t*>(w2_scale.value().data_ptr()) : nullptr,
                stream);
            dispatched = true;
        }
    }

    if (!dispatched) {
        throw std::runtime_error("Unsupported input/weight type combination");
    }

    return output;
}
