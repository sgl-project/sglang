#pragma once

#include <vector>
#include <torch/all.h>

// Tiled moe fused gate dynamic dispatcher (VPT>32 fallback)
std::vector<at::Tensor> moe_fused_gate_tiled(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor);

// Tiled moe fused gate static specializations (compile-time params inside)
std::vector<at::Tensor> moe_fused_gate_tiled_static(
    at::Tensor& input,
    at::Tensor& bias,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    double routed_scaling_factor);

// Keep a consistent launch style with LAUNCH_MOE_GATE_CONFIG
#define LAUNCH_MOE_GATE_TILED_CONFIG(EXPERTS, EXPERT_GROUP, TILE)                                        \
  do {                                                                                                   \
    (void)(EXPERTS);                                                                                     \
    (void)(EXPERT_GROUP);                                                                                \
    (void)(TILE);                                                                                        \
    return moe_fused_gate_tiled_static(                                                                  \
        input, bias, num_expert_group, topk_group, topk, num_fused_shared_experts, routed_scaling_factor); \
  } while (0)


