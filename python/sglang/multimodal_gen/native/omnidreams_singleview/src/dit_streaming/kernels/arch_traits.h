// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace omnidreams_singleview {

// ============================================================================
// Architecture Traits for Multi-Model Extensibility
// ============================================================================
//
// ArchTraits encode the architectural decisions that differ between transformer
// model families (Wan, Flux, Hunyuan, etc.). The transformer block
// (run_transformer_block) and model forward (transformer_forward) are templated
// on an ArchTraits type so that new models can be supported by defining a new
// traits struct and instantiating the templates, without modifying core logic.
//
// Each traits struct must provide the following static constexpr members:
//   NormType  kSelfAttnNorm    - normalization before self-attention
//   NormType  kCrossAttnNorm   - normalization before cross-attention
//   ActType   kFFNActivation   - FFN hidden-layer activation function
//   bool      kHasImageCrossAttn - whether I2V image cross-attention is supported
//   int       kFFNGateStyle    - 0 = no gate, 1 = gated residual from scale_shift_table
//   int       kScaleShiftSlots - number of entries in per-block scale_shift_table (e.g. 6)

enum class NormType {
    RMSNorm,    // x / sqrt(mean(x^2) + eps)  -- no learnable affine (Wan)
    LayerNorm,  // (x - mean) / sqrt(var + eps) * gamma + beta (standard)
};

enum class ActType {
    GELU,       // Gaussian Error Linear Unit (Wan FFN)
    SiLU,       // Sigmoid Linear Unit / Swish (Flux, Hunyuan)
    GELUTanh,   // GELU with tanh approximation
};

// ---------------------------------------------------------------------------
// WanArchTraits -- Wan 2.1 (T2V-1.3B, I2V-14B) architecture
// ---------------------------------------------------------------------------
struct WanArchTraits {
    static constexpr NormType kSelfAttnNorm    = NormType::RMSNorm;
    static constexpr NormType kCrossAttnNorm   = NormType::LayerNorm;
    static constexpr ActType  kFFNActivation   = ActType::GELU;
    static constexpr bool     kHasImageCrossAttn = true;
    static constexpr int      kFFNGateStyle    = 1;  // gated residual
    static constexpr int      kScaleShiftSlots = 6;  // [scale1, shift1, gate1, shift2, scale2, gate2]
};

// ---------------------------------------------------------------------------
// Future model traits (not yet implemented -- placeholders for extensibility)
// ---------------------------------------------------------------------------
// struct FluxArchTraits {
//     static constexpr NormType kSelfAttnNorm    = NormType::LayerNorm;
//     static constexpr NormType kCrossAttnNorm   = NormType::LayerNorm;
//     static constexpr ActType  kFFNActivation   = ActType::SiLU;
//     static constexpr bool     kHasImageCrossAttn = false;
//     static constexpr int      kFFNGateStyle    = 0;
//     static constexpr int      kScaleShiftSlots = 2;
// };
//
// struct HunyuanArchTraits {
//     static constexpr NormType kSelfAttnNorm    = NormType::RMSNorm;
//     static constexpr NormType kCrossAttnNorm   = NormType::RMSNorm;
//     static constexpr ActType  kFFNActivation   = ActType::SiLU;
//     static constexpr bool     kHasImageCrossAttn = false;
//     static constexpr int      kFFNGateStyle    = 1;
//     static constexpr int      kScaleShiftSlots = 6;
// };

} // namespace omnidreams_singleview
