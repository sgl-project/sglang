// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "kernels.cuh"
#include "kernels_ops.cuh"
#include "kernels_exec.cuh"

namespace omnidreams_singleview {

struct MemBlockWeights {
    const half* w1; const half* b1;
    const half* w2; const half* b2;
    const half* w3; const half* b3;
    const half* wskip; // nullable if Cin==Cout (identity skip)
};

struct EncoderStageWeights {
    // TPool conv 1x1: [Cout=C, Cin=stride*C]
    const half* tpool_w; // bias is typically nullptr (bias=False in PyTorch)
    const half* tpool_b; // nullable
    int tpool_stride;    // 2, 2, 1 for the three stages in reference model
    // Spatial downsample conv 3x3 stride=2
    const half* down_w; const half* down_b; // bias may be nullptr
    // Three MemBlocks at this spatial scale
    MemBlockWeights mb[3];
};

struct TAEWeightsEncoder {
    // Initial conv 3x3: 3->64
    const half* c0_w; const half* c0_b;
    // Three encoder stages
    EncoderStageWeights stage[3];
    // Final conv to latent channels (e.g., 64->16)
    const half* final_w; const half* final_b;
};

// Channel constants to mirror Python defaults
static constexpr int kImageChannels = 3;
static constexpr int kLatentChannels = 16;

inline void build_encoder_ops(/*in*/ const TAEWeightsEncoder& w, /*out*/ OpDesc* ops, int* num_ops, int c0_cin_padded = 8)
{
    int i = 0;
    // Initial conv 3x3 + ReLU (use padded Cin for Tensor Core alignment)
    ops[i++] = MakeConv3x3(w.c0_w, w.c0_b, /*Cin=*/c0_cin_padded, /*Cout=*/64, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::ReLU);
    int C = 64; // feature channels throughout encoder
    // Three stages matching the reference: strides {2,2,1}
    for (int s = 0; s < 3; ++s) {
        const EncoderStageWeights& st = w.stage[s];
        // TPool conv 1x1 (caller must reshape NT and Cin to (N*T/stride, stride*C))
        ops[i++] = MakeTPool(st.tpool_stride);
        ops[i++] = MakeConv1x1(st.tpool_w, st.tpool_b, /*Cin=*/st.tpool_stride*C, /*Cout=*/C, /*groups=*/1, Activation::None);
        // Spatial downsample conv 3x3 stride=2
        ops[i++] = MakeConv3x3(st.down_w, st.down_b, /*Cin=*/C, /*Cout=*/C, /*groups=*/1, /*stride=*/2, /*pad=*/1, Activation::None);
        // Three MemBlocks
        for (int k = 0; k < 3; ++k) {
            const MemBlockWeights& m = st.mb[k];
            ops[i++] = MakeMemBlock(m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, /*Cout=*/C);
        }
    }
    // Final conv to latent channels (no activation)
    ops[i++] = MakeConv3x3(w.final_w, w.final_b, /*Cin=*/C, /*Cout=*/kLatentChannels, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::None);
    *num_ops = i;
}

struct DecoderStageWeights {
    const half* tgrow_w; const half* tgrow_b; // bias typically nullptr
    int tgrow_stride; // 1 or 2 depending on time upscale flag
    const half* next_w; const half* next_b;
    MemBlockWeights mb[3];
};

struct TAEWeightsDecoder {
    const half* c0_w; const half* c0_b;
    MemBlockWeights mb0[3];
    DecoderStageWeights stage[3];
    const half* final_w; const half* final_b;
    int final_cout_padded;  // Padded channel count (e.g., 8 for RGB=3)
    int final_cout_orig;    // Original channel count (e.g., 3 for RGB)
};

inline int compute_frames_to_trim(bool time_up0, bool time_up1)
{
    int sum = (time_up0 ? 1 : 0) + (time_up1 ? 1 : 0);
    return (1 << sum) - 1;
}

inline void build_decoder_ops(/*in*/ const TAEWeightsDecoder& w, /*out*/ OpDesc* ops, int* num_ops,
    bool time_up0=true, bool time_up1=true, bool space_up0=true, bool space_up1=true, bool space_up2=true)
{
    int i = 0;
    // Clamp first (elementwise), then first conv 3x3 + ReLU to 256
    ops[i++] = MakeClamp();
    ops[i++] = MakeConv3x3(w.c0_w, w.c0_b, /*Cin=*/kLatentChannels, /*Cout=*/256, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::ReLU);
    int C = 256;
    // Three MemBlocks at 256
    for (int k = 0; k < 3; ++k) {
        const MemBlockWeights& m = w.mb0[k];
        ops[i++] = MakeMemBlock(m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, /*Cout=*/C);
    }
    // Three decoder stages mapping 256->128->64->64
    // Stage 0: no temporal growth (stride=1)
    if (space_up0) ops[i++] = MakeUpsample2x();
    ops[i++] = MakeTGrow(/*stride=*/1);
    ops[i++] = MakeConv1x1(w.stage[0].tgrow_w, w.stage[0].tgrow_b, /*Cin=*/C, /*Cout=*/C, /*groups=*/1, Activation::None);
    ops[i++] = MakeConv3x3(w.stage[0].next_w, w.stage[0].next_b, /*Cin=*/C, /*Cout=*/128, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::None);
    C = 128;
    for (int k = 0; k < 3; ++k) {
        const MemBlockWeights& m = w.stage[0].mb[k];
        ops[i++] = MakeMemBlock(m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, /*Cout=*/C);
    }
    // Stage 1: temporal growth based on time_up0
    if (space_up1) ops[i++] = MakeUpsample2x();
    ops[i++] = MakeTGrow(time_up0 ? 2 : 1);
    ops[i++] = MakeConv1x1(w.stage[1].tgrow_w, w.stage[1].tgrow_b, /*Cin=*/C, /*Cout=*/(time_up0 ? 2*C : C), /*groups=*/1, Activation::None);
    ops[i++] = MakeConv3x3(w.stage[1].next_w, w.stage[1].next_b, /*Cin=*/C, /*Cout=*/64, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::None);
    C = 64;
    for (int k = 0; k < 3; ++k) {
        const MemBlockWeights& m = w.stage[1].mb[k];
        ops[i++] = MakeMemBlock(m.w1, m.b1, m.w2, m.b2, m.w3, m.b3, m.wskip, /*Cout=*/C);
    }
    // Stage 2: temporal growth based on time_up1
    if (space_up2) ops[i++] = MakeUpsample2x();
    ops[i++] = MakeTGrow(time_up1 ? 2 : 1);
    ops[i++] = MakeConv1x1(w.stage[2].tgrow_w, w.stage[2].tgrow_b, /*Cin=*/C, /*Cout=*/(time_up1 ? 2*C : C), /*groups=*/1, Activation::None);
    // Match Python decoder semantics (ReLU before final): fuse ReLU into s2 next conv
    ops[i++] = MakeConv3x3(w.stage[2].next_w, w.stage[2].next_b, /*Cin=*/C, /*Cout=*/64, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::ReLU);
    C = 64;
    // Final conv to RGB (padded to meet Tensor Core alignment, will be sliced later)
    int final_cout = (w.final_cout_padded > 0) ? w.final_cout_padded : kImageChannels;
    ops[i++] = MakeConv3x3(w.final_w, w.final_b, /*Cin=*/C, /*Cout=*/final_cout, /*groups=*/1, /*stride=*/1, /*pad=*/1, Activation::None);
    *num_ops = i;
}

} // namespace omnidreams_singleview
