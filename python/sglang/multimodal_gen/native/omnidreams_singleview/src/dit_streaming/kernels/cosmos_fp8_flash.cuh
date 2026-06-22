// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cutlass/numeric_types.h"
#include <cuda_runtime.h>

namespace omnidreams_singleview {

// Fused FP8 attention bring-up path for Cosmos-layout tensors:
//   Q: [B, Mq, H, D] raw E4M3 bytes
//   K: [B, Mk, H, D] raw E4M3 bytes
//   V: [B, Mk, H, D] raw E4M3 bytes
//   O: [B, Mq, H, D] fp16
//
// This kernel keeps the softmax online within a single kernel and does not
// materialize the Mq x Mk score/probability matrix.
cudaError_t run_cosmos_cute_flash_fp8(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    cudaStream_t stream);

}  // namespace omnidreams_singleview
