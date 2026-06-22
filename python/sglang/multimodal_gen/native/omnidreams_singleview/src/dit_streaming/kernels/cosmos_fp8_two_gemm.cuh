// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cutlass/numeric_types.h"
#include <cuda_runtime.h>

namespace omnidreams_singleview {

cudaError_t run_cosmos_fp8_dense_ref(
    const cutlass::float_e4m3_t* q,
    const cutlass::float_e4m3_t* k,
    const cutlass::float_e4m3_t* v,
    cutlass::float_e4m3_t* q_bhmd,
    cutlass::float_e4m3_t* k_bhmd,
    cutlass::float_e4m3_t* v_bhmd,
    cutlass::half_t* scores,
    cutlass::float_e4m3_t* probs,
    cutlass::half_t* o_bhmd,
    cutlass::half_t* o,
    int B,
    int Mq,
    int Mk,
    int H,
    int D,
    bool causal,
    cudaStream_t stream);

}  // namespace omnidreams_singleview
