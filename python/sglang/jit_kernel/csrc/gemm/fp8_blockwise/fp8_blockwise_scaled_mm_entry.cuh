/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "fp8_blockwise_scaled_mm_sm120.cuh"  // For fp8_blockwise_scaled_mm_sm120

// Exported entry point. Inputs are expected to be pre-padded (mat_a / scales_a
// row-padded to a multiple of 4) and `out` pre-allocated by the Python wrapper.
void fp8_blockwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  fp8_blockwise_scaled_mm_sm120(out, mat_a, mat_b, scales_a, scales_b, /*force_noswap=*/false);
}

// Benchmark/debug entry: forces the legacy non-swapped 128-tile path for all M.
void fp8_blockwise_scaled_mm_noswap(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  fp8_blockwise_scaled_mm_sm120(out, mat_a, mat_b, scales_a, scales_b, /*force_noswap=*/true);
}

// Benchmark/debug entries: force a fixed swapAB tile N regardless of M.
void fp8_blockwise_scaled_mm_swapab32(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  fp8_blockwise_scaled_mm_sm120_swapab_fixed<32>(out, mat_a, mat_b, scales_a, scales_b);
}

void fp8_blockwise_scaled_mm_swapab64(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  fp8_blockwise_scaled_mm_sm120_swapab_fixed<64>(out, mat_a, mat_b, scales_a, scales_b);
}
