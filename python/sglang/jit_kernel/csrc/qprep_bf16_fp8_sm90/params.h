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

// Parameters for the SM90 Q8KV8 born-fp8 q-prep kernel (absorbed-q bmm +
// nope/rope concat + fp32 -> bf16 -> fp8_e4m3 cast).  All strides are in
// ELEMENTS of the respective tensor's dtype (fp8 strides == byte strides).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

struct QprepBf16Fp8Sm90Params {
  int num_tokens;  // T (runtime; m-tiles are masked)
  int num_heads;   // H (grid dim)

  // q_nope: [T, H, K] bf16 (strided view OK; innermost dim contiguous)
  const void* q_nope;
  int64_t a_s0, a_s1;

  // w_kc: [H, K, N] bf16 with K contiguous (stride(1) == 1; production layout
  // is (K*N, 1, K), i.e. the N-major absorbed weight)
  const void* w_kc;
  int64_t b_s0, b_s2;

  // q_rope: [T, H, R] bf16 (strided view OK; innermost dim contiguous)
  const void* q_rope;
  int64_t r_s0, r_s1;
  // 16B-aligned rope rows (base pointer and both strides) -> uint4 loads
  bool rope_vec16;

  // out: [T, pad_heads, N + R] fp8_e4m3; only [:, :H, :] is written
  void* out;
  int64_t o_s0, o_s1;
  // 16B-aligned out rows (base pointer and both strides) -> smem-staged
  // coalesced uint4 stores for the nope half (else direct u16 stores)
  bool out_vec16;

  cudaStream_t stream;
};
