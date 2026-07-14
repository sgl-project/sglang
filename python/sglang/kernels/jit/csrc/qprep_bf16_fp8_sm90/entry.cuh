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

// JIT dispatch entry for the SM90 Q8KV8 born-fp8 q-prep kernel.
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "kernel.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

// All strides are in elements; validation of dtypes/shapes/alignment happens
// in the Python wrapper (sglang/jit_kernel/qprep_bf16_fp8_sm90.py).
void qprep_bf16_fp8_dispatch(
    tvm::ffi::TensorView q_nope,
    tvm::ffi::TensorView w_kc,
    tvm::ffi::TensorView q_rope,
    tvm::ffi::TensorView out,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t k_dim,
    int64_t a_s0,
    int64_t a_s1,
    int64_t b_s0,
    int64_t b_s2,
    int64_t r_s0,
    int64_t r_s1,
    int64_t o_s0,
    int64_t o_s1,
    int64_t rope_vec16,
    int64_t out_vec16,
    int64_t cuda_stream) {
  QprepBf16Fp8Sm90Params params;
  params.num_tokens = (int)num_tokens;
  params.num_heads = (int)num_heads;
  params.q_nope = q_nope.data_ptr();
  params.a_s0 = a_s0;
  params.a_s1 = a_s1;
  params.w_kc = w_kc.data_ptr();
  params.b_s0 = b_s0;
  params.b_s2 = b_s2;
  params.q_rope = q_rope.data_ptr();
  params.r_s0 = r_s0;
  params.r_s1 = r_s1;
  params.rope_vec16 = (bool)rope_vec16;
  params.out = out.data_ptr();
  params.o_s0 = o_s0;
  params.o_s1 = o_s1;
  params.out_vec16 = (bool)out_vec16;

  DLDevice dev = q_nope.device();
  cudaSetDevice(dev.device_id);
  params.stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  switch (k_dim) {
    case 128:
      qprep_sm90::run_qprep_bf16_fp8_sm90<128>(params);
      return;
    case 192:
      qprep_sm90::run_qprep_bf16_fp8_sm90<192>(params);
      return;
    default:
      fprintf(stderr, "qprep_bf16_fp8_sm90: unsupported k_dim=%ld (must be 128 or 192)\n", (long)k_dim);
      exit(1);
  }
}

}  // namespace
