/* Copyright 2025 SGLang Team. All Rights Reserved.

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

// Fused SiLU-Mul + Per-Token FP8 Quantization Kernel
//
// Eliminates the BF16 intermediate between silu_and_mul and per_token_quant_fp8.
// Instead of:
//   Kernel 1: read gate_up [2d] → silu(gate) * up → write BF16 [d]
//   Kernel 2: read BF16 [d]   → find max → quantize → write FP8 [d]
// We do:
//   Kernel:   read gate_up [2d] → silu(gate) * up → find max → quantize → write FP8 [d]
//
// Memory savings: eliminates 1 global BF16 write + 1 global BF16 read per token.
// Kernel launch savings: eliminates 1 kernel launch (~30-70µs gap on H100).

#include <ATen/cuda/CUDAContext.h>

#include <flashinfer/vec_dtypes.cuh>

#include "utils.h"

using namespace flashinfer;

namespace {

template <typename T>
__forceinline__ __device__ T ceil_div_dev(T a, T b) {
  return (a + b - 1) / b;
}

// Register-resident kernel: activated values stay in registers across the
// abs_max reduction, eliminating all SMEM traffic except the reduction slot.
// ROUNDS >= 1: compile-time unrolled register path (covers d up to ROUNDS*VEC_SIZE*1024).
// ROUNDS == 0: dynamic fallback with SMEM buffer for very large hidden sizes.
template <uint32_t VEC_SIZE, uint32_t ROUNDS, typename T>
__global__ void FusedSiLUMulQuantFP8Kernel(
    const T* __restrict__ input,        // [M, 2*d]  gate_up projection
    __nv_fp8_e4m3* __restrict__ output_q,  // [M, d]  FP8 output
    float* __restrict__ output_s,       // [M, 1]  per-token scales
    const uint32_t d) {                 // intermediate_size
  const uint32_t bx = blockIdx.x;  // token index
  const uint32_t tx = threadIdx.x;
  const uint32_t ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;

  extern __shared__ float smem[];
  const uint32_t reduce_slots = ((num_warps + 3) / 4) * 4;
  float* smem_reduce = smem;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  if constexpr (ROUNDS >= 1) {
    // =================================================================
    // REGISTER-RESIDENT: all rounds' values in registers, no SMEM buffer.
    // Fully unrolled at compile time for ROUNDS=1,2,etc.
    // =================================================================
    float act_reg[ROUNDS][VEC_SIZE];
    float abs_max = 0.f;

    // Compute SiLU(gate) * up for each round → keep in registers
#pragma unroll
    for (uint32_t i = 0; i < ROUNDS; i++) {
      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;

      vec_t<T, VEC_SIZE> gate_vec;
      gate_vec.fill(0.f);
      vec_t<T, VEC_SIZE> up_vec;
      up_vec.fill(0.f);

      if (offset < d) {
        gate_vec.load(input + bx * 2 * d + offset);
        up_vec.load(input + bx * 2 * d + d + offset);
      }

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float gate_f = float(gate_vec[j]);
        float act = (gate_f / (1.0f + expf(-gate_f))) * float(up_vec[j]);
        abs_max = fmaxf(abs_max, fabsf(act));
        act_reg[i][j] = act;
      }
    }

    // Warp-level abs_max reduction
#pragma unroll
    for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
      abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, s));
    }

    // Cross-warp reduction
    smem_reduce[ty] = abs_max;
    __syncthreads();
    if (ty == 0) {
      abs_max = (tx < num_warps) ? smem_reduce[tx] : 0.f;
#pragma unroll
      for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, s));
      }
      smem_reduce[0] = abs_max;
    }
    __syncthreads();

    float scale = smem_reduce[0] / FP8_E4M3_MAX;
    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    if (thread_id == 0) {
      output_s[bx] = scale;
    }

    // Quantize from registers → FP8 store
#pragma unroll
    for (uint32_t i = 0; i < ROUNDS; i++) {
      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
      if (offset < d) {
        __nv_fp8_e4m3 fp8_arr[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          float val = fmaxf(fminf(act_reg[i][j] * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
          fp8_arr[j] = static_cast<__nv_fp8_e4m3>(val);
        }

        __nv_fp8_e4m3* out_ptr = output_q + bx * d + offset;
        if constexpr (VEC_SIZE >= 8) {
          static_assert(VEC_SIZE == 8, "VEC_SIZE > 8 not expected for BF16/FP16");
          *(uint2*)out_ptr = *(uint2*)fp8_arr;
        } else if constexpr (VEC_SIZE == 4) {
          *(uint32_t*)out_ptr = *(uint32_t*)fp8_arr;
        } else if constexpr (VEC_SIZE == 2) {
          *(uint16_t*)out_ptr = *(uint16_t*)fp8_arr;
        } else {
          for (uint32_t k = 0; k < VEC_SIZE; k++) {
            out_ptr[k] = fp8_arr[k];
          }
        }
      }
    }
  } else {
    // =================================================================
    // MULTI-ROUND: SMEM buffer fallback for exotic hidden sizes
    // =================================================================
    const uint32_t rounds = ROUNDS > 0 ? ROUNDS : ceil_div_dev(d, VEC_SIZE * num_threads);
    float* smem_x = smem + reduce_slots;

    float abs_max = 0.f;

#pragma unroll
    for (uint32_t i = 0; i < rounds; i++) {
      vec_t<T, VEC_SIZE> gate_vec;
      gate_vec.fill(0.f);
      vec_t<T, VEC_SIZE> up_vec;
      up_vec.fill(0.f);
      vec_t<float, VEC_SIZE> act_vec;
      act_vec.fill(0.f);

      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
      if (offset < d) {
        gate_vec.load(input + bx * 2 * d + offset);
        up_vec.load(input + bx * 2 * d + d + offset);
      }

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float gate_f = float(gate_vec[j]);
        float act = (gate_f / (1.0f + expf(-gate_f))) * float(up_vec[j]);
        abs_max = fmaxf(abs_max, fabsf(act));
        act_vec[j] = act;
      }

      if (offset < d) {
        act_vec.store(smem_x + offset);
      }
    }

    // Warp-level abs_max reduction
#pragma unroll
    for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
      abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, s));
    }

    // Cross-warp reduction
    smem_reduce[ty] = abs_max;
    __syncthreads();
    if (ty == 0) {
      abs_max = (tx < num_warps) ? smem_reduce[tx] : 0.f;
#pragma unroll
      for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, s));
      }
      smem_reduce[0] = abs_max;
    }
    __syncthreads();

    float scale = smem_reduce[0] / FP8_E4M3_MAX;
    float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

    if (thread_id == 0) {
      output_s[bx] = scale;
    }

    // Quantize from SMEM → FP8 output
#pragma unroll
    for (uint32_t i = 0; i < rounds; i++) {
      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;

      vec_t<float, VEC_SIZE> act_vec;
      act_vec.fill(0.f);
      if (offset < d) {
        act_vec.load(smem_x + offset);
      }

      __nv_fp8_e4m3 fp8_arr[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float val = fmaxf(fminf(act_vec[j] * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        fp8_arr[j] = static_cast<__nv_fp8_e4m3>(val);
      }

      if (offset < d) {
        __nv_fp8_e4m3* out_ptr = output_q + bx * d + offset;
        if constexpr (VEC_SIZE >= 8) {
          static_assert(VEC_SIZE == 8, "VEC_SIZE > 8 not expected for BF16/FP16");
          *(uint2*)out_ptr = *(uint2*)fp8_arr;
        } else if constexpr (VEC_SIZE == 4) {
          *(uint32_t*)out_ptr = *(uint32_t*)fp8_arr;
        } else if constexpr (VEC_SIZE == 2) {
          *(uint16_t*)out_ptr = *(uint16_t*)fp8_arr;
        } else {
          for (uint32_t k = 0; k < VEC_SIZE; k++) {
            out_ptr[k] = fp8_arr[k];
          }
        }
      }
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T>
__host__ inline uint32_t compute_vec_size(uint32_t d) {
  uint32_t type_vec = 16 / sizeof(T);
  return std::__gcd(type_vec, d);
}

}  // namespace

void sgl_fused_silu_mul_quant_fp8(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    bool enable_pdl) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  auto device = input.device();
  CHECK_EQ(output_q.device(), device);
  CHECK_EQ(output_s.device(), device);

  CHECK_DIM(2, input);     // (num_tokens, 2 * intermediate_size)
  CHECK_DIM(2, output_q);  // (num_tokens, intermediate_size)
  CHECK_DIM(2, output_s);  // (num_tokens, 1)

  uint32_t d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);

  CHECK_EQ(output_q.size(0), num_tokens);
  CHECK_EQ(static_cast<int64_t>(output_q.size(1)), static_cast<int64_t>(d));
  CHECK_EQ(output_s.size(0), num_tokens);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    const uint32_t vec_size = compute_vec_size<c_type>(d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = (block_size + 31) / 32;
    const uint32_t num_threads = num_warps * 32;
    const uint32_t rounds = (d + vec_size * num_threads - 1) / (vec_size * num_threads);
    dim3 nblks(num_tokens);
    dim3 nthrs(32, num_warps);

    // Shared memory: reduce_slots only for register-resident path (rounds<=4),
    // + d floats buffer for SMEM fallback path (rounds>4)
    const uint32_t reduce_slots = ((num_warps + 3) / 4) * 4;
    const uint32_t smem_size = (rounds <= 4)
        ? reduce_slots * sizeof(float)
        : (reduce_slots + d) * sizeof(float);

    // Configure launch with PDL support
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto launch_kernel = [&](auto kernel_fn) {
      TORCH_CHECK(
          cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess,
          "Failed to set max dynamic shared memory size");
      TORCH_CHECK(
          cudaLaunchKernelEx(
              &config,
              kernel_fn,
              static_cast<const c_type*>(input.data_ptr()),
              static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
              static_cast<float*>(output_s.data_ptr()),
              d) == cudaSuccess,
          "FusedSiLUMulQuantFP8 kernel launch failed");
    };

    // Dispatch with compile-time ROUNDS:
    //   ROUNDS=1 → register-resident, 1 round  (d ≤ 1*VEC_SIZE*num_threads)
    //   ROUNDS=2 → register-resident, 2 rounds (d ≤ 2*VEC_SIZE*num_threads)
    //   ROUNDS=3 → register-resident, 3 rounds (d ≤ 3*VEC_SIZE*num_threads)
    //   ROUNDS=4 → register-resident, 4 rounds (d ≤ 4*VEC_SIZE*num_threads)
    //   ROUNDS=0 → dynamic SMEM fallback       (d > 4*VEC_SIZE*num_threads)
    if (rounds == 1) {
      switch (vec_size) {
        case 8: launch_kernel(FusedSiLUMulQuantFP8Kernel<8, 1, c_type>); break;
        case 4: launch_kernel(FusedSiLUMulQuantFP8Kernel<4, 1, c_type>); break;
        case 2: launch_kernel(FusedSiLUMulQuantFP8Kernel<2, 1, c_type>); break;
        case 1: launch_kernel(FusedSiLUMulQuantFP8Kernel<1, 1, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    } else if (rounds == 2) {
      switch (vec_size) {
        case 8: launch_kernel(FusedSiLUMulQuantFP8Kernel<8, 2, c_type>); break;
        case 4: launch_kernel(FusedSiLUMulQuantFP8Kernel<4, 2, c_type>); break;
        case 2: launch_kernel(FusedSiLUMulQuantFP8Kernel<2, 2, c_type>); break;
        case 1: launch_kernel(FusedSiLUMulQuantFP8Kernel<1, 2, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    } else if (rounds == 3) {
      switch (vec_size) {
        case 8: launch_kernel(FusedSiLUMulQuantFP8Kernel<8, 3, c_type>); break;
        case 4: launch_kernel(FusedSiLUMulQuantFP8Kernel<4, 3, c_type>); break;
        case 2: launch_kernel(FusedSiLUMulQuantFP8Kernel<2, 3, c_type>); break;
        case 1: launch_kernel(FusedSiLUMulQuantFP8Kernel<1, 3, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    } else if (rounds == 4) {
      switch (vec_size) {
        case 8: launch_kernel(FusedSiLUMulQuantFP8Kernel<8, 4, c_type>); break;
        case 4: launch_kernel(FusedSiLUMulQuantFP8Kernel<4, 4, c_type>); break;
        case 2: launch_kernel(FusedSiLUMulQuantFP8Kernel<2, 4, c_type>); break;
        case 1: launch_kernel(FusedSiLUMulQuantFP8Kernel<1, 4, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    } else {
      switch (vec_size) {
        case 8: launch_kernel(FusedSiLUMulQuantFP8Kernel<8, 0, c_type>); break;
        case 4: launch_kernel(FusedSiLUMulQuantFP8Kernel<4, 0, c_type>); break;
        case 2: launch_kernel(FusedSiLUMulQuantFP8Kernel<2, 0, c_type>); break;
        case 1: launch_kernel(FusedSiLUMulQuantFP8Kernel<1, 0, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    }

    return true;
  });
}
