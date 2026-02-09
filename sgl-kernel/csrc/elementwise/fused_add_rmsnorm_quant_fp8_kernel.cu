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

// Fused AddRMSNorm + Per-Token FP8 Quantization Kernel
//
// Eliminates the BF16 intermediate between fused_add_rmsnorm and
// per_token_quant_fp8. Instead of:
//   Kernel 1: input + residual → normalize → write BF16
//   Kernel 2: read BF16 → find max → quantize → write FP8
// We do:
//   Kernel:   input + residual → normalize → find max → quantize → write FP8
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

// Fused 2-pass kernel (was 3-pass): pass 2 keeps normalized values in registers
// across the abs_max reduction and quantizes directly, eliminating the third
// SMEM read+write pass. SMEM is still needed for pass 1→2 (x values before
// rms_rcp is known).
template <uint32_t VEC_SIZE, uint32_t ROUNDS, typename T>
__global__ void FusedAddRMSNormQuantFP8Kernel(
    const T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    const uint32_t d,
    const uint32_t stride_input,
    const uint32_t stride_residual,
    float eps) {
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
  float* smem_x = smem + reduce_slots;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  if constexpr (ROUNDS == 1) {
    // =================================================================
    // ROUNDS=1: register-optimized path (all standard hidden sizes)
    // Pass 1: load+add → SMEM + sum_sq
    // Pass 2: normalize from SMEM → register → abs_max → quantize from register
    // =================================================================
    const uint32_t offset = thread_id * VEC_SIZE;

    // --- Pass 1: Residual add + store x to smem + accumulate sum_sq ---
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    float x_reg[VEC_SIZE];

    float sum_sq = 0.f;

    if (offset < d) {
      input_vec.load(input + bx * stride_input + offset);
      residual_vec.load(residual + bx * stride_residual + offset);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]) + float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_reg[j] = x;
    }

    if (offset < d) {
      residual_vec.store(residual + bx * stride_residual + offset);
      // Store x to SMEM — needed after sum_sq reduction for normalize
      vec_t<float, VEC_SIZE> x_vec;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_vec[j] = x_reg[j];
      }
      x_vec.store(smem_x + offset);
    }

    // Reduce sum_sq: warp-level
#pragma unroll
    for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
      sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, s);
    }

    // Cross-warp reduction
    smem_reduce[ty] = sum_sq;
    __syncthreads();
    if (ty == 0) {
      sum_sq = (tx < num_warps) ? smem_reduce[tx] : 0.f;
#pragma unroll
      for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, s);
      }
      smem_reduce[0] = sum_sq;
    }
    __syncthreads();

    float rms_rcp = rsqrtf(smem_reduce[0] / float(d) + eps);

    // --- Pass 2: Normalize + abs_max + quantize (register-resident) ---
    vec_t<T, VEC_SIZE> weight_vec;
    weight_vec.fill(0.f);
    float norm_reg[VEC_SIZE];

    if (offset < d) {
      weight_vec.load(weight + offset);
      // Reload x from SMEM (avoids register pressure across sync barriers)
      vec_t<float, VEC_SIZE> x_vec;
      x_vec.load(smem_x + offset);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_reg[j] = x_vec[j];
      }
    }

    float abs_max = 0.f;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float norm = x_reg[j] * rms_rcp * float(weight_vec[j]);
      abs_max = fmaxf(abs_max, fabsf(norm));
      norm_reg[j] = norm;
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

    // Quantize directly from registers → FP8 store
    if (offset < d) {
      __nv_fp8_e4m3 fp8_arr[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float val = fmaxf(fminf(norm_reg[j] * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
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
  } else {
    // =================================================================
    // MULTI-ROUND: SMEM buffer fallback for exotic hidden sizes
    // =================================================================
    const uint32_t rounds = ROUNDS > 0 ? ROUNDS : ceil_div_dev(d, VEC_SIZE * num_threads);

    // --- Pass 1: Residual add + store x to smem + accumulate sum_sq ---
    float sum_sq = 0.f;

#pragma unroll
    for (uint32_t i = 0; i < rounds; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      input_vec.fill(0.f);
      vec_t<T, VEC_SIZE> residual_vec;
      residual_vec.fill(0.f);
      vec_t<float, VEC_SIZE> x_vec;
      x_vec.fill(0.f);

      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
      if (offset < d) {
        input_vec.load(input + bx * stride_input + offset);
        residual_vec.load(residual + bx * stride_residual + offset);
      }

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float x = float(input_vec[j]) + float(residual_vec[j]);
        sum_sq += x * x;
        residual_vec[j] = (T)x;
        x_vec[j] = x;
      }

      if (offset < d) {
        residual_vec.store(residual + bx * stride_residual + offset);
        x_vec.store(smem_x + offset);
      }
    }

    // Reduce sum_sq: warp-level
#pragma unroll
    for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
      sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, s);
    }

    // Cross-warp reduction
    smem_reduce[ty] = sum_sq;
    __syncthreads();
    if (ty == 0) {
      sum_sq = (tx < num_warps) ? smem_reduce[tx] : 0.f;
#pragma unroll
      for (uint32_t s = warp_size / 2; s > 0; s /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, s);
      }
      smem_reduce[0] = sum_sq;
    }
    __syncthreads();

    float rms_rcp = rsqrtf(smem_reduce[0] / float(d) + eps);

    // --- Pass 2: Normalize + abs_max (write normalized to SMEM) ---
    float abs_max = 0.f;

#pragma unroll
    for (uint32_t i = 0; i < rounds; i++) {
      vec_t<T, VEC_SIZE> weight_vec;
      weight_vec.fill(0.f);
      vec_t<float, VEC_SIZE> x_vec;
      x_vec.fill(0.f);

      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;
      if (offset < d) {
        weight_vec.load(weight + offset);
        x_vec.load(smem_x + offset);
      }

      vec_t<float, VEC_SIZE> norm_vec;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float norm = x_vec[j] * rms_rcp * float(weight_vec[j]);
        abs_max = fmaxf(abs_max, fabsf(norm));
        norm_vec[j] = norm;
      }

      if (offset < d) {
        norm_vec.store(smem_x + offset);
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

    // --- Pass 3: Quantize from SMEM → FP8 output ---
#pragma unroll
    for (uint32_t i = 0; i < rounds; i++) {
      const uint32_t offset = i * num_threads * VEC_SIZE + thread_id * VEC_SIZE;

      vec_t<float, VEC_SIZE> norm_vec;
      norm_vec.fill(0.f);
      if (offset < d) {
        norm_vec.load(smem_x + offset);
      }

      __nv_fp8_e4m3 fp8_arr[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float val = fmaxf(fminf(norm_vec[j] * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
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
  uint32_t type_vec = 16 / sizeof(T);  // max elements per 16-byte load
  return std::__gcd(type_vec, d);
}

}  // namespace

void sgl_fused_add_rmsnorm_quant_fp8(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor output_q,
    torch::Tensor output_s,
    double eps,
    bool enable_pdl) {
  CHECK_INPUT(input);
  CHECK_INPUT(residual);
  CHECK_INPUT(weight);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);

  auto device = input.device();
  CHECK_EQ(residual.device(), device);
  CHECK_EQ(weight.device(), device);
  CHECK_EQ(output_q.device(), device);
  CHECK_EQ(output_s.device(), device);

  CHECK_DIM(2, input);     // (num_tokens, hidden_size)
  CHECK_DIM(2, residual);  // (num_tokens, hidden_size)
  CHECK_DIM(1, weight);    // (hidden_size,)
  CHECK_DIM(2, output_q);  // (num_tokens, hidden_size)
  CHECK_DIM(2, output_s);  // (num_tokens, 1)

  CHECK_EQ(input.size(0), residual.size(0));
  CHECK_EQ(input.size(1), residual.size(1));
  CHECK_EQ(input.size(1), weight.size(0));
  CHECK_EQ(input.size(0), output_q.size(0));
  CHECK_EQ(input.size(1), output_q.size(1));
  CHECK_EQ(input.size(0), output_s.size(0));

  unsigned int batch_size = input.size(0);
  unsigned int d = input.size(1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    const uint32_t vec_size = compute_vec_size<c_type>(d);
    const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
    const uint32_t num_warps = (block_size + 31) / 32;
    const uint32_t num_threads = num_warps * 32;
    const uint32_t rounds = (d + vec_size * num_threads - 1) / (vec_size * num_threads);
    dim3 nblks(batch_size);
    dim3 nthrs(32, num_warps);

    // Shared memory: reduce_slots + d floats (SMEM buffer needed for x values
    // between pass 1 sum_sq reduction and pass 2 normalize, regardless of ROUNDS)
    const uint32_t reduce_slots = ((num_warps + 3) / 4) * 4;
    const uint32_t smem_size = (reduce_slots + d) * sizeof(float);

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
              static_cast<c_type*>(residual.data_ptr()),
              static_cast<const c_type*>(weight.data_ptr()),
              static_cast<__nv_fp8_e4m3*>(output_q.data_ptr()),
              static_cast<float*>(output_s.data_ptr()),
              d,
              static_cast<uint32_t>(input.stride(0)),
              static_cast<uint32_t>(residual.stride(0)),
              static_cast<float>(eps)) == cudaSuccess,
          "FusedAddRMSNormQuantFP8 kernel launch failed");
    };

    // Dispatch with compile-time ROUNDS:
    //   ROUNDS=1 → 2-pass (pass 2+3 merged via registers)
    //   ROUNDS=0 → 3-pass fallback with SMEM buffer
    if (rounds == 1) {
      switch (vec_size) {
        case 8: launch_kernel(FusedAddRMSNormQuantFP8Kernel<8, 1, c_type>); break;
        case 4: launch_kernel(FusedAddRMSNormQuantFP8Kernel<4, 1, c_type>); break;
        case 2: launch_kernel(FusedAddRMSNormQuantFP8Kernel<2, 1, c_type>); break;
        case 1: launch_kernel(FusedAddRMSNormQuantFP8Kernel<1, 1, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    } else {
      switch (vec_size) {
        case 8: launch_kernel(FusedAddRMSNormQuantFP8Kernel<8, 0, c_type>); break;
        case 4: launch_kernel(FusedAddRMSNormQuantFP8Kernel<4, 0, c_type>); break;
        case 2: launch_kernel(FusedAddRMSNormQuantFP8Kernel<2, 0, c_type>); break;
        case 1: launch_kernel(FusedAddRMSNormQuantFP8Kernel<1, 0, c_type>); break;
        default: TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
      }
    }

    return true;
  });
}
