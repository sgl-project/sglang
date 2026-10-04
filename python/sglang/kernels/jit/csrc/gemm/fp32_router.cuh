// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from Inferact/lin fp32_router_gemm.cu.
// BF16 activations [M, 5120] x FP32 weights [256, 5120] -> FP32 [M, 256].

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang {

/// Load full-precision weights without a reduced-precision intermediate.
template <int VPT>
__device__ __forceinline__ void load_weight(float const* ptr, float* dst) {
#pragma unroll
  for (int v = 0; v < VPT / 4; ++v) {
    device::AlignedVector<float, 4> value;
    value.load(ptr, v);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      dst[v * 4 + i] = value[i];
  }
}

template <typename T, int VPT>
__device__ __forceinline__ void load_activation(T const* ptr, float* dst) {
  device::AlignedVector<T, VPT> value;
  value.load(ptr);
#pragma unroll
  for (int i = 0; i < VPT; ++i)
    dst[i] = static_cast<float>(value[i]);
}

// Each block computes EPB expert columns. Token groups share the same
// weight addresses but keep fewer accumulators per thread. Python chooses
// the geometry; the kernel retains FP32 weights and reduction arithmetic.
template <typename InputT, int kBlockSize, int kNumTokens, int kEPB, int kNumExperts, int kHiddenDim, int kTGroups = 1>
__global__ __launch_bounds__(kBlockSize* kTGroups, 1) void fp32_router_gemm_kernel(
    float* out, InputT const* mat_a, float const* mat_b) {
  constexpr int VPT = 16 / sizeof(InputT);
  constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
  constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration;
  static_assert(kHiddenDim % k_elems_per_k_iteration == 0);
  static_assert(kNumTokens % kTGroups == 0);
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = kBlockSize / kWarpSize;  // per token group
  constexpr int kMG = kNumTokens / kTGroups;         // tokens per group

  int const e_base = blockIdx.x * kEPB;
  int const tid = threadIdx.x % kBlockSize;
  int const m0 = (threadIdx.x / kBlockSize) * kMG;
  int const warpId = tid / kWarpSize;
  int const laneId = tid % kWarpSize;

  float acc[kMG][kEPB] = {};
  __shared__ float sm_reduction[kNumTokens][kEPB][kNumWarps];

  device::PDLWaitPrimary<true>();
  // A dependent reader must wait for grid completion before reading outputs.
  device::PDLTriggerSecondary<true>();

#pragma unroll
  for (int ki = 0; ki < k_iterations; ki++) {
    int const k_base = ki * k_elems_per_k_iteration + tid * VPT;

    float b_float[kEPB][VPT];
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      load_weight<VPT>(mat_b + (e_base + e) * kHiddenDim + k_base, b_float[e]);
    }

#pragma unroll
    for (int m_idx = 0; m_idx < kMG; m_idx++) {
      float a_float[VPT];
      load_activation<InputT, VPT>(mat_a + static_cast<size_t>(m0 + m_idx) * kHiddenDim + k_base, a_float);
#pragma unroll
      for (int e = 0; e < kEPB; e++) {
#pragma unroll
        for (int k = 0; k < VPT; k++) {
          acc[m_idx][e] += a_float[k] * b_float[e][k];
        }
      }
    }
  }

  // Warp-level butterfly reduction
#pragma unroll
  for (int m = 0; m < kMG; m++) {
#pragma unroll
    for (int e = 0; e < kEPB; e++) {
      float sum = acc[m][e];
      sum = device::warp::reduce_sum(sum);
      if (laneId == 0) sm_reduction[m0 + m][e][warpId] = sum;
    }
  }

  __syncthreads();

  // Parallel finalize: one thread per (m, e) output.
  for (int idx = threadIdx.x; idx < kNumTokens * kEPB; idx += kBlockSize * kTGroups) {
    int const m = idx / kEPB;
    int const e = idx % kEPB;
    float final_sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kNumWarps; w++)
      final_sum += sm_reduction[m][e][w];
    out[m * kNumExperts + e_base + e] = final_sum;
  }
}

template <int M, int Block, int EPB, int Groups>
struct Fp32RouterKernel {
  static_assert(M > 0 && M <= 64);
  static_assert(Block % 32 == 0 && Block * Groups <= 1024);
  static_assert(256 % EPB == 0);
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, 5120}).with_dtype<__nv_bfloat16>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({256, 5120}).with_dtype<float>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({M, 256}).with_dtype<float>().with_device(device).verify(out);
    LaunchKernel(256 / EPB, Block * Groups, device.unwrap())
        .enable_pdl(true)(
            fp32_router_gemm_kernel<__nv_bfloat16, Block, M, EPB, 256, 5120, Groups>,
            static_cast<float*>(out.data_ptr()),
            static_cast<const __nv_bfloat16*>(x.data_ptr()),
            static_cast<const float*>(w.data_ptr()));
  }
};
}  // namespace sglang
