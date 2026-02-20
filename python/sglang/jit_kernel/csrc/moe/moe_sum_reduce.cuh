// Adapt from sgl-kernel/csrc/moe/moe_sum_reduce.cu
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <type_traits>

using tvm::ffi::TensorView;

namespace {

// ---------------------------------------------------------------------------
// Accumulation helpers — always accumulate in float
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_acc(T x) {
  return static_cast<float>(x);
}
template <>
__device__ __forceinline__ float to_acc<__half>(__half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float to_acc<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_acc(float x) {
  return static_cast<T>(x);
}
template <>
__device__ __forceinline__ __half from_acc<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_acc<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

// ---------------------------------------------------------------------------
// Vectorised BF16 kernel — 16 elements (2 × uint4) per iteration
// Only instantiated when T == __nv_bfloat16
// ---------------------------------------------------------------------------
union Pack16B {
  uint4 v;
  __nv_bfloat16 u16[8];
};

template <int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_per_token_vec_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t topk_num,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  constexpr int VEC = 16;
  constexpr int PACKS = VEC / 8;

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  const int64_t n_chunks = hidden_dim / VEC;

  for (int64_t chunk = (int64_t)blockIdx.x * 32 + lane; chunk < n_chunks; chunk += (int64_t)gridDim.x * 32) {
    const int64_t d = chunk * VEC;
    const int64_t base = t * stride_token + d;

    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.f;

    for (int64_t k = 0; k < topk_num; ++k) {
#pragma unroll
      for (int p = 0; p < PACKS; ++p) {
        const int64_t offset = base + k * stride_topk + p * 8;
        Pack16B pack = {ldg_cg(reinterpret_cast<const uint4*>(x + offset))};
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[p * 8 + i] += __bfloat162float(pack.u16[i]);
      }
    }

#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] *= scale;

#pragma unroll
    for (int p = 0; p < PACKS; ++p) {
      Pack16B outp;
#pragma unroll
      for (int i = 0; i < 8; ++i) outp.u16[i] = __float2bfloat16_rn(acc[p * 8 + i]);
      *reinterpret_cast<uint4*>(y + t * out_stride_token + d + p * 8) = outp.v;
    }
  }
}

// ---------------------------------------------------------------------------
// Warp-per-token kernels with compile-time TOPK
// ---------------------------------------------------------------------------
template <typename T, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_topk(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    float acc = 0.f;
    const int64_t base = t * stride_token + d;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) acc += to_acc<T>(x[base + (int64_t)k * stride_topk]);
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<T>(acc);
  }
}

// ---------------------------------------------------------------------------
// Small-token kernels with compile-time TOPK
// ---------------------------------------------------------------------------
template <typename T, int TOPK>
__global__ void moe_sum_reduce_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; ++k) acc += to_acc<T>(x[base + (int64_t)k * stride_topk]);
      acc *= scale;
      y[t * out_stride_token + d] = from_acc<T>(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// General fallback kernels (runtime topk_num)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void moe_sum_reduce_kernel_general(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const float scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      float acc = 0.f;
#pragma unroll 1
      for (int k = 0; k < topk_num; ++k) acc += to_acc<T>(x[base + (int64_t)k * stride_topk]);
      acc *= scale;
      y[t * out_stride_token + d] = from_acc<T>(acc);
    }
  }
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_general(
    const T* __restrict__ x,
    T* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const float scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    float acc = 0.f;
    const int64_t base = t * stride_token + d;
#pragma unroll 1
    for (int k = 0; k < topk_num; ++k) acc += to_acc<T>(x[base + (int64_t)k * stride_topk]);
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<T>(acc);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host launcher (tvm-ffi interface)
// ---------------------------------------------------------------------------
template <typename T>
void moe_sum_reduce(TensorView input, TensorView output, double routed_scaling_factor) {
  using namespace host;

  // --- Input validation ---
  RuntimeCheck(input.dim() == 3, "input must be 3-D [token_num, topk_num, hidden_dim]");
  RuntimeCheck(output.dim() == 2, "output must be 2-D [token_num, hidden_dim]");
  RuntimeCheck(input.shape()[0] == output.shape()[0], "token dim mismatch");
  RuntimeCheck(input.shape()[2] == output.shape()[1], "hidden_dim mismatch");
  RuntimeCheck(input.is_contiguous(), "input must be contiguous");
  RuntimeCheck(output.is_contiguous(), "output must be contiguous");

  const int64_t token_num = input.shape()[0];
  const int64_t topk_num = input.shape()[1];
  const int64_t hidden_dim = input.shape()[2];

  // Contiguous strides
  const int64_t in_stride_token = topk_num * hidden_dim;
  const int64_t in_stride_topk = hidden_dim;
  const int64_t out_stride_token = hidden_dim;

  const float scale = static_cast<float>(routed_scaling_factor);

  const T* x_ptr = static_cast<const T*>(input.data_ptr());
  T* y_ptr = static_cast<T*>(output.data_ptr());

  cudaStream_t stream = LaunchKernel::resolve_device(input.device());

  // ---------------------------------------------------------------------------
  // Fast vectorised BF16 path: warp-per-token, 16-element loads
  // Only compiled/entered when T == __nv_bfloat16
  // ---------------------------------------------------------------------------
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if ((token_num > 256) && (hidden_dim % 8 == 0)) {
      constexpr int WARPS_PER_BLOCK = 8;
      constexpr int THREADS = WARPS_PER_BLOCK * 32;

      int64_t grid_x = (hidden_dim / 8 + 32 - 1) / 32;
      if (grid_x > 65535) grid_x = 65535;
      int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      if (grid_y > 65535) grid_y = 65535;

      moe_sum_reduce_warp_per_token_vec_kernel<WARPS_PER_BLOCK>
          <<<dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y)), dim3(THREADS), 0, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(x_ptr),
              reinterpret_cast<__nv_bfloat16*>(y_ptr),
              token_num,
              hidden_dim,
              topk_num,
              in_stride_token,
              in_stride_topk,
              out_stride_token,
              scale);
      return;
    }
  }

  const bool per_token_use_one_warp = (token_num > 128);

  if (!per_token_use_one_warp) {
    // -------------------------------------------------------------------------
    // Small-token path: one CTA covers many tokens
    // -------------------------------------------------------------------------
    const int block_size = 256;
    int64_t grid_x = (hidden_dim + block_size - 1) / block_size;
    if (grid_x > 65535) grid_x = 65535;
    int64_t grid_y = token_num < 65535 ? token_num : 65535;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

#define LAUNCH_SMALL(TOPK)                                                                                 \
  moe_sum_reduce_kernel<T, TOPK><<<grid, block, 0, stream>>>(x_ptr, y_ptr, token_num, hidden_dim,         \
                                                              in_stride_token, in_stride_topk,             \
                                                              out_stride_token, scale);

    switch (topk_num) {
      case 2:
        LAUNCH_SMALL(2);
        break;
      case 4:
        LAUNCH_SMALL(4);
        break;
      case 8:
        LAUNCH_SMALL(8);
        break;
      case 9:
        LAUNCH_SMALL(9);
        break;
      default:
        moe_sum_reduce_kernel_general<T><<<grid, block, 0, stream>>>(
            x_ptr, y_ptr, token_num, hidden_dim, in_stride_token, in_stride_topk, out_stride_token,
            static_cast<int>(topk_num), scale);
    }
#undef LAUNCH_SMALL

  } else {
    // -------------------------------------------------------------------------
    // Warp-per-token path: one warp handles one token
    // -------------------------------------------------------------------------
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int64_t gx = (hidden_dim + 32 - 1) / 32;
    if (gx > 65535) gx = 65535;
    int64_t gy = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (gy > 65535) gy = 65535;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));

#define LAUNCH_WARP(TOPK)                                                                                         \
  moe_sum_reduce_kernel_warp_token_topk<T, TOPK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(                   \
      x_ptr, y_ptr, token_num, hidden_dim, in_stride_token, in_stride_topk, out_stride_token, scale);

    switch (topk_num) {
      case 2:
        LAUNCH_WARP(2);
        break;
      case 4:
        LAUNCH_WARP(4);
        break;
      case 8:
        LAUNCH_WARP(8);
        break;
      case 9:
        LAUNCH_WARP(9);
        break;
      default:
        moe_sum_reduce_kernel_warp_token_general<T, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            x_ptr, y_ptr, token_num, hidden_dim, in_stride_token, in_stride_topk, out_stride_token,
            static_cast<int>(topk_num), scale);
    }
#undef LAUNCH_WARP
  }
}
