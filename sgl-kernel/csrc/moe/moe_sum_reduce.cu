#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <iostream>
#include <type_traits>

#include "cutlass/array.h"
#include "utils.h"

template <typename T>
using opmath_t = at::opmath_type<T>;

template <typename T>
__device__ __forceinline__ opmath_t<T> to_acc(T x) {
  return static_cast<opmath_t<T>>(x);
}

template <typename T>
__device__ __forceinline__ T from_acc(opmath_t<T> x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ opmath_t<at::Half> to_acc<at::Half>(at::Half x) {
  return __half2float(__nv_half(x));
}
template <>
__device__ __forceinline__ at::Half from_acc<at::Half>(opmath_t<at::Half> x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ opmath_t<at::BFloat16> to_acc<at::BFloat16>(at::BFloat16 x) {
  return __bfloat162float(__nv_bfloat16(x));
}
template <>
__device__ __forceinline__ at::BFloat16 from_acc<at::BFloat16>(opmath_t<at::BFloat16> x) {
  return __float2bfloat16_rn(x);
}

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

union Pack16B {
  uint4 v;
  __nv_bfloat16 u16[8];
};

template <int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_per_token_vec_kernel(
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t topk_num,
    const int64_t stride_token,      // in elements
    const int64_t stride_topk,       // in elements
    const int64_t out_stride_token,  // in elements
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
    for (int i = 0; i < VEC; ++i)
      acc[i] = 0.f;

#pragma unroll
    for (int k = 0; k < topk_num; ++k) {
#pragma unroll
      for (int p = 0; p < PACKS; ++p) {
        const int64_t offset = base + (int64_t)k * stride_topk + p * 8;
        Pack16B pack = {ldg_cg(reinterpret_cast<const uint4*>(x + offset))};

#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[p * 8 + i] += __bfloat162float(pack.u16[i]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] *= scale;

#pragma unroll
    for (int p = 0; p < PACKS; ++p) {
      Pack16B outp;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        outp.u16[i] = __float2bfloat16_rn(acc[p * 8 + i]);
      }
      const int64_t dst = t * out_stride_token + d + p * 8;
      *reinterpret_cast<uint4*>(y + dst) = outp.v;
    }
  }
}

template <typename scalar_t, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_topk(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_reduce_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);

#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }

      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

// -------------------- general-topk fallback kernels --------------------
// small-token
template <typename scalar_t>
__global__ void moe_sum_reduce_kernel_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
#pragma unroll 1
      for (int k = 0; k < topk_num; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }
      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

// warp-per-token
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;
#pragma unroll 1
    for (int k = 0; k < topk_num; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

void moe_sum_reduce(at::Tensor& input, at::Tensor& output, double routed_scaling_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor like [token_num, topk_num, hidden_dim]");
  TORCH_CHECK(output.dim() == 2, "output must be [token_num, hidden_dim]");
  TORCH_CHECK(input.size(0) == output.size(0), "token dim mismatch");
  TORCH_CHECK(input.size(2) == output.size(1), "hidden_dim mismatch");

  TORCH_CHECK(input.is_contiguous(), "expect input to be contiguous");
  TORCH_CHECK(output.is_contiguous(), "expect output to be contiguous");

  const int64_t token_num = input.size(0);
  const int64_t topk_num = input.size(1);
  const int64_t hidden_dim = input.size(2);

  const int64_t in_stride_token = input.stride(0);
  const int64_t in_stride_topk = input.stride(1);
  const int64_t out_stride_token = output.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  const bool fast_bf16_vec_ok = (input.scalar_type() == at::kBFloat16) && (token_num > 256) && (hidden_dim % 8 == 0);

  // Fast path for bf16 vectorize
  if (fast_bf16_vec_ok) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    const int64_t n_chunks = hidden_dim / 8;
    int64_t grid_x = (n_chunks + 32 - 1) / 32;
    if (grid_x > 65535) grid_x = 65535;

    int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (grid_y > 65535) grid_y = 65535;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    auto stream = at::cuda::getCurrentCUDAStream();

    const float scale = static_cast<float>(routed_scaling_factor);
    moe_sum_reduce_warp_per_token_vec_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::BFloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<at::BFloat16*>(output.data_ptr<at::BFloat16>()),
        token_num,
        hidden_dim,
        topk_num,
        in_stride_token,
        in_stride_topk,
        out_stride_token,
        scale);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (bf16 vec) launch failed");
    return;
  }

  const bool per_token_use_one_warp = (token_num > 128);

  if (!per_token_use_one_warp) {
    // ---------- small-token ----------
    const int block_size = 256;
    int64_t grid_x = (hidden_dim + block_size - 1) / block_size;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num < 65535 ? token_num : 65535;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

#define LAUNCH_SMALL_TOKEN_KERNEL(TOPK)                               \
  moe_sum_reduce_kernel<scalar_t_, TOPK><<<grid, block, 0, stream>>>( \
      input.data_ptr<scalar_t_>(),                                    \
      output.data_ptr<scalar_t_>(),                                   \
      token_num,                                                      \
      hidden_dim,                                                     \
      in_stride_token,                                                \
      in_stride_topk,                                                 \
      out_stride_token,                                               \
      scale);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "moe_sum_reduce_cuda_small_token", [&] {
          using scalar_t_ = scalar_t;
          using acc_t_ = opmath_t<scalar_t_>;
          const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

          switch (topk_num) {
            case 2:
              LAUNCH_SMALL_TOKEN_KERNEL(2);
              break;
            case 4:
              LAUNCH_SMALL_TOKEN_KERNEL(4);
              break;
            case 8:
              LAUNCH_SMALL_TOKEN_KERNEL(8);
              break;
            case 9:
              LAUNCH_SMALL_TOKEN_KERNEL(9);
              break;
            default:  // launch general kernel
              moe_sum_reduce_kernel_general<scalar_t_><<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t_>(),
                  output.data_ptr<scalar_t_>(),
                  token_num,
                  hidden_dim,
                  in_stride_token,
                  in_stride_topk,
                  out_stride_token,
                  static_cast<int>(topk_num),
                  scale);
          }
        });
#undef LAUNCH_SMALL_TOKEN_KERNEL

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (small-token) launch failed");

  } else {
    // ---------- warp-per-token ----------
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int64_t gx = (hidden_dim + 32 - 1) / 32;
    gx = gx > 65535 ? 65535 : gx;

    int64_t gy = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gy = gy > 65535 ? 65535 : gy;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));

#define LAUNCH_WARP_PER_TOKEN_KERNEL(TOPK)                                                             \
  moe_sum_reduce_kernel_warp_token_topk<scalar_t_, TOPK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>( \
      input.data_ptr<scalar_t_>(),                                                                     \
      output.data_ptr<scalar_t_>(),                                                                    \
      token_num,                                                                                       \
      hidden_dim,                                                                                      \
      in_stride_token,                                                                                 \
      in_stride_topk,                                                                                  \
      out_stride_token,                                                                                \
      scale);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "moe_sum_reduce_cuda_large_token", [&] {
          using scalar_t_ = scalar_t;
          using acc_t_ = opmath_t<scalar_t_>;
          const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

          switch (topk_num) {
            case 2:
              LAUNCH_WARP_PER_TOKEN_KERNEL(2);
              break;
            case 4:
              LAUNCH_WARP_PER_TOKEN_KERNEL(4);
              break;
            case 8:
              LAUNCH_WARP_PER_TOKEN_KERNEL(8);
              break;
            case 9:
              LAUNCH_WARP_PER_TOKEN_KERNEL(9);
              break;
            default:  // launch general kernel
              moe_sum_reduce_kernel_warp_token_general<scalar_t_, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t_>(),
                  output.data_ptr<scalar_t_>(),
                  token_num,
                  hidden_dim,
                  in_stride_token,
                  in_stride_topk,
                  out_stride_token,
                  static_cast<int>(topk_num),
                  scale);
          }
        });
#undef LAUNCH_WARP_PER_TOKEN_KERNEL

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (warp-token) launch failed");
  }
}
