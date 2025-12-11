#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 900
#endif

#include <cooperative_groups.h>

#include <cublasdx.hpp>
#include <cuda/atomic>
#include <cusolverdx.hpp>

#ifdef WITH_NVSHMEM
#undef uint64_t
#define uint64_t unsigned long
#include <device/nvshmem_defines.h>
#include <device/nvshmemx_defines.h>
#endif

namespace cg = cooperative_groups;

// External Definitions Required:
// - SM_Ver
// - BLOCK_DIM
// - NC
// - NV

template <int N>
__device__ void gaussian_elimination_solve(float a[N][N], float b[N]) {
  int status;
  decltype(cusolverdx::Size<N>() + cusolverdx::Function<cusolverdx::function::posv>() + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>() + cusolverdx::SM<SM_Ver>() + cusolverdx::Block() + cusolverdx::FillMode<cusolverdx::lower>() + cusolverdx::BlockDim<BLOCK_DIM>())()
      .execute(a[0], b, &status);
}

template <int M, int N, int K>
__device__ void matmulNT(float* a, float* b, float* c) {
  decltype(cublasdx::Size<M, N, K>() + cublasdx::Function<cublasdx::function::MM>() + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() + cublasdx::SM<SM_Ver>() + cublasdx::Block() + cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int M, int N, int K>
__device__ void matmulNN(float* a, float* b, float* c) {
  decltype(cublasdx::Size<M, N, K>() + cublasdx::Function<cublasdx::function::MM>() + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::SM<SM_Ver>() + cublasdx::Block() + cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

struct smem_variables {
  float b[NC];
  float a[NC][NV];
  float c[NV];
  float ax2[NC][NV];
  float ax2a[NC][NC];
  float x[NV];
  float ax2c[NC];
  float r[NV];
  float d[NV];
  float alpha;
  bool avail_flag;
};

extern "C" __global__ void get_smem_size(int* size_output) {
  *size_output = sizeof(smem_variables);
}

extern "C" __global__ void kernel_solve(int* avail_num, float* result, float* input_a, float* input_b, float* input_c) {
  extern __shared__ smem_variables smem[];

  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int dim = blockDim.x;

  // Load matrices from input into shared memory
  auto&& a = smem->a;
  auto&& b = smem->b;
  auto&& c = smem->c;

  // Copy matrix a from input
  for (int i = tid; i < NC * NV; i += dim) {
    int ic = i / NV, iv = i % NV;
    a[ic][iv] = input_a[pid * NC * NV + i];
  }
  __syncthreads();

  // Copy matrix b from input
  for (int i = tid; i < NC; i += dim) {
    b[i] = input_b[pid * NC + i];
  }
  __syncthreads();

  // Copy matrix c from input
  for (int i = tid; i < NV; i += dim) {
    c[i] = input_c[pid * NV + i];
  }
  __syncthreads();

  auto&& ax2 = smem->ax2;
  auto&& ax2a = smem->ax2a;
  auto&& x = smem->x;
  auto&& ax2c = smem->ax2c;
  auto&& r = smem->r;
  auto&& d = smem->d;
  auto&& alpha = smem->alpha;
  float d_max, max_residual;  // 在第一个 warp 可用
  for (int j = tid; j < NV; j += dim)
    x[j] = 1;
  __syncthreads();

  for (int step = 0; step < 5; step++) {
    for (int ij = tid; ij < NC * NV; ij += dim) {
      int i = ij / NV, j = ij % NV;
      ax2[i][j] = a[i][j] * x[j] * x[j];
    }
    __syncthreads();

    // ax2a = ax2 @ a.T
    matmulNT<NC, NC, NV>(ax2[0], a[0], ax2a[0]);

    // ax2c = ax2 @ c.unsqueeze(1)
    matmulNT<NC, 1, NV>(ax2[0], c, ax2c);

    // solve(ax2a, ax2c);
    gaussian_elimination_solve<NC>(ax2a, ax2c);

    // r = ax2c.unsqueeze(0) @ a
    matmulNN<1, NV, NC>(ax2c, a[0], r);

    if (tid < 32) {
      d_max = 0.f;
      for (int j = tid; j < NV; j += 32)
        d_max = d[j] = x[j] * (c[j] - r[j]);
      for (int offset = 16; offset > 0; offset >>= 1)
        d_max = fmaxf(d_max, __shfl_xor_sync(0xffffffff, d_max, offset));
      if (tid == 0) alpha = 0.999 / d_max;
    }
    __syncthreads();
    for (int j = tid; j < NV; j += dim)
      x[j] *= 1 - alpha * d[j];
    __syncthreads();
  }

  // ax2c -> residual
  matmulNT<NC, 1, NV>(a[0], x, ax2c);
  if (tid < 32) {
    max_residual = 0;
    for (int i = tid; i < NC; i += 32)
      max_residual = fmaxf(max_residual, fabsf(ax2c[i] - b[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
      max_residual = fmaxf(max_residual, __shfl_down_sync(0xffffffff, max_residual, offset));
  }
  auto&& avail_flag = smem->avail_flag;
  if (tid == 0) {
    avail_flag = (d_max < 0.1 && 0 <= x[NV - 1] && x[NV - 1] < 1e-4 && max_residual < 0.05);
    atomicAdd(avail_num, (int)avail_flag);
  }
  __syncthreads();
#ifdef DEBUG_DUMP
  if (tid == 0 && pid == 0) {
    // print all of x
    printf("x:\n");
    for (int iv = 0; iv < NV; iv++)
      printf("%.2f ", x[iv]);
    printf("\n");
  }
#endif
  if (!avail_flag)
    for (int i = tid; i < NC; i += dim)
      result[i] = 0.5;
  else
    for (int i = tid; i < NC; i += dim)
      result[i] = x[i];
  __syncthreads();
}
