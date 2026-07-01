// Single-SM Interior Point Method (IPM) LP solver.
//
// Solves min c^T x subject to A x = b, x >= 0 with a barrier method:
//   for step in 0..NUM_ITERS:
//     ax2  = A * x^2
//     ax2a = ax2 @ A^T          (cuBLASDx GEMM)
//     ax2c = ax2 @ c            (cuBLASDx GEMM)
//     d    = solve(ax2a, ax2c)  (cuSolverDx Cholesky/POSV)
//     r    = d^T @ A
//     d    = x * (c - r)
//     x   *= 1 - 0.999 * d / max(d)
//
// Convergence is checked at the end and the kernel writes 0.5 to every
// element on non-convergence (matches the historical Numba behavior).
//
// Adapted from DeepSeek-AI/LPLB's `minilp.cu`. Templated on (NC, NV,
// BLOCK_DIM, SM_VER, NUM_ITERS) so each unique shape is compiled once via
// sglang's tvm-ffi load_jit cache.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cublasdx.hpp>
#include <cusolverdx.hpp>

namespace {

template <int NC, int NV>
struct ipm_smem {
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
  int solve_status;
  bool avail_flag;
};

template <int N, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void posv_solve(float a[N][N], float b[N], int* status) {
  if (threadIdx.x == 0) {
    *status = 0;
  }
  __syncthreads();

  using Posv = decltype(cusolverdx::Size<N, N, 1>() +
                        cusolverdx::Function<cusolverdx::function::posv>() +
                        cusolverdx::Precision<float>() +
                        cusolverdx::Type<cusolverdx::type::real>() +
                        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
                        cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::col_major>() +
                        cusolverdx::SM<SM_VER>() +
                        cusolverdx::Block() +
                        cusolverdx::BlockDim<BLOCK_DIM>() +
                        cusolverdx::BatchesPerBlock<1>());

  Posv().execute(a[0], /*runtime_lda=*/N, b, /*runtime_ldb=*/N, status);
  __syncthreads();

  if (*status != 0) {
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM) {
      b[i] = __int_as_float(0x7fffffff);
    }
  }
  __syncthreads();
}

template <int M, int N, int K, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void matmul_NT(float* a, float* b, float* c) {
  decltype(cublasdx::Size<M, N, K>() + cublasdx::Function<cublasdx::function::MM>() + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() + cublasdx::SM<SM_VER>() + cublasdx::Block() + cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int M, int N, int K, int SM_VER, int BLOCK_DIM>
__device__ __forceinline__ void matmul_NN(float* a, float* b, float* c) {
  decltype(cublasdx::Size<M, N, K>() + cublasdx::Function<cublasdx::function::MM>() + cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major>() + cublasdx::SM<SM_VER>() + cublasdx::Block() + cublasdx::BlockDim<BLOCK_DIM>())()
      .execute(1.f, a, b, 0.f, c);
}

template <int NC, int NV, int BLOCK_DIM, int SM_VER, int NUM_ITERS>
__global__ void ipm_solve_kernel(
    float* __restrict__ result,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    const float* __restrict__ input_c) {
  using SMem = ipm_smem<NC, NV>;
  extern __shared__ unsigned char raw_smem[];
  SMem* smem = reinterpret_cast<SMem*>(raw_smem);

  const int tid = threadIdx.x;
  const int dim = blockDim.x;

  auto& a = smem->a;
  auto& b = smem->b;
  auto& c = smem->c;

  // Load A, b, c into shared memory (single block, no grid index).
  for (int i = tid; i < NC * NV; i += dim) {
    int ic = i / NV, iv = i % NV;
    a[ic][iv] = input_a[i];
  }
  __syncthreads();
  for (int i = tid; i < NC; i += dim) {
    b[i] = input_b[i];
  }
  __syncthreads();
  for (int i = tid; i < NV; i += dim) {
    c[i] = input_c[i];
  }
  __syncthreads();

  auto& ax2 = smem->ax2;
  auto& ax2a = smem->ax2a;
  auto& x = smem->x;
  auto& ax2c = smem->ax2c;
  auto& r = smem->r;
  auto& d = smem->d;
  auto& alpha = smem->alpha;
  // d_max and max_residual are warp-scope reductions; only valid for tid<32.
  float d_max = 0.f;
  float max_residual = 0.f;

  for (int j = tid; j < NV; j += dim) {
    x[j] = 1.f;
  }
  __syncthreads();

  for (int step = 0; step < NUM_ITERS; step++) {
    for (int ij = tid; ij < NC * NV; ij += dim) {
      int i = ij / NV, j = ij % NV;
      ax2[i][j] = a[i][j] * x[j] * x[j];
    }
    __syncthreads();

    matmul_NT<NC, NC, NV, SM_VER, BLOCK_DIM>(ax2[0], a[0], ax2a[0]);
    matmul_NT<NC, 1, NV, SM_VER, BLOCK_DIM>(ax2[0], c, ax2c);
    posv_solve<NC, SM_VER, BLOCK_DIM>(ax2a, ax2c, &smem->solve_status);
    matmul_NN<1, NV, NC, SM_VER, BLOCK_DIM>(ax2c, a[0], r);

    if (tid < 32) {
      d_max = 0.f;
      for (int j = tid; j < NV; j += 32) {
        float val = x[j] * (c[j] - r[j]);
        d[j] = val;
        d_max = fmaxf(d_max, val);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        d_max = fmaxf(d_max, __shfl_xor_sync(0xffffffff, d_max, offset));
      }
      if (tid == 0) {
        // Guard against d_max <= 0 from a degenerate / numerically-stuck
        // iteration. A non-positive d_max would yield inf/NaN from the
        // division and corrupt x on the next update. The 1.0 fallback
        // produces a no-op step (x *= 1 - 1*0 = x) so the solver simply
        // stalls rather than diverges, and the convergence check at the
        // end of the kernel writes 0.5 if d_max stays small.
        alpha = (d_max > 1e-9f) ? (0.999f / d_max) : 1.0f;
      }
    }
    __syncthreads();

    for (int j = tid; j < NV; j += dim) {
      x[j] *= 1.f - alpha * d[j];
    }
    __syncthreads();
  }

  // Compute residual ‖A x - b‖_inf for the convergence check.
  matmul_NT<NC, 1, NV, SM_VER, BLOCK_DIM>(a[0], x, ax2c);
  if (tid < 32) {
    max_residual = 0.f;
    for (int i = tid; i < NC; i += 32) {
      max_residual = fmaxf(max_residual, fabsf(ax2c[i] - b[i]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      max_residual = fmaxf(max_residual, __shfl_down_sync(0xffffffff, max_residual, offset));
    }
  }

  auto& avail_flag = smem->avail_flag;
  if (tid == 0) {
    avail_flag = (d_max < 0.1f && x[NV - 1] >= 0.f && x[NV - 1] < 1e-4f && max_residual < 0.05f);
  }
  __syncthreads();

  if (!avail_flag) {
    for (int i = tid; i < NV; i += dim) {
      result[i] = 0.5f;
    }
  } else {
    for (int i = tid; i < NV; i += dim) {
      result[i] = x[i];
    }
  }
}

template <int NC, int NV, int BLOCK_DIM, int SM_VER, int NUM_ITERS>
void ipm_solve(tvm::ffi::TensorView A, tvm::ffi::TensorView b, tvm::ffi::TensorView c, tvm::ffi::TensorView result) {
  using namespace host;

  SymbolicDevice device_;
  TensorMatcher({NC, NV}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(A);
  TensorMatcher({NC}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(b);
  TensorMatcher({NV}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(c);
  TensorMatcher({NV}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(result);

  const DLDevice device = device_.unwrap();
  const size_t smem_bytes = sizeof(ipm_smem<NC, NV>);

  using KernelT = void (*)(float*, const float*, const float*, const float*);
  KernelT kernel = ipm_solve_kernel<NC, NV, BLOCK_DIM, SM_VER, NUM_ITERS>;

  // Opt in to >48 KB dynamic shared memory if needed (Hopper supports up to
  // 228 KB per block).
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  }

  LaunchKernel(/*grid_dim=*/1, /*block_dim=*/BLOCK_DIM, device, smem_bytes)(
      kernel,
      static_cast<float*>(result.data_ptr()),
      static_cast<const float*>(A.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<const float*>(c.data_ptr()));
}

}  // namespace
