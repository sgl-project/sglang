// Single-block Interior Point Method (IPM) LP solver — HIP/ROCm port.
//
// ROCm has no cuBLASDx (the device-side header-only GEMM that the CUDA
// kernel in ``ipm.cuh`` uses), so this variant keeps the exact same
// barrier-method IPM but does the tiny GEMMs with hand-written shared-memory
// block GEMMs. Everything runs on-chip in one kernel per solve, which makes
// it CUDA-graph-capturable (unlike rocSOLVER's dense factorizations, whose
// mid-call workspace allocation is illegal during HIP stream capture).
//
//   for step in 0..NUM_ITERS:
//     ax2  = A * x^2
//     ax2a = ax2 @ A^T           (gemm_ABt)
//     ax2c = ax2 @ c             (gemm_ABt, N=1)
//     d    = solve(ax2a, ax2c)   (hand-written block Cholesky / POSV)
//     r    = ax2c @ A            (gemm_AB, M=1)
//     d    = x * (c - r)
//     x   *= 1 - 0.999 * d / max(d)
//
// On non-convergence the kernel writes 0.5 to every element (matches
// ipm.cuh / the historical Numba behavior).
//
// Reductions use a single thread (NC/NV are <= a few dozen, dwarfed by the
// GEMMs) rather than warp shuffles, so the kernel is wavefront-size agnostic
// (AMD wavefronts are 64 wide, not 32).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template <int NC, int NV>
struct ipm_hip_smem {
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
  float d_max;
  float max_residual;
  bool avail_flag;
};

// In-place Cholesky factorization a = L L^T (lower triangle). Identical to
// the CUDA version — plain C++, no external linkage.
template <int N, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_factor(float a[N][N]) {
  const int tid = threadIdx.x;
  for (int k = 0; k < N; k++) {
    if (tid == 0) {
      // Clamp the pivot away from zero before sqrtf; IPM drift can push a
      // diagonal slightly negative on an otherwise-PSD matrix.
      a[k][k] = sqrtf(fmaxf(a[k][k], 1e-12f));
    }
    __syncthreads();
    const float pivot = a[k][k];
    for (int i = k + 1 + tid; i < N; i += BLOCK_DIM) {
      a[i][k] /= pivot;
    }
    __syncthreads();
    for (int idx = tid; idx < N * N; idx += BLOCK_DIM) {
      const int i = idx / N, j = idx % N;
      if (j > k && i >= j && i < N) {
        a[i][j] -= a[i][k] * a[j][k];
      }
    }
    __syncthreads();
  }
}

// Solve L L^T x = b in-place on b (single thread; N is small and the
// substitutions carry loop dependencies).
template <int N, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_apply(const float a[N][N], float b[N]) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < N; i++) {
      float s = b[i];
      for (int j = 0; j < i; j++) {
        s -= a[i][j] * b[j];
      }
      b[i] = s / a[i][i];
    }
    for (int i = N - 1; i >= 0; i--) {
      float s = b[i];
      for (int j = i + 1; j < N; j++) {
        s -= a[j][i] * b[j];
      }
      b[i] = s / a[i][i];
    }
  }
  __syncthreads();
}

template <int N, int BLOCK_DIM>
__device__ __forceinline__ void cholesky_solve(float a[N][N], float b[N]) {
  cholesky_factor<N, BLOCK_DIM>(a);
  cholesky_apply<N, BLOCK_DIM>(a, b);
}

// out[m][n] = sum_k a[m][k] * b[n][k]   (a is MxK, b is NxK — "A @ B^T")
template <int M, int N, int K, int BLOCK_DIM>
__device__ __forceinline__ void gemm_ABt(const float* a, const float* b, float* out) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < M * N; idx += BLOCK_DIM) {
    const int m = idx / N, n = idx % N;
    float s = 0.f;
    for (int k = 0; k < K; k++) {
      s += a[m * K + k] * b[n * K + k];
    }
    out[idx] = s;
  }
  __syncthreads();
}

// out[m][n] = sum_k a[m][k] * b[k][n]   (a is MxK, b is KxN — "A @ B")
template <int M, int N, int K, int BLOCK_DIM>
__device__ __forceinline__ void gemm_AB(const float* a, const float* b, float* out) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < M * N; idx += BLOCK_DIM) {
    const int m = idx / N, n = idx % N;
    float s = 0.f;
    for (int k = 0; k < K; k++) {
      s += a[m * K + k] * b[k * N + n];
    }
    out[idx] = s;
  }
  __syncthreads();
}

template <int NC, int NV, int BLOCK_DIM, int NUM_ITERS>
__global__ void ipm_hip_solve_kernel(
    float* __restrict__ result,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    const float* __restrict__ input_c) {
  using SMem = ipm_hip_smem<NC, NV>;
  extern __shared__ unsigned char raw_smem[];
  SMem* smem = reinterpret_cast<SMem*>(raw_smem);

  const int tid = threadIdx.x;
  const int dim = blockDim.x;

  auto& a = smem->a;
  auto& b = smem->b;
  auto& c = smem->c;

  for (int i = tid; i < NC * NV; i += dim) {
    a[i / NV][i % NV] = input_a[i];
  }
  for (int i = tid; i < NC; i += dim) {
    b[i] = input_b[i];
  }
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
  auto& d_max = smem->d_max;
  auto& max_residual = smem->max_residual;

  for (int j = tid; j < NV; j += dim) {
    x[j] = 1.f;
  }
  __syncthreads();

  for (int step = 0; step < NUM_ITERS; step++) {
    for (int ij = tid; ij < NC * NV; ij += dim) {
      const int i = ij / NV, j = ij % NV;
      ax2[i][j] = a[i][j] * x[j] * x[j];
    }
    __syncthreads();

    gemm_ABt<NC, NC, NV, BLOCK_DIM>(ax2[0], a[0], ax2a[0]);
    gemm_ABt<NC, 1, NV, BLOCK_DIM>(ax2[0], c, ax2c);
    cholesky_solve<NC, BLOCK_DIM>(ax2a, ax2c);
    gemm_AB<1, NV, NC, BLOCK_DIM>(ax2c, a[0], r);

    for (int j = tid; j < NV; j += dim) {
      d[j] = x[j] * (c[j] - r[j]);
    }
    __syncthreads();

    if (tid == 0) {
      float dm = 0.f;
      for (int j = 0; j < NV; j++) {
        dm = fmaxf(dm, d[j]);
      }
      d_max = dm;
      // 1.0 fallback on a non-positive d_max makes the step a no-op so the
      // solver stalls rather than diverging; the end-of-kernel convergence
      // check then writes 0.5.
      alpha = (dm > 1e-9f) ? (0.999f / dm) : 1.0f;
    }
    __syncthreads();

    for (int j = tid; j < NV; j += dim) {
      x[j] *= 1.f - alpha * d[j];
    }
    __syncthreads();
  }

  // Residual ‖A x - b‖_inf for the convergence check (reuse ax2c as scratch).
  gemm_ABt<NC, 1, NV, BLOCK_DIM>(a[0], x, ax2c);
  if (tid == 0) {
    float mr = 0.f;
    for (int i = 0; i < NC; i++) {
      mr = fmaxf(mr, fabsf(ax2c[i] - b[i]));
    }
    max_residual = mr;
    smem->avail_flag = (d_max < 0.1f && x[NV - 1] >= 0.f && x[NV - 1] < 1e-4f && mr < 0.05f);
  }
  __syncthreads();

  const bool ok = smem->avail_flag;
  for (int i = tid; i < NV; i += dim) {
    result[i] = ok ? x[i] : 0.5f;
  }
}

template <int NC, int NV, int BLOCK_DIM, int NUM_ITERS>
void ipm_hip_solve(
    tvm::ffi::TensorView A, tvm::ffi::TensorView b, tvm::ffi::TensorView c, tvm::ffi::TensorView result) {
  using namespace host;

  // HIP-safe device match: set_options<kDLCUDA>() resolves to the ROCm device
  // kind under a HIP build, and LaunchKernel takes the tensor's own device
  // (matches the per_token_group_quant_8bit.cuh idiom).
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({NC, NV}).with_dtype<float>().with_device(device).verify(A);
  TensorMatcher({NC}).with_dtype<float>().with_device(device).verify(b);
  TensorMatcher({NV}).with_dtype<float>().with_device(device).verify(c);
  TensorMatcher({NV}).with_dtype<float>().with_device(device).verify(result);

  const size_t smem_bytes = sizeof(ipm_hip_smem<NC, NV>);

  // One block solves the whole (tiny) LP on-chip. smem for the LPLB shapes is
  // ~30 KB, comfortably under the 64 KB LDS/CU, so no max-dynamic-smem opt-in
  // is needed (unlike the >48 KB Hopper path in ipm.cuh).
  LaunchKernel(/*grid_dim=*/1, /*block_dim=*/BLOCK_DIM, result.device(), smem_bytes)(
      ipm_hip_solve_kernel<NC, NV, BLOCK_DIM, NUM_ITERS>,
      static_cast<float*>(result.data_ptr()),
      static_cast<const float*>(A.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<const float*>(c.data_ptr()));
}

}  // namespace
