// Native CUDA M=1 FP8 GEMV for the per-token / per-channel W8A8 decode regime.
//
// The SM100 `fp8_scaled_mm` path routes the M==1 (decode) shape to SM89 CUTLASS
// GEMM kernels (a 64-row MMA tile), wasting ~63/64 of the tensor-core M dimension
// on a single-row problem. At M==1 the op is a memory-bound GEMV, not a
// compute-bound GEMM, so a kernel that simply streams B once with fully-coalesced
// vector loads wins.
//
// Layout exploited: B is column-major [K, N] (stride (1, K)), i.e. physically an
// [N, K] contiguous weight Bphys with Bphys[n, k] == B[k, n]. So each output
// column n maps to a CONTIGUOUS K-length row Bphys[n, :]. One warp owns one
// column n and reads that row in 16-byte (uint4 = 16xfp8) chunks; A[0, :] is
// preloaded to shared memory and reused by all warps. Per-lane fp32 accumulation
// -> warp-shuffle reduction -> lane 0 applies scale_a[0] * scale_b[n] and stores
// bf16. Destination-passing: the result is written into the pre-allocated `out`.
//
// Coverage is intentionally narrow (M==1, bf16 out, no bias, exactly-packed
// column-major B, K%16==0, N%8==0, 16-byte-aligned bases); the Python gate
// `can_use_fp8_gemv` enforces it and everything else falls back to the
// existing `fp8_scaled_mm`.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarps = 8;  // warps per CTA (one output column each)
constexpr int kBlock = kWarps * 32;
constexpr int kVec = 16;  // fp8 elements per uint4 (16-byte) load

// One warp per output column n. A[0, :] (K fp8) lives in shared, reused by all warps.
__global__ void fp8_gemv_kernel(
    const fp8_e4m3_t* __restrict__ A,    // [K]            (row of A; M==1)
    const fp8_e4m3_t* __restrict__ B,    // [N, K] (Bphys; B[k,n] = B[n*K + k])
    const fp32_t* __restrict__ scale_a,  // [1]
    const fp32_t* __restrict__ scale_b,  // [N]
    bf16_t* __restrict__ out,            // [N]
    int K,
    int N) {
  extern __shared__ __align__(16) char smem_raw[];
  fp8_e4m3_t* sA = reinterpret_cast<fp8_e4m3_t*>(smem_raw);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Cooperative vectorized preload of A[0, :] into shared (K % 16 == 0 is
  // guaranteed by the dispatch gate, so the uint4 copies are safe and aligned).
  for (int i = tid * kVec; i < K; i += kBlock * kVec) {
    *reinterpret_cast<uint4*>(&sA[i]) = *reinterpret_cast<const uint4*>(&A[i]);
  }
  __syncthreads();

  const int n = blockIdx.x * kWarps + warp;
  if (n >= N) return;

  const fp8_e4m3_t* Brow = B + static_cast<size_t>(n) * K;
  fp32_t acc = 0.f;
  // Lanes cover consecutive 16-fp8 chunks -> 512 contiguous B bytes / warp / step.
  for (int k = lane * kVec; k < K; k += 32 * kVec) {
    uint4 bvec = *reinterpret_cast<const uint4*>(Brow + k);
    uint4 avec = *reinterpret_cast<const uint4*>(&sA[k]);
    const fp8_e4m3_t* bb = reinterpret_cast<const fp8_e4m3_t*>(&bvec);
    const fp8_e4m3_t* aa = reinterpret_cast<const fp8_e4m3_t*>(&avec);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      acc += static_cast<fp32_t>(bb[j]) * static_cast<fp32_t>(aa[j]);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(device::kFullMask, acc, off);
  }
  if (lane == 0) {
    out[n] = __float2bfloat16(acc * scale_a[0] * scale_b[n]);
  }
}

template <typename T>
const T* cptr(const tvm::ffi::TensorView& t) {
  return reinterpret_cast<const T*>(static_cast<const char*>(t.data_ptr()) + t.byte_offset());
}
template <typename T>
T* mptr(const tvm::ffi::TensorView& t) {
  return reinterpret_cast<T*>(static_cast<char*>(t.data_ptr()) + t.byte_offset());
}

// Destination-passing entry exported to Python as `fp8_gemv`.
// out: [1, N] bf16; a: [1, K] fp8_e4m3 row-major; b: [K, N] fp8_e4m3 column-major
// (physically [N, K] contiguous); scale_a: [1] fp32; scale_b: [N] fp32.
void fp8_gemv(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scale_a,
    tvm::ffi::TensorView scale_b) {
  using namespace host;
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  RuntimeCheck(a.size(0) == 1, "fp8_gemv requires M == 1, got M = ", a.size(0));
  RuntimeCheck(K % kVec == 0, "fp8_gemv requires K % 16 == 0, got K = ", K);

  const DLDevice device = a.device();
  const int grid = (N + kWarps - 1) / kWarps;
  const size_t smem = static_cast<size_t>(K) * sizeof(fp8_e4m3_t);
  LaunchKernel(grid, kBlock, device, smem)(
      fp8_gemv_kernel,
      cptr<fp8_e4m3_t>(a),
      cptr<fp8_e4m3_t>(b),
      cptr<fp32_t>(scale_a),
      cptr<fp32_t>(scale_b),
      mptr<bf16_t>(out),
      K,
      N);
}

}  // namespace
