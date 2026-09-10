// GEMM for decode: C[M,N] = A[M,K] @ B[N,K]^T, bf16 in/out, fp32 acc.
//
// Tuned for small-M decode workloads (M<=512) on gfx942/gfx950.
// Uses mfma_f32_16x16x16bf16_1k with LDS-staged cooperative tile loading.
// Tile: BM=64 x BN=64 x BK=32.  Block: 256 threads (4 wavefronts).
//
// Entry point: sglang_gemm_decode(A, B, C, M, N, K, stream)

#include <hip/hip_runtime.h>
#include <stdint.h>

#if __has_include("aiter_stream.h")
#  include "aiter_stream.h"
#  include "opus/opus.hpp"
using bf16_t = opus::bf16_t;
static __device__ __forceinline__ float  cvt_f32(bf16_t v)  { return opus::upcast_s(v); }
static __device__ __forceinline__ bf16_t cvt_bf16(float v)  {
    uint16_t h = (uint16_t)(__float_as_uint(v) >> 16);
    return *reinterpret_cast<bf16_t*>(&h);
}
#else
using bf16_t = __hip_bfloat16;
static __device__ __forceinline__ float  cvt_f32(bf16_t v)  {
    uint32_t b = (uint32_t)(*reinterpret_cast<const uint16_t*>(&v)) << 16;
    return __uint_as_float(b);
}
static __device__ __forceinline__ bf16_t cvt_bf16(float v)  {
    uint16_t h = (uint16_t)(__float_as_uint(v) >> 16);
    return *reinterpret_cast<bf16_t*>(&h);
}
#endif

using bf16x4 = __attribute__((ext_vector_type(4))) bf16_t;
using f32x4  = __attribute__((ext_vector_type(4))) float;

static __device__ __forceinline__ f32x4 mfma16(bf16x4 a, bf16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

static constexpr int BM = 64, BN = 64, BK = 32, TNUM = 256;
static constexpr int LDS_AS = BK + 4, LDS_BS = BK + 4;
static constexpr int LDS_AN = BM * LDS_AS, LDS_BN = BN * LDS_BS;

__global__ void __launch_bounds__(TNUM)
gemm_decode_kernel(const bf16_t* __restrict__ A,
                   const bf16_t* __restrict__ B,
                   bf16_t*       __restrict__ C,
                   int M, int N, int K) {
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int tid = threadIdx.x, wave = tid / 64, lane = tid % 64;
    const int rb = m0 + wave * 16;

    __shared__ bf16_t lds[2 * (LDS_AN + LDS_BN)];
    bf16_t *la[2] = {lds, lds + (LDS_AN + LDS_BN)};
    bf16_t *lb[2] = {lds + LDS_AN, lds + (LDS_AN + LDS_BN) + LDS_AN};

    f32x4 acc[4] = {};
    int buf = 0;

    for (int k = 0; k < K; k += BK) {
        for (int li = 0; li < 2; li++) {
            int flat = tid + li * TNUM;
            int r = m0 + flat / (BK / 4), c = k + (flat % (BK / 4)) * 4;
            bf16x4 v = {};
            if (r < M && c + 3 < K)
                *reinterpret_cast<uint64_t*>(&v) = *reinterpret_cast<const uint64_t*>(A + r * K + c);
            *reinterpret_cast<uint64_t*>(la[buf] + (flat / (BK / 4)) * LDS_AS + (flat % (BK / 4)) * 4) =
                *reinterpret_cast<uint64_t*>(&v);
        }
        for (int li = 0; li < 2; li++) {
            int flat = tid + li * TNUM;
            int r = n0 + flat / (BK / 4), c = k + (flat % (BK / 4)) * 4;
            bf16x4 v = {};
            if (r < N && c + 3 < K)
                *reinterpret_cast<uint64_t*>(&v) = *reinterpret_cast<const uint64_t*>(B + r * K + c);
            *reinterpret_cast<uint64_t*>(lb[buf] + (flat / (BK / 4)) * LDS_BS + (flat % (BK / 4)) * 4) =
                *reinterpret_cast<uint64_t*>(&v);
        }
        __syncthreads();

        int row = lane & 15, kg = (lane >> 4) * 4;
        bf16x4 af; *reinterpret_cast<uint64_t*>(&af) = *reinterpret_cast<uint64_t*>(la[buf] + row * LDS_AS + kg);
        for (int nf = 0; nf < 4; nf++) {
            bf16x4 bf_; *reinterpret_cast<uint64_t*>(&bf_) = *reinterpret_cast<uint64_t*>(lb[buf] + (nf * 16 + row) * LDS_BS + kg);
            acc[nf] = mfma16(af, bf_, acc[nf]);
        }
        buf ^= 1;
        __syncthreads();
    }

    for (int nf = 0; nf < 4; nf++) {
        int col = n0 + nf * 16 + (lane & 15);
        if (col >= N) continue;
        for (int i = 0; i < 4; i++) {
            int r = rb + (lane >> 4) * 4 + i;
            if (r >= M) continue;
            C[r * N + col] = cvt_bf16(acc[nf][i]);
        }
    }
}

extern "C" hipError_t sglang_gemm_decode(
    const void* A, const void* B, void* C,
    int M, int N, int K, hipStream_t stream) {
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    gemm_decode_kernel<<<grid, TNUM, 0, stream>>>(
        (const bf16_t*)A, (const bf16_t*)B, (bf16_t*)C, M, N, K);
    return hipGetLastError();
}
