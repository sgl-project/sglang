/// Dual bf16 GEMV for the gfx950 DSA indexer: Q [M,4096] and KW [M,160] in one launch.

#pragma once

#ifndef USE_ROCM
#error "dual_gemv_bf16.cuh targets CDNA3/4; it uses MFMA and wave64 v_dot2c"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

// Local bf16_t is the builtin __bf16 that ext_vector_type needs, not sglang::bf16_t.
namespace dsa_gfx950::dual_gemv {

using namespace ::sglang::host;

typedef unsigned short ushort_t;
typedef __bf16 bf16_t;
typedef bf16_t bf16x2_t __attribute__((ext_vector_type(2)));
typedef bf16_t bf16x8_t __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

constexpr int kThreads = 256;
constexpr int kKq = 2048;
constexpr int kKk = 6144;
constexpr int kMaxRows = 48;  // fused_decode.MAX_ROWS
constexpr int kQCols = 16;
constexpr int kQSteps = kKq / 128;
constexpr int kQStageRows = 16;
constexpr int kQStageLd = kThreads + 1;  // 16-byte pad per row against LDS bank conflicts
constexpr int kKwVecs = kKk / (8 * kThreads);

__device__ __forceinline__ float dot2_bf16(unsigned int x, unsigned int w, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, x), __builtin_bit_cast(bf16x2_t, w), acc, false);
}

// round-to-nearest-even, matching torch's fp32 -> bf16 cast
__device__ __forceinline__ ushort_t f32_to_bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN -> quiet NaN
    return static_cast<ushort_t>((u >> 16) | 0x0040u);
  }
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<ushort_t>(u >> 16);
}

// K offset of MFMA step s, in hipBLASLt MT16x16x512's reduction order so Q matches it bit for bit.
__device__ __forceinline__ int q_step_k(int s) {
  return (s >> 2) * 512 + (s & 3) * 32;
}

// 16 Q columns for all M rows; the weights load once and the first 16 x rows go through LDS.
__device__ __forceinline__ void gemv_q(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    ushort_t* __restrict__ O,
    long ldo,
    int M,
    int col0,
    float* __restrict__ smem) {
  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int frag = lane & 15;
  const int kofs = wave * 128 + (lane >> 4) * 8;
  const bf16_t* wp = w + (long)(col0 + frag) * kKq + kofs;
  const bf16_t* xp = x + (long)frag * kKq + kofs;
  bf16x8_t a[kQSteps];
  bf16x8_t b[kQSteps];
  {
    bf16x8_t xv[kQStageRows];
#pragma unroll
    for (int r = 0; r < kQStageRows; ++r)
      if (r < M) xv[r] = *reinterpret_cast<const bf16x8_t*>(x + (long)r * kKq + threadIdx.x * 8);
#pragma unroll
    for (int s = 0; s < kQSteps; ++s)
      b[s] = *reinterpret_cast<const bf16x8_t*>(wp + q_step_k(s));
    bf16x8_t* xs = reinterpret_cast<bf16x8_t*>(smem);
#pragma unroll
    for (int r = 0; r < kQStageRows; ++r)
      if (r < M) xs[r * kQStageLd + threadIdx.x] = xv[r];
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kQSteps; ++s)
      a[s] = frag < M ? xs[frag * kQStageLd + (kofs + q_step_k(s)) / 8] : bf16x8_t{};
    __syncthreads();
  }
  const int groups = (M + 15) >> 4;
  for (int g = 0; g < groups; ++g) {
    const bool more = g + 1 < groups;
    const bool next_valid = 16 * (g + 1) + frag < M;
    const bf16_t* xn = xp + (long)16 * (g + 1) * kKq;
    float4_t acc = {};
#pragma unroll
    for (int s = 0; s < kQSteps; ++s) {
      acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a[s], b[s], acc, 0, 0, 0);
      if (more) {
        a[s] = next_valid ? *reinterpret_cast<const bf16x8_t*>(xn + q_step_k(s)) : bf16x8_t{};
      }
    }
    // [group][wave][column][row], so the wave-order sum below reads stride 256.
    *reinterpret_cast<float4_t*>(smem + ((g * 4 + wave) * 16 + frag) * 16 + (lane >> 4) * 4) = acc;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * kQCols; i += kThreads) {
    const int r = i >> 4;
    const int c = i & 15;
    const float* p = smem + ((r >> 4) * 64 + c) * 16 + (r & 15);
    float sum = p[0] + p[256];
    sum += p[512];
    sum += p[768];
    O[(long)r * ldo + col0 + c] = f32_to_bf16_rne(sum);
  }
}

// NCOL KW columns for R rows from row0.
template <int R, int NCOL>
__device__ __forceinline__ void gemv_kw(
    const uint4* __restrict__ Xv,
    long ldxv,
    const uint4* __restrict__ Wv,
    long ldwv,
    ushort_t* __restrict__ O,
    long ldo,
    int M,
    int row0,
    int col0,
    int N,
    float* __restrict__ smem) {
  const int tid = threadIdx.x;
  unsigned int xr[R][kKwVecs][4];
#pragma unroll
  for (int m = 0; m < R; ++m) {
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      uint4 t = {0u, 0u, 0u, 0u};
      if (row0 + m < M) t = Xv[(long)(row0 + m) * ldxv + tid + p * kThreads];
      xr[m][p][0] = t.x;
      xr[m][p][1] = t.y;
      xr[m][p][2] = t.z;
      xr[m][p][3] = t.w;
    }
  }
  uint4 wr[NCOL][kKwVecs];
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
    const int col = col0 + c;
    // clamp instead of branch: out-of-range columns are computed and dropped
    const uint4* wp = Wv + (long)(col < N ? col : 0) * ldwv + tid;
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      wr[c][p] = wp[p * kThreads];
    }
  }
  float acc[NCOL][R] = {};
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int p = 0; p < kKwVecs; ++p) {
      unsigned int w4[4] = {wr[c][p].x, wr[c][p].y, wr[c][p].z, wr[c][p].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int m = 0; m < R; ++m)
          acc[c][m] = dot2_bf16(xr[m][p][e], w4[e], acc[c][m]);
      }
    }
  }
  const int wave = tid >> 6;
  const int wlane = tid & 63;
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) {
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
      for (int m = 0; m < R; ++m)
        acc[c][m] += __shfl_down(acc[c][m], off, 64);
  }
  if (wlane == 0) {
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
      for (int m = 0; m < R; ++m)
        smem[(wave * NCOL + c) * R + m] = acc[c][m];
  }
  __syncthreads();
  if (tid < NCOL * R) {
    const int c = tid / R;
    const int m = tid - c * R;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w)
      s += smem[(w * NCOL + c) * R + m];
    const int col = col0 + c;
    if (col < N && row0 + m < M) O[(long)(row0 + m) * ldo + col] = f32_to_bf16_rne(s);
  }
}

// The KW blocks lead the grid so their loads queue ahead of the Q weight stream.
template <int R, int NCOL>
__global__ __launch_bounds__(kThreads) void dual_gemv_kernel(
    const bf16_t* __restrict__ Xq,
    const bf16_t* __restrict__ Wq,
    ushort_t* __restrict__ Oq,
    long ldoq,
    int nblk_q,
    const uint4* __restrict__ Xk,
    long ldxk,
    const uint4* __restrict__ Wk,
    long ldwk,
    ushort_t* __restrict__ Ok,
    long ldok,
    int Nk,
    int nblk_kc,
    int M) {
  constexpr int kSmemQ = kQStageRows * kQStageLd * 4;
  static_assert(kSmemQ >= (kMaxRows / 16) * 4 * 16 * 16 && kSmemQ >= 4 * NCOL * R);
  __shared__ __attribute__((aligned(16))) float smem[kSmemQ];
  const int nblk_k = gridDim.x - nblk_q;
  int bid = blockIdx.x;
  bid = bid < nblk_k ? bid + nblk_q : bid - nblk_k;
  if (bid < nblk_q) {
    gemv_q(Xq, Wq, Oq, ldoq, M, bid * kQCols, smem);
  } else {
    const int b = bid - nblk_q;
    const int rc = b / nblk_kc;
    const int cb = b - rc * nblk_kc;
    gemv_kw<R, NCOL>(Xk, ldxk, Wk, ldwk, Ok, ldok, M, rc * R, cb * NCOL, Nk, smem);
  }
}

struct Args {
  const bf16_t* Xq;
  const bf16_t* Wq;
  ushort_t* Oq;
  long ldoq;
  int Nq;
  const uint4* Xk;
  long ldxk;
  const uint4* Wk;
  long ldwk;
  ushort_t* Ok;
  long ldok;
  int Nk;
  int M;
  hipStream_t stream;
};

template <int R, int NCOL>
void launch(const Args& a) {
  const int nblk_q = a.Nq / kQCols;
  const int nblk_kc = (a.Nk + NCOL - 1) / NCOL;
  const int nblk_kr = (a.M + R - 1) / R;
  dual_gemv_kernel<R, NCOL><<<dim3(nblk_q + nblk_kc * nblk_kr), dim3(kThreads), 0, a.stream>>>(
      a.Xq, a.Wq, a.Oq, a.ldoq, nblk_q, a.Xk, a.ldxk, a.Wk, a.ldwk, a.Ok, a.ldok, a.Nk, nblk_kc, a.M);
}

// KW block shape (rows x columns) per M range.
inline void dispatch(const Args& a) {
  if (a.M == 1) {
    launch<1, 1>(a);
  } else if (a.M == 2) {
    launch<2, 1>(a);
  } else if (a.M <= 4) {
    launch<4, 1>(a);
  } else if (a.M <= 6) {
    launch<2, 2>(a);
  } else if (a.M <= 8) {
    launch<4, 2>(a);
  } else if (a.M <= 24) {
    launch<4, 4>(a);
  } else {
    launch<8, 4>(a);
  }
}
}  // namespace dsa_gfx950::dual_gemv

struct DualGemvBf16Kernel {
  /// Oq = Xq @ Wq^T and Ok = Xk @ Wk^T, all bf16; outputs are overwritten.
  static void
  run(const tvm::ffi::TensorView Xq,
      const tvm::ffi::TensorView Wq,
      const tvm::ffi::TensorView Oq,
      const tvm::ffi::TensorView Xk,
      const tvm::ffi::TensorView Wk,
      const tvm::ffi::TensorView Ok) {
    using namespace host;
    namespace impl = dsa_gfx950::dual_gemv;

    auto M_ = SymbolicSize{"num_rows"};
    auto Kq_ = SymbolicSize{"k_q"};
    auto Nq_ = SymbolicSize{"n_q"};
    auto Kk_ = SymbolicSize{"k_k"};
    auto Nk_ = SymbolicSize{"n_k"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    // Q reads x at row * K, so a padded Xq or Wq would silently read wrong rows.
    TensorMatcher({M_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Xq);
    TensorMatcher({Nq_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Wq);
    TensorMatcher({M_, Nq_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Oq);
    TensorMatcher({M_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Xk);
    TensorMatcher({Nk_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Wk);
    TensorMatcher({M_, Nk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Ok);

    const auto M = static_cast<int>(M_.unwrap());
    RuntimeCheck(M >= 1 && M <= impl::kMaxRows, "dual_gemv supports M in [1,", impl::kMaxRows, "], got ", M);
    const auto Kq = static_cast<int>(Kq_.unwrap());
    const auto Kk = static_cast<int>(Kk_.unwrap());
    RuntimeCheck(
        Kk == impl::kKk && Kq == impl::kKq, "dual_gemv is built for Kk=6144, Kq=2048; got Kk=", Kk, " Kq=", Kq);

    impl::Args a;
    a.Xq = reinterpret_cast<const impl::bf16_t*>(Xq.data_ptr());
    a.Wq = reinterpret_cast<const impl::bf16_t*>(Wq.data_ptr());
    a.Oq = reinterpret_cast<impl::ushort_t*>(Oq.data_ptr());
    a.Nq = static_cast<int>(Nq_.unwrap());
    a.ldoq = Oq.stride(0);
    a.Xk = reinterpret_cast<const uint4*>(Xk.data_ptr());
    a.Wk = reinterpret_cast<const uint4*>(Wk.data_ptr());
    a.Ok = reinterpret_cast<impl::ushort_t*>(Ok.data_ptr());
    a.Nk = static_cast<int>(Nk_.unwrap());
    a.ldxk = Xk.stride(0) / 8;
    a.ldwk = Wk.stride(0) / 8;
    a.ldok = Ok.stride(0);
    a.M = M;
    a.stream = LaunchKernel::resolve_device(device_.unwrap());

    RuntimeCheck(a.Nq % impl::kQCols == 0, "dual_gemv needs n_q % 16 == 0, got ", a.Nq);
    RuntimeCheck(Xk.stride(0) % 8 == 0 && Wk.stride(0) % 8 == 0, "the K operands must start on an 8-element boundary");

    impl::dispatch(a);
    RuntimeDeviceCheck();
  }
};

}  // namespace sglang
