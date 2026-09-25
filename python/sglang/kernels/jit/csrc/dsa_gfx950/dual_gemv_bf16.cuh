/// Fused dual bf16 GEMV for the gfx950 DSA indexer: Q [M,4096] from K=2048 and
/// KW [M,160] from K=6144 in one grid, since the 160-column half alone leaves
/// the card idle.  M is specialised in [1,8], the decode batch of one step.

#pragma once

#ifndef USE_ROCM
#error "dual_gemv_bf16.cuh targets CDNA3/4; it uses MFMA and wave64 v_dot2c"
#endif

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

/// The kernel's own scalar aliases live here rather than at ``sglang`` scope:
/// it wants the compiler builtin ``__bf16`` for its ext_vector_type packs,
/// which is a different type from ``sglang::bf16_t`` (``__hip_bfloat16``).
namespace dsa_gfx950::dual_gemv {

using namespace ::sglang::host;

typedef unsigned short ushort_t;
typedef __bf16 bf16_t;
typedef bf16_t bf16x2_t __attribute__((ext_vector_type(2)));
typedef bf16_t bf16x8_t __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

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

// gfx950 exact-Q path.  This is the reduction topology used by the production
// hipBLASLt MT16x16x512 kernel: four local-split-U accumulators, each owning a
// contiguous 128-K slice of every DepthU=512 tile, reduced in wave order.
template <int M>
__device__ __forceinline__ void load_q_mfma_operands(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    int col0,
    int N,
    int K,
    int wave,
    int lane,
    int step,
    bf16x8_t& a,
    bf16x8_t& b) {
  const int operand_row = lane & 15;
  const int kg = lane >> 4;
  const int base = (step >> 2) * 512;
  const int off = (step & 3) * 32;
  const int kk = base + wave * 128 + off;
  a = {};
  b = {};
  if (operand_row < M) {
    a = *reinterpret_cast<const bf16x8_t*>(x + (long)operand_row * K + kk + kg * 8);
  }
  const int wc = col0 + operand_row;
  if (wc < N) {
    b = *reinterpret_cast<const bf16x8_t*>(w + (long)wc * K + kk + kg * 8);
  }
}

// ``THREADS`` runs this body as ``THREADS/256`` independent 256-thread column
// groups inside one workgroup, each with its own waves, columns, LDS slice and
// reduction.
template <int M, int THREADS, int PIPE = 0>
__device__ __forceinline__ void gemv_q_mfma_lsu4(
    const uint4* __restrict__ Xv,
    const uint4* __restrict__ Wv,
    ushort_t* __restrict__ O,
    long ldo,
    int col0,
    int N,
    int K,
    float* __restrict__ smem) {
  static_assert(THREADS % 256 == 0, "MFMA Q path needs a multiple of 256");
  constexpr int kQGroups = THREADS / 256;
  const int gid = (kQGroups == 1) ? 0 : (int)(threadIdx.x >> 8);
  const int tid = (kQGroups == 1) ? (int)threadIdx.x : (int)(threadIdx.x & 255);
  if constexpr (kQGroups > 1) {
    smem += gid * (64 * M);  // 4 waves x 16 columns x M floats per group
    col0 += gid * 16;
  }
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const auto* x = reinterpret_cast<const bf16_t*>(Xv);
  const auto* w = reinterpret_cast<const bf16_t*>(Wv);
  float4_t acc = {};
  // Shift the next global-load pair ahead of the current dependent MFMA.
  // The accumulator visits the same 16 K fragments in the same order; only
  // the operand lifetime changes.  This is therefore bit-exact to LSU4.
  constexpr int kStages = PIPE + 1;
  bf16x8_t abuf[kStages];
  bf16x8_t bbuf[kStages];
#pragma unroll
  for (int preload = 0; preload < kStages - 1; ++preload) {
    load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, preload, abuf[preload], bbuf[preload]);
  }
#pragma unroll
  for (int step = 0; step < 16; ++step) {
    const int cur = step % kStages;
    const int ahead = step + kStages - 1;
    if (ahead < 16) {
      const int next = ahead % kStages;
      load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, ahead, abuf[next], bbuf[next]);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(abuf[cur], bbuf[cur], acc, 0, 0, 0);
    asm volatile("" : "+v"(acc));
  }
  const int out_col_local = lane & 15;
  const int out_row0 = (lane >> 4) * 4;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (out_row0 + j < M) {
      smem[(wave * 16 + out_col_local) * M + out_row0 + j] = acc[j];
    }
  }
  __syncthreads();

  if (tid < 16 * M) {
    const int c = tid / M;
    const int r = tid - c * M;
    float sum = smem[c * M + r] + smem[(16 + c) * M + r];
    sum += smem[(32 + c) * M + r];
    sum += smem[(48 + c) * M + r];
    if (col0 + c < N) O[(long)r * ldo + col0 + c] = f32_to_bf16_rne(sum);
  }
}

// one column-group body.  smem is [THREADS/64][NCOL][M] floats.
template <int M, int THREADS, int TPC, int NCOL, int VPT>
__device__ __forceinline__ void gemv_group(
    const uint4* __restrict__ Xv,
    long ldxv,
    const uint4* __restrict__ Wv,
    long ldwv,
    ushort_t* __restrict__ O,
    long ldo,
    int blk_col0,
    int N,
    float* smem) {
  constexpr int kGroups = THREADS / TPC;
  constexpr int kWPG = TPC / 64;  // waves per group (>=1 by construction)

  const int tid = threadIdx.x;
  const int lane = tid % TPC;  // K-slot inside the group
  const int gid = tid / TPC;   // which column group
  const int cbase = blk_col0 + gid * NCOL;

  // ---- X into registers, packed as 4 x bf16x2 per 16-byte vector ----------
  unsigned int xr[M][VPT][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      uint4 t;
      t = Xv[m * ldxv + lane + p * TPC];
      xr[m][p][0] = t.x;
      xr[m][p][1] = t.y;
      xr[m][p][2] = t.z;
      xr[m][p][3] = t.w;
    }
  }

  // ---- all NCOL x VPT weight vectors in flight ---------------------------
  uint4 wr[NCOL][VPT];
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
    int col = cbase + c;
    // clamp instead of branch: out-of-range columns are computed and dropped
    const uint4* wp = Wv + (long)(col < N ? col : 0) * ldwv + lane;
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      wr[c][p] = wp[p * TPC];
    }
  }

  float acc[NCOL][M];
#pragma unroll
  for (int c = 0; c < NCOL; ++c)
#pragma unroll
    for (int m = 0; m < M; ++m)
      acc[c][m] = 0.0f;

#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      unsigned int w4[4] = {wr[c][p].x, wr[c][p].y, wr[c][p].z, wr[c][p].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int m = 0; m < M; ++m)
          acc[c][m] = dot2_bf16(xr[m][p][e], w4[e], acc[c][m]);
      }
    }
  }

  // ---- reduce: 64-lane butterfly, then cross-wave through LDS -------------
  const int wave = tid >> 6;
  const int wlane = tid & 63;
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[c][m];
#pragma unroll
      for (int off = 32; off > 0; off >>= 1)
        v += __shfl_down(v, off, 64);
      if (wlane == 0) smem[(wave * NCOL + c) * M + m] = v;
    }
  }
  __syncthreads();

  constexpr int kOut = kGroups * NCOL * M;
  if (tid < kOut) {
    const int g = tid / (NCOL * M);
    const int r = tid - g * (NCOL * M);
    const int c = r / M;
    const int m = r - c * M;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWPG; ++w)
      s += smem[((g * kWPG + w) * NCOL + c) * M + m];
    const int col = blk_col0 + g * NCOL + c;
    if (col < N) {
      const ushort_t v = f32_to_bf16_rne(s);
      O[(long)m * ldo + col] = v;
    }
  }
}

template <int M, int THREADS, int TPCK, int NCOLK, int VPTK, int TPCQ, int NCOLQ, int VPTQ, int QPIPE = 0>
__global__ __launch_bounds__(THREADS) void dual_gemv_kernel(
    const uint4* __restrict__ Xq,
    const uint4* __restrict__ Wq,
    ushort_t* __restrict__ Oq,
    long ldoq,
    int Nq,
    const uint4* __restrict__ Xk,
    long ldxk,
    const uint4* __restrict__ Wk,
    long ldwk,
    ushort_t* __restrict__ Ok,
    long ldok,
    int Nk,
    int nblk_k) {
  constexpr int kColsPerBlkK = (THREADS / TPCK) * NCOLK;
  constexpr int kColsPerBlkQ = (THREADS / TPCQ) * NCOLQ;
  constexpr int kSmemK = (THREADS / 64) * NCOLK * M;
  constexpr int kSmemQ = (THREADS / 64) * NCOLQ * M;
  __shared__ float smem[kSmemK > kSmemQ ? kSmemK : kSmemQ];

  const int bid = blockIdx.x;
  // The K blocks sit at the END of the grid, so the Q blocks -- which carry the
  // longer dependent chain -- are dispatched first.
  const int first_k = gridDim.x - nblk_k;
  const bool do_k = bid >= first_k;
  const int b = do_k ? bid - first_k : bid;

  if (do_k) {
    gemv_group<M, THREADS, TPCK, NCOLK, VPTK>(Xk, ldxk, Wk, ldwk, Ok, ldok, b * kColsPerBlkK, Nk, smem);
  } else {
    gemv_q_mfma_lsu4<M, THREADS, QPIPE>(Xq, Wq, Oq, ldoq, b * kColsPerBlkQ, Nq, VPTQ * TPCQ * 8, smem);
  }
}

struct Args {
  const uint4* Xq;
  const uint4* Wq;
  ushort_t* Oq;
  long ldoq;
  int Nq;
  int Kq;
  const uint4* Xk;
  long ldxk;
  const uint4* Wk;
  long ldwk;
  ushort_t* Ok;
  long ldok;
  int Nk;
  int Kk;
  hipStream_t stream;
};

template <int M, int THREADS, int TPCK, int NCOLK, int TPCQ, int NCOLQ, int QPIPE = 0>
void launch(const Args& a) {
  constexpr int kColsK = (THREADS / TPCK) * NCOLK;
  constexpr int kColsQ = (THREADS / TPCQ) * NCOLQ;
  // The shipped configuration is the only one this file builds: 16-byte
  // vectors per thread, 3 on the K half and 1 on the Q half.
  constexpr int VPTK = 3;
  constexpr int VPTQ = 1;
  RuntimeCheck(
      a.Kk == VPTK * TPCK * 8 && a.Kq == VPTQ * TPCQ * 8,
      "dual_gemv is built for Kk=6144, Kq=2048; got Kk=",
      a.Kk,
      " Kq=",
      a.Kq);
  const int nblk_k = (a.Nk + kColsK - 1) / kColsK;
  const int nblk_q = (a.Nq + kColsQ - 1) / kColsQ;
  dual_gemv_kernel<M, THREADS, TPCK, NCOLK, VPTK, TPCQ, NCOLQ, VPTQ, QPIPE>
      <<<dim3(nblk_k + nblk_q), dim3(THREADS), 0, a.stream>>>(
          a.Xq, a.Wq, a.Oq, a.ldoq, a.Nq, a.Xk, a.ldxk, a.Wk, a.ldwk, a.Ok, a.ldok, a.Nk, nblk_k);
}

// The accepted configuration.  Its parameters are template arguments, not a
// selector: retuning means editing this line.
//                       THR TPCK NK TPCQ NQ  PIPE
template <int M>
void dispatch(const Args& a) {
  launch<M, 256, 256, 2, 256, 16, 2>(a);
}
}  // namespace dsa_gfx950::dual_gemv

struct DualGemvBf16Kernel {
  /// Oq = Xq @ Wq^T and Ok = Xk @ Wk^T in one launch.  All six tensors are
  /// bf16; outputs are written, never accumulated.
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

    // The Q half addresses x + operand_row * K, so a padded tensor would read
    // the wrong rows with no other symptom; hence packed strides on Xq/Wq.
    TensorMatcher({M_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Xq);
    TensorMatcher({Nq_, Kq_}).with_dtype<bf16_t>().with_device(device_).with_strides({Kq_, 1}).verify(Wq);
    TensorMatcher({M_, Nq_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Oq);
    TensorMatcher({M_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Xk);
    TensorMatcher({Nk_, Kk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Wk);
    TensorMatcher({M_, Nk_}).with_dtype<bf16_t>().with_device(device_).with_strides({-1, 1}).verify(Ok);

    const auto M = static_cast<int>(M_.unwrap());
    RuntimeCheck(M >= 1 && M <= 8, "dual_gemv is specialised for M in [1,8], got ", M);

    impl::Args a;
    a.Xq = reinterpret_cast<const uint4*>(Xq.data_ptr());
    a.Wq = reinterpret_cast<const uint4*>(Wq.data_ptr());
    a.Oq = reinterpret_cast<impl::ushort_t*>(Oq.data_ptr());
    a.Kq = static_cast<int>(Kq_.unwrap());
    a.Nq = static_cast<int>(Nq_.unwrap());
    a.ldoq = Oq.stride(0);
    a.Xk = reinterpret_cast<const uint4*>(Xk.data_ptr());
    a.Wk = reinterpret_cast<const uint4*>(Wk.data_ptr());
    a.Ok = reinterpret_cast<impl::ushort_t*>(Ok.data_ptr());
    a.Kk = static_cast<int>(Kk_.unwrap());
    a.Nk = static_cast<int>(Nk_.unwrap());
    a.ldxk = Xk.stride(0) / 8;
    a.ldwk = Wk.stride(0) / 8;
    a.ldok = Ok.stride(0);
    a.stream = LaunchKernel::resolve_device(device_.unwrap());

    // A 16-byte (8 bf16) load is the unit, so every K-half row start has to
    // land on one; the Q half is packed with Kq == 2048.
    RuntimeCheck(Xk.stride(0) % 8 == 0 && Wk.stride(0) % 8 == 0, "the K operands must start on an 8-element boundary");

    switch (M) {
      case 1:
        impl::dispatch<1>(a);
        break;
      case 2:
        impl::dispatch<2>(a);
        break;
      case 3:
        impl::dispatch<3>(a);
        break;
      case 4:
        impl::dispatch<4>(a);
        break;
      case 5:
        impl::dispatch<5>(a);
        break;
      case 6:
        impl::dispatch<6>(a);
        break;
      case 7:
        impl::dispatch<7>(a);
        break;
      case 8:
        impl::dispatch<8>(a);
        break;
    }
    RuntimeDeviceCheck();
  }
};

}  // namespace sglang
