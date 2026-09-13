// Fused dual bf16 GEMV: Q [M,4096] from K=2048 and KW [M,160] from K=6144 in
// one grid, since the 160-column half alone leaves the card idle.  The K half
// is register-blocked v_dot2c; the Q half is MFMA and ignores the ld* args.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

namespace {

typedef unsigned short ushort_t;
typedef __bf16 bf16_t;
typedef bf16_t bf16x2_t __attribute__((ext_vector_type(2)));
typedef bf16_t bf16x8_t __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float dot2_bf16(unsigned int x, unsigned int w,
                                           float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, x),
                                         __builtin_bit_cast(bf16x2_t, w), acc,
                                         false);
}

// round-to-nearest-even, matching torch's fp32 -> bf16 cast
__device__ __forceinline__ ushort_t f32_to_bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) { // NaN -> quiet NaN
    return static_cast<ushort_t>((u >> 16) | 0x0040u);
  }
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<ushort_t>(u >> 16);
}

__device__ __forceinline__ uint4 load_w(const uint4 *p) { return *p; }

// gfx950 exact-Q path.  This is the reduction topology used by the production
// hipBLASLt MT16x16x512 kernel: four local-split-U accumulators, each owning a
// contiguous 128-K slice of every DepthU=512 tile, reduced in wave order.
template <int M>
__device__ __forceinline__ void
load_q_mfma_operands(const bf16_t *__restrict__ x, const bf16_t *__restrict__ w,
                     int col0, int N, int K, int wave, int lane, int step,
                     bf16x8_t &a, bf16x8_t &b) {
  const int operand_row = lane & 15;
  const int kg = lane >> 4;
  const int base = (step >> 2) * 512;
  const int off = (step & 3) * 32;
  const int kk = base + wave * 128 + off;
  a = {};
  b = {};
  if (operand_row < M) {
    a = *reinterpret_cast<const bf16x8_t *>(x + (long)operand_row * K + kk +
                                            kg * 8);
  }
  const int wc = col0 + operand_row;
  if (wc < N) {
    b = *reinterpret_cast<const bf16x8_t *>(w + (long)wc * K + kk + kg * 8);
  }
}

// ``THREADS`` runs this body as ``THREADS/256`` independent 256-thread column
// groups inside one workgroup, each with its own waves, columns, LDS slice and
// reduction.
template <int M, int THREADS, int PIPE = 0>
__device__ __forceinline__ void
gemv_q_mfma_lsu4(const uint4 *__restrict__ Xv, const uint4 *__restrict__ Wv,
                 ushort_t *__restrict__ O, long ldo, int col0, int N, int K,
                 float *__restrict__ smem) {
  static_assert(THREADS % 256 == 0, "MFMA Q path needs a multiple of 256");
  constexpr int kQGroups = THREADS / 256;
  const int gid = (kQGroups == 1) ? 0 : (int)(threadIdx.x >> 8);
  const int tid = (kQGroups == 1) ? (int)threadIdx.x : (int)(threadIdx.x & 255);
  if constexpr (kQGroups > 1) {
    smem += gid * (64 * M); // 4 waves x 16 columns x M floats per group
    col0 += gid * 16;
  }
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const auto *x = reinterpret_cast<const bf16_t *>(Xv);
  const auto *w = reinterpret_cast<const bf16_t *>(Wv);
  const int operand_row = lane & 15;
  const int kg = lane >> 4;
  float4_t acc = {};
  // Shift the next global-load pair ahead of the current dependent MFMA.
  // The accumulator visits the same 16 K fragments in the same order; only
  // the operand lifetime changes.  This is therefore bit-exact to LSU4.
  constexpr int kStages = PIPE + 1;
  bf16x8_t abuf[kStages];
  bf16x8_t bbuf[kStages];
#pragma unroll
  for (int preload = 0; preload < kStages - 1; ++preload) {
    load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, preload,
                            abuf[preload], bbuf[preload]);
  }
#pragma unroll
  for (int step = 0; step < 16; ++step) {
    const int cur = step % kStages;
    const int ahead = step + kStages - 1;
    if (ahead < 16) {
      const int next = ahead % kStages;
      load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, ahead, abuf[next],
                              bbuf[next]);
    }
    acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(abuf[cur], bbuf[cur], acc, 0,
                                                  0, 0);
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
    if (col0 + c < N)
      O[(long)r * ldo + col0 + c] = f32_to_bf16_rne(sum);
  }
}

// one column-group body.  smem is [THREADS/64][NCOL][M] floats.
template <int M, int THREADS, int TPC, int NCOL, int VPT>
__device__ __forceinline__ void
gemv_group(const uint4 *__restrict__ Xv, long ldxv,
           const uint4 *__restrict__ Wv, long ldwv, ushort_t *__restrict__ O,
           long ldo, int blk_col0, int N, float *smem) {
  constexpr int kGroups = THREADS / TPC;
  constexpr int kWPG = TPC / 64; // waves per group (>=1 by construction)

  const int tid = threadIdx.x;
  const int lane = tid % TPC; // K-slot inside the group
  const int gid = tid / TPC;  // which column group
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
    const uint4 *wp = Wv + (long)(col < N ? col : 0) * ldwv + lane;
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      wr[c][p] = load_w(wp + p * TPC);
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
      if (wlane == 0)
        smem[(wave * NCOL + c) * M + m] = v;
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

template <int M, int THREADS, int TPCK, int NCOLK, int VPTK, int TPCQ,
          int NCOLQ, int VPTQ, int QPIPE = 0>
__global__ __launch_bounds__(THREADS) void dual_gemv_kernel(
    const uint4 *__restrict__ Xq, long ldxq, const uint4 *__restrict__ Wq,
    long ldwq, ushort_t *__restrict__ Oq, long ldoq, int Nq,
    const uint4 *__restrict__ Xk, long ldxk, const uint4 *__restrict__ Wk,
    long ldwk, ushort_t *__restrict__ Ok, long ldok, int Nk, int nblk_k) {
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
    gemv_group<M, THREADS, TPCK, NCOLK, VPTK>(Xk, ldxk, Wk, ldwk, Ok, ldok,
                                              b * kColsPerBlkK, Nk, smem);
  } else {
    gemv_q_mfma_lsu4<M, THREADS, QPIPE>(Xq, Wq, Oq, ldoq, b * kColsPerBlkQ, Nq,
                                        VPTQ * TPCQ * 8, smem);
  }
}

struct Args {
  const uint4 *Xq;
  long ldxq;
  const uint4 *Wq;
  long ldwq;
  ushort_t *Oq;
  long ldoq;
  int Nq;
  int Kq;
  const uint4 *Xk;
  long ldxk;
  const uint4 *Wk;
  long ldwk;
  ushort_t *Ok;
  long ldok;
  int Nk;
  int Kk;
  hipStream_t stream;
};

template <int M, int THREADS, int TPCK, int NCOLK, int TPCQ, int NCOLQ,
          int QPIPE = 0>
void launch(const Args &a) {
  constexpr int kColsK = (THREADS / TPCK) * NCOLK;
  constexpr int kColsQ = (THREADS / TPCQ) * NCOLQ;
  const int VPTK = (a.Kk / 8) / TPCK;
  const int VPTQ = (a.Kq / 8) / TPCQ;
  TORCH_CHECK(VPTK * TPCK * 8 == a.Kk, "Kk not divisible by 8*TPCK");
  TORCH_CHECK(VPTQ * TPCQ * 8 == a.Kq, "Kq not divisible by 8*TPCQ");
  const int nblk_k = (a.Nk + kColsK - 1) / kColsK;
  const int nblk_q = (a.Nq + kColsQ - 1) / kColsQ;

#define DG_CALL(VK, VQ)                                                        \
  do {                                                                         \
    dual_gemv_kernel<M, THREADS, TPCK, NCOLK, VK, TPCQ, NCOLQ, VQ, QPIPE>      \
        <<<dim3(nblk_k + nblk_q), dim3(THREADS), 0, a.stream>>>(               \
            a.Xq, a.ldxq, a.Wq, a.ldwq, a.Oq, a.ldoq, a.Nq, a.Xk, a.ldxk,      \
            a.Wk, a.ldwk, a.Ok, a.ldok, a.Nk, nblk_k);                         \
  } while (0)

  // The shipped configuration is the only one this file builds; the template
  // arguments record it rather than select it.
  TORCH_CHECK(VPTK == 3 && VPTQ == 1,
              "dual_gemv is built for Kk=6144, Kq=2048; got Kk=", a.Kk,
              " Kq=", a.Kq);
  DG_CALL(3, 1);
#undef DG_CALL
}

// The accepted configuration.  Its parameters are template arguments, not a
// selector: retuning means editing this line.
//                       THR TPCK NK TPCQ NQ  PIPE
template <int M> void dispatch(const Args &a) {
  launch<M, 256, 256, 2, 256, 16, 2>(a);
}

} // namespace

void dual_gemv_bf16(at::Tensor Xq, at::Tensor Wq, at::Tensor Oq, at::Tensor Xk,
                    at::Tensor Wk, at::Tensor Ok) {
  for (auto *t : {&Xq, &Wq, &Oq, &Xk, &Wk, &Ok})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "all tensors must be bf16");
  TORCH_CHECK(Xq.dim() == 2 && Wq.dim() == 2 && Oq.dim() == 2, "2-D");
  TORCH_CHECK(Xk.dim() == 2 && Wk.dim() == 2 && Ok.dim() == 2, "2-D");
  const int M = static_cast<int>(Xq.size(0));
  TORCH_CHECK(Xk.size(0) == M && Oq.size(0) == M && Ok.size(0) == M,
              "row count must match");
  TORCH_CHECK(M >= 1 && M <= 8, "specialised for M in [1,8]");

  Args a;
  a.Xq = reinterpret_cast<const uint4 *>(Xq.data_ptr());
  a.Wq = reinterpret_cast<const uint4 *>(Wq.data_ptr());
  a.Oq = reinterpret_cast<ushort_t *>(Oq.data_ptr());
  a.Kq = static_cast<int>(Xq.size(1));
  a.Nq = static_cast<int>(Wq.size(0));
  a.ldxq = Xq.stride(0) / 8;
  a.ldwq = Wq.stride(0) / 8;
  a.ldoq = Oq.stride(0);
  a.Xk = reinterpret_cast<const uint4 *>(Xk.data_ptr());
  a.Wk = reinterpret_cast<const uint4 *>(Wk.data_ptr());
  a.Ok = reinterpret_cast<ushort_t *>(Ok.data_ptr());
  a.Kk = static_cast<int>(Xk.size(1));
  a.Nk = static_cast<int>(Wk.size(0));
  a.ldxk = Xk.stride(0) / 8;
  a.ldwk = Wk.stride(0) / 8;
  a.ldok = Ok.stride(0);
  a.stream = c10::cuda::getCurrentCUDAStream();

  TORCH_CHECK(Wq.size(1) == a.Kq && Wk.size(1) == a.Kk, "K mismatch");
  TORCH_CHECK(Oq.size(1) == a.Nq && Ok.size(1) == a.Nk, "N mismatch");
  // The Q half addresses x + operand_row * K, so a padded tensor would read the
  // wrong rows with no other symptom.  Hence the stride check, not ldxq.
  TORCH_CHECK(
      Xq.stride(0) == a.Kq && Wq.stride(0) == a.Kq,
      "the Q path reads packed rows: Xq/Wq stride(0) must equal Kq, got ",
      Xq.stride(0), "/", Wq.stride(0), " for Kq=", a.Kq);
  TORCH_CHECK(Xq.stride(0) % 8 == 0 && Wq.stride(0) % 8 == 0, "align Q");
  TORCH_CHECK(Xk.stride(0) % 8 == 0 && Wk.stride(0) % 8 == 0, "align K");
  TORCH_CHECK(Xq.stride(1) == 1 && Wq.stride(1) == 1 && Oq.stride(1) == 1,
              "contig");
  TORCH_CHECK(Xk.stride(1) == 1 && Wk.stride(1) == 1 && Ok.stride(1) == 1,
              "contig");

  switch (M) {
  case 1:
    dispatch<1>(a);
    break;
  case 2:
    dispatch<2>(a);
    break;
  case 3:
    dispatch<3>(a);
    break;
  case 4:
    dispatch<4>(a);
    break;
  case 5:
    dispatch<5>(a);
    break;
  case 6:
    dispatch<6>(a);
    break;
  case 7:
    dispatch<7>(a);
    break;
  case 8:
    dispatch<8>(a);
    break;
  default:
    TORCH_CHECK(false, "unreachable");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemv_bf16", &dual_gemv_bf16,
        "fused dual bf16 GEMV: Oq = Xq @ Wq^T and Ok = Xk @ Wk^T in one launch",
        pybind11::arg("Xq"), pybind11::arg("Wq"), pybind11::arg("Oq"),
        pybind11::arg("Xk"), pybind11::arg("Wk"), pybind11::arg("Ok"));
}
