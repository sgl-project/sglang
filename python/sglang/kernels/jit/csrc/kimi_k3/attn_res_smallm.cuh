// K3 attention-residual aggregation for decode/verify token counts (ROCm).
//
// Same contract as the Triton _agg_kernel in
// sglang/kernels/ops/attention/attn_res_hip.py: per token, score the nvb bank
// rows and the (optionally residual-added, bf16-rounded) prefix row against
// cw with an RMS-normalized dot product, take a global-max softmax over the
// nvb + 1 scores, mix the rows, then optionally apply the output RMSNorm and
// emit a saturating E4M3 copy. Also optionally writes the bf16 pre-norm
// mixture (the dspark aux-capture value) to a strided buffer.
//
// One block per token, every bank row held in registers (kNVB is a template
// parameter so the row arrays stay in VGPRs): each thread owns kVPT 8-wide
// bf16 vectors of every row, so the whole step is
//   issue all loads -> per-thread partials -> ONE block reduction of the
//   2 * (nvb + 1) score partials -> softmax (redundant per thread, scalar)
//   -> mix in registers -> ONE block reduction of ||acc||^2 -> stores.
// The Triton kernel spends ~1 us per bank row on layout round trips through
// LDS (45 s_barrier at nvb = 8); this one has two barriers regardless of nvb.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#ifdef USE_ROCM
#include <hip/hip_fp8.h>
#endif

namespace sglang {

struct AttnResSmallMParams {
  const bf16_t* __restrict__ prefix;  // [T, H]
  const bf16_t* __restrict__ addend;  // [T, H] or null
  bf16_t* __restrict__ prefix_out;    // [T, H] (written when addend)
  bf16_t* __restrict__ bank;          // [T, NB, H]
  const float* __restrict__ cw;       // [H]
  const bf16_t* __restrict__ ow;      // [H] or null (no output norm)
  bf16_t* __restrict__ out;           // [T, H]
  uint8_t* __restrict__ out8;         // [T, H] e4m3 or null
  bf16_t* __restrict__ stream;        // [T, H] pre-norm mixture or null
  int64_t stride_p, stride_a, stride_po, stride_bm, stride_bb, stride_o, stride_o8, stride_s;
  int32_t hidden;  // H
  int32_t n_vecs;  // H / 8
  int32_t write_bank;
  float score_eps, out_eps;
};

namespace attn_res_smallm {

constexpr int kVec = 8;
using vec_t = device::AlignedVector<bf16_t, kVec>;

// All-reduce N per-thread values over the block, leaving the totals in every
// thread. Transposes through LDS so the cross-lane work is one shuffle tree
// per value-group instead of one per value (a wave64 shuffle is an LDS
// permute; 2 * (nvb + 1) separate trees cost ~2 us at nvb = 8).
// smem: N * kThreads + N floats.
template <int N, int kThreads>
__device__ __forceinline__ void block_sum(float (&v)[N], float* smem) {
  constexpr int G = (N * 32 <= kThreads) ? 32 : (N * 16 <= kThreads) ? 16 : (N * 8 <= kThreads) ? 8 : 4;
  static_assert(N * G <= kThreads, "block too small for this reduction");
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < N; ++i)
    smem[i * kThreads + tid] = v[i];
  __syncthreads();
  const int i = tid / G, l = tid % G;
  float s = 0.f;
  if (i < N) {
#pragma unroll
    for (int k = 0; k < kThreads / G; ++k)
      s += smem[i * kThreads + k * G + l];
  }
#pragma unroll
  for (int off = G / 2; off > 0; off >>= 1)
    s += __shfl_xor(s, off, 64);
  if (i < N && l == 0) smem[N * kThreads + i] = s;
  __syncthreads();
#pragma unroll
  for (int k = 0; k < N; ++k)
    v[k] = smem[N * kThreads + k];
}

// Single-value all-reduce (wave shuffle tree + one LDS hop).
template <int kThreads>
__device__ __forceinline__ float block_sum1(float v, float* smem) {
  constexpr int kWarps = kThreads / 64;
#pragma unroll
  for (int off = 32; off > 0; off >>= 1)
    v += __shfl_xor(v, off, 64);
  if ((threadIdx.x & 63) == 0) smem[threadIdx.x >> 6] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    s += smem[w];
  return s;
}

typedef float f2_t __attribute__((ext_vector_type(2)));

// bf16 pair (packed in a u32) -> two fp32 (exact).
__device__ __forceinline__ f2_t bf2_to_f2(uint32_t u) {
  f2_t r;
  r.x = __uint_as_float(u << 16);
  r.y = __uint_as_float(u & 0xffff0000u);
  return r;
}

__device__ __forceinline__ uint8_t to_e4m3(float x) {
#ifdef USE_ROCM
  return static_cast<uint8_t>(__hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3));
#else
  return 0;
#endif
}

}  // namespace attn_res_smallm

template <int kNVB, int kVPT, int kThreads>
__global__ void __launch_bounds__(kThreads) attn_res_smallm_kernel(const AttnResSmallMParams __grid_constant__ p) {
  using namespace device;
  using namespace attn_res_smallm;
  constexpr int kR = kNVB + 1;  // bank rows + prefix (row kNVB)
  __shared__ float smem_a[2 * kR * kThreads + 2 * kR];
  __shared__ float smem_b[(kThreads / 64)];

  const int64_t t = blockIdx.x;
  const bf16_t* bank_t = p.bank + t * p.stride_bm;

  vec_t rows[kR][kVPT];
  vec_t addv[kVPT];
  float cwv[kVPT][kVec];
  const bool has_add = p.addend != nullptr;

  // ---- issue every load up front ----
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int v = threadIdx.x + j * kThreads;
    if (v < p.n_vecs) {
#pragma unroll
      for (int r = 0; r < kNVB; ++r)
        rows[r][j].load(bank_t + r * p.stride_bb, v);
      rows[kNVB][j].load(p.prefix + t * p.stride_p, v);
      if (has_add) addv[j].load(p.addend + t * p.stride_a, v);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        cwv[j][i] = p.cw[v * kVec + i];
    } else {
#pragma unroll
      for (int r = 0; r < kR; ++r)
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          rows[r][j][i] = cast<bf16_t>(0.f);
#pragma unroll
      for (int i = 0; i < kVec; ++i)
        cwv[j][i] = 0.f;
    }
  }

  // ---- materialize the prefix: bf16(prefix + addend), stored and banked ----
  if (has_add) {
#pragma unroll
    for (int j = 0; j < kVPT; ++j) {
      const int v = threadIdx.x + j * kThreads;
      if (v < p.n_vecs) {
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          rows[kNVB][j][i] = cast<bf16_t>(cast<fp32_t>(rows[kNVB][j][i]) + cast<fp32_t>(addv[j][i]));
        rows[kNVB][j].store(p.prefix_out + t * p.stride_po, v);
      }
    }
  }
  if (p.write_bank) {
#pragma unroll
    for (int j = 0; j < kVPT; ++j) {
      const int v = threadIdx.x + j * kThreads;
      if (v < p.n_vecs) rows[kNVB][j].store(p.bank + t * p.stride_bm + kNVB * p.stride_bb, v);
    }
  }

  // ---- score partials: <x_r, cw> and <x_r, x_r> (packed fp32 math) ----
  float part[2 * kR];
  f2_t xf[kR][kVPT][kVec / 2];  // fp32 copies, reused by the mix
#pragma unroll
  for (int r = 0; r < kR; ++r) {
    f2_t d = {0.f, 0.f}, s = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < kVPT; ++j) {
      const uint32_t* u = reinterpret_cast<const uint32_t*>(&rows[r][j]);
#pragma unroll
      for (int i = 0; i < kVec / 2; ++i) {
        const f2_t x = bf2_to_f2(u[i]);
        xf[r][j][i] = x;
        const f2_t c = {cwv[j][2 * i], cwv[j][2 * i + 1]};
        d = x * c + d;
        s = x * x + s;
      }
    }
    part[2 * r] = d.x + d.y;
    part[2 * r + 1] = s.x + s.y;
  }
  block_sum<2 * kR, kThreads>(part, smem_a);

  // ---- global-max softmax (redundantly per thread) ----
  const float inv_h = 1.0f / static_cast<float>(p.hidden);
  float score[kR];
#pragma unroll
  for (int r = 0; r < kR; ++r)
    score[r] = part[2 * r] / sqrtf(part[2 * r + 1] * inv_h + p.score_eps);
  float m = score[kNVB];
#pragma unroll
  for (int r = 0; r < kNVB; ++r)
    m = fmaxf(m, score[r]);
  float w[kR];
  float den = 0.f;
#pragma unroll
  for (int r = 0; r < kNVB; ++r) {
    w[r] = expf(score[r] - m);
    den += w[r];
  }
  w[kNVB] = expf(score[kNVB] - m);
  const float inv = 1.0f / (den + w[kNVB]);
#pragma unroll
  for (int r = 0; r < kR; ++r)
    w[r] *= inv;

  // ---- mix (packed fp32, registers) ----
  float acc[kVPT][kVec];
  f2_t asq2 = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
#pragma unroll
    for (int i = 0; i < kVec / 2; ++i) {
      f2_t b = {0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kNVB; ++r) {
        const f2_t x = xf[r][j][i];
        const f2_t wr = {w[r], w[r]};
        b = wr * x + b;
      }
      const f2_t pv = xf[kNVB][j][i];
      const f2_t wp = {w[kNVB], w[kNVB]};
      const f2_t a = pv * wp + b;
      acc[j][2 * i] = a.x;
      acc[j][2 * i + 1] = a.y;
      asq2 = a * a + asq2;
    }
  }

  if (p.stream != nullptr) {
#pragma unroll
    for (int j = 0; j < kVPT; ++j) {
      const int v = threadIdx.x + j * kThreads;
      if (v < p.n_vecs) {
        vec_t o;
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          o[i] = cast<bf16_t>(acc[j][i]);
        o.store(p.stream + t * p.stride_s, v);
      }
    }
  }

  float scale = 1.f;
  const bool norm = p.ow != nullptr;
  if (norm) {
    const float asq = block_sum1<kThreads>(asq2.x + asq2.y, smem_b);
    scale = 1.0f / sqrtf(asq * inv_h + p.out_eps);
  }
#pragma unroll
  for (int j = 0; j < kVPT; ++j) {
    const int v = threadIdx.x + j * kThreads;
    if (v < p.n_vecs) {
      vec_t o;
      if (norm) {
        vec_t owv;
        owv.load(p.ow, v);
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          o[i] = cast<bf16_t>(acc[j][i] * scale * cast<fp32_t>(owv[i]));
      } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i)
          o[i] = cast<bf16_t>(acc[j][i]);
      }
      o.store(p.out + t * p.stride_o, v);
      if (p.out8 != nullptr) {
        uint32_t q[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t packed = 0;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float x = fminf(fmaxf(cast<fp32_t>(o[h * 4 + i]), -448.f), 448.f);
            packed |= static_cast<uint32_t>(to_e4m3(x)) << (8 * i);
          }
          q[h] = packed;
        }
        *reinterpret_cast<uint2*>(p.out8 + t * p.stride_o8 + v * kVec) = make_uint2(q[0], q[1]);
      }
    }
  }
}

template <int kVPT, int kThreads>
struct AttnResSmallMKernel {
  template <int kNVB>
  static void launch(const AttnResSmallMParams& params, int64_t T, DLDevice dev) {
    host::LaunchKernel(static_cast<uint32_t>(T), kThreads, dev)(attn_res_smallm_kernel<kNVB, kVPT, kThreads>, params);
  }

  // Optional operands are passed as real tensors plus a has_* flag (the
  // TVM FFI wrapper here has no optional TensorView).
  static void
  run(const tvm::ffi::TensorView prefix,
      const tvm::ffi::TensorView addend,
      const tvm::ffi::TensorView prefix_out,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView out8,
      const tvm::ffi::TensorView stream,
      int64_t nvb,
      int64_t flags,  // bit0 add, bit1 write_bank, bit2 out norm, bit3 out8, bit4 stream
      double score_eps,
      double out_eps) {
    using namespace host;
    const bool has_add = flags & 1, write_bank = flags & 2, has_norm = flags & 4, has_out8 = flags & 8,
               has_stream = flags & 16;
    const int64_t T = prefix.size(0);
    const int64_t H = prefix.size(1);
    RuntimeCheck(H % 8 == 0, "H must be a multiple of 8");
    RuntimeCheck((H / 8 + kThreads - 1) / kThreads <= kVPT, "H too large for this instantiation");
    RuntimeCheck(prefix.stride(1) == 1 && bank.stride(2) == 1 && out.stride(1) == 1, "inner dim must be contiguous");
    RuntimeCheck(bank.size(1) > nvb || !write_bank, "bank has no free row for the snapshot");
    if (T == 0) return;
    AttnResSmallMParams params{};
    params.prefix = static_cast<const bf16_t*>(prefix.data_ptr());
    params.stride_p = prefix.stride(0);
    if (has_add) {
      params.addend = static_cast<const bf16_t*>(addend.data_ptr());
      params.stride_a = addend.stride(0);
      params.prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr());
      params.stride_po = prefix_out.stride(0);
    }
    params.bank = static_cast<bf16_t*>(bank.data_ptr());
    params.stride_bm = bank.stride(0);
    params.stride_bb = bank.stride(1);
    params.cw = static_cast<const float*>(cw.data_ptr());
    if (has_norm) params.ow = static_cast<const bf16_t*>(ow.data_ptr());
    params.out = static_cast<bf16_t*>(out.data_ptr());
    params.stride_o = out.stride(0);
    if (has_out8) {
      params.out8 = static_cast<uint8_t*>(out8.data_ptr());
      params.stride_o8 = out8.stride(0);
    }
    if (has_stream) {
      params.stream = static_cast<bf16_t*>(stream.data_ptr());
      params.stride_s = stream.stride(0);
    }
    params.hidden = static_cast<int32_t>(H);
    params.n_vecs = static_cast<int32_t>(H / 8);
    params.write_bank = write_bank ? 1 : 0;
    params.score_eps = static_cast<float>(score_eps);
    params.out_eps = static_cast<float>(out_eps);
    const DLDevice dev = prefix.device();
    switch (nvb) {
      case 1:
        launch<1>(params, T, dev);
        break;
      case 2:
        launch<2>(params, T, dev);
        break;
      case 3:
        launch<3>(params, T, dev);
        break;
      case 4:
        launch<4>(params, T, dev);
        break;
      case 5:
        launch<5>(params, T, dev);
        break;
      case 6:
        launch<6>(params, T, dev);
        break;
      case 7:
        launch<7>(params, T, dev);
        break;
      case 8:
        launch<8>(params, T, dev);
        break;
      default:
        RuntimeCheck(false, "nvb must be in [1, 8]");
    }
  }
};

}  // namespace sglang
