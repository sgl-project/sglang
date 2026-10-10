#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp8.h>

namespace sglang {

using namespace device;

// Small-M blockwise-FP8 GEMV for Hopper decode (128x128 weight blocks, 1x128
// activation groups, the DeepGEMM / DeepSeek-V3 layout):
//
//   out[m, n] = sum_k xq[m, k] * wq[n, k] * xs[m, k / 128] * ws[n / 128, k / 128]
//
// DeepGEMM's TMA/WGMMA pipeline is built for large tiles; at M <= 8 it moves the
// few-MB decode weights at ~1 TB/s on H20 (18 us for a 21 MB projection that
// streams in 6 us), and a CUDA-core FMA GEMV is instruction-bound past M = 2.
// This kernel streams the weights straight from global memory into
// mma.sync.m16n8k32 e4m3 fragments: one warp owns a 16-row tile for all 8 token
// columns per n8 group (M <= 16 as two groups; unused columns are zero), and per 128-wide scale block lane
// (g = lane / 4, t = lane % 4) reads k = kb * 128 + 32 * t .. + 32 of rows g and
// g + 8 with two 16-byte evict-first loads each, plus the same 32-byte run of
// token g from the shared-memory activation slab. Feeding those runs to four
// MMAs relabels k identically on both operands (a dot product is invariant to
// the k order), so no shuffles are needed. The fp32 128-block partial is scaled
// by ws * xs before it joins the running sum, matching the block-scaled GEMM
// numerics up to summation order. Warps in a block split K (kKSplit) and reduce
// through shared memory so small-N projections still fill the machine.

constexpr uint32_t kFp8BlockK = 128;
constexpr uint32_t kMaxSideN = 8;

__device__ __forceinline__ void mma_m16n8k32_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// side_out[m, col] = x[m, :] . side_w[col, :] for one column: the block's
// threads split K in 16-byte chunks with every load issued up front, then a
// shuffle + shared-memory reduce. One extra block per side column; it finishes
// inside the main GEMV's runtime.
template <uint32_t kMaxM, uint32_t kThreads>
__device__ __forceinline__ void side_column(
    bf16_t* __restrict__ side_out,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ side_w,
    float* sred,  // [kThreads / 32][kMaxM]
    const uint32_t M,
    const uint32_t K,
    const uint32_t side_n,
    const uint32_t col) {
  constexpr uint32_t kSideWarps = kThreads / 32;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const bf16_t* wrow = side_w + static_cast<size_t>(col) * K;
  const uint32_t nvec = K / 8;
  float acc[kMaxM];
#pragma unroll
  for (uint32_t m = 0; m < kMaxM; ++m) {
    acc[m] = 0.0f;
  }
#pragma unroll 4
  for (uint32_t v = tid; v < nvec; v += kThreads) {
    const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(wrow + static_cast<size_t>(v) * 8));
    const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      if (m < M) {
        const uint4 xv =
            __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(m) * K + static_cast<size_t>(v) * 8));
        const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
          const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
          p = fmaf(w0, x0, p);
          p = fmaf(w1, x1, p);
        }
        acc[m] += p;
      }
    }
  }
#pragma unroll
  for (uint32_t m = 0; m < kMaxM; ++m) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
      acc[m] += __shfl_xor_sync(0xffffffff, acc[m], off);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t m = 0; m < kMaxM; ++m) {
      sred[warp * kMaxM + m] = acc[m];
    }
  }
  __syncthreads();
  if (tid < kMaxM && tid < M) {
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kSideWarps; ++w) {
      s += sred[w * kMaxM + tid];
    }
    side_out[static_cast<size_t>(tid) * side_n + col] = cast<bf16_t>(s);
  }
}

template <uint32_t kMaxM, uint32_t kRowTiles, uint32_t kKSplit, uint32_t kUnroll>
__global__ void __launch_bounds__(32 * kRowTiles * kKSplit) hopper_fp8_block_gemv_kernel(
    bf16_t* __restrict__ out,
    const bf16_t* __restrict__ x,
    const uint8_t* __restrict__ xq,
    const float* __restrict__ xs,
    const uint8_t* __restrict__ wq,
    const float* __restrict__ ws,
    const bf16_t* __restrict__ side_w,
    bf16_t* __restrict__ side_out,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t side_n,
    const uint32_t num_main_blocks) {
  constexpr uint32_t kWarps = kRowTiles * kKSplit;
  constexpr uint32_t kThreads = 32 * kWarps;
  constexpr uint32_t kMGroups = kMaxM / 8;  // n8 MMA column groups
  extern __shared__ __align__(16) uint8_t smem_raw[];
  if (blockIdx.x >= num_main_blocks) {
    // Side blocks never touch the slab; reuse the front of the dynamic buffer
    // for their tiny cross-warp reduction.
    side_column<kMaxM, kThreads>(
        side_out, x, side_w, reinterpret_cast<float*>(smem_raw), M, K, side_n, blockIdx.x - num_main_blocks);
    return;
  }
  const uint32_t nkb = K / kFp8BlockK;
  // Row stride K + 16: lanes (g, t) of a quarter-warp read 16-byte slots at
  // g * S + 32 * t; S = 16 (mod 32) keeps those slots on distinct banks.
  const uint32_t sx_stride = K + 16;
  uint8_t* sx = smem_raw;                                                                    // [kMaxM][K + 16] e4m3
  float* sxs = reinterpret_cast<float*>(smem_raw + static_cast<size_t>(kMaxM) * sx_stride);  // [kMaxM][nkb]
  float* sred = sxs + kMaxM * nkb;  // [kKSplit][kRowTiles][32][4 * kMGroups]

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t tile = warp / kKSplit;
  const uint32_t ks = warp % kKSplit;
  const uint32_t g = lane / 4;
  const uint32_t t = lane % 4;
  // A tile past N (only the last block) still runs on clamped rows so every
  // warp reaches the K-split barrier; its stores are guarded below.
  const uint32_t row_base = (blockIdx.x * kRowTiles + tile) * 16;
  const uint32_t r_lo = min(row_base + g, N - 1);
  const uint32_t r_hi = min(row_base + g + 8, N - 1);
  const uint8_t* w_lo = wq + static_cast<size_t>(r_lo) * K + t * 32;
  const uint8_t* w_hi = wq + static_cast<size_t>(r_hi) * K + t * 32;
  const uint32_t kb_per_warp = nkb / kKSplit;  // host-checked multiple of kUnroll
  const uint32_t kb_begin = ks * kb_per_warp;
  const uint32_t kb_end = kb_begin + kb_per_warp;

  // Weight fragments for kUnroll scale blocks: [u][0..1] row g, [u][2..3] row g + 8.
  uint4 buf_a[kUnroll][4], buf_b[kUnroll][4];
  auto load = [&](uint4(&buf)[kUnroll][4], const uint32_t kb) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const size_t koff = static_cast<size_t>(kb + u) * kFp8BlockK;
      buf[u][0] = __ldcs(reinterpret_cast<const uint4*>(w_lo + koff));
      buf[u][1] = __ldcs(reinterpret_cast<const uint4*>(w_lo + koff + 16));
      buf[u][2] = __ldcs(reinterpret_cast<const uint4*>(w_hi + koff));
      buf[u][3] = __ldcs(reinterpret_cast<const uint4*>(w_hi + koff + 16));
    }
  };
  // Issue the first weight loads before staging so the DRAM latency overlaps
  // the shared-memory fill.
  load(buf_a, kb_begin);

  for (uint32_t i = tid * 16; i < kMaxM * K; i += kThreads * 16) {
    const uint32_t m = i / K;
    const uint32_t kk = i - m * K;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (m < M) {
      v = *reinterpret_cast<const uint4*>(xq + i);
    }
    *reinterpret_cast<uint4*>(sx + static_cast<size_t>(m) * sx_stride + kk) = v;
  }
  for (uint32_t i = tid; i < kMaxM * nkb; i += kThreads) {
    sxs[i] = (i / nkb < M) ? xs[i] : 0.0f;
  }
  __syncthreads();

  const float* wsrow = ws + (row_base / kFp8BlockK) * nkb;  // 16 | 128: one scale row per tile

  float acc[kMGroups][4];
#pragma unroll
  for (uint32_t mg = 0; mg < kMGroups; ++mg) {
    acc[mg][0] = acc[mg][1] = acc[mg][2] = acc[mg][3] = 0.0f;
  }
  auto compute = [&](const uint4(&buf)[kUnroll][4], const uint32_t kb) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const size_t koff = static_cast<size_t>(kb + u) * kFp8BlockK;
      const uint32_t r[8] = {
          buf[u][0].x, buf[u][0].y, buf[u][0].z, buf[u][0].w, buf[u][1].x, buf[u][1].y, buf[u][1].z, buf[u][1].w};
      const uint32_t s[8] = {
          buf[u][2].x, buf[u][2].y, buf[u][2].z, buf[u][2].w, buf[u][3].x, buf[u][3].y, buf[u][3].z, buf[u][3].w};
      const float wsc = __ldg(wsrow + kb + u);
#pragma unroll
      for (uint32_t mg = 0; mg < kMGroups; ++mg) {
        // Token column group mg: lane (g, t) feeds token mg * 8 + g.
        const uint8_t* xg = sx + static_cast<size_t>(mg * 8 + g) * sx_stride + t * 32 + koff;
        const uint4 x0 = *reinterpret_cast<const uint4*>(xg);
        const uint4 x1 = *reinterpret_cast<const uint4*>(xg + 16);
        const uint32_t q[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t a[4] = {r[2 * j], s[2 * j], r[2 * j + 1], s[2 * j + 1]};
          const uint32_t b[2] = {q[2 * j], q[2 * j + 1]};
          mma_m16n8k32_e4m3(c, a, b);
        }
        const float s0 = wsc * sxs[(mg * 8 + 2 * t) * nkb + kb + u];
        const float s1 = wsc * sxs[(mg * 8 + 2 * t + 1) * nkb + kb + u];
        acc[mg][0] = fmaf(c[0], s0, acc[mg][0]);
        acc[mg][1] = fmaf(c[1], s1, acc[mg][1]);
        acc[mg][2] = fmaf(c[2], s0, acc[mg][2]);
        acc[mg][3] = fmaf(c[3], s1, acc[mg][3]);
      }
    }
  };

  // Two-stage register pipeline: the next kUnroll blocks stream in while the
  // current ones feed the MMAs.
  for (uint32_t kb = kb_begin; kb < kb_end; kb += 2 * kUnroll) {
    const bool has_b = kb + kUnroll < kb_end;
    if (has_b) load(buf_b, kb + kUnroll);
    compute(buf_a, kb);
    if (has_b) {
      if (kb + 2 * kUnroll < kb_end) load(buf_a, kb + 2 * kUnroll);
      compute(buf_b, kb + kUnroll);
    }
  }

  if constexpr (kKSplit > 1) {
    float* slot = sred + ((ks * kRowTiles + tile) * 32 + lane) * (4 * kMGroups);
#pragma unroll
    for (uint32_t mg = 0; mg < kMGroups; ++mg) {
      slot[4 * mg + 0] = acc[mg][0];
      slot[4 * mg + 1] = acc[mg][1];
      slot[4 * mg + 2] = acc[mg][2];
      slot[4 * mg + 3] = acc[mg][3];
    }
    __syncthreads();
    if (ks != 0) {
      return;
    }
#pragma unroll
    for (uint32_t o = 1; o < kKSplit; ++o) {
      const float* other = sred + ((o * kRowTiles + tile) * 32 + lane) * (4 * kMGroups);
#pragma unroll
      for (uint32_t mg = 0; mg < kMGroups; ++mg) {
        acc[mg][0] += other[4 * mg + 0];
        acc[mg][1] += other[4 * mg + 1];
        acc[mg][2] += other[4 * mg + 2];
        acc[mg][3] += other[4 * mg + 3];
      }
    }
  }

  const uint32_t row_lo = row_base + g;
  const uint32_t row_hi = row_base + g + 8;
#pragma unroll
  for (uint32_t mg = 0; mg < kMGroups; ++mg) {
    const uint32_t m0 = mg * 8 + 2 * t;
    const uint32_t m1 = m0 + 1;
    if (m0 < M) {
      if (row_lo < N) out[static_cast<size_t>(m0) * N + row_lo] = cast<bf16_t>(acc[mg][0]);
      if (row_hi < N) out[static_cast<size_t>(m0) * N + row_hi] = cast<bf16_t>(acc[mg][2]);
    }
    if (m1 < M) {
      if (row_lo < N) out[static_cast<size_t>(m1) * N + row_lo] = cast<bf16_t>(acc[mg][1]);
      if (row_hi < N) out[static_cast<size_t>(m1) * N + row_hi] = cast<bf16_t>(acc[mg][3]);
    }
  }
}

template <uint32_t kMaxM, uint32_t kRowTiles, uint32_t kKSplit, uint32_t kUnroll>
struct HopperFp8BlockGemvKernel {
  static_assert(kRowTiles >= 1 && kKSplit >= 1 && kRowTiles * kKSplit <= 32, "block too large");
  static_assert(kMaxM == 8 || kMaxM == 16, "kMaxM must be 8 or 16 (n8 MMA column groups)");

  // out[M, N] = xq[M, K] @ wq[N, K]^T with 1x128 activation and 128x128 weight
  // scales; with side tensors, one extra block per side column also writes
  // side_out[M, S] = x[M, K] @ side_w[S, K]^T (bf16 in, fp32 accumulation).
  static void run_impl(
      const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView xs,
      const tvm::ffi::TensorView wq,
      const tvm::ffi::TensorView ws,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView* x,
      const tvm::ffi::TensorView* side_w,
      const tvm::ffi::TensorView* side_out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"M"};
    auto N = SymbolicSize{"N"};
    auto K = SymbolicSize{"K"};
    auto KB = SymbolicSize{"K/128"};
    auto NB = SymbolicSize{"ceil(N/128)"};
    TensorMatcher({M, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(xq);
    TensorMatcher({M, KB}).with_dtype<fp32_t>().with_device(device).verify(xs);
    TensorMatcher({N, K}).with_dtype<fp8_e4m3_t>().with_device(device).verify(wq);
    TensorMatcher({NB, KB}).with_dtype<fp32_t>().with_device(device).verify(ws);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto m = static_cast<uint32_t>(M.unwrap());
    const auto n = static_cast<uint32_t>(N.unwrap());
    const auto k = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(m >= 1 && m <= kMaxM, "M must be in [1, kMaxM]");
    RuntimeCheck(k % kFp8BlockK == 0, "K must be a multiple of 128");
    const uint32_t nkb = k / kFp8BlockK;
    RuntimeCheck(nkb % (kKSplit * kUnroll) == 0, "K/128 must be a multiple of kKSplit * kUnroll");
    RuntimeCheck(static_cast<uint32_t>(KB.unwrap()) == nkb, "activation scale must have K/128 groups");
    RuntimeCheck(
        static_cast<uint32_t>(NB.unwrap()) == (n + kFp8BlockK - 1) / kFp8BlockK,
        "weight scale must have ceil(N/128) rows");

    const bf16_t* x_ptr = nullptr;
    const bf16_t* side_w_ptr = nullptr;
    bf16_t* side_out_ptr = nullptr;
    uint32_t side_n = 0;
    if (side_w != nullptr) {
      auto S = SymbolicSize{"S"};
      TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(*x);
      TensorMatcher({S, K}).with_dtype<bf16_t>().with_device(device).verify(*side_w);
      TensorMatcher({M, S}).with_dtype<bf16_t>().with_device(device).verify(*side_out);
      side_n = static_cast<uint32_t>(S.unwrap());
      RuntimeCheck(side_n >= 1 && side_n <= kMaxSideN, "side projection is at most kMaxSideN columns");
      x_ptr = static_cast<const bf16_t*>(x->data_ptr());
      side_w_ptr = static_cast<const bf16_t*>(side_w->data_ptr());
      side_out_ptr = static_cast<bf16_t*>(side_out->data_ptr());
    }

    const size_t smem_bytes = static_cast<size_t>(kMaxM) * (k + 16) + static_cast<size_t>(kMaxM) * nkb * sizeof(float) +
                              static_cast<size_t>(kKSplit) * kRowTiles * 32 * (kMaxM / 2) * sizeof(float);
    auto* kernel = hopper_fp8_block_gemv_kernel<kMaxM, kRowTiles, kKSplit, kUnroll>;
    static size_t s_smem_opt_in = 48 * 1024;
    if (smem_bytes > s_smem_opt_in) {
      RuntimeCheck(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)) ==
              cudaSuccess,
          "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed");
      s_smem_opt_in = smem_bytes;
    }

    constexpr uint32_t kRowsPerBlock = 16 * kRowTiles;
    const uint32_t num_main_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(num_main_blocks + side_n, 32 * kRowTiles * kKSplit, device.unwrap(), smem_bytes)(
        kernel,
        static_cast<bf16_t*>(out.data_ptr()),
        x_ptr,
        static_cast<const uint8_t*>(xq.data_ptr()),
        static_cast<const float*>(xs.data_ptr()),
        static_cast<const uint8_t*>(wq.data_ptr()),
        static_cast<const float*>(ws.data_ptr()),
        side_w_ptr,
        side_out_ptr,
        m,
        n,
        k,
        side_n,
        num_main_blocks);
  }

  static void
  run(const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView xs,
      const tvm::ffi::TensorView wq,
      const tvm::ffi::TensorView ws,
      const tvm::ffi::TensorView out) {
    run_impl(xq, xs, wq, ws, out, nullptr, nullptr, nullptr);
  }

  static void run_side(
      const tvm::ffi::TensorView xq,
      const tvm::ffi::TensorView xs,
      const tvm::ffi::TensorView wq,
      const tvm::ffi::TensorView ws,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView side_w,
      const tvm::ffi::TensorView side_out) {
    run_impl(xq, xs, wq, ws, out, &x, &side_w, &side_out);
  }
};

}  // namespace sglang
