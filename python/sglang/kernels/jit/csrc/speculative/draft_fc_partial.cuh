// Row-parallel MTP fc for draft-width batches: per token, the Gemma RMSNorm of
// one input half (fp32 statistics, bf16 output as the stock norm stores it),
// then this rank's slice of the fc GEMV with an fp32 result per output row.
// One CTA per SM owns a contiguous block of weight rows; the static rows are
// copied into shared memory before the PDL wait, so they stream in while the
// previous grid finishes.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace draft_fc {

// global -> shared bulk copy completed on an smem mbarrier (PTX ISA cp.async.bulk);
// size and both addresses must be 16-byte aligned.
SGL_DEVICE void bulk_g2s(void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          device::ptx::to_shared(smem_dst)),
      "l"(gmem_src),
      "r"(bytes),
      "r"(device::ptx::to_shared(bar))
      : "memory");
}

}  // namespace draft_fc

struct DraftFcPartialParams {
  const bf16_t* x;                    // [M, H] input half, written by an earlier grid
  const bf16_t* __restrict__ norm_w;  // [H] Gemma norm weight (scale = 1 + w)
  const bf16_t* __restrict__ w;       // [N, W] this rank's fc input-column slice
  float* out;                         // [M, N] fp32 partial
  float eps;
  int32_t col_start;  // first column of the slice inside the half
  int32_t rows;       // M
};

/**
 * \brief Fused RMSNorm + slice GEMV. \tparam kH half width (norm length),
 * \tparam kW slice width, \tparam kN output rows, \tparam kMaxRows weight rows per CTA,
 * \tparam kTokens rows of x (exact). The whole CTA computes the norm statistics of
 * every token in one pass; each warp then holds one weight row in registers and
 * dots it with every normalized token.
 */
template <int kH, int kW, int kN, int kMaxRows, int kTokens, bool kUsePDL>
__global__ void __launch_bounds__(256, 1) draft_fc_partial_kernel(const DraftFcPartialParams __grid_constant__ p) {
  using namespace device;
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  using vec_t = AlignedVector<bf16_t, 8>;
  static_assert(kH % (8 * kThreads) == 0, "norm length must tile the CTA in 16-byte vectors");
  static_assert(kW % (8 * kWarpThreads) == 0, "slice width must tile a warp in 16-byte vectors");
  constexpr int kNormVecs = kH / (8 * kThreads);
  constexpr int kDotVecs = kW / (8 * kWarpThreads);
  constexpr int kRowBytes = kW * static_cast<int>(sizeof(bf16_t));

  extern __shared__ __align__(128) uint8_t smem[];  // [kMaxRows, kW] weight rows
  __shared__ alignas(16) bf16_t s_x[kTokens][kW];
  __shared__ float s_red[kTokens][kWarps];
  __shared__ uint64_t bar;

  const int warp = static_cast<int>(threadIdx.x / kWarpThreads);
  const int lane = static_cast<int>(get_lane_id());
  const int G = static_cast<int>(gridDim.x);
  const int c = static_cast<int>(blockIdx.x);
  const int base = kN / G, rem = kN % G;
  const int r0 = c * base + min(c, rem);
  const int nrows = base + (c < rem ? 1 : 0);

  if (threadIdx.x == 0) {
    ptx::mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (nrows > 0) {
      ptx::mbar_arrive_expect_tx(&bar, static_cast<uint32_t>(nrows) * kRowBytes);
      draft_fc::bulk_g2s(smem, p.w + static_cast<int64_t>(r0) * kW, static_cast<uint32_t>(nrows) * kRowBytes, &bar);
    }
  }
  __syncthreads();
  PDLWaitPrimary<kUsePDL>();

  // Norm statistics of every token over the whole half.
#pragma unroll
  for (int m = 0; m < kTokens; ++m) {
    const bf16_t* xm = p.x + static_cast<int64_t>(m) * kH;
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormVecs; ++k) {
      vec_t v;
      v.load(xm, k * kThreads + static_cast<int>(threadIdx.x));
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float f = cast<fp32_t>(v[j]);
        ss = fmaf(f, f, ss);
      }
    }
    ss = warp::reduce_sum(ss);
    if (lane == 0) s_red[m][warp] = ss;
  }
  __syncthreads();
  // Normalized slices, rounded to bf16 as the norm output is stored.
#pragma unroll
  for (int m = 0; m < kTokens; ++m) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) tot += s_red[m][w];
    const float inv = rsqrtf(tot / static_cast<float>(kH) + p.eps);
    const bf16_t* xm = p.x + static_cast<int64_t>(m) * kH + p.col_start;
    for (int k = static_cast<int>(threadIdx.x); k < kW / 8; k += kThreads) {
      vec_t v, g, o;
      v.load(xm, k);
      g.load(p.norm_w + p.col_start, k);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        o[j] = cast<bf16_t>(cast<fp32_t>(v[j]) * inv * (1.0f + cast<fp32_t>(g[j])));
      }
      o.store(s_x[m], k);
    }
  }
  __syncthreads();
  if (nrows > 0) ptx::mbar_wait_parity(&bar, 0);
  for (int r = warp; r < nrows; r += kWarps) {
    const auto* row = reinterpret_cast<const bf16_t*>(smem + static_cast<int64_t>(r) * kRowBytes);
    vec_t wv[kDotVecs];
#pragma unroll
    for (int k = 0; k < kDotVecs; ++k) wv[k].load(row, k * kWarpThreads + lane);
#pragma unroll
    for (int m = 0; m < kTokens; ++m) {
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kDotVecs; ++k) {
        vec_t xv;
        xv.load(s_x[m], k * kWarpThreads + lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(cast<fp32_t>(wv[k][j]), cast<fp32_t>(xv[j]), acc);
      }
      acc = warp::reduce_sum(acc);
      if (lane == 0) p.out[static_cast<int64_t>(m) * kN + r0 + r] = acc;
    }
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int kH, int kW, int kN, int kMaxRows, int kTokens, bool kUsePDL>
struct DraftFcPartial {
  static constexpr std::size_t kSmem = static_cast<std::size_t>(kMaxRows) * kW * sizeof(bf16_t);

  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView norm_w,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView out,
      double eps,
      int64_t col_start) {
    using namespace host;
    auto M = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({kH}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(norm_w);
    TensorMatcher({kN, kW}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({M, kN}).with_dtype<fp32_t>().with_device(device).verify(out);
    CHECK_HOST(col_start >= 0 && col_start + kW <= kH && col_start % 8 == 0) << "slice out of range";
    CHECK_HOST(M.unwrap() == kTokens) << "row count differs from kTokens";
    const DLDevice dev = device.unwrap();
    const uint32_t grid = runtime::get_sm_count(dev.device_id);
    CHECK_HOST((kN + grid - 1) / grid <= static_cast<uint32_t>(kMaxRows)) << "more weight rows per CTA than kMaxRows";
    const auto params = DraftFcPartialParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .norm_w = static_cast<const bf16_t*>(norm_w.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .eps = static_cast<float>(eps),
        .col_start = static_cast<int32_t>(col_start),
        .rows = static_cast<int32_t>(M.unwrap()),
    };
    auto kernel = draft_fc_partial_kernel<kH, kW, kN, kMaxRows, kTokens, kUsePDL>;
    static const bool attr_set = [&] {
      RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      return true;
    }();
    (void)attr_set;
    LaunchKernel(grid, 256, dev, kSmem).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
