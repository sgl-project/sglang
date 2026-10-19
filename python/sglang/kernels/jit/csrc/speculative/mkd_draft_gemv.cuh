// One-row draft GEMV y[n] = bf16(sum_k W[n, k] * x[k]): bf16 operands, fp32 accumulation, output rounded
// to bf16 as the served GEMM writes it (only the fp32 summation order differs). One warp per row with the
// whole row in flight, the row loaded before the PDL wait (weights are static). With a hop site, every CTA
// owns one 8-row fragment of the output and pushes it into the q35 hop before storing it.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include "sgl_kernel/distributed/q35_hop.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

struct MkdGemvParams {
  const bf16_t* x;               // [K], written by the previous grid
  const bf16_t* __restrict__ w;  // [N, K]
  bf16_t* y;                     // [N]
  int32_t n;
  q35hop::Site hop;   // kPush: the hop the fragments go to
  int32_t hop_token;  // kPush: the output's row in the hop
};

/**
 * \brief kRows warps per CTA; warp w of CTA c takes rows kRows * c + w, then a grid stride further on, the
 * next row loaded while the current one is reduced (as the draft head does).
 * \tparam kK    reduction length (a multiple of 256)
 * \tparam kRows warps per CTA; 8 when pushing, with one row per warp so a CTA owns one hop fragment
 */
template <int kK, int kRows, bool kUsePDL, bool kPush>
__global__ void __launch_bounds__(kRows * 32) mkd_gemv_kernel(const MkdGemvParams __grid_constant__ p) {
  using namespace device;
  static_assert(kK % 256 == 0, "K must tile 32 lanes x 8 bf16");
  static_assert(!kPush || kRows == 8, "a pushing CTA owns exactly one 8-row fragment");
  constexpr int kVecs = kK / 8 / 32;
  __shared__ alignas(16) bf16_t s_x[kK];
  __shared__ alignas(16) bf16_t s_y[kRows];
  __shared__ uint32_t s_stage;
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int stride = static_cast<int>(gridDim.x) * kRows;
  int row = static_cast<int>(blockIdx.x) * kRows + warp;
  uint4 wv[kVecs];
  if (row < p.n) {
    const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(row) * kK);
#pragma unroll
    for (int j = 0; j < kVecs; ++j)
      wv[j] = __ldg(wr + j * 32 + lane);
  }
  PDLWaitPrimary<kUsePDL>();
  for (int i = static_cast<int>(threadIdx.x); i < kK / 8; i += kRows * 32) {
    reinterpret_cast<uint4*>(s_x)[i] = reinterpret_cast<const uint4*>(p.x)[i];
  }
  if constexpr (kPush) {
    if (threadIdx.x == 0) s_stage = q35hop::stage_peek(p.hop);
  }
  __syncthreads();
  for (; row < p.n; row += stride) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
      const uint4 xv = reinterpret_cast<const uint4*>(s_x)[j * 32 + lane];
      const bf16_t* a = reinterpret_cast<const bf16_t*>(&wv[j]);
      const bf16_t* b = reinterpret_cast<const bf16_t*>(&xv);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc = fmaf(cast<fp32_t>(a[e]), cast<fp32_t>(b[e]), acc);
    }
    const int next = row + stride;
    if (next < p.n) {
      const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(next) * kK);
#pragma unroll
      for (int j = 0; j < kVecs; ++j)
        wv[j] = __ldg(wr + j * 32 + lane);
    }
    acc = warp::reduce_sum(acc);
    const bf16_t v = cast<bf16_t>(acc);
    if constexpr (kPush) {
      if (lane == 0) s_y[warp] = v;
      __syncthreads();
      // Push before the store: the push starts the flight to every rank.
      if (threadIdx.x == 0)
        q35hop::push_frag(
            p.hop, s_stage, p.hop_token, static_cast<int>(blockIdx.x), *reinterpret_cast<const uint4*>(s_y));
    }
    if (lane == 0) p.y[row] = v;
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <int kK, int kRows, bool kUsePDL, bool kPush>
struct MkdGemv {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView y,
      const tvm::ffi::TensorView hop_site,
      int64_t hop_token) {
    using namespace host;
    auto N = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({N, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(y);
    const int n = static_cast<int>(N.unwrap());
    auto params = MkdGemvParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .y = static_cast<bf16_t*>(y.data_ptr()),
        .n = n,
        .hop = {},
        .hop_token = static_cast<int32_t>(hop_token),
    };
    if constexpr (kPush) {
      // [mc, uc, ctrl, rank, world, m_cap, timeout_ns] of Q35Hop.site_fields(), host int64.
      CHECK_HOST(hop_site.device().device_type == kDLCPU && hop_site.numel() == 7) << "hop_site must be 7 host int64";
      CHECK_HOST(n == 4096) << "a pushed partial is one hidden row of 4096";
      const auto* f = static_cast<const int64_t*>(hop_site.data_ptr());
      params.hop = q35hop::Site{
          reinterpret_cast<void*>(f[0]),
          reinterpret_cast<void*>(f[1]),
          reinterpret_cast<q35hop::Ctrl*>(f[2]),
          static_cast<int>(f[3]),
          static_cast<int>(f[4]),
          static_cast<int>(f[5]),
          static_cast<unsigned long long>(f[6])};
      CHECK_HOST(hop_token >= 0 && hop_token < params.hop.m_cap) << "hop row out of range";
    }
    const auto kernel = mkd_gemv_kernel<kK, kRows, kUsePDL, kPush>;
    uint32_t grid = static_cast<uint32_t>((n + kRows - 1) / kRows);
    if constexpr (!kPush) {
      // Rows beyond one resident wave loop inside the warps instead of waiting for a second wave.
      static const uint32_t per_sm = runtime::get_blocks_per_sm(kernel, kRows * 32, 0);
      grid = std::min(grid, per_sm * runtime::get_sm_count(device.unwrap().device_id));
    }
    LaunchKernel(grid, kRows * 32, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
