// Draft lm_head of one token on this rank's vocabulary shard, with the shard argmax folded in:
// logits[n] = bf16(sum_k W[n, k] * x[k]) with fp32 accumulation (bf16 inputs and weights, logits
// rounded to bf16 as the served GEMM writes them; only the fp32 summation order differs), and per
// CTA the packed order key of its best row (vocab_argmax order keys, -0.0 == +0.0, ties to the
// lowest row) for the step-boundary exchange, which then skips its own scan.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace mkd_head {

// Packed partial of a CTA whose rows hold a NaN logit (no NaN-free packed key reaches it).
inline constexpr uint32_t kNanPart = 0xffffffffu;

/// \brief 16-bit order key of one bf16 value (-0.0 as +0.0); sets nan for a NaN.
SGL_DEVICE uint32_t key16(bf16_t v, bool& nan) {
  uint32_t b = static_cast<uint32_t>(__bfloat16_as_ushort(v));
  nan |= (b & 0x7fffu) > 0x7f80u;
  b = b == 0x8000u ? 0u : b;
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

}  // namespace mkd_head

struct MkdHeadParams {
  const bf16_t* x;               // [K] final-norm hidden state, written by the previous grid
  const bf16_t* __restrict__ w;  // [N, K] lm_head shard
  bf16_t* logits;                // [N]
  uint32_t* part;                // [gridDim.x] packed (key16 << 16) | (0xffff - row), or kNanPart
  int32_t n;
  int32_t reverse;  // 1: walk the rows from the end (alternate steps, see the caller)
};

/**
 * \brief One warp per weight row at a time, all 16 lane vectors of the row in flight; x in shared
 * memory. The first row of every warp is loaded before the PDL wait (weights are static).
 */
template <int kK, int kWarps, bool kUsePDL>
__global__ void __launch_bounds__(kWarps * 32) mkd_head_kernel(const MkdHeadParams __grid_constant__ p) {
  using namespace device;
  constexpr int kVecs = kK / 8 / 32;
  __shared__ alignas(16) bf16_t s_x[kK];
  __shared__ uint32_t s_best[kWarps];
  const int warp = static_cast<int>(threadIdx.x / 32);
  const int lane = static_cast<int>(threadIdx.x % 32);
  const int stride = static_cast<int>(gridDim.x) * kWarps;
  int row = static_cast<int>(blockIdx.x) * kWarps + warp;
  // Walk position i maps to weight row i, or n - 1 - i on reversed calls.
  auto wrow = [&](int i) { return p.reverse ? p.n - 1 - i : i; };
  uint4 wv[kVecs];
  if (row < p.n) {
    const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(wrow(row)) * kK);
#pragma unroll
    for (int j = 0; j < kVecs; ++j)
      wv[j] = __ldg(wr + j * 32 + lane);
  }
  PDLWaitPrimary<kUsePDL>();
  PDLTriggerSecondary<kUsePDL>();
  for (int i = static_cast<int>(threadIdx.x); i < kK / 8; i += kWarps * 32) {
    reinterpret_cast<uint4*>(s_x)[i] = reinterpret_cast<const uint4*>(p.x)[i];
  }
  __syncthreads();
  uint32_t best = 0u;
  bool nan = false;
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
      const uint4* wr = reinterpret_cast<const uint4*>(p.w + static_cast<int64_t>(wrow(next)) * kK);
#pragma unroll
      for (int j = 0; j < kVecs; ++j)
        wv[j] = __ldg(wr + j * 32 + lane);
    }
    acc = warp::reduce_sum(acc);
    if (lane == 0) {
      const int r = wrow(row);
      const bf16_t v = cast<bf16_t>(acc);
      p.logits[r] = v;
      const uint32_t k = mkd_head::key16(v, nan);
      best = max(best, (k << 16) | (0xffffu - static_cast<uint32_t>(r)));
    }
  }
  if (lane == 0) s_best[warp] = nan ? mkd_head::kNanPart : best;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t b = s_best[0];
#pragma unroll
    for (int i = 1; i < kWarps; ++i)
      b = max(b, s_best[i]);
    p.part[blockIdx.x] = b;
  }
}

template <bool kUsePDL>
struct MkdHead {
  static constexpr int kK = 4096;
  static constexpr int kWarps = 16;

  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView part,
      int64_t reverse) {
    using namespace host;
    auto N = SymbolicSize{"rows"};
    auto G = SymbolicSize{"ctas"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({1, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(x);
    TensorMatcher({N, kK}).with_dtype<bf16_t>().with_device(device).ensure_alignment(16).verify(w);
    TensorMatcher({1, N}).with_dtype<bf16_t>().with_device(device).verify(logits);
    TensorMatcher({G}).with_dtype<int32_t>().with_device(device).verify(part);
    CHECK_HOST(N.unwrap() <= 65536) << "rows must fit the 16-bit row of the order key";
    const auto params = MkdHeadParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .logits = static_cast<bf16_t*>(logits.data_ptr()),
        .part = static_cast<uint32_t*>(part.data_ptr()),
        .n = static_cast<int32_t>(N.unwrap()),
        .reverse = static_cast<int32_t>(reverse != 0),
    };
    LaunchKernel(static_cast<uint32_t>(G.unwrap()), kWarps * 32, device.unwrap())
        .enable_pdl(kUsePDL)(mkd_head_kernel<kK, kWarps, kUsePDL>, params);
  }
};

}  // namespace sglang
