/**
 * \brief Skinny fp32 router GEMM for MuseSpark v1.2 decode: out[T, N] = x[T, K] @ w[K, N].
 *
 * x:   [T, K] fp32, T <= kMaxTokens.
 * w:   [K, N] fp32 row-major (checkpoint-native [hidden, experts]).
 * out: [T, N] fp32.
 * workspace: fp32 partials [kSplitK, kMaxTokens, N] followed by int32 counters [N / kNTile].
 *
 * The weight is read exactly once, spread over kSplitK x (N / kNTile) CTAs; the
 * last CTA to finish a column tile sums that tile's kSplitK partials in a fixed
 * order, so the result is deterministic and no second launch is needed. The
 * counters are self-resetting, which is what makes the launch graph-replayable.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct RouterGemmParams {
  const float* __restrict__ x;
  const float* __restrict__ w;
  float* __restrict__ out;
  float* __restrict__ partials;
  int* __restrict__ counters;
  int num_tokens;
  int64_t k;
  int64_t n;
};

template <int kMaxTokens, int kHidden, int kSplitK, int kNTile, int kThreads, bool kUsePDL>
__global__ void __launch_bounds__(kThreads)
    muse_spark_router_gemm_kernel(const __grid_constant__ RouterGemmParams params) {
  using namespace device;
  constexpr int kColGroups = kNTile / 4;
  constexpr int kPhases = kThreads / kColGroups;
  constexpr int kRows = kHidden / kSplitK;
  constexpr int kIters = kRows / kPhases;
  constexpr int kRedPhases = kPhases / 2;
  static_assert(kThreads % kColGroups == 0 && kRows % kPhases == 0 && kPhases % 2 == 0);

  const int split = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int group = threadIdx.x % kColGroups;
  const int phase = threadIdx.x / kColGroups;
  const int n = n_tile * kNTile + group * 4;
  const int64_t k0 = static_cast<int64_t>(split) * kRows;
  const int T = params.num_tokens;

  __shared__ __align__(16) float x_s[kMaxTokens * kRows];
  __shared__ __align__(16) float4 red[kRedPhases * kMaxTokens * kColGroups];
  __shared__ int is_last;

  // All weight loads go out before the PDL wait; one load per iteration was 2x slower.
  float4 wv[kIters];
  const float* w_col = params.w + k0 * params.n + n;
#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    wv[i] = __ldg(reinterpret_cast<const float4*>(w_col + static_cast<int64_t>(phase + i * kPhases) * params.n));
  }

  PDLWaitPrimary<kUsePDL>();

  for (int i = threadIdx.x; i < T * kRows; i += kThreads) {
    const int t = i / kRows;
    const int kk = i - t * kRows;
    x_s[t * kRows + kk] = params.x[t * params.k + k0 + kk];
  }
  __syncthreads();

  float4 acc[kMaxTokens];
#pragma unroll
  for (int t = 0; t < kMaxTokens; ++t)
    acc[t] = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
  for (int i = 0; i < kIters; ++i) {
    const int kk = phase + i * kPhases;
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      if (t < T) {
        const float xv = x_s[t * kRows + kk];
        acc[t].x = fmaf(xv, wv[i].x, acc[t].x);
        acc[t].y = fmaf(xv, wv[i].y, acc[t].y);
        acc[t].z = fmaf(xv, wv[i].z, acc[t].z);
        acc[t].w = fmaf(xv, wv[i].w, acc[t].w);
      }
    }
  }

  if (phase >= kRedPhases) {
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t)
      if (t < T) red[((phase - kRedPhases) * kMaxTokens + t) * kColGroups + group] = acc[t];
  }
  __syncthreads();
  if (phase < kRedPhases) {
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      if (t < T) {
        const float4 o = red[(phase * kMaxTokens + t) * kColGroups + group];
        acc[t].x += o.x;
        acc[t].y += o.y;
        acc[t].z += o.z;
        acc[t].w += o.w;
        red[(phase * kMaxTokens + t) * kColGroups + group] = acc[t];
      }
    }
  }
  __syncthreads();

  float* partial = params.partials + static_cast<int64_t>(split) * kMaxTokens * params.n;
  if (phase == 0) {
#pragma unroll
    for (int t = 0; t < kMaxTokens; ++t) {
      if (t < T) {
        float4 s = red[t * kColGroups + group];
#pragma unroll
        for (int p = 1; p < kRedPhases; ++p) {
          const float4 o = red[(p * kMaxTokens + t) * kColGroups + group];
          s.x += o.x;
          s.y += o.y;
          s.z += o.z;
          s.w += o.w;
        }
        *reinterpret_cast<float4*>(partial + t * params.n + n) = s;
      }
    }
  }

  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(params.counters + n_tile, 1);
    is_last = (prev == kSplitK - 1);
    if (is_last) params.counters[n_tile] = 0;
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  // Fixed summation order keeps the result deterministic.
  for (int i = threadIdx.x; i < T * kNTile; i += kThreads) {
    const int t = i / kNTile;
    const int c = i - t * kNTile;
    const int64_t idx = static_cast<int64_t>(t) * params.n + n_tile * kNTile + c;
    float s = 0.0f;
#pragma unroll 8
    for (int sp = 0; sp < kSplitK; ++sp) {
      s += __ldcg(params.partials + static_cast<int64_t>(sp) * kMaxTokens * params.n + idx);
    }
    params.out[idx] = s;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kMaxTokens, int kHidden, int kSplitK, int kNTile, int kThreads, bool kUsePDL>
struct MuseSparkRouterGemm {
  static void
  run(const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView workspace) {
    using namespace host;

    auto T = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    K.set_value(kHidden);
    auto N = SymbolicSize{"num_experts"};
    auto W = SymbolicSize{"workspace"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({T, K}).with_dtype<fp32_t>().with_device(device_).verify(x);
    TensorMatcher({K, N}).with_dtype<fp32_t>().with_device(device_).verify(w);
    TensorMatcher({T, N}).with_dtype<fp32_t>().with_device(device_).verify(out);
    TensorMatcher({W}).with_dtype<fp32_t>().with_device(device_).verify(workspace);

    const auto num_tokens = static_cast<int>(T.unwrap());
    const auto k = K.unwrap();
    const auto n = N.unwrap();
    RuntimeCheck(num_tokens >= 1 && num_tokens <= kMaxTokens, "num_tokens must be in [1, ", kMaxTokens, "]");
    RuntimeCheck(n % kNTile == 0, "num_experts must be a multiple of ", kNTile);
    const int64_t n_tiles = n / kNTile;
    const int64_t partial_elems = static_cast<int64_t>(kSplitK) * kMaxTokens * n;
    RuntimeCheck(
        W.unwrap() >= partial_elems + n_tiles, "workspace too small: need ", partial_elems + n_tiles, " floats");

    auto* ws = static_cast<float*>(workspace.data_ptr());
    const auto params = RouterGemmParams{
        .x = static_cast<const float*>(x.data_ptr()),
        .w = static_cast<const float*>(w.data_ptr()),
        .out = static_cast<float*>(out.data_ptr()),
        .partials = ws,
        .counters = reinterpret_cast<int*>(ws + partial_elems),
        .num_tokens = num_tokens,
        .k = k,
        .n = n,
    };

    const dim3 grid(kSplitK, static_cast<unsigned>(n_tiles));
    LaunchKernel(grid, kThreads, device_.unwrap())
        .enable_pdl(kUsePDL)(
            muse_spark_router_gemm_kernel<kMaxTokens, kHidden, kSplitK, kNTile, kThreads, kUsePDL>, params);
  }
};

}  // namespace sglang
