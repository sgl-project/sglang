#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sglang {

constexpr uint32_t kMaxBlockThreads = 1024;

template <uint32_t N_, uint32_t K_, uint32_t N_SPLIT_, uint32_t kBytes_>
struct GEMMTraitN {
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t N_SPLIT = N_SPLIT_;  // per block n size
  static constexpr uint32_t kBytes = kBytes_;

  static_assert(N % N_SPLIT == 0, "N must be divisible by n_split");
  static_assert((K * sizeof(bf16_t)) % kBytes == 0, "K must be divisible by kBytes");
  static_assert(kBytes % device::kMaxVecBytes == 0);
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static constexpr uint32_t kBlockSize = (K * sizeof(bf16_t)) / kBytes;
  // always use the largest possible vector
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  // may reduce block size for less thread usage
  static constexpr uint32_t kUnroll = kBytes / device::kMaxVecBytes;
  static_assert(kBlockSize % device::kWarpThreads == 0, "block size must be divisible by warp size");
  static_assert(kBlockSize <= kMaxBlockThreads, "block size exceeds the maximum block size");
};

template <uint32_t N_, uint32_t K_, uint32_t N_SPLIT_, uint32_t kBytes_>
struct GEMMTraitK {
  static constexpr uint32_t N = N_;
  static constexpr uint32_t K = K_;
  static constexpr uint32_t N_SPLIT = N_SPLIT_;  // per block n size
  static constexpr uint32_t kBytes = kBytes_;

  static_assert(N % N_SPLIT == 0, "N must be divisible by n_split");
  static_assert((K * sizeof(bf16_t)) % kBytes == 0, "K must be divisible by kBytes");
  static_assert(device::kMaxVecBytes % kBytes == 0);
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static constexpr uint32_t kNumKLanes = (K * sizeof(bf16_t)) / kBytes;
  static constexpr uint32_t kVecSize = kBytes / sizeof(bf16_t);
  static constexpr uint32_t kBlockSize = N_SPLIT * K / kVecSize;
  static_assert(device::kWarpThreads % kNumKLanes == 0, "K reduction must fit in a warp");
  // A partial warp would leave reduce_sum's kFullMask naming absent lanes.
  static_assert(kBlockSize % device::kWarpThreads == 0, "block size must be divisible by warp size");
  static_assert(kBlockSize <= kMaxBlockThreads, "block size exceeds the maximum block size");
};

#define TINY_GEMM_KERNEL __global__ __launch_bounds__(Trait::kBlockSize, 1)  // grid: 1 block per SM

struct TinyGEMMParams {
  void* __restrict__ out;
  const bf16_t* __restrict__ x;
  const bf16_t* __restrict__ w;
  int64_t stride_x;
};

template <std::size_t N>
SGL_DEVICE void dot_product(device::AlignedVector<bf16x2_t, N> a, device::AlignedVector<bf16x2_t, N> b, float& acc) {
  using namespace device;
#pragma unroll
  for (uint32_t i = 0; i < N; ++i) {
#if SGL_ARCH_BLACKWELL_OR_GREATER
    acc = device::math::fma_f32_bf16(a[i].x, b[i].x, acc);
    acc = device::math::fma_f32_bf16(a[i].y, b[i].y, acc);
#else
    const auto [a0, a1] = cast<fp32x2_t>(a[i]);
    const auto [b0, b1] = cast<fp32x2_t>(b[i]);
    acc += a0 * b0;
    acc += a1 * b1;
#endif
  }
}

template <typename Trait, uint32_t M, typename Out, bool kUsePDL>
TINY_GEMM_KERNEL void tiny_n_gemm_kernel(const TinyGEMMParams params) {
  using namespace device;
  constexpr uint32_t N = Trait::N;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t N_SPLIT = Trait::N_SPLIT;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kUnroll = Trait::kUnroll;
  constexpr uint32_t kBlockSize = Trait::kBlockSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const bf16_t* w_tile = params.w + bx * (N_SPLIT * K);

  // prefetch weight before PDL
  vec_t wv[N_SPLIT][kUnroll];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      wv[n][u].load(w_tile + n * K, tx + u * kBlockSize);
    }
  }

  PDLWaitPrimary<kUsePDL>();

  vec_t xv[M][kUnroll];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      xv[m][u].load(params.x + m * params.stride_x, tx + u * kBlockSize);
    }
  }

  __shared__ float s_acc[kNumWarps][M * N_SPLIT];
  const uint32_t warp_id = tx / kWarpThreads;

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t n = 0; n < N_SPLIT; ++n) {
      float acc = 0.0f;
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        dot_product(xv[m][u], wv[n][u], acc);
      }
      s_acc[warp_id][m * N_SPLIT + n] = warp::reduce_sum(acc);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  static_assert(M * N_SPLIT <= kBlockSize);
  if (tx < M * N_SPLIT) {
    float acc[kNumWarps];
#pragma unroll
    for (uint32_t i = 0; i < kNumWarps; ++i) {
      acc[i] = s_acc[i][tx];
    }
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      acc[0] += acc[i];
    }
    const uint32_t m = tx / N_SPLIT;
    const uint32_t n = tx % N_SPLIT;
    static_cast<Out*>(params.out)[m * N + bx * N_SPLIT + n] = cast<Out>(acc[0]);
  }
}

template <typename Trait, uint32_t M, typename Out, bool kUsePDL>
TINY_GEMM_KERNEL void tiny_k_gemm_kernel(const TinyGEMMParams params) {
  using namespace device;
  constexpr uint32_t N = Trait::N;
  constexpr uint32_t K = Trait::K;
  constexpr uint32_t N_SPLIT = Trait::N_SPLIT;
  constexpr uint32_t kVecSize = Trait::kVecSize;
  constexpr uint32_t kNumKLanes = Trait::kNumKLanes;
  using vec_t = AlignedVector<bf16x2_t, kVecSize / 2>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t n_idx = bx * N_SPLIT + tx / kNumKLanes;
  const uint32_t work_id = tx % kNumKLanes;
  const bf16_t* w_tile = params.w + n_idx * K;

  // Weight prefetch: address is input-independent, load before the PDL wait.
  vec_t wv;
  wv.load(w_tile, work_id);

  PDLWaitPrimary<kUsePDL>();
  vec_t xv[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    xv[m].load(params.x + m * params.stride_x, work_id);
  }
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    float acc = 0.0f;
    dot_product(xv[m], wv, acc);
    // Broadcast store: every lane of the group holds the reduced sum.
    const auto sum = warp::reduce_sum<kNumKLanes>(acc);
    static_cast<Out*>(params.out)[m * N + n_idx] = cast<Out>(sum);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t N, uint32_t K, uint32_t kMaxM, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
struct TinyNGemmKernel {
  using Trait = GEMMTraitN<N, K, N_SPLIT, 32>;
  using KernelFn = void (*)(TinyGEMMParams);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxM + 1>{nullptr, tiny_n_gemm_kernel<Trait, I + 1, OutT, kUsePDL>...};
  }

  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxM>{});

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM);
    // x may be a row-sliced view of a wider buffer, but the rows are loaded as
    // whole vectors, so every row start must stay vector-aligned.
    CHECK_HOST(x.stride(0) % Trait::kVecSize == 0)
        << "x rows must stay aligned to the vector width, got stride " << x.stride(0);
    const auto params = TinyGEMMParams{
        .out = out.data_ptr(),
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .stride_x = static_cast<int64_t>(x.stride(0)),
    };
    LaunchKernel(Trait::kNumBlocks, Trait::kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kTable[num_tokens], params);
  }
};

template <uint32_t N, uint32_t K, uint32_t kMaxM, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
struct TinyKGemmKernel {
  using Trait = GEMMTraitK<N, K, N_SPLIT, 16>;
  using KernelFn = void (*)(TinyGEMMParams);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxM + 1>{nullptr, tiny_k_gemm_kernel<Trait, I + 1, OutT, kUsePDL>...};
  }

  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxM>{});

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;
    CHECK_HOST(num_tokens >= 1 && num_tokens <= kMaxM);
    // x may be a row-sliced view of a wider buffer, but the rows are loaded as
    // whole vectors, so every row start must stay vector-aligned.
    CHECK_HOST(x.stride(0) % Trait::kVecSize == 0)
        << "x rows must stay aligned to the vector width, got stride " << x.stride(0);
    const auto params = TinyGEMMParams{
        .out = out.data_ptr(),
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .stride_x = static_cast<int64_t>(x.stride(0)),
    };
    LaunchKernel(Trait::kNumBlocks, Trait::kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kTable[num_tokens], params);
  }
};

}  // namespace sglang
