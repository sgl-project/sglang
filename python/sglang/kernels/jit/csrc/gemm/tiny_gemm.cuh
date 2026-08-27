#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <bit>
#include <cstdint>
#include <utility>

namespace sglang {

using namespace device;

constexpr uint32_t kTinyNGemmVecSize = kMaxVecBytes / sizeof(bf16_t);

template <uint32_t M, uint32_t N, uint32_t K, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
__global__ __launch_bounds__(K / kTinyNGemmVecSize, 1)  // 1 block per SM
    void tiny_n_gemm_kernel(OutT* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w) {
  constexpr uint32_t kBlockSize = K / kTinyNGemmVecSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  static_assert(M * N_SPLIT <= kBlockSize, "output tile must fit one thread each for the final reduce");
  using vec_t = AlignedVector<bf16_t, kTinyNGemmVecSize>;

  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const bf16_t* w_tile = w + bx * (N_SPLIT * K);

  // Weight prefetch: address is input-independent, load before the PDL wait.
  vec_t wv[N_SPLIT];
#pragma unroll
  for (uint32_t n = 0; n < N_SPLIT; ++n) {
    wv[n].load(w_tile + n * K, tx);
  }

  PDLWaitPrimary<kUsePDL>();

  vec_t xv[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    xv[m].load(x + m * K, tx);
  }

  __shared__ float s_acc[kNumWarps][M * N_SPLIT];
  const uint32_t warp_id = tx / kWarpThreads;

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
    for (uint32_t n = 0; n < N_SPLIT; ++n) {
      float acc = 0.0f;
#if SGL_ARCH_BLACKWELL_OR_GREATER
#pragma unroll
      for (uint32_t i = 0; i < kTinyNGemmVecSize; ++i) {
        acc = device::math::fma_f32_bf16(xv[m][i], wv[n][i], acc);
      }
#else
      for (uint32_t i = 0; i < kTinyNGemmVecSize / 2; ++i) {
        const auto [x0, x1] = cast<fp32x2_t>(bf16x2_t{xv[m][2 * i], xv[m][2 * i + 1]});
        const auto [w0, w1] = cast<fp32x2_t>(bf16x2_t{wv[n][2 * i], wv[n][2 * i + 1]});
        acc = fmaf(x0, w0, acc);
        acc = fmaf(x1, w1, acc);
      }
#endif
      // NOTE: broadcast write (all lanes hold the reduced value), safe here.
      s_acc[warp_id][m * N_SPLIT + n] = warp::reduce_sum(acc);
    }
  }
  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

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
    out[m * N + bx * N_SPLIT + n] = cast<OutT>(acc[0]);
  }
}

SGL_DEVICE void cp_async_cg_16(void* smem_dst, const void* gmem_src, int32_t vec_offset) {
  const uint32_t offset = static_cast<uint32_t>(vec_offset * 16);
#if defined(USE_ROCM)
  *reinterpret_cast<uint4*>(static_cast<char*>(smem_dst) + offset) =
      *reinterpret_cast<const uint4*>(static_cast<const char*>(gmem_src) + offset);
#else
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst)) + offset;
  const uint64_t gmem_addr = static_cast<uint64_t>(__cvta_generic_to_global(gmem_src)) + offset;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_addr) : "memory");
#endif
}

constexpr uint32_t kTinyKGemmVecSize = 16 / sizeof(bf16_t);  // NOTE: no need to be large

template <uint32_t M, uint32_t N, uint32_t K, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
__global__ __launch_bounds__(N_SPLIT* K / kTinyKGemmVecSize, 1)  // control the block size
    void tiny_k_gemm_kernel(
        OutT* __restrict__ out, const bf16_t* __restrict__ x, const bf16_t* __restrict__ w, const int64_t dx) {
  using vec_t = AlignedVector<bf16_t, kTinyKGemmVecSize>;
  constexpr uint32_t kNumKLanes = K / kTinyKGemmVecSize;
  static_assert(std::has_single_bit(kNumKLanes), "K / vec_size must be a power of 2");
  static_assert(kNumKLanes <= kWarpThreads, "require in-warp reduction");
  static_assert((N_SPLIT * K / kTinyKGemmVecSize) % kWarpThreads == 0);
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  const uint32_t n_idx = bx * N_SPLIT + tx / kNumKLanes;
  const uint32_t work_id = tx % kNumKLanes;
  const bf16_t* w_tile = w + n_idx * K;

  // Weight prefetch: address is input-independent, load before the PDL wait.
  vec_t wv;
  wv.load(w_tile, work_id);

  PDLWaitPrimary<kUsePDL>();
  vec_t xv[M];
#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    xv[m].load(x + m * dx, work_id);
  }

#pragma unroll
  for (uint32_t m = 0; m < M; ++m) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kTinyKGemmVecSize; ++i) {
      acc = device::math::fma_f32_bf16(xv[m][i], wv[i], acc);
    }
    // Broadcast store: every lane of the group holds the reduced sum.
    out[m * N + n_idx] = cast<OutT>(warp::reduce_sum<kNumKLanes>(acc));
  }
  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace sglang

using namespace sglang;

template <uint32_t N, uint32_t K, uint32_t kMaxM, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
struct TinyNGemmKernel {
  static constexpr uint32_t kBlockSize = K / kTinyNGemmVecSize;
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static_assert(K % kTinyNGemmVecSize == 0, "K must be divisible by the vector width");
  static_assert(kBlockSize % kWarpThreads == 0, "K / vec_size must be a multiple of the warp size");
  static_assert(kBlockSize <= 1024, "K / vec_size exceeds the maximum block size");
  static_assert(N % N_SPLIT == 0, "N must be divisible by split_n");
  static_assert(kMaxM * N_SPLIT <= kBlockSize, "max_m * split_n must fit one thread each for the final reduce");

  using KernelFn = void (*)(OutT*, const bf16_t*, const bf16_t*);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxM + 1>{nullptr, tiny_n_gemm_kernel<I + 1, N, K, N_SPLIT, OutT, kUsePDL>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxM>{});

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    RuntimeCheck(num_tokens >= 1 && num_tokens <= kMaxM);
    LaunchKernel(kNumBlocks, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(
            kTable[num_tokens],
            static_cast<OutT*>(out.data_ptr()),
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(w.data_ptr()));
  }
};

template <uint32_t N, uint32_t K, uint32_t kMaxM, uint32_t N_SPLIT, typename OutT, bool kUsePDL>
struct TinyKGemmKernel {
  static constexpr uint32_t kNumKLanes = K / kTinyKGemmVecSize;
  static constexpr uint32_t kBlockSize = N_SPLIT * kNumKLanes;
  static constexpr uint32_t kNumBlocks = N / N_SPLIT;
  static_assert(K % kTinyKGemmVecSize == 0, "K must be divisible by the vector width");
  static_assert(N % N_SPLIT == 0, "N must be divisible by split_n");
  static_assert(kBlockSize % kWarpThreads == 0, "split_n * K-lanes must fill whole warps");
  static_assert(kBlockSize <= 1024, "split_n * K-lanes exceeds the maximum block size");

  using KernelFn = void (*)(OutT*, const bf16_t*, const bf16_t*, int64_t);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxM + 1>{nullptr, tiny_k_gemm_kernel<I + 1, N, K, N_SPLIT, OutT, kUsePDL>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxM>{});

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    // x may be a row-sliced view of a wider fused buffer: allow stride != K.
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_strides({-1, 1}).with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<OutT>().with_device(device).verify(out);
    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const auto x_stride = static_cast<int64_t>(x.stride(0));
    RuntimeCheck(num_tokens >= 1 && num_tokens <= kMaxM);
    RuntimeCheck(
        x_stride * sizeof(bf16_t) % (kTinyKGemmVecSize * sizeof(bf16_t)) == 0,
        "x rows must stay aligned to the vector width, got stride ",
        x_stride);
    LaunchKernel(kNumBlocks, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(
            kTable[num_tokens],
            static_cast<OutT*>(out.data_ptr()),
            static_cast<const bf16_t*>(x.data_ptr()),
            static_cast<const bf16_t*>(w.data_ptr()),
            x_stride);
  }
};
