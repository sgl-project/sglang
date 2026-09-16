/// \file n32k5120.cuh
/// \brief Small bf16 GEMM for N = 32, K = 5120: `out[m, n] = sum_k a[m, k] * b[n, k]`.
///
/// One CTA owns an output column; threads share its 10 KB weight row and
/// prefetch it before the PDL wait. Groups of at most 8 input rows map to
/// blockIdx.y, with batch size read at run time.
///
/// The thread-to-K mapping and dot_product_vec reduction order match tiny_gemm's
/// N-variant, giving bitwise parity where both apply (m <= 16) and row invariance
/// across M. Accumulation is fp32 with one final bf16 rounding; cuBLAS uses
/// a different reduction order.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/gemm/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sglang {

template <uint32_t M_SPLIT_>
struct N32K5120Trait {
  static constexpr uint32_t N = 32;
  static constexpr uint32_t K = 5120;
  static constexpr uint32_t kMaxMSplit = 8;
  static constexpr uint32_t M_SPLIT = M_SPLIT_;  // rows of `a` one CTA handles
  static_assert(M_SPLIT >= 1 && M_SPLIT <= kMaxMSplit, "M_SPLIT must be in [1, 8]");
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  // Ten warps preserve tiny_gemm's mapping: thread t owns [16t, 16t + 16).
  // B200 measurements found no latency benefit from a 160-thread alternative.
  static constexpr uint32_t kBlockSize = 320;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kNumVecs = K / (kVecSize * kBlockSize);  // vectors per thread
  static_assert(K % (kVecSize * kBlockSize) == 0, "K must be a whole number of CTA-wide vectors");
  static_assert(kBlockSize % device::kWarpThreads == 0, "the reduction needs whole warps");
  static_assert(M_SPLIT <= kBlockSize, "one thread finishes one row");
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  static constexpr uint32_t kGridX = N;  // one CTA per output column
};

struct N32K5120Params {
  bf16_t* __restrict__ out;
  const bf16_t* __restrict__ a;
  const bf16_t* __restrict__ b;
  int64_t stride_a;  // row stride of `a`, in elements
  uint32_t m;        // batch size
};

template <typename T, bool kUsePDL>
__global__ __launch_bounds__(T::kBlockSize, 1) void n32k5120_kernel(const N32K5120Params params) {
  using namespace device;
  constexpr uint32_t kNumVecs = T::kNumVecs;
  constexpr uint32_t kNumWarps = T::kNumWarps;
  constexpr uint32_t M_SPLIT = T::M_SPLIT;
  constexpr uint32_t K = T::K;
  constexpr uint32_t N = T::N;
  using vec_t = typename T::vec_t;

  const uint32_t M = params.m;
  const uint32_t tx = threadIdx.x;
  const uint32_t warp_id = tx / kWarpThreads;
  const uint32_t n = blockIdx.x;
  const uint32_t m_start = blockIdx.y * M_SPLIT;
  const auto gmem = tile::Memory<vec_t>::cta(T::kBlockSize);

  // The weight column does not depend on the previous kernel: load it before the PDL wait.
  vec_t b[kNumVecs];
#pragma unroll
  for (uint32_t j = 0; j < kNumVecs; ++j) {
    b[j] = gmem.load(params.b + n * K, j);
  }

  PDLWaitPrimary<kUsePDL>();

  // Clamp padded loads to the last valid row to keep vector loads branch-free;
  // only stores are guarded.
  vec_t a[M_SPLIT][kNumVecs];
#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    const uint32_t m = min(m_start + i, M - 1);
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      a[i][j] = gmem.load(params.a + m * params.stride_a, j);
    }
  }

  __shared__ float s_acc[kNumWarps][M_SPLIT];
#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      dot_product_vec(a[i][j], b[j], acc);
    }
    acc = warp::reduce_sum(acc);
    if (tx % kWarpThreads == 0) s_acc[warp_id][i] = acc;
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();

  // One thread per row sums the warps in a fixed order, so a row's result does
  // not depend on which rows share its CTA.
  if (tx < M_SPLIT) {
    float acc = s_acc[0][tx];
#pragma unroll
    for (uint32_t w = 1; w < kNumWarps; ++w) {
      acc += s_acc[w][tx];
    }
    const uint32_t m = m_start + tx;
    if (m < M) {
      params.out[m * N + n] = cast<bf16_t>(acc);
    }
  }
}

template <bool kUsePDL>
struct N32K5120Kernel {
  using Trait1 = N32K5120Trait<1>;
  static constexpr uint32_t kMaxMSplit = Trait1::kMaxMSplit;
  using KernelFn = void (*)(N32K5120Params);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxMSplit + 1>{nullptr, n32k5120_kernel<N32K5120Trait<I + 1>, kUsePDL>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxMSplit>{});

  static void run(const tvm::ffi::TensorView a, const tvm::ffi::TensorView b, const tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, Trait1::K}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(a);
    TensorMatcher({Trait1::N, Trait1::K}).with_dtype<bf16_t>().with_device(device).verify(b);
    TensorMatcher({M, Trait1::N}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto m = static_cast<uint32_t>(M.unwrap());
    if (m == 0) return;
    // Rows are loaded as whole vectors, so a row-sliced view must keep its row starts vector-aligned.
    CHECK_HOST(a.stride(0) % Trait1::kVecSize == 0)
        << "a rows must stay aligned to the vector width, got stride " << a.stride(0);
    // Spread the rows evenly over as few y-blocks as eight rows per CTA allow.
    const uint32_t grid_y = div_ceil(m, kMaxMSplit);
    const uint32_t m_split = div_ceil(m, grid_y);
    const auto params = N32K5120Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .a = static_cast<const bf16_t*>(a.data_ptr()),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .stride_a = static_cast<int64_t>(a.stride(0)),
        .m = m,
    };
    LaunchKernel(dim3(Trait1::kGridX, grid_y), dim3(Trait1::kBlockSize), device.unwrap())
        .enable_pdl(kUsePDL)(kTable[m_split], params);
  }
};

}  // namespace sglang
