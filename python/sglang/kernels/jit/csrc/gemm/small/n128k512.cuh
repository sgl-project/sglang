/// \file n128k512.cuh
/// \brief Small bf16 GEMM specialised for N = 128, K = 512: `out[m, n] = sum_k a[m, k] * b[n, k]`.
///
/// One warp owns one output column n and keeps that whole 1 KB weight row in
/// registers (32 lanes x 32 bytes); it is prefetched before the PDL wait so the
/// load overlaps the tail of the previous kernel. Rows of `a` are handled in
/// groups of `M_SPLIT` (at most 8) per warp along `blockIdx.y`; the batch size is
/// read from the params at run time, so eight compiled kernels serve every M.
/// The reduction is a per-lane in-order fma over 16 elements followed by a warp
/// butterfly, the same as `tiny_n_gemm_kernel`: results are row-invariant across
/// M and bitwise equal to tiny_gemm where both apply.

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
struct N128K512Trait {
  static constexpr uint32_t N = 128;
  static constexpr uint32_t K = 512;
  static constexpr uint32_t kMaxMSplit = 8;
  static constexpr uint32_t M_SPLIT = M_SPLIT_;  // rows of `a` one warp handles
  static_assert(M_SPLIT >= 1 && M_SPLIT <= kMaxMSplit, "M_SPLIT must be in [1, 8]");
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kNumVecs = K / (kVecSize * device::kWarpThreads);
  static_assert(K % (kVecSize * device::kWarpThreads) == 0, "K must be a whole number of warp-wide vectors");
  using vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  static constexpr uint32_t kBlockSize = 128;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(N % kNumWarps == 0, "every warp of a block owns one output column");
  static constexpr uint32_t kGridX = N / kNumWarps;
};

struct N128K512Params {
  bf16_t* __restrict__ out;
  const bf16_t* __restrict__ a;
  const bf16_t* __restrict__ b;
  int64_t stride_a;  // row stride of `a`, in elements
  uint32_t m;        // batch size
};

template <typename T, bool kUsePDL>
__global__ __launch_bounds__(T::kBlockSize, 1) void n128k512_kernel(const N128K512Params params) {
  using namespace device;
  constexpr uint32_t kNumVecs = T::kNumVecs;
  constexpr uint32_t M_SPLIT = T::M_SPLIT;
  constexpr uint32_t K = T::K;
  constexpr uint32_t N = T::N;
  using vec_t = typename T::vec_t;

  const uint32_t M = params.m;
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t n = blockIdx.x * T::kNumWarps + warp_id;
  const uint32_t m_start = blockIdx.y * M_SPLIT;
  const auto gmem = tile::Memory<vec_t>::warp();

  // The weight row does not depend on the previous kernel: load it before the PDL wait.
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

#pragma unroll
  for (uint32_t i = 0; i < M_SPLIT; ++i) {
    float acc = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kNumVecs; ++j) {
      dot_product_vec(a[i][j], b[j], acc);
    }
    acc = warp::reduce_sum(acc);
    const uint32_t m = m_start + i;
    if (m < M) {
      params.out[m * N + n] = cast<bf16_t>(acc);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct N128K512Kernel {
  using Trait1 = N128K512Trait<1>;
  static constexpr uint32_t kMaxMSplit = Trait1::kMaxMSplit;
  using KernelFn = void (*)(N128K512Params);

  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxMSplit + 1>{nullptr, n128k512_kernel<N128K512Trait<I + 1>, kUsePDL>...};
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
    // Spread the rows evenly over as few y-blocks as eight rows per warp allow.
    const uint32_t grid_y = div_ceil(m, kMaxMSplit);
    const uint32_t m_split = div_ceil(m, grid_y);
    const auto params = N128K512Params{
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
