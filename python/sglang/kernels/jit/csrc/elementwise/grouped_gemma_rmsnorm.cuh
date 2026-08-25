#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct GroupedGemmaRMSNormParams {
  const void* input;
  const void* __restrict__ weight;
  void* output;
  uint32_t num_groups;
  float eps;
};

/**
 * \brief Grouped Gemma-style RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + w).
 *
 * The last dim of the input is split into `num_groups` chunks of `kGroupSize`
 * elements. Variance is computed per (token, group) chunk, so a [M, H] input
 * with H = num_groups * kGroupSize behaves like M * num_groups independent
 * RMSNorm rows whose weight rows are the matching kGroupSize slice of `weight`.
 *
 * One CTA handles one (token, group) chunk. Since chunks are contiguous in
 * memory, block `bid` reads/writes elements [bid * kGroupSize, (bid + 1) *
 * kGroupSize) and uses weight slice (bid % num_groups) * kGroupSize.
 *
 * \tparam kGroupSize Elements per group. Must be a multiple of 512.
 * \tparam kUsePDL    Whether to emit the PDL wait/trigger pair.
 * \tparam Float      Element type: bf16_t | fp16_t.
 */
template <int64_t kGroupSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kGroupSize / 16) void grouped_gemma_rmsnorm_kernel(
    const GroupedGemmaRMSNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  // Blackwell: 32B vector, each thread loads/stores once
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kNumLoads = 1;
#else
  // Pre-Blackwell: 16B vector, each thread loads/stores twice
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kNumLoads = 2;
#endif
  constexpr uint32_t kVecLen = kNumLoads == 1 ? 8 : 4;
  constexpr auto kNumThreads = kGroupSize / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;

  const uint32_t bid = blockIdx.x;
  const uint32_t group = bid % params.num_groups;
  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  // Warp 0 writes smem[tx] for all 32 lanes in the cross-warp reduce below,
  // so this must hold kWarpThreads entries, not kNumWarps.
  __shared__ float smem[kWarpThreads];

  PDLWaitPrimary<kUsePDL>();

  const auto input_ptr =
      pointer::offset<Float>(params.input, static_cast<int64_t>(bid) * kGroupSize);
  const auto output_ptr =
      pointer::offset<Float>(params.output, static_cast<int64_t>(bid) * kGroupSize);
  const auto weight_ptr =
      pointer::offset<Float>(params.weight, static_cast<int64_t>(group) * kGroupSize);

  Storage input_vec[kNumLoads];
  Storage weight_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    input_vec[j] = gmem.load(input_ptr, j);
    weight_vec[j] = gmem.load(weight_ptr, j);
  }

  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);
      sum_of_squares += x * x + y * y;
    }
  }

  sum_of_squares = warp::reduce_sum(sum_of_squares);
  float norm_factor;
  if constexpr (kNumWarps == 1) {
    norm_factor = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
  } else {
    const auto warp_id = threadIdx.x / kWarpThreads;
    smem[warp_id] = sum_of_squares;
    __syncthreads();
    if (warp_id == 0) {
      const auto tx = threadIdx.x;
      const auto local_sum = tx < kNumWarps ? smem[tx] : 0.0f;
      sum_of_squares = warp::reduce_sum(local_sum);
      smem[tx] = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
    }
    __syncthreads();
    norm_factor = smem[warp_id];
  }

#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    Storage output_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [ix, iy] = cast<fp32x2_t>(input_vec[j][i]);
      const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j][i]);
      output_vec[i] = cast<Float2>(
          fp32x2_t{ix * norm_factor * (1.0f + wx), iy * norm_factor * (1.0f + wy)});
    }
    gmem.store(output_ptr, output_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kGroupSize, bool kUsePDL, typename DType>
struct GroupedGemmaRMSNormKernel {
  static_assert(sizeof(DType) == 2, "GroupedGemmaRMSNorm only supports 2-byte dtypes");
  static_assert(kGroupSize % 512 == 0, "kGroupSize must be a multiple of 512");
  static constexpr auto kernel = grouped_gemma_rmsnorm_kernel<kGroupSize, kUsePDL, DType>;
  static constexpr auto kBlockSize = static_cast<uint32_t>(kGroupSize / 16);

  /**
   * \brief Validate tensors and launch one CTA per (token, group) chunk.
   * \param input  [M, H] contiguous, H % kGroupSize == 0
   * \param weight [H]
   * \param output [M, H] contiguous, same shape/dtype/device as input
   * \param eps    RMSNorm epsilon
   */
  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto H = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, H})  // input
        .with_dtype<DType>()
        .with_device(device)
        .verify(input);
    TensorMatcher({H})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);
    TensorMatcher({M, H})  // output
        .with_dtype<DType>()
        .with_device(device)
        .verify(output);

    const int64_t hidden_size = H.unwrap();
    CHECK_HOST(hidden_size % kGroupSize == 0)
        << "grouped_gemma_rmsnorm: hidden_size (" << hidden_size
        << ") must be divisible by group_size (" << kGroupSize << ")";

    const auto params = GroupedGemmaRMSNormParams{
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .num_groups = static_cast<uint32_t>(hidden_size / kGroupSize),
        .eps = eps,
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const uint32_t num_blocks = num_tokens * params.num_groups;
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
