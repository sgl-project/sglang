#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct GatedResidualCombineNormParams {
  const void* block_output;     // [M, H]
  const void* residual;         // [M, HC * H]
  const void* inject_logits;    // [M, HC]
  const void* weight;           // [H] or [group_size]
  void* output;                 // [M, HC * H]
  void* normed_output;          // [M, HC * H]
  uint32_t num_groups;
  float eps;
};

/**
 * \brief Fused gated residual combine + grouped Gemma RMSNorm.
 *
 *   combined[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]
 *   where a[m, c] = 2 * sigmoid(inject_logits[m, c] / kHcCount)
 *
 *   Then per-group RMSNorm on combined:
 *   normed[m, g*G + j] = combined[m, g*G + j] * rsqrt(mean(combined[m, g*G:(g+1)*G]^2) + eps) * (1 + w[g*G + j])
 *
 * One CTA handles one (token, group) chunk. Phase 1 computes gate values from
 * inject_logits (small, fits in registers). Phase 2 loads residual + block_output,
 * computes combined, then RMSNorm, writing both combined and normed outputs.
 *
 * \tparam kHcCount    Number of hyper-connection branches (4 in production).
 * \tparam kHiddenSize Per-branch hidden size H.
 * \tparam kGroupSize  Elements per norm group (multiple of 512).
 * \tparam kUsePDL     Whether to emit the PDL wait/trigger pair.
 * \tparam Float       Element type: bf16_t | fp16_t.
 */
template <int64_t kHcCount, int64_t kHiddenSize, int64_t kGroupSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(kGroupSize / 16) void gated_residual_combine_norm_kernel(
    const GatedResidualCombineNormParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  using Storage = AlignedVector<Float2, 8>;
  constexpr uint32_t kNumLoads = 1;
#else
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kNumLoads = 2;
#endif
  constexpr uint32_t kVecLen = kNumLoads == 1 ? 8 : 4;
  constexpr auto kNumThreads = kGroupSize / 16;
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerGroup = kGroupSize / kVecLen;

  const uint32_t bid = blockIdx.x;
  const uint32_t group = bid % params.num_groups;
  const uint32_t token = bid / params.num_groups;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  __shared__ float smem[kWarpThreads];

  // Load gate values for this token (small, fits in registers)
  const auto logits_ptr = pointer::offset<Float>(params.inject_logits, static_cast<int64_t>(token) * kHcCount);
  float a[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const float logit = cast<float>(logits_ptr[c]);
    a[c] = 2.0f / (1.0f + math::exp(-logit / kHcCount));
  }

  // PDL wait after loading small gate values, before loading large tensors
  PDLWaitPrimary<kUsePDL>();

  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(token) * kRowSize + group * kGroupSize);
  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(token) * kHiddenSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(token) * kRowSize + group * kGroupSize);
  const auto norm_ptr = pointer::offset<Float>(params.normed_output, static_cast<int64_t>(token) * kRowSize + group * kGroupSize);
  const auto w_ptr = pointer::offset<Float>(params.weight, static_cast<int64_t>(group) * kGroupSize);

  // Load residual and block_output, compute combined
  Storage r_vec[kNumLoads];
  Storage y_vec[kNumLoads];
  Storage combined_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    r_vec[j] = gmem.load(r_ptr, j);
    // block_output is [M, H], we need the slice for this group
    y_vec[j] = gmem.load(y_ptr + group * kGroupSize, j);
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[j][i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[j][i]);
      const uint32_t branch = (threadIdx.x * kVecLen + j * kNumThreads * kVecLen + i * 2) / kHiddenSize;
      combined_vec[j][i] = cast<Float2>(fp32x2_t{rx + a[branch] * yx, ry + a[branch] * yy});
    }
  }

  // Store combined output
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    gmem.store(out_ptr, combined_vec[j], j);
  }

  // Compute RMSNorm on combined
  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [x, y] = cast<fp32x2_t>(combined_vec[j][i]);
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

  // Load weight and apply normalization
  Storage w_vec[kNumLoads];
#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    w_vec[j] = gmem.load(w_ptr, j);
  }

#pragma unroll
  for (uint32_t j = 0; j < kNumLoads; ++j) {
    Storage norm_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen; ++i) {
      const auto [cx, cy] = cast<fp32x2_t>(combined_vec[j][i]);
      const auto [wx, wy] = cast<fp32x2_t>(w_vec[j][i]);
      norm_vec[i] = cast<Float2>(fp32x2_t{cx * norm_factor * (1.0f + wx), cy * norm_factor * (1.0f + wy)});
    }
    gmem.store(norm_ptr, norm_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, int64_t kGroupSize, bool kUsePDL, typename DType>
struct GatedResidualCombineNormKernel {
  static_assert(sizeof(DType) == 2, "GatedResidualCombineNorm only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert(kGroupSize > 0 && kGroupSize % 512 == 0, "kGroupSize must be a multiple of 512");
  static_assert(kHiddenSize % kGroupSize == 0, "kHiddenSize must be divisible by kGroupSize");
  static constexpr auto kernel = gated_residual_combine_norm_kernel<kHcCount, kHiddenSize, kGroupSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = static_cast<uint32_t>(kGroupSize / 16);

  /**
   * \brief Validate tensors and launch one CTA per (token, group) chunk.
   * \param block_output   [M, H] contiguous
   * \param residual       [M, HC * H] contiguous
   * \param inject_logits  [M, HC] contiguous
   * \param weight         [H] contiguous
   * \param output         [M, HC * H] contiguous
   * \param normed_output  [M, HC * H] contiguous
   * \param eps            RMSNorm epsilon
   */
  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView inject_logits,
      const tvm::ffi::TensorView weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView normed_output,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize})  // block_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})  // residual, output, normed_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(output)
        .verify(normed_output);
    TensorMatcher({M, kHcCount})  // inject_logits
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_logits);
    TensorMatcher({kHiddenSize})  // weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(weight);

    const auto params = GatedResidualCombineNormParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .inject_logits = inject_logits.data_ptr(),
        .weight = weight.data_ptr(),
        .output = output.data_ptr(),
        .normed_output = normed_output.data_ptr(),
        .num_groups = static_cast<uint32_t>(kHiddenSize / kGroupSize),
        .eps = eps,
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    const uint32_t num_blocks = num_tokens * params.num_groups;
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
