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

struct HcCombineParams {
  const void* block_output;     // [M, H]
  const void* residual;         // [M, HC * H]
  const void* normed_residual;  // [M, HC * H]
  const void* inject_weight;    // [HC, HC * H]
  void* output;                 // [M, HC * H]
};

/**
 * \brief Fused HyperConnection (gated residual) combine:
 *
 *   a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / kHcCount)
 *   output[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]
 *
 * One CTA handles one token row. Phase 1 computes the kHcCount gate values with a
 * block reduction over the full HC*H row (fp32 accumulation). Phase 2 streams the
 * HC*H output elements with vectorized 16B accesses; the block_output row is only
 * H elements, so its re-read per branch stays in L2.
 *
 * \tparam kHcCount    Number of hyper-connection branches (4 in production).
 * \tparam kHiddenSize Per-branch hidden size H. HC*H must be a multiple of
 *                     kNumThreads * kVecLen so the row maps exactly onto the CTA.
 * \tparam kUsePDL     Whether to emit the PDL wait/trigger pair.
 * \tparam Float       Element type: bf16_t | fp16_t.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_combine_kernel(const HcCombineParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;            // 1280 for 4x2560
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;  // 5 for 4x2560
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;      // 320 for 2560
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: gate values a_c, accumulated in fp32 and reduced across the CTA.
  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j] = gmem.load(n_ptr, j);
  }

  float acc[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    acc[c] = warp::reduce_sum(sum);
  }

  __shared__ float smem[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      smem[c][warp_id] = acc[c];
    }
  }
  __syncthreads();
  __shared__ float a_shared[kHcCount];
  if (threadIdx.x < kHcCount) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += smem[threadIdx.x][w];
    }
    a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
  }
  __syncthreads();

  // Phase 2: stream the output row. Vector `vec_idx` lies entirely inside
  // branch `vec_idx / kVecsPerBranch` (H is a multiple of kVecLen).
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_shared[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, out_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel = hc_combine_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  /**
   * \brief Validate tensors and launch one CTA per token row.
   * \param block_output    [M, H] contiguous
   * \param residual        [M, HC * H] contiguous
   * \param normed_residual [M, HC * H] contiguous, same dtype/device as residual
   * \param inject_weight   [HC, HC * H] contiguous, same dtype/device as residual
   * \param output          [M, HC * H] contiguous, same shape/dtype/device as residual
   */
  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, kHiddenSize})  // block_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})  // residual, normed_residual, output
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize})  // inject_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_weight);

    const auto params = HcCombineParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

struct HcCombineSplitParams {
  const void* block_output;
  const void* residual;
  const void* normed_residual;
  const void* inject_weight;
  void* output;
  float* partials;
};

namespace hc_combine_split_detail {

// One CTA per warp of the original kernel: CTA s replays exactly the work of
// threads [32s, 32s+32) of the 256-thread reference, so every float is
// accumulated in the reference order and results stay bit-identical.
constexpr uint32_t kSplit = 8;
constexpr uint32_t kRefThreads = 256;
constexpr uint32_t kGateThreads = 32;
constexpr uint32_t kApplyThreads = 160;
constexpr uint32_t kVecLen = 8;

}  // namespace hc_combine_split_detail

/**
 * \brief Stage 1 of the split combine: partial gate dots over a K slice.
 *
 * Grid is (rows, kSplit) so the [HC, HC*H] inject weight is read once across
 * the whole grid instead of once per row, and the row's traffic is spread over
 * kSplit CTAs. Each CTA writes its own partials slot, so no atomics and no
 * buffer clearing are needed.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kGateThreads) void hc_combine_gate_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kVecLen = 8;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kRefThreads;
  static_assert(kVecsPerRow % kRefThreads == 0);
  static_assert(kRefThreads / kGateThreads == kSplit);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y / kHcCount;
  const uint32_t c = blockIdx.y % kHcCount;
  const uint32_t ref_tid = split * kGateThreads + threadIdx.x;

  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);

  PDLWaitPrimary<kUsePDL>();

  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j].load(n_ptr, ref_tid + j * kRefThreads);
  }

  {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      Storage w_vec;
      w_vec.load(wc_ptr, ref_tid + j * kRefThreads);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    sum = warp::reduce_sum(sum);
    if (threadIdx.x == 0) {
      params.partials[(static_cast<int64_t>(m) * kSplit + split) * kHcCount + c] = sum;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Stage 2: reduce the partial dots and stream the combined row.
 *
 * Each CTA owns one contiguous vector slice of the row. kSplit divides
 * kHiddenSize, so a slice never straddles two branches and the gate is a
 * per-CTA scalar.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kApplyThreads) void hc_combine_apply_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerSplit = kVecsPerRow / kSplit;
  constexpr uint32_t kVecsPerThread = kVecsPerSplit / kApplyThreads;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  static_assert(kVecsPerBranch % kVecsPerSplit == 0);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t vec_base = split * kVecsPerSplit;
  const uint32_t branch = vec_base / kVecsPerBranch;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  float total = 0.0f;
#pragma unroll
  for (uint32_t s = 0; s < kSplit; ++s) {
    total += params.partials[(static_cast<int64_t>(m) * kSplit + s) * kHcCount + branch];
  }
  const float a = 2.0f / (1.0f + math::exp(-total / kHcCount));

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = vec_base + threadIdx.x + j * kApplyThreads;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch);
    Storage r_vec;
    r_vec.load(r_ptr, vec_idx);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    out_vec.store(out_ptr, vec_idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineSplitKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static constexpr auto gate_kernel = hc_combine_gate_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr auto apply_kernel = hc_combine_apply_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView partials) {
    using namespace host;
    using namespace hc_combine_split_detail;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(inject_weight);
    auto part_rows = SymbolicSize{"partial_rows"};
    TensorMatcher({part_rows, kSplit, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);

    const auto params = HcCombineSplitParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
        .partials = static_cast<float*>(partials.data_ptr()),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;
    LaunchKernel(dim3(num_tokens, kSplit * kHcCount, 1), kGateThreads, device.unwrap())
        .enable_pdl(kUsePDL)(gate_kernel, params);
    LaunchKernel(dim3(num_tokens, kSplit, 1), kApplyThreads, device.unwrap()).enable_pdl(kUsePDL)(apply_kernel, params);
  }
};

struct HcCombineLogitsParams {
  const void* block_output;  // [M, H]
  const void* residual;      // [M, HC * H]
  const void* gate_logits;   // [M, HC] with row stride gate_stride
  int64_t gate_stride;
  const void* norm_weight;   // [HC * H] or nullptr
  void* combined;            // [M, HC * H]
  void* normalized;          // [M, HC * H] or nullptr
  float eps;
};

namespace hc_combine_logits_detail {

template <int64_t kHiddenSize>
constexpr bool use_32b_vec() {
  return (device::kMaxVecBytes >= 32) && (kHiddenSize % 16 == 0) &&
         ((kHiddenSize / 16) % 32 == 0) && ((kHiddenSize / 16) <= 256);
}

template <int64_t kHiddenSize>
constexpr uint32_t select_vec_len() {
  return use_32b_vec<kHiddenSize>() ? 16u : 8u;
}

template <int64_t kHiddenSize>
constexpr uint32_t select_num_threads() {
  constexpr uint32_t vecs = static_cast<uint32_t>(kHiddenSize / select_vec_len<kHiddenSize>());
  if constexpr (use_32b_vec<kHiddenSize>()) {
    return vecs;
  } else if constexpr (vecs <= 256u && (vecs % 32u == 0u)) {
    return vecs;
  } else {
    return 256u;
  }
}

}  // namespace hc_combine_logits_detail

/**
 * \brief Fused HyperConnection combine (with optional GroupedGemmaRMSNorm epilogue)
 *        consuming precomputed TensorCore gate logits [M, HC].
 *
 * Grid is (M,): one CTA per token. block_output [M, H] is loaded once into
 * registers and reused across all kHcCount branches, avoiding redundant HBM
 * reads while keeping each branch's RMSNorm reduction within one CTA.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kFuseNorm, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_logits_detail::select_num_threads<kHiddenSize>()) void hc_combine_logits_kernel(
    const HcCombineLogitsParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  constexpr uint32_t kVecLen = hc_combine_logits_detail::select_vec_len<kHiddenSize>();
  constexpr uint32_t kPairsPerVec = kVecLen / 2;
  using Storage = AlignedVector<Float2, kPairsPerVec>;
  constexpr uint32_t kNumThreads = hc_combine_logits_detail::select_num_threads<kHiddenSize>();
  constexpr uint32_t kVecsPerBranch = static_cast<uint32_t>(kHiddenSize / kVecLen);
  constexpr uint32_t kIters = (kVecsPerBranch + kNumThreads - 1) / kNumThreads;
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;
  static_assert(kNumWarps <= kWarpThreads, "kNumThreads must not exceed 1024");

  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(row) * kHiddenSize);
  const auto logits_ptr = static_cast<const Float*>(params.gate_logits);
  const int64_t row_base = static_cast<int64_t>(row) * kHcCount * kHiddenSize;

  PDLWaitPrimary<kUsePDL>();

  // Load block_output once into registers and reuse across all kHcCount branches.
  Storage y_reg[kIters];
#pragma unroll
  for (uint32_t j = 0; j < kIters; ++j) {
    const uint32_t vec_idx = tid + j * kNumThreads;
    if (kVecsPerBranch % kNumThreads == 0 || vec_idx < kVecsPerBranch) {
      y_reg[j].load(y_ptr, vec_idx);
    }
  }

  float gates[kHcCount];
  const int64_t logit_row_offset = static_cast<int64_t>(row) * params.gate_stride;
#pragma unroll
  for (uint32_t b = 0; b < static_cast<uint32_t>(kHcCount); ++b) {
    const float logit = cast<float>(logits_ptr[logit_row_offset + b]);
    gates[b] = 2.0f / (1.0f + math::exp(-logit / static_cast<float>(kHcCount)));
  }

  __shared__ float warp_sums[kNumWarps];
  __shared__ float inv_rms_shared;
  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t lane = tid % kWarpThreads;

#pragma unroll
  for (uint32_t branch = 0; branch < static_cast<uint32_t>(kHcCount); ++branch) {
    const int64_t branch_offset = row_base + static_cast<int64_t>(branch) * kHiddenSize;
    const auto r_ptr = pointer::offset<Float>(params.residual, branch_offset);
    const auto out_ptr = pointer::offset<Float>(params.combined, branch_offset);
    const float gate = gates[branch];

    if constexpr (!kFuseNorm) {
#pragma unroll
      for (uint32_t j = 0; j < kIters; ++j) {
        const uint32_t vec_idx = tid + j * kNumThreads;
        if (kVecsPerBranch % kNumThreads == 0 || vec_idx < kVecsPerBranch) {
          Storage r_vec, c_vec;
          r_vec.load(r_ptr, vec_idx);
#pragma unroll
          for (uint32_t i = 0; i < kPairsPerVec; ++i) {
            const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
            const auto [yx, yy] = cast<fp32x2_t>(y_reg[j][i]);
            c_vec[i] = cast<Float2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
          }
          c_vec.store(out_ptr, vec_idx);
        }
      }
    } else {
      const auto w_ptr = pointer::offset<Float>(params.norm_weight, static_cast<int64_t>(branch) * kHiddenSize);
      const auto norm_ptr = pointer::offset<Float>(params.normalized, branch_offset);
      Storage combined_reg[kIters];
      float sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < kIters; ++j) {
        const uint32_t vec_idx = tid + j * kNumThreads;
        if (kVecsPerBranch % kNumThreads == 0 || vec_idx < kVecsPerBranch) {
          Storage r_vec, c_vec;
          r_vec.load(r_ptr, vec_idx);
#pragma unroll
          for (uint32_t i = 0; i < kPairsPerVec; ++i) {
            const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
            const auto [yx, yy] = cast<fp32x2_t>(y_reg[j][i]);
            // Match the BF16/FP16 store rounding before RMSNorm accumulation.
            c_vec[i] = cast<Float2>(fp32x2_t{rx + gate * yx, ry + gate * yy});
            const auto [cx, cy] = cast<fp32x2_t>(c_vec[i]);
            sum += cx * cx + cy * cy;
          }
          c_vec.store(out_ptr, vec_idx);
          combined_reg[j] = c_vec;
        }
      }

      sum = warp::reduce_sum(sum);
      if (lane == 0) {
        warp_sums[warp_id] = sum;
      }
      __syncthreads();
      if (warp_id == 0) {
        const float local_sum = lane < kNumWarps ? warp_sums[lane] : 0.0f;
        const float total = warp::reduce_sum(local_sum);
        if (lane == 0) {
          inv_rms_shared = math::rsqrt(total / static_cast<float>(kHiddenSize) + params.eps);
        }
      }
      __syncthreads();
      const float inv_rms = inv_rms_shared;

#pragma unroll
      for (uint32_t j = 0; j < kIters; ++j) {
        const uint32_t vec_idx = tid + j * kNumThreads;
        if (kVecsPerBranch % kNumThreads == 0 || vec_idx < kVecsPerBranch) {
          Storage w_vec, n_vec;
          w_vec.load(w_ptr, vec_idx);
#pragma unroll
          for (uint32_t i = 0; i < kPairsPerVec; ++i) {
            const auto [cx, cy] = cast<fp32x2_t>(combined_reg[j][i]);
            const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
            n_vec[i] = cast<Float2>(fp32x2_t{cx * inv_rms * (1.0f + wx), cy * inv_rms * (1.0f + wy)});
          }
          n_vec.store(norm_ptr, vec_idx);
        }
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineLogitsKernel {
  static_assert(sizeof(DType) == 2, "HcCombineLogits only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static constexpr uint32_t kNumThreads = hc_combine_logits_detail::select_num_threads<kHiddenSize>();
  static constexpr size_t kVecBytes = hc_combine_logits_detail::select_vec_len<kHiddenSize>() * sizeof(DType);

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView gate_logits,
      const tvm::ffi::TensorView combined) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto gate_stride = SymbolicSize{"gate_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(combined);
    TensorMatcher({M, kHcCount})
        .with_strides({gate_stride, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(gate_logits);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;

    RuntimeCheck(
        reinterpret_cast<uintptr_t>(block_output.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(residual.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(combined.data_ptr()) % kVecBytes == 0,
        "HcCombineLogits tensors must be ",
        kVecBytes,
        "-byte aligned");

    const auto params = HcCombineLogitsParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .gate_logits = gate_logits.data_ptr(),
        .gate_stride = gate_stride.unwrap(),
        .norm_weight = nullptr,
        .combined = combined.data_ptr(),
        .normalized = nullptr,
        .eps = 0.0f,
    };
    LaunchKernel(dim3(num_tokens, 1, 1), kNumThreads, device.unwrap())
        .enable_pdl(kUsePDL)(hc_combine_logits_kernel<kHcCount, kHiddenSize, false, kUsePDL, DType>, params);
  }

  static void
  run_norm(
      const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView gate_logits,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView combined,
      const tvm::ffi::TensorView normalized,
      float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto gate_stride = SymbolicSize{"gate_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(combined)
        .verify(normalized);
    TensorMatcher({kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(norm_weight);
    TensorMatcher({M, kHcCount})
        .with_strides({gate_stride, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(gate_logits);

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) return;

    RuntimeCheck(
        reinterpret_cast<uintptr_t>(block_output.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(residual.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(norm_weight.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(combined.data_ptr()) % kVecBytes == 0 &&
            reinterpret_cast<uintptr_t>(normalized.data_ptr()) % kVecBytes == 0,
        "HcCombineLogitsNorm tensors must be ",
        kVecBytes,
        "-byte aligned");

    const auto params = HcCombineLogitsParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .gate_logits = gate_logits.data_ptr(),
        .gate_stride = gate_stride.unwrap(),
        .norm_weight = norm_weight.data_ptr(),
        .combined = combined.data_ptr(),
        .normalized = normalized.data_ptr(),
        .eps = eps,
    };
    LaunchKernel(dim3(num_tokens, 1, 1), kNumThreads, device.unwrap())
        .enable_pdl(kUsePDL)(hc_combine_logits_kernel<kHcCount, kHiddenSize, true, kUsePDL, DType>, params);
  }
};

}  // namespace sglang
