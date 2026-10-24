#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

/// Level-one keys of the two-level indexer: the max of each kBlockTokens-score
/// block, the row's newest block forced to +inf. Scores and keys are each fp32
/// or bf16; a bf16 score's max is exact in either key type. Contract: BlockAmaxKernel.
template <typename InT, typename OutT>
struct BlockAmaxConfig {
  using DType = InT;
  using KeyT = OutT;
  static constexpr uint32_t kBlockTokens = 8;  // scores per key
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kOccupancy = 4;
  // One fp32 block is 32 B: a single load on Blackwell, two 16 B loads before it.
  // A bf16 block is one 16 B load anywhere.
  static constexpr uint32_t kBlockBytes = kBlockTokens * sizeof(DType);
  static constexpr uint32_t kVecBytes = std::min<uint32_t>(kBlockBytes, device::kMaxVecBytes);
  static constexpr uint32_t kVecSize = kVecBytes / sizeof(DType);
  static constexpr uint32_t kVecsPerBlock = kBlockTokens / kVecSize;
  static constexpr uint32_t kNumItems = 64 / kBlockBytes;  // 64B scores (16 registers) per thread
  static constexpr uint32_t kKeysPerCTA = kBlockSize * kNumItems;
  static_assert(kVecsPerBlock * kVecSize == kBlockTokens);
  using vec_t = device::AlignedVector<DType, kVecSize>;
};

template <typename InT, typename OutT>
struct BlockAmaxParams {
  const InT* __restrict__ scores;
  OutT* __restrict__ amax_scores;
  const int32_t* __restrict__ seq_len;
  int64_t stride_scores;       // in elements
  int64_t stride_amax_scores;  // in elements
  uint32_t topk;               // rows with <= topk blocks are skipped, 0 = never skip
};

/// grid = (rows, ceil(max_keys / kKeysPerCTA)); a CTA owns kKeysPerCTA consecutive
/// keys of one row, a thread kNumItems keys kBlockSize apart (coalesced loads).
template <bool kUsePDL, typename InT, typename OutT>
__global__ __launch_bounds__(BlockAmaxConfig<InT, OutT>::kBlockSize, BlockAmaxConfig<InT, OutT>::kOccupancy)  //
    void amax8_varlen_kernel(const __grid_constant__ BlockAmaxParams<InT, OutT> params) {
  using namespace device;
  using C = BlockAmaxConfig<InT, OutT>;
  using vec_t = typename C::vec_t;
  const auto bx = blockIdx.x;
  const auto by = blockIdx.y;
  const auto tx = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();  // seq_len and scores are the previous kernels' outputs
  const auto seq_len = static_cast<uint32_t>(params.seq_len[bx]);
  const auto num_keys = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  const auto first_key = by * C::kKeysPerCTA;
  if (num_keys <= params.topk || first_key >= num_keys) {
    return PDLTriggerSecondary<kUsePDL>();
  }
  const auto* __restrict__ in = params.scores + bx * params.stride_scores;
  auto* __restrict__ out = params.amax_scores + bx * params.stride_amax_scores;

  vec_t vec[C::kNumItems][C::kVecsPerBlock];
#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto idx = first_key + tx + i * C::kBlockSize;
    if (idx < num_keys) {
#pragma unroll
      for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
        vec[i][v].load(in, idx * C::kVecsPerBlock + v);
      }
    }
  }
  // The dependent grid may start its prologue now; its griddepcontrol.wait still
  // covers every store below (it waits for this grid to complete).
  PDLTriggerSecondary<kUsePDL>();

#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto idx = first_key + tx + i * C::kBlockSize;
    if (idx < num_keys) {
      // fmaxf and __hmax2 ignore a NaN score, torch.amax would propagate it
      if constexpr (std::is_same_v<InT, float>) {
        float key = vec[i][0][0];
#pragma unroll
        for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
#pragma unroll
          for (uint32_t j = 0; j < C::kVecSize; ++j) {
            key = fmaxf(key, vec[i][v][j]);
          }
        }
        const auto fp32_inf = std::numeric_limits<float>::infinity();
        out[idx] = cast<OutT>(idx + 1 == num_keys ? fp32_inf : key);
      } else {
        const auto vec_ptr = reinterpret_cast<bf16x2_t*>(vec[i][0].data());
        static_assert(C::kVecSize % 2 == 0);
        bf16x2_t key2 = vec_ptr[0];
#pragma unroll
        for (uint32_t v = 1; v < C::kVecsPerBlock * C::kVecSize / 2; ++v) {
          key2 = __hmax2(key2, vec_ptr[v]);
        }
        const auto key = __hmax(key2.x, key2.y);
        const auto bf16_inf = cast<bf16_t>(std::numeric_limits<float>::infinity());
        out[idx] = cast<OutT>(idx + 1 == num_keys ? bf16_inf : key);
      }
    }
  }
}

/// Host entry: `amax_scores[b, i] = max(scores[b, 8 i : 8 i + 8])` for
/// `i < ceil(seq_len[b] / 8)`, the last of them +inf; rows with at most `topk`
/// blocks untouched. `scores` (`InT`) rows must keep every block of 8 aligned
/// (stride % 8 == 0); `amax_scores` is `OutT`. Each type is fp32 or bf16.
/// The grid covers `amax_scores`' width, so the caller sizes it for the longest
/// row: `seq_len[b] <= 8 * amax_scores.shape[1]` for every row (not checked).
template <bool kPDL, typename InT, typename OutT>
struct BlockAmaxKernel {
  static void amax8_varlen(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView amax_scores,
      const uint32_t topk) {
    using namespace host;
    using C = BlockAmaxConfig<InT, OutT>;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto K = SymbolicSize{"max_keys"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({-1, 1})
        .with_dtype<typename C::DType>()
        .with_device(device_)
        .ensure_alignment(C::kBlockBytes)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, K})  // amax_scores
        .with_strides({-1, 1})
        .with_dtype<typename C::KeyT>()
        .with_device(device_)
        .verify(amax_scores);
    const auto max_keys = K.unwrap();  // ceil(longest row / 8), sized by the caller
    const auto params = BlockAmaxParams<InT, OutT>{
        .scores = static_cast<const InT*>(scores.data_ptr()),
        .amax_scores = static_cast<OutT*>(amax_scores.data_ptr()),
        .seq_len = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .stride_scores = scores.stride(0),
        .stride_amax_scores = amax_scores.stride(0),
        .topk = topk,
    };
    const auto grid = dim3(
        static_cast<uint32_t>(B.unwrap()),
        static_cast<uint32_t>(div_ceil(max_keys, static_cast<int64_t>(C::kKeysPerCTA))));
    LaunchKernel(grid, C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(amax8_varlen_kernel<kPDL, InT, OutT>, params);
  }
};

}  // namespace sglang
