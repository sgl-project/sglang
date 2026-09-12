#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

/// Level-one keys of the DeepSeek-V4.1 two-level indexer: one key per block of
/// kBlockTokens consecutive scores, the block's maximum score (`amax` in the
/// model code, a plain max). Row `b` has `ceil(seq_len[b] / kBlockTokens)` keys;
/// its newest block is written as +inf so the block top-k can never drop it,
/// which also makes the scores past `seq_len` inside that block irrelevant.
/// Nothing is written past the key count: the consumer takes the same count as
/// the row length. Rows with at most `topk` blocks are skipped entirely (every
/// block is selected anyway; `topk = 0` disables the skip).
struct AmaxConfig {
  using DType = float;                         // TODO: support bf16
  static constexpr uint32_t kBlockTokens = 8;  // scores per key
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kNumItems = 2;  // keys per thread
  static constexpr uint32_t kOccupancy = 4;
  static constexpr uint32_t kKeysPerCTA = kBlockSize * kNumItems;
  // One block is 32 B: a single load on Blackwell, two 16 B loads before it.
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(DType);
  static constexpr uint32_t kVecsPerBlock = kBlockTokens / kVecSize;
  static_assert(kVecsPerBlock * kVecSize == kBlockTokens);
  using vec_t = device::AlignedVector<DType, kVecSize>;
};

struct AmaxParams {
  const AmaxConfig::DType* __restrict__ scores;
  AmaxConfig::DType* __restrict__ amax_scores;
  const int32_t* __restrict__ seq_len;
  int64_t stride_scores;       // in elements
  int64_t stride_amax_scores;  // in elements
  uint32_t topk;               // rows with <= topk blocks are skipped, 0 = never skip
};

/// grid = (rows, ceil(max_keys / kKeysPerCTA)); a CTA owns kKeysPerCTA consecutive
/// keys of one row, a thread kNumItems keys kBlockSize apart (coalesced loads).
template <bool kUsePDL>
__global__ __launch_bounds__(AmaxConfig::kBlockSize, AmaxConfig::kOccupancy)  //
    void amax8_varlen_kernel(const __grid_constant__ AmaxParams params) {
  using namespace device;
  using C = AmaxConfig;
  using T = typename C::DType;
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
      T key = vec[i][0][0];
#pragma unroll
      for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
#pragma unroll
        for (uint32_t j = 0; j < C::kVecSize; ++j) {
          key = fmaxf(key, vec[i][v][j]);  // a NaN score is ignored, torch.amax would propagate it
        }
      }
      out[idx] = idx + 1 == num_keys ? std::numeric_limits<T>::infinity() : key;
    }
  }
}

/// Host entry: `amax_scores[b, i] = max(scores[b, 8 i : 8 i + 8])` for
/// `i < ceil(seq_len[b] / 8)`, the last of them +inf; rows with at most `topk`
/// blocks untouched. `scores` rows must stay 32 B aligned (stride % 8 == 0).
/// The grid covers `amax_scores`' width, so the caller sizes it for the longest
/// row: `seq_len[b] <= 8 * amax_scores.shape[1]` for every row (not checked).
template <bool kPDL>
struct AmaxCopyKernel {
  static void amax8_varlen(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView amax_scores,
      const uint32_t topk) {
    using namespace host;
    using C = AmaxConfig;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"stride_scores"};
    auto K = SymbolicSize{"max_keys"};
    auto O = SymbolicSize{"stride_amax_scores"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({S, 1})
        .with_dtype<typename C::DType>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, K})  // amax_scores
        .with_strides({O, 1})
        .with_dtype<typename C::DType>()
        .with_device(device_)
        .verify(amax_scores);
    RuntimeCheck(S.unwrap() % C::kBlockTokens == 0, "stride_scores must keep every block 32 B aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(scores.data_ptr()) % (C::kBlockTokens * sizeof(typename C::DType)) == 0,
        "scores must be 32 B aligned");
    RuntimeCheck(K.unwrap() > 0, "amax_scores must hold at least one key per row");
    const auto max_keys = K.unwrap();  // ceil(longest row / 8), sized by the caller
    const auto params = AmaxParams{
        .scores = static_cast<const typename C::DType*>(scores.data_ptr()),
        .amax_scores = static_cast<typename C::DType*>(amax_scores.data_ptr()),
        .seq_len = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .stride_scores = S.unwrap(),
        .stride_amax_scores = O.unwrap(),
        .topk = topk,
    };
    const auto grid = dim3(
        static_cast<uint32_t>(B.unwrap()),
        static_cast<uint32_t>(div_ceil(max_keys, static_cast<int64_t>(C::kKeysPerCTA))));
    LaunchKernel(grid, C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(amax8_varlen_kernel<kPDL>, params);
  }
};

}  // namespace sglang
