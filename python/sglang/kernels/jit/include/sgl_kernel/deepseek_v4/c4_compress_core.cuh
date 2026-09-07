/**
 * \brief Shared device-side core of the C4 (ratio-4) compressor.
 *
 * The standalone c4 kernels (c4_v2.cuh) and the fused HIP decode epilogue
 * (fused_compress4_norm_rope_hip.cuh) run the exact same compress: load eight
 * (kv, score) tiles, apply the ape bias, and reduce them into one row with a
 * safe online softmax. Keeping that math in one place means the fused path is
 * bit-identical to the two-kernel path by construction rather than by review.
 *
 * The caller supplies a `Trait` exposing kTileElements / kHeadDim / kElementSize
 * / kScoreOffset / kOverlapOffset; everything here is parameterized on it, so the
 * same code serves both the compressor's 4-elements-per-lane decomposition and
 * the fused epilogue's 2-elements-per-lane (flashmla) / 4 (indexer) layouts.
 */
#pragma once

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace sglang {

/// \brief Load one lane's tile from `src` and widen it to `Dst`. The compressor
/// reads rows stored in two dtypes -- the ring buffer holds BufferFloat, the
/// un-staged wkv_gate gemm tail holds InputFloat -- so the widen-on-load happens
/// here at each call site. The constexpr fast path returns the raw tile untouched
/// when no conversion is needed.
template <typename Trait, typename Dst, typename Src>
SGL_DEVICE device::AlignedVector<Dst, Trait::kTileElements> c4_tile_load(const Src* src) {
  using namespace device;
  using StorageSrc = AlignedVector<Src, Trait::kTileElements>;
  const auto gmem = tile::Memory<StorageSrc>::warp();
  const auto raw = gmem.load(src);
  if constexpr (std::is_same_v<Dst, Src>) {
    return raw;
  } else {
    AlignedVector<Dst, Trait::kTileElements> out;
#pragma unroll
    for (int32_t j = 0; j < Trait::kTileElements; ++j) {
      out[j] = cast<Dst>(raw[j]);
    }
    return out;
  }
}

/// \brief Stage this step's kv/score row into the ring buffer. Runs for every
/// token, including the ones that do not compress.
template <typename Trait, typename BufferFloat, typename InputFloat>
SGL_DEVICE void c4_write_decode(BufferFloat* kv_buf, const InputFloat* kv_src) {
  using namespace device;

  using StorageInput = AlignedVector<InputFloat, Trait::kTileElements>;
  const auto gmem_input = tile::Memory<StorageInput>::warp();

  StorageInput data[4];
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    data[i] = gmem_input.load(kv_src + Trait::kHeadDim * i);
  }

  if constexpr (std::is_same_v<BufferFloat, InputFloat>) {
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
      gmem_input.store(kv_buf + Trait::kHeadDim * i, data[i]);
    }
  } else {
    using StorageBuffer = AlignedVector<BufferFloat, Trait::kTileElements>;
    const auto gmem_buffer = tile::Memory<StorageBuffer>::warp();

    StorageBuffer data_cast[4];
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
#pragma unroll
      for (int32_t j = 0; j < Trait::kTileElements; ++j) {
        data_cast[i][j] = cast<BufferFloat>(data[i][j]);
      }
      gmem_buffer.store(kv_buf + Trait::kHeadDim * i, data_cast[i]);
    }
  }
}

/// \brief Compress one token's eight (kv, score) tiles into a single row and
/// return it in registers as fp32. `c4_forward` casts the result to OutFloat and
/// stores it; the fused decode kernel hands it straight to its norm + RoPE +
/// fp8-store epilogue with no global round trip. BufferFloat is the ring-buffer
/// storage dtype and InputFloat is the compute dtype (== score_bias/ape dtype ==
/// un-staged kv_src dtype); the load site widens BufferFloat -> InputFloat when
/// they differ.
template <typename Trait, typename BufferFloat, typename InputFloat>
SGL_DEVICE device::AlignedVector<float, Trait::kTileElements> c4_compress_core(
    const BufferFloat* kv_buf_0,  // overlap [4n - 4, 4n - 1]
    const BufferFloat* kv_buf_1,  // normal [4n + 0, 4n + 3]
    const InputFloat* kv_src,     // ragged pointer at position = 4n + 3
    const InputFloat* score_bias,
    const bool should_overlap,
    const int32_t buffer_len) {
  using namespace device;
  constexpr int32_t kTile = Trait::kTileElements;

  AlignedVector<InputFloat, kTile> kv[8];
  AlignedVector<InputFloat, kTile> score[8];
  AlignedVector<InputFloat, kTile> bias[8];

#pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    bias[i] = c4_tile_load<Trait, InputFloat>(score_bias + i * Trait::kHeadDim);
  }

  const auto kv_start_0 = kv_src - 7 * Trait::kElementSize;  // point to start
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    if (should_overlap && i < buffer_len) {
      const auto base = kv_buf_0 + i * Trait::kElementSize;
      kv[i] = c4_tile_load<Trait, InputFloat>(base);
      score[i] = c4_tile_load<Trait, InputFloat>(base + Trait::kScoreOffset);
    } else if (should_overlap) {
      const auto base = kv_start_0 + i * Trait::kElementSize;
      kv[i] = c4_tile_load<Trait, InputFloat>(base);
      score[i] = c4_tile_load<Trait, InputFloat>(base + Trait::kScoreOffset);
    } else {
      [[unlikely]];
      constexpr float kFloatNegInf = -FLT_MAX;
      kv[i].fill(cast<InputFloat>(0.0f));
      score[i].fill(cast<InputFloat>(kFloatNegInf));
    }
  }

  const auto kv_start = kv_src - 3 * Trait::kElementSize;  // point to start
#pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    if (i + 4 < buffer_len) {
      const auto base = kv_buf_1 + i * Trait::kElementSize + Trait::kOverlapOffset;
      kv[i + 4] = c4_tile_load<Trait, InputFloat>(base);
      score[i + 4] = c4_tile_load<Trait, InputFloat>(base + Trait::kScoreOffset);
    } else {
      const auto base = kv_start + i * Trait::kElementSize + Trait::kOverlapOffset;
      kv[i + 4] = c4_tile_load<Trait, InputFloat>(base);
      score[i + 4] = c4_tile_load<Trait, InputFloat>(base + Trait::kScoreOffset);
    }
  }

  /// safe online softmax + weighted sum, accumulated in fp32
  AlignedVector<float, kTile> result;
  float score_fp32[kTile][8];  // consume 32 fp registers

#pragma unroll
  for (int32_t i = 0; i < kTile; ++i) {
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      score_fp32[i][j] = cast<float>(score[j][i]) + cast<float>(bias[j][i]);
    }
  }

#pragma unroll
  for (int32_t i = 0; i < kTile; ++i) {
    const auto& s = score_fp32[i];
    float max_value = s[0];
#pragma unroll
    for (int32_t j = 1; j < 8; ++j) {
      max_value = fmaxf(max_value, s[j]);
    }

    float sum_exp_value = 0.0f;
    float sum_product = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < 8; ++j) {
      const auto exp_score = __expf(s[j] - max_value);
      sum_product += cast<float>(kv[j][i]) * exp_score;
      sum_exp_value += exp_score;
    }

    result[i] = sum_product / sum_exp_value;
  }
  return result;
}

}  // namespace sglang
