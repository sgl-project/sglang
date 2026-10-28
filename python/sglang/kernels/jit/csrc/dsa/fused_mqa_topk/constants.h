// Shapes and workspace layout of the fused MQA-logits + top-k kernel, shared by
// the kernel and its host entry. kernels/ops/attention/dsa/fused_mqa_topk.py
// mirrors the workspace sizes; the host entry checks them.
#pragma once

#include <cstdint>

namespace sglang::fused_mqa_topk {

// Indexer geometry (GLM-5.2): 32 index heads of dimension 128, FP8 Q and K.
inline constexpr uint32_t kNumHeads = 32, kHeadDim = 128;
// The only top-k the kernel is compiled for.
inline constexpr uint32_t kTopK = 2048;
// Query rows per block (the unit a CTA claims), threads per CTA, keys per tile.
inline constexpr uint32_t kBlockRows = 8, kNumThreads = 384, kTileKeys = 256;
// Key range of one call. The gates are tuned above kMinKeys - 1 keys only; key
// indices take 20 bits in a candidate record (candidates.cuh).
inline constexpr uint32_t kMinKeys = 20481, kMaxKeys = 1u << 20;
// Per-row workspace: the gated candidate slab (a row with more candidates is
// rescanned exactly) and the exact rescan's retained set.
inline constexpr uint32_t kCandidatesPerRow = 16384, kRetainedPerRow = 4096;
// Candidate / retained slots (of kBlockRows rows each) per persistent CTA: one
// block in selection, one claimed ahead.
inline constexpr uint32_t kSlotsPerCta = 2;

}  // namespace sglang::fused_mqa_topk
