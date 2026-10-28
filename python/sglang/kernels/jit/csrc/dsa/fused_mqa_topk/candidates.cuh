// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Candidate records of the gated scan: a uint16 value (bits 0-15 of the
// candidate's fp24 score code) and an int32 word (key index in bits 0-19,
// code bits 16-23 in bits 20-27).
#pragma once

#include <cutlass/cutlass.h>

#include "constants.h"
#include <cstdint>

namespace sglang::fused_mqa_topk {

using CandidateValue = uint16_t;
inline constexpr uint32_t kCandidateIndexBits = 20;
inline constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;
static_assert(kMaxKeys <= (1u << kCandidateIndexBits), "key indices must fit the candidate index field");

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload, const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) | ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value, const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 | static_cast<uint32_t>(value);
}

CUTLASS_DEVICE int32_t candidate_decode_index(const int32_t packed_idx) {
  return static_cast<int32_t>(static_cast<uint32_t>(packed_idx) & kCandidateIndexMask);
}

// Stores a candidate whose fp24 code `payload` was already computed.
CUTLASS_DEVICE void
store_candidate_record(CandidateValue* value_dst, int32_t* index_dst, const uint32_t payload, const uint32_t kv_index) {
  __stcg(value_dst, static_cast<CandidateValue>(payload));
  __stcg(index_dst, static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

}  // namespace sglang::fused_mqa_topk
