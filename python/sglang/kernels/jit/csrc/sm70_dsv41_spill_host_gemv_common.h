// SPDX-License-Identifier: Apache-2.0
// Shared mailbox for the spilled-expert host GEMV.
#pragma once

#include <cstdint>

namespace sglang::sm70_dsv41 {

inline constexpr int32_t kHidden = 5120;
inline constexpr int32_t kIntermediate = 2304;
inline constexpr int32_t kGateUp = 2 * kIntermediate;
inline constexpr int32_t kTopK = 6;
inline constexpr int32_t kGroupSize = 32;
inline constexpr int32_t kMaxTok = 2;
inline constexpr int32_t kMaxHits = kMaxTok * kTopK;
inline constexpr int32_t kPackedMacroN = 256;
inline constexpr int32_t kQuantTileK = 16;
inline constexpr int32_t kQuantTileN = 64;
inline constexpr int32_t kGroupTiles = kPackedMacroN / kQuantTileN;
inline constexpr int32_t kW13KTiles = kHidden / kQuantTileK;
inline constexpr int32_t kW2KTiles = kIntermediate / kQuantTileK;
inline constexpr int32_t kW13Groups = kHidden / kGroupSize;
inline constexpr int32_t kW2Groups = kIntermediate / kGroupSize;
inline constexpr int32_t kMaxHostThreads = 16;

struct alignas(64) SpillHostMailbox {
  volatile uint32_t seq;
  volatile uint32_t done;
  volatile int32_t error;
  int32_t n_tok;
  int32_t n_hits;
  int32_t n_host;
  int32_t host_rows[kMaxHits];
  int32_t tok_of[kMaxHits];
  float weights[kMaxHits];
  int64_t w13_ptr;
  int64_t s13_ptr;
  int64_t w2_ptr;
  int64_t s2_ptr;
  uint16_t x[kMaxTok][kHidden];
  uint16_t y[kMaxTok][kHidden];
};

extern SpillHostMailbox* dsv41_spill_mbox_host;
extern SpillHostMailbox* dsv41_spill_mbox_dev;

void dsv41_host_workers_start(int nthreads);
void dsv41_host_workers_stop();
void dsv41_mxfp4_moe_expert_serial(const uint16_t* x,
                                   const uint32_t* w13,
                                   const uint8_t* s13,
                                   const uint32_t* w2,
                                   const uint8_t* s2,
                                   float weight,
                                   uint16_t* y);

}  // namespace sglang::sm70_dsv41
