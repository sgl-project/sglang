/*
 * Copyright (c) 2026 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

#include "params.h"

namespace sm90 {

// This is the sparse SM90 FlashMLA kernel with an NVFP4 K/V producer.  The
// query/compute/softmax/epilogue path is inherited from the pinned FlashMLA
// sparse FP8 implementation.  Only the 416-byte cache-row loader differs.
void run_flash_splitkv_mla_nvfp4_sparse_kernel(
    DecodingParams& params, const float* kv_global_scale_ptr, cudaStream_t stream);

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)

// Benchmark-only output layout.  Each CTA owns one [record, metric] slot, so
// the kernel never needs atomics.  Records 0 and 1 are the two consumer
// warpgroups; records 2..5 are the four producer warps.  Consumers execute in
// parallel, therefore their service time is intentionally kept separate and
// must not be added together by the caller.  Cycle fields are cumulative; the
// kernel excludes each CTA's first tile and kTimedTileCount is the divisor for
// per-tile summaries.  kConsumerSyncWaitCycles covers the named-barrier wait
// between the two consumer warpgroups, not either of the two K-buffer waits.
inline constexpr int kStageTimingRecordsPerCta = 6;
inline constexpr int kStageTimingMetricsPerRecord = 8;

enum StageTimingRecord : int {
  kConsumerLocalRecord = 0,
  kConsumerRemoteRecord = 1,
  kProducerWarp0Record = 2,
};

enum StageTimingMetric : int {
  kTimedTileCount = 0,
  kLoadCycles = 1,
  kDequantCycles = 2,
  kHandoffCycles = 3,
  kConsumerCycles = 4,
  kConsumerReadyWaitCycles = 5,
  kProducerAvailableWaitCycles = 6,
  kConsumerSyncWaitCycles = 7,
};

// The profile launcher is only compiled when the explicit CMake profile
// option is enabled.  It launches a separately-instantiated kernel and leaves
// the production launcher and public operator unchanged.
int get_flash_splitkv_mla_nvfp4_stage_timing_num_ctas(const DecodingParams& params);

void run_flash_splitkv_mla_nvfp4_sparse_profile_kernel(
    DecodingParams& params, const float* kv_global_scale_ptr, uint64_t* stage_timing_ptr, cudaStream_t stream);

#endif

}  // namespace sm90
