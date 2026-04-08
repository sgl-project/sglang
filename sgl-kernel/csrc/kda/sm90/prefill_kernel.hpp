// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace kda::sm90 {

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState>
void launch_kda_fwd_prefill_kernel(
    cudaStream_t stream,
    TO* output,
    TState* output_state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    TState const* input_state,
    float const* alpha,
    float const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    bool safe_gate,
    int32_t sm_count = 0);

}  // namespace kda::sm90
