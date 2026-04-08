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

// Dispatch function only — does NOT include the .cuh to avoid
// implicit instantiation of all kernel variants in one TU.
// Each SafeGate variant is explicitly instantiated in its own .cu file.

#include <cutlass/arch/arch.h>

#include <cute/numeric/numeric_types.hpp>

namespace kda::sm90 {

using namespace cute;

// Forward declaration of the per-variant launcher (defined in .cuh, instantiated in separate TUs)
template <
    bool NeedsBeta,
    bool NeedsAlpha,
    bool InitStateFromInput,
    bool SafeGate,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState>
void launch_kda_fwd_prefill_kernel_gbai(
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
    int32_t sm_count);

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
    int32_t sm_count = 0) {
  bool needs_beta = beta != nullptr;
  bool needs_alpha = alpha != nullptr;
  bool init_state = input_state != nullptr;

#define LAUNCH(needs_beta, needs_alpha, init_state, safe_gate)                                 \
  launch_kda_fwd_prefill_kernel_gbai<needs_beta, needs_alpha, init_state, safe_gate, ArchTag>( \
      stream,                                                                                  \
      output,                                                                                  \
      output_state,                                                                            \
      q,                                                                                       \
      k,                                                                                       \
      v,                                                                                       \
      input_state,                                                                             \
      alpha,                                                                                   \
      beta,                                                                                    \
      cu_seqlens,                                                                              \
      workspace_buffer,                                                                        \
      num_seqs,                                                                                \
      num_heads,                                                                               \
      head_size,                                                                               \
      total_seqlen,                                                                            \
      scale,                                                                                   \
      sm_count);
  if (init_state) {
    if (needs_beta && needs_alpha && safe_gate) {
      LAUNCH(true, true, true, true);
    } else {
      throw std::runtime_error("unreachable");
    }
  } else {
    if (needs_beta && needs_alpha && safe_gate) {
      LAUNCH(true, true, false, true);
    } else {
      throw std::runtime_error("unreachable");
    }
  }

#undef LAUNCH
}

using bf16 = cute::bfloat16_t;

template void launch_kda_fwd_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t stream,
    bf16* output,
    float* state,
    bf16 const* q,
    bf16 const* k,
    bf16 const* v,
    float const* input_state,
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
    int32_t sm_count);

}  // namespace kda::sm90
