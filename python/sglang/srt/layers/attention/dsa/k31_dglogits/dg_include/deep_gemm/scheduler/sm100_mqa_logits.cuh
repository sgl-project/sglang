#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

// SM100 contiguous-KV scheduler; translates row bounds into core task geometry
// Two scheduler modes:
// - Grid-stride: SM `s` owns Q blocks `s`, `s + kNumSMs`, never split K
// - With metadata: consumes metadata prebuilt by sm100_mqa_logits_metadata,
//   may split K to equalize workload among SMs. Metadata workspace layout, count as int32:
//  [0, 2 * kNumSMs)                    per-SM start points, uint2 = (q_block_idx, split_offset)
//  [2 * kNumSMs, 3 * kNumSMs)          per-SM split counts
//  [3 * kNumSMs, kMetaSpanWordOffset)  pad kMetaSpanWordOffset to 8-byte aligned
//  [kMetaSpanWordOffset, ...)          per-block MQALogitsKVSpan

namespace deep_gemm::sched {

// KV range one Q block must visit, cover all [ks, ke)
// precomputed by metadata kernel, consumed by (Q TMA, KV TMA, UMMA, math)
struct MQALogitsKVSpan {
    uint32_t kv_token_base; // min k_start over rows, rounded down to 4 tokens
    uint32_t num_kv_splits; // ceil((max k_end - kv_token_base) / SPLIT_KV)
};

// NOTE: Out-of-range rows are clamped to the last valid row
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSpanUnrolls = 8, bool kRematerializeSpan = false>
CUTLASS_DEVICE MQALogitsKVSpan get_mqa_logits_kv_span(const uint32_t& q_block_idx,
                                                      const uint32_t& num_q_tokens, const uint32_t& num_kv_tokens,
                                                      const uint32_t* cu_seq_len_k_start,
                                                      const uint32_t* cu_seq_len_k_end) {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
    #pragma unroll kNumSpanUnrolls
    for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
        auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
        // Keep each row index local to its load pair in the fully unrolled scan.
        if constexpr (kRematerializeSpan)
            asm volatile("mov.b32 %0, %1;" : "=r"(row_idx) : "r"(row_idx));
        const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
        const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
        start = cute::min(start, k_start);
        end = cute::max(end, k_end);
    }
    const uint32_t kv_token_base = start / 4 * 4;
    return {kv_token_base, math::ceil_div(end - kv_token_base, SPLIT_KV)};
}

template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumSMs,
          bool kUseSchedule = false, uint32_t kNumSpanUnrolls = 8, bool kRematerializeSpan = false>
struct SM100MQALogitsScheduler {
    static constexpr bool kIsPaged = false;
    static constexpr bool kHasPartialBlock = false;
    static constexpr uint32_t kPageKV = 0;
    static constexpr uint32_t kMetaSpanWordOffset = (3 * kNumSMs + 1) / 2 * 2;

    uint32_t current_q_block_idx;
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    const uint32_t* cu_seq_len_k_start;
    const uint32_t* cu_seq_len_k_end;
    uint32_t current_split_offset = 0;
    uint32_t remaining_splits = 0;
    const MQALogitsKVSpan* kv_spans = nullptr;

    CUTLASS_DEVICE SM100MQALogitsScheduler(const uint32_t& sm_idx,
                                           const uint32_t& num_q_tokens,
                                           const uint32_t& num_kv_tokens,
                                           const uint32_t* cu_seq_len_k_start,
                                           const uint32_t* cu_seq_len_k_end,
                                           const uint32_t* schedule_meta = nullptr):
            current_q_block_idx(sm_idx),
            num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
            num_q_tokens(num_q_tokens),
            num_kv_tokens(num_kv_tokens),
            cu_seq_len_k_start(cu_seq_len_k_start),
            cu_seq_len_k_end(cu_seq_len_k_end) {
        DG_STATIC_ASSERT(kNumSMs > 0, "Invalid SM count");
        if constexpr (kUseSchedule) {
            const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
            current_q_block_idx = start.x;
            current_split_offset = start.y;
            remaining_splits = schedule_meta[2 * kNumSMs + sm_idx];
            kv_spans = reinterpret_cast<const MQALogitsKVSpan*>(schedule_meta + kMetaSpanWordOffset);
        }
    }

    CUTLASS_DEVICE auto make_cleaner(const uint32_t& sm_idx) const {
        return SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end);
    }

    template <bool kLoadSeqBounds = false>
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits,
                                     uint32_t* seq_k_start = nullptr, uint32_t* seq_k_end = nullptr) {
        if constexpr (kUseSchedule) {
            while (remaining_splits > 0 and current_q_block_idx < num_q_blocks) {
                const auto span = kv_spans[current_q_block_idx];
                const auto split_offset = current_split_offset;
                current_split_offset = 0;
                if (split_offset < span.num_kv_splits) {
                    if constexpr (kLoadSeqBounds) {
                        #pragma unroll
                        for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                            const auto row_idx = cute::min(current_q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                            seq_k_start[token_idx] = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                            seq_k_end[token_idx] = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                        }
                    }
                    q_block_idx = current_q_block_idx;
                    kv_token_base = span.kv_token_base + split_offset * SPLIT_KV;
                    num_kv_splits = cute::min(span.num_kv_splits - split_offset, remaining_splits);
                    remaining_splits -= num_kv_splits;
                    ++ current_q_block_idx;
                    return true;
                }
                ++ current_q_block_idx;
            }
            return false;
        } else {
            if (current_q_block_idx >= num_q_blocks)
                return false;
            q_block_idx = current_q_block_idx;
            current_q_block_idx += kNumSMs;
            if constexpr (kLoadSeqBounds) {
                uint32_t start = cute::numeric_limits<uint32_t>::max();
                uint32_t end = cute::numeric_limits<uint32_t>::min();
                #pragma unroll
                for (uint32_t token_idx = 0; token_idx < BLOCK_Q; ++ token_idx) {
                    const auto row_idx = cute::min(q_block_idx * BLOCK_Q + token_idx, num_q_tokens - 1);
                    const auto k_start = cute::min(cu_seq_len_k_start[row_idx], num_kv_tokens);
                    const auto k_end = cute::min(cu_seq_len_k_end[row_idx], num_kv_tokens);
                    seq_k_start[token_idx] = k_start;
                    seq_k_end[token_idx] = k_end;
                    start = cute::min(start, k_start);
                    end = cute::max(end, k_end);
                }
                kv_token_base = start / 4 * 4;
                num_kv_splits = math::ceil_div(end - kv_token_base, SPLIT_KV);
            } else {
                const auto span = get_mqa_logits_kv_span<BLOCK_Q, SPLIT_KV, kNumSpanUnrolls, kRematerializeSpan>(q_block_idx, num_q_tokens, num_kv_tokens,
                                                                            cu_seq_len_k_start, cu_seq_len_k_end);
                kv_token_base = span.kv_token_base;
                num_kv_splits = span.num_kv_splits;
            }
            return true;
        }
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t& q_block_idx) const {
        return q_block_idx * BLOCK_Q;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_token_base, const uint32_t& kv_split_idx) {
        return kv_token_base + kv_split_idx * SPLIT_KV;
    }

    CUTLASS_DEVICE static uint32_t get_logits_row(const uint32_t& q_block_idx, const uint32_t& token_idx) {
        return q_block_idx * BLOCK_Q + token_idx;
    }

    CUTLASS_DEVICE static uint32_t get_logits_col(const uint32_t& kv_token_base,
                                                  const uint32_t& kv_split_idx,
                                                  const uint32_t& math_thread_idx) {
        return kv_token_base + kv_split_idx * SPLIT_KV + math_thread_idx;
    }
};

} // namespace deep_gemm::sched
