// SGLang Apple Silicon Metal kernel: block-table paged decode attention.

#include <metal_stdlib>
using namespace metal;

constant uint  HEAD_DIM     [[function_constant(0)]];
constant uint  NUM_QO_HEADS [[function_constant(1)]];
constant uint  NUM_KV_HEADS [[function_constant(2)]];
constant float SM_SCALE     [[function_constant(3)]];
constant uint  BLOCK_SIZE   [[function_constant(4)]];

constant uint PAGED_ATTN_NUM_THREADS = 256;
constant uint PAGED_ATTN_SIMD_SIZE = 32;
constant uint PAGED_ATTN_NUM_WARPS = PAGED_ATTN_NUM_THREADS / PAGED_ATTN_SIMD_SIZE;
constant uint PAGED_ATTN_MAX_HEAD_DIM = 256;

inline float simd_max_32(float value) {
    value = max(value, simd_shuffle_xor(value, ushort(16)));
    value = max(value, simd_shuffle_xor(value, ushort(8)));
    value = max(value, simd_shuffle_xor(value, ushort(4)));
    value = max(value, simd_shuffle_xor(value, ushort(2)));
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    return value;
}

inline float simd_sum_32(float value) {
    value += simd_shuffle_xor(value, ushort(16));
    value += simd_shuffle_xor(value, ushort(8));
    value += simd_shuffle_xor(value, ushort(4));
    value += simd_shuffle_xor(value, ushort(2));
    value += simd_shuffle_xor(value, ushort(1));
    return value;
}

inline float warp_state_scale(float warp_max, float row_max) {
    return warp_max == -INFINITY ? 0.0f : metal::exp(warp_max - row_max);
}

template <typename T>
inline void block_paged_attention_decode_impl(
    const device T*       q,
    const device T*       k_blocks,
    const device T*       v_blocks,
    const device int32_t* block_tables,
    const device int32_t* seq_lens,
    const constant uint&  max_blocks,
    device T*             out,
    uint3 tg_pos,
    uint simd_tid,
    uint simd_lid,
    threadgroup float*    warp_maxes,
    threadgroup float*    warp_sums,
    threadgroup float*    warp_accs
) {
    const uint batch_id = tg_pos.x;
    const uint q_head = tg_pos.y;
    const uint warp = simd_tid;
    const uint lane = simd_lid;

    const uint kv_group = NUM_QO_HEADS / NUM_KV_HEADS;
    const uint kv_head = q_head / kv_group;
    const int32_t seq_len = seq_lens[batch_id];
    const uint q_base = (batch_id * NUM_QO_HEADS + q_head) * HEAD_DIM;

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_acc[PAGED_ATTN_MAX_HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {
        local_acc[d] = 0.0f;
    }

    for (int32_t token_idx = int32_t(warp * PAGED_ATTN_SIMD_SIZE + lane);
         token_idx < seq_len;
         token_idx += int32_t(PAGED_ATTN_NUM_THREADS)) {
        const uint logical_block = uint(token_idx) / BLOCK_SIZE;
        const uint block_offset = uint(token_idx) - logical_block * BLOCK_SIZE;
        const int32_t physical_block = block_tables[batch_id * max_blocks + logical_block];
        if (physical_block < 0) {
            continue;
        }
        const uint kv_base =
            (((uint)physical_block * BLOCK_SIZE + block_offset) * NUM_KV_HEADS + kv_head) * HEAD_DIM;

        float logit = 0.0f;
        uint d = 0;
        for (; d + 3 < HEAD_DIM; d += 4) {
            const float4 qv = float4(
                float(q[q_base + d + 0]),
                float(q[q_base + d + 1]),
                float(q[q_base + d + 2]),
                float(q[q_base + d + 3]));
            const float4 kv = float4(
                float(k_blocks[kv_base + d + 0]),
                float(k_blocks[kv_base + d + 1]),
                float(k_blocks[kv_base + d + 2]),
                float(k_blocks[kv_base + d + 3]));
            logit += dot(qv, kv);
        }
        for (; d < HEAD_DIM; ++d) {
            logit += float(q[q_base + d]) * float(k_blocks[kv_base + d]);
        }
        logit *= SM_SCALE;

        const float new_max = max(local_max, logit);
        const float old_scale = metal::exp(local_max - new_max);
        const float weight = metal::exp(logit - new_max);
        local_sum = local_sum * old_scale + weight;
        d = 0;
        for (; d + 3 < HEAD_DIM; d += 4) {
            local_acc[d + 0] = local_acc[d + 0] * old_scale + weight * float(v_blocks[kv_base + d + 0]);
            local_acc[d + 1] = local_acc[d + 1] * old_scale + weight * float(v_blocks[kv_base + d + 1]);
            local_acc[d + 2] = local_acc[d + 2] * old_scale + weight * float(v_blocks[kv_base + d + 2]);
            local_acc[d + 3] = local_acc[d + 3] * old_scale + weight * float(v_blocks[kv_base + d + 3]);
        }
        for (; d < HEAD_DIM; ++d) {
            local_acc[d] = local_acc[d] * old_scale + weight * float(v_blocks[kv_base + d]);
        }
        local_max = new_max;
    }

    const float warp_max = simd_max_32(local_max);
    const bool warp_has_tokens = warp_max != -INFINITY;
    const float lane_scale = warp_has_tokens ? metal::exp(local_max - warp_max) : 0.0f;
    const float warp_sum = warp_has_tokens ? simd_sum_32(local_sum * lane_scale) : 0.0f;

    if (lane == 0) {
        warp_maxes[warp] = warp_max;
        warp_sums[warp] = warp_sum;
    }
    for (uint d = 0; d < HEAD_DIM; ++d) {
        const float warp_acc = simd_sum_32(local_acc[d] * lane_scale);
        if (lane == 0) {
            warp_accs[warp * PAGED_ATTN_MAX_HEAD_DIM + d] =
                warp_has_tokens ? warp_acc : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp != 0) {
        return;
    }

    float row_max = -INFINITY;
    for (uint w = 0; w < PAGED_ATTN_NUM_WARPS; ++w) {
        row_max = max(row_max, warp_maxes[w]);
    }
    if (row_max == -INFINITY) {
        row_max = 0.0f;
    }

    float row_sum = 0.0f;
    for (uint w = 0; w < PAGED_ATTN_NUM_WARPS; ++w) {
        row_sum += warp_sums[w] * warp_state_scale(warp_maxes[w], row_max);
    }

    const uint out_base = (batch_id * NUM_QO_HEADS + q_head) * HEAD_DIM;
    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
    for (uint d = lane; d < HEAD_DIM; d += PAGED_ATTN_SIMD_SIZE) {
        float acc = 0.0f;
        for (uint w = 0; w < PAGED_ATTN_NUM_WARPS; ++w) {
            acc += warp_accs[w * PAGED_ATTN_MAX_HEAD_DIM + d] *
                   warp_state_scale(warp_maxes[w], row_max);
        }
        out[out_base + d] = static_cast<T>(acc * inv_sum);
    }
}

#define INSTANTIATE_BLOCK(NAME, T)                                                      \
    [[host_name("block_paged_attention_decode_" #NAME)]] [[kernel]] void               \
    block_paged_attention_decode_##NAME(                                                \
        const device T*       q            [[buffer(0)]],                               \
        const device T*       k_blocks     [[buffer(1)]],                               \
        const device T*       v_blocks     [[buffer(2)]],                               \
        const device int32_t* block_tables [[buffer(3)]],                               \
        const device int32_t* seq_lens     [[buffer(4)]],                               \
        device T*             out          [[buffer(5)]],                               \
        const constant uint&  max_blocks   [[buffer(6)]],                               \
        uint3 tg_pos [[threadgroup_position_in_grid]],                                  \
        uint simd_tid [[simdgroup_index_in_threadgroup]],                               \
        uint simd_lid [[thread_index_in_simdgroup]]) {                                  \
        threadgroup float warp_maxes[PAGED_ATTN_NUM_WARPS];                             \
        threadgroup float warp_sums[PAGED_ATTN_NUM_WARPS];                              \
        threadgroup float warp_accs[PAGED_ATTN_NUM_WARPS * PAGED_ATTN_MAX_HEAD_DIM];    \
        block_paged_attention_decode_impl<T>(q, k_blocks, v_blocks, block_tables,       \
                                             seq_lens, max_blocks, out, tg_pos,         \
                                             simd_tid, simd_lid, warp_maxes, warp_sums, \
                                             warp_accs);                                \
    }

INSTANTIATE_BLOCK(f16,  half)
INSTANTIATE_BLOCK(bf16, bfloat)
INSTANTIATE_BLOCK(f32,  float)
