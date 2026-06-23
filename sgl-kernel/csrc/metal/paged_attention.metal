// SGLang Apple Silicon Metal kernel: page-size-1 decode attention over a flat KV pool.

#include <metal_stdlib>
using namespace metal;

constant uint  HEAD_DIM     [[function_constant(0)]];
constant uint  NUM_QO_HEADS [[function_constant(1)]];
constant uint  NUM_KV_HEADS [[function_constant(2)]];
constant float SM_SCALE     [[function_constant(3)]];

constant uint PAGED_ATTN_TG_SIZE = 32;
constant uint PAGED_ATTN_MAX_HEAD_DIM = 128;

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

template <typename T>
inline void paged_attention_decode_impl(
    const device T*       q,
    const device T*       k_pool,
    const device T*       v_pool,
    const device int32_t* kv_indptr,
    const device int32_t* kv_indices,
    device T*             out,
    uint3 tid,
    uint3 tg_pos
) {
    const uint batch_id = tg_pos.x;
    const uint q_head = tg_pos.y;
    const uint lane = tid.z;

    const uint kv_group = NUM_QO_HEADS / NUM_KV_HEADS;
    const uint kv_head = q_head / kv_group;
    const int32_t start = kv_indptr[batch_id];
    const int32_t end = kv_indptr[batch_id + 1];

    const uint q_base = (batch_id * NUM_QO_HEADS + q_head) * HEAD_DIM;

    float local_max = -INFINITY;
    for (int32_t i = start + int32_t(lane); i < end; i += int32_t(PAGED_ATTN_TG_SIZE)) {
        const int32_t slot = kv_indices[i];
        const uint k_base = ((uint)slot * NUM_KV_HEADS + kv_head) * HEAD_DIM;
        float logit = 0.0f;
        uint d = 0;
        for (; d + 3 < HEAD_DIM; d += 4) {
            const float4 qv = float4(
                float(q[q_base + d + 0]),
                float(q[q_base + d + 1]),
                float(q[q_base + d + 2]),
                float(q[q_base + d + 3]));
            const float4 kv = float4(
                float(k_pool[k_base + d + 0]),
                float(k_pool[k_base + d + 1]),
                float(k_pool[k_base + d + 2]),
                float(k_pool[k_base + d + 3]));
            logit += dot(qv, kv);
        }
        for (; d < HEAD_DIM; ++d) {
            logit += float(q[q_base + d]) * float(k_pool[k_base + d]);
        }
        logit *= SM_SCALE;
        local_max = max(local_max, logit);
    }
    const float row_max = simd_max_32(local_max);

    float local_sum = 0.0f;
    float local_acc[PAGED_ATTN_MAX_HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {
        local_acc[d] = 0.0f;
    }

    for (int32_t i = start + int32_t(lane); i < end; i += int32_t(PAGED_ATTN_TG_SIZE)) {
        const int32_t slot = kv_indices[i];
        const uint k_base = ((uint)slot * NUM_KV_HEADS + kv_head) * HEAD_DIM;

        float logit = 0.0f;
        uint d = 0;
        for (; d + 3 < HEAD_DIM; d += 4) {
            const float4 qv = float4(
                float(q[q_base + d + 0]),
                float(q[q_base + d + 1]),
                float(q[q_base + d + 2]),
                float(q[q_base + d + 3]));
            const float4 kv = float4(
                float(k_pool[k_base + d + 0]),
                float(k_pool[k_base + d + 1]),
                float(k_pool[k_base + d + 2]),
                float(k_pool[k_base + d + 3]));
            logit += dot(qv, kv);
        }
        for (; d < HEAD_DIM; ++d) {
            logit += float(q[q_base + d]) * float(k_pool[k_base + d]);
        }
        logit *= SM_SCALE;

        const float weight = metal::exp(logit - row_max);
        local_sum += weight;
        const uint v_base = ((uint)slot * NUM_KV_HEADS + kv_head) * HEAD_DIM;
        d = 0;
        for (; d + 3 < HEAD_DIM; d += 4) {
            local_acc[d + 0] += weight * float(v_pool[v_base + d + 0]);
            local_acc[d + 1] += weight * float(v_pool[v_base + d + 1]);
            local_acc[d + 2] += weight * float(v_pool[v_base + d + 2]);
            local_acc[d + 3] += weight * float(v_pool[v_base + d + 3]);
        }
        for (; d < HEAD_DIM; ++d) {
            local_acc[d] += weight * float(v_pool[v_base + d]);
        }
    }

    const float row_sum = simd_sum_32(local_sum);

    const uint out_base = (batch_id * NUM_QO_HEADS + q_head) * HEAD_DIM;
    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f / row_sum;
    for (uint d = 0; d < HEAD_DIM; ++d) {
        float acc = simd_sum_32(local_acc[d]);
        if (lane == 0) {
            out[out_base + d] = static_cast<T>(
                acc * inv_sum);
        }
    }
}

#define INSTANTIATE(NAME, T)                                                            \
    [[host_name("paged_attention_decode_" #NAME)]] [[kernel]] void                     \
    paged_attention_decode_##NAME(                                                      \
        const device T*       q          [[buffer(0)]],                                 \
        const device T*       k_pool     [[buffer(1)]],                                 \
        const device T*       v_pool     [[buffer(2)]],                                 \
        const device int32_t* kv_indptr  [[buffer(3)]],                                 \
        const device int32_t* kv_indices [[buffer(4)]],                                 \
        device T*             out        [[buffer(5)]],                                 \
        uint3 tid [[thread_position_in_threadgroup]],                                   \
        uint3 tg_pos [[threadgroup_position_in_grid]]) {                                \
        paged_attention_decode_impl<T>(q, k_pool, v_pool, kv_indptr, kv_indices, out,   \
                                       tid, tg_pos);                                    \
    }

INSTANTIATE(f16,  half)
INSTANTIATE(bf16, bfloat)
INSTANTIATE(f32,  float)
