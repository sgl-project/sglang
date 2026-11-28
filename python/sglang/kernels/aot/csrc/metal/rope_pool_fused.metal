// SGLang Apple Silicon Metal kernel: NeoX RoPE fused with KV pool scatter.

#include <metal_stdlib>
using namespace metal;

constant uint  HEAD_DIM           [[function_constant(0)]];
constant uint  NUM_QO_HEADS       [[function_constant(1)]];
constant uint  NUM_KV_HEADS       [[function_constant(2)]];
constant float INV_DIM_LOG2_BASE  [[function_constant(3)]];
// Heads-per-thread amortization (MLX uses 8). Each thread computes cos/sin
// once for its (token,dim) and reuses it across N heads. Saves N-1 trig calls.
constant uint  HEADS_PER_THREAD   [[function_constant(4)]];

constant uint HALF_DIM = HEAD_DIM / 2;

// ----------------------------------------------------------------------
// Kernel 1: Q rope (no branch, no pool write) - heads-per-thread amortized
//   grid: (HALF_DIM, num_tokens, NUM_QO_HEADS / HEADS_PER_THREAD)
//   Each thread processes HEADS_PER_THREAD consecutive Q heads, sharing one
//   cos/sin computation across them.
// ----------------------------------------------------------------------
template <typename T>
inline void rope_q_impl(
    const device T*       q_in,
    device T*             q_out,
    const device int32_t* positions,
    uint3 pos
) {
    const uint dim_idx  = pos.x;
    const uint token_id = pos.y;
    const uint head_block = pos.z;
    const uint head_start = head_block * HEADS_PER_THREAD;

    // Trig is independent of head_id, compute once and reuse.
    const float pos_f = float(positions[token_id]);
    const float theta = pos_f * metal::exp2(
        -float(2u * dim_idx) * INV_DIM_LOG2_BASE);
    const float c     = metal::fast::cos(theta);
    const float s     = metal::fast::sin(theta);

    // Apply to HEADS_PER_THREAD heads. Compiler unrolls when N is a fn-const.
    for (uint h = 0; h < HEADS_PER_THREAD; ++h) {
        const uint head_id = head_start + h;
        // Boundary: when num_qo_heads is not a multiple of N, skip extras.
        if (head_id >= NUM_QO_HEADS) break;

        const uint base = (token_id * NUM_QO_HEADS + head_id) * HEAD_DIM;
        const uint i1   = base + dim_idx;
        const uint i2   = base + HALF_DIM + dim_idx;

        const float x1 = float(q_in[i1]);
        const float x2 = float(q_in[i2]);
        q_out[i1] = static_cast<T>(x1 * c - x2 * s);
        q_out[i2] = static_cast<T>(x1 * s + x2 * c);
    }
}

// ----------------------------------------------------------------------
// Kernel 2: K rope + write rotated K to pool slots (no Q branch)
//   grid: (HALF_DIM, num_tokens, NUM_KV_HEADS)
//   - Same as Kernel 1 but reads from k_in, writes both k_out and k_pool[slot]
//   - slots[token_id] < 0 means "skip pool write"
// ----------------------------------------------------------------------
template <typename T>
inline void rope_k_pool_impl(
    const device T*       k_in,
    device T*             k_out,
    device T*             k_pool,
    const device int32_t* positions,
    const device int32_t* slots,
    uint3 pos
) {
    const uint dim_idx  = pos.x;
    const uint token_id = pos.y;
    const uint head_block = pos.z;
    const uint head_start = head_block * HEADS_PER_THREAD;

    const float pos_f = float(positions[token_id]);
    const float theta = pos_f * metal::exp2(
        -float(2u * dim_idx) * INV_DIM_LOG2_BASE);
    const float c     = metal::fast::cos(theta);
    const float s     = metal::fast::sin(theta);

    // Hoist slot lookup; same for all heads of this token.
    const int32_t slot = slots[token_id];
    const bool write_pool = slot >= 0;

    for (uint h = 0; h < HEADS_PER_THREAD; ++h) {
        const uint head_id = head_start + h;
        if (head_id >= NUM_KV_HEADS) break;

        const uint base = (token_id * NUM_KV_HEADS + head_id) * HEAD_DIM;
        const uint i1   = base + dim_idx;
        const uint i2   = base + HALF_DIM + dim_idx;

        const float x1 = float(k_in[i1]);
        const float x2 = float(k_in[i2]);
        const T r1 = static_cast<T>(x1 * c - x2 * s);
        const T r2 = static_cast<T>(x1 * s + x2 * c);
        k_out[i1] = r1;
        k_out[i2] = r2;

        if (write_pool) {
            const uint pool_base =
                ((uint)slot * NUM_KV_HEADS + head_id) * HEAD_DIM;
            k_pool[pool_base + dim_idx]            = r1;
            k_pool[pool_base + HALF_DIM + dim_idx] = r2;
        }
    }
}

// ----------------------------------------------------------------------
// Kernel 3: V copy to pool slots
//   grid: (HEAD_DIM, num_tokens, NUM_KV_HEADS)
//   - Pure memcpy from v_in[token, head, dim] to v_pool[slot, head, dim]
//   - No trig, no rotation
// ----------------------------------------------------------------------
template <typename T>
inline void v_to_pool_impl(
    const device T*       v_in,
    device T*             v_pool,
    const device int32_t* slots,
    uint3 pos
) {
    const uint dim_idx  = pos.x;
    const uint token_id = pos.y;
    const uint head_id  = pos.z;

    const int32_t slot = slots[token_id];
    if (slot < 0) return;

    const uint src = (token_id * NUM_KV_HEADS + head_id) * HEAD_DIM + dim_idx;
    const uint dst = ((uint)slot * NUM_KV_HEADS + head_id) * HEAD_DIM + dim_idx;
    v_pool[dst] = v_in[src];
}

// ----------------------------------------------------------------------
// Experimental single-dispatch rectangular kernel.
//   grid: (HEAD_DIM, num_tokens, max(NUM_QO_HEADS, NUM_KV_HEADS) / HPT)
//   - dim < HALF_DIM lanes rotate Q for Q heads
//   - dim < HALF_DIM lanes rotate K and write K pool for KV heads
//   - all dim lanes copy V for KV heads
//   This avoids packed div/mod region decoding but wastes lanes when Q and KV
//   head counts differ.
// ----------------------------------------------------------------------
template <typename T>
inline void rope_pool_fused_rect_impl(
    const device T*       q_in,
    const device T*       k_in,
    const device T*       v_in,
    device T*             q_out,
    device T*             k_out,
    device T*             k_pool,
    device T*             v_pool,
    const device int32_t* positions,
    const device int32_t* slots,
    uint3 pos
) {
    const uint dim_idx = pos.x;
    const uint token_id = pos.y;
    const uint head_start = pos.z * HEADS_PER_THREAD;

    const bool rope_lane = dim_idx < HALF_DIM;
    float c = 0.0f;
    float s = 0.0f;
    if (rope_lane) {
        const float pos_f = float(positions[token_id]);
        const float theta = pos_f * metal::exp2(
            -float(2u * dim_idx) * INV_DIM_LOG2_BASE);
        c = metal::fast::cos(theta);
        s = metal::fast::sin(theta);
    }

    const int32_t slot = slots[token_id];
    const bool write_pool = slot >= 0;

    for (uint h = 0; h < HEADS_PER_THREAD; ++h) {
        const uint head_id = head_start + h;

        if (rope_lane && head_id < NUM_QO_HEADS) {
            const uint q_base = (token_id * NUM_QO_HEADS + head_id) * HEAD_DIM;
            const uint q_i1 = q_base + dim_idx;
            const uint q_i2 = q_base + HALF_DIM + dim_idx;
            const float x1 = float(q_in[q_i1]);
            const float x2 = float(q_in[q_i2]);
            q_out[q_i1] = static_cast<T>(x1 * c - x2 * s);
            q_out[q_i2] = static_cast<T>(x1 * s + x2 * c);
        }

        if (head_id >= NUM_KV_HEADS) continue;

        const uint kv_base = (token_id * NUM_KV_HEADS + head_id) * HEAD_DIM;
        const uint pool_base =
            write_pool ? ((uint)slot * NUM_KV_HEADS + head_id) * HEAD_DIM : 0u;

        if (write_pool) {
            v_pool[pool_base + dim_idx] = v_in[kv_base + dim_idx];
        }

        if (!rope_lane) continue;

        const uint k_i1 = kv_base + dim_idx;
        const uint k_i2 = kv_base + HALF_DIM + dim_idx;
        const float x1 = float(k_in[k_i1]);
        const float x2 = float(k_in[k_i2]);
        const T r1 = static_cast<T>(x1 * c - x2 * s);
        const T r2 = static_cast<T>(x1 * s + x2 * c);
        k_out[k_i1] = r1;
        k_out[k_i2] = r2;

        if (write_pool) {
            k_pool[pool_base + dim_idx] = r1;
            k_pool[pool_base + HALF_DIM + dim_idx] = r2;
        }
    }
}

// ----------------------------------------------------------------------
// dtype-specialized entry points
// ----------------------------------------------------------------------
#define INSTANTIATE(NAME, T)                                                     \
    [[host_name("rope_pool_rect_" #NAME)]] [[kernel]] void rope_pool_rect_##NAME(\
        const device T*       q_in       [[buffer(0)]],                          \
        const device T*       k_in       [[buffer(1)]],                          \
        const device T*       v_in       [[buffer(2)]],                          \
        device T*             q_out      [[buffer(3)]],                          \
        device T*             k_out      [[buffer(4)]],                          \
        device T*             k_pool     [[buffer(5)]],                          \
        device T*             v_pool     [[buffer(6)]],                          \
        const device int32_t* positions  [[buffer(7)]],                          \
        const device int32_t* slots      [[buffer(8)]],                          \
        uint3 pos [[thread_position_in_grid]]) {                                 \
        rope_pool_fused_rect_impl<T>(q_in, k_in, v_in, q_out, k_out, k_pool,     \
                                     v_pool, positions, slots, pos);             \
    }                                                                            \
    [[host_name("rope_q_" #NAME)]] [[kernel]] void rope_q_##NAME(                \
        const device T*       q_in       [[buffer(0)]],                          \
        device T*             q_out      [[buffer(1)]],                          \
        const device int32_t* positions  [[buffer(2)]],                          \
        uint3 pos [[thread_position_in_grid]]) {                                 \
        rope_q_impl<T>(q_in, q_out, positions, pos);                             \
    }                                                                            \
    [[host_name("rope_k_pool_" #NAME)]] [[kernel]] void rope_k_pool_##NAME(      \
        const device T*       k_in       [[buffer(0)]],                          \
        device T*             k_out      [[buffer(1)]],                          \
        device T*             k_pool     [[buffer(2)]],                          \
        const device int32_t* positions  [[buffer(3)]],                          \
        const device int32_t* slots      [[buffer(4)]],                          \
        uint3 pos [[thread_position_in_grid]]) {                                 \
        rope_k_pool_impl<T>(k_in, k_out, k_pool, positions, slots, pos);         \
    }                                                                            \
    [[host_name("v_to_pool_" #NAME)]] [[kernel]] void v_to_pool_##NAME(          \
        const device T*       v_in       [[buffer(0)]],                          \
        device T*             v_pool     [[buffer(1)]],                          \
        const device int32_t* slots      [[buffer(2)]],                          \
        uint3 pos [[thread_position_in_grid]]) {                                 \
        v_to_pool_impl<T>(v_in, v_pool, slots, pos);                             \
    }

INSTANTIATE(f16,  half)
INSTANTIATE(bf16, bfloat)
INSTANTIATE(f32,  float)
