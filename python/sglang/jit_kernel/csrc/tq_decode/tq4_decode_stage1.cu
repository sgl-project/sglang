/*
 * TQ4 Decode Attention Stage1 v2 — 3-Phase CUDA Kernel
 *
 * Phase 1 (QK):  lane→token, loop over packed_dim, compute qk per token
 * Phase 2 (softmax): warp reduce max/sum, compute p, store to smem
 * Phase 3 (V accum): lane→dimension, loop over tokens, load V + weighted accum
 *
 * This avoids per-token warp reductions for V and enables coalesced V loads.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <torch/extension.h>

constexpr int HEAD_DIM = 128;
constexpr int PACKED_DIM = 64;
constexpr int BLOCK_N = 32;     // tokens per tile (doubled from Triton)
constexpr int BLOCK_H = 4;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_H;  // 1 warp per Q head
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 128
constexpr int MIN_BLOCK_KV = 32;

// packed positions per lane: PACKED_DIM / WARP_SIZE = 2
constexpr int PP_PER_LANE = PACKED_DIM / WARP_SIZE;  // 2

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__global__ void tq4_decode_stage1_v2(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t* __restrict__ K_packed,
    const uint8_t* __restrict__ V_packed,
    const __nv_bfloat16* __restrict__ K_dscale,
    const __nv_bfloat16* __restrict__ V_dscale,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ num_kv_splits,
    float* __restrict__ att_out,
    float* __restrict__ att_lse,
    const float* __restrict__ k_centroids,
    const float* __restrict__ v_centroids,
    float sm_scale,
    int H_Q, int H_KV, int max_kv_splits,
    int stride_q_batch, int stride_q_head,
    int stride_kp_pool, int stride_kp_head,
    int stride_vp_pool, int stride_vp_head,
    int stride_kds_pool, int stride_vds_pool,
    int stride_att_batch, int stride_att_head, int stride_att_split,
    int kv_group_num
) {
    const int batch_id = blockIdx.x;
    const int head_group_id = blockIdx.y;
    const int split_id = blockIdx.z;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int q_head = head_group_id * BLOCK_H + warp_id;
    if (q_head >= H_Q) return;
    const int kv_head = q_head / kv_group_num;

    // KV split range
    const int kv_start = kv_indptr[batch_id];
    const int kv_end = kv_indptr[batch_id + 1];
    const int kv_len = kv_end - kv_start;
    const int n_splits = num_kv_splits[batch_id];
    const int split_len = (kv_len + n_splits - 1) / n_splits;
    const int aligned_len = ((split_len + MIN_BLOCK_KV - 1) / MIN_BLOCK_KV) * MIN_BLOCK_KV;
    const int my_start = kv_start + split_id * aligned_len;
    const int my_end = min(my_start + aligned_len, kv_end);

    if (my_start >= kv_end) {
        if (lane_id == 0) {
            att_lse[batch_id * H_Q * max_kv_splits + q_head * max_kv_splits + split_id] = -FLT_MAX;
        }
        return;
    }

    // ---- Shared memory ----
    __shared__ float s_k_cb[16];    // K codebook
    __shared__ float s_v_cb[16];    // V codebook
    // Q in shared memory: [BLOCK_H][PACKED_DIM] for even and odd
    __shared__ float s_q_even[BLOCK_H][PACKED_DIM];
    __shared__ float s_q_odd[BLOCK_H][PACKED_DIM];
    // Per-tile: softmax weights and token indices
    __shared__ float s_p[BLOCK_H][BLOCK_N];
    __shared__ int s_kv_idx[BLOCK_N];
    __shared__ float s_kds[BLOCK_N];  // K dequant scales
    __shared__ float s_vds[BLOCK_N];  // V dequant scales

    // Load codebook (first 16 threads)
    if (threadIdx.x < 16) {
        s_k_cb[threadIdx.x] = k_centroids[threadIdx.x];
        s_v_cb[threadIdx.x] = v_centroids[threadIdx.x];
    }

    // Load Q to shared memory
    // Each lane loads 2 positions for its warp's Q head
    {
        const __nv_bfloat16* q_ptr = Q + batch_id * stride_q_batch + q_head * stride_q_head;
        for (int i = 0; i < PP_PER_LANE; i++) {
            int pp = lane_id * PP_PER_LANE + i;
            if (pp < PACKED_DIM) {
                s_q_even[warp_id][pp] = __bfloat162float(q_ptr[2 * pp]);
                s_q_odd[warp_id][pp] = __bfloat162float(q_ptr[2 * pp + 1]);
            }
        }
    }
    __syncthreads();

    // Accumulators for V (lane→dimension mapping in phase 3)
    // Each lane handles PP_PER_LANE=2 packed positions → 4 output values
    float acc_even[PP_PER_LANE] = {0.0f, 0.0f};
    float acc_odd[PP_PER_LANE] = {0.0f, 0.0f};
    float e_max = -FLT_MAX;
    float e_sum = 0.0f;

    // Main loop over token tiles
    for (int tok_start = my_start; tok_start < my_end; tok_start += BLOCK_N) {
        const int n_valid = min(BLOCK_N, my_end - tok_start);

        // Load token indices and dequant scales to shared memory
        if (lane_id < BLOCK_N) {
            int tok = tok_start + lane_id;
            bool valid = tok < my_end;
            int idx = valid ? kv_indices[tok] : 0;
            s_kv_idx[lane_id] = idx;
            s_kds[lane_id] = valid ? __bfloat162float(K_dscale[idx * stride_kds_pool + kv_head]) : 0.0f;
            s_vds[lane_id] = valid ? __bfloat162float(V_dscale[idx * stride_vds_pool + kv_head]) : 0.0f;
        }
        __syncwarp();

        // ======== Phase 1: QK computation (lane→token) ========
        // Each lane handles 1 token, loops over packed_dim
        float qk;
        {
            bool valid = lane_id < n_valid;
            int kv_idx = s_kv_idx[lane_id];

            float local_qk = 0.0f;
            if (valid) {
                const uint8_t* k_ptr = K_packed + kv_idx * stride_kp_pool + kv_head * stride_kp_head;
                #pragma unroll 8
                for (int p = 0; p < PACKED_DIM; p++) {
                    uint8_t kb = k_ptr[p];
                    float k_e = s_k_cb[kb & 0x0F];
                    float k_o = s_k_cb[(kb >> 4) & 0x0F];
                    local_qk += s_q_even[warp_id][p] * k_e + s_q_odd[warp_id][p] * k_o;
                }
                local_qk *= s_kds[lane_id] * sm_scale;
            }
            qk = valid ? local_qk : -FLT_MAX;
        }

        // ======== Phase 2: Online softmax ========
        float tile_max = warp_reduce_max(qk);
        float old_max = e_max;
        e_max = fmaxf(e_max, tile_max);
        float exp_correction = expf(old_max - e_max);

        float p = expf(qk - e_max);
        if (lane_id >= n_valid) p = 0.0f;

        float p_sum = warp_reduce_sum(p);

        // Rescale accumulators
        #pragma unroll
        for (int i = 0; i < PP_PER_LANE; i++) {
            acc_even[i] *= exp_correction;
            acc_odd[i] *= exp_correction;
        }
        e_sum = e_sum * exp_correction + p_sum;

        // Store p * v_dscale to shared memory for phase 3
        float p_scaled = (lane_id < n_valid) ? p * s_vds[lane_id] : 0.0f;
        s_p[warp_id][lane_id] = p_scaled;
        __syncwarp();

        // ======== Phase 3: V accumulation (lane→dimension) ========
        // Each lane handles PP_PER_LANE=2 packed positions, loops over n_valid tokens
        for (int n = 0; n < n_valid; n++) {
            float pn = s_p[warp_id][n];
            if (pn == 0.0f) continue;

            int kv_idx = s_kv_idx[n];
            const uint8_t* v_ptr = V_packed + kv_idx * stride_vp_pool + kv_head * stride_vp_head;

            #pragma unroll
            for (int i = 0; i < PP_PER_LANE; i++) {
                int pp = lane_id * PP_PER_LANE + i;
                uint8_t vb = v_ptr[pp];
                acc_even[i] += pn * s_v_cb[vb & 0x0F];
                acc_odd[i] += pn * s_v_cb[(vb >> 4) & 0x0F];
            }
        }
    }

    // ======== Store output (interleaved) ========
    float inv_sum = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;
    int out_base = batch_id * stride_att_batch + q_head * stride_att_head + split_id * stride_att_split;

    #pragma unroll
    for (int i = 0; i < PP_PER_LANE; i++) {
        int pp = lane_id * PP_PER_LANE + i;
        att_out[out_base + 2 * pp] = acc_even[i] * inv_sum;
        att_out[out_base + 2 * pp + 1] = acc_odd[i] * inv_sum;
    }

    // Store LSE
    if (lane_id == 0) {
        int lse_idx = batch_id * H_Q * max_kv_splits + q_head * max_kv_splits + split_id;
        att_lse[lse_idx] = e_max + logf(fmaxf(e_sum, 1e-10f));
    }
}


void tq4_decode_stage1(
    torch::Tensor Q, torch::Tensor K_packed, torch::Tensor V_packed,
    torch::Tensor K_dscale, torch::Tensor V_dscale,
    torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor num_kv_splits_tensor,
    torch::Tensor att_out, torch::Tensor att_lse,
    torch::Tensor k_centroids, torch::Tensor v_centroids,
    float sm_scale, int max_kv_splits
) {
    const int batch = Q.size(0);
    const int h_q = Q.size(1);
    const int h_kv = K_packed.size(1);
    const int kv_group = h_q / h_kv;

    dim3 grid(batch, (h_q + BLOCK_H - 1) / BLOCK_H, max_kv_splits);
    dim3 block(BLOCK_SIZE);

    tq4_decode_stage1_v2<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        K_packed.data_ptr<uint8_t>(),
        V_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(K_dscale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V_dscale.data_ptr()),
        kv_indptr.data_ptr<int32_t>(),
        kv_indices.data_ptr<int32_t>(),
        num_kv_splits_tensor.data_ptr<int32_t>(),
        att_out.data_ptr<float>(),
        att_lse.data_ptr<float>(),
        k_centroids.data_ptr<float>(),
        v_centroids.data_ptr<float>(),
        sm_scale, h_q, h_kv, max_kv_splits,
        Q.stride(0), Q.stride(1),
        K_packed.stride(0), K_packed.stride(1),
        V_packed.stride(0), V_packed.stride(1),
        K_dscale.stride(0), V_dscale.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        kv_group
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tq4_decode_stage1", &tq4_decode_stage1, "TQ4 decode attention stage1 v2");
}
