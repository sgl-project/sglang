// Stage 4 sparse MQA — sm_90a — v3: DG-style register reduce + 3-stage pipe.
//
// Differences from v2:
//   - No S_smem [64][64] spill (saves 16KB)
//   - Reduce done by all 128 threads via per-thread accum walk + shfl_xor
//   - kNumKVStages = 3 (now fits because S_smem freed)
//
// Reduce derivation (CuTe CLayout 64x64, accum[32] per thread):
//   m_acc(s) = warp*16 + (lane/4) + ((s/2)%2)*8
//   n_acc(s) = (lane%4)*2 + (s%2) + (s/4)*8
//   half = (s/2)%2  → 16 accum cover m_lo (=v_0_offset), 16 cover m_hi (=v_1_offset)
//   For per-row scale + reduce over n=H:
//     For each thread, sum the 16 accum at m_lo (× weight × relu) → partial sum_lo
//     Same for m_hi → partial sum_hi
//     shfl_xor over offset 1, 2 reduces across lane%4 ∈ {0..3} (same m, different n)
//   weight pre-load: weights_reg[j_dg] = W_smem[(j_dg/2)*8 + (j_dg&1) + (lane%4)*2]
//     for accum[s], w_idx = 2*(s/4) + (s%2)

#include <cutlass/detail/helper_macros.hpp>
#undef CUTLASS_DEVICE
#define CUTLASS_DEVICE __forceinline__ __device__ __host__
#include <cutlass/arch/synclog.hpp>
#undef CUTLASS_DEVICE
#define CUTLASS_DEVICE __forceinline__ __device__

#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include <deep_gemm/common/sm90_utils.cuh>

namespace {

constexpr int H = 64;
constexpr int D = 128;
constexpr int K_BLK = 8;
constexpr int GROUP = 8;
constexpr int GEMM_TILE = K_BLK * GROUP;       // 64
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;            // 128

constexpr int kNumKVStages = 3;

using WGMMA = deep_gemm::sm90::FP8MMASelector<64>::type;
constexpr int kNumAccum = WGMMA::kNumAccum;     // 32

__device__ __forceinline__ int b128_swiz_offset16(int m, int kc) {
    return m * 128 + (kc ^ (m & 7)) * 16;
}

__device__ __forceinline__ void cp_async_16(uint8_t* smem, const void* gmem) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_4(float* smem, const float* gmem) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile(
        "cp.async.ca.shared::cta.global [%0], [%1], 4;\n"
        :: "r"(smem_int), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

extern "C" __global__ void block_sparse_mqa_sm90_v3(
    const __nv_fp8_e4m3* __restrict__ Q,
    const __nv_fp8_e4m3* __restrict__ K,
    const float* __restrict__ KS,
    const int32_t* __restrict__ TopK,
    float* __restrict__ Logits,
    const float* __restrict__ W,
    const int32_t* __restrict__ CuKS,
    const int32_t* __restrict__ CuKE,
    int seq_kv,
    int topk,
    int K_CHUNKS
) {
    using namespace deep_gemm::sm90;

    int seq_i = blockIdx.x;
    int outer = blockIdx.y;
    int tid   = threadIdx.x;
    int warp  = tid / 32;
    int lane  = tid % 32;

    constexpr int TILE_BYTES = GEMM_TILE * D;        // 8192
    __shared__ alignas(1024) uint8_t Q_smem[H * D];
    __shared__ alignas(1024) uint8_t K_smem_buf[kNumKVStages][TILE_BYTES];
    __shared__ float W_smem[H];
    __shared__ float KS_smem_buf[kNumKVStages][GEMM_TILE];

    const int ks_min = CuKS[seq_i];
    const int ke_max = CuKE[seq_i];
    const int32_t* __restrict__ topk_seq = TopK + seq_i * topk;

    constexpr int CHUNKS_PER_ROW = D / 16;           // 8

    // ---- Q + W: load once at start ----
    constexpr int Q_VECS = H * CHUNKS_PER_ROW;
    constexpr int Q_VECS_PER_THREAD = Q_VECS / THREADS;
    #pragma unroll
    for (int p = 0; p < Q_VECS_PER_THREAD; ++p) {
        int vidx = p * THREADS + tid;
        int h = vidx / CHUNKS_PER_ROW;
        int kc = vidx % CHUNKS_PER_ROW;
        const uint4* src = reinterpret_cast<const uint4*>(Q + seq_i * H * D + h * D + kc * 16);
        *reinterpret_cast<uint4*>(Q_smem + b128_swiz_offset16(h, kc)) = *src;
    }
    if (tid < H) W_smem[tid] = W[seq_i * H + tid];
    fence_proxy_async();
    __syncthreads();

    // Pre-load per-thread weights into registers (DG-style indexing).
    float weights_reg[H / 4];
    #pragma unroll
    for (int j = 0; j < H / 4; ++j) {
        weights_reg[j] = W_smem[(j / 2) * 8 + (j & 1) + (lane % 4) * 2];
    }

    auto k_row_for = [&](int chunk, int m) -> int {
        int g = m / K_BLK;
        int b = m % K_BLK;
        int g_pos = chunk * GROUP + g;
        int block_id = (g_pos < topk) ? topk_seq[g_pos] : -1;
        return block_id * K_BLK + b;
    };

    auto issue_loads = [&](int c, int s) {
        constexpr int K_VECS = GEMM_TILE * CHUNKS_PER_ROW;
        constexpr int K_VECS_PER_THREAD = K_VECS / THREADS;
        #pragma unroll
        for (int p = 0; p < K_VECS_PER_THREAD; ++p) {
            int vidx = p * THREADS + tid;
            int m = vidx / CHUNKS_PER_ROW;
            int kc = vidx % CHUNKS_PER_ROW;
            int row = k_row_for(c, m);
            int safe = (row >= 0 && row < seq_kv) ? row : 0;
            const void* src = K + safe * D + kc * 16;
            cp_async_16(K_smem_buf[s] + b128_swiz_offset16(m, kc), src);
        }
        if (tid < GEMM_TILE) {
            int row = k_row_for(c, tid);
            int safe = (row >= 0 && row < seq_kv) ? row : 0;
            cp_async_4(KS_smem_buf[s] + tid, KS + safe);
        }
        cp_async_commit();
    };

    // ---- Bootstrap: pre-issue first kNumKVStages-1 stages ----
    int bootstrap = min(kNumKVStages - 1, K_CHUNKS);
    #pragma unroll 1
    for (int b = 0; b < bootstrap; ++b) {
        issue_loads(outer * K_CHUNKS + b, b);
    }

    // Per-thread output positions (m positions this thread is responsible for).
    int warp_offset = warp * 16;
    int v_0_offset = warp_offset + (lane / 4);        // 0..7 within warp
    int v_1_offset = warp_offset + (lane / 4) + 8;    // 8..15 within warp

    // ---- Main K_CHUNKS loop ----
    #pragma unroll 1
    for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
        int chunk_idx = outer * K_CHUNKS + k_iter;
        int n_i_start = chunk_idx * GROUP;
        int stage = k_iter % kNumKVStages;

        int prefetch_iter = k_iter + bootstrap;
        if (prefetch_iter < K_CHUNKS) {
            int p_stage = prefetch_iter % kNumKVStages;
            issue_loads(outer * K_CHUNKS + prefetch_iter, p_stage);
        }

        cp_async_wait_group<kNumKVStages - 1>();
        fence_proxy_async();
        __syncthreads();

        // ---- WGMMA on K_smem[stage], Q_smem ----
        float accum[kNumAccum];
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_arrive();
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            auto desc_a = make_smem_desc(K_smem_buf[stage] + kt * 32, /*layout=*/1, /*LBO=*/0, /*SBO=*/1024);
            auto desc_b = make_smem_desc(Q_smem + kt * 32, /*layout=*/1, /*LBO=*/0, /*SBO=*/1024);
            WGMMA::wgmma(desc_a, desc_b, accum, /*scale_d=*/(kt != 0));
        }
        warpgroup_commit_batch();
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_wait<0>();

        // ---- Per-row KS read + register reduce ----
        int row_lo = k_row_for(chunk_idx, v_0_offset);
        int row_hi = k_row_for(chunk_idx, v_1_offset);
        float ks_lo = (row_lo >= 0 && row_lo < seq_kv) ? KS_smem_buf[stage][v_0_offset] : 0.f;
        float ks_hi = (row_hi >= 0 && row_hi < seq_kv) ? KS_smem_buf[stage][v_1_offset] : 0.f;

        // Reduce 16 accum per half (m_lo / m_hi) over n=H.
        // s_lo = 4*j_outer + inner       (half=0)
        // s_hi = 4*j_outer + 2 + inner   (half=1)
        // w_idx = 2*j_outer + inner      (same for both halves)
        float sum_lo = 0.f, sum_hi = 0.f;
        #pragma unroll
        for (int j_outer = 0; j_outer < H / 8; ++j_outer) {
            #pragma unroll
            for (int inner = 0; inner < 2; ++inner) {
                int s_lo = j_outer * 4 + inner;
                int s_hi = j_outer * 4 + 2 + inner;
                int w_idx = 2 * j_outer + inner;
                float w = weights_reg[w_idx];
                sum_lo += fmaxf(accum[s_lo], 0.f) * w;
                sum_hi += fmaxf(accum[s_hi], 0.f) * w;
            }
        }
        sum_lo *= ks_lo;
        sum_hi *= ks_hi;

        // Inter-thread reduce: shfl_xor across lane%4 (4-way), 2 levels.
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int offset = 1u << j;
            sum_lo += __shfl_xor_sync(0xffffffffu, sum_lo, offset);
            sum_hi += __shfl_xor_sync(0xffffffffu, sum_hi, offset);
        }

        bool valid_lo = (row_lo >= ks_min) && (row_lo < ke_max);
        bool valid_hi = (row_hi >= ks_min) && (row_hi < ke_max);
        float out_lo = valid_lo ? sum_lo : -INFINITY;
        float out_hi = valid_hi ? sum_hi : -INFINITY;

        int out_col_lo = n_i_start * K_BLK + v_0_offset;
        int out_col_hi = n_i_start * K_BLK + v_1_offset;
        if (out_col_lo < topk * K_BLK) {
            Logits[seq_i * topk * K_BLK + out_col_lo] = out_lo;
        }
        if (out_col_hi < topk * K_BLK) {
            Logits[seq_i * topk * K_BLK + out_col_hi] = out_hi;
        }
    }
}

}  // namespace

void block_sparse_mqa_sm90_v3_launcher(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& k_scale, const at::Tensor& topk_idx,
    at::Tensor& logits, const at::Tensor& weights,
    const at::Tensor& cu_ks, const at::Tensor& cu_ke
) {
    int seq    = q.size(0);
    int seq_kv = k.size(0);
    int topk   = topk_idx.size(-1);
    int num_chunks = (topk + GROUP - 1) / GROUP;
    int K_CHUNKS = num_chunks;
    int outer = 1;

    block_sparse_mqa_sm90_v3<<<dim3(seq, outer), THREADS>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr()),
        k_scale.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_ks.data_ptr<int32_t>(),
        cu_ke.data_ptr<int32_t>(),
        seq_kv, topk, K_CHUNKS
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_mqa_sm90_v3", &block_sparse_mqa_sm90_v3_launcher,
          "Stage 4 sparse MQA, sm_90a, v3 (DG-style reduce + 3-stage pipe)");
}
