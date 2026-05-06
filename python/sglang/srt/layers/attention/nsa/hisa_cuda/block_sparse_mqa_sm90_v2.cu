// Stage 4 sparse MQA — sm_90a — v2: cp.async multi-stage pipeline.
//
// Differences from v1:
//   - K and KS loaded via cp.async (multi-stage ring buffer), so chunk N+1's
//     load overlaps with chunk N's WGMMA + reduce.
//   - TopK indices read inline by each thread from gmem on demand (no smem
//     staging) — saves one __syncthreads per prefetch.
//
// Stage scheme:
//   bootstrap: issue cp.async for chunks 0..bootstrap-1 → stages 0..bootstrap-1
//   iter i in [0, K_CHUNKS):
//     if i+bootstrap < K_CHUNKS: issue cp.async for chunk i+bootstrap
//     wait_group<kNumKVStages-1>      // current iter's group must be done
//     fence.proxy.async.shared::cta
//     __syncthreads
//     WGMMA on K_smem[i % S]
//     reduce + store

// Override CUTLASS_DEVICE for synclog stubs so cute headers compile in our env.
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

constexpr int kNumKVStages = 2;                 // 3 needs >48KB smem opt-in

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

extern "C" __global__ void block_sparse_mqa_sm90_v2(
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

    // ---- Q + W: load once at start (sync) ----
    constexpr int Q_VECS = H * CHUNKS_PER_ROW;       // 512
    constexpr int Q_VECS_PER_THREAD = Q_VECS / THREADS;  // 4
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

    // ---- Helper: per-thread row computation, no smem staging ----
    // For (m, kc): the K row = TopK[seq_i, chunk*GROUP + m/K_BLK] * K_BLK + m%K_BLK
    auto k_row_for = [&](int chunk, int m) -> int {
        int g = m / K_BLK;
        int b = m % K_BLK;
        int g_pos = chunk * GROUP + g;
        int block_id = (g_pos < topk) ? topk_seq[g_pos] : -1;
        return block_id * K_BLK + b;
    };

    // Issue cp.async for K + KS into stage `s` for chunk `c`.
    auto issue_loads = [&](int c, int s) {
        // K cp.async: 512 vecs / 128 threads = 4 each.
        constexpr int K_VECS = GEMM_TILE * CHUNKS_PER_ROW;     // 512
        constexpr int K_VECS_PER_THREAD = K_VECS / THREADS;    // 4
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
        // KS cp.async: 64 floats / 128 threads, first 64 threads each load 1.
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

    // ---- Main K_CHUNKS loop ----
    #pragma unroll 1
    for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
        int chunk_idx = outer * K_CHUNKS + k_iter;
        int n_i_start = chunk_idx * GROUP;
        int stage = k_iter % kNumKVStages;

        // Prefetch chunk k_iter+bootstrap into stage (k_iter+bootstrap) % S
        int prefetch_iter = k_iter + bootstrap;
        if (prefetch_iter < K_CHUNKS) {
            int p_stage = prefetch_iter % kNumKVStages;
            issue_loads(outer * K_CHUNKS + prefetch_iter, p_stage);
        }

        // Wait for current stage's load to complete.
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

        // ---- Spill accum to S_smem ----
        __shared__ float S_smem[GEMM_TILE][H];
        #pragma unroll
        for (int s = 0; s < kNumAccum; ++s) {
            int m = warp * 16 + (lane / 4) + ((s / 2) % 2) * 8;
            int n = (lane % 4) * 2 + (s % 2) + (s / 4) * 8;
            S_smem[m][n] = accum[s];
        }
        __syncthreads();

        // ---- Reduce + store ----
        if (tid < GEMM_TILE) {
            int row = k_row_for(chunk_idx, tid);
            float ks_val = (row >= 0 && row < seq_kv) ? KS_smem_buf[stage][tid] : 0.f;
            float acc = 0.f;
            #pragma unroll
            for (int h = 0; h < H; ++h) {
                float v = S_smem[tid][h] * ks_val;
                if (v < 0.f) v = 0.f;
                acc += v * W_smem[h];
            }
            bool pos_valid = (row >= ks_min) && (row < ke_max);
            float out = pos_valid ? acc : -INFINITY;
            int out_col = n_i_start * K_BLK + tid;
            if (out_col < topk * K_BLK) {
                Logits[seq_i * topk * K_BLK + out_col] = out;
            }
        }
        __syncthreads();
    }
}

}  // namespace

void block_sparse_mqa_sm90_v2_launcher(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& k_scale, const at::Tensor& topk_idx,
    at::Tensor& logits, const at::Tensor& weights,
    const at::Tensor& cu_ks, const at::Tensor& cu_ke
) {
    int seq    = q.size(0);
    int seq_kv = k.size(0);
    int topk   = topk_idx.size(-1);
    int num_chunks = (topk + GROUP - 1) / GROUP;
    int K_CHUNKS = std::min(32, num_chunks);
    int outer = (num_chunks + K_CHUNKS - 1) / K_CHUNKS;

    block_sparse_mqa_sm90_v2<<<dim3(seq, outer), THREADS>>>(
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
    m.def("block_sparse_mqa_sm90_v2", &block_sparse_mqa_sm90_v2_launcher,
          "Stage 4 sparse MQA, sm_90a, v2 (cp.async multi-stage pipeline)");
}
