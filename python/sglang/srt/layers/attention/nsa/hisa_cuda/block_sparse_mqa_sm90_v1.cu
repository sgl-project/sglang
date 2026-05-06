// Stage 4 sparse MQA — sm_90a — v1: DG wgmma wrappers + B128-swizzled smem.
//
// STATUS: Correct (rel 3e-7 vs Triton). Slow (~4.5× Triton) — single
// warpgroup, no producer/consumer split, no TMA, no multi-stage pipeline.
// v2 adds those for perf.
//
// Key fence requirement: smem stores from generic proxy (regular STS / STG
// stores) are NOT visible to async proxy (WGMMA's smem read path) without an
// explicit `fence.proxy.async.shared::cta`. __syncthreads() alone is a
// generic-proxy fence; without the proxy fence, WGMMA can read stale smem
// for some lanes (which is what burned us on the m=51 bug for hours).
// DG dodges this by TMA-loading into smem — TMA writes are async-proxy and
// the mbarrier wait gives implicit proxy visibility.
//
// Override CUTLASS_DEVICE for synclog stubs so cute headers compile in our env
// (stubs are __device__-only but called from __host__ __device__ fma).
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

using WGMMA = deep_gemm::sm90::FP8MMASelector<64>::type;     // m=64,n=64,k=32, 32 fp32 accum
constexpr int kNumAccum = WGMMA::kNumAccum;                   // 32

// fp8 [M, D=128] tile, physically B128-swizzled (matches TMA-with-128B-swizzle
// output, which is what cute's B128 GMMA descriptor expects to read).
//
// Swizzle<3,4,3>: for byte address `a` within a 1024-byte block,
//   physical_a = a XOR (((a >> 7) & 7) << 4)
// In our [M, D=128] tile (row stride 128 = 2^7), bits[7..9] of `a` are the
// low 3 bits of m. So per-16-byte chunk at logical (m, kc):
//   logical:  m * 128 + kc * 16
//   physical: m * 128 + (kc XOR (m & 7)) * 16
// LBO unused, SBO = m_outer stride = 8 rows * 128 = 1024 bytes.

__device__ __forceinline__ int b128_swiz_offset16(int m, int kc) {
    return m * 128 + (kc ^ (m & 7)) * 16;
}

extern "C" __global__ void block_sparse_mqa_sm90_v1(
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
    // B128 swizzle operates within 1024-byte blocks; smem must be 1024-aligned.
    __shared__ alignas(1024) uint8_t K_smem[TILE_BYTES];
    __shared__ alignas(1024) uint8_t Q_smem[H * D];
    __shared__ float W_smem[H];
    __shared__ int32_t topk_block_ids[GROUP];
    __shared__ int32_t k_rows_smem[GEMM_TILE];

    const int ks_min = CuKS[seq_i];
    const int ke_max = CuKE[seq_i];

    // B128-swizzled store: 16-byte vec per (h, kc). 128 threads × 4 vecs = 512 vecs.
    constexpr int CHUNKS_PER_ROW = D / 16;           // 8
    constexpr int Q_VECS = (H * CHUNKS_PER_ROW);     // 512
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
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
        int chunk_idx = outer * K_CHUNKS + k_iter;
        int n_i_start = chunk_idx * GROUP;

        if (tid < GROUP) {
            int g_pos = n_i_start + tid;
            topk_block_ids[tid] = (g_pos < topk) ? TopK[seq_i * topk + g_pos] : -1;
        }
        __syncthreads();
        if (tid < GEMM_TILE) {
            int g = tid / K_BLK, b = tid % K_BLK;
            k_rows_smem[tid] = topk_block_ids[g] * K_BLK + b;
        }
        __syncthreads();

        constexpr int K_VECS = GEMM_TILE * CHUNKS_PER_ROW;          // 512
        constexpr int K_VECS_PER_THREAD = K_VECS / THREADS;         // 4
        #pragma unroll
        for (int p = 0; p < K_VECS_PER_THREAD; ++p) {
            int vidx = p * THREADS + tid;
            int m = vidx / CHUNKS_PER_ROW;
            int kc = vidx % CHUNKS_PER_ROW;
            int row = k_rows_smem[m];
            int safe = (row >= 0 && row < seq_kv) ? row : 0;
            const uint4* src = reinterpret_cast<const uint4*>(K + safe * D + kc * 16);
            *reinterpret_cast<uint4*>(K_smem + b128_swiz_offset16(m, kc)) = *src;
        }
        // Generic-proxy smem writes must be made visible to async (WGMMA) proxy.
        // __syncthreads() alone is generic-proxy; need explicit fence.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        // ---- WGMMA: 4 iters of k=32 → k_total=128 ----
        float accum[kNumAccum];

        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_arrive();
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            auto desc_a = make_smem_desc(K_smem + kt * 32, /*layout=*/1, /*LBO=*/0, /*SBO=*/1024);
            auto desc_b = make_smem_desc(Q_smem + kt * 32, /*layout=*/1, /*LBO=*/0, /*SBO=*/1024);
            // First kt: scale_d=false (overwrite). Subsequent: accumulate.
            WGMMA::wgmma(desc_a, desc_b, accum, /*scale_d=*/(kt != 0));
        }
        warpgroup_commit_batch();
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_wait<0>();

        // ---- Spill accum to smem ----
        // CLayout_64xN per cute: m = warp*16 + (l/4) + ((s/2)%2)*8,
        //                       n = (l%4)*2 + (s%2) + (s/4)*8.
        __shared__ float S_smem[GEMM_TILE][H];
        #pragma unroll
        for (int s = 0; s < kNumAccum; ++s) {
            int m = warp * 16 + (lane / 4) + ((s / 2) % 2) * 8;
            int n = (lane % 4) * 2 + (s % 2) + (s / 4) * 8;
            S_smem[m][n] = accum[s];
        }
        __syncthreads();

        if (tid < GEMM_TILE) {
            int row = k_rows_smem[tid];
            float ks_val = (row >= 0 && row < seq_kv) ? KS[row] : 0.f;
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

void block_sparse_mqa_sm90_v1_launcher(
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

    block_sparse_mqa_sm90_v1<<<dim3(seq, outer), THREADS>>>(
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
    m.def("block_sparse_mqa_sm90_v1", &block_sparse_mqa_sm90_v1_launcher,
          "Stage 4 sparse MQA, sm_90a, v1 (DG wgmma + INTERLEAVE smem)");
}
