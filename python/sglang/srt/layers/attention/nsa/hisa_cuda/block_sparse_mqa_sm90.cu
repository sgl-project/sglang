// Minimal sparse stage-4 MQA kernel for Hopper sm_90a.
//
// Design (v0 — no TMA, no wgmma yet, baseline only):
//   Grid:   (seq_q, outer)         — one CTA per (query, K_CHUNKS slice)
//   Threads: 128                    — 4 warps
//
// Per CTA:
//   1. Load Q[H, D] fp8 + W[H] f32 once into smem/regs
//   2. For each of K_CHUNKS chunks:
//        a. For g in 0..G: load topk_id, gather K[topk_id*K_BLK : (topk_id+1)*K_BLK, D]
//           into smem[g*K_BLK : (g+1)*K_BLK, D]
//        b. Run wgmma K @ Q^T → s[GEMM_TILE, H]
//        c. relu(s * ks) * w, sum over H → logits[GEMM_TILE]
//        d. Write logits with mask
//
// This v0 uses regular __ldg gather (no TMA) and m16n8k16 mma (no wgmma). It's a
// correctness scaffold. Future versions add TMA + wgmma + producer/consumer.

#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>

namespace {

// Hardcoded for the user's K=8 production path; can templatize later.
constexpr int H = 64;
constexpr int D = 128;
constexpr int K_BLK = 8;
constexpr int GROUP = 8;
constexpr int GEMM_TILE = K_BLK * GROUP;  // 64
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;       // 4

__device__ __forceinline__ float to_fp32(__nv_fp8_e4m3 x) {
    return static_cast<float>(x);
}

// Naive scalar fp8 GEMM: C[m,n] = sum_k A[m,k] * B[n,k]  (B is [N, K]).
// For correctness/baseline only — slow but works.
__device__ __forceinline__ void gemm_naive(
    const __nv_fp8_e4m3 K_smem[GEMM_TILE][D],
    const __nv_fp8_e4m3 Q_smem[H][D],
    float S[GEMM_TILE][H],
    int tid
) {
    // Each thread computes ~(GEMM_TILE * H) / 128 = 32 output elements.
    constexpr int OUT = GEMM_TILE * H;            // 64 * 64 = 4096
    constexpr int PER_THREAD = OUT / THREADS;     // 32
    for (int p = 0; p < PER_THREAD; ++p) {
        int idx = p * THREADS + tid;
        int m = idx / H;
        int h = idx % H;
        float acc = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            acc += to_fp32(K_smem[m][d]) * to_fp32(Q_smem[h][d]);
        }
        S[m][h] = acc;
    }
}

extern "C" __global__ void block_sparse_mqa_sm90_v0(
    const __nv_fp8_e4m3* __restrict__ Q,    // [seq, H, D]
    const __nv_fp8_e4m3* __restrict__ K,    // [seq_kv, D]
    const float* __restrict__ KS,           // [seq_kv]
    const int32_t* __restrict__ TopK,       // [seq, topk] i32 expected
    float* __restrict__ Logits,             // [seq, topk * K_BLK]
    const float* __restrict__ W,            // [seq, H]
    const int32_t* __restrict__ CuKS,       // [seq]
    const int32_t* __restrict__ CuKE,       // [seq]
    int seq_kv,
    int topk,
    int K_CHUNKS,
    int outer_total
) {
    int seq_i = blockIdx.x;
    int outer = blockIdx.y;
    int tid   = threadIdx.x;

    __shared__ __nv_fp8_e4m3 Q_smem[H][D];
    __shared__ __nv_fp8_e4m3 K_smem[GEMM_TILE][D];
    __shared__ float W_smem[H];
    __shared__ float S_smem[GEMM_TILE][H];
    __shared__ int32_t topk_block_ids[GROUP];
    __shared__ int32_t k_rows_smem[GEMM_TILE];

    const int ks_min = CuKS[seq_i];
    const int ke_max = CuKE[seq_i];

    // Load Q [H, D] cooperatively (4 warps, 128 threads, H*D = 8192 elements; 64 per thread)
    constexpr int Q_PER_THREAD = (H * D) / THREADS;   // 64
    #pragma unroll
    for (int p = 0; p < Q_PER_THREAD; ++p) {
        int idx = p * THREADS + tid;
        int h = idx / D;
        int d = idx % D;
        Q_smem[h][d] = Q[seq_i * H * D + h * D + d];
    }
    // Load W [H]
    if (tid < H) {
        W_smem[tid] = W[seq_i * H + tid];
    }
    __syncthreads();

    for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
        int chunk_idx = outer * K_CHUNKS + k_iter;
        int n_i_start = chunk_idx * GROUP;

        // Load G topk_block_ids
        if (tid < GROUP) {
            int g_pos = n_i_start + tid;
            topk_block_ids[tid] = (g_pos < topk)
                ? TopK[seq_i * topk + g_pos]
                : -1;
        }
        __syncthreads();

        // Build k_rows_smem[GEMM_TILE]: rows[g*K + b] = topk_id[g]*K + b
        // 4 warps cooperative: each warp does 16 rows.
        #pragma unroll
        for (int p = 0; p < GEMM_TILE / THREADS + 1; ++p) {
            int idx = p * THREADS + tid;
            if (idx < GEMM_TILE) {
                int g = idx / K_BLK;
                int b = idx % K_BLK;
                int tid_val = topk_block_ids[g];
                k_rows_smem[idx] = tid_val * K_BLK + b;
            }
        }
        __syncthreads();

        // Gather K[k_row, :] into K_smem[GEMM_TILE, D]
        // 128 threads, GEMM_TILE * D = 64*128 = 8192 elements; 64 per thread.
        constexpr int K_PER_THREAD = (GEMM_TILE * D) / THREADS;  // 64
        #pragma unroll
        for (int p = 0; p < K_PER_THREAD; ++p) {
            int idx = p * THREADS + tid;
            int m = idx / D;
            int d = idx % D;
            int row = k_rows_smem[m];
            int safe = (row >= 0 && row < seq_kv) ? row : 0;
            K_smem[m][d] = K[safe * D + d];
        }
        __syncthreads();

        // GEMM (naive scalar fp8 for v0)
        gemm_naive(K_smem, Q_smem, S_smem, tid);
        __syncthreads();

        // Post-process: relu(s * ks) * w, sum over H → logits[GEMM_TILE]
        // Each thread handles GEMM_TILE/THREADS rows = 0.5 → use first 64 threads
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
            // Mask + store
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

void block_sparse_mqa_sm90_v0_launcher(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& k_scale,
    const at::Tensor& topk_idx,    // i32 [seq, topk]
    at::Tensor& logits,
    const at::Tensor& weights,
    const at::Tensor& cu_ks,
    const at::Tensor& cu_ke
) {
    int seq    = q.size(0);
    int seq_kv = k.size(0);
    int topk   = topk_idx.size(-1);

    int num_chunks = (topk + GROUP - 1) / GROUP;
    int K_CHUNKS = 32;
    int outer = (num_chunks + K_CHUNKS - 1) / K_CHUNKS;

    dim3 grid(seq, outer);
    dim3 block(THREADS);

    block_sparse_mqa_sm90_v0<<<grid, block>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr()),
        k_scale.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_ks.data_ptr<int32_t>(),
        cu_ke.data_ptr<int32_t>(),
        seq_kv,
        topk,
        K_CHUNKS,
        outer
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_mqa_sm90_v0", &block_sparse_mqa_sm90_v0_launcher,
          "Stage 4 sparse MQA, sm_90a, v0 (no TMA/wgmma)");
}
