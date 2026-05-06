// Stage 4 sparse MQA — sm_90a — v6: TMA + 2-warpgroup design.
//
// Design (DG-style):
//   - 9 warps: 8 math (= 2 warpgroups, 256 threads) + 1 TMA producer (32 threads).
//   - BLOCK_KV = 128 K rows per chunk (= 16 K_BLKs of 8). Each warpgroup handles
//     64 K rows (its half of K_smem); producer issues 16 TMAs in parallel.
//   - K_CHUNKS = topk / 16 = 32 (was 64 with BLOCK_KV=64) — halves per-chunk overhead.
//   - KS via cp.async.cg (16-byte vec, 32 lanes loading 128 floats per stage).
//   - mbarriers (full / empty) per stage; math waits full, arrives empty.
//   - 4-stage K pipeline (dynamic smem 64KB).
//   - DG-style register reduce + shfl_xor (no S_smem spill).
//   - `bar.sync 1, 256` (math-scoped) for KS visibility — __syncthreads
//     would deadlock waiting on still-looping producer warp.

#include <cutlass/detail/helper_macros.hpp>
#undef CUTLASS_DEVICE
#define CUTLASS_DEVICE __forceinline__ __device__ __host__
#include <cutlass/arch/synclog.hpp>
#undef CUTLASS_DEVICE
#define CUTLASS_DEVICE __forceinline__ __device__

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

#include <cutlass/arch/barrier.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <deep_gemm/common/sm90_utils.cuh>

namespace {

constexpr int H = 64;
constexpr int D = 128;
constexpr int K_BLK = 8;
constexpr int GROUP = 8;
constexpr int GEMM_TILE = K_BLK * GROUP;       // 64
constexpr int kNumMathThreads = 128;
constexpr int kNumTMAThreads  = 32;
constexpr int kNumThreads     = kNumMathThreads + kNumTMAThreads;  // 160

constexpr int kNumKVStages = 4;

using WGMMA = deep_gemm::sm90::FP8MMASelector<64>::type;
constexpr int kNumAccum = WGMMA::kNumAccum;     // 32
using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr int K_STAGE_BYTES  = GEMM_TILE * D;        // 8192

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

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_copy_1d(
    const void* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr, uint32_t crd_0
) {
    constexpr uint64_t cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    cute::SM90_TMA_LOAD_1D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0);
}

__device__ __forceinline__ void tma_copy_2d_last(
    const void* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
    uint32_t crd_0, uint32_t crd_1
) {
    constexpr uint64_t cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_LAST);
    cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
}

extern "C" __global__ __launch_bounds__(kNumThreads, 1) void block_sparse_mqa_sm90_v6(
    const __nv_fp8_e4m3* __restrict__ Q,
    const float* __restrict__ KS,
    const int32_t* __restrict__ TopK,
    float* __restrict__ Logits,
    const float* __restrict__ W,
    const int32_t* __restrict__ CuKS,
    const int32_t* __restrict__ CuKE,
    int seq_kv,
    int topk,
    int K_CHUNKS,
    const __grid_constant__ CUtensorMap tensor_map_k
) {
    using namespace deep_gemm::sm90;

    int seq_i = blockIdx.x;
    int outer = blockIdx.y;
    int tid   = threadIdx.x;
    int warp  = tid / 32;
    int lane  = tid % 32;

    bool is_tma     = tid >= kNumMathThreads;
    bool is_tma_lead = is_tma && cute::elect_one_sync();

    if (tid == 0) {
        cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&tensor_map_k));
    }

    extern __shared__ __align__(1024) uint8_t K_smem_dyn[];
    auto K_smem_buf = reinterpret_cast<uint8_t (*)[K_STAGE_BYTES]>(K_smem_dyn);
    __shared__ alignas(1024) uint8_t Q_smem[H * D];
    __shared__ alignas(16) float W_smem[H];
    __shared__ alignas(128) float KS_smem_buf[kNumKVStages][GEMM_TILE];
    __shared__ alignas(8) uint64_t full_kv_bars[kNumKVStages];
    __shared__ alignas(8) uint64_t empty_kv_bars[kNumKVStages];

    const int ks_min = CuKS[seq_i];
    const int ke_max = CuKE[seq_i];
    const int32_t* __restrict__ topk_seq = TopK + seq_i * topk;

    // Init mbarriers (1 thread). full needs 1 arrive (from arrive_and_expect_tx).
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < kNumKVStages; ++i) {
            Barrier::init(&full_kv_bars[i], 1);
            Barrier::init(&empty_kv_bars[i], kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }

    // ---- Q + W: load once at start, all threads cooperate via cp.async ----
    constexpr int CHUNKS_PER_ROW = D / 16;           // 8
    constexpr int Q_VECS = H * CHUNKS_PER_ROW;       // 512
    {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            int vidx = p * kNumThreads + tid;
            if (vidx < Q_VECS) {
                int h = vidx / CHUNKS_PER_ROW;
                int kc = vidx % CHUNKS_PER_ROW;
                const uint4* src = reinterpret_cast<const uint4*>(Q + seq_i * H * D + h * D + kc * 16);
                cp_async_16(Q_smem + b128_swiz_offset16(h, kc), reinterpret_cast<const void*>(src));
            }
        }
        if (tid < H) W_smem[tid] = W[seq_i * H + tid];
        cp_async_commit();
        cp_async_wait_all();
    }
    fence_proxy_async();
    __syncthreads();

    if (is_tma) {
        // ============ TMA PRODUCER WARP (K via TMA) ============
        int prod_lane = tid - kNumMathThreads;  // 0..31

        for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
            int chunk_idx = outer * K_CHUNKS + k_iter;
            int stage = k_iter % kNumKVStages;
            int phase = (k_iter / kNumKVStages) & 1;

            if (k_iter >= kNumKVStages && is_tma_lead) {
                Barrier::wait(&empty_kv_bars[stage], phase ^ 1);
            }
            __syncwarp();

            if (prod_lane < GROUP) {
                int g = prod_lane;
                int g_pos = chunk_idx * GROUP + g;
                int block_id = (g_pos < topk) ? topk_seq[g_pos] : -1;
                int row = (block_id >= 0 && block_id * K_BLK + K_BLK <= seq_kv)
                          ? block_id * K_BLK : 0;
                tma_copy_2d_last(&tensor_map_k, &full_kv_bars[stage],
                         K_smem_buf[stage] + g * K_BLK * D,
                         /*crd_0=*/0, /*crd_1=*/row);
            }
            __syncwarp();

            if (is_tma_lead) {
                Barrier::arrive_and_expect_tx(&full_kv_bars[stage], K_STAGE_BYTES);
            }
        }
        return;
    }

    // ============ MATH WARPGROUP ============

    // Pre-load per-thread weights into registers (DG-style).
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

    int warp_offset = warp * 16;
    int v_0_offset = warp_offset + (lane / 4);
    int v_1_offset = warp_offset + (lane / 4) + 8;

    // Vectorized KS load: 16 lanes × 16-byte cp.async.cg (4 floats each).
    auto issue_ks = [&](int chunk, int stage_) {
        if (tid < GEMM_TILE / 4) {
            int g = tid / 2;
            int sub = (tid & 1) * 4;
            int g_pos = chunk * GROUP + g;
            int block_id = (g_pos < topk) ? topk_seq[g_pos] : -1;
            int row = (block_id >= 0 && block_id * K_BLK + K_BLK <= seq_kv)
                      ? block_id * K_BLK + sub : 0;
            cp_async_16(reinterpret_cast<uint8_t*>(KS_smem_buf[stage_] + g * K_BLK + sub),
                        reinterpret_cast<const void*>(KS + row));
        }
        cp_async_commit();
    };

    issue_ks(outer * K_CHUNKS, 0);

    bool is_lane_writer = (lane & 3) == 0;

    for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
        int chunk_idx = outer * K_CHUNKS + k_iter;
        int n_i_start = chunk_idx * GROUP;
        int stage = k_iter % kNumKVStages;
        int phase = (k_iter / kNumKVStages) & 1;

        Barrier::wait(&full_kv_bars[stage], phase);

        float accum[kNumAccum];
        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_arrive();
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            auto desc_a = make_smem_desc(K_smem_buf[stage] + kt * 32, 1, 0, 1024);
            auto desc_b = make_smem_desc(Q_smem + kt * 32, 1, 0, 1024);
            WGMMA::wgmma(desc_a, desc_b, accum, /*scale_d=*/(kt != 0));
        }
        warpgroup_commit_batch();

        if (k_iter + 1 < K_CHUNKS) {
            int next_stage = (k_iter + 1) % kNumKVStages;
            issue_ks(outer * K_CHUNKS + k_iter + 1, next_stage);
        }
        asm volatile("cp.async.wait_group 1;\n");
        asm volatile("bar.sync 1, 128;\n");

        #pragma unroll
        for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
        warpgroup_wait<0>();

        // ---- Per-row scale + register reduce ----
        int row_lo = k_row_for(chunk_idx, v_0_offset);
        int row_hi = k_row_for(chunk_idx, v_1_offset);
        float ks_lo = (row_lo >= 0 && row_lo < seq_kv) ? KS_smem_buf[stage][v_0_offset] : 0.f;
        float ks_hi = (row_hi >= 0 && row_hi < seq_kv) ? KS_smem_buf[stage][v_1_offset] : 0.f;

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

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int offset = 1u << j;
            sum_lo += __shfl_xor_sync(0xffffffffu, sum_lo, offset);
            sum_hi += __shfl_xor_sync(0xffffffffu, sum_hi, offset);
        }

        if (is_lane_writer) {
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

        // Release stage to producer.
        Barrier::arrive(&empty_kv_bars[stage]);
    }
}

}  // namespace

// ----- Host-side TMA descriptor encoding -----

static void encode_tma_k(CUtensorMap* desc, const void* k_ptr, int seq_kv) {
    cuuint64_t globalDim[2]      = {(cuuint64_t)D, (cuuint64_t)seq_kv};
    cuuint64_t globalStrides[1]  = {(cuuint64_t)D};
    cuuint32_t boxDim[2]         = {(cuuint32_t)D, (cuuint32_t)K_BLK};
    cuuint32_t elementStrides[2] = {1, 1};
    auto result = cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, const_cast<void*>(k_ptr),
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "TMA K encode failed: ", (int)result);
}

// KS as 1D layout: TMA fetches K_BLK contiguous floats per call.
static void encode_tma_ks(CUtensorMap* desc, const void* ks_ptr, int seq_kv) {
    cuuint64_t globalDim[1]      = {(cuuint64_t)seq_kv};
    cuuint64_t globalStrides[0]  = {};
    cuuint32_t boxDim[1]         = {(cuuint32_t)K_BLK};
    cuuint32_t elementStrides[1] = {1};
    auto result = cuTensorMapEncodeTiled(
        desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        1, const_cast<void*>(ks_ptr),
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "TMA KS encode failed: ", (int)result);
}

void block_sparse_mqa_sm90_v6_launcher(
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

    CUtensorMap tma_k_host{};
    encode_tma_k(&tma_k_host, k.data_ptr(), seq_kv);

    constexpr int kDynSmem = kNumKVStages * K_STAGE_BYTES;
    cudaFuncSetAttribute(block_sparse_mqa_sm90_v6,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmem);
    block_sparse_mqa_sm90_v6<<<dim3(seq, outer), kNumThreads, kDynSmem,
                               at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        k_scale.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_ks.data_ptr<int32_t>(),
        cu_ke.data_ptr<int32_t>(),
        seq_kv, topk, K_CHUNKS,
        tma_k_host
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_mqa_sm90_v6", &block_sparse_mqa_sm90_v6_launcher,
          "Stage 4 sparse MQA, sm_90a, v6 (TMA + producer/consumer)");
}
