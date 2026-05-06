// Stage 4 sparse MQA — sm_90a — v5: v4 design + persistent CTA.
//
// Same single-warpgroup TMA + cp.async design as v4. The only change:
//   - Launch num_sms * occupancy CTAs; each loops over assigned queries.
//   - Mbarriers init ONCE; phase preserves across queries (K_CHUNKS=64, stages=4
//     → 16 fires per stage per query, even, so phase returns to 0).

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
constexpr int kNumAccum = WGMMA::kNumAccum;
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

extern "C" __global__ __launch_bounds__(kNumThreads, 1) void block_sparse_mqa_sm90_v5(
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
    int num_seq,
    const CUtensorMap* tensor_map_k_ptr
) {
    const CUtensorMap& tensor_map_k = *tensor_map_k_ptr;
    using namespace deep_gemm::sm90;

    int tid   = threadIdx.x;
    int warp  = tid / 32;
    int lane  = tid % 32;

    bool is_tma     = tid >= kNumMathThreads;
    bool is_tma_lead = is_tma && cute::elect_one_sync();

    extern __shared__ __align__(1024) uint8_t K_smem_dyn[];
    auto K_smem_buf = reinterpret_cast<uint8_t (*)[K_STAGE_BYTES]>(K_smem_dyn);
    __shared__ alignas(1024) uint8_t Q_smem[H * D];
    __shared__ alignas(16) float W_smem[H];
    __shared__ alignas(16) float KS_smem_buf[kNumKVStages][GEMM_TILE];
    __shared__ alignas(8) uint64_t full_kv_bars[kNumKVStages];
    __shared__ alignas(8) uint64_t empty_kv_bars[kNumKVStages];

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < kNumKVStages; ++i) {
            Barrier::init(&full_kv_bars[i], 1);
            Barrier::init(&empty_kv_bars[i], kNumMathThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Global k_iter counter — mbarrier phase tracking across queries.
    int g_kiter = 0;

    // Persistent loop
    for (int seq_i = blockIdx.x; seq_i < num_seq; seq_i += gridDim.x) {
        const int ks_min = CuKS[seq_i];
        const int ke_max = CuKE[seq_i];
        const int32_t* __restrict__ topk_seq = TopK + seq_i * topk;

        // ---- Q + W load ----
        constexpr int CHUNKS_PER_ROW = D / 16;
        constexpr int Q_VECS = H * CHUNKS_PER_ROW;
        {
            #pragma unroll
            for (int p = 0; p < (Q_VECS + kNumThreads - 1) / kNumThreads; ++p) {
                int vidx = p * kNumThreads + tid;
                if (vidx < Q_VECS) {
                    int h = vidx / CHUNKS_PER_ROW;
                    int kc = vidx % CHUNKS_PER_ROW;
                    const uint4* src = reinterpret_cast<const uint4*>(
                        Q + seq_i * H * D + h * D + kc * 16);
                    cp_async_16(Q_smem + b128_swiz_offset16(h, kc),
                                reinterpret_cast<const void*>(src));
                }
            }
            if (tid < H) W_smem[tid] = W[seq_i * H + tid];
            cp_async_commit();
            cp_async_wait_all();
        }
        fence_proxy_async();
        __syncthreads();

        if (is_tma) {
            int prod_lane = tid - kNumMathThreads;
            for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
                int chunk_idx = k_iter;
                int gk = g_kiter + k_iter;
                int stage = gk % kNumKVStages;
                int phase = (gk / kNumKVStages) & 1;

                if (gk >= kNumKVStages && is_tma_lead) {
                    Barrier::wait(&empty_kv_bars[stage], phase ^ 1);
                }
                __syncwarp();

                if (prod_lane < GROUP) {
                    int g = prod_lane;
                    int g_pos = chunk_idx * GROUP + g;
                    int block_id = (g_pos < topk) ? topk_seq[g_pos] : -1;
                    int row = (block_id >= 0 && block_id * K_BLK + K_BLK <= seq_kv)
                              ? block_id * K_BLK : 0;
                    tma_copy(&tensor_map_k, &full_kv_bars[stage],
                             K_smem_buf[stage] + g * K_BLK * D,
                             /*crd_0=*/0, /*crd_1=*/row);
                }
                __syncwarp();

                if (is_tma_lead) {
                    Barrier::arrive_and_expect_tx(&full_kv_bars[stage], K_STAGE_BYTES);
                }
            }
        } else {
            float weights_reg[H / 4];
            #pragma unroll
            for (int j = 0; j < H / 4; ++j) {
                weights_reg[j] = W_smem[(j / 2) * 8 + (j & 1) + (lane % 4) * 2];
            }

            int warp_offset = warp * 16;
            int v_0_offset = warp_offset + (lane / 4);
            int v_1_offset = warp_offset + (lane / 4) + 8;

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

            int boot_stage = g_kiter % kNumKVStages;
            issue_ks(0, boot_stage);
            bool is_lane_writer = (lane & 3) == 0;

            for (int k_iter = 0; k_iter < K_CHUNKS; ++k_iter) {
                int chunk_idx = k_iter;
                int n_i_start = chunk_idx * GROUP;
                int gk = g_kiter + k_iter;
                int stage = gk % kNumKVStages;
                int phase = (gk / kNumKVStages) & 1;

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
                    int next_stage = (gk + 1) % kNumKVStages;
                    issue_ks(k_iter + 1, next_stage);
                }
                asm volatile("cp.async.wait_group 1;\n");
                asm volatile("bar.sync 1, 128;\n");

                #pragma unroll
                for (int i = 0; i < kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                int row_lo, row_hi;
                {
                    int g_lo = v_0_offset / K_BLK, b_lo = v_0_offset % K_BLK;
                    int g_hi = v_1_offset / K_BLK, b_hi = v_1_offset % K_BLK;
                    int gp_lo = chunk_idx * GROUP + g_lo;
                    int gp_hi = chunk_idx * GROUP + g_hi;
                    int bid_lo = (gp_lo < topk) ? topk_seq[gp_lo] : -1;
                    int bid_hi = (gp_hi < topk) ? topk_seq[gp_hi] : -1;
                    row_lo = bid_lo * K_BLK + b_lo;
                    row_hi = bid_hi * K_BLK + b_hi;
                }
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

                Barrier::arrive(&empty_kv_bars[stage]);
            }
        }
        // Sync before next query (smem reused).
        __syncthreads();
        g_kiter += K_CHUNKS;
    }
}

}  // namespace

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

void block_sparse_mqa_sm90_v5_launcher(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& k_scale, const at::Tensor& topk_idx,
    at::Tensor& logits, const at::Tensor& weights,
    const at::Tensor& cu_ks, const at::Tensor& cu_ke
) {
    int seq    = q.size(0);
    int seq_kv = k.size(0);
    int topk   = topk_idx.size(-1);
    int K_CHUNKS = (topk + GROUP - 1) / GROUP;

    CUtensorMap tma_k_host{};
    encode_tma_k(&tma_k_host, k.data_ptr(), seq_kv);

    auto desc_buf = torch::empty(
        {(int64_t)sizeof(CUtensorMap)},
        torch::TensorOptions().dtype(torch::kUInt8).device(q.device())
    );
    auto* d_tma_k = reinterpret_cast<CUtensorMap*>(desc_buf.data_ptr());
    cudaMemcpyAsync(d_tma_k, &tma_k_host, sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream());

    constexpr int kDynSmem = kNumKVStages * K_STAGE_BYTES;
    cudaFuncSetAttribute(block_sparse_mqa_sm90_v5,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kDynSmem);

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    // Persistent: launch num_sms * 5 (matches v4 occupancy = 5 CTA/SM)
    int grid = prop.multiProcessorCount * 5;
    if (grid > seq) grid = seq;

    block_sparse_mqa_sm90_v5<<<dim3(grid), kNumThreads, kDynSmem,
                               at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        k_scale.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        weights.data_ptr<float>(),
        cu_ks.data_ptr<int32_t>(),
        cu_ke.data_ptr<int32_t>(),
        seq_kv, topk, K_CHUNKS, seq,
        d_tma_k
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_mqa_sm90_v5", &block_sparse_mqa_sm90_v5_launcher,
          "Stage 4 sparse MQA, sm_90a, v5 (v4 + persistent CTA)");
}
