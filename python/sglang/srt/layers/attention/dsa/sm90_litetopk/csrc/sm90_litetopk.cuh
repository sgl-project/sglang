// SPDX-License-Identifier: MIT
// Hopper scoring structure and FP8 WGMMA reduction derived from DeepGEMM
// (deepseek-ai/DeepGEMM) and the local sm90_dsa_marsco prototype.
// HOT seed plus exact-once suffix evidence; monotone online bucket gate.
#pragma once
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

#include <type_traits>
#include <cuda/atomic>
#ifndef LT_INTERIOR
#define LT_INTERIOR 0
#endif
#ifndef LT_UNROLL
#define LT_UNROLL 0
#endif
#ifndef LT_UNIFORM_WARP
#define LT_UNIFORM_WARP 0
#endif
#ifndef LT_COMBINED_EMITTER
#define LT_COMBINED_EMITTER 0
#endif
#ifndef LT_OVERLAP_EMIT
#define LT_OVERLAP_EMIT 0
#endif
#ifndef LT_DOUBLE_MMA
#define LT_DOUBLE_MMA 0
#endif
#ifndef LT_WARP_BUFFER
#define LT_WARP_BUFFER 0
#endif
#ifndef LT_MERGE_REDUCE
#define LT_MERGE_REDUCE 0
#endif
#ifndef LT_PRODUCER_REGS
#define LT_PRODUCER_REGS 32
#endif
#ifndef LT_ONLINE
#define LT_ONLINE 1
#endif
#ifndef LT_REFRESH
#define LT_REFRESH 8
#endif
#ifndef LT_PENDING
#define LT_PENDING 128
#endif
#ifndef LT_GLOBAL_GATE
#define LT_GLOBAL_GATE 1
#endif
#define LT_PRAGMA_IMPL(x) _Pragma(#x)
#define LT_PRAGMA(x) LT_PRAGMA_IMPL(x)

namespace litetopk_sm90 {
using namespace deep_gemm;
template<class T>
__device__ __forceinline__ T shared_load(T* p) {
    return cuda::atomic_ref<T, cuda::thread_scope_block>(*p).load(cuda::memory_order_relaxed);
}
__device__ __forceinline__ unsigned global_load(float* p) {
    return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(
        *reinterpret_cast<unsigned*>(p)).load(cuda::memory_order_relaxed);
}

template<int BQ, int BK, int STAGES, int MATH_THREADS, int MATH_REGS, bool DENSE, bool BUCKET = false>
CUTLASS_GLOBAL __launch_bounds__(MATH_THREADS + 128, 1)
void score(const uint32_t Q, const uint32_t S,
           const int32_t* __restrict__ starts,
           const int32_t* __restrict__ ends,
           float* __restrict__ threshold,
           const uint32_t splits,
           float* __restrict__ values,
           int32_t* __restrict__ indices,
           int32_t* __restrict__ counts,
           const uint32_t capacity,
           const int32_t* __restrict__ active,
           const uint32_t range_start, const uint32_t range_end,
           const float* origin, const float* inv_delta, const int* seed_hist,
           const __grid_constant__ cute::TmaDescriptor tm_q,
           const __grid_constant__ cute::TmaDescriptor tm_k,
           const __grid_constant__ cute::TmaDescriptor tm_s,
           const __grid_constant__ cute::TmaDescriptor tm_w) {
    constexpr int H = 32, D = 128;
    using MMA = typename mma::sm90::FP8MMASelector<BQ * H>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr int WG_PER_KV = BK / MMA::M;
    constexpr int GROUPS = MATH_THREADS / 128 / WG_PER_KV;
    constexpr int GROUP_THREADS = 128 * WG_PER_KV;
    static_assert(BK % 64 == 0 && MATH_THREADS % GROUP_THREADS == 0);
    constexpr int Q_BYTES = BQ * H * D;
    constexpr int K_BYTES = BK * D;
    constexpr int W_BYTES = BQ * H * 4;
    constexpr int S_BYTES = BK * 4;
    extern __shared__ __align__(1024) uint8_t memory[];
    auto* sq = reinterpret_cast<__nv_fp8_e4m3*>(memory);
    auto sk = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(memory + Q_BYTES + i * K_BYTES);
    });
    auto* sw = reinterpret_cast<float*>(memory + Q_BYTES + STAGES * K_BYTES);
    auto ss = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(memory + Q_BYTES + STAGES * K_BYTES + W_BYTES + i * S_BYTES);
    });
    auto* qfull = reinterpret_cast<Barrier*>(ss[STAGES]);
    auto kfull = utils::PatternVisitor([&](const uint32_t& i) { return qfull + 1 + i; });
    auto kempty = utils::PatternVisitor([&](const uint32_t& i) { return qfull + 1 + STAGES + i; });
    auto* shared_queue = reinterpret_cast<uint2*>(kempty[STAGES]);

    constexpr bool ONLINE = BUCKET && LT_ONLINE > 0;
    auto* hist = reinterpret_cast<int*>(shared_queue + MATH_THREADS / 32 * BQ * 64);
    auto* gates = reinterpret_cast<unsigned*>(hist + BQ * 256);
    auto* pending = reinterpret_cast<int*>(gates + BQ);
    auto* done = pending + BQ;
    auto* seq = reinterpret_cast<unsigned*>(done + 1);
    auto* snapshots = seq + MATH_THREADS / 32 * BQ;
    auto* shadows = snapshots + MATH_THREADS / 32 * BQ * 32;

    uint32_t row_start[BQ], row_end[BQ];
    uint32_t begin = S, end = 0, full_begin = 0, full_end = S;
    #pragma unroll
    for (int i = 0; i < BQ; ++i) {
        const uint32_t row = blockIdx.x * BQ + i;
        const bool enabled = row < Q && (active == nullptr || active[row] != 0);
        row_start[i] = enabled ? min(max(static_cast<uint32_t>(starts[row]), range_start), S) : S;
        row_end[i] = enabled ? min(min(static_cast<uint32_t>(ends[row]), range_end), S) : 0;
        begin = min(begin, row_start[i]);
        end = max(end, row_end[i]);
        full_begin = max(full_begin, row_start[i]);
        full_end = min(full_end, row_end[i]);
    }
    const uint32_t blocks_per_split = math::ceil_div(math::ceil_div(range_end - range_start, uint32_t(BK)), splits);
    begin = max(begin / 4 * 4, range_start + blockIdx.y * blocks_per_split * BK);
    end = min(end, range_start + (blockIdx.y + 1) * blocks_per_split * BK);
    const uint32_t nblocks = end > begin ? math::ceil_div(end - begin, uint32_t(BK)) : 0;
    if (nblocks == 0) return;

    const bool producer = threadIdx.x >= MATH_THREADS;
    const bool load_warp = threadIdx.x / 32 == MATH_THREADS / 32;
    if (load_warp && cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tm_q);
        cute::prefetch_tma_descriptor(&tm_k);
        cute::prefetch_tma_descriptor(&tm_s);
        cute::prefetch_tma_descriptor(&tm_w);
        qfull->init(1);
        #pragma unroll
        for (int i = 0; i < STAGES; ++i) {
            kfull[i]->init(1);
            kempty[i]->init(GROUP_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    }
    if constexpr (ONLINE) {
        for (int j = threadIdx.x; j < BQ * 256; j += blockDim.x) {
            const int row = blockIdx.x * BQ + j / 256;
            hist[j] = row < Q ? seed_hist[row * 256 + j % 256] : 0;
        }
        if (threadIdx.x < BQ) {
            const int row = blockIdx.x * BQ + threadIdx.x;
            // Concurrent CTAs can only lower this positive float-bit edge.
            gates[threadIdx.x] = row < Q
                ? global_load(threshold + row) : 0u;
            pending[threadIdx.x] = 0;
        }
        if (threadIdx.x == 0) *done = 0;
        if constexpr (LT_ONLINE == 3) {
            if (threadIdx.x < MATH_THREADS / 32 * BQ) {
                seq[threadIdx.x] = 0;
                shadows[threadIdx.x] = 0;
            }
        }
    }
    __syncthreads();

    // One warp scans a row. Independent, monotonically increasing bins are
    // safe under concurrent updates: a mixed-time snapshot is still a subset.
    const auto refresh_row = [&](int i) {
        const int lane = threadIdx.x % 32;
        const int row = blockIdx.x * BQ + i;
        if (row >= Q) return;
        unsigned gate = 0;
        if (lane == 0) gate = shared_load(gates + i);
        gate = __shfl_sync(0xffffffffu, gate, 0);
        const int limit = min(255, int(__uint_as_float(gate)) - 1);
        int carry = 0;
        for (int base = 0; base < limit; base += 32) {
            const int b = base + lane;
            const int v = b < limit ? shared_load(hist + i * 256 + b) : 0;
            const int total = __reduce_add_sync(0xffffffffu, v);
            if (carry + total < 2048) { carry += total; continue; }
            int prefix = v;
            #pragma unroll
            for (int d = 1; d < 32; d *= 2) {
                const int x = __shfl_up_sync(0xffffffffu, prefix, d);
                if (lane >= d) prefix += x;
            }
            const unsigned mask = __ballot_sync(0xffffffffu, carry + prefix >= 2048);
            const unsigned edge = __float_as_uint(float(base + __ffs(mask)));
            if (lane == 0) {
                atomicMin(gates + i, edge);
                if constexpr (LT_GLOBAL_GATE)
                    atomicMin(reinterpret_cast<unsigned*>(threshold + row), edge);
            }
            break;
        }
    };

    if (producer) {
        cutlass::arch::warpgroup_reg_dealloc<LT_PRODUCER_REGS>();
        if constexpr (ONLINE && (LT_ONLINE == 1 || LT_ONLINE == 3)) {
            const int spare = int(threadIdx.x / 32) - MATH_THREADS / 32 - 1;
            if (spare >= 0 && spare < 2) {

                while (true) {
                    int finished = 0;
                    if (threadIdx.x % 32 == 0) finished = shared_load(done);
                    if (__shfl_sync(0xffffffffu, finished, 0) == MATH_THREADS / 32) break;
                    for (int i = spare; i < BQ; i += 2) {
                        if constexpr (LT_ONLINE == 3) {
                            const int lane = threadIdx.x % 32;
                            const int row = blockIdx.x * BQ + i;
                            if (row >= Q) continue;
                            const float inv = inv_delta[row], bias = -origin[row] * inv;
                            int fresh = 0;
                            for (int w = 0; w < MATH_THREADS / 32; ++w) {
                                const int pair = w * BQ + i;
                                unsigned before = 0;
                                if (lane == 0) before = cuda::atomic_ref<unsigned,
                                    cuda::thread_scope_block>(seq[pair]).load(cuda::memory_order_acquire);
                                before = __shfl_sync(0xffffffffu, before, 0);
                                if (before == 0 || (before & 1) || before == shadows[pair]) continue;
                                const unsigned bits = shared_load(snapshots + pair * 32 + lane);
                                __threadfence_block();
                                __syncwarp();
                                unsigned after = 0;
                                if (lane == 0) after = cuda::atomic_ref<unsigned,
                                    cuda::thread_scope_block>(seq[pair]).load(cuda::memory_order_acquire);
                                after = __shfl_sync(0xffffffffu, after, 0);
                                if (after != before) continue;
                                if (lane == 0) shadows[pair] = before;
                                __syncwarp();
                                const float score = __uint_as_float(bits);
                                const bool valid = isfinite(score);
                                if (valid) {
                                    const float b = fmaf(-score, inv, bias);
                                    const int bin = b < 0 ? 0 : (b < 255 ? int(b) : 255);
                                    atomicAdd(hist + i * 256 + bin, 1);
                                }
                                fresh += __popc(__ballot_sync(0xffffffffu, valid));
                            }
                            if (lane == 0) pending[i] += fresh;
                            __syncwarp();
                        }
                        int n = 0;
                        if (threadIdx.x % 32 == 0) n = shared_load(pending + i);
                        n = __shfl_sync(0xffffffffu, n, 0);
                        if (n >= LT_PENDING) {
                            if (threadIdx.x % 32 == 0) atomicExch(pending + i, 0);
                            refresh_row(i);
                        }
                    }
                    __nanosleep(1024);
                }
            }
        }
        if (load_warp) {
          if (cute::elect_one_sync()) {
            tma::copy<D, BQ * H, D>(&tm_q, qfull, sq, 0, blockIdx.x * BQ * H);
            tma::copy<H, BQ, 0>(&tm_w, qfull, sw, 0, blockIdx.x * BQ);
            qfull->arrive_and_expect_tx(Q_BYTES + W_BYTES);
          }
            #if LT_UNROLL > 0
            LT_PRAGMA(unroll LT_UNROLL)
            #endif
            for (uint32_t b = 0; b < nblocks; ++b) {
                const uint32_t st = b % STAGES, phase = (b / STAGES) & 1;
              if (cute::elect_one_sync()) {
                kempty[st]->wait(phase ^ 1);
                tma::copy<D, BK, D>(&tm_k, kfull[st], sk[st], 0, begin + b * BK);
                tma::copy<BK, 1, 0>(&tm_s, kfull[st], ss[st], begin + b * BK, 0);
                kfull[st]->arrive_and_expect_tx(K_BYTES + S_BYTES);
              }
              if constexpr (ONLINE && LT_ONLINE == 2) {
                  if (b % (LT_REFRESH ? LT_REFRESH : 64) == 0) {
                      #pragma unroll
                      for (int i = 0; i < BQ; ++i) refresh_row(i);
                  }
              }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<MATH_REGS>();
        const uint32_t lane = ptx::get_lane_idx();
        #if LT_UNIFORM_WARP
        const uint32_t warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
        #else
        const uint32_t warp = threadIdx.x / 32;
        #endif
        const uint32_t wg = warp / 4;
        const uint32_t group = wg / WG_PER_KV;
        const uint32_t wg_in_group = wg % WG_PER_KV;
        const uint32_t warp_in_group = warp - group * WG_PER_KV * 4;
        const uint32_t off0 = warp_in_group * 16 + lane / 4;
        const uint32_t off1 = off0 + 8;
        float acc[MMA::kNumAccum], weight[BQ][H / 4], gate[BQ];
        float bucket_inv[BQ], bucket_bias[BQ];
        float queued_value[BQ];
        uint32_t queued_index[BQ], queued_count[BQ];
        qfull->wait(0);
        #pragma unroll
        for (int i = 0; i < BQ; ++i) {
            const uint32_t row = blockIdx.x * BQ + i;
            if constexpr (ONLINE) gate[i] = __uint_as_float(shared_load(gates + i));
            else gate[i] = (!DENSE && row < Q) ? threshold[row] : 0;
            queued_count[i] = 0;
            if constexpr (BUCKET) {
                bucket_inv[i] = row < Q ? inv_delta[row] : 0;
                bucket_bias[i] = row < Q ? -origin[row] * bucket_inv[i] : 0;
            }
            #pragma unroll
            for (int j = 0; j < H / 4; ++j)
                weight[i][j] = ptx::ld_shared(sw + i * H + (j / 2) * 8 + (j & 1) + (lane % 4) * 2);
        }
        const auto passes = [&](float v, int i) {
            if constexpr (BUCKET) {
                return __float_as_int(fmaf(-v, bucket_inv[i], bucket_bias[i])) < __float_as_int(gate[i]);
            } else return v >= gate[i];
        };
        const auto flush_queue = [&](int i, uint32_t n) {
            if (n) {
                if constexpr (ONLINE && LT_ONLINE == 3) {
                    const int pair = warp * BQ + i;
                    if (lane == 0) atomicAdd(seq + pair, 1u);
                    __threadfence_block();
                    __syncwarp();
                    const unsigned bits = lane < n
                        ? shared_queue[pair * 64 + lane].x : 0x7fffffffu;
                    cuda::atomic_ref<unsigned, cuda::thread_scope_block>(
                        snapshots[pair * 32 + lane]).store(bits, cuda::memory_order_relaxed);
                    __threadfence_block();
                    __syncwarp();
                    if (lane == 0) atomicAdd(seq + pair, 1u);
                }
                const uint32_t row = blockIdx.x * BQ + i;
                int dest = 0;
                if (lane == 0) dest = atomicAdd(counts + row, n);
                dest = __shfl_sync(0xffffffffu, dest, 0);
                const uint64_t base = static_cast<uint64_t>(row) * capacity;
                if (lane < n && dest + lane < capacity) {
                    if constexpr (LT_WARP_BUFFER == 2) {
                        const uint2 item = shared_queue[(warp * BQ + i) * 64 + lane];
                        if constexpr (ONLINE && LT_ONLINE != 3) {
                            const float b = fmaf(-__uint_as_float(item.x), bucket_inv[i], bucket_bias[i]);
                            const int bin = b < 0 ? 0 : (b < 255 ? int(b) : 255);
                            atomicAdd(hist + i * 256 + bin, 1);
                        }
                        __stcs(values + base + dest + lane, __uint_as_float(item.x));
                        __stcs(indices + base + dest + lane, static_cast<int>(item.y));
                    } else {
                        __stcs(values + base + dest + lane, queued_value[i]);
                        __stcs(indices + base + dest + lane, static_cast<int>(queued_index[i]));
                    }
                }
                if constexpr (ONLINE && LT_ONLINE != 3) {
                    if (lane == 0) atomicAdd(pending + i, n);
                }
                if constexpr (ONLINE && LT_REFRESH == 0) {
                    unsigned bits = 0;
                    if (lane == 0) {
                        if constexpr (LT_GLOBAL_GATE) bits = global_load(threshold + row);
                        else bits = shared_load(gates + i);
                    }
                    gate[i] = __uint_as_float(__shfl_sync(0xffffffffu, bits, 0));
                }
            }
        };
        const auto issue_mma = [&](uint32_t b, float* frag) {
            const uint32_t st = b % STAGES, phase = (b / STAGES) & 1;
            kfull[st]->wait(phase);
            #pragma unroll
            for (int i = 0; i < MMA::kNumAccum; ++i) ptx::warpgroup_fence_operand(frag[i]);
            ptx::warpgroup_arrive();
            #pragma unroll
            for (int k = 0; k < D / MMA::K; ++k) {
                auto a = mma::sm90::make_smem_desc(
                    sk[st] + wg_in_group * MMA::M * D + k * MMA::K,
                    mma::sm90::to_swizzle_cute_type<D>(), 0, D * 8);
                auto q = mma::sm90::make_smem_desc(sq + k * MMA::K,
                    mma::sm90::to_swizzle_cute_type<D>(), 0, D * 8);
                MMA::wgmma(a, q, frag, k);
            }
            ptx::warpgroup_commit_batch();
            #pragma unroll
            for (int i = 0; i < MMA::kNumAccum; ++i) ptx::warpgroup_fence_operand(frag[i]);
        };
        const auto emit_tile_impl = [&](uint32_t b, float* frag, auto full_tag) {
            constexpr bool FULL = decltype(full_tag)::value;
            const uint32_t st = b % STAGES;
            const float scale0 = ptx::ld_shared(ss[st] + off0);
            const float scale1 = ptx::ld_shared(ss[st] + off1);
            kempty[st]->arrive();
            const uint32_t col0 = begin + b * BK + off0;
            const uint32_t col1 = begin + b * BK + off1;
            float reduced0[BQ], reduced1[BQ];
            #pragma unroll
            for (int i = 0; i < BQ; ++i) {
                auto a = frag + i * (H / 2);
                const auto transform = [&](int j) {
                    return fmaxf(a[j], 0) * weight[i][(j / 4) * 2 + (j & 1)];
                };
                float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                #pragma unroll
                for (int j = 1; j < H / 8; ++j) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) sum[k] += transform(j * 4 + k);
                }
                float v0 = (sum[0] + sum[1]) * scale0;
                float v1 = (sum[2] + sum[3]) * scale1;
                if constexpr (LT_MERGE_REDUCE && !DENSE && (LT_COMBINED_EMITTER || LT_WARP_BUFFER)) {
                    // After the first pair reduction, parity lanes can carry
                    // separate keys. The last shuffle then reduces both keys.
                    v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
                    v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
                    float v;
                    if constexpr (LT_MERGE_REDUCE == 2) {
                        asm volatile("{ .reg .pred p; setp.eq.u32 p, %3, 0; selp.f32 %0, %1, %2, p; }"
                                     : "=f"(v) : "f"(v0), "f"(v1), "r"(lane & 1));
                    } else {
                        v = (lane & 1) ? v1 : v0;
                    }
                    v += __shfl_xor_sync(0xffffffffu, v, 2);
                    reduced0[i] = v;
                    reduced1[i] = v;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        v0 += __shfl_xor_sync(0xffffffffu, v0, 1 << j);
                        v1 += __shfl_xor_sync(0xffffffffu, v1, 1 << j);
                    }
                    reduced0[i] = v0;
                    reduced1[i] = v1;
                }
            }
            if constexpr (LT_OVERLAP_EMIT) {
                if (b + GROUPS < nblocks) issue_mma(b + GROUPS, acc);
            }
            #pragma unroll
            for (int i = 0; i < BQ; ++i) {
                const float v0 = reduced0[i], v1 = reduced1[i];
                const uint32_t row = blockIdx.x * BQ + i;
                const uint64_t base = static_cast<uint64_t>(row) * capacity;
                bool valid0 = FULL || (row < Q && row_start[i] <= col0 && col0 < row_end[i]);
                bool valid1 = FULL || (row < Q && row_start[i] <= col1 && col1 < row_end[i]);
                if constexpr (DENSE) {
                    if (valid0 && (lane & 3) == 0) values[base + col0] = v0;
                    if (valid1 && (lane & 3) == 0) values[base + col1] = v1;
                } else if constexpr (LT_WARP_BUFFER == 2) {
                    const bool second = lane & 1;
                    const float v = second ? v1 : v0;
                    const uint32_t col = second ? col1 : col0;
                    const bool valid = second ? valid1 : valid0;
                    const bool p = valid && passes(v, i) && (lane & 2) == 0;
                    const uint32_t mask = __ballot_sync(0xffffffffu, p);
                    const uint32_t n = __popc(mask);
                    if (n) {
                        auto* queue = shared_queue + (warp * BQ + i) * 64;
                        const uint32_t previous = queued_count[i];
                        const uint32_t rank = __popc(mask & ((1u << lane) - 1u));
                        if (p) queue[previous + rank] = make_uint2(__float_as_uint(v), col);
                        __syncwarp();
                        const uint32_t total = previous + n;
                        if (total >= 32) {
                            flush_queue(i, 32);
                            if (lane < total - 32) queue[lane] = queue[lane + 32];
                            __syncwarp();
                            queued_count[i] = total - 32;
                        } else {
                            queued_count[i] = total;
                        }
                    }
                } else if constexpr (LT_WARP_BUFFER == 1) {
                    const bool second = lane & 1;
                    const float v = second ? v1 : v0;
                    const uint32_t col = second ? col1 : col0;
                    const bool valid = second ? valid1 : valid0;
                    const bool p = valid && passes(v, i) && (lane & 2) == 0;
                    const uint32_t mask = __ballot_sync(0xffffffffu, p);
                    const uint32_t n = __popc(mask);
                    if (n) {
                        const uint32_t previous = queued_count[i];
                        const uint32_t src = __fns(mask, 0, static_cast<int>(lane) - previous + 1);
                        const float incoming_value = __shfl_sync(0xffffffffu, v, src);
                        const uint32_t incoming_index = __shfl_sync(0xffffffffu, col, src);
                        if (lane >= previous && lane < previous + n) {
                            queued_value[i] = incoming_value;
                            queued_index[i] = incoming_index;
                        }
                        const uint32_t total = previous + n;
                        if (total >= 32) {
                            flush_queue(i, 32);
                            const uint32_t tail_src = __fns(mask, 0, lane + 32 - previous + 1);
                            const float tail_value = __shfl_sync(0xffffffffu, v, tail_src);
                            const uint32_t tail_index = __shfl_sync(0xffffffffu, col, tail_src);
                            queued_value[i] = tail_value;
                            queued_index[i] = tail_index;
                            queued_count[i] = total - 32;
                        } else {
                            queued_count[i] = total;
                        }
                    }
                } else if constexpr (LT_COMBINED_EMITTER) {
                    // Four lanes hold identical reduced scores. Use two lanes
                    // for the two keys, allowing one ballot for both values.
                    const bool second = lane & 1;
                    const float v = second ? v1 : v0;
                    const uint32_t col = second ? col1 : col0;
                    const bool valid = second ? valid1 : valid0;
                    const bool p = valid && passes(v, i) && (lane & 2) == 0;
                    const uint32_t mask = __ballot_sync(0xffffffffu, p);
                    const int n = __popc(mask);
                    if (n) {
                        int dest = 0;
                        if (lane == 0) dest = atomicAdd(counts + row, n);
                        dest = __shfl_sync(0xffffffffu, dest, 0);
                        const int j = dest + __popc(mask & ((1u << lane) - 1u));
                        if (p && j < capacity) {
                            __stcs(values + base + j, v);
                            __stcs(indices + base + j, static_cast<int32_t>(col));
                        }
                    }
                } else {
                    const bool p0 = valid0 && passes(v0, i) && (lane & 3) == 0;
                    const bool p1 = valid1 && passes(v1, i) && (lane & 3) == 0;
                    const uint32_t m0 = __ballot_sync(0xffffffffu, p0);
                    const uint32_t m1 = __ballot_sync(0xffffffffu, p1);
                    const int n = __popc(m0) + __popc(m1);
                    if (n) {
                        int dest = 0;
                        if (lane == 0) dest = atomicAdd(counts + row, n);
                        dest = __shfl_sync(0xffffffffu, dest, 0);
                        const uint32_t below = (1u << lane) - 1u;
                        if (p0) {
                            const int j = dest + __popc(m0 & below);
                            if (j < capacity) {
                                __stcs(values + base + j, v0);
                                __stcs(indices + base + j, static_cast<int32_t>(col0));
                            }
                        }
                        if (p1) {
                            const int j = dest + __popc(m0) + __popc(m1 & below);
                            if (j < capacity) {
                                __stcs(values + base + j, v1);
                                __stcs(indices + base + j, static_cast<int32_t>(col1));
                            }
                        }
                    }
                }
            }
        };
        const auto emit_tile = [&](uint32_t b, float* frag) {
            if constexpr (ONLINE && LT_REFRESH > 0) {
                if (b % LT_REFRESH == 0) {
                    #pragma unroll
                    for (int i = 0; i < BQ; ++i) {
                        const int row = blockIdx.x * BQ + i;
                        unsigned bits = 0;
                        if (lane == 0 && row < Q) {
                            if constexpr (LT_GLOBAL_GATE)
                                bits = global_load(threshold + row);
                            else bits = shared_load(gates + i);
                        }
                        gate[i] = __uint_as_float(__shfl_sync(0xffffffffu, bits, 0));
                    }
                }
            }
            if constexpr (LT_INTERIOR && !DENSE) {
                const uint32_t start = begin + b * BK;
                if (start >= full_begin && start + BK <= full_end)
                    emit_tile_impl(b, frag, std::true_type{});
                else
                    emit_tile_impl(b, frag, std::false_type{});
            } else {
                emit_tile_impl(b, frag, std::false_type{});
            }
        };
        if constexpr (LT_DOUBLE_MMA) {
            static_assert(!LT_OVERLAP_EMIT && STAGES >= 2 * GROUPS);
            float next[MMA::kNumAccum];
            if (group < nblocks) issue_mma(group, acc);
            uint32_t b = group;
            #if LT_UNROLL > 0
            LT_PRAGMA(unroll LT_UNROLL)
            #endif
            for (; b + GROUPS < nblocks; b += 2 * GROUPS) {
                issue_mma(b + GROUPS, next);
                ptx::warpgroup_wait<1>();
                emit_tile(b, acc);
                if (b + 2 * GROUPS < nblocks) {
                    issue_mma(b + 2 * GROUPS, acc);
                    ptx::warpgroup_wait<1>();
                } else {
                    ptx::warpgroup_wait<0>();
                }
                emit_tile(b + GROUPS, next);
            }
            if (b < nblocks) {
                ptx::warpgroup_wait<0>();
                emit_tile(b, acc);
            }
        } else {
            if constexpr (LT_OVERLAP_EMIT) {
                if (group < nblocks) issue_mma(group, acc);
            }
            #if LT_UNROLL > 0
            LT_PRAGMA(unroll LT_UNROLL)
            #endif
            for (uint32_t b = group; b < nblocks; b += GROUPS) {
                if constexpr (!LT_OVERLAP_EMIT) issue_mma(b, acc);
                ptx::warpgroup_wait<0>();
                emit_tile(b, acc);
            }
        }
        if constexpr (LT_WARP_BUFFER && !DENSE) {
            #pragma unroll
            for (int i = 0; i < BQ; ++i) flush_queue(i, queued_count[i]);
        }
        if constexpr (ONLINE) {
            if (lane == 0) atomicAdd(done, 1);
        }
    }
}
}  // namespace litetopk_sm90
