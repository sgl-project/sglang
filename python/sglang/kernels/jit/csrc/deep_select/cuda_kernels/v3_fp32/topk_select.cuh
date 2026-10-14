/*
TopK select kernel fp32 variant. 

Usage caveats:
  - vocab_size must be < 2^23.
  - Do NOT assume that the kernel will select a prefix index on ties.
*/

#pragma once

#include "topk_select.h"

#include <cutlass/kernel_launch.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_fp32 {

template<typename Config>
class TopkSelectKernelFP32 : public topk_select_common::TopkSelectKernelBase<Config> {
    using Base = topk_select_common::TopkSelectKernelBase<Config>;
public:
    using ValueT = typename Base::ValueT;
    using OutIdxT = typename Base::OutIdxT;
    using TmaParams = typename Base::TmaParams;
    using SharedMemoryPlanBase = typename Base::SharedMemoryPlanBase;
    using EpilogueT = typename Base::EpilogueT;
    using Base::NUM_THREADS;
    using Base::NUM_WARPS;
    using Base::MAX_TOPK;
    using Base::NUM_ELEMS_PER_128b;
    using Base::NUM_UINT32_PER_128b;
    using Base::NUM_ELEMS_PER_ROUND;
    using Base::NUM_ELEMS_PER_THREAD_PER_ROUND;
    using Base::ELEMS_PER_THREAD_PER_ROUND_MASK;
    using Base::NUM_128b_PER_THREAD_PER_ROUND;
    using Base::NUM_ELEMS_PER_SEG;
    using Base::NUM_TAIL_ELEMS;
    using Base::NUM_TAIL_SEGS;
    using Base::NUM_ELEMS_IN_INIT_WINDOW;
    using Base::NUM_SEGS_IN_INIT_WINDOW;
    using Base::NUM_PERM_SEGS_IN_INIT_WINDOW;
    using Base::NUM_INIT_ROUNDS_MAX;
    using Base::NUM_UINT32_IN_INIT_WINDOW_PER_THREAD;
    using Base::NUM_128b_INIT_PER_THREAD;
    using Base::NUM_SEGS_PER_ROUND;
    using Base::NUM_SEGS_PER_ISSUE_WARP;
    using Base::NUM_ISSUE_WARPS;
    using Base::NUM_TMA_LOAD_BUFS;
    using Base::TMA_PREFETCH_DEPTH;
    using Base::RECONSTRUCT_THRESHOLD;
    using Base::PERM_ADD_BASE;
    using Base::PERM_MUL_PRIME;
    using Base::PLACEHOLDER_PAIR;
    using Base::NUM_RECONSTRUCT_BUCKETS;

    static constexpr uint32_t PLACEHOLDER = 0xff800000; // -INF
    static constexpr uint32_t NEG_INF_BITS = 0xFF800000;
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096);
    static constexpr bool HAS_PARTIAL_ROUNDS = NUM_SEGS_PER_ROUND > NUM_TAIL_SEGS;

    struct SharedMemoryPlanFP32 : SharedMemoryPlanBase {};
    
    static __device__ __forceinline__
    void append_pair_at(uint32_t &dst_ptr, uint32_t index, uint32_t val_bits) {
        *(uint64_t*)__cvta_shared_to_generic(dst_ptr) = (uint64_t)val_bits << 32 | index;
        dst_ptr = __float_as_uint(__uint_as_float(dst_ptr) + __uint_as_float(8u));
    }

    static __device__ __forceinline__ void topk_select_kernel_devfunc(const TopkSelectArgs &args, const TmaParams &tma_params) {
        uint32_t batch_idx = blockIdx.x;
        uint32_t end_vocab_idx = args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlanFP32 &smem = *reinterpret_cast<SharedMemoryPlanFP32*>(wksp_buf);

        uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
        uint32_t lane_idx = threadIdx.x % 32;

        if (end_vocab_idx <= args.topk) {
            EpilogueT::template topk_select_epilogue<true>(
                (ValueT*)smem.surviving_topk_pairs[0],
                (uint32_t*)(smem.surviving_topk_pairs[1]),
                args,
                batch_idx, end_vocab_idx, warp_idx,
                *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs)
            );
            return;
        }

        uint32_t threshold_bits = NEG_INF_BITS;
        uint32_t num_incomers = 0;
        uint32_t survivor_buf_idx = 0;  // current candidate buffer (A/B, swapped by reconstruct)
        bool have_nan = false;

        Base::init_shared_memory(smem, warp_idx);

        uint32_t num_input_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
        uint32_t num_perm_segs = num_input_segs <= NUM_SEGS_IN_INIT_WINDOW ? 0u : (num_input_segs - 1) / NUM_TAIL_SEGS * NUM_TAIL_SEGS;
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_local_tail_elems_padded = num_perm_segs != 0 ? (uint32_t)NUM_TAIL_ELEMS : num_input_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_rounds = ku::ceil_div(num_local_tail_elems_padded + num_perm_elems, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_init_rounds = min(num_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
        ValueT *init_buf = (ValueT*)smem.incoming_topk_pairs;   // aliased

        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t perm_mul = PERM_MUL_PRIME % perm_len;
        static_assert(NUM_SEGS_PER_ROUND == NUM_SEGS_PER_ISSUE_WARP * NUM_ISSUE_WARPS);
        __shared__ bool should_select_whole_bucket_shared;

        auto find_pivot_in_histogram = [&](uint32_t topk, const uint32_t *bucket_counter) {
            bool should_select_whole_bucket =
                Base::template find_pivot_in_histogram<true>(smem, bucket_counter, topk, lane_idx);
            if (threadIdx.x == 0) {
                should_select_whole_bucket_shared = should_select_whole_bucket;
            }
        };
        // First radix pass: histogram the most significant byte of each distorted
        // fp32 value. Used to locate the top-k bucket prefix.
        uint32_t hist_base = cute::cast_smem_ptr_to_uint(smem.reconstruct_bucket_counter[0]);
        // mad forms the bucket address and red.shared.add does the increment: no return value, one address add.
        auto histogram_radix_msb_one = [&](uint32_t value_bits) {
            uint32_t addr;
            asm volatile ("mad.lo.u32 %0, %1, 4, %2; red.shared.add.u32 [%0], 1;"
                          : "=r"(addr)
                          : "r"(topk_select_common::distort(value_bits) >> 24), "r"(hist_base)
                          : "memory");
        };
        auto histogram_radix_msb = [&]<uint32_t N>(const uint32_t (&vals)[N], uint32_t m_end) {
            CUTE_UNROLL
            for (uint32_t i = 0; i < N; i++) {
                if (i == m_end) break;
                histogram_radix_msb_one(vals[i]);
            }
        };
        struct PivotAndQuota {
            uint32_t pivot_value_bits;     
            uint32_t start_pos_in_collector;            
            uint32_t eq_quota;              
            uint32_t cnt_nan;
        };

        auto compute_pivot_and_quota = [&](uint32_t topk, const auto &for_each_value) -> PivotAndQuota {
            if (warp_idx == 0) {
                find_pivot_in_histogram(topk, smem.reconstruct_bucket_counter[0]);
            }
            __syncthreads();
            // `pivot_prefix` tracks the pivot's leading bits in the *undistorted* (raw) domain.
            // Inside a round every element that survived the previous round must has the pivot's sign, so the distortion rule (to xor what) is the same.
            // So we can optimize scanning & picking & distorting into "comparing the raw leading bits" and "xor-ing `neg_mask`"
            uint32_t pivot_prefix_dist = smem.reconstruct_pivot_bucket;    // distorted top 8 bits
            bool pivot_negative = pivot_prefix_dist < 0x80u;               // distorted >= 0x80 <=> positive
            uint32_t neg_mask = pivot_negative ? 0xFFu : 0u;
            uint32_t pivot_prefix = pivot_prefix_dist ^ (pivot_negative ? 0xFFu : 0x80u);
            uint32_t num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;
            bool should_select_whole_bucket = should_select_whole_bucket_shared;      // whether round 1 is already a clean boundary
            uint32_t rounds_done = 1;

            // Early exit: once topk lands exactly on a bucket's lower edge, the pivot is that bucket's floor.
            CUTE_UNROLL
            for (uint32_t r = 1; r < 4; ++r) {
                if (should_select_whole_bucket) break;
                uint32_t *dst = smem.reconstruct_bucket_counter[r & 1];
                uint32_t *clr = smem.reconstruct_bucket_counter[(r & 1) ^ 1];
                uint32_t dst_base = cute::cast_smem_ptr_to_uint(dst);
                for_each_value([&](uint32_t value_bits) {
                    // Elements whose prefix differs from the pivot still add, but to this warp's own sink slot. This let PTXAS generate ATOMS.INC instructions which is faster
                    bool match = (value_bits >> (32 - 8 * r)) == pivot_prefix;
                    // bfe extracts the next raw byte in one instruction (ptxas otherwise emits shf + lop3)
                    uint32_t bucket_raw, bucket;
                    asm volatile ("bfe.u32 %0, %1, %2, 8;" : "=r"(bucket_raw) : "r"(value_bits), "r"(24 - 8 * r));
                    bucket = match ? (bucket_raw ^ neg_mask) : NUM_RECONSTRUCT_BUCKETS;
                    uint32_t addr;
                    asm volatile ("mad.lo.u32 %0, %1, 4, %2; red.shared.add.u32 [%0], 1;"
                                  : "=r"(addr) : "r"(bucket), "r"(dst_base) : "memory");
                });
                constexpr uint32_t NUM_BUCKET_CLEAR_128b = NUM_RECONSTRUCT_BUCKETS / 4;
                static_assert(NUM_THREADS >= NUM_BUCKET_CLEAR_128b);
                if (threadIdx.x < NUM_BUCKET_CLEAR_128b) {
                    reinterpret_cast<__int128_t*>(clr)[threadIdx.x] = (__int128_t)0ull;
                }
                __syncthreads();
                if (warp_idx == 0) {
                    find_pivot_in_histogram(num_elem_should_select_in_pivot_bucket, dst);
                }
                __syncthreads();
                // The histogram is keyed by distorted bytes, so undo the sign map on the bucket id.
                pivot_prefix = (pivot_prefix << 8) | (smem.reconstruct_pivot_bucket ^ neg_mask);
                num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;
                should_select_whole_bucket = should_select_whole_bucket_shared;
                rounds_done++;
            }
            // Take the lowest element in the selected bucket
            // Be careful that `pivot` may be negative, and we may have to fill the lower bits with 1
            uint32_t pivot_low_shift = (4 - rounds_done) * 8;
            uint32_t pivot_value_bits = pivot_prefix << pivot_low_shift;
            if (pivot_negative) {
                pivot_value_bits |= (1u << pivot_low_shift) - 1u;
            }
            float pivot_value = __uint_as_float(pivot_value_bits);

            // set.xx.f32.f32 yields 1.0/0.0, so the counts accumulate on the FP pipe instead of the
            // (busiest) integer pipe; the counts stay exact because FP32 can represent every integer within 0 ~ 2**23
            float gt_accum = 0.0f, eq_accum = 0.0f;
            float nan_accum = 0.0f;
            for_each_value([&](uint32_t value_bits) {
                float v = __uint_as_float(value_bits);
                asm volatile (
                    "{\n"
                    ".reg .f32 g, e;\n"
                    "set.gt.f32.f32 g, %3, %4;\n"
                    "add.f32 %0, %0, g;\n"
                    "set.eq.f32.f32 e, %3, %4;\n"
                    "add.f32 %1, %1, e;\n"
                    "min.NaN.f32 %2, %2, %3;\n" // We use `min.NaN` for NaN detection
                    "}\n"
                    : "+f"(gt_accum), "+f"(eq_accum), "+f"(nan_accum)
                    : "f"(v), "f"(pivot_value));
            });
            uint32_t cnt_gt = (uint32_t)gt_accum;
            uint32_t cnt_eq = (uint32_t)eq_accum;
            float nan_flag;
            asm ("set.nan.f32.f32 %0, %1, %1;" : "=f"(nan_flag) : "f"(nan_accum));
            uint32_t cnt_nan = (uint32_t)nan_flag;

            static_assert(NUM_WARPS <= NUM_RECONSTRUCT_BUCKETS);
            uint32_t *eq_pass_warp_cnt = smem.reconstruct_bucket_counter[0];
            auto eqgt = Base::compute_equal_quota_and_prefix(cnt_gt, cnt_eq, topk, warp_idx, lane_idx, eq_pass_warp_cnt);
            return {pivot_value_bits, eqgt.start_pos_in_collector, eqgt.eq_quota, cnt_nan};
        };

        auto reconstruct = [&](uint32_t cur_extra_len) -> uint32_t {
            uint32_t tid = threadIdx.x;
            Base::clear_reconstruct_histograms(smem, tid);

            uint32_t padded_extra_len = (cur_extra_len + 1u) & ~1u;
            if (warp_idx == 0 && (cur_extra_len & 1u)) {
                smem.incoming_topk_pairs[cur_extra_len] = PLACEHOLDER_PAIR;
            }
            __syncthreads();

            uint32_t num_128b = (MAX_TOPK + padded_extra_len) / 2;
            uint32_t num_128b_per_thread = ((num_128b + NUM_THREADS - 1) / NUM_THREADS) | 1u;   // force odd => bank-conflict-free smem gathers
            uint32_t b128_base = tid * num_128b_per_thread;
            uint32_t num_my_128b = b128_base < num_128b ? min(num_128b_per_thread, num_128b - b128_base) : 0u;

            auto b128_to_pair_addr = [&](uint32_t u) -> uint32_t {
                // The first MAX_TOPK/2 b128 words live in the current candidate buffer, the rest in the incoming region
                return u < MAX_TOPK / 2
                     ? cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx] + 2 * u)
                     : cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs + (2 * u - MAX_TOPK));
            };

            // One b128 word is 16 bytes = 2 pairs {index0, value0, index1, value1};
            constexpr uint32_t NUM_128b_MAX = (MAX_TOPK + RECONSTRUCT_THRESHOLD + NUM_ELEMS_PER_ROUND) / 2;
            constexpr uint32_t NUM_128b_PER_THREAD = ((NUM_128b_MAX + NUM_THREADS - 1) / NUM_THREADS) | 1u;   // | 1u since we may | 1u when generating `num_128b_per_thread`
            auto for_each_value = [&](const auto &fn) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_128b_PER_THREAD; i++) {
                    if (i == num_my_128b) break;
                    uint32_t pair2[4];
                    topk_select_common::ld_shared<4>(pair2, (const uint32_t*)b128_to_pair_addr(b128_base + i));
                    fn(pair2[1]);
                    fn(pair2[3]);
                }
            };

            for_each_value(histogram_radix_msb_one);
            __syncthreads();

            uint32_t topk = args.topk;
            auto [pivot_value_bits, start_pos_in_collector, eq_quota, cnt_nan] = compute_pivot_and_quota(topk, for_each_value);
            // Every NaN the main loop collected is part of the buffer this census just walked (the hit test
            // is `.gtu`, so NaN always becomes an incomer); the CTA-wide OR happens at the end.
            have_nan |= cnt_nan != 0;

            {
                uint32_t out_ptr = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[survivor_buf_idx ^ 1]) + start_pos_in_collector * (uint32_t)sizeof(uint64_t);
                float pivot_value = __uint_as_float(pivot_value_bits);
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_128b_PER_THREAD; i++) {
                    if (i == num_my_128b) break;
                    uint32_t pair2[4];
                    topk_select_common::ld_shared<4>(pair2, (const uint32_t*)b128_to_pair_addr(b128_base + i));
                    float v0 = __uint_as_float(pair2[1]);
                    float v1 = __uint_as_float(pair2[3]);
                    bool t0 = v0 == pivot_value && eq_quota != 0;
                    eq_quota -= t0;
                    if (v0 > pivot_value || t0) { append_pair_at(out_ptr, pair2[0], pair2[1]); }
                    bool t1 = v1 == pivot_value && eq_quota != 0;
                    eq_quota -= t1;
                    if (v1 > pivot_value || t1) { append_pair_at(out_ptr, pair2[2], pair2[3]); }
                }
            }

            survivor_buf_idx ^= 1;
            return pivot_value_bits;
        };

        uint32_t tma_permuted_segment_stride = NUM_ISSUE_WARPS * perm_mul % perm_len;
        uint32_t next_tma_permuted_segment = Base::get_permuted_seg_idx(warp_idx, perm_len, perm_mul);
        auto issue_tma_loads_for_round = [&]<bool IS_INIT, bool HAVE_TAIL>(uint32_t idx) {
            Base::template issue_tma_loads_for_round<IS_INIT, HAVE_TAIL>(
                smem, tma_params.tensor_map, batch_idx, end_vocab_idx,
                num_perm_segs, idx, next_tma_permuted_segment, tma_permuted_segment_stride, warp_idx);
        };

        if (cute::elect_one_sync()) {
            issue_tma_loads_for_round.template operator()<true, true>(0);
            CUTE_UNROLL
            for (uint32_t i = 1; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i < num_init_rounds) {
                    issue_tma_loads_for_round.template operator()<true, false>(i);
                }
            }
            {
                uint32_t num_main_prefetch = num_rounds - num_init_rounds;
                CUTE_UNROLL
                for (uint32_t i = 0; i < TMA_PREFETCH_DEPTH; i++) {
                    if (i < num_main_prefetch) {
                        issue_tma_loads_for_round.template operator()<false, false>(i);
                    }
                }
            }
        }

        { // INIT
            uint32_t num_elems_in_init_window_padded = num_local_tail_elems_padded + min(num_perm_segs, (uint32_t)NUM_PERM_SEGS_IN_INIT_WINDOW) * NUM_ELEMS_PER_SEG;
            uint32_t num_128b_in_init_window_padded = num_elems_in_init_window_padded / NUM_ELEMS_PER_128b;
            uint32_t cnt_floor = num_128b_in_init_window_padded / NUM_THREADS;
            uint32_t cnt_rem = num_128b_in_init_window_padded % NUM_THREADS;
            uint32_t my_elem_start_idx = (threadIdx.x * cnt_floor + min(threadIdx.x, cnt_rem)) * NUM_ELEMS_PER_128b;
            uint32_t num_my_elems = NUM_UINT32_PER_128b * cnt_floor + (threadIdx.x < cnt_rem ? NUM_UINT32_PER_128b : 0u);
            uint32_t num_local_tail_elems = end_vocab_idx - num_perm_elems;

            if (num_perm_segs != 0) {
                Base::fill_padded_tail_segments(init_buf, end_vocab_idx - num_perm_elems);
                __syncthreads();
            }

            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i == num_init_rounds) break;
                smem.init_full_bar[i].wait(0);
                if (num_local_tail_elems % NUM_ELEMS_PER_SEG != 0 && i == num_local_tail_elems / NUM_ELEMS_PER_ROUND) {
                    uint32_t box_end = ku::ceil_div(num_local_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG) * NUM_ELEMS_PER_SEG;
                    for (uint32_t e = num_local_tail_elems + threadIdx.x; e < box_end; e += NUM_THREADS) {
                        init_buf[Base::sw_elem(e)] = __uint_as_float(PLACEHOLDER);
                    }
                    __syncthreads();
                }
                uint32_t my_values[4 * NUM_128b_PER_THREAD_PER_ROUND];
                uint32_t num_my_values = 0;
                CUTE_UNROLL
                for (uint32_t k = 0; k < NUM_128b_PER_THREAD_PER_ROUND; ++k) {
                    uint32_t u = threadIdx.x + (i * NUM_128b_PER_THREAD_PER_ROUND + k) * NUM_THREADS;
                    if (u < num_128b_in_init_window_padded) {
                        topk_select_common::ld_shared<4>(my_values + k * 4, (const uint32_t*)(init_buf + Base::sw_b128(u) * NUM_ELEMS_PER_128b));
                        num_my_values = 4 * (k + 1);
                    }
                }
                histogram_radix_msb(my_values, num_my_values);
            }

            uint32_t init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
                if (j * NUM_UINT32_PER_128b == num_my_elems) break;
                uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
                topk_select_common::ld_shared<4>(init_values + j * NUM_UINT32_PER_128b, (const uint32_t*)(init_buf + Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
            }

            __syncthreads();    // publish the round-1 histogram; also the last read of init_buf

            static_assert(NUM_ELEMS_IN_INIT_WINDOW <= 0xFFFF);
            
            auto for_each_init_value = [&](const auto &fn) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i++) {
                    if (i == num_my_elems) break;
                    fn(init_values[i]);
                }
            };
            auto [pivot_value_bits, start_pos_in_collector, eq_quota, cnt_nan] = compute_pivot_and_quota(args.topk, for_each_init_value);
            // The init census walks the thread's whole slice of the window, i.e. every element of the init
            // window exactly once, so NaNs inside the window are caught here.
            have_nan |= cnt_nan != 0;

            {
                uint32_t out_ptr = cute::cast_smem_ptr_to_uint(smem.surviving_topk_pairs[0]) + start_pos_in_collector * (uint32_t)sizeof(uint64_t);
                float pivot_value = __uint_as_float(pivot_value_bits);
                uint32_t s0 = (max(my_elem_start_idx, num_local_tail_elems_padded) - num_local_tail_elems_padded) / NUM_ELEMS_PER_SEG;
                uint32_t perm_state = Base::get_permuted_seg_idx(s0, perm_len, perm_mul);
                float b128_base_f = 0.0f;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i += 2) {
                    if (i == num_my_elems) break;
                    if (i % NUM_ELEMS_PER_128b == 0) {
                        uint32_t g_u = my_elem_start_idx + i;      // fp32: word index == element index
                        if (g_u < num_local_tail_elems_padded) {
                            b128_base_f = __uint_as_float(num_perm_elems + g_u);
                        } else {
                            b128_base_f = __uint_as_float(perm_state * NUM_ELEMS_PER_SEG + g_u % NUM_ELEMS_PER_SEG);
                            if (g_u % NUM_ELEMS_PER_SEG == NUM_ELEMS_PER_SEG - NUM_ELEMS_PER_128b) {
                                Base::advance_perm_state(perm_state, perm_mul, perm_len);
                            }
                        }
                    }
                    // fp32-subnormal FP-pipe int-add (see file-head): index = base + element offset
                    uint32_t index0 = __float_as_uint(b128_base_f + __uint_as_float(i % NUM_ELEMS_PER_128b));
                    uint32_t index1 = __float_as_uint(b128_base_f + __uint_as_float(i % NUM_ELEMS_PER_128b + 1));
                    float v0 = __uint_as_float(init_values[i]);
                    float v1 = __uint_as_float(init_values[i + 1]);
                    bool t0 = v0 == pivot_value && eq_quota != 0;
                    eq_quota -= t0;
                    if (v0 > pivot_value || t0) { append_pair_at(out_ptr, index0, init_values[i]); }
                    // the quota is re-tested after element 0's decrement
                    bool t1 = v1 == pivot_value && eq_quota != 0;
                    eq_quota -= t1;
                    if (v1 > pivot_value || t1) { append_pair_at(out_ptr, index1, init_values[i + 1]); }
                }
            }

            threshold_bits = pivot_value_bits;
            __syncthreads();
        }

        uint32_t linear_segment_start =
            num_init_rounds * NUM_SEGS_PER_ROUND
            - NUM_TAIL_SEGS
            + warp_idx;
        uint32_t current_permuted_segment = Base::get_permuted_seg_idx(linear_segment_start, perm_len, perm_mul);
        uint32_t permuted_segment_stride_per_round = NUM_SEGS_PER_ROUND % perm_len * perm_mul % perm_len;

        uint32_t num_main_rounds = num_rounds - num_init_rounds;
        // Track the TMA buffer index and its phase incrementally instead of dividing every round.
        uint32_t tma_buf_idx = 0;
        uint32_t tma_buf_phase = 0;
        for (uint32_t main_round_idx = 0; main_round_idx < num_main_rounds; ++main_round_idx) {
            // This thread's contiguous chunk starts at logical_elem_offset in this round.
            uint32_t logical_elem_offset = threadIdx.x * NUM_ELEMS_PER_THREAD_PER_ROUND;
            // Offset inside the 512-element segment; used later to rebuild original indices.
            uint32_t offset_in_segment = logical_elem_offset % NUM_ELEMS_PER_SEG;
            // TMA stores the round data in a swizzled layout. swizzle_mask describes how the
            // logical element offset is mapped to the actual shared-memory offset.
            uint32_t swizzle_mask = Base::sw_msk(logical_elem_offset);
            uint32_t chunk_swizzle_mask = swizzle_mask & ELEMS_PER_THREAD_PER_ROUND_MASK;
            uint32_t smem_read_offset = logical_elem_offset ^ (swizzle_mask & ~ELEMS_PER_THREAD_PER_ROUND_MASK);

            if (main_round_idx + TMA_PREFETCH_DEPTH < num_main_rounds) {
                if (cute::elect_one_sync()) {
                    issue_tma_loads_for_round.template operator()<false, false>(main_round_idx + TMA_PREFETCH_DEPTH);
                }
            }

            bool is_warp_active = true;
            if constexpr (HAS_PARTIAL_ROUNDS) {
                is_warp_active = (num_init_rounds + main_round_idx) * NUM_SEGS_PER_ROUND + warp_idx < NUM_TAIL_SEGS + num_perm_segs;
            }

            const ValueT *buf = smem.tma_load_buf[tma_buf_idx];
            // One bit per element of this thread's slice, in element order.
            // Hits are rare, so appending through the set bits of this mask is faster than testing + storing every element again.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32);        // hit_mask is a uint32_t
            uint32_t hit_mask = 0;
            if (is_warp_active) {
                smem.tma_load_full_bar[tma_buf_idx].wait(tma_buf_phase);
                ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
                Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(values, buf, smem_read_offset, chunk_swizzle_mask);

                float threshold = __uint_as_float(threshold_bits);
                CUTE_UNROLL
                for (uint32_t j = 0; j < NUM_ELEMS_PER_THREAD_PER_ROUND; ++j) {
                    asm volatile (
                        "{\n"
                        ".reg .pred p;\n"
                        "setp.gtu.f32 p, %1, %2;\n" // .gtu: NaN always counts as a hit, so every NaN reaches the incoming buffer and the census reports it
                        "@p or.b32 %0, %0, %3;\n"
                        "}\n"
                        : "+r"(hit_mask)
                        : "f"((float)values[j]), "f"(threshold), "r"(1u << j));
                }
            }
            uint32_t num_new_incomers = __popc(hit_mask);

            uint32_t warp_total_hits = __reduce_add_sync(0xFFFFFFFF, num_new_incomers);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total_hits;
            }
            __syncthreads();

            uint32_t seg_elem_base = current_permuted_segment * NUM_ELEMS_PER_SEG + offset_in_segment;
            // Element e of this thread's slice lives at smem_read_offset + (e ^ chunk_swizzle_mask)
            uint32_t elem_addr_base = cute::cast_smem_ptr_to_uint(buf + smem_read_offset);

            // Start the first hit's smem load before the count exchange below, so that its latency
            // (and the barrier wait) overlaps with them instead of delaying the first store.
            uint32_t first_hit_e = hit_mask != 0 ? __ffs(hit_mask) - 1u : 0u;
            uint32_t first_hit_val = 0;
            if (is_warp_active && warp_total_hits != 0) {
                asm volatile ("ld.shared.u32 %0, [%1];" : "=r"(first_hit_val) : "r"(elem_addr_base + ((first_hit_e ^ chunk_swizzle_mask) << 2)));
            }

            static_assert(NUM_WARPS <= 32);
            uint32_t stored_warp_hits = lane_idx < NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
            uint32_t num_total_hits_in_this_round = __reduce_add_sync(0xFFFFFFFF, stored_warp_hits);

            uint32_t dst_slot =
                num_incomers +
                __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_hits : 0u) +
                warp_level_exclusive_prefix_sum(num_new_incomers, lane_idx);

            if (is_warp_active && warp_total_hits != 0) {
                uint32_t dst_ptr = cute::cast_smem_ptr_to_uint(smem.incoming_topk_pairs) + dst_slot * (uint32_t)sizeof(uint64_t);
                if (hit_mask != 0) {
                    asm volatile ("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(dst_ptr), "r"(seg_elem_base + first_hit_e), "r"(first_hit_val) : "memory");
                    dst_ptr += (uint32_t)sizeof(uint64_t);
                }
                uint32_t mask = hit_mask & (hit_mask - 1u);
                while (mask != 0) {
                    uint32_t e = __ffs(mask) - 1u;
                    mask &= mask - 1u;
                    uint32_t val_word;
                    asm volatile ("ld.shared.u32 %0, [%1];" : "=r"(val_word) : "r"(elem_addr_base + ((e ^ chunk_swizzle_mask) << 2)));
                    asm volatile ("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(dst_ptr), "r"(seg_elem_base + e), "r"(val_word) : "memory");
                    dst_ptr += (uint32_t)sizeof(uint64_t);
                }
            }
            Base::advance_perm_state(current_permuted_segment, permuted_segment_stride_per_round, perm_len);

            num_incomers += num_total_hits_in_this_round;
            __syncthreads();

            if (++tma_buf_idx == NUM_TMA_LOAD_BUFS) {
                tma_buf_idx = 0;
                tma_buf_phase ^= 1u;
            }

            if (num_incomers >= RECONSTRUCT_THRESHOLD) {
                threshold_bits = reconstruct(num_incomers);
                num_incomers = 0;
            }
        }

        if (num_incomers > 0) {
            reconstruct(num_incomers);
            num_incomers = 0;
        }

        have_nan = __syncthreads_or(have_nan) != 0;
        if (have_nan) {
            Base::take_action_when_have_nan(args, batch_idx);
            return;
        }

        Base::template stage_output_and_epilogue<Config::sorted_index>(
            smem, *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs), args,
            batch_idx, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx);

    }

};


template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, 1)
__global__ void topk_kernel(__grid_constant__ const TopkSelectArgs args, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::topk_select_kernel_devfunc(args, tma_params);
}

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args) {
    KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
    KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
    KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
    static_assert(cute::is_same_v<typename Config::ValueT, float>);
    KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

    using Kernel = TopkSelectKernelFP32<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlanFP32);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    KU_ASSERT(args.stride_input_batch % 4 == 0, "stride_input_batch must be 16B-aligned");
    typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

    ku::launch_kernel(ku::KernelLaunchConfig {
        dim3(args.batch_size),
        dim3(Kernel::NUM_THREADS),
        smem_size,
        args.stream
    }, kernel, args, tma_params);
}

}   // topk_select_fp32
