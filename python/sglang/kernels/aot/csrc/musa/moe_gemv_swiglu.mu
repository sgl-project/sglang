/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <musa_runtime.h>
#include <cassert>
#include <mutex>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include "common.muh"
#include "dtype.muh"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

using namespace musa::dnn;

#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ == 310
#define ThreadNumPerWarp 32
#else
#define ThreadNumPerWarp 128
#endif

#define SYNC_IF_NEEDED() \
    if constexpr (BLOCK_N * BLOCK_K > ThreadNumPerWarp) { \
        __SYNCTHREADS_LM; \
    }

#define CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, _IS_FP8, _IS_RMSNORM) \
    if (scale_k_group_tile == 128) { \
        musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, _IS_FP8, 128, _IS_RMSNORM> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_CDTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
                static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
    } else { \
        musa_gemv_kernel<_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, _IS_FP8, 64, _IS_RMSNORM> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_CDTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
                static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
    } \
    return;

#define RUN_SCALE_ROUTE_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _IS_FP8) \
    if (mul_routed_weight) { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, true, _IS_FP8, false) \
        } else if(use_rms_norm) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, false, _IS_FP8, true) \
        } else { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, false, _IS_FP8, false) \
        } \
    } else { \
        if (use_swigelu) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, true, _IS_FP8, false) \
        } else if (use_rms_norm) { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, false, _IS_FP8, true) \
        } else { \
            CAL_MOE_GEMV_FP8(_ADTYPE, _BDTYPE, _CDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, false, _IS_FP8, false) \
        } \
    }

#define CAL_MOE_GEMV_W4A16(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, _IS_RMS_NROM) \
    if (is_pergroup_scale) { \
        if (scale_k_group_tile == 128) { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, true, true, false, 128, _IS_RMS_NROM> \
                <<<grid_size, block_size, shmem_size, stream>>>( \
                    static_cast<_ADTYPE*>(C.data_ptr()), \
                    static_cast<_ADTYPE*>(A.data_ptr()), \
                    static_cast<_BDTYPE*>(B.data_ptr()), \
                    static_cast<int*>(topk_ids_ptr), \
                    static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                    static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
                    static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                    topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                    static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
            return; \
        } else { \
            musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, true, true, false, 64, _IS_RMS_NROM> \
                <<<grid_size, block_size, shmem_size, stream>>>( \
                    static_cast<_ADTYPE*>(C.data_ptr()), \
                    static_cast<_ADTYPE*>(A.data_ptr()), \
                    static_cast<_BDTYPE*>(B.data_ptr()), \
                    static_cast<int*>(topk_ids_ptr), \
                    static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                    static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
                    static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                    topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                    static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
            return; \
        } \
    } else { \
        musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, true, false, false, 1, _IS_RMS_NROM> \
            <<<grid_size, block_size, shmem_size, stream>>>( \
                static_cast<_ADTYPE*>(C.data_ptr()), \
                static_cast<_ADTYPE*>(A.data_ptr()), \
                static_cast<_BDTYPE*>(B.data_ptr()), \
                static_cast<int*>(topk_ids_ptr), \
                static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
                static_cast<_SCALE_DTYPE*>(a_scale_ptr), \
                static_cast<_SCALE_DTYPE*>(b_scale_ptr), \
                topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
                static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
        return; \
    }

#define CAL_MOE_GEMV(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, _IS_RMS_NROM) \
    musa_gemv_kernel<_ADTYPE, _BDTYPE, _ADTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, block_n, block_k, iobit, _IS_MUL_ROUTED_WEIGHT, _IS_SWGELU, false, false, false, 1, _IS_RMS_NROM> \
        <<<grid_size, block_size, shmem_size, stream>>>( \
            static_cast<_ADTYPE*>(C.data_ptr()), \
            static_cast<_ADTYPE*>(A.data_ptr()), \
            static_cast<_BDTYPE*>(B.data_ptr()), \
            static_cast<int*>(topk_ids_ptr), \
            static_cast<_TOPK_WEIGHT_DTYPE*>(topk_weights_ptr), \
            nullptr, \
            nullptr, \
            topk, expert_offset_stride, nr_n, hidden_size, num_experts, half_n_idx, scale_k_len, \
            static_cast<bfloat16_t*>(rms_gamma_ptr), static_cast<float*>(rms_sum_out_ptr), static_cast<int*>(rms_count_ptr), eps); \
    return;

#define RUN_SCALE_ROUTE(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, _CAL_FUNC) \
    if (mul_routed_weight) { \
        if (use_swigelu) { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, true, false) \
        } else if (use_rms_norm) { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, false, true) \
        } else { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, true, false, false) \
        } \
    } else { \
        if (use_swigelu) { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, true, false) \
        } else if (use_rms_norm) { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, false, true) \
        } else { \
            _CAL_FUNC(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _SCALE_DTYPE, false, false, false) \
        } \
    }

#define RUN_ROUNTE_WEIGHT(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, _CAL_FUNC) \
    if (!B_scale.has_value() || B_scale->scalar_type() == at::ScalarType::Float) { \
        RUN_SCALE_ROUTE(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, float, _CAL_FUNC) \
    } else if (B_scale.has_value() && B_scale->scalar_type() == at::ScalarType::BFloat16) { \
        RUN_SCALE_ROUTE(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, bfloat16_t, _CAL_FUNC) \
    } else if (B_scale.has_value() && B_scale->scalar_type() == at::ScalarType::Half) { \
        RUN_SCALE_ROUTE(_ADTYPE, _BDTYPE, _TOPK_WEIGHT_DTYPE, float16_t, _CAL_FUNC) \
    }

#define GEN_LAUNCH_KERN_GEMV(_BLK_N, _BLK_K) \
    { \
        launch_kernel = [&]() { \
            constexpr int block_n = _BLK_N; \
            constexpr int block_k = _BLK_K; \
            TORCH_CHECK(nr_n % block_n == 0, "gemv n need align"); \
            TORCH_CHECK(hidden_size % block_k == 0, "gemv k need align"); \
            dim3 block_size{block_n * block_k, 1, 1}; \
            dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)topk, (uint32_t)bseqlen}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (use_int4_w4a16) { \
                if (A.scalar_type() == at::ScalarType::BFloat16) { \
                    RUN_ROUNTE_WEIGHT(bfloat16_t, int8_t, float, CAL_MOE_GEMV_W4A16) \
                } else if (A.scalar_type() == at::ScalarType::Half) { \
                    RUN_ROUNTE_WEIGHT(float16_t, int8_t, float, CAL_MOE_GEMV_W4A16) \
                } \
            } else if (is_fp8) { \
                if (A.dtype() == at::ScalarType::BFloat16) { \
                    RUN_SCALE_ROUTE_FP8(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, float, float, true) \
                } else { \
                    RUN_SCALE_ROUTE_FP8(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, float, float, true) \
                } \
            } else { \
                if (A.scalar_type() == at::ScalarType::BFloat16) { \
                    RUN_ROUNTE_WEIGHT(bfloat16_t, bfloat16_t, float, CAL_MOE_GEMV) \
                } else if (A.scalar_type() == at::ScalarType::Half) { \
                    RUN_ROUNTE_WEIGHT(float16_t, float16_t, float, CAL_MOE_GEMV) \
                } \
            } \
            TORCH_CHECK(false, "no support on moe gemv"); \
        }; \
    }

#define GEN_LAUNCH_KERN(_BLK_N, _BLK_K) \
    { \
        launch_kernel = [&]() { \
            constexpr int block_n = _BLK_N; \
            constexpr int block_k = _BLK_K; \
            TORCH_CHECK(nr_n % block_n == 0, "gemv n need align"); \
            TORCH_CHECK(hidden_size % block_k == 0, "gemv k need align"); \
            dim3 block_size{block_n * block_k, 1, 1}; \
            dim3 grid_size{(uint32_t)ceil_div(reduce_size, block_n), (uint32_t)topk, (uint32_t)bseqlen}; \
            int shmem_size = block_n * sizeof(float) * block_k; \
            if (use_int4_w4a16) { \
                if (A.scalar_type() == at::ScalarType::BFloat16) { \
                    if (topk_weights.scalar_type() == at::ScalarType::Float) { \
                        RUN_ROUNTE_WEIGHT(bfloat16_t, int8_t, float, CAL_MOE_GEMV_W4A16) \
                    } else if (topk_weights.scalar_type() == at::ScalarType::BFloat16) { \
                        RUN_ROUNTE_WEIGHT(bfloat16_t, int8_t, bfloat16_t, CAL_MOE_GEMV_W4A16) \
                    } \
                } else if (A.scalar_type() == at::ScalarType::Half) { \
                    if (topk_weights.scalar_type() == at::ScalarType::Float) { \
                        RUN_ROUNTE_WEIGHT(float16_t, int8_t, float, CAL_MOE_GEMV_W4A16) \
                    } else if (topk_weights.scalar_type() == at::ScalarType::Half) { \
                        RUN_ROUNTE_WEIGHT(float16_t, int8_t, float16_t, CAL_MOE_GEMV_W4A16) \
                    } \
                } \
            } else if (is_fp8) { \
                if (A.dtype() == at::ScalarType::BFloat16) { \
                    RUN_SCALE_ROUTE_FP8(bfloat16_t, __mt_fp8_e4m3, bfloat16_t, float, float, true) \
                } else { \
                    RUN_SCALE_ROUTE_FP8(__mt_fp8_e4m3, __mt_fp8_e4m3, bfloat16_t, float, float, true) \
                } \
            } else { \
                if (A.scalar_type() == at::ScalarType::BFloat16) { \
                    if (topk_weights.scalar_type() == at::ScalarType::Float) { \
                        RUN_ROUNTE_WEIGHT(bfloat16_t, bfloat16_t, float, CAL_MOE_GEMV) \
                    } else if (topk_weights.scalar_type() == at::ScalarType::BFloat16) { \
                        RUN_ROUNTE_WEIGHT(bfloat16_t, bfloat16_t, bfloat16_t, CAL_MOE_GEMV) \
                    } \
                } else if (A.scalar_type() == at::ScalarType::Half) { \
                    if (topk_weights.scalar_type() == at::ScalarType::Float) { \
                        RUN_ROUNTE_WEIGHT(float16_t, float16_t, float, CAL_MOE_GEMV) \
                    } else if (topk_weights.scalar_type() == at::ScalarType::Half) { \
                        RUN_ROUNTE_WEIGHT(float16_t, float16_t, float16_t, CAL_MOE_GEMV) \
                    } \
                } \
            } \
            TORCH_CHECK(false, "no support on moe gemv"); \
        }; \
    }

template <typename AType, typename BType, typename CType, typename ScoreType, typename ScaleType,
          int BLOCK_N, int BLOCK_K, int iobit, bool mul_routed_weight, bool is_swigelu,
          bool is_w4a16, bool is_per_group_scale, bool is_fp8, int scale_block, bool use_rms_norm>
__global__ void musa_gemv_kernel(
    CType *c_ptr,
    const AType *a_ptr,
    const BType *b_ptr,
    int *expert_idx_table,
    ScoreType *score_ptr,
    ScaleType *scale_a,
    ScaleType *scale_b,
    int topk,
    int expert_offset_stride,
    int n, int k,
    int nr_expert,
    int half_n_idx,
    int scale_k_len,
    bfloat16_t *gamma,
    float* sum_out,
    volatile int *count,
    float eps) {

    constexpr int bits_of_byte = 8;
    constexpr int half_blockn = BLOCK_N / 2;
    constexpr int b_vec_bits = 128;
    constexpr int Vlen = is_w4a16 ? b_vec_bits / 4 : b_vec_bits / (sizeof(BType) * bits_of_byte);
    constexpr int w4a16_shift = is_w4a16 ? 2 : 1;
    constexpr int scale_k_load_cntdown_init = ceil_div(scale_block, (BLOCK_K * Vlen));
    constexpr bool fuse_castfp8 = (is_fp8 && !std::is_same_v<AType, __mt_fp8_e4m3>);

    using AVecSType = std::conditional_t<std::is_same_v<AType, __mt_fp8_e4m3>, uint8_t, AType>;
    using BVecSType = std::conditional_t<is_fp8, uint8_t, BType>;
    using v16f32_t = float __attribute__((vector_size(64)));
    using v8f32_t = float __attribute__((vector_size(32)));
    using AVec = typename std::conditional_t<
        is_w4a16,
        v16f32_t,
        typename std::conditional_t<
            fuse_castfp8,
            v8f32_t,
            typename VecType<AVecSType, 128>::Ttype
        >
    >;
    using BVec = typename VecType<BVecSType, b_vec_bits>::Ttype;
    using fp8x4_vec = unsigned char __attribute__((vector_size(4)));

    int token_idx = blockIdx.z;
    int expert_idx = blockIdx.y;
    int real_expert_idx = 0;
    int t_n_idx = threadIdx.x / BLOCK_K;
    int t_k_idx = threadIdx.x % BLOCK_K;
    int n_idx = blockIdx.x * BLOCK_N + t_n_idx;

    if (expert_idx_table != nullptr) {
        real_expert_idx = expert_idx_table[token_idx * topk + expert_idx];
        if (real_expert_idx < 0 || real_expert_idx >= nr_expert) {
              if constexpr (is_swigelu) {
                if (n_idx < half_n_idx) {
                  int offsets = (token_idx * topk + expert_idx) * half_n_idx + n_idx;
                  c_ptr[offsets] = 0;
                }
              } else {
                int offsets = (token_idx * topk + expert_idx) * half_n_idx * 2 + n_idx;
                c_ptr[offsets] = 0;
              }
            return;
        }
    }

    constexpr int thread_sum_len = is_fp8 ? Vlen / 4 : Vlen;
    float cur_thread_sum[thread_sum_len];
    int scale_k_load_cntdown = scale_k_load_cntdown_init;

    extern __shared__ float shared_array[];

    if constexpr (is_swigelu) {
        if (t_n_idx < half_blockn) {
            n_idx = blockIdx.x * half_blockn + t_n_idx;
        } else {
            n_idx = blockIdx.x * half_blockn + t_n_idx - half_blockn + half_n_idx;
        }
    }


    #pragma unroll
    for (int i = 0; i < thread_sum_len; i++) {
        cur_thread_sum[i] = 0.0f;
    }

    float scale_a_val = 1.0f;
    float scale_b_val = 1.0f;
    int scale_a_offset = 0;
    int scale_b_offset = 0;

    if constexpr (is_w4a16) {
        scale_b_offset = (real_expert_idx * n + n_idx) * scale_k_len + t_k_idx * Vlen / scale_block;
        if constexpr (is_swigelu) {
            scale_b_offset = (real_expert_idx * 2 * n + n_idx) * scale_k_len + t_k_idx * Vlen / scale_block;
        }
        scale_b_val = scale_b[scale_b_offset];
        scale_k_load_cntdown -= 1;
    } else if (is_fp8) {
        scale_a_offset = token_idx * scale_k_len + t_k_idx * Vlen / scale_block;
        scale_b_offset = (real_expert_idx * n + n_idx) / scale_block * scale_k_len + t_k_idx * Vlen / scale_block;
        if constexpr (is_swigelu) {
            scale_b_offset = (real_expert_idx * 2 * n + n_idx) / scale_block * scale_k_len + t_k_idx * Vlen / scale_block;
        }
        if constexpr (!fuse_castfp8) {
            scale_a_val = scale_a[scale_a_offset];
        }
        scale_b_val = scale_b[scale_b_offset];
        scale_k_load_cntdown -= 1;
    }

    const BType *b_base_ptr = b_ptr + ((size_t)real_expert_idx * expert_offset_stride + n_idx * k + t_k_idx * Vlen) / w4a16_shift;

    for (int k_idx = 0; k_idx < k; k_idx += Vlen * BLOCK_K) {
        AType a_reg[Vlen];
        BType b_reg[Vlen / w4a16_shift];
        *(AVec *)(a_reg) = *(AVec *)(a_ptr + token_idx * k + t_k_idx * Vlen + k_idx);
        *(BVec *)(b_reg) = *(BVec *)(b_base_ptr + k_idx / w4a16_shift);

        if constexpr (is_w4a16 && !is_fp8) {
            float b_reg_float[Vlen];
            #pragma unroll
            for (int i = 0; i < Vlen / 2; i++) {
                if constexpr (is_per_group_scale) {
                    uint8_t read_u8 = b_reg[i];
                    b_reg_float[i * 2 + 0] = scale_b_val * ((float)(read_u8 & 0xF) - 8.f);
                    b_reg_float[i * 2 + 1] = scale_b_val * ((float)(read_u8 >> 4) - 8.f);
                } else {
                    int8_t read_s8 = b_reg[i];
                    b_reg_float[i * 2 + 0] = scale_b_val * (float)((int8_t)(read_s8 << 4));
                    b_reg_float[i * 2 + 1] = scale_b_val * (float)((int8_t)(read_s8 & 0xF0));
                }
            }
            if constexpr (is_per_group_scale) {
                if (scale_k_load_cntdown == 0 && (k_idx + Vlen * BLOCK_K) < k) {
                    scale_b_offset += ceil_div(BLOCK_K * Vlen, scale_block);
                    scale_b_val = scale_b[scale_b_offset];
                    scale_k_load_cntdown = scale_k_load_cntdown_init;
                }
                scale_k_load_cntdown -= 1;
            }
            #pragma unroll
            for (int i = 0; i < thread_sum_len; i++) {
                cur_thread_sum[i] += b_reg_float[i] * (float)a_reg[i];
            }
        } else if constexpr (is_fp8) {
            float scale_val = scale_a_val * scale_b_val;
            if (scale_k_load_cntdown == 0 && (k_idx + Vlen * BLOCK_K) < k) {
                scale_a_offset += ceil_div(BLOCK_K * Vlen, scale_block);
                scale_b_offset += ceil_div(BLOCK_K * Vlen, scale_block);
                if constexpr (!fuse_castfp8) {
                    scale_a_val = scale_a[scale_a_offset];
                }
                scale_b_val = scale_b[scale_b_offset];
                scale_k_load_cntdown = scale_k_load_cntdown_init;
            }
            scale_k_load_cntdown -= 1;
            for (int i = 0; i < thread_sum_len; i++) {
                typedef _Float16 _half_v4 __attribute__((ext_vector_type(4)));
                typedef _Float32 _float_v4 __attribute__((ext_vector_type(4)));
                _half_v4 a_halfv4;
                _half_v4 b_halfv4;
                _float_v4 a_float4;
                _float_v4 b_float4;
                if constexpr (fuse_castfp8) {
                    b_halfv4 = __musa_e4m32f16_rn_bst4(reinterpret_cast<const fp8x4_vec*>(b_reg)[i]);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        cur_thread_sum[i] += scale_val * float(a_reg[i * 4 + j]) * (b_halfv4[j]);
                    }
                } else {
                    a_halfv4 = __musa_e4m32f16_rn_bst4(reinterpret_cast<const fp8x4_vec*>(a_reg)[i]);
                    b_halfv4 = __musa_e4m32f16_rn_bst4(reinterpret_cast<const fp8x4_vec*>(b_reg)[i]);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        cur_thread_sum[i] += scale_val * (a_halfv4[j]) * (b_halfv4[j]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < thread_sum_len; i++) {
                cur_thread_sum[i] += (float)b_reg[i] * (float)a_reg[i];
            }
        }
    }

    float rst = 0;
    #pragma unroll
    for (int i = 0; i < thread_sum_len; i++) {
        rst += cur_thread_sum[i];
    }

    if constexpr (is_w4a16 && !is_per_group_scale) {
        rst = rst / 16.f;
    }

    if constexpr (BLOCK_K > 1) {
        shared_array[threadIdx.x] = rst;
        SYNC_IF_NEEDED()
        if (threadIdx.x < BLOCK_N) {
            rst = 0;
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i++) {
                rst += shared_array[threadIdx.x * BLOCK_K + i];
            }
        }
        if constexpr (is_swigelu) {
            SYNC_IF_NEEDED()
        }
    }

    if constexpr (BLOCK_N > ThreadNumPerWarp) {
        return;
    }

    if (threadIdx.x < BLOCK_N) {
        int dst_n_idx = blockIdx.x * BLOCK_N + threadIdx.x;
        if constexpr (is_swigelu) {
            dst_n_idx = blockIdx.x * half_blockn + threadIdx.x;
        }

        if constexpr (mul_routed_weight) {
            float score = (float)score_ptr[token_idx * topk + expert_idx];
            rst = rst * score;
        }

        if constexpr (is_swigelu) {
            shared_array[threadIdx.x] = rst;
            if (threadIdx.x < half_blockn) {
                float b = shared_array[threadIdx.x + half_blockn];
                rst = rst * sigmoid(rst) * b;
                c_ptr[token_idx * topk * n + expert_idx * n + dst_n_idx] = rst;
            }
        } else if constexpr (use_rms_norm) {
            float rms = rst * rst;
            int count_val = 0;

            for (int offset = 1; offset < BLOCK_N; offset *= 2) {
                float peer = __shfl_xor_sync(BLOCK_N, rms, offset);
                rms += peer;
            }

            if (threadIdx.x == 0) {
                atomicAdd(sum_out, rms);
                __threadfence_block();
                atomicAdd((int*)(count), 1);
            }

            while (count_val < gridDim.x) {
                count_val = count[0];
            }

            rms = sum_out[0];
            rst = rst * rsqrtf(rms / n + eps) * float(gamma[dst_n_idx]);
            c_ptr[token_idx * topk * n + expert_idx * n + dst_n_idx] = rst;
        } else {
            c_ptr[token_idx * topk * n + expert_idx * n + dst_n_idx] = rst;
        }
    }
}

struct BlockConfig {
    int block_n;
    int block_k;
    float score;
    bool valid;
};

void musa_fused_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor> &gamma,
    double eps) {

    TORCH_CHECK(A.dim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.dim() == 2, "B must be dim 2.")

    bool mul_routed_weight = false;
    int topk = 1;
    int32_t bseqlen = A.size(0);
    int32_t hidden_size = A.size(1);
    int32_t num_experts = 1;
    int32_t reduce_size = B.size(0);
    bool is_fp8 = false;

    if (B.dtype() == torch::kFloat8_e4m3fn) {
        is_fp8 = true;
    }

    int current_arch = at::musa::getMUSAArch();
    if (current_arch < 300) {
        if (is_fp8) {
            TORCH_CHECK(false, "gemv moe not support Float8_e4m3fn on MUSA arch ", current_arch);
        }
    }

    const at::musa::OptionalMUSAGuard device_guard(device_of(A));
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    void *topk_ids_ptr = nullptr;
    void *topk_weights_ptr = nullptr;
    void *a_scale_ptr = nullptr;
    void *b_scale_ptr = nullptr;

    if (A_scale.has_value()) {
        a_scale_ptr = A_scale.value().data_ptr();
    }
    if (B_scale.has_value()) {
        b_scale_ptr = B_scale.value().data_ptr();
    }

    void *rms_gamma_ptr = nullptr;
    void *rms_sum_out_ptr = nullptr;
    void *rms_count_ptr = nullptr;

    if (use_rms_norm && gamma.has_value()) {
        torch::Tensor sum_out = torch::zeros({1}, A.options().dtype(torch::kFloat));
        torch::Tensor count = torch::zeros({1}, A.options().dtype(torch::kInt));
        rms_gamma_ptr = gamma.value().data_ptr();
        rms_sum_out_ptr = sum_out.data_ptr();
        rms_count_ptr = count.data_ptr();
    }

    int device;
    musaGetDevice(&device);
    musaDeviceProp device_prop;
    musaGetDeviceProperties(&device_prop, device);
    int num_mp = device_prop.multiProcessorCount;
    int expert_offset_stride = reduce_size * hidden_size;
    int half_n_idx = reduce_size / 2;
    int scale_k_len = 1;
    int scale_k_group_tile = 128;

    if (use_int4_w4a16 || is_fp8) {
        scale_k_len = B_scale->size(1);
        if (scale_k_len != 1) {
            scale_k_group_tile = ceil_div(hidden_size, scale_k_len);
            TORCH_CHECK(scale_k_group_tile == 128 || scale_k_group_tile == 64, "scale_k_group_tile only support 128 or 64");
        }
    }

    bool is_pergroup_scale = scale_k_len != 1;
    int nr_n = use_swigelu ? reduce_size / 2 : reduce_size;

    std::function<void()> launch_kernel;

    BlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = use_int4_w4a16 ?
                      (iobit / 4):
                      (iobit / (torch::elementSize(B.scalar_type()) * bits_of_byte));

    float target_ratio = static_cast<float>(reduce_size) / hidden_size;

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (reduce_size % config.block_n == 0) && (hidden_size % load_size == 0) && (load_size % scale_k_group_tile == 0);

        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    BlockConfig best_config_storage;
    if (current_arch < 300) {
        best_config_storage = {128, 1, -1.0f, false};
    } else {
        best_config_storage = {32, 1, -1.0f, false};
    }
    BlockConfig* best_config = &best_config_storage;
    for (auto& config : configs) {
        if (config.valid && config.score > best_config->score) {
            best_config = &config;
        }
    }

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: GEN_LAUNCH_KERN_GEMV(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: GEN_LAUNCH_KERN_GEMV(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: GEN_LAUNCH_KERN_GEMV(16, 8); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: GEN_LAUNCH_KERN_GEMV(32, 4); break;
                case 1: GEN_LAUNCH_KERN_GEMV(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        case 128:
            switch (best_config->block_k) {
                case 1: GEN_LAUNCH_KERN_GEMV(128, 1);
                    break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=128");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();
}

void fused_moe_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu) {

    TORCH_CHECK(A.dim() == 2, "A must be dim 2.")
    TORCH_CHECK(B.dim() == 3, "B must be dim 3.")

    int32_t bseqlen = A.size(0);
    bool is_fp8 = false;
    if (B.dtype() == torch::kFloat8_e4m3fn) {
        is_fp8 = true;
    }

    bool use_rms_norm = false;
    void *rms_gamma_ptr = nullptr;
    void *rms_sum_out_ptr = nullptr;
    void *rms_count_ptr = nullptr;
    float eps = 1e-6;

    int current_arch = at::musa::getMUSAArch();
    if (current_arch < 300) {
        if (is_fp8) {
            TORCH_CHECK(false, "gemv moe not support Float8_e4m3fn on MUSA arch ", current_arch);
        }
    }

    int32_t hidden_size = A.size(1);
    int32_t num_experts = B.size(0);
    int32_t reduce_size = B.size(1);

    const at::musa::OptionalMUSAGuard device_guard(device_of(A));
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    void *topk_ids_ptr = topk_ids.data_ptr();
    void *topk_weights_ptr = topk_weights.data_ptr();
    void *a_scale_ptr = nullptr;
    void *b_scale_ptr = nullptr;

    if (A_scale.has_value()) {
        a_scale_ptr = A_scale.value().data_ptr();
    }
    if (B_scale.has_value()) {
        b_scale_ptr = B_scale.value().data_ptr();
    }

    int device;
    musaGetDevice(&device);
    musaDeviceProp device_prop;
    musaGetDeviceProperties(&device_prop, device);
    int num_mp = device_prop.multiProcessorCount;
    int expert_offset_stride = reduce_size * hidden_size;
    int half_n_idx = reduce_size / 2;
    int scale_k_len = 1;
    int scale_k_group_tile = 128;

    bool is_pergroup_scale = false;
    if (use_int4_w4a16 || is_fp8) {
        scale_k_len = B_scale->size(2);
        if (scale_k_len != 1) {
            is_pergroup_scale = true;
            scale_k_group_tile = ceil_div(hidden_size, scale_k_len);
            TORCH_CHECK(scale_k_group_tile == 128 || scale_k_group_tile == 64, "scale_k_group_tile only support 128 or 64");
        }
    }

    int nr_n = use_swigelu ? reduce_size / 2 : reduce_size;

    std::function<void()> launch_kernel;

    BlockConfig configs[] = {
        {8, 16, 0.f, false},
        {16, 8, 0.f, false},
        {32, 4, 0.f, false},
        {4, 32, 0.f, false},
    };

    constexpr int iobit = 128;
    const int bits_of_byte = 8;
    const int vlen = use_int4_w4a16 ?
                      (iobit / 4):
                      (iobit / (torch::elementSize(B.scalar_type()) * bits_of_byte));

    float target_ratio = static_cast<float>(reduce_size) / hidden_size;

    for (auto& config : configs) {
        int load_size = config.block_k * vlen;
        config.valid = (reduce_size % config.block_n == 0) && (hidden_size % load_size == 0) && (load_size % scale_k_group_tile == 0);

        if (config.valid) {
            float block_ratio = static_cast<float>(config.block_n) / config.block_k;
            config.score = 1.0f / (1.0f + fabsf(block_ratio - target_ratio));
        }
    }

    BlockConfig best_config_storage;
    if (current_arch < 300) {
        best_config_storage = {128, 1, -1.0f, false};
    } else {
        best_config_storage = {32, 1, -1.0f, false};
    }
    BlockConfig* best_config = &best_config_storage;

    for (auto& config : configs) {
        if (config.valid && config.score > best_config->score) {
            best_config = &config;
        }
    }

    switch (best_config->block_n) {
        case 4:
            switch (best_config->block_k) {
                case 32: GEN_LAUNCH_KERN(4, 32); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=4");
            }
            break;
        case 8:
            switch (best_config->block_k) {
                case 16: GEN_LAUNCH_KERN(8, 16); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=8");
            }
            break;
        case 16:
            switch (best_config->block_k) {
                case 8: GEN_LAUNCH_KERN(16, 8); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=16");
            }
            break;
        case 32:
            switch (best_config->block_k) {
                case 4: GEN_LAUNCH_KERN(32, 4); break;
                case 1: GEN_LAUNCH_KERN(32, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=32");
            }
            break;
        case 128:
            switch (best_config->block_k) {
                case 1: GEN_LAUNCH_KERN(128, 1); break;
                default: TORCH_CHECK(false, "Unsupported block_k for block_n=128");
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported block configuration");
    }

    launch_kernel();
}
