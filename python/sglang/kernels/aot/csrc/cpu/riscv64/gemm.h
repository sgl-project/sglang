#pragma once

#include <ATen/core/Tensor.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "riscv64/vector_helpers.h"

// 4 m4-accumulators × 4 phys-regs = 16 of 32 VRs; leaves 16 for operands.
static constexpr int GEMM_TILE_M = 4;

// Weight packing (RVV block-N format)
at::Tensor convert_weight_packed(at::Tensor& weight);

inline int64_t get_int8_packed_row_size(int64_t K) {
  return K + static_cast<int64_t>(sizeof(int32_t));
}

inline int64_t get_int8_packed_block_size(int64_t K) {
  return rvv_constants::BLOCK_N * get_int8_packed_row_size(K);
}

// GEMM Kernel Declarations

template <typename scalar_t>
void int8_scaled_mm_kernel(
    scalar_t* __restrict__ out,
    const uint8_t* __restrict__ mat1,
    const int8_t* __restrict__ mat2,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<float>(
    float* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<at::Half>(
    at::Half* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

extern template void int8_scaled_mm_kernel<at::BFloat16>(
    at::BFloat16* out,
    const uint8_t* mat1,
    const int8_t* mat2,
    const float* scales1,
    const float* scales2,
    const float* bias,
    int64_t M,
    int64_t N,
    int64_t K,
    bool is_packed);

// TinyGEMM Interface for RVV
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

// TinyGEMM Interface for INT8 (RVV W8A8) - activation is uint8, weight is int8.
template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

// Inline Attention Kernels (Template Implementations)

#if defined(CPU_CAPABILITY_RVV)
#include <riscv_vector.h>

#include "vector_math.h"

template <typename scalar_t>
inline void gemm_nt_tiled_transposed(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_trans,
    float* __restrict__ C,
    int M,
    int N,
    int head_size,
    int q_strideM,
    int block_n,
    int ldc,
    float scale) {
  size_t vl_max = __riscv_vsetvlmax_e32m4();

  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);

    for (int n_base = 0; n_base < N; n_base += vl_max) {
      size_t vl = __riscv_vsetvl_e32m4(N - n_base);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int k = 0; k < head_size; ++k) {
        const scalar_t* k_ptr = K_trans + k * block_n + n_base;

        vfloat16m2_t v_k_f16;
        vfloat32m4_t v_k_f32;

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          v_k_f16 = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(k_ptr), vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          v_k_f32 = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(k_ptr), vl);
        } else {
          // scalar_t=float: identity cast, load directly
          v_k_f32 = __riscv_vle32_v_f32m4(reinterpret_cast<const float*>(k_ptr), vl);
        }

        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (m_count > 0) q0 = static_cast<float>(Q[(m_base + 0) * q_strideM + k]);
        if (m_count > 1) q1 = static_cast<float>(Q[(m_base + 1) * q_strideM + k]);
        if (m_count > 2) q2 = static_cast<float>(Q[(m_base + 2) * q_strideM + k]);
        if (m_count > 3) q3 = static_cast<float>(Q[(m_base + 3) * q_strideM + k]);

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)q0, v_k_f16, vl);
          if (m_count > 1) acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)q1, v_k_f16, vl);
          if (m_count > 2) acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)q2, v_k_f16, vl);
          if (m_count > 3) acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)q3, v_k_f16, vl);
        } else {
          acc0 = __riscv_vfmacc_vf_f32m4(acc0, q0, v_k_f32, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, q1, v_k_f32, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, q2, v_k_f32, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, q3, v_k_f32, vl);
        }
      }

      auto store = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          __riscv_vse32_v_f32m4(C + (m_base + idx) * ldc + n_base, __riscv_vfmul_vf_f32m4(acc, scale, vl), vl);
        }
      };

      store(0, acc0);
      store(1, acc1);
      store(2, acc2);
      store(3, acc3);
    }
  }
}

// NOTE: O must be zero-initialized by the caller before the first call.
// This function accumulates (load-add-store) onto O across tiled K iterations.
template <typename scalar_t>
inline void gemm_nn_tiled(
    const float* __restrict__ P,
    const scalar_t* __restrict__ V,
    float* __restrict__ O,
    int M,
    int N,
    int head_size_v,
    int p_strideN,
    int v_strideH) {
  for (int m_base = 0; m_base < M; m_base += GEMM_TILE_M) {
    int m_count = std::min(GEMM_TILE_M, M - m_base);
    size_t vl;
    for (int d = 0; d < head_size_v; d += vl) {
      vl = __riscv_vsetvl_e32m4(head_size_v - d);

      vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
      vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      for (int n = 0; n < N; ++n) {
        const scalar_t* v_ptr = V + n * v_strideH + d;

        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        if (m_count > 0) p0 = P[(m_base + 0) * p_strideN + n];
        if (m_count > 1) p1 = P[(m_base + 1) * p_strideN + n];
        if (m_count > 2) p2 = P[(m_base + 2) * p_strideN + n];
        if (m_count > 3) p3 = P[(m_base + 3) * p_strideN + n];

        if constexpr (std::is_same_v<scalar_t, at::Half>) {
          vfloat16m2_t v_v = __riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(v_ptr), vl);

          acc0 = vfwmacc_f16_scalar_to_f32m4(acc0, (_Float16)p0, v_v, vl);
          if (m_count > 1) acc1 = vfwmacc_f16_scalar_to_f32m4(acc1, (_Float16)p1, v_v, vl);
          if (m_count > 2) acc2 = vfwmacc_f16_scalar_to_f32m4(acc2, (_Float16)p2, v_v, vl);
          if (m_count > 3) acc3 = vfwmacc_f16_scalar_to_f32m4(acc3, (_Float16)p3, v_v, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
          vfloat32m4_t v_v = bf16_to_f32m4(reinterpret_cast<const uint16_t*>(v_ptr), vl);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v, vl);
        } else {
          // scalar_t=float: identity cast, load directly
          vfloat32m4_t v_v = __riscv_vle32_v_f32m4(reinterpret_cast<const float*>(v_ptr), vl);

          acc0 = __riscv_vfmacc_vf_f32m4(acc0, p0, v_v, vl);
          if (m_count > 1) acc1 = __riscv_vfmacc_vf_f32m4(acc1, p1, v_v, vl);
          if (m_count > 2) acc2 = __riscv_vfmacc_vf_f32m4(acc2, p2, v_v, vl);
          if (m_count > 3) acc3 = __riscv_vfmacc_vf_f32m4(acc3, p3, v_v, vl);
        }
      }

      auto store_o = [&](int idx, vfloat32m4_t acc) {
        if (idx < m_count) {
          float* o_ptr = O + (m_base + idx) * head_size_v + d;
          vfloat32m4_t old_o = __riscv_vle32_v_f32m4(o_ptr, vl);
          acc = __riscv_vfadd_vv_f32m4(old_o, acc, vl);
          __riscv_vse32_v_f32m4(o_ptr, acc, vl);
        }
      };

      store_o(0, acc0);
      store_o(1, acc1);
      store_o(2, acc2);
      store_o(3, acc3);
    }
  }
}

#endif  // CPU_CAPABILITY_RVV
