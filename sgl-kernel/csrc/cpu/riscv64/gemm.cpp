#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include "riscv64/gemm.h"

#include <ATen/core/Tensor.h>

#include <cstdint>

#include "common.h"
#include "vector_helpers.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)

#include <riscv_vector.h>

#include <algorithm>
#include <cstring>

namespace {

template <typename scalar_t>
void pack_weight(scalar_t* packed_w, const scalar_t* orig_w, int64_t N, int64_t K) {
  constexpr int64_t BN = rvv_constants::BLOCK_N;
  int64_t NB = (N + BN - 1) / BN;

  at::parallel_for(0, NB * K, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t nb = i / K;
      const int64_t k = i % K;
      int64_t n_start = nb * BN;
      int64_t n_size = std::min(BN, N - n_start);

      scalar_t* dst = packed_w + nb * K * BN + k * BN;

      if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
        size_t vl = 0;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);
          vfloat16m2_t v_data = __riscv_vlse16_v_f16m2(
              reinterpret_cast<const _Float16*>(orig_w + (n_start + j) * K + k), K * sizeof(_Float16), vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + j), v_data, vl);
        }
        if (n_size < BN) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BN - n_size);
          vfloat16m2_t v_zero = __riscv_vfmv_v_f_f16m2((_Float16)0.0f, vl_pad);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(dst + n_size), v_zero, vl_pad);
        }
#else
        for (int64_t j = 0; j < n_size; ++j)
          dst[j] = orig_w[(n_start + j) * K + k];
        for (int64_t j = n_size; j < BN; ++j)
          dst[j] = static_cast<scalar_t>(0);
#endif
      } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        size_t vl = 0;
        for (int64_t j = 0; j < n_size; j += vl) {
          vl = __riscv_vsetvl_e16m2(n_size - j);
          vuint16m2_t v_data = __riscv_vlse16_v_u16m2(
              reinterpret_cast<const uint16_t*>(orig_w + (n_start + j) * K + k), K * sizeof(uint16_t), vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + j), v_data, vl);
        }
        if (n_size < BN) {
          size_t vl_pad = __riscv_vsetvl_e16m2(BN - n_size);
          vuint16m2_t v_zero = __riscv_vmv_v_x_u16m2(0, vl_pad);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(dst + n_size), v_zero, vl_pad);
        }
      } else {
        for (int64_t j = 0; j < n_size; ++j) {
          dst[j] = orig_w[(n_start + j) * K + k];
        }
        for (int64_t j = n_size; j < BN; ++j) {
          dst[j] = static_cast<scalar_t>(0);
        }
      }
    }
  });
}

template <>
void pack_weight<int8_t>(int8_t* packed_w, const int8_t* orig_w, int64_t N, int64_t K) {
  pack_weight_int8_with_comp(packed_w, orig_w, N, K, rvv_constants::BLOCK_N);
}

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static_assert(BLOCK_M >= 1 && BLOCK_M <= 4, "BLOCK_M must be 1-4");
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
      "tinygemm_kernel_nn: only supports float, Half, and BFloat16");

  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldc,
      int64_t n_size) {
    size_t vl_max = __riscv_vsetvlmax_e32m4();

    // Float accumulator: holds partial sums across k-tiles so the k-tile loop
    // can be placed outside the j (N) loop.  n_size ≤ BLOCK_N (compile-time),
    // so this is at most BLOCK_M × BLOCK_N × 4 B = 1 KB on stack.
    alignas(64) float C_f32[BLOCK_M][BLOCK_N];
    if constexpr (has_bias) {
      for (int m = 0; m < BLOCK_M; ++m)
        std::copy(bias, bias + n_size, C_f32[m]);
    } else {
      std::memset(C_f32, 0, sizeof(C_f32));
    }

    constexpr int64_t UNROLL = (BLOCK_M <= 3) ? 4 : 2;
    constexpr int64_t PREFETCH_DIST = UNROLL * 2;
    // K-tile size: fit A[BLOCK_M, KB] in L1 while all N-tiles are swept (loop order: k_tile → j → k).
    // KB = L1_CACHE_BYTES / (BLOCK_N * sizeof(scalar_t)); BLOCK_N * sizeof = VLEN_bits/8 for fp32, /16 for fp16/bf16.
    constexpr int64_t KB = std::is_same_v<scalar_t, float> ? (rvv_constants::L1_CACHE_BYTES / __riscv_v_fixed_vlen)
                                                           : (rvv_constants::L1_CACHE_BYTES * 2 / __riscv_v_fixed_vlen);

    for (int64_t k_tile = 0; k_tile < K; k_tile += KB) {
      const int64_t k_end = std::min(k_tile + KB, K);

      for (int64_t j = 0; j < n_size; j += (int64_t)vl_max) {
        size_t vl = (j + (int64_t)vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);

        const scalar_t* b_ptr_base = B + j;
        auto load_b_as_f32 = [&](const scalar_t* ptr) -> vfloat32m4_t {
          if constexpr (std::is_same_v<scalar_t, float>) {
            return __riscv_vle32_v_f32m4(ptr, vl);
          } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
            return __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(ptr), vl), vl);
#else
            alignas(64) float tmp[rvv_constants::MAX_VL_ELEMENTS_M4];
            for (size_t i = 0; i < vl; ++i)
              tmp[i] = static_cast<float>(ptr[i]);
            return __riscv_vle32_v_f32m4(tmp, vl);
#endif
          } else {
            return bf16_to_f32m4(reinterpret_cast<const uint16_t*>(ptr), vl);
          }
        };

        // Load partial sums from C_f32 into vector accumulators.
        vfloat32m4_t v_acc0, v_acc1, v_acc2, v_acc3;
        vfloat32m4_t* v_acc[4] = {&v_acc0, &v_acc1, &v_acc2, &v_acc3};
        for (int m = 0; m < BLOCK_M; ++m)
          *v_acc[m] = __riscv_vle32_v_f32m4(C_f32[m] + j, vl);

        int64_t k = k_tile;
        for (; k + UNROLL - 1 < k_end; k += UNROLL) {
          if (k + PREFETCH_DIST < K) __builtin_prefetch(b_ptr_base + (k + PREFETCH_DIST) * BLOCK_N, 0, 1);
          if constexpr (UNROLL == 4) {
            vfloat32m4_t vb0 = load_b_as_f32(b_ptr_base + (k + 0) * BLOCK_N);
            vfloat32m4_t vb1 = load_b_as_f32(b_ptr_base + (k + 1) * BLOCK_N);
            vfloat32m4_t vb2 = load_b_as_f32(b_ptr_base + (k + 2) * BLOCK_N);
            vfloat32m4_t vb3 = load_b_as_f32(b_ptr_base + (k + 3) * BLOCK_N);
            for (int m = 0; m < BLOCK_M; ++m) {
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 0]), vb0, vl);
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 1]), vb1, vl);
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 2]), vb2, vl);
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 3]), vb3, vl);
            }
          } else {
            vfloat32m4_t vb0 = load_b_as_f32(b_ptr_base + (k + 0) * BLOCK_N);
            vfloat32m4_t vb1 = load_b_as_f32(b_ptr_base + (k + 1) * BLOCK_N);
            for (int m = 0; m < BLOCK_M; ++m) {
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 0]), vb0, vl);
              *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k + 1]), vb1, vl);
            }
          }
        }
        for (; k < k_end; ++k) {
          vfloat32m4_t v_b = load_b_as_f32(b_ptr_base + k * BLOCK_N);
          for (int m = 0; m < BLOCK_M; ++m)
            *v_acc[m] = __riscv_vfmacc_vf_f32m4(*v_acc[m], static_cast<float>(A[m * lda + k]), v_b, vl);
        }

        // Write partial sums back to C_f32.
        for (int m = 0; m < BLOCK_M; ++m)
          __riscv_vse32_v_f32m4(C_f32[m] + j, *v_acc[m], vl);
      }
    }

    // Final pass: convert float C_f32 → scalar_t output C.
    for (int m = 0; m < BLOCK_M; ++m) {
      scalar_t* c_ptr = C + m * ldc;
      for (int64_t j = 0; j < n_size; j += (int64_t)vl_max) {
        size_t vl = (j + (int64_t)vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m4(n_size - j);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(C_f32[m] + j, vl);
        if constexpr (std::is_same_v<scalar_t, float>) {
          __riscv_vse32_v_f32m4(c_ptr + j, v, vl);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
#if defined(__riscv_zvfh) || defined(__riscv_v)
          vfloat16m2_t v_f16 = f32m4_to_f16(v, vl);
          __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(c_ptr + j), v_f16, vl);
#else
          alignas(64) float tmp[rvv_constants::MAX_VL_ELEMENTS_M4];
          __riscv_vse32_v_f32m4(tmp, v, vl);
          for (size_t i = 0; i < vl; ++i)
            c_ptr[j + i] = static_cast<scalar_t>(tmp[i]);
#endif
        } else {
          vuint16m2_t v_bf16 = f32m4_to_bf16(v, vl);
          __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(c_ptr + j), v_bf16, vl);
        }
      }
    }
  }
};

// GEMV specialization (M=1) using LMUL=8. BF16/FP16 use UNROLL=4; FP32 uses UNROLL=2.
template <typename scalar_t, bool has_bias, int BLOCK_N>
struct tinygemm_kernel_gemv_m8 {
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
      "tinygemm_kernel_gemv_m8: only supports float, Half, and BFloat16");

  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      const float* __restrict__ bias,
      int64_t K,
      int64_t lda,
      int64_t ldc,
      int64_t n_size) {
    UNUSED(lda);
    UNUSED(ldc);

    const size_t vl_max = __riscv_vsetvlmax_e32m8();

    // Float accumulator: n_size ≤ BLOCK_N (compile-time), max 64 × 4 = 256 B on stack.
    alignas(64) float C_f32[BLOCK_N];
    if constexpr (has_bias) {
      std::copy(bias, bias + n_size, C_f32);
    } else {
      std::memset(C_f32, 0, sizeof(C_f32));
    }

    // UNROLL: FP32 uses m8 B-loads (8 regs each) → cap at 2.
    // BF16/FP16 use m4 B-loads (4 regs each) → UNROLL=4.
    constexpr int64_t UNROLL = std::is_same_v<scalar_t, float> ? 2 : 4;
    constexpr int64_t PREFETCH_DIST = UNROLL * 2;

    constexpr int64_t KB = std::is_same_v<scalar_t, float> ? (rvv_constants::L1_CACHE_BYTES / __riscv_v_fixed_vlen)
                                                           : (rvv_constants::L1_CACHE_BYTES * 2 / __riscv_v_fixed_vlen);

    for (int64_t k_tile = 0; k_tile < K; k_tile += KB) {
      const int64_t k_end = std::min(k_tile + KB, K);

      for (int64_t j = 0; j < n_size; j += (int64_t)vl_max) {
        const size_t vl = (j + (int64_t)vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m8(n_size - j);
        const scalar_t* b_ptr_base = B + j;

        auto load_b_as_f32m8 = [&](const scalar_t* ptr) -> vfloat32m8_t {
          alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
          return load_as_float_m8<scalar_t>(ptr, vl, scratch);
        };

        vfloat32m8_t v_acc = __riscv_vle32_v_f32m8(C_f32 + j, vl);

        int64_t k = k_tile;
        for (; k + UNROLL - 1 < k_end; k += UNROLL) {
          if (k + PREFETCH_DIST < K) __builtin_prefetch(b_ptr_base + (k + PREFETCH_DIST) * BLOCK_N, 0, 1);
          if constexpr (UNROLL == 4) {
            vfloat32m8_t vb0 = load_b_as_f32m8(b_ptr_base + (k + 0) * BLOCK_N);
            vfloat32m8_t vb1 = load_b_as_f32m8(b_ptr_base + (k + 1) * BLOCK_N);
            vfloat32m8_t vb2 = load_b_as_f32m8(b_ptr_base + (k + 2) * BLOCK_N);
            vfloat32m8_t vb3 = load_b_as_f32m8(b_ptr_base + (k + 3) * BLOCK_N);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 0]), vb0, vl);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 1]), vb1, vl);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 2]), vb2, vl);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 3]), vb3, vl);
          } else {
            vfloat32m8_t vb0 = load_b_as_f32m8(b_ptr_base + (k + 0) * BLOCK_N);
            vfloat32m8_t vb1 = load_b_as_f32m8(b_ptr_base + (k + 1) * BLOCK_N);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 0]), vb0, vl);
            v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k + 1]), vb1, vl);
          }
        }
        for (; k < k_end; ++k) {
          vfloat32m8_t v_b = load_b_as_f32m8(b_ptr_base + k * BLOCK_N);
          v_acc = __riscv_vfmacc_vf_f32m8(v_acc, static_cast<float>(A[k]), v_b, vl);
        }

        __riscv_vse32_v_f32m8(C_f32 + j, v_acc, vl);
      }
    }

    // Final store with type conversion.
    alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
    for (int64_t j = 0; j < n_size; j += (int64_t)vl_max) {
      const size_t vl = (j + (int64_t)vl_max <= n_size) ? vl_max : __riscv_vsetvl_e32m8(n_size - j);
      vfloat32m8_t v = __riscv_vle32_v_f32m8(C_f32 + j, vl);
      store_from_float_m8<scalar_t>(C + j, v, vl, scratch);
    }
  }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                \
  tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                          \
      B + nb_start * K,                                            \
      C + mb_start * ldc + nb_start,                               \
      has_bias ? bias + nb_start : nullptr,                        \
      K,                                                           \
      lda,                                                         \
      ldc,                                                         \
      nb_size);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldc) {
  constexpr int64_t BLOCK_M = GEMM_TILE_M;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size) {
        case 1:
          tinygemm_kernel_gemv_m8<scalar_t, has_bias, BLOCK_N>::apply(
              A + mb_start * lda,
              B + nb_start * K,
              C + mb_start * ldc + nb_start,
              has_bias ? bias + nb_start : nullptr,
              K,
              lda,
              ldc,
              nb_size);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(2, BLOCK_N);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NN(3, BLOCK_N);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(4, BLOCK_N);
          break;
        default:
          TORCH_CHECK(false, "Unexpected mb_size: ", mb_size);
      }
    }
  }
}

template <typename scalar_t>
void weight_packed_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const scalar_t* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  constexpr int64_t BLOCK_M = GEMM_TILE_M;
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int64_t mb = i / NB;
        const int64_t nb = i % NB;
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            mat1 + mb_start * mat1_strideM,
            mat2 + nb_start * K,
            out + mb_start * out_strideM + nb_start,
            bias ? bias + nb_start : nullptr,
            mb_size,
            nb_size,
            K,
            mat1_strideM,
            out_strideM);
      }
    });
  });
}

template <typename scalar_t>
void weight_packed_linear_fma_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const float* __restrict__ mat2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM) {
  const size_t vl_max = __riscv_vsetvlmax_e32m4();
  const int64_t NB = div_up(N, (int64_t)vl_max);

  // Parallel over M × NB so all 8 cores are used even when M < num_threads.
  at::parallel_for(0, M * NB, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float linear_row[rvv_constants::MAX_VL_ELEMENTS_M4];
    for (int64_t i = begin; i < end; ++i) {
      const int64_t m = i / NB;
      const int64_t nb = i % NB;
      const int64_t n_start = nb * (int64_t)vl_max;
      size_t vl = __riscv_vsetvl_e32m4(N - n_start);

      const scalar_t* a_row = mat1 + m * mat1_strideM;
      scalar_t* c_row = out + m * out_strideM;

      // Initialize accumulator with bias or zero
      vfloat32m4_t v_acc =
          (bias != nullptr) ? __riscv_vle32_v_f32m4(bias + n_start, vl) : __riscv_vfmv_v_f_f32m4(0.0f, vl);

      // K-loop: broadcast A scalar, load contiguous floats from B row
      for (int64_t k = 0; k < K; ++k) {
        const float a_val = static_cast<float>(a_row[k]);
        const float* b_row = mat2 + k * N + n_start;
        v_acc = __riscv_vfmacc_vf_f32m4(v_acc, a_val, __riscv_vle32_v_f32m4(b_row, vl), vl);
      }

      __riscv_vse32_v_f32m4(linear_row, v_acc, vl);
      copy_stub(c_row + n_start, linear_row, vl);
    }
  });
}

}  // anonymous namespace

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
    bool brg) {
  UNUSED(Ctmp);
  UNUSED(ldb);
  TORCH_CHECK(!brg, "RVV does not use brgemm");
  // Forward to internal implementation (no bias)
  tinygemm_kernel<scalar_t, false>(A, B, C, nullptr, M, N, K, lda, ldc);
}

#define INSTANTIATE_TINYGEMM_TEMPLATE(TYPE) \
  template void tinygemm_kernel<TYPE>(      \
      const TYPE* __restrict__ A,           \
      const TYPE* __restrict__ B,           \
      TYPE* __restrict__ C,                 \
      float* __restrict__ Ctmp,             \
      int64_t M,                            \
      int64_t N,                            \
      int64_t K,                            \
      int64_t lda,                          \
      int64_t ldb,                          \
      int64_t ldc,                          \
      bool brg)

INSTANTIATE_TINYGEMM_TEMPLATE(float);
INSTANTIATE_TINYGEMM_TEMPLATE(at::BFloat16);
INSTANTIATE_TINYGEMM_TEMPLATE(at::Half);

at::Tensor convert_weight_packed(at::Tensor& weight) {
  CHECK_INPUT(weight);

  TORCH_CHECK(weight.ndimension() == 2, "expect weight to be 2d, got ", weight.ndimension(), "d tensor.");

  constexpr int64_t BLOCK_N_RVV = rvv_constants::BLOCK_N;
  // data [K, BLOCK_N] followed by one int32 compensation per output channel.
  // The float32-transpose fallback is only valid for FP types.
  const bool is_int8 = (weight.scalar_type() == at::kChar);
  if (!is_int8 && (weight.size(0) < BLOCK_N_RVV || weight.size(0) % BLOCK_N_RVV != 0)) {
    // Small OC shape: use fma linear path, which needs transpose not pack.
    return weight.to(at::kFloat).t().contiguous();
  }

  const auto st = weight.scalar_type();
  const int64_t OC = weight.size(0);
  const int64_t IC = weight.size(1);
  const int64_t NB = div_up(OC, BLOCK_N_RVV);

  TORCH_CHECK(
      st == at::kBFloat16 || st == at::kHalf || st == at::kChar, "expect weight to be bfloat16, float16, int8 ");

  auto packed_weight = at::empty({}, weight.options());

  CPU_DISPATCH_PACKED_TYPES_RVV(st, [&] {
    const int64_t packed_block_size =
        std::is_same_v<packed_t, int8_t> ? get_int8_packed_block_size(IC) : IC * BLOCK_N_RVV;
    packed_weight.resize_({NB, packed_block_size});
    packed_t* packed_data = packed_weight.data_ptr<packed_t>();
    const packed_t* w_data = weight.data_ptr<packed_t>();

    at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
      for (int64_t nb = begin; nb < end; ++nb) {
        int64_t n = nb * BLOCK_N_RVV;
        int64_t n_size = std::min(BLOCK_N_RVV, OC - n);
        pack_weight<packed_t>(packed_data + nb * packed_block_size, w_data + n * IC, n_size, IC);
      }
    });
  });
  return packed_weight;
}

at::Tensor
weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2, const std::optional<at::Tensor>& bias, bool is_packed) {
  RECORD_FUNCTION("sgl-kernel::weight_packed_linear", std::vector<c10::IValue>({mat1, mat2, bias}));

  auto packed_w = is_packed ? mat2 : convert_weight_packed(mat2);
  bool use_fma_gemm = false;
  if (packed_w.scalar_type() == at::kFloat) {
    use_fma_gemm = true;
  }

  int64_t M = mat1.size(0);
  int64_t K = mat1.size(1);
  int64_t N;
  if (use_fma_gemm) {
    N = packed_w.size(1);
  } else {
    if (is_packed) {
      TORCH_CHECK(
          packed_w.scalar_type() != at::kChar,
          "weight_packed_linear: packed int8 weights are not supported through this API; ",
          "use int8_scaled_mm_cpu/int8_scaled_mm_with_quant for RVV W8A8.");
      // RVV packed int8 weight: [NB, BLOCK_N * (IC + sizeof(int32_t))].
      // Floating-point packed weight remains [NB, IC * BLOCK_N].
      TORCH_CHECK(packed_w.dim() == 2, "RVV packed weight must be 2D");
      N = packed_w.size(0) * rvv_constants::BLOCK_N;
    } else {
      N = mat2.size(0);
    }
  }

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_DIM(2, mat1);
  if (!is_packed) {
    CHECK_DIM(2, mat2);
    if (!use_fma_gemm) {
      CHECK_EQ(mat1.size(1), mat2.size(1));
    }
  }

  if (use_fma_gemm) {
    TORCH_CHECK(
        packed_w.scalar_type() == at::kFloat,
        "weight_packed_linear: float packed weights required for FMA path, got ",
        packed_w.scalar_type());
  } else {
    TORCH_CHECK(
        packed_w.scalar_type() == mat1.scalar_type(),
        "weight_packed_linear: mat1 and weight must have the same dtype for RVV packed GEMM, got mat1=",
        mat1.scalar_type(),
        ", weight=",
        packed_w.scalar_type());
  }
  if (is_packed && !use_fma_gemm && packed_w.scalar_type() != at::kChar) {
    TORCH_CHECK(
        packed_w.size(1) == K * rvv_constants::BLOCK_N,
        "weight_packed_linear: packed floating weight must have shape [NB, K * BLOCK_N], expected second dim ",
        K * rvv_constants::BLOCK_N,
        ", got ",
        packed_w.size(1));
  }

  auto dispatch_type = mat1.scalar_type();
  auto out = at::empty({M, N}, mat1.options());
  int64_t out_strideM = out.stride(0);
  int64_t mat1_strideM = mat1.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_INPUT(bias.value());
    CHECK_DIM(1, bias.value());
    CHECK_EQ(bias.value().size(0), N);
    TORCH_CHECK(
        bias.value().scalar_type() == at::kFloat,
        "weight_packed_linear: bias must be float32, got ",
        bias.value().scalar_type());
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_RVV_TYPES(dispatch_type, "weight_packed_linear_kernel_impl", [&] {
    if (use_fma_gemm) {
      weight_packed_linear_fma_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<float>(),
          bias_data,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    } else {
      weight_packed_linear_kernel_impl<scalar_t>(
          out.data_ptr<scalar_t>(),
          mat1.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          bias_data,
          M,
          N,
          K,
          mat1_strideM,
          out_strideM);
    }
  });

  return out;
}

#endif  // CPU_CAPABILITY_RVV
