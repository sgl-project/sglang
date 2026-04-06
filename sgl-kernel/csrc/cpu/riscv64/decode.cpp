#if defined(CPU_CAPABILITY_RVV) && defined(__clang__)
#pragma clang riscv intrinsic vector
#endif

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "common.h"
#include "riscv64/gemm.h"
#include "vector_helpers.h"
#include "vector_math.h"

#if defined(CPU_CAPABILITY_RVV)

namespace {

// Generic scalar fallback for tinygemm_kernel_nt: computes Q(M,K) @ B(N,K) → C(M,N).
template <typename scalar_t, typename kv_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const kv_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    for (int64_t m = 0; m < BLOCK_M; ++m) {
      for (int64_t n = 0; n < BLOCK_N; ++n) {
        float sum = 0.f;
        int64_t b_idx = indices[n];
        TORCH_CHECK(b_idx < max_tokens, "token index out of scope!");
        for (int64_t k = 0; k < K; ++k) {
          float a_val = static_cast<float>(A[m * lda + k]);
          float b_val = static_cast<float>(B[b_idx * ldb + k]);
          sum += scale * a_val * b_val;
        }
        C[m * ldc + n] = sum;
      }
    }
  }
};

template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt<at::BFloat16, at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N;

    size_t vl_max = __riscv_vsetvlmax_e32m1();

    vf32m1_t vc[ROWS * COLS];

    // Initialize accumulators with zero
    auto init_c = [&](auto i) { vc[i] = __riscv_vfmv_v_f_f32m1(0.0f, vl_max); };
    Unroll<ROWS * COLS>{}(init_c);

    // Validate indices once
    for (int n = 0; n < COLS; ++n) {
      TORCH_CHECK(indices[n] < max_tokens, "token index out of scope!");
    }

    // Prefetch first 2 tokens before K loop (cache warmup)
    if (COLS >= 1) __builtin_prefetch(B + indices[0] * ldb, 0, 3);
    if (COLS >= 2) __builtin_prefetch(B + indices[1] * ldb, 0, 3);

    size_t vl;
    for (int64_t k = 0; k < K; k += vl) {
      vl = __riscv_vsetvl_e32m1(K - k);

      vf32m1_t vb[COLS];
      for (int n = 0; n < COLS; ++n) {
        int64_t b_idx = indices[n];
        vb[n] = bf16_to_f32m1(reinterpret_cast<const uint16_t*>(B + b_idx * ldb + k), vl);

        if (k + vl < K) {
          __builtin_prefetch(B + b_idx * ldb + k + vl, 0, 1);
        }
        if (k == 0 && n + 2 < COLS) {
          __builtin_prefetch(B + indices[n + 2] * ldb, 0, 2);
        }
      }

      for (int m = 0; m < ROWS; ++m) {
        vfloat32m1_t va = bf16_to_f32m1(reinterpret_cast<const uint16_t*>(A + m * lda + k), vl);
        for (int n = 0; n < COLS; ++n) {
          vc[m * COLS + n] = __riscv_vfmacc_vv_f32m1_tu(vc[m * COLS + n], va, vb[n], vl);
        }
      }
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      C[row * ldc + col] = reduce_sum_f32m1(vc[i], vl_max) * scale;
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};

#if defined(__riscv_zvfh)
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nt<at::Half, at::Half, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const at::Half* __restrict__ A,
      const at::Half* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      float scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N;

    size_t vl_max = __riscv_vsetvlmax_e32m1();

    vf32m1_t vc[ROWS * COLS];

    // Zero init accumulators
    auto init_c = [&](auto i) { vc[i] = __riscv_vfmv_v_f_f32m1(0.0f, vl_max); };
    Unroll<ROWS * COLS>{}(init_c);

    // Validate indices once
    for (int n = 0; n < COLS; ++n) {
      TORCH_CHECK(indices[n] < max_tokens, "token index out of scope!");
    }

    // Prefetch first 2 tokens before K loop (cache warmup)
    if (COLS >= 1) __builtin_prefetch(B + indices[0] * ldb, 0, 3);
    if (COLS >= 2) __builtin_prefetch(B + indices[1] * ldb, 0, 3);

    size_t vl;
    for (int64_t k = 0; k < K; k += vl) {
      vl = __riscv_vsetvl_e32m1(K - k);

      vf32m1_t vb[COLS];
      for (int n = 0; n < COLS; ++n) {
        int64_t b_idx = indices[n];
        vfloat16mf2_t vb_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(B + b_idx * ldb + k), vl);
        vb[n] = __riscv_vfwcvt_f_f_v_f32m1(vb_f16, vl);

        if (k + vl < K) {
          __builtin_prefetch(B + b_idx * ldb + k + vl, 0, 1);
        }
        if (k == 0 && n + 2 < COLS) {
          __builtin_prefetch(B + indices[n + 2] * ldb, 0, 2);
        }
      }

      for (int m = 0; m < ROWS; ++m) {
        vfloat16mf2_t va_f16 = __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(A + m * lda + k), vl);
        vfloat32m1_t va = __riscv_vfwcvt_f_f_v_f32m1(va_f16, vl);

        for (int n = 0; n < COLS; ++n) {
          vc[m * COLS + n] = __riscv_vfmacc_vv_f32m1_tu(vc[m * COLS + n], va, vb[n], vl);
        }
      }
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      C[row * ldc + col] = reduce_sum_f32m1(vc[i], vl_max) * scale;
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif  // __riscv_zvfh

#define LAUNCH_TINYGEMM_KERNEL_NT(MB_SIZE, NB_SIZE)                        \
  tinygemm_kernel_nt<scalar_t, kv_type, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda, B, C + mb_start * ldc + nb_start, indices + nb_start, scale, lda, ldb, ldc, K, max_tokens);

// Kernel: Value Aggregate (GEMM NN)
// GEMM handles v' * scale + attn @ value (indexed)
//   A : [M, K]
//   B : [K, N] indexed
//   C : [M, N]

template <typename scalar_t, typename kv_t, typename index_t>
inline void tinygemm_kernel_nn_scalar(
    const float* __restrict__ A,
    const kv_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  // Validate all V-cache token indices once before entering the computation loops.
  for (int64_t k = 0; k < K; ++k) {
    TORCH_CHECK(indices[k] < max_tokens, "token index out of scope!");
  }
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      C[m * ldc + n] *= scale[m];
      for (int64_t k = 0; k < K; ++k) {
        int64_t b_idx = indices[k];
        float b_val = static_cast<float>(B[b_idx * ldb + n]);
        C[m * ldc + n] += A[m * lda + k] * b_val;
      }
    }
  }
}
// GEMM handles v' * scale + attn @ value (indexed)
//   A : [M, K]
//   B : [K, N] indexed
//   C : [M, N]
//
// NOTE: C must be zero-initialized by the caller before the first call.
// This kernel accumulates: C = C * scale + A @ B_indexed.
// Callers must ensure zero-init via init_v_prime_split().
//
// Generic scalar fallback: delegates to tinygemm_kernel_nn_scalar.
template <typename scalar_t, typename kv_t, typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
      const float* __restrict__ A,
      const kv_t* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    tinygemm_kernel_nn_scalar<scalar_t, kv_t, index_t>(
        A, B, C, indices, scale, BLOCK_M, BLOCK_N, K, lda, ldb, ldc, max_tokens);
  }
};

// RVV-optimized specialization for BFloat16
template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, at::BFloat16, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const float* __restrict__ A,
      const at::BFloat16* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = 4;  // 4 m1 vectors per tile

    size_t vl_max = __riscv_vsetvlmax_e32m1();
    int64_t tile_n = COLS * vl_max;

    // Validate all V-cache token indices once before entering the tile loop.
    for (int64_t k = 0; k < K; ++k) {
      TORCH_CHECK(indices[k] < max_tokens, "token index out of scope!");
    }

    // Outer loop: Tiling over N
    for (int64_t nb = 0; nb < BLOCK_N; nb += tile_n) {
      vf32m1_t vc[ROWS * COLS];
      size_t vls[COLS];

      // 1. Pre-calculate VLs for this tile
      for (int col = 0; col < COLS; ++col) {
        int64_t n_curr = nb + col * vl_max;
        if (n_curr < BLOCK_N) {
          vls[col] = __riscv_vsetvl_e32m1(BLOCK_N - n_curr);
        } else {
          vls[col] = 0;
        }
      }

      // 2. Load C and Scale
      for (int m = 0; m < ROWS; ++m) {
        float s = scale[m];
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            vc[m * COLS + col] = __riscv_vle32_v_f32m1(C + m * ldc + n_curr, vls[col]);
            vc[m * COLS + col] = __riscv_vfmul_vf_f32m1(vc[m * COLS + col], s, vls[col]);
          }
        }
      }

      // 3. Main Computation Loop (K)
      for (int64_t k = 0; k < K; ++k) {
        int64_t b_idx = indices[k];

        // Prefetch next B row
        if (k + 1 < K) {
          __builtin_prefetch(B + indices[k + 1] * ldb + nb, 0, 1);
        }

        // 3a. Prepare B vectors (Load & Convert BF16 -> FP32)
        vf32m1_t vb[COLS];
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            // Helper from vector_helpers.h: bf16_to_f32m1
            vb[col] = bf16_to_f32m1(reinterpret_cast<const uint16_t*>(B + b_idx * ldb + n_curr), vls[col]);
          }
        }

        // 3b. FMA for each row
        for (int m = 0; m < ROWS; ++m) {
          float va = A[m * lda + k];
          for (int col = 0; col < COLS; ++col) {
            if (vls[col] > 0) {
              vc[m * COLS + col] = __riscv_vfmacc_vf_f32m1(vc[m * COLS + col], va, vb[col], vls[col]);
            }
          }
        }
      }

      // 4. Store Results
      for (int m = 0; m < ROWS; ++m) {
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            __riscv_vse32_v_f32m1(C + m * ldc + n_curr, vc[m * COLS + col], vls[col]);
          }
        }
      }
    }
  }
};

template <typename index_t, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::Half, at::Half, index_t, BLOCK_M, BLOCK_N> {
  static inline void apply(
      const float* __restrict__ A,
      const at::Half* __restrict__ B,
      float* __restrict__ C,
      const index_t* __restrict__ indices,
      const float* __restrict__ scale,
      int64_t lda,
      int64_t ldb,
      int64_t ldc,
      int64_t K,
      int64_t max_tokens) {
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = 4;

    size_t vl_max = __riscv_vsetvlmax_e32m1();
    int64_t tile_n = COLS * vl_max;

    // Validate all V-cache token indices once before entering the tile loop.
    for (int64_t k = 0; k < K; ++k) {
      TORCH_CHECK(indices[k] < max_tokens, "token index out of scope!");
    }

    // Outer loop: Tiling over N
    for (int64_t nb = 0; nb < BLOCK_N; nb += tile_n) {
      vf32m1_t vc[ROWS * COLS];
      size_t vls[COLS];

      // 1. Pre-calculate VLs
      for (int col = 0; col < COLS; ++col) {
        int64_t n_curr = nb + col * vl_max;
        if (n_curr < BLOCK_N) {
          vls[col] = __riscv_vsetvl_e32m1(BLOCK_N - n_curr);
        } else {
          vls[col] = 0;
        }
      }

      // 2. Load C and Scale
      for (int m = 0; m < ROWS; ++m) {
        float s = scale[m];
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            vc[m * COLS + col] = __riscv_vle32_v_f32m1(C + m * ldc + n_curr, vls[col]);
            vc[m * COLS + col] = __riscv_vfmul_vf_f32m1(vc[m * COLS + col], s, vls[col]);
          }
        }
      }

      // 3. Main Computation Loop (K)
      for (int64_t k = 0; k < K; ++k) {
        int64_t b_idx = indices[k];

        // Prefetch next B row
        if (k + 1 < K) {
          __builtin_prefetch(B + indices[k + 1] * ldb + nb, 0, 1);
        }

        // 3a. Prepare B vectors (Load FP16 -> Convert FP32)
        vf32m1_t vb[COLS];
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            vfloat16mf2_t vb_f16 =
                __riscv_vle16_v_f16mf2(reinterpret_cast<const _Float16*>(B + b_idx * ldb + n_curr), vls[col]);
            vb[col] = __riscv_vfwcvt_f_f_v_f32m1(vb_f16, vls[col]);
          }
        }

        // 3b. FMA for each row
        for (int m = 0; m < ROWS; ++m) {
          float a_val = A[m * lda + k];
          for (int col = 0; col < COLS; ++col) {
            if (vls[col] > 0) {
              vc[m * COLS + col] = __riscv_vfmacc_vf_f32m1(vc[m * COLS + col], a_val, vb[col], vls[col]);
            }
          }
        }
      }

      // 4. Store Results
      for (int m = 0; m < ROWS; ++m) {
        for (int col = 0; col < COLS; ++col) {
          if (vls[col] > 0) {
            int64_t n_curr = nb + col * vl_max;
            __riscv_vse32_v_f32m1(C + m * ldc + n_curr, vc[m * COLS + col], vls[col]);
          }
        }
      }
    }
  }
};

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                        \
  tinygemm_kernel_nn<scalar_t, kv_type, index_t, MB_SIZE, NB_SIZE>::apply( \
      A + mb_start * lda,                                                  \
      B + nb_start,                                                        \
      C + mb_start * ldc + nb_start,                                       \
      indices,                                                             \
      scale + mb_start,                                                    \
      lda,                                                                 \
      ldb,                                                                 \
      ldc,                                                                 \
      K,                                                                   \
      max_tokens);

template <typename scalar_t, typename kv_t, typename index_t>
void index_gemm_kernel_nt(
    const scalar_t* __restrict__ A,
    const kv_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  using kv_type = kv_t;
  // pattern: 1-8-8 for decode (M==1)
  if (M == 1) {
    constexpr int64_t BLOCK_N = 8;
    const int64_t NB = div_up(N, BLOCK_N);
    // M==1: mb_start=0, so A+mb_start*lda==A and C+mb_start*ldc==C regardless
    // of lda/ldc; the function-scope parameters are used directly via the macro.
    int64_t mb_start = 0;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 7);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 8);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", nb_size);
      }
    }
    return;
  }
  // Non-Half (BF16/FP32): BLOCK_M=4, BLOCK_N=6; dispatches 4x6=24 (mb,nb) tile cases.
  // Half (FP16):          BLOCK_M=4, BLOCK_N=4; dispatches 4x4=16 (mb,nb) tile cases.
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = std::is_same_v<scalar_t, at::Half> ? 4 : 6;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (mb_size << 4 | nb_size) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 1);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 2);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 3);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 4);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 5);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NT(1, 6);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 1);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 2);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 3);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 4);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 5);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NT(2, 6);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 1);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 2);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 3);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 4);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 5);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NT(3, 6);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 1);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 2);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 3);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 4);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 5);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NT(4, 6);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", nb_size);
      }
    }
  }
}

template <typename scalar_t, typename kv_t, typename index_t>
void index_gemm_kernel_nn(
    const float* __restrict__ A,
    const kv_t* __restrict__ B,
    float* __restrict__ C,
    const index_t* __restrict__ indices,
    float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t max_tokens) {
  using kv_type = kv_t;
  // kVecSize = vl_max_e32m1 = VLEN/32.
  // __riscv_v_fixed_vlen is a compile-time constant injected by -mrvv-vector-bits=N,
  constexpr int kVecSize = static_cast<int>(__riscv_v_fixed_vlen) / 32;
  constexpr int kVecShift = __builtin_ctz(static_cast<unsigned int>(kVecSize));
  // N must be a multiple of kVecSize; non-aligned sizes fall through to scalar path.
  if ((N & (kVecSize - 1)) != 0) {
    tinygemm_kernel_nn_scalar<scalar_t, kv_t, index_t>(A, B, C, indices, scale, M, N, K, lda, ldb, ldc, max_tokens);
    return;
  }

  // pattern: 1-8-8 for decode (M==1)
  if (M == 1) {
    // BLOCK_N = 8 * kVecSize: 8 full vl_max-wide strips per tile.
    constexpr int64_t BLOCK_N = 8 * kVecSize;
    const int64_t NB = div_up(N, BLOCK_N);
    // M==1: mb_start=0, so A+mb_start*lda==A and C+mb_start*ldc==C regardless
    // of lda/ldc; the function-scope parameters are used directly via the macro.
    int64_t mb_start = 0;

    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch (nb_size >> kVecShift) {
        case 1:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 1);
          break;
        case 2:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 2);
          break;
        case 3:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 3);
          break;
        case 4:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 4);
          break;
        case 5:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 5);
          break;
        case 6:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 6);
          break;
        case 7:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 7);
          break;
        case 8:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 8);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, 1x", nb_size);
      }
    }
    return;
  }

  // Batch pattern: BLOCK_M x BLOCK_N tiling.
  // BLOCK_N = 6 * kVecSize: 6 full vl_max-wide strips per tile.
  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 6 * kVecSize;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  for (int64_t mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      // Upper nibble: mb_size (1–4); lower nibble: nb_size / kVecSize (1–6).
      switch (mb_size << 4 | nb_size >> kVecShift) {
        // mb_size = 1
        case 0x11:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 1);
          break;
        case 0x12:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 2);
          break;
        case 0x13:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 3);
          break;
        case 0x14:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 4);
          break;
        case 0x15:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 5);
          break;
        case 0x16:
          LAUNCH_TINYGEMM_KERNEL_NN(1, kVecSize * 6);
          break;
        // mb_size = 2
        case 0x21:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 1);
          break;
        case 0x22:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 2);
          break;
        case 0x23:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 3);
          break;
        case 0x24:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 4);
          break;
        case 0x25:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 5);
          break;
        case 0x26:
          LAUNCH_TINYGEMM_KERNEL_NN(2, kVecSize * 6);
          break;
        // mb_size = 3
        case 0x31:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 1);
          break;
        case 0x32:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 2);
          break;
        case 0x33:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 3);
          break;
        case 0x34:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 4);
          break;
        case 0x35:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 5);
          break;
        case 0x36:
          LAUNCH_TINYGEMM_KERNEL_NN(3, kVecSize * 6);
          break;
        // mb_size = 4
        case 0x41:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 1);
          break;
        case 0x42:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 2);
          break;
        case 0x43:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 3);
          break;
        case 0x44:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 4);
          break;
        case 0x45:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 5);
          break;
        case 0x46:
          LAUNCH_TINYGEMM_KERNEL_NN(4, kVecSize * 6);
          break;
        default:
          TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", nb_size);
      }
    }
  }
}

template <typename scalar_t>
void decode_set_kv_buffer(
    scalar_t* __restrict__ k_buffer,
    scalar_t* __restrict__ v_buffer,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ loc,
    int64_t batches,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    int64_t nk_strideN,
    int64_t nk_strideH,
    int64_t nv_strideN,
    int64_t nv_strideH) {
  at::parallel_for(0, batches * num_heads_kv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv);

    for (int64_t i = begin; i < end; i++) {
      int64_t loc_val = loc[bs];
      scalar_t* k_buffer_ptr = k_buffer + loc_val * k_strideN + head_kv_id * k_strideH;
      const scalar_t* new_key_ptr = key + bs * nk_strideN + head_kv_id * nk_strideH;

      if (i + 1 < end) {
        int64_t next_bs = bs, next_head = head_kv_id;
        data_index_step(next_bs, batches, next_head, num_heads_kv);
        __builtin_prefetch(key + next_bs * nk_strideN + next_head * nk_strideH, 0, 1);
        __builtin_prefetch(value + next_bs * nv_strideN + next_head * nv_strideH, 0, 1);
      }

      copy_stub<scalar_t>(k_buffer_ptr, new_key_ptr, head_size);
      scalar_t* v_buffer_ptr = v_buffer + loc_val * v_strideN + head_kv_id * v_strideH;
      const scalar_t* new_value_ptr = value + bs * nv_strideN + head_kv_id * nv_strideH;
      copy_stub<scalar_t>(v_buffer_ptr, new_value_ptr, head_size_v);
      data_index_step(bs, batches, head_kv_id, num_heads_kv);
    }
  });
}

// Initialize a v_prime split slot: zero the value accumulator and write the
// -inf sentinel so decode_accumulate_kv_splits can detect empty splits.
inline void init_v_prime_split(float* __restrict__ vp, int64_t head_size_v) {
  fill_stub(vp, 0.f, head_size_v);
  vp[head_size_v] = -std::numeric_limits<float>::infinity();
}

template <typename scalar_t>
inline void decode_accumulate_kv_splits(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    int64_t batches,
    int64_t num_heads,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t l_stride1,
    int64_t l_stride2) {
  at::parallel_for(0, batches * num_heads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      float* __restrict__ acc = attn_logits + i * l_stride1;

      float s_prime = 0.f;
      float m_prime = -std::numeric_limits<float>::infinity();

      for (int64_t kv_id = 0; kv_id < num_kv_splits; ++kv_id) {
        float* __restrict__ tv = acc + kv_id * l_stride2;
        const float tlogic = tv[head_size_v];

        float m_i = std::max(tlogic, m_prime);
        float m_delta = expf(m_prime - m_i);
        float e_logic = expf(tlogic - m_i);

        if (kv_id != 0) {
          size_t vl_max = __riscv_vsetvlmax_e32m8();
          for (int64_t d = 0; d < head_size_v; d += vl_max) {
            size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + d, vl);
            vfloat32m8_t vtv = __riscv_vle32_v_f32m8(tv + d, vl);

            vacc = __riscv_vfmul_vf_f32m8(vacc, m_delta, vl);
            vacc = __riscv_vfmacc_vf_f32m8(vacc, e_logic, vtv, vl);

            __riscv_vse32_v_f32m8(acc + d, vacc, vl);
          }
        }

        s_prime = s_prime * m_delta + e_logic;
        m_prime = m_i;
      }

      float scale = (s_prime > 0.0f && std::isfinite(s_prime)) ? (1.0f / s_prime) : 0.0f;

      size_t vl_max = __riscv_vsetvlmax_e32m8();
      alignas(64) float scratch[rvv_constants::MAX_VL_ELEMENTS_M8];
      // Assumes contiguous output: stride[0]=num_heads*head_size_v, stride[1]=head_size_v
      scalar_t* o_ptr = output + i * head_size_v;
      for (int64_t d = 0; d < head_size_v; d += vl_max) {
        size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(acc + d, vl);
        vacc = __riscv_vfmul_vf_f32m8(vacc, scale, vl);

        store_from_float_m8(o_ptr + d, vacc, vl, scratch);
      }
    }
  });
}

template <typename scalar_t, typename kv_t, typename index_t>
void decode_attention_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const kv_t* __restrict__ k_buffer,
    const kv_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    int64_t o_strideM,
    int64_t o_strideH) {
  // Strides for accumulation
  // attn_logits layout: [batches, num_heads, num_kv_splits, head_size_v + 1]
  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  // TARGET_BLOCK_N = 2 * vl_max_e32m4 = __riscv_v_fixed_vlen / 4 (compile-time).
  // VLEN=128 → 32, VLEN=256 → 64, VLEN=512 → 128, VLEN=1024 → 256.
  constexpr int64_t TARGET_BLOCK_N = rvv_constants::BLOCK_N;

  at::parallel_for(0, batches * num_heads * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_id, num_heads, kv_id, num_kv_splits);

    alignas(64) float s_i[TARGET_BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + head_id * q_strideH;

      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(seq_len_kv <= max_context_len, "seq_len_kv out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");
      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      float m_prime = -std::numeric_limits<float>::infinity();
      float s_prime = 0.f;

      // Use correct stride calculation: [bs, head_id, kv_id, d]
      float* __restrict__ v_prime = attn_logits + bs * l_stride0 + head_id * l_stride1 + kv_id * l_stride2;
      init_v_prime_split(v_prime, head_size_v);

      // Loop over tokens in blocks
      for (int64_t n = kv_start; n < kv_end; n += TARGET_BLOCK_N) {
        int64_t n_size = std::min(TARGET_BLOCK_N, kv_end - n);

        const index_t* cur_indices = req_to_token + req_pool_id * max_context_len + n;

        // Calculate s_i <- scale * Q @ K using generic dispatch
        index_gemm_kernel_nt<scalar_t, kv_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ k_buffer + head_id * k_strideH,
            /* C   */ s_i,
            /* ind */ cur_indices,
            /* scl */ scaling,
            /* M   */ 1,
            /* N   */ n_size,
            /* K   */ head_size,
            /* lda */ 1,
            /* ldb */ k_strideN,
            /* ldc */ 1,
            /* mtt */ max_total_num_tokens);

        // 2. Logit Cap & Softmax parts
        if (has_logit_cap) {
          size_t vl_max = __riscv_vsetvlmax_e32m4();
          for (int64_t idx = 0; idx < n_size; idx += vl_max) {
            size_t vl = __riscv_vsetvl_e32m4(n_size - idx);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(s_i + idx, vl);
            vx = __riscv_vfmul_vf_f32m4(vx, rlogit_cap, vl);
            vx = vftanh_f32m4(vx, vl);
            vx = __riscv_vfmul_vf_f32m4(vx, logit_cap, vl);
            __riscv_vse32_v_f32m4(s_i + idx, vx, vl);
          }
        }

        // m_i: max value across block
        float m_i = rvv_reduce_max_f32(s_i, n_size);
        m_i = std::max(m_i, m_prime);

        // m_delta <- exp(m' - m_i)
        float m_delta = expf(m_prime - m_i);

        // s_delta <- exp(s_i - m_i), returns sum(s_delta)
        float local_sum = exp_and_sum(s_i, n_size, m_i);

        // s' <- s' * m_delta + sum(s_delta)
        s_prime = s_prime * m_delta + local_sum;
        m_prime = m_i;

        // 3. V' <- V' * m_delta + s_delta @ V
        index_gemm_kernel_nn<scalar_t, kv_t, index_t>(
            /* A   */ s_i,
            /* B   */ v_buffer + head_id * v_strideH,
            /* C   */ v_prime,
            /* ind */ cur_indices,
            /* scl */ &m_delta,
            /* M   */ 1,
            /* N   */ head_size_v,
            /* K   */ n_size,
            /* lda */ 1,
            /* ldb */ v_strideN,
            /* ldc */ 1,
            /* mtt */ max_total_num_tokens);
      }

      if (kv_end > kv_start) {
        const float safe_s_prime = (s_prime > 1e-38f) ? s_prime : 1e-38f;
        float s = 1.0f / safe_s_prime;
        size_t vl_max = __riscv_vsetvlmax_e32m8();
        for (int64_t d = 0; d < head_size_v; d += vl_max) {
          size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
          vfloat32m8_t vval = __riscv_vle32_v_f32m8(v_prime + d, vl);
          vval = __riscv_vfmul_vf_f32m8(vval, s, vl);
          __riscv_vse32_v_f32m8(v_prime + d, vval, vl);
        }
        v_prime[head_size_v] = m_prime + std::log(safe_s_prime);
      }

      data_index_step(bs, batches, head_id, num_heads, kv_id, num_kv_splits);
    }
  });

  decode_accumulate_kv_splits(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

template <typename scalar_t, typename kv_t, typename index_t>
void decode_attention_grouped_kernel_impl(
    scalar_t* __restrict__ output,
    float* __restrict__ attn_logits,
    const scalar_t* __restrict__ query,
    const kv_t* __restrict__ k_buffer,
    const kv_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    int64_t batches,
    int64_t num_heads,
    int64_t num_heads_kv,
    int64_t head_size,
    int64_t head_size_v,
    int64_t num_kv_splits,
    int64_t q_strideM,
    int64_t q_strideH,
    int64_t k_strideN,
    int64_t k_strideH,
    int64_t v_strideN,
    int64_t v_strideH,
    float scaling,
    float logit_cap,
    int64_t max_num_reqs,
    int64_t max_context_len,
    int64_t max_total_num_tokens,
    int64_t o_strideM,
    int64_t o_strideH) {
  constexpr int64_t BLOCK_N = rvv_constants::BLOCK_N;
  // Block length for heads - parallel on [batches, divup(num_heads, BLOCK_H), num_kv_splits]
  constexpr int64_t kBLOCK_H = 16;
  const int64_t BLOCK_H = std::min(4 * batches, kBLOCK_H);

  // Strides for attn_logits
  const int64_t l_stride0 = num_heads * num_kv_splits * (head_size_v + 1);
  const int64_t l_stride1 = num_kv_splits * (head_size_v + 1);
  const int64_t l_stride2 = head_size_v + 1;

  const bool has_logit_cap = logit_cap > 0;
  float rlogit_cap = has_logit_cap ? 1 / logit_cap : 0.f;

  // Partition heads into blocks for parallel
  const int64_t num_groups = num_heads / num_heads_kv;
  const int64_t num_blocks = div_up(num_groups, BLOCK_H);

  // Parallel on [batches, num_heads_kv, num_blocks, num_kv_splits]
  at::parallel_for(0, batches * num_heads_kv * num_blocks * num_kv_splits, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, head_kv_id{0}, block_id{0}, kv_id{0};
    data_index_init(begin, bs, batches, head_kv_id, num_heads_kv, block_id, num_blocks, kv_id, num_kv_splits);

    alignas(64) float s_i[kBLOCK_H * BLOCK_N];
    float* __restrict__ s_delta = s_i;

    alignas(64) float s_prime[kBLOCK_H];
    alignas(64) float m_prime[kBLOCK_H];
    alignas(64) float m_delta[kBLOCK_H];

    for (int64_t i = begin; i < end; ++i) {
      const int64_t h_start = head_kv_id * num_groups + block_id * BLOCK_H;
      const int64_t h_end = head_kv_id * num_groups + std::min(block_id * BLOCK_H + BLOCK_H, num_groups);
      const int64_t h_size = h_end - h_start;

      // Get query
      const scalar_t* __restrict__ q_ptr = query + bs * q_strideM + h_start * q_strideH;

      int64_t seq_len_kv = seq_lens[bs];
      int64_t req_pool_id = req_pool_indices[bs];
      TORCH_CHECK(seq_len_kv <= max_context_len, "seq_len_kv out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      const int64_t SPLIT_SIZE = div_up(seq_len_kv, num_kv_splits);
      const int64_t kv_start = kv_id * SPLIT_SIZE;
      const int64_t kv_end = std::min(kv_start + SPLIT_SIZE, seq_len_kv);

      // Initialize accumulators
      for (int64_t h = 0; h < h_size; ++h) {
        s_prime[h] = 0.f;
        m_prime[h] = -std::numeric_limits<float>::infinity();
      }

      // Get v_prime, and init to zero (sentinel -inf for decode_accumulate_kv_splits)
      float* __restrict__ v_prime = attn_logits + bs * l_stride0 + h_start * l_stride1 + kv_id * l_stride2;
      for (int64_t h = 0; h < h_size; ++h) {
        init_v_prime_split(v_prime + h * l_stride1, head_size_v);
      }

      // Loop over K and V sequence with BLOCK_N
      for (int64_t n = kv_start; n < kv_end; n += BLOCK_N) {
        int64_t n_size = std::min(BLOCK_N, kv_end - n);

        const index_t* cur_indices = req_to_token + req_pool_id * max_context_len + n;

        // Calculate Q @ K using RVV dispatch
        index_gemm_kernel_nt<scalar_t, kv_t, index_t>(
            /* A   */ q_ptr,
            /* B   */ k_buffer + head_kv_id * k_strideH,
            /* C   */ s_i,
            /* ind */ cur_indices,
            /* scl */ scaling,
            /* M   */ h_size,
            /* N   */ n_size,
            /* K   */ head_size,
            /* lda */ q_strideH,
            /* ldb */ k_strideN,
            /* ldc */ BLOCK_N,
            /* mtt */ max_total_num_tokens);

        if (has_logit_cap) {
          // Apply per-row, up to n_size only. h_size rows × BLOCK_N columns,
          // but only n_size elements per row are valid (partial last block).
          // Reading s_i[h*BLOCK_N + n_size..BLOCK_N-1] is C++ UB (uninitialized).
          size_t vl_max = __riscv_vsetvlmax_e32m4();
          for (int64_t h = 0; h < h_size; ++h) {
            float* row = s_i + h * BLOCK_N;
            for (int64_t idx = 0; idx < n_size; idx += vl_max) {
              size_t vl = __riscv_vsetvl_e32m4(n_size - idx);
              vfloat32m4_t vx = __riscv_vle32_v_f32m4(row + idx, vl);
              vx = __riscv_vfmul_vf_f32m4(vx, rlogit_cap, vl);
              vx = vftanh_f32m4(vx, vl);
              vx = __riscv_vfmul_vf_f32m4(vx, logit_cap, vl);
              __riscv_vse32_v_f32m4(row + idx, vx, vl);
            }
          }
        }

        // Update the scaling coefficients
        for (int64_t h = 0; h < h_size; ++h) {
          float* s_row = s_i + h * BLOCK_N;

          // m_i: vectorized max reduction across row
          float local_max = rvv_reduce_max_f32(s_row, n_size);
          float m_i = std::max(local_max, m_prime[h]);

          // m_delta <- exp(m' - m_i)
          m_delta[h] = expf(m_prime[h] - m_i);

          // s_delta <- exp(s_i - m_i) and sum
          float row_sum = 0.0f;
          size_t vl_max = __riscv_vsetvlmax_e32m4();
          for (int64_t j = 0; j < n_size; j += vl_max) {
            size_t vl = __riscv_vsetvl_e32m4(n_size - j);
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(s_row + j, vl);
            vx = __riscv_vfsub_vf_f32m4(vx, m_i, vl);
            vfloat32m4_t vex = vfexp_f32m4(vx, vl);
            __riscv_vse32_v_f32m4(s_delta + h * BLOCK_N + j, vex, vl);

            vfloat32m1_t vzero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(vex, vzero, vl);
            row_sum += __riscv_vfmv_f_s_f32m1_f32(vsum);
          }

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[h] = s_prime[h] * m_delta[h] + row_sum;
          m_prime[h] = m_i;
        }

        // Calculate V' <- s_delta @ V + V' * m_delta
        index_gemm_kernel_nn<scalar_t, kv_t, index_t>(
            /* A   */ s_delta,
            /* B   */ v_buffer + head_kv_id * v_strideH,
            /* C   */ v_prime,
            /* ind */ cur_indices,
            /* scl */ m_delta,
            /* M   */ h_size,
            /* N   */ head_size_v,
            /* K   */ n_size,
            /* lda */ BLOCK_N,
            /* ldb */ v_strideN,
            /* ldc */ l_stride1,
            /* mtt */ max_total_num_tokens);
      }  // loop with KV blocks

      // Only update v' when kv_split_size > 0
      if (kv_end > kv_start) {
        for (int64_t h = 0; h < h_size; ++h) {
          // Guard against division by very small numbers to prevent Inf
          const float safe_s_prime = (s_prime[h] > 1e-38f) ? s_prime[h] : 1e-38f;
          float s = 1.0f / safe_s_prime;
          float* v_ptr = v_prime + h * l_stride1;

          // Scale each element of v' by 1/s_prime to produce the normalized partial sum.
          size_t vl_max = __riscv_vsetvlmax_e32m8();
          for (int64_t d = 0; d < head_size_v; d += vl_max) {
            size_t vl = __riscv_vsetvl_e32m8(head_size_v - d);
            vfloat32m8_t vv = __riscv_vle32_v_f32m8(v_ptr + d, vl);
            vv = __riscv_vfmul_vf_f32m8(vv, s, vl);
            __riscv_vse32_v_f32m8(v_ptr + d, vv, vl);
          }
          v_ptr[head_size_v] = m_prime[h] + std::log(safe_s_prime);
        }
      }

      data_index_step(bs, batches, head_kv_id, num_heads_kv, block_id, num_blocks, kv_id, num_kv_splits);
    }
  });

  decode_accumulate_kv_splits(
      output, attn_logits, batches, num_heads, head_size_v, num_kv_splits, l_stride1, l_stride2);
}

// Tensor shapes:
//   query:            [num_tokens, num_heads, head_size]
//   output:           [num_tokens, num_heads, head_size]
//   k_buffer:         [max_total_num_tokens, num_heads, head_size]
//   v_buffer:         [max_total_num_tokens, num_heads, head_size_v]
//   attn_logits:      [num_seqs, num_heads, num_kv_splits, head_size_v + 1]
//   req_to_token:     [max_num_reqs, max_context_len] int32 or int64
//   req_pool_indices: [num_seqs] int64
//   seq_lens:         [num_seqs] int64
}  // anonymous namespace

void decode_attention_cpu(
    at::Tensor& query,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& output,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& loc,
    at::Tensor& attn_logits,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    double sm_scale,
    double logit_cap) {
  RECORD_FUNCTION(
      "sgl-kernel::decode_attention_cpu",
      std::vector<c10::IValue>(
          {query, output, k_buffer, v_buffer, attn_logits, req_to_token, req_pool_indices, seq_lens}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_DIM(3, query);
  CHECK_DIM(3, k_buffer);
  CHECK_DIM(3, v_buffer);
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);
  CHECK_DIM(1, loc);
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == k_buffer.scalar_type() && query.scalar_type() == v_buffer.scalar_type(),
      "decode_attention_cpu: expect query, key, value, k_buffer, and v_buffer to have the same dtype");

  int64_t num_seqs = seq_lens.size(0);
  int64_t max_num_reqs = req_to_token.size(0);
  int64_t max_context_len = req_to_token.size(1);
  int64_t max_total_num_tokens = k_buffer.size(0);

  int64_t num_heads = query.size(1);
  int64_t num_heads_kv = k_buffer.size(1);
  int64_t head_size = query.size(2);
  int64_t head_size_v = v_buffer.size(2);

  TORCH_CHECK(
      head_size <= MAX_HEAD_SIZE && head_size_v <= MAX_HEAD_SIZE,
      "decode_attention_cpu: head_size (",
      head_size,
      ") or head_size_v (",
      head_size_v,
      ") exceeds MAX_HEAD_SIZE (",
      MAX_HEAD_SIZE,
      "). Recompile with a larger MAX_HEAD_SIZE to support this configuration.");

  int64_t num_kv_splits = attn_logits.size(2);

  CHECK_EQ(loc.numel(), num_seqs);
  CHECK_EQ(attn_logits.size(0), num_seqs);
  CHECK_EQ(attn_logits.size(1), num_heads);
  CHECK_EQ(attn_logits.size(3), head_size_v + 1);
  CHECK_EQ(attn_logits.scalar_type(), at::kFloat);
  // decode_accumulate_kv_splits writes output as output + i*head_size_v
  // (flat contiguous indexing). Verify the tensor is actually contiguous.
  TORCH_CHECK(output.is_contiguous(), "decode_attention_cpu: output must be contiguous");

  int64_t q_strideM = query.stride(0);
  int64_t q_strideH = query.stride(1);
  int64_t o_strideM = output.stride(0);
  int64_t o_strideH = output.stride(1);
  int64_t k_strideN = k_buffer.stride(0);
  int64_t k_strideH = k_buffer.stride(1);
  int64_t v_strideN = v_buffer.stride(0);
  int64_t v_strideH = v_buffer.stride(1);
  int64_t nk_strideN = key.stride(0);
  int64_t nk_strideH = key.stride(1);
  int64_t nv_strideN = value.stride(0);
  int64_t nv_strideH = value.stride(1);

  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "decode: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "decode: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "decode: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());

  void* k_buffer_data = k_buffer.data_ptr();
  void* v_buffer_data = v_buffer.data_ptr();

  int num_threads = at::get_num_threads();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "decode_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "decode_attention_indices", [&] {
      decode_set_kv_buffer(
          (scalar_t*)k_buffer_data,
          (scalar_t*)v_buffer_data,
          key.data_ptr<scalar_t>(),
          value.data_ptr<scalar_t>(),
          loc.data_ptr<int64_t>(),
          num_seqs,
          num_heads_kv,
          head_size,
          head_size_v,
          k_strideN,
          k_strideH,
          v_strideN,
          v_strideH,
          nk_strideN,
          nk_strideH,
          nv_strideN,
          nv_strideH);

      if (num_heads == num_heads_kv) {
        decode_attention_kernel_impl<scalar_t, scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const scalar_t*)k_buffer_data,
            (const scalar_t*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            o_strideM,
            o_strideH);
      } else {
        decode_attention_grouped_kernel_impl<scalar_t, scalar_t, index_t>(
            output.data_ptr<scalar_t>(),
            attn_logits.data_ptr<float>(),
            query.data_ptr<scalar_t>(),
            (const scalar_t*)k_buffer_data,
            (const scalar_t*)v_buffer_data,
            req_to_token.data_ptr<index_t>(),
            req_pool_indices.data_ptr<int64_t>(),
            seq_lens.data_ptr<int64_t>(),
            num_seqs,
            num_heads,
            num_heads_kv,
            head_size,
            head_size_v,
            num_kv_splits,
            q_strideM,
            q_strideH,
            k_strideN,
            k_strideH,
            v_strideN,
            v_strideH,
            sm_scale,
            logit_cap,
            max_num_reqs,
            max_context_len,
            max_total_num_tokens,
            o_strideM,
            o_strideH);
      }
    });
  });
}

#endif  // CPU_CAPABILITY_RVV
