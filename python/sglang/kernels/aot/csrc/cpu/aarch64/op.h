#pragma once

#include <arm_neon.h>

constexpr int64_t kL1Size = 64 * 1024;
constexpr int64_t kL2Size = 1 * 1024 * 1024;

// simd optimized operators
namespace op {

// do matmul in "R rows x C cols" tile with sdot
// - a is the full [M, K] matrix, row major
// - b is one slice of a [K, N] matrix, column major ([N, K] row major)
// - c is one slice of a [M, N] matrix, row major
//
//                            slice_width             slice_width
//   a                     b  /                    c  /
//   |                     |----|                  |----|
//   v                     |    |                  v    v
//   / ------ \            v    v            /     ------     \
//   | ------ |     /      ||||||      \     |     ------     |
// M | ------ |  @  |      ||||||      |  =  |     ------     |
//   | ------ |     |      ||||||      |     |     ------     |
//   | ------ |     \      ||||||      /     |     ------     |
//   \ ------ /                              \     ------     /
//       K                   N
//
template <int R = 4, int C = 8, typename T>
__attribute__((target("+dotprod+bf16"))) void sdot_matmul(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    T* c,
    int64_t M,
    int64_t K,
    int64_t N,
    int slice_width,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>);

  int row = 0;
  for (; row + R <= M; row += R) {
    const int8_t* a_rows[R];
    T* c_rows[R];
    for (int i = 0; i < R; ++i) {
      a_rows[i] = a + (row + i) * K;
      c_rows[i] = c + (row + i) * N;
    }

    int col = 0;
    for (; col + C <= slice_width; col += C) {
      const int8_t* b_cols[C];
      for (int i = 0; i < C; ++i) {
        b_cols[i] = b + (col + i) * K;
      }

      int32x4_t vsums[R][C]{};

      // TODO: accumulated integer sum may overflow when K >= 65536
      int k = 0;
      for (; k + 16 <= K; k += 16) {
        int8x16_t va[R];
        int8x16_t vb[C];
        for (int i = 0; i < R; ++i) {
          va[i] = vld1q_s8(a_rows[i] + k);
        }
        for (int i = 0; i < C; ++i) {
          vb[i] = vld1q_s8(b_cols[i] + k);
        }
        for (int i = 0; i < R; ++i) {
          for (int j = 0; j < C; ++j) {
            vsums[i][j] = vdotq_s32(vsums[i][j], va[i], vb[j]);
          }
        }
      }

      if (k < K) {
        int8_t abuf[16]{};
        int8_t bbuf[16]{};
        for (int i = 0; i < R; ++i) {
          memcpy(abuf, a_rows[i] + k, K - k);
          const int8x16_t va = vld1q_s8(abuf);
          for (int j = 0; j < C; ++j) {
            memcpy(bbuf, b_cols[j] + k, K - k);
            const int8x16_t vb = vld1q_s8(bbuf);
            vsums[i][j] = vdotq_s32(vsums[i][j], va, vb);
          }
        }
      }

      for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
          const float sum = vaddvq_s32(vsums[i][j]);
          const float sum_scaled = sum * scales1[row + i] * scales2[col + j];
          if constexpr (std::is_same_v<T, bfloat16_t>) {
            c_rows[i][col + j] = vcvth_bf16_f32(sum_scaled);
          } else {
            c_rows[i][col + j] = sum_scaled;
          }
        }
      }
    }

    if (col < slice_width) {
      sdot_matmul<R, 1>(a, b + col * K, c + col, M, K, N, slice_width - col, scales1, scales2 + col);
    }
  }

  if (row < M) {
    sdot_matmul<1, C>(a + row * K, b, c + row * N, M - row, K, N, slice_width, scales1 + row, scales2);
  }
}

// do matmul in "R rows x C cols" tile with i8mm
template <int R = 4, int C = 8, typename T>
__attribute__((target("+i8mm+bf16"))) void i8mm_matmul(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    T* c,
    int64_t M,
    int64_t K,
    int64_t N,
    int slice_width,
    const float* __restrict__ scales1,
    const float* __restrict__ scales2) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>);
  static_assert(R % 2 == 0 && C % 2 == 0);

  int row = 0;
  for (; row + R <= M; row += R) {
    const int8_t* a_rows[R];
    T* c_rows[R];
    for (int i = 0; i < R; ++i) {
      a_rows[i] = a + (row + i) * K;
      c_rows[i] = c + (row + i) * N;
    }

    int col = 0;
    for (; col + C <= slice_width; col += C) {
      const int8_t* b_cols[C];
      for (int i = 0; i < C; ++i) {
        b_cols[i] = b + (col + i) * K;
      }

      int8x16_t va[R], vb[C];
      int32x4_t vsums[R / 2][C / 2]{};

      // TODO: accumulated integer sum may overflow when K >= 65536
      int k = 0;
      for (; k + 16 <= K; k += 16) {
        for (int i = 0; i < R; i += 2) {
          const int64x2_t va0_s64 = vreinterpretq_s64_s8(vld1q_s8(a_rows[i + 0] + k));
          const int64x2_t va1_s64 = vreinterpretq_s64_s8(vld1q_s8(a_rows[i + 1] + k));
          va[i + 0] = vreinterpretq_s8_s64(vzip1q_s64(va0_s64, va1_s64));
          va[i + 1] = vreinterpretq_s8_s64(vzip2q_s64(va0_s64, va1_s64));
        }
        for (int i = 0; i < C; i += 2) {
          const int64x2_t vb0_s64 = vreinterpretq_s64_s8(vld1q_s8(b_cols[i + 0] + k));
          const int64x2_t vb1_s64 = vreinterpretq_s64_s8(vld1q_s8(b_cols[i + 1] + k));
          vb[i + 0] = vreinterpretq_s8_s64(vzip1q_s64(vb0_s64, vb1_s64));
          vb[i + 1] = vreinterpretq_s8_s64(vzip2q_s64(vb0_s64, vb1_s64));
        }
        for (int i = 0; i < R / 2; ++i) {
          for (int j = 0; j < C / 2; ++j) {
            vsums[i][j] = vmmlaq_s32(vsums[i][j], va[i * 2 + 0], vb[j * 2 + 0]);
            vsums[i][j] = vmmlaq_s32(vsums[i][j], va[i * 2 + 1], vb[j * 2 + 1]);
          }
        }
      }

      if (k < K) {
        int8_t buf0[16]{}, buf1[16]{};
        for (int i = 0; i < R; i += 2) {
          memcpy(buf0, a_rows[i + 0] + k, (K - k) * sizeof(int8_t));
          memcpy(buf1, a_rows[i + 1] + k, (K - k) * sizeof(int8_t));
          const int64x2_t va0_s64 = vreinterpretq_s64_s8(vld1q_s8(buf0));
          const int64x2_t va1_s64 = vreinterpretq_s64_s8(vld1q_s8(buf1));
          va[i + 0] = vreinterpretq_s8_s64(vzip1q_s64(va0_s64, va1_s64));
          va[i + 1] = vreinterpretq_s8_s64(vzip2q_s64(va0_s64, va1_s64));
        }
        for (int i = 0; i < C; i += 2) {
          memcpy(buf0, b_cols[i + 0] + k, (K - k) * sizeof(int8_t));
          memcpy(buf1, b_cols[i + 1] + k, (K - k) * sizeof(int8_t));
          const int64x2_t vb0_s64 = vreinterpretq_s64_s8(vld1q_s8(buf0));
          const int64x2_t vb1_s64 = vreinterpretq_s64_s8(vld1q_s8(buf1));
          vb[i + 0] = vreinterpretq_s8_s64(vzip1q_s64(vb0_s64, vb1_s64));
          vb[i + 1] = vreinterpretq_s8_s64(vzip2q_s64(vb0_s64, vb1_s64));
        }
        for (int i = 0; i < R / 2; ++i) {
          for (int j = 0; j < C / 2; ++j) {
            vsums[i][j] = vmmlaq_s32(vsums[i][j], va[i * 2 + 0], vb[j * 2 + 0]);
            vsums[i][j] = vmmlaq_s32(vsums[i][j], va[i * 2 + 1], vb[j * 2 + 1]);
          }
        }
      }

      for (int i = 0; i < R; i += 2) {
        for (int j = 0; j < C; j += 2) {
          float32x4_t vsum_f32 = vcvtq_f32_s32(vsums[i / 2][j / 2]);
          const float32x4_t scales = {
              scales1[row + i + 0] * scales2[col + j + 0],
              scales1[row + i + 0] * scales2[col + j + 1],
              scales1[row + i + 1] * scales2[col + j + 0],
              scales1[row + i + 1] * scales2[col + j + 1],
          };
          vsum_f32 = vmulq_f32(vsum_f32, scales);
          if constexpr (std::is_same_v<T, bfloat16_t>) {
            const bfloat16x4_t vsum_bf16 = vcvt_bf16_f32(vsum_f32);
            c_rows[i + 0][col + j + 0] = vget_lane_bf16(vsum_bf16, 0);
            c_rows[i + 0][col + j + 1] = vget_lane_bf16(vsum_bf16, 1);
            c_rows[i + 1][col + j + 0] = vget_lane_bf16(vsum_bf16, 2);
            c_rows[i + 1][col + j + 1] = vget_lane_bf16(vsum_bf16, 3);
          } else {
            c_rows[i + 0][col + j + 0] = vgetq_lane_f32(vsum_f32, 0);
            c_rows[i + 0][col + j + 1] = vgetq_lane_f32(vsum_f32, 1);
            c_rows[i + 1][col + j + 0] = vgetq_lane_f32(vsum_f32, 2);
            c_rows[i + 1][col + j + 1] = vgetq_lane_f32(vsum_f32, 3);
          }
        }
      }
    }

    if (col < slice_width) {
      sdot_matmul<R, 1>(a, b + col * K, c + col, M, K, N, slice_width - col, scales1, scales2 + col);
    }
  }

  if (row < M) {
    sdot_matmul<1, C>(a + row * K, b, c + row * N, M - row, K, N, slice_width, scales1 + row, scales2);
  }
}

__attribute__((target("+bf16"))) inline void
add_bias(bfloat16_t* __restrict__ out, const float* __restrict__ bias, int64_t M, int64_t N, int width) {
  int col = 0;

  for (; col + 4 <= width; col += 4) {
    const float32x4_t vbias32 = vld1q_f32(bias + col);
    bfloat16_t* out_ptr = out + col;
    for (int64_t i = 0; i < M; ++i) {
      bfloat16x4_t vout16 = vld1_bf16(out_ptr);
      float32x4_t vout32 = vcvt_f32_bf16(vout16);
      vout32 = vaddq_f32(vout32, vbias32);
      vout16 = vcvt_bf16_f32(vout32);
      vst1_bf16(out_ptr, vout16);
      out_ptr += N;
    }
  }

  for (; col < width; ++col) {
    const float vbias32 = bias[col];
    bfloat16_t* out_ptr = out + col;
    for (int64_t i = 0; i < M; ++i) {
      bfloat16_t vout16 = *out_ptr;
      float vout32 = vcvtah_f32_bf16(vout16);
      vout32 += vbias32;
      vout16 = vcvth_bf16_f32(vout32);
      *out_ptr = vout16;
      out_ptr += N;
    }
  }
}

constexpr float eps = 1e-7;

template <typename scalar_t>
inline void quantize_row_int8(int8_t* __restrict__ q, float* scale, const scalar_t* __restrict__ x, int64_t n) {
  float max_abs_val = eps;
  for (int64_t i = 0; i < n; ++i) {
    max_abs_val = std::max(std::abs(static_cast<float>(x[i])), max_abs_val);
  }
  *scale = max_abs_val / 127.0f;

  const float inv_scale = 127.0f / max_abs_val;
  for (int64_t i = 0; i < n; ++i) {
    q[i] = static_cast<int8_t>(std::round(static_cast<float>(x[i]) * inv_scale));
  }
}

// manually optimize for bf16
template <>
__attribute__((target("+bf16"))) inline void
quantize_row_int8<bfloat16_t>(int8_t* __restrict__ q, float* scale, const bfloat16_t* __restrict__ x, int64_t n) {
  float max_abs_val = eps;
  for (int64_t i = 0; i < n; ++i) {
    max_abs_val = std::max(std::abs(vcvtah_f32_bf16(x[i])), max_abs_val);
  }
  *scale = max_abs_val / 127.0f;

  const float inv_scale = 127.0f / max_abs_val;

  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    int32x4_t qv_s32[4];
    {
      const bfloat16x8x2_t xv_bf16 = vld1q_bf16_x2(x + i);
      const float32x4_t xv_f32[4] = {
          vcvtq_low_f32_bf16(xv_bf16.val[0]),
          vcvtq_high_f32_bf16(xv_bf16.val[0]),
          vcvtq_low_f32_bf16(xv_bf16.val[1]),
          vcvtq_high_f32_bf16(xv_bf16.val[1]),
      };
      for (int j = 0; j < 4; ++j) {
        float32x4_t qv_f32 = vmulq_n_f32(xv_f32[j], inv_scale);
        qv_f32 = vrndaq_f32(qv_f32);
        qv_s32[j] = vcvtq_s32_f32(qv_f32);
      }
    }

    const int16x8_t qv_s16[2] = {
        vuzp1q_s16(vreinterpretq_s16_s32(qv_s32[0]), vreinterpretq_s16_s32(qv_s32[1])),
        vuzp1q_s16(vreinterpretq_s16_s32(qv_s32[2]), vreinterpretq_s16_s32(qv_s32[3])),
    };
    const int8x16_t qv_s8 = vuzp1q_s8(vreinterpretq_s8_s16(qv_s16[0]), vreinterpretq_s8_s16(qv_s16[1]));
    vst1q_s8(q + i, qv_s8);
  }

  for (; i < n; ++i) {
    q[i] = static_cast<int8_t>(std::round(vcvtah_f32_bf16(x[i]) * inv_scale));
  }
}

template <>
inline void
quantize_row_int8<at::BFloat16>(int8_t* __restrict__ q, float* scale, const at::BFloat16* __restrict__ x, int64_t n) {
  quantize_row_int8(q, scale, reinterpret_cast<const bfloat16_t*>(x), n);
}

__attribute__((target("+bf16"))) inline void f32_to_bf16(const float* f32, bfloat16_t* bf16, int64_t n) {
  int64_t i = 0;

  for (; i + 4 <= n; i += 4) {
    const float32x4_t vf32 = vld1q_f32(f32 + i);
    const bfloat16x4_t vbf16 = vcvt_bf16_f32(vf32);
    vst1_bf16(bf16 + i, vbf16);
  }

  for (; i < n; ++i) {
    bf16[i] = vcvth_bf16_f32(f32[i]);
  }
}

}  // namespace op
