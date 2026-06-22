#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

namespace {
// For this cpu kernel, we have some innovations aside from the existing gpu kernels:
// 1) Use less parallel loops, i.e. 4 including l2_norm.
// 2) Fuse part of l2_norm with the rest of the computation.

#define THREAD_BUFFER_ALLOC(dst, base_ptr, offset, type, size) \
  type* dst = reinterpret_cast<type*>((base_ptr) + (offset));  \
  offset += (size);

// * convert to vnni format， expect contiguous input and output
//     from [K/2, 2, N] FP32 to [K/2, N, 2] BF16
// * update src = src * exp(g_last)
template <typename scalar_t, int K, int N>
void pack_vnni2(scalar_t* __restrict__ dst, float* __restrict__ src, const float g_last, int ld_src, int ld_dst) {
  static_assert(K % 32 == 0);
  static_assert(N % 32 == 0);

  const float scale = std::exp(g_last);
#if defined(CPU_CAPABILITY_AVX512)
  constexpr int KB = K / 2;
  constexpr int NB = N / 32;

  __m512i s0, s1, d0, d1;
  __m512 vd = _mm512_set1_ps(scale);

  const auto trans = [&](auto i) {
    constexpr int kb = i / NB;
    constexpr int nb = i % NB;

    // [K/2, 2, N/32, 32] -> [K/2, N/32, 32, 2]
    constexpr int k0 = kb * 2 + 0;
    constexpr int k1 = kb * 2 + 1;
    __m512 v00 = _mm512_loadu_ps(src + k0 * ld_src + nb * 32);
    __m512 v01 = _mm512_loadu_ps(src + k0 * ld_src + nb * 32 + 16);
    __m512 v10 = _mm512_loadu_ps(src + k1 * ld_src + nb * 32);
    __m512 v11 = _mm512_loadu_ps(src + k1 * ld_src + nb * 32 + 16);
    s0 = (__m512i)_mm512_cvtne2ps_pbh(v01, v00);
    s1 = (__m512i)_mm512_cvtne2ps_pbh(v11, v10);

    std::tie(d0, d1) = transpose_2x32_16bit(s0, s1);
    _mm512_storeu_si512(dst + kb * ld_dst * 2 + nb * 32 * 2, d0);
    _mm512_storeu_si512(dst + kb * ld_dst * 2 + nb * 32 * 2 + 32, d1);

    // update src = src * exp(g_last)
    _mm512_storeu_ps(src + k0 * ld_src + nb * 32, _mm512_mul_ps(v00, vd));
    _mm512_storeu_ps(src + k0 * ld_src + nb * 32 + 16, _mm512_mul_ps(v01, vd));
    _mm512_storeu_ps(src + k1 * ld_src + nb * 32, _mm512_mul_ps(v10, vd));
    _mm512_storeu_ps(src + k1 * ld_src + nb * 32 + 16, _mm512_mul_ps(v11, vd));
  };
  Unroll<KB * NB>{}(trans);
#else
  // [K/2, 2, N] -> [K/2, N, 2]
  for (int k = 0; k < K; k += 2) {
    for (int n = 0; n < N; ++n) {
      const float v0 = src[(k + 0) * ld_src + n];
      const float v1 = src[(k + 1) * ld_src + n];
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = static_cast<scalar_t>(v0);
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = static_cast<scalar_t>(v1);
      src[(k + 0) * ld_src + n] = v0 * scale;
      src[(k + 1) * ld_src + n] = v1 * scale;
    }
  }
#endif
}

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, float val, int size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  const Vec data_vec = Vec(static_cast<scalar_t>(val));
  int d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    data_vec.store(out + d);
  }
  if (size - d > 0) {
    data_vec.store(out + d, size - d);
  }
}

template <typename scalar_t, int D, bool has_scale>
struct l2norm_kernel {
  static inline void apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float eps) {
    TORCH_CHECK(false, "l2norm_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D, bool has_scale>
struct l2norm_kernel<at::BFloat16, D, has_scale> {
  static inline void apply(at::BFloat16* __restrict__ out, const at::BFloat16* __restrict__ input, float eps) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 32;

    __m512bh va[COLS];
    __m512 vrscale;

    constexpr float scale = 1.f / std::sqrt(D);
    __m512 vscale = _mm512_set1_ps(scale);

    // step 1: load input and do reduce with avx512-bf16
    __m512 vsum = _mm512_set1_ps(0.f);
    auto reduce = [&](auto col) {
      va[col] = (__m512bh)(_mm512_loadu_si512(input + col * 32));
      vsum = _mm512_dpbf16_ps(vsum, va[col], va[col]);
    };
    Unroll<COLS>{}(reduce);

    float sqsum = _mm512_reduce_add_ps(vsum);
    float rscale = 1.f / std::sqrt(sqsum + eps);
    vrscale = _mm512_set1_ps(rscale);

    // step 2: apply scale to output
    auto map = [&](auto col) {
      __m512i a16 = (__m512i)va[col];
      __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
      __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
      va0 = _mm512_mul_ps(va0, vrscale);
      va1 = _mm512_mul_ps(va1, vrscale);
      // keep the mul order same as torch code:
      //   query = l2norm(query) * scale
      if constexpr (has_scale) {
        va0 = _mm512_mul_ps(va0, vscale);
        va1 = _mm512_mul_ps(va1, vscale);
      }
      _mm512_storeu_si512(out + col * 32, (__m512i)(_mm512_cvtne2ps_pbh(va1, va0)));
    };
    Unroll<COLS>{}(map);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int BLOCK_H>
struct cumsum_kernel {
  static inline void
  apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int size, int ld_src, int ld_dst) {
    TORCH_CHECK(false, "cumsum_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int BLOCK_H>
struct cumsum_kernel<float, CHUNK_SIZE, BLOCK_H> {
  static inline void apply(float* __restrict__ out, const float* __restrict__ input, int size, int ld_src, int ld_dst) {
    // vector length of fp32 for avx512
    static_assert(BLOCK_H == 16);

    __m512i va[16];
    __m512 vsum = _mm512_set1_ps(0.f);

    for (int i = 0; i < CHUNK_SIZE; i += 16) {
      // load input data
      Unroll<16>{}([&](auto j) {
        __m512 v = (i + j < size) ? _mm512_loadu_ps(input + (i + j) * ld_src) : _mm512_setzero_ps();
        vsum = _mm512_add_ps(vsum, v);
        va[j] = _mm512_castps_si512(vsum);
      });
      // transpose
      transpose_16x16_32bit(va);
      // store output data
      Unroll<16>{}([&](auto j) { _mm512_storeu_si512(out + j * ld_dst + i, va[j]); });
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE>
struct decay_mask_kernel {
  static inline void apply(scalar_t* __restrict__ out, const scalar_t* __restrict__ input) {
    TORCH_CHECK(false, "decay_mask_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE>
struct decay_mask_kernel<float, CHUNK_SIZE> {
  static inline void apply(float* __restrict__ out, const float* __restrict__ input) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int ROWS = CHUNK_SIZE;
    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 va;
    __m512 vb[COLS];

    // step 1: load g[j]
    auto loadb = [&](auto i) { vb[i] = _mm512_loadu_ps(input + i * 16); };
    Unroll<COLS>{}(loadb);

    // step2: exp(g[i] - g[j])
    auto compute = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = _mm512_set1_ps(input[row]);
      }

      // mask vb[col] (already loaded in step 1) for the lower-triangular region
      constexpr int len = std::max(0, std::min(row + 1 - col * 16, 16));

      __m512 vc;
      if constexpr (len == 16) {
        vc = _mm512_fexp_u20_ps(va - vb[col]);
      } else if constexpr (len == 0) {
        vc = _mm512_setzero_ps();
      } else {
        vc = _mm512_fexp_u20_ps(va - vb[col]);
        // do mask for vc
        constexpr __mmask16 vmask = (1 << len) - 1;
        vc = _mm512_mask_blend_ps(vmask, _mm512_setzero_ps(), vc);
      }
      _mm512_storeu_ps(out + row * CHUNK_SIZE + col * 16, vc);
    };
    Unroll<ROWS * COLS>{}(compute);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE>
struct apply_mask_kernel {
  static inline void apply(
      scalar_t* __restrict__ attn2,
      const float* __restrict__ attn,
      const scalar_t* __restrict__ beta,
      const float* __restrict__ d,
      int size,
      int b_stride) {
    TORCH_CHECK(false, "apply_mask_kernel: scalar path not implemented!");
  }
  static inline void
  apply2(scalar_t* __restrict__ attn2, const float* __restrict__ attn, const float* __restrict__ d, int size) {
    TORCH_CHECK(false, "apply_mask_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE>
struct apply_mask_kernel<at::BFloat16, CHUNK_SIZE> {
  static inline void apply(
      at::BFloat16* __restrict__ attn2,
      const float* __restrict__ attn,
      const at::BFloat16* __restrict__ beta,
      const float* __restrict__ d,
      int size,
      int b_stride) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int ROWS = CHUNK_SIZE;
    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 vbeta;

    // attn2 = -attn * beta * d
    auto compute = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      constexpr int len = std::max(0, std::min(row - col * 16, 16));
      if (row < size) {
        if constexpr (col == 0) {
          vbeta = _mm512_set1_ps(-static_cast<float>(beta[row * b_stride]));
        }

        __m512 vc;
        if constexpr (len == 0) {
          vc = _mm512_setzero_ps();
        } else {
          constexpr __mmask16 vmask = (1 << len) - 1;
          __m512 va = _mm512_maskz_loadu_ps(vmask, attn + row * CHUNK_SIZE + col * 16);
          __m512 vd = _mm512_maskz_loadu_ps(vmask, d + row * CHUNK_SIZE + col * 16);
          vc = _mm512_mul_ps(_mm512_mul_ps(va, vbeta), vd);
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(attn2 + row * CHUNK_SIZE + col * 16), (__m256i)(_mm512_cvtneps_pbh(vc)));
      }
    };
    Unroll<ROWS * COLS>{}(compute);
  }

  static inline void
  apply2(at::BFloat16* __restrict__ attn2, const float* __restrict__ attn, const float* __restrict__ d, int size) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int ROWS = CHUNK_SIZE;
    constexpr int COLS = CHUNK_SIZE / 16;

    // attn2 = attn * d
    auto compute = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      constexpr int len = std::max(0, std::min(row + 1 - col * 16, 16));
      if (row < size) {
        __m512 vc;
        if constexpr (len == 0) {
          vc = _mm512_setzero_ps();
        } else {
          constexpr __mmask16 vmask = (1 << len) - 1;
          __m512 va = _mm512_maskz_loadu_ps(vmask, attn + row * CHUNK_SIZE + col * 16);
          __m512 vd = _mm512_maskz_loadu_ps(vmask, d + row * CHUNK_SIZE + col * 16);
          vc = _mm512_mul_ps(va, vd);
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(attn2 + row * CHUNK_SIZE + col * 16), (__m256i)(_mm512_cvtneps_pbh(vc)));
      }
    };
    Unroll<ROWS * COLS>{}(compute);
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE>
struct solve_tril_kernel {
  static inline void apply(scalar_t* __restrict__ attn2, int size) {
    TORCH_CHECK(false, "solve_tril_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE>
struct solve_tril_kernel<at::BFloat16, CHUNK_SIZE> {
  static inline void apply(at::BFloat16* __restrict__ attn2, int size) {
    static_assert(CHUNK_SIZE % 16 == 0);

    constexpr int COLS = CHUNK_SIZE / 16;

    __m512 va;
    __m512 vb[COLS];
    __m512 vsum[COLS];

    // for len == 0 and row < size, we don't have to write back zero again
    // as in `apply_mask_kernel`, we already set zero for the upper-triangular region
    for (int i = 1; i < size; ++i) {
      // load row attn[..., i, :i]
      at::BFloat16* __restrict__ row_ptr = attn2 + i * CHUNK_SIZE;
      Unroll<COLS>{}([&](auto col) {
        int len = std::min(i - col * 16, 16);
        if (len > 0) {
          const __mmask16 vmask = (1 << len) - 1;
          vsum[col] = CVT_BF16_TO_FP32(_mm256_maskz_loadu_epi16(vmask, row_ptr + col * 16));
        }
      });

      // row = attn[..., i, :i].clone()
      // sub = attn[..., :i, :i].clone()
      // vsum = row + (row.unsqueeze(-1) * sub).sum(-2)
      for (int k = 0; k < i; ++k) {
        va = _mm512_set1_ps(static_cast<float>(row_ptr[k]));

        const at::BFloat16* __restrict__ row_k_ptr = attn2 + k * CHUNK_SIZE;
        Unroll<COLS>{}([&](auto col) {
          int len = std::min(k - col * 16, 16);
          if (len > 0) {
            const __mmask16 vmask = (1 << len) - 1;
            vb[col] = CVT_BF16_TO_FP32(_mm256_maskz_loadu_epi16(vmask, row_k_ptr + col * 16));
            vsum[col] = _mm512_fmadd_ps(va, vb[col], vsum[col]);
          }
        });
      }

      // attn[..., i, :i] = vsum
      Unroll<COLS>{}([&](auto col) {
        int len = std::min(i - col * 16, 16);
        if (len > 0) {
          const __mmask16 vmask = (1 << len) - 1;
          _mm256_mask_storeu_epi16(row_ptr + col * 16, vmask, (__m256i)(_mm512_cvtneps_pbh(vsum[col])));
        }
      });
    }

    // attn = attn + torch.eye(chunk_size)
    for (int i = 0; i < size; ++i) {
      attn2[i * CHUNK_SIZE + i] += 1.f;
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int D, bool has_beta, bool has_g>
struct apply_beta_kernel {
  static inline void apply(
      scalar_t* __restrict__ out,
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ beta,
      const float* __restrict__ g,
      int size,
      int ld_src,
      int ld_dst,
      int b_stride) {
    TORCH_CHECK(false, "apply_beta_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int D, bool has_beta, bool has_g>
struct apply_beta_kernel<at::BFloat16, CHUNK_SIZE, D, has_beta, has_g> {
  static inline void apply(
      at::BFloat16* __restrict__ out,
      const at::BFloat16* __restrict__ input,
      const at::BFloat16* __restrict__ beta,
      const float* __restrict__ g,
      int size,
      int ld_src,
      int ld_dst,
      int b_stride) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    // get g.exp() and g is padded to CHUNK_SIZE
    alignas(64) float g_arr[CHUNK_SIZE];
    if constexpr (has_g) {
      Unroll<CHUNK_SIZE / 16>{}([&](auto col) {
        __m512 vg = _mm512_loadu_ps(g + col * 16);
        __m512 vg_exp = _mm512_fexp_u20_ps(vg);
        _mm512_storeu_ps(g_arr + col * 16, vg_exp);
      });
    }

    for (int i = 0; i < size; ++i) {
      __m512 vbeta;
      if constexpr (has_beta) {
        vbeta = _mm512_set1_ps(static_cast<float>(beta[i * b_stride]));
      }
      __m512 vg;
      if constexpr (has_g) {
        vg = _mm512_set1_ps(g_arr[i]);
      }

      Unroll<COLS>{}([&](auto col) {
        // load for 0, 2, 4, 6
        if constexpr (col % 2 == 0) {
          __m512i a16 = _mm512_loadu_si512(input + i * ld_src + col * 16);
          __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 0));
          __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(a16, 1));
          if constexpr (has_beta) {
            va0 = _mm512_mul_ps(va0, vbeta);
            va1 = _mm512_mul_ps(va1, vbeta);
          }
          if constexpr (has_g) {
            va0 = _mm512_mul_ps(va0, vg);
            va1 = _mm512_mul_ps(va1, vg);
          }
          _mm512_storeu_si512(out + i * ld_dst + col * 16, (__m512i)(_mm512_cvtne2ps_pbh(va1, va0)));
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int D>
struct update_kernel {
  static inline void
  apply(scalar_t* __restrict__ out, const float* __restrict__ input, int size, int ld_src, int ld_dst) {
    TORCH_CHECK(false, "update_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D>
struct update_kernel<at::BFloat16, D> {
  static inline void
  apply(at::BFloat16* __restrict__ out, const float* __restrict__ input, int size, int ld_src, int ld_dst) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    for (int i = 0; i < size; ++i) {
      Unroll<COLS>{}([&](auto col) {
        if constexpr (col % 2 == 0) {
          __m512 va0 = _mm512_loadu_ps(input + i * ld_src + (col + 0) * 16);
          __m512 va1 = _mm512_loadu_ps(input + i * ld_src + (col + 1) * 16);
          __m512i a16 = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
          _mm512_storeu_si512(out + i * ld_dst + col * 16, a16);
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int D>
struct update_value_kernel {
  static inline void apply(
      scalar_t* __restrict__ v_prime2,
      const scalar_t* __restrict__ v,
      const float* __restrict__ v_prime,
      int size,
      int v_strideT) {
    TORCH_CHECK(false, "update_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int D>
struct update_value_kernel<at::BFloat16, D> {
  static inline void apply(
      at::BFloat16* __restrict__ v_prime2,
      const at::BFloat16* __restrict__ v,
      const float* __restrict__ v_prime,
      int size,
      int v_strideT) {
    static_assert(D % 32 == 0);
    constexpr int COLS = D / 16;

    // v2' = v - v'
    for (int i = 0; i < size; ++i) {
      Unroll<COLS>{}([&](auto col) {
        // load for 0, 2, 4, 6
        if constexpr (col % 2 == 0) {
          __m512i v16 = _mm512_loadu_si512(v + i * v_strideT + col * 16);
          __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(v16, 0));
          __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(v16, 1));

          __m512 v_prime0 = _mm512_loadu_ps(v_prime + i * D + col * 16);
          __m512 v_prime1 = _mm512_loadu_ps(v_prime + i * D + col * 16 + 16);
          va0 = _mm512_sub_ps(va0, v_prime0);
          va1 = _mm512_sub_ps(va1, v_prime1);
          __m512i o16 = (__m512i)(_mm512_cvtne2ps_pbh(va1, va0));
          _mm512_storeu_si512(v_prime2 + i * D + col * 16, o16);
        }
      });
    }
  }
};
#endif

template <typename scalar_t, int CHUNK_SIZE, int D>
struct update_key_kernel {
  static inline void apply(
      scalar_t* __restrict__ k_updated,
      const scalar_t* __restrict__ k,
      const float* __restrict__ g,
      int size,
      int k_strideT) {
    TORCH_CHECK(false, "update_key_kernel: scalar path not implemented!");
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int CHUNK_SIZE, int D>
struct update_key_kernel<at::BFloat16, CHUNK_SIZE, D> {
  static inline void apply(
      at::BFloat16* __restrict__ k_updated,
      const at::BFloat16* __restrict__ k,
      const float* __restrict__ g,
      int size,
      int k_strideT) {
    static_assert(D % 32 == 0);
    const int MB = div_up(size, 16);
    const int KB = D / 16;

    const float g_last = g[size - 1];
    const __m512 vg_last = _mm512_set1_ps(g_last);

    float scale_arr[16];
    __m256i va[16];

    // from [C, D](MB, KB) to [D, C](KB, MB)
    // pad size to 16 in this kernel so that transpose can be done in one loop
    for (int mb = 0; mb < MB; ++mb) {
      const int mb_size = std::min(size - mb * 16, 16);
      // prepare exp(g_last - g)
      __m512 vg = _mm512_loadu_ps(g + mb * 16);
      _mm512_storeu_ps(scale_arr, _mm512_fexp_u20_ps(_mm512_sub_ps(vg_last, vg)));
      for (int kb = 0; kb < KB; ++kb) {
        const at::BFloat16* __restrict__ k_ptr = k + mb * 16 * k_strideT + kb * 16;
        at::BFloat16* __restrict__ k_updated_ptr = k_updated + kb * 16 * CHUNK_SIZE + mb * 16;
        // load 16 regs
        Unroll<16>{}([&](auto m) {
          if (m < mb_size) {
            __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_ptr + m * k_strideT));
            __m512 v32 = _mm512_mul_ps(CVT_BF16_TO_FP32(v16), _mm512_set1_ps(scale_arr[m]));
            va[m] = (__m256i)_mm512_cvtneps_pbh(v32);
          } else {
            va[m] = _mm256_setzero_si256();
          }
        });
        // transpose 16x16
        transpose_16x16_16bit(va);
        // store 16 regs
        Unroll<16>{}(
            [&](auto k) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(k_updated_ptr + k * CHUNK_SIZE), va[k]); });
      }
    }
  }
};
#endif

// template head_dim here to reduce extra read
//   * normal approach: read inputs 2 times:
//     - reduce: 1R
//     - scale: 1R + 1W
//   * keep input data in register:
//     - reduce: 1R
//     - scale: 1W
template <typename scalar_t, int D>
void l2norm_fwd_kernel_impl(
    scalar_t* __restrict__ query_norm,
    scalar_t* __restrict__ key_norm,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    float eps,
    int64_t T,
    int64_t H,
    int64_t q_strideT,
    int64_t q_strideH,
    int64_t k_strideT,
    int64_t k_strideH) {
  // expected to be contuguous
  int64_t qn_strideH = D;
  int64_t kn_strideH = D;

  // parallel on [B, T, H]
  at::parallel_for(0, T * H, 0, [&](int64_t begin, int64_t end) {
    int64_t t{0}, h{0};
    data_index_init(begin, t, T, h, H);

    for (int64_t i = begin; i < end; ++i) {
      const scalar_t* __restrict__ q_ptr = query + t * q_strideT + h * q_strideH;
      const scalar_t* __restrict__ k_ptr = key + t * k_strideT + h * k_strideH;
      scalar_t* __restrict__ qn_ptr = query_norm + i * qn_strideH;
      scalar_t* __restrict__ kn_ptr = key_norm + i * kn_strideH;

      l2norm_kernel<scalar_t, D, true>::apply(qn_ptr, q_ptr, eps);
      l2norm_kernel<scalar_t, D, false>::apply(kn_ptr, k_ptr, eps);

      // move to the next index
      data_index_step(t, T, h, H);
    }
  });
}

// g  : [B, T, Hv]
// g_ : [B, Hv, NT, C] -> [B, NT, HB, BLOCK_H, C]
// cu_seqlens : [num_seqs + 1]
// chunk_indices : [NT * 2]
template <typename scalar_t, int CHUNK_SIZE>
void chunk_local_cumsum_kernel_impl(
    scalar_t* __restrict__ g_,
    const scalar_t* __restrict__ g,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_indices,
    int64_t Hv,
    int64_t NT) {
  constexpr int BLOCK_H = 16;
  // TODO: now we only support qwen3.5 configs (H/Hv == 16/32)
  TORCH_CHECK(Hv % BLOCK_H == 0);
  int64_t HB = Hv / BLOCK_H;

  // parallel on [NT * HB] to increase parallelism
  at::parallel_for(0, NT * HB, 0, [&](int64_t begin, int64_t end) {
    int64_t nt{0}, hb{0};
    data_index_init(begin, nt, NT, hb, HB);

    for (int64_t i = begin; i < end; ++i) {
      int32_t bs = chunk_indices[nt * 2 + 0];
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t mb_start = chunk_indices[nt * 2 + 1] * CHUNK_SIZE;
      int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));

      const scalar_t* __restrict__ g_ptr = g + (batch_offset + mb_start) * Hv + hb * BLOCK_H;
      scalar_t* __restrict__ gsum_ptr = g_ + nt * (Hv * CHUNK_SIZE) + hb * (BLOCK_H * CHUNK_SIZE);
      cumsum_kernel<scalar_t, CHUNK_SIZE, BLOCK_H>::apply(gsum_ptr, g_ptr, mb_size, Hv, CHUNK_SIZE);

      // move to the next index
      data_index_step(nt, NT, hb, HB);
    }
  });
}

// w : [B, T, Hv, D]
// u : [B, T, Hv, Dv]
// d : [B, NT, Hv, C, C]
// k : [B, T, H, D]
// v : [B, T, Hv, Dv]
// g : [B, NT, Hv, C]
// beta : [B, T, Hv]
// cu_seqlens : [num_seqs + 1]
// chunk_indices : [NT * 2]
template <typename scalar_t, int D, int CHUNK_SIZE>
void chunk_gated_delta_rule_fwd_intra_kernel_impl(
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ u,
    float* __restrict__ d,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ g,
    const scalar_t* __restrict__ beta,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_indices,
    int64_t H,
    int64_t Hv,
    int64_t NT,
    int64_t k_strideT,
    int64_t k_strideH,
    int64_t v_strideT,
    int64_t v_strideH) {
  // head group, expect to be 1，2，4 for qwen3.5
  const int64_t HG = Hv / H;

  // strides
  const int64_t w_strideT = Hv * D;
  const int64_t w_strideH = D;
  const int64_t u_strideT = Hv * D;
  const int64_t u_strideH = D;

  // [NB]: parallel on [NT, H]
  //   * parallel on num_heads and go sequential on num_heads_v,
  //   * avoid instantialize k_beta (beta * k)
  //   * compute key @ key^T * beta instead of k_beta @ key^T, same as triton impl
  //   * compute key @ key^T once for each k head index and reuse for v head index
  at::parallel_for(0, NT * H, 0, [&](int64_t begin, int64_t end) {
    int64_t nt{0}, h{0};
    data_index_init(begin, nt, NT, h, H);

    // thread local temp buffer
    alignas(64) scalar_t tmp[CHUNK_SIZE * D], tmp2[CHUNK_SIZE * D];
    alignas(64) float attn[CHUNK_SIZE * CHUNK_SIZE];
    alignas(64) scalar_t attn2[CHUNK_SIZE * CHUNK_SIZE];
    alignas(64) float tmp3[CHUNK_SIZE * D];

    // init temp buffer just once for each thread to prevent NaN
    fill_stub(tmp, 0.f, CHUNK_SIZE * D);
    fill_stub(tmp2, 0.f, CHUNK_SIZE * D);
    fill_stub(attn, 0.f, CHUNK_SIZE * CHUNK_SIZE);
    fill_stub(attn2, 0.f, CHUNK_SIZE * CHUNK_SIZE);
    fill_stub(tmp3, 0.f, CHUNK_SIZE * D);

    // alias
    scalar_t* __restrict__ k_packed = tmp;
    scalar_t* __restrict__ k_beta = tmp;
    scalar_t* __restrict__ v_beta = tmp;
    scalar_t* __restrict__ k_beta_packed = tmp2;
    scalar_t* __restrict__ v_beta_packed = tmp2;
    float* __restrict__ k_updated = tmp3;
    float* __restrict__ v_updated = tmp3;

    for (int64_t i = begin; i < end; ++i) {
      int32_t bs = chunk_indices[nt * 2 + 0];
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t mb_start = chunk_indices[nt * 2 + 1] * CHUNK_SIZE;
      int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));

      // mb_size` is K in 5.c, 5.g, pad to TILE_K;
      const int64_t padded_mb_size = div_up((int)mb_size, TILE_K) * TILE_K;

      // step 1: decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
      for (int64_t hv = h * HG; hv < h * HG + HG; ++hv) {
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * CHUNK_SIZE;
        float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        decay_mask_kernel<float, CHUNK_SIZE>::apply(d_ptr, g_ptr);
      }

      // step 2: attn = key @ key^T
      const scalar_t* __restrict__ k_ptr = k + (batch_offset + mb_start) * k_strideT + h * k_strideH;
      pack_vnni<scalar_t>(
          /*    dst */ k_packed,
          /*    src */ k_ptr,
          /*     N  */ mb_size,
          /*     K  */ D,
          /* ld_src */ k_strideT,
          /* ld_dst */ CHUNK_SIZE);

      at::native::cpublas::brgemm(
          /*     M */ mb_size,
          /*     N */ mb_size,
          /*     K */ D,
          /*   lda */ k_strideT,
          /*   ldb */ CHUNK_SIZE,
          /*   ldc */ CHUNK_SIZE,
          /* add_C */ false,
          /*     A */ k_ptr,
          /*     B */ k_packed,
          /*     C */ attn);

      for (int64_t hv = h * HG; hv < h * HG + HG; ++hv) {
        // step 3: attn2 = -attn * beta * d
        const scalar_t* __restrict__ beta_ptr = beta + (batch_offset + mb_start) * Hv + hv;
        const float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        apply_mask_kernel<scalar_t, CHUNK_SIZE>::apply(attn2, attn, beta_ptr, d_ptr, mb_size, Hv);

        // step 4: solve_tril(attn2) -> (I + L)^{-1}, L = strict-lower from step 3
        //   for i in 1..C-1: attn2[i, :i] += (attn2[i, :i] * attn2[:i, :i]).sum(-1)
        //   attn2 += eye(C)
        solve_tril_kernel<scalar_t, CHUNK_SIZE>::apply(attn2, mb_size);

        // step 5: recompute_w_u
        //   w = attn2 @ (k_beta * g.exp().unsqueeze(-1))
        //   u = attn2 @ value * beta.unsqueeze(-1)
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * CHUNK_SIZE;
        const scalar_t* __restrict__ v_ptr = v + (batch_offset + mb_start) * v_strideT + hv * v_strideH;

        //  5.a key = key * beta * g.exp
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, true, true>::apply(
            k_beta, k_ptr, beta_ptr, g_ptr, mb_size, k_strideT, D, Hv);

        //  5.b pack key
        pack_vnni2<scalar_t>(
            /*    dst */ k_beta_packed,
            /*    src */ k_beta,
            /*     K  */ mb_size,
            /*     N  */ D,
            /* ld_src */ D,
            /* ld_dst */ D);

        // 5.c w = attn2 @ k_beta
        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ D,
            /*     K */ padded_mb_size,  // mb_size
            /*   lda */ CHUNK_SIZE,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ false,
            /*     A */ attn2,
            /*     B */ k_beta_packed,
            /*     C */ k_updated);

        // 5.d k_updated -> w
        scalar_t* __restrict__ w_ptr = w + (batch_offset + mb_start) * w_strideT + hv * w_strideH;
        update_kernel<scalar_t, D>::apply(w_ptr, k_updated, mb_size, D, w_strideT);

        // 5.e value = value * beta
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, true, false>::apply(
            v_beta, v_ptr, beta_ptr, nullptr, mb_size, v_strideT, D, Hv);

        // 5.f pack value
        pack_vnni2<scalar_t>(
            /*    dst */ v_beta_packed,
            /*    src */ v_beta,
            /*     K  */ mb_size,
            /*     N  */ D,
            /* ld_src */ D,
            /* ld_dst */ D);

        // 5.g u = attn2 @ v_beta
        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ D,
            /*     K */ padded_mb_size,  // mb_size
            /*   lda */ CHUNK_SIZE,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ false,
            /*     A */ attn2,
            /*     B */ v_beta_packed,
            /*     C */ v_updated);

        // 5.h v_updated -> u
        scalar_t* __restrict__ u_ptr = u + (batch_offset + mb_start) * u_strideT + hv * u_strideH;
        update_kernel<scalar_t, D>::apply(u_ptr, v_updated, mb_size, D, u_strideT);
      }

      // move to the next index
      data_index_step(nt, NT, h, H);
    }
    at::native::cpublas::brgemm_release();
  });
}

//
// out           : [B, T, Hv, Dv]
// state         : [num_seqs, Hv, D, Dv]
// q             : [B, T, H, D]
// k             : [B, T, H, D]
// w             : [B, T, Hv, D]
// u             : [B, T, Hv, Dv]
// g             : [B, NT, Hv, C]
// d             : [B, NT, Hv, C, C]
// cu_seqlens    : [num_seqs + 1]
// chunk_offsets : [num_seqs + 1]
template <typename scalar_t, int D, int CHUNK_SIZE>
void chunk_gated_delta_rule_fwd_inter_kernel_impl(
    scalar_t* __restrict__ out,
    float* __restrict__ state,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ u,
    const float* __restrict__ g,
    const float* __restrict__ d,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    int64_t H,
    int64_t Hv,
    int64_t num_seqs,
    int64_t q_strideT,
    int64_t q_strideH,
    int64_t k_strideT,
    int64_t k_strideH) {
  // head group, expect to be 1，2，4 for qwen3.5
  const int64_t HG = Hv / H;

  // strides
  const int64_t w_strideT = Hv * D;
  const int64_t w_strideH = D;
  const int64_t u_strideT = Hv * D;
  const int64_t u_strideH = D;
  const int64_t o_strideT = Hv * D;
  const int64_t o_strideH = D;

  // [NB]: parallel on [num_seqs, Hv]
  //  * choose to parallel on Hv instead of H, though this means q @ kT has duplicated compute
  //  * H might be 16 which is not enough to use 32C when num_seqs is small
  at::parallel_for(0, num_seqs * Hv, 0, [&](int64_t begin, int64_t end) {
    int64_t bs{0}, hv{0};
    data_index_init(begin, bs, num_seqs, hv, Hv);

    // thread local temp buffer
    alignas(64) scalar_t tmp[CHUNK_SIZE * D];
    alignas(64) scalar_t tmp2[D * D];
    alignas(64) float tmp3[CHUNK_SIZE * D];
    alignas(64) scalar_t tmp4[CHUNK_SIZE * D];
    alignas(64) float attn[CHUNK_SIZE * CHUNK_SIZE];
    alignas(64) scalar_t attn2[CHUNK_SIZE * CHUNK_SIZE];

    // init temp buffer just once for each thread to prevent NaN
    fill_stub(tmp, 0.f, CHUNK_SIZE * D);
    fill_stub(tmp2, 0.f, D * D);
    fill_stub(tmp3, 0.f, CHUNK_SIZE * D);
    fill_stub(tmp4, 0.f, CHUNK_SIZE * D);
    fill_stub(attn, 0.f, CHUNK_SIZE * CHUNK_SIZE);
    fill_stub(attn2, 0.f, CHUNK_SIZE * CHUNK_SIZE);

    // alias
    scalar_t* __restrict__ k_packed = tmp;
    scalar_t* __restrict__ s_packed = tmp2;
    float* __restrict__ v_prime = tmp3;
    scalar_t* __restrict__ v_prime2 = tmp;
    float* __restrict__ attn_inter = tmp3;
    scalar_t* __restrict__ qg_exp = tmp4;
    scalar_t* __restrict__ v_packed = tmp4;
    scalar_t* __restrict__ k_updated = tmp;

    for (int64_t i = begin; i < end; ++i) {
      int64_t h = hv / HG;
      int32_t batch_offset = cu_seqlens[bs];
      int32_t seqlen = cu_seqlens[bs + 1] - cu_seqlens[bs];
      int64_t nt = chunk_offsets[bs];

      for (int64_t mb_start = 0; mb_start < seqlen; mb_start += CHUNK_SIZE, ++nt) {
        int64_t mb_size = std::min(seqlen - mb_start, int64_t(CHUNK_SIZE));

        // mb_size` is K in 4.a, pad to TILE_K;
        const int64_t padded_mb_size = div_up((int)mb_size, TILE_K) * TILE_K;

        // step 1.a: attn = query @ key^T
        // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        const scalar_t* __restrict__ q_ptr = q + (batch_offset + mb_start) * q_strideT + h * q_strideH;
        const scalar_t* __restrict__ k_ptr = k + (batch_offset + mb_start) * k_strideT + h * k_strideH;
        pack_vnni<scalar_t>(
            /*    dst */ k_packed,
            /*    src */ k_ptr,
            /*     N  */ mb_size,
            /*     K  */ D,
            /* ld_src */ k_strideT,
            /* ld_dst */ CHUNK_SIZE);

        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ mb_size,
            /*     K */ D,
            /*   lda */ q_strideT,
            /*   ldb */ CHUNK_SIZE,
            /*   ldc */ CHUNK_SIZE,
            /* add_C */ false,
            /*     A */ q_ptr,
            /*     B */ k_packed,
            /*     C */ attn);

        // step 1.b: attn = attn * decay_mask.masked_fill_(mask, 0)
        const float* __restrict__ d_ptr = d + nt * (Hv * CHUNK_SIZE * CHUNK_SIZE) + hv * (CHUNK_SIZE * CHUNK_SIZE);
        apply_mask_kernel<scalar_t, CHUNK_SIZE>::apply2(attn2, attn, d_ptr, mb_size);

        // step 2.a: v' = w @ state (fuse state *= exp(g_last) with packing)
        float* __restrict__ s_ptr = state + bs * (Hv * D * D) + hv * (D * D);
        const float* __restrict__ g_ptr = g + nt * (Hv * CHUNK_SIZE) + hv * (CHUNK_SIZE);
        float g_last = g_ptr[mb_size - 1];
        pack_vnni2<scalar_t, D, D>(
            /*    dst */ s_packed,
            /*    src */ s_ptr,
            /* g_last */ g_last,
            /* ld_src */ D,
            /* ld_dst */ D);

        const scalar_t* __restrict__ w_ptr = w + (batch_offset + mb_start) * w_strideT + hv * w_strideH;
        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ D,
            /*     K */ D,
            /*   lda */ w_strideT,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ false,
            /*     A */ w_ptr,
            /*     B */ s_packed,
            /*     C */ v_prime);

        // step 2.b: v2' = u - v'
        const scalar_t* __restrict__ u_ptr = u + (batch_offset + mb_start) * u_strideT + hv * u_strideH;
        update_value_kernel<scalar_t, D>::apply(v_prime2, u_ptr, v_prime, mb_size, u_strideT);

        // step 3.a: qg_exp = q * exp(g)
        apply_beta_kernel<scalar_t, CHUNK_SIZE, D, false, true>::apply(
            qg_exp, q_ptr, nullptr, g_ptr, mb_size, q_strideT, D, /*b_stride*/ 0);

        // step 3.b: attn_inter = qg_exp @ state
        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ D,
            /*     K */ D,
            /*   lda */ D,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ false,
            /*     A */ qg_exp,
            /*     B */ s_packed,
            /*     C */ attn_inter);

        // step 4.a: attn_inter += attn2 @ v2'
        pack_vnni2<scalar_t>(
            /*    dst */ v_packed,
            /*    src */ v_prime2,
            /*     K  */ mb_size,
            /*     N  */ D,
            /* ld_src */ D,
            /* ld_dst */ D);

        at::native::cpublas::brgemm(
            /*     M */ mb_size,
            /*     N */ D,
            /*     K */ padded_mb_size,
            /*   lda */ CHUNK_SIZE,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ true,
            /*     A */ attn2,
            /*     B */ v_packed,
            /*     C */ attn_inter);

        // step 4.b: write attn_inter -> out
        scalar_t* __restrict__ o_ptr = out + (batch_offset + mb_start) * o_strideT + hv * o_strideH;
        update_kernel<scalar_t, D>::apply(o_ptr, attn_inter, mb_size, D, o_strideT);

        // step 5: update state
        //   state_new = state * exp(g_last) + (k * exp(g_last - g)).T @ v2'

        // step 5.1 state *= exp(g_last) fused with step 2.a

        // step 5.2 k' = k * exp(g_last - g).T; TODO: fuse this with 1.a
        update_key_kernel<scalar_t, CHUNK_SIZE, D>::apply(k_updated, k_ptr, g_ptr, mb_size, k_strideT);

        // step 5.3 state += k' @ v2'
        at::native::cpublas::brgemm(
            /*     M */ D,
            /*     N */ D,
            /*     K */ padded_mb_size,  // mb_size
            /*   lda */ CHUNK_SIZE,
            /*   ldb */ D,
            /*   ldc */ D,
            /* add_C */ true,
            /*     A */ k_updated,
            /*     B */ v_packed,
            /*     C */ s_ptr);
      }

      // move to the next index
      data_index_step(bs, num_seqs, hv, Hv);
    }
  });
}

template <typename scalar_t, int64_t chunk_size = 64>
void chunk_gated_delta_rule_kernel_impl(
    scalar_t* __restrict__ out,                  // [B, T, HV, EV]
    float* __restrict__ final_state_data,        // [N, HV, EK, EV]
    const scalar_t* __restrict__ q_orig,         // [B, T, HK, EK]
    const scalar_t* __restrict__ k_orig,         // [B, T, HK, EK]
    const scalar_t* __restrict__ v_orig,         // [B, T, HV, EV]
    const float* __restrict__ g_orig,            // [B, T, HV] FP32
    const scalar_t* __restrict__ b_orig,         // [B, T, HV]
    const int32_t* __restrict__ cu_seqlens_ptr,  // [N + 1] INT32
    float* __restrict__ buff,
    scalar_t* __restrict__ reduced_buff,
    scalar_t* __restrict__ thread_buff,
    const int32_t* __restrict__ chunk_offsets_ptr,
    const int32_t* __restrict__ chunk_indices_ptr,
    bool use_qk_l2norm_in_kernel,
    const int64_t& batch_size,
    const int64_t& global_seq_len,
    const int64_t& qk_num_head,
    const int64_t& v_num_head,
    const int64_t& qk_head_size,
    const int64_t& v_head_size,
    const int64_t& qStrideH,
    const int64_t& qStrideT,
    const int64_t& kStrideH,
    const int64_t& kStrideT,
    const int64_t& vStrideH,
    const int64_t& vStrideT,
    const int64_t& oStrideH,
    const int64_t& oStrideT,
    const int64_t& global_total_seq_length,
    const int64_t& global_num_chunk,
    const int64_t& buff_size_16bit_per_thread,
    double eps = 1e-5) {
  int64_t gStrideH = 1;
  int64_t gStrideT = v_num_head;
  int64_t bStrideH = 1;
  int64_t bStrideT = v_num_head;
  int64_t final_state_StrideN = v_num_head * qk_head_size * v_head_size;
  int64_t final_state_StrideH = qk_head_size * v_head_size;
  int64_t final_state_StrideE = v_head_size;
  int64_t head_group = v_num_head / qk_num_head;
  float scale = 1.0 / std::sqrt(qk_head_size);
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();

  // Data pointers
  float* g_pad = buff;
  float* core_attn_out = g_pad + v_num_head * global_total_seq_length;
  float* decay_mask = core_attn_out + batch_size * v_num_head * global_total_seq_length * v_head_size;
  float* v_beta_attn = decay_mask + v_num_head * global_total_seq_length * chunk_size;

  scalar_t* q_pad = reduced_buff;
  scalar_t* k_pad = q_pad + qk_num_head * global_total_seq_length * qk_head_size;
  scalar_t* v_pad = k_pad + qk_num_head * global_total_seq_length * qk_head_size;
  scalar_t* k_beta = v_pad + v_num_head * global_total_seq_length * v_head_size;
  scalar_t* v_beta = k_beta + v_num_head * global_total_seq_length * qk_head_size;
  scalar_t* k_cumdecay_reduced = v_beta + v_num_head * global_total_seq_length * v_head_size;
  scalar_t* q_norm_sum = k_cumdecay_reduced + v_num_head * global_total_seq_length * qk_head_size;
  scalar_t* k_norm_sum = q_norm_sum + qk_num_head * global_seq_len;

  if (use_qk_l2norm_in_kernel) {
    at::parallel_for(0, qk_num_head * global_seq_len, 0, [&](int64_t begin, int64_t end) {
      int64_t h_qk = 0, l = 0;
      data_index_init(begin, h_qk, qk_num_head, l, global_seq_len);
      for (int64_t i = begin; i < end; ++i) {
        auto q_norm_sum_ptr = q_norm_sum + h_qk * global_seq_len + l;
        auto k_norm_sum_ptr = k_norm_sum + h_qk * global_seq_len + l;
        float sum_q = float(0);
        float sum_k = float(0);
        fVec sum_q_fvec = fVec(float(0));
        fVec sum_k_fvec = fVec(float(0));
        int64_t q_offset = l * qStrideT + h_qk * qStrideH;
        int64_t k_offset = l * qStrideT + h_qk * qStrideH;
        int64_t d;
        for (d = 0; d <= qk_head_size - VecSize; d += VecSize) {
          bVec q_bvec = bVec::loadu(q_orig + q_offset + d);
          fVec q_fvec0, q_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_bvec);
          sum_q_fvec += q_fvec0 * q_fvec0;
          sum_q_fvec += q_fvec1 * q_fvec1;
          bVec k_bvec = bVec::loadu(k_orig + k_offset + d);
          fVec k_fvec0, k_fvec1;
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_bvec);
          sum_k_fvec += k_fvec0 * k_fvec0;
          sum_k_fvec += k_fvec1 * k_fvec1;
        }
        sum_q += vec_reduce_sum(sum_q_fvec);
        sum_k += vec_reduce_sum(sum_k_fvec);
        q_norm_sum_ptr[0] = static_cast<scalar_t>(float(1) / std::sqrt(sum_q + eps));
        k_norm_sum_ptr[0] = static_cast<scalar_t>(float(1) / std::sqrt(sum_k + eps));
        data_index_step(h_qk, qk_num_head, l, global_seq_len);
      }
    });
  }

  // query = query * scale
  // k_beta = key * beta.unsqueeze(-1)
  // v_beta = value * beta.unsqueeze(-1)
  // Padding for q/k/v/beta
  at::parallel_for(0, qk_num_head * global_num_chunk, 1, [&](int64_t begin, int64_t end) {
    int ompIdx = at::get_thread_num();
    int64_t h_qk = 0, c = 0;
    data_index_init(begin, h_qk, qk_num_head, c, global_num_chunk);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t ib = chunk_indices_ptr[c * 2];      // idx_batch
      int64_t ic = chunk_indices_ptr[c * 2 + 1];  // idx_chunk
      int64_t l_orig = cu_seqlens_ptr[ib] + ic * chunk_size;
      int64_t l = c * chunk_size;
      bool is_tail = (c + 1 == chunk_offsets_ptr[ib + 1]);
      int64_t seq_len = cu_seqlens_ptr[ib + 1] - cu_seqlens_ptr[ib];
      int64_t real_chunk_size = is_tail ? seq_len - ic * chunk_size : chunk_size;
      auto q_orig_ptr = q_orig + h_qk * qStrideH + l_orig * qStrideT;
      auto k_orig_ptr = k_orig + h_qk * kStrideH + l_orig * kStrideT;
      auto v_orig_ptr = v_orig + l_orig * vStrideT;
      auto b_orig_ptr = b_orig + l_orig * bStrideT;
      auto q_pad_ptr = q_pad + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
      auto k_pad_ptr = k_pad + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
      auto v_pad_ptr = v_pad + l * v_head_size;
      auto k_beta_ptr = k_beta + l * qk_head_size;
      auto v_beta_ptr = v_beta + l * v_head_size;

      for (int64_t j = 0; j < real_chunk_size; j++) {
        auto curr_q_orig = q_orig_ptr + j * qStrideT;
        auto curr_k_orig = k_orig_ptr + j * kStrideT;
        auto curr_q_pad = q_pad_ptr + j * qk_head_size;
        auto curr_k_pad = k_pad_ptr + j * qk_head_size;
        auto q_scale =
            use_qk_l2norm_in_kernel ? *(q_norm_sum + h_qk * global_seq_len + l_orig + j) : static_cast<scalar_t>(1);
        auto k_scale =
            use_qk_l2norm_in_kernel ? *(k_norm_sum + h_qk * global_seq_len + l_orig + j) : static_cast<scalar_t>(1);
        auto q_scale_vec = bVec(q_scale);
        auto k_scale_vec = bVec(k_scale);
        int64_t i = 0;
        scalar_t scale_reduced = static_cast<scalar_t>(scale);
        auto vec_scale_reduced = bVec(scale_reduced);
        for (; i < fVecSize * (qk_head_size / fVecSize); i += fVecSize) {
          auto tmp0 = bVec::loadu(curr_q_orig + i, fVecSize);
          auto tmp1 = tmp0 * q_scale_vec * vec_scale_reduced;
          tmp1.store(curr_q_pad + i, fVecSize);
          auto tmp3 = bVec::loadu(curr_k_orig + i, fVecSize);
          auto tmp4 = tmp3 * k_scale_vec;
          tmp4.store(curr_k_pad + i, fVecSize);
        }

        for (auto hi = 0; hi < head_group; hi++) {
          int64_t h = h_qk * head_group + hi;
          auto curr_v_orig = v_orig_ptr + h * vStrideH + j * vStrideT;
          auto curr_b_orig = b_orig_ptr + h * bStrideH + j * bStrideT;
          scalar_t b_orig_val_reduced = *(curr_b_orig);
          auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + j * qk_head_size;
          auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;

          // query = query * scale
          // k_beta = key * beta.unsqueeze(-1)
          int64_t i = 0;
          auto vec_b_reduced = bVec(b_orig_val_reduced);
          for (; i < fVecSize * (qk_head_size / fVecSize); i += fVecSize) {
            auto tmp0 = bVec::loadu(curr_k_orig + i, fVecSize);
            auto tmp2 = tmp0 * k_scale_vec * vec_b_reduced;
            tmp2.store(curr_k_beta + i, fVecSize);
          }
          // v_beta = value * beta.unsqueeze(-1)
          i = 0;
          for (; i < VecSize * (v_head_size / VecSize); i += VecSize) {
            auto tmp3 = bVec::loadu(curr_v_orig + i);
            tmp3.store(curr_v_pad + i);
            auto tmp5 = tmp3 * vec_b_reduced;
            tmp5.store(curr_v_beta + i);
          }
        }
      }

      for (int64_t j = real_chunk_size; j < chunk_size; j++) {
        auto curr_q_pad = q_pad_ptr + j * qk_head_size;
        auto curr_k_pad = k_pad_ptr + j * qk_head_size;
        int64_t i = 0;
        auto vec_zero = bVec(0.0);
        for (; i < VecSize * (qk_head_size / VecSize); i += VecSize) {
          vec_zero.store(curr_q_pad + i);
          vec_zero.store(curr_k_pad + i);
        }
        for (auto hi = 0; hi < head_group; hi++) {
          int64_t h = h_qk * head_group + hi;
          auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + j * qk_head_size;
          auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + j * v_head_size;
          int64_t i = 0;
          for (; i < VecSize * (qk_head_size / VecSize); i += VecSize) {
            vec_zero.store(curr_k_beta + i);
          }
          i = 0;
          for (; i < VecSize * (v_head_size / VecSize); i += VecSize) {
            vec_zero.store(curr_v_pad + i);
            vec_zero.store(curr_v_beta + i);
          }
        }
      }
      // Move to the next query
      data_index_step(h_qk, qk_num_head, c, global_num_chunk);
    }
  });

  at::parallel_for(0, v_num_head * global_num_chunk, 1, [&](int64_t begin, int64_t end) {
    int64_t h = 0, c = 0;
    data_index_init(begin, h, v_num_head, c, global_num_chunk);
    int ompIdx = at::get_thread_num();
    int64_t offset = 0;
    scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
    THREAD_BUFFER_ALLOC(k_transpose, thread_buff_ptr, offset, scalar_t, qk_head_size * chunk_size);
    THREAD_BUFFER_ALLOC(v_pack, thread_buff_ptr, offset, scalar_t, chunk_size * v_head_size);
    THREAD_BUFFER_ALLOC(k_beta_g, thread_buff_ptr, offset, scalar_t, chunk_size * qk_head_size);
    THREAD_BUFFER_ALLOC(k_beta_g_pack, thread_buff_ptr, offset, scalar_t, chunk_size * qk_head_size);
    THREAD_BUFFER_ALLOC(curr_attn, thread_buff_ptr, offset, float, chunk_size* chunk_size * 2);
    THREAD_BUFFER_ALLOC(curr_attn_reduced, thread_buff_ptr, offset, scalar_t, chunk_size * chunk_size);
    THREAD_BUFFER_ALLOC(k_cumdecay, thread_buff_ptr, offset, float, chunk_size* qk_head_size * 2);
    THREAD_BUFFER_ALLOC(row, thread_buff_ptr, offset, float, chunk_size * 2);
    THREAD_BUFFER_ALLOC(updated, thread_buff_ptr, offset, float, chunk_size * 2);
    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t ib = chunk_indices_ptr[c * 2];      // idx_batch
      int64_t ic = chunk_indices_ptr[c * 2 + 1];  // idx_chunk
      int64_t l_orig = cu_seqlens_ptr[ib] + ic * chunk_size;
      int64_t seq_len = cu_seqlens_ptr[ib + 1] - cu_seqlens_ptr[ib];
      int64_t h_qk = h / head_group;
      auto curr_g_orig = g_orig + h * gStrideH + l_orig * gStrideT;
      auto curr_g_pad = g_pad + h * global_total_seq_length + c * chunk_size;
      auto curr_decay_mask = decay_mask + h * global_total_seq_length * chunk_size + c * chunk_size * chunk_size;
      auto curr_k_pad = k_pad + h_qk * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_k_beta = k_beta + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_k_cumdecay_reduced =
          k_cumdecay_reduced + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
      auto curr_v_beta = v_beta + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;
      auto curr_value = v_beta_attn + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;

      float acc_val = 0;
      for (int64_t i = 0; i < chunk_size; i++) {
        // Padding for g
        // g = g.cumsum(dim=-1)
        // g: [B, HV, num_chunk, chunk_size]
        if (ic * chunk_size + i < seq_len) {
          acc_val += curr_g_orig[i * gStrideT];
        }
        curr_g_pad[i] = acc_val;
        // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        // decay_mask: [B, HV, num_chunk, chunk_size, chunk_size]
        float curr_g_pad_i = static_cast<float>(curr_g_pad[i]);
        auto vec_curr_g_pad_i = fVec(curr_g_pad_i);
        int64_t j = 0;
        int64_t len = i + 1;
        for (; j < fVecSize * (len / fVecSize); j += fVecSize) {
          auto tmp0 = fVec::loadu(curr_g_pad + j);
          auto tmp1 = vec_curr_g_pad_i - tmp0;
          auto tmp2 = tmp1.exp_u20();
          tmp2.store(curr_decay_mask + i * chunk_size + j);
        }
        if (j < len) {
          auto tmp0 = fVec::loadu(curr_g_pad + j, len - j);
          auto tmp1 = vec_curr_g_pad_i - tmp0;
          auto tmp2 = tmp1.exp_u20();
          tmp2.store(curr_decay_mask + i * chunk_size + j, len - j);
        }
      }

      // attn = k_beta @ key.transpose(-1, -2)
      // attn: [B, HV, num_chunk, chunk_size, chunk_size]
      // transpose and pack for key
      pack_vnni<scalar_t>(
          /*    dst */ k_transpose,
          /*    src */ curr_k_pad,
          /*     N  */ chunk_size,
          /*     K  */ qk_head_size,
          /* ld_src */ qk_head_size,
          /* ld_dst */ chunk_size);
      // k_beta @ key.transpose(-1, -2)
      at::native::cpublas::brgemm(
          /*     M */ chunk_size,
          /*     N */ chunk_size,
          /*     K */ qk_head_size,
          /*   lda */ qk_head_size,
          /*   ldb */ chunk_size,
          /*   ldc */ chunk_size,
          /* add_C */ false,
          /*     A */ curr_k_beta,
          /*     B */ k_transpose,
          /*     C */ curr_attn);
      // attn = attn * decay_mask
      for (int64_t m = 0; m < chunk_size; m++) {
        at::vec::map2<float>(
            [](fVec x, fVec y) { return fVec(0) - x * y; },
            curr_attn + m * chunk_size,
            curr_attn + m * chunk_size,
            curr_decay_mask + m * chunk_size,
            chunk_size);
      }

      // chunk decay
      // attn: [B, HV, num_chunk, chunk_size, chunk_size]
      // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
      // attn = -attn.masked_fill(mask, 0)
      // attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2) [B, HV, num_chunk, i]
      // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
      // attn = -attn.masked_fill(mask, 0)
      for (int i = 0; i < chunk_size; i++) {
        const auto vec_zero = fVec(0);
        int64_t len = chunk_size - i;
        int64_t front = len % fVecSize;
        int64_t j = i;
        // first masked vec for alignment
        if (front > 0) {
          vec_zero.store(curr_attn + i * chunk_size + j, front);
          j += front;
        }
        for (; j < fVecSize * (chunk_size / fVecSize); j += fVecSize) {
          vec_zero.store(curr_attn + i * chunk_size + j);
        }
      }
      for (int i = 1; i < chunk_size; i++) {
        // row = attn[..., i, :i] [B, HK, num_chunk, i]
        int64_t j = 0;
        int64_t len = i;
        for (; j < fVecSize * (len / fVecSize); j += fVecSize) {
          auto tmp0 = fVec::loadu(curr_attn + i * chunk_size + j);
          tmp0.store(row + j);
        }
        if (j < len) {
          auto tmp0 = fVec::loadu(curr_attn + i * chunk_size + j, len - j);
          tmp0.store(row + j, len - j);
        }
        // (row.unsqueeze(-1) * sub).sum(-2)
        fill_stub(updated, 0, i);
        for (int k = 0; k < i; k++) {
          float row_k = row[k];
          auto vec_row_k = fVec(row_k);
          int64_t j = 0;
          int64_t len = i;
          for (; j < fVecSize * (len / fVecSize); j += fVecSize) {
            auto tmp0 = fVec::loadu(curr_attn + k * chunk_size + j);
            auto tmp1 = vec_row_k * tmp0;
            auto tmp2 = fVec::loadu(updated + j);
            auto tmp3 = tmp1 + tmp2;
            tmp3.store(updated + j);
          }
          if (j < len) {
            auto tmp0 = fVec::loadu(curr_attn + k * chunk_size + j, len - j);
            auto tmp1 = vec_row_k * tmp0;
            auto tmp2 = fVec::loadu(updated + j);
            auto tmp3 = tmp1 + tmp2;
            tmp3.store(updated + j, len - j);
          }
        }
        // attn[..., i, :i] = row + sum(...)
        j = 0;
        len = i;
        for (; j < fVecSize * (len / fVecSize); j += fVecSize) {
          auto tmp0 = fVec::loadu(row + j);
          auto tmp1 = fVec::loadu(updated + j);
          auto tmp2 = tmp0 + tmp1;
          tmp2.store(curr_attn + i * chunk_size + j);
        }
        if (j < len) {
          auto tmp0 = fVec::loadu(row + j, len - j);
          auto tmp1 = fVec::loadu(updated + j, len - j);
          auto tmp2 = tmp0 + tmp1;
          tmp2.store(curr_attn + i * chunk_size + j, len - j);
        }
      }
      for (int i = 0; i < chunk_size; i++) {
        curr_attn[i * chunk_size + i] += 1.0f;
        at::vec::map<scalar_t>(
            [](fVec x) { return x; }, curr_attn_reduced + i * chunk_size, curr_attn + i * chunk_size, chunk_size);
      }

      // v_beta_attn = attn @ v_beta
      // k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
      // v_beta_attn: [B, HV, num_chunk, chunk_size, EV]
      // k_beta_g = k_beta * g: [B, HV, num_chunk, chunk_size, EK]
      // k_cumdecay: [B, HV, num_chunk, chunk_size, EK]
      // pack for value
      pack_vnni2<scalar_t>(
          /*    dst */ v_pack,
          /*    src */ curr_v_beta,
          /*     N  */ chunk_size,
          /*     K  */ v_head_size,
          /* ld_src */ v_head_size,
          /* ld_dst */ v_head_size);
      // value = attn @ v_beta
      at::native::cpublas::brgemm(
          /*     M */ chunk_size,
          /*     N */ v_head_size,
          /*     K */ chunk_size,
          /*   lda */ chunk_size,
          /*   ldb */ v_head_size,
          /*   ldc */ v_head_size,
          /* add_C */ false,
          /*     A */ curr_attn_reduced,
          /*     B */ v_pack,
          /*     C */ curr_value);
      // k_beta_g = k_beta * g.exp().unsqueeze(-1)
      for (int64_t j = 0; j < chunk_size; j++) {
        int64_t i = 0;
        float g_exp = std::exp(curr_g_pad[j]);
        scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
        auto vec_g_exp_reduced = bVec(g_exp_reduced);
        for (; i < VecSize * (qk_head_size / VecSize); i += VecSize) {
          auto tmp0 = bVec::loadu(curr_k_beta + j * qk_head_size + i);
          auto tmp1 = tmp0 * vec_g_exp_reduced;
          tmp1.store(k_beta_g + j * qk_head_size + i);
        }
      }
      // pack for k_beta_g
      pack_vnni2<scalar_t>(
          /*    dst */ k_beta_g_pack,
          /*    src */ k_beta_g,
          /*     N  */ chunk_size,
          /*     K  */ qk_head_size,
          /* ld_src */ qk_head_size,
          /* ld_dst */ qk_head_size);
      // k_cumdecay = attn @ k_beta_g
      at::native::cpublas::brgemm(
          /*     M */ chunk_size,
          /*     N */ qk_head_size,
          /*     K */ chunk_size,
          /*   lda */ chunk_size,
          /*   ldb */ qk_head_size,
          /*   ldc */ qk_head_size,
          /* add_C */ false,
          /*     A */ curr_attn_reduced,
          /*     B */ k_beta_g_pack,
          /*     C */ k_cumdecay);
      for (int i = 0; i < chunk_size; i++) {
        at::vec::map<scalar_t>(
            [](fVec x) { return x; },
            curr_k_cumdecay_reduced + i * qk_head_size,
            k_cumdecay + i * qk_head_size,
            qk_head_size);
      }

      // Move to the next query
      data_index_step(h, v_num_head, c, global_num_chunk);
    }
  });

  // for each chunk
  at::parallel_for(0, batch_size * v_num_head, 1, [&](int64_t begin, int64_t end) {
    int64_t b = 0, h = 0;
    data_index_init(begin, b, batch_size, h, v_num_head);
    int ompIdx = at::get_thread_num();
    int64_t offset =
        /* k_transpose */ qk_head_size * chunk_size +
        /* v_pack */ chunk_size * v_head_size +
        /* k_beta_g  */ chunk_size * qk_head_size +
        /* k_beta_g_pack  */ chunk_size * qk_head_size +
        /* attn */ chunk_size * chunk_size * 2 +
        /* attn_reduced */ chunk_size * chunk_size +
        /* k_cumdecay */ chunk_size * qk_head_size * 2 +
        /* row */ chunk_size * 2 +
        /* updated */ chunk_size * 2;
    scalar_t* thread_buff_ptr = thread_buff + ompIdx * buff_size_16bit_per_thread;
    THREAD_BUFFER_ALLOC(
        curr_last_recurrent_state_reduced, thread_buff_ptr, offset, scalar_t, qk_head_size * v_head_size);
    THREAD_BUFFER_ALLOC(
        curr_last_recurrent_state_pack_reduced, thread_buff_ptr, offset, scalar_t, qk_head_size * v_head_size);
    THREAD_BUFFER_ALLOC(k_transpose_i, thread_buff_ptr, offset, scalar_t, qk_head_size * chunk_size);
    THREAD_BUFFER_ALLOC(attn_i, thread_buff_ptr, offset, float, chunk_size* chunk_size * 2);
    THREAD_BUFFER_ALLOC(attn_i_reduced, thread_buff_ptr, offset, scalar_t, chunk_size * chunk_size);
    THREAD_BUFFER_ALLOC(v_prime, thread_buff_ptr, offset, float, chunk_size* v_head_size * 2);
    THREAD_BUFFER_ALLOC(v_prime_reduced, thread_buff_ptr, offset, scalar_t, chunk_size * v_head_size);
    THREAD_BUFFER_ALLOC(v_prime_pack_reduced, thread_buff_ptr, offset, scalar_t, chunk_size * v_head_size);
    THREAD_BUFFER_ALLOC(qg, thread_buff_ptr, offset, scalar_t, chunk_size * qk_head_size);
    THREAD_BUFFER_ALLOC(attn_inter, thread_buff_ptr, offset, float, chunk_size* v_head_size * 2);
    THREAD_BUFFER_ALLOC(kg, thread_buff_ptr, offset, scalar_t, chunk_size * qk_head_size);
    THREAD_BUFFER_ALLOC(kg_transpose, thread_buff_ptr, offset, scalar_t, qk_head_size * chunk_size);
    THREAD_BUFFER_ALLOC(kgv, thread_buff_ptr, offset, float, qk_head_size* v_head_size * 2);

    for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
      int64_t start_q = cu_seqlens_ptr[b];
      int64_t seq_len = cu_seqlens_ptr[b + 1] - start_q;
      int64_t num_chunk = chunk_offsets_ptr[b + 1] - chunk_offsets_ptr[b];
      int64_t chunk_offset = chunk_offsets_ptr[b];
      int64_t len_offset = chunk_offset * chunk_size;

      int64_t h_qk = h / head_group;
      auto out_ptr = out + start_q * oStrideT;
      auto curr_q = q_pad + len_offset * qk_head_size +
                    h_qk * global_total_seq_length * qk_head_size;  // [num_chunk, chunk_size, EK]
      auto curr_k = k_pad + len_offset * qk_head_size +
                    h_qk * global_total_seq_length * qk_head_size;            // [num_chunk, chunk_size, EK]
      auto curr_v = v_beta_attn + h * global_total_seq_length * v_head_size;  // [num_chunk, chunk_size, EV]
      auto curr_decay_mask =
          decay_mask + h * global_total_seq_length * chunk_size;  // [num_chunk, chunk_size, chunk_size]
      auto curr_k_cumdecay_reduced =
          k_cumdecay_reduced + h * global_total_seq_length * qk_head_size;  // [num_chunk, chunk_size, EK]
      auto curr_last_recurrent_state =
          final_state_data + b * final_state_StrideN + h * final_state_StrideH;  // [EK, EV]
      auto curr_g_pad = g_pad + len_offset + h * global_total_seq_length;        // [num_chunk, chunk_size]
      auto curr_core_attn_out = core_attn_out + len_offset * v_head_size +
                                h * global_total_seq_length * v_head_size;  // [num_chunk, chunk_size, EV]
      for (int64_t c = 0; c < num_chunk; c++) {
        for (int i = 0; i < qk_head_size; i++) {
          at::vec::map<scalar_t>(
              [](fVec x) { return x; },
              curr_last_recurrent_state_reduced + i * v_head_size,
              curr_last_recurrent_state + i * v_head_size,
              v_head_size);
        }
        auto q_i = curr_q + c * chunk_size * qk_head_size;                                   // [chunk_size, EK]
        auto k_i = curr_k + c * chunk_size * qk_head_size;                                   // [chunk_size, EK]
        auto v_i = curr_v + (chunk_offset + c) * chunk_size * v_head_size;                   // [chunk_size, EV]
        auto decay_mask_i = curr_decay_mask + (chunk_offset + c) * chunk_size * chunk_size;  // [chunk_size, chunk_size]
        auto k_cumdecay_i_reduced =
            curr_k_cumdecay_reduced + (chunk_offset + c) * chunk_size * qk_head_size;  // [chunk_size, EK]
        auto g_pad_i = curr_g_pad + c * chunk_size;                                    // [chunk_size]
        auto core_attn_out_i = curr_core_attn_out + c * chunk_size * v_head_size;      // [chunk_size, EV]

        // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        // k_transpose_i = k_i.transpose(-1, -2)
        pack_vnni<scalar_t>(
            /*    dst */ k_transpose_i,
            /*    src */ k_i,
            /*     N  */ chunk_size,
            /*     K  */ qk_head_size,
            /* ld_src */ qk_head_size,
            /* ld_dst */ chunk_size);
        // attn_i = q_i @ k_transpose_i
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ chunk_size,
            /* K */ qk_head_size,
            /* lda */ qk_head_size,
            /* ldb */ chunk_size,
            /* ldc */ chunk_size,
            /* add_C */ false,
            /* A */ q_i,
            /* B */ k_transpose_i,
            /* C */ attn_i);
        // attn_i = attn_i * decay_mask_i
        for (int64_t m = 0; m < chunk_size; m++) {
          auto attn_i_m = attn_i + m * chunk_size;
          auto attn_i_reduced_m = attn_i_reduced + m * chunk_size;
          auto decay_mask_i_m = decay_mask_i + m * chunk_size;
          int64_t n = 0;
          for (; n < fVecSize * (chunk_size / fVecSize); n += fVecSize) {
            auto tmp0 = fVec::loadu(attn_i_m + n);
            auto tmp1 = fVec::loadu(decay_mask_i_m + n);
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(attn_i_reduced_m + n, fVecSize);
          }
          if (n < chunk_size) {
            auto tmp0 = fVec::loadu(attn_i_m + n, chunk_size - n);
            auto tmp1 = fVec::loadu(decay_mask_i_m + n, chunk_size - n);
            auto tmp2 = tmp0 * tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(attn_i_reduced_m + n, chunk_size - n);
          }
        }
        // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
        // attn_i = attn_i.masked_fill_(mask, 0)
        for (int i = 0; i < chunk_size - 1; i++) {
          const auto vec_zero = bVec(0);
          int64_t len = chunk_size - i - 1;
          int64_t front = len % VecSize;
          int64_t j = i + 1;
          // first masked vec for alignment
          if (front > 0) {
            vec_zero.store(attn_i_reduced + i * chunk_size + j, front);
            j += front;
          }
          for (; j < VecSize * (chunk_size / VecSize); j += VecSize) {
            vec_zero.store(attn_i_reduced + i * chunk_size + j);
          }
        }

        // pack for curr_last_recurrent_state
        pack_vnni2<scalar_t>(
            /*    dst */ curr_last_recurrent_state_pack_reduced,
            /*    src */ curr_last_recurrent_state_reduced,
            /*     N  */ qk_head_size,
            /*     K  */ v_head_size,
            /* ld_src */ v_head_size,
            /* ld_dst */ v_head_size);

        // v_prime = k_cumdecay_i @ curr_last_recurrent_state: [chunk_size, EV]
        // k_cumdecay_i: [chunk_size, EK]
        // curr_last_recurrent_state: [EK, EV]
        at::native::cpublas::brgemm(
            /*     M */ chunk_size,
            /*     N */ v_head_size,
            /*     K */ qk_head_size,
            /*   lda */ qk_head_size,
            /*   ldb */ v_head_size,
            /*   ldc */ v_head_size,
            /* add_C */ false,
            /*     A */ k_cumdecay_i_reduced,
            /*     B */ curr_last_recurrent_state_pack_reduced,
            /*     C */ v_prime);

        // v_new = v_prime = v_i - v_prime
        // v_i: [chunk_size, EV]
        for (int64_t m = 0; m < chunk_size; m++) {
          int64_t i = 0;
          for (; i < fVecSize * (v_head_size / fVecSize); i += fVecSize) {
            auto tmp0 = fVec::loadu(v_i + m * v_head_size + i);
            auto tmp1 = fVec::loadu(v_prime + m * v_head_size + i);
            auto tmp2 = tmp0 - tmp1;
            auto tmp3 = at::vec::convert<scalar_t>(tmp2);
            tmp3.store(v_prime_reduced + m * v_head_size + i, fVecSize);
          }
        }

        // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        // qg = q_i * g[:, :, i, :, None].exp(): [chunk_size, EK]
        // q_i: [chunk_size, EK]
        // g[:, :, i, :, None]: [chunk_size, 1]
        for (int64_t m = 0; m < chunk_size; m++) {
          auto g_pad_i_m = g_pad_i + m;
          auto g_exp = std::exp(*g_pad_i_m);
          int64_t i = 0;
          scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
          auto vec_g_exp_reduced = bVec(g_exp_reduced);
          for (; i < VecSize * (qk_head_size / VecSize); i += VecSize) {
            auto tmp0 = bVec::loadu(q_i + m * qk_head_size + i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(qg + m * qk_head_size + i);
          }
        }
        // attn_inter = qg @ curr_last_recurrent_state: [chunk_size, EV]
        // curr_last_recurrent_state: [EK, EV]
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ v_head_size,
            /* K */ qk_head_size,
            /* lda */ qk_head_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ false,
            /* A */ qg,
            /* B */ curr_last_recurrent_state_pack_reduced,
            /* C */ attn_inter);

        // core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        // pack for v_prime
        pack_vnni2<scalar_t>(
            /*    dst */ v_prime_pack_reduced,
            /*    src */ v_prime_reduced,
            /*     N  */ chunk_size,
            /*     K  */ v_head_size,
            /* ld_src */ v_head_size,
            /* ld_dst */ v_head_size);
        // attn_inter = attn_inter + attn_i @ v_new: [chunk_size, EV]
        // attn_i: [chunk_size, chunk_size]
        // v_new: [chunk_size, EV]
        at::native::cpublas::brgemm(
            /* M */ chunk_size,
            /* N */ v_head_size,
            /* K */ chunk_size,
            /* lda */ chunk_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ true,
            /* A */ attn_i_reduced,
            /* B */ v_prime_pack_reduced,
            /* C */ attn_inter);

        // core_attn_out[:, :, i] = attn_inter
        for (int64_t m = 0; m < chunk_size; m++) {
          at::vec::map<float>(
              [](fVec x) { return x; }, core_attn_out_i + m * v_head_size, attn_inter + m * v_head_size, v_head_size);
        }

        // last_recurrent_state = (
        //     last_recurrent_state * g[:, :, i, -1, None, None].exp()
        //     + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        // )
        // 1) last_recurrent_state * g[:, :, i, -1, None, None].exp()
        // curr_last_recurrent_state: [EK, EV]
        // g[:, :, i, -1, None, None]: [1, 1]
        // last_recurrent_state * g[:, :, i, -1, None, None].exp(): [EK, EV]
        auto g_pad_i_last = g_pad_i + chunk_size - 1;
        auto g_exp_last = std::exp(g_pad_i_last[0]);
        for (int64_t m = 0; m < qk_head_size; m++) {
          int64_t i = 0;
          auto vec_g_exp_last = fVec(g_exp_last);
          for (; i < fVecSize * (v_head_size / fVecSize); i += fVecSize) {
            auto tmp0 = bVec::loadu(curr_last_recurrent_state_reduced + m * v_head_size + i);
            auto tmp1 = at::vec::convert<float>(tmp0);
            auto tmp2 = tmp1 * vec_g_exp_last;
            tmp2.store(curr_last_recurrent_state + m * v_head_size + i);
          }
          if (i < v_head_size) {
            auto tmp0 = bVec::loadu(curr_last_recurrent_state_reduced + m * v_head_size + i, v_head_size - i);
            auto tmp1 = at::vec::convert<float>(tmp0);
            auto tmp2 = tmp1 * vec_g_exp_last;
            tmp2.store(curr_last_recurrent_state + m * v_head_size + i, v_head_size - i);
          }
        }
        // 2) (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        // k_i: [chunk_size, EK]
        // g[:, :, i, -1, None]: [1]
        // g[:, :, i]: [chunk_size]
        // (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, 1]
        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, EK]
        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2): [EK, chunk_size]
        // v_new: [chunk_size, EV]
        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new: [EK, EV]
        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
        for (int64_t m = 0; m < chunk_size; m++) {
          auto g_exp = std::exp((g_pad_i_last[0] - g_pad_i[m]));
          int64_t i = 0;
          scalar_t g_exp_reduced = static_cast<scalar_t>(g_exp);
          auto vec_g_exp_reduced = bVec(g_exp_reduced);
          for (; i < VecSize * (qk_head_size / VecSize); i += VecSize) {
            auto tmp0 = bVec::loadu(k_i + m * qk_head_size + i);
            auto tmp2 = tmp0 * vec_g_exp_reduced;
            tmp2.store(kg + m * qk_head_size + i);
          }
        }
        // kg.transpose(-1, -2): [EK, chunk_size]
        at::native::utils::transpose<scalar_t>(
            /* M */ chunk_size,
            /* N */ qk_head_size,
            /* src */ kg,
            /* ld_src */ qk_head_size,
            /* dst */ kg_transpose,
            /* ld_dst */ chunk_size);
        // kgv = kg.transpose(-1, -2) @ v_new
        // v_new: [chunk_size, EV]
        at::native::cpublas::brgemm(
            /* M */ qk_head_size,
            /* N */ v_head_size,
            /* K */ chunk_size,
            /* lda */ chunk_size,
            /* ldb */ v_head_size,
            /* ldc */ v_head_size,
            /* add_C */ false,
            /* A */ kg_transpose,
            /* B */ v_prime_pack_reduced,
            /* C */ kgv);
        // last_recurrent_state = 1) + 2)
        for (int64_t m = 0; m < qk_head_size; m++) {
          at::vec::map2<float>(
              [](fVec x, fVec y) { return x + y; },
              curr_last_recurrent_state + m * v_head_size,
              curr_last_recurrent_state + m * v_head_size,
              kgv + m * v_head_size,
              v_head_size);
        }
      }

      // core_attn_out -> output
      // output: [B, T, HV, EV]
      // core_attn_out: [B, HV, padded_T, EV]
      auto curr_out = out_ptr + h * oStrideH;
      for (int64_t m = 0; m < seq_len; m++) {
        at::vec::map<scalar_t>(
            [](fVec x) { return x; }, curr_out + m * oStrideT, curr_core_attn_out + m * v_head_size, v_head_size);
      }

      // Move to the next query
      data_index_step(b, batch_size, h, v_num_head);
    }
  });
}

inline float softplus(float x, double threshold = 20.0) {
  if (x > threshold)
    return x;
  else if (x < -threshold)
    return std::exp(x);
  else
    return std::log1p(std::exp(x));
}

inline at::vec::Vectorized<float> softplus(const at::vec::Vectorized<float>& x, double threshold = 20.0) {
  using Vec = at::vec::Vectorized<float>;
  Vec mask_hi = x > Vec(threshold);
  Vec mask_lo = x < Vec(-threshold);

  Vec expx = x.exp_u20();
  Vec log1pex = (expx + Vec(1.0f)).log();

  return Vec::blendv(Vec::blendv(log1pex, expx, mask_lo), x, mask_hi);
}

template <typename scalar_t, typename param_t>
void fused_sigmoid_gating_delta_rule_update_kernel_impl(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const param_t* __restrict__ A_log_ptr,
    const scalar_t* __restrict__ a_ptr,
    const scalar_t* __restrict__ dt_bias_ptr,
    const scalar_t* __restrict__ b_ptr,
    const int32_t* __restrict__ indices_ptr,
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,
    float* __restrict__ qk_scale_buf,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t k_strideS,
    int64_t k_strideH,
    int64_t v_strideB,
    int64_t v_strideS,
    int64_t v_strideH,
    bool use_qk_l2norm_in_kernel,
    double softplus_threshold) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt(head_dim);
  fVec scale_vec = fVec(scale);
  if (use_qk_l2norm_in_kernel) {
    float eps = 1e-5;
    at::parallel_for(0, batch_size * seq_len * num_heads, 0, [&](int64_t begin, int64_t end) {
      int64_t bi{0}, si{0}, ni{0};
      data_index_init(begin, bi, batch_size, si, seq_len, ni, num_heads);
      for (int64_t i = begin; i < end; ++i) {
        float sum_q = float(0);
        float sum_k = float(0);
        fVec sum_q_fvec = fVec(float(0));
        fVec sum_k_fvec = fVec(float(0));
        int64_t q_offset = bi * q_strideB + si * q_strideS + ni * q_strideH;
        int64_t k_offset = bi * k_strideB + si * k_strideS + ni * k_strideH;
        int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + ni;
        int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
        int64_t d;
#pragma GCC unroll 4
        for (d = 0; d <= head_dim - VecSize; d += VecSize) {
          bVec q_bvec = bVec::loadu(q_ptr + q_offset + d);
          fVec q_fvec0, q_fvec1;
          std::tie(q_fvec0, q_fvec1) = at::vec::convert_to_float(q_bvec);
          sum_q_fvec += q_fvec0 * q_fvec0;
          sum_q_fvec += q_fvec1 * q_fvec1;
          bVec k_bvec = bVec::loadu(k_ptr + k_offset + d);
          fVec k_fvec0, k_fvec1;
          std::tie(k_fvec0, k_fvec1) = at::vec::convert_to_float(k_bvec);
          sum_k_fvec += k_fvec0 * k_fvec0;
          sum_k_fvec += k_fvec1 * k_fvec1;
        }
#pragma GCC unroll 4
        for (; d < head_dim; ++d) {
          float q_val = static_cast<float>(q_ptr[q_offset + d]);
          sum_q += q_val * q_val;
          float k_val = static_cast<float>(k_ptr[k_offset + d]);
          sum_k += k_val * k_val;
        }

        sum_q += vec_reduce_sum(sum_q_fvec);
        sum_k += vec_reduce_sum(sum_k_fvec);
        qk_scale_buf[q_scale_offset] = float(1) / std::sqrt(sum_q + eps);
        qk_scale_buf[k_scale_offset] = float(1) / std::sqrt(sum_k + eps);

        data_index_step(bi, batch_size, si, seq_len, ni, num_heads);
      }
    });
  }
  at::parallel_for(0, batch_size * seq_len * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, si{0}, ni{0};
    data_index_init(begin, bi, batch_size, si, seq_len, ni, v_num_heads);
    for (int64_t i = begin; i < end; ++i) {
      int64_t cache_index = indices_ptr[bi];
      int64_t state_offset = (cache_index * v_num_heads + ni) * head_dim * v_head_dim;
      float g_val = -std::exp(float(A_log_ptr[ni])) *
                    softplus(float(a_ptr[bi * v_num_heads + ni]) + float(dt_bias_ptr[ni]), softplus_threshold);
      float g_val_exp = std::exp(g_val);
      fVec g_val_exp_vec = fVec(g_val_exp);
      int64_t q_offset = si * q_strideS + bi * q_strideB + (ni / group_size) * q_strideH;
      int64_t k_offset = si * k_strideS + bi * k_strideB + (ni / group_size) * k_strideH;
      int64_t q_scale_offset = bi * seq_len * num_heads + si * num_heads + (ni / group_size);
      int64_t k_scale_offset = q_scale_offset + batch_size * seq_len * num_heads;
      float q_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[q_scale_offset] : 1.0f;
      float k_scale = use_qk_l2norm_in_kernel ? qk_scale_buf[k_scale_offset] : 1.0f;
      int64_t v_offset = si * v_strideS + bi * v_strideB + ni * v_strideH;
      int64_t o_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
      float beta_val = 1 / (1 + std::exp(-b_ptr[bi * v_num_heads + ni]));
      fVec beta_vec = fVec(beta_val);
      int64_t dvi = 0;
      for (; dvi <= v_head_dim - VecSize; dvi += VecSize) {
        fVec kv_mem_vec0 = fVec(float(0));
        fVec kv_mem_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec k_val_vec = fVec(k_ptr[k_offset + di] * k_scale);
          fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
          fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          kv_mem_vec0 = kv_mem_vec0 + state_vec0 * g_val_exp_vec * k_val_vec;
          kv_mem_vec1 = kv_mem_vec1 + state_vec1 * g_val_exp_vec * k_val_vec;
        }
        bVec v_bvec = bVec::loadu(v_ptr + v_offset + dvi);
        fVec v_vec0, v_vec1;
        std::tie(v_vec0, v_vec1) = at::vec::convert_to_float(v_bvec);
        fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
        fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
        fVec o_vec0 = fVec(float(0));
        fVec o_vec1 = fVec(float(0));
        for (int di = 0; di < head_dim; ++di) {
          fVec q_vec = fVec(q_ptr[q_offset + di] * q_scale);
          fVec k_vec = fVec(k_ptr[k_offset + di] * k_scale);
          fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
          fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          state_vec0 = state_vec0 * g_val_exp_vec + k_vec * dt_vec0;
          state_vec1 = state_vec1 * g_val_exp_vec + k_vec * dt_vec1;
          o_vec0 = o_vec0 + state_vec0 * q_vec * scale_vec;
          o_vec1 = o_vec1 + state_vec1 * q_vec * scale_vec;
          state_vec0.store(state_ptr + state_offset + di * v_head_dim + dvi);
          state_vec1.store(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
        }
        bVec o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
        o_vec.store(o_ptr + o_offset + dvi);
      }
      for (; dvi < v_head_dim; ++dvi) {
        float kv_mem_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] *= g_val_exp;
          kv_mem_val += state_ptr[state_offset + di * v_head_dim + dvi] * k_val;
        }
        float v_val = v_ptr[v_offset + dvi];
        float dt_val = (v_val - kv_mem_val) * beta_val;
        float o_val = 0;
        for (int di = 0; di < head_dim; ++di) {
          float q_val = q_ptr[q_offset + di] * q_scale;
          float k_val = k_ptr[k_offset + di] * k_scale;
          state_ptr[state_offset + di * v_head_dim + dvi] += k_val * dt_val;
          o_val += state_ptr[state_offset + di * v_head_dim + dvi] * q_val * scale;
        }
        o_ptr[o_offset + dvi] = o_val;
      }
      data_index_step(bi, batch_size, si, seq_len, ni, v_num_heads);
    }
  });
}

template <typename scalar_t>
void fused_gdn_gating_kernel_impl(
    float* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    float* __restrict__ out,
    scalar_t* __restrict__ beta,
    int64_t batch,
    int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  const fVec neg_one(-1.0f);
  const fVec one(1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t j = 0;
      for (; j < num_heads - (num_heads % vec_size); j += vec_size) {
        fVec A_log_vec0 = fVec::loadu(A_log + j);
        fVec A_log_vec1 = fVec::loadu(A_log + j + fvec_size);
        bVec dt_bias_vec = bVec::loadu(dt_bias + j);
        bVec a_bvec = bVec::loadu(a + i * num_heads + j);
        bVec b_bvec = bVec::loadu(b + i * num_heads + j);
        fVec a0, a1, dt_bias_vec0, dt_bias_vec1, b0, b1;
        std::tie(a0, a1) = at::vec::convert_to_float(a_bvec);
        std::tie(b0, b1) = at::vec::convert_to_float(b_bvec);
        std::tie(dt_bias_vec0, dt_bias_vec1) = at::vec::convert_to_float(dt_bias_vec);

        fVec g0 = neg_one * A_log_vec0.exp_u20() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp_u20() * softplus(a1 + dt_bias_vec1);
        fVec beta0 = one / (one + (neg_one * b0).exp_u20());
        fVec beta1 = one / (one + (neg_one * b1).exp_u20());

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
        bVec beta_vec = at::vec::convert_from_float<scalar_t>(beta0, beta1);
        beta_vec.store(beta + i * num_heads + j);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(A_log[j]) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
        beta[i * num_heads + j] = 1 / (1 + std::exp(-b[i * num_heads + j]));
      }
    }
  });
}

template <typename scalar_t>
void fused_gdn_gating_kernel_impl(
    scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ dt_bias,
    float* __restrict__ out,
    scalar_t* __restrict__ beta,
    int64_t batch,
    int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  const fVec neg_one(-1.0f);
  const fVec one(1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      int64_t j = 0;
      for (; j < num_heads - (num_heads % vec_size); j += vec_size) {
        bVec A_log_bvec = bVec::loadu(A_log + j);
        fVec A_log_vec0, A_log_vec1;
        std::tie(A_log_vec0, A_log_vec1) = at::vec::convert_to_float(A_log_bvec);
        bVec dt_bias_vec = bVec::loadu(dt_bias + j);
        bVec a_bvec = bVec::loadu(a + i * num_heads + j);
        bVec b_bvec = bVec::loadu(b + i * num_heads + j);
        fVec a0, a1, dt_bias_vec0, dt_bias_vec1, b0, b1;
        std::tie(a0, a1) = at::vec::convert_to_float(a_bvec);
        std::tie(b0, b1) = at::vec::convert_to_float(b_bvec);
        std::tie(dt_bias_vec0, dt_bias_vec1) = at::vec::convert_to_float(dt_bias_vec);

        fVec g0 = neg_one * A_log_vec0.exp_u20() * softplus(a0 + dt_bias_vec0);
        fVec g1 = neg_one * A_log_vec1.exp_u20() * softplus(a1 + dt_bias_vec1);
        fVec beta0 = one / (one + (neg_one * b0).exp_u20());
        fVec beta1 = one / (one + (neg_one * b1).exp_u20());

        g0.store(out + i * num_heads + j);
        g1.store(out + i * num_heads + j + fvec_size);
        bVec beta_vec = at::vec::convert_from_float<scalar_t>(beta0, beta1);
        beta_vec.store(beta + i * num_heads + j);
      }
      for (; j < num_heads; ++j) {
        out[i * num_heads + j] = -std::exp(float(A_log[j])) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
        beta[i * num_heads + j] = 1 / (1 + std::exp(-b[i * num_heads + j]));
      }
    }
  });
}

}  // anonymous namespace

// TODO: remove this one
template <bool is_last_dim_contiguous>
inline void
CHECK_INPUT_SHAPE_DTYPE(const at::Tensor& tensor, const int64_t& dim, const at::IntArrayRef& sizes, at::ScalarType st) {
  TORCH_CHECK(tensor.sizes() == sizes, "Input tensor shape mismatch: expected ", sizes, ", got ", tensor.sizes());
  TORCH_CHECK(tensor.dtype() == st, "Input tensor dtype mismatch");
  CHECK_DIM(dim, tensor);
  if (is_last_dim_contiguous) {
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor);
  } else {
    CHECK_CONTIGUOUS(tensor);
  }
}

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor> prepare_chunk_indices(const at::Tensor& cu_seqlens) {
  int64_t num_seqs = cu_seqlens.size(0) - 1;
  at::Tensor chunk_offsets = at::empty({num_seqs + 1}, cu_seqlens.options());
  // get number of chunks and chunk offsets
  const int32_t* offsets_data = cu_seqlens.data_ptr<int32_t>();
  int32_t num_chunks = 0;
  chunk_offsets[0] = 0;
  for (int64_t row = 0; row < num_seqs; ++row) {
    num_chunks += div_up(offsets_data[row + 1] - offsets_data[row], CHUNK_SIZE);
    chunk_offsets[row + 1] = num_chunks;
  }
  // get chunk indices
  at::Tensor chunk_indices = at::empty({num_chunks, 2}, cu_seqlens.options());
  int32_t* indices_data = chunk_indices.data_ptr<int32_t>();

  int64_t idx = 0;
  for (int32_t row = 0; row < num_seqs; ++row) {
    int32_t num_chunks = div_up(offsets_data[row + 1] - offsets_data[row], CHUNK_SIZE);

    for (int32_t col = 0; col < num_chunks; ++col) {
      indices_data[idx * 2 + 0] = row;
      indices_data[idx * 2 + 1] = col;
      idx++;
    }
  }
  return std::make_tuple(chunk_indices, chunk_offsets);
}

#define LAUNCH_L2NORM_KERNEL(HD)        \
  l2norm_fwd_kernel_impl<scalar_t, HD>( \
      query_norm.data_ptr<scalar_t>(),  \
      key_norm.data_ptr<scalar_t>(),    \
      query.data_ptr<scalar_t>(),       \
      key.data_ptr<scalar_t>(),         \
      eps,                              \
      T,                                \
      H,                                \
      query.stride(1),                  \
      query.stride(2),                  \
      key.stride(1),                    \
      key.stride(2));

std::tuple<at::Tensor, at::Tensor> l2norm_fwd(const at::Tensor& query, const at::Tensor& key, double eps) {
  int64_t B = query.size(0);
  int64_t T = query.size(1);
  int64_t H = query.size(2);
  int64_t D = query.size(3);

  at::Tensor query_norm = at::empty_like(query);
  at::Tensor key_norm = at::empty_like(key);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "l2norm_fwd", [&] {
    switch (D) {
      case 64:
        LAUNCH_L2NORM_KERNEL(64);
        break;
      case 128:
        LAUNCH_L2NORM_KERNEL(128);
        break;
      default:
        TORCH_CHECK(false, "Unexpected head dim size, ", D);
    }
  });

  return std::make_tuple(query_norm, key_norm);
}

// TODO: debug code remove me!!!
at::Tensor l2norm(const at::Tensor& x, double eps, bool has_scale) {
  auto xf = x.to(at::kFloat);
  auto inv_norm = at::rsqrt((xf * xf).sum(-1, true) + eps);
  auto out = xf * inv_norm;
  if (has_scale) {
    out.mul_(1.0 / std::sqrt(static_cast<double>(x.size(-1))));
  }
  return out.to(x.scalar_type());
}

// [NB]: instantiate decay_mask to avoid heavy recomputation in the kernel with exp
template <int CHUNK_SIZE>
at::Tensor chunk_local_cumsum(const at::Tensor& g, const at::Tensor& cu_seqlens, const at::Tensor& chunk_indices) {
  int64_t B = g.size(0);
  // int64_t T = g.size(1);
  int64_t Hv = g.size(2);
  int64_t NT = chunk_indices.size(0);

  std::cout << "### chunk_local_cumsum" << std::endl;

  at::Tensor g_ = at::empty({B, NT, Hv, CHUNK_SIZE}, g.options());
  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "chunk_local_cumsum", [&] {
    chunk_local_cumsum_kernel_impl<scalar_t, CHUNK_SIZE>(
        g_.data_ptr<scalar_t>(),
        g.data_ptr<scalar_t>(),
        cu_seqlens.data_ptr<int32_t>(),
        chunk_indices.data_ptr<int32_t>(),
        Hv,
        NT);
  });
  return g_;
}

// TODO: debug code remove me!!!
// Reference for chunk_local_cumsum on a single sequence.
// Matches test_mamba.py: F.pad(g, (0, pad_size)) + reshape + g.cumsum(dim=-1)
// on head-first g with shape [B, Hv, T], returning [B, Hv, NT, chunk_size].
at::Tensor cumsum(const at::Tensor& g, int chunk_size) {
  TORCH_CHECK(g.dim() == 3, "cumsum: expect g with shape [B, T, Hv]");
  TORCH_CHECK(chunk_size > 0, "cumsum: chunk_size must be positive");

  auto g_hf = g.transpose(1, 2).contiguous();
  const int64_t B = g_hf.size(0);
  const int64_t Hv = g_hf.size(1);
  const int64_t seq_len = g_hf.size(2);
  const int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
  if (pad_size > 0) {
    auto padding = at::zeros({B, Hv, pad_size}, g.options());
    g_hf = at::cat({g_hf, padding}, 2);
  }
  const int64_t NT = g_hf.size(2) / chunk_size;
  return g_hf.view({B, Hv, NT, chunk_size}).cumsum(-1);
}

// Varlen packed batch reference: run single-sequence cumsum per cu_seqlens entry
// and concatenate along the global chunk dimension.
at::Tensor cumsum(const at::Tensor& g, const at::Tensor& cu_seqlens, int chunk_size) {
  TORCH_CHECK(g.dim() == 3, "cumsum: expect g with shape [B, T, Hv]");
  TORCH_CHECK(g.size(0) == 1, "cumsum: varlen expects batch size 1");
  TORCH_CHECK(cu_seqlens.dim() == 1, "cumsum: expect cu_seqlens to be 1D");

  const int64_t num_seqs = cu_seqlens.size(0) - 1;
  std::vector<at::Tensor> outputs;
  outputs.reserve(num_seqs);
  for (int64_t i = 0; i < num_seqs; ++i) {
    const int64_t start = cu_seqlens[i].item<int32_t>();
    const int64_t end = cu_seqlens[i + 1].item<int32_t>();
    outputs.push_back(cumsum(g.narrow(1, start, end - start), chunk_size));
  }
  return at::cat(outputs, 2).transpose_(1, 2);
}

// TODO: debug code remove me!!!
// Reference for decay_mask on chunked local cumsum g.
// Matches test_mamba.py: ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
// on g with shape [B, NT, Hv, chunk_size], returning [B, NT, Hv, chunk_size, chunk_size].
at::Tensor decay_mask_ref(const at::Tensor& g, int chunk_size) {
  TORCH_CHECK(g.dim() == 4, "decay_mask: expect g with shape [B, NT, Hv, chunk_size]");
  TORCH_CHECK(chunk_size > 0, "decay_mask: chunk_size must be positive");
  TORCH_CHECK(g.size(-1) == chunk_size, "decay_mask: g last dim must equal chunk_size");

  auto gf = g.to(at::kFloat);
  return (gf.unsqueeze(-1) - gf.unsqueeze(-2)).tril().exp().tril();
}

// TODO: debug code remove me!!!
// Reference for recompute_w_u (intra), mirrors torch_chunk_gated_delta_rule:
//   attn = -((k*beta) @ k^T * decay_mask).tril(-1); forward-subst; attn += eye
//   w = attn @ (k * beta * exp(g))
//   u = attn @ (v * beta)
// key_:[B,T,H,D] value:[B,T,Hv,Dv] g_:[B,NT,Hv,C] beta:[B,T,Hv]
// returns w_ref:[B,T,Hv,D], u_ref:[B,T,Hv,Dv] (float)
std::tuple<at::Tensor, at::Tensor> recompute_w_u_ref(
    const at::Tensor& key_,
    const at::Tensor& value,
    const at::Tensor& g_,
    const at::Tensor& beta,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_indices,
    int chunk_size) {
  const int64_t T = key_.size(1);
  const int64_t H = key_.size(2);
  const int64_t D = key_.size(3);
  const int64_t Hv = value.size(2);
  const int64_t Dv = value.size(3);
  const int64_t NT = chunk_indices.size(0);
  const int64_t HG = Hv / H;

  auto key_f = key_.to(at::kFloat)[0];                            // [T, H, D]
  auto value_f = value.to(at::kFloat)[0];                         // [T, Hv, Dv]
  auto beta_f = beta.to(at::kFloat)[0];                           // [T, Hv]
  auto g_f = g_.to(at::kFloat)[0];                                // [NT, Hv, C]
  auto decay = decay_mask_ref(g_, chunk_size).to(at::kFloat)[0];  // [NT, Hv, C, C]

  auto w_ref = at::zeros({1, T, Hv, D}, at::kFloat);
  auto u_ref = at::zeros({1, T, Hv, Dv}, at::kFloat);
  auto w0 = w_ref[0];  // [T, Hv, D]
  auto u0 = u_ref[0];  // [T, Hv, Dv]

  auto ci = chunk_indices.to(at::kCPU).contiguous();
  auto cs = cu_seqlens.to(at::kCPU).contiguous();
  const int32_t* ci_p = ci.data_ptr<int32_t>();
  const int32_t* cs_p = cs.data_ptr<int32_t>();

  for (int64_t c = 0; c < NT; ++c) {
    int32_t bs = ci_p[c * 2 + 0];
    int32_t cpos = ci_p[c * 2 + 1];
    int32_t bos = cs_p[bs];
    int32_t seqlen = cs_p[bs + 1] - cs_p[bs];
    int64_t mb_start = static_cast<int64_t>(cpos) * chunk_size;
    int64_t mb_size = std::min<int64_t>(seqlen - mb_start, chunk_size);
    int64_t tok = bos + mb_start;

    for (int64_t hv = 0; hv < Hv; ++hv) {
      int64_t h = hv / HG;
      auto K = key_f.narrow(0, tok, mb_size).select(1, h);                                          // [mb, D]
      auto V = value_f.narrow(0, tok, mb_size).select(1, hv);                                       // [mb, Dv]
      auto beta_c = beta_f.narrow(0, tok, mb_size).select(1, hv).unsqueeze(-1);                     // [mb, 1]
      auto g_c = g_f.select(0, c).select(0, hv).narrow(0, 0, mb_size);                              // [mb]
      auto decay_c = decay.select(0, c).select(0, hv).narrow(0, 0, mb_size).narrow(1, 0, mb_size);  // [mb, mb]

      // attn = -((k*beta) @ k^T) * decay_mask, strict-lower
      auto kkt = at::matmul(K, K.transpose(0, 1));       // [mb, mb]
      auto attn = (-(beta_c * kkt) * decay_c).tril(-1);  // [mb, mb]
      // forward substitution: attn[i, :i] += attn[i, :i] @ attn[:i, :i]
      for (int64_t i = 1; i < mb_size; ++i) {
        auto row = attn.select(0, i).narrow(0, 0, i);                       // [i]
        auto sub = attn.narrow(0, 0, i).narrow(1, 0, i);                    // [i, i]
        auto updated = row + at::matmul(row.unsqueeze(0), sub).squeeze(0);  // [i]
        attn.select(0, i).narrow(0, 0, i).copy_(updated);
      }
      attn = attn + at::eye(mb_size, attn.options());

      auto u_c = at::matmul(attn, V * beta_c);                            // [mb, Dv]
      auto w_c = at::matmul(attn, K * beta_c * g_c.exp().unsqueeze(-1));  // [mb, D]
      u0.narrow(0, tok, mb_size).select(1, hv).copy_(u_c);
      w0.narrow(0, tok, mb_size).select(1, hv).copy_(w_c);
    }
  }
  return std::make_tuple(w_ref, u_ref);
}

#define LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTRA_KERNEL(HD)                \
  chunk_gated_delta_rule_fwd_intra_kernel_impl<scalar_t, HD, CHUNK_SIZE>( \
      w.data_ptr<scalar_t>(),                                             \
      u.data_ptr<scalar_t>(),                                             \
      decay_mask.data_ptr<float>(),                                       \
      k.data_ptr<scalar_t>(),                                             \
      v.data_ptr<scalar_t>(),                                             \
      g.data_ptr<float>(),                                                \
      beta.data_ptr<scalar_t>(),                                          \
      cu_seqlens.data_ptr<int32_t>(),                                     \
      chunk_indices.data_ptr<int32_t>(),                                  \
      H,                                                                  \
      Hv,                                                                 \
      NT,                                                                 \
      k.stride(1),                                                        \
      k.stride(2),                                                        \
      v.stride(1),                                                        \
      v.stride(2));

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_intra(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t H = k.size(2);
  int64_t D = k.size(3);
  int64_t Hv = v.size(2);
  int64_t Dv = v.size(3);
  int64_t NT = chunk_indices.size(0);

  // TODO: now only optimized for qwen3.5 D == Dv
  CHECK_EQ(D, Dv);

  std::cout << "\n### chunk_gated_delta_rule_fwd_intra ..." << std::endl;

  at::Tensor w = at::empty({B, T, Hv, D}, k.options());                                 // BFloat16
  at::Tensor u = at::empty({B, T, Hv, Dv}, k.options());                                // BFloat16
  at::Tensor decay_mask = at::empty({B, NT, Hv, CHUNK_SIZE, CHUNK_SIZE}, g.options());  // Float
  AT_DISPATCH_REDUCED_FLOATING_TYPES(k.scalar_type(), "chunk_gated_delta_rule_fwd_intra", [&] {
    switch (D) {
      case 64:
        LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTRA_KERNEL(64);
        break;
      case 128:
        LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTRA_KERNEL(128);
        break;
      default:
        TORCH_CHECK(false, "Unexpected head dim size, ", D);
    }
  });

  return std::make_tuple(w, u, decay_mask);
}

#define LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTER_KERNEL(HD)                \
  chunk_gated_delta_rule_fwd_inter_kernel_impl<scalar_t, HD, CHUNK_SIZE>( \
      o.data_ptr<scalar_t>(),                                             \
      initial_state.data_ptr<float>(),                                    \
      q.data_ptr<scalar_t>(),                                             \
      k.data_ptr<scalar_t>(),                                             \
      w.data_ptr<scalar_t>(),                                             \
      u.data_ptr<scalar_t>(),                                             \
      g.data_ptr<float>(),                                                \
      decay_mask.data_ptr<float>(),                                       \
      cu_seqlens.data_ptr<int32_t>(),                                     \
      chunk_offsets.data_ptr<int32_t>(),                                  \
      H,                                                                  \
      Hv,                                                                 \
      num_seqs,                                                           \
      q.stride(1),                                                        \
      q.stride(2),                                                        \
      k.stride(1),                                                        \
      k.stride(2));

template <int CHUNK_SIZE>
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_inter(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& w,
    const at::Tensor& u,
    const at::Tensor& g,
    const at::Tensor& decay_mask,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_offsets) {
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t Hv = w.size(2);
  const int64_t Dv = u.size(3);
  const int64_t num_seqs = initial_state.size(0);

  std::cout << "### chunk_gated_delta_rule_fwd_inter ..." << std::endl;

  at::Tensor o = at::empty({B, T, Hv, Dv}, q.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "chunk_gated_delta_rule_fwd_inter", [&] {
    switch (D) {
      case 64:
        LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTER_KERNEL(64);
        break;
      case 128:
        LAUNCH_CHUNK_GATED_DELTA_RULE_FWD_INTER_KERNEL(128);
        break;
    }
  });

  return std::make_tuple(o, initial_state);
}

// [NB]: Support only varlen inputs
//   B: packed batch dim of q/k/v (== 1)
//   num_seqs: number of variable-length sequences
//
//   query: [B, T, H, D]
//   key: [B, T, H, D]
//   value: [B, T, Hv, Dv]
//   g: [B, T, Hv] FP32
//   beta: [B, T, Hv]
//   initial_state: [num_seqs, Hv, D, Dv] FP32
//   cu_seqlens: [num_seqs + 1] INT32
//
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    bool head_first,
    bool use_qk_l2norm_in_kernel,
    double eps = 1e-6) {
  TORCH_CHECK(!head_first, "chunk_gated_delta_rule_cpu: does not support head first");

  int64_t B = query.size(0);
  int64_t T = query.size(1);
  int64_t H = query.size(2);
  int64_t D = query.size(3);
  int64_t Hv = value.size(2);
  int64_t Dv = value.size(3);
  int64_t num_seqs = initial_state.size(0);

  TORCH_CHECK(B == 1, __func__, ": expect batch size to be 1");
  TORCH_CHECK(Hv % H == 0, __func__, ": expect num_heads_kv multiple of num_heads.");
  TORCH_CHECK(D % 32 == 0, __func__, ": expect head_dim to be multiples of 32.");
  TORCH_CHECK(Dv % 32 == 0, __func__, ": expect head_dim_v to be multiples of 32.");
  CHECK_INPUT_SHAPE_DTYPE<true>(query, {B, T, H, D}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(key, {B, T, H, D}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(value, {B, T, Hv, Dv}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(g, {B, T, Hv}, at::kFloat);
  CHECK_INPUT_SHAPE_DTYPE<false>(beta, {B, T, Hv}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(cu_seqlens, {num_seqs + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<false>(initial_state, {num_seqs, Hv, D, Dv}, at::kFloat);

  std::cout << "### chunk_gated_delta_rule_cpu ..." << std::endl;

  constexpr int CHUNK_SIZE = 64;

  // prepare chunk indices
  auto [chunk_indices, chunk_offsets] = prepare_chunk_indices<CHUNK_SIZE>(cu_seqlens);
  std::cout << "### cu_seqlens" << cu_seqlens << std::endl;
  std::cout << "### chunk_indices" << chunk_indices << std::endl;

  float scale = 1.0 / std::sqrt(D);
  auto [query_, key_] = use_qk_l2norm_in_kernel ? l2norm_fwd(query, key, eps) : std::make_tuple(query.mul(scale), key);

  bool debug = true;
  if (debug && use_qk_l2norm_in_kernel) {
    auto q_ref = l2norm(query, eps, true);
    auto k_ref = l2norm(key, eps, false);
    std::cout << "### check l2norm q max_err: "
              << (query_.to(at::kFloat) - q_ref.to(at::kFloat)).abs().max().item<float>() << std::endl;
    std::cout << "### check l2norm k max_err: "
              << (key_.to(at::kFloat) - k_ref.to(at::kFloat)).abs().max().item<float>() << std::endl;
  }

  auto g_ = chunk_local_cumsum<CHUNK_SIZE>(g, cu_seqlens, chunk_indices);

  if (debug) {
    auto g_ref = cumsum(g, cu_seqlens, CHUNK_SIZE);
    std::cout << "### check cumsum g max_err: " << (g_.to(at::kFloat) - g_ref.to(at::kFloat)).abs().max().item<float>()
              << std::endl;
    // std::cout << "### g_ref[0][0]: " << g_ref[0][0].view({-1, CHUNK_SIZE}) << std::endl;
    // std::cout << "### g_[0][0]: " << g_[0][0].view({-1, CHUNK_SIZE}) << std::endl;
  }

  // fused kkt + solve_tril + recompute_w_u
  auto [w, u, decay_mask] =
      chunk_gated_delta_rule_fwd_intra<CHUNK_SIZE>(key_, value, g_, beta, cu_seqlens, chunk_indices);

  if (debug) {
    auto decay_mask2 = decay_mask_ref(g_, CHUNK_SIZE);
    auto decay_mask_f = decay_mask.to(at::kFloat);
    auto decay_mask2_f = decay_mask2.to(at::kFloat);
    // fexp_u20 vs torch.exp: use relaxed rtol/atol (same spirit as torch.allclose)
    constexpr double rtol = 1e-3;
    constexpr double atol = 1e-2;
    const bool decay_mask_ok = at::allclose(decay_mask_f, decay_mask2_f, rtol, atol);
    std::cout << "### check decay_mask allclose: " << (decay_mask_ok ? "true" : "false") << " (rtol=" << rtol
              << ", atol=" << atol << ")" << std::endl;
  }

  if (debug) {
    auto [w_ref, u_ref] = recompute_w_u_ref(key_, value, g_, beta, cu_seqlens, chunk_indices, CHUNK_SIZE);
    auto w_f = w.to(at::kFloat);
    auto u_f = u.to(at::kFloat);
    // w/u are stored in bf16; vs fp32 reference, use bf16-appropriate tolerance.
    // NOTE: exp(cumsum(g)) has huge dynamic range when g has positive values,
    // which destroys bf16 precision; realistic GDN gates are <= 0 (bounded).
    constexpr double rtol = 1e-2;
    constexpr double atol = 1e-2;
    const bool w_ok = at::allclose(w_f, w_ref, rtol, atol);
    const bool u_ok = at::allclose(u_f, u_ref, rtol, atol);
    std::cout << "### check w allclose: " << (w_ok ? "true" : "false")
              << " max_err: " << (w_f - w_ref).abs().max().item<float>() << " (rtol=" << rtol << ", atol=" << atol
              << ")" << std::endl;
    std::cout << "### check u allclose: " << (u_ok ? "true" : "false")
              << " max_err: " << (u_f - u_ref).abs().max().item<float>() << " (rtol=" << rtol << ", atol=" << atol
              << ")" << std::endl;
  }

  // fused `chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o`
  auto [output, final_state] = chunk_gated_delta_rule_fwd_inter<CHUNK_SIZE>(
      query_, key_, w, u, g_, decay_mask, initial_state, output_final_state, cu_seqlens, chunk_offsets);

  return std::make_tuple(output, final_state);
}

// query: [B, T, HK, EK]
// key: [B, T, HK, EK]
// value: [B, T, HV, EV]
// g: [B, T, HV] FP32
// beta: [B, T, HV]
// initial_state: [N, HV, EK, EV] FP32
// output_final_state: bool
// cu_seqlens: [N + 1] INT32
// head_first: bool
// use_qk_l2norm_in_kernel: bool
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu1(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    bool head_first,
    bool use_qk_l2norm_in_kernel,
    double eps = 1e-5) {
  TORCH_CHECK(head_first == false, "chunk_gated_delta_rule_cpu does not support head first");
  int64_t B = query.size(0);
  int64_t global_seq_len = query.size(1);
  int64_t qk_num_head = query.size(2);
  int64_t qk_head_size = query.size(3);
  int64_t v_num_head = value.size(2);
  int64_t v_head_size = value.size(3);
  int64_t batch_size = initial_state.size(0);
  CHECK_EQ(B, 1);
  TORCH_CHECK(v_num_head % qk_num_head == 0, "expect v_num_head multiple of qk_num_head.");
  TORCH_CHECK(qk_head_size % 32 == 0, "expect qk_head_size to be multiples of 32.");
  TORCH_CHECK(v_head_size % 32 == 0, "expect v_head_size to be multiples of 32.");
  CHECK_INPUT_SHAPE_DTYPE<true>(query, 4, {B, global_seq_len, qk_num_head, qk_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(key, 4, {B, global_seq_len, qk_num_head, qk_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<true>(value, 4, {B, global_seq_len, v_num_head, v_head_size}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(g, 3, {B, global_seq_len, v_num_head}, at::kFloat);
  CHECK_INPUT_SHAPE_DTYPE<false>(beta, 3, {B, global_seq_len, v_num_head}, at::kBFloat16);
  CHECK_INPUT_SHAPE_DTYPE<false>(cu_seqlens, 1, {batch_size + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<false>(initial_state, 4, {batch_size, v_num_head, qk_head_size, v_head_size}, at::kFloat);

  at::Tensor output = at::empty_like(value, value.options());  // [B, T, HV, EV]
  at::Tensor final_state = initial_state.to(at::kFloat);       // [N, HV, EK, EV]

  // Strides
  int64_t qStrideH = query.stride(2);
  int64_t qStrideT = query.stride(1);
  int64_t kStrideH = key.stride(2);
  int64_t kStrideT = key.stride(1);
  int64_t vStrideH = value.stride(2);
  int64_t vStrideT = value.stride(1);
  int64_t oStrideH = output.stride(2);
  int64_t oStrideT = output.stride(1);

  constexpr int64_t chunk_size = 64;
  // Deduce the global chunks
  // e.g. cu_seqlens: [0, 5, 13, 16], chunk_size = 4
  // chunk_offsets: [0, 2, 4, 5]
  // chunk_indices (batch_id, local_chunk_id): [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
  at::Tensor chunk_offsets = at::empty(batch_size + 1, cu_seqlens.options());
  auto chunk_offsets_ptr = chunk_offsets.data_ptr<int32_t>();
  chunk_offsets_ptr[0] = 0;
  int32_t* cu_seqlens_ptr = cu_seqlens.data_ptr<int32_t>();
  int64_t s = 0;
  int64_t e = 0;
  int64_t s_pad = 0;
  int64_t e_pad = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    e = cu_seqlens_ptr[b + 1];
    int64_t seq_len = e - s;
    int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
    int64_t total_seq_length = seq_len + pad_size;
    e_pad = s_pad + total_seq_length;
    chunk_offsets[b + 1] = e_pad / chunk_size;
    s = e;
    s_pad = e_pad;
  }
  int64_t global_total_seq_length = e_pad;
  int64_t global_num_chunk = chunk_offsets_ptr[batch_size];
  at::Tensor chunk_indices = at::empty(global_num_chunk * 2, cu_seqlens.options());
  auto chunk_indices_ptr = chunk_indices.data_ptr<int32_t>();
  int64_t curr_c = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    int64_t batch_chunk_num = chunk_offsets_ptr[b + 1] - chunk_offsets_ptr[b];
    for (int64_t c = 0; c < batch_chunk_num; c++) {
      chunk_indices_ptr[curr_c * 2] = b;
      chunk_indices_ptr[curr_c * 2 + 1] = c;
      curr_c += 1;
    }
  }

  // Allocate buffer
  int64_t buff_size = v_num_head * global_total_seq_length                               // g_pad_data
                      + batch_size * v_num_head * global_total_seq_length * v_head_size  // core_attn
                      + v_num_head * global_total_seq_length * chunk_size                // decay_mask
                      + v_num_head * global_total_seq_length * v_head_size;              // v_beta_attn
  at::Tensor buff_data = at::empty({buff_size}, query.options().dtype(at::kFloat));
  int64_t reduced_buff_size = qk_num_head * global_total_seq_length * qk_head_size    // q_pad_data
                              + qk_num_head * global_total_seq_length * qk_head_size  // k_pad_data
                              + v_num_head * global_total_seq_length * v_head_size    // v_pad_data
                              + v_num_head * global_total_seq_length * qk_head_size   // k_beta_data
                              + v_num_head * global_total_seq_length * v_head_size    // v_beta_data
                              + v_num_head * global_total_seq_length * qk_head_size   // k_cumdecay_reduced
                              + qk_num_head * global_seq_len                          // q_norm_sum
                              + qk_num_head * global_seq_len;                         // k_norm_sum
  at::Tensor reduced_buff_data = at::empty({reduced_buff_size}, query.options());
  int64_t num_thread = at::get_num_threads();
  int64_t buff_size_16bit_per_thread =
      /* k_transpose */ qk_head_size * chunk_size +
      /* v_pack */ chunk_size * v_head_size +
      /* k_beta_g  */ chunk_size * qk_head_size +
      /* k_beta_g_pack  */ chunk_size * qk_head_size +
      /* attn */ chunk_size * chunk_size * 2 +
      /* attn_reduced */ chunk_size * chunk_size +
      /* k_cumdecay */ chunk_size * qk_head_size * 2 +
      /* row */ chunk_size * 2 +
      /* updated */ chunk_size * 2 +
      /* curr_last_recurrent_state_reduced  */ qk_head_size * v_head_size +
      /* curr_last_recurrent_state_pack_reduced   */ qk_head_size * v_head_size +
      /* k_transpose_i  */ qk_head_size * chunk_size +
      /* attn_i   */ chunk_size * chunk_size * 2 +
      /* attn_i_reduced     */ chunk_size * chunk_size +
      /* v_prime */ chunk_size * v_head_size * 2 +
      /* v_prime_reduced */ chunk_size * v_head_size +
      /* v_prime_pack_reduced */ chunk_size * v_head_size +
      /* qg */ chunk_size * qk_head_size +
      /* attn_inter */ chunk_size * v_head_size * 2 +
      /* kg */ chunk_size * qk_head_size +
      /* kg_transpose */ qk_head_size * chunk_size +
      /* kgv */ qk_head_size * v_head_size * 2;
  at::Tensor thread_buff_data = at::empty({num_thread, buff_size_16bit_per_thread}, query.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "chunk_gated_delta_rule_kernel", [&] {
    chunk_gated_delta_rule_kernel_impl<scalar_t, chunk_size>(
        output.data_ptr<scalar_t>(),
        final_state.data_ptr<float>(),
        query.data_ptr<scalar_t>(),
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        g.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        cu_seqlens_ptr,
        buff_data.data_ptr<float>(),
        reduced_buff_data.data_ptr<scalar_t>(),
        thread_buff_data.data_ptr<scalar_t>(),
        chunk_offsets_ptr,
        chunk_indices_ptr,
        use_qk_l2norm_in_kernel,
        batch_size,
        global_seq_len,
        qk_num_head,
        v_num_head,
        qk_head_size,
        v_head_size,
        qStrideH,
        qStrideT,
        kStrideH,
        kStrideT,
        vStrideH,
        vStrideT,
        oStrideH,
        oStrideT,
        global_total_seq_length,
        global_num_chunk,
        buff_size_16bit_per_thread,
        eps);
  });
  return std::make_tuple(std::move(output), std::move(final_state));
}

// A_log: [v_num_heads]
// dt_bias: [v_num_heads]
// query: [seq_len, batch_size, num_heads, head_dim]
// key: [seq_len, batch_size, num_heads, head_dim]
// value: [seq_len, batch_size, v_num_heads, v_head_dim]
// a: [batch_size, v_num_heads]
// b: [batch_size, v_num_heads]
// initial_state_source:[num_tokens, v_num_heads, head_dim, v_head_dim]
// initial_state_indices: [batch_size]
// cu_seqlens: [batch_size + 1]
at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices,
    const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel,
    double softplus_beta = 1.0,
    double softplus_threshold = 20.0) {
  CHECK_DIM(4, q);
  CHECK_DIM(4, v);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  int64_t seq_len = q.size(0);
  int64_t batch_size = q.size(1);
  int64_t num_heads = q.size(2);
  int64_t head_dim = q.size(3);
  int64_t v_num_heads = v.size(2);
  int64_t v_head_dim = v.size(3);
  CHECK_INPUT_SHAPE_DTYPE<true>(k, {seq_len, batch_size, num_heads, head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(v, {seq_len, batch_size, v_num_heads, v_head_dim}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(a, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(dt_bias, {v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(b, {batch_size, v_num_heads}, q.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<true>(initial_state_indices, {batch_size}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true>(cu_seqlens, {batch_size + 1}, at::kInt);
  CHECK_INPUT_SHAPE_DTYPE<true>(
      initial_state_source, {initial_state_source.size(0), v_num_heads, head_dim, v_head_dim}, at::kFloat);
  CHECK(initial_state_source.size(0) >= batch_size);
  CHECK_EQ(v_num_heads % num_heads, 0);
  TORCH_CHECK(
      A_log.sizes() == at::IntArrayRef({v_num_heads}),
      "Input tensor shape mismatch: expected ",
      at::IntArrayRef({v_num_heads}),
      ", got ",
      A_log.sizes());

  int64_t q_strideB = q.stride(1);
  int64_t q_strideS = q.stride(0);
  int64_t q_strideH = q.stride(2);
  int64_t k_strideB = k.stride(1);
  int64_t k_strideS = k.stride(0);
  int64_t k_strideH = k.stride(2);
  int64_t v_strideB = v.stride(1);
  int64_t v_strideS = v.stride(0);
  int64_t v_strideH = v.stride(2);
  at::Tensor core_attn_out = at::empty({batch_size, seq_len, v_num_heads, v_head_dim}, q.options());
  at::Tensor qk_scale_buf = at::empty({2 * batch_size, seq_len, num_heads}, at::kFloat);

  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(
      q.scalar_type(), A_log.scalar_type(), "fused_sigmoid_gating_delta_rule_update_kernel_impl", [&] {
        fused_sigmoid_gating_delta_rule_update_kernel_impl<scalar_t, param_t>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            A_log.data_ptr<param_t>(),
            a.data_ptr<scalar_t>(),
            dt_bias.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            initial_state_indices.data_ptr<int32_t>(),
            initial_state_source.data_ptr<float>(),
            core_attn_out.data_ptr<scalar_t>(),
            qk_scale_buf.data_ptr<float>(),
            seq_len,
            batch_size,
            num_heads,
            head_dim,
            v_num_heads,
            v_head_dim,
            q_strideB,
            q_strideS,
            q_strideH,
            k_strideB,
            k_strideS,
            k_strideH,
            v_strideB,
            v_strideS,
            v_strideH,
            use_qk_l2norm_in_kernel,
            softplus_threshold);
      });
  return core_attn_out;
}

// A_log: [num_v_heads]
// a: [batch, num_v_heads]
// b: [batch, num_v_heads]
// dt_bias: [num_v_heads]
// -A_log.float().exp() * F.softplus(a.float() + dt_bias)
std::tuple<at::Tensor, at::Tensor>
fused_gdn_gating_cpu(const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b, const at::Tensor& dt_bias) {
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(1, dt_bias);
  CHECK_CONTIGUOUS(a);
  CHECK_EQ(A_log.size(0), a.size(1));
  CHECK_EQ(A_log.size(0), dt_bias.size(0));
  int batch = a.size(0);
  int num_heads = a.size(1);
  CHECK_EQ(b.size(0), batch);
  CHECK_EQ(b.size(1), num_heads);
  at::Tensor out = at::empty({1, batch, num_heads}, a.options().dtype(at::kFloat));
  at::Tensor beta = at::empty({1, batch, num_heads}, b.options());
  CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(a.scalar_type(), A_log.scalar_type(), "fused_gdn_gating_kernel", [&] {
    fused_gdn_gating_kernel_impl<scalar_t>(
        A_log.data_ptr<param_t>(),
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        dt_bias.data_ptr<scalar_t>(),
        out.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        batch,
        num_heads);
  });
  return std::make_tuple(out, beta);
}
