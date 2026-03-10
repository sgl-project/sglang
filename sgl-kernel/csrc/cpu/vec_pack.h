// To use the transpose functions
#include <ATen/native/cpu/utils.h>

#include "vec.h"

namespace {

using namespace at::vec;

template <typename index_t>
inline index_t get_index(index_t* ind, int i) {
  return (ind == nullptr) ? (index_t)i : ind[i];
}

#if defined(CPU_CAPABILITY_AVX512)

template <typename scalar_t, typename func_t>
void MM512_LOAD_VEC(
    const scalar_t* __restrict__ src,
    const float* __restrict__ src_scale,
    int64_t ld_src,
    int64_t index,
    const func_t& f,
    __m512i& dst) {
  dst = f(src + index * ld_src);
}

const __m512 vexp = _mm512_castsi512_ps(_mm512_set1_epi32(kFP8_BIAS));

template <typename func_t>
void MM512_LOAD_VEC(
    const at::Float8_e4m3fn* __restrict__ src,
    const float* __restrict__ src_scale,
    int64_t ld_src,
    int64_t index,
    const func_t& f,
    __m512i& dst) {
  const __m512 scale = _mm512_mul_ps(_mm512_set1_ps(src_scale[index]), vexp);
  __m512i s8 = f(src + index * ld_src);
  __m256i s8_0 = _mm512_extracti32x8_epi32(s8, 0);
  __m512bh bf16_0 = CVT_FP8_TO_BF16_EXT(s8_0);
  __m512 f_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 0));
  __m512 f_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 1));
  f_lo = _mm512_mul_ps(f_lo, scale);
  f_hi = _mm512_mul_ps(f_hi, scale);
  bf16_0 = _mm512_cvtne2ps_pbh(f_hi, f_lo);
  dst = (__m512i)bf16_0;
}

template <typename func_t>
void MM512_LOAD_VEC(
    const at::Float8_e5m2* __restrict__ src,
    const float* __restrict__ src_scale,
    int64_t ld_src,
    int64_t index,
    const func_t& f,
    __m512i& dst) {
  __m512i s8 = f(src + index * ld_src);
  __m256i s8_0 = _mm512_extracti32x8_epi32(s8, 0);
  __m512i a = _mm512_slli_epi16(_mm512_cvtepi8_epi16(s8_0), 8);
  __m256i ah = _mm512_extracti64x4_epi64(a, 0);
  __m256i bh = _mm512_extracti64x4_epi64(a, 1);
  __m512 a_ = _mm512_cvtph_ps(ah);
  __m512 b_ = _mm512_cvtph_ps(bh);
  __m512bh bf16_0 = _mm512_cvtne2ps_pbh(b_, a_);
  dst = (__m512i)bf16_0;
}

// key: from [N, 32] to [32/2, N, 2]
template <typename scalar_t, typename packed_t, typename index_t>
inline void pack_vnni_Nx32(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    MM512_LOAD_VEC(
        src,
        src_scale,
        ld_src,
        index,
        [](const packed_t* __restrict__ x) { return _mm512_loadu_si512(x); },
        vinputs[n]);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask = (1 << N) - 1;
  for (int k = 0; k < 16; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask, vinputs[k]);
  }
}

template <typename scalar_t, typename packed_t, typename index_t>
inline void pack_vnni_N_remainder(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[16];

  int K2 = K >> 1;
  const __mmask16 vmask = (1 << K2) - 1;

  int n = 0;
  for (; n < N; ++n) {
    index_t index = get_index(ind, n);
    MM512_LOAD_VEC(
        src,
        src_scale,
        ld_src,
        index,
        [vmask](const packed_t* __restrict__ x) { return _mm512_maskz_loadu_epi32((__mmask16)vmask, x); },
        vinputs[n]);
  }
  // padding with zero to avoid uninitialized vectors
  for (; n < 16; ++n) {
    vinputs[n] = _mm512_set1_epi32(0);
  }

  // pack key
  transpose_16x16_32bit(vinputs);

  const __mmask16 vmask2 = (1 << N) - 1;
  for (int k = 0; k < K2; ++k) {
    _mm512_mask_storeu_epi32(dst + k * ld_dst * 2, vmask2, vinputs[k]);
  }
}

// value: from [K, 32] to [K/2, 32, 2]
template <typename scalar_t, typename packed_t, typename index_t>
inline void pack_vnni_Kx32(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int K,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  int k = 0;
  for (; k < K; ++k) {
    index_t index = get_index(ind, k);
    MM512_LOAD_VEC(
        src,
        src_scale,
        ld_src,
        index,
        [](const packed_t* __restrict__ x) { return _mm512_loadu_si512(x); },
        vinputs[k]);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2, d0);
  _mm512_storeu_si512(dst + 0 * ld_dst * 2 + 32, d1);
}

template <typename scalar_t, typename packed_t, typename index_t>
inline void pack_vnni_K_remainder(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  __m512i vinputs[2];

  const __mmask32 vmask = (1 << N) - 1;

  int k = 0;
  for (; k < K; ++k) {
    index_t index = get_index(ind, k);
    MM512_LOAD_VEC(
        src,
        src_scale,
        ld_src,
        index,
        [vmask](const packed_t* __restrict__ x) { return _mm512_maskz_loadu_epi16((__mmask32)vmask, x); },
        vinputs[k]);
  }
  // padding with zero to avoid uninitialized vectors
  for (; k < 2; ++k) {
    vinputs[k] = _mm512_set1_epi32(0);
  }

  // pack value
  __m512i d0, d1;
  std::tie(d0, d1) = transpose_2x32_16bit(vinputs[0], vinputs[1]);

  if (N <= 16) {
    // 2N * 16bits: N * 32bits
    const __mmask16 vmask2 = (1 << N) - 1;
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2, vmask2, d0);
  } else {
    // 2(N-16) * 16bits: (N-16) * 32bits
    const __mmask16 vmask2 = (1 << (N - 16)) - 1;
    _mm512_storeu_epi32(dst + 0 * ld_dst * 2, d0);
    _mm512_mask_storeu_epi32(dst + 0 * ld_dst * 2 + 32, vmask2, d1);
  }
}
#endif

// convert to vnni format
// from [N, K/2, 2] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename packed_t, typename index_t, bool is_indexed>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int NB = div_up(N, 16);
  const int KB = K / 32;
  const int K_remainder = K - KB * 32;

  for (int nb = 0; nb < NB; ++nb) {
    int nb_size = std::min(N - nb * 16, 16);
    for (int kb = 0; kb < KB; ++kb) {
      // handle 16x512bits each block
      pack_vnni_Nx32<scalar_t, packed_t, index_t>(
          /*    dst */ dst + ((kb * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + kb * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /* src_scale*/ src_scale,
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
    if (K_remainder > 0) {
      pack_vnni_N_remainder<scalar_t, packed_t, index_t>(
          /*    dst */ dst + ((KB * 32) >> 1) * ld_dst * 2 + nb * 16 * 2,
          /*    src */ src + KB * 32 + (is_indexed ? 0 : nb * 16 * ld_src),
          /* src_scale */ src_scale,
          /*    ind */ is_indexed ? ind + nb * 16 : nullptr,
          /*      N */ nb_size,
          /*      K */ K_remainder,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  for (int n = 0; n < N; ++n) {
    index_t index = get_index(ind, n);
    float scale = src_scale != nullptr ? src_scale[index] : 1.0f;
    for (int k = 0; k < K / 2; ++k) {
      for (int d = 0; d < 2; ++d) {
        dst[k * ld_dst * 2 + n * 2 + d] = src[index * ld_src + k * 2 + d] * scale;
      }
    }
  }
#endif
}

template <typename scalar_t>
void pack_vnni(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int N, int K, int ld_src, int ld_dst) {
  pack_vnni<scalar_t, scalar_t, int32_t, false>(dst, src, nullptr, nullptr, N, K, ld_src, ld_dst);
}

template <typename scalar_t, typename packed_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  pack_vnni<scalar_t, packed_t, int32_t, false>(dst, src, src_scale, nullptr, N, K, ld_src, ld_dst);
}

template <typename scalar_t, typename packed_t, typename index_t>
void pack_vnni(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int N,
    int K,
    int ld_src,
    int ld_dst) {
  assert(ind != nullptr);
  pack_vnni<scalar_t, packed_t, index_t, true>(dst, src, src_scale, ind, N, K, ld_src, ld_dst);
}

// convert to vnni format
// from [K/2, 2, N] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t, typename packed_t, typename index_t, bool is_indexed>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
#if defined(CPU_CAPABILITY_AVX512)
  const int KB = div_up(K, 2);
  const int NB = N / 32;
  const int N_remainder = N - NB * 32;

  for (int kb = 0; kb < KB; ++kb) {
    int kb_size = std::min(K - kb * 2, 2);
    for (int nb = 0; nb < NB; ++nb) {
      // handle 2x512bits each block
      pack_vnni_Kx32<scalar_t, packed_t, index_t>(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + nb * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + nb * 32,
          /* src_scale */ src_scale,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
    if (N_remainder > 0) {
      pack_vnni_K_remainder(
          /*    dst */ dst + ((kb * 2) >> 1) * ld_dst * 2 + NB * 32 * 2,
          /*    src */ src + (is_indexed ? 0 : kb * 2 * ld_src) + NB * 32,
          /* src_scale */ src_scale,
          /*    ind */ is_indexed ? ind + kb * 2 : nullptr,
          /*      K */ kb_size,
          /*      N */ N_remainder,
          /* ld_src */ ld_src,
          /* ld_dst */ ld_dst);
    }
  }
#else
  int k = 0;
  for (; k < (K >> 1) * 2; k += 2) {
    index_t index0 = get_index(ind, k + 0);
    index_t index1 = get_index(ind, k + 1);
    float scale0 = src_scale != nullptr ? src_scale[index0] : 1.0f;
    float scale1 = src_scale != nullptr ? src_scale[index1] : 1.0f;
    for (int n = 0; n < N; ++n) {
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 0] = src[index0 * ld_src + n] * scale0;
      dst[(k >> 1) * ld_dst * 2 + n * 2 + 1] = src[index1 * ld_src + n] * scale1;
    }
  }
  if (K % 2 != 0) {
    index_t index = get_index(ind, K - 1);
    float scale = src_scale != nullptr ? src_scale[index] : 1.0f;
    for (int n = 0; n < N; ++n) {
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 0] = src[index * ld_src + n] * scale;
      dst[(K >> 1) * ld_dst * 2 + n * 2 + 1] = 0;
    }
    k += 2;
  }
#endif
}

template <typename scalar_t>
void pack_vnni2(scalar_t* __restrict__ dst, const scalar_t* __restrict__ src, int K, int N, int ld_src, int ld_dst) {
  pack_vnni2<scalar_t, scalar_t, int32_t, false>(dst, src, nullptr, nullptr, K, N, ld_src, ld_dst);
}

template <typename scalar_t, typename packed_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  pack_vnni2<scalar_t, packed_t, int32_t, false>(dst, src, src_scale, nullptr, K, N, ld_src, ld_dst);
}

template <typename scalar_t, typename packed_t, typename index_t>
void pack_vnni2(
    scalar_t* __restrict__ dst,
    const packed_t* __restrict__ src,
    const float* __restrict__ src_scale,
    const index_t* __restrict__ ind,
    int K,
    int N,
    int ld_src,
    int ld_dst) {
  assert(ind != nullptr);
  pack_vnni2<scalar_t, packed_t, index_t, true>(dst, src, src_scale, ind, K, N, ld_src, ld_dst);
}

}  // anonymous namespace
