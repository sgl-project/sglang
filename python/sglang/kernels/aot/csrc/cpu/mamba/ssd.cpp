#include <ATen/record_function.h>

#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

namespace {

using fVec = at::vec::Vectorized<float>;

struct ScratchArena {
  char* ptr;
  int64_t offset{0};

  explicit ScratchArena(void* base = nullptr) : ptr(static_cast<char*>(base)) {}

  template <typename T>
  T* alloc(int64_t count) {
    T* res = ptr != nullptr ? reinterpret_cast<T*>(ptr + offset) : nullptr;
    offset += div_up(count * static_cast<int64_t>(sizeof(T)), int64_t(64)) * 64;
    return res;
  }
};

// same threshold as the triton kernels
inline float softplus(float v) {
  return v <= 20.f ? std::log1p(std::exp(v)) : v;
}

inline fVec softplus(const fVec& x) {
  constexpr float kThreshold = 20.f;
  fVec mask_hi = x > fVec(kThreshold);
  fVec mask_lo = x < fVec(-kThreshold);
  fVec expx = x.exp_u20();
  fVec log1pex = (expx + fVec(1.f)).log();
  return fVec::blendv(fVec::blendv(log1pex, expx, mask_lo), x, mask_hi);
}

template <typename scalar_t>
inline void transpose_16bit(int64_t M, int64_t N, const scalar_t* src, int64_t ld_src, scalar_t* dst, int64_t ld_dst) {
  static_assert(sizeof(scalar_t) == sizeof(uint16_t));
  at::native::utils::transpose<uint16_t>(
      M, N, reinterpret_cast<const uint16_t*>(src), ld_src, reinterpret_cast<uint16_t*>(dst), ld_dst);
}

// pack [K, N] rows (leading dim ld_src) to vnni [Kp/2, N, 2], zero padding K to Kp
template <typename scalar_t>
inline void pack_rows_vnni(scalar_t* dst, const scalar_t* src, int64_t K, int64_t Kp, int64_t N, int64_t ld_src) {
  pack_vnni2<scalar_t>(dst, src, K, N, ld_src, N);
  const int64_t packed_pairs = div_up(K, int64_t(2));
  if (packed_pairs < Kp / 2) {
    std::fill(dst + packed_pairs * N * 2, dst + (Kp / 2) * N * 2, scalar_t(0));
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <typename scalar_t>
inline void store_vnni2_block(scalar_t* __restrict__ dst, __m512i s0, __m512i s1) {
  auto [d0, d1] = transpose_2x32_16bit(s0, s1);
  _mm512_storeu_si512(dst, d0);
  _mm512_storeu_si512(dst + 32, d1);
}

template <typename scalar_t>
inline void store_vnni2_rem(scalar_t* __restrict__ dst, __m512i s0, __m512i s1, int64_t N_rem) {
  auto [d0, d1] = transpose_2x32_16bit(s0, s1);
  if (N_rem <= 16) {
    const __mmask16 vmask = static_cast<__mmask16>((1u << N_rem) - 1);
    _mm512_mask_storeu_epi32(dst, vmask, d0);
  } else {
    const __mmask16 vmask = static_cast<__mmask16>((1u << (N_rem - 16)) - 1);
    _mm512_storeu_epi32(dst, d0);
    _mm512_mask_storeu_epi32(dst + 32, vmask, d1);
  }
}
#endif

// pack [K, N] rows (leading dim ld_src) scaled by scales[k] to vnni [Kp/2, N, 2], zero padding K to Kp
template <typename scalar_t>
inline void pack_rows_vnni_scaled(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const float* __restrict__ scales,
    int64_t K,
    int64_t Kp,
    int64_t N,
    int64_t ld_src) {
  const int64_t KB = div_up(K, int64_t(2));
#if defined(CPU_CAPABILITY_AVX512)
  using bVec = at::vec::Vectorized<scalar_t>;
  const int64_t NB = N / 32;
  const int64_t N_rem = N - NB * 32;
  const __m512i vzero = _mm512_setzero_si512();

  for (int64_t kb = 0; kb < KB; ++kb) {
    const int64_t k0 = kb * 2;
    const int64_t k1 = k0 + 1;
    const bool has_k1 = k1 < K;
    const fVec vs0(scales[k0]);
    const fVec vs1(has_k1 ? scales[k1] : 0.f);
    const scalar_t* __restrict__ r0 = src + k0 * ld_src;
    const scalar_t* __restrict__ r1 = has_k1 ? src + k1 * ld_src : nullptr;
    scalar_t* __restrict__ out = dst + kb * N * 2;

    for (int64_t nb = 0; nb < NB; ++nb) {
      const int64_t n = nb * 32;
      auto [a0, a1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(r0 + n));
      __m512i s0 = (__m512i)convert_from_float_ext<scalar_t>(a0 * vs0, a1 * vs0);
      __m512i s1 = vzero;
      if (has_k1) {
        auto [b0, b1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(r1 + n));
        s1 = (__m512i)convert_from_float_ext<scalar_t>(b0 * vs1, b1 * vs1);
      }
      store_vnni2_block(out + n * 2, s0, s1);
    }
    if (N_rem > 0) {
      const int64_t n = NB * 32;
      auto [a0, a1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(r0 + n, N_rem));
      __m512i s0 = (__m512i)convert_from_float_ext<scalar_t>(a0 * vs0, a1 * vs0);
      __m512i s1 = vzero;
      if (has_k1) {
        auto [b0, b1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(r1 + n, N_rem));
        s1 = (__m512i)convert_from_float_ext<scalar_t>(b0 * vs1, b1 * vs1);
      }
      store_vnni2_rem(out + n * 2, s0, s1, N_rem);
    }
  }
#else
  for (int64_t kb = 0; kb < KB; ++kb) {
    const int64_t k0 = kb * 2;
    const int64_t k1 = k0 + 1;
    const bool has_k1 = k1 < K;
    const float s0 = scales[k0];
    const float s1 = has_k1 ? scales[k1] : 0.f;
    const scalar_t* __restrict__ r0 = src + k0 * ld_src;
    const scalar_t* __restrict__ r1 = has_k1 ? src + k1 * ld_src : nullptr;
    scalar_t* __restrict__ out = dst + kb * N * 2;
    for (int64_t n = 0; n < N; ++n) {
      out[n * 2 + 0] = static_cast<scalar_t>(static_cast<float>(r0[n]) * s0);
      out[n * 2 + 1] = has_k1 ? static_cast<scalar_t>(static_cast<float>(r1[n]) * s1) : scalar_t(0);
    }
  }
#endif
  if (KB < Kp / 2) {
    std::fill(dst + KB * N * 2, dst + (Kp / 2) * N * 2, scalar_t(0));
  }
}

// pack [K, N] fp32 rows (leading dim ld_src) to vnni [Kp/2, N, 2] scalar_t, zero padding K to Kp
template <typename scalar_t>
inline void pack_rows_vnni_fp32(
    scalar_t* __restrict__ dst, const float* __restrict__ src, int64_t K, int64_t Kp, int64_t N, int64_t ld_src) {
  const int64_t KB = div_up(K, int64_t(2));
#if defined(CPU_CAPABILITY_AVX512)
  const int64_t NB = N / 32;
  const int64_t N_rem = N - NB * 32;
  const __m512i vzero = _mm512_setzero_si512();

  for (int64_t kb = 0; kb < KB; ++kb) {
    const int64_t k0 = kb * 2;
    const int64_t k1 = k0 + 1;
    const bool has_k1 = k1 < K;
    const float* __restrict__ r0 = src + k0 * ld_src;
    const float* __restrict__ r1 = has_k1 ? src + k1 * ld_src : nullptr;
    scalar_t* __restrict__ out = dst + kb * N * 2;

    for (int64_t nb = 0; nb < NB; ++nb) {
      const int64_t n = nb * 32;
      __m512i s0 = (__m512i)convert_from_float_ext<scalar_t>(fVec::loadu(r0 + n), fVec::loadu(r0 + n + 16));
      __m512i s1 =
          has_k1 ? (__m512i)convert_from_float_ext<scalar_t>(fVec::loadu(r1 + n), fVec::loadu(r1 + n + 16)) : vzero;
      store_vnni2_block(out + n * 2, s0, s1);
    }
    if (N_rem > 0) {
      const int64_t n = NB * 32;
      const int64_t rem0 = std::min(N_rem, int64_t(16));
      const int64_t rem1 = std::max(N_rem - 16, int64_t(0));
      fVec v00 = fVec::loadu(r0 + n, rem0);
      fVec v01 = rem1 > 0 ? fVec::loadu(r0 + n + 16, rem1) : fVec(0.f);
      __m512i s0 = (__m512i)convert_from_float_ext<scalar_t>(v00, v01);
      __m512i s1 = vzero;
      if (has_k1) {
        fVec v10 = fVec::loadu(r1 + n, rem0);
        fVec v11 = rem1 > 0 ? fVec::loadu(r1 + n + 16, rem1) : fVec(0.f);
        s1 = (__m512i)convert_from_float_ext<scalar_t>(v10, v11);
      }
      store_vnni2_rem(out + n * 2, s0, s1, N_rem);
    }
  }
#else
  for (int64_t kb = 0; kb < KB; ++kb) {
    const int64_t k0 = kb * 2;
    const int64_t k1 = k0 + 1;
    const bool has_k1 = k1 < K;
    const float* __restrict__ r0 = src + k0 * ld_src;
    const float* __restrict__ r1 = has_k1 ? src + k1 * ld_src : nullptr;
    scalar_t* __restrict__ out = dst + kb * N * 2;
    for (int64_t n = 0; n < N; ++n) {
      out[n * 2 + 0] = static_cast<scalar_t>(r0[n]);
      out[n * 2 + 1] = has_k1 ? static_cast<scalar_t>(r1[n]) : scalar_t(0);
    }
  }
#endif
  if (KB < Kp / 2) {
    std::fill(dst + KB * N * 2, dst + (Kp / 2) * N * 2, scalar_t(0));
  }
}

// transpose B [K, N] (leading dim ld_bc) to [N, Kp], zero padding K to Kp
template <typename scalar_t>
inline void transpose_pad_B(
    scalar_t* __restrict__ Bt, const scalar_t* __restrict__ B_src, int64_t K, int64_t Kp, int64_t N, int64_t ld_bc) {
  transpose_16bit(K, N, B_src, ld_bc, Bt, Kp);
  if (K < Kp) {
    for (int64_t n = 0; n < N; ++n) {
      std::fill(Bt + n * Kp + K, Bt + n * Kp + Kp, scalar_t(0));
    }
  }
}

// compute local chunk state: dst [N, P] = Bt [N, Kp] @ (x [K, P] * dt * exp(cum_last - cum))
template <typename scalar_t>
inline void compute_chunk_state(
    float* __restrict__ dst,
    const scalar_t* __restrict__ Bt,
    const scalar_t* __restrict__ x_src,
    const float* __restrict__ dt_row,
    const float* __restrict__ cum_row,
    float* __restrict__ w_buf,
    scalar_t* __restrict__ xw_packed,
    int64_t K,
    int64_t Kp,
    int64_t N,
    int64_t P,
    int64_t ld_x) {
  constexpr int64_t kVecSize = fVec::size();
  const fVec vcum_last(cum_row[K - 1]);
  int64_t l = 0;
  for (; l <= K - kVecSize; l += kVecSize) {
    fVec decay = (vcum_last - fVec::loadu(cum_row + l)).exp_u20();
    (fVec::loadu(dt_row + l) * decay).store(w_buf + l);
  }
  if (l < K) {
    const int64_t rem = K - l;
    fVec decay = (vcum_last - fVec::loadu(cum_row + l, rem)).exp_u20();
    (fVec::loadu(dt_row + l, rem) * decay).store(w_buf + l, rem);
  }
  pack_rows_vnni_scaled(xw_packed, x_src, w_buf, K, Kp, P, ld_x);
  at::native::cpublas::brgemm(
      /*     M */ N,
      /*     N */ P,
      /*     K */ Kp,
      /*   lda */ Kp,
      /*   ldb */ P,
      /*   ldc */ P,
      /* add_C */ false,
      /*     A */ Bt,
      /*     B */ xw_packed,
      /*     C */ dst);
}

// data[i] *= s
inline void scale_row(float* __restrict__ data, float s, int64_t size) {
  constexpr int64_t kVecSize = fVec::size();
  const fVec vs(s);
  int64_t d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    (fVec::loadu(data + d) * vs).store(data + d);
  }
  for (; d < size; ++d) {
    data[d] *= s;
  }
}

template <typename scalar_t>
inline fVec load_fp32x16(const scalar_t* __restrict__ ptr) {
  return load_float_vec<scalar_t>(ptr);
}

template <>
inline fVec load_fp32x16<float>(const float* __restrict__ ptr) {
  return fVec::loadu(ptr);
}

template <typename scalar_t>
inline void store_fp32x16(scalar_t* __restrict__ ptr, const fVec& v) {
  store_from_float_ext<scalar_t>(ptr, v);
}

template <>
inline void store_fp32x16<float>(float* __restrict__ ptr, const fVec& v) {
  v.store(ptr);
}

// fp32 row <-> state storage (fp32, bf16 or fp16)
template <typename scalar_t>
inline void load_row_fp32(float* __restrict__ dst, const scalar_t* __restrict__ src, int64_t size) {
  constexpr int64_t kVecSize = fVec::size();
  int64_t d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    load_fp32x16(src + d).store(dst + d);
  }
  for (; d < size; ++d) {
    dst[d] = static_cast<float>(src[d]);
  }
}

template <typename scalar_t>
inline void store_row_fp32(scalar_t* __restrict__ dst, const float* __restrict__ src, int64_t size) {
  constexpr int64_t kVecSize = fVec::size();
  int64_t d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    store_fp32x16(dst + d, fVec::loadu(src + d));
  }
  for (; d < size; ++d) {
    dst[d] = static_cast<scalar_t>(src[d]);
  }
}

// w[s] = cb[s] * exp(cum_l - cum[s]) * dt[s] for s <= l, converted to scalar_t and zero padded to Kp
template <typename scalar_t>
inline void decay_row(
    scalar_t* __restrict__ w,
    const float* __restrict__ cb,
    const float* __restrict__ cum,
    const float* __restrict__ dt,
    int64_t l,
    int64_t Kp) {
  constexpr int64_t kVecSize = fVec::size();
  const fVec vcum_l(cum[l]);
  const int64_t len = l + 1;
  int64_t s = 0;
  for (; s <= len - kVecSize; s += kVecSize) {
    fVec decay = (vcum_l - fVec::loadu(cum + s)).exp_u20();
    store_fp32x16(w + s, fVec::loadu(cb + s) * decay * fVec::loadu(dt + s));
  }
  if (s < len) {
    const int64_t rem = len - s;
    alignas(64) float tmp[kVecSize];
    fVec decay = (vcum_l - fVec::loadu(cum + s, rem)).exp_u20();
    (fVec::loadu(cb + s, rem) * decay * fVec::loadu(dt + s, rem)).store(tmp);
    for (int64_t j = 0; j < rem; ++j) {
      w[s + j] = static_cast<scalar_t>(tmp[j]);
    }
  }
  if (len < Kp) {
    std::fill(w + len, w + Kp, scalar_t(0));
  }
}

// out = acc + x * D, then out *= silu(z)
template <typename scalar_t>
inline void scan_epilogue_row(
    scalar_t* __restrict__ out,
    const float* __restrict__ acc,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ z,
    const float* __restrict__ D,
    bool D_is_scalar,
    int64_t P) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();
  const fVec vd_scalar(D != nullptr && D_is_scalar ? D[0] : 0.f);
  int64_t p = 0;
  for (; p <= P - kVecSize; p += kVecSize) {
    fVec o0 = fVec::loadu(acc + p);
    fVec o1 = fVec::loadu(acc + p + fVec::size());
    if (D != nullptr) {
      auto [x0, x1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(x + p));
      if (D_is_scalar) {
        o0 = at::vec::fmadd(x0, vd_scalar, o0);
        o1 = at::vec::fmadd(x1, vd_scalar, o1);
      } else {
        o0 = at::vec::fmadd(x0, fVec::loadu(D + p), o0);
        o1 = at::vec::fmadd(x1, fVec::loadu(D + p + fVec::size()), o1);
      }
    }
    if (z != nullptr) {
      auto [z0, z1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(z + p));
      o0 = o0 * fast_silu(z0);
      o1 = o1 * fast_silu(z1);
    }
    convert_from_float_ext<scalar_t>(o0, o1).store(out + p);
  }
  for (; p < P; ++p) {
    float o = acc[p];
    if (D != nullptr) {
      o += static_cast<float>(x[p]) * (D_is_scalar ? D[0] : D[p]);
    }
    if (z != nullptr) {
      const float g = static_cast<float>(z[p]);
      o *= g / (1.f + std::exp(-g));
    }
    out[p] = static_cast<scalar_t>(o);
  }
}

template <typename scalar_t>
inline int64_t mamba_chunk_scan_scratch_bytes(int64_t N, int64_t P, int64_t L, bool has_req) {
  const int64_t Lp = div_up(L, int64_t(TILE_K)) * TILE_K;
  const int64_t Np = div_up(N, int64_t(TILE_K)) * TILE_K;

  ScratchArena s1;
  s1.alloc<scalar_t>(N * Lp);
  s1.alloc<scalar_t>(Lp * P);
  s1.alloc<float>(L);

  ScratchArena s2;
  s2.alloc<float>(N * P);

  ScratchArena s3;
  s3.alloc<scalar_t>(N == Np ? 0 : L * Np);
  s3.alloc<scalar_t>(Np * L);
  s3.alloc<float>(L * L);
  s3.alloc<scalar_t>(L * Lp);
  s3.alloc<scalar_t>(Lp * P);
  s3.alloc<scalar_t>(Np * P);
  s3.alloc<float>(L * P);

  int64_t bytes = std::max({s1.offset, s2.offset, s3.offset});
  if (has_req) {
    ScratchArena s4;
    s4.alloc<scalar_t>(N * Lp);
    s4.alloc<scalar_t>(Lp * P);
    s4.alloc<float>(L);
    s4.alloc<float>(N * P);
    bytes = std::max(bytes, s4.offset);
  }
  return bytes;
}

// Mamba2 SSD chunked scan, following the 3 stages of the triton implementation:
//   1. chunk state:   S_c^T = B_c^T @ (x_c * dt * exp(cum_last - cum))      [brgemm, amx]
//   2. state passing: S_in[c + 1] = S_in[c] * exp(cum_last[c]) + S_c        [avx512]
//   3. chunk scan:    y = (C @ S_in^T) * exp(cum) + ((C @ B^T) * L) @ x      [brgemm, amx]
// states are kept transposed as [N, P] so that they serve directly as the vnni B operand.
//
// Varlen: all sequences are packed along time (T = total tokens), sequence s spans
// [cu_seqlens[s], cu_seqlens[s + 1]) and is split into its own chunks of size L, so the
// numerics are identical to scanning every sequence separately, but all (seq, chunk)
// pairs run inside one parallel region.
//
// Optional stage 4 extracts the state after `req_pos - cu_seqlens[req_seq]` tokens of
// sequence `req_seq` from the materialized chunk states (O(1) chunk per request),
// instead of re-running the scan on the prefix.
//
// x, z  : [T, H, P]
// B, C  : [T, G, N]
// dt    : [T, H]
// init  : [S, H, P, N]
// final : [S, H, P, N]
// req   : [R, H, P, N]
// states: [NCt, H, N, P] workspace
// dt_buf, cum_buf : [NCt, H, L] workspace
template <typename scalar_t>
void mamba_chunk_scan_kernel_impl(
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ req_state,
    float* __restrict__ states,
    float* __restrict__ dt_buf,
    float* __restrict__ cum_buf,
    void* __restrict__ scratch,
    int64_t scratch_per_thread,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ Bm,
    const scalar_t* __restrict__ Cm,
    const scalar_t* __restrict__ z,
    const float* __restrict__ dt,
    const float* __restrict__ A,
    const float* __restrict__ D,
    int64_t D_head_stride,
    int64_t D_dim_stride,
    const float* __restrict__ dt_bias,
    const float* __restrict__ init,
    bool dt_softplus,
    float dt_min,
    float dt_max,
    const int64_t* __restrict__ cu_seqlens,   // [S + 1]
    const int64_t* __restrict__ chunk_start,  // [S + 1], first global chunk of each seq
    const int64_t* __restrict__ chunk_seq,    // [NCt], seq of each global chunk
    const int64_t* __restrict__ req_seq,      // [R]
    const int64_t* __restrict__ req_pos,      // [R], global token position
    int64_t S,
    int64_t NCt,
    int64_t R,
    int64_t H,
    int64_t P,
    int64_t G,
    int64_t N,
    int64_t L) {
  const int64_t HG = H / G;
  const int64_t Lp = div_up(L, int64_t(TILE_K)) * TILE_K;
  const int64_t Np = div_up(N, int64_t(TILE_K)) * TILE_K;
  const int64_t ld_x = H * P;
  const int64_t ld_bc = G * N;

  // (global token offset, length) of global chunk c
  auto chunk_range = [&](int64_t c, int64_t& t0, int64_t& Lc) {
    const int64_t s = chunk_seq[c];
    const int64_t local = (c - chunk_start[s]) * L;
    t0 = cu_seqlens[s] + local;
    Lc = std::min(L, cu_seqlens[s + 1] - cu_seqlens[s] - local);
  };

  // split heads of a group into HB blocks when [NCt, G] alone cannot occupy all threads;
  // the group-shared B^T / CB are recomputed per block.
  const int64_t HB =
      std::min(HG, std::max(int64_t(1), div_up(int64_t(at::get_num_threads()), std::max(int64_t(1), NCt * G))));
  const int64_t HB_size = div_up(HG, HB);

  // stage 1: dt, cumsum and chunk local states, parallel on [NCt, G, HB]
  at::parallel_for(0, NCt * G * HB, 0, [&](int64_t begin, int64_t end) {
    ScratchArena arena(static_cast<char*>(scratch) + at::get_thread_num() * scratch_per_thread);
    scalar_t* __restrict__ Bt = arena.alloc<scalar_t>(N * Lp);
    scalar_t* __restrict__ xw_packed = arena.alloc<scalar_t>(Lp * P);
    float* __restrict__ w_buf = arena.alloc<float>(L);

    constexpr int64_t kVecSize = fVec::size();
    const fVec vmin(dt_min);
    const fVec vmax(dt_max);

    int64_t c{0}, g{0}, hb{0};
    data_index_init(begin, c, NCt, g, G, hb, HB);
    for (int64_t i = begin; i < end; ++i) {
      int64_t t0, Lc;
      chunk_range(c, t0, Lc);
      const int64_t Kp = div_up(Lc, int64_t(TILE_K)) * TILE_K;

      // B^T: [N, Kp], shared by all heads in the group
      transpose_pad_B(Bt, Bm + (t0 * G + g) * N, Lc, Kp, N, ld_bc);

      for (int64_t h = g * HG + hb * HB_size; h < std::min(g * HG + (hb + 1) * HB_size, (g + 1) * HG); ++h) {
        float* __restrict__ dt_row = dt_buf + (c * H + h) * L;
        float* __restrict__ cum_row = cum_buf + (c * H + h) * L;
        const float bias = dt_bias != nullptr ? dt_bias[h] : 0.f;
        const fVec vbias(bias);

        for (int64_t l = 0; l < Lc; ++l) {
          dt_row[l] = dt[(t0 + l) * H + h];
        }
        int64_t l = 0;
        for (; l <= Lc - kVecSize; l += kVecSize) {
          fVec v = fVec::loadu(dt_row + l) + vbias;
          if (dt_softplus) {
            v = softplus(v);
          }
          at::vec::minimum(at::vec::maximum(v, vmin), vmax).store(dt_row + l);
        }
        if (l < Lc) {
          const int64_t rem = Lc - l;
          fVec v = fVec::loadu(dt_row + l, rem) + vbias;
          if (dt_softplus) {
            v = softplus(v);
          }
          at::vec::minimum(at::vec::maximum(v, vmin), vmax).store(dt_row + l, rem);
        }

        const float ah = A[h];
        float running = 0.f;
        for (int64_t j = 0; j < Lc; ++j) {
          running += dt_row[j] * ah;
          cum_row[j] = running;
        }

        compute_chunk_state(
            states + (c * H + h) * N * P,
            Bt,
            x + t0 * ld_x + h * P,
            dt_row,
            cum_row,
            w_buf,
            xw_packed,
            Lc,
            Kp,
            N,
            P,
            ld_x);
      }
      data_index_step(c, NCt, g, G, hb, HB);
    }
    at::native::cpublas::brgemm_release();
  });

  // stage 2: inter-chunk recurrence, parallel on [S, H]
  //   states[c] is replaced in-place by the state entering chunk c
  at::parallel_for(0, S * H, 0, [&](int64_t begin, int64_t end) {
    constexpr int64_t kVecSize = fVec::size();
    const int64_t size = N * P;
    ScratchArena arena(static_cast<char*>(scratch) + at::get_thread_num() * scratch_per_thread);
    float* __restrict__ run = arena.alloc<float>(size);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t s = i / H;
      const int64_t h = i % H;
      if (init != nullptr) {
        at::native::utils::transpose<float>(P, N, init + i * P * N, N, run, P);
      } else {
        std::fill(run, run + size, 0.f);
      }
      for (int64_t c = chunk_start[s]; c < chunk_start[s + 1]; ++c) {
        int64_t t0, Lc;
        chunk_range(c, t0, Lc);
        const float decay = std::exp(cum_buf[(c * H + h) * L + Lc - 1]);
        const fVec vdecay(decay);
        float* __restrict__ st = states + (c * H + h) * size;
        int64_t j = 0;
        for (; j <= size - kVecSize; j += kVecSize) {
          fVec sv = fVec::loadu(st + j);
          fVec r = fVec::loadu(run + j);
          r.store(st + j);
          at::vec::fmadd(r, vdecay, sv).store(run + j);
        }
        for (; j < size; ++j) {
          const float sv = st[j];
          st[j] = run[j];
          run[j] = run[j] * decay + sv;
        }
      }
      at::native::utils::transpose<float>(N, P, run, P, final_state + i * P * N, N);
    }
  });

  // stage 3: chunk output, parallel on [NCt, G, HB]
  at::parallel_for(0, NCt * G * HB, 0, [&](int64_t begin, int64_t end) {
    ScratchArena arena(static_cast<char*>(scratch) + at::get_thread_num() * scratch_per_thread);
    scalar_t* __restrict__ C_pad = arena.alloc<scalar_t>(N == Np ? 0 : L * Np);
    scalar_t* __restrict__ B_packed = arena.alloc<scalar_t>(Np * L);
    float* __restrict__ CB = arena.alloc<float>(L * L);
    scalar_t* __restrict__ W = arena.alloc<scalar_t>(L * Lp);
    scalar_t* __restrict__ x_packed = arena.alloc<scalar_t>(Lp * P);
    scalar_t* __restrict__ s_packed = arena.alloc<scalar_t>(Np * P);
    float* __restrict__ acc = arena.alloc<float>(L * P);

    int64_t c{0}, g{0}, hb{0};
    data_index_init(begin, c, NCt, g, G, hb, HB);
    for (int64_t i = begin; i < end; ++i) {
      int64_t t0, Lc;
      chunk_range(c, t0, Lc);
      const int64_t Kp = div_up(Lc, int64_t(TILE_K)) * TILE_K;
      const scalar_t* __restrict__ C_ptr = Cm + (t0 * G + g) * N;
      const scalar_t* __restrict__ B_ptr = Bm + (t0 * G + g) * N;

      // C as A operand: [Lc, Np]
      const scalar_t* A_C = C_ptr;
      int64_t lda_C = ld_bc;
      if (N != Np) {
        for (int64_t l = 0; l < Lc; ++l) {
          std::copy(C_ptr + l * ld_bc, C_ptr + l * ld_bc + N, C_pad + l * Np);
          std::fill(C_pad + l * Np + N, C_pad + l * Np + Np, scalar_t(0));
        }
        A_C = C_pad;
        lda_C = Np;
      }

      // CB = C @ B^T: [Lc, Lc], shared by all heads in the group
      pack_vnni<scalar_t>(B_packed, B_ptr, Lc, N, ld_bc, Lc);
      if (N < Np) {
        std::fill(B_packed + (N / 2) * Lc * 2, B_packed + (Np / 2) * Lc * 2, scalar_t(0));
      }
      at::native::cpublas::brgemm(
          /*     M */ Lc,
          /*     N */ Lc,
          /*     K */ Np,
          /*   lda */ lda_C,
          /*   ldb */ Lc,
          /*   ldc */ Lc,
          /* add_C */ false,
          /*     A */ A_C,
          /*     B */ B_packed,
          /*     C */ CB);

      for (int64_t h = g * HG + hb * HB_size; h < std::min(g * HG + (hb + 1) * HB_size, (g + 1) * HG); ++h) {
        const float* __restrict__ dt_row = dt_buf + (c * H + h) * L;
        const float* __restrict__ cum_row = cum_buf + (c * H + h) * L;

        // W = tril(CB * exp(cum_l - cum_s) * dt_s): [Lc, Kp]
        for (int64_t l = 0; l < Lc; ++l) {
          decay_row(W + l * Kp, CB + l * Lc, cum_row, dt_row, l, Kp);
        }
        pack_rows_vnni(x_packed, x + t0 * ld_x + h * P, Lc, Kp, P, ld_x);

        // state entering this chunk as vnni B operand: [Np/2, P, 2]
        const float* __restrict__ st = states + (c * H + h) * N * P;
        pack_rows_vnni_fp32(s_packed, st, N, Np, P, P);

        // acc = (C @ S_in^T) * exp(cum)
        at::native::cpublas::brgemm(
            /*     M */ Lc,
            /*     N */ P,
            /*     K */ Np,
            /*   lda */ lda_C,
            /*   ldb */ P,
            /*   ldc */ P,
            /* add_C */ false,
            /*     A */ A_C,
            /*     B */ s_packed,
            /*     C */ acc);
        for (int64_t l = 0; l < Lc; ++l) {
          scale_row(acc + l * P, std::exp(cum_row[l]), P);
        }

        // acc += W @ x
        at::native::cpublas::brgemm(
            /*     M */ Lc,
            /*     N */ P,
            /*     K */ Kp,
            /*   lda */ Kp,
            /*   ldb */ P,
            /*   ldc */ P,
            /* add_C */ true,
            /*     A */ W,
            /*     B */ x_packed,
            /*     C */ acc);

        const float* D_head = D != nullptr ? D + h * D_head_stride : nullptr;
        const bool D_is_scalar = (D_dim_stride == 0);
        for (int64_t l = 0; l < Lc; ++l) {
          const int64_t offset = (t0 + l) * ld_x + h * P;
          scan_epilogue_row(
              out + offset, acc + l * P, x + offset, z != nullptr ? z + offset : nullptr, D_head, D_is_scalar, P);
        }
      }
      data_index_step(c, NCt, g, G, hb, HB);
    }
    at::native::cpublas::brgemm_release();
  });

  if (R == 0) {
    return;
  }

  // stage 4: states at requested positions, parallel on [R, H]
  //   local = pos - seq_start, k = local / L, r = local % L
  //   r == 0          : S_in[k] (or final when k == nchunks)
  //   otherwise       : S = S_in[k] * exp(cum[r-1]) + B[:r]^T @ (x[:r] * dt * exp(cum[r-1] - cum))
  at::parallel_for(0, R * H, 0, [&](int64_t begin, int64_t end) {
    ScratchArena arena(static_cast<char*>(scratch) + at::get_thread_num() * scratch_per_thread);
    scalar_t* __restrict__ Bt = arena.alloc<scalar_t>(N * Lp);
    scalar_t* __restrict__ xw_packed = arena.alloc<scalar_t>(Lp * P);
    float* __restrict__ w_buf = arena.alloc<float>(L);
    float* __restrict__ tmp = arena.alloc<float>(N * P);
    bool used_brgemm = false;

    for (int64_t i = begin; i < end; ++i) {
      const int64_t q = i / H;
      const int64_t h = i % H;
      const int64_t s = req_seq[q];
      const int64_t seq_len = cu_seqlens[s + 1] - cu_seqlens[s];
      const int64_t local = req_pos[q] - cu_seqlens[s];
      float* __restrict__ dst = req_state + i * P * N;

      const int64_t k = local / L;
      const int64_t r = local % L;
      if (local == seq_len) {
        std::copy(final_state + (s * H + h) * P * N, final_state + (s * H + h + 1) * P * N, dst);
        continue;
      }
      const int64_t c = chunk_start[s] + k;
      const float* __restrict__ st = states + (c * H + h) * N * P;
      if (r == 0) {
        at::native::utils::transpose<float>(N, P, st, P, dst, N);
        continue;
      }

      int64_t t0, Lc;
      chunk_range(c, t0, Lc);
      const int64_t Kp = div_up(r, int64_t(TILE_K)) * TILE_K;
      const float* __restrict__ dt_row = dt_buf + (c * H + h) * L;
      const float* __restrict__ cum_row = cum_buf + (c * H + h) * L;
      const float cum_last = cum_row[r - 1];

      transpose_pad_B(Bt, Bm + (t0 * G + h / HG) * N, r, Kp, N, ld_bc);
      compute_chunk_state(tmp, Bt, x + t0 * ld_x + h * P, dt_row, cum_row, w_buf, xw_packed, r, Kp, N, P, ld_x);
      used_brgemm = true;

      const float decay = std::exp(cum_last);
      for (int64_t j = 0; j < N * P; ++j) {
        tmp[j] += st[j] * decay;
      }
      at::native::utils::transpose<float>(N, P, tmp, P, dst, N);
    }
    if (used_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

// Fused recurrent update for decode. Each row of the state (d) is independent:
//   s = s * exp(dt * A) + dt * x * B ;  y = s . C + D * x ;  y *= silu(z)
// the row stays in fp32 across the T tokens and is written back once.
//
// x, dt, z, out : [Bs, T, H, Dm] strided, activation dtype
// B, C          : [Bs, T, G, N]  contiguous, activation dtype
// A             : [H, Dm, N] strided, `A_tied` when strides of Dm and N are 0
using Strides4 = std::array<int64_t, 4>;

inline int64_t offset4(const Strides4& s, int64_t b, int64_t t, int64_t h, int64_t d) {
  return b * s[0] + t * s[1] + h * s[2] + d * s[3];
}

template <typename state_t, typename act_t>
void selective_state_update_kernel_impl(
    state_t* __restrict__ state,
    int64_t state_stride_slot,
    int64_t state_stride_head,
    int64_t state_stride_dim,
    int64_t num_slots,
    act_t* __restrict__ out,
    Strides4 out_strides,
    const act_t* __restrict__ x,
    Strides4 x_strides,
    const act_t* __restrict__ dt,
    Strides4 dt_strides,
    const float* __restrict__ A,
    int64_t A_stride_head,
    int64_t A_stride_dim,
    bool A_tied,
    const act_t* __restrict__ Bm,
    const act_t* __restrict__ Cm,
    const float* __restrict__ D,
    int64_t D_stride_head,
    int64_t D_stride_dim,
    const act_t* __restrict__ z,
    Strides4 z_strides,
    const float* __restrict__ dt_bias,
    int64_t bias_stride_head,
    int64_t bias_stride_dim,
    const int64_t* __restrict__ indices,
    int64_t pad_slot_id,
    bool dt_softplus,
    bool disable_state_update,
    float* __restrict__ scratch,
    int64_t scratch_stride,
    int64_t Bs,
    int64_t T,
    int64_t H,
    int64_t Dm,
    int64_t G,
    int64_t N) {
  constexpr int64_t kVecSize = fVec::size();
  constexpr int64_t BLOCK_D = 16;
  const int64_t HG = H / G;
  const int64_t DB = div_up(Dm, BLOCK_D);

  // parallel on [Bs, H, Dm / BLOCK_D]
  at::parallel_for(0, Bs * H * DB, 0, [&](int64_t begin, int64_t end) {
    float* __restrict__ row = scratch + at::get_thread_num() * scratch_stride;
    int64_t b{0}, h{0}, db{0};
    data_index_init(begin, b, Bs, h, H, db, DB);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t slot = indices == nullptr ? b : indices[b];
      if (slot != pad_slot_id) {
        TORCH_CHECK(slot >= 0 && slot < num_slots, "state batch index out of range");
        const int64_t g = h / HG;
        state_t* __restrict__ state_head = state + slot * state_stride_slot + h * state_stride_head;

        for (int64_t d = db * BLOCK_D; d < std::min(Dm, db * BLOCK_D + BLOCK_D); ++d) {
          state_t* __restrict__ state_row = state_head + d * state_stride_dim;
          load_row_fp32(row, state_row, N);
          const float* __restrict__ A_row = A + h * A_stride_head + d * A_stride_dim;
          const float bias = dt_bias != nullptr ? dt_bias[h * bias_stride_head + d * bias_stride_dim] : 0.f;

          for (int64_t t = 0; t < T; ++t) {
            const act_t* __restrict__ B_row = Bm + ((b * T + t) * G + g) * N;
            const act_t* __restrict__ C_row = Cm + ((b * T + t) * G + g) * N;
            float delta = static_cast<float>(dt[offset4(dt_strides, b, t, h, d)]) + bias;
            if (dt_softplus) {
              delta = softplus(delta);
            }
            const float xv = static_cast<float>(x[offset4(x_strides, b, t, h, d)]);
            const fVec vdx(delta * xv);
            fVec vacc(0.f);
            float acc = 0.f;
            int64_t n = 0;
            if (A_tied) {
              const float decay = std::exp(delta * A_row[0]);
              const fVec vdecay(decay);
              for (; n <= N - kVecSize; n += kVecSize) {
                fVec s = at::vec::fmadd(fVec::loadu(row + n), vdecay, vdx * load_fp32x16(B_row + n));
                s.store(row + n);
                vacc = at::vec::fmadd(s, load_fp32x16(C_row + n), vacc);
              }
              for (; n < N; ++n) {
                row[n] = row[n] * decay + delta * xv * static_cast<float>(B_row[n]);
                acc += row[n] * static_cast<float>(C_row[n]);
              }
            } else {
              const fVec vdelta(delta);
              for (; n <= N - kVecSize; n += kVecSize) {
                fVec decay = (vdelta * fVec::loadu(A_row + n)).exp_u20();
                fVec s = at::vec::fmadd(fVec::loadu(row + n), decay, vdx * load_fp32x16(B_row + n));
                s.store(row + n);
                vacc = at::vec::fmadd(s, load_fp32x16(C_row + n), vacc);
              }
              for (; n < N; ++n) {
                row[n] = row[n] * std::exp(delta * A_row[n]) + delta * xv * static_cast<float>(B_row[n]);
                acc += row[n] * static_cast<float>(C_row[n]);
              }
            }
            acc += vec_reduce_sum(vacc);
            if (D != nullptr) {
              acc += xv * D[h * D_stride_head + d * D_stride_dim];
            }
            if (z != nullptr) {
              const float gate = static_cast<float>(z[offset4(z_strides, b, t, h, d)]);
              acc *= gate / (1.f + std::exp(-gate));
            }
            out[offset4(out_strides, b, t, h, d)] = static_cast<act_t>(acc);
          }
          if (!disable_state_update) {
            store_row_fp32(state_row, row, N);
          }
        }
      }
      data_index_step(b, Bs, h, H, db, DB);
    }
  });
}

}  // anonymous namespace

at::Tensor selective_state_update_cpu(
    at::Tensor& state,
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const std::optional<at::Tensor>& D,
    const std::optional<at::Tensor>& z,
    const std::optional<at::Tensor>& dt_bias,
    bool dt_softplus,
    const std::optional<at::Tensor>& state_batch_indices,
    int64_t pad_slot_id,
    bool disable_state_update,
    const std::optional<at::Tensor>& out) {
  RECORD_FUNCTION("sgl-kernel::selective_state_update_cpu", std::vector<c10::IValue>({state, x}));
  TORCH_CHECK(state.dim() == 4, "state must have shape [slots, heads, dim, dstate]");
  TORCH_CHECK(x.dim() == 4, "x must have shape [batch, time, heads, dim]");
  TORCH_CHECK(dt.sizes() == x.sizes(), "dt must have the same shape as x");
  TORCH_CHECK(A.dim() == 3, "A must have shape [heads, dim, dstate]");
  TORCH_CHECK(B.dim() == 4 && C.sizes() == B.sizes(), "B and C must have shape [batch, time, groups, dstate]");
  TORCH_CHECK(state.stride(3) == 1, "state must be contiguous in the last dimension");

  const int64_t Bs = x.size(0);
  const int64_t T = x.size(1);
  const int64_t H = x.size(2);
  const int64_t Dm = x.size(3);
  const int64_t N = state.size(3);
  const int64_t G = B.size(2);
  TORCH_CHECK(H % G == 0, "heads must be divisible by groups");
  TORCH_CHECK(state.size(1) == H && state.size(2) == Dm, "state shape does not match x");
  TORCH_CHECK(A.sizes() == at::IntArrayRef({H, Dm, N}), "A shape does not match state");
  TORCH_CHECK(B.size(0) == Bs && B.size(1) == T && B.size(3) == N, "B shape does not match x/state");
  if (state_batch_indices.has_value()) {
    TORCH_CHECK(state_batch_indices->numel() == Bs, "state_batch_indices must have one entry per batch");
  }

  // activations stay in their dtype and strides; A, D, dt_bias keep their (possibly expanded) strides
  const auto act_dtype = x.scalar_type();
  TORCH_CHECK(
      act_dtype == at::kFloat || act_dtype == at::kBFloat16 || act_dtype == at::kHalf,
      "x must be float32, bfloat16 or float16");
  auto dt_a = dt.to(act_dtype);
  auto B_a = B.to(act_dtype).contiguous();
  auto C_a = C.to(act_dtype).contiguous();
  auto z_a = z.has_value() ? z->to(act_dtype) : at::Tensor();
  auto A_f = A.to(at::kFloat);
  const bool A_tied = A_f.stride(1) == 0 && A_f.stride(2) == 0;
  if (!A_tied && A_f.stride(2) != 1) {
    A_f = A_f.contiguous();
  }
  auto D_f = D.has_value() ? D->to(at::kFloat) : at::Tensor();
  auto bias_f = dt_bias.has_value() ? dt_bias->to(at::kFloat) : at::Tensor();
  auto indices = state_batch_indices.has_value() ? state_batch_indices->to(at::kLong).contiguous() : at::Tensor();
  auto output = out.has_value() ? *out : at::empty_like(x);
  TORCH_CHECK(output.sizes() == x.sizes(), "out must have the same shape as x");
  // padded entries are left untouched, so a temporary must start from the original values
  auto out_a = output.scalar_type() == act_dtype ? output : output.to(act_dtype);

  const int64_t scratch_stride = div_up(N, int64_t(16)) * 16;
  auto scratch = at::empty({at::get_num_threads() * scratch_stride}, x.options().dtype(at::kFloat));

  auto strides4 = [](const at::Tensor& t) { return Strides4{t.stride(0), t.stride(1), t.stride(2), t.stride(3)}; };

  AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, state.scalar_type(), "selective_state_update_cpu", [&] {
    using state_t = scalar_t;
    AT_DISPATCH_REDUCED_FLOATING_TYPES_AND(at::kFloat, act_dtype, "selective_state_update_cpu_act", [&] {
      selective_state_update_kernel_impl<state_t, scalar_t>(
          state.data_ptr<state_t>(),
          state.stride(0),
          state.stride(1),
          state.stride(2),
          state.size(0),
          out_a.data_ptr<scalar_t>(),
          strides4(out_a),
          x.data_ptr<scalar_t>(),
          strides4(x),
          dt_a.data_ptr<scalar_t>(),
          strides4(dt_a),
          A_f.data_ptr<float>(),
          A_f.stride(0),
          A_f.stride(1),
          A_tied,
          B_a.data_ptr<scalar_t>(),
          C_a.data_ptr<scalar_t>(),
          D.has_value() ? D_f.data_ptr<float>() : nullptr,
          D.has_value() ? D_f.stride(0) : 0,
          D.has_value() ? D_f.stride(1) : 0,
          z.has_value() ? z_a.data_ptr<scalar_t>() : nullptr,
          z.has_value() ? strides4(z_a) : Strides4{0, 0, 0, 0},
          dt_bias.has_value() ? bias_f.data_ptr<float>() : nullptr,
          dt_bias.has_value() ? bias_f.stride(0) : 0,
          dt_bias.has_value() ? bias_f.stride(1) : 0,
          state_batch_indices.has_value() ? indices.data_ptr<int64_t>() : nullptr,
          pad_slot_id,
          dt_softplus,
          disable_state_update,
          scratch.data_ptr<float>(),
          scratch_stride,
          Bs,
          T,
          H,
          Dm,
          G,
          N);
    });
  });

  if (!out_a.is_same(output)) {
    output.copy_(out_a);
  }
  return output;
}

namespace {

// Shared launcher for dense [Bs, T, ...] and packed varlen [1, T, ...] scans.
//   cu [S + 1] int64 (cpu), req_seq/req_pos [R] int64 (cpu, optional).
// Returns (output, final_states [S, H, P, N], req_states [R, H, P, N]).
std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba_chunk_scan_varlen_launch(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    int64_t L,
    const std::optional<at::Tensor>& D,
    const std::optional<at::Tensor>& z,
    const std::optional<at::Tensor>& dt_bias,
    const std::optional<at::Tensor>& initial_states,
    const at::Tensor& cu,
    const at::Tensor& req_seq,
    const at::Tensor& req_pos,
    bool dt_softplus,
    double dt_min,
    double dt_max,
    const std::optional<at::ScalarType>& state_dtype,
    const std::optional<at::Tensor>& out) {
  const auto dtype = x.scalar_type();
  const int64_t T = x.size(0) * x.size(1);
  const int64_t H = x.size(2);
  const int64_t P = x.size(3);
  const int64_t G = B.size(2);
  const int64_t N = B.size(3);
  const int64_t S = cu.numel() - 1;
  const int64_t R = req_seq.defined() ? req_seq.numel() : 0;

  auto x_c = x.contiguous();
  auto output = out.has_value() ? *out : at::empty_like(x);
  const bool direct_out = output.scalar_type() == dtype && output.is_contiguous();
  auto out_buf = direct_out ? output : at::empty_like(x_c);

  auto dt_f = dt.to(at::kFloat).contiguous();
  auto A_f = A.to(at::kFloat).reshape({H}).contiguous();
  auto B_c = B.to(dtype).contiguous();
  auto C_c = C.to(dtype).contiguous();
  auto D_f = D.has_value() ? D->to(at::kFloat).contiguous() : at::Tensor();
  auto z_c = z.has_value() ? z->to(dtype).contiguous() : at::Tensor();
  auto bias_f = dt_bias.has_value() ? dt_bias->to(at::kFloat).reshape({H}).contiguous() : at::Tensor();
  auto init_f = initial_states.has_value() ? initial_states->to(at::kFloat).contiguous() : at::Tensor();
  auto cu_c = cu.to(at::kLong).contiguous();
  const int64_t* cu_ptr = cu_c.data_ptr<int64_t>();
  TORCH_CHECK(S >= 0 && cu_ptr[0] == 0 && cu_ptr[S] == T, "cu_seqlens must start at 0 and end at total tokens");

  auto chunk_start_t = at::empty({S + 1}, cu_c.options());
  int64_t* chunk_start = chunk_start_t.data_ptr<int64_t>();
  chunk_start[0] = 0;
  for (int64_t s = 0; s < S; ++s) {
    const int64_t len = cu_ptr[s + 1] - cu_ptr[s];
    TORCH_CHECK(len >= 0, "cu_seqlens must be non-decreasing");
    chunk_start[s + 1] = chunk_start[s] + div_up(len, L);
  }
  if (R > 0) {
    const int64_t* rs = req_seq.data_ptr<int64_t>();
    const int64_t* rp = req_pos.data_ptr<int64_t>();
    for (int64_t q = 0; q < R; ++q) {
      TORCH_CHECK(
          rp[q] >= cu_ptr[rs[q]] && rp[q] <= cu_ptr[rs[q] + 1], "state_pos must lie within its sequence [start, end]");
    }
  }
  const int64_t NCt = chunk_start[S];
  auto chunk_seq_t = at::empty({NCt}, cu_c.options());
  int64_t* chunk_seq = chunk_seq_t.data_ptr<int64_t>();
  for (int64_t s = 0; s < S; ++s) {
    std::fill(chunk_seq + chunk_start[s], chunk_seq + chunk_start[s + 1], s);
  }

  auto float_options = x_c.options().dtype(at::kFloat);
  auto final_f = at::empty({S, H, P, N}, float_options);
  auto req_f = at::empty({R, H, P, N}, float_options);
  const auto result_type = state_dtype.value_or(C.scalar_type());

  if (NCt == 0) {
    // every sequence is empty: final state = initial state
    final_f.copy_(init_f.defined() ? init_f : at::zeros_like(final_f));
    if (R > 0) {
      req_f.copy_(final_f.index_select(0, req_seq));
    }
    if (!direct_out) {
      output.copy_(out_buf);
    }
    return {output, final_f.to(result_type), req_f.to(result_type)};
  }

  auto states = at::empty({NCt, H, N, P}, float_options);
  auto dt_buf = at::empty({NCt, H, L}, float_options);
  auto cum_buf = at::empty({NCt, H, L}, float_options);
  const int num_threads = at::get_num_threads();

  AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "mamba_chunk_scan_combined_cpu", [&] {
    const int64_t scratch_per_thread = mamba_chunk_scan_scratch_bytes<scalar_t>(N, P, L, R > 0);
    auto scratch = at::empty({num_threads * scratch_per_thread}, x_c.options().dtype(at::kChar));
    mamba_chunk_scan_kernel_impl<scalar_t>(
        out_buf.data_ptr<scalar_t>(),
        final_f.data_ptr<float>(),
        R > 0 ? req_f.data_ptr<float>() : nullptr,
        states.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        cum_buf.data_ptr<float>(),
        scratch.data_ptr(),
        scratch_per_thread,
        x_c.data_ptr<scalar_t>(),
        B_c.data_ptr<scalar_t>(),
        C_c.data_ptr<scalar_t>(),
        z_c.defined() ? z_c.data_ptr<scalar_t>() : nullptr,
        dt_f.data_ptr<float>(),
        A_f.data_ptr<float>(),
        D_f.defined() ? D_f.data_ptr<float>() : nullptr,
        D_f.defined() && D_f.dim() == 2 ? P : 1,
        D_f.defined() && D_f.dim() == 2 ? 1 : 0,
        bias_f.defined() ? bias_f.data_ptr<float>() : nullptr,
        init_f.defined() ? init_f.data_ptr<float>() : nullptr,
        dt_softplus,
        static_cast<float>(dt_min),
        static_cast<float>(dt_max),
        cu_ptr,
        chunk_start,
        chunk_seq,
        R > 0 ? req_seq.data_ptr<int64_t>() : nullptr,
        R > 0 ? req_pos.data_ptr<int64_t>() : nullptr,
        S,
        NCt,
        R,
        H,
        P,
        G,
        N,
        L);
  });

  if (!direct_out) {
    output.copy_(out_buf);
  }
  return {output, final_f.to(result_type), req_f.to(result_type)};
}

void check_scan_inputs(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    int64_t chunk_size,
    const std::optional<at::Tensor>& D) {
  TORCH_CHECK(x.dim() == 4, "x must have shape [batch, time, heads, dim]");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x must be bfloat16 or float16 on CPU");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({x.size(0), x.size(1), x.size(2)}), "dt shape does not match x");
  TORCH_CHECK(B.dim() == 4 && C.sizes() == B.sizes(), "B and C must have shape [batch, time, groups, dstate]");
  TORCH_CHECK(B.size(0) == x.size(0) && B.size(1) == x.size(1), "B shape does not match x");
  TORCH_CHECK(x.size(2) % B.size(2) == 0, "heads must be divisible by groups");
  TORCH_CHECK(B.size(3) % 2 == 0, "dstate must be even");
  TORCH_CHECK(A.numel() == x.size(2), "A must have one value per head");
  if (D.has_value()) {
    const int64_t H = x.size(2);
    const int64_t P = x.size(3);
    TORCH_CHECK(
        D->sizes() == at::IntArrayRef({H}) || D->sizes() == at::IntArrayRef({H, P}),
        "D must be [heads] or [heads, dim]");
  }
}

}  // anonymous namespace

std::tuple<at::Tensor, at::Tensor> mamba_chunk_scan_combined_cpu(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    int64_t chunk_size,
    const std::optional<at::Tensor>& D,
    const std::optional<at::Tensor>& z,
    const std::optional<at::Tensor>& dt_bias,
    const std::optional<at::Tensor>& initial_states,
    bool dt_softplus,
    double dt_min,
    double dt_max,
    const std::optional<at::ScalarType>& state_dtype,
    const std::optional<at::Tensor>& out) {
  RECORD_FUNCTION("sgl-kernel::mamba_chunk_scan_combined_cpu", std::vector<c10::IValue>({x, B, C}));
  check_scan_inputs(x, dt, A, B, C, chunk_size, D);
  const int64_t Bs = x.size(0);
  const int64_t T = x.size(1);
  if (initial_states.has_value()) {
    TORCH_CHECK(
        initial_states->sizes() == at::IntArrayRef({Bs, x.size(2), x.size(3), B.size(3)}),
        "initial_states shape mismatch");
  }

  // a dense batch is a varlen batch with equal lengths
  auto cu = at::arange(Bs + 1, at::TensorOptions().dtype(at::kLong)).mul_(T);
  auto [output, final_state, req_state] = mamba_chunk_scan_varlen_launch(
      x,
      dt,
      A,
      B,
      C,
      chunk_size,
      D,
      z,
      dt_bias,
      initial_states,
      cu,
      at::Tensor(),
      at::Tensor(),
      dt_softplus,
      dt_min,
      dt_max,
      state_dtype,
      out);
  return {output, final_state};
}

// Packed varlen scan: one launch for all sequences of a prefill batch.
//   x [1, T, H, P], dt [1, T, H], B/C [1, T, G, N], z [1, T, H, P]
//   initial_states [S, H, P, N], cu_seqlens [S + 1]
//   state_seq_idx / state_pos [R]: optionally also return the state of sequence
//   state_seq_idx[r] after global token position state_pos[r] (cu_seqlens[s] <= pos <= cu_seqlens[s + 1]),
//   computed from the materialized chunk states instead of re-scanning the prefix.
// returns (out [1, T, H, P], final_states [S, H, P, N], states [R, H, P, N])
std::tuple<at::Tensor, at::Tensor, at::Tensor> mamba_chunk_scan_combined_varlen_cpu(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    int64_t chunk_size,
    const std::optional<at::Tensor>& D,
    const std::optional<at::Tensor>& z,
    const std::optional<at::Tensor>& dt_bias,
    const std::optional<at::Tensor>& initial_states,
    const at::Tensor& cu_seqlens,
    const std::optional<at::Tensor>& state_seq_idx,
    const std::optional<at::Tensor>& state_pos,
    bool dt_softplus,
    double dt_min,
    double dt_max,
    const std::optional<at::ScalarType>& state_dtype,
    const std::optional<at::Tensor>& out) {
  RECORD_FUNCTION("sgl-kernel::mamba_chunk_scan_combined_varlen_cpu", std::vector<c10::IValue>({x, B, C, cu_seqlens}));
  check_scan_inputs(x, dt, A, B, C, chunk_size, D);
  TORCH_CHECK(x.size(0) == 1, "varlen scan expects batch == 1");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 1, "cu_seqlens must be 1D [num_seqs + 1]");
  TORCH_CHECK(state_seq_idx.has_value() == state_pos.has_value(), "state_seq_idx and state_pos must be given together");
  const int64_t S = cu_seqlens.numel() - 1;
  if (initial_states.has_value()) {
    TORCH_CHECK(
        initial_states->sizes() == at::IntArrayRef({S, x.size(2), x.size(3), B.size(3)}),
        "initial_states shape mismatch");
  }

  at::Tensor req_seq, req_pos;
  if (state_seq_idx.has_value()) {
    TORCH_CHECK(state_seq_idx->numel() == state_pos->numel(), "state_seq_idx and state_pos size mismatch");
    req_seq = state_seq_idx->to(at::kLong).contiguous();
    req_pos = state_pos->to(at::kLong).contiguous();
    if (req_seq.numel() > 0) {
      TORCH_CHECK(
          req_seq.min().item<int64_t>() >= 0 && req_seq.max().item<int64_t>() < S, "state_seq_idx out of range");
    }
  }

  return mamba_chunk_scan_varlen_launch(
      x,
      dt,
      A,
      B,
      C,
      chunk_size,
      D,
      z,
      dt_bias,
      initial_states,
      cu_seqlens,
      req_seq,
      req_pos,
      dt_softplus,
      dt_min,
      dt_max,
      state_dtype,
      out);
}
