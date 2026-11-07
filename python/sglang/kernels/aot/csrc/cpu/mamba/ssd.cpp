#include <ATen/record_function.h>

#include <array>

#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

namespace {

using fVec = at::vec::Vectorized<float>;

// same threshold as the triton kernels
inline float softplus(float v) {
  return v <= 20.f ? std::log1p(std::exp(v)) : v;
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
  std::fill(dst + div_up(K, int64_t(2)) * N * 2, dst + (Kp / 2) * N * 2, scalar_t(0));
}

// out[i] = in[i] * s
template <typename scalar_t>
inline void scale_row(scalar_t* __restrict__ out, const scalar_t* __restrict__ in, float s, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();
  const fVec vs(s);
  int64_t d = 0;
  for (; d <= size - kVecSize; d += kVecSize) {
    auto [a0, a1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(in + d));
    convert_from_float_ext<scalar_t>(a0 * vs, a1 * vs).store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(static_cast<float>(in[d]) * s);
  }
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

// w[s] = cb[s] * exp(cum_l - cum[s]) * dt[s] for s <= l, zero padded to Kp
inline void decay_row(
    float* __restrict__ w,
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
    (fVec::loadu(cb + s) * decay * fVec::loadu(dt + s)).store(w + s);
  }
  if (s < len) {
    const int64_t rem = len - s;
    fVec decay = (vcum_l - fVec::loadu(cum + s, rem)).exp_u20();
    (fVec::loadu(cb + s, rem) * decay * fVec::loadu(dt + s, rem)).store(w + s, rem);
  }
  std::fill(w + len, w + Kp, 0.f);
}

// out = acc + x * D, then out *= silu(z)
template <typename scalar_t>
inline void scan_epilogue_row(
    scalar_t* __restrict__ out,
    const float* __restrict__ acc,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ z,
    const float* __restrict__ D,
    int64_t P) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int64_t kVecSize = bVec::size();
  const fVec one(1.f);
  int64_t p = 0;
  for (; p <= P - kVecSize; p += kVecSize) {
    fVec o0 = fVec::loadu(acc + p);
    fVec o1 = fVec::loadu(acc + p + fVec::size());
    if (D != nullptr) {
      auto [x0, x1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(x + p));
      o0 = at::vec::fmadd(x0, fVec::loadu(D + p), o0);
      o1 = at::vec::fmadd(x1, fVec::loadu(D + p + fVec::size()), o1);
    }
    if (z != nullptr) {
      auto [z0, z1] = at::vec::convert_to_float<scalar_t>(bVec::loadu(z + p));
      o0 = o0 * z0 / (one + z0.neg().exp_u20());
      o1 = o1 * z1 / (one + z1.neg().exp_u20());
    }
    convert_from_float_ext<scalar_t>(o0, o1).store(out + p);
  }
  for (; p < P; ++p) {
    float o = acc[p];
    if (D != nullptr) {
      o += static_cast<float>(x[p]) * D[p];
    }
    if (z != nullptr) {
      const float g = static_cast<float>(z[p]);
      o *= g / (1.f + std::exp(-g));
    }
    out[p] = static_cast<scalar_t>(o);
  }
}

// Mamba2 SSD chunked scan, following the 3 stages of the triton implementation:
//   1. chunk state:   S_c^T = B_c^T @ (x_c * dt * exp(cum_last - cum))      [brgemm, amx]
//   2. state passing: S_in[c + 1] = S_in[c] * exp(cum_last[c]) + S_c        [avx512]
//   3. chunk scan:    y = (C @ S_in^T) * exp(cum) + ((C @ B^T) * L) @ x      [brgemm, amx]
// states are kept transposed as [N, P] so that they serve directly as the vnni B operand.
//
// x, z  : [Bs, T, H, P]
// B, C  : [Bs, T, G, N]
// dt    : [Bs, T, H]
// init  : [Bs, H, P, N]
// final : [Bs, H, P, N]
// states: [Bs, NC, H, N, P] workspace
// dt_buf, cum_buf : [Bs, H, NC, L] workspace
template <typename scalar_t>
void mamba_chunk_scan_kernel_impl(
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ states,
    float* __restrict__ dt_buf,
    float* __restrict__ cum_buf,
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
    int64_t Bs,
    int64_t T,
    int64_t H,
    int64_t P,
    int64_t G,
    int64_t N,
    int64_t L) {
  const int64_t NC = div_up(T, L);
  const int64_t HG = H / G;
  const int64_t Lp = div_up(L, int64_t(TILE_K)) * TILE_K;
  const int64_t Np = div_up(N, int64_t(TILE_K)) * TILE_K;
  const int64_t ld_x = H * P;
  const int64_t ld_bc = G * N;

  // split heads of a group into HB blocks when [Bs, NC, G] alone cannot occupy all threads;
  // the group-shared B^T / CB are recomputed per block.
  const int64_t HB = std::min(HG, std::max(int64_t(1), div_up(int64_t(at::get_num_threads()), Bs * NC * G)));
  const int64_t HB_size = div_up(HG, HB);

  // stage 1: dt, cumsum and chunk local states, parallel on [Bs, NC, G, HB]
  at::parallel_for(0, Bs * NC * G * HB, 0, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> Bt(N * Lp);
    std::vector<scalar_t> xw(L * P);
    std::vector<scalar_t> xw_packed(Lp * P);

    int64_t b{0}, c{0}, g{0}, hb{0};
    data_index_init(begin, b, Bs, c, NC, g, G, hb, HB);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t t0 = c * L;
      const int64_t Lc = std::min(L, T - t0);
      const int64_t Kp = div_up(Lc, int64_t(TILE_K)) * TILE_K;

      // B^T: [N, Kp], shared by all heads in the group
      transpose_16bit(Lc, N, Bm + ((b * T + t0) * G + g) * N, ld_bc, Bt.data(), Kp);
      for (int64_t n = 0; n < N; ++n) {
        std::fill(Bt.data() + n * Kp + Lc, Bt.data() + n * Kp + Kp, scalar_t(0));
      }

      for (int64_t h = g * HG + hb * HB_size; h < std::min(g * HG + (hb + 1) * HB_size, (g + 1) * HG); ++h) {
        float* __restrict__ dt_row = dt_buf + ((b * H + h) * NC + c) * L;
        float* __restrict__ cum_row = cum_buf + ((b * H + h) * NC + c) * L;
        const float bias = dt_bias != nullptr ? dt_bias[h] : 0.f;
        float running = 0.f;
        for (int64_t l = 0; l < Lc; ++l) {
          float v = dt[(b * T + t0 + l) * H + h] + bias;
          if (dt_softplus) {
            v = softplus(v);
          }
          v = std::min(std::max(v, dt_min), dt_max);
          dt_row[l] = v;
          running += v * A[h];
          cum_row[l] = running;
        }

        // x * dt * exp(cum_last - cum): [Lc, P]
        for (int64_t l = 0; l < Lc; ++l) {
          const float w = dt_row[l] * std::exp(running - cum_row[l]);
          scale_row(xw.data() + l * P, x + (b * T + t0 + l) * ld_x + h * P, w, P);
        }
        pack_rows_vnni(xw_packed.data(), xw.data(), Lc, Kp, P, P);

        at::native::cpublas::brgemm(
            /*     M */ N,
            /*     N */ P,
            /*     K */ Kp,
            /*   lda */ Kp,
            /*   ldb */ P,
            /*   ldc */ P,
            /* add_C */ false,
            /*     A */ Bt.data(),
            /*     B */ xw_packed.data(),
            /*     C */ states + ((b * NC + c) * H + h) * N * P);
      }
      data_index_step(b, Bs, c, NC, g, G, hb, HB);
    }
    at::native::cpublas::brgemm_release();
  });

  // stage 2: inter-chunk recurrence, parallel on [Bs, H]
  //   states[c] is replaced in-place by the state entering chunk c
  at::parallel_for(0, Bs * H, 0, [&](int64_t begin, int64_t end) {
    constexpr int64_t kVecSize = fVec::size();
    const int64_t size = N * P;
    std::vector<float> run(size);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t b = i / H;
      const int64_t h = i % H;
      if (init != nullptr) {
        at::native::utils::transpose<float>(P, N, init + i * P * N, N, run.data(), P);
      } else {
        std::fill(run.begin(), run.end(), 0.f);
      }
      for (int64_t c = 0; c < NC; ++c) {
        const int64_t Lc = std::min(L, T - c * L);
        const float decay = std::exp(cum_buf[((b * H + h) * NC + c) * L + Lc - 1]);
        const fVec vdecay(decay);
        float* __restrict__ st = states + ((b * NC + c) * H + h) * size;
        int64_t j = 0;
        for (; j <= size - kVecSize; j += kVecSize) {
          fVec s = fVec::loadu(st + j);
          fVec r = fVec::loadu(run.data() + j);
          r.store(st + j);
          at::vec::fmadd(r, vdecay, s).store(run.data() + j);
        }
        for (; j < size; ++j) {
          const float s = st[j];
          st[j] = run[j];
          run[j] = run[j] * decay + s;
        }
      }
      at::native::utils::transpose<float>(N, P, run.data(), P, final_state + i * P * N, N);
    }
  });

  // stage 3: chunk output, parallel on [Bs, NC, G, HB]
  at::parallel_for(0, Bs * NC * G * HB, 0, [&](int64_t begin, int64_t end) {
    std::vector<scalar_t> C_pad(N == Np ? 0 : L * Np);
    std::vector<scalar_t> B_packed(Np * L);
    std::vector<float> CB(L * L);
    std::vector<float> w_row(Lp);
    std::vector<scalar_t> W(L * Lp);
    std::vector<scalar_t> x_packed(Lp * P);
    std::vector<scalar_t> s_cvt(N * P);
    std::vector<scalar_t> s_packed(Np * P);
    std::vector<float> acc(L * P);
    std::vector<float> D_row(P);

    int64_t b{0}, c{0}, g{0}, hb{0};
    data_index_init(begin, b, Bs, c, NC, g, G, hb, HB);
    for (int64_t i = begin; i < end; ++i) {
      const int64_t t0 = c * L;
      const int64_t Lc = std::min(L, T - t0);
      const int64_t Kp = div_up(Lc, int64_t(TILE_K)) * TILE_K;
      const scalar_t* __restrict__ C_ptr = Cm + ((b * T + t0) * G + g) * N;
      const scalar_t* __restrict__ B_ptr = Bm + ((b * T + t0) * G + g) * N;

      // C as A operand: [Lc, Np]
      const scalar_t* A_C = C_ptr;
      int64_t lda_C = ld_bc;
      if (N != Np) {
        for (int64_t l = 0; l < Lc; ++l) {
          std::copy(C_ptr + l * ld_bc, C_ptr + l * ld_bc + N, C_pad.data() + l * Np);
          std::fill(C_pad.data() + l * Np + N, C_pad.data() + l * Np + Np, scalar_t(0));
        }
        A_C = C_pad.data();
        lda_C = Np;
      }

      // CB = C @ B^T: [Lc, Lc], shared by all heads in the group
      pack_vnni<scalar_t>(B_packed.data(), B_ptr, Lc, N, ld_bc, Lc);
      std::fill(B_packed.data() + (N / 2) * Lc * 2, B_packed.data() + (Np / 2) * Lc * 2, scalar_t(0));
      at::native::cpublas::brgemm(
          /*     M */ Lc,
          /*     N */ Lc,
          /*     K */ Np,
          /*   lda */ lda_C,
          /*   ldb */ Lc,
          /*   ldc */ Lc,
          /* add_C */ false,
          /*     A */ A_C,
          /*     B */ B_packed.data(),
          /*     C */ CB.data());

      for (int64_t h = g * HG + hb * HB_size; h < std::min(g * HG + (hb + 1) * HB_size, (g + 1) * HG); ++h) {
        const float* __restrict__ dt_row = dt_buf + ((b * H + h) * NC + c) * L;
        const float* __restrict__ cum_row = cum_buf + ((b * H + h) * NC + c) * L;

        // W = tril(CB * exp(cum_l - cum_s) * dt_s): [Lc, Kp]
        for (int64_t l = 0; l < Lc; ++l) {
          decay_row(w_row.data(), CB.data() + l * Lc, cum_row, dt_row, l, Kp);
          store_row_fp32(W.data() + l * Kp, w_row.data(), Kp);
        }
        pack_rows_vnni(x_packed.data(), x + (b * T + t0) * ld_x + h * P, Lc, Kp, P, ld_x);

        // state entering this chunk as vnni B operand: [Np/2, P, 2]
        const float* __restrict__ st = states + ((b * NC + c) * H + h) * N * P;
        store_row_fp32(s_cvt.data(), st, N * P);
        pack_rows_vnni(s_packed.data(), s_cvt.data(), N, Np, P, P);

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
            /*     B */ s_packed.data(),
            /*     C */ acc.data());
        for (int64_t l = 0; l < Lc; ++l) {
          scale_row(acc.data() + l * P, std::exp(cum_row[l]), P);
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
            /*     A */ W.data(),
            /*     B */ x_packed.data(),
            /*     C */ acc.data());

        const float* D_ptr = nullptr;
        if (D != nullptr) {
          for (int64_t p = 0; p < P; ++p) {
            D_row[p] = D[h * D_head_stride + p * D_dim_stride];
          }
          D_ptr = D_row.data();
        }
        for (int64_t l = 0; l < Lc; ++l) {
          const int64_t offset = (b * T + t0 + l) * ld_x + h * P;
          scan_epilogue_row(
              out + offset, acc.data() + l * P, x + offset, z != nullptr ? z + offset : nullptr, D_ptr, P);
        }
      }
      data_index_step(b, Bs, c, NC, g, G, hb, HB);
    }
    at::native::cpublas::brgemm_release();
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
    std::vector<float> row(N);
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
          load_row_fp32(row.data(), state_row, N);
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
                fVec s = at::vec::fmadd(fVec::loadu(row.data() + n), vdecay, vdx * load_fp32x16(B_row + n));
                s.store(row.data() + n);
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
                fVec s = at::vec::fmadd(fVec::loadu(row.data() + n), decay, vdx * load_fp32x16(B_row + n));
                s.store(row.data() + n);
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
            store_row_fp32(state_row, row.data(), N);
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
  TORCH_CHECK(x.dim() == 4, "x must have shape [batch, time, heads, dim]");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x must be bfloat16 or float16 on CPU");
  TORCH_CHECK(dt.sizes() == at::IntArrayRef({x.size(0), x.size(1), x.size(2)}), "dt shape does not match x");
  TORCH_CHECK(B.dim() == 4 && C.sizes() == B.sizes(), "B and C must have shape [batch, time, groups, dstate]");
  const int64_t Bs = x.size(0);
  const int64_t T = x.size(1);
  const int64_t H = x.size(2);
  const int64_t P = x.size(3);
  const int64_t G = B.size(2);
  const int64_t N = B.size(3);
  TORCH_CHECK(B.size(0) == Bs && B.size(1) == T, "B shape does not match x");
  TORCH_CHECK(H % G == 0, "heads must be divisible by groups");
  TORCH_CHECK(N % 2 == 0, "dstate must be even");
  TORCH_CHECK(A.numel() == H, "A must have one value per head");
  if (initial_states.has_value()) {
    TORCH_CHECK(initial_states->sizes() == at::IntArrayRef({Bs, H, P, N}), "initial_states shape mismatch");
  }
  if (D.has_value()) {
    TORCH_CHECK(
        D->sizes() == at::IntArrayRef({H}) || D->sizes() == at::IntArrayRef({H, P}),
        "D must be [heads] or [heads, dim]");
  }

  const auto dtype = x.scalar_type();
  auto x_c = x.contiguous();
  auto B_c = B.to(dtype).contiguous();
  auto C_c = C.to(dtype).contiguous();
  auto z_c = z.has_value() ? z->to(dtype).contiguous() : at::Tensor();
  auto dt_f = dt.to(at::kFloat).contiguous();
  auto A_f = A.to(at::kFloat).reshape({H}).contiguous();
  auto D_f = D.has_value() ? D->to(at::kFloat).contiguous() : at::Tensor();
  auto bias_f = dt_bias.has_value() ? dt_bias->to(at::kFloat).reshape({H}).contiguous() : at::Tensor();
  auto init_f = initial_states.has_value() ? initial_states->to(at::kFloat).contiguous() : at::Tensor();

  auto output = out.has_value() ? *out : at::empty_like(x);
  const bool direct_out = output.scalar_type() == dtype && output.is_contiguous();
  auto out_buf = direct_out ? output : at::empty_like(x_c);

  const int64_t L = chunk_size;
  const int64_t NC = div_up(T, L);
  auto float_options = x.options().dtype(at::kFloat);
  auto final_f = at::empty({Bs, H, P, N}, float_options);
  auto states = at::empty({Bs, NC, H, N, P}, float_options);
  auto dt_buf = at::empty({Bs, H, NC, L}, float_options);
  auto cum_buf = at::empty({Bs, H, NC, L}, float_options);

  if (T == 0) {
    final_f.copy_(init_f.defined() ? init_f : at::zeros_like(final_f));
  } else {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "mamba_chunk_scan_combined_cpu", [&] {
      mamba_chunk_scan_kernel_impl<scalar_t>(
          out_buf.data_ptr<scalar_t>(),
          final_f.data_ptr<float>(),
          states.data_ptr<float>(),
          dt_buf.data_ptr<float>(),
          cum_buf.data_ptr<float>(),
          x_c.data_ptr<scalar_t>(),
          B_c.data_ptr<scalar_t>(),
          C_c.data_ptr<scalar_t>(),
          z.has_value() ? z_c.data_ptr<scalar_t>() : nullptr,
          dt_f.data_ptr<float>(),
          A_f.data_ptr<float>(),
          D.has_value() ? D_f.data_ptr<float>() : nullptr,
          D.has_value() && D_f.dim() == 2 ? P : 1,
          D.has_value() && D_f.dim() == 2 ? 1 : 0,
          dt_bias.has_value() ? bias_f.data_ptr<float>() : nullptr,
          initial_states.has_value() ? init_f.data_ptr<float>() : nullptr,
          dt_softplus,
          static_cast<float>(dt_min),
          static_cast<float>(dt_max),
          Bs,
          T,
          H,
          P,
          G,
          N,
          L);
    });
  }

  if (!direct_out) {
    output.copy_(out_buf);
  }
  const auto result_type = state_dtype.value_or(C.scalar_type());
  return {output, final_f.to(result_type)};
}
