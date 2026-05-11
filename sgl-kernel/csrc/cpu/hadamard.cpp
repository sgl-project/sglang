#include "common.h"
#include "vec.h"

namespace {

using fVec = at::vec::Vectorized<float>;

// ── Unified scalar FWHT ──
// For float: buf unused, work=out; casts are no-ops.
// For bf16/f16: work=buf as scratch; casts do dtype conversion.
template <typename scalar_t>
inline void fwht_row_scalar(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    float* __restrict__ buf,
    int64_t n,
    float scale) {
  if (n == 1) {
    out[0] = static_cast<scalar_t>(static_cast<float>(in[0]) * scale);
    return;
  }
  if (n == 2) {
    float a = static_cast<float>(in[0]), b = static_cast<float>(in[1]);
    out[0] = static_cast<scalar_t>((a + b) * scale);
    out[1] = static_cast<scalar_t>((a - b) * scale);
    return;
  }
  float* work;
  if constexpr (std::is_same_v<scalar_t, float>) {
    work = out;
  } else {
    work = buf;
  }
  // h=1: read from in → work
  for (int64_t j = 0; j < n; j += 2) {
    float a = static_cast<float>(in[j]), b = static_cast<float>(in[j + 1]);
    work[j]     = a + b;
    work[j + 1] = a - b;
  }
  // h=2..n/4: in-place on work
  const int64_t last_h = n >> 1;
  for (int64_t h = 2; h < last_h; h <<= 1)
    for (int64_t j = 0; j < n; j += 2 * h)
      for (int64_t k = 0; k < h; ++k) {
        float a = work[j + k], b = work[j + k + h];
        work[j + k]     = a + b;
        work[j + k + h] = a - b;
      }
  // h=n/2: butterfly + scale → out
  for (int64_t k = 0; k < last_h; ++k) {
    float a = work[k], b = work[k + last_h];
    out[k]          = static_cast<scalar_t>((a + b) * scale);
    out[k + last_h] = static_cast<scalar_t>((a - b) * scale);
  }
}

#if defined(CPU_CAPABILITY_AVX512)

// Phase-1: in-register butterflies h=1,2,4,8.
// Requires permute/blend intrinsics with no Vectorized equivalent.
static inline __m512 fwht_phase1_fused(__m512 v) {
  __m512 p, s, d;
  p = _mm512_permute_ps(v, 0b10'11'00'01);                          // h=1
  s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
  v = _mm512_mask_blend_ps((__mmask16)0xAAAA, s, d);
  p = _mm512_permute_ps(v, 0b01'00'11'10);                          // h=2
  s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
  v = _mm512_mask_blend_ps((__mmask16)0xCCCC, s, d);
  p = _mm512_permutexvar_ps(                                         // h=4
      _mm512_set_epi32(11,10,9,8, 15,14,13,12, 3,2,1,0, 7,6,5,4), v);
  s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
  v = _mm512_mask_blend_ps((__mmask16)0xF0F0, s, d);
  p = _mm512_permutexvar_ps(                                         // h=8
      _mm512_set_epi32(7,6,5,4, 3,2,1,0, 15,14,13,12, 11,10,9,8), v);
  s = _mm512_add_ps(v, p); d = _mm512_sub_ps(p, v);
  v = _mm512_mask_blend_ps((__mmask16)0xFF00, s, d);
  return v;
}

// fVec wrapper for fwht_phase1_fused
static inline fVec fwht_phase1(fVec v) {
  return fVec(fwht_phase1_fused(__m512(v)));
}

// ── Vectorized dtype conversion (16 elements) ──
// For float: identity load/store. For bf16/f16: intrinsic conversion.
template <typename scalar_t> static inline fVec load_cvt(const scalar_t*);
template <typename scalar_t> static inline void store_cvt(scalar_t*, fVec);

template <> inline fVec load_cvt<float>(const float* p) {
  return fVec::loadu(p);
}
template <> inline fVec load_cvt<at::BFloat16>(const at::BFloat16* p) {
  return fVec(CVT_BF16_TO_FP32(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
}
template <> inline fVec load_cvt<at::Half>(const at::Half* p) {
  return fVec(CVT_FP16_TO_FP32(
      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))));
}
template <> inline void store_cvt<float>(float* p, fVec v) {
  v.store(p);
}
template <> inline void store_cvt<at::BFloat16>(at::BFloat16* p, fVec v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
      (__m256i)_mm512_cvtneps_pbh(__m512(v)));
}
template <> inline void store_cvt<at::Half>(at::Half* p, fVec v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),
      _mm512_cvtps_ph(__m512(v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Phase-2: h=16..last_h-1 vectorized butterflies on float buffer.
static inline void fwht_phase2(float* __restrict__ buf, int64_t n, int64_t last_h) {
  for (int64_t h = 16; h < last_h; h <<= 1)
    for (int64_t j = 0; j < n; j += 2 * h)
      for (int64_t k = 0; k < h; k += fVec::size()) {
        fVec a = fVec::loadu(buf + j + k);
        fVec b = fVec::loadu(buf + j + k + h);
        (a + b).store(buf + j + k);
        (a - b).store(buf + j + k + h);
      }
}

// Unified AVX-512 FWHT with optional dtype conversion.
// For float: load_cvt/store_cvt are identity, work=out.
// For bf16/f16: load_cvt/store_cvt handle conversion, work=buf.
template <typename scalar_t>
inline void fwht_row_avx512(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    float* __restrict__ buf,
    int64_t n,
    float scale) {
  fVec sv(scale);
  if (n == 16) {
    store_cvt(out, fwht_phase1(load_cvt(in)) * sv);
    return;
  }

  float* work;
  if constexpr (std::is_same_v<scalar_t, float>) {
    work = out;
  } else {
    work = buf;
  }

  // Phase 1: load (+ convert) → fused butterflies → work
  for (int64_t j = 0; j < n; j += fVec::size())
    fwht_phase1(load_cvt(in + j)).store(work + j);

  // Phase 2: h=16..n/4 butterflies on work
  const int64_t last_h = n >> 1;
  fwht_phase2(work, n, last_h);

  // Last stage: butterfly + scale (+ convert) → out
  for (int64_t k = 0; k < last_h; k += fVec::size()) {
    fVec a = fVec::loadu(work + k);
    fVec b = fVec::loadu(work + k + last_h);
    store_cvt(out + k,          (a + b) * sv);
    store_cvt(out + k + last_h, (a - b) * sv);
  }
}

#endif  // CPU_CAPABILITY_AVX512

template <typename scalar_t>
void fast_hadamard_transform_kernel_impl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t rows,
    int64_t n,
    float scale) {
  at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
    float* buf = nullptr;
    std::vector<float> scratch;
    if constexpr (!std::is_same_v<scalar_t, float>) {
      scratch.resize(n);
      buf = scratch.data();
    }
    for (int64_t r = begin; r < end; ++r) {
#if defined(CPU_CAPABILITY_AVX512)
      if (n >= 16)
        fwht_row_avx512(output + r * n, input + r * n, buf, n, scale);
      else
#endif
        fwht_row_scalar(output + r * n, input + r * n, buf, n, scale);
    }
  });
}

}  // anonymous namespace

at::Tensor fast_hadamard_transform_cpu(const at::Tensor& x, double scale) {
  CHECK_INPUT(x);
  const int64_t n = x.size(-1);
  TORCH_CHECK(n > 0 && (n & (n - 1)) == 0,
              "fast_hadamard_transform: last dim must be a power of 2, got ", n);

  const int64_t rows = x.numel() / n;
  auto out = at::empty_like(x);
  const float s = static_cast<float>(scale);

  CPU_DISPATCH_FLOATING_TYPES(x.scalar_type(),
      "fast_hadamard_transform_cpu", [&] {
    fast_hadamard_transform_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), rows, n, s);
  });

  return out;
}
