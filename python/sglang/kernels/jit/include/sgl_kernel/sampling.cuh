/**
 * \file sampling.cuh
 * \brief Counter-based random numbers for sampling kernels: Philox4x32-10 and the transforms
 *        from one 32-bit Philox word to a uniform or a base-2 Gumbel.
 */

#pragma once
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

namespace device::ptx {

/// \brief lg2.approx.ftz.f32: one MUFU.LG2 without the denormal fix-up; for a normal
///        argument it returns the same bits as lg2.approx.f32.
SGL_DEVICE float lg2_approx_ftz(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace device::ptx

namespace device::sampling {

/// \brief Philox4x32-10 (Salmon et al. 2011): four random words for counter `c` under key `k`.
SGL_DEVICE uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int32_t i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

/**
 * \brief The first four words of the stream `curand_init(seed, subsequence, offset)` starts:
 *        Philox4x32-10 under key `seed` at counter (offset / 4, subsequence).
 * \note `offset` must be a multiple of 4 (torch's Philox offsets are), so torch's generator state
 *       maps onto it directly.
 */
SGL_DEVICE uint4 curand_philox4(uint64_t seed, uint64_t subsequence, uint64_t offset) {
  const auto counter = offset / 4;
  return philox(
      make_uint4(
          static_cast<uint32_t>(counter),
          static_cast<uint32_t>(counter >> 32),
          static_cast<uint32_t>(subsequence),
          static_cast<uint32_t>(subsequence >> 32)),
      make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)));
}

/**
 * \brief Base-2 Gumbel for keys compared in log2 units: -log2(-log2(1 - v)) = G / ln 2 + log2(ln 2),
 *        a constant shift that does not change an argmax. Range (-5.1, 33], max |error| 1.3e-5.
 * \note One MUFU lg2 for E whichever half x is in (the half is a per-lane random bit, so a warp calling
 *       a different log per half would run both); E by its series for v < 2^-7, so the large-Gumbel end
 *       stays exact. The lg2 arguments t in [2^-33, 1/2], 1 - t in [1/2, 1 - 2^-7] and E in [2^-33, 34)
 *       are all normal.
 */
SGL_DEVICE float gumbel2_fast(uint32_t x) {
  // min(v, 1 - v) for v = (x + 0.5) 2^-32, in [2^-33, 1/2]; the top bit of x says which one it is.
  // 1 - v comes from ~x, since 1 - v formed in fp32 drops v's low bits, so E is finite and > 0.
  constexpr auto folded_uniform = [](uint32_t w) {
    constexpr float kTwoM32 = 2.3283064365386963e-10f;  // 2^-32
    return (__uint2float_rn(w >> 31 ? ~w : w) + 0.5f) * kTwoM32;
  };
  const float t = folded_uniform(x);
  const bool hi = x >> 31;
  const float l = -ptx::lg2_approx_ftz(hi ? t : 1.f - t);
  const float e =
      !hi && t < 0.0078125f ? t * 1.4426950408889634f * (1.f + t * (0.5f + t * (0.33333334f + t * 0.25f))) : l;
  return -ptx::lg2_approx_ftz(e);
}

/// \brief Uniform in (0, 1) for an inverse CDF: u = ((x >> 9) + 0.5) 2^-23, every value exact and in
///        [2^-24, 1 - 2^-24], so u * c < c for every normal fp32 c and an element of zero mass is never drawn.
SGL_DEVICE float uniform_23bit(uint32_t x) {
  return (__uint2float_rn(x >> 9) + 0.5f) * 1.1920928955078125e-7f;
}

}  // namespace device::sampling

}  // namespace sglang
