// SPDX-License-Identifier: Apache-2.0
// Shared by the int8 DSpark markov-walk kernels: small_batch (bs 1..4) and wgmma (bs 5..64, and every bs for big
// vocabularies). python/sglang/kernels/ops/speculative/dspark/markov_walk.py builds the weight layouts.
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace sglang::device::ptx {

SGL_DEVICE void cp_async_cg_16B(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem)), "l"(gmem));
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int32_t N>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// The barrier counts this thread's earlier cp.async as one arrival once they land
SGL_DEVICE void cp_async_mbar_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(to_shared(bar)) : "memory");
}

SGL_DEVICE uint64_t ld_relaxed_u64(const uint64_t* ptr) {
  uint64_t value;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
  return value;
}

SGL_DEVICE void st_relaxed_u64(uint64_t* ptr, uint64_t value) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(ptr), "l"(value) : "memory");
}

// Cross-CTA exchange. Per (step, request) a 16-B aligned pair {key, count}: every CTA posts red.max(key, its best)
// then red.release.add(count, 1), and polls the pair with ld_relaxed_v2_u64 until count == gridDim.x; that load's
// key is the global max.
// HARDWARE ASSUMPTION: a 16-B aligned ld.relaxed.gpu.v2.u64 observes both words at one point in L2, so the load that
// sees the final count also sees every max ordered before it. PTX does not promise this (vector accesses are
// unordered scalar accesses, and a relaxed load does not synchronize with the release); it holds on H100, where an
// aligned 16-B load is one L2 sector access.
SGL_DEVICE void ld_relaxed_v2_u64(const uint64_t* ptr, uint64_t (&value)[2]) {
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];" : "=l"(value[0]), "=l"(value[1]) : "l"(ptr) : "memory");
}

// CUDA has no atomicMax for uint64_t (unsigned long)
SGL_DEVICE void red_relaxed_max_u64(uint64_t* ptr, uint64_t value) {
  asm volatile("red.relaxed.gpu.global.max.u64 [%0], %1;" ::"l"(ptr), "l"(value) : "memory");
}

// No return value: a returning atom would make the following poll wait for its round trip
SGL_DEVICE void red_release_add_u64(uint64_t* ptr, uint64_t value) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(ptr), "l"(value) : "memory");
}

SGL_DEVICE uint64_t create_policy_evict_first() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

SGL_DEVICE uint4 ld_cg_hint_v4(const uint4* ptr, uint64_t policy) {
  uint4 value;
  asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(ptr), "l"(policy));
  return value;
}

SGL_DEVICE void prefetch_l2(const void* ptr) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

SGL_DEVICE void prefetch_l2_evict_last(const void* ptr) {
  asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(ptr));
}

SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// mbar_wait_parity with a memory clobber, so the SMEM the barrier tracks is read after it
SGL_DEVICE void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(to_shared(bar)), "r"(parity)
        : "memory");
  } while (!done);
}

// global -> this CTA's SMEM (the .shared::cluster destination is this CTA's SMEM without a cluster launch)
SGL_DEVICE void cp_async_bulk_g2s(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(to_shared(smem)),
      "l"(gmem),
      "r"(bytes),
      "r"(to_shared(bar)),
      "l"(policy)
      : "memory");
}

// %tid.x through volatile asm: unlike threadIdx.x, every read is a fresh S2R the compiler cannot keep live
SGL_DEVICE uint32_t tid_x() {
  uint32_t tid;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
  return tid;
}

// An 8-row run's bf16 logits as 4 packed pairs p0..p3 (pair i = rows 2i (low half), 2i + 1 (high half)).
// bf16x8_max: their max in both halves; bf16x8_first_eq: the first row whose value equals m (ties take the smallest
// row, as torch.argmax). HMNMX2 / HSETP2 on the packed values, no unpacking to fp32.
SGL_DEVICE uint32_t bf16x8_max(uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3) {
  uint32_t m;
  asm("{\n\t.reg .b32 a, b;\n\t"
      "max.bf16x2 a, %1, %2;\n\tmax.bf16x2 b, %3, %4;\n\tmax.bf16x2 a, a, b;\n\t"
      "prmt.b32 b, a, a, 0x1032;\n\tmax.bf16x2 %0, a, b;\n\t}"
      : "=r"(m)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3));
  return m;
}

SGL_DEVICE int32_t bf16x8_first_eq(uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, uint32_t m) {
  int32_t row;
  asm("{\n\t.reg .pred e0, e1, e2, e3, e4, e5, e6, e7;\n\t"
      "setp.eq.bf16x2 e0|e1, %1, %5;\n\tsetp.eq.bf16x2 e2|e3, %2, %5;\n\t"
      "setp.eq.bf16x2 e4|e5, %3, %5;\n\tsetp.eq.bf16x2 e6|e7, %4, %5;\n\t"
      "mov.b32 %0, 7;\n\t@e6 mov.b32 %0, 6;\n\t@e5 mov.b32 %0, 5;\n\t@e4 mov.b32 %0, 4;\n\t"
      "@e3 mov.b32 %0, 3;\n\t@e2 mov.b32 %0, 2;\n\t@e1 mov.b32 %0, 1;\n\t@e0 mov.b32 %0, 0;\n\t}"
      : "=r"(row)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(m));
  return row;
}

// MUFU lg2 / ex2 without the denormal fix-up around every MUFU: for a normal argument .ftz returns the same bits, and
// every caller passes normal numbers
SGL_DEVICE float lg2_approx_ftz(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE float ex2_approx_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

SGL_DEVICE void mma_m16n8k32_s8(int32_t (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

}  // namespace sglang::device::ptx

namespace sglang::dspark_markov_walk {

namespace ptx = device::ptx;

// The Philox counters pack the step into 8 bits (step | request << 8)
inline constexpr int32_t kMaxSteps = 256;
inline constexpr int32_t kThreads = 256;
inline constexpr int32_t kWarps = kThreads / 32;
// A 16-row W2 tile is 8 k32-chunks x 32 lanes of 16-B mma A fragments
inline constexpr int32_t kTileRows = 16;
inline constexpr int32_t kTileVecs = 256;
// W1q row: q_hi[256] | q_lo[256] | s_hi f32 | s_lo f32 | 8 B pad
inline constexpr int32_t kURowWords = 132;
inline constexpr int32_t kURowBytes = kURowWords * 4;
// The CTA's W2 slice: per warp kRegTiles tiles in registers and kSmemTiles in SMEM, 72 tiles = 1152 rows per CTA
inline constexpr int32_t kRegTiles = 6;
inline constexpr int32_t kSmemTiles = 3;
inline constexpr int32_t kWarpTiles = kRegTiles + kSmemTiles;
inline constexpr int32_t kTilesSmem = kSmemTiles * kWarps;
inline constexpr int32_t kTilesPerCta = kWarpTiles * kWarps;
inline constexpr int32_t kRowsCta = kTilesPerCta * kTileRows;
// Batched kernels' state: one {key, count} pair per 128-B line, since atomics to one line serialize in its L2 slice
inline constexpr int32_t kMaxBatch = 64;
inline constexpr int32_t kPairWords = 16;
inline constexpr int32_t kStepWords = kMaxBatch * kPairWords;

// An out-of-range anchor would make the first W1 gather an illegal address
SGL_DEVICE uint32_t clamp_row(int64_t row, int32_t valid_rows) {
  return static_cast<uint32_t>(row < 0 ? 0 : row >= valid_rows ? valid_rows - 1 : row);
}

// 1 / T, with T clamped to [1e-5, 1e4] so +inf and denormals stay finite; 0 (greedy) for T <= 0 or NaN
SGL_DEVICE float inv_temperature(float t) {
  return t > 0.f ? 1.f / fminf(fmaxf(t, 1e-5f), 1e4f) : 0.f;
}

// Argmax key (order-preserving fp32 << 32) | (0xFFFFFFFF - row): the max is the best score, ties take the smallest row
SGL_DEVICE uint64_t pack_key(float score, uint32_t row) {
  const auto bits = __float_as_uint(score);
  const auto ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return (static_cast<uint64_t>(ordered) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - row);
}

SGL_DEVICE uint32_t key_row(uint64_t key) {
  return 0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFu);
}

// The bits of a bf16 pair as F2FP packs them, .x in the low half
SGL_DEVICE uint32_t bf16x2_bits(bf16x2_t value) {
  return *reinterpret_cast<const uint32_t*>(&value);
}

// acc[i] <- the int32 dots of the warp's tile i with B columns from u (a W1q row's q_hi | q_lo words). All chains
// advance one k32 chunk at a time, the register tiles' mma running while the chunk's SMEM fragments load.
SGL_DEVICE void mma_tiles(
    int32_t (&acc)[kWarpTiles][4],
    const uint4 (&w2_reg)[kRegTiles][8],
    const uint4* w2_smem,
    const uint32_t* u,
    int32_t warp_id,
    int32_t lane_id) {
  const int32_t g = lane_id >> 2, tig = lane_id & 3;
#pragma unroll
  for (int32_t i = 0; i < kWarpTiles; ++i) {
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0;
  }
#pragma unroll
  for (int32_t j = 0; j < 8; ++j) {
    uint4 a_smem[kSmemTiles];
#pragma unroll
    for (int32_t s = 0; s < kSmemTiles; ++s) {
      a_smem[s] = w2_smem[((warp_id + kWarps * s) * 8 + j) * 32 + lane_id];
    }
    // B column n takes q_hi for even n and q_lo for odd n: every lane holds the (hi, lo) dots of its rows g and g + 8
    const uint32_t b0 = u[(g & 1) * 64 + j * 8 + tig];
    const uint32_t b1 = u[(g & 1) * 64 + j * 8 + 4 + tig];
#pragma unroll
    for (int32_t i = 0; i < kRegTiles; ++i) {
      ptx::mma_m16n8k32_s8(acc[i], w2_reg[i][j], b0, b1);
    }
#pragma unroll
    for (int32_t s = 0; s < kSmemTiles; ++s) {
      ptx::mma_m16n8k32_s8(acc[kRegTiles + s], a_smem[s], b0, b1);
    }
  }
}

// Philox4x32-10 (Salmon et al. 2011)
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

// The Gumbel transforms take v = (x + 0.5) 2^-32 in (0, 1) at full 32-bit resolution and need E = -log(1 - v). For
// x >= 2^31 they use 1 - v = (~x + 0.5) 2^-32 directly, since 1 - v formed in fp32 drops v's low bits. So this
// returns v for x < 2^31 and 1 - v otherwise, in [2^-33, 1/2]: E is finite and > 0 for every x.
SGL_DEVICE float half_uniform(uint32_t x) {
  constexpr float kTwoM32 = 2.3283064365386963e-10f;  // 2^-32
  return (__uint2float_rn(x >> 31 ? ~x : x) + 0.5f) * kTwoM32;
}

// The natural log as __logf computes it (lg2.approx times ln 2 rounded to fp32), without the denormal fix-up
SGL_DEVICE float logf_ftz(float x) {
  return ptx::lg2_approx_ftz(x) * 0.693147182464599609375f;
}

// Gumbel(0, 1) = -log(E) with MUFU logs (small_batch, where the T > 0 epilogue is mostly these logs). One log for E
// whichever half x is in (the half is a per-lane random bit, so a warp calling a different log per half would run
// both); E by its series for v < 2^-7, so the large-Gumbel end stays exact. Max |G error| over all x: 9.0e-6. The log
// arguments t in [2^-33, 1/2], 1 - t in [1/2, 1 - 2^-7] and E in [2^-33, 23] are all normal.
SGL_DEVICE float gumbel_fast(uint32_t x) {
  const float t = half_uniform(x);
  const bool hi = x >> 31;
  const float l = -logf_ftz(hi ? t : 1.f - t);
  const float e = !hi && t < 0.0078125f ? t * (1.f + t * (0.5f + t * (0.33333334f + t * 0.25f))) : l;
  return -logf_ftz(e);
}

// Base-2 Gumbel for keys compared in log2 units (wgmma): -log2(-log2(1 - v)) = G / ln 2 + log2(ln 2), a constant shift
// that does not change an argmax; gumbel_fast with lg2. Max |error| over all x: 1.3e-5 (log2 units).
SGL_DEVICE float gumbel2_fast(uint32_t x) {
  const float t = half_uniform(x);
  const bool hi = x >> 31;
  const float l = -ptx::lg2_approx_ftz(hi ? t : 1.f - t);
  const float e =
      !hi && t < 0.0078125f ? t * 1.4426950408889634f * (1.f + t * (0.5f + t * (0.33333334f + t * 0.25f))) : l;
  return -ptx::lg2_approx_ftz(e);
}

// Uniform in (0, 1) for an inverse CDF (wgmma, inside an 8-row item): u = ((x >> 9) + 0.5) 2^-23, every value exact
// and in [2^-24, 1 - 2^-24], so u * c < c for every normal fp32 c and a row of zero mass is never drawn
SGL_DEVICE float uniform23(uint32_t x) {
  return (__uint2float_rn(x >> 9) + 0.5f) * 1.1920928955078125e-7f;
}

// Every CTA must be co-resident, since they spin on each other's exchange slots: with the cooperative attribute the
// driver rejects a grid that is not, instead of letting it hang. kMaxSmem is the most any launch of kKernel uses.
template <auto kKernel, std::size_t kMaxSmem, typename... Args>
inline void
launch_cooperative(DLDevice device, uint32_t num_ctas, uint32_t num_threads, std::size_t smem, Args&&... args) {
  using namespace host;
  CHECK_HOST(num_ctas <= runtime::get_sm_count(device.device_id)) << "markov walk: more CTAs than SMs";
  runtime::set_max_dynamic_smem_per_device<kKernel>(device, kMaxSmem);
  LaunchKernel(num_ctas, num_threads, device, smem)  //
      .enable_cooperative()(kKernel, std::forward<Args>(args)...);
}

}  // namespace sglang::dspark_markov_walk
