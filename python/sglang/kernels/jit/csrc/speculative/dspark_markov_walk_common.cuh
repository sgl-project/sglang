// SPDX-License-Identifier: Apache-2.0
// Shared by the int8 DSpark markov-walk kernels: dspark_markov_walk_single.cuh (bs = 1),
// dspark_markov_walk_small_batch.cuh (bs 2..4), dspark_markov_walk_wgmma.cuh (bs 5..64, and every bs for big
// vocabularies).  Inline PTX, device helpers, then the cooperative launch.  The weight layouts are built by
// python/sglang/kernels/ops/speculative/dspark/markov_walk.py.
#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <utility>

// The kernels' inline PTX, added to the `sglang::device::ptx` namespace of sgl_kernel/mbarrier.cuh.
namespace sglang::device::ptx {

// ---- global memory

SGL_DEVICE void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to_shared(smem)), "l"(gmem));
}
SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
template <int N>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
// the mbarrier at `mbar` (a shared address) counts this thread's earlier cp.async as one arrival when they land
SGL_DEVICE void cp_async_mbar_arrive_noinc(unsigned mbar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(mbar) : "memory");
}

SGL_DEVICE unsigned long long ld_relaxed(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
SGL_DEVICE void st_relaxed(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// Cross-CTA exchange (all three kernels).  Per (step, request) a 16-B aligned pair {key, count}: every CTA posts
// atomicMax(key, its best) and then red.release.add(count, 1) (the release orders the CTA's max before its arrival),
// and polls the pair with ld_relaxed_v2 until count == gridDim.x; that load's key is the global max.
// HARDWARE ASSUMPTION: a 16-B aligned ld.relaxed.gpu.v2.u64 observes both words at one point in L2 (single-copy
// atomic), so the load that sees the final count also sees every max ordered before it.  PTX does not promise this
// (vector accesses are modelled as unordered scalar accesses, and a relaxed load does not synchronize with the
// release); it holds on H100 (sm_90), where an aligned 16-B load is served by one L2 sector access.
SGL_DEVICE ulonglong2 ld_relaxed_v2(const unsigned long long* p) {
  ulonglong2 v;
  asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  return v;
}
// release add without a return value (a returning atom would make the following poll wait for its round trip)
SGL_DEVICE void red_add_release(unsigned long long* p, unsigned long long v) {
  asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// W2 is read once per round: L2 evict_first, so it evicts its own lines rather than the freshly written base logits
SGL_DEVICE unsigned long long evict_first_policy() {
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
SGL_DEVICE uint4 ldcg_hint(const uint4* p, unsigned long long pol) {
  uint4 v;
  asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
SGL_DEVICE void prefetch_l2(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}
SGL_DEVICE void prefetch_l2_evict_last(const void* p) {
  asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p));
}

// ---- mbarriers, TMA bulk copies (shared addresses from to_shared)

SGL_DEVICE void mbar_init(unsigned addr, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count) : "memory");
}
SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
SGL_DEVICE void mbar_expect_tx(unsigned addr, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes) : "memory");
}
// CTA-scope acquire: the waits order this CTA's own TMA / cp.async writes into its SMEM (the launch has no cluster
// attribute, so .cluster would mean the same 1-CTA cluster -- and its acquire invalidates L1, CCTL.IVALL per wait)
SGL_DEVICE void mbar_wait_cta(unsigned addr, unsigned parity) {
  unsigned ok = 0;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(ok)
        : "r"(addr), "r"(parity)
        : "memory");
  } while (!ok);
}
// global -> this CTA's SMEM (PTX names the destination state space .shared::cluster; with no cluster launch that is
// this CTA's SMEM), completion counted on mbar, L2 cache policy pol
SGL_DEVICE void bulk_g2s_hint(unsigned dst, const void* src, unsigned bytes, unsigned mbar, unsigned long long pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
          "r"(dst),
      "l"(src),
      "r"(bytes),
      "r"(mbar),
      "l"(pol)
      : "memory");
}

// ---- registers

// %tid.x through volatile asm: unlike threadIdx.x, every read is a fresh S2R that the compiler cannot keep live
SGL_DEVICE unsigned tid_x() {
  unsigned t;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
  return t;
}

// ---- bf16x2, int8 mma, MUFU

// An 8-row run's bf16 logits as 4 packed pairs p0..p3 (pair i = rows 2i (low half), 2i + 1 (high half)).
// bf16x8_max: their max in both halves.  bf16x8_first_eq: the first row whose value equals m (m in both halves; ties ->
// smallest row, as torch.argmax).  HMNMX2 / HSETP2 on the packed values, no bf16 -> fp32 unpacking.
SGL_DEVICE unsigned bf16x8_max(unsigned p0, unsigned p1, unsigned p2, unsigned p3) {
  unsigned m;
  asm("{\n\t.reg .b32 a, b;\n\t"
      "max.bf16x2 a, %1, %2;\n\tmax.bf16x2 b, %3, %4;\n\tmax.bf16x2 a, a, b;\n\t"
      "prmt.b32 b, a, a, 0x1032;\n\tmax.bf16x2 %0, a, b;\n\t}"
      : "=r"(m)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3));
  return m;
}
SGL_DEVICE int bf16x8_first_eq(unsigned p0, unsigned p1, unsigned p2, unsigned p3, unsigned m) {
  int qi;
  asm("{\n\t.reg .pred e0, e1, e2, e3, e4, e5, e6, e7;\n\t"
      "setp.eq.bf16x2 e0|e1, %1, %5;\n\tsetp.eq.bf16x2 e2|e3, %2, %5;\n\t"
      "setp.eq.bf16x2 e4|e5, %3, %5;\n\tsetp.eq.bf16x2 e6|e7, %4, %5;\n\t"
      "mov.b32 %0, 7;\n\t@e6 mov.b32 %0, 6;\n\t@e5 mov.b32 %0, 5;\n\t@e4 mov.b32 %0, 4;\n\t"
      "@e3 mov.b32 %0, 3;\n\t@e2 mov.b32 %0, 2;\n\t@e1 mov.b32 %0, 1;\n\t@e0 mov.b32 %0, 0;\n\t}"
      : "=r"(qi)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(m));
  return qi;
}

// single / small_batch tensor-core GEMV: one m16n8k32 s8.s8.s32 mma, A = a 16-row W2 tile chunk, B = two W1q planes
SGL_DEVICE void mma_s8(int (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// MUFU lg2 / ex2 without the denormal fix-up (FSETP / FMUL / FSEL / FADD around every MUFU.LG2): for a normal
// argument .ftz returns the same bits, and every caller passes normal numbers.
SGL_DEVICE float lg2_ftz(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
SGL_DEVICE float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace sglang::device::ptx

namespace sglang::dspark_markov_walk {

namespace ptx = device::ptx;
using u64 = unsigned long long;

// Draft steps per launch; the Philox counters pack the step into 8 bits (step | request << 8).
inline constexpr int kMaxSteps = 256;
// single / small_batch CTA and W2 layout: 256 threads; a 16-row W2 tile = 8 k32-chunks x 32 lanes of 16-B mma A
// fragments
inline constexpr int kThreads = 256;
inline constexpr int kWarps = kThreads / 32;
inline constexpr int kTileVecs = 256;  // uint4 per tile
// single / small_batch W1q row: q_hi[256] | q_lo[256] | s_hi f32 | s_lo f32 | 8 B pad = 528 B
inline constexpr int kURowWords = 132;
inline constexpr int kURowBytes = kURowWords * 4;

// ---- per-request inputs (read once per launch)

// anchor token -> a row in [0, valid_rows): an out-of-range anchor would make the first W1 gather an illegal address
SGL_DEVICE unsigned clamp_row(int64_t a, int valid_rows) {
  return static_cast<unsigned>(a < 0 ? 0 : a >= valid_rows ? valid_rows - 1 : a);
}
// 1 / T of a sampling request, T clamped to [1e-5, 1e4] (T = +inf or a denormal T stays finite, the same way in all
// three kernels); 0 = greedy (T <= 0 or NaN)
SGL_DEVICE float inv_temperature(float t) {
  return t > 0.f ? 1.f / fminf(fmaxf(t, 1e-5f), 1e4f) : 0.f;
}

// ---- argmax keys: (order-preserving fp32 << 32) | (0xFFFFFFFF - row): max = best score, ties -> smallest row

SGL_DEVICE u64 pack_key(float v, unsigned row) {
  unsigned b = __float_as_uint(v);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return (static_cast<u64>(b) << 32) | static_cast<u64>(0xFFFFFFFFu - row);
}
SGL_DEVICE unsigned key_row(u64 key) {
  return 0xFFFFFFFFu - static_cast<unsigned>(key & 0xFFFFFFFFull);
}

// ---- sampling noise

// Philox4x32-10 (Salmon et al. 2011)
SGL_DEVICE uint4 philox(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const unsigned hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}
// Uniforms from a 32-bit Philox word x.  The Gumbel transforms use v = (x + 0.5) 2^-32 in (0, 1) at full 32-bit
// resolution and need E = -log(1 - v) ~ Exp(1).  For x >= 2^31 (v >= 1/2) they take 1 - v = (~x + 0.5) 2^-32
// directly: 1 - v formed in fp32 drops v's low bits, and v itself rounds to 1.0f for x >= 2^32 - 128, which made
// E = inf and G = -inf.  half_uniform returns that operand, in [2^-33, 1/2]; E is then finite and > 0 for every x,
// and every Gumbel value lies in [-log(33 ln 2), 33 ln 2] (natural-log units), checked over all 2^32 x.
inline constexpr float kTwoM32 = 2.3283064365386963e-10f;  // 2^-32
SGL_DEVICE float half_uniform(unsigned x) {                // x < 2^31: v;  x >= 2^31: 1 - v
  return (__uint2float_rn(x >> 31 ? ~x : x) + 0.5f) * kTwoM32;
}
// Each transform below evaluates ONE log for E whichever half x is in (the half is a per-lane random bit, so a warp
// that called a different log per half would execute both).
// Gumbel(0,1) = -log(E) with accurate logs (single).  x >= 2^31: E = -logf(t).  x < 2^31: E = -log1p(-t) from the same
// logf: u = fl(1 - t), r = (1 - u) - t = (1 - t) - u exactly (Sterbenz, twice), log(1 - t) = log(u) + log1p(r / u)
// with |r / u| <= 2^-24, and r / u = r (1 + t + t^2) to well below an ulp of E -- no division, no branch; the
// large-Gumbel end (t -> 0: u = 1, E = t (1 + t + t^2)) stays exact.  Max |G error| over all x: 1.0e-6 (half an ulp of
// G near its maximum).
SGL_DEVICE float gumbel(unsigned x) {
  const float t = half_uniform(x);
  const bool hi = x >> 31;
  const float u = 1.f - t;
  const float l = logf(hi ? t : u);
  const float r = (1.f - u) - t;
  const float e = hi ? -l : -fmaf(r, 1.f + t * (1.f + t), l);
  return -logf(e);
}
// natural log as __logf computes it (lg2.approx times ln 2 rounded to fp32), without the denormal fix-up
SGL_DEVICE float logf_ftz(float x) {
  return ptx::lg2_ftz(x) * 0.693147182464599609375f;
}
// Gumbel(0,1) with MUFU logs (small_batch: the epilogue's cost at T > 0 is mostly these logs).  E by its series for v <
// 2^-7 (the large-Gumbel end stays exact).  Max |G error| over all x: 9.0e-6.  Log arguments: t in [2^-33, 1/2], 1 - t
// in [1/2, 1 - 2^-7], E in [2^-33, 23]: all normal.
SGL_DEVICE float gumbel_fast(unsigned x) {
  const float t = half_uniform(x);
  const bool hi = x >> 31;
  const float l = -logf_ftz(hi ? t : 1.f - t);
  const float e = !hi && t < 0.0078125f ? t * (1.f + t * (0.5f + t * (0.33333334f + t * 0.25f))) : l;
  return -logf_ftz(e);
}
// Base-2 Gumbel for keys compared in log2 units (wgmma): -log2(-log2(1 - v)) = G / ln 2 + log2(ln 2), i.e. G / ln 2 up
// to a constant shift that does not change an argmax; computed as gumbel_fast with lg2.  Max |error| over all x: 1.3e-5
// (log2 units).
SGL_DEVICE float gumbel2_fast(unsigned x) {
  const float t = half_uniform(x);
  const bool hi = x >> 31;
  const float l = -ptx::lg2_ftz(hi ? t : 1.f - t);
  const float e =
      !hi && t < 0.0078125f ? t * 1.4426950408889634f * (1.f + t * (0.5f + t * (0.33333334f + t * 0.25f))) : l;
  return -ptx::lg2_ftz(e);
}
// Uniform in (0, 1) for an inverse CDF (wgmma, inside an 8-row item): u = ((x >> 9) + 0.5) 2^-23, every value exact,
// u in [2^-24, 1 - 2^-24].  So u * c < c for every normal fp32 c (the product rounds to at most c - ulp), and a row of
// zero mass is never drawn.  (24 bits, ((x >> 8) + 0.5) 2^-24, rounds to 1.0f at x >> 8 = 2^24 - 1.)
SGL_DEVICE float uniform23(unsigned x) {
  return (__uint2float_rn(x >> 9) + 0.5f) * 1.1920928955078125e-7f;
}

// ---- host

/**
 * \brief Cooperative launch on `device`'s current stream.
 *
 * Every CTA must be co-resident -- the kernels spin on each other's exchange slots -- which the cooperative
 * attribute enforces: the launch fails instead of hanging.
 *
 * \param smem     Dynamic SMEM bytes of this launch.
 * \param smem_max The most any launch of `kernel` uses, set as its max-dynamic-SMEM attribute (never lowered: a CUDA
 *                 graph captured earlier with a larger smem stays valid).
 */
template <typename... KArgs, typename... Args>
inline void launch_cooperative(
    void (*kernel)(KArgs...),
    DLDevice device,
    uint32_t grid,
    uint32_t threads,
    std::size_t smem,
    std::size_t smem_max,
    Args&&... args) {
  using namespace host;
  [[maybe_unused]] const tvm::ffi::CUDADeviceGuard guard(device.device_id);
  const uint32_t num_sms = runtime::get_sm_count(device.device_id);
  CHECK_HOST(grid <= num_sms) << "markov walk: grid " << grid << " CTAs > " << num_sms
                              << " SMs (one co-resident CTA per SM)";
  CHECK_HOST(smem <= smem_max) << "markov walk: dynamic SMEM " << smem << " > " << smem_max;
  cudaFuncAttributes attrs;
  CHECK_CUDA(cudaFuncGetAttributes(&attrs, kernel));
  const std::size_t max_smem = runtime::get_max_smem_per_block(device.device_id);
  CHECK_HOST(smem_max + attrs.sharedSizeBytes <= max_smem)
      << "markov walk: SMEM per CTA " << smem_max << " + " << attrs.sharedSizeBytes << " static > " << max_smem << " B";
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_max)));
  LaunchKernel(grid, threads, device, smem).enable_cooperative()(kernel, std::forward<Args>(args)...);
}

}  // namespace sglang::dspark_markov_walk
