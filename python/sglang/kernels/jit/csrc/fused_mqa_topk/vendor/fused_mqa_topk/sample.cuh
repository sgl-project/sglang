// Gate sample of the persistent kernel: each math warp histograms its row's
// sampled scores relative to a per-row reference (the row's maximum over its
// first sample tile), in bins of fp16(x - ref) (1/64 octave of the distance to
// the reference: 4096 bins as 16-bit counters, two per word, 2048 words padded
// to 33 per 32), and finish_sample_histogram picks the gate from them. Absolute
// fp16 bins are ~2 wide at |x| ~ 100, wider than the whole top-K tail of
// layers whose scores sit at a large common offset (GLM-5.2 layer 1: ~+100,
// ranks 1024..4096 within 1.6-2.6), which put the gate far too low.
#pragma once
#include <cutlass/arch/barrier.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
constexpr unsigned SampleQ = 8;
struct alignas(16) SampleShared {
  unsigned hist[SampleQ][2112];
  // Reference of each row: ordered fp32 code of its maximum valid sample in
  // its first sample tile holding any (0: none yet / no sample, reference 0).
  unsigned ref[SampleQ];
  // The samples of a reference tile (NaN: not valid) until its maximum is known.
  float stage[SampleQ][256];
};
__device__ __forceinline__ void sample_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(256, 2).sync();
  asm volatile("" ::: "memory");
}
// Ordered code of fp16 round-down(d) >> 4 (bins ascend with d).
__device__ __forceinline__ unsigned sample_bin(float d) {
  unsigned h = __half_as_ushort(__float2half_rd(d));
  return ((h & 0x8000u) ? ((~h) & 65535u) : (h ^ 0x8000u)) >> 4;
}
// Lower edge of bin b.
__device__ __forceinline__ float sample_floor(unsigned b) {
  unsigned o = b << 4, h = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(h));
}
// Counts one sample in bin b (16-bit half b & 1 of word b >> 1; a row holds at
// most 255 sample tiles = 65280 samples, see launch_config).
__device__ __forceinline__ void sample_add(unsigned* hist, unsigned b) {
  const unsigned w = b >> 1;
  atomicAdd(hist + w + w / 32, 1u << ((b & 1u) * 16));
}
__device__ __forceinline__ unsigned sample_ref_code(float x) {
  const unsigned u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ float sample_ref_value(unsigned code) {
  return code == 0u ? 0.0f
                    : __uint_as_float((code & 0x80000000u) ? (code ^ 0x80000000u)
                                                           : ~code);
}
