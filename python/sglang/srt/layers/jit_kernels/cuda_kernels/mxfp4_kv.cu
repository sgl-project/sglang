// MXFP4 KV cache quantize/dequantize kernels for sm86 (Ampere).
//
// Layout (block_size = 32, MXFP4 spec, E8M0 exponent-only scale):
//   data  [S, H, D/2]  uint8   <- two E2M1 values packed per byte (lo = even idx)
//   scale [S, H, D/32] uint8   <- E8M0: bits = exp + 127, value = 2^(bits-127)
//
// Quantize formula (per block of 32 elements along head_dim):
//   exp = ceil(log2(block_max / 6.0))   (clamped, 0 if block_max == 0)
//   x_scaled = x * 2^-exp, rounded to nearest E2M1
//
// Dequant:  value = E2M1(x) * 2^(bits - 127)   (E8M0 via int-as-float bit shift)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MXFP4_HEAD_DIM 128
#define MXFP4_BLOCK 32  // MXFP4 standard scale granularity; buffer shape adapts

// E2M1 positive magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6
__constant__ float c_e2m1_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint8_t float_to_e2m1(float x) {
  // Round-to-nearest-even magnitude index via 7 comparisons, sign bit = bit 3.
  // Midpoint values (exactly on a boundary) round to the even index.
  float ax = fabsf(x);
  uint8_t mag = 0;
  mag += (ax > 0.25f);
  mag += (ax > 0.75f);
  mag += (ax > 1.25f);
  mag += (ax > 1.75f);
  mag += (ax > 2.5f);
  mag += (ax > 3.5f);
  mag += (ax > 5.0f);
  // Half-way cases: round to even magnitude index (LSB 0) -> odd rounds up.
  if (ax == 0.25f && (mag & 1)) mag++;
  else if (ax == 0.75f && (mag & 1)) mag++;
  else if (ax == 1.25f && (mag & 1)) mag++;
  else if (ax == 1.75f && (mag & 1)) mag++;
  else if (ax == 2.5f && (mag & 1)) mag++;
  else if (ax == 3.5f && (mag & 1)) mag++;
  else if (ax == 5.0f && (mag & 1)) mag++;
  uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;
  // Preserve -0.0 as -0.0 for exact round-trip.
  if (x == 0.0f && __float_as_uint(x) & 0x80000000u) sign = 0x8;
  return sign | mag;
}

__device__ __forceinline__ float e2m1_to_float(uint8_t v) {
  float mag = c_e2m1_lut[v & 0x7];
  return (v & 0x8) ? -mag : mag;
}

// Per-thread: 8 contiguous elements of one (token, head) row.
// A warp handles 2 rows (16 threads each); a block32 = 4 consecutive threads.
// Computes the block max, reduces over the 4 threads, returns shared exp.
__device__ __forceinline__ float block_absmax4(const float* vals /* 8 elems */) {
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(vals[i]));
  return m;
}

__device__ __forceinline__ int exp_from_max(float block_max) {
  if (block_max == 0.0f) return -127;
  float e = ceilf(log2f(block_max / 6.0f));
  int ei = (int)e;
  ei = min(ei, 127);
  ei = max(ei, -127);
  return ei;
}

// ============================================================================
// quantize_and_store: bf16 [T, H, 128] -> packed fp4 + scale, scattered to
// pool slots via loc[T].  Grid: ceil(T*H / ROWS_PER_CTA), 256 threads.
// Each CTA: ROWS_PER_CTA (token, head) rows. Each warp: 2 rows.
// ============================================================================
#define ROWS_PER_CTA 16

__global__ void mxfp4_quantize_store_kernel(
    const __nv_bfloat16* __restrict__ cache_kv,  // [T, H, 128] bf16
    int64_t* __restrict__ loc,            // [T] slot per token (int64, no conversion)
    uint8_t* __restrict__ data,           // [S, H, 64]
    uint8_t* __restrict__ scale,          // [S, H, 4]
    int T, int H) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_in_warp = lane >> 4;    // 0..1
  const int sub_lane = lane & 15;       // 0..15
  // block = MXFP4_BLOCK/8 threads' 8 elements each; 16 -> 2 threads, 32 -> 4
  const int blk = sub_lane / (MXFP4_BLOCK / 8);
  const int sub_blk = sub_lane % (MXFP4_BLOCK / 8);

  const int row = blockIdx.x * ROWS_PER_CTA + warp_id * 2 + row_in_warp;
  if (row >= T * H) return;
  const int token = row / H;
  const int head = row % H;
  const int slot = (int)loc[token];

  // Load 8 bf16 (128-bit) starting at element sub_lane*8.
  const int elem0 = sub_lane * 8;
  const __nv_bfloat16* src = cache_kv + (long long)row * MXFP4_HEAD_DIM + elem0;
  float vals[8];
  {
    const float4 v = *reinterpret_cast<const float4*>(src);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; i++) vals[i] = __bfloat162float(h[i]);
  }

  // Block abs-max over the MXFP4_BLOCK/8 threads (all in one warp).
  float m = block_absmax4(vals);
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1, 16));
  if (MXFP4_BLOCK == 32)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2, 16));
  const int exp_ = exp_from_max(m);
  const float inv_scale = exp2f(-(float)exp_);

  // Quantize 8 elements -> 4 packed bytes.
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint8_t lo = float_to_e2m1(vals[i] * inv_scale);
    const uint8_t hi = float_to_e2m1(vals[i + 1] * inv_scale);
    packed |= (uint32_t)((hi << 4) | lo) << (i / 2 * 8);
  }

  // data row = slot*H + head, 64 bytes: this thread writes 4B at sub_lane*4.
  uint8_t* dst = data + ((long long)(slot * H + head)) * (MXFP4_HEAD_DIM / 2) + sub_lane * 4;
  *reinterpret_cast<uint32_t*>(dst) = packed;

  // scale: 1 byte per block32, written by sub_blk == 0.
  if (sub_blk == 0) {
    scale[(long long)(slot * H + head) * (MXFP4_HEAD_DIM / MXFP4_BLOCK) + blk] =
        (uint8_t)(exp_ + 127);
  }
}

// K and V in one launch (halves per-layer launches in eager decode).
__global__ void mxfp4_quantize_store_kv_kernel(
    const __nv_bfloat16* __restrict__ cache_k, const __nv_bfloat16* __restrict__ cache_v,
    int64_t* __restrict__ loc,
    uint8_t* __restrict__ k_data, uint8_t* __restrict__ k_scale,
    uint8_t* __restrict__ v_data, uint8_t* __restrict__ v_scale,
    int T, int H) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_in_warp = lane >> 4;
  const int sub_lane = lane & 15;
  const int blk = sub_lane / (MXFP4_BLOCK / 8);
  const int sub_blk = sub_lane % (MXFP4_BLOCK / 8);

  const int row = blockIdx.x * ROWS_PER_CTA + warp_id * 2 + row_in_warp;
  if (row >= T * H) return;
  const int token = row / H;
  const int head = row % H;
  const int slot = (int)loc[token];

  const int elem0 = sub_lane * 8;
  const long long row_off = (long long)row * MXFP4_HEAD_DIM + elem0;
  // K
  float vals[8];
  {
    const float4 v = *reinterpret_cast<const float4*>(cache_k + row_off);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; i++) vals[i] = __bfloat162float(h[i]);
  }
  float m = block_absmax4(vals);
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1, 16));
  if (MXFP4_BLOCK == 32) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2, 16));
  const int exp_ = exp_from_max(m);
  const float inv_scale = exp2f(-(float)exp_);
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint8_t lo = float_to_e2m1(vals[i] * inv_scale);
    const uint8_t hi = float_to_e2m1(vals[i + 1] * inv_scale);
    packed |= (uint32_t)((hi << 4) | lo) << (i / 2 * 8);
  }
  uint8_t* dst = k_data + ((long long)(slot * H + head)) * (MXFP4_HEAD_DIM / 2) + sub_lane * 4;
  *reinterpret_cast<uint32_t*>(dst) = packed;
  if (sub_blk == 0) {
    k_scale[(long long)(slot * H + head) * (MXFP4_HEAD_DIM / MXFP4_BLOCK) + blk] =
        (uint8_t)(exp_ + 127);
  }
  // V
  {
    const float4 v = *reinterpret_cast<const float4*>(cache_v + row_off);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; i++) vals[i] = __bfloat162float(h[i]);
  }
  m = block_absmax4(vals);
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1, 16));
  if (MXFP4_BLOCK == 32) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2, 16));
  const int exp_v = exp_from_max(m);
  const float inv_scale_v = exp2f(-(float)exp_v);
  packed = 0;
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint8_t lo = float_to_e2m1(vals[i] * inv_scale_v);
    const uint8_t hi = float_to_e2m1(vals[i + 1] * inv_scale_v);
    packed |= (uint32_t)((hi << 4) | lo) << (i / 2 * 8);
  }
  uint8_t* vdst = v_data + ((long long)(slot * H + head)) * (MXFP4_HEAD_DIM / 2) + sub_lane * 4;
  *reinterpret_cast<uint32_t*>(vdst) = packed;
  if (sub_blk == 0) {
    v_scale[(long long)(slot * H + head) * (MXFP4_HEAD_DIM / MXFP4_BLOCK) + blk] =
        (uint8_t)(exp_v + 127);
  }
}

// ============================================================================
// dequantize: packed fp4 [T, H, 64] + scale [T, H, 4] -> bf16 [T, H, 128].
// Same thread mapping as quantize (16 threads per row).
// ============================================================================
__global__ void mxfp4_dequantize_kernel(
    const uint8_t* __restrict__ data,    // [T, H, 64]
    const uint8_t* __restrict__ scale,   // [T, H, 4]
    __nv_bfloat16* __restrict__ out,            // [T, H, 128]
    int T, int H) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_in_warp = lane >> 4;
  const int sub_lane = lane & 15;
  const int blk = sub_lane / (MXFP4_BLOCK / 8);
  const int sub_blk = sub_lane % (MXFP4_BLOCK / 8);

  const int row = blockIdx.x * ROWS_PER_CTA + warp_id * 2 + row_in_warp;
  if (row >= T * H) return;
  const int token = row / H;
  const int head = row % H;

  // Load 4 packed bytes (32-bit) + the shared E8M0 scale byte.
  const uint8_t* src = data + (long long)row * (MXFP4_HEAD_DIM / 2) + sub_lane * 4;
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(src);
  const uint8_t s =
      scale[(long long)row * (MXFP4_HEAD_DIM / MXFP4_BLOCK) + blk];
  const float sscale = __int_as_float((uint32_t)s << 23);  // E8M0 -> fp32

  __nv_bfloat16* dst = out + (long long)row * MXFP4_HEAD_DIM + sub_lane * 8;
  float vals[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint8_t b = (packed >> ((i / 2) * 8)) & 0xFF;
    const uint8_t v = (i & 1) ? (b >> 4) : (b & 0xF);
    vals[i] = e2m1_to_float(v) * sscale;
  }
  // 8 bf16 = 16 bytes = 128-bit store.
  __nv_bfloat16 h8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) h8[i] = __float2bfloat16_rn(vals[i]);
  *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(h8);
}

// ============================================================================
// dequantize_indices: gather rows in `indices` order (flashinfer kv_indices
// layout) into a contiguous bf16 workspace. indices[I] = slot id.
// out [I, H, 128], I = total number of kv tokens this step.
// ============================================================================
__global__ void mxfp4_dequantize_indices_kernel(
    const uint8_t* __restrict__ data,    // [S, H, 64]
    const uint8_t* __restrict__ scale,   // [S, H, 4]
    const int* __restrict__ indices,     // [I]
    __nv_bfloat16* __restrict__ out,            // [I, H, 128]
    int I, int H) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_in_warp = lane >> 4;
  const int sub_lane = lane & 15;
  const int blk = sub_lane / (MXFP4_BLOCK / 8);

  const int out_row = blockIdx.x * ROWS_PER_CTA + warp_id * 2 + row_in_warp;
  if (out_row >= I * H) return;
  const int token = out_row / H;
  const int head = out_row % H;
  const int slot = indices[token];

  const uint8_t* src = data + ((long long)(slot * H + head)) * (MXFP4_HEAD_DIM / 2) + sub_lane * 4;
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(src);
  const uint8_t s =
      scale[((long long)(slot * H + head)) * (MXFP4_HEAD_DIM / MXFP4_BLOCK) + blk];
  const float sscale = __int_as_float((uint32_t)s << 23);

  __nv_bfloat16* dst = out + (long long)out_row * MXFP4_HEAD_DIM + sub_lane * 8;
  float vals[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint8_t b = (packed >> ((i / 2) * 8)) & 0xFF;
    const uint8_t v = (i & 1) ? (b >> 4) : (b & 0xF);
    vals[i] = e2m1_to_float(v) * sscale;
  }
  __nv_bfloat16 h8[8];
#pragma unroll
  for (int i = 0; i < 8; i++) h8[i] = __float2bfloat16_rn(vals[i]);
  *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(h8);
}
