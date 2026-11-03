// Fused NVFP4 SwiGLU MoE for SM120 (BF16 in/out, EP = TP = 1), in two launches:
//   prologue : routing -> per-expert token slots grouped in units of up to 32 tokens; NVFP4
//              quantization of every input row (once per token); dependency counters reset.
//   main     : persistent CTAs pull tiles from one atomic queue.
//              GEMM1 tile (unit, 64 channels): FP4 GEMM over H for 64 up + 64 gate rows, alpha,
//                BF16 round, SwiGLU, NVFP4 quantization of the intermediate.
//              GEMM2 tile (unit, 128 output columns): waits for the unit's GEMM1 tiles, FP4 GEMM
//                over I, alpha, BF16 round. The last contribution to a token's column block runs
//                that token's weighted top-k reduction in k order.
//
// The arithmetic is the one FlashInfer's CUTLASS NVFP4 MoE performs with the unfused finalize
// (w13 = [up, gate], FP4 quantization with the fast-math reciprocal, mma.sync m16n8k64 block-scaled
// GEMMs accumulated in ascending K with no split-K, D = bf16(alpha * acc), finalize in FP32 over
// k = 0..TopK-1). Build with --use_fast_math so CUTLASS's SiLu compiles to the same instructions.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sglang {
namespace nvfp4_swiglu_moe {

constexpr int kRows = 128;            // weight rows per tile
constexpr int kSeg = kRows / 2;       // GEMM1 tile = kSeg up rows + kSeg gate rows
constexpr int kWarpRows = kRows / 4;  // weight rows per warp
constexpr int kUnit = 32;             // tokens per unit
constexpr int kThreads = 128;
constexpr int kStages = 2;         // GEMM1 pipeline stages (GEMM2 uses twice as many half-size ones)
constexpr int kStage1 = 128;       // bytes of every row per GEMM1 stage
constexpr int kStage2 = 64;        // bytes of every row per GEMM2 stage
constexpr int kOutLd = kRows + 8;  // bf16 elements per row of the epilogue tile
constexpr int kPrologueThreads = 1024;
constexpr int kStagedIds = 5120;  // routing ids staged in shared memory when T * TopK fits

template <int KS>
constexpr int stage_bytes() {
  return kRows * KS + kUnit * KS + (KS / 32) * kRows * 4 + kUnit * (KS / 32) * 4;
}
constexpr int kSlot = stage_bytes<kStage1>();
static_assert(stage_bytes<kStage2>() <= kSlot / 2, "GEMM2 stage must fit half a GEMM1 stage");
constexpr int kMainSmem = kStages * kSlot;
static_assert(kUnit * kOutLd * 2 <= kMainSmem, "epilogue tile must fit the stage buffers");

template <int E, int TopK, int H, int I>
struct Shape {
  static_assert(E % 32 == 0 && E < (1 << 16), "expert count must be a multiple of 32");
  static_assert(H % 256 == 0 && I % 128 == 0, "hidden must be a multiple of 256, intermediate of 128");
  static_assert(TopK >= 1, "top-k must be positive");
  static_assert(4 * (4 * E + 1) + 4 * kStagedIds <= 48 * 1024, "routing shared memory exceeds 48 KB");
  static constexpr int K1B = H / 2;   // bytes per x / w13 row
  static constexpr int K2B = I / 2;   // bytes per intermediate / w2 row
  static constexpr int KP1 = H / 16;  // scale factors per x / w13 row
  static constexpr int KP2 = I / 16;  // scale factors per intermediate / w2 row
  static constexpr int NT1 = I / kSeg;
  static constexpr int NT2 = H / kRows;
  static constexpr int NS2 = K2B / kStage2;
  // sum over experts of ceil(count / kUnit)
  static constexpr int64_t max_units(int64_t T) {
    return (T * TopK < E ? T * TopK : E) + (T * TopK + kUnit - 1) / kUnit;
  }
  static constexpr int64_t align(int64_t n) {
    return (n + 255) / 256 * 256;
  }
  static constexpr int64_t u8_bytes(int64_t T) {
    return align(T * K1B) + align(T * KP1) + align(max_units(T) * kUnit * K2B) + align(max_units(T) * NS2 * kUnit * 8);
  }
  static constexpr int64_t i32_count(int64_t T) {
    return 2 * T * TopK + 2 * max_units(T) + 2 * E + T * NT2 + T + 2;
  }
};

struct Params {
  const __nv_bfloat16* x;  // [T, H]
  const int32_t* ids;      // [T, TopK]
  const float* weights;    // [T, TopK]
  const uint8_t* w13;      // [E, 2I, H/2]
  const uint8_t* w13_sf;   // [E, 2I, H/16], 128x4 swizzled
  const uint8_t* w2;       // [E, H, I/2]
  const uint8_t* w2_sf;    // [E, H, I/16], 128x4 swizzled
  const float* alpha1;     // [E]
  const float* alpha2;     // [E]
  const float* gs1;        // [1] FC1 input global scale
  const float* gs2;        // [1] FC2 input global scale
  __nv_bfloat16* out;      // [T, H]
  int T;
  uint8_t* xq;        // [T, H/2]
  uint8_t* xsf;       // [T, H/16]
  uint8_t* aq;        // [units, kUnit, I/2]
  uint8_t* asf;       // [units, NS2, kUnit, 8]
  __nv_bfloat16* y2;  // [T * TopK, H]
  int* slot_tok;      // [T * TopK]
  int* tk_slot;       // [T * TopK], -1 for skipped experts
  int* units;         // [units] expert | group << 16
  int* counts;        // [E]
  int* offsets;       // [E]
  int* g1done;        // [units]
  int* fcnt;          // [T, NT2]
  int* nvalid;        // [T]
  int* n_units;       // [1]
  int* queue;         // [1]
};

struct PackedVec {
  __nv_bfloat162 elts[4];
};

__device__ __forceinline__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1(const float2 (&v)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(v[0].x), "f"(v[0].y), "f"(v[1].x), "f"(v[1].y), "f"(v[2].x), "f"(v[2].y), "f"(v[3].x), "f"(v[3].y));
  return val;
}

// FlashInfer's cvt_warp_fp16_to_fp4 for 8 BF16 values per thread and 16-element blocks (fast-math
// variant): the xor-1 lane holds the other half of the block, so all 32 lanes must call it.
__device__ __forceinline__ uint32_t quantize_bf16x8(const PackedVec& vec, float global_scale, uint8_t* sf_out) {
  __nv_bfloat162 local_max = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; i++)
    local_max = __hmax2(local_max, __habs2(vec.elts[i]));
  local_max = __hmax2(__shfl_xor_sync(uint32_t(-1), local_max, 1), local_max);
  const float vec_max = float(__hmax(local_max.x, local_max.y));
  float sf = global_scale * (vec_max * rcp_approx_ftz(6.0f));
  const __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sf);
  sf = static_cast<float>(sf8);
  const float scale = vec_max != 0 ? rcp_approx_ftz(sf * rcp_approx_ftz(global_scale)) : 0.0f;
  if (sf_out) *sf_out = sf8.__x;
  float2 v[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    v[i] = __bfloat1622float2(vec.elts[i]);
    v[i].x *= scale;
    v[i].y *= scale;
  }
  return fp32x8_to_e2m1(v);
}

template <class U, class Tin>
__device__ __forceinline__ U array_convert(const Tin& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename Tin::Element, U::kElements> converter;
  return converter(input);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void
mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// Stage rows are KS bytes; 16-byte piece p of row r lives at piece p ^ swz(r) so the 8 rows x 4
// words of an MMA fragment load hit 32 distinct banks.
template <int KS>
__device__ __forceinline__ int swz(int r) {
  return KS == 128 ? (r & 7) : ((r >> 1) & 3);
}
template <int KS>
__device__ __forceinline__ uint32_t lds_frag(const uint8_t* buf, int r, int c, int h, int tig) {
  const int p = (2 * c + h) ^ swz<KS>(r);
  return *reinterpret_cast<const uint32_t*>(buf + r * KS + p * 16 + tig * 4);
}

template <int E, int TopK, int H, int I>
__global__ void __launch_bounds__(kPrologueThreads) prologue(Params p) {
  using S = Shape<E, TopK, H, I>;
  const int tid = threadIdx.x;
  const int T = p.T;
  for (int i = blockIdx.x * kPrologueThreads + tid; i < T * S::NT2; i += gridDim.x * kPrologueThreads)
    p.fcnt[i] = 0;
  // CTAs 1.. quantize x while CTA 0 routes. A unit is 8 elements; lanes (2i, 2i + 1) share a block.
  if (blockIdx.x != 0) {
    const float gs = *p.gs1;
    const int n = T * (H / 8);  // multiple of 32, so whole warps run the loop together
    for (int u = (blockIdx.x - 1) * kPrologueThreads + tid; u - (tid & 31) < n;
         u += (gridDim.x - 1) * kPrologueThreads) {
      const int t = u / (H / 8), j = u % (H / 8);
      const PackedVec v = *reinterpret_cast<const PackedVec*>(p.x + static_cast<size_t>(t) * H + j * 8);
      uint8_t* sf_out = (j & 1) ? nullptr : p.xsf + static_cast<size_t>(t) * S::KP1 + (j >> 1);
      reinterpret_cast<uint32_t*>(p.xq)[static_cast<size_t>(t) * (H / 8) + j] = quantize_bf16x8(v, gs, sf_out);
    }
    return;
  }
  // Global loads interleaved with shared-memory atomics serialize, so stage the ids first.
  __shared__ int cnt[E], off[E], cur[E], uoff[E + 1];
  __shared__ __align__(16) int s_ids[kStagedIds];
  const bool staged = T * TopK <= kStagedIds;
  for (int e = tid; e < E; e += kPrologueThreads)
    cnt[e] = 0, cur[e] = 0;
  if (staged) {
    for (int i = tid; i < T * TopK; i += kPrologueThreads)
      s_ids[i] = p.ids[i];
  }
  __syncthreads();
  const int* ids = staged ? s_ids : p.ids;
  for (int i = tid; i < T * TopK; i += kPrologueThreads) {
    const int e = ids[i];
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid < 32) {  // warp scan over the experts, E / 32 per lane
    constexpr int kPerLane = E / 32;
    int s = 0, su = 0;
    int lc[kPerLane], lu[kPerLane];
#pragma unroll
    for (int k = 0; k < kPerLane; k++) {
      const int c = cnt[tid * kPerLane + k];
      lc[k] = s, lu[k] = su;
      s += c, su += (c + kUnit - 1) / kUnit;
    }
    int is = s, isu = su;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffff, is, d), b = __shfl_up_sync(0xffffffff, isu, d);
      if (tid >= d) is += a, isu += b;
    }
#pragma unroll
    for (int k = 0; k < kPerLane; k++)
      off[tid * kPerLane + k] = is - s + lc[k], uoff[tid * kPerLane + k] = isu - su + lu[k];
    if (tid == 31) uoff[E] = isu, *p.n_units = isu, *p.queue = 0;
  }
  __syncthreads();
  for (int i = tid; i < T * TopK; i += kPrologueThreads) {
    const int e = ids[i];
    int slot = -1;
    if (e >= 0 && e < E) {
      slot = off[e] + atomicAdd(&cur[e], 1);
      p.slot_tok[slot] = i / TopK;
    }
    p.tk_slot[i] = slot;
  }
  for (int e = tid; e < E; e += kPrologueThreads) {
    p.counts[e] = cnt[e];
    p.offsets[e] = off[e];
    for (int u = uoff[e]; u < uoff[e + 1]; u++) {
      p.units[u] = e | ((u - uoff[e]) << 16);
      p.g1done[u] = 0;
    }
  }
  // A token whose experts are all skipped is never finalized; its output is zero.
  for (int t = tid; t < T; t += kPrologueThreads) {
    int n = 0;
    for (int k = 0; k < TopK; k++) {
      const int e = ids[t * TopK + k];
      n += (e >= 0 && e < E);
    }
    p.nvalid[t] = n;
    if (n == 0) {
      for (int c = 0; c < H; c += 8)
        *reinterpret_cast<uint4*>(p.out + static_cast<size_t>(t) * H + c) = make_uint4(0, 0, 0, 0);
    }
  }
}

struct TileCtx {
  int u, mcount, seg0, seg1;  // unit, valid tokens, first rows of the two 64-row weight segments
  const uint8_t* wbase;       // expert weight matrix
  const uint8_t* sfbase;      // expert swizzled scale factors
  const int* s_tok;           // smem: token of each unit row (clamped to a valid row)
};

// K-pipelined FP4 GEMM of one tile: 128 weight rows (MMA A operand) x kUnit tokens (B operand),
// full K, accumulators in registers. One thread owns each output element; k64 chunks accumulate
// in ascending order.
template <int E, int TopK, int H, int I, int NSTG, int KS, int SLOTB, bool G1>
__device__ __forceinline__ void
mainloop(float (&acc)[kWarpRows / 16][kUnit / 8][4], uint8_t* smem, const Params& p, const TileCtx& tc) {
  using S = Shape<E, TopK, H, I>;
  constexpr int DA = kWarpRows / 16, DB = kUnit / 8;
  constexpr int CPS = KS / 32;  // k64 chunks per stage
  constexpr int ROWB = G1 ? S::K1B : S::K2B;
  constexpr int KP4 = (G1 ? S::KP1 : S::KP2) / 4;
  constexpr int NST = ROWB / KS;
  constexpr int OFF_A = kRows * KS, OFF_WSF = OFF_A + kUnit * KS, OFF_ASF = OFF_WSF + CPS * kRows * 4;
  static_assert(OFF_ASF + kUnit * CPS * 4 <= SLOTB, "stage overflow");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
  const int mcount = tc.mcount;

  auto issue = [&](int s) {
    uint8_t* st = smem + (s % NSTG) * SLOTB;
    uint8_t *sw = st, *sa = st + OFF_A, *swsf = st + OFF_WSF, *sasf = st + OFF_ASF;
    constexpr int PPR = KS / 16;  // 16-byte pieces per row
#pragma unroll
    for (int i = 0; i < kRows * PPR / kThreads; i++) {
      const int idx = tid + i * kThreads;
      const int r = idx / PPR, pc = idx % PPR;
      const int grow = (r < kSeg ? tc.seg0 : tc.seg1) + (r & (kSeg - 1));
      cp_async16(
          sw + r * KS + ((pc ^ swz<KS>(r)) * 16), tc.wbase + static_cast<size_t>(grow) * ROWB + s * KS + pc * 16);
    }
#pragma unroll
    for (int i = 0; i < kUnit * PPR / kThreads; i++) {
      const int idx = tid + i * kThreads;
      const int r = idx / PPR, pc = idx % PPR;
      const uint8_t* src = G1 ? p.xq + static_cast<size_t>(tc.s_tok[r]) * S::K1B + s * KS + pc * 16
                              : p.aq + (static_cast<size_t>(tc.u) * kUnit + r) * S::K2B + s * KS + pc * 16;
      cp_async16(sa + r * KS + ((pc ^ swz<KS>(r)) * 16), src);
    }
    // Weight SF: per (segment, i in 0..31, chunk c) 8 bytes = SF(row seg + i) and SF(row seg + 32 + i).
#pragma unroll
    for (int j = 0; j < 64 * CPS / kThreads; j++) {
      const int idx = tid + j * kThreads;
      const int sgi = idx / (32 * CPS), i = (idx / CPS) % 32, c = idx % CPS;
      const int n = (sgi ? tc.seg1 : tc.seg0) + i;
      const uint8_t* src =
          tc.sfbase + (static_cast<size_t>(n >> 7) * KP4 + (CPS * s + c)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4;
      cp_async8(swsf + ((c * 2 + sgi) * 32 + i) * 8, src);
    }
    if constexpr (G1) {  // xsf [T][H/16]: CPS * 4 = 16 bytes per row per stage
      if (tid < kUnit) cp_async16(sasf + tid * 16, p.xsf + static_cast<size_t>(tc.s_tok[tid]) * S::KP1 + s * 16);
    } else {  // asf [unit][NS2][kUnit][8]: 256 contiguous bytes per stage
      if (tid < kUnit / 2)
        cp_async16(sasf + tid * 16, p.asf + (static_cast<size_t>(tc.u) * S::NS2 + s) * (kUnit * 8) + tid * 16);
    }
  };
  auto wsf = [&](const uint8_t* swsf, int R, int c) -> uint32_t {
    const int sgi = R >> 6, h = (R >> 5) & 1, i = R & 31;
    return *reinterpret_cast<const uint32_t*>(swsf + ((c * 2 + sgi) * 32 + i) * 8 + h * 4);
  };

#pragma unroll
  for (int a = 0; a < DA; a++)
#pragma unroll
    for (int b = 0; b < DB; b++)
#pragma unroll
      for (int c = 0; c < 4; c++)
        acc[a][b][c] = 0.f;

#pragma unroll
  for (int s = 0; s < NSTG - 1; s++) {
    if (s < NST) issue(s);
    cp_commit();
  }
  for (int s = 0; s < NST; s++) {
    cp_wait<NSTG - 2>();
    __syncthreads();
    if (s + NSTG - 1 < NST) issue(s + NSTG - 1);
    cp_commit();
    const uint8_t* st = smem + (s % NSTG) * SLOTB;
    const uint8_t *sw = st, *sa = st + OFF_A, *swsf = st + OFF_WSF, *sasf = st + OFF_ASF;
#pragma unroll
    for (int c = 0; c < CPS; c++) {
      uint32_t af[DA][4], sfa[DA];
#pragma unroll
      for (int mt = 0; mt < DA; mt++) {
        const int R = warp * kWarpRows + mt * 16 + g;
        af[mt][0] = lds_frag<KS>(sw, R, c, 0, tig);
        af[mt][1] = lds_frag<KS>(sw, R + 8, c, 0, tig);
        af[mt][2] = lds_frag<KS>(sw, R, c, 1, tig);
        af[mt][3] = lds_frag<KS>(sw, R + 8, c, 1, tig);
        sfa[mt] = wsf(swsf, R + 8 * (lane & 1), c);
      }
#pragma unroll
      for (int nt = 0; nt < DB; nt++) {
        if (nt * 8 < mcount) {
          const int t0 = nt * 8 + g;
          const uint32_t b0 = lds_frag<KS>(sa, t0, c, 0, tig), b1 = lds_frag<KS>(sa, t0, c, 1, tig);
          const uint32_t sfb = *reinterpret_cast<const uint32_t*>(sasf + t0 * (CPS * 4) + c * 4);
#pragma unroll
          for (int mt = 0; mt < DA; mt++)
            mma_fp4(acc[mt][nt], af[mt], b0, b1, sfa[mt], sfb);
        }
      }
    }
  }
  cp_wait<0>();
}

template <int E, int TopK, int H, int I>
__device__ __forceinline__ void
swiglu_quant_epilogue(const Params& p, const __nv_bfloat16* sm_out, int u, int sub, float gs) {
  using S = Shape<E, TopK, H, I>;
  using CE = cutlass::Array<float, 8>;
  using GE = cutlass::Array<__nv_bfloat16, 8>;
  constexpr int GQ = kSeg / 8;
  const int tid = threadIdx.x;
#pragma unroll
  for (int it = 0; it < kUnit * GQ / kThreads; it++) {
    const int item = it * kThreads + tid;
    const int tok = item / GQ, q = item % GQ;
    const GE up = *reinterpret_cast<const GE*>(sm_out + tok * kOutLd + 8 * q);
    const GE gate = *reinterpret_cast<const GE*>(sm_out + tok * kOutLd + kSeg + 8 * q);
    cutlass::epilogue::thread::SiLu<CE> silu{};
    const CE act = silu(array_convert<CE>(gate)) * array_convert<CE>(up);
    PackedVec pv;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      pv.elts[i].x = static_cast<__nv_bfloat16>(act[i * 2 + 0]);
      pv.elts[i].y = static_cast<__nv_bfloat16>(act[i * 2 + 1]);
    }
    const int ch = sub * kSeg + 8 * q;  // intermediate channel
    const int blk = ch >> 4;            // SF block: stage blk / 8, byte blk % 8
    uint8_t* sf_out =
        (q & 1) ? nullptr : p.asf + ((static_cast<size_t>(u) * S::NS2 + (blk >> 3)) * kUnit + tok) * 8 + (blk & 7);
    *reinterpret_cast<uint32_t*>(p.aq + (static_cast<size_t>(u) * kUnit + tok) * S::K2B + ch / 2) =
        quantize_bf16x8(pv, gs, sf_out);
  }
}

template <int E, int TopK, int H, int I>
__device__ __forceinline__ void finalize_rows(const Params& p, const int* s_fin, int nfin, int sub) {
  using CE = cutlass::Array<float, 8>;
  using GE = cutlass::Array<__nv_bfloat16, 8>;
  constexpr int PC = kRows / 8;
  for (int item = threadIdx.x; item < nfin * PC; item += kThreads) {
    const int t = s_fin[item / PC], pc = item % PC;
    CE sum;
    sum.fill(0);
    for (int k = 0; k < TopK; ++k) {
      const int slot = p.tk_slot[t * TopK + k];
      if (slot < 0) continue;
      const float scale = p.weights[t * TopK + k];
      const uint4 raw =
          __ldcg(reinterpret_cast<const uint4*>(p.y2 + static_cast<size_t>(slot) * H + sub * kRows + pc * 8));
      sum = sum + scale * array_convert<CE>(*reinterpret_cast<const GE*>(&raw));
    }
    *reinterpret_cast<GE*>(p.out + static_cast<size_t>(t) * H + sub * kRows + pc * 8) = array_convert<GE>(sum);
  }
}

template <int E, int TopK, int H, int I>
__global__ void __launch_bounds__(kThreads) persistent_main(Params p) {
  using S = Shape<E, TopK, H, I>;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_tile, s_tok[kUnit], s_fin[kUnit], s_nfin;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
  const int n_units = *reinterpret_cast<volatile int*>(p.n_units);
  const int total1 = n_units * S::NT1, total = total1 + n_units * S::NT2;

  for (;;) {
    if (tid == 0) s_tile = atomicAdd(p.queue, 1);
    __syncthreads();
    const int tile = s_tile;
    if (tile >= total) break;
    const bool g1 = tile < total1;
    const int u = g1 ? tile / S::NT1 : (tile - total1) / S::NT2;
    const int sub = g1 ? tile % S::NT1 : (tile - total1) % S::NT2;
    const int unit_info = p.units[u];
    const int e = unit_info & 0xffff, tg = unit_info >> 16;
    const int m0 = p.offsets[e] + tg * kUnit;
    const int mcount = min(kUnit, p.counts[e] - tg * kUnit);
    if (tid < kUnit) s_tok[tid] = p.slot_tok[m0 + min(tid, mcount - 1)];
    if (tid == 0) s_nfin = 0;
    if (!g1 && tid == 0) {
      while (atomicAdd(&p.g1done[u], 0) < S::NT1)
        __nanosleep(64);
      __threadfence();
    }
    __syncthreads();

    float acc[kWarpRows / 16][kUnit / 8][4];
    TileCtx tc;
    tc.u = u, tc.mcount = mcount, tc.s_tok = s_tok;
    if (g1) {
      tc.seg0 = sub * kSeg, tc.seg1 = I + sub * kSeg;
      tc.wbase = p.w13 + static_cast<size_t>(e) * (2 * I) * S::K1B;
      tc.sfbase = p.w13_sf + static_cast<size_t>(e) * (2 * I) * S::KP1;
      mainloop<E, TopK, H, I, kStages, kStage1, kSlot, true>(acc, smem, p, tc);
    } else {
      tc.seg0 = sub * kRows, tc.seg1 = sub * kRows + kSeg;
      tc.wbase = p.w2 + static_cast<size_t>(e) * H * S::K2B;
      tc.sfbase = p.w2_sf + static_cast<size_t>(e) * H * S::KP2;
      mainloop<E, TopK, H, I, 2 * kStages, kStage2, kSlot / 2, false>(acc, smem, p, tc);
    }
    __syncthreads();  // stage buffers are free: reuse them for the epilogue tile

    __nv_bfloat16* sm_out = reinterpret_cast<__nv_bfloat16*>(smem);
    const float alpha = (g1 ? p.alpha1 : p.alpha2)[e];
#pragma unroll
    for (int a = 0; a < kWarpRows / 16; a++)
#pragma unroll
      for (int b = 0; b < kUnit / 8; b++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int row = warp * kWarpRows + a * 16 + g + 8 * (i >> 1);
          const int tok = b * 8 + 2 * tig + (i & 1);
          sm_out[tok * kOutLd + row] = __float2bfloat16_rn(alpha * acc[a][b][i]);
        }
    __syncthreads();

    if (g1) {
      swiglu_quant_epilogue<E, TopK, H, I>(p, sm_out, u, sub, *p.gs2);
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicAdd(&p.g1done[u], 1);
    } else {
      constexpr int PC = kRows / 8;
#pragma unroll
      for (int i = 0; i < kUnit * PC / kThreads; i++) {
        const int idx = tid + i * kThreads;
        const int tok = idx / PC, pc = idx % PC;
        if (tok < mcount) {
          *reinterpret_cast<uint4*>(p.y2 + static_cast<size_t>(m0 + tok) * H + sub * kRows + pc * 8) =
              *reinterpret_cast<const uint4*>(sm_out + tok * kOutLd + pc * 8);
        }
      }
      __threadfence();
      __syncthreads();
      if (tid < mcount) {
        const int t = s_tok[tid];
        if (atomicAdd(&p.fcnt[t * S::NT2 + sub], 1) == p.nvalid[t] - 1) s_fin[atomicAdd(&s_nfin, 1)] = t;
      }
      __syncthreads();
      if (s_nfin) {
        __threadfence();
        finalize_rows<E, TopK, H, I>(p, s_fin, s_nfin, sub);
      }
    }
    __syncthreads();  // shared memory is reused by the next tile
  }
}

}  // namespace nvfp4_swiglu_moe

template <int E, int TopK, int H, int I>
struct Sm120Nvfp4MoeSwiglu {
  using S = nvfp4_swiglu_moe::Shape<E, TopK, H, I>;

  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView w13,
      tvm::ffi::TensorView w13_sf,
      tvm::ffi::TensorView w2,
      tvm::ffi::TensorView w2_sf,
      tvm::ffi::TensorView alpha1,
      tvm::ffi::TensorView alpha2,
      tvm::ffi::TensorView gs1,
      tvm::ffi::TensorView gs2,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView ws_u8,
      tvm::ffi::TensorView ws_bf16,
      tvm::ffi::TensorView ws_i32,
      tvm::ffi::TensorView out) {
    using namespace host;
    namespace k = nvfp4_swiglu_moe;
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    auto rows = SymbolicSize{"num_tokens"};
    TensorMatcher({rows, H}).with_dtype<bf16_t>().with_device(dev).verify(x);
    const int64_t T = rows.unwrap();
    RuntimeCheck(T >= 1 && T * TopK < (int64_t{1} << 30), "unsupported token count ", T);
    TensorMatcher({E, 2 * I, H / 2}).with_dtype<uint8_t>().with_device(dev).verify(w13);
    TensorMatcher({E, H, I / 2}).with_dtype<uint8_t>().with_device(dev).verify(w2);
    TensorMatcher({E * 2 * I * (H / 16)}).with_dtype<uint8_t>().with_device(dev).verify(w13_sf);
    TensorMatcher({E * H * (I / 16)}).with_dtype<uint8_t>().with_device(dev).verify(w2_sf);
    TensorMatcher({E}).with_dtype<float>().with_device(dev).verify(alpha1).verify(alpha2);
    TensorMatcher({1}).with_dtype<float>().with_device(dev).verify(gs1).verify(gs2);
    TensorMatcher({T, TopK}).with_dtype<int32_t>().with_device(dev).verify(ids);
    TensorMatcher({T, TopK}).with_dtype<float>().with_device(dev).verify(weights);
    TensorMatcher({S::u8_bytes(T)}).with_dtype<uint8_t>().with_device(dev).verify(ws_u8);
    TensorMatcher({T * TopK, H}).with_dtype<bf16_t>().with_device(dev).verify(ws_bf16);
    TensorMatcher({S::i32_count(T)}).with_dtype<int32_t>().with_device(dev).verify(ws_i32);
    TensorMatcher({T, H}).with_dtype<bf16_t>().with_device(dev).verify(out);

    k::Params p{};
    p.x = static_cast<const __nv_bfloat16*>(x.data_ptr());
    p.ids = static_cast<const int32_t*>(ids.data_ptr());
    p.weights = static_cast<const float*>(weights.data_ptr());
    p.w13 = static_cast<const uint8_t*>(w13.data_ptr());
    p.w13_sf = static_cast<const uint8_t*>(w13_sf.data_ptr());
    p.w2 = static_cast<const uint8_t*>(w2.data_ptr());
    p.w2_sf = static_cast<const uint8_t*>(w2_sf.data_ptr());
    p.alpha1 = static_cast<const float*>(alpha1.data_ptr());
    p.alpha2 = static_cast<const float*>(alpha2.data_ptr());
    p.gs1 = static_cast<const float*>(gs1.data_ptr());
    p.gs2 = static_cast<const float*>(gs2.data_ptr());
    p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    p.T = static_cast<int>(T);
    const int64_t units = S::max_units(T);
    uint8_t* u8 = static_cast<uint8_t*>(ws_u8.data_ptr());
    p.xq = u8;
    p.xsf = p.xq + S::align(T * S::K1B);
    p.aq = p.xsf + S::align(T * S::KP1);
    p.asf = p.aq + S::align(units * k::kUnit * S::K2B);
    p.y2 = static_cast<__nv_bfloat16*>(ws_bf16.data_ptr());
    int* i32 = static_cast<int*>(ws_i32.data_ptr());
    p.slot_tok = i32;
    p.tk_slot = p.slot_tok + T * TopK;
    p.units = p.tk_slot + T * TopK;
    p.counts = p.units + units;
    p.offsets = p.counts + E;
    p.g1done = p.offsets + E;
    p.fcnt = p.g1done + units;
    p.nvalid = p.fcnt + T * S::NT2;
    p.n_units = p.nvalid + T;
    p.queue = p.n_units + 1;

    const DLDevice device = dev.unwrap();
    const auto main_kernel = k::persistent_main<E, TopK, H, I>;
    static const uint32_t grid =
        runtime::get_sm_count(device.device_id) * runtime::get_blocks_per_sm(main_kernel, k::kThreads, k::kMainSmem);
    const uint32_t prologue_grid =
        1 + static_cast<uint32_t>((T * (H / 8) + k::kPrologueThreads - 1) / k::kPrologueThreads);
    LaunchKernel(prologue_grid, k::kPrologueThreads, device)(k::prologue<E, TopK, H, I>, p);
    LaunchKernel(grid, k::kThreads, device, k::kMainSmem)(main_kernel, p);
  }
};

}  // namespace sglang
