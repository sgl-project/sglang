"""CUDA source of the K-sliced one-launch HC boundary (see hc_boundary_ks.py).

Kept as a Python string so the module ships with the Python sources; the loader writes it to a
content-addressed file under the JIT cache root before compiling (as hc_combine_extend does).
"""

CUDA_SOURCE = r'''#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if !SGL_ARCH_BLACKWELL_OR_GREATER
#error "hc_boundary_ks requires SM100-class tensor cores"
#endif

namespace sglang {

namespace hcks {

constexpr int kHC = 4;
constexpr int kH = 2560;
constexpr int kW = kHC * kH;
constexpr int kL = 320;
constexpr int kThreads = 512;
constexpr int kRT = 64;                // token rows per CTA: the MMA M of both GEMMs
constexpr int kKS = 256;               // phase 1: K columns per CTA, inside one branch
constexpr int kS = kW / kKS;           // 40 K slices: CTAs per row tile
constexpr int kSPG = kH / kKS;         // 10 slices per branch
constexpr int kBK = 64;                // K per 128B-swizzled box
constexpr int kKB1 = kKS / kBK;        // 4
constexpr int kKB2 = kL / kBK;         // 5
constexpr int kOB = kL / kS;           // 8 low-rank outputs each CTA reduces
constexpr int kCols2 = kH / kS;        // 64 hidden columns per CTA in phase 2
constexpr int kLog2 = kHC * kCols2;    // 256 permuted up logits: the phase-2 MMA N
constexpr int kMaxRT = 5;
constexpr int kMaxRows = kMaxRT * kRT; // 320
constexpr int kNQ = 8;                 // phase-1 MMA columns past the low rank: the 4 next gate rows, 4 zero rows
constexpr int kWBox1 = (kL + kNQ) * 128;  // one 64-column box of the W_down slice and the next gate rows
constexpr int kWBox2 = kLog2 * 128;    // one 64-column box of the W_up tile (256 rows)
constexpr int kXBox = kRT * 128;       // one 64-column box of a 64-row operand
constexpr int kWBytes = kKB1 * kWBox1;
static_assert(kWBytes >= kKB2 * kWBox2, "the W_up tile reuses the W_down slice's buffer");
constexpr int kXBytes = kKB2 * kXBox;
constexpr int kTmemCols = 512;
constexpr int kPBlock = kRT * kOB;     // floats per (row tile, source slice, output block)

// Counters (int32), zero between launches; the last CTA to exit re-arms them.
constexpr int kCtrP = 0;      // [kMaxRT]: phase-1 CTAs done (partials, sums, combined stored)
constexpr int kCtrT = 8;      // [kMaxRT]: reducers done (t stored)
constexpr int kCtrExit = 16;
constexpr int kCtrSize = 32;
constexpr int kDbgSlots = 16;
constexpr int kSmemCap = 232448;

struct Params {
  const void* block_output;        // [M, H]
  const void* residual;            // [M, W]
  const float* partials;           // [M, HC] this read's gate dot totals
  const void* norm_weight;         // [W] the next read's per-branch norm weight
  const void* next_inject_weight;  // [HC, W] (kNextGate)
  void* output;                    // [M, W] combined streams
  void* normed_output;             // [M, W] (kWriteNormed)
  float* next_partials;            // [M, HC] (kNextGate)
  void* mixed;                     // [M, H]
  float* ws_p;                     // [kMaxRT][kS src][kS block][kOB][kRT] down partials
  float* ws_ss;                    // [kMaxRT][kS][kRT] sums of squares
  float* ws_q;                     // [kMaxRT][kS][kRT][kHC] next gate partial dots
  void* ws_t;                      // [kMaxRows, kL] bf16 silu'd low-rank activations
  int* ctr;                        // [kCtrSize]
  long long* dbg;                  // optional [grid, kDbgSlots]
  int rows;
  int flags;                       // kFlagDiscard | kFlagTanh
  float eps;
};

constexpr int kFlagDiscard = 1;  // drop consumed scratch lines from L2 (no DRAM write-back)
constexpr int kFlagTanh = 2;     // gate sigmoid as 0.5 tanh(x / 2) + 0.5 (one MUFU op)

struct alignas(1024) Smem {
  alignas(1024) uint8_t w[kWBytes];  // phase 1: [4 boxes][320 W_down + 8 next gate rows]; phase 3: W_up [5][256]
  alignas(1024) uint8_t x[kXBytes];  // phase 1: bf16(x * (1 + w_norm)) [4][64 rows]; phase 3: t [5][64]
  float wn2[kHC][kCols2];            // 1 + norm weight over the phase-3 columns of each branch
  float rs[kRT][kHC];                // inverse RMS per (row, branch)
  alignas(16) bf16_t tb[kRT][kOB];   // this CTA's t block, staged for 16-byte row stores
  uint64_t w_full;
  uint64_t d1_full;   // low-rank outputs [0, 256)
  uint64_t d1b_full;  // low-rank outputs [256, 320) and the next gate dots
  uint64_t wu_full;
  uint64_t t_full;
  uint64_t d2_full;   // logits [0, 128)
  uint64_t d2b_full;  // logits [128, 256)
  uint32_t taddr;
};

// ---------------------------------------------------------------------------------------------- PTX
// TMA and tcgen05 wrappers follow kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh (measured on SM100).
SGL_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

SGL_DEVICE void prefetch_tensormap(const void* tmap) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory");
}

SGL_DEVICE void tma_2d(uint32_t dst, const CUtensorMap* tmap, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(tmap),
      "r"(x),
      "r"(y),
      "r"(smem_u32(bar))
      : "memory");
}

SGL_DEVICE void prefetch_l2_2d(const CUtensorMap* tmap, int32_t x, int32_t y) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(tmap), "r"(x), "r"(y)
               : "memory");
}

SGL_DEVICE void signal_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

SGL_DEVICE int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

SGL_DEVICE void wait_ge(const int* p, int target) {
  while (ld_acquire(p) < target) {
  }
}

// Drop a consumed scratch line from L2 without writing it back to DRAM.
SGL_DEVICE void discard_l2(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;" ::"l"(p) : "memory");
}

SGL_DEVICE void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
SGL_DEVICE void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

SGL_DEVICE void tc_alloc(uint32_t smem_dst, uint32_t n_cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_dst), "r"(n_cols));
}
SGL_DEVICE void tc_relinquish() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}
SGL_DEVICE void tc_dealloc(uint32_t taddr, uint32_t n_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(n_cols));
}
SGL_DEVICE void tc_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" ::"r"(smem_u32(bar)));
}
SGL_DEVICE void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
SGL_DEVICE void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
SGL_DEVICE void tc_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

SGL_DEVICE void tc_ld_x8(uint32_t taddr, uint32_t* d) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
               : "r"(taddr));
}

SGL_DEVICE void tc_ld_x16(uint32_t taddr, uint32_t* d) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
        "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15])
      : "r"(taddr));
}

SGL_DEVICE void tc_ld_x32(uint32_t taddr, uint32_t* d) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%"
      "28,%29,%30,%31}, [%32];"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]),
        "=r"(d[8]), "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]),
        "=r"(d[16]), "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23]),
        "=r"(d[24]), "=r"(d[25]), "=r"(d[26]), "=r"(d[27]), "=r"(d[28]), "=r"(d[29]), "=r"(d[30]), "=r"(d[31])
      : "r"(taddr));
}

// Dense kind::f16 shapes: M in {64, 128}, N in {8..256 step 8}, K = 16.
SGL_DEVICE void tc_mma(uint32_t d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
      "  tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" ::"r"(d),
      "l"(desc_a),
      "l"(desc_b),
      "r"(idesc),
      "r"(accumulate));
}

// K-major, 128B-swizzled operand in 8-row groups of 1024 B (what a 64-column TMA box writes): addr>>4 | SBO
// 1024>>4 at bit 32 | version 1 at bit 46 | layout 2 (128B) at bit 61. A K step of 16 adds 32 B.
SGL_DEVICE uint64_t smem_desc(uint32_t addr) {
  auto enc = [](uint32_t v) -> uint64_t { return uint64_t((v & 0x3FFFFu) >> 4); };
  return enc(addr) | (enc(1024u) << 32) | (uint64_t(1) << 46) | (uint64_t(2) << 61);
}

// kind::f16 instruction descriptor: F32 accumulator, BF16 A and B, both K-major.
constexpr uint32_t idesc_bf16(uint32_t M, uint32_t N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (((N >> 3) & 0x3Fu) << 17) | (((M >> 4) & 0x1Fu) << 24);
}

SGL_DEVICE long long global_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

SGL_DEVICE void trace(const Params& p, int slot) {
  if (p.dbg != nullptr) p.dbg[static_cast<int64_t>(blockIdx.x) * kDbgSlots + slot] = global_ns();
}

// Triton's tl.sigmoid on fp32 as the up/gate kernel's PTX has it: ex2.approx(x * -log2 e), add, div.full.
SGL_DEVICE float sigmoid_triton(float x) {
  float e, r;
  const float xm = __fmul_rn(x, -1.44269502162933349609375f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(xm));
  const float d = __fadd_rn(1.0f, e);
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(1.0f), "f"(d));
  return r;
}

SGL_DEVICE float sigmoid_tanh(float x) {
  float th;
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(0.5f * x));
  return __fmaf_rn(0.5f, th, 0.5f);
}

SGL_DEVICE float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

using Vec8 = device::AlignedVector<packed_t<bf16_t>, 4>;  // 8 bf16, 16 bytes

// ---------------------------------------------------------------------------------------------- kernel
// CTA (rt, s): token rows [64 rt, 64 rt + 64), K slice [256 s, 256 s + 256) inside branch s / 10.
//   1  combine its slice (hc_combine_extend's arithmetic, so the stored streams are the stock bits) and its rows'
//      sums of squares, then the mix down of bf16(x * (1 + w_norm)) on tcgen05 (M = 64 rows, N = 320 low-rank
//      outputs + the 4 next gate rows); the FP32 partial goes to a scratch buffer and the inverse RMS, a
//      per-(row, branch) scalar of this K range, is applied after the K sum;
//   2  once the row tile's 40 partials are in, reduce low-rank outputs [8 s, 8 s + 8) in a fixed order,
//      t = bf16(silu(x / 4)) (the CuTe split-K silu down's epilogue); the last CTA of the tile also sums the next
//      gate totals, after publishing its t;
//   3  once the row tile's t is in, the mix up on tcgen05 (M = 64 rows, N = 256 permuted logits, K = 320) and the
//      gated mean over hidden columns [64 s, 64 s + 64), the normalized streams recomputed from the combined ones.
// Every CTA must be resident at once (the host checks the grid against the device's resident CTAs).
template <bool kUsePDL, bool kNextGate, bool kWriteNormed>
__global__ __launch_bounds__(kThreads, 1) void hc_boundary_ks_kernel(
    const __grid_constant__ CUtensorMap wd_map,
    const __grid_constant__ CUtensorMap wu_map,
    const __grid_constant__ CUtensorMap t_map,
    const __grid_constant__ CUtensorMap inj_map,
    const Params __grid_constant__ p) {
  using namespace device;
  using Float2 = packed_t<bf16_t>;
  extern __shared__ __align__(1024) uint8_t hcks_smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(hcks_smem_raw + ((1024u - (smem_u32(hcks_smem_raw) & 1023u)) & 1023u));

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int G = gridDim.x;
  const int cta = blockIdx.x;
  const int rt = cta / kS;
  const int s = cta % kS;
  const int branch = s / kSPG;
  const int k0 = s * kKS;              // column in the [M, W] streams
  const int kh0 = k0 - branch * kH;    // column in the block output
  const int R = p.rows;
  const int row0 = rt * kRT;
  constexpr uint32_t kW1Bytes = kKB1 * (kNextGate ? kWBox1 : kL * 128);

  if (tid == 0) trace(p, 0);
  if (warp == 0) {
    tc_alloc(smem_u32(&sm.taddr), kTmemCols);
    tc_relinquish();
  } else if (warp == 1 && lane == 0) {
    device::ptx::mbar_init(&sm.w_full, 1);
    device::ptx::mbar_init(&sm.d1_full, 1);
    device::ptx::mbar_init(&sm.d1b_full, 1);
    device::ptx::mbar_init(&sm.wu_full, 1);
    device::ptx::mbar_init(&sm.t_full, 1);
    device::ptx::mbar_init(&sm.d2_full, 1);
    device::ptx::mbar_init(&sm.d2b_full, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    prefetch_tensormap(&wd_map);
    prefetch_tensormap(&wu_map);
    prefetch_tensormap(&t_map);
    if constexpr (kNextGate) prefetch_tensormap(&inj_map);
    // The weights are not the producer's output: load the W_down slice (and the next gate rows, whose box rows
    // 4-7 fall outside the [4, W] map and land as zeros) now, and pull the W_up tile into L2.
    device::ptx::mbar_arrive_expect_tx(&sm.w_full, kW1Bytes);
    for (int kb = 0; kb < kKB1; ++kb) {
      tma_2d(smem_u32(sm.w + kb * kWBox1), &wd_map, k0 + kb * kBK, 0, &sm.w_full);
      tma_2d(smem_u32(sm.w + kb * kWBox1 + kL * 64), &wd_map, k0 + kb * kBK, kL / 2, &sm.w_full);
      if constexpr (kNextGate) tma_2d(smem_u32(sm.w + kb * kWBox1 + kL * 128), &inj_map, k0 + kb * kBK, 0, &sm.w_full);
    }
    for (int kb = 0; kb < kKB2; ++kb) prefetch_l2_2d(&wu_map, kb * kBK, s * kLog2);
  }
  // Constant inputs, before the PDL wait.
  const bf16_t* nw = static_cast<const bf16_t*>(p.norm_weight);
  if (tid < kHC * kCols2) {
    const int g = tid / kCols2, c = tid % kCols2;
    sm.wn2[g][c] = 1.0f + __bfloat162float(nw[g * kH + s * kCols2 + c]);
  }
  const int prow = tid >> 3;  // phase 1: row of the tile
  const int cg = tid & 7;     // phase 1: 16-byte chunks cg + 8 i of the slice
  Vec8 w_vec[kKB1];
  {
    const auto wp = reinterpret_cast<const Vec8*>(nw + k0);
#pragma unroll
    for (int i = 0; i < kKB1; ++i) w_vec[i] = wp[cg + 8 * i];
  }
  __syncthreads();
  const uint32_t taddr = sm.taddr;
  device::PDLWaitPrimary<kUsePDL>();
  if (tid == 0) trace(p, 1);

  // ---- phase 1: combine, sums of squares, down operand
  {
    constexpr int64_t kHcCount = kHC;
    const int m = row0 + prow;
    const bool valid = m < R;
    Vec8 r_vec[kKB1], y_vec[kKB1];
    float a = 0.0f;
    if (valid) {
      const auto rp =
          reinterpret_cast<const Vec8*>(static_cast<const bf16_t*>(p.residual) + static_cast<int64_t>(m) * kW + k0);
      const auto yp = reinterpret_cast<const Vec8*>(static_cast<const bf16_t*>(p.block_output) +
                                                    static_cast<int64_t>(m) * kH + kh0);
#pragma unroll
      for (int i = 0; i < kKB1; ++i) {
        r_vec[i] = rp[cg + 8 * i];
        y_vec[i] = yp[cg + 8 * i];
      }
      const float total = p.partials[static_cast<int64_t>(m) * kHcCount + branch];
      a = 2.0f / (1.0f + math::exp(-total / kHcCount));
    }
    float ss = 0.0f;
    const auto out_ptr =
        reinterpret_cast<Vec8*>(static_cast<bf16_t*>(p.output) + static_cast<int64_t>(m) * kW + k0);
#pragma unroll
    for (int i = 0; i < kKB1; ++i) {
      Vec8 xw_vec;
      if (valid) {
        Vec8 out_vec;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const auto [rx, ry] = cast<fp32x2_t>(r_vec[i][e]);
          const auto [yx, yy] = cast<fp32x2_t>(y_vec[i][e]);
          out_vec[e] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
        }
        out_ptr[cg + 8 * i] = out_vec;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const auto [x0, x1] = cast<fp32x2_t>(out_vec[e]);
          const auto [w0, w1] = cast<fp32x2_t>(w_vec[i][e]);
          ss = __fmaf_rn(x0, x0, ss);
          ss = __fmaf_rn(x1, x1, ss);
          xw_vec[e] = cast<Float2>(fp32x2_t{__fmul_rn(x0, 1.0f + w0), __fmul_rn(x1, 1.0f + w1)});
        }
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) xw_vec[e] = cast<Float2>(fp32x2_t{0.0f, 0.0f});
      }
      *reinterpret_cast<Vec8*>(sm.x + i * kXBox + prow * 128 + ((cg ^ (prow & 7)) << 4)) = xw_vec;
    }
    // Row total over the 8 threads of the row, a fixed butterfly.
#pragma unroll
    for (int off = 1; off < 8; off <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if (cg == 0) p.ws_ss[(static_cast<int64_t>(rt) * kS + s) * kRT + prow] = ss;
  }
  // The MMA reads the operand through the async proxy; the threads wrote it generically.
  fence_proxy_async_shared();
  __syncthreads();
  if (tid == 0) trace(p, 2);

  if (warp == 0) {
    if (lane == 0) {
      device::ptx::mbar_wait_parity(&sm.w_full, 0);
      trace(p, 3);
      tc_fence_after();
      constexpr uint32_t idesc_a = idesc_bf16(kRT, 256);
      constexpr uint32_t idesc_b = idesc_bf16(kRT, kL - 256 + (kNextGate ? kNQ : 0));
      // Outputs [0, 256) first, so drain quarters 0-2 start while the rest runs.
#pragma unroll
      for (int kb = 0; kb < kKB1; ++kb) {
        const uint64_t dx = smem_desc(smem_u32(sm.x + kb * kXBox));
        const uint64_t dw = smem_desc(smem_u32(sm.w + kb * kWBox1));
#pragma unroll
        for (int k = 0; k < kBK / 16; ++k) tc_mma(taddr, dx + 2 * k, dw + 2 * k, idesc_a, (kb | k) != 0);
      }
      tc_commit(&sm.d1_full);
#pragma unroll
      for (int kb = 0; kb < kKB1; ++kb) {
        const uint64_t dx = smem_desc(smem_u32(sm.x + kb * kXBox));
        const uint64_t dw2 = smem_desc(smem_u32(sm.w + kb * kWBox1 + 256 * 128));
#pragma unroll
        for (int k = 0; k < kBK / 16; ++k) tc_mma(taddr + 256, dx + 2 * k, dw2 + 2 * k, idesc_b, (kb | k) != 0);
      }
      tc_commit(&sm.d1b_full);
    }
    __syncwarp();
  } else if (warp == 1) {
    if (lane == 0) {
      // The W_down slice is free once the down MMAs are done: bring in this CTA's W_up tile.
      device::ptx::mbar_wait_parity(&sm.d1b_full, 0);
      device::ptx::mbar_arrive_expect_tx(&sm.wu_full, kKB2 * kWBox2);
      for (int kb = 0; kb < kKB2; ++kb) {
        tma_2d(smem_u32(sm.w + kb * kWBox2), &wu_map, kb * kBK, s * kLog2, &sm.wu_full);
      }
    }
    __syncwarp();
  }

  // ---- drain 1: TMEM lane band (warp % 4) holds rows 16 band + lane (lanes 0-15); quarter warp / 4 takes outputs
  // [80 q, 80 q + 80) to ws_p[rt][s][output block][row][8]; quarter 3 also the next gate dots to ws_q.
  {
    const int band = warp & 3;
    const int quarter = warp >> 2;
    const int row = band * 16 + lane;
    device::ptx::mbar_wait_parity(quarter == 3 ? &sm.d1b_full : &sm.d1_full, 0);
    if (tid == 128) trace(p, 4);
    tc_fence_after();
    const uint32_t tb = taddr + ((band * 32) << 16);
    uint32_t v[80];
    tc_ld_x32(tb + quarter * 80, v);
    tc_ld_x32(tb + quarter * 80 + 32, v + 32);
    tc_ld_x16(tb + quarter * 80 + 64, v + 64);
    uint32_t qv[8];
    if (kNextGate && quarter == 3) tc_ld_x8(tb + kL, qv);
    tc_wait_ld();
    if (tid == 128) trace(p, 5);
    {
      // One plane of 64 rows per output. Lanes 16-31 take outputs 4-7 of lane - 16's row, so each store
      // instruction writes two 64-byte runs.
      const int hi = lane >> 4;
      const int srow = band * 16 + (lane & 15);
      float* dst = p.ws_p + ((static_cast<int64_t>(rt) * kS + s) * kS + quarter * 10) * kPBlock + hi * 4 * kRT + srow;
      const bool st_ok = row0 + srow < R;
#pragma unroll
      for (int b = 0; b < 10; ++b) {
#pragma unroll
        for (int o = 0; o < 4; ++o) {
          const uint32_t up = __shfl_sync(0xffffffffu, v[8 * b + 4 + o], lane & 15);
          if (st_ok) dst[b * kPBlock + o * kRT] = __uint_as_float(hi ? up : v[8 * b + o]);
        }
      }
    }
    if (lane < 16 && row0 + row < R) {
      if (kNextGate && quarter == 3) {
        *reinterpret_cast<float4*>(p.ws_q + ((static_cast<int64_t>(rt) * kS + s) * kRT + row) * kHC) =
            make_float4(__uint_as_float(qv[0]), __uint_as_float(qv[1]), __uint_as_float(qv[2]),
                        __uint_as_float(qv[3]));
      }
    }
    tc_fence_before();
  }
  __syncthreads();
  // The barrier orders every thread's stores before thread 0's release (cumulative), as H469's hand-offs do.
  if (tid == 0) {
    trace(p, 6);
    signal_add(&p.ctr[kCtrP + rt], 1);
    trace(p, 13);
  }

  // ---- phase 2: reduce outputs [8 s, 8 s + 8) of the row tile in a fixed order
  if (tid == 0) wait_ge(&p.ctr[kCtrP + rt], kS);
  __syncthreads();
  (void)ld_acquire(&p.ctr[kCtrP + rt]);
  if (tid == 0) trace(p, 7);
  {
    // Thread = (row tid % 64, output 8 s + tid / 64): a warp reads 32 consecutive rows of one output plane.
    const int row = tid & (kRT - 1);
    const int o8 = tid / kRT;
    const float* src = p.ws_p + ((static_cast<int64_t>(rt) * kS) * kS + s) * kPBlock + o8 * kRT + row;
    constexpr int64_t kSrcStride = static_cast<int64_t>(kS) * kPBlock;
    float pv[kS];
#pragma unroll
    for (int j = 0; j < kS; ++j) pv[j] = __ldcg(src + j * kSrcStride);
    if (o8 < kHC) {
      // Branch o8's inverse RMS for this row, in the same load round as the partials.
      float sv[kSPG];
      const float* ssrc = p.ws_ss + (static_cast<int64_t>(rt) * kS + o8 * kSPG) * kRT + row;
#pragma unroll
      for (int j = 0; j < kSPG; ++j) sv[j] = __ldcg(ssrc + j * kRT);
      float tot_ss = sv[0];
#pragma unroll
      for (int j = 1; j < kSPG; ++j) tot_ss += sv[j];
      constexpr int64_t kHiddenSize = kH;
      sm.rs[row][o8] = math::rsqrt(tot_ss / kHiddenSize + p.eps);
    }
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int g = 0; g < kHC; ++g) {
      float gs = pv[g * kSPG];
#pragma unroll
      for (int j = 1; j < kSPG; ++j) gs = __fadd_rn(gs, pv[g * kSPG + j]);
      tot = __fmaf_rn(sm.rs[row][g], gs, tot);
    }
    const float v = __fmul_rn(tot, 0.25f);
    const float sig = __frcp_rn(__fadd_rn(expf(-v), 1.0f));
    sm.tb[row][o8] = __float2bfloat16_rn(__fmul_rn(v, sig));
  }
  __syncthreads();
  if (tid < kRT && row0 + tid < R) {
    *reinterpret_cast<uint4*>(static_cast<bf16_t*>(p.ws_t) + static_cast<int64_t>(row0 + tid) * kL + s * kOB) =
        *reinterpret_cast<const uint4*>(&sm.tb[tid][0]);
  }
  // t is read by TMA in other CTAs.
  fence_proxy_async_global();
  __syncthreads();
  if (tid == 0) {
    trace(p, 8);
    signal_add(&p.ctr[kCtrT + rt], 1);
    trace(p, 15);
  }
  if constexpr (kNextGate) {
    // The next gate totals are read by the next boundary only: off this one's path, after t is published; the
    // tile's 256 (row, gate) totals are spread over its CTAs, kQPerCta each.
    constexpr int kQPerCta = (kRT * kHC + kS - 1) / kS;
    const int pi = s * kQPerCta + tid;
    if (tid < kQPerCta && pi < kRT * kHC) {
      const int row = pi >> 2;
      const int g2 = pi & 3;
      const int m = row0 + row;
      const float* src = p.ws_q + (static_cast<int64_t>(rt) * kS) * kRT * kHC + row * kHC + g2;
      float qv[kS];
#pragma unroll
      for (int j = 0; j < kS; ++j) qv[j] = __ldcg(src + static_cast<int64_t>(j) * kRT * kHC);
      float tot = 0.0f;
#pragma unroll
      for (int g = 0; g < kHC; ++g) {
        float gs = qv[g * kSPG];
#pragma unroll
        for (int j = 1; j < kSPG; ++j) gs = __fadd_rn(gs, qv[g * kSPG + j]);
        tot = __fmaf_rn(sm.rs[row][g], gs, tot);
      }
      if (m < R) p.next_partials[static_cast<int64_t>(m) * kHC + g2] = tot;
    }
  }
  if (p.flags & kFlagDiscard) {
    // Each warp drops the 40 scratch lines it read in phase 2 (one 128-byte run of one plane per source; the
    // loads completed before the barrier above).
    const int line = ((tid / kRT) * kRT + (warp & 1) * 32);
    for (int j = lane; j < kS; j += 32) {
      discard_l2(p.ws_p + ((static_cast<int64_t>(rt) * kS + j) * kS + s) * kPBlock + line);
    }
  }
  if (tid == 64) trace(p, 14);
  device::PDLTriggerSecondary<kUsePDL>();

  // ---- phase 3: mix up and gated mean
  if (warp == 2 && lane == 0) {
    wait_ge(&p.ctr[kCtrT + rt], kS);
    trace(p, 9);
    fence_proxy_async_global();
    device::ptx::mbar_arrive_expect_tx(&sm.t_full, kXBytes);
    for (int kb = 0; kb < kKB2; ++kb) {
      tma_2d(smem_u32(sm.x + kb * kXBox), &t_map, kb * kBK, row0, &sm.t_full);
    }
  } else if (warp == 0 && lane == 0) {
    device::ptx::mbar_wait_parity(&sm.wu_full, 0);
    device::ptx::mbar_wait_parity(&sm.t_full, 0);
    trace(p, 10);
    tc_fence_after();
    // Logits [0, 128) first, so drain quarters 0-1 start while the rest runs.
    constexpr uint32_t idesc = idesc_bf16(kRT, kLog2 / 2);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int kb = 0; kb < kKB2; ++kb) {
        const uint64_t dt = smem_desc(smem_u32(sm.x + kb * kXBox));
        const uint64_t dw = smem_desc(smem_u32(sm.w + kb * kWBox2 + half * (kLog2 / 2) * 128));
#pragma unroll
        for (int k = 0; k < kBK / 16; ++k) {
          tc_mma(taddr + half * (kLog2 / 2), dt + 2 * k, dw + 2 * k, idesc, (kb | k) != 0);
        }
      }
      tc_commit(half == 0 ? &sm.d2_full : &sm.d2b_full);
    }
  }
  __syncwarp();
  {
    // Thread = row 16 band + lane (lanes 0-15), hidden columns [64 s + 16 q, + 16) of each branch.
    const int band = warp & 3;
    const int quarter = warp >> 2;
    const int row = band * 16 + lane;
    const int m = row0 + row;
    const bool live = lane < 16 && m < R;
    const int jc0 = s * kCols2 + quarter * 16;
    const bf16_t* comb = static_cast<const bf16_t*>(p.output);
    uint4 xn[kHC][2];
    if (live) {
#pragma unroll
      for (int g = 0; g < kHC; ++g) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          xn[g][u] = __ldcg(reinterpret_cast<const uint4*>(comb + static_cast<int64_t>(m) * kW + g * kH + jc0 + 8 * u));
        }
      }
      // The next read's normalized streams, grouped_gemma_rmsnorm's expression on the stored combined streams.
#pragma unroll
      for (int g = 0; g < kHC; ++g) {
        const float nf = sm.rs[row][g];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          bf16_t* xb = reinterpret_cast<bf16_t*>(&xn[g][u]);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float ix = __bfloat162float(xb[e]);
            xb[e] = __float2bfloat16_rn(ix * nf * sm.wn2[g][quarter * 16 + 8 * u + e]);
          }
          if constexpr (kWriteNormed) {
            *reinterpret_cast<uint4*>(static_cast<bf16_t*>(p.normed_output) + static_cast<int64_t>(m) * kW + g * kH +
                                      jc0 + 8 * u) = xn[g][u];
          }
        }
      }
    }
    const bool use_tanh = (p.flags & kFlagTanh) != 0;
    device::ptx::mbar_wait_parity(quarter < 2 ? &sm.d2_full : &sm.d2b_full, 0);
    tc_fence_after();
    if (tid == 128) trace(p, 11);
    bf16_t* out = static_cast<bf16_t*>(p.mixed) + static_cast<int64_t>(m) * kH + jc0;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      // TMEM column 4 j + g of this quarter: hidden column jc0 + j, branch g.
      uint32_t v[32];
      tc_ld_x32(taddr + ((band * 32) << 16) + quarter * 64 + 32 * u, v);
      tc_wait_ld();
      if (live) {
        uint4 o;
        bf16_t* ob = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float sg[kHC], x[kHC];
#pragma unroll
          for (int g = 0; g < kHC; ++g) {
            const float logit = round_bf16(__uint_as_float(v[4 * e + g]));
            sg[g] = use_tanh ? sigmoid_tanh(logit) : sigmoid_triton(logit);
            x[g] = __bfloat162float(reinterpret_cast<const bf16_t*>(&xn[g][u])[e]);
          }
          float total = __fadd_rn(__fmul_rn(sg[0], x[0]), __fmul_rn(sg[1], x[1]));
          total = __fmaf_rn(sg[2], x[2], total);
          total = __fmaf_rn(sg[3], x[3], total);
          ob[e] = __float2bfloat16_rn(__fmul_rn(total, 0.25f));
        }
        *reinterpret_cast<uint4*>(out + 8 * u) = o;
      }
    }
    tc_fence_before();
  }

  __syncthreads();
  if (tid == 0) trace(p, 12);
  if (warp == 0) {
    tc_fence_after();
    tc_dealloc(taddr, kTmemCols);
  }

  if (tid == 0) {
    __threadfence();
    const int done = atomicAdd(&p.ctr[kCtrExit], 1);
    if (done == G - 1) {
      __threadfence();
      const int n_rt = (R + kRT - 1) / kRT;
      for (int i = 0; i < n_rt; ++i) {
        p.ctr[kCtrP + i] = 0;
        p.ctr[kCtrT + i] = 0;
      }
      p.ctr[kCtrExit] = 0;
      __threadfence();
    }
  }
}

// ---------------------------------------------------------------------------------------------- host
inline CUtensorMap make_map(const void* base, uint64_t cols, uint64_t rows, uint64_t row_stride_bytes,
                            uint32_t box_cols, uint32_t box_rows) {
  CUtensorMap map{};
  uint64_t dim[2] = {cols, rows};
  uint64_t stride[1] = {row_stride_bytes};
  uint32_t box[2] = {box_cols, box_rows};
  uint32_t elem_stride[2] = {1, 1};
  const CUresult res = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dim, stride, box, elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  host::RuntimeCheck(res == CUDA_SUCCESS, "hc_boundary_ks: cuTensorMapEncodeTiled failed");
  return map;
}

}  // namespace hcks

template <bool kUsePDL, bool kNextGate, bool kWriteNormed>
struct HcBoundaryKsKernel {
  static constexpr auto kernel = hcks::hc_boundary_ks_kernel<kUsePDL, kNextGate, kWriteNormed>;
  static constexpr int kSmem = sizeof(hcks::Smem) + 1024;
  static_assert(kSmem <= hcks::kSmemCap, "shared memory");

  // CTAs of this kernel the device holds resident at once.
  static int resident_ctas() {
    static int cached = 0;
    if (cached == 0) {
      host::RuntimeCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == cudaSuccess,
                         "hc_boundary_ks: smem attribute");
      int per_sm = 0;
      host::RuntimeCheck(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, hcks::kThreads, kSmem) == cudaSuccess,
          "hc_boundary_ks: occupancy");
      int dev = 0, sms = 0;
      host::RuntimeCheck(cudaGetDevice(&dev) == cudaSuccess, "hc_boundary_ks: device");
      host::RuntimeCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess,
                         "hc_boundary_ks: SM count");
      cached = per_sm * sms;
    }
    return cached;
  }

  static int64_t capacity(const tvm::ffi::TensorView probe) {
    (void)probe;
    return static_cast<int64_t>(resident_ctas());
  }

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView norm_weight,
      const tvm::ffi::TensorView next_inject_weight,
      const tvm::ffi::TensorView w_down,
      const tvm::ffi::TensorView w_up,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView normed_output,
      const tvm::ffi::TensorView next_partials,
      const tvm::ffi::TensorView mixed,
      const tvm::ffi::TensorView ws_p,
      const tvm::ffi::TensorView ws_ss,
      const tvm::ffi::TensorView ws_q,
      const tvm::ffi::TensorView ws_t,
      const tvm::ffi::TensorView ctr,
      const tvm::ffi::TensorView dbg,
      int64_t use_dbg,
      int64_t flags,
      double eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, hcks::kH}).with_dtype<bf16_t>().with_device(device).verify(block_output).verify(mixed);
    TensorMatcher({M, hcks::kW}).with_dtype<bf16_t>().with_device(device).verify(residual).verify(output);
    if constexpr (kWriteNormed) {
      TensorMatcher({M, hcks::kW}).with_dtype<bf16_t>().with_device(device).verify(normed_output);
    }
    TensorMatcher({M, hcks::kHC}).with_dtype<fp32_t>().with_device(device).verify(partials);
    TensorMatcher({hcks::kW}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    if constexpr (kNextGate) {
      TensorMatcher({hcks::kHC, hcks::kW}).with_dtype<bf16_t>().with_device(device).verify(next_inject_weight);
      TensorMatcher({M, hcks::kHC}).with_dtype<fp32_t>().with_device(device).verify(next_partials);
    }
    TensorMatcher({hcks::kL, hcks::kW}).with_dtype<bf16_t>().with_device(device).verify(w_down);
    auto LDU = SymbolicSize{"ldu"};
    TensorMatcher({hcks::kW, LDU}).with_dtype<bf16_t>().with_device(device).verify(w_up);
    TensorMatcher({hcks::kMaxRT * hcks::kS * hcks::kS * hcks::kPBlock}).with_dtype<fp32_t>().with_device(device).verify(
        ws_p);
    TensorMatcher({hcks::kMaxRT * hcks::kS * hcks::kRT}).with_dtype<fp32_t>().with_device(device).verify(ws_ss);
    TensorMatcher({hcks::kMaxRT * hcks::kS * hcks::kRT * hcks::kHC}).with_dtype<fp32_t>().with_device(device).verify(
        ws_q);
    TensorMatcher({hcks::kMaxRows, hcks::kL}).with_dtype<bf16_t>().with_device(device).verify(ws_t);
    TensorMatcher({hcks::kCtrSize}).with_dtype<int32_t>().with_device(device).verify(ctr);
    const auto rows = static_cast<int64_t>(M.unwrap());
    const auto ldu = static_cast<int64_t>(LDU.unwrap());
    RuntimeCheck(rows >= 1 && rows <= hcks::kMaxRows, "hc_boundary_ks serves 1..320 rows");
    RuntimeCheck(ldu >= hcks::kL && ldu % 8 == 0, "w_up must hold >= 320 columns, 16-byte rows");
    const int64_t n_rt = (rows + hcks::kRT - 1) / hcks::kRT;
    const int64_t grid = n_rt * hcks::kS;
    if (use_dbg) {
      auto DN = SymbolicSize{"dbg_numel"};
      TensorMatcher({DN}).with_dtype<int64_t>().with_device(device).verify(dbg);
      RuntimeCheck(static_cast<int64_t>(DN.unwrap()) >= grid * hcks::kDbgSlots, "trace buffer too small");
    }
    RuntimeCheck(grid <= resident_ctas(), "hc_boundary_ks: the grid would not be resident at once");
    const auto dev = device.unwrap();

    const auto wd_map = hcks::make_map(w_down.data_ptr(), hcks::kW, hcks::kL, hcks::kW * 2, hcks::kBK, hcks::kL / 2);
    const auto wu_map = hcks::make_map(w_up.data_ptr(), ldu, hcks::kW, ldu * 2, hcks::kBK, hcks::kLog2);
    const auto t_map = hcks::make_map(ws_t.data_ptr(), hcks::kL, rows, hcks::kL * 2, hcks::kBK, hcks::kRT);
    // Box rows 4-7 of the [4, W] next gate weight are out of bounds: zero-filled, and counted by complete_tx.
    const auto inj_map = hcks::make_map(kNextGate ? next_inject_weight.data_ptr() : w_down.data_ptr(), hcks::kW,
                                        hcks::kHC, hcks::kW * 2, hcks::kBK, hcks::kNQ);
    const auto params = hcks::Params{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .partials = static_cast<const float*>(partials.data_ptr()),
        .norm_weight = norm_weight.data_ptr(),
        .next_inject_weight = kNextGate ? next_inject_weight.data_ptr() : nullptr,
        .output = output.data_ptr(),
        .normed_output = kWriteNormed ? normed_output.data_ptr() : nullptr,
        .next_partials = kNextGate ? static_cast<float*>(next_partials.data_ptr()) : nullptr,
        .mixed = mixed.data_ptr(),
        .ws_p = static_cast<float*>(ws_p.data_ptr()),
        .ws_ss = static_cast<float*>(ws_ss.data_ptr()),
        .ws_q = static_cast<float*>(ws_q.data_ptr()),
        .ws_t = ws_t.data_ptr(),
        .ctr = static_cast<int*>(ctr.data_ptr()),
        .dbg = use_dbg ? static_cast<long long*>(dbg.data_ptr()) : nullptr,
        .rows = static_cast<int>(rows),
        .flags = static_cast<int>(flags),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(static_cast<uint32_t>(grid), hcks::kThreads, dev, kSmem)
        .config({.use_pdl = kUsePDL})(kernel, wd_map, wu_map, t_map, inj_map, params);
  }
};

}  // namespace sglang
'''
