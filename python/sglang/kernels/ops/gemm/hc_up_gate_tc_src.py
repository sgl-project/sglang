"""CUDA source of the prefill HC mix up + gated mean on tcgen05 (see hc_up_gate_tc.py).

Kept as a Python string so the module ships with the Python sources; the loader writes it to a
content-addressed file under the JIT cache root before compiling. The PTX wrappers are hc_boundary_ks_src's.
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
#error "hc_up_gate_tc requires SM100-class tensor cores"
#endif

namespace sglang {

namespace hcug {

constexpr int kHC = 4;
constexpr int kH = 2560;
constexpr int kW = kHC * kH;
constexpr int kL = 320;
constexpr int kBM = 128;                // token rows per tile: the MMA M (TMEM lane = row)
constexpr int kBJ = 64;                 // hidden columns per tile
constexpr int kBN = kHC * kBJ;          // 256 permuted logits per tile: the MMA N
constexpr int kNJ = kH / kBJ;           // 40 column tiles
constexpr int kBK = 64;                 // K per 128B-swizzled box
constexpr int kKB = kL / kBK;           // 5
constexpr int kTBox = kBM * 128;        // one 64-column box of the t tile (16 KB)
constexpr int kWBox = kBN * 128;        // one 64-column box of the W_up tile (32 KB)
constexpr int kStage = kTBox + kWBox;   // one K step: t box + W_up box
constexpr int kStages = 2;
constexpr int kXBox = kBM * 128;        // one branch's [128 rows x 64 columns] of the normalized streams (16 KB)
constexpr int kXBuf = kHC * kXBox;      // the tile's normalized streams, 4 branches (64 KB)
constexpr int kEpiWarps = 16;           // four per TMEM lane quarter, each a quarter of the tile's columns
constexpr int kThreads = 32 * (2 + kEpiWarps);
constexpr int kTmemCols = 512;          // two accumulators of 256 columns
constexpr int kSmemCap = 232448;

struct Params {
  void* out;          // [M, H]
  int rows;
};

// Kernel modes (template): 0 bitwise to the compiled chain (Triton's ex2 + div.full sigmoid, products summed as
// fadd(fmul, fmul) then two fma); 4 the sigmoid as 0.5 tanh(x / 2) + 0.5 (one MUFU op, not bitwise).
constexpr int kModeTanh = 4;

struct alignas(1024) Smem {
  alignas(1024) uint8_t ab[kStages * kStage];  // K steps in flight: t box, then W_up box
  alignas(1024) uint8_t x[2 * kXBuf];          // two tiles' normalized streams
  uint64_t ab_full[kStages];
  uint64_t ab_empty[kStages];
  uint64_t x_full[2];
  uint64_t x_empty[2];
  uint64_t acc_full[2];
  uint64_t acc_empty[2];
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
// out[m, j] = bf16(sum_g sigmoid(bf16(logit[m, 4 j + g])) * x[m, g H + j] / 4) with logit = t @ w_up_perm.T in FP32
// over K ascending (one tcgen05 MMA chain per tile, no split): the compiled chain's GEMM2 + sigmoid gated mean
// without the [M, 4 H] logits in memory. Persistent: CTA c takes tiles [c T / G, (c + 1) T / G) of
// (row tile, column tile), column tile fastest. Warp 0 issues the TMA loads (t and W_up per K step, the tile's
// normalized streams into one of two buffers), warp 1 the MMAs into one of two TMEM accumulators, warps 2-17 the
// gated mean of the other, reading the streams from shared memory (128B-swizzled TMA boxes, conflict-free by row).
template <int kMode>
__global__ __launch_bounds__(kThreads, 1) void hc_up_gate_tc_kernel(
    const __grid_constant__ CUtensorMap t_map,
    const __grid_constant__ CUtensorMap w_map,
    const __grid_constant__ CUtensorMap x_map,
    const Params __grid_constant__ p) {
  using namespace device;
  extern __shared__ __align__(1024) uint8_t ug_smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(ug_smem_raw + ((1024u - (smem_u32(ug_smem_raw) & 1023u)) & 1023u));
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int G = gridDim.x;
  const int R = p.rows;
  const int n_mt = (R + kBM - 1) / kBM;
  const int T = n_mt * kNJ;
  const int tile_begin = static_cast<int>((static_cast<int64_t>(blockIdx.x) * T) / G);
  const int tile_end = static_cast<int>((static_cast<int64_t>(blockIdx.x + 1) * T) / G);

  if (warp == 0) {
    if (lane == 0) {
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        device::ptx::mbar_init(&sm.ab_full[s], 1);
        device::ptx::mbar_init(&sm.ab_empty[s], 1);
      }
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        device::ptx::mbar_init(&sm.x_full[b], 1);
        device::ptx::mbar_init(&sm.x_empty[b], kEpiWarps);
        device::ptx::mbar_init(&sm.acc_full[b], 1);
        device::ptx::mbar_init(&sm.acc_empty[b], kEpiWarps);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      prefetch_tensormap(&t_map);
      prefetch_tensormap(&w_map);
      prefetch_tensormap(&x_map);
    }
    __syncwarp();
  } else if (warp == 1) {
    tc_alloc(smem_u32(&sm.taddr), kTmemCols);
    tc_relinquish();
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t taddr = sm.taddr;
  device::PDLWaitPrimary<true>();

  if (warp == 0) {
    if (lane == 0) {
      int s = 0, xb = 0;
      uint32_t sph = 0, xph = 0;
      for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int mt = tile / kNJ;
        const int jt = tile % kNJ;
        {
          // The tile's streams: its buffer is free once the gated mean two tiles back has read it.
          device::ptx::mbar_wait_parity(&sm.x_empty[xb], xph ^ 1);
          device::ptx::mbar_arrive_expect_tx(&sm.x_full[xb], kXBuf);
#pragma unroll
          for (int g = 0; g < kHC; ++g) {
            tma_2d(smem_u32(sm.x + xb * kXBuf + g * kXBox), &x_map, g * kH + jt * kBJ, mt * kBM, &sm.x_full[xb]);
          }
          if (++xb == 2) {
            xb = 0;
            xph ^= 1;
          }
        }
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb) {
          device::ptx::mbar_wait_parity(&sm.ab_empty[s], sph ^ 1);
          device::ptx::mbar_arrive_expect_tx(&sm.ab_full[s], kStage);
          tma_2d(smem_u32(sm.ab + s * kStage), &t_map, kb * kBK, mt * kBM, &sm.ab_full[s]);
          tma_2d(smem_u32(sm.ab + s * kStage + kTBox), &w_map, kb * kBK, jt * kBN, &sm.ab_full[s]);
          if (++s == kStages) {
            s = 0;
            sph ^= 1;
          }
        }
      }
    }
  } else if (warp == 1) {
    if (lane == 0) {
      int s = 0, ab = 0;
      uint32_t sph = 0, aph = 0;
      constexpr uint32_t idesc = idesc_bf16(kBM, kBN);
      for (int tile = tile_begin; tile < tile_end; ++tile) {
        device::ptx::mbar_wait_parity(&sm.acc_empty[ab], aph ^ 1);
        tc_fence_after();
        const uint32_t d = taddr + ab * kBN;
#pragma unroll
        for (int kb = 0; kb < kKB; ++kb) {
          device::ptx::mbar_wait_parity(&sm.ab_full[s], sph);
          tc_fence_after();
          const uint64_t dt = smem_desc(smem_u32(sm.ab + s * kStage));
          const uint64_t dw = smem_desc(smem_u32(sm.ab + s * kStage + kTBox));
#pragma unroll
          for (int k = 0; k < kBK / 16; ++k) tc_mma(d, dt + 2 * k, dw + 2 * k, idesc, (kb | k) != 0);
          tc_commit(&sm.ab_empty[s]);
          if (++s == kStages) {
            s = 0;
            sph ^= 1;
          }
        }
        tc_commit(&sm.acc_full[ab]);
        if (++ab == 2) {
          ab = 0;
          aph ^= 1;
        }
      }
    }
  } else {
    // Epilogue: TMEM lane quarter warp % 4 (rows 32 sub + lane of the tile), logit columns [64 part, + 64) =
    // hidden columns [16 part, + 16) of the tile, 2 chunks of 8 (16-byte chunks 2 part + q of each stream row).
    const int sub = warp & 3;
    const int part = (warp - 2) >> 2;
    const int r = sub * 32 + lane;
    bf16_t* out = static_cast<bf16_t*>(p.out);
    int ab = 0;
    uint32_t aph = 0;
    for (int tile = tile_begin; tile < tile_end; ++tile) {
      const int mt = tile / kNJ;
      const int jt = tile % kNJ;
      const int m = mt * kBM + r;
      const bool live = m < R;
      const int j0 = jt * kBJ + 16 * part;
      device::ptx::mbar_wait_parity(&sm.acc_full[ab], aph);
      tc_fence_after();
      const uint8_t* xr = sm.x + ab * kXBuf + r * 128;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        uint32_t v[32];
        tc_ld_x32(taddr + ab * kBN + part * (kBN / 4) + 32 * q + ((sub * 32) << 16), v);
        tc_wait_ld();
        if (q == 1) {
          // This warp's columns of the accumulator are in registers: release it to the MMA warp.
          tc_fence_before();
          __syncwarp();
          if (lane == 0) device::ptx::mbar_arrive(&sm.acc_empty[ab]);
        }
        uint4 xv[kHC];
        {
          if (q == 0) {
            device::ptx::mbar_wait_parity(&sm.x_full[ab], aph);
            // mbar_wait_parity / mbar_arrive carry no memory clobber: keep the shared-memory reads between them.
            asm volatile("" ::: "memory");
          }
#pragma unroll
          for (int g = 0; g < kHC; ++g) {
            xv[g] = *reinterpret_cast<const uint4*>(xr + g * kXBox + (((2 * part + q) ^ (r & 7)) << 4));
          }
          if (q == 1) {
            // The next TMA into this buffer is an async-proxy write after these generic reads.
            fence_proxy_async_shared();
            __syncwarp();
            if (lane == 0) device::ptx::mbar_arrive(&sm.x_empty[ab]);
          }
        }
        if (live) {
          uint4 o;
          bf16_t* ob = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float sg[kHC], xf[kHC];
#pragma unroll
            for (int g = 0; g < kHC; ++g) {
              const float logit = round_bf16(__uint_as_float(v[4 * e + g]));
              if constexpr (kMode == kModeTanh) {
                sg[g] = sigmoid_tanh(logit);
              } else {
                sg[g] = sigmoid_triton(logit);
              }
              xf[g] = __bfloat162float(reinterpret_cast<const bf16_t*>(&xv[g])[e]);
            }
            float total = __fadd_rn(__fmul_rn(sg[0], xf[0]), __fmul_rn(sg[1], xf[1]));
            total = __fmaf_rn(sg[2], xf[2], total);
            total = __fmaf_rn(sg[3], xf[3], total);
            ob[e] = __float2bfloat16_rn(__fmul_rn(total, 0.25f));
          }
          *reinterpret_cast<uint4*>(out + static_cast<int64_t>(m) * kH + j0 + 8 * q) = o;
        }
      }
      if (++ab == 2) {
        ab = 0;
        aph ^= 1;
      }
    }
  }
  device::PDLTriggerSecondary<true>();
  __syncthreads();
  if (warp == 1) {
    tc_fence_after();
    tc_dealloc(taddr, kTmemCols);
  }
}

// ---------------------------------------------------------------------------------------------- host
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
  host::RuntimeCheck(res == CUDA_SUCCESS, "hc_up_gate_tc: cuTensorMapEncodeTiled failed");
  return map;
}


}  // namespace hcug

template <int kMode>
struct HcUpGateTcKernel {
  static constexpr auto kernel = hcug::hc_up_gate_tc_kernel<kMode>;
  static constexpr int kSmem = sizeof(hcug::Smem) + 1024;
  static_assert(kSmem <= hcug::kSmemCap, "shared memory");

  static void run(const tvm::ffi::TensorView t,
                  const tvm::ffi::TensorView w_up,
                  const tvm::ffi::TensorView x,
                  const tvm::ffi::TensorView out,
                  int64_t num_ctas) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto LDU = SymbolicSize{"ldu"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, hcug::kL}).with_dtype<bf16_t>().with_device(device).verify(t);
    TensorMatcher({hcug::kW, LDU}).with_dtype<bf16_t>().with_device(device).verify(w_up);
    TensorMatcher({M, hcug::kW}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({M, hcug::kH}).with_dtype<bf16_t>().with_device(device).verify(out);
    const auto rows = static_cast<int64_t>(M.unwrap());
    const auto ldu = static_cast<int64_t>(LDU.unwrap());
    RuntimeCheck(ldu >= hcug::kL && ldu % 8 == 0, "w_up must hold >= 320 columns, 16-byte rows");
    if (rows == 0) return;
    static bool attr_set = false;
    if (!attr_set) {
      RuntimeCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) == cudaSuccess,
                   "hc_up_gate_tc: smem attribute");
      attr_set = true;
    }
    const int64_t tiles = (rows + hcug::kBM - 1) / hcug::kBM * hcug::kNJ;
    const int64_t grid = num_ctas < tiles ? num_ctas : tiles;
    const auto t_map = hcug::make_map(t.data_ptr(), hcug::kL, rows, hcug::kL * 2, hcug::kBK, hcug::kBM);
    const auto w_map = hcug::make_map(w_up.data_ptr(), ldu, hcug::kW, ldu * 2, hcug::kBK, hcug::kBN);
    const auto x_map = hcug::make_map(x.data_ptr(), hcug::kW, rows, hcug::kW * 2, hcug::kBK, hcug::kBM);
    const auto params = hcug::Params{
        .out = out.data_ptr(),
        .rows = static_cast<int>(rows),
    };
    LaunchKernel(static_cast<uint32_t>(grid), hcug::kThreads, device.unwrap(), kSmem)
        .config({.use_pdl = true})(kernel, t_map, w_map, x_map, params);
  }
};

}  // namespace sglang
'''
