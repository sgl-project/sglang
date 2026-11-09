// HiCache load-back of an NVFP4 host copy into an FP8 device KV pool, in one pass
// (sglang.srt.mem_cache.pool_host.mha_fp8_nvfp4, SGLANG_HICACHE_FP4_HOST_TMA_LOAD).
//
// Host layout (page_first, per K and per V): packed E2M1 rows [tokens][layers][kRow / 2] (element 2j in
// the low nibble of byte j) and E4M3 block scales [tokens][layers][kRow / 16], one per 16 values. The
// device rows are FP8 [tokens][kRow].
//
// Copy: the TMA ring of hicache_tma.cuh (sgl-project/sglang#40278). One loader warp fills
// shared-memory stages with bulk copies, a whole page run of rows as one 3D tensor-map box for the
// packed bytes and one for the scales (one bulk op per row otherwise), so the host loads stay in flight
// without registers. Store warps dequantize each stage from shared memory straight into the device
// rows: a chunk is read from host once and written to the device once.
//
// Dequantization is a table lookup. An FP8 value restored from NVFP4 depends only on its E2M1 nibble
// and its block's E4M3 scale, so a 256 x 16 byte table, built on the GPU by the reference load
// (mha_fp8_nvfp4._load_tile: cvt.rn.f16x2.e2m1x2, mul.rn.f16x2 by the scale, cvt.rn.satfinite.e4m3x2)
// and checked by the caller on all 65,536 (scale, byte) pairs, gives the same bytes. One prmt looks up
// four nibbles on the integer pipe; the reference's float conversions run on the narrower conversion
// pipe, which is what held the Triton load to ~19 GB/s of FP8 output per SM.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace sglang {

namespace device::nvfp4_load {

using ptx::to_shared;

// global -> shared::cta, completion counted on `bar` (arm with mbar_arrive_expect_tx).
SGL_DEVICE void bulk_g2s(void* dst_smem, const void* src_gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   to_shared(dst_smem)),
               "l"(src_gmem),
               "r"(bytes),
               "r"(to_shared(bar))
               : "memory");
}

// 3D tiled tensor-map box at element coords (x, y, z) -> shared::cta.
SGL_DEVICE void box_3d_g2s(void* dst_smem, const CUtensorMap* map, int32_t x, int32_t y, int32_t z, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];" ::"r"(to_shared(dst_smem)),
      "l"(map),
      "r"(x),
      "r"(y),
      "r"(z),
      "r"(to_shared(bar))
      : "memory");
}

SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Barrier 1 among the store warps only (barrier 0 is __syncthreads).
SGL_DEVICE void store_warps_sync(uint32_t threads) {
  asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
}

// prmt.b32 in its generic mode: byte i of the result is byte (c >> 4i) & 7 of {b, a}, or, when bit
// (4i + 3) of c is set, that byte's bit 7 replicated over all eight bits. Only c[15:0] is read.
SGL_DEVICE uint32_t prmt(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Eight E2M1 nibbles (four packed bytes, element 2j in the low nibble of byte j) -> eight FP8 bytes in
// element order, through one scale's table: t.x, t.y = nibbles 0-7 (sign clear), t.z, t.w = nibbles
// 8-15 (sign set). Returns elements 0-3 in .x and 4-7 in .y.
SGL_DEVICE uint2 dequant8(uint32_t x, uint4 t) {
  const uint32_t mag_lo = x & 0x7777u;           // nibbles 0-3 without their sign bit
  const uint32_t mag_hi = (x >> 16) & 0x7777u;   // nibbles 4-7
  // A sign mask per output byte: the sign of nibble 2k (bit 3 of byte k) is bit 7 of byte k of x << 4,
  // the sign of nibble 2k + 1 is bit 7 of byte k of x; prmt's sign mode spreads it over the byte.
  const uint32_t up = x << 4;
  const uint32_t neg_lo = prmt(x, up, 0x9D8Cu);  // bytes <- sign(n0), sign(n1), sign(n2), sign(n3)
  const uint32_t neg_hi = prmt(x, up, 0xBFAEu);  // bytes <- sign(n4), sign(n5), sign(n6), sign(n7)
  const uint32_t pos_lo = prmt(t.x, t.y, mag_lo), negv_lo = prmt(t.z, t.w, mag_lo);
  const uint32_t pos_hi = prmt(t.x, t.y, mag_hi), negv_hi = prmt(t.z, t.w, mag_hi);
  return make_uint2((pos_lo & ~neg_lo) | (negv_lo & neg_lo), (pos_hi & ~neg_hi) | (negv_hi & neg_hi));
}

}  // namespace device::nvfp4_load

// Rows per chunk at most: one 64-token page, so a page run is one box per buffer.
inline constexpr uint32_t kNvfp4LoadMaxRows = 64;
inline constexpr uint32_t kNvfp4LoadRowsPerLane = kNvfp4LoadMaxRows / device::kWarpThreads;

struct Nvfp4LoadParams {
  // [tokens][layers][row bytes / 8] uint64 views of the host buffers: K packed, V packed, K scales,
  // V scales; a box is (row bytes / 8, 1, rows_per_chunk).
  CUtensorMap maps[4];
  const void* src[4];  // the same buffers' base pointers, for chunks that are not one run of rows
  void* dst[2];        // this layer's device FP8 rows, K and V
  const int64_t* host_idx;
  const int64_t* dev_idx;
  const void* lut;     // 256 x 16 B on the device: lut[scale][nibble]
  int64_t dst_stride;  // bytes between device rows
  uint32_t length;
  uint32_t layer;
  uint32_t layers;
  uint32_t rows_per_chunk;  // a power of two <= 64 that divides the page size
};

template <uint32_t kRow, uint32_t kNumStages>
struct Nvfp4LoadSmem {
  alignas(128) uint8_t data[kNumStages][kNvfp4LoadMaxRows * (kRow / 2)];
  alignas(128) uint8_t scale[kNumStages][kNvfp4LoadMaxRows * (kRow / 16)];
  uint4 lut[256];
  int64_t dst_idx[kNumStages][kNvfp4LoadMaxRows];
  uint64_t full[kNumStages];   // loader -> storers: stage filled (count 1)
  uint64_t empty[kNumStages];  // storers -> loader: stage drained (count kStoreWarps)
};

template <uint32_t kRow, uint32_t kNumStages, uint32_t kStoreWarps>
__global__ void __launch_bounds__((1 + kStoreWarps) * device::kWarpThreads, 1)
    hicache_nvfp4_load_kernel(const __grid_constant__ Nvfp4LoadParams p) {
#if SGL_ARCH_HOPPER_OR_GREATER
  using namespace device;
  using namespace device::nvfp4_load;
  using Smem = Nvfp4LoadSmem<kRow, kNumStages>;
  constexpr uint32_t kPacked = kRow / 2;   // E2M1 bytes per row
  constexpr uint32_t kBlocks = kRow / 16;  // 16-value blocks (= scales) per row; 8 packed bytes each
  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem*>(smem_raw);

  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;

  if (threadIdx.x == 0) {
    for (uint32_t s = 0; s < kNumStages; ++s) {
      ptx::mbar_init(&smem.full[s], 1);
      ptx::mbar_init(&smem.empty[s], kStoreWarps);
    }
    fence_mbarrier_init();
  }
  __syncthreads();

  const uint32_t rows_per_chunk = p.rows_per_chunk;
  const uint32_t token_chunks = div_ceil(p.length, rows_per_chunk);
  const uint32_t num_chunks = 2u * token_chunks;  // K chunks, then V chunks

  if (warp == 0) {
    // ---- loader: host -> smem ring via TMA (index prefetch as in hicache_tma.cuh)
    struct Prefetch {
      int64_t src[kNvfp4LoadRowsPerLane];  // rows lane, lane + 32
      int64_t dst[kNvfp4LoadRowsPerLane];
    };
    auto prefetch = [&](uint32_t chunk) {
      const uint32_t t0 = (chunk % token_chunks) * rows_per_chunk;
      const uint32_t rows = min(rows_per_chunk, p.length - t0);
      Prefetch pf;
#pragma unroll
      for (uint32_t k = 0; k < kNvfp4LoadRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        pf.src[k] = r < rows ? p.host_idx[t0 + r] : 0;
        pf.dst[k] = r < rows ? p.dev_idx[t0 + r] : 0;
      }
      return pf;
    };

    uint32_t chunk = blockIdx.x;
    Prefetch next = chunk < num_chunks ? prefetch(chunk) : Prefetch{};
    for (uint32_t it = 0; chunk < num_chunks; chunk += gridDim.x, ++it) {
      const uint32_t t0 = (chunk % token_chunks) * rows_per_chunk;
      const uint32_t rows = min(rows_per_chunk, p.length - t0);
      const uint32_t kv = chunk / token_chunks;
      const Prefetch cur = next;
      if (chunk + gridDim.x < num_chunks) next = prefetch(chunk + gridDim.x);

      // A run: the chunk's host rows are consecutive tokens (whole pages in order).
      const int64_t first = __shfl_sync(warp::kFullMask, cur.src[0], 0);
      bool run = true;
#pragma unroll
      for (uint32_t k = 0; k < kNvfp4LoadRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        run &= r >= rows || cur.src[k] == first + static_cast<int64_t>(r);
      }
      run = __all_sync(warp::kFullMask, run);

      const uint32_t s = it % kNumStages;
      ptx::mbar_wait_parity(&smem.empty[s], ((it / kNumStages) & 1) ^ 1);  // a fresh ring passes
#pragma unroll
      for (uint32_t k = 0; k < kNvfp4LoadRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        if (r < rows) smem.dst_idx[s][r] = cur.dst[k];
      }
      __syncwarp();  // every lane's dst_idx before lane 0's arrive releases the stage
      // A box always lands rows_per_chunk rows (rows past the tensor are zero-filled).
      const uint32_t tx_rows = run ? rows_per_chunk : rows;
      if (lane == 0) ptx::mbar_arrive_expect_tx(&smem.full[s], tx_rows * (kPacked + kBlocks));
      __syncwarp();

      if (run) {
        if (lane == 0) {
          const int32_t z = static_cast<int32_t>(first), y = static_cast<int32_t>(p.layer);
          box_3d_g2s(smem.data[s], &p.maps[kv], 0, y, z, &smem.full[s]);
          box_3d_g2s(smem.scale[s], &p.maps[2 + kv], 0, y, z, &smem.full[s]);
        }
      } else {
#pragma unroll
        for (uint32_t k = 0; k < kNvfp4LoadRowsPerLane; ++k) {
          const uint32_t r = lane + k * kWarpThreads;
          if (r < rows) {
            const int64_t row = cur.src[k] * p.layers + p.layer;
            bulk_g2s(smem.data[s] + r * kPacked, pointer::offset(p.src[kv], row * kPacked), kPacked, &smem.full[s]);
            bulk_g2s(
                smem.scale[s] + r * kBlocks, pointer::offset(p.src[2 + kv], row * kBlocks), kBlocks, &smem.full[s]);
          }
        }
      }
    }
  } else {
    // ---- store warps: smem stage -> dequantize -> device FP8 rows. Unit u of a chunk is block
    // u % kBlocks of row u / kBlocks: 8 packed bytes at data + 8u, its scale at scale + u, 16 FP8 bytes out.
    constexpr uint32_t kStoreThreads = kStoreWarps * kWarpThreads;
    constexpr uint32_t kFullUnits = kNvfp4LoadMaxRows * kBlocks;
    constexpr uint32_t kUnroll = 4;
    static_assert(kFullUnits % (kStoreThreads * kUnroll) == 0, "a full chunk splits evenly over the store threads");
    const uint32_t tid = threadIdx.x - kWarpThreads;

    for (uint32_t i = tid; i < 256; i += kStoreThreads) {
      smem.lut[i] = static_cast<const uint4*>(p.lut)[i];
    }
    store_warps_sync(kStoreThreads);

    auto unit = [&](const uint8_t* data, const uint8_t* scale, const int64_t* dst_idx, void* dst_base, uint32_t u,
                    uint2& x, uint32_t& sc, uint4*& out) {
      x = *reinterpret_cast<const uint2*>(data + u * 8);
      sc = scale[u];
      out = static_cast<uint4*>(
          pointer::offset(dst_base, dst_idx[u / kBlocks] * p.dst_stride + static_cast<int64_t>(u % kBlocks) * 16));
    };
    auto convert = [&](uint2 x, uint32_t sc) {
      const uint4 t = smem.lut[sc];
      const uint2 lo = dequant8(x.x, t), hi = dequant8(x.y, t);
      return make_uint4(lo.x, lo.y, hi.x, hi.y);
    };

    uint32_t it = 0;
    for (uint32_t chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x, ++it) {
      const uint32_t t0 = (chunk % token_chunks) * rows_per_chunk;
      const uint32_t rows = min(rows_per_chunk, p.length - t0);
      void* dst_base = p.dst[chunk / token_chunks];
      const uint32_t s = it % kNumStages;
      const uint8_t* data = smem.data[s];
      const uint8_t* scale = smem.scale[s];
      const int64_t* dst_idx = smem.dst_idx[s];

      ptx::mbar_wait_parity(&smem.full[s], (it / kNumStages) & 1);
      if (rows == kNvfp4LoadMaxRows) {
#pragma unroll
        for (uint32_t i = 0; i < kFullUnits / (kStoreThreads * kUnroll); ++i) {
          const uint32_t u0 = tid + i * kStoreThreads * kUnroll;
          uint2 x[kUnroll];
          uint32_t sc[kUnroll];
          uint4* out[kUnroll];
#pragma unroll
          for (uint32_t k = 0; k < kUnroll; ++k)
            unit(data, scale, dst_idx, dst_base, u0 + k * kStoreThreads, x[k], sc[k], out[k]);
#pragma unroll
          for (uint32_t k = 0; k < kUnroll; ++k)
            __stcs(out[k], convert(x[k], sc[k]));
        }
      } else {
        for (uint32_t u = tid; u < rows * kBlocks; u += kStoreThreads) {
          uint2 x;
          uint32_t sc;
          uint4* out;
          unit(data, scale, dst_idx, dst_base, u, x, sc, out);
          __stcs(out, convert(x, sc));
        }
      }
      __syncwarp();
      if (lane == 0) ptx::mbar_arrive(&smem.empty[s]);
    }
  }
#endif
}

template <uint32_t kRow, uint32_t kNumStages, uint32_t kStoreWarps>
struct HiCacheNvfp4LoadKernel {
  using Smem = Nvfp4LoadSmem<kRow, kNumStages>;
  static_assert(kRow % 256 == 0 && kRow <= 4096, "row of 256..4096 FP8 values: 16 B scale rows, boxes <= 256 elems");
  static_assert(kNumStages >= 2 && kStoreWarps >= 1, "a ring of stages and at least one store warp");
  static constexpr uint32_t kThreads = (1 + kStoreWarps) * device::kWarpThreads;
  static constexpr auto kernel = hicache_nvfp4_load_kernel<kRow, kNumStages, kStoreWarps>;

  // Bytes of the per-pool parameter blob `encode` fills and `launch` reads (a CPU uint8 tensor).
  static int64_t params_bytes() {
    return static_cast<int64_t>(sizeof(Nvfp4LoadParams));
  }

  static bool fits_device(int64_t device_id) {
    int max_smem = 0;
    host::RuntimeDeviceCheck(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, static_cast<int>(device_id)));
    return static_cast<std::size_t>(max_smem) >= sizeof(Smem);
  }

  static auto encode_tiled_fn() -> PFN_cuTensorMapEncodeTiled_v12000 {
    static const auto fn = [] {
      void* sym = nullptr;
      cudaDriverEntryPointQueryResult status;
      host::RuntimeDeviceCheck(
          cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &sym, 12000, cudaEnableDefault, &status));
      host::RuntimeCheck(status == cudaDriverEntryPointSuccess && sym != nullptr, "cuTensorMapEncodeTiled unavailable");
      return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(sym);
    }();
    return fn;
  }

  // [tokens][layers][row_bytes / 8] uint64 elements; a box is one layer's rows_per_chunk consecutive tokens.
  static void encode_map(CUtensorMap* map, uint64_t base, int64_t tokens, int64_t layers, uint32_t row_bytes,
                         uint32_t rows_per_chunk) {
    const cuuint64_t gdim[3] = {row_bytes / 8, static_cast<cuuint64_t>(layers), static_cast<cuuint64_t>(tokens)};
    const cuuint64_t gstride[2] = {row_bytes, static_cast<cuuint64_t>(layers) * row_bytes};
    const cuuint32_t box[3] = {row_bytes / 8, 1, rows_per_chunk};
    const cuuint32_t estride[3] = {1, 1, 1};
    const CUresult res = encode_tiled_fn()(
        map,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        3,
        reinterpret_cast<void*>(base),
        gdim,
        gstride,
        box,
        estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    host::RuntimeCheck(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(res));
  }

  static auto verify_blob(const tvm::ffi::TensorView& blob) -> Nvfp4LoadParams* {
    using namespace host;
    TensorMatcher({static_cast<int64_t>(sizeof(Nvfp4LoadParams))})
        .with_dtype<uint8_t>()
        .with_device<kDLCPU>()
        .verify(blob);
    return static_cast<Nvfp4LoadParams*>(blob.data_ptr());
  }

  // Once per host pool: the four boxes over its host buffers, their bases, the table and the geometry.
  static void encode(
      const tvm::ffi::TensorView blob,
      const int64_t k_data,
      const int64_t v_data,
      const int64_t k_scale,
      const int64_t v_scale,
      const int64_t tokens,
      const int64_t layers,
      const int64_t rows_per_chunk,
      const int64_t lut,
      const int64_t dst_stride) {
    using namespace host;
    RuntimeCheck(tokens > 0 && layers > 0, "NVFP4 load: empty host pool");
    RuntimeCheck(
        rows_per_chunk > 0 && rows_per_chunk <= kNvfp4LoadMaxRows && is_pow2<uint64_t>(rows_per_chunk),
        "NVFP4 load: rows per chunk must be a power of two <= 64");
    RuntimeCheck(dst_stride >= kRow && dst_stride % 16 == 0, "NVFP4 load: device rows must be 16 B aligned");
    for (const int64_t ptr : {k_data, v_data, k_scale, v_scale, lut})
      RuntimeCheck(ptr % 16 == 0, "NVFP4 load: buffers must be 16 B aligned");
    Nvfp4LoadParams params;
    std::memset(&params, 0, sizeof(params));
    const int64_t data[4] = {k_data, v_data, k_scale, v_scale};
    for (int i = 0; i < 4; ++i) {
      encode_map(
          &params.maps[i],
          static_cast<uint64_t>(data[i]),
          tokens,
          layers,
          i < 2 ? kRow / 2 : kRow / 16,
          static_cast<uint32_t>(rows_per_chunk));
      params.src[i] = reinterpret_cast<const void*>(data[i]);
    }
    params.lut = reinterpret_cast<const void*>(lut);
    params.dst_stride = dst_stride;
    params.layers = static_cast<uint32_t>(layers);
    params.rows_per_chunk = static_cast<uint32_t>(rows_per_chunk);
    std::memcpy(verify_blob(blob), &params, sizeof(params));
  }

  // One layer's load-back: K and V of `length` tokens on `stream`, at most `block_quota` CTAs.
  static void launch(
      const tvm::ffi::TensorView blob,
      const tvm::ffi::TensorView host_idx,
      const tvm::ffi::TensorView dev_idx,
      const int64_t k_dst,
      const int64_t v_dst,
      const int64_t layer,
      const int64_t block_quota,
      const int64_t stream) {
    using namespace host;
    auto L = SymbolicSize{"indices length"};
    auto device_ = SymbolicDevice{};
    TensorMatcher({L}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(host_idx).verify(dev_idx);
    Nvfp4LoadParams params;
    std::memcpy(&params, verify_blob(blob), sizeof(params));
    RuntimeCheck(layer >= 0 && layer < params.layers, "NVFP4 load: layer out of range");
    RuntimeCheck(block_quota > 0, "NVFP4 load: block quota must be positive");
    RuntimeCheck(k_dst % 16 == 0 && v_dst % 16 == 0, "NVFP4 load: device rows must be 16 B aligned");
    const auto length = L.unwrap();
    if (length == 0) return;
    RuntimeCheck(length <= 0xFFFFFFFFll, "NVFP4 load: too many tokens");
    params.host_idx = static_cast<const int64_t*>(host_idx.data_ptr());
    params.dev_idx = static_cast<const int64_t*>(dev_idx.data_ptr());
    params.dst[0] = reinterpret_cast<void*>(k_dst);
    params.dst[1] = reinterpret_cast<void*>(v_dst);
    params.length = static_cast<uint32_t>(length);
    params.layer = static_cast<uint32_t>(layer);

    const auto device = device_.unwrap();
    constexpr std::size_t kSmemBytes = sizeof(Smem);
    runtime::init_per_device(device, [=] { runtime::set_max_dynamic_smem(kernel, kSmemBytes); });
    const uint32_t chunks = 2u * static_cast<uint32_t>(div_ceil(length, static_cast<int64_t>(params.rows_per_chunk)));
    const uint32_t grid = std::min<uint32_t>(chunks, static_cast<uint32_t>(block_quota));
    LaunchKernel(grid, kThreads, reinterpret_cast<cudaStream_t>(stream), kSmemBytes)(kernel, params);
  }
};

}  // namespace sglang
