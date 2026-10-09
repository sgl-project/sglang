// HiCache host<->device KV transfer staged through shared memory by the TMA
// bulk-copy engine (sm_90+).
//
// One CTA owns a ring of shared-memory stages. A single loader warp fills
// stages with `cp.async.bulk` (global -> shared, completion counted on an
// mbarrier), which keeps the whole ring in flight with no registers or issue
// slots; the register-staging kernel in hicache.cuh cannot hold enough host
// loads in flight per SM for that. Store warps drain filled stages. Row size is
// a runtime parameter (multiple of 16 B), so one compiled module serves every
// KV shape.
//
// Two hardware facts fix the shape of the kernel (numbers in the PR): every SM
// has a fixed write port to L2, so one CTA cannot exceed it and the block
// quota decides how much of the host link is used; and the TMA unit processes
// bulk ops at a fixed per-op rate, so a source run must move as one op
// (contiguous run -> one 1D bulk copy; strided page run -> one 2D tensor-map
// box), never one op per row. Revisit both if a future part widens the SM
// write port or the TMA op rate.
//
// Work unit ("chunk") = (buffer K|V, layer, run of consecutive positions of the
// index arrays). The loader prefetches the next chunk's indices before blocking
// on the ring so their latency overlaps the wait, and stashes the destination
// indices in smem for the store warps.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>

namespace sglang {

namespace device::ptx {

// global -> shared::cta, completion counted on `bar` (arm with mbar_arrive_expect_tx).
SGL_DEVICE void bulk_g2s(void* dst_smem, const void* src_gmem, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   to_shared(dst_smem)),
               "l"(src_gmem),
               "r"(bytes),
               "r"(to_shared(bar))
               : "memory");
}

// 2D tiled tensor-map box at element coords (x, y) -> shared::cta.
SGL_DEVICE void bulk_tensor_2d_g2s(void* dst_smem, const CUtensorMap* map, int32_t x, int32_t y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
          "r"(to_shared(dst_smem)),
      "l"(map),
      "r"(x),
      "r"(y),
      "r"(to_shared(bar))
      : "memory");
}

// shared::cta -> global, tracked by the issuing thread's bulk group.
SGL_DEVICE void bulk_s2g(void* dst_gmem, const void* src_smem, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_gmem),
               "r"(to_shared(src_smem)),
               "r"(bytes)
               : "memory");
}

SGL_DEVICE void bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// Block until every committed bulk group has finished reading its smem source.
SGL_DEVICE void bulk_wait_group_read_all() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// Same, but the most recent group may still be reading.
SGL_DEVICE void bulk_wait_group_read_one() {
  asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
}

// Block until every committed bulk group has fully landed in global memory.
SGL_DEVICE void bulk_wait_group_all() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

SGL_DEVICE void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

}  // namespace device::ptx

struct HicacheTmaParams {
  // Either a direct base pointer (`*_is_table == false`) or a device array of
  // `num_layers` uint64 base pointers. `v_*` is unused when `has_v == false` (MLA).
  const void* __restrict__ k_src;
  const void* __restrict__ v_src;
  void* __restrict__ k_dst;
  void* __restrict__ v_dst;
  const void* __restrict__ indices_src;
  const void* __restrict__ indices_dst;
  int64_t src_stride;  // bytes between consecutive token rows
  int64_t dst_stride;
  uint32_t row_bytes;  // bytes copied per token row, multiple of 16
  uint32_t length;     // number of token indices
  uint32_t num_layers;
  uint64_t units_per_row_magic;  // ceil(2^32 / (row_bytes / 16)); see store loop
  bool src_is_table;
  bool dst_is_table;
  bool has_v;
  // Strided source rows ([K, V] views): one box per chunk instead of one op per row.
  bool has_src_map;
  CUtensorMap src_map[2];
};

// Rows of one chunk are spread over the loader lanes; each lane prefetches at
// most this many row indices, which bounds rows per chunk to 32x this.
inline constexpr uint32_t kHicacheTmaRowsPerLane = 4;
inline constexpr uint32_t kHicacheTmaMaxRows = kHicacheTmaRowsPerLane * device::kWarpThreads;
// Tensor-map boxes are limited to 256 elements per dimension; rows are mapped as
// 8-byte elements so this is the widest row a 2D box can cover.
inline constexpr uint32_t kHicacheTmaMaxMapRowBytes = 256 * 8;

// Rows per chunk: largest power of two that fits the stage, so chunks never
// straddle a (power-of-two) page and a page run stays one bulk op.
__host__ __device__ constexpr uint32_t hicache_tma_rows_per_chunk(uint32_t stage_bytes, uint32_t row_bytes) {
  uint32_t rows = 1;
  while (rows * 2 <= stage_bytes / row_bytes && rows * 2 <= kHicacheTmaMaxRows)
    rows *= 2;
  return rows;
}

template <uint32_t kStageBytes, uint32_t kNumStages>
struct HicacheTmaSmem {
  alignas(128) uint8_t stages[kNumStages][kStageBytes];
  int64_t dst_idx[kNumStages][kHicacheTmaMaxRows];  // destination row indices of the staged chunk
  uint32_t dst_run[kNumStages];                     // destination rows form one contiguous span
  uint64_t full[kNumStages];                        // loader -> storers: stage filled (count 1)
  uint64_t empty[kNumStages];                       // storers -> loader: stage drained (count kStoreWarps)
};

template <typename T, uint32_t kStageBytes, uint32_t kNumStages, uint32_t kStoreWarps>
__global__ void __launch_bounds__((1 + kStoreWarps) * device::kWarpThreads, 1)
    hicache_tma_transfer_kernel(const __grid_constant__ HicacheTmaParams p) {
#if SGL_ARCH_HOPPER_OR_GREATER
  using namespace device;
  using Smem = HicacheTmaSmem<kStageBytes, kNumStages>;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  auto& smem = *reinterpret_cast<Smem*>(smem_raw);

  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;

  if (threadIdx.x == 0) {
    for (uint32_t s = 0; s < kNumStages; ++s) {
      ptx::mbar_init(&smem.full[s], 1);
      ptx::mbar_init(&smem.empty[s], kStoreWarps);
    }
    ptx::fence_mbarrier_init();
  }
  __syncthreads();

  const uint32_t rows_per_chunk = hicache_tma_rows_per_chunk(kStageBytes, p.row_bytes);
  const uint32_t token_chunks = div_ceil(p.length, rows_per_chunk);
  const uint32_t num_chunks = (p.has_v ? 2u : 1u) * p.num_layers * token_chunks;
  const T* idx_src = static_cast<const T*>(p.indices_src);
  const T* idx_dst = static_cast<const T*>(p.indices_dst);

  struct ChunkInfo {
    uint32_t t0;    // first index position
    uint32_t rows;  // rows in this chunk
    uint32_t layer;
    bool is_v;
  };
  auto describe = [&](uint32_t chunk) {
    const uint32_t tc = chunk % token_chunks;
    const uint32_t rest = chunk / token_chunks;
    const uint32_t t0 = tc * rows_per_chunk;
    return ChunkInfo{t0, min(rows_per_chunk, p.length - t0), rest % p.num_layers, (rest / p.num_layers) != 0};
  };
  auto base_ptr = [&](const void* direct_or_table, bool is_table, uint32_t layer) -> const void* {
    return is_table ? reinterpret_cast<const void*>(static_cast<const uint64_t*>(direct_or_table)[layer])
                    : direct_or_table;
  };

  if (warp == 0) {
    // ---- loader: global -> smem ring via TMA bulk copies
    struct Prefetch {
      const void* src_base;
      T src[kHicacheTmaRowsPerLane];  // rows lane, lane + 32, ...
      T dst[kHicacheTmaRowsPerLane];
    };
    auto prefetch = [&](uint32_t chunk) {
      const ChunkInfo c = describe(chunk);
      Prefetch pf;
      pf.src_base = base_ptr(c.is_v ? p.v_src : p.k_src, p.src_is_table, c.layer);
#pragma unroll
      for (uint32_t k = 0; k < kHicacheTmaRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        pf.src[k] = r < c.rows ? idx_src[c.t0 + r] : T{0};
        pf.dst[k] = r < c.rows ? idx_dst[c.t0 + r] : T{0};
      }
      return pf;
    };

    uint32_t chunk = blockIdx.x;
    Prefetch next = chunk < num_chunks ? prefetch(chunk) : Prefetch{};
    for (uint32_t it = 0; chunk < num_chunks; chunk += gridDim.x, ++it) {
      const ChunkInfo c = describe(chunk);
      const Prefetch cur = next;
      if (chunk + gridDim.x < num_chunks) next = prefetch(chunk + gridDim.x);

      // A run: every row sits at first + r (whole pages in order), on either side.
      const T first = __shfl_sync(warp::kFullMask, cur.src[0], 0);
      const T first_dst = __shfl_sync(warp::kFullMask, cur.dst[0], 0);
      bool run = true, run_dst = true;
#pragma unroll
      for (uint32_t k = 0; k < kHicacheTmaRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        run &= r >= c.rows || cur.src[k] == first + static_cast<T>(r);
        run_dst &= r >= c.rows || cur.dst[k] == first_dst + static_cast<T>(r);
      }
      run = __all_sync(warp::kFullMask, run);
      run_dst = __all_sync(warp::kFullMask, run_dst);
      const bool contiguous = run && p.src_stride == p.row_bytes;
      const bool boxed = run && !contiguous && p.has_src_map;

      const uint32_t s = it % kNumStages;
      ptx::mbar_wait_parity(&smem.empty[s], ((it / kNumStages) & 1) ^ 1);  // fresh ring passes
#pragma unroll
      for (uint32_t k = 0; k < kHicacheTmaRowsPerLane; ++k) {
        const uint32_t r = lane + k * kWarpThreads;
        if (r < c.rows) smem.dst_idx[s][r] = static_cast<int64_t>(cur.dst[k]);
      }
      if (lane == 0) smem.dst_run[s] = run_dst && p.dst_stride == p.row_bytes;
      // A box always lands rows_per_chunk rows (out-of-range rows are zero-filled).
      const uint32_t tx_bytes = (boxed ? rows_per_chunk : c.rows) * p.row_bytes;
      if (lane == 0) ptx::mbar_arrive_expect_tx(&smem.full[s], tx_bytes);
      __syncwarp();

      uint8_t* stage = smem.stages[s];
      if (contiguous) {
        if (lane == 0) {
          ptx::bulk_g2s(
              stage,
              pointer::offset(cur.src_base, static_cast<int64_t>(first) * p.src_stride),
              c.rows * p.row_bytes,
              &smem.full[s]);
        }
      } else if (boxed) {
        if (lane == 0) {
          ptx::bulk_tensor_2d_g2s(stage, &p.src_map[c.is_v ? 1 : 0], 0, static_cast<int32_t>(first), &smem.full[s]);
        }
      } else {
#pragma unroll
        for (uint32_t k = 0; k < kHicacheTmaRowsPerLane; ++k) {
          const uint32_t r = lane + k * kWarpThreads;
          if (r < c.rows) {
            ptx::bulk_g2s(
                stage + r * p.row_bytes,
                pointer::offset(cur.src_base, static_cast<int64_t>(cur.src[k]) * p.src_stride),
                p.row_bytes,
                &smem.full[s]);
          }
        }
      }
    }
  } else {
    // ---- storers: smem ring -> global, 16-byte units interleaved across all
    // store threads. Row addressing comes from the loader-staged dst_idx.
    constexpr uint32_t kUnroll = 4;
    constexpr uint32_t kStoreThreads = kStoreWarps * kWarpThreads;
    constexpr uint32_t kUnitsPerIter = kStoreThreads * kUnroll;
    const uint32_t tid = threadIdx.x - kWarpThreads;
    const uint32_t units_per_row = p.row_bytes / 16;

    // unit -> (row, col) without a hardware divide: magic multiply plus a one-step fixup.
    auto locate = [&](uint32_t u, uint32_t& row, uint32_t& col) {
      row = static_cast<uint32_t>((static_cast<uint64_t>(u) * p.units_per_row_magic) >> 32);
      int32_t rem = static_cast<int32_t>(u - row * units_per_row);
      if (rem < 0) {
        --row;
        rem += units_per_row;
      }
      col = static_cast<uint32_t>(rem);
    };
    auto unit_dst = [&](void* dst_base, const int64_t* dst_idx, uint32_t u) -> uint4* {
      uint32_t row, col;
      locate(u, row, col);
      return static_cast<uint4*>(pointer::offset(dst_base, dst_idx[row] * p.dst_stride, col * 16));
    };

    constexpr uint32_t kNoStage = ~0u;
    uint32_t bulk_pending = kNoStage;  // stage of the last bulk store not yet released (warp 0)
    uint32_t it = 0;
    for (uint32_t chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x, ++it) {
      const ChunkInfo c = describe(chunk);
      void* dst_base = const_cast<void*>(base_ptr(c.is_v ? p.v_dst : p.k_dst, p.dst_is_table, c.layer));
      const uint32_t s = it % kNumStages;
      const uint32_t n_units = c.rows * units_per_row;
      const uint32_t n_full = n_units - n_units % kUnitsPerIter;
      const uint4* stage = reinterpret_cast<const uint4*>(smem.stages[s]);
      const int64_t* dst_idx = smem.dst_idx[s];

      ptx::mbar_wait_parity(&smem.full[s], (it / kNumStages) & 1);
      if (smem.dst_run[s]) {
        // Contiguous span: one bulk store from store warp 0 runs at the SM's write
        // port; it releases the previous bulk stage once that stage's smem read is
        // done, keeping two stores in flight. The other store warps have nothing
        // to read and release the stage right away.
        if (tid < kWarpThreads) {
          if (lane == 0) {
            ptx::bulk_s2g(pointer::offset(dst_base, dst_idx[0] * p.dst_stride), stage, n_units * 16);
            ptx::bulk_commit_group();
            if (bulk_pending != kNoStage) {
              ptx::bulk_wait_group_read_one();
              ptx::mbar_arrive(&smem.empty[bulk_pending]);
            }
          }
          bulk_pending = s;
        } else if (lane == 0) {
          ptx::mbar_arrive(&smem.empty[s]);
        }
        continue;
      }
      if (tid < kWarpThreads && bulk_pending != kNoStage) {
        if (lane == 0) {
          ptx::bulk_wait_group_read_all();
          ptx::mbar_arrive(&smem.empty[bulk_pending]);
        }
        bulk_pending = kNoStage;
      }
      for (uint32_t u0 = tid; u0 < n_full; u0 += kUnitsPerIter) {
        uint4 v[kUnroll];
        uint4* dst[kUnroll];
#pragma unroll
        for (uint32_t k = 0; k < kUnroll; ++k) {
          v[k] = stage[u0 + k * kStoreThreads];
          dst[k] = unit_dst(dst_base, dst_idx, u0 + k * kStoreThreads);
        }
#pragma unroll
        for (uint32_t k = 0; k < kUnroll; ++k)
          __stcs(dst[k], v[k]);
      }
      for (uint32_t u = n_full + tid; u < n_units; u += kStoreThreads) {
        __stcs(unit_dst(dst_base, dst_idx, u), stage[u]);
      }
      __syncwarp();
      if (lane == 0) ptx::mbar_arrive(&smem.empty[s]);
    }
    if (tid == 0) ptx::bulk_wait_group_all();  // bulk stores must land before the grid completes
    (void)bulk_pending;                        // the final stage is never reused
  }
#endif
}

template <uint32_t kStageBytes, uint32_t kNumStages, uint32_t kStoreWarps, uint32_t kBlockQuota>
struct HiCacheTmaKernel {
  using Smem = HicacheTmaSmem<kStageBytes, kNumStages>;
  static_assert(kStageBytes % 128 == 0, "stage must stay 128-byte aligned for bulk copies");
  static_assert(kNumStages >= 3 && kStoreWarps >= 1, "two bulk stores in flight plus one loading stage");
  static constexpr uint32_t kThreads = (1 + kStoreWarps) * device::kWarpThreads;

  template <typename T>
  static constexpr auto kernel = hicache_tma_transfer_kernel<T, kStageBytes, kNumStages, kStoreWarps>;

  static uint32_t rows_per_chunk(uint32_t row_bytes) {
    return hicache_tma_rows_per_chunk(kStageBytes, row_bytes);
  }

  // Whether the device can hold the smem ring in one CTA; sm_90+ parts with
  // small opt-in shared memory (consumer Blackwell) must keep the register kernel.
  static bool fits_device(int64_t device_id) {
    int max_smem = 0;
    host::RuntimeDeviceCheck(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, static_cast<int>(device_id)));
    return static_cast<std::size_t>(max_smem) >= sizeof(Smem);
  }

  // ceil(2^32 / units_per_row): (u * magic) >> 32 overestimates u / units_per_row
  // by at most one for the unit counts a stage can hold; the kernel fixes that up.
  static uint64_t units_per_row_magic(uint32_t row_bytes) {
    const uint64_t upr = row_bytes / 16;
    return ((uint64_t{1} << 32) + upr - 1) / upr;
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

  // 2D view of a strided row buffer: [rows][row_bytes / 8] uint64 elements with
  // pitch `stride_bytes`; one box covers rows_per_chunk consecutive rows.
  static void encode_src_map(CUtensorMap* map, const void* base, int64_t num_rows, uint32_t row_bytes, int64_t stride) {
    const cuuint64_t gdim[2] = {row_bytes / 8, static_cast<cuuint64_t>(num_rows)};
    const cuuint64_t gstride[1] = {static_cast<cuuint64_t>(stride)};
    const cuuint32_t box[2] = {row_bytes / 8, rows_per_chunk(row_bytes)};
    const cuuint32_t estride[2] = {1, 1};
    const CUresult res = encode_tiled_fn()(
        map,
        CU_TENSOR_MAP_DATA_TYPE_UINT64,
        2,
        const_cast<void*>(base),
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

  static void launch(const HicacheTmaParams& params, bool use_int32, DLDevice device) {
    using namespace host;
    RuntimeCheck(params.row_bytes > 0 && params.row_bytes % 16 == 0, "HiCache TMA: row bytes must be a multiple of 16");
    RuntimeCheck(params.row_bytes <= kStageBytes, "HiCache TMA: row bytes exceed the smem stage");
    RuntimeCheck(
        params.src_stride % 16 == 0 && params.dst_stride % 16 == 0, "HiCache TMA: strides must be multiples of 16");
    if (params.length == 0 || params.num_layers == 0) return;

    const uint32_t chunks =
        (params.has_v ? 2u : 1u) * params.num_layers * div_ceil(params.length, rows_per_chunk(params.row_bytes));
    constexpr std::size_t kSmemBytes = sizeof(Smem);

    static const bool attr_set = [] {
      for (auto fn : {kernel<int32_t>, kernel<int64_t>}) {
        RuntimeDeviceCheck(
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes)));
      }
      return true;
    }();
    (void)attr_set;
    LaunchKernel(std::min(chunks, kBlockQuota), kThreads, device, kSmemBytes)(
        use_int32 ? kernel<int32_t> : kernel<int64_t>, params);
  }

  // Cache operand viewed as [-1, D] rows; binds row dim, stride and dtype.
  static void verify_cache(
      const tvm::ffi::TensorView& t, host::SymbolicSize& D, host::SymbolicSize& stride, host::SymbolicDType& dtype) {
    using namespace host;
    TensorMatcher({-1, D})  //
        .with_strides({stride, 1})
        .with_dtype(dtype)
        .with_device<kDLGPU, kDLGPUHost, kDLCPU>()
        .verify(t);
  }

  static void verify_indices(
      const tvm::ffi::TensorView& a,
      const tvm::ffi::TensorView& b,
      host::SymbolicSize& L,
      host::SymbolicDType& dtype,
      host::SymbolicDevice& device) {
    using namespace host;
    TensorMatcher({L})  //
        .with_dtype<int32_t, int64_t>(dtype)
        .with_device<kDLGPU>(device)
        .verify(a)
        .verify(b);
  }

  // One layer, direct pointers. `v_*` are ignored when `has_v == false`.
  static void run_one_impl(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src,
      bool has_v) {
    using namespace host;
    auto D = SymbolicSize{"row dim"};
    auto N = SymbolicSize{"src stride"};
    auto M = SymbolicSize{"dst stride"};
    auto L = SymbolicSize{"indices length"};
    auto cache_dtype = SymbolicDType{};
    auto indices_dtype = SymbolicDType{};
    auto indices_device = SymbolicDevice{};

    verify_cache(k_cache_src, D, N, cache_dtype);
    verify_cache(k_cache_dst, D, M, cache_dtype);
    if (has_v) {
      verify_cache(v_cache_src, D, N, cache_dtype);
      verify_cache(v_cache_dst, D, M, cache_dtype);
    }
    verify_indices(indices_src, indices_dst, L, indices_dtype, indices_device);

    const auto dtype_size = dtype_bytes(cache_dtype.unwrap());
    const auto row_bytes = static_cast<uint32_t>(D.unwrap() * dtype_size);
    const auto src_stride = static_cast<int64_t>(N.unwrap() * dtype_size);
    HicacheTmaParams params{
        .k_src = k_cache_src.data_ptr(),
        .v_src = has_v ? v_cache_src.data_ptr() : nullptr,
        .k_dst = k_cache_dst.data_ptr(),
        .v_dst = has_v ? v_cache_dst.data_ptr() : nullptr,
        .indices_src = indices_src.data_ptr(),
        .indices_dst = indices_dst.data_ptr(),
        .src_stride = src_stride,
        .dst_stride = static_cast<int64_t>(M.unwrap() * dtype_size),
        .row_bytes = row_bytes,
        .length = static_cast<uint32_t>(L.unwrap()),
        .num_layers = 1,
        .units_per_row_magic = units_per_row_magic(row_bytes),
        .src_is_table = false,
        .dst_is_table = false,
        .has_v = has_v,
        .has_src_map = false,
    };
    if (src_stride != row_bytes && row_bytes <= kHicacheTmaMaxMapRowBytes) {
      params.has_src_map = true;
      encode_src_map(&params.src_map[0], params.k_src, k_cache_src.shape()[0], row_bytes, src_stride);
      if (has_v) encode_src_map(&params.src_map[1], params.v_src, v_cache_src.shape()[0], row_bytes, src_stride);
    }
    launch(params, indices_dtype.unwrap().bits == 32, indices_device.unwrap());
  }

  // All layers through device-side pointer tables; strides and row bytes explicit.
  static void run_all_impl(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      int64_t src_stride_bytes,
      int64_t dst_stride_bytes,
      int64_t row_bytes,
      bool has_v) {
    using namespace host;
    auto N = SymbolicSize{"num_layers"};
    auto L = SymbolicSize{"indices length"};
    auto indices_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    auto verify_table = [&](const tvm::ffi::TensorView& t) {
      TensorMatcher({N}).with_dtype<uint64_t>().with_device<kDLGPU>(device_).verify(t);
    };
    verify_table(k_ptr_src);
    verify_table(k_ptr_dst);
    if (has_v) {
      verify_table(v_ptr_src);
      verify_table(v_ptr_dst);
    }
    verify_indices(indices_src, indices_dst, L, indices_dtype, device_);

    const HicacheTmaParams params{
        .k_src = k_ptr_src.data_ptr(),
        .v_src = has_v ? v_ptr_src.data_ptr() : nullptr,
        .k_dst = k_ptr_dst.data_ptr(),
        .v_dst = has_v ? v_ptr_dst.data_ptr() : nullptr,
        .indices_src = indices_src.data_ptr(),
        .indices_dst = indices_dst.data_ptr(),
        .src_stride = src_stride_bytes,
        .dst_stride = dst_stride_bytes,
        .row_bytes = static_cast<uint32_t>(row_bytes),
        .length = static_cast<uint32_t>(L.unwrap()),
        .num_layers = static_cast<uint32_t>(N.unwrap()),
        .units_per_row_magic = units_per_row_magic(static_cast<uint32_t>(row_bytes)),
        .src_is_table = true,
        .dst_is_table = true,
        .has_v = has_v,
        .has_src_map = false,
    };
    launch(params, indices_dtype.unwrap().bits == 32, device_.unwrap());
  }

  static void run_one(
      const tvm::ffi::TensorView k_cache_dst,
      const tvm::ffi::TensorView v_cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_cache_src,
      const tvm::ffi::TensorView v_cache_src,
      const tvm::ffi::TensorView indices_src) {
    run_one_impl(k_cache_dst, v_cache_dst, indices_dst, k_cache_src, v_cache_src, indices_src, true);
  }

  static void run_one_mla(
      const tvm::ffi::TensorView cache_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView cache_src,
      const tvm::ffi::TensorView indices_src) {
    run_one_impl(cache_dst, cache_dst, indices_dst, cache_src, cache_src, indices_src, false);
  }

  static void run_all(
      const tvm::ffi::TensorView k_ptr_dst,
      const tvm::ffi::TensorView v_ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView k_ptr_src,
      const tvm::ffi::TensorView v_ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t src_stride_bytes,
      const int64_t dst_stride_bytes,
      const int64_t row_bytes) {
    run_all_impl(
        k_ptr_dst,
        v_ptr_dst,
        indices_dst,
        k_ptr_src,
        v_ptr_src,
        indices_src,
        src_stride_bytes,
        dst_stride_bytes,
        row_bytes,
        true);
  }

  static void run_all_mla(
      const tvm::ffi::TensorView ptr_dst,
      const tvm::ffi::TensorView indices_dst,
      const tvm::ffi::TensorView ptr_src,
      const tvm::ffi::TensorView indices_src,
      const int64_t src_stride_bytes,
      const int64_t dst_stride_bytes,
      const int64_t row_bytes) {
    run_all_impl(
        ptr_dst,
        ptr_dst,
        indices_dst,
        ptr_src,
        ptr_src,
        indices_src,
        src_stride_bytes,
        dst_stride_bytes,
        row_bytes,
        false);
  }
};

}  // namespace sglang
