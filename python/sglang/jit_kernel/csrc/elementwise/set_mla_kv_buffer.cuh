#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct SetMlaKVBufferParams {
  const void* __restrict__ k_nope;
  const void* __restrict__ k_rope;
  void* __restrict__ kv_buffer;
  const void* __restrict__ loc;
  int64_t stride_nope_bytes;
  int64_t stride_rope_bytes;
  int64_t stride_buffer_bytes;
  uint32_t batch_size;
};

// Cooperative warp-wide memcpy of kBytes bytes. Picks the widest vector type
// that divides both the per-thread share and is implied by the byte width.
// Issues all loads first, then all stores so the warp scheduler can keep
// multiple LDG transactions in flight before the first STG retires.
template <int64_t kBytes>
SGL_DEVICE void copy_one_warp(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  constexpr int64_t kAlignment = (kBytes % (16 * kWarpThreads) == 0) ? 16
                                 : (kBytes % (8 * kWarpThreads) == 0)
                                     ? 8
                                     : (kBytes % (4 * kWarpThreads) == 0) ? 4 : (kBytes % 4 == 0) ? 4 : 0;
  static_assert(kAlignment > 0, "kBytes must be a multiple of 4");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr auto kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kLoopBytes;
  constexpr int64_t kTailVecs = (kBytes - kLoopCount * kLoopBytes) / sizeof(vec_t);
  constexpr int64_t kTotalVecs = kLoopCount + (kTailVecs > 0 ? 1 : 0);

  const auto gmem = tile::Memory<vec_t>::warp();

  vec_t buf[kTotalVecs == 0 ? 1 : kTotalVecs];

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    buf[i] = gmem.load(src, i);
  }
  if constexpr (kTailVecs > 0) {
    if (gmem.in_bound(kLoopCount * kWarpThreads + kTailVecs, kLoopCount)) {
      buf[kLoopCount] = gmem.load(src, kLoopCount);
    }
  }

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    gmem.store(dst, buf[i], i);
  }
  if constexpr (kTailVecs > 0) {
    if (gmem.in_bound(kLoopCount * kWarpThreads + kTailVecs, kLoopCount)) {
      gmem.store(dst, buf[kLoopCount], kLoopCount);
    }
  }
}

// Do the per-item copy for a single (item_id, split_id).
template <int64_t kNopeBytes, int64_t kRopeBytes, int kSplit>
SGL_DEVICE void copy_one_item(
    const void* __restrict__ k_nope_base,
    const void* __restrict__ k_rope_base,
    void* __restrict__ kv_buffer_base,
    int64_t stride_nope_bytes,
    int64_t stride_rope_bytes,
    int64_t stride_buffer_bytes,
    int64_t item_id,
    int64_t loc,
    uint32_t split_id) {
  using namespace device;
  const auto nope_src = pointer::offset(k_nope_base, item_id * stride_nope_bytes);
  const auto rope_src = pointer::offset(k_rope_base, item_id * stride_rope_bytes);
  const auto buf_dst = pointer::offset(kv_buffer_base, loc * stride_buffer_bytes);
  const auto rope_dst = pointer::offset(buf_dst, static_cast<int64_t>(kNopeBytes));

  if constexpr (kSplit == 1) {
    copy_one_warp<kNopeBytes>(nope_src, buf_dst);
    copy_one_warp<kRopeBytes>(rope_src, rope_dst);
  } else {
    constexpr int kNopeSplits = kSplit - 1;
    static_assert(
        kNopeBytes % kNopeSplits == 0,
        "kNopeBytes must divide evenly among kSplit-1 nope-warps");
    constexpr int64_t kChunkBytes = kNopeBytes / kNopeSplits;
    if (split_id < static_cast<uint32_t>(kNopeSplits)) {
      const int64_t chunk_off = static_cast<int64_t>(split_id) * kChunkBytes;
      copy_one_warp<kChunkBytes>(
          pointer::offset(nope_src, chunk_off), pointer::offset(buf_dst, chunk_off));
    } else {
      copy_one_warp<kRopeBytes>(rope_src, rope_dst);
    }
  }
}

// Tunables exposed via the launcher:
//   kSplit:        warps cooperating on one item. =1 → fat warp (large bs).
//                  >1 → kSplit-1 warps do equal nope chunks, last warp does rope.
//                  This multiplies the live CTA count which saturates more SMs at small bs.
//   kNumWarps:     warps per CTA. 1 keeps each CTA on its own SM (small bs); 4-8 packs work
//                  to amortise launch overhead at large bs.
//   kItemsPerWarp: items this warp processes in a tight unrolled loop. >1 gives the warp
//                  scheduler more in-flight memory ops, improving HBM utilisation at
//                  bs ≥ 4096 where we are memory-bound.
template <
    int64_t kNopeBytes,
    int64_t kRopeBytes,
    int kSplit,
    int kNumWarps,
    int kItemsPerWarp,
    bool kUsePDL,
    typename TLoc>
__global__ void set_mla_kv_buffer_kernel(const __grid_constant__ SetMlaKVBufferParams params) {
  using namespace device;
  const uint32_t warp_id_global = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t first_item = (warp_id_global / kSplit) * kItemsPerWarp;
  const uint32_t split_id = warp_id_global % kSplit;

  PDLWaitPrimary<kUsePDL>();

#pragma unroll
  for (int i = 0; i < kItemsPerWarp; ++i) {
    const uint32_t item_id = first_item + i;
    if (item_id >= params.batch_size) break;
    const int64_t loc = static_cast<int64_t>(static_cast<const TLoc*>(params.loc)[item_id]);
    copy_one_item<kNopeBytes, kRopeBytes, kSplit>(
        params.k_nope,
        params.k_rope,
        params.kv_buffer,
        params.stride_nope_bytes,
        params.stride_rope_bytes,
        params.stride_buffer_bytes,
        item_id,
        loc,
        split_id);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL>
struct SetMlaKVBufferKernel {
  static_assert(kNopeBytes > 0 && kNopeBytes % 4 == 0, "kNopeBytes must be a positive multiple of 4");
  static_assert(kRopeBytes > 0 && kRopeBytes % 4 == 0, "kRopeBytes must be a positive multiple of 4");

  template <int kSplit, int kNumWarps, int kItemsPerWarp, typename TLoc>
  static constexpr auto kernel = set_mla_kv_buffer_kernel<
      kNopeBytes,
      kRopeBytes,
      kSplit,
      kNumWarps,
      kItemsPerWarp,
      kUsePDL,
      TLoc>;

  static void
  run(tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      int64_t k_split,
      int64_t num_warps_per_block,
      int64_t items_per_warp) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto D_nope = SymbolicSize{"nope_dim"};
    auto D_rope = SymbolicSize{"rope_dim"};
    auto D_buf = SymbolicSize{"buffer_last_dim"};
    auto S_nope = SymbolicSize{"nope_stride"};
    auto S_rope = SymbolicSize{"rope_stride"};
    auto S_buf = SymbolicSize{"buffer_stride"};
    auto S_loc = SymbolicSize{"loc_stride"};
    auto dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA, kDLROCM>();

    TensorMatcher({B, D_nope})  //
        .with_strides({S_nope, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k_nope);
    TensorMatcher({B, D_rope})  //
        .with_strides({S_rope, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(k_rope);
    TensorMatcher({-1, D_buf})  //
        .with_strides({S_buf, 1})
        .with_dtype(dtype)
        .with_device(device)
        .verify(kv_buffer);
    TensorMatcher({B})  //
        .with_strides({S_loc})
        .with_dtype<int32_t, int64_t>(loc_dtype)
        .with_device(device)
        .verify(loc);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    RuntimeCheck(
        kNopeBytes == dtype_size * D_nope.unwrap(),
        "kNopeBytes mismatch: expected ",
        kNopeBytes,
        ", got ",
        dtype_size * D_nope.unwrap());
    RuntimeCheck(
        kRopeBytes == dtype_size * D_rope.unwrap(),
        "kRopeBytes mismatch: expected ",
        kRopeBytes,
        ", got ",
        dtype_size * D_rope.unwrap());
    RuntimeCheck(
        dtype_size * D_buf.unwrap() >= kNopeBytes + kRopeBytes,
        "kv_buffer last dim too small: bytes ",
        dtype_size * D_buf.unwrap(),
        " < ",
        kNopeBytes + kRopeBytes);

    const uint32_t batch = static_cast<uint32_t>(B.unwrap());
    if (batch == 0) return;

    const auto params = SetMlaKVBufferParams{
        .k_nope = k_nope.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .kv_buffer = kv_buffer.data_ptr(),
        .loc = loc.data_ptr(),
        .stride_nope_bytes = S_nope.unwrap() * dtype_size,
        .stride_rope_bytes = S_rope.unwrap() * dtype_size,
        .stride_buffer_bytes = S_buf.unwrap() * dtype_size,
        .batch_size = batch,
    };

    const auto use_int32 = loc_dtype.is_type<int32_t>();

    auto launch = [&]<int kS, int kNW, int kIPW>() {
      const auto kernel_ptr =
          use_int32 ? kernel<kS, kNW, kIPW, int32_t> : kernel<kS, kNW, kIPW, int64_t>;
      // Each warp handles kItemsPerWarp items via kSplit warps per item; pack kNumWarps
      // warps per CTA.
      const uint32_t item_groups = div_ceil(batch, static_cast<uint32_t>(kIPW));
      const uint32_t total_warps = item_groups * static_cast<uint32_t>(kS);
      const uint32_t num_blocks = div_ceil(total_warps, static_cast<uint32_t>(kNW));
      const uint32_t threads_per_block = static_cast<uint32_t>(kNW) * device::kWarpThreads;
      LaunchKernel(num_blocks, threads_per_block, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel_ptr, params);
    };

    auto bad = [&]() {
      Panic(
          "Unsupported (kSplit=",
          k_split,
          ", num_warps=",
          num_warps_per_block,
          ", items_per_warp=",
          items_per_warp,
          ")");
    };

    // kSplit > 1 only meaningful with items_per_warp == 1 (multi-item-per-warp only
    // exercised in the s=1 fat-warp path where memory bandwidth saturates).
    switch (k_split) {
      case 1:
        switch (items_per_warp) {
          case 1:
            switch (num_warps_per_block) {
              case 1: launch.template operator()<1, 1, 1>(); break;
              case 2: launch.template operator()<1, 2, 1>(); break;
              case 4: launch.template operator()<1, 4, 1>(); break;
              case 8: launch.template operator()<1, 8, 1>(); break;
              case 16: launch.template operator()<1, 16, 1>(); break;
              default: bad();
            }
            break;
          case 2:
            switch (num_warps_per_block) {
              case 4: launch.template operator()<1, 4, 2>(); break;
              case 8: launch.template operator()<1, 8, 2>(); break;
              default: bad();
            }
            break;
          case 4:
            switch (num_warps_per_block) {
              case 4: launch.template operator()<1, 4, 4>(); break;
              case 8: launch.template operator()<1, 8, 4>(); break;
              default: bad();
            }
            break;
          default:
            bad();
        }
        break;
      case 3:
        if constexpr (kNopeBytes % 2 == 0) {
          if (items_per_warp != 1) bad();
          switch (num_warps_per_block) {
            case 1: launch.template operator()<3, 1, 1>(); break;
            case 3: launch.template operator()<3, 3, 1>(); break;
            case 6: launch.template operator()<3, 6, 1>(); break;
            default: bad();
          }
        } else {
          bad();
        }
        break;
      case 5:
        if constexpr (kNopeBytes % 4 == 0) {
          if (items_per_warp != 1) bad();
          switch (num_warps_per_block) {
            case 1: launch.template operator()<5, 1, 1>(); break;
            case 5: launch.template operator()<5, 5, 1>(); break;
            default: bad();
          }
        } else {
          bad();
        }
        break;
      default:
        bad();
    }
  }
};

}  // namespace
