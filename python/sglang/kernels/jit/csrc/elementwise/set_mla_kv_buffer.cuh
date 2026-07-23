// JIT TMA bulk-store kernel for MLA paged-KV scatter writes.
//
// Each warp:
//   1. Cooperatively loads one item's (nope, rope) row into a per-warp slot in
//      shared memory via vectorised ld/st.
//   2. Lane 0 issues a single ``cp.async.bulk.global.shared::cta`` (TMA bulk
//      store, non-tensor variant) to scatter the row to
//      ``kv_buffer + loc[item] * stride_buffer``.
//
// End-of-CTA: ``cp.async.bulk.commit_group`` + ``wait_group<0>`` ensures all
// in-flight stores commit before the kernel exits so the writes are visible
// to subsequent kernels and the host.
//
// Two correctness gotchas worth a comment (easy to lose):
//   - ``fence.proxy.async.shared::cta`` between the smem fill and the TMA
//     store. The TMA engine reads via the async proxy; without the fence it
//     observes stale smem under heavy concurrency (manifests as zero rows at
//     large bs).
//   - ``wait_group`` not ``wait_group_read`` — the latter only allows early
//     smem reuse; it does not wait for the gmem store to commit globally.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cuda/ptx>
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

// Warp-cooperative gmem -> smem copy. Picks the widest vec width that divides
// both the per-thread share and the byte total. Caller guarantees src is
// 16-byte aligned (PyTorch tensors are) and dst is the start of a per-warp
// smem slot (also 16-byte aligned by ``alignas(16)``).
template <int64_t kBytes>
SGL_DEVICE void warp_g2s_copy(const void* __restrict__ src, void* __restrict__ dst) {
  using namespace device;
  constexpr int64_t kAlignment = (kBytes % (16 * kWarpThreads) == 0)  ? 16
                                 : (kBytes % (8 * kWarpThreads) == 0) ? 8
                                 : (kBytes % (4 * kWarpThreads) == 0) ? 4
                                 : (kBytes % 4 == 0)                  ? 4
                                                                      : 0;
  static_assert(kAlignment > 0, "kBytes must be a multiple of 4");

  using vec_t = AlignedStorage<uint32_t, kAlignment / 4>;
  constexpr auto kLoopBytes = sizeof(vec_t) * kWarpThreads;
  constexpr auto kLoopCount = kBytes / kLoopBytes;
  constexpr int64_t kTailVecs = (kBytes - kLoopCount * kLoopBytes) / sizeof(vec_t);

  const auto gmem = tile::Memory<vec_t>::warp();

#pragma unroll
  for (int64_t i = 0; i < kLoopCount; ++i) {
    const auto v = gmem.load(src, i);
    gmem.store(dst, v, i);
  }
  if constexpr (kTailVecs > 0) {
    if (gmem.in_bound(kLoopCount * kWarpThreads + kTailVecs, kLoopCount)) {
      const auto v = gmem.load(src, kLoopCount);
      gmem.store(dst, v, kLoopCount);
    }
  }
}

template <int64_t kNopeBytes, int64_t kRopeBytes, int kNumWarps, bool kUsePDL, typename TLoc>
__global__ void set_mla_kv_buffer_kernel(const __grid_constant__ SetMlaKVBufferParams params) {
  using namespace device;
  static_assert((kNopeBytes + kRopeBytes) % 16 == 0, "TMA bulk store requires total row to be 16-byte aligned");

  constexpr int64_t kRowBytes = kNopeBytes + kRopeBytes;

  // One contiguous smem slot per warp; align to 16 for TMA.
  __shared__ alignas(16) uint8_t smem[kNumWarps][kRowBytes];

  const uint32_t warp_in_cta = threadIdx.x / kWarpThreads;
  const uint32_t item_id = blockIdx.x * kNumWarps + warp_in_cta;
  if (item_id >= params.batch_size) return;

  PDLWaitPrimary<kUsePDL>();

  const int64_t loc = static_cast<int64_t>(static_cast<const TLoc*>(params.loc)[item_id]);

  const auto nope_src = pointer::offset(params.k_nope, item_id * params.stride_nope_bytes);
  const auto rope_src = pointer::offset(params.k_rope, item_id * params.stride_rope_bytes);
  void* const gmem_dst = pointer::offset(params.kv_buffer, loc * params.stride_buffer_bytes);

  // Warp-cooperative load (nope, rope) into the per-warp smem slot.
  warp_g2s_copy<kNopeBytes>(nope_src, &smem[warp_in_cta][0]);
  warp_g2s_copy<kRopeBytes>(rope_src, &smem[warp_in_cta][kNopeBytes]);

  // Fence required: TMA reads smem via the async proxy, normal sts writes
  // through the generic proxy. Without this the TMA engine can observe stale
  // values at large bs.
  __syncwarp();
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

  // Lane 0 issues one bulk store from the smem slot to the scattered gmem row.
  if (threadIdx.x % kWarpThreads == 0) {
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_global,
        cuda::ptx::space_shared,
        gmem_dst,
        &smem[warp_in_cta][0],
        static_cast<uint32_t>(kRowBytes));
  }

  // Commit and wait for the CTA's bulk-stores to be globally visible before
  // returning. ``wait_group`` (not ``_read``) is the one that waits for gmem
  // commit; ``_read`` only releases smem for reuse.
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL>
struct SetMlaKVBufferKernel {
  static_assert(kNopeBytes > 0 && kNopeBytes % 4 == 0, "kNopeBytes must be a positive multiple of 4");
  static_assert(kRopeBytes > 0 && kRopeBytes % 4 == 0, "kRopeBytes must be a positive multiple of 4");
  static_assert(
      (kNopeBytes + kRopeBytes) % 16 == 0, "TMA bulk store requires (kNopeBytes + kRopeBytes) to be a multiple of 16");

  template <int kNumWarps, typename TLoc>
  static constexpr auto kernel = set_mla_kv_buffer_kernel<kNopeBytes, kRopeBytes, kNumWarps, kUsePDL, TLoc>;

  static void
  run(tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      int64_t num_warps_per_block) {
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
    device.set_options<kDLCUDA>();

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
    RuntimeCheck(dtype_size * D_buf.unwrap() >= kNopeBytes + kRopeBytes, "kv_buffer last dim too small");
    RuntimeCheck(
        (S_buf.unwrap() * dtype_size) % 16 == 0,
        "kv_buffer row stride must be a multiple of 16 bytes for TMA bulk store; got ",
        S_buf.unwrap() * dtype_size);

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

    auto launch = [&]<int kNW>() {
      const auto kernel_ptr = use_int32 ? kernel<kNW, int32_t> : kernel<kNW, int64_t>;
      const uint32_t num_blocks = div_ceil(batch, static_cast<uint32_t>(kNW));
      const uint32_t threads_per_block = static_cast<uint32_t>(kNW) * device::kWarpThreads;
      LaunchKernel(num_blocks, threads_per_block, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel_ptr, params);
    };

    switch (num_warps_per_block) {
      case 1:
        launch.template operator()<1>();
        break;
      case 2:
        launch.template operator()<2>();
        break;
      case 4:
        launch.template operator()<4>();
        break;
      case 8:
        launch.template operator()<8>();
        break;
      default:
        Panic("Unsupported num_warps_per_block=", num_warps_per_block);
    }
  }
};

}  // namespace
