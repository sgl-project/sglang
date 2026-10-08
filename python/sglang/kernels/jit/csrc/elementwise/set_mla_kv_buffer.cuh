#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#if defined(SGL_CUDA_ARCH) && SGL_CUDA_ARCH >= 900
#include <cuda/ptx>
#endif

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang {

struct SetMlaKVBufferParams {
  const void* __restrict__ k_nope;
  const void* __restrict__ k_rope;
  void* __restrict__ kv_buffer;
  const void* __restrict__ loc;
  int64_t stride_nope_bytes;
  int64_t stride_rope_bytes;
  int64_t stride_buffer_bytes;
  uint32_t batch_size;
  int64_t reserved_skip_index;
};

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL, typename TLoc>
__global__ void set_mla_kv_buffer_kernel(const __grid_constant__ SetMlaKVBufferParams params) {
  using namespace device;
  using enum warp::LoadStorePattern::type;
  const auto global_warp_id = threadIdx.y + blockIdx.x * blockDim.y;
  const auto input_nope = pointer::offset(params.k_nope, params.stride_nope_bytes * global_warp_id);
  const auto input_rope = pointer::offset(params.k_rope, params.stride_rope_bytes * global_warp_id);
  if (global_warp_id >= params.batch_size) return;

  PDLWaitPrimary<kUsePDL>();
  const int64_t loc = static_cast<int64_t>(static_cast<const TLoc*>(params.loc)[global_warp_id]);
  const auto nope = warp::load_bytes<kNopeBytes, WARP_UNIFORM_16B>(input_nope);
  const auto rope = warp::load_bytes<kRopeBytes, WARP_UNIFORM_16B>(input_rope);

  PDLTriggerSecondary<kUsePDL>();
  if (loc != params.reserved_skip_index) {
    const auto output_nope = pointer::offset(params.kv_buffer, params.stride_buffer_bytes * loc);
    const auto output_rope = pointer::offset(output_nope, kNopeBytes);
    warp::store_bytes<kNopeBytes, WARP_UNIFORM_16B>(output_nope, nope);
    warp::store_bytes<kRopeBytes, WARP_UNIFORM_16B>(output_rope, rope);
  }
}

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL>
struct SetMlaKVBufferKernel {
  template <typename TLoc>
  static constexpr auto set_kernel = set_mla_kv_buffer_kernel<kNopeBytes, kRopeBytes, kUsePDL, TLoc>;

  static void
  run(tvm::ffi::TensorView kv_buffer,
      tvm::ffi::TensorView loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      int64_t,
      int64_t reserved_skip_index) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto dtype = SymbolicDType{};
    auto loc_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    using device::warp::LoadStorePattern;
    using enum LoadStorePattern::type;
    constexpr int64_t kAlignNope = LoadStorePattern::get_vec_bytes<kNopeBytes, WARP_UNIFORM_16B>();
    constexpr int64_t kAlignRope = LoadStorePattern::get_vec_bytes<kRopeBytes, WARP_UNIFORM_16B>();
    // The buffer row carries both halves, so it has to satisfy the WIDER of the
    // two -- the narrower one alone would let a nope-misaligned stride through.
    constexpr int64_t kAlignBuffer = std::max(kAlignNope, kAlignRope);
    // The rope half starts kNopeBytes into the row, so that offset must not
    // break the rope alignment the buffer was just checked for.
    static_assert(kNopeBytes % kAlignRope == 0, "nope width must not misalign the rope half");

    TensorMatcher({B, -1})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignNope)
        .verify(k_nope);
    TensorMatcher({B, -1})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignRope)
        .verify(k_rope);
    TensorMatcher({-1, -1})  //
        .with_strides({-1, 1})
        .with_dtype(dtype)
        .with_device(device_)
        .ensure_alignment(kAlignBuffer)
        .verify(kv_buffer);
    TensorMatcher({B})  //
        .with_strides({-1})
        .with_dtype<int32_t, int64_t>(loc_dtype)
        .with_device(device_)
        .verify(loc);

    const auto dtype_size = static_cast<int64_t>(dtype_bytes(dtype.unwrap()));
    CHECK_HOST(kv_buffer.size(1) >= k_nope.size(1) + k_rope.size(1));
    CHECK_HOST(k_nope.size(1) * dtype_size == kNopeBytes);
    CHECK_HOST(k_rope.size(1) * dtype_size == kRopeBytes);
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto params = SetMlaKVBufferParams{
        .k_nope = k_nope.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .kv_buffer = kv_buffer.data_ptr(),
        .loc = loc.data_ptr(),
        .stride_nope_bytes = k_nope.stride(0) * dtype_size,
        .stride_rope_bytes = k_rope.stride(0) * dtype_size,
        .stride_buffer_bytes = kv_buffer.stride(0) * dtype_size,
        .batch_size = batch_size,
        .reserved_skip_index = reserved_skip_index,
    };
    const auto device = device_.unwrap();
    const auto kernel = loc_dtype.is_type<int32_t>() ? set_kernel<int32_t> : set_kernel<int64_t>;
    const auto num_warps = [&] {
      const auto sm_count = runtime::get_sm_count(device.device_id);
#pragma unroll
      for (uint32_t n : {1, 2, 4}) {
        if (batch_size <= sm_count * n) return n;
      }
      return 8u;
    }();
    const auto num_blocks = div_ceil(batch_size, num_warps);
    LaunchKernel(num_blocks, {device::kWarpSize, num_warps}, device)  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

#if defined(SGL_CUDA_ARCH) && SGL_CUDA_ARCH >= 900
// The single-destination writer above intentionally retains its existing API
// and vector-store implementation. Only this dual-destination specialization
// stages a row once and issues TMA stores to scratch and its persistent owner.
struct SetShardedMlaKVBufferParams {
  const void* k_nope;
  const void* k_rope;
  void* scratch;
  void* shard_buffer;
  const void* scratch_loc;
  const void* logical_loc;
  int64_t stride_nope_bytes;
  int64_t stride_rope_bytes;
  int64_t stride_scratch_bytes;
  int64_t stride_shard_bytes;
  int64_t page_size;
  int64_t shard_size;
  int64_t shard_rank;
  int64_t scratch_reserved_skip_index;
  int64_t local_reserved_skip_index;
  uint32_t batch_size;
};

template <int64_t kNopeBytes, int64_t kRopeBytes, int kNumWarps, bool kUsePDL, typename TLoc>
__global__ void set_sharded_mla_kv_buffer_kernel(const __grid_constant__ SetShardedMlaKVBufferParams params) {
  using namespace device;
  using enum warp::LoadStorePattern::type;
  constexpr int64_t kRowBytes = kNopeBytes + kRopeBytes;
  static_assert(kRowBytes % 16 == 0);
  __shared__ alignas(16) uint8_t smem[kNumWarps][kRowBytes];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t item = blockIdx.x * kNumWarps + warp_id;
  if (item >= params.batch_size) return;

  PDLWaitPrimary<kUsePDL>();
  const int64_t scratch_loc = static_cast<const TLoc*>(params.scratch_loc)[item];
  const int64_t logical_loc = static_cast<const TLoc*>(params.logical_loc)[item];
  const auto nope =
      warp::load_bytes<kNopeBytes, WARP_UNIFORM_16B>(pointer::offset(params.k_nope, item * params.stride_nope_bytes));
  const auto rope =
      warp::load_bytes<kRopeBytes, WARP_UNIFORM_16B>(pointer::offset(params.k_rope, item * params.stride_rope_bytes));
  warp::store_bytes<kNopeBytes, WARP_UNIFORM_16B>(&smem[warp_id][0], nope);
  warp::store_bytes<kRopeBytes, WARP_UNIFORM_16B>(&smem[warp_id][kNopeBytes], rope);
  // Every writer publishes to the async proxy before the elected lane can
  // issue TMA. Each warp owns a separate row, so warp synchronization suffices.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();

  if (warp::elect_one_lane()) {
    if (scratch_loc != params.scratch_reserved_skip_index) {
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          pointer::offset(params.scratch, scratch_loc * params.stride_scratch_bytes),
          &smem[warp_id][0],
          static_cast<uint32_t>(kRowBytes));
    }
    const auto page = logical_loc / params.page_size;
    if (page % params.shard_size == params.shard_rank) {
      const auto local = (page / params.shard_size) * params.page_size + logical_loc % params.page_size;
      if (local != params.local_reserved_skip_index) {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_global,
            cuda::ptx::space_shared,
            pointer::offset(params.shard_buffer, local * params.stride_shard_bytes),
            &smem[warp_id][0],
            static_cast<uint32_t>(kRowBytes));
      }
    }
    // wait_group (not wait_group_read) waits for destination visibility, not
    // merely for TMA to stop reading the shared staging row.
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
  }
  __syncwarp();
  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kNopeBytes, int64_t kRopeBytes, bool kUsePDL>
struct SetShardedMlaKVBufferKernel {
  static_assert(kNopeBytes == 512 && kRopeBytes == 64, "only raw FP8 MLA rows are supported");

  static void
  run(tvm::ffi::TensorView scratch,
      tvm::ffi::TensorView scratch_loc,
      tvm::ffi::TensorView k_nope,
      tvm::ffi::TensorView k_rope,
      tvm::ffi::TensorView shard_buffer,
      tvm::ffi::TensorView logical_loc,
      int64_t page_size,
      int64_t shard_size,
      int64_t shard_rank,
      int64_t num_warps,
      int64_t scratch_reserved_skip_index,
      int64_t local_reserved_skip_index) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto loc_dtype = SymbolicDType{};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({B, kNopeBytes})
        .with_strides({-1, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(k_nope);
    TensorMatcher({B, kRopeBytes})
        .with_strides({-1, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .ensure_alignment(16)
        .verify(k_rope);
    for (const auto& buffer : {scratch, shard_buffer}) {
      TensorMatcher({-1, -1})
          .with_strides({-1, 1})
          .with_dtype<uint8_t>()
          .with_device(device_)
          .ensure_alignment(16)
          .verify(buffer);
      CHECK_HOST(buffer.size(1) >= kNopeBytes + kRopeBytes);
      CHECK_HOST(buffer.stride(0) >= buffer.size(1));
    }
    TensorMatcher({B}).with_strides({1}).with_dtype<int32_t, int64_t>(loc_dtype).with_device(device_).verify(
        scratch_loc);
    TensorMatcher({B}).with_strides({1}).with_dtype(loc_dtype).with_device(device_).verify(logical_loc);
    CHECK_HOST(page_size > 0 && shard_size > 1 && shard_rank >= 0 && shard_rank < shard_size);
    CHECK_HOST(scratch_reserved_skip_index >= -1 && local_reserved_skip_index >= -1);
    CHECK_HOST(B.unwrap() <= UINT32_MAX);
    CHECK_HOST(k_nope.stride(0) >= kNopeBytes && k_rope.stride(0) >= kRopeBytes);
    const auto batch = static_cast<uint32_t>(B.unwrap());
    if (batch == 0) return;
    const auto params = SetShardedMlaKVBufferParams{
        .k_nope = k_nope.data_ptr(),
        .k_rope = k_rope.data_ptr(),
        .scratch = scratch.data_ptr(),
        .shard_buffer = shard_buffer.data_ptr(),
        .scratch_loc = scratch_loc.data_ptr(),
        .logical_loc = logical_loc.data_ptr(),
        .stride_nope_bytes = k_nope.stride(0),
        .stride_rope_bytes = k_rope.stride(0),
        .stride_scratch_bytes = scratch.stride(0),
        .stride_shard_bytes = shard_buffer.stride(0),
        .page_size = page_size,
        .shard_size = shard_size,
        .shard_rank = shard_rank,
        .scratch_reserved_skip_index = scratch_reserved_skip_index,
        .local_reserved_skip_index = local_reserved_skip_index,
        .batch_size = batch,
    };
    auto launch = [&]<int kNW>() {
      const auto kernel = loc_dtype.is_type<int32_t>()
                              ? set_sharded_mla_kv_buffer_kernel<kNopeBytes, kRopeBytes, kNW, kUsePDL, int32_t>
                              : set_sharded_mla_kv_buffer_kernel<kNopeBytes, kRopeBytes, kNW, kUsePDL, int64_t>;
      LaunchKernel(div_ceil(batch, static_cast<uint32_t>(kNW)), kNW * device::kWarpThreads, device_.unwrap())
          .enable_pdl(kUsePDL)(kernel, params);
    };
    switch (num_warps) {
      case 4:
        launch.template operator()<4>();
        break;
      case 8:
        launch.template operator()<8>();
        break;
      default:
        Panic("Unsupported num_warps=", num_warps);
    }
  }
};
#endif  // SGL_CUDA_ARCH >= 900

}  // namespace sglang
