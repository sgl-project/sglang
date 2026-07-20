// Fused gather and scatter into sconv_cache.
//
// For each batch element b where mask[b] is true, copy the W1 = W-1 token rows
// hidden_states[track_idx[b, w]]  ->  sconv_cache[dst[b], w]  (w = 0..W1-1).
// Masked-out lanes are left untouched. Pure copy (no arithmetic) => BIT-EXACT.
// 2 channels/thread packed as bf16x2. Requires bf16 + even D + channel-contiguous.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

struct GatherScatterParams {
  const void* __restrict__ hidden;     // [T, D], channel-contiguous
  void* __restrict__ cache;            // [pool, W1, D], in-place scatter target
  const void* __restrict__ track_idx;  // int32 [B, W1]
  const void* __restrict__ mask;       // bool  [B]
  const void* __restrict__ dst;        // int64 [B]
  int64_t hs_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  int64_t track_stride_b;
  int64_t track_stride_w;
  int64_t dst_stride_b;
  uint32_t D;
};

constexpr uint32_t kGSThreads = 256;

template <int W1, typename DType>
__global__ void gather_scatter_kernel(const __grid_constant__ GatherScatterParams p) {
  const int b = blockIdx.y;
  if (!static_cast<const bool*>(p.mask)[b]) return;  // masked-out lane: untouched

  const int c0 = (blockIdx.x * kGSThreads + threadIdx.x) * 2;
  if (c0 >= static_cast<int>(p.D)) return;

  const auto* hp = static_cast<const __nv_bfloat16*>(p.hidden);
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const int64_t dst_slot = static_cast<const int64_t*>(p.dst)[static_cast<int64_t>(b) * p.dst_stride_b];
  const int64_t cache_base = dst_slot * p.cache_stride_slot + c0;
  const int64_t track_base = static_cast<int64_t>(b) * p.track_stride_b;

#pragma unroll
  for (int w = 0; w < W1; ++w) {
    const int64_t src_t =
        static_cast<const int32_t*>(p.track_idx)[track_base + static_cast<int64_t>(w) * p.track_stride_w];
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&hp[src_t * p.hs_stride_t + c0]);
    *reinterpret_cast<__nv_bfloat162*>(&cp[cache_base + static_cast<int64_t>(w) * p.cache_stride_w]) = v;
  }
}

template <int W1, typename DType>
struct GatherScatterSconvKernel {
  static void
  run(tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView track_idx,
      tvm::ffi::TensorView mask,
      tvm::ffi::TensorView dst) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto B = SymbolicSize{"B"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    W1s.set_value(W1);

    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(hidden);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({B, W1s}).with_dtype<int32_t>().with_device(dev).verify(track_idx);
    TensorMatcher({B}).with_device(dev).verify(mask);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(dev).verify(dst);
    RuntimeCheck(sizeof(DType) == 2, "gather_scatter: bf16x2 kernel requires 16-bit dtype");
    RuntimeCheck(D.unwrap() % 2 == 0, "gather_scatter: D must be even for the bf16x2 kernel");
    RuntimeCheck(cache.stride(2) == 1, "gather_scatter: cache must be channel-contiguous");

    const auto params = GatherScatterParams{
        .hidden = hidden.data_ptr(),
        .cache = cache.data_ptr(),
        .track_idx = track_idx.data_ptr(),
        .mask = mask.data_ptr(),
        .dst = dst.data_ptr(),
        .hs_stride_t = hidden.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .track_stride_b = track_idx.stride(0),
        .track_stride_w = track_idx.stride(1),
        .dst_stride_b = dst.stride(0),
        .D = static_cast<uint32_t>(D.unwrap()),
    };

    const uint32_t d_pairs = params.D / 2;
    const dim3 grid{div_ceil(d_pairs, kGSThreads), static_cast<uint32_t>(B.unwrap())};
    const dim3 block{kGSThreads};
    constexpr auto kernel = gather_scatter_kernel<W1, DType>;
    LaunchKernel(grid, block, dev.unwrap())(kernel, params);
  }
};

}  // namespace
