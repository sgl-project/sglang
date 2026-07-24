// Update the convolution cache from an extend/prefill token stream.
//
// For each sequence b with slot ci = cache_indices[b] and query range
// [start, end) (query_start_loc), the new conv state is the last W1 = W-1 entries of
// the virtual stream  [ old_state (W1 rows, gated by has_initial_state) ++ x[start:end] ]:
//     new_state[w] = virtual[qlen + w]   for w in 0..W1-1  (qlen = end - start)
//       qlen + w >= W1  -> x[end - W1 + w, d]                          (a "current" token)
//       qlen + w <  W1  -> old_cache[slot, w + qlen, d] * has_state    (shifted state)
// PAD (ci == -1) or empty (qlen <= 0) lanes are left untouched. This is a pure
// select/copy (no arithmetic) => must be BIT-EXACT (bf16 values moved verbatim).
//
// RAW-safe: each thread loads all W1 old_cache rows into registers BEFORE writing any,
// so the in-place writes never clobber a not-yet-read shift source. 2 channels/thread
// are packed as bf16x2 (32-bit) to halve the moves. Requires bf16 + even D.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

struct UpdateSconvParams {
  const void* __restrict__ x;              // [T, D], channel-contiguous
  void* __restrict__ cache;                // [max_slots, W1, D], in-place update
  const void* __restrict__ cache_indices;  // int32 [B]
  const void* __restrict__ has_state;      // bool  [B]
  const void* __restrict__ qsl;            // int32 [B+1] query_start_loc
  int64_t x_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  uint32_t D;
};

constexpr uint32_t kUpdThreads = 256;  // threads/block, each owns a channel pair
constexpr int kPadSlot = -1;

template <int W1, typename DType>
__global__ void update_sconv_cache_kernel(const __grid_constant__ UpdateSconvParams p) {
  const int b = blockIdx.y;
  const int ci = static_cast<const int32_t*>(p.cache_indices)[b];
  const int start = static_cast<const int32_t*>(p.qsl)[b];
  const int end = static_cast<const int32_t*>(p.qsl)[b + 1];
  const int qlen = end - start;
  if (ci == kPadSlot || qlen <= 0) return;  // PAD / empty lane: untouched

  const int c0 = (blockIdx.x * kUpdThreads + threadIdx.x) * 2;
  if (c0 >= static_cast<int>(p.D)) return;

  const bool hs = static_cast<const bool*>(p.has_state)[b];
  const auto* xp = static_cast<const __nv_bfloat16*>(p.x);
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const int cw = static_cast<int>(p.cache_stride_w);
  const int64_t slot_base = static_cast<int64_t>(ci) * p.cache_stride_slot + c0;

  // Load all old-state rows into registers first (RAW-safe against the writes below).
  __nv_bfloat162 old_reg[W1];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    old_reg[w] = *reinterpret_cast<const __nv_bfloat162*>(&cp[slot_base + static_cast<int64_t>(w) * cw]);
  }
  const __nv_bfloat162 zero = __float2bfloat162_rn(0.0f);

#pragma unroll
  for (int w = 0; w < W1; ++w) {
    __nv_bfloat162 nv;
    if (qlen >= (W1 - w)) {
      // current token from x: index end - W1 + w >= start >= 0
      const int x_idx = end - W1 + w;
      nv = *reinterpret_cast<const __nv_bfloat162*>(&xp[static_cast<int64_t>(x_idx) * p.x_stride_t + c0]);
    } else {
      // shifted state old_cache[w + qlen] (w+qlen in [0, W1)), gated by has_state
      __nv_bfloat162 shift = zero;
#pragma unroll
      for (int src = 0; src < W1; ++src) {
        if (src == w + qlen) shift = old_reg[src];
      }
      nv = hs ? shift : zero;
    }
    *reinterpret_cast<__nv_bfloat162*>(&cp[slot_base + static_cast<int64_t>(w) * cw]) = nv;
  }
}

template <int W1, typename DType>
struct UpdateSconvCacheKernel {
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView has_state,
      tvm::ffi::TensorView qsl) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto B = SymbolicSize{"B"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    W1s.set_value(W1);

    // x channel-contiguous (may be a non-contiguous row view); cache contiguous
    // [slots, W1, D]. cache_indices/qsl int32, has_state torch-bool (shape/device only).
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(x);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(cache_indices);
    TensorMatcher({B}).with_device(dev).verify(has_state);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(dev).verify(qsl);
    RuntimeCheck(qsl.size(0) == B.unwrap() + 1, "qsl must have length B+1");
    RuntimeCheck(sizeof(DType) == 2, "update_sconv_cache: bf16x2 kernel requires 16-bit dtype");
    RuntimeCheck(D.unwrap() % 2 == 0, "update_sconv_cache: D must be even for the bf16x2 kernel");
    RuntimeCheck(cache.stride(2) == 1, "update_sconv_cache: cache must be channel-contiguous");

    const auto params = UpdateSconvParams{
        .x = x.data_ptr(),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .has_state = has_state.data_ptr(),
        .qsl = qsl.data_ptr(),
        .x_stride_t = x.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .D = static_cast<uint32_t>(D.unwrap()),
    };

    const uint32_t d_pairs = params.D / 2;
    const dim3 grid{div_ceil(d_pairs, kUpdThreads), static_cast<uint32_t>(B.unwrap())};
    const dim3 block{kUpdThreads};
    constexpr auto kernel = update_sconv_cache_kernel<W1, DType>;
    LaunchKernel(grid, block, dev.unwrap())(kernel, params);
  }
};

}  // namespace
