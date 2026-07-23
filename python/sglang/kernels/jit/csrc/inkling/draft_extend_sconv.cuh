// Fused draft-extend convolution-cache update.
//
// Speculative draft-extend: for each sequence b (slot ci = cache_indices[b]) the new
// conv state is the length-W1 window of the "virtual padded" stream
//     virtual = [ sconv_cache[ci] (W1 rows) ++ hidden[b, 0:T] (T rows) ]
// starting at num_accepted_tokens[b]:  new[w] = virtual[n_acc + w], w in 0..W1-1
//     n_acc + w <  W1  -> sconv_cache[ci, n_acc + w]   (initial state)
//     n_acc + w >= W1  -> hidden[b*T + (n_acc + w - W1)]  (a draft token)
// written back to sconv_cache[ci]. With tracking, the window at track_step[b] is also
// written to sconv_cache[mamba_track_indices[b]] wherever crossed[b].
// Pure copy/select (BIT-EXACT). Init state loaded to registers before writes (RAW-safe);
// 2 channels/thread as bf16x2. Requires bf16 + even D + channel-contiguous.
#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace {

struct DraftExtendParams {
  const void* __restrict__ hidden;         // [B*T, D], channel-contiguous
  void* __restrict__ cache;                // [pool, W1, D], in-place
  const void* __restrict__ cache_indices;  // int32 [B]
  const void* __restrict__ num_accepted;   // int32 [B]
  const void* __restrict__ crossed;        // bool  [B]   (DO_TRACK only)
  const void* __restrict__ track_step;     // int32 [B]   (DO_TRACK only)
  const void* __restrict__ track_indices;  // int64 [B]   (DO_TRACK only)
  int64_t hs_stride_t;
  int64_t cache_stride_slot;
  int64_t cache_stride_w;
  uint32_t D;
  uint32_t T;  // draft_token_num
};

constexpr uint32_t kDEThreads = 256;

template <int W1, bool DO_TRACK, typename DType>
__global__ void draft_extend_kernel(const __grid_constant__ DraftExtendParams p) {
  const int b = blockIdx.y;
  const int c0 = (blockIdx.x * kDEThreads + threadIdx.x) * 2;
  if (c0 >= static_cast<int>(p.D)) return;

  const int ci = static_cast<const int32_t*>(p.cache_indices)[b];
  const auto* hp = static_cast<const __nv_bfloat16*>(p.hidden);
  auto* cp = static_cast<__nv_bfloat16*>(p.cache);
  const int cw = static_cast<int>(p.cache_stride_w);
  const int T = static_cast<int>(p.T);
  const int b_off = b * T;  // hidden row base for this sequence
  const int64_t src_slot_base = static_cast<int64_t>(ci) * p.cache_stride_slot + c0;

  // Initial state -> registers (RAW-safe against the cache[ci] writes below).
  __nv_bfloat162 init_reg[W1];
#pragma unroll
  for (int w = 0; w < W1; ++w) {
    init_reg[w] = *reinterpret_cast<const __nv_bfloat162*>(&cp[src_slot_base + static_cast<int64_t>(w) * cw]);
  }

  // Select the window at `at` from the virtual stream and write it to cache[dst_base].
  auto emit = [&](int at, int64_t dst_base) {
#pragma unroll
    for (int w = 0; w < W1; ++w) {
      const int pos = at + w;
      __nv_bfloat162 v;
      if (pos < W1) {
        v = init_reg[0];
#pragma unroll
        for (int src = 0; src < W1; ++src) {
          if (src == pos) v = init_reg[src];
        }
      } else {
        const int row = b_off + (pos - W1);
        v = *reinterpret_cast<const __nv_bfloat162*>(&hp[static_cast<int64_t>(row) * p.hs_stride_t + c0]);
      }
      *reinterpret_cast<__nv_bfloat162*>(&cp[dst_base + static_cast<int64_t>(w) * cw]) = v;
    }
  };

  const int n_acc = static_cast<const int32_t*>(p.num_accepted)[b];
  if constexpr (DO_TRACK) {
    // Track window first (reads init_reg, distinct dst slot) then the main window.
    if (static_cast<const bool*>(p.crossed)[b]) {
      const int tstep = static_cast<const int32_t*>(p.track_step)[b];
      const int64_t tslot = static_cast<const int64_t*>(p.track_indices)[b];
      emit(tstep, tslot * p.cache_stride_slot + c0);
    }
  }
  emit(n_acc, src_slot_base);
}

template <int W1, bool DO_TRACK, typename DType>
struct DraftExtendSconvKernel {
  static void
  run(tvm::ffi::TensorView hidden,
      tvm::ffi::TensorView cache,
      tvm::ffi::TensorView cache_indices,
      tvm::ffi::TensorView num_accepted,
      int64_t draft_token_num,
      tvm::ffi::TensorView crossed,
      tvm::ffi::TensorView track_step,
      tvm::ffi::TensorView track_indices) {
    using namespace host;
    auto BT = SymbolicSize{"B_times_T"};
    auto D = SymbolicSize{"D"};
    auto W1s = SymbolicSize{"W_minus_1"};
    auto B = SymbolicSize{"B"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    W1s.set_value(W1);

    TensorMatcher({BT, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(hidden);
    TensorMatcher({-1, W1s, D}).with_dtype<DType>().with_device(dev).verify(cache);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(cache_indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(num_accepted);
    RuntimeCheck(sizeof(DType) == 2, "draft_extend: bf16x2 kernel requires 16-bit dtype");
    RuntimeCheck(D.unwrap() % 2 == 0, "draft_extend: D must be even for the bf16x2 kernel");
    RuntimeCheck(cache.stride(2) == 1, "draft_extend: cache must be channel-contiguous");

    const auto params = DraftExtendParams{
        .hidden = hidden.data_ptr(),
        .cache = cache.data_ptr(),
        .cache_indices = cache_indices.data_ptr(),
        .num_accepted = num_accepted.data_ptr(),
        .crossed = DO_TRACK ? crossed.data_ptr() : nullptr,
        .track_step = DO_TRACK ? track_step.data_ptr() : nullptr,
        .track_indices = DO_TRACK ? track_indices.data_ptr() : nullptr,
        .hs_stride_t = hidden.stride(0),
        .cache_stride_slot = cache.stride(0),
        .cache_stride_w = cache.stride(1),
        .D = static_cast<uint32_t>(D.unwrap()),
        .T = static_cast<uint32_t>(draft_token_num),
    };

    const uint32_t d_pairs = params.D / 2;
    const dim3 grid{div_ceil(d_pairs, kDEThreads), static_cast<uint32_t>(B.unwrap())};
    const dim3 block{kDEThreads};
    constexpr auto kernel = draft_extend_kernel<W1, DO_TRACK, DType>;
    LaunchKernel(grid, block, dev.unwrap())(kernel, params);
  }
};

}  // namespace
