#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/once.h>
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace sglang {

using ArgMaxHandle = unsigned long long;

/// Balanced config on both B200 and H200
inline constexpr uint32_t kArgMaxThreads = 512;

/// \brief Map a float onto an unsigned key whose integer order matches the float order.
SGL_DEVICE uint32_t to_ordered(float val) {
  const auto bits = __float_as_uint(val);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

struct alignas(8) ArgMaxPair {
  float val;
  uint32_t idx;
};

/// \brief Reduce one candidate per lane to the warp's max; a value tie takes the lowest index.
SGL_DEVICE ArgMaxPair reduce_warp_argmax(ArgMaxPair pair) {
  constexpr uint32_t kPadIdx = std::numeric_limits<uint32_t>::max();
  const auto max_val = device::warp::reduce_max(pair.val);
  const auto max_idx = device::warp::reduce_min(pair.val == max_val ? pair.idx : kPadIdx);
  return ArgMaxPair{max_val, max_idx};
}

struct ArgMaxParams {
  const void* input;
  void* output;  // IdxType*, resolved by the kernel's template parameter
  ArgMaxHandle* handle;
  int64_t input_stride;
  uint32_t num_vecs;  // vocab_size / kVecSize; exact, the launcher rejects a remainder
  uint32_t batch_size;
};

template <typename DType, typename IdxType, bool kUsePDL>
__global__ void argmax_kernel(const ArgMaxParams params) {
  constexpr float kPadVal = -std::numeric_limits<float>::infinity();
  constexpr uint32_t kPadIdx = std::numeric_limits<uint32_t>::max();
  constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(DType);
  constexpr uint32_t kNumWarps = kArgMaxThreads / device::kWarpSize;
  using vec_t = device::AlignedVector<packed_t<DType>, kVecSize / 2>;
  // Loads issued before the compare chain: 2 x 32 B for fp32, 1 for 16-bit inputs, so one batch
  // is 16 elements either way (unrolling 16-bit inputs too measured slower).
  constexpr uint32_t kUnroll = kVecSize < 16 ? 16 / kVecSize : 1;

  const auto tx = threadIdx.x;
  const auto token_id = blockIdx.y;
  const auto global_id = blockIdx.x * kArgMaxThreads + tx;
  const auto num_threads = gridDim.x * kArgMaxThreads;
  const auto input_ptr = static_cast<const DType*>(params.input) + params.input_stride * token_id;

  device::PDLWaitPrimary<kUsePDL>();

  float thr_max_val = kPadVal;
  uint32_t thr_max_idx = 0;
  for (uint32_t base = global_id; base < params.num_vecs; base += kUnroll * num_threads) {
    vec_t vecs[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      if (base + u * num_threads < params.num_vecs) vecs[u].load(input_ptr, base + u * num_threads);
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const auto vid = base + u * num_threads;
      if (vid >= params.num_vecs) break;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        const auto [x, y] = device::cast<fp32x2_t>(vecs[u][i]);
        // Strict `>` while scanning upwards is what keeps the lowest index on a tie.
        if (x > thr_max_val) {
          thr_max_val = x;
          thr_max_idx = vid * kVecSize + i * 2;
        }
        if (y > thr_max_val) {
          thr_max_val = y;
          thr_max_idx = vid * kVecSize + i * 2 + 1;
        }
      }
    }
  }
  __shared__ ArgMaxPair smem[kNumWarps];
  const auto warp_max = reduce_warp_argmax({thr_max_val, thr_max_idx});
  smem[tx / device::kWarpSize] = warp_max;  // broadcast write
  __syncthreads();
  if (tx < device::kWarpSize) {
    const auto seed = tx < kNumWarps ? smem[tx] : ArgMaxPair{kPadVal, kPadIdx};
    const auto [max_val, max_idx] = reduce_warp_argmax(seed);
    if (tx == 0) {
      if (gridDim.x == 1) {
        static_cast<IdxType*>(params.output)[token_id] = static_cast<IdxType>(max_idx);
      } else {
        // -0.0 and +0.0 tie under `>` but not under `to_ordered`; key both as +0.0 so the
        // lowest-index rule also holds across CTAs.
        const auto key_val = (__float_as_uint(max_val) << 1) == 0 ? 0.0f : max_val;
        const auto payload = static_cast<ArgMaxHandle>(to_ordered(key_val)) << 32 | ~max_idx;
        atomicMax(params.handle + token_id, payload);
      }
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename IdxType, bool kUsePDL>
__global__ void argmax_finalize_kernel(const ArgMaxParams params) {
  const auto token_id = blockIdx.x * blockDim.x + threadIdx.x;
  device::PDLWaitPrimary<kUsePDL>();
  if (token_id < params.batch_size) {
    const auto payload = atomicExch(params.handle + token_id, ArgMaxHandle{0});
    static_cast<IdxType*>(params.output)[token_id] = static_cast<IdxType>(~static_cast<uint32_t>(payload));
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t kMaxBatchSize>
inline tvm::ffi::TensorView allocate_argmax_zero_buffer(cudaStream_t stream, int32_t device_id) {
  return host::init_once(std::make_pair(stream, device_id), [&] {
    constexpr auto kNumSlots = static_cast<int64_t>(kMaxBatchSize);
    auto buffer = host::ffi::empty({kNumSlots}, {kDLInt, 64, 1}, {kDLCUDA, device_id});
    CHECK_CUDA(cudaMemsetAsync(buffer.data_ptr(), 0, kNumSlots * sizeof(ArgMaxHandle), stream));
    return buffer;
  });
}

/**
 * \brief Blocks per SM for `kernel` at `kArgMaxThreads`, memoized per (kernel, device).
 *
 * `runtime::get_blocks_per_sm` is a driver query and is not itself memoized, and the kernel
 * is chosen per call by index width, so a function-local static would pin whichever width
 * ran first.
 */
inline uint32_t argmax_blocks_per_sm(void (*kernel)(const ArgMaxParams), int32_t device_id) {
  const auto key = std::make_pair(reinterpret_cast<const void*>(kernel), device_id);
  return host::init_once(key, [&] { return host::runtime::get_blocks_per_sm(kernel, kArgMaxThreads); });
}

template <typename DType, bool kUsePDL>
struct ArgmaxKernel {
  static void
  run(tvm::ffi::TensorView input, tvm::ffi::TensorView output, tvm::ffi::Optional<tvm::ffi::TensorView> workspace) {
    using namespace host;
    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    TensorMatcher({N, -1})  //
        .with_strides({-1, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<DType>()
        .ensure_alignment(device::kMaxVecBytes)
        .verify(input);
    auto index_dtype = SymbolicDType{};
    TensorMatcher({N})  //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(index_dtype)
        .verify(output);

    constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(DType);
    constexpr uint32_t kMaxGridDimY = 65535;  // CUDA limit
    const auto batch_size = static_cast<uint32_t>(N.unwrap());
    const auto vocab_size = static_cast<uint32_t>(input.size(1));
    CHECK_HOST(batch_size <= kMaxGridDimY);
    if (batch_size == 0) return;
    CHECK_HOST(vocab_size > 0) << "argmax: vocab_size must be > 0";
    CHECK_HOST(vocab_size % kVecSize == 0);
    const auto device = device_.unwrap();
    const auto stream = LaunchKernel::resolve_device(device);
    // Both widths are instantiated into the module; the index type changes nothing but
    // which of them we launch, so it is resolved here and nowhere else.
    const auto use_i32 = index_dtype.is_type<int32_t>();
    const auto kernel_0 = use_i32 ? argmax_kernel<DType, int32_t, kUsePDL>  //
                                  : argmax_kernel<DType, int64_t, kUsePDL>;
    const auto kernel_1 = use_i32 ? argmax_finalize_kernel<int32_t, kUsePDL>  //
                                  : argmax_finalize_kernel<int64_t, kUsePDL>;
    const auto num_vecs = vocab_size / kVecSize;
    const auto num_split = [&] {
      const auto blocks_per_sm = argmax_blocks_per_sm(kernel_0, device.device_id);
      const auto target_blocks = runtime::get_sm_count(device.device_id) * blocks_per_sm;
      const auto max_split = std::max(div_ceil(num_vecs, kArgMaxThreads), 1u);
      return std::clamp(target_blocks / batch_size, 1u, max_split);
    }();
    if (num_split > 1 && !workspace.has_value()) {
      constexpr uint32_t kMaxBatchSize = 16384;
      CHECK_HOST(batch_size <= kMaxBatchSize)
          << "argmax: batch_size " << batch_size << " exceeds the built-in workspace of " << kMaxBatchSize
          << " rows; pass an explicit `workspace`";
      workspace = allocate_argmax_zero_buffer<kMaxBatchSize>(stream, device.device_id);
    }
    auto handle = static_cast<ArgMaxHandle*>(nullptr);
    if (workspace.has_value()) {
      auto slots = SymbolicSize{"num_slots"};
      TensorMatcher({slots})  //
          .with_device<kDLCUDA>(device_)
          .with_dtype<int64_t>()
          .verify(workspace.value());
      CHECK_HOST(slots.unwrap() >= batch_size)
          << "argmax: workspace holds " << slots.unwrap() << " slots, need " << batch_size;
      handle = static_cast<ArgMaxHandle*>(workspace.value().data_ptr());
    }
    const auto params = ArgMaxParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .handle = handle,
        .input_stride = input.stride(0),
        .num_vecs = num_vecs,
        .batch_size = batch_size,
    };
    LaunchKernel({num_split, batch_size}, kArgMaxThreads, stream)  //
        .enable_pdl(kUsePDL)(kernel_0, params);
    if (num_split > 1) {
      constexpr uint32_t kFinalizeThreads = 256;
      LaunchKernel(div_ceil(batch_size, kFinalizeThreads), kFinalizeThreads, stream)  //
          .enable_pdl(kUsePDL)(kernel_1, params);
    }
  }
};

}  // namespace sglang
