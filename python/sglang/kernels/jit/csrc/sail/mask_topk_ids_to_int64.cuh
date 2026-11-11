#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

struct MaskCastTopKParams {
  const int32_t* __restrict__ in_topk_ids;
  int64_t* __restrict__ out_topk_ids;
  const int32_t* __restrict__ ntn_ptr;
  int32_t ntn_value;
  int64_t in_stride;
  int64_t out_stride;
  uint32_t topk;
  uint32_t num_tokens;
};

__global__ void mask_topk_ids_padded_region_cast_to_i64(const MaskCastTopKParams __grid_constant__ params) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / device::kWarpThreads;
  const uint32_t lane_id = tid % device::kWarpThreads;
  if (warp_id >= params.num_tokens || lane_id >= params.topk) return;
  device::PDLWaitPrimary<true>();
  const uint32_t num = (params.ntn_ptr != nullptr)  //
                           ? static_cast<uint32_t>(params.ntn_ptr[0])
                           : static_cast<uint32_t>(params.ntn_value);
  const auto in_off = static_cast<int64_t>(warp_id) * params.in_stride + lane_id;
  const auto out_off = static_cast<int64_t>(warp_id) * params.out_stride + lane_id;
  const int64_t v = (warp_id >= num) ? int64_t{-1} : static_cast<int64_t>(params.in_topk_ids[in_off]);
  params.out_topk_ids[out_off] = v;
  device::PDLTriggerSecondary<true>();
}

struct MaskCastInt64Kernel {
  static constexpr auto kernel = mask_topk_ids_padded_region_cast_to_i64;

  static void
  run(tvm::ffi::TensorView in_topk_ids, tvm::ffi::TensorView out_topk_ids, tvm::ffi::TensorView num_token_non_padded) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"topk"};
    auto D_in = SymbolicSize{"in_stride"};
    auto D_out = SymbolicSize{"out_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, K})  //
        .with_strides({D_in, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(in_topk_ids);
    TensorMatcher({N, K})  //
        .with_strides({D_out, 1})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(out_topk_ids);
    CHECK_HOST(num_token_non_padded.numel() == 1) << "num_token_non_padded must be a scalar";
    CHECK_HOST(K.unwrap() <= device::kWarpThreads) << "topk must not exceed the logical warp width";
    CHECK_HOST(N.unwrap() <= std::numeric_limits<int32_t>::max() / device::kWarpThreads);
    const int32_t* ntn_ptr = nullptr;
    int32_t ntn_value = 0;
    const auto ntn_dev = num_token_non_padded.device().device_type;
    if (ntn_dev == kDLCUDA) {
      CHECK_HOST(is_type<int32_t>(num_token_non_padded.dtype())) << "CUDA cutoff must be int32";
      CHECK_HOST(num_token_non_padded.device().device_id == device.unwrap().device_id)
          << "cutoff and TopK IDs must share a device";
      ntn_ptr = static_cast<const int32_t*>(num_token_non_padded.data_ptr());
    } else if (ntn_dev == kDLCPU) {
      int64_t value;
      if (is_type<int32_t>(num_token_non_padded.dtype())) {
        value = *static_cast<const int32_t*>(num_token_non_padded.data_ptr());
      } else if (is_type<int64_t>(num_token_non_padded.dtype())) {
        value = *static_cast<const int64_t*>(num_token_non_padded.data_ptr());
      } else {
        Panic("CPU cutoff must be int32 or int64");
      }
      CHECK_HOST(value >= 0 && value <= N.unwrap()) << "cutoff must be in [0, num_tokens]";
      ntn_value = static_cast<int32_t>(value);
    } else {
      Panic("cutoff must be on CPU or CUDA");
    }
    if (N.unwrap() == 0 || K.unwrap() == 0) return;

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = MaskCastTopKParams{
        .in_topk_ids = static_cast<const int32_t*>(in_topk_ids.data_ptr()),
        .out_topk_ids = static_cast<int64_t*>(out_topk_ids.data_ptr()),
        .ntn_ptr = ntn_ptr,
        .ntn_value = ntn_value,
        .in_stride = static_cast<int64_t>(D_in.unwrap()),
        .out_stride = static_cast<int64_t>(D_out.unwrap()),
        .topk = static_cast<uint32_t>(K.unwrap()),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128u;
    const auto kNumWarps = kBlockSize / device::kWarpThreads;
    const auto num_blocks = div_ceil(num_tokens, kNumWarps);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(true)(kernel, params);
  }
};

}  // namespace sglang
