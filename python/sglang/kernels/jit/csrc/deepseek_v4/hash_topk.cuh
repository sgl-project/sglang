#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace {

[[maybe_unused]]
SGL_DEVICE float act_sqrt_softplus(float x) {
  const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
  return sqrtf(softplus);
}

struct MoEHashTopKParams {
  const float* __restrict__ router_logits;
  const int64_t* __restrict__ input_id;
  const int32_t* __restrict__ tid2eid;
  int32_t* __restrict__ topk_ids;
  float* __restrict__ topk_weights;
  uint32_t num_tokens;
  uint32_t topk;
  uint32_t num_routed_experts;
  uint32_t num_shared_experts;
  float routed_scaling_factor;
};

template <auto Fn, bool kUsePDL>
__global__ void moe_hash_topk_fused(const MoEHashTopKParams __grid_constant__ params) {
  using namespace device;
  const auto& [
    router_logits, input_id, tid2eid, topk_ids, topk_weights, // pointers
    num_tokens, topk, num_routed_experts, num_shared_experts, routed_scaling_factor] =
      params;

  const uint32_t topk_fused = topk + num_shared_experts;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t lane_id = tid % kWarpThreads;
  if (warp_id >= num_tokens) return;
  // we can safely prefetch the token id
  const auto token_id = input_id[warp_id];

  PDLWaitPrimary<kUsePDL>();

  float routed_weight = 0.0f;
  int32_t expert_id = 0;
  if (lane_id < topk) {
    expert_id = tid2eid[token_id * topk + lane_id];
    routed_weight = Fn(router_logits[warp_id * num_routed_experts + expert_id]);
  }

  const auto routed_sum = device::warp::reduce_sum(routed_weight);
  if (lane_id < topk_fused) {
    const bool is_shared = lane_id >= topk;
    const auto output_offset = warp_id * topk_fused + lane_id;
    topk_ids[output_offset] = is_shared ? num_routed_experts + lane_id - topk : expert_id;
    topk_weights[output_offset] = is_shared ? 1.0f / routed_scaling_factor : routed_weight / routed_sum;
  }

  PDLTriggerSecondary<kUsePDL>();
}

struct TopKParams {
  int32_t* __restrict__ topk_ids;
  // Exactly one is active: ntn_ptr == nullptr means use ntn_value.
  const int32_t* __restrict__ ntn_ptr;
  int32_t ntn_value;
  int64_t stride;
  uint32_t topk;
  uint32_t num_tokens;
};

__global__ void mask_topk_ids_padded_region(const TopKParams __grid_constant__ params) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid / device::kWarpThreads;
  const uint32_t lane_id = tid % device::kWarpThreads;
  if (warp_id >= params.num_tokens || lane_id >= params.topk) return;
  device::PDLWaitPrimary<true>();
  const uint32_t num = (params.ntn_ptr != nullptr)  //
                           ? static_cast<uint32_t>(params.ntn_ptr[0])
                           : static_cast<uint32_t>(params.ntn_value);
  if (warp_id >= num) params.topk_ids[warp_id * params.stride + lane_id] = -1;
  device::PDLTriggerSecondary<true>();
}

template <auto Fn, bool kUsePDL>
struct HashTopKKernel {
  static constexpr auto kernel = moe_hash_topk_fused<Fn, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView router_logits,
      const tvm::ffi::TensorView input_id,
      const tvm::ffi::TensorView tid2eid,
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_ids,
      float routed_scaling_factor) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto E = SymbolicSize{"num_routed_experts"};
    auto K = SymbolicSize{"topk_fused"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({N, E})  //
        .with_dtype<float>()
        .with_device(device)
        .verify(router_logits);
    TensorMatcher({N})  //
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(input_id);
    TensorMatcher({-1, -1})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(tid2eid);
    TensorMatcher({N, K})  //
        .with_dtype<float>()
        .with_device(device)
        .verify(topk_weights);
    TensorMatcher({N, K})  //
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_ids);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto topk_fused = static_cast<uint32_t>(K.unwrap());
    const auto topk = static_cast<uint32_t>(tid2eid.size(1));
    const auto shared_experts = topk_fused - topk;
    RuntimeCheck(topk <= topk_fused, "HashTopKKernel requires topk <= topk_fused");
    RuntimeCheck(topk_fused <= device::kWarpThreads, "HashTopKKernel requires topk_fused <= warp size");

    const auto params = MoEHashTopKParams{
        .router_logits = static_cast<const float*>(router_logits.data_ptr()),
        .input_id = static_cast<const int64_t*>(input_id.data_ptr()),
        .tid2eid = static_cast<const int32_t*>(tid2eid.data_ptr()),
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
        .num_tokens = num_tokens,
        .topk = topk,
        .num_routed_experts = static_cast<uint32_t>(E.unwrap()),
        .num_shared_experts = shared_experts,
        .routed_scaling_factor = routed_scaling_factor,
    };
    const auto kBlockSize = 128u;
    const auto kNumWarps = kBlockSize / device::kWarpThreads;
    const auto num_blocks = div_ceil(num_tokens, kNumWarps);
    LaunchKernel(num_blocks, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

// TODO this may not be related to *hash* topk, thus may move
struct MaskKernel {
  static constexpr auto kernel = mask_topk_ids_padded_region;

  static void run(tvm::ffi::TensorView topk_ids, tvm::ffi::TensorView num_token_non_padded) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"topk"};
    auto D = SymbolicSize{"stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({N, K})  //
        .with_strides({D, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_ids);
    RuntimeCheck(num_token_non_padded.numel() == 1, "num_token_non_padded should be a scalar");
    RuntimeCheck(K.unwrap() <= device::kWarpThreads, "MaskKernel requires topk <= warp size");
    const int32_t* ntn_ptr = nullptr;
    int32_t ntn_value = 0;
    const auto ntn_dev = num_token_non_padded.device().device_type;
    if (ntn_dev == kDLCUDA) {
      RuntimeCheck(is_type<int32_t>(num_token_non_padded.dtype()), "num_token_non_padded on CUDA must be int32");
      ntn_ptr = static_cast<const int32_t*>(num_token_non_padded.data_ptr());
    } else if (ntn_dev == kDLCPU) {
      if (is_type<int32_t>(num_token_non_padded.dtype())) {
        ntn_value = *static_cast<const int32_t*>(num_token_non_padded.data_ptr());
      } else if (is_type<int64_t>(num_token_non_padded.dtype())) {
        ntn_value = static_cast<int32_t>(*static_cast<const int64_t*>(num_token_non_padded.data_ptr()));
      } else {
        RuntimeCheck(false, "num_token_non_padded on CPU must be int32 or int64");
      }
    } else {
      RuntimeCheck(false, "num_token_non_padded must be on CPU or CUDA");
    }

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = TopKParams{
        .topk_ids = static_cast<int32_t*>(topk_ids.data_ptr()),
        .ntn_ptr = ntn_ptr,
        .ntn_value = ntn_value,
        .stride = static_cast<int64_t>(D.unwrap()),
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

}  // namespace
