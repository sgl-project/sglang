// Tiny moe_align_block_size for M == 1 decode: one warp replaces the
// moe_align_block_size + count_and_sort_expert_tokens kernel pair (~4.4us)
// with a single ~1.5us launch.
//
// For a single token the top-k expert ids are distinct, so the aligned
// layout is exactly: experts sorted ascending, one block per expert,
// slot i of block b = flat topk index for b's expert, remaining block
// slots padded with numel (= topk). num_tokens_post_padded = topk * block.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct AlignSingleTokenParams {
  const int32_t* __restrict__ topk_ids;  // [1, topk]
  int32_t* __restrict__ sorted_ids;      // [topk * block_size]
  int32_t* __restrict__ expert_ids;      // [topk]
  int32_t* __restrict__ num_post;        // [1]
  uint32_t topk;
  uint32_t block_size;
};

template <bool kUsePDL>
__global__ void align_single_token_kernel(const AlignSingleTokenParams __grid_constant__ params) {
  using namespace device;
  const uint32_t lane = threadIdx.x;  // one warp
  const uint32_t topk = params.topk;
  const uint32_t bs = params.block_size;

  PDLWaitPrimary<kUsePDL>();

  int32_t my_id = (lane < topk) ? params.topk_ids[lane] : INT32_MAX;

  // Rank of my expert id among the topk (ids are distinct for one token;
  // tie-break on lane keeps this robust anyway).
  uint32_t rank = 0;
  for (uint32_t j = 0; j < topk; ++j) {
    int32_t other = __shfl_sync(0xffffffff, my_id, j);
    if (other < my_id || (other == my_id && j < lane)) {
      rank++;
    }
  }

  if (lane < topk) {
    params.expert_ids[rank] = my_id;
    // block `rank`: first slot is my flat index (token 0, slot `lane`),
    // rest padded with numel (= topk).
    params.sorted_ids[rank * bs] = static_cast<int32_t>(lane);
  }
  // Fill padding cooperatively: positions not equal to a block start.
  for (uint32_t p = lane; p < topk * bs; p += 32) {
    if (p % bs != 0) {
      params.sorted_ids[p] = static_cast<int32_t>(topk);
    }
  }
  if (lane == 0) {
    params.num_post[0] = static_cast<int32_t>(topk * bs);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct AlignSingleTokenKernel {
  static constexpr auto kernel = align_single_token_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView sorted_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_post,
      int64_t block_size) {
    using namespace host;

    auto One_ = SymbolicSize{"one"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({One_, K_}).with_dtype<int32_t>().with_device(device).verify(topk_ids);

    const auto topk = static_cast<uint32_t>(K_.unwrap());
    RuntimeCheck(One_.unwrap() == 1, "moe_align_single_token requires M == 1");
    RuntimeCheck(topk <= 32, "moe_align_single_token requires topk <= 32");

    const auto params = AlignSingleTokenParams{
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .sorted_ids = static_cast<int32_t*>(sorted_ids.data_ptr()),
        .expert_ids = static_cast<int32_t*>(expert_ids.data_ptr()),
        .num_post = static_cast<int32_t*>(num_post.data_ptr()),
        .topk = topk,
        .block_size = static_cast<uint32_t>(block_size),
    };

    LaunchKernel(dim3(1), 32, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
