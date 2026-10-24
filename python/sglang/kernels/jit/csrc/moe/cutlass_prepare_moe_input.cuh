#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {
namespace cutlass_moe {

template <bool kTransposeProblemSizes>
__global__ void count_expert_routes(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ problem_sizes2,
    int32_t num_routes,
    int32_t n,
    int32_t k) {
  __shared__ int32_t count;
  if (threadIdx.x == 0) count = 0;
  __syncthreads();

  const int32_t expert = blockIdx.x;
  int32_t local_count = 0;
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(num_routes); i += blockDim.x) {
    local_count += topk_ids[i] == expert;
  }
  atomicAdd(&count, local_count);
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr int kCountColumn = kTransposeProblemSizes ? 1 : 0;
    constexpr int kOutputColumn = 1 - kCountColumn;
    problem_sizes1[expert * 3 + kCountColumn] = count;
    problem_sizes1[expert * 3 + kOutputColumn] = 2 * n;
    problem_sizes1[expert * 3 + 2] = k;
    problem_sizes2[expert * 3 + kCountColumn] = count;
    problem_sizes2[expert * 3 + kOutputColumn] = k;
    problem_sizes2[expert * 3 + 2] = n;
  }
}

template <bool kTransposeProblemSizes>
__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ blockscale_offsets,
    int32_t num_experts) {
  int32_t offset = 0;
  int32_t blockscale_offset = 0;
  expert_offsets[0] = 0;
  if (blockscale_offsets != nullptr) blockscale_offsets[0] = 0;
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    constexpr int kCountColumn = kTransposeProblemSizes ? 1 : 0;
    const int32_t count = problem_sizes1[expert * 3 + kCountColumn];
    offset += count;
    expert_offsets[expert + 1] = offset;
    if (blockscale_offsets != nullptr) {
      blockscale_offset += (count + 127) / 128 * 128;
      blockscale_offsets[expert + 1] = blockscale_offset;
    }
  }
}

__global__ void complete_route_maps(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ input_permutation,
    int32_t* __restrict__ output_permutation,
    int32_t num_routes,
    int32_t topk,
    int32_t num_experts) {
  const int32_t expert = blockIdx.x;
  // Exactly one CTA owns each expert's interval, so its cursor is block-local.
  __shared__ int32_t cursor;
  if (threadIdx.x == 0) cursor = expert_offsets[expert];
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(num_routes); i += blockDim.x) {
    const int32_t route = topk_ids[i];
    if (route == expert) {
      const int32_t destination = atomicAdd(&cursor, 1);
      input_permutation[destination] = i / topk;
      output_permutation[i] = destination;
    } else if (expert == 0 && (route < 0 || route >= num_experts)) {
      output_permutation[i] = -1;
    }
  }

  // The grouped GEMMs use only [0, live); full-capacity gathers also read the tail.
  const int32_t live = expert_offsets[num_experts];
  const int64_t thread = static_cast<int64_t>(expert) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(num_experts) * blockDim.x;
  for (int64_t i = live + thread; i < num_routes; i += stride) {
    input_permutation[i] = 0;
  }
}

}  // namespace cutlass_moe

/// \brief Prepare CUTLASS grouped-GEMM shapes and complete route maps.
/// \param input_permutation Destination-to-token map, with zero-filled unused capacity.
/// \param output_permutation Route-to-destination map, with -1 for ids outside [0, num_experts).
/// \param blockscale_offsets Optional MXFP8 row offsets, rounded up to 128 per expert.
template <bool kTransposeProblemSizes = false>
inline void cutlass_prepare_moe_input(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView expert_offsets,
    tvm::ffi::TensorView problem_sizes1,
    tvm::ffi::TensorView problem_sizes2,
    tvm::ffi::TensorView input_permutation,
    tvm::ffi::TensorView output_permutation,
    int64_t num_experts,
    int64_t n,
    int64_t k,
    tvm::ffi::Optional<tvm::ffi::TensorView> blockscale_offsets) {
  using namespace host;
  constexpr int64_t max_index = std::numeric_limits<int32_t>::max();
  CHECK_HOST(num_experts > 0 && num_experts <= max_index / 3);
  CHECK_HOST(n > 0 && n <= max_index / 2 && k > 0 && k <= max_index);

  auto device = SymbolicDevice{};
  auto tokens = SymbolicSize{"tokens"};
  auto topk = SymbolicSize{"topk"};
  TensorMatcher({tokens, topk})
      .with_strides({-1, -1})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(topk_ids);
  const int64_t num_routes = topk_ids.numel();
  // Empty PyTorch dimensions can have noncanonical strides; no ID is read.
  RuntimeCheck(num_routes == 0 || topk_ids.is_contiguous(), "topk_ids must be contiguous when nonempty");
  CHECK_HOST(num_routes <= max_index);
  TensorMatcher({num_experts + 1}).with_dtype<int32_t>().with_device(device).verify(expert_offsets);
  TensorMatcher({num_experts, 3})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(problem_sizes1)
      .verify(problem_sizes2);
  TensorMatcher({num_routes})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(input_permutation)
      .verify(output_permutation);

  int32_t* blockscale_ptr = nullptr;
  if (blockscale_offsets.has_value()) {
    TensorMatcher({num_experts + 1}).with_dtype<int32_t>().with_device(device).verify(blockscale_offsets.value());
    CHECK_HOST(num_routes + 127 * std::min(num_experts, num_routes) <= max_index);
    blockscale_ptr = static_cast<int32_t*>(blockscale_offsets.value().data_ptr());
  }

  const auto threads = static_cast<uint32_t>(std::min<int64_t>(512, std::max<int64_t>(1, num_routes)));
  const auto experts = static_cast<uint32_t>(num_experts);
  const auto routes = static_cast<int32_t>(num_routes);
  const auto* ids = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* sizes1 = static_cast<int32_t*>(problem_sizes1.data_ptr());
  auto* sizes2 = static_cast<int32_t*>(problem_sizes2.data_ptr());
  auto* offsets = static_cast<int32_t*>(expert_offsets.data_ptr());
  const auto stream = LaunchKernel::resolve_device(device.unwrap());

  LaunchKernel(dim3(experts), dim3(threads), stream)(
      cutlass_moe::count_expert_routes<kTransposeProblemSizes>,
      ids,
      sizes1,
      sizes2,
      routes,
      static_cast<int32_t>(n),
      static_cast<int32_t>(k));
  LaunchKernel(dim3(1), dim3(1), stream)(
      cutlass_moe::compute_expert_offsets<kTransposeProblemSizes>,
      sizes1,
      offsets,
      blockscale_ptr,
      static_cast<int32_t>(num_experts));
  if (num_routes != 0) {
    LaunchKernel(dim3(experts), dim3(threads), stream)(
        cutlass_moe::complete_route_maps,
        ids,
        offsets,
        static_cast<int32_t*>(input_permutation.data_ptr()),
        static_cast<int32_t*>(output_permutation.data_ptr()),
        routes,
        static_cast<int32_t>(topk.unwrap()),
        static_cast<int32_t>(num_experts));
  }
}

}  // namespace sglang
