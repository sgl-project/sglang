#pragma once

#include <cstdint>
#include <type_traits>

#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_tile_scheduler_group_precomputed.hpp"

namespace sgl_kernel::swg_detail {

using SwgWorkTile = cutlass::gemm::kernel::detail::PrecomputedGroupWorkTile;

// Match the Humming single-warpgroup scheduler: use a 2-way launch-grid
// swizzle, but do not pad each expert's chunk-major work-map tiles.
constexpr int kSwgSchedulerMaxSwizzle = 2;
constexpr int kSwgWorkMapMaxSwizzle = 1;
constexpr int kSwgWorkMapBuilderThreads = 128;

enum class ExpertRowPolicy {
  All,
  PreferN32,
  PreferN64,
};

template <ExpertRowPolicy Policy>
CUTLASS_HOST_DEVICE bool swg_select_expert_rows(uint64_t rows) {
  if (rows == 0) {
    return false;
  }
  uint64_t const padded_n32 = ((rows + 31) / 32) * 32;
  uint64_t const padded_n64 = ((rows + 63) / 64) * 64;
  if constexpr (Policy == ExpertRowPolicy::PreferN32) {
    return padded_n32 < padded_n64;
  } else if constexpr (Policy == ExpertRowPolicy::PreferN64) {
    return padded_n64 <= padded_n32;
  }
  return true;
}

CUTLASS_HOST_DEVICE uint64_t swg_div_up(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

CUTLASS_HOST_DEVICE uint64_t swg_round_up(uint64_t value, uint64_t multiple) {
  return swg_div_up(value, multiple) * multiple;
}

CUTLASS_HOST_DEVICE int
swg_log_swizzle_size(uint64_t problem_blocks_m, uint64_t problem_blocks_n, int max_swizzle_size) {
  uint64_t const min_cta_dim = problem_blocks_m < problem_blocks_n ? problem_blocks_m : problem_blocks_n;
  if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
    return 3;
  }
  if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
    return 2;
  }
  if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
    return 1;
  }
  return 0;
}

template <int TileM, int TileN>
uint64_t swg_max_work_tiles(int groups, uint64_t total_tokens, uint64_t channels) {
  if (groups <= 0 || total_tokens == 0 || channels == 0) {
    return 0;
  }

  uint64_t const channel_tiles = swg_div_up(channels, uint64_t(TileM));
  uint64_t const total_token_tiles = swg_div_up(total_tokens, uint64_t(TileN));
  int const swizzle_log = swg_log_swizzle_size(channel_tiles, total_token_tiles, kSwgWorkMapMaxSwizzle);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const padded_channel_tiles = swg_round_up(channel_tiles, swizzle);
  uint64_t const tokens_per_padded_group = uint64_t(TileN) * swizzle;
  uint64_t const nonempty_groups = uint64_t(groups) < total_tokens ? uint64_t(groups) : total_tokens;
  uint64_t const extra_tokens = total_tokens - nonempty_groups;
  // Every non-empty group opens one swizzle-padded token-tile band. Each
  // additional band needs TileN * swizzle more routed tokens in that group.
  uint64_t const max_token_tiles = swizzle * (nonempty_groups + extra_tokens / tokens_per_padded_group);
  return padded_channel_tiles * max_token_tiles;
}

struct SwgPrecomputedWorkMap {
  torch::Tensor storage;
  torch::Tensor prebuilt_tma_desc_a;
  torch::Tensor prebuilt_tma_desc_b;
  uint32_t worker_count = 0;
  uint32_t grid_x = 0;
  uint32_t grid_y = 0;
  uint32_t tiles_per_worker = 0;
};

constexpr int kSwgPrebuiltTmaDescriptorCount = 2;
constexpr size_t kSwgPrebuiltTmaDescriptorScratchBytes = kSwgPrebuiltTmaDescriptorCount * sizeof(cute::TmaDescriptor);

CUTE_DEVICE void swg_publish_prebuilt_tma_descriptor(
    cute::TmaDescriptor const* gmem_desc, cute::TmaDescriptor& smem_desc, int publisher_warp) {
  if ((threadIdx.x >> 5) == publisher_warp) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    cute::tma_descriptor_cp_fence_release(gmem_desc, smem_desc);
    __syncwarp();
  }
}

template <class MainloopParams, class Problem>
__device__ __forceinline__ void swg_build_prebuilt_tma_descriptors(
    MainloopParams const& mainloop_params,
    Problem const& problem,
    int group,
    bool weight_desc_per_group,
    cute::TmaDescriptor* smem_descs,
    cute::TmaDescriptor* prebuilt_tma_desc_a,
    cute::TmaDescriptor* prebuilt_tma_desc_b) {
  if (weight_desc_per_group || group == 0) {
    cute::TmaDescriptor& smem_desc = smem_descs[0];
    if (threadIdx.x == 0) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> shape_a = {1, 1, 1, 1, 1};
      cute::array<uint64_t, MaxTensorRank> stride_a = {0, 0, 0, 0, 0};
      using PtrA = std::remove_reference_t<decltype(mainloop_params.ptr_A[group])>;
      PtrA ptr_a = nullptr;
      uint32_t const M = static_cast<uint32_t>(cute::get<0>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto d_a = mainloop_params.ptr_dA[group];
      auto stride_m = cute::get<0>(d_a);
      auto stride_k = cute::get<1>(d_a);
      int64_t const term_m = static_cast<int64_t>(M) * static_cast<int64_t>(stride_m);
      int64_t const term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t const stride_l = weight_desc_per_group ? int64_t(0) : (term_m > term_k ? term_m : term_k);
      auto tensor_a = cute::make_tensor(
          ptr_a,
          cute::make_layout(
              cute::make_shape(
                  M, K, weight_desc_per_group ? uint32_t(1) : static_cast<uint32_t>(mainloop_params.num_groups)),
              cute::make_stride(stride_m, stride_k, stride_l)));

      smem_desc = *mainloop_params.tma_load_a.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(
          smem_desc, mainloop_params.ptr_A[weight_desc_per_group ? group : 0]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, shape_a, stride_a);
      using ElementA = std::remove_cv_t<std::remove_pointer_t<PtrA>>;
      for (uint64_t& stride : stride_a) {
        stride = (stride * cutlass::sizeof_bits<ElementA>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, shape_a, stride_a);
    }
    swg_publish_prebuilt_tma_descriptor(prebuilt_tma_desc_a + (weight_desc_per_group ? group : 0), smem_desc, 0);
  }

  if (cute::get<1>(problem) == 0) {
    return;
  }

  cute::TmaDescriptor& smem_desc = smem_descs[1];
  if (threadIdx.x == 32) {
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> shape_b = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> stride_b = {0, 0, 0, 0, 0};
    using PtrB = std::remove_reference_t<decltype(mainloop_params.ptr_B[group])>;
    PtrB ptr_b = nullptr;
    uint32_t const N = static_cast<uint32_t>(cute::get<1>(problem));
    uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
    auto d_b = mainloop_params.ptr_dB[group];
    auto stride_n = cute::get<0>(d_b);
    auto stride_k = cute::get<1>(d_b);
    auto tensor_b = cute::make_tensor(
        ptr_b,
        cute::make_layout(cute::make_shape(N, K, uint32_t(1)), cute::make_stride(stride_n, stride_k, int64_t(0))));

    smem_desc = *mainloop_params.tma_load_b.get_tma_descriptor();
    cute::tma_descriptor_replace_addr_in_shared_mem(smem_desc, mainloop_params.ptr_B[group]);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, shape_b, stride_b);
    using ElementB = std::remove_cv_t<std::remove_pointer_t<PtrB>>;
    for (uint64_t& stride : stride_b) {
      stride = (stride * cutlass::sizeof_bits<ElementB>::value) / 8;
    }
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, shape_b, stride_b);
  }
  swg_publish_prebuilt_tma_descriptor(prebuilt_tma_desc_b + group, smem_desc, 1);
}

__device__ __forceinline__ uint64_t swg_make_work_tile(
    uint64_t global_linear_idx,
    uint64_t local_linear_idx,
    int group,
    uint64_t problem_blocks_m,
    int swizzle_log,
    int gemm_grid_x,
    int gemm_grid_y) {
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);
  uint64_t const worker_id = total_grid_size == 0 ? 0 : global_linear_idx % total_grid_size;
  uint64_t const cluster_minor_offset = worker_id % uint64_t(gemm_grid_y);
  uint64_t const cluster_id = local_linear_idx;

  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const offset = cluster_id & (swizzle - 1);
  uint64_t const extra = cluster_id >> swizzle_log;
  uint64_t const cluster_idx_minor_div_swizzle = extra / problem_blocks_m;
  uint64_t const cluster_idx_major = extra % problem_blocks_m;
  uint64_t const cluster_idx_minor = cluster_idx_minor_div_swizzle * swizzle + offset;

  return SwgWorkTile::pack(cluster_idx_major, cluster_idx_minor + cluster_minor_offset, uint64_t(group));
}

struct SwgGroupInfo {
  uint64_t problem_blocks_m = 0;
  uint64_t group_tiles = 0;
  int swizzle_log = 0;
};

template <int TileM, int TileN, ExpertRowPolicy RowPolicy, class Problem>
__device__ __forceinline__ SwgGroupInfo swg_group_info(Problem const& problem) {
  uint64_t const rows = uint64_t(cute::get<1>(problem));
  if (!swg_select_expert_rows<RowPolicy>(rows)) {
    return {};
  }
  uint64_t const channel_tiles = swg_div_up(uint64_t(cute::get<0>(problem)), uint64_t(TileM));
  uint64_t const token_tiles = swg_div_up(rows, uint64_t(TileN));
  int const swizzle_log = swg_log_swizzle_size(channel_tiles, token_tiles, kSwgWorkMapMaxSwizzle);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const problem_blocks_m = swg_round_up(channel_tiles, swizzle);
  uint64_t const problem_blocks_n = swg_round_up(token_tiles, swizzle);

  if (problem_blocks_m > uint64_t(SwgWorkTile::ChannelMask + 1) ||
      problem_blocks_n > uint64_t(SwgWorkTile::TokenMask + 1)) {
    asm volatile("trap;");
  }
  return {problem_blocks_m, problem_blocks_m * problem_blocks_n, swizzle_log};
}

template <int TileM, int TileN, bool ChunkMajorWorkMap, ExpertRowPolicy RowPolicy, class Problem, class MainloopParams>
__global__ void build_swg_precomputed_work_map_kernel(
    Problem const* problem_shapes,
    int groups,
    int gemm_grid_x,
    int gemm_grid_y,
    uint32_t work_tiles_per_worker,
    uint64_t* work_tiles,
    MainloopParams mainloop_params,
    int32_t const* expert_offsets,
    int32_t const* expert_ids,
    uint64_t* activation_ptrs,
    uint64_t* weight_ptrs,
    uint64_t* output_ptrs,
    uint64_t* activation_scale_ptrs,
    uint64_t* weight_scale_ptrs,
    uint8_t* activation_base,
    uint8_t* weight_base,
    uint8_t* output_base,
    uint8_t* activation_scale_base,
    uint8_t* weight_scale_base,
    uint64_t output_element_bytes,
    uint64_t output_channels,
    uint64_t weight_channels,
    uint64_t reduction_channels,
    bool weight_desc_per_group,
    cute::TmaDescriptor* prebuilt_tma_desc_a,
    cute::TmaDescriptor* prebuilt_tma_desc_b) {
  int const tid = threadIdx.x;
  uint64_t const worker_count = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);

  if (groups <= 0) {
    if (blockIdx.x == 0) {
      for (uint64_t worker = uint64_t(tid); worker < worker_count; worker += uint64_t(blockDim.x)) {
        work_tiles[worker * uint64_t(work_tiles_per_worker)] = SwgWorkTile::Invalid;
      }
    }
    return;
  }

  int const group = int(blockIdx.x);
  if (group >= groups) {
    return;
  }

  extern __shared__ __align__(64) unsigned char shared_storage[];
  auto* smem_descs = reinterpret_cast<cute::TmaDescriptor*>(shared_storage);
  auto* prefix_partials = reinterpret_cast<unsigned long long*>(shared_storage + kSwgPrebuiltTmaDescriptorScratchBytes);
  auto* total_partials = prefix_partials + blockDim.x;
  auto* group_info_storage = total_partials + blockDim.x;

  if (tid == 0) {
    int const expert = expert_ids != nullptr ? expert_ids[group] : group;
    uint64_t const token_offset = static_cast<uint64_t>(expert_offsets[group]);
    activation_ptrs[group] = reinterpret_cast<uint64_t>(activation_base + token_offset * reduction_channels);
    weight_ptrs[group] =
        reinterpret_cast<uint64_t>(weight_base + uint64_t(expert) * weight_channels * reduction_channels / 2);
    output_ptrs[group] =
        reinterpret_cast<uint64_t>(output_base + token_offset * output_channels * output_element_bytes);
    activation_scale_ptrs[group] = reinterpret_cast<uint64_t>(activation_scale_base + token_offset * sizeof(float));
    weight_scale_ptrs[group] =
        reinterpret_cast<uint64_t>(weight_scale_base + uint64_t(expert) * weight_channels * reduction_channels / 32);

    SwgGroupInfo const info = swg_group_info<TileM, TileN, RowPolicy>(problem_shapes[group]);
    group_info_storage[0] = static_cast<unsigned long long>(info.problem_blocks_m);
    group_info_storage[1] = static_cast<unsigned long long>(info.group_tiles);
    group_info_storage[2] = static_cast<unsigned long long>(info.swizzle_log);
  }

  __syncthreads();
  swg_build_prebuilt_tma_descriptors(
      mainloop_params,
      problem_shapes[group],
      group,
      weight_desc_per_group,
      smem_descs,
      prebuilt_tma_desc_a,
      prebuilt_tma_desc_b);

  uint64_t prefix_sum = 0;
  uint64_t total_sum = 0;
  for (int scan_group = tid; scan_group < groups; scan_group += blockDim.x) {
    SwgGroupInfo const info = swg_group_info<TileM, TileN, RowPolicy>(problem_shapes[scan_group]);
    total_sum += info.group_tiles;
    if (scan_group < group) {
      prefix_sum += info.group_tiles;
    }
  }
  prefix_partials[tid] = static_cast<unsigned long long>(prefix_sum);
  total_partials[tid] = static_cast<unsigned long long>(total_sum);
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      prefix_partials[tid] += prefix_partials[tid + offset];
      total_partials[tid] += total_partials[tid + offset];
    }
    __syncthreads();
  }

  uint64_t const group_start = static_cast<uint64_t>(prefix_partials[0]);
  uint64_t const total_tiles = static_cast<uint64_t>(total_partials[0]);
  uint64_t const problem_blocks_m = static_cast<uint64_t>(group_info_storage[0]);
  uint64_t const group_tiles = static_cast<uint64_t>(group_info_storage[1]);
  int const swizzle_log = static_cast<int>(group_info_storage[2]);
  uint64_t const tiles_per_worker = total_tiles == 0 ? 1 : swg_div_up(total_tiles, worker_count);

  for (uint64_t local_tile = uint64_t(tid); local_tile < group_tiles; local_tile += uint64_t(blockDim.x)) {
    uint64_t const global_tile = group_start + local_tile;
    uint64_t const packed_tile =
        swg_make_work_tile(global_tile, local_tile, group, problem_blocks_m, swizzle_log, gemm_grid_x, gemm_grid_y);
    if constexpr (ChunkMajorWorkMap) {
      uint64_t const worker = global_tile / tiles_per_worker;
      uint64_t const worker_tile = global_tile % tiles_per_worker;
      work_tiles[worker * uint64_t(work_tiles_per_worker) + worker_tile] = packed_tile;
    } else {
      work_tiles[global_tile] = packed_tile;
    }
  }

  if (group == groups - 1) {
    for (uint64_t worker = uint64_t(tid); worker < worker_count; worker += uint64_t(blockDim.x)) {
      if constexpr (ChunkMajorWorkMap) {
        uint64_t const worker_start = worker * tiles_per_worker;
        uint64_t const worker_tile_count =
            worker_start < total_tiles
                ? ((total_tiles - worker_start) < tiles_per_worker ? total_tiles - worker_start : tiles_per_worker)
                : 0;
        work_tiles[worker * uint64_t(work_tiles_per_worker) + worker_tile_count] = SwgWorkTile::Invalid;
      } else {
        uint64_t invalid_tile = worker;
        if (invalid_tile < total_tiles) {
          invalid_tile += swg_div_up(total_tiles - invalid_tile, worker_count) * worker_count;
        }
        work_tiles[invalid_tile] = SwgWorkTile::Invalid;
      }
    }
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class Gemm, class Problem>
SwgPrecomputedWorkMap build_swg_precomputed_work_map(
    Problem const* problem_shapes,
    int groups,
    uint64_t total_tokens,
    uint64_t channels,
    dim3 const& grid_shape,
    bool weight_desc_per_group,
    torch::Device device) {
  uint64_t const worker_count_u64 = uint64_t(grid_shape.x) * uint64_t(grid_shape.y) * uint64_t(grid_shape.z);
  TORCH_CHECK(worker_count_u64 > 0, "SWG precomputed work map requires workers");
  TORCH_CHECK(worker_count_u64 <= uint64_t(UINT32_MAX), "SWG precomputed work map worker count exceeds uint32");
  TORCH_CHECK(grid_shape.z == 1, "SWG precomputed grouped scheduler requires a 2D launch grid");
  TORCH_CHECK(groups <= int(SwgWorkTile::ExpertMask + 1), "SWG precomputed work map expert index exceeds packed limit");

  constexpr int TileM = Gemm::SingleWarpgroupTileM;
  constexpr int TileN = Gemm::SingleWarpgroupTileN;
  uint64_t const channel_tiles = swg_div_up(channels, uint64_t(TileM));
  uint64_t const total_token_tiles = swg_div_up(total_tokens, uint64_t(TileN));
  uint64_t const max_swizzle = uint64_t(1)
                               << swg_log_swizzle_size(channel_tiles, total_token_tiles, kSwgWorkMapMaxSwizzle);
  uint64_t const max_tiles = swg_max_work_tiles<TileM, TileN>(groups, total_tokens, channels);
  TORCH_CHECK(
      swg_round_up(channel_tiles, max_swizzle) <= SwgWorkTile::ChannelMask + 1,
      "SWG precomputed work map channel index exceeds packed limit");
  TORCH_CHECK(
      swg_round_up(total_token_tiles, max_swizzle) <= SwgWorkTile::TokenMask + 1,
      "SWG precomputed work map token index exceeds packed limit");

  uint64_t const work_tiles_per_worker_u64 = swg_div_up(max_tiles, worker_count_u64) + 1;
  uint64_t const capacity =
      Gemm::UseChunkMajorWorkMap ? worker_count_u64 * work_tiles_per_worker_u64 : max_tiles + worker_count_u64;
  TORCH_CHECK(
      work_tiles_per_worker_u64 <= uint64_t(UINT32_MAX), "SWG precomputed work map worker chunk exceeds uint32");
  TORCH_CHECK(
      capacity <= uint64_t(INT64_MAX) / sizeof(uint64_t), "SWG precomputed work map capacity exceeds tensor size");

  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  SwgPrecomputedWorkMap result;
  result.storage = torch::empty(int64_t(capacity * sizeof(uint64_t)), options);
  result.prebuilt_tma_desc_a =
      torch::empty(int64_t((weight_desc_per_group && groups > 0 ? groups : 1) * sizeof(cute::TmaDescriptor)), options);
  result.prebuilt_tma_desc_b = torch::empty(int64_t((groups > 0 ? groups : 1) * sizeof(cute::TmaDescriptor)), options);
  result.worker_count = static_cast<uint32_t>(worker_count_u64);
  result.grid_x = grid_shape.x;
  result.grid_y = grid_shape.y;
  result.tiles_per_worker = static_cast<uint32_t>(work_tiles_per_worker_u64);
  return result;
}

template <class Gemm, class Problem, class MainloopParams>
void launch_swg_precomputed_work_map(
    SwgPrecomputedWorkMap const& work_map,
    Problem const* problem_shapes,
    int groups,
    MainloopParams const& mainloop_params,
    torch::Tensor const& expert_offsets,
    std::optional<torch::Tensor> const& expert_ids,
    torch::Tensor const& activation_ptrs,
    torch::Tensor const& weight_ptrs,
    torch::Tensor const& output_ptrs,
    torch::Tensor const& activation_scale_ptrs,
    torch::Tensor const& weight_scale_ptrs,
    torch::Tensor const& activation,
    torch::Tensor const& weight,
    torch::Tensor const& output,
    torch::Tensor const& activation_scale,
    torch::Tensor const& weight_scale,
    bool weight_desc_per_group,
    cudaStream_t stream) {
  constexpr int TileM = Gemm::SingleWarpgroupTileM;
  constexpr int TileN = Gemm::SingleWarpgroupTileN;
  size_t const scheduler_smem =
      kSwgPrebuiltTmaDescriptorScratchBytes + size_t(kSwgWorkMapBuilderThreads * 2 + 3) * sizeof(unsigned long long);
  dim3 const scheduler_grid(groups > 0 ? groups : 1);
  build_swg_precomputed_work_map_kernel<TileM, TileN, Gemm::UseChunkMajorWorkMap, Gemm::ExpertRows>
      <<<scheduler_grid, kSwgWorkMapBuilderThreads, scheduler_smem, stream>>>(
          problem_shapes,
          groups,
          static_cast<int>(work_map.grid_x),
          static_cast<int>(work_map.grid_y),
          work_map.tiles_per_worker,
          static_cast<uint64_t*>(work_map.storage.data_ptr()),
          mainloop_params,
          static_cast<int32_t const*>(expert_offsets.data_ptr()),
          expert_ids.has_value() ? static_cast<int32_t const*>(expert_ids->data_ptr()) : nullptr,
          static_cast<uint64_t*>(activation_ptrs.data_ptr()),
          static_cast<uint64_t*>(weight_ptrs.data_ptr()),
          static_cast<uint64_t*>(output_ptrs.data_ptr()),
          static_cast<uint64_t*>(activation_scale_ptrs.data_ptr()),
          static_cast<uint64_t*>(weight_scale_ptrs.data_ptr()),
          static_cast<uint8_t*>(activation.data_ptr()),
          static_cast<uint8_t*>(weight.data_ptr()),
          static_cast<uint8_t*>(output.data_ptr()),
          static_cast<uint8_t*>(activation_scale.data_ptr()),
          static_cast<uint8_t*>(weight_scale.data_ptr()),
          static_cast<uint64_t>(output.element_size()),
          static_cast<uint64_t>(output.size(1)),
          static_cast<uint64_t>(weight.size(1)),
          static_cast<uint64_t>(activation.size(1)),
          weight_desc_per_group,
          static_cast<cute::TmaDescriptor*>(work_map.prebuilt_tma_desc_a.data_ptr()),
          static_cast<cute::TmaDescriptor*>(work_map.prebuilt_tma_desc_b.data_ptr()));
  TORCH_CHECK(cudaPeekAtLastError() == cudaSuccess, "Failed to launch SWG precomputed work-map kernel");
}

}  // namespace sgl_kernel::swg_detail
