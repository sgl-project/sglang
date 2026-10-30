#pragma once

#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

namespace sglang {

__global__ void dummy_probe_kernel() {}

uint32_t get_max_active_clusters(uint32_t cluster_size, uint32_t num_waves) {
#if !SGL_ARCH_HOPPER_OR_GREATER
  host::Panic("cluster is not supported on arch before CUDA sm90");
#else
  int device;
  int max_threads_per_sm;
  int smem_per_sm;
  int smem_per_block;
  int num_clusters;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // A block caps at 1024 threads, so threads alone cannot pin `num_waves` blocks
  // per SM; spend the shared budget instead, less the per-block driver reserve,
  // floored to the 1 KiB granularity so the driver cannot round it back up.
  const auto reserved = static_cast<uint32_t>(smem_per_sm - smem_per_block);
  const auto budget = static_cast<uint32_t>(smem_per_sm) / num_waves;
  const auto smem = (std::min(budget - std::min(budget, reserved), static_cast<uint32_t>(smem_per_block))) & ~1023u;
  const auto num_warps = std::max(1u, std::min(1024u, static_cast<uint32_t>(max_threads_per_sm) / num_waves) / 32);

  const auto kernel = reinterpret_cast<const void*>(dummy_probe_kernel);
  if (cluster_size > 8) {
    // Widths above 8 are non-portable and the query rejects them without this.
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  }
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int probed_occupancy = 0;
  const auto block_size = 32 * num_warps;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&probed_occupancy, dummy_probe_kernel, block_size, smem));
  CHECK_HOST(probed_occupancy == static_cast<int>(num_waves));
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3{cluster_size};
  config.blockDim = dim3{block_size};
  config.dynamicSmemBytes = smem;
  config.numAttrs = 1;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {cluster_size, 1, 1};
  config.attrs = &attr;
  CHECK_CUDA(cudaOccupancyMaxActiveClusters(&num_clusters, kernel, &config));
  return num_clusters;
#endif
}

}  // namespace sglang
