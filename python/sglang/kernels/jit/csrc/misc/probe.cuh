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

  // Threads alone cannot pin the probe to `num_waves` blocks per SM: a block
  // caps at 1024, so num_waves == 1 still leaves room for a second block. Spend
  // the shared budget instead -- what one block gets at this occupancy, less the
  // per-block driver reserve, floored to the 1 KiB allocation granularity so the
  // driver cannot round it back up and squeeze out a block.
  const auto reserved = static_cast<uint32_t>(smem_per_sm - smem_per_block);
  const auto budget = static_cast<uint32_t>(smem_per_sm) / num_waves;
  const auto smem = (min(budget - min(budget, reserved), static_cast<uint32_t>(smem_per_block))) & ~uint32_t{1023};
  const auto num_warps = max(1u, min(1024u, static_cast<uint32_t>(max_threads_per_sm) / num_waves) / 32);

  // Widths above 8 are non-portable and the query rejects them without this.
  CHECK_CUDA(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(dummy_probe_kernel), cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CHECK_CUDA(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(dummy_probe_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem)));

  cudaLaunchConfig_t config = {};  // stream/dynamicSmemBytes must not be garbage
  config.gridDim = dim3{cluster_size, 1024u};
  config.blockDim = dim3{32, num_warps};
  config.dynamicSmemBytes = smem;
  config.numAttrs = 1;
  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {cluster_size, 1, 1};
  config.attrs = &attr;
  CHECK_CUDA(cudaOccupancyMaxActiveClusters(&num_clusters, dummy_probe_kernel, &config));
  return num_clusters;
#endif
}

}  // namespace sglang
