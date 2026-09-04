#pragma once

namespace sglang {
namespace mtp_demand_writeback {

__global__ void backup_window_mla_kernel(
    const int64_t* __restrict__ src_layers,
    const int64_t* __restrict__ dst_layers,
    const int64_t* __restrict__ src_indices,
    const int64_t* __restrict__ dst_indices,
    const int32_t* __restrict__ accept_index,
    int64_t num_items,
    int64_t item_size_bytes,
    int64_t num_layers) {
  constexpr int kWarp = 32;
  const int lane = threadIdx.x & (kWarp - 1);
  const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
  const int64_t total = num_items * num_layers;
  if (warp >= total) return;
  const int64_t item = warp % num_items;
  if (accept_index[item] < 0) return;
  const int64_t layer = warp / num_items;
  const int64_t chunks = item_size_bytes / static_cast<int64_t>(sizeof(uint64_t));
  const auto* src = reinterpret_cast<const uint64_t*>(src_layers[layer]) + src_indices[item] * chunks;
  auto* dst = reinterpret_cast<uint64_t*>(dst_layers[layer]) + dst_indices[item] * chunks;
  for (int64_t chunk = lane; chunk < chunks; chunk += kWarp) {
    uint64_t value;
    asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(value) : "l"(src + chunk) : "memory");
    asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(dst + chunk), "l"(value) : "memory");
  }
}

}  // namespace mtp_demand_writeback

void backup_mtp_demand_window_mla(
    tvm::ffi::TensorView src_layers,
    tvm::ffi::TensorView dst_layers,
    tvm::ffi::TensorView src_indices,
    tvm::ffi::TensorView dst_indices,
    tvm::ffi::TensorView accept_index,
    int64_t item_size_bytes,
    int64_t num_layers) {
  using namespace host;
  constexpr int kThreads = 128;
  constexpr int kWarps = kThreads / 32;
  RuntimeCheck(
      item_size_bytes > 0 && item_size_bytes % 8 == 0,
      "MTP Demand writeback item size must be positive and 8-byte aligned");
  RuntimeCheck(
      src_indices.numel() == dst_indices.numel() && src_indices.numel() == accept_index.numel(),
      "MTP Demand writeback index shape mismatch");
  RuntimeCheck(
      src_layers.numel() >= num_layers && dst_layers.numel() >= num_layers,
      "MTP Demand writeback layer pointer table is too small");
  const int64_t num_items = src_indices.numel();
  const int64_t total_warps = num_items * num_layers;
  const auto device = LaunchKernel::resolve_device(src_indices.device());
  LaunchKernel((total_warps + kWarps - 1) / kWarps, kThreads, device)(
      mtp_demand_writeback::backup_window_mla_kernel,
      static_cast<const int64_t*>(src_layers.data_ptr()),
      static_cast<const int64_t*>(dst_layers.data_ptr()),
      static_cast<const int64_t*>(src_indices.data_ptr()),
      static_cast<const int64_t*>(dst_indices.data_ptr()),
      static_cast<const int32_t*>(accept_index.data_ptr()),
      num_items,
      item_size_bytes,
      num_layers);
}

}  // namespace sglang
