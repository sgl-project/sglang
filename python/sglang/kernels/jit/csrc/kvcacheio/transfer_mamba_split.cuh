#pragma once

// Mamba state host<->device copies over a CTA quota (SGLANG_HICACHE_MAMBA_{LOAD,BACKUP}_BLOCK_QUOTA). The kernels in
// transfer_mamba.cuh copy each (slot, layer) item with one CTA on min(2, items) CTAs, so one request's per-layer
// state crosses the host link through a single SM. Here each item is cut into chunk_bytes pieces spread over
// min(quota, pieces) CTAs; every 16-byte word is copied once with the same load_nc/store_nc pair, so the bytes match.

#include <sgl_kernel/runtime.cuh>

#include "hicache.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

namespace mamba_split {

constexpr int kThreads = 1024;
constexpr int64_t kBytesPerThreadPerStep = 16;
constexpr int64_t kBytesPerBlockPerStep = kThreads * kBytesPerThreadPerStep;
// Measured on a C2C-attached host: 4 independent loads per thread give ~1.6x the per-CTA rate of the stock
// one-ahead loop; 8 gave nothing more.
constexpr int kWordsInFlight = 4;

struct Params {
  const char* __restrict__ src_base;         // load: page_first host pool
  char* __restrict__ dst_base;               // load: one device layer; backup: page_first host pool
  const uintptr_t* __restrict__ layer_ptrs;  // backup: device layer base pointers
  const int64_t* __restrict__ src_indices;
  const int64_t* __restrict__ dst_indices;
  int64_t item_size;       // bytes of one slot's state in one layer
  int64_t src_layout_dim;  // load: bytes of one host slot over all layers
  int64_t dst_layout_dim;  // backup: bytes of one host slot over all layers
  int64_t layer_id;        // load only
  int64_t num_items;
  int64_t num_layers;   // backup only
  int64_t chunk_bytes;  // multiple of 16
  int64_t chunks_per_item;
};

// Copies `len` bytes (a multiple of 16): each round a thread loads kWordsInFlight words one block step apart,
// then stores them.
SGL_DEVICE void copy_span(const char* src, char* dst, const int64_t len) {
  for (int64_t off0 = static_cast<int64_t>(threadIdx.x) * kBytesPerThreadPerStep; off0 < len;
       off0 += kWordsInFlight * kBytesPerBlockPerStep) {
    uint4 v[kWordsInFlight];
#pragma unroll
    for (int k = 0; k < kWordsInFlight; ++k) {
      const int64_t off = off0 + k * kBytesPerBlockPerStep;
      if (off < len) v[k] = device::details::load_nc(reinterpret_cast<const uint4*>(src + off));
    }
#pragma unroll
    for (int k = 0; k < kWordsInFlight; ++k) {
      const int64_t off = off0 + k * kBytesPerBlockPerStep;
      if (off < len) device::details::store_nc(reinterpret_cast<uint4*>(dst + off), v[k]);
    }
  }
}

// Load, one layer. Work unit = (item, chunk), chunk fastest.
__global__ __launch_bounds__(kThreads, 1) void load_kernel(const __grid_constant__ Params p) {
  const int64_t total = p.num_items * p.chunks_per_item;
  for (int64_t unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int64_t item_id = unit / p.chunks_per_item;
    const int64_t offset = (unit - item_id * p.chunks_per_item) * p.chunk_bytes;
    const int64_t len = min(p.chunk_bytes, p.item_size - offset);
    const char* src = p.src_base + p.src_indices[item_id] * p.src_layout_dim + p.layer_id * p.item_size + offset;
    char* dst = p.dst_base + p.dst_indices[item_id] * p.item_size + offset;
    copy_span(src, dst, len);
  }
}

// Backup, all layers. Work unit = (item, layer, chunk), chunk fastest.
__global__ __launch_bounds__(kThreads, 1) void backup_kernel(const __grid_constant__ Params p) {
  const int64_t total = p.num_items * p.num_layers * p.chunks_per_item;
  for (int64_t unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int64_t rest = unit / p.chunks_per_item;
    const int64_t offset = (unit - rest * p.chunks_per_item) * p.chunk_bytes;
    const int64_t layer_id = rest % p.num_layers;
    const int64_t item_id = rest / p.num_layers;
    const int64_t len = min(p.chunk_bytes, p.item_size - offset);
    const char* src =
        reinterpret_cast<const char*>(p.layer_ptrs[layer_id]) + p.src_indices[item_id] * p.item_size + offset;
    char* dst = p.dst_base + p.dst_indices[item_id] * p.dst_layout_dim + layer_id * p.item_size + offset;
    copy_span(src, dst, len);
  }
}

}  // namespace mamba_split

struct TransferMambaSplitKernel {
  static int64_t check_split(const int64_t item_size, const int64_t block_quota, const int64_t chunk_bytes) {
    using namespace host;
    RuntimeCheck(item_size > 0, "transfer_mamba_split: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba_split: item_size must be 16-byte aligned (uint4)");
    RuntimeCheck(block_quota > 0, "transfer_mamba_split: block_quota must be positive");
    RuntimeCheck(
        chunk_bytes > 0 && chunk_bytes % 16 == 0,
        "transfer_mamba_split: chunk_bytes must be a positive multiple of 16");
    return (item_size + chunk_bytes - 1) / chunk_bytes;
  }

  /// \brief Load one layer of Mamba state items, page_first host -> layer_first device, over up to
  ///        `block_quota` CTAs. Same arguments as TransferMambaKernel::run_pf_lf, plus:
  /// \param block_quota Maximum number of CTAs (> 0).
  /// \param chunk_bytes Bytes per work unit, a positive multiple of 16.
  static void run_pf_lf(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t layer_id,
      const int64_t item_size,
      const int64_t src_layout_dim,
      const int64_t block_quota,
      const int64_t chunk_bytes) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    const int64_t chunks_per_item = check_split(item_size, block_quota, chunk_bytes);
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const dim3 grid(static_cast<unsigned>(std::min(block_quota, num_items * chunks_per_item)));

    const auto params = mamba_split::Params{
        .src_base = static_cast<const char*>(runtime::get_device_accessible_ptr(src)),
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = nullptr,
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = src_layout_dim,
        .dst_layout_dim = 0,
        .layer_id = layer_id,
        .num_items = num_items,
        .num_layers = 1,
        .chunk_bytes = chunk_bytes,
        .chunks_per_item = chunks_per_item,
    };

    LaunchKernel(grid, mamba_split::kThreads, device)(mamba_split::load_kernel, params);
  }

  /// \brief Back up Mamba state items of all layers, layer_first device -> page_first host, over up to
  ///        `block_quota` CTAs. Same arguments as TransferMambaKernel::run_lf_pf, plus:
  /// \param block_quota Maximum number of CTAs (> 0).
  /// \param chunk_bytes Bytes per work unit, a positive multiple of 16.
  static void run_lf_pf(
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t item_size,
      const int64_t dst_layout_dim,
      const int64_t num_layers,
      const int64_t block_quota,
      const int64_t chunk_bytes) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);
    TensorMatcher({static_cast<int64_t>(num_layers)})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_ptrs);

    const int64_t chunks_per_item = check_split(item_size, block_quota, chunk_bytes);
    RuntimeCheck(num_layers > 0, "transfer_mamba_split: num_layers must be positive");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const dim3 grid(static_cast<unsigned>(std::min(block_quota, num_items * num_layers * chunks_per_item)));

    const auto params = mamba_split::Params{
        .src_base = nullptr,
        .dst_base = static_cast<char*>(runtime::get_device_accessible_ptr(dst)),
        .layer_ptrs = static_cast<const uintptr_t*>(src_ptrs.data_ptr()),
        .src_indices = static_cast<const int64_t*>(src_indices.data_ptr()),
        .dst_indices = static_cast<const int64_t*>(dst_indices.data_ptr()),
        .item_size = item_size,
        .src_layout_dim = 0,
        .dst_layout_dim = dst_layout_dim,
        .layer_id = 0,
        .num_items = num_items,
        .num_layers = num_layers,
        .chunk_bytes = chunk_bytes,
        .chunks_per_item = chunks_per_item,
    };

    LaunchKernel(grid, mamba_split::kThreads, device)(mamba_split::backup_kernel, params);
  }
};

}  // namespace sglang
