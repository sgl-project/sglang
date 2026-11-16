#pragma once

#include <sgl_kernel/runtime.cuh>

#include "hicache.cuh"
#include <algorithm>
#include <cstdint>

namespace sglang {

constexpr int kBlockSize = 1024;
constexpr int kBlockQuotaBackup = 2;
constexpr int kBlockQuotaLoad = 2;

constexpr int kBytesPerThreadPerStep = 16;
constexpr int kBytesPerBlockPerStep = kBlockSize * kBytesPerThreadPerStep;

struct MambaTransferParams {
  const char* __restrict__ src_base;
  char* __restrict__ dst_base;

  const uintptr_t* __restrict__ layer_ptrs;
  const int64_t* __restrict__ src_indices;
  const int64_t* __restrict__ dst_indices;
  int64_t item_size;
  int64_t src_layout_dim;
  int64_t dst_layout_dim;
  int64_t layer_id;
  int64_t num_items;
  int64_t num_layers;
};

__global__
__launch_bounds__(kBlockSize, 1) void transfer_mamba_load_kernel(const __grid_constant__ MambaTransferParams params) {
  const int tid = threadIdx.x;

  for (int64_t item_id = static_cast<int64_t>(blockIdx.x); item_id < params.num_items; item_id += gridDim.x) {
    const int64_t src_page = params.src_indices[item_id];
    const int64_t dst_page = params.dst_indices[item_id];

    const char* src = params.src_base + src_page * params.src_layout_dim + params.layer_id * params.item_size;

    char* dst = params.dst_base + dst_page * params.item_size;

    const int64_t base = static_cast<int64_t>(tid) * kBytesPerThreadPerStep;
    if (base < params.item_size) {
      uint4 v_cur = device::details::load_nc(reinterpret_cast<const uint4*>(src + base));
      int64_t off;
      for (off = base + kBytesPerBlockPerStep; off < params.item_size; off += kBytesPerBlockPerStep) {
        uint4 v_next = device::details::load_nc(reinterpret_cast<const uint4*>(src + off));
        device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
        v_cur = v_next;
      }
      device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
    }
  }
}

__global__
__launch_bounds__(kBlockSize, 1) void transfer_mamba_backup_kernel(const __grid_constant__ MambaTransferParams params) {
  const int tid = threadIdx.x;
  const int64_t total_work = params.num_items * params.num_layers;

  for (int64_t work_id = static_cast<int64_t>(blockIdx.x); work_id < total_work; work_id += gridDim.x) {
    const int64_t layer_id = work_id % params.num_layers;
    const int64_t item_id = work_id / params.num_layers;

    const int64_t src_page = params.src_indices[item_id];
    const int64_t dst_page = params.dst_indices[item_id];

    const char* src = reinterpret_cast<const char*>(params.layer_ptrs[layer_id]) + src_page * params.item_size;

    char* dst = params.dst_base + dst_page * params.dst_layout_dim + layer_id * params.item_size;

    const int64_t base = static_cast<int64_t>(tid) * kBytesPerThreadPerStep;
    if (base < params.item_size) {
      uint4 v_cur = device::details::load_nc(reinterpret_cast<const uint4*>(src + base));
      int64_t off;
      for (off = base + kBytesPerBlockPerStep; off < params.item_size; off += kBytesPerBlockPerStep) {
        uint4 v_next = device::details::load_nc(reinterpret_cast<const uint4*>(src + off));
        device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
        v_cur = v_next;
      }
      device::details::store_nc(reinterpret_cast<uint4*>(dst + off - kBytesPerBlockPerStep), v_cur);
    }
  }
}

// One row of the piece table: device source, offset in the device slot,
// offset in the host slot, byte length.
constexpr int kPieceFields = 4;
constexpr int kWarpSize = 32;

struct MambaPieceParams {
  const uint64_t* __restrict__ device_ptrs;
  const int64_t* __restrict__ device_strides;
  const int64_t* __restrict__ pieces;
  char* __restrict__ host_base;
  const int64_t* __restrict__ host_indices;
  const int64_t* __restrict__ device_indices;
  int64_t host_stride;
  int64_t num_items;
  int64_t num_pieces;
};

// A warp moves one (slot, piece) pair; pieces are too small for a block each.
template <bool kToHost>
__global__
__launch_bounds__(kBlockSize, 1) void transfer_mamba_pieces_kernel(const __grid_constant__ MambaPieceParams params) {
  const int lane = threadIdx.x % kWarpSize;
  const int64_t warps_per_block = blockDim.x / kWarpSize;
  const int64_t total_work = params.num_items * params.num_pieces;

  for (int64_t work_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + threadIdx.x / kWarpSize;
       work_id < total_work;
       work_id += gridDim.x * warps_per_block) {
    const int64_t item_id = work_id / params.num_pieces;
    const int64_t* piece = params.pieces + (work_id % params.num_pieces) * kPieceFields;
    const int64_t source = piece[0];

    char* device_ptr = reinterpret_cast<char*>(params.device_ptrs[source]) +
                       params.device_indices[item_id] * params.device_strides[source] + piece[1];
    char* host_ptr = params.host_base + params.host_indices[item_id] * params.host_stride + piece[2];
    const char* src = kToHost ? device_ptr : host_ptr;
    char* dst = kToHost ? host_ptr : device_ptr;

    for (int64_t off = static_cast<int64_t>(lane) * kBytesPerThreadPerStep; off < piece[3];
         off += kWarpSize * kBytesPerThreadPerStep) {
      device::details::store_nc(
          reinterpret_cast<uint4*>(dst + off), device::details::load_nc(reinterpret_cast<const uint4*>(src + off)));
    }
  }
}

struct TransferMambaKernel {
  // Load: page_first -> layer_first (single layer at a time)
  static void run_pf_lf(
      const tvm::ffi::TensorView src,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t layer_id,
      const int64_t item_size,
      const int64_t src_layout_dim) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const int grid_x = static_cast<int>(std::min(static_cast<int64_t>(kBlockQuotaLoad), num_items));
    dim3 grid(grid_x);

    const auto params = MambaTransferParams{
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
    };

    LaunchKernel(grid, kBlockSize, device)(transfer_mamba_load_kernel, params);
  }

  // Backup: layer_first -> page_first (all layers at once)
  static void run_lf_pf(
      const tvm::ffi::TensorView src_ptrs,
      const tvm::ffi::TensorView dst,
      const tvm::ffi::TensorView src_indices,
      const tvm::ffi::TensorView dst_indices,
      const int64_t item_size,
      const int64_t dst_layout_dim,
      const int64_t num_layers) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_indices)
        .verify(dst_indices);
    // src_ptrs is a 1D tensor of device pointers (uint64) on CUDA
    TensorMatcher({static_cast<int64_t>(num_layers)})  //
        .with_dtype<uint64_t>()
        .with_device<kDLGPU>(device_)
        .verify(src_ptrs);

    RuntimeCheck(item_size > 0, "transfer_mamba: item_size must be positive");
    RuntimeCheck(item_size % 16 == 0, "transfer_mamba: item_size must be 16-byte aligned (uint4)");
    RuntimeCheck(num_layers > 0, "transfer_mamba: num_layers must be positive");
    const auto num_items = L.unwrap();
    if (num_items == 0) return;

    const auto device = device_.unwrap();
    const int64_t total_work = num_items * num_layers;
    const int grid_x = static_cast<int>(std::min(static_cast<int64_t>(kBlockQuotaBackup), total_work));
    dim3 grid(grid_x);

    const auto params = MambaTransferParams{
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
    };

    LaunchKernel(grid, kBlockSize, device)(transfer_mamba_backup_kernel, params);
  }

  // page_unified: copy every piece of each (host slot, device slot) pair.
  static void run_pieces(
      const tvm::ffi::TensorView device_ptrs,
      const tvm::ffi::TensorView device_strides,
      const tvm::ffi::TensorView pieces,
      const tvm::ffi::TensorView host_buffer,
      const tvm::ffi::TensorView host_indices,
      const tvm::ffi::TensorView device_indices,
      const int64_t host_stride,
      const int64_t to_host) {
    using namespace host;

    auto L = SymbolicSize{"num_indices"};
    auto S = SymbolicSize{"num_sources"};
    auto P = SymbolicSize{"num_pieces"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({L})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(host_indices)
        .verify(device_indices);
    TensorMatcher({S}).with_dtype<uint64_t>().with_device<kDLGPU>(device_).verify(device_ptrs);
    TensorMatcher({S}).with_dtype<int64_t>().with_device<kDLGPU>(device_).verify(device_strides);
    TensorMatcher({P, kPieceFields})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .with_strides({kPieceFields, 1})
        .verify(pieces);

    RuntimeCheck(host_stride > 0 && host_stride % 16 == 0, "transfer_mamba: host slots must be 16-byte aligned");
    const auto num_items = L.unwrap();
    const auto num_pieces = P.unwrap();
    if (num_items == 0 || num_pieces == 0) return;

    const auto device = device_.unwrap();
    const int64_t warps = num_items * num_pieces;
    const int64_t warps_per_block = kBlockSize / kWarpSize;
    const int grid_x = static_cast<int>(
        std::min(static_cast<int64_t>(kBlockQuotaBackup), (warps + warps_per_block - 1) / warps_per_block));

    const auto params = MambaPieceParams{
        .device_ptrs = static_cast<const uint64_t*>(device_ptrs.data_ptr()),
        .device_strides = static_cast<const int64_t*>(device_strides.data_ptr()),
        .pieces = static_cast<const int64_t*>(pieces.data_ptr()),
        .host_base = static_cast<char*>(runtime::get_device_accessible_ptr(host_buffer)),
        .host_indices = static_cast<const int64_t*>(host_indices.data_ptr()),
        .device_indices = static_cast<const int64_t*>(device_indices.data_ptr()),
        .host_stride = host_stride,
        .num_items = num_items,
        .num_pieces = num_pieces,
    };

    const auto kernel = to_host ? transfer_mamba_pieces_kernel<true> : transfer_mamba_pieces_kernel<false>;
    LaunchKernel(dim3(grid_x), kBlockSize, device)(kernel, params);
  }
};

}  // namespace sglang
