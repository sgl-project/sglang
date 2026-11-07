#pragma once

#include "hicache.cuh"

namespace sglang {

// Opaque page-row copies between pinned host memory and device pools: one row of
// row_bytes per (layer, page). A 64-lane worker moves one segment of a row with up
// to kVecsPerLane independent 16 B loads per lane in flight, so a few CTAs (the
// caller's block quota) saturate the host link and leave every other SM to compute.
constexpr uint32_t kPageRowLanes = 64;
constexpr int64_t kPageRowLineVecs = 128 / 16;

#if defined(USE_ROCM) && !defined(cudaDevAttrPageableMemoryAccessUsesHostPageTables)
#define cudaDevAttrPageableMemoryAccessUsesHostPageTables hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#endif
#if defined(USE_ROCM) && !defined(cudaDevAttrCanUseHostPointerForRegisteredMem)
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#endif

/// \brief Whether the GPU reaches pageable host memory through the host page tables,
///        as on a coherent NVLink-C2C link.
inline bool page_row_host_link_coherent(const int64_t device_id) {
  int coherent = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(
      &coherent, cudaDevAttrPageableMemoryAccessUsesHostPageTables, static_cast<int>(device_id)));
  return coherent != 0;
}

/// \brief Whether kernels may use the host address of cudaHostRegister-ed memory,
///        rather than a separate device alias from cudaHostGetDevicePointer.
inline bool page_row_host_pointer_usable(const int64_t device_id) {
  int usable = 0;
  host::RuntimeDeviceCheck(
      cudaDeviceGetAttribute(&usable, cudaDevAttrCanUseHostPointerForRegisteredMem, static_cast<int>(device_id)));
  return usable != 0;
}

struct PageRowParams {
  const uint64_t* __restrict__ host_ptrs;
  const uint64_t* __restrict__ device_ptrs;
  const void* __restrict__ host_pages;
  const void* __restrict__ device_pages;
  int64_t pages;
  int64_t layers;
  int64_t host_stride;
  int64_t device_stride;
  int64_t row_lines;
  int64_t segments;
  // Page `p` is `pages[p * index_stride] / index_stride`: 1 for page ids, page_size
  // for page-aligned token slots.
  int64_t index_stride;
};

// Stride 1 gets its own instantiation: a per-page runtime branch costs 1-2% on issue-bound 2-CTA copies.
template <bool kStridedIndices, typename T>
SGL_DEVICE int64_t page_row_page(const T* __restrict__ pages, int64_t selected, int64_t index_stride) {
  if constexpr (kStridedIndices) {
    return int64_t(pages[selected * index_stride]) / index_stride;
  } else {
    return int64_t(pages[selected]);
  }
}

template <typename T, bool kLoad, uint32_t kVecsPerLane, uint32_t kBlockSize, bool kStridedIndices>
__global__ __launch_bounds__(kBlockSize, 1) void page_row_copy(const __grid_constant__ PageRowParams params) {
  constexpr uint32_t kWorkersPerBlock = kBlockSize / kPageRowLanes;
  const auto hp = static_cast<const T*>(params.host_pages);
  const auto dp = static_cast<const T*>(params.device_pages);
  const auto lane = threadIdx.x % kPageRowLanes;
  const auto total = params.layers * params.pages * params.segments;
  for (int64_t work = blockIdx.x * kWorkersPerBlock + threadIdx.x / kPageRowLanes; work < total;
       work += gridDim.x * kWorkersPerBlock) {
    const auto segment = work % params.segments;
    const auto selected = (work / params.segments) % params.pages;
    const auto layer = work / (params.segments * params.pages);
    const auto host_page = page_row_page<kStridedIndices>(hp, selected, params.index_stride);
    const auto device_page = page_row_page<kStridedIndices>(dp, selected, params.index_stride);
    const auto host = reinterpret_cast<uint4*>(params.host_ptrs[layer] + host_page * params.host_stride);
    const auto device = reinterpret_cast<uint4*>(params.device_ptrs[layer] + device_page * params.device_stride);
    const auto src = kLoad ? host : device;
    const auto dst = kLoad ? device : host;
    // Balanced segments on 128 B line boundaries: no worker idles on a short tail
    // and no PCIe read straddles two lines.
    const auto begin = segment * params.row_lines / params.segments * kPageRowLineVecs;
    const auto end = (segment + 1) * params.row_lines / params.segments * kPageRowLineVecs;

    uint4 data[kVecsPerLane];
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerLane; ++i) {
      const auto v = begin + lane + i * kPageRowLanes;
      if (v < end) data[i] = device::details::load_nc(src + v);
    }
#pragma unroll
    for (uint32_t i = 0; i < kVecsPerLane; ++i) {
      const auto v = begin + lane + i * kPageRowLanes;
      if (v < end) device::details::store_nc(dst + v, data[i]);
    }
  }
}

template <bool kLoad, uint32_t kVecsPerLane, uint32_t kBlockSize>
struct PageRowKernel {
  static_assert(kBlockSize % kPageRowLanes == 0, "block must hold whole workers");

  static void
  run(tvm::ffi::TensorView host_ptrs,
      tvm::ffi::TensorView device_ptrs,
      tvm::ffi::TensorView host_pages,
      tvm::ffi::TensorView device_pages,
      int64_t host_stride,
      int64_t device_stride,
      int64_t row_bytes,
      int64_t index_stride,
      int64_t block_quota) {
    using namespace host;
    auto layers = SymbolicSize{"layers"};
    auto num_indices = SymbolicSize{"num_indices"};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    TensorMatcher({layers}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(host_ptrs).verify(device_ptrs);
    TensorMatcher({num_indices})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device<kDLGPU>(device)
        .verify(host_pages)
        .verify(device_pages);
    RuntimeCheck(row_bytes > 0 && row_bytes % 128 == 0, "page-row bytes must be a positive multiple of 128");
    RuntimeCheck(host_stride % 128 == 0 && device_stride % 128 == 0, "page-row strides must be 128B aligned");
    RuntimeCheck(index_stride > 0 && num_indices.unwrap() % index_stride == 0, "indices must be whole pages");
    RuntimeCheck(block_quota >= 0, "block quota must not be negative");
    const auto pages = num_indices.unwrap() / index_stride;
    if (pages == 0 || layers.unwrap() == 0) return;

    constexpr int64_t kSegmentLines = kPageRowLanes * kVecsPerLane / kPageRowLineVecs;
    constexpr int64_t kWorkersPerBlock = kBlockSize / kPageRowLanes;
    const auto row_lines = row_bytes / 128;
    const auto segments = div_ceil(row_lines, kSegmentLines);
    const auto work = layers.unwrap() * pages * segments;
    const auto needed = div_ceil(work, kWorkersPerBlock);
    const auto blocks = block_quota > 0 ? std::min<int64_t>(needed, block_quota) : needed;
    const auto params = PageRowParams{
        .host_ptrs = static_cast<const uint64_t*>(host_ptrs.data_ptr()),
        .device_ptrs = static_cast<const uint64_t*>(device_ptrs.data_ptr()),
        .host_pages = host_pages.data_ptr(),
        .device_pages = device_pages.data_ptr(),
        .pages = pages,
        .layers = layers.unwrap(),
        .host_stride = host_stride,
        .device_stride = device_stride,
        .row_lines = row_lines,
        .segments = segments,
        .index_stride = index_stride,
    };
    const bool int32_indices = index_dtype.unwrap().bits == 32;
    const auto kernel = index_stride == 1
                            ? (int32_indices ? page_row_copy<int32_t, kLoad, kVecsPerLane, kBlockSize, false>
                                             : page_row_copy<int64_t, kLoad, kVecsPerLane, kBlockSize, false>)
                            : (int32_indices ? page_row_copy<int32_t, kLoad, kVecsPerLane, kBlockSize, true>
                                             : page_row_copy<int64_t, kLoad, kVecsPerLane, kBlockSize, true>);
    LaunchKernel(blocks, kBlockSize, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
