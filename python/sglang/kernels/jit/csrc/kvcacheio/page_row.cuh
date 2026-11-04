#pragma once

#include "hicache.cuh"

namespace sglang {

// Opaque page-row copies between pinned host memory and device pools: one row of
// row_bytes per (layer, page). A 64-lane worker moves one segment of a row with up
// to kVecsPerLane independent 16 B loads per lane in flight, so two CTAs saturate
// PCIe and leave every other SM to compute.
constexpr uint32_t kPageRowLanes = 64;
constexpr int64_t kPageRowLineVecs = 128 / 16;
constexpr uint32_t kPageRowBlockQuota = 2;

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
};

template <typename T, bool kLoad, uint32_t kVecsPerLane, uint32_t kBlockSize>
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
    const auto host = reinterpret_cast<uint4*>(params.host_ptrs[layer] + int64_t(hp[selected]) * params.host_stride);
    const auto device =
        reinterpret_cast<uint4*>(params.device_ptrs[layer] + int64_t(dp[selected]) * params.device_stride);
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
      int64_t row_bytes) {
    using namespace host;
    auto layers = SymbolicSize{"layers"};
    auto pages = SymbolicSize{"pages"};
    auto device = SymbolicDevice{};
    auto index_dtype = SymbolicDType{};
    TensorMatcher({layers}).with_dtype<uint64_t>().with_device<kDLGPU>(device).verify(host_ptrs).verify(device_ptrs);
    TensorMatcher({pages})
        .with_dtype<int32_t, int64_t>(index_dtype)
        .with_device<kDLGPU>(device)
        .verify(host_pages)
        .verify(device_pages);
    RuntimeCheck(row_bytes > 0 && row_bytes % 128 == 0, "page-row bytes must be a positive multiple of 128");
    RuntimeCheck(host_stride % 128 == 0 && device_stride % 128 == 0, "page-row strides must be 128B aligned");
    if (pages.unwrap() == 0 || layers.unwrap() == 0) return;

    constexpr int64_t kSegmentLines = kPageRowLanes * kVecsPerLane / kPageRowLineVecs;
    constexpr int64_t kWorkersPerBlock = kBlockSize / kPageRowLanes;
    const auto row_lines = row_bytes / 128;
    const auto segments = div_ceil(row_lines, kSegmentLines);
    const auto work = layers.unwrap() * pages.unwrap() * segments;
    const auto blocks = std::min<int64_t>(div_ceil(work, kWorkersPerBlock), kPageRowBlockQuota);
    const auto params = PageRowParams{
        .host_ptrs = static_cast<const uint64_t*>(host_ptrs.data_ptr()),
        .device_ptrs = static_cast<const uint64_t*>(device_ptrs.data_ptr()),
        .host_pages = host_pages.data_ptr(),
        .device_pages = device_pages.data_ptr(),
        .pages = pages.unwrap(),
        .layers = layers.unwrap(),
        .host_stride = host_stride,
        .device_stride = device_stride,
        .row_lines = row_lines,
        .segments = segments,
    };
    const auto kernel = index_dtype.unwrap().bits == 32 ? page_row_copy<int32_t, kLoad, kVecsPerLane, kBlockSize>
                                                        : page_row_copy<int64_t, kLoad, kVecsPerLane, kBlockSize>;
    LaunchKernel(blocks, kBlockSize, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
