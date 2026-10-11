#pragma once

#include <sgl_kernel/copy_batch.h>
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/extra/stl.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda.h>
#include <limits>
#include <tuple>
#include <vector>

namespace sglang {
using namespace host;
using tvm::ffi::TensorView;

inline std::vector<tvm::ffi::Tensor> dma_page_indices(const std::vector<std::tuple<TensorView, int64_t>>& inputs) {
  std::vector<tvm::ffi::Tensor> outputs;
  outputs.reserve(inputs.size());
  for (const auto& [indices, page_size] : inputs) {
    if (page_size <= 0 || indices.device().device_type != kDLCPU || indices.dtype() != DLDataType{kDLInt, 64, 1} ||
        indices.ndim() != 1 || !indices.is_contiguous() || indices.numel() % page_size != 0) {
      TVM_FFI_THROW(ValueError) << "Direct DMA requires complete pages of CPU int64 indices";
    }
    int64_t count = indices.numel() / page_size;
    auto pages = host::ffi::empty(tvm::ffi::ShapeView(&count, 1), indices.dtype(), indices.device());
    const auto* tokens = static_cast<const int64_t*>(indices.data_ptr());
    auto* ids = static_cast<int64_t*>(pages.data_ptr());
    // Pool allocators supply aligned, contiguous pages; read only their first token.
    for (int64_t page = 0; page < count; ++page) {
      ids[page] = tokens[page * page_size] / page_size;
    }
    outputs.push_back(std::move(pages));
  }
  return outputs;
}

// Query once when preparing a layout; a tensor may span several independent
// host registrations, or occupy only part of a torch-pinned allocation.
inline std::tuple<std::vector<int64_t>, std::vector<int64_t>> host_registration_bounds(TensorView buffer) {
  RuntimeCheck(
      buffer.is_contiguous() && buffer.dtype().bits % 8 == 0 && buffer.dtype().lanes == 1 &&
          (buffer.device().device_type == kDLCPU || buffer.device().device_type == kDLCUDAHost),
      "Direct DMA requires a contiguous host allocation");
  auto ptr = reinterpret_cast<uintptr_t>(buffer.data_ptr());
  const auto end = ptr + buffer.numel() * (buffer.dtype().bits / 8);
  std::vector<int64_t> starts, ends;
  CUpointer_attribute attributes[] = {
      CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, CU_POINTER_ATTRIBUTE_RANGE_SIZE};
  while (ptr < end) {
    CUmemorytype memory_type{};
    CUdeviceptr start = 0;
    size_t size = 0;
    void* values[] = {&memory_type, &start, &size};
    const auto status = cuPointerGetAttributes(3, attributes, values, ptr);
    RuntimeCheck(
        status == CUDA_SUCCESS && memory_type == CU_MEMORYTYPE_HOST && start <= ptr && size > ptr - start &&
            start + size > ptr,
        "Direct DMA requires pinned host memory with queryable registration bounds");
    starts.push_back(ptr);
    ptr = std::min(end, static_cast<uintptr_t>(start + size));
    ends.push_back(ptr);
  }
  return {starts, ends};
}

// Host-only planning and driver submission: this module launches no SM kernels.
namespace block_dma {
struct HostAllocation {
  uintptr_t start, end;
  const std::vector<int64_t>& registration_starts;
  const std::vector<int64_t>& registration_ends;

  HostAllocation(TensorView buffer, const std::vector<int64_t>& starts, const std::vector<int64_t>& ends)
      : start(reinterpret_cast<uintptr_t>(buffer.data_ptr())),
        end(start + buffer.numel() * (buffer.dtype().bits / 8)),
        registration_starts(starts),
        registration_ends(ends) {
    RuntimeCheck(
        buffer.is_contiguous() && buffer.dtype().bits % 8 == 0 && buffer.dtype().lanes == 1 &&
            (buffer.device().device_type == kDLCPU || buffer.device().device_type == kDLCUDAHost),
        "Block DMA requires a contiguous host allocation");
    RuntimeCheck(starts.size() == ends.size() && end >= start, "Invalid host registrations");
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, buffer.data_ptr());
    RuntimeCheck(status == cudaSuccess, "Block DMA host pointer query failed: ", cudaGetErrorString(status));
    RuntimeCheck(attributes.type == cudaMemoryTypeHost, "Block DMA requires pinned host memory");
    for (size_t i = 0; i < starts.size(); ++i) {
      RuntimeCheck(
          starts[i] >= start && ends[i] <= end && starts[i] < ends[i] && (i == 0 || starts[i] >= ends[i - 1]),
          "Block DMA invalid registration bounds");
    }
  }

  size_t registered_bytes(uintptr_t ptr) const {
    RuntimeCheck(ptr >= start && ptr < end, "Block DMA host view escapes its allocation");
    if (registration_starts.empty()) return end - ptr;
    auto it = std::upper_bound(registration_starts.begin(), registration_starts.end(), ptr);
    RuntimeCheck(it != registration_starts.begin(), "Block DMA pointer is not registered");
    const auto registration_end = registration_ends[it - registration_starts.begin() - 1];
    RuntimeCheck(registration_end > ptr, "Block DMA pointer is not registered");
    return registration_end - ptr;
  }
};

struct CopyPlan {
  std::vector<direct::Copy> copies;

  explicit CopyPlan(size_t count) {
    copies.reserve(count);
  }

  void add(CUdeviceptr src, CUdeviceptr dst, size_t remaining, bool host_is_src, const HostAllocation& host) {
    const auto host_ptr = host_is_src ? src : dst;
    RuntimeCheck(
        host_ptr >= host.start && host_ptr <= host.end && remaining <= host.end - host_ptr,
        "Block DMA host view escapes its allocation");
    while (remaining) {
      // Copies cannot cross cudaHostRegister boundaries, even for adjacent ranges.
      const size_t size = std::min(remaining, host.registered_bytes(host_is_src ? src : dst));
      copies.push_back({src, dst, size});
      src += size;
      dst += size;
      remaining -= size;
    }
  }

  void check_destinations() const {
    std::vector<std::pair<CUdeviceptr, CUdeviceptr>> spans;
    spans.reserve(copies.size());
    for (const auto& copy : copies)
      spans.emplace_back(copy.dst, copy.dst + copy.size);
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i) {
      RuntimeCheck(spans[i - 1].second <= spans[i].first, "Block DMA pool destinations overlap");
    }
  }

  std::vector<int64_t> submit(int64_t stream_handle) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const auto result = direct::submit_copies(copies, stream);
    RuntimeCheck(result == cudaSuccess, "HiCache direct copy failed: ", cudaGetErrorString(result));
    const auto largest = copies.empty() ? 0 : copies.front().size;
    return {static_cast<int64_t>(copies.size()), static_cast<int64_t>(largest)};
  }
};

// KV and Mamba share the layout [slot, strided subrow, contiguous payload].
inline void append_rows(
    CopyPlan& plan,
    const std::vector<TensorView>& host_rows,
    const std::vector<TensorView>& device_rows,
    TensorView host_buffer,
    const std::vector<int64_t>& registration_starts,
    const std::vector<int64_t>& registration_ends,
    TensorView host_indices,
    TensorView device_indices,
    bool host_to_device,
    CUdevice current_device) {
  TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCPU>().verify(host_indices);
  TensorMatcher({host_indices.numel()}).with_dtype<int64_t>().with_device<kDLCPU>().verify(device_indices);
  RuntimeCheck(host_rows.size() == device_rows.size(), "Direct DMA row counts differ");
  RuntimeCheck(
      host_indices.is_contiguous() && device_indices.is_contiguous(), "Direct DMA requires contiguous indices");
  const auto n = host_indices.numel();
  if (n == 0 || host_rows.empty()) return;
  const auto* host_ids = static_cast<const int64_t*>(host_indices.data_ptr());
  const auto* device_ids = static_cast<const int64_t*>(device_indices.data_ptr());
  int64_t host_limit = std::numeric_limits<int64_t>::max(), device_limit = host_limit;
  for (size_t row = 0; row < host_rows.size(); ++row) {
    RuntimeCheck(host_rows[row].ndim() > 0 && device_rows[row].ndim() > 0, "Direct DMA requires row tensors");
    host_limit = std::min(host_limit, host_rows[row].size(0));
    device_limit = std::min(device_limit, device_rows[row].size(0));
  }
  std::vector<std::array<int64_t, 3>> runs;
  int64_t first = 0;
  for (int64_t i = 0; i < n; ++i) {
    RuntimeCheck(
        host_ids[i] >= 0 && device_ids[i] >= 0 && host_ids[i] < host_limit && device_ids[i] < device_limit,
        "Direct DMA index out of bounds");
    if (i + 1 == n || host_ids[i + 1] != host_ids[i] + 1 || device_ids[i + 1] != device_ids[i] + 1) {
      runs.push_back({host_ids[first], device_ids[first], i + 1 - first});
      first = i + 1;
    }
  }
  HostAllocation host(host_buffer, registration_starts, registration_ends);
  for (size_t row = 0; row < host_rows.size(); ++row) {
    const auto cpu = host_rows[row], gpu = device_rows[row];
    RuntimeCheck(
        gpu.device().device_type == kDLCUDA && gpu.device().device_id == current_device &&
            (cpu.device().device_type == kDLCPU || cpu.device().device_type == kDLCUDAHost),
        "Direct DMA requires one CPU and one CUDA view");
    RuntimeCheck(
        cpu.ndim() == gpu.ndim() && cpu.dtype() == gpu.dtype() && cpu.dtype().bits % 8 == 0 && cpu.dtype().lanes == 1,
        "Direct DMA row dimensions and dtypes must match");
    int64_t row_elements = 1;
    for (int dim = cpu.ndim() - 1; dim > 1; --dim) {
      RuntimeCheck(cpu.size(dim) == gpu.size(dim), "Direct DMA row shapes differ");
      RuntimeCheck(
          cpu.size(dim) == 1 || (cpu.stride(dim) == row_elements && gpu.stride(dim) == row_elements),
          "Direct DMA requires contiguous row payloads");
      row_elements *= cpu.size(dim);
    }
    int64_t subrows = 1;
    if (cpu.ndim() > 1) {
      RuntimeCheck(cpu.size(1) == gpu.size(1), "Direct DMA row shapes differ");
      subrows = cpu.size(1);
      if (subrows == 1 || (cpu.stride(1) == row_elements && gpu.stride(1) == row_elements)) {
        row_elements *= subrows;
        subrows = 1;
      }
    }
    RuntimeCheck(
        row_elements > 0 && subrows > 0 && cpu.stride(0) >= row_elements && gpu.stride(0) >= row_elements &&
            (subrows == 1 || (cpu.stride(1) >= row_elements && gpu.stride(1) >= row_elements)),
        "Direct DMA requires non-overlapping rows");
    const size_t itemsize = cpu.dtype().bits / 8;
    const size_t row_bytes = row_elements * itemsize;
    const size_t host_stride = cpu.stride(0) * itemsize, device_stride = gpu.stride(0) * itemsize;
    const size_t host_substride = subrows == 1 ? 0 : cpu.stride(1) * itemsize;
    const size_t device_substride = subrows == 1 ? 0 : gpu.stride(1) * itemsize;
    const auto host_base = reinterpret_cast<uintptr_t>(cpu.data_ptr());
    const auto device_base = reinterpret_cast<uintptr_t>(gpu.data_ptr());
    const auto host_extent = (subrows - 1) * host_substride + row_bytes;
    RuntimeCheck(
        host_base >= host.start && host_base <= host.end && host_extent <= host.end - host_base &&
            cpu.size(0) - 1 <= (host.end - host_base - host_extent) / host_stride,
        "Direct DMA host view escapes its allocation");
    const bool contiguous_slots = host_stride == row_bytes && device_stride == row_bytes;
    plan.copies.reserve(plan.copies.size() + subrows * (contiguous_slots ? runs.size() : n));
    for (int64_t subrow = 0; subrow < subrows; ++subrow) {
      for (const auto& run : runs) {
        for (int64_t offset = 0; offset < run[2]; offset += contiguous_slots ? run[2] : 1) {
          const auto host_ptr = host_base + subrow * host_substride + (run[0] + offset) * host_stride;
          const auto device_ptr = device_base + subrow * device_substride + (run[1] + offset) * device_stride;
          plan.add(
              host_to_device ? host_ptr : device_ptr,
              host_to_device ? device_ptr : host_ptr,
              row_bytes * (contiguous_slots ? run[2] : 1),
              host_to_device,
              host);
        }
      }
    }
  }
}

using RowLayout = std::
    tuple<std::vector<TensorView>, std::vector<TensorView>, TensorView, std::vector<int64_t>, std::vector<int64_t>>;
using RowTransfer = std::tuple<RowLayout, TensorView, TensorView>;
}  // namespace block_dma

inline std::vector<int64_t> transfer_row_blocks(
    const std::vector<std::vector<block_dma::RowTransfer>>& blocks,
    bool host_to_device,
    int64_t stream_handle,
    std::optional<tvm::ffi::Function> on_block_done) {
  CUdevice current_device;
  RuntimeCheck(cuCtxGetDevice(&current_device) == CUDA_SUCCESS, "Direct DMA requires a CUDA context");
  std::vector<block_dma::CopyPlan> plans;
  plans.reserve(blocks.size());
  // Validate the entire transfer before writing any pool or publishing a block.
  // Actual copy spans allow disjoint views into the same Mamba envelope.
  for (const auto& block : blocks) {
    auto& plan = plans.emplace_back(0);
    for (const auto& transfer : block) {
      const auto& [layout, host_indices, device_indices] = transfer;
      std::apply(
          [&](const auto&... args) {
            block_dma::append_rows(plan, args..., host_indices, device_indices, host_to_device, current_device);
          },
          layout);
    }
    plan.check_destinations();
  }
  int64_t count = 0, largest = 0;
  for (size_t block = 0; block < plans.size(); ++block) {
    const auto result = plans[block].submit(stream_handle);
    count += result[0];
    largest = std::max(largest, result[1]);
    if (on_block_done) (*on_block_done)(static_cast<int64_t>(block));
  }
  return {count, largest};
}
}  // namespace sglang
