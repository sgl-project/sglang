#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include "memcpy_batch.cuh"
#include <cstdint>
#include <vector>

namespace sglang {

/// \brief Copies whole pages between per-layer device rows and per-layer host rows.
///
/// Pages whose host and device rows both advance by one page form a run, copied with one
/// descriptor per (run, layer). All descriptors of a call go out in one cudaMemcpyBatchAsync
/// on the given stream, or as one cudaMemcpyAsync each where the batch API is unavailable.
struct PageRunCopy {
  /// \param host_indices    CPU int64 [n]: host row of each index; whole pages of
  ///                        rows_per_page consecutive rows (checked).
  /// \param device_indices  CPU int64 [n]: device row of each index, same structure.
  /// \param host_ptrs       CPU uint64 [layers]: address of host row 0 per layer.
  /// \param device_ptrs     CPU uint64 [layers]: address of device row 0 per layer.
  /// \param host_rows       rows in each host layer.
  /// \param device_rows     rows in each device layer.
  /// \param rows_per_page   rows per page.
  /// \param row_bytes       bytes per row, the same on both sides.
  /// \param device_id       the CUDA device of the device rows.
  /// \param stream          cudaStream_t to issue on (the caller's current stream); every
  ///                        tensor argument is on the CPU, so it cannot be inferred.
  /// \param to_host         copy device -> host, else host -> device.
  static void
  run(tvm::ffi::TensorView host_indices,
      tvm::ffi::TensorView device_indices,
      tvm::ffi::TensorView host_ptrs,
      tvm::ffi::TensorView device_ptrs,
      int64_t host_rows,
      int64_t device_rows,
      int64_t rows_per_page,
      int64_t row_bytes,
      int64_t device_id,
      int64_t stream,
      bool to_host) {
    using namespace host;
    auto n = SymbolicSize{"indices"};
    auto layers = SymbolicSize{"layers"};
    TensorMatcher({n})  //
        .with_dtype<int64_t>()
        .with_device<kDLCPU>()
        .verify(host_indices)
        .verify(device_indices);
    TensorMatcher({layers})  //
        .with_dtype<uint64_t>()
        .with_device<kDLCPU>()
        .verify(host_ptrs)
        .verify(device_ptrs);
    RuntimeCheck(rows_per_page > 0 && row_bytes > 0, "rows_per_page and row_bytes must be positive");
    RuntimeCheck(n.unwrap() % rows_per_page == 0, "index count ", n.unwrap(), " is not a whole number of pages");
    const int64_t num_pages = n.unwrap() / rows_per_page;
    if (num_pages == 0 || layers.unwrap() == 0) return;

    const auto* h = static_cast<const int64_t*>(host_indices.data_ptr());
    const auto* d = static_cast<const int64_t*>(device_indices.data_ptr());

    // Paired runs: (first host row, first device row, pages).
    std::vector<int64_t> run_host, run_device, run_pages;
    for (int64_t p = 0; p < num_pages; ++p) {
      const int64_t first = p * rows_per_page;
      const int64_t host_row = h[first], device_row = d[first];
      // Written so no term can overflow for any index value.
      RuntimeCheck(
          host_row >= 0 && host_row <= host_rows - rows_per_page && device_row >= 0 &&
              device_row <= device_rows - rows_per_page,
          "page ",
          p,
          " (host row ",
          host_row,
          ", device row ",
          device_row,
          ") is out of range");
      // Branch-free OR of every row's offset error, so the loop vectorizes. Unsigned: an
      // arbitrary invalid index must not overflow, and a zero difference modulo 2^64 is an
      // exact match because the page start is in range.
      const auto host_base_row = static_cast<uint64_t>(host_row);
      const auto device_base_row = static_cast<uint64_t>(device_row);
      uint64_t offset_error = 0;
      for (int64_t k = 1; k < rows_per_page; ++k) {
        const auto uk = static_cast<uint64_t>(k);
        offset_error |= (static_cast<uint64_t>(h[first + k]) - host_base_row - uk) |
                        (static_cast<uint64_t>(d[first + k]) - device_base_row - uk);
      }
      RuntimeCheck(offset_error == 0, "page ", p, " does not span consecutive rows");
      if (!run_pages.empty()) {
        const int64_t span = run_pages.back() * rows_per_page;
        if (host_row == run_host.back() + span && device_row == run_device.back() + span) {
          ++run_pages.back();
          continue;
        }
      }
      run_host.push_back(host_row);
      run_device.push_back(device_row);
      run_pages.push_back(1);
    }

    const auto* host_base = static_cast<const uint64_t*>(host_ptrs.data_ptr());
    const auto* device_base = static_cast<const uint64_t*>(device_ptrs.data_ptr());
    const auto page_bytes = static_cast<size_t>(row_bytes * rows_per_page);
    const size_t runs = run_pages.size();
    const size_t count = runs * static_cast<size_t>(layers.unwrap());
    std::vector<void*> dsts, srcs;
    std::vector<size_t> sizes;
    dsts.reserve(count);
    srcs.reserve(count);
    sizes.reserve(count);
    for (int64_t l = 0; l < layers.unwrap(); ++l) {
      for (size_t r = 0; r < runs; ++r) {
        auto* host_ptr = reinterpret_cast<char*>(host_base[l] + run_host[r] * row_bytes);
        auto* device_ptr = reinterpret_cast<char*>(device_base[l] + run_device[r] * row_bytes);
        dsts.push_back(to_host ? host_ptr : device_ptr);
        srcs.push_back(to_host ? device_ptr : host_ptr);
        sizes.push_back(run_pages[r] * page_bytes);
      }
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (try_submit_batch(dsts, srcs, sizes, cuda_stream, static_cast<int>(device_id), to_host)) return;
    for (size_t i = 0; i < count; ++i) {
      check_copy(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, cuda_stream));
    }
  }

 private:
  // Clears the CUDA error state before throwing, so a caller that falls back to another copy
  // path does not trip over this error at its next CUDA call.
  static void check_copy(cudaError_t err) {
    if (err == cudaSuccess) return;
    (void)cudaGetLastError();
    host::RuntimeCheck(false, "page run copy failed: ", cudaGetErrorString(err));
  }

  // False when the batch API is unavailable here; the caller then issues one copy per descriptor.
  static bool try_submit_batch(
      std::vector<void*>& dsts,
      std::vector<void*>& srcs,
      std::vector<size_t>& sizes,
      cudaStream_t stream,
      int device_id,
      bool to_host) {
#if defined(USE_ROCM) || !defined(CUDA_VERSION) || CUDA_VERSION < 12080
    return false;
#else
    // cudaMemcpyBatchAsync rejects the legacy default stream.
    if (stream == nullptr || !memcpy_batch::available()) return false;

    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    const cudaMemLocation device_loc{cudaMemLocationTypeDevice, device_id};
    const cudaMemLocation host_loc{cudaMemLocationTypeHost, 0};
    attrs.srcLocHint = to_host ? device_loc : host_loc;
    attrs.dstLocHint = to_host ? host_loc : device_loc;
    attrs.flags = 0;
    const cudaError_t err = memcpy_batch::copy(dsts.data(), srcs.data(), sizes.data(), sizes.size(), &attrs, stream);
    if (err == cudaErrorNotSupported || err == cudaErrorCallRequiresNewerDriver || err == cudaErrorInvalidValue) {
      (void)cudaGetLastError();
      return false;
    }
    check_copy(err);
    return true;
#endif
  }
};

}  // namespace sglang
