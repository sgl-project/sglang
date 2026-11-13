// Top-k index page-table transform, in place: a request-relative key position p >= 0 becomes the
// physical KV slot `page_table[row_batch[r]][p / kPageSize] * kPageSize + p % kPageSize`; -1 stays -1.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

struct TopkPageTableTransformParams {
  int32_t* __restrict__ indices;           // [rows, topk]
  const int32_t* __restrict__ page_table;  // [batches, width]
  const int32_t* __restrict__ row_batch;   // [rows]
  int64_t indices_stride;
  int64_t page_table_stride;
  uint32_t total_quads;  // rows * topk / 4
};

template <uint32_t kPageSize, uint32_t kTopK, bool kUsePDL>
__global__ void topk_page_table_transform_kernel(const __grid_constant__ TopkPageTableTransformParams params) {
  using namespace device;
  constexpr uint32_t kNumQuads = kTopK / 4;
  const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row = q / kNumQuads;
  const auto q4 = q % kNumQuads;
  PDLWaitPrimary<kUsePDL>();
  if (q < params.total_quads) {
    const auto page_row = params.page_table + params.row_batch[row] * params.page_table_stride;
    const auto indices_row = params.indices + row * params.indices_stride;
    AlignedVector<int32_t, 4> pos;
    pos.load(indices_row, q4);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      if (pos[i] >= 0) pos[i] = page_row[pos[i] / kPageSize] * kPageSize + pos[i] % kPageSize;
    }
    pos.store(indices_row, q4);
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <uint32_t kPageSize, uint32_t kTopK, bool kUsePDL>
struct TopkPageTableTransformKernel {
  static_assert(kTopK > 0 && kTopK % 4 == 0, "topk must be a positive multiple of 4");
  static constexpr uint32_t kBlockSize = 128;
  static constexpr auto kernel = topk_page_table_transform_kernel<kPageSize, kTopK, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView indices, const tvm::ffi::TensorView page_table, const tvm::ffi::TensorView row_batch) {
    using namespace host;
    auto R = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    TensorMatcher({R, kTopK})  // indices
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device)
        .ensure_alignment(16)
        .verify(indices);
    TensorMatcher({-1, -1})  // page_table
        .with_strides({-1, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_table);
    TensorMatcher({R})  // row_batch
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(row_batch);
    const auto num_rows = static_cast<uint32_t>(R.unwrap());
    if (num_rows == 0) return;
    const auto total_quads = static_cast<uint64_t>(num_rows) * (kTopK / 4);
    CHECK_HOST(total_quads <= UINT32_MAX) << "too many indices: " << num_rows << " x " << kTopK;
    const auto params = TopkPageTableTransformParams{
        .indices = static_cast<int32_t*>(indices.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .row_batch = static_cast<const int32_t*>(row_batch.data_ptr()),
        .indices_stride = indices.stride(0),
        .page_table_stride = page_table.stride(0),
        .total_quads = static_cast<uint32_t>(total_quads),
    };
    const auto grid = div_ceil(params.total_quads, kBlockSize);
    LaunchKernel(grid, kBlockSize, device.unwrap()).config({.use_pdl = kUsePDL})(kernel, params);
  }
};

}  // namespace sglang
