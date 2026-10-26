#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <fused_mqa_topk/launch.cuh>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <climits>
#include <cstdint>

namespace sglang {

/**
 * \brief Fused FP8 MQA-logits + per-row top-k of the DSA indexer (SM100).
 *
 * Row `r` scores every key `j` in `[starts[r], ends[r])` as
 * `k_scale[j] * sum_h weights[r, h] * relu(q[r, h] . k[j])` and writes the
 * `topk` best as `j - starts[r] (+ out_offsets[r])`, or as
 * `page_table[row_batch[r], j - starts[r]]`, unordered and `-1` padded when
 * fewer keys are visible. The `[rows, keys]` logits are never materialized.
 * The launch runs on the current stream and neither allocates nor synchronizes,
 * so it can be captured; the workspace must not be shared with a launch that
 * may run concurrently.
 *
 * \param q                [rows, 32, 128] fp8 e4m3
 * \param k                [keys, 128] fp8 e4m3, keys in [20481, 2^20]
 * \param k_scale          [keys] fp32 per-key dequant scale
 * \param weights          [rows, 32] fp32 per-head gate, q scale folded in
 * \param starts           [rows] int32, first visible key of each row
 * \param ends             [rows] int32, one past the last visible key
 * \param out_offsets      optional [rows] int32 added to every selected index
 * \param page_table       optional [requests, width] int32 page_size=1 table with
 *                         unit inner stride and width >= the keys any row of the
 *                         request sees; excludes `out_offsets`
 * \param row_batch        [rows] int32 page-table row of each query row, given
 *                         exactly when `page_table` is
 * \param out              [rows, topk] int32, topk = 2048 (the only instantiation)
 * \param counter          [2] int32 claim counter, zero before the first launch;
 *                         every launch leaves it zero
 * \param values           [8 * slots, 16384] fp16 candidate slab, slots even
 * \param indices          [8 * slots, 16384] int32
 * \param retained         [8 * slots, 4096] fp32 exact-rescan slab
 * \param retained_indices [8 * slots, 4096] int32
 * \param counts           [>= rows rounded up to 8] int32
 * \param defer            [>= ceil(rows / 8)] int32
 */
inline void fused_mqa_topk_prefill(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView k_scale,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView starts,
    tvm::ffi::TensorView ends,
    tvm::ffi::Optional<tvm::ffi::TensorView> out_offsets,
    tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
    tvm::ffi::Optional<tvm::ffi::TensorView> row_batch,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView counter,
    tvm::ffi::TensorView values,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView retained,
    tvm::ffi::TensorView retained_indices,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView defer) {
  using namespace host;
  namespace kernel = ::fused_mqa_topk;

  auto R = SymbolicSize{"rows"};
  auto K = SymbolicSize{"keys"};
  auto TOPK = SymbolicSize{"topk"};
  auto SLOT_ROWS = SymbolicSize{"slot_rows"};
  auto COUNTS = SymbolicSize{"counts"};
  auto DEFER = SymbolicSize{"defer"};
  auto device_ = SymbolicDevice{};

  // TMA tensor maps need 16-byte aligned bases and row strides; the output and
  // the offset pass use 16-byte stores.
  TensorMatcher({R, kernel::kHeads, kernel::kHeadDim})
      .with_dtype<fp8_e4m3_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(q);
  TensorMatcher({K, kernel::kHeadDim})
      .with_dtype<fp8_e4m3_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(k);
  TensorMatcher({K}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).ensure_alignment(16).verify(k_scale);
  TensorMatcher({R, kernel::kHeads})
      .with_dtype<fp32_t>()
      .with_device<kDLCUDA>(device_)
      .ensure_alignment(16)
      .verify(weights);
  TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(starts).verify(ends);
  TensorMatcher({R, TOPK}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).ensure_alignment(16).verify(out);
  const int32_t* out_offsets_ptr = nullptr;
  if (out_offsets.has_value()) {
    TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_offsets.value());
    out_offsets_ptr = static_cast<const int32_t*>(out_offsets.value().data_ptr());
  }
  CHECK_HOST(page_table.has_value() == row_batch.has_value()) << "page_table and row_batch go together";
  CHECK_HOST(!(page_table.has_value() && out_offsets.has_value())) << "page_table excludes out_offsets";
  const int32_t* page_table_ptr = nullptr;
  const int32_t* row_batch_ptr = nullptr;
  int64_t page_table_stride = 0;
  if (page_table.has_value()) {
    const auto table = page_table.value();
    TensorMatcher({-1, -1}).with_strides({-1, 1}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(table);
    TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(row_batch.value());
    page_table_ptr = static_cast<const int32_t*>(table.data_ptr());
    row_batch_ptr = static_cast<const int32_t*>(row_batch.value().data_ptr());
    page_table_stride = table.stride(0);
  }

  TensorMatcher({2}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(counter);
  TensorMatcher({SLOT_ROWS, kernel::kCandidates}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(values);
  TensorMatcher({SLOT_ROWS, kernel::kCandidates}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(indices);
  TensorMatcher({SLOT_ROWS, kernel::kRetained}).with_dtype<fp32_t>().with_device<kDLCUDA>(device_).verify(retained);
  TensorMatcher({SLOT_ROWS, kernel::kRetained})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_)
      .verify(retained_indices);
  TensorMatcher({COUNTS}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(counts);
  TensorMatcher({DEFER}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(defer);

  const int64_t rows = R.unwrap();
  const int64_t keys = K.unwrap();
  const int64_t topk = TOPK.unwrap();
  const int64_t blocks = div_ceil(rows, int64_t{8});
  // rows * 32 is the int32 outer extent of the Q tensor map.
  CHECK_HOST(rows > 0 && rows <= INT32_MAX / kernel::kHeads)
      << "rows must be in [1, " << INT32_MAX / kernel::kHeads << "]";
  CHECK_HOST(keys >= ::fused_persist::kMinKeys && keys <= kernel::kMaxKeys)
      << "keys must be in [" << ::fused_persist::kMinKeys << ", " << kernel::kMaxKeys << "]";
  CHECK_HOST(topk == ::fused_persist::kTopK) << "topk must be " << ::fused_persist::kTopK << ", got " << topk;
  CHECK_HOST(SLOT_ROWS.unwrap() > 0 && SLOT_ROWS.unwrap() % 16 == 0)
      << "candidate slabs must hold an even number of 8-row slots";
  CHECK_HOST(COUNTS.unwrap() >= blocks * 8 && DEFER.unwrap() >= blocks) << "per-row workspace too small";

  const DLDevice device = device_.unwrap();
  int current_device = -1;
  CHECK_CUDA(cudaGetDevice(&current_device));
  // The launcher sets kernel attributes on the current device.
  CHECK_HOST(current_device == device.device_id)
      << "tensors are on cuda:" << device.device_id << " but the current device is cuda:" << current_device;

  kernel::FusedMqaTopkParams params{};
  params.q = q.data_ptr();
  params.k = k.data_ptr();
  params.scales = static_cast<const float*>(k_scale.data_ptr());
  params.weights = static_cast<const float*>(weights.data_ptr());
  params.starts = static_cast<const int32_t*>(starts.data_ptr());
  params.ends = static_cast<const int32_t*>(ends.data_ptr());
  params.out_offsets = out_offsets_ptr;
  params.page_table = page_table_ptr;
  params.page_table_stride = page_table_stride;
  params.row_batch = row_batch_ptr;
  params.out = static_cast<int32_t*>(out.data_ptr());
  params.rows = rows;
  params.keys = keys;
  params.topk = topk;
  params.relative = true;
  params.counter = static_cast<int32_t*>(counter.data_ptr());
  params.values = static_cast<uint16_t*>(values.data_ptr());
  params.indices = static_cast<int32_t*>(indices.data_ptr());
  params.retained = static_cast<float*>(retained.data_ptr());
  params.retained_indices = static_cast<int32_t*>(retained_indices.data_ptr());
  params.counts = static_cast<int32_t*>(counts.data_ptr());
  params.defer = static_cast<int32_t*>(defer.data_ptr());
  params.slots = SLOT_ROWS.unwrap() / 8;
  params.num_sms = static_cast<int>(runtime::get_sm_count(device.device_id));
  params.stream = LaunchKernel::resolve_device(device);
  CHECK_HOST(kernel::params_supported(params)) << "unsupported fused_mqa_topk_prefill parameters";
  CHECK_CUDA(kernel::launch(params)) << "fused_mqa_topk_prefill launch failed";
}

}  // namespace sglang
