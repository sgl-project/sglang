#pragma once

#include <sgl_kernel/bits.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cuda_kernels/v3/topk_select.cuh>
#include <cuda_kernels/v3_cluster/topk_select.cuh>
#include <cuda_kernels/v3_fp32/topk_select.cuh>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "structs.h"
#include <cstdint>
#include <type_traits>

namespace sglang {

/// NOTE: input stride can be largely relaxed
inline constexpr uint32_t kInputAlignmentBytes = 128u;
/// NOTE: fix this value for SM90 and SM100 (SM90 is actually 16, not 32)
inline constexpr uint32_t kOutputAlignmentBytes = OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT;

/**
 * \brief Host entry points for the vendored DeepSelect top-k kernels.
 *
 * Upstream resolves a `TopkSelectConfig` from the call at runtime, expanding
 * every boolean and index-type axis through nested `BOOL_SWITCH` macros. Here
 * those axes -- value dtype, index dtype, `sorted_value`, `sorted_index`,
 * `return_value`, `page_transform`, the top-k bucket and the cluster size -- are template
 * arguments picked by the Python module factory, which leaves exactly one
 * decision for C++: which of a bucket's two tunings the launch shape wants.
 */
namespace deep_select {

namespace details {

/// The two families of one-CTA-per-row kernels live in separate namespaces.
template <typename Config>
inline auto run_normal(const TopkSelectArgs& args) -> void {
  if constexpr (std::is_same_v<typename Config::ValueT, float>) {
    topk_select_fp32::run_topk_select_kernel<Config>(args);
  } else {
    topk_select_bf16_normal::run_topk_select_kernel<Config>(args);
  }
}

/// Validated launch arguments plus the device they were validated against.
struct LaunchPlan {
  TopkSelectArgs args;
  DLDevice device;
};

/**
 * \brief Validate the tensors and build the argument block for one launch.
 *
 * \tparam Config The specialisation the caller is about to run. Its
 *         `sorted_value` / `sorted_index` / `return_value` become the runtime
 *         flags, which the kernel launcher then asserts back against them.
 */
template <typename Config>
auto make_plan(
    tvm::ffi::TensorView input,
    tvm::ffi::Optional<tvm::ffi::TensorView> output_value,
    tvm::ffi::TensorView output_index,
    tvm::ffi::Optional<tvm::ffi::TensorView> begin,
    tvm::ffi::Optional<tvm::ffi::TensorView> end,
    tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
    tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
    uint32_t page_size,
    uint32_t topk,
    int64_t input_storage_bytes,
    int32_t idx_oob_fill_value,
    float value_oob_fill_value,
    bool abort_when_nan_found) -> LaunchPlan {
  using namespace host;
  using ValueT = typename Config::ValueT;
  using OutIdxT = typename Config::OutIdxT;

  auto B = SymbolicSize{"batch_size"};
  auto device_ = SymbolicDevice{};
  CHECK_HOST(topk > 0 && topk <= Config::max_topk) << topk << " out of (0, " << Config::max_topk << "]";
  CHECK_HOST(!begin.has_value()) << "`begin` is not supported currently";
  CHECK_HOST(input_storage_bytes >= 0) << "input_storage_bytes must be non-negative";
  TensorMatcher({B, -1})
      .with_strides({-1, 1})
      .with_device<kDLCUDA>(device_)
      .with_dtype<ValueT>()
      .ensure_alignment(kInputAlignmentBytes)
      .verify(input);
  TensorMatcher({B, topk})
      .with_strides({-1, 1})
      .with_device<kDLCUDA>(device_)
      .with_dtype<OutIdxT>()
      .ensure_alignment(kOutputAlignmentBytes)
      .verify(output_index);
  CHECK_HOST((topk * sizeof(OutIdxT)) % kOutputAlignmentBytes == 0);
  void* output_value_ptr = nullptr;
  uint64_t stride_output_value = 0;
  if constexpr (Config::return_value) {
    CHECK_HOST(output_value.has_value());
    const auto output_value_ = output_value.value();
    TensorMatcher({B, topk})
        .with_strides({-1, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype<ValueT>()
        .ensure_alignment(kOutputAlignmentBytes)
        .verify(output_value_);
    CHECK_HOST((topk * sizeof(ValueT)) % kOutputAlignmentBytes == 0);
    output_value_ptr = output_value_.data_ptr();
    stride_output_value = output_value_.stride(0);
  } else {
    CHECK_HOST(!output_value.has_value());
  }

  int* end_ptr = nullptr;
  if (end.has_value()) {
    TensorMatcher({B}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(end.value());
    end_ptr = static_cast<int*>(end.value().data_ptr());
  }
  int* index_offset_ptr = nullptr;
  if (output_index_offset.has_value()) {
    TensorMatcher({B}).with_device<kDLCUDA>(device_).with_dtype<int32_t>().verify(output_index_offset.value());
    index_offset_ptr = static_cast<int*>(output_index_offset.value().data_ptr());
  }

  const auto batch_size = static_cast<uint32_t>(B.unwrap());
  const auto vocab_size = static_cast<uint32_t>(input.size(1));
  CHECK_HOST(vocab_size < MAX_VOCAB_SIZE) << "vocab_size must be < " << MAX_VOCAB_SIZE << ", got " << vocab_size;

  const int* page_table_ptr = nullptr;
  uint64_t stride_page_table = 0;
  uint32_t page_bits = 0;
  if constexpr (Config::page_transform) {
    CHECK_HOST(page_table.has_value() && !output_index_offset.has_value());
    const auto page_table_ = page_table.value();
    TensorMatcher({B, -1}).with_strides({-1, 1}).with_device<kDLCUDA>(device_).with_dtype<int>().verify(page_table_);
    const auto num_pages = page_table_.size(1);
    CHECK_HOST(num_pages >= 1 && num_pages * page_size >= static_cast<int64_t>(vocab_size));
    CHECK_HOST(page_size > 0 && is_pow2(page_size)) << page_size;
    page_table_ptr = static_cast<const int*>(page_table_.data_ptr());
    stride_page_table = page_table_.stride(0);
    page_bits = log2_ceil(page_size);
  } else {
    CHECK_HOST(!page_table.has_value());
  }

  const auto device = device_.unwrap();
  const auto input_stride = batch_size > 1 ? input.stride(0) : kInputAlignmentBytes;
  CHECK_HOST(input_stride >= 0) << "input row stride must be non-negative";
  CHECK_HOST((input_stride * sizeof(ValueT)) % kInputAlignmentBytes == 0)
      << "input row stride must be a multiple of " << kInputAlignmentBytes << " bytes";
  static_assert(kInputAlignmentBytes % 128 == 0);

  if (batch_size != 0 && vocab_size != 0) {
    const auto row_bytes = static_cast<uint64_t>(vocab_size) * sizeof(ValueT);
    const auto padded_row_bytes = div_ceil(row_bytes, 128) * 128;
    const auto front_bytes = static_cast<size_t>(batch_size - 1) * input_stride * sizeof(ValueT);
    const auto required_bytes = front_bytes + padded_row_bytes;
    CHECK_HOST(required_bytes <= static_cast<uint64_t>(input_storage_bytes))
        << "input storage must include the final row padded to 128 bytes";
  }

  return LaunchPlan{
      TopkSelectArgs{
          batch_size,
          vocab_size,
          topk,
          input.data_ptr(),
          output_value_ptr,
          output_index.data_ptr(),
          nullptr,  // `begin` is unimplemented upstream
          end_ptr,
          index_offset_ptr,
          static_cast<uint64_t>(input_stride),
          stride_output_value,
          static_cast<uint64_t>(output_index.stride(0)),
          Config::sorted_value,
          Config::sorted_index,
          Config::return_value,
          static_cast<int>(idx_oob_fill_value),
          static_cast<float>(value_oob_fill_value),
          abort_when_nan_found,
          runtime::get_max_smem_per_block(device.device_id),
          LaunchKernel::resolve_device(device),
          page_table_ptr,
          stride_page_table,
          page_bits,
      },
      device,
  };
}

}  // namespace details

/**
 * \brief Top-k through the one-CTA-per-row kernels.
 *
 * \tparam ConfigWave1 Tuning for a batch that fits in a single wave.
 * \tparam ConfigWaves Tuning for anything larger. A bucket with no wave split
 *         leaves it defaulted; one kernel is then instantiated and the SM-count
 *         query drops out.
 */
template <typename ConfigWave1, typename ConfigWaves = ConfigWave1>
struct TopkNormal {
  static_assert(ConfigWave1::cluster_size == 1 && ConfigWaves::cluster_size == 1);
  static_assert(std::is_same_v<typename ConfigWave1::ValueT, typename ConfigWaves::ValueT>);
  static_assert(std::is_same_v<typename ConfigWave1::OutIdxT, typename ConfigWaves::OutIdxT>);
  static_assert(ConfigWave1::max_topk == ConfigWaves::max_topk);
  static_assert(ConfigWave1::sorted_value == ConfigWaves::sorted_value);
  static_assert(ConfigWave1::sorted_index == ConfigWaves::sorted_index);
  static_assert(ConfigWave1::return_value == ConfigWaves::return_value);
  static_assert(ConfigWave1::page_transform == ConfigWaves::page_transform);

  static auto topk(
      tvm::ffi::TensorView input,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_value,
      tvm::ffi::TensorView output_index,
      tvm::ffi::Optional<tvm::ffi::TensorView> begin,
      tvm::ffi::Optional<tvm::ffi::TensorView> end,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
      tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      uint32_t page_size,
      uint32_t topk,
      int64_t input_storage_bytes,
      int32_t idx_oob_fill_value,
      float value_oob_fill_value,
      bool abort_when_nan_found) -> void {
    const auto plan = details::make_plan<ConfigWave1>(
        input,
        output_value,
        output_index,
        begin,
        end,
        output_index_offset,
        page_table,
        page_size,
        topk,
        input_storage_bytes,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found);
    if (plan.args.batch_size == 0) return;
    if constexpr (std::is_same_v<ConfigWave1, ConfigWaves>) {
      details::run_normal<ConfigWave1>(plan.args);
    } else {
      const auto num_sm = host::runtime::get_sm_count(plan.device.device_id);
      if (plan.args.batch_size <= num_sm) {
        details::run_normal<ConfigWave1>(plan.args);
      } else {
        details::run_normal<ConfigWaves>(plan.args);
      }
    }
  }
};

/**
 * \brief Top-k through the cluster kernel: `Config::cluster_size` CTAs per row.
 *
 * Whether a launch belongs here -- small batch, long rows, bf16 -- is decided
 * in Python, which also picks the cluster size: 8 CTAs on SM90, 16 on
 * SM100/SM103, where 16 needs the non-portable cluster attribute.
 */
template <typename Config>
struct TopkCluster {
  static_assert(Config::cluster_size > 1);

  static auto topk(
      tvm::ffi::TensorView input,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_value,
      tvm::ffi::TensorView output_index,
      tvm::ffi::Optional<tvm::ffi::TensorView> begin,
      tvm::ffi::Optional<tvm::ffi::TensorView> end,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
      tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      uint32_t page_size,
      uint32_t topk,
      int64_t input_storage_bytes,
      int32_t idx_oob_fill_value,
      float value_oob_fill_value,
      bool abort_when_nan_found) -> void {
    const auto plan = details::make_plan<Config>(
        input,
        output_value,
        output_index,
        begin,
        end,
        output_index_offset,
        page_table,
        page_size,
        topk,
        input_storage_bytes,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found);
    if (plan.args.batch_size == 0) return;
    topk_select_bf16_cluster::run_topk_select_kernel<Config>(plan.args);
  }
};

}  // namespace deep_select

}  // namespace sglang
