#pragma once

// The vendored DeepSelect sources declare everything at global scope, so they
// are included before `namespace sglang` is opened.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "vendor/cuda_kernels/v3/topk_select.cuh"
#include "vendor/cuda_kernels/v3_cluster/topk_select.cuh"
#include "vendor/cuda_kernels/v3_fp32/topk_select.cuh"
#include <cstdint>
#include <type_traits>

namespace sglang {

/**
 * \brief Host entry points for the vendored DeepSelect top-k kernels.
 *
 * Upstream resolves a `TopkSelectConfig` from the call at runtime, expanding
 * every boolean and index-type axis through nested `BOOL_SWITCH` macros. Here
 * those axes -- value dtype, index dtype, `sorted_value`, `sorted_index`,
 * `return_value`, the top-k bucket and the cluster size -- are template
 * arguments picked by the Python module factory, which leaves exactly one
 * decision for C++: which of a bucket's two tunings the launch shape wants.
 */
namespace deepselect {

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
    tvm::ffi::Optional<tvm::ffi::TensorView> end,
    tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
    int64_t topk,
    int64_t idx_oob_fill_value,
    double value_oob_fill_value,
    bool abort_when_nan_found) -> LaunchPlan {
  using namespace host;
  using ValueT = typename Config::ValueT;
  using OutIdxT = typename Config::OutIdxT;

  auto batch = SymbolicSize{"batch_size"};
  auto vocab = SymbolicSize{"vocab_size"};
  auto input_stride = SymbolicSize{"input_row_stride"};
  auto index_stride = SymbolicSize{"index_row_stride"};
  auto value_stride = SymbolicSize{"value_row_stride"};
  auto device = SymbolicDevice{};

  TensorMatcher({batch, vocab})
      .with_strides({input_stride, 1})
      .with_device<kDLCUDA>(device)
      .with_dtype<ValueT>()
      .ensure_alignment(INPUT_STRIDE_ALIGNMENT_REQUIREMENT)
      .verify(input);
  TensorMatcher({batch, topk})
      .with_strides({index_stride, 1})
      .with_device<kDLCUDA>(device)
      .with_dtype<OutIdxT>()
      .ensure_alignment(OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT)
      .verify(output_index);

  void* output_value_ptr = nullptr;
  uint64_t stride_output_value = 0;
  if constexpr (Config::return_value) {
    CHECK_HOST(output_value.has_value()) << "output_value is required when return_value is enabled";
    TensorMatcher({batch, topk})
        .with_strides({value_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype<ValueT>()
        .ensure_alignment(OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT)
        .verify(output_value.value());
    output_value_ptr = output_value.value().data_ptr();
    stride_output_value = static_cast<uint64_t>(value_stride.unwrap());
  } else {
    CHECK_HOST(!output_value.has_value()) << "output_value must be absent when return_value is disabled";
  }

  int* end_ptr = nullptr;
  if (end.has_value()) {
    TensorMatcher({batch}).with_device<kDLCUDA>(device).with_dtype<int32_t>().verify(end.value());
    end_ptr = static_cast<int*>(end.value().data_ptr());
  }
  int* index_offset_ptr = nullptr;
  if (output_index_offset.has_value()) {
    TensorMatcher({batch}).with_device<kDLCUDA>(device).with_dtype<int32_t>().verify(output_index_offset.value());
    index_offset_ptr = static_cast<int*>(output_index_offset.value().data_ptr());
  }

  const auto batch_size = static_cast<uint32_t>(batch.unwrap());
  const auto vocab_size = static_cast<uint32_t>(vocab.unwrap());
  CHECK_HOST(topk > 0 && topk <= static_cast<int64_t>(Config::max_topk))
      << "topk must be in (0, " << Config::max_topk << "] for this module, got " << topk;
  CHECK_HOST(vocab_size < MAX_VOCAB_SIZE) << "vocab_size must be < " << MAX_VOCAB_SIZE << ", got " << vocab_size;

  const DLDevice dev = device.unwrap();
  return LaunchPlan{
      TopkSelectArgs{
          batch_size,
          vocab_size,
          static_cast<uint32_t>(topk),
          input.data_ptr(),
          output_value_ptr,
          output_index.data_ptr(),
          nullptr,  // `begin` is unimplemented upstream
          end_ptr,
          index_offset_ptr,
          static_cast<uint64_t>(input_stride.unwrap()),
          stride_output_value,
          static_cast<uint64_t>(index_stride.unwrap()),
          Config::sorted_value,
          Config::sorted_index,
          Config::return_value,
          static_cast<int>(idx_oob_fill_value),
          static_cast<float>(value_oob_fill_value),
          abort_when_nan_found,
          host::runtime::get_max_smem_per_block(dev.device_id),
          host::LaunchKernel::resolve_device(dev),
      },
      dev,
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

  static auto topk(
      tvm::ffi::TensorView input,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_value,
      tvm::ffi::TensorView output_index,
      tvm::ffi::Optional<tvm::ffi::TensorView> end,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
      int64_t topk,
      int64_t idx_oob_fill_value,
      double value_oob_fill_value,
      bool abort_when_nan_found) -> void {
    const auto plan = details::make_plan<ConfigWave1>(
        input,
        output_value,
        output_index,
        end,
        output_index_offset,
        topk,
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
      tvm::ffi::Optional<tvm::ffi::TensorView> end,
      tvm::ffi::Optional<tvm::ffi::TensorView> output_index_offset,
      int64_t topk,
      int64_t idx_oob_fill_value,
      double value_oob_fill_value,
      bool abort_when_nan_found) -> void {
    const auto plan = details::make_plan<Config>(
        input,
        output_value,
        output_index,
        end,
        output_index_offset,
        topk,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found);
    if (plan.args.batch_size == 0) return;
    topk_select_bf16_cluster::run_topk_select_kernel<Config>(plan.args);
  }
};

}  // namespace deepselect

}  // namespace sglang
