/*
 * SM70 FP16 MoE GEMM using TurboMind s884h kernels.
 * Completely self-contained — no AWQ dependency.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace sglang::sm70_fp16_moe {

namespace {

struct WorkspaceHolder {
  torch::Tensor barriers;
  torch::Tensor partials;
  torch::Tensor tensormaps;
  torch::Tensor flags;
  turbomind::gemm::Workspace workspace{};
};

struct GemmHolder {
  std::unique_ptr<turbomind::gemm::Gemm> gemm;
};

struct StreamWorkspaceKey {
  int device;
  cudaStream_t stream;

  bool operator==(const StreamWorkspaceKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct StreamWorkspaceKeyHash {
  std::size_t operator()(const StreamWorkspaceKey& key) const {
    return std::hash<int>()(key.device) ^
           (std::hash<cudaStream_t>()(key.stream) << 1);
  }
};

std::mutex workspace_mutex;
std::mutex gemm_mutex;
std::unordered_map<StreamWorkspaceKey, std::unique_ptr<WorkspaceHolder>,
                   StreamWorkspaceKeyHash>
    workspace_cache;
std::unordered_map<int, GemmHolder> gemm_cache;
std::once_flag fast_targets_once;

void install_qwen35b_fast_targets() {
  std::call_once(fast_targets_once, [] {
    const char* enabled =
        std::getenv("SGLANG_SM70_FP16_MOE_FAST_SELECTOR");
    if (enabled && std::atoi(enabled) == 0) {
      return;
    }

    // The pinned TurboMind dispatcher accepts exact launch targets through
    // this compatibility variable.  Its generic heuristic selects an 8-row
    // CTA for Qwen3.6-35B-A3B's 4096/8192-token routed-MoE chunks; the measured
    // 128-row CTAs below are substantially faster on V100 and remain within
    // the normal FP16 accumulation tolerance.
    constexpr const char* targets =
        "sm70_f16_f16_f16_tnt_bbb_32768x256x2048_256|"
        "128x256x16x1x0x1@s884_2x4x1;"
        "sm70_f16_f16_f16_tnt_bbb_32768x2048x128_256|"
        "128x128x16x1x4x1@s884_2x2x1;"
        "sm70_f16_f16_f16_tnt_bbb_65536x256x2048_256|"
        "128x256x16x1x0x1@s884_2x4x1;"
        "sm70_f16_f16_f16_tnt_bbb_65536x2048x128_256|"
        "128x256x16x1x3x1@s884_2x4x1";
    const char* existing =
        std::getenv("VLLM_SM70_AWQ_TP2_FAST_TARGETS");
    std::string combined = existing && *existing
                               ? std::string(existing) + ";" + targets
                               : std::string(targets);
    setenv("VLLM_SM70_AWQ_TP2_FAST_TARGETS", combined.c_str(), 1);
  });
}

const std::unordered_set<int64_t>& get_tune_rows() {
  static const std::unordered_set<int64_t> rows = [] {
    std::unordered_set<int64_t> result;
    const char* value = std::getenv("SGLANG_SM70_FP16_MOE_TUNE_ROWS");
    if (!value) {
      return result;
    }
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
      try {
        const int64_t row_count = std::stoll(item);
        if (row_count > 0) {
          result.insert(row_count);
        }
      } catch (...) {
      }
    }
    return result;
  }();
  return rows;
}

WorkspaceHolder& get_workspace(int device, cudaStream_t stream) {
  StreamWorkspaceKey key{device, stream};
  {
    std::lock_guard<std::mutex> lock(workspace_mutex);
    auto it = workspace_cache.find(key);
    if (it != workspace_cache.end()) {
      return *it->second;
    }
  }

  auto holder = std::make_unique<WorkspaceHolder>();
  auto byte_opts = torch::TensorOptions()
                       .device(torch::Device(torch::kCUDA, device))
                       .dtype(torch::kUInt8);
  auto int_opts = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device))
                      .dtype(torch::kInt32);

  holder->barriers = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kBarriersSize}, byte_opts);
  holder->partials = torch::zeros(
      {(long long)turbomind::gemm::Gemm::kPartialsSize}, byte_opts);
  holder->tensormaps = torch::empty({(long long)(8192 * 128)}, byte_opts);
  holder->flags = torch::zeros({1}, int_opts);

  holder->workspace.barriers = holder->barriers.data_ptr();
  holder->workspace.barriers_size = holder->barriers.numel();
  holder->workspace.partials = holder->partials.data_ptr();
  holder->workspace.partials_size = holder->partials.numel();
  holder->workspace.tensormaps = holder->tensormaps.data_ptr();
  holder->workspace.tensormaps_size = holder->tensormaps.numel();
  holder->workspace.flags = holder->flags.data_ptr<int>();

  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto [insert_it, _] = workspace_cache.emplace(key, std::move(holder));
  return *insert_it->second;
}

turbomind::gemm::Gemm& get_gemm(int device) {
  std::lock_guard<std::mutex> lock(gemm_mutex);
  auto it = gemm_cache.find(device);
  if (it != gemm_cache.end()) {
    return *it->second.gemm;
  }
  GemmHolder holder;
  holder.gemm = std::make_unique<turbomind::gemm::Gemm>();
  auto [insert_it, _] = gemm_cache.emplace(device, std::move(holder));
  return *insert_it->second.gemm;
}

}  // namespace

std::vector<torch::Tensor> prepare_weight(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cuda(), "FP16 MoE weight must be CUDA.");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              "FP16 MoE weight must be float16.");
  TORCH_CHECK(weight.dim() == 2, "FP16 MoE weight must be two-dimensional.");
  TORCH_CHECK(weight.is_contiguous(), "FP16 MoE weight must be contiguous.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(weight));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* converter = converters[0];
  TORCH_CHECK(converter,
              "No compatible TurboMind FP16 weight converter was found.");

  const auto order = converter->order;
  const bool is_a = turbomind::gemm::get_operand_tag(converter->pack) ==
                    turbomind::gemm::OPERAND_A;
  turbomind::gemm::MatrixLayout source_desc{
      turbomind::kHalf,
      order,
      static_cast<int>(n),
      static_cast<int>(k),
      order == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                         : static_cast<int>(n),
  };
  if (!is_a) {
    std::swap(source_desc.rows, source_desc.cols);
    source_desc.order = ~source_desc.order;
  }

  turbomind::gemm::MatrixLayout destination_desc = source_desc;
  destination_desc.pack = converter->pack;
  if (is_a) {
    destination_desc = turbomind::gemm::transpose(destination_desc);
  }

  auto prepared = torch::empty_like(weight);
  TORCH_CHECK(converter->Convert(weight.data_ptr(), source_desc,
                                 prepared.data_ptr(), destination_desc,
                                 stream) == 0,
              "TurboMind FP16 weight conversion failed.");
  auto meta = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64));
  meta.index_put_({0}, destination_desc.ld);
  return {prepared, meta};
}

std::vector<torch::Tensor> build_strided_ptrs(torch::Tensor tm_weights,
                                               int64_t k_ld,
                                               int64_t num_experts) {
  TORCH_CHECK(tm_weights.is_cuda(), "FP16 MoE weights must be CUDA.");
  TORCH_CHECK(num_experts > 0, "FP16 MoE num_experts must be positive.");
  TORCH_CHECK(tm_weights.size(0) == num_experts,
              "FP16 MoE weights dim0 must equal num_experts.");
  TORCH_CHECK(tm_weights.scalar_type() == torch::kFloat16,
              "FP16 MoE weights must be float16.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(tm_weights));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  std::vector<std::pair<void*, int>> weight_ptrs;
  weight_ptrs.reserve(num_experts);
  const int64_t expert_stride =
      tm_weights.stride(0) * tm_weights.element_size();
  char* weight_base = static_cast<char*>(tm_weights.data_ptr());
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    weight_ptrs.emplace_back(weight_base + expert * expert_stride,
                             static_cast<int>(k_ld));
  }

  void* device_ptrs = turbomind::gemm::MakeStridedPtrs(weight_ptrs, stream);
  const int64_t ptr_bytes = num_experts * 16;
  auto result = torch::empty(
      {ptr_bytes},
      torch::TensorOptions().device(tm_weights.device()).dtype(torch::kUInt8));
  cudaMemcpyAsync(result.data_ptr(), device_ptrs, ptr_bytes,
                  cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(device_ptrs, stream);
  return {result};
}

void gemm(torch::Tensor out, torch::Tensor sorted_input,
          torch::Tensor expert_offsets, torch::Tensor strided_weight_ptrs,
          int64_t num_experts, int64_t k, int64_t n, bool gated_silu) {
  TORCH_CHECK(
      sorted_input.is_cuda() && sorted_input.scalar_type() == torch::kFloat16,
      "FP16 MoE input must be CUDA float16.");
  TORCH_CHECK(expert_offsets.is_cuda() &&
                  expert_offsets.scalar_type() == torch::kInt32,
              "FP16 MoE expert offsets must be CUDA int32.");
  TORCH_CHECK(strided_weight_ptrs.is_cuda(),
              "FP16 MoE strided weight pointers must be CUDA.");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat16,
              "FP16 MoE output must be CUDA float16.");
  TORCH_CHECK(num_experts > 0 && k > 0 && n > 0,
              "FP16 MoE dimensions must be positive.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(sorted_input));
  const int device = sorted_input.get_device();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t total_tokens = sorted_input.size(0);

  TORCH_CHECK(out.size(0) == total_tokens,
              "FP16 MoE output rows must match input rows.");
  TORCH_CHECK(out.stride(1) == 1,
              "FP16 MoE output must be row-major contiguous.");
  if (gated_silu) {
    TORCH_CHECK((n % 2) == 0, "FP16 MoE gated SiLU requires even n.");
    TORCH_CHECK(out.size(1) == n / 2,
                "FP16 MoE gated SiLU output has the wrong width.");
  } else {
    TORCH_CHECK(out.size(1) == n,
                "FP16 MoE output has the wrong width.");
  }
  if (total_tokens == 0) {
    return;
  }

  install_qwen35b_fast_targets();

  const auto converters = turbomind::gemm::GetConverters(
      turbomind::kHalf, turbomind::kHalf, turbomind::kHalf, true, 70);
  const auto* weight_converter = converters[0];
  TORCH_CHECK(weight_converter,
              "No compatible TurboMind FP16 MoE converter was found.");

  turbomind::gemm::MatrixLayout input_desc{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(k),
      static_cast<int>(k),
  };
  input_desc.num = static_cast<int>(num_experts);
  input_desc.offsets = expert_offsets.data_ptr<int>();
  turbomind::gemm::MatrixLayout unused_desc{};

  const auto weight_order = weight_converter->order;
  const bool weight_is_a =
      turbomind::gemm::get_operand_tag(weight_converter->pack) ==
      turbomind::gemm::OPERAND_A;
  turbomind::gemm::MatrixLayout unpacked_weight_desc{
      turbomind::kHalf,
      weight_order,
      static_cast<int>(n),
      static_cast<int>(k),
      weight_order == turbomind::gemm::kRowMajor ? static_cast<int>(k)
                                                 : static_cast<int>(n),
  };
  if (!weight_is_a) {
    std::swap(unpacked_weight_desc.rows, unpacked_weight_desc.cols);
    unpacked_weight_desc.order = ~unpacked_weight_desc.order;
  }

  turbomind::gemm::MatrixLayout weight_desc = unpacked_weight_desc;
  weight_desc.pack = weight_converter->pack;
  if (weight_is_a) {
    weight_desc = turbomind::gemm::transpose(weight_desc);
  }
  weight_desc.ld = 0;
  weight_desc.num = static_cast<int>(num_experts);

  turbomind::gemm::MatrixLayout output_desc{
      turbomind::kHalf,
      turbomind::gemm::kRowMajor,
      static_cast<int>(total_tokens),
      static_cast<int>(n),
      static_cast<int>(out.stride(0)),
  };
  output_desc.num = static_cast<int>(num_experts);
  output_desc.offsets = expert_offsets.data_ptr<int>();

  turbomind::gemm::Operation operation{};
  operation.dispatch = get_tune_rows().count(total_tokens)
                           ? turbomind::gemm::DispatchPolicy::kMeasure
                           : turbomind::gemm::DispatchPolicy::kDefault;
  operation.epilogue = gated_silu ? turbomind::gemm::Epilogue::kGatedSilu
                                  : turbomind::gemm::Epilogue::kNone;
  operation.quant_a = {turbomind::gemm::QuantType::kNone, 0};
  operation.quant_b = {turbomind::gemm::QuantType::kNone, 0};
  operation.batch_dim = 0;

  auto& workspace = get_workspace(device, stream);
  auto& runner = get_gemm(device);
  const int error = runner.Run(
      operation, 1.f, sorted_input.data_ptr(), input_desc, nullptr, unused_desc,
      strided_weight_ptrs.data_ptr(), weight_desc, nullptr, unused_desc, 0.f,
      out.data_ptr(), output_desc, out.data_ptr(), output_desc,
      workspace.workspace, stream);
  TORCH_CHECK(error == 0, "TurboMind FP16 MoE GEMM failed (error=", error,
              ").");
}

}  // namespace sglang::sm70_fp16_moe

std::vector<torch::Tensor> sglang_sm70_f16_prepare(torch::Tensor weight) {
  return sglang::sm70_fp16_moe::prepare_weight(weight);
}

std::vector<torch::Tensor> sm70_f16_moe_build_strided_ptrs(
    torch::Tensor tm_weights, int64_t k_ld, int64_t num_experts) {
  return sglang::sm70_fp16_moe::build_strided_ptrs(tm_weights, k_ld,
                                                   num_experts);
}

void sm70_f16_moe_gemm_sm70_out(
    torch::Tensor out, torch::Tensor sorted_input, torch::Tensor expert_offsets,
    torch::Tensor strided_weight_ptrs, int64_t num_experts, int64_t k,
    int64_t n, bool gated_silu) {
  sglang::sm70_fp16_moe::gemm(out, sorted_input, expert_offsets,
                              strided_weight_ptrs, num_experts, k, n,
                              gated_silu);
}
