/*
 * Copyright (c) 2026 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "flashmla/sm90/sparse_nvfp4_dsv4/splitkv_mla.h"
#include "params.h"
#include "smxx/decode/combine/combine.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

namespace {

using Dsv4Nvfp4Result = std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>;

constexpr int kHeadDimQk = 512;
constexpr int kHeadDimV = 512;
constexpr int kBytesPerToken = 380;
constexpr int kBlockSizeTopk = 64;
constexpr int kFixedOverheadNumBlocks = 5;
constexpr float kLog2E = 1.4426950408889634074f;

void check_cuda(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_same_device(const at::Tensor& q, const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same CUDA device as q");
}

void check_optional_device(const at::Tensor& q, const std::optional<at::Tensor>& tensor, const char* name) {
  if (tensor.has_value()) {
    check_cuda(*tensor, name);
    check_same_device(q, *tensor, name);
  }
}

int checked_stride(int64_t stride, const char* name) {
  TORCH_CHECK(stride >= 0 && stride <= std::numeric_limits<int>::max(), name, " stride does not fit in an int32");
  return static_cast<int>(stride);
}

void check_scale(const at::Tensor& q, const at::Tensor& scale, const char* name) {
  check_cuda(scale, name);
  check_same_device(q, scale, name);
  TORCH_CHECK(scale.scalar_type() == at::kFloat, name, " must have dtype float32");
  TORCH_CHECK(scale.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      scale.numel() == 1 && (scale.dim() == 0 || (scale.dim() == 1 && scale.size(0) == 1)),
      name,
      " must be a device scalar or one-element tensor");
}

void check_cache(const at::Tensor& q, const at::Tensor& cache, const char* name) {
  check_cuda(cache, name);
  check_same_device(q, cache, name);
  TORCH_CHECK(cache.scalar_type() == at::kByte, name, " must have dtype uint8");
  TORCH_CHECK(cache.dim() == 4, name, " must have shape [num_pages, page_size, 1, 380]");
  TORCH_CHECK(cache.size(0) > 0, name, " must contain at least one page");
  TORCH_CHECK(cache.size(1) > 0, name, " page size must be positive");
  TORCH_CHECK(cache.size(2) == 1, name, " must contain exactly one KV head");
  TORCH_CHECK(cache.size(3) == kBytesPerToken, name, " rows must use the 380-byte DSV4 NVFP4 layout");
  TORCH_CHECK(
      cache.stride(3) == 1 && cache.stride(2) == kBytesPerToken && cache.stride(1) == kBytesPerToken &&
          cache.stride(0) == cache.size(1) * kBytesPerToken,
      name,
      " must be tightly packed as [num_pages, page_size, 1, 380]");
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(cache.data_ptr()) % alignof(uint32_t) == 0,
      name,
      " data pointer must be 4-byte aligned for the SM90 NVFP4 producer loads");
}

void check_indices(
    const at::Tensor& q, const at::Tensor& indices, const char* name, int64_t batch_size, int64_t seqlen_q) {
  check_cuda(indices, name);
  check_same_device(q, indices, name);
  TORCH_CHECK(indices.scalar_type() == at::kInt, name, " must have dtype int32");
  TORCH_CHECK(
      indices.dim() == 3 && indices.size(0) == batch_size && indices.size(1) == seqlen_q,
      name,
      " must have shape [B, Sq, topk]");
  TORCH_CHECK(indices.size(2) > 0, name, " topk must be positive");
  TORCH_CHECK(indices.size(2) % kBlockSizeTopk == 0, name, " topk must be padded to a multiple of 64");
  TORCH_CHECK(indices.stride(2) == 1, name, " last dimension must be contiguous");
}

void check_length(const at::Tensor& q, const std::optional<at::Tensor>& length, const char* name, int64_t batch_size) {
  if (!length.has_value()) {
    return;
  }
  check_cuda(*length, name);
  check_same_device(q, *length, name);
  TORCH_CHECK(length->scalar_type() == at::kInt, name, " must have dtype int32");
  TORCH_CHECK(length->is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(length->dim() == 1 && length->size(0) == batch_size, name, " must have shape [B]");
}

template <typename T>
T* optional_data_ptr(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? tensor->data_ptr<T>() : nullptr;
}

}  // namespace

Dsv4Nvfp4Result dsv4_sparse_decode_fwd_nvfp4(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& kv_global_scale,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor> tile_scheduler_metadata,
    std::optional<at::Tensor> num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_kv_global_scale,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    const int64_t d_v,
    const double sm_scale) {
  check_cuda(q, "q");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must have dtype bfloat16");
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, Sq, H, 512]");
  TORCH_CHECK(q.stride(3) == 1, "q last dimension must be contiguous");

  const int b = static_cast<int>(q.size(0));
  const int s_q = static_cast<int>(q.size(1));
  const int h_q = static_cast<int>(q.size(2));
  TORCH_CHECK(b > 0, "q batch size must be positive");
  TORCH_CHECK(s_q > 0, "q sequence length must be positive");
  TORCH_CHECK(h_q == 64 || h_q == 128, "q must contain 64 or 128 heads");
  TORCH_CHECK(q.size(3) == kHeadDimQk, "q must have head dimension 512");
  TORCH_CHECK(d_v == kHeadDimV, "d_v must be 512");
  TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0.0, "sm_scale must be finite and positive");

  check_cache(q, kv, "kv");
  check_scale(q, kv_global_scale, "kv_global_scale");
  check_indices(q, indices, "indices", b, s_q);
  check_length(q, topk_length, "topk_length", b);

  check_optional_device(q, attn_sink, "attn_sink");
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->scalar_type() == at::kFloat, "attn_sink must have dtype float32");
    TORCH_CHECK(attn_sink->is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(attn_sink->dim() == 1 && attn_sink->size(0) == h_q, "attn_sink must have shape [H]");
  }

  const bool have_extra = extra_kv.has_value();
  TORCH_CHECK(
      have_extra == extra_kv_global_scale.has_value() && have_extra == extra_indices.has_value() &&
          have_extra == extra_topk_length.has_value(),
      "extra_kv, extra_kv_global_scale, extra_indices, and extra_topk_length must be provided together");

  int extra_num_blocks = 0;
  int extra_page_block_size = 0;
  int extra_topk = 0;
  if (have_extra) {
    check_cache(q, *extra_kv, "extra_kv");
    check_scale(q, *extra_kv_global_scale, "extra_kv_global_scale");
    check_indices(q, *extra_indices, "extra_indices", b, s_q);
    check_length(q, extra_topk_length, "extra_topk_length", b);
    extra_num_blocks = static_cast<int>(extra_kv->size(0));
    extra_page_block_size = static_cast<int>(extra_kv->size(1));
    extra_topk = static_cast<int>(extra_indices->size(2));
  }

  const bool have_metadata = tile_scheduler_metadata.has_value();
  TORCH_CHECK(
      have_metadata == num_splits.has_value(),
      "tile_scheduler_metadata and num_splits must either both be provided or both be omitted");
  check_optional_device(q, tile_scheduler_metadata, "tile_scheduler_metadata");
  check_optional_device(q, num_splits, "num_splits");

  at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
  const auto* device_properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      device_properties->major == 9 && device_properties->minor == 0,
      "dsv4_sparse_decode_fwd_nvfp4 only supports SM90");

  const int num_sm_parts = std::max(device_properties->multiProcessorCount / s_q / (h_q / 64), 1);
  const auto options = q.options();
  at::Tensor out = torch::empty({b, s_q, h_q, kHeadDimV}, options);
  at::Tensor lse = torch::empty({b, s_q, h_q}, options.dtype(at::kFloat));

  if (!have_metadata) {
    tile_scheduler_metadata = torch::empty(
        {num_sm_parts, static_cast<int64_t>(sizeof(DecodingSchedMeta) / sizeof(int))}, options.dtype(at::kInt));
    num_splits = torch::empty({b + 1}, options.dtype(at::kInt));

    // Length values live on the device and are intentionally not synchronized
    // back to the host. The private scheduler clamps each value to its padded
    // index width, and the producer applies the identical clamp during replay.
    GetDecodeSchedMetaParams sched_params = {};
    sched_params.b = b;
    sched_params.s_q = s_q;
    sched_params.block_size_n = kBlockSizeTopk;
    sched_params.fixed_overhead_num_blocks = kFixedOverheadNumBlocks;
    sched_params.topk = static_cast<int>(indices.size(2));
    sched_params.extra_topk = have_extra ? extra_topk : 0;
    sched_params.topk_length = optional_data_ptr<int>(topk_length);
    sched_params.extra_topk_length = optional_data_ptr<int>(extra_topk_length);
    sched_params.seqlens_k_ptr = nullptr;
    sched_params.tile_scheduler_metadata_ptr =
        reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata->data_ptr<int>());
    sched_params.num_splits_ptr = num_splits->data_ptr<int>();
    sched_params.num_sm_parts = num_sm_parts;
    sched_params.stream = at::cuda::getCurrentCUDAStream().stream();
    sm90::decode::sparse_nvfp4_dsv4::run_get_dsv4_nvfp4_decoding_sched_meta_kernel(sched_params);
  }

  TORCH_CHECK(tile_scheduler_metadata->scalar_type() == at::kInt, "tile_scheduler_metadata must have dtype int32");
  TORCH_CHECK(num_splits->scalar_type() == at::kInt, "num_splits must have dtype int32");
  TORCH_CHECK(tile_scheduler_metadata->is_contiguous(), "tile_scheduler_metadata must be contiguous");
  TORCH_CHECK(num_splits->is_contiguous(), "num_splits must be contiguous");
  TORCH_CHECK(
      tile_scheduler_metadata->dim() == 2 && tile_scheduler_metadata->size(0) == num_sm_parts &&
          tile_scheduler_metadata->size(1) == static_cast<int64_t>(sizeof(DecodingSchedMeta) / sizeof(int)),
      "tile_scheduler_metadata has an invalid shape for this device and query shape");
  TORCH_CHECK(num_splits->dim() == 1 && num_splits->size(0) == b + 1, "num_splits must have shape [B + 1]");

  const int total_num_splits = b + num_sm_parts;
  at::Tensor lse_accum = torch::empty({total_num_splits, s_q, h_q}, options.dtype(at::kFloat));
  at::Tensor o_accum = torch::empty({total_num_splits, s_q, h_q, kHeadDimV}, options.dtype(at::kFloat));

  SparseAttnDecodeParams params = {};
  params.b = b;
  params.s_q = s_q;
  params.h_q = h_q;
  params.h_kv = 1;
  params.d_qk = kHeadDimQk;
  params.d_v = kHeadDimV;
  params.sm_scale = static_cast<float>(sm_scale);
  params.sm_scale_div_log2 = static_cast<float>(sm_scale) * kLog2E;
  params.num_blocks = static_cast<int>(kv.size(0));
  params.page_block_size = static_cast<int>(kv.size(1));
  params.topk = static_cast<int>(indices.size(2));
  params.model_type = ModelType::MODEL1;

  params.q = reinterpret_cast<cutlass::bfloat16_t*>(q.data_ptr());
  params.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv.data_ptr());
  params.indices = indices.data_ptr<int>();
  params.topk_length = optional_data_ptr<int>(topk_length);
  params.attn_sink = optional_data_ptr<float>(attn_sink);
  params.lse = lse.data_ptr<float>();
  params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());

  params.extra_num_blocks = extra_num_blocks;
  params.extra_page_block_size = extra_page_block_size;
  params.extra_topk = extra_topk;
  params.extra_kv = have_extra ? reinterpret_cast<cutlass::bfloat16_t*>(extra_kv->data_ptr()) : nullptr;
  params.extra_indices = have_extra ? extra_indices->data_ptr<int>() : nullptr;
  params.extra_topk_length = optional_data_ptr<int>(extra_topk_length);

  params.stride_q_b = checked_stride(q.stride(0), "q batch");
  params.stride_q_s_q = checked_stride(q.stride(1), "q sequence");
  params.stride_q_h_q = checked_stride(q.stride(2), "q head");
  params.stride_kv_block = checked_stride(kv.stride(0), "kv block");
  params.stride_kv_row = checked_stride(kv.stride(1), "kv row");
  params.stride_indices_b = checked_stride(indices.stride(0), "indices batch");
  params.stride_indices_s_q = checked_stride(indices.stride(1), "indices sequence");
  params.stride_lse_b = checked_stride(lse.stride(0), "lse batch");
  params.stride_lse_s_q = checked_stride(lse.stride(1), "lse sequence");
  params.stride_o_b = checked_stride(out.stride(0), "out batch");
  params.stride_o_s_q = checked_stride(out.stride(1), "out sequence");
  params.stride_o_h_q = checked_stride(out.stride(2), "out head");
  params.stride_extra_kv_block = have_extra ? checked_stride(extra_kv->stride(0), "extra_kv block") : 0;
  params.stride_extra_kv_row = have_extra ? checked_stride(extra_kv->stride(1), "extra_kv row") : 0;
  params.stride_extra_indices_b = have_extra ? checked_stride(extra_indices->stride(0), "extra_indices batch") : 0;
  params.stride_extra_indices_s_q = have_extra ? checked_stride(extra_indices->stride(1), "extra_indices sequence") : 0;
  params.stream = at::cuda::getCurrentCUDAStream().stream();

  params.lse_accum = lse_accum.data_ptr<float>();
  params.o_accum = o_accum.data_ptr<float>();
  params.stride_lse_accum_split = checked_stride(lse_accum.stride(0), "lse_accum split");
  params.stride_lse_accum_s_q = checked_stride(lse_accum.stride(1), "lse_accum sequence");
  params.stride_o_accum_split = checked_stride(o_accum.stride(0), "o_accum split");
  params.stride_o_accum_s_q = checked_stride(o_accum.stride(1), "o_accum sequence");
  params.stride_o_accum_h_q = checked_stride(o_accum.stride(2), "o_accum head");
  params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata->data_ptr<int>());
  params.num_splits_ptr = num_splits->data_ptr<int>();
  params.num_sm_parts = num_sm_parts;

  sm90::decode::sparse_nvfp4_dsv4::run_flash_splitkv_mla_nvfp4_dsv4_sparse_kernel(
      params, kv_global_scale.data_ptr<float>(), have_extra ? extra_kv_global_scale->data_ptr<float>() : nullptr);

  CombineParams combine_params = {};
  combine_params.b = b;
  combine_params.s_q = s_q;
  combine_params.h_q = h_q;
  combine_params.d_v = kHeadDimV;
  combine_params.lse = params.lse;
  combine_params.out = params.out;
  combine_params.stride_lse_b = params.stride_lse_b;
  combine_params.stride_lse_s_q = params.stride_lse_s_q;
  combine_params.stride_o_b = params.stride_o_b;
  combine_params.stride_o_s_q = params.stride_o_s_q;
  combine_params.stride_o_h_q = params.stride_o_h_q;
  combine_params.lse_accum = params.lse_accum;
  combine_params.o_accum = params.o_accum;
  combine_params.stride_lse_accum_split = params.stride_lse_accum_split;
  combine_params.stride_lse_accum_s_q = params.stride_lse_accum_s_q;
  combine_params.stride_o_accum_split = params.stride_o_accum_split;
  combine_params.stride_o_accum_s_q = params.stride_o_accum_s_q;
  combine_params.stride_o_accum_h_q = params.stride_o_accum_h_q;
  combine_params.tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
  combine_params.num_splits_ptr = params.num_splits_ptr;
  combine_params.num_sm_parts = params.num_sm_parts;
  combine_params.attn_sink = params.attn_sink;
  combine_params.stream = params.stream;
  smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);

  return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
