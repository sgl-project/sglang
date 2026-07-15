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

#include <cmath>
#include <cstdint>

#include "flashmla/sm90/sparse_nvfp4/layout.h"
#include "flashmla/sm90/sparse_nvfp4/legacy_params.h"
#include "flashmla/sm90/sparse_nvfp4/splitkv_mla.h"
#include "params.h"
#include "smxx/decode/combine/combine.h"

namespace {

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

void check_same_device(const at::Tensor& reference, const at::Tensor& tensor, const char* tensor_name) {
  TORCH_CHECK(tensor.device() == reference.device(), tensor_name, " must be on the same CUDA device as q");
}

}  // namespace

namespace {

std::vector<at::Tensor> fwd_kvcache_mla_nvfp4_impl(
    at::Tensor& q,
    const at::Tensor& kcache,
    const at::Tensor& kv_global_scale,
    const int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const double softmax_scale,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const at::Tensor& indices,
    const bool enable_stage_timing) {
  // This entry point is deliberately separate from fwd_kvcache_mla.  Its
  // cache ABI is fixed and cannot be confused with FlashMLA's FP8 layout.
  CHECK_CUDA_TENSOR(q);
  CHECK_CUDA_TENSOR(kcache);
  CHECK_CUDA_TENSOR(kv_global_scale);
  CHECK_CUDA_TENSOR(seqlens_k);
  CHECK_CUDA_TENSOR(tile_scheduler_metadata);
  CHECK_CUDA_TENSOR(num_splits);
  CHECK_CUDA_TENSOR(indices);

  check_same_device(q, kcache, "kcache");
  check_same_device(q, kv_global_scale, "kv_global_scale");
  check_same_device(q, seqlens_k, "seqlens_k");
  check_same_device(q, tile_scheduler_metadata, "tile_scheduler_metadata");
  check_same_device(q, num_splits, "num_splits");
  check_same_device(q, indices, "indices");

  at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
  const auto* dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major == 9 && dprops->minor == 0, "fwd_kvcache_mla_nvfp4 only supports SM90");

  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must have dtype bfloat16");
  TORCH_CHECK(kcache.scalar_type() == at::kByte, "kcache must have dtype uint8");
  TORCH_CHECK(kv_global_scale.scalar_type() == at::kFloat, "kv_global_scale must have dtype float32");
  TORCH_CHECK(seqlens_k.scalar_type() == at::kInt, "seqlens_k must have dtype int32");
  TORCH_CHECK(tile_scheduler_metadata.scalar_type() == at::kInt, "tile_scheduler_metadata must have dtype int32");
  TORCH_CHECK(num_splits.scalar_type() == at::kInt, "num_splits must have dtype int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must have dtype int32");

  CHECK_CONTIGUOUS_TENSOR(q);
  CHECK_CONTIGUOUS_TENSOR(kcache);
  CHECK_CONTIGUOUS_TENSOR(kv_global_scale);
  CHECK_CONTIGUOUS_TENSOR(seqlens_k);
  CHECK_CONTIGUOUS_TENSOR(tile_scheduler_metadata);
  CHECK_CONTIGUOUS_TENSOR(num_splits);
  CHECK_CONTIGUOUS_TENSOR(indices);

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, Sq, H, 576]");
  const int batch_size = static_cast<int>(q.size(0));
  const int seqlen_q = static_cast<int>(q.size(1));
  const int num_heads_q = static_cast<int>(q.size(2));
  TORCH_CHECK(q.size(3) == 576, "q must have head dimension 576");
  TORCH_CHECK(batch_size > 0, "q batch size must be positive");
  TORCH_CHECK(seqlen_q > 0, "q sequence length must be positive");
  TORCH_CHECK(num_heads_q > 0, "q must contain at least one head");
  TORCH_CHECK(head_size_v == 512, "head_size_v must be 512");
  TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0, "softmax_scale must be finite and positive");

  TORCH_CHECK(kcache.dim() == 4, "kcache must have shape [num_pages, 64, 1, 416]");
  TORCH_CHECK(kcache.size(0) > 0, "kcache must contain at least one page");
  TORCH_CHECK(kcache.size(1) == 64, "kcache page size must be 64");
  TORCH_CHECK(kcache.size(2) == 1, "kcache must contain exactly one KV head");
  TORCH_CHECK(kcache.size(3) == sm90::nvfp4::kBytesPerToken, "kcache rows must use the 416-byte NVFP4 layout");
  TORCH_CHECK(
      kcache.stride(0) == 64 * sm90::nvfp4::kBytesPerToken && kcache.stride(1) == sm90::nvfp4::kBytesPerToken &&
          kcache.stride(2) == sm90::nvfp4::kBytesPerToken && kcache.stride(3) == 1,
      "kcache must be tightly packed as [num_pages, 64, 1, 416]");

  TORCH_CHECK(kv_global_scale.numel() == 1, "kv_global_scale must be a device float32 scalar");
  TORCH_CHECK(
      kv_global_scale.dim() == 0 || (kv_global_scale.dim() == 1 && kv_global_scale.size(0) == 1),
      "kv_global_scale must be a scalar or one-element tensor");

  TORCH_CHECK(seqlens_k.dim() == 1 && seqlens_k.size(0) == batch_size, "seqlens_k must have shape [B]");
  TORCH_CHECK(
      tile_scheduler_metadata.dim() == 2 && tile_scheduler_metadata.size(0) > 0 &&
          tile_scheduler_metadata.size(1) == sm90::nvfp4_legacy::kTileSchedulerMetadataSize,
      "tile_scheduler_metadata must have shape [num_sm_parts, 8]");
  TORCH_CHECK(num_splits.dim() == 1 && num_splits.size(0) == batch_size + 1, "num_splits must have shape [B + 1]");
  TORCH_CHECK(
      indices.dim() == 3 && indices.size(0) == batch_size && indices.size(1) == seqlen_q,
      "indices must have shape [B, Sq, topk]");
  const int topk = static_cast<int>(indices.size(2));
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk % 64 == 0, "topk must be a multiple of 64");

  // h_k is fixed to one by the latent-cache ABI.  Fold q heads into the row
  // dimension exactly as the pinned FlashMLA sparse implementation does.
  constexpr int num_heads_k = 1;
  const int q_seq_per_hk = seqlen_q * num_heads_q;
  at::Tensor q_kernel = q.view({batch_size, seqlen_q, num_heads_k, num_heads_q, 576})
                            .transpose(2, 3)
                            .reshape({batch_size, q_seq_per_hk, num_heads_k, 576});

  const auto options = q.options();
  at::Tensor out = torch::empty({batch_size, q_seq_per_hk, num_heads_k, head_size_v}, options);
  at::Tensor softmax_lse = torch::empty({batch_size, num_heads_k, q_seq_per_hk}, options.dtype(at::kFloat));

  sm90::nvfp4_legacy::DecodingParams params = {};
  params.b = batch_size;
  params.s_q = seqlen_q;
  params.q_seq_per_hk = q_seq_per_hk;
  params.d = 576;
  params.d_v = static_cast<int>(head_size_v);
  params.h_q = num_heads_q;
  params.h_k = num_heads_k;
  params.num_blocks = static_cast<int>(kcache.size(0));
  params.q_head_per_hk = num_heads_q;
  params.is_causal = false;
  params.scale_softmax = static_cast<float>(softmax_scale);
  params.scale_softmax_log2 = static_cast<float>(softmax_scale * M_LOG2E);
  params.topk = topk;

  params.q_ptr = q_kernel.data_ptr();
  params.k_ptr = kcache.data_ptr();
  params.o_ptr = out.data_ptr();
  params.softmax_lse_ptr = softmax_lse.data_ptr();
  params.indices_ptr = indices.data_ptr<int>();
  params.seqlens_k_ptr = seqlens_k.data_ptr<int>();

  params.q_batch_stride = q_kernel.stride(0);
  params.q_row_stride = q_kernel.stride(1);
  params.q_head_stride = q_kernel.stride(2);
  params.k_batch_stride = kcache.stride(0);
  params.k_row_stride = kcache.stride(1);
  params.k_head_stride = kcache.stride(2);
  params.o_batch_stride = out.stride(0);
  params.o_row_stride = out.stride(1);
  params.o_head_stride = out.stride(2);
  params.indices_batch_stride = indices.stride(0);
  params.indices_row_stride = indices.stride(1);

  params.block_table = nullptr;
  params.block_table_batch_stride = 0;
  params.page_block_size = 64;
  params.tile_scheduler_metadata_ptr = tile_scheduler_metadata.data_ptr<int>();
  params.num_sm_parts = static_cast<int>(tile_scheduler_metadata.size(0));
  params.num_splits_ptr = num_splits.data_ptr<int>();

  const int total_num_splits = batch_size + params.num_sm_parts;
  at::Tensor softmax_lse_accum = torch::empty({total_num_splits, num_heads_k, q_seq_per_hk}, options.dtype(at::kFloat));
  at::Tensor out_accum =
      torch::empty({total_num_splits, num_heads_k, q_seq_per_hk, head_size_v}, options.dtype(at::kFloat));
  params.total_num_splits = total_num_splits;
  params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
  params.oaccum_ptr = out_accum.data_ptr();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  at::Tensor stage_timing;
  if (enable_stage_timing) {
    const int num_ctas = sm90::get_flash_splitkv_mla_nvfp4_stage_timing_num_ctas(params);
    stage_timing = torch::zeros(
        {num_ctas, sm90::kStageTimingRecordsPerCta, sm90::kStageTimingMetricsPerRecord}, options.dtype(at::kLong));
    sm90::run_flash_splitkv_mla_nvfp4_sparse_profile_kernel(
        params,
        kv_global_scale.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(stage_timing.data_ptr<int64_t>()),
        stream);
  } else {
    sm90::run_flash_splitkv_mla_nvfp4_sparse_kernel(params, kv_global_scale.data_ptr<float>(), stream);
  }
#else
  TORCH_CHECK(!enable_stage_timing, "NVFP4 stage timing was not enabled in this flashmla_ops build");
  sm90::run_flash_splitkv_mla_nvfp4_sparse_kernel(params, kv_global_scale.data_ptr<float>(), stream);
#endif
  CombineParams combine_params = {
      batch_size,
      seqlen_q,
      num_heads_q,
      static_cast<int>(head_size_v),

      static_cast<float*>(params.softmax_lse_ptr),
      params.o_ptr,
      seqlen_q * num_heads_q,
      num_heads_q,
      seqlen_q * num_heads_q * static_cast<int>(head_size_v),
      num_heads_q * static_cast<int>(head_size_v),
      static_cast<int>(head_size_v),

      static_cast<float*>(params.softmax_lseaccum_ptr),
      static_cast<float*>(params.oaccum_ptr),
      seqlen_q * num_heads_q,
      num_heads_q,
      seqlen_q * num_heads_q * static_cast<int>(head_size_v),
      num_heads_q * static_cast<int>(head_size_v),
      static_cast<int>(head_size_v),

      reinterpret_cast<DecodingSchedMeta*>(params.tile_scheduler_metadata_ptr),
      params.num_splits_ptr,
      params.num_sm_parts,

      nullptr,
      stream,
  };
  smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(combine_params);

  out = out.view({batch_size, seqlen_q, num_heads_q, num_heads_k, head_size_v})
            .transpose(2, 3)
            .reshape({batch_size, seqlen_q, num_heads_q, head_size_v});
  softmax_lse = softmax_lse.view({batch_size, num_heads_k, seqlen_q, num_heads_q})
                    .transpose(2, 3)
                    .reshape({batch_size, num_heads_q, seqlen_q});
#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)
  if (enable_stage_timing) {
    return {out, softmax_lse, stage_timing};
  }
#endif
  return {out, softmax_lse};
}

}  // namespace

std::vector<at::Tensor> fwd_kvcache_mla_nvfp4(
    at::Tensor& q,
    const at::Tensor& kcache,
    const at::Tensor& kv_global_scale,
    const int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const double softmax_scale,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const at::Tensor& indices) {
  return fwd_kvcache_mla_nvfp4_impl(
      q,
      kcache,
      kv_global_scale,
      head_size_v,
      seqlens_k,
      softmax_scale,
      tile_scheduler_metadata,
      num_splits,
      indices,
      false);
}

#if defined(SGLANG_FLASHMLA_NVFP4_STAGE_TIMING)

std::vector<at::Tensor> fwd_kvcache_mla_nvfp4_stage_timing(
    at::Tensor& q,
    const at::Tensor& kcache,
    const at::Tensor& kv_global_scale,
    const int64_t head_size_v,
    const at::Tensor& seqlens_k,
    const double softmax_scale,
    const at::Tensor& tile_scheduler_metadata,
    const at::Tensor& num_splits,
    const at::Tensor& indices) {
  return fwd_kvcache_mla_nvfp4_impl(
      q,
      kcache,
      kv_global_scale,
      head_size_v,
      seqlens_k,
      softmax_scale,
      tile_scheduler_metadata,
      num_splits,
      indices,
      true);
}

#endif
