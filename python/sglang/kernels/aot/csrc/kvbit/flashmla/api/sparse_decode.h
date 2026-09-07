#pragma once

#include <cmath>

#include "common.h"
#include "packed_layout.h"
#include "params.h"
#include "sm90/decode/sparse_fp8/combine.h"
#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
  HEAD_64,
  HEAD_DIM_512,
  MODEL1_KVCACHE_FORMAT,

  ATTN_SINK,
  TOPK_LENGTH,
  EXTRA_KVCACHE,
  EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
  int num_sm_parts;
  int fixed_overhead_num_blocks;
  int block_size_topk;
};

class DecodeImplBase : public ImplBase<SparseAttnDecodeParams, DecodeFeatures> {
 public:
  virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class Decode_Int4_Sm90_Impl : public DecodeImplBase {
  DECLARE_SUPPORTED_FEATURES(
      DecodeFeatures::HEAD_64,
      DecodeFeatures::HEAD_DIM_512,
      DecodeFeatures::MODEL1_KVCACHE_FORMAT,
      DecodeFeatures::ATTN_SINK,
      DecodeFeatures::TOPK_LENGTH,
      DecodeFeatures::EXTRA_KVCACHE,
      DecodeFeatures::EXTRA_TOPK_LENGTH)

 public:
  DecodeImplMeta get_meta(int h_q, int s_q) override {
    Arch arch = Arch();
    return {std::max(arch.num_sms / s_q / (h_q / 64), 1), 5, 64};
  }

 protected:
  void run_(const SparseAttnDecodeParams& params, const std::vector<FeatureT>& required_features) override {
    sm90::decode::sparse_fp8::run_flash_splitkv_mla_int4_sparse_kernel<ModelType::MODEL1, 64>(params);
  }
};

inline void
sparse_validate_int4_buffer(const at::Tensor& packed_kcache, int64_t kv_num_rows, int row_bytes, const char* name) {
  KU_CHECK_DEVICE(packed_kcache);
  TORCH_CHECK(packed_kcache.dtype() == at::kByte, name, " must be uint8");
  TORCH_CHECK(packed_kcache.dim() == 2, name, " must be rank-2 [num_rows, row_bytes]");
  TORCH_CHECK(packed_kcache.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(packed_kcache.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
  TORCH_CHECK(
      packed_kcache.size(0) == kv_num_rows,
      name,
      " row count ",
      packed_kcache.size(0),
      " must equal KV num_rows ",
      kv_num_rows);
  TORCH_CHECK(
      packed_kcache.size(1) == row_bytes,
      name,
      " must match the shape-carrier row stride, got ",
      packed_kcache.size(1));
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor& q,                                          // [b, s_q, h_q, d_qk]
    const at::Tensor& kv,                                         // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor& indices,                                    // [b, s_q, topk]
    const std::optional<at::Tensor>& topk_length,                 // [b], shared across query positions
    const std::optional<at::Tensor>& attn_sink,                   // [h_q]
    const std::optional<at::Tensor>& tile_scheduler_metadata_in,  // num_sm_parts x (DecodingSchedMetaSize/4)
    const std::optional<at::Tensor>& num_splits_in,               // batch_size + 1
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_v,
    float sm_scale,
    const at::Tensor& packed_kcache,
    const std::optional<at::Tensor>& extra_packed_kcache = std::nullopt) {
  using bf16 = cutlass::bfloat16_t;

  // Return newly allocated metadata without mutating optional inputs.
  std::optional<at::Tensor> tile_scheduler_metadata = tile_scheduler_metadata_in;
  std::optional<at::Tensor> num_splits = num_splits_in;

  KU_CHECK_NDIM(q, 4);
  KU_CHECK_NDIM(kv, 4);
  KU_CHECK_NDIM(indices, 3);
  KU_CHECK_NDIM(extra_kv, 4);
  KU_CHECK_NDIM(extra_indices, 3);
  TORCH_CHECK(
      tile_scheduler_metadata.has_value() == num_splits.has_value(),
      "tile_scheduler_metadata and num_splits must be supplied together");

  int b = q.size(0);
  int s_q = q.size(1);
  int h_q = q.size(2);
  int d_qk = q.size(3);
  int num_blocks = kv.size(0);
  int page_block_size = kv.size(1);
  int h_kv = kv.size(2);
  int topk = indices.size(2);
  const int row_bytes = kv.size(3);
  TORCH_CHECK(
      row_bytes == kvbit::dsv4::COMPACT_ROW_BYTES || row_bytes == kvbit::dsv4::ALIGNED_ROW_BYTES,
      "INT4 row width must be 368 or 384, got ",
      row_bytes);
  TORCH_CHECK(
      page_block_size == 2 || page_block_size == 64 || page_block_size == 256, "INT4 page size must be 2, 64 or 256");

  bool have_topk_length = topk_length.has_value();
  bool have_extra_kcache = extra_kv.has_value();
  bool have_extra_topk_length = extra_topk_length.has_value();
  bool have_attn_sink = attn_sink.has_value();

  int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
  if (have_extra_kcache) {
    extra_num_blocks = extra_kv->size(0);
    extra_page_block_size = extra_kv->size(1);
  }
  if (extra_indices.has_value()) {
    extra_topk = extra_indices->size(-1);
  }

  // metadata sanity check
  TORCH_CHECK(b > 0);
  TORCH_CHECK(s_q > 0);
  TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0, "INT4 softmax scale must be finite and positive");
  TORCH_CHECK(h_q == 64, "KVBit sparse decode supports exactly 64 query heads, got ", h_q);
  TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
  TORCH_CHECK(
      d_qk == 512,
      "KVBit sparse decode supports only MODEL1 head_size_k=512 "
      "(448 NoPE + 64 RoPE), got ",
      d_qk);
  TORCH_CHECK(d_v == 512, "KVBit sparse decode supports only head_size_v=512");
  TORCH_CHECK(topk > 0 && topk % 64 == 0, "INT4 sparse width must be positive and divisible by 64");

  if (have_extra_kcache) {
    TORCH_CHECK(
        extra_indices.has_value(),
        "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    TORCH_CHECK(extra_topk > 0 && extra_topk % 64 == 0, "INT4 extra sparse width must be positive and divisible by 64");
    TORCH_CHECK(
        extra_page_block_size == 2 || extra_page_block_size == 64 || extra_page_block_size == 256,
        "INT4 extra page size must be 2, 64 or 256");
  } else {
    TORCH_CHECK(
        !extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
    TORCH_CHECK(
        !extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
  }

  // Check device
  KU_CHECK_DEVICE(q);
  KU_CHECK_DEVICE(kv);
  KU_CHECK_DEVICE(indices);
  KU_CHECK_DEVICE(topk_length);
  KU_CHECK_DEVICE(attn_sink);
  KU_CHECK_DEVICE(tile_scheduler_metadata);
  KU_CHECK_DEVICE(num_splits);
  KU_CHECK_DEVICE(extra_kv);
  KU_CHECK_DEVICE(extra_indices);
  KU_CHECK_DEVICE(extra_topk_length);
  KU_CHECK_DEVICE(packed_kcache);
  KU_CHECK_DEVICE(extra_packed_kcache);

  const auto q_device = q.device();
  auto check_same_device = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(
        tensor.device() == q_device, name, " must be on the same device as q (", q_device, "), got ", tensor.device());
  };
  check_same_device(kv, "kv");
  check_same_device(indices, "indices");
  if (topk_length.has_value()) check_same_device(*topk_length, "topk_length");
  if (attn_sink.has_value()) check_same_device(*attn_sink, "attn_sink");
  if (tile_scheduler_metadata.has_value()) check_same_device(*tile_scheduler_metadata, "tile_scheduler_metadata");
  if (num_splits.has_value()) check_same_device(*num_splits, "num_splits");
  if (extra_kv.has_value()) check_same_device(*extra_kv, "extra_kv");
  if (extra_indices.has_value()) check_same_device(*extra_indices, "extra_indices");
  if (extra_topk_length.has_value()) check_same_device(*extra_topk_length, "extra_topk_length");
  check_same_device(packed_kcache, "packed_kcache");
  if (extra_packed_kcache.has_value()) check_same_device(*extra_packed_kcache, "extra_packed_kcache");

  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  Arch arch = Arch();
  TORCH_CHECK(arch.is_sm90a(), "KVBit sparse decode supports only SM90");

  // Check data type
  KU_CHECK_DTYPE(q, torch::kBFloat16);
  TORCH_CHECK(
      kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8,
      "key must have dtype fp8_e4m3fn, int8 or uint8");
  if (extra_kv.has_value()) {
    TORCH_CHECK(
        extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 ||
            extra_kv->dtype() == torch::kUInt8,
        "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
  }
  KU_CHECK_DTYPE(indices, torch::kInt32);
  KU_CHECK_DTYPE(topk_length, torch::kInt32);
  KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
  KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
  KU_CHECK_DTYPE(num_splits, torch::kInt32);
  KU_CHECK_DTYPE(extra_indices, torch::kInt32);
  KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

  // Check layout
  KU_CHECK_LAST_DIM_CONTIGUOUS(q);
  KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
  KU_CHECK_CONTIGUOUS(topk_length);
  KU_CHECK_CONTIGUOUS(attn_sink);

  KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
  KU_CHECK_CONTIGUOUS(num_splits);

  KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
  KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
  KU_CHECK_CONTIGUOUS(extra_topk_length);

  // Check shape
  KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
  {
    const int bytes_per_token = row_bytes;
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
    if (extra_kv.has_value()) {
      const int extra_bpt = static_cast<int>(extra_kv->size(3));
      TORCH_CHECK(extra_bpt == bytes_per_token, "INT4 sources must use the same physical row layout");
      KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, extra_bpt);
      TORCH_CHECK(
          extra_kv->stride(1) == extra_bpt,
          "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
    }
    TORCH_CHECK(
        kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for kv cache");
  }
  KU_CHECK_SHAPE(indices, b, s_q, topk);
  KU_CHECK_SHAPE(topk_length, b);
  KU_CHECK_SHAPE(attn_sink, h_q);
  KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
  KU_CHECK_SHAPE(extra_topk_length, b);
  sparse_validate_int4_buffer(
      packed_kcache, static_cast<int64_t>(num_blocks) * page_block_size, row_bytes, "packed_kcache");
  TORCH_CHECK(
      have_extra_kcache == extra_packed_kcache.has_value(),
      "extra_kv and extra_packed_kcache must be supplied together");
  if (have_extra_kcache) {
    sparse_validate_int4_buffer(
        *extra_packed_kcache,
        static_cast<int64_t>(extra_num_blocks) * extra_page_block_size,
        row_bytes,
        "extra_packed_kcache");
  }

  auto opts = q.options();

  at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts);
  at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

  constexpr ModelType model_type = ModelType::MODEL1;

  std::vector<DecodeFeatures> features;
  features.push_back(DecodeFeatures::HEAD_64);
  features.push_back(DecodeFeatures::HEAD_DIM_512);
  features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
  if (have_attn_sink) {
    features.push_back(DecodeFeatures::ATTN_SINK);
  }
  if (have_topk_length) {
    features.push_back(DecodeFeatures::TOPK_LENGTH);
  }
  if (have_extra_kcache) {
    features.push_back(DecodeFeatures::EXTRA_KVCACHE);
  }
  if (have_extra_topk_length) {
    features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
  }

  Decode_Int4_Sm90_Impl impl;

  DecodeImplMeta impl_meta = impl.get_meta(h_q, s_q);

  SparseAttnDecodeParams params = {
      b,
      s_q,
      h_q,
      h_kv,
      d_qk,
      d_v,
      sm_scale,
      sm_scale * LOG_2_E,
      num_blocks,
      page_block_size,
      topk,
      model_type,

      (bf16*)q.data_ptr(),
      nullptr,
      (bf16*)kv.data_ptr(),
      (int*)indices.data_ptr(),
      ku::get_optional_tensor_ptr<int>(topk_length),
      ku::get_optional_tensor_ptr<float>(attn_sink),
      (float*)lse.data_ptr(),
      (bf16*)out.data_ptr(),

      extra_num_blocks,
      extra_page_block_size,
      extra_topk,
      ku::get_optional_tensor_ptr<bf16>(extra_kv),
      ku::get_optional_tensor_ptr<int>(extra_indices),
      ku::get_optional_tensor_ptr<int>(extra_topk_length),

      int64_stride_to_int(q.stride(0)),
      int64_stride_to_int(q.stride(1)),
      int64_stride_to_int(q.stride(2)),
      0,
      0,
      0,
      int64_stride_to_int(kv.stride(0)),
      int64_stride_to_int(kv.stride(1)),
      int64_stride_to_int(indices.stride(0)),
      int64_stride_to_int(indices.stride(1)),
      int64_stride_to_int(lse.stride(0)),
      int64_stride_to_int(lse.stride(1)),
      int64_stride_to_int(out.stride(0)),
      int64_stride_to_int(out.stride(1)),
      int64_stride_to_int(out.stride(2)),

      have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
      have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
      at::cuda::getCurrentCUDAStream().stream()};

  // Get MLA metadata if necessary
  at::Tensor o_accum, lse_accum;
  if (!tile_scheduler_metadata.has_value()) {
    tile_scheduler_metadata =
        torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta) / 4}, opts.dtype(torch::kInt32));
    num_splits = torch::empty({b + 1}, opts.dtype(torch::kInt32));
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    GetDecodeSchedMetaParams get_sched_meta_params = {
        b,
        s_q,
        impl_meta.block_size_topk,
        impl_meta.fixed_overhead_num_blocks,
        topk,
        extra_topk,
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),
        nullptr,
        (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
        num_splits->data_ptr<int>(),
        impl_meta.num_sm_parts,
        at::cuda::getCurrentCUDAStream().stream()};
    smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
  }
  // Stick the metadata pointers to `params`
  KU_CHECK_DEVICE(tile_scheduler_metadata);
  KU_CHECK_DEVICE(num_splits);
  KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
  KU_CHECK_DTYPE(num_splits, torch::kInt32);
  KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
  KU_CHECK_CONTIGUOUS(num_splits);
  KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int));
  KU_CHECK_SHAPE(num_splits, b + 1);
  params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
  params.num_splits_ptr = num_splits->data_ptr<int>();
  params.num_sm_parts = impl_meta.num_sm_parts;

  // Allocate intermediate buffers for split-KV
  const int total_num_splits = b + impl_meta.num_sm_parts;
  lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
  o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
  KU_CHECK_CONTIGUOUS(lse_accum);
  KU_CHECK_CONTIGUOUS(o_accum);
  params.lse_accum = lse_accum.data_ptr<float>();
  params.o_accum = o_accum.data_ptr<float>();
  params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
  params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
  params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
  params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
  params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

  // Physical stride varies; payload offsets and quantization do not.
  {
    TORCH_CHECK(
        !have_extra_kcache || extra_packed_kcache.has_value(),
        "KVBit INT4 sparse decode requires extra_packed_kcache when extra_kv is present");

    params.packed_kcache_ptr = packed_kcache.data_ptr();
    params.packed_row_bytes = row_bytes;

    if (extra_packed_kcache.has_value()) {
      TORCH_CHECK(extra_kv.has_value(), "extra_packed_kcache requires extra_kv");
      const at::Tensor& epk = extra_packed_kcache.value();
      params.extra_packed_kcache_ptr = epk.data_ptr();
    }
  }

  impl.run(params, features);

  CombineParams combine_params = {
      b,
      s_q,
      h_q,
      d_v,

      params.lse,
      params.out,
      params.stride_lse_b,
      params.stride_lse_s_q,
      params.stride_o_b,
      params.stride_o_s_q,
      params.stride_o_h_q,

      params.lse_accum,
      params.o_accum,
      params.stride_lse_accum_split,
      params.stride_lse_accum_s_q,
      params.stride_o_accum_split,
      params.stride_o_accum_s_q,
      params.stride_o_accum_h_q,

      params.tile_scheduler_metadata_ptr,
      params.num_splits_ptr,
      params.num_sm_parts,

      ku::get_optional_tensor_ptr<float>(attn_sink),
      at::cuda::getCurrentCUDAStream().stream()};
  kvbit::dsv4::run_int4_combine(combine_params);

  return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
