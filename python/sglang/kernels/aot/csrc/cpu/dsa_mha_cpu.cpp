#include "common.h"

// [NOTE]: CPU kernels backing DeepseekSparseAttnBackend's MHA_ONE_SHOT path
// (dsa_backend.py / forward_batch_deepseek_mha_mixin.py).

// Gather per-request KV-cache slot indices into one flat buffer:
//   kv_indices[kv_indptr[i] : kv_indptr[i+1]] =
//       req_to_token[req_pool_indices[i], kv_start : kv_start + page_kernel_lens[i]]
at::Tensor create_flashinfer_kv_indices_cpu(
    const at::Tensor& req_to_token,  // [max_batch, max_context_len]
    const at::Tensor& req_pool_indices,  // [num_seqs], int64
    const at::Tensor& page_kernel_lens,  // [num_seqs], int64
    const at::Tensor& kv_indptr,  // [num_seqs + 1], int32
    const std::optional<at::Tensor>& kv_start_idx  // [num_seqs], int32, or None
) {
  CHECK_INPUT(req_to_token);
  CHECK_DIM(2, req_to_token);
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(page_kernel_lens);
  CHECK_INPUT(kv_indptr);
  CHECK_EQ(req_pool_indices.scalar_type(), at::kLong);
  CHECK_EQ(page_kernel_lens.scalar_type(), at::kLong);
  CHECK_EQ(kv_indptr.scalar_type(), at::kInt);

  const int64_t num_seqs = req_pool_indices.size(0);
  CHECK_EQ(page_kernel_lens.size(0), num_seqs);
  CHECK_EQ(kv_indptr.size(0), num_seqs + 1);

  const bool has_start = kv_start_idx.has_value();
  if (has_start) {
    CHECK_INPUT(kv_start_idx.value());
    CHECK_EQ(kv_start_idx->scalar_type(), at::kInt);
    CHECK_EQ(kv_start_idx->size(0), num_seqs);
  }

  const int64_t stride = req_to_token.size(1);
  const int64_t total = kv_indptr[num_seqs].item<int32_t>();
  at::Tensor kv_indices = at::empty({total}, req_to_token.options());

  const int64_t* req_pool_indices_ptr = req_pool_indices.data_ptr<int64_t>();
  const int64_t* page_kernel_lens_ptr = page_kernel_lens.data_ptr<int64_t>();
  const int32_t* kv_indptr_ptr = kv_indptr.data_ptr<int32_t>();
  const int32_t* kv_start_idx_ptr = has_start ? kv_start_idx->data_ptr<int32_t>() : nullptr;

  AT_DISPATCH_INDEX_TYPES(req_to_token.scalar_type(), "create_flashinfer_kv_indices_cpu", [&] {
    const index_t* req_to_token_ptr = req_to_token.data_ptr<index_t>();
    index_t* kv_indices_ptr = kv_indices.data_ptr<index_t>();

    at::parallel_for(0, num_seqs, 0, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int64_t req_pool_id = req_pool_indices_ptr[i];
        const int64_t kv_start = has_start ? kv_start_idx_ptr[i] : 0;
        const int64_t len = page_kernel_lens_ptr[i];
        const int64_t dst_off = kv_indptr_ptr[i];
        std::memcpy(
            kv_indices_ptr + dst_off, req_to_token_ptr + req_pool_id * stride + kv_start, len * sizeof(index_t));
      }
    });
  });

  return kv_indices;
}

// De-quantize the DSA fp8 latent KV cache (per-token byte layout: 512
// nope-fp8 bytes | 4 x fp32 per-128-group scales (16 bytes) | 64 bf16 rope
// values (128 bytes) = 656 bytes/token), gathered by page index. Mirrors
// dequantize_k_cache_paged (dequant_k_cache.py).
at::Tensor dequantize_k_cache_paged_cpu(
    const at::Tensor& quant_k_cache,  // [total_tokens, dim_quant=656] uint8 byte view
    const at::Tensor& page_table_1_flattened,  // [num_tokens], int32 or int64
    int64_t group_size) {
  CHECK_INPUT(quant_k_cache);
  CHECK_INPUT(page_table_1_flattened);
  CHECK_EQ(quant_k_cache.scalar_type(), at::kByte);

  constexpr int64_t dim_nope = 512;
  constexpr int64_t dim_rope = 64;
  const int64_t dim_quant = quant_k_cache.size(-1);
  TORCH_CHECK(dim_quant == 656, "dequantize_k_cache_paged_cpu: dim_quant must be 656, got ", dim_quant);
  TORCH_CHECK(dim_nope % group_size == 0);
  const int64_t num_tiles = dim_nope / group_size;

  const int64_t num_tokens = page_table_1_flattened.size(0);
  at::Tensor output =
      at::empty({num_tokens, 1, dim_nope + dim_rope}, quant_k_cache.options().dtype(at::kBFloat16));

  const uint8_t* cache_ptr = quant_k_cache.data_ptr<uint8_t>();
  at::BFloat16* out_ptr = output.data_ptr<at::BFloat16>();
  const int64_t out_stride = dim_nope + dim_rope;

  AT_DISPATCH_INDEX_TYPES(page_table_1_flattened.scalar_type(), "dequantize_k_cache_paged_cpu", [&] {
    const index_t* page_ptr = page_table_1_flattened.data_ptr<index_t>();

    at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
      for (int64_t t = begin; t < end; ++t) {
        const int64_t src_row = static_cast<int64_t>(page_ptr[t]);
        const uint8_t* row_ptr = cache_ptr + src_row * dim_quant;
        const c10::Float8_e4m3fn* nope_q = reinterpret_cast<const c10::Float8_e4m3fn*>(row_ptr);
        const float* nope_s = reinterpret_cast<const float*>(row_ptr + dim_nope);
        const at::BFloat16* rope = reinterpret_cast<const at::BFloat16*>(row_ptr + dim_nope + num_tiles * 4);

        at::BFloat16* dst = out_ptr + t * out_stride;
        for (int64_t tile = 0; tile < num_tiles; ++tile) {
          const float scale = nope_s[tile];
          const int64_t base = tile * group_size;
          for (int64_t d = 0; d < group_size; ++d) {
            dst[base + d] = static_cast<at::BFloat16>(static_cast<float>(nope_q[base + d]) * scale);
          }
        }
        for (int64_t d = 0; d < dim_rope; ++d) {
          dst[dim_nope + d] = rope[d];
        }
      }
    });
  });

  return output;
}
