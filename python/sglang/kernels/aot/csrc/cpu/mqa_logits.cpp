#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

// Ragged (non-paged) fp8 MQA logits: q and k are both flat/concatenated across
// all requests in the batch. Query token `i` may only attend to key tokens in
// the half-open range [ks[i], ke[i)) of the shared k buffer (causal, and
// scoped to its own request via the request-local offset baked into ks/ke).
//
// This is a GEMM (Q @ K^T per head, relu, per-head weighted reduce, scale by
// k_scale), computed via convert-to-bf16 + mqa_logits_gemm_kv_range (gemm.cpp)
// - a standalone AMX/vector GEMM kernel (no relation to fused_linear_relu_reduce,
// used by fp8_index.cpp / fp8_paged_mqa_logits_cpu) that skips whole KV tiles
// outside a query row-tile's own [ks, ke) bounding box, with row-tiles aligned
// to cu_seqlens_q request boundaries so a tile can never straddle two
// requests' disjoint k-ranges.
//
// clean_logits=False (the only supported mode): entries outside each row's own
// [ks[i], ke[i)) are left uninitialized; the caller (topk_transform with
// ks=ks) is responsible for only selecting within that range per row.

namespace {

constexpr int64_t kHeadDim = 128;

}  // namespace

// Request-boundary-aligned KV-range-pruned GEMM + relu + per-head weighted
// reduction, defined in gemm.cpp. Standalone - does not share code with
// fused_linear_relu_reduce.
template <bool is_vnni>
void mqa_logits_gemm_kv_range(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale,
    at::Tensor& ks,
    at::Tensor& ke,
    at::Tensor& cu_seqlens_q);

// Fuses fp8_e4m3fn -> bf16 conversion with VNNI packing, defined in gemm.cpp.
template <bool parallelize_internally>
at::Tensor convert_fp8_to_bf16_packed(at::Tensor& weight_fp8);

at::Tensor fp8_mqa_logits_cpu(
    at::Tensor& q_fp8,
    at::Tensor& k_fp8,
    at::Tensor& k_scale,
    at::Tensor& weight,
    at::Tensor& ks,
    at::Tensor& ke,
    at::Tensor& cu_seqlens_q,
    bool clean_logits,
    int64_t max_seqlen_k) {
  // max_seqlen_k: reserved for API parity with the CUDA path, not used yet.
  TORCH_CHECK(!clean_logits, "fp8_mqa_logits_cpu only supports clean_logits == false");
  CHECK_INPUT(q_fp8);
  CHECK_INPUT(k_fp8);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(weight);
  CHECK_INPUT(ks);
  CHECK_INPUT(ke);
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "q_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "k_fp8 must be torch.float8_e4m3fn");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat, "k_scale must be torch.float32");
  TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be torch.float32");

  TORCH_CHECK(q_fp8.dim() == 3, "q_fp8 must have shape [num_q_tokens, num_heads, head_dim]");
  TORCH_CHECK(q_fp8.size(2) == kHeadDim, "q_fp8 head_dim must be 128");
  TORCH_CHECK(k_fp8.dim() == 2 && k_fp8.size(1) == kHeadDim, "k_fp8 must have shape [num_k_tokens, 128]");
  TORCH_CHECK(k_scale.dim() == 1 && k_scale.size(0) == k_fp8.size(0), "k_scale must have shape [num_k_tokens]");

  const int64_t num_q = q_fp8.size(0);
  const int64_t num_heads = q_fp8.size(1);
  const int64_t num_k = k_fp8.size(0);

  TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == num_q && weight.size(1) == num_heads,
      "weight must have shape [num_q_tokens, num_heads]");
  TORCH_CHECK(ks.dim() == 1 && ks.size(0) == num_q, "ks must have shape [num_q_tokens]");
  TORCH_CHECK(ke.sizes() == ks.sizes(), "ke must have the same shape as ks");
  TORCH_CHECK(ks.scalar_type() == at::kInt, "ks must be torch.int32");
  TORCH_CHECK(ke.scalar_type() == at::kInt, "ke must be torch.int32");
  CHECK_INPUT(cu_seqlens_q);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.size(0) >= 1, "cu_seqlens_q must be 1-d and non-empty");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == at::kInt, "cu_seqlens_q must be torch.int32");

  auto logits = at::empty({num_q, num_k}, weight.options().dtype(at::kFloat));
  if (num_q == 0 || num_k == 0) {
    return logits;
  }

  const auto bf16_opts = q_fp8.options().dtype(at::kBFloat16);

  auto q_bf16 = at::empty({num_q, num_heads, kHeadDim}, bf16_opts);
  fp8_to_bf16(
      q_bf16.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint8_t*>(q_fp8.const_data_ptr()),
      num_q * num_heads * kHeadDim);

  auto k_packed = convert_fp8_to_bf16_packed<true>(k_fp8);

  mqa_logits_gemm_kv_range<true>(logits, q_bf16, weight, k_packed, k_scale, ks, ke, cu_seqlens_q);

  return logits;
}
