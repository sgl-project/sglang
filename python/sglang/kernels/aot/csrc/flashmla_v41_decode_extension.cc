#include <torch/all.h>
#include <torch/library.h>

#include "api/sparse_decode.cpp"

namespace {

std::tuple<at::Tensor, at::Tensor> v41_sparse_decode_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int64_t d_v,
    double sm_scale) {
  std::optional<at::Tensor> tile_scheduler_metadata, num_splits;
  auto [out, lse, meta, splits] = sparse_decode_fwd(
      q,
      kv,
      indices,
      topk_length,
      attn_sink,
      tile_scheduler_metadata,
      num_splits,
      extra_kv,
      extra_indices,
      extra_topk_length,
      static_cast<int>(d_v),
      static_cast<float>(sm_scale),
      false);
  return {out, lse};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_flashmla_v41, m) {
  m.def(
      "sparse_decode_fwd(Tensor q, Tensor kv, Tensor indices, Tensor? topk_length, Tensor? attn_sink, "
      "Tensor? extra_kv, Tensor? extra_indices, Tensor? extra_topk_length, int d_v, float sm_scale) "
      "-> (Tensor, Tensor)");
  m.impl("sparse_decode_fwd", torch::kCUDA, &v41_sparse_decode_fwd);
}
