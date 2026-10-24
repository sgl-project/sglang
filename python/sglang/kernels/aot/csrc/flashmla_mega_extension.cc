#include <torch/all.h>
#include <torch/library.h>

#include "api/fused_norm_rope_attn_rope_cast_fwd.cpp"

static std::vector<at::Tensor> sgl_flash_mla_mega_prefill(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    double sm_scale,
    int64_t n_wv_group,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length) {
  TORCH_CHECK(n_wv_group > 0 && q.size(1) == n_wv_group * 8, "MegaAttention requires eight heads per WO-A group");
  return fused_norm_rope_attn_rope_cast_fwd(
      q,
      kv,
      indices,
      static_cast<float>(sm_scale),
      512,
      attn_sink,
      topk_length,
      false,
      0.0f,
      positions,
      false,
      64,
      cos_sin_cache,
      static_cast<uint32_t>(n_wv_group),
      32,
      true,
      true,
      true);
}

static std::vector<at::Tensor> sgl_flash_mla_mega_decode(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    double sm_scale,
    int64_t n_wv_group,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length) {
  TORCH_CHECK(n_wv_group > 0 && q.size(1) == n_wv_group * 8, "MegaAttention requires eight heads per WO-A group");
  return fused_norm_rope_attn_rope_cast_decode(
      q,
      kv,
      indices,
      static_cast<float>(sm_scale),
      512,
      attn_sink,
      topk_length,
      extra_kv,
      extra_indices,
      extra_topk_length,
      false,
      0.0f,
      positions,
      false,
      64,
      cos_sin_cache,
      static_cast<uint32_t>(n_wv_group),
      32,
      true,
      true,
      true);
}

static std::vector<at::Tensor>
sgl_flash_mla_mega_permute_q(const at::Tensor& weight, const at::Tensor& scale, int64_t heads) {
  TORCH_CHECK(heads == 64 || heads == 128, "MegaAttention requires 64 or 128 query heads");
  return permute_q_b_proj(weight, scale, static_cast<int>(heads), 512);
}

static std::vector<at::Tensor> sgl_flash_mla_mega_permute_wo(const at::Tensor& weight, const at::Tensor& scale) {
  return permute_wv_proj(weight, scale, 8, 512);
}

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "flash_mla_mega_prefill(Tensor q, Tensor kv, Tensor indices, Tensor positions, Tensor cos_sin_cache, "
      "float sm_scale, int n_wv_group, Tensor? attn_sink=None, Tensor? topk_length=None) -> Tensor[]");
  m.impl("flash_mla_mega_prefill", torch::kCUDA, &sgl_flash_mla_mega_prefill);
  m.def(
      "flash_mla_mega_decode(Tensor q, Tensor kv, Tensor indices, Tensor positions, Tensor cos_sin_cache, "
      "float sm_scale, int n_wv_group, Tensor? attn_sink=None, Tensor? topk_length=None, "
      "Tensor? extra_kv=None, Tensor? extra_indices=None, Tensor? extra_topk_length=None) -> Tensor[]");
  m.impl("flash_mla_mega_decode", torch::kCUDA, &sgl_flash_mla_mega_decode);
  m.def("flash_mla_mega_permute_q(Tensor weight, Tensor scale, int heads) -> Tensor[]");
  m.impl("flash_mla_mega_permute_q", torch::kCUDA, &sgl_flash_mla_mega_permute_q);
  m.def("flash_mla_mega_permute_wo(Tensor weight, Tensor scale) -> Tensor[]");
  m.impl("flash_mla_mega_permute_wo", torch::kCUDA, &sgl_flash_mla_mega_permute_wo);
}
