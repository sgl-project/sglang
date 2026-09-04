#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"
#include "vec.h"

// DSA indexer FP8 index score (ragged, single-batch loop path).
//
// Mirrors the semantics of the tilelang/CUDA ``fp8_index`` kernel and the
// PyTorch reference in dsa/cpu_kernel.py:
//   1) fp8 q @ fp8 k -> fp32 logits          (per head)
//   2) relu(logits) * q_s (head gate)        (per head)
//   3) sum over heads -> logits_sum
//   4) logits_sum * k_s (per-token scale)    -> index_score
//
// Shapes (all contiguous):
//   q   : [B, M, H, D]  float8_e4m3fn
//   q_s : [B, M, H]     float32
//   k   : [B, N, D]     float8_e4m3fn
//   k_s : [B, N]        float32
//   out : [B, M, N]     float32

// Fused GEMM + relu + per-head weighted reduction, defined in gemm.cpp.
template <bool parallelize_internally, bool is_vnni>
void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale);

// Fuses fp8_e4m3fn -> bf16 conversion with VNNI packing, defined in gemm.cpp.
template <bool parallelize_internally>
at::Tensor convert_fp8_to_bf16_packed(at::Tensor& weight_fp8);

at::Tensor fp8_index_cpu(at::Tensor& q, at::Tensor& q_s, at::Tensor& k, at::Tensor& k_s) {
  CHECK_INPUT(q);
  CHECK_INPUT(q_s);
  CHECK_INPUT(k);
  CHECK_INPUT(k_s);

  TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn, "q must be torch.float8_e4m3fn");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn, "k must be torch.float8_e4m3fn");
  TORCH_CHECK(q_s.scalar_type() == at::kFloat, "q_s must be torch.float32");
  TORCH_CHECK(k_s.scalar_type() == at::kFloat, "k_s must be torch.float32");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B, M, H, D]");
  TORCH_CHECK(k.dim() == 3, "k must have shape [B, N, D]");
  TORCH_CHECK(q_s.dim() == 3, "q_s must have shape [B, M, H]");
  TORCH_CHECK(k_s.dim() == 2, "k_s must have shape [B, N]");

  const int64_t B = q.size(0);
  // Caller always passes B=1 (one batch item per forward_indexer iteration).
  TORCH_CHECK(B == 1, "fp8_index_cpu only supports B=1, got B=", B);

  const int64_t M = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t N = k.size(1);

  TORCH_CHECK(k.size(0) == 1 && k.size(2) == D, "k must have shape [1, N, D] matching q");
  TORCH_CHECK(
      q_s.size(0) == 1 && q_s.size(1) == M && q_s.size(2) == H,
      "q_s must have shape [1, M, H] matching q");
  TORCH_CHECK(k_s.size(0) == 1 && k_s.size(1) == N, "k_s must have shape [1, N] matching k");

  auto out = at::empty({1, M, N}, q.options().dtype(at::kFloat));
  if (M == 0 || N == 0) {
    return out;
  }

  const auto bf16_opts = q.options().dtype(at::kBFloat16);

  // Q[0]: [1, M, H, D] fp8 → [M, H, D] bf16.
  // B=1 + contiguous: q.data_ptr() == q[0].data_ptr(), no select needed.
  auto q_bf16 = at::empty({M, H, D}, bf16_opts);
  fp8_to_bf16(
      q_bf16.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint8_t*>(q.const_data_ptr()),
      M * H * D);

  // K[0]: [1, N, D] fp8 → [N_pad, D] bf16, VNNI-packed in one fused pass.
  at::Tensor k_fp8 = k.select(0, 0);  // [N, D]
  at::Tensor k_packed = convert_fp8_to_bf16_packed<true>(k_fp8);

  // Views into q_s/k_s/out — no data copies.
  at::Tensor q_scale = q_s.select(0, 0);  // [M, H]
  at::Tensor k_scale = k_s.select(0, 0);  // [N]
  at::Tensor out_b   = out.select(0, 0);  // [M, N]
  fused_linear_relu_reduce<true, true>(out_b, q_bf16, q_scale, k_packed, k_scale);

  return out;
}
