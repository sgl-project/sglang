#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>

#include "common.h"

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
extern void fused_linear_relu_reduce(
    at::Tensor& out,
    at::Tensor& q,
    at::Tensor& q_scale,
    at::Tensor& k,
    at::Tensor& k_scale,
    bool is_vnni);

// AMX weight packing requires the GEMM output dimension (here the number of
// key tokens N) to be a multiple of TILE_N (16). Pad the key rows up so the
// packed path is always taken; the padding rows are never read since
// fused_linear_relu_reduce only produces k_scale.size(0) output columns.
constexpr int64_t kTileN = 16;


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
  const int64_t M = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t N = k.size(1);

  TORCH_CHECK(k.size(0) == B && k.size(2) == D, "k must have shape [B, N, D] matching q");
  TORCH_CHECK(
      q_s.size(0) == B && q_s.size(1) == M && q_s.size(2) == H, "q_s must have shape [B, M, H] matching q");
  TORCH_CHECK(k_s.size(0) == B && k_s.size(1) == N, "k_s must have shape [B, N] matching k");

  auto out = at::empty({B, M, N}, q.options().dtype(at::kFloat));
  if (B == 0 || M == 0 || N == 0) {
    return out;
  }

  const int64_t N_pad = ((N + kTileN - 1) / kTileN) * kTileN;
  const auto bf16_opts = q.options().dtype(at::kBFloat16);

  for (int64_t b = 0; b < B; ++b) {
    at::Tensor q_bf16, k_bf16;
    // Q[b] : [M, H, D] bf16.
    q_bf16 = q.select(0, b).to(at::kBFloat16);
    // K[b] : [N_pad, D] bf16, key rows padded with zeros so N_pad % 16 == 0.
    if (N_pad == N) {
      k_bf16 = k.select(0, b).to(at::kBFloat16);
    } else {
      k_bf16 = at::zeros({N_pad, D}, bf16_opts);
      k_bf16.narrow(0, 0, N).copy_(k.select(0, b).to(at::kBFloat16));
    }

    at::Tensor q_scale_b = q_s.select(0, b);  // [M, H]
    at::Tensor k_scale_b = k_s.select(0, b);  // [N]
    at::Tensor out_b = out.select(0, b);      // [M, N]
    fused_linear_relu_reduce(out_b, q_bf16, q_scale_b, k_bf16, k_scale_b, /*is_vnni=*/false);
  }

  return out;
}
