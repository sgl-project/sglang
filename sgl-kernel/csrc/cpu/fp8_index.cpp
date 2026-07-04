#include <c10/util/Float8_e4m3fn.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "common.h"
#include "gemm.h"
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
// Step (1) is the expensive contraction over head_dim. It is expressed as a
// single AMX bf16 GEMM by folding (m, h) into the row dimension:
//   L[m*H + h, n] = Q[m*H + h, :] @ K[n, :]^T
// and dispatched through the efficient ``weight_packed_linear`` path from
// gemm.cpp (bf16 inputs, fp32 accumulation/output). The relu + head-weighted
// reduction (steps 2-4) stays as a vectorized elementwise pass because the
// relu nonlinearity between the two contractions cannot be folded into a GEMM.
//
// Shapes (all contiguous):
//   q   : [B, M, H, D]  float8_e4m3fn
//   q_s : [B, M, H]     float32
//   k   : [B, N, D]     float8_e4m3fn
//   k_s : [B, N]        float32
//   out : [B, M, N]     float32

// Efficient AMX bf16 GEMM from gemm.cpp: out = mat1 @ mat2^T.
extern at::Tensor weight_packed_linear(
    at::Tensor& mat1,
    at::Tensor& mat2,
    const std::optional<at::Tensor>& bias,
    bool is_vnni,
    std::optional<at::ScalarType> out_dtype);

namespace {

// AMX weight packing requires the GEMM output dimension (here the number of
// key tokens N) to be a multiple of TILE_N (16). Pad the key rows up so the
// packed path is always taken; the extra logit columns are dropped during the
// head-weighted reduction below.
constexpr int64_t kTileN = 16;

inline int64_t round_up(int64_t x, int64_t m) {
  return ((x + m - 1) / m) * m;
}

// out[m, n] = k_s[n] * sum_h relu(logits[m, h, n]) * q_s[m, h]
void reduce_heads(
    const float* __restrict__ logits,  // [M, H, N_pad]
    const float* __restrict__ qs,      // [M, H]
    const float* __restrict__ ks,      // [N]
    float* __restrict__ out,           // [M, N]
    int64_t M,
    int64_t H,
    int64_t N,
    int64_t N_pad) {
  using fVec = at::vec::Vectorized<float>;
  const fVec zero(0.0f);
  const int64_t kVecSize = fVec::size();

  at::parallel_for(0, M, 0, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const float* logits_m = logits + m * H * N_pad;
      const float* qs_m = qs + m * H;
      float* out_m = out + m * N;

      std::memset(out_m, 0, N * sizeof(float));

      for (int64_t h = 0; h < H; ++h) {
        const float* logits_h = logits_m + h * N_pad;
        const float w = qs_m[h];
        const fVec wv(w);
        int64_t n = 0;
        for (; n <= N - kVecSize; n += kVecSize) {
          fVec acc = fVec::loadu(out_m + n);
          fVec l = at::vec::maximum(fVec::loadu(logits_h + n), zero);
          acc = at::vec::fmadd(l, wv, acc);
          acc.store(out_m + n);
        }
        for (; n < N; ++n) {
          out_m[n] += std::max(logits_h[n], 0.0f) * w;
        }
      }

      int64_t n = 0;
      for (; n <= N - kVecSize; n += kVecSize) {
        fVec acc = fVec::loadu(out_m + n) * fVec::loadu(ks + n);
        acc.store(out_m + n);
      }
      for (; n < N; ++n) {
        out_m[n] *= ks[n];
      }
    }
  });
}

}  // namespace

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

  const int64_t N_pad = round_up(N, kTileN);
  const auto bf16_opts = q.options().dtype(at::kBFloat16);
  const auto* qs_ptr = q_s.const_data_ptr<float>();
  const auto* ks_ptr = k_s.const_data_ptr<float>();
  auto* out_ptr = out.data_ptr<float>();

  for (int64_t b = 0; b < B; ++b) {
    // Q[b] : [M*H, D] bf16
    at::Tensor q_bf16 = q.select(0, b).reshape({M * H, D}).to(at::kBFloat16);

    // K[b] : [N_pad, D] bf16, key rows padded with zeros so N_pad % 16 == 0.
    at::Tensor k_bf16;
    if (N_pad == N) {
      k_bf16 = k.select(0, b).to(at::kBFloat16);
    } else {
      k_bf16 = at::zeros({N_pad, D}, bf16_opts);
      k_bf16.narrow(0, 0, N).copy_(k.select(0, b).to(at::kBFloat16));
    }

    // L : [M*H, N_pad] fp32  ==  Q @ K^T 
    at::Tensor logits = weight_packed_linear(q_bf16, k_bf16, c10::nullopt, /*is_vnni=*/false, at::kFloat);

    reduce_heads(
        logits.const_data_ptr<float>(),
        qs_ptr + b * M * H,
        ks_ptr + b * N,
        out_ptr + b * M * N,
        M,
        H,
        N,
        N_pad);
  }

  return out;
}
