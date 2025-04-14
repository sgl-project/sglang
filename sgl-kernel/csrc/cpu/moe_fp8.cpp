#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// out = input + input2 * scale
template <typename scalar_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x_bvec);

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

}  // namespace

template <typename scalar_t>
void shared_expert_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K) {
  // handle 2 tiles per block
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();

  // stage 1: intermediate_cache1 = silu(hidden_states @ w1)
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(2 * N, BLOCK_N);
  int64_t scale_size_N = div_up(2 * N, block_size_N);
  int64_t scale_size_K = div_up(K, block_size_K);
  int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  // TODO: add the support for use_brgemm = false;
  // use avx512-bf16 when a) M is small; b) dtype is bfloat16, otherwise use amx
  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  int64_t mat1_strideM = K;
  int64_t out_strideM = 2 * N;
  alignas(64) scalar_t my_output[M * 2 * N];
  alignas(64) scalar_t output_ic1[M * N];

  // here we only parallel on half of 2N to fuse silu_and_mul with gemm
  at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
    alignas(64) scalar_t Btmp[BLOCK_N * BLOCK_K];
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB;
      int64_t nb = i % NB;

      const float* scale_ptr = w1s + (nb / blocks_n_per_group) * scale_size_K;
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(2 * N - nb_start, BLOCK_N);

      // 1.b gemm: C = A @ B
      tinygemm_kernel<scalar_t>(
          /*   A                  */ input + mb_start * mat1_strideM,
          /*   B                  */ packed_w1 + nb_start * K,
          /*   C                  */ my_output + mb_start * out_strideM + nb_start,
          /*   Btmp               */ Btmp,
          /*   Ctmp               */ Ctmp,
          /*   scale              */ scale_ptr,
          /*   M                  */ mb_size,
          /*   N                  */ nb_size,
          /*   K                  */ K,
          /*   lda                */ mat1_strideM,
          /*   ldb                */ nb_size,
          /*   ldc                */ out_strideM,
          /*   brg                */ use_brgemm,
          /*   block_size_K       */ block_size_K);
    }
  });
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t d = 0; d < N; d += bVec::size()) {
      bVec x = bVec::loadu(my_output + m * 2 * N + d);
      fVec x0, x1;
      std::tie(x0, x1) = at::vec::convert_to_float(x);
      bVec y = bVec::loadu(my_output + m * 2 * N + N + d);
      fVec y0, y1;
      std::tie(y0, y1) = at::vec::convert_to_float(y);
      x0 = x0 / (one + x0.neg().exp_u20());
      x1 = x1 / (one + x1.neg().exp_u20());
      x0 = x0 * y0;
      x1 = x1 * y1;
      bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
      out_vec.store(output_ic1 + m * N + d);
    }
  }

  // stage 2: intermediate_cache2 = intermediate_cache1 @ w2
  //   w2 : [K, N] as [OC, IC]
  const int64_t OC = K;  // rename K as OC
  const int64_t IC = N;  // rename N as IC
  const int64_t MB2 = MB;
  const int64_t NB2 = div_up(OC, BLOCK_N);
  const int64_t stride_oc = N;
  mat1_strideM = IC;
  out_strideM = OC;
  scale_size_N = div_up(K, block_size_N);
  scale_size_K = div_up(N, block_size_K);
  blocks_n_per_group = block_size_N / BLOCK_N;

  // parallel on [MB2, NB2]
  at::parallel_for(0, MB2 * NB2, 0, [&](int64_t begin, int64_t end) {
    alignas(64) scalar_t Btmp2[BLOCK_K * BLOCK_N];
    alignas(64) scalar_t my_output2[M * K];
    alignas(64) scalar_t C2[BLOCK_M * BLOCK_K];
    alignas(64) float Ctmp2[BLOCK_M * BLOCK_K];

    for (int64_t i = begin; i < end; ++i) {
      int64_t mb = i / NB2;
      int64_t nb = i % NB2;
      const float* scale_ptr_2 = w2s + (nb / blocks_n_per_group) * scale_size_K;
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(OC - nb_start, BLOCK_N);

      // 2.a gemm: C = A @ B
      tinygemm_kernel<scalar_t>(
          /*   A                  */ output_ic1 + mb_start * mat1_strideM,
          /*   B                  */ packed_w2 + nb_start * N,
          /*   C                  */ C2,
          /*   Btmp               */ Btmp2,
          /*   Ctmp               */ Ctmp2,
          /*   scale              */ scale_ptr_2,
          /*   M                  */ mb_size,
          /*   N                  */ nb_size,
          /*   K                  */ IC,
          /*   lda                */ mat1_strideM,
          /*   ldb                */ nb_size,
          /*   ldc                */ BLOCK_N,
          /*   brg                */ use_brgemm,
          /*   block_size_K       */ block_size_K);

      // 2.b copy from C to output and add fused_experts_out
      scalar_t* __restrict__ out = output + mb_start * out_strideM + nb_start;
      const scalar_t* __restrict__ fused_out = fused_experts_out + mb_start * out_strideM + nb_start;
      for (int64_t m = 0; m < mb_size; ++m) {
        add_mul_stub(out + m * K, C2 + m * BLOCK_N, fused_out + m * K, routed_scaling_factor, nb_size);
      }
    }
  });

  if (use_brgemm) {
    at::native::cpublas::brgemm_release();
  }
}

#define INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(TYPE)   \
  template void shared_expert_fp8_kernel_impl<TYPE>(   \
      TYPE* __restrict__ output,                       \
      TYPE* __restrict__ ic1,                          \
      float* __restrict__ C_tmp,                       \
      const TYPE* __restrict__ input,                  \
      const at::Float8_e4m3fn* __restrict__ packed_w1, \
      const at::Float8_e4m3fn* __restrict__ packed_w2, \
      const float* __restrict__ w1s,                   \
      const float* __restrict__ w2s,                   \
      int64_t block_size_N,                            \
      int64_t block_size_K,                            \
      const TYPE* __restrict__ fused_experts_out,      \
      float routed_scaling_factor,                     \
      int64_t M,                                       \
      int64_t N,                                       \
      int64_t K)

INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::BFloat16);
INSTANTIATE_SHARED_EXPERT_FP8_TEMPLATE(at::Half);
