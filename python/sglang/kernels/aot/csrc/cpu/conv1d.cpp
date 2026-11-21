#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
inline void pack_vnni_conv1d(
    scalar_t* __restrict__ packed, const scalar_t* __restrict__ weight, int64_t N, int64_t K, int64_t lda) {
  const int64_t VNNI_BLK = 2;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K / VNNI_BLK; ++k) {
      for (int64_t d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * lda + k * VNNI_BLK + d];
      }
    }
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void pack_vnni_conv1d(
    at::BFloat16* __restrict__ packed, const at::BFloat16* __restrict__ weight, int64_t N, int64_t K, int64_t lda) {
  const float* src = reinterpret_cast<const float*>(weight);
  float* dst = reinterpret_cast<float*>(packed);
  int64_t K2 = K >> 1;
  int64_t lda2 = lda >> 1;
  int64_t ldb2 = N;

  __m512i vinputs[16];

  for (int64_t n = 0; n < N; n += 16) {
    for (int64_t k2 = 0; k2 < K2; k2 += 16) {
      int64_t k2_size = std::min((int64_t)16, K2 - k2);
      int64_t n_size = std::min((int64_t)16, N - n);
      for (int64_t d = 0; d < n_size; ++d) {
        if (k2_size == 16) {
          vinputs[d] = _mm512_loadu_si512(src + (n + d) * lda2 + k2);
        } else {
          __mmask16 mask = (1 << k2_size) - 1;
          vinputs[d] = _mm512_maskz_loadu_epi32(mask, src + (n + d) * lda2 + k2);
        }
      }
      for (int64_t d = n_size; d < 16; ++d) {
        vinputs[d] = _mm512_setzero_si512();
      }
      transpose_16x16_32bit(vinputs);
      for (int64_t d = 0; d < k2_size; ++d) {
        if (n_size == 16) {
          _mm512_storeu_si512(dst + (k2 + d) * ldb2 + n, vinputs[d]);
        } else {
          __mmask16 mask = (1 << n_size) - 1;
          _mm512_mask_storeu_epi32(dst + (k2 + d) * ldb2 + n, mask, vinputs[d]);
        }
      }
    }
  }
}
#endif

// apply bias and convert: Ctmp [M, N] float -> C [M, N_out_stride] scalar_t
template <typename scalar_t>
inline void copy_add_bias_conv1d(
    scalar_t* __restrict__ C,
    const float* __restrict__ Ctmp,
    const scalar_t* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t ldc) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  for (int64_t d = 0; d < N; d += kVecSize) {
    fVec bias0, bias1;
    bVec bias_vec = bVec::loadu(bias + d);
    std::tie(bias0, bias1) = at::vec::convert_to_float(bias_vec);

    for (int64_t m = 0; m < M; ++m) {
      fVec data0 = fVec::loadu(Ctmp + m * N + d) + bias0;
      fVec data1 = fVec::loadu(Ctmp + m * N + d + fVec::size()) + bias1;
      bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
      out_vec.store(C + m * ldc + d);
    }
  }
}

// copy without bias
template <typename scalar_t>
inline void
copy_no_bias_conv1d(scalar_t* __restrict__ C, const float* __restrict__ Ctmp, int64_t M, int64_t N, int64_t ldc) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t d = 0; d < N; d += kVecSize) {
      fVec data0 = fVec::loadu(Ctmp + m * N + d);
      fVec data1 = fVec::loadu(Ctmp + m * N + d + fVec::size());
      bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
      out_vec.store(C + m * ldc + d);
    }
  }
}

// im2col for Conv1d: input [N, IC, L] padded -> col [N*L_out, IC*kernel_size]
//
// The im2col collects elements for each output position as:
//   col[n*L_out + l, ic*kernel_size + k] = padded_input[n, ic, l*stride + k]
//
// For GEMM weight layout [OC, IC*kernel_size], K dimension = (ic, k) in row major.
//
template <typename scalar_t>
void im2col_conv1d(
    scalar_t* __restrict__ col,
    const scalar_t* __restrict__ input,
    int64_t N,
    int64_t IC,
    int64_t L,
    int64_t L_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {
  // input layout: [N, IC, L]
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  const int64_t K = IC * kernel_size;

  at::parallel_for(0, N * L_out, 0, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t n = idx / L_out;
      int64_t l = idx % L_out;

      scalar_t* col_ptr = col + idx * K;
      const scalar_t* inp_n = input + n * IC * L;

      for (int64_t ic = 0; ic < IC; ++ic) {
        const scalar_t* inp_ic = inp_n + ic * L;
        scalar_t* col_ic = col_ptr + ic * kernel_size;

        for (int64_t k = 0; k < kernel_size; ++k) {
          int64_t l_in = l * stride + k - padding;
          if (l_in >= 0 && l_in < L) {
            col_ic[k] = inp_ic[l_in];
          } else {
            col_ic[k] = scalar_t(0);
          }
        }
      }
    }
  });
}

// Conv1d kernel implementation using AMX brgemm
//
// Computes: output[n, oc, l] = sum_{ic, k} input[n, ic, l*stride+k-padding] * weight[oc, ic, k] + bias[oc]
//
// Maps to GEMM:
//   A = im2col(input) : [N*L_out, IC*kernel_size]
//   B = weight        : [OC, IC*kernel_size] packed in VNNI
//   C = output        : [N*L_out, OC]
//
template <typename scalar_t, bool has_bias>
void conv1d_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ col,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int64_t M,  // N_batch * L_out
    int64_t OC,
    int64_t K) {  // IC * kernel_size

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(OC, BLOCK_N);

  // col   : [M, K]
  // weight: [OC/BLOCK_N][K/2, BLOCK_N, 2] in VNNI format
  // out   : [M, OC]
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(OC - nb_start, BLOCK_N);

      const scalar_t* __restrict__ A = col + mb_start * K;
      const scalar_t* __restrict__ B = weight + nb_start * K;

      at::native::cpublas::brgemm(mb_size, nb_size, K, K, BLOCK_N, BLOCK_N, false, A, B, Ctmp);

      if constexpr (has_bias) {
        copy_add_bias_conv1d(out + mb_start * OC + nb_start, Ctmp, bias + nb_start, mb_size, nb_size, OC);
      } else {
        copy_no_bias_conv1d(out + mb_start * OC + nb_start, Ctmp, mb_size, nb_size, OC);
      }
    });

    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

// Pack weight for conv1d from [OC, IC, kernel_size] to VNNI format
//
//   from [OC, IC, kernel_size]
//   view [OC, IC * kernel_size]  (K = IC * kernel_size)
//   pack [OC / BLOCK_N, BLOCK_N, K]
//   to   [OC / BLOCK_N][K / 2, BLOCK_N, 2]
//
at::Tensor conv1d_weight_pack(const at::Tensor& weight) {
  CHECK_INPUT(weight);
  CHECK_DIM(3, weight);

  int64_t OC = weight.size(0);
  int64_t IC = weight.size(1);
  int64_t kernel_size = weight.size(2);
  int64_t K = IC * kernel_size;

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(OC % BLOCK_N == 0, "conv1d_weight_pack: expect OC divisible by ", BLOCK_N);
  // K needs to be even for VNNI bf16 packing
  TORCH_CHECK(K % 2 == 0, "conv1d_weight_pack: expect IC*kernel_size to be even, got ", K);

  const int64_t NB = div_up(OC, BLOCK_N);
  at::Tensor weight_contig = weight.contiguous().view({OC, K});
  at::Tensor packed_weight = at::empty_like(weight_contig);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(weight.scalar_type(), "conv1d_weight_pack", [&] {
    at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
      const scalar_t* w_data = weight_contig.data_ptr<scalar_t>();
      scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();

      for (int64_t nb = begin; nb < end; ++nb) {
        int64_t n = nb * BLOCK_N;
        int64_t n_size = std::min(BLOCK_N, OC - n);

        pack_vnni_conv1d<scalar_t>(packed_data + nb * BLOCK_N * K, w_data + n * K, n_size, K, K);
      }
    });
  });

  return packed_weight;
}

// Conv1d using AMX GEMM
//
// input:  [N, IC, L]
// weight: [OC, IC, kernel_size] or pre-packed VNNI
// bias:   [OC] or None
// output: [N, OC, L_out]
//
at::Tensor conv1d_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    bool is_vnni) {
  CHECK_CONTIGUOUS(input);
  CHECK_CPU(input);
  CHECK_DIM(3, input);

  const int64_t N = input.size(0);
  const int64_t IC = input.size(1);
  const int64_t L = input.size(2);

  // weight is [OC, IC, kernel_size] or packed [OC, IC * kernel_size]
  int64_t OC, kernel_size;
  if (is_vnni) {
    // packed weight is [OC, IC * kernel_size] but we need to know kernel_size
    // infer from weight shape
    CHECK_DIM(2, weight);
    OC = weight.size(0);
    int64_t K = weight.size(1);
    TORCH_CHECK(K % IC == 0, "conv1d_cpu: packed weight K dimension must be divisible by IC");
    kernel_size = K / IC;
  } else {
    CHECK_DIM(3, weight);
    OC = weight.size(0);
    TORCH_CHECK(weight.size(1) == IC, "conv1d_cpu: weight IC mismatch");
    kernel_size = weight.size(2);
  }

  const int64_t L_out = (L + 2 * padding - kernel_size) / stride + 1;
  const int64_t K = IC * kernel_size;
  const int64_t M = N * L_out;

  const auto st = input.scalar_type();
  const bool has_bias = bias.has_value();

  if (has_bias) {
    CHECK_INPUT(bias.value());
    TORCH_CHECK(bias.value().size(0) == OC, "conv1d_cpu: bias size mismatch");
    TORCH_CHECK(bias.value().scalar_type() == st, "conv1d_cpu: bias dtype mismatch");
  }

  auto packed_w = is_vnni ? weight : conv1d_weight_pack(weight);

  // allocate im2col buffer
  at::Tensor col = at::empty({M, K}, input.options());

  // allocate output [N, L_out, OC] then transpose to [N, OC, L_out]
  at::Tensor out_flat = at::empty({M, OC}, input.options());

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "conv1d_cpu", [&] {
    // im2col
    im2col_conv1d<scalar_t>(
        col.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), N, IC, L, L_out, kernel_size, stride, padding);

    // GEMM
    if (has_bias) {
      conv1d_kernel_impl<scalar_t, true>(
          out_flat.data_ptr<scalar_t>(),
          col.data_ptr<scalar_t>(),
          packed_w.data_ptr<scalar_t>(),
          bias.value().data_ptr<scalar_t>(),
          M,
          OC,
          K);
    } else {
      conv1d_kernel_impl<scalar_t, false>(
          out_flat.data_ptr<scalar_t>(), col.data_ptr<scalar_t>(), packed_w.data_ptr<scalar_t>(), nullptr, M, OC, K);
    }
  });

  // reshape from [N*L_out, OC] to [N, L_out, OC]
  return out_flat.view({N, L_out, OC});
}
