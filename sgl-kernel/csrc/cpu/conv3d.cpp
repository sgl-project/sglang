#include "common.h"
#include "gemm.h"
#include "vec.h"

namespace {

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
inline void
pack_vnni(scalar_t* __restrict__ packed, const scalar_t* __restrict__ weight, int64_t N, int64_t K, int64_t lda) {
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
inline void pack_vnni(
    at::BFloat16* __restrict__ packed, const at::BFloat16* __restrict__ weight, int64_t N, int64_t K, int64_t lda) {
  const float* src = reinterpret_cast<const float*>(weight);
  float* dst = reinterpret_cast<float*>(packed);
  int64_t K2 = K >> 1;
  int64_t lda2 = lda >> 1;
  int64_t ldb2 = N * 2 >> 1;

  __m512i vinputs[16];

  for (int64_t n = 0; n < N; n += 16) {
    for (int64_t k2 = 0; k2 < K2; k2 += 16) {
      for (int64_t d = 0; d < 16; ++d) {
        vinputs[d] = _mm512_loadu_si512(src + (n + d) * lda2 + k2);
      }
      transpose_16x16_32bit(vinputs);
      for (int64_t d = 0; d < 16; ++d) {
        _mm512_storeu_si512(dst + (k2 + d) * ldb2 + n, vinputs[d]);
      }
    }
  }
}
#endif

// apply bias: C [M, N] ldc, Ctmp: [M, N]
template <typename scalar_t>
inline void copy_add_stub(
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

template <typename scalar_t>
void conv3d_embed_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    int64_t N,
    int64_t IC,
    int64_t OC,
    int64_t D,
    int64_t H,
    int64_t W) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(N, BLOCK_M);
  const int64_t NB = div_up(OC, BLOCK_N);

  // K in gemm
  const int64_t K = IC * D * H * W;

  // input : [ N/BLOCK_M, BLOCK_M, IC, D, H, W]
  // weight: [OC/BLOCK_N, IC, D, H*W/2, BLOCK_N, 2]
  // out   : [N/BLOCK_M, BLOCK_M, OC/BLOCK_N, BLOCK_N]
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(N - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(OC - nb_start, BLOCK_N);

      const scalar_t* __restrict__ A = input + mb_start * K;
      const scalar_t* __restrict__ B = weight + nb_start * K;
#if 0
      // only access 1st index of D dimension
      for (int64_t ic = 0; ic < IC; ++ic) {
        for (int64_t d = 0; d < D; ++d) {
          at::native::cpublas::brgemm(
              mb_size,
              nb_size,
              H * W,
              K,
              BLOCK_N,
              BLOCK_N,
              /* add_C */ ic > 0 || d > 0,
              A + ic * (D * H * W) + /* d */ 0 * (H * W), // dimension D for input is repeated
              B + ic * (D * BLOCK_N * H * W) + d * (BLOCK_N * H * W),
              Ctmp);
      }
#else
      // accumulates K normally, this is still marginally faster than above
      at::native::cpublas::brgemm(mb_size, nb_size, K, K, BLOCK_N, BLOCK_N, false, A, B, Ctmp);
#endif
      // update bias
      copy_add_stub(out + mb_start * OC + nb_start, Ctmp, bias + nb_start, mb_size, nb_size, OC);
    });

    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

// [NB]: use blocked format for weight of OIDHW
//
//   from [OC, Cin, D, H, W]
//   view [OC / BLOCK_N, BLOCK_N, Cin, D, H * W]
//   view [OC / BLOCK_N, IC, D, BLOCK_N, H * W]
//   to   [OC / BLOCK_N][IC, D][H * W / 2, BLOCK_N, 2]
//        +- parallel -+- seq -+------ mma ----------+
//
at::Tensor conv3d_embed_weight_pack(const at::Tensor& weight) {
  CHECK_INPUT(weight);

  int64_t OC = weight.size(0);
  int64_t IC = weight.size(1);
  int64_t D = weight.size(2);
  int64_t H = weight.size(3);
  int64_t W = weight.size(4);

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(OC % BLOCK_N == 0, "conv3d_embed_weight_pack: expect OC dividable by ", BLOCK_N);
  TORCH_CHECK((H * W) % TILE_K == 0, "conv3d_embed_weight_pack: expect IC dividable by ", TILE_K);

  // strides
  int64_t stride_nb = BLOCK_N * IC * D * H * W;
  int64_t stride_ic = D * H * W;
  int64_t stride_d = H * W;

  const int64_t NB = div_up(OC, BLOCK_N);
  at::Tensor packed_weight = at::empty_like(weight);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(weight.scalar_type(), "conv3d_embed_weight_pack", [&] {
    // parallel {NB, IC, D}
    at::parallel_for(0, NB * IC * D, 0, [&](int64_t begin, int64_t end) {
      int64_t nb{0}, ic{0}, d{0};
      data_index_init(begin, nb, NB, ic, IC, d, D);

      const scalar_t* w_data = weight.data_ptr<scalar_t>();
      scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();

      for (int64_t i = begin; i < end; ++i) {
        int64_t n = nb * BLOCK_N;
        int64_t n_size = std::min(BLOCK_N, OC - n);  // BLOCK_N

        pack_vnni<scalar_t>(
            packed_data + i * (BLOCK_N * H * W),
            w_data + nb * stride_nb + ic * stride_ic + d * stride_d,
            n_size,
            H * W,
            IC * D * H * W);

        // move to the next index
        data_index_step(nb, NB, ic, IC, d, D);
      }
    });
  });

  return packed_weight;
}

// conv3d mapped to gemm in embedding
at::Tensor conv3d_embed_cpu(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, bool is_vnni) {
  RECORD_FUNCTION("sgl_kernel::conv3d_embed_cpu", std::vector<c10::IValue>({input, weight, bias}));

  auto packed_w = is_vnni ? weight : conv3d_embed_weight_pack(weight);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);
  CHECK_DIM(5, input);
  CHECK_DIM(5, weight);

  const int64_t N = input.size(0);
  const int64_t IC = input.size(1);
  const int64_t OC = weight.size(0);
  const int64_t D = input.size(2);
  const int64_t H = input.size(3);
  const int64_t W = input.size(4);

  const auto st = input.scalar_type();
  CHECK_INPUT_SHAPE_DTYPE<false>(weight, {OC, IC, D, H, W}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(bias, {OC}, st);

  // allocate {D, H, W} for out is 1
  at::Tensor out = at::empty({N, OC}, input.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "conv3d_embed_kernel_impl", [&] {
    conv3d_embed_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        packed_w.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        N,
        IC,
        OC,
        D,
        H,
        W);
  });

  return out;
}
