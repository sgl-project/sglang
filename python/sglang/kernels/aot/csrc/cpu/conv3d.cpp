#include "common.h"
#include "gemm.h"
#include "vec.h"
#include "vec_pack.h"

namespace {

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
    auto [bias0, bias1] = load_float_vec2(bias + d);

    for (int64_t m = 0; m < M; ++m) {
      auto [data0, data1] = load_float_vec2(Ctmp + m * N + d);
      data0 = data0 + bias0;
      data1 = data1 + bias1;
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

  // input : [N, K]
  // weight: [OC/BLOCK_N, K/2, BLOCK_N, 2]
  // out   : [N, OC]
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

// [NB]: use blocked format for weight of OIDHW.
//
//   from [OC / BLOCK_N, BLOCK_N, K]
//   to   [OC / BLOCK_N, K / 2, BLOCK_N, 2]
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

  const int64_t K = IC * D * H * W;
  TORCH_CHECK(K % 2 == 0, "conv3d_embed_weight_pack: expect K divisible by 2, got ", K);
  const int64_t NB = div_up(OC, BLOCK_N);
  at::Tensor packed_weight = at::empty_like(weight);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(weight.scalar_type(), "conv3d_embed_weight_pack", [&] {
    at::parallel_for(0, NB, 0, [&](int64_t begin, int64_t end) {
      const scalar_t* w_data = weight.data_ptr<scalar_t>();
      scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();

      for (int64_t nb = begin; nb < end; ++nb) {
        int64_t n = nb * BLOCK_N;
        scalar_t* packed_block = packed_data + nb * BLOCK_N * K;
        const scalar_t* weight_block = w_data + n * K;

        pack_vnni<scalar_t>(packed_block, weight_block, BLOCK_N, K, K, BLOCK_N);
      }
    });
  });

  return packed_weight;
}

// conv3d mapped to gemm in embedding
at::Tensor conv3d_embed_cpu(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, bool is_vnni) {
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
