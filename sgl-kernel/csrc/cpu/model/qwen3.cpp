#include "common.h"
#include "gemm.h"
#include "vec.h"
namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec out_bvec = bVec::loadu(src + d);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = src[d];
  }
}

template <typename scalar_t>
void fused_qkvzba_split_reshape_cat_impl(
    const scalar_t* __restrict__ mixed_qkvz,
    const scalar_t* __restrict__ mixed_ba,
    scalar_t* __restrict__ mixed_qkv,
    scalar_t* __restrict__ z,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ a,
    int64_t batch,
    int64_t num_heads_qk,
    int64_t num_heads_v,
    int64_t head_qk,
    int64_t group,
    int64_t head_v,
    int64_t qkv_strideB,
    int64_t qkvz_strideB,
    int64_t ba_strideB) {
  int64_t qkvz_stride_per_head = head_qk * 2 + head_v * 2 * group;
  at::parallel_for(0, batch * num_heads_qk, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, hi{0};
    data_index_init(begin, bi, batch, hi, num_heads_qk);
    for (int64_t i = begin; i < end; ++i) {
      scalar_t* __restrict__ q_out_ptr = mixed_qkv + bi * qkv_strideB + hi * head_qk;
      const scalar_t* __restrict__ q_in_ptr = mixed_qkvz + bi * qkvz_strideB + hi * qkvz_stride_per_head;
      scalar_t* __restrict__ k_out_ptr = q_out_ptr + num_heads_qk * head_qk;
      const scalar_t* __restrict__ k_in_ptr = q_in_ptr + head_qk;
      scalar_t* __restrict__ v_out_ptr = k_out_ptr + num_heads_qk * head_qk + hi * head_qk * (group - 1);
      const scalar_t* __restrict__ v_in_ptr = k_in_ptr + head_qk;
      scalar_t* __restrict__ z_out_ptr = z + bi * num_heads_v * head_v + hi * group * head_v;
      const scalar_t* __restrict__ z_in_ptr = v_in_ptr + head_qk * group;
      copy_stub(q_out_ptr, q_in_ptr, head_qk);
      copy_stub(k_out_ptr, k_in_ptr, head_qk);
      copy_stub(v_out_ptr, v_in_ptr, head_qk * group);
      copy_stub(z_out_ptr, z_in_ptr, head_qk * group);
      scalar_t* __restrict__ b_out_ptr = b + bi * num_heads_v + hi * group;
      const scalar_t* __restrict__ b_in_ptr = mixed_ba + bi * ba_strideB + hi * group * 2;
      scalar_t* __restrict__ a_out_ptr = a + bi * num_heads_v + hi * group;
      const scalar_t* __restrict__ a_in_ptr = b_in_ptr + group;
      copy_stub(b_out_ptr, b_in_ptr, group);
      copy_stub(a_out_ptr, a_in_ptr, group);
      data_index_step(bi, batch, hi, num_heads_qk);
    }
  });
}

template <typename scalar_t>
void fused_qkvzba_split_reshape_cat_contiguous_impl(
    const scalar_t* __restrict__ mixed_qkvz,
    const scalar_t* __restrict__ mixed_ba,
    scalar_t* __restrict__ mixed_qkv,
    scalar_t* __restrict__ z,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ a,
    int64_t batch,
    int64_t v_tp,
    int64_t num_heads_v,
    int64_t qkv_dim,
    int64_t qkv_strideB,
    int64_t qkvz_strideB,
    int64_t ba_strideB) {
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t bi = begin; bi < end; ++bi) {
      scalar_t* __restrict__ qkv_out_ptr = mixed_qkv + bi * qkv_strideB;
      const scalar_t* __restrict__ qkv_in_ptr = mixed_qkvz + bi * qkvz_strideB;
      scalar_t* __restrict__ z_out_ptr = z + bi * v_tp;
      const scalar_t* __restrict__ z_in_ptr = qkv_in_ptr + qkv_dim;
      copy_stub(qkv_out_ptr, qkv_in_ptr, qkv_dim);
      copy_stub(z_out_ptr, z_in_ptr, v_tp);
      scalar_t* __restrict__ b_out_ptr = b + bi * num_heads_v;
      const scalar_t* __restrict__ b_in_ptr = mixed_ba + bi * ba_strideB;
      scalar_t* __restrict__ a_out_ptr = a + bi * num_heads_v;
      const scalar_t* __restrict__ a_in_ptr = b_in_ptr + num_heads_v;
      copy_stub(b_out_ptr, b_in_ptr, num_heads_v);
      copy_stub(a_out_ptr, a_in_ptr, num_heads_v);
    }
  });
}

template <typename scalar_t>
void fused_input_proj_kernel_impl(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ out2,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ weight2,
    int64_t M,
    int64_t N,
    int64_t N2,
    int64_t K) {
  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N + N2, BLOCK_N);

  const bool use_brgemm = can_use_brgemm<scalar_t>(M);

  // parallel on [MB, NB]
  parallel_2d(MB, NB, [&](int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1) {
    // for brgemm, use float32 for accumulate
    alignas(64) float Ctmp[BLOCK_M * BLOCK_N];

    loop_2d<scalar_t>(mb0, mb1, nb0, nb1, BLOCK_N * K, [&](int64_t mb, int64_t nb, int64_t nb_offset) {
      int64_t mb_start = mb * BLOCK_M;
      int64_t mb_size = std::min(M - mb_start, BLOCK_M);
      int64_t nb_start = nb * BLOCK_N;
      const bool is_first = nb_start < N;
      int64_t local_nb_start = is_first ? nb_start : nb_start - N;
      int64_t nb_size = std::min((is_first ? N : N2) - local_nb_start, BLOCK_N);
      scalar_t* __restrict__ curr_out = is_first ? out : out2;
      const scalar_t* __restrict__ curr_weight = is_first ? weight : weight2;
      int64_t local_out_strideM = is_first ? N : N2;

      tinygemm_kernel<scalar_t>(
          /*   A */ input + mb_start * K,
          /*   B */ curr_weight + local_nb_start * K,
          /*   C */ curr_out + mb_start * local_out_strideM + local_nb_start,
          /* Ctmp*/ Ctmp,
          /*   M */ mb_size,
          /*   N */ nb_size,
          /*   K */ K,
          /* lda */ K,
          /* ldb */ nb_size,
          /* ldc */ local_out_strideM,
          /* brg */ use_brgemm);
    });

    if (use_brgemm) {
      at::native::cpublas::brgemm_release();
    }
  });
}

}  // anonymous namespace

// mixed_qkvz: [batch, num_heads_qk * head_qk * 2 + num_heads_v * head_v * 2]
// mixed_ba: [batch, num_heads_v * 2]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_qkvzba_split_reshape_cat_cpu(
    const at::Tensor& mixed_qkvz,
    const at::Tensor& mixed_ba,
    int64_t num_heads_qk,
    int64_t num_heads_v,
    int64_t head_qk,
    int64_t head_v) {
  int64_t batch = mixed_qkvz.size(0);
  int64_t qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v;
  int64_t ba_dim = num_heads_v * 2;
  int64_t expected_dim = qkv_dim + num_heads_v * head_v;
  CHECK_INPUT_SHAPE_DTYPE<false>(mixed_qkvz, {batch, expected_dim}, mixed_qkvz.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(mixed_ba, {batch, ba_dim}, mixed_qkvz.scalar_type());
  CHECK_EQ(num_heads_v % num_heads_qk, 0);
  at::Tensor mixed_qkv = at::empty({batch, qkv_dim}, mixed_qkvz.options());
  at::Tensor z = at::empty({batch, num_heads_v, head_v}, mixed_qkvz.options());
  at::Tensor b = at::empty({batch, num_heads_v}, mixed_ba.options());
  at::Tensor a = at::empty({batch, num_heads_v}, mixed_ba.options());
  int64_t group = num_heads_v / num_heads_qk;
  int64_t qkvz_strideB = mixed_qkvz.size(1);
  int64_t qkv_strideB = mixed_qkv.size(1);
  int64_t ba_strideB = mixed_ba.size(1);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(mixed_qkvz.scalar_type(), "fused_qkvzba_split_reshape_cat_impl", [&] {
    fused_qkvzba_split_reshape_cat_impl<scalar_t>(
        mixed_qkvz.data_ptr<scalar_t>(),
        mixed_ba.data_ptr<scalar_t>(),
        mixed_qkv.data_ptr<scalar_t>(),
        z.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        a.data_ptr<scalar_t>(),
        batch,
        num_heads_qk,
        num_heads_v,
        head_qk,
        group,
        head_v,
        qkv_strideB,
        qkvz_strideB,
        ba_strideB);
  });
  return std::make_tuple(mixed_qkv, z, b, a);
}

// mixed_qkvz: [batch, num_heads_qk * head_qk * 2 + num_heads_v * head_v * 2]
// mixed_ba: [batch, num_heads_v * 2]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_qkvzba_split_reshape_cat_contiguous_cpu(
    const at::Tensor& mixed_qkvz,
    const at::Tensor& mixed_ba,
    int64_t num_heads_qk,
    int64_t num_heads_v,
    int64_t head_qk,
    int64_t head_v) {
  int64_t batch = mixed_qkvz.size(0);
  int64_t k_tp = num_heads_qk * head_qk;
  int64_t v_tp = num_heads_v * head_v;
  int64_t qkv_dim = k_tp * 2 + v_tp;
  int64_t ba_dim = num_heads_v * 2;
  int64_t expected_dim = qkv_dim + v_tp;
  CHECK_INPUT_SHAPE_DTYPE<false>(mixed_qkvz, {batch, expected_dim}, mixed_qkvz.scalar_type());
  CHECK_INPUT_SHAPE_DTYPE<false>(mixed_ba, {batch, ba_dim}, mixed_qkvz.scalar_type());
  at::Tensor mixed_qkv = at::empty({batch, qkv_dim}, mixed_qkvz.options());
  at::Tensor z = at::empty({batch, num_heads_v, head_v}, mixed_qkvz.options());
  at::Tensor b = at::empty({batch, num_heads_v}, mixed_ba.options());
  at::Tensor a = at::empty({batch, num_heads_v}, mixed_ba.options());
  int64_t qkvz_strideB = mixed_qkvz.size(1);
  int64_t qkv_strideB = mixed_qkv.size(1);
  int64_t ba_strideB = mixed_ba.size(1);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(mixed_qkvz.scalar_type(), "fused_qkvzba_split_reshape_cat_contiguous_impl", [&] {
    fused_qkvzba_split_reshape_cat_contiguous_impl<scalar_t>(
        mixed_qkvz.data_ptr<scalar_t>(),
        mixed_ba.data_ptr<scalar_t>(),
        mixed_qkv.data_ptr<scalar_t>(),
        z.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        a.data_ptr<scalar_t>(),
        batch,
        v_tp,
        num_heads_v,
        qkv_dim,
        qkv_strideB,
        qkvz_strideB,
        ba_strideB);
  });
  return std::make_tuple(mixed_qkv, z, b, a);
}

// [projected_states_qkvz |projected_states_ba]
//   = hidden_states @ [qkvz_weight.T | ba_weight.T]
//
//   hidden_states         : [batch, hidden_size]
//   qkvz_weight           : [qkvz_dim, hidden_size]
//   ba_weight             : [ba_dim, hidden_size]
//   projected_states_qkvz : [batch, qkvz_dim]
//   projected_states_ba   : [batch, ba_dim]
//
std::tuple<at::Tensor, at::Tensor>
fused_input_proj_cpu(at::Tensor& hidden_states, at::Tensor& qkvz_weight, at::Tensor& ba_weight, bool is_vnni) {
  const auto st = hidden_states.scalar_type();
  TORCH_CHECK(st == at::ScalarType::BFloat16, "fused_input_proj_cpu only supports BFloat16");

  int64_t batch = hidden_states.size(0);
  int64_t hidden_size = hidden_states.size(1);
  int64_t qkvz_dim = qkvz_weight.size(0);
  int64_t ba_dim = ba_weight.size(0);
  CHECK_INPUT(hidden_states);
  CHECK_INPUT_SHAPE_DTYPE<false>(qkvz_weight, {qkvz_dim, hidden_size}, st);
  CHECK_INPUT_SHAPE_DTYPE<false>(ba_weight, {ba_dim, hidden_size}, st);
  TORCH_CHECK(qkvz_dim % block_size_n() == 0, "qkvz_weight out features must be divisible by ", block_size_n());
  TORCH_CHECK(ba_dim % block_size_n() == 0, "ba_weight out features must be divisible by ", block_size_n());
  TORCH_CHECK(hidden_size % TILE_K == 0, "hidden_size must be divisible by ", TILE_K);

  // weight prepacking if necessary
  at::Tensor packed_w = is_vnni ? qkvz_weight : convert_weight_packed(qkvz_weight);
  at::Tensor packed_w2 = is_vnni ? ba_weight : convert_weight_packed(ba_weight);

  at::Tensor projected_states_qkvz = at::empty({batch, qkvz_dim}, hidden_states.options());
  at::Tensor projected_states_ba = at::empty({batch, ba_dim}, hidden_states.options());
  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "fused_input_proj_cpu", [&] {
    fused_input_proj_kernel_impl<scalar_t>(
        projected_states_qkvz.data_ptr<scalar_t>(),
        projected_states_ba.data_ptr<scalar_t>(),
        hidden_states.data_ptr<scalar_t>(),
        packed_w.data_ptr<scalar_t>(),
        packed_w2.data_ptr<scalar_t>(),
        batch,
        qkvz_dim,
        ba_dim,
        hidden_size);
  });
  return std::make_tuple(projected_states_qkvz, projected_states_ba);
}
