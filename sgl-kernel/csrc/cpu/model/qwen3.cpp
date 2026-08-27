#include "common.h"
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
  RECORD_FUNCTION("sgl-kernel::fused_qkvzba_split_reshape_cat_cpu", std::vector<c10::IValue>({mixed_qkvz, mixed_ba}));
  CHECK_DIM(2, mixed_qkvz);
  CHECK_DIM(2, mixed_ba);
  CHECK_INPUT(mixed_qkvz);
  CHECK_INPUT(mixed_ba);
  int64_t batch = mixed_qkvz.size(0);
  int64_t qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v;
  int64_t ba_dim = num_heads_v * 2;
  int64_t expected_dim = qkv_dim + num_heads_v * head_v;
  CHECK_EQ(mixed_qkvz.size(1), expected_dim);
  CHECK_EQ(mixed_ba.size(0), batch);
  CHECK_EQ(mixed_ba.size(1), ba_dim);
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
