#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

// convert to vnni format
// from [N, K] to [K/2, N, 2] for bfloat16 and float16
template <typename scalar_t>
inline void pack_vnni(scalar_t* __restrict__ packed, const scalar_t* __restrict__ weight, int N, int K) {
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K / VNNI_BLK; ++k) {
      for (int d = 0; d < VNNI_BLK; ++d) {
        packed[k * N * VNNI_BLK + n * VNNI_BLK + d] = weight[n * K + k * VNNI_BLK + d];
      }
    }
  }
}

} // anonymous namespace

at::Tensor convert_weight_packed(at::Tensor& weight) {
  // weight : [E, OC, IC]
  //     w1 : [E, 2N,  K]
  //     w2 : [E,  K,  N]
  CHECK_DIM(3, weight);
  CHECK_INPUT(weight);
  const auto st = weight.scalar_type();
  const int E = weight.size(0);
  const int OC = weight.size(1);
  const int IC = weight.size(2);

  // we handle 2 TILE_N at a time.
  TORCH_CHECK(OC % TILE_N == 0, "invalid weight out features ", OC);
  TORCH_CHECK(IC % TILE_K == 0, "invalid weight input features ", IC);

  constexpr int BLOCK_N = block_size_n();

  // use phony sizes here [E, OC, IC], for each [E], [OC, IC] -> [IC / 2, OC, 2]
  auto packed_weight = at::empty({E, OC, IC}, weight.options());
  const int stride = OC * IC;

  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "expect weight to be bfloat16, float16.");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(st, "convert_weight_packed", [&] {
    const scalar_t* w_data = weight.data_ptr<scalar_t>();
    scalar_t* packed_data = packed_weight.data_ptr<scalar_t>();

    // parallel on {E}
    at::parallel_for(0, E, 0, [&](int begin, int end) {
      for (int e = begin; e < end; ++e) {
        for (int n = 0; n < OC; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, OC - n);
          pack_vnni<scalar_t>(
              packed_data + e * stride + n * IC,
              w_data + e * stride + n * IC,
              n_size,
              IC);
        }
      }
    });
  });
  return packed_weight;
}
