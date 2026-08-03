#pragma once
#include "cute/tensor.hpp"
#include "es_sm100_mxfp8_blockscaled_moe_group_gemm_traits.cuh"
#include <cuda.h>

namespace expert_specialization {

using namespace cute;

template <typename GemmTraits>
struct Sm100Mxfp8BlockScaledMoeGroupGemmOffsetFunctor {
  using ElementB = typename GemmTraits::Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementD;
  // Input
  int* expert_offsets{nullptr};
  int* blockscale_offsets{nullptr};
  // Output
  ElementB* b_base{nullptr};
  ElementSF* sfb_base{nullptr};
  ElementD* d_base{nullptr};
  ElementB** b_offsets{nullptr};
  ElementSF** sfb_offsets{nullptr};
  ElementD** d_offsets{nullptr};

  Sm100Mxfp8BlockScaledMoeGroupGemmOffsetFunctor() = default;
  Sm100Mxfp8BlockScaledMoeGroupGemmOffsetFunctor(
      int* _expert_offsets,
      int* _blockscale_offsets,
      ElementB* _b_base,
      ElementSF* _sfb_base,
      ElementD* _d_base,
      ElementB** _b_offsets,
      ElementSF** _sfb_offsets,
      ElementD** _d_offsets)
      : expert_offsets{_expert_offsets},
        blockscale_offsets{_blockscale_offsets},
        b_base(_b_base),
        sfb_base(_sfb_base),
        d_base(_d_base),
        b_offsets(_b_offsets),
        sfb_offsets(_sfb_offsets),
        d_offsets(_d_offsets) {}

  void CUTE_DEVICE operator()(int expert_id, int m, int k) {
    int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
    int64_t blockscale_offset = static_cast<int64_t>(blockscale_offsets[expert_id]);
    int64_t b_stride = expert_offset * k;
    int64_t sfb_stride = blockscale_offset * (k / 32);
    int64_t d_stride = expert_offset * m;

    b_offsets[expert_id] = b_base + b_stride;
    sfb_offsets[expert_id] = sfb_base + sfb_stride;
    d_offsets[expert_id] = d_base + d_stride;
  }
};

template <typename OffsetFunctor>
__global__ void sm100_mxfp8_blockscaled_moe_group_gemm_pre_compute_kernel(OffsetFunctor offset_functor, int m, int k) {
  int expert_id = static_cast<int>(threadIdx.x);
  offset_functor(expert_id, m, k);
}

}  // namespace expert_specialization
