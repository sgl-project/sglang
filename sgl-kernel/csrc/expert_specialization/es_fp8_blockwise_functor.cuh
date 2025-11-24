#pragma once
#include <cuda.h>

#include <iostream>

#include "cute/tensor.hpp"
#include "es_fp8_blockwise_traits.cuh"

namespace expert_specialization {

using namespace cute;

template <typename ElementAB, typename ElementSF, typename ElementD>
struct Fp8BlockwiseGroupedGemmOffsetFunctor {
  // Input
  int* expert_offsets{nullptr};
  // Base pointers
  ElementAB* a_base{nullptr};
  ElementAB* b_base{nullptr};
  ElementD* out_base{nullptr};
  ElementSF* a_scales_base{nullptr};
  ElementSF* b_scales_base{nullptr};

  // Output
  // Pointer Array for A/B
  ElementAB** a_offsets{nullptr};
  ElementAB** b_offsets{nullptr};
  ElementSF** a_scales_offsets{nullptr};
  ElementSF** b_scales_offsets{nullptr};
  ElementD** out_offsets{nullptr};

  Fp8BlockwiseGroupedGemmOffsetFunctor() = default;
  Fp8BlockwiseGroupedGemmOffsetFunctor(
      int* _expert_offsets,
      ElementAB* _a_base,
      ElementAB* _b_base,
      ElementD* _out_base,
      ElementSF* _a_scales_base,
      ElementSF* _b_scales_base,
      ElementAB** _a_offsets,
      ElementAB** _b_offsets,
      ElementSF** _a_scales_offsets,
      ElementSF** _b_scales_offsets,
      ElementD** _out_offsets)
      : expert_offsets(_expert_offsets),
        a_base(_a_base),
        b_base(_b_base),
        out_base(_out_base),
        a_scales_base(_a_scales_base),
        b_scales_base(_b_scales_base),
        a_offsets(_a_offsets),
        b_offsets(_b_offsets),
        a_scales_offsets(_a_scales_offsets),
        b_scales_offsets(_b_scales_offsets),
        out_offsets(_out_offsets) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
    int64_t a_stride = 0;
    int64_t b_stride = 0;
    int64_t a_scale_stride = 0;
    int64_t b_scale_stride = 0;

    a_stride = expert_offset * k;
    b_stride = expert_id * k * n;
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = expert_id * k * n / 128 / 128;

    a_offsets[expert_id] = a_base + a_stride;
    b_offsets[expert_id] = b_base + b_stride;
    a_scales_offsets[expert_id] = a_scales_base + a_scale_stride;
    b_scales_offsets[expert_id] = b_scales_base + b_scale_stride;
    out_offsets[expert_id] = out_base + expert_offset * n;
  }
};

template <typename PerfConfig>
struct Fp8BlockwiseGroupedGemmSFLayoutFunctor {
  using ScaleConfig = typename PerfConfig::ScaleConfig;
  using LayoutSFA = typename PerfConfig::LayoutSFA;
  using LayoutSFB = typename PerfConfig::LayoutSFB;
  LayoutSFA* layout_sfa_base{nullptr};
  LayoutSFB* layout_sfb_base{nullptr};

  Fp8BlockwiseGroupedGemmSFLayoutFunctor() = default;
  Fp8BlockwiseGroupedGemmSFLayoutFunctor(LayoutSFA* _layout_sfa_base, LayoutSFB* _layout_sfb_base)
      : layout_sfa_base(_layout_sfa_base), layout_sfb_base(_layout_sfb_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    LayoutSFA* layout_sfa_ptr = layout_sfa_base + expert_id;
    LayoutSFB* layout_sfb_ptr = layout_sfb_base + expert_id;
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  }
};

// [Unused]: Specialization for Swap A/B
template <>
struct Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigLowMH20> {
  using ScaleConfig = typename PerfConfigLowMH20::ScaleConfig;
  using LayoutSFA = typename PerfConfigLowMH20::LayoutSFA;
  using LayoutSFB = typename PerfConfigLowMH20::LayoutSFB;
  LayoutSFA* layout_sfa_base{nullptr};
  LayoutSFB* layout_sfb_base{nullptr};

  Fp8BlockwiseGroupedGemmSFLayoutFunctor() = default;
  Fp8BlockwiseGroupedGemmSFLayoutFunctor(LayoutSFA* _layout_sfa_base, LayoutSFB* _layout_sfb_base)
      : layout_sfa_base(_layout_sfa_base), layout_sfb_base(_layout_sfb_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    LayoutSFA* layout_sfa_ptr = layout_sfa_base + expert_id;
    LayoutSFB* layout_sfb_ptr = layout_sfb_base + expert_id;
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }
};

template <typename PerfConfig>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor;

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMH20> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m <= 48) {
      // Swap A/B
      problem_sizes[expert_id * 3 + 0] = n;
      problem_sizes[expert_id * 3 + 1] = m;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMHx00> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m <= 32) {
      // Swap A/B
      problem_sizes[expert_id * 3 + 0] = n;
      problem_sizes[expert_id * 3 + 1] = m;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMH20> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m > 48 && m <= 96) {
      problem_sizes[expert_id * 3 + 0] = m;
      problem_sizes[expert_id * 3 + 1] = n;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMHx00> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m > 32 && m <= 64) {
      problem_sizes[expert_id * 3 + 0] = n;
      problem_sizes[expert_id * 3 + 1] = m;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMH20> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m > 96) {
      problem_sizes[expert_id * 3 + 0] = m;
      problem_sizes[expert_id * 3 + 1] = n;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <>
struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMHx00> {
  int* problem_sizes{nullptr};

  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor() = default;
  Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor(int* _problem_sizes) : problem_sizes(_problem_sizes) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    if (m > 64) {
      problem_sizes[expert_id * 3 + 0] = m;
      problem_sizes[expert_id * 3 + 1] = n;
      problem_sizes[expert_id * 3 + 2] = k;
    } else {
      problem_sizes[expert_id * 3 + 0] = 0;
      problem_sizes[expert_id * 3 + 1] = 0;
      problem_sizes[expert_id * 3 + 2] = 0;
    }
  }
};

template <
    typename OffsetFunctor,
    typename ScaleLayoutFunctor,
    typename LowMProblemSizeFilterFunctor,
    typename MiddleMProblemSizeFilterFunctor,
    typename HighMProblemSizeFilterFunctor>
__global__ void groupedGemmPreComputeKernel(
    int* problem_sizes,
    OffsetFunctor offset_functor,
    ScaleLayoutFunctor sf_functor,
    LowMProblemSizeFilterFunctor lm_psf_functor,
    MiddleMProblemSizeFilterFunctor mm_psf_functor,
    HighMProblemSizeFilterFunctor hm_psf_functor) {
  int64_t expert_id = static_cast<int64_t>(threadIdx.x);
  int m = problem_sizes[expert_id * 3 + 0];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];

  offset_functor(expert_id, m, n, k);
  sf_functor(expert_id, m, n, k);
  lm_psf_functor(expert_id, m, n, k);
  mm_psf_functor(expert_id, m, n, k);
  hm_psf_functor(expert_id, m, n, k);
}

}  // namespace expert_specialization
