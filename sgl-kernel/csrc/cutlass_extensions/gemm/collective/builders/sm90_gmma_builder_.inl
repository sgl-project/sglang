// Support scale B from cutlass
#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Returns the maximum number of smem tiles that can be used with a given smem capacity (with an optional scale matrix),
// or overrides with manual count.
template <
    int capacity_bytes_,
    class ElementA,
    class ElementB,
    class ElementScaleA,
    class ElementScaleB,
    class ElementZero,
    class TileShapeMNK,
    int carveout_bytes_,
    int alignment = 128>
constexpr int
compute_stage_count_or_override_single_affine_transformed_input_(StageCountAutoCarveout<carveout_bytes_> stage_count) {
  // 32 bytes to account for barriers etc.
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
  constexpr int scale_zero_k_tile = 1;
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto sA_bits = get_bits_for_possibly_void_element<ElementScaleA>();
  constexpr auto sB_bits = get_bits_for_possibly_void_element<ElementScaleB>();
  constexpr auto z_bits = get_bits_for_possibly_void_element<ElementZero>();

  constexpr auto scaleA_bytes = cutlass::bits_to_bytes(sA_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);
  constexpr auto scaleB_bytes =
      cutlass::bits_to_bytes(sB_bits * size<1>(TileShapeMNK{}) * 4 * scale_zero_k_tile);  // K need padding to 4
  constexpr auto zero_bytes = cutlass::bits_to_bytes(z_bits * size<0>(TileShapeMNK{}) * scale_zero_k_tile);

  static_assert(scaleA_bytes % 128 == 0, "Scale A bytes must be a multiple of 128");
  static_assert(scaleB_bytes % 128 == 0, "Scale B bytes must be a multiple of 128");
  static_assert(zero_bytes % 128 == 0, "Zero bytes must be a multiple of 128");

  // When scales are void, s_bits will be 0 so no smem will be allocated for scales.
  constexpr int stage_bytes_ = cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
                               cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
                               scaleA_bytes + scaleB_bytes + zero_bytes;

  constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) + static_cast<int>(mainloop_pipeline_bytes);
  constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
  constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;

  return (capacity_bytes - carveout_bytes) / stage_bytes;
}

}  // namespace detail
}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
