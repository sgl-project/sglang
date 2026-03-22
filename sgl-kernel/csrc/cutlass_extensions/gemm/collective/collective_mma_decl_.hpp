// Support scale B from cutlass
#pragma once

#include <cute/numeric/integral_constant.hpp>
#include <cutlass/detail/dependent_false.hpp>

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy,
    class TileShape,
    class ElementA,
    class StrideA,
    class ElementB,
    class StrideB,
    class TiledMma,
    class GmemTiledCopyA,
    class SmemLayoutAtomA,
    class SmemCopyAtomA,
    class TransformA,
    class GmemTiledCopyB,
    class SmemLayoutAtomB,
    class SmemCopyAtomB,
    class TransformB>
struct CollectiveMma_ {
  static_assert(cutlass::detail::dependent_false<ElementA>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective
