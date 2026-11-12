#pragma once

#include <cute/util/type_traits.hpp>
#include <cutlass/float_subbyte.h>

namespace deep_gemm {

template <typename dtype_t>
constexpr CUTLASS_HOST_DEVICE uint32_t get_smem_pack_factor() {
    // Packed FP4 stores two logical elements per byte in shared memory.
    return cute::is_same_v<dtype_t, cutlass::float_e2m1_t> ? 2 : 1;
}

} // namespace deep_gemm
