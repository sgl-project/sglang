#pragma once

#include <utility>

#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumValid, uint32_t... kIdx, typename FuncT>
CUTLASS_DEVICE void for_each_static_until(std::integer_sequence<uint32_t, kIdx...>,
                                          FuncT&& func) {
    ((kIdx < kNumValid ? func.template operator()<kIdx>() : void()), ...);
}

template <uint32_t... kIdx, typename FuncT>
CUTLASS_DEVICE void for_each_static_prefix(std::integer_sequence<uint32_t, kIdx...>,
                                           const uint32_t& num_valid,
                                           FuncT&& func) {
    using seq_t = std::integer_sequence<uint32_t, kIdx...>;
    constexpr uint32_t kNumIndices = sizeof...(kIdx);
    if constexpr (kNumIndices <= 4) {
        switch (num_valid) {
            case 0: break;
            case 1: for_each_static_until<1>(seq_t(), func); break;
            case 2: for_each_static_until<2>(seq_t(), func); break;
            case 3: for_each_static_until<3>(seq_t(), func); break;
            default: for_each_static_until<kNumIndices>(seq_t(), func); break;
        }
    } else {
        ((kIdx < num_valid ? func.template operator()<kIdx>() : void()), ...);
    }
}

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

} // namespace deep_gemm::utils
