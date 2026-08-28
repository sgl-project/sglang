/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace sglang::sm100_q8kv8 {

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

CUTE_DEVICE void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTE_DEVICE void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE void umma_arrive(transac_bar_t& bar) {
  uint32_t const addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" : : "r"(addr));
}

template <class TiledMMA, class TensorA, class TensorB, class TensorC>
CUTE_DEVICE void umma_ss(TiledMMA& tiled_mma, TensorA s_a, TensorB s_b, TensorC t_c, bool clear_accum) {
  using namespace cute;
  tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
  auto thr_mma = tiled_mma.get_slice(_0{});
  auto a_frag = thr_mma.partition_fragment_A(s_a);
  auto b_frag = thr_mma.partition_fragment_B(s_b);
  static_assert(size<2>(a_frag) == size<2>(b_frag));
  CUTE_UNROLL
  for (int k = 0; k < size<2>(a_frag); ++k) {
    cute::gemm(tiled_mma, a_frag(_, _, k), b_frag(_, _, k), t_c);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

template <int N>
CUTE_DEVICE void tmem_load(uint32_t col, float* values) {
  static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64 || N == 128);
  auto* data = reinterpret_cast<uint32_t*>(values);
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (N == 1) {
      cute::SM100_TMEM_LOAD_32dp32b1x::copy(col, data[Is]...);
    } else if constexpr (N == 2) {
      cute::SM100_TMEM_LOAD_32dp32b2x::copy(col, data[Is]...);
    } else if constexpr (N == 4) {
      cute::SM100_TMEM_LOAD_32dp32b4x::copy(col, data[Is]...);
    } else if constexpr (N == 8) {
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(col, data[Is]...);
    } else if constexpr (N == 16) {
      cute::SM100_TMEM_LOAD_32dp32b16x::copy(col, data[Is]...);
    } else if constexpr (N == 32) {
      cute::SM100_TMEM_LOAD_32dp32b32x::copy(col, data[Is]...);
    } else if constexpr (N == 64) {
      cute::SM100_TMEM_LOAD_32dp32b64x::copy(col, data[Is]...);
    } else if constexpr (N == 128) {
      cute::SM100_TMEM_LOAD_32dp32b128x::copy(col, data[Is]...);
    }
  }(cute::make_index_sequence<N>{});
}

template <int N>
CUTE_DEVICE void tmem_store(uint32_t col, float const* values) {
  static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64 || N == 128);
  auto const* data = reinterpret_cast<uint32_t const*>(values);
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (N == 1) {
      cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., col);
    } else if constexpr (N == 2) {
      cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., col);
    } else if constexpr (N == 4) {
      cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., col);
    } else if constexpr (N == 8) {
      cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., col);
    } else if constexpr (N == 16) {
      cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., col);
    } else if constexpr (N == 32) {
      cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., col);
    } else if constexpr (N == 64) {
      cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., col);
    } else if constexpr (N == 128) {
      cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., col);
    }
  }(cute::make_index_sequence<N>{});
}

}  // namespace sglang::sm100_q8kv8
