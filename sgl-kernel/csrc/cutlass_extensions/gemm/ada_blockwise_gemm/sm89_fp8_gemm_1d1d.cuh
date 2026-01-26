/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "sm89_utils.cuh"

namespace ada_blockwise_gemm {

template <typename GemmKernel>
CUTLASS_GLOBAL void sm89_fp8_gemm_1d1d_impl(
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    void const* A,
    void const* B,
    void* D,
    float const* scales_a,
    float const* scales_b) {
  GemmKernel op;
  op.invoke(shape_m, shape_n, shape_k, A, B, D, scales_a, scales_b);
}

template <typename GemmKernel>
CUTLASS_GLOBAL void sm89_fp8_bmm_1d1d_impl(
    uint32_t shape_m,
    uint32_t shape_n,
    uint32_t shape_k,
    __nv_fp8_e4m3* A,
    __nv_fp8_e4m3* B,
    __nv_bfloat16* D,
    float* scales_a,
    float* scales_b,
    uint64_t stride_a,
    uint64_t stride_b,
    uint64_t stride_d,
    uint64_t stride_scales_a,
    uint64_t stride_scales_b) {
  GemmKernel op;

  auto ptr_a = reinterpret_cast<typename GemmKernel::ElementInput const*>(A + blockIdx.z * stride_a);
  auto ptr_b = reinterpret_cast<typename GemmKernel::ElementInput const*>(B + blockIdx.z * stride_b);
  auto ptr_scale_a =
      reinterpret_cast<typename GemmKernel::ElementBlockScale const*>(scales_a + blockIdx.z * stride_scales_a);
  auto ptr_scale_b =
      reinterpret_cast<typename GemmKernel::ElementBlockScale const*>(scales_b + blockIdx.z * stride_scales_b);
  auto ptr_output = reinterpret_cast<typename GemmKernel::ElementOutput*>(D + blockIdx.z * stride_d);

  op(ptr_a, ptr_b, ptr_scale_a, ptr_scale_b, ptr_output, shape_m, shape_n, shape_k);
}

template <typename KT>
struct AdaBlockwiseGemmKernel {
  using SharedStorage = typename KT::SharedStorage;
  using ElementInput = typename KT::ElementInput;
  using ElementOutput = typename KT::ElementOutput;
  using ElementBlockScale = typename KT::ElementBlockScale;

  // Factory invocation
  CUTLASS_DEVICE
  void invoke(
      uint32_t shape_m,
      uint32_t shape_n,
      uint32_t shape_k,
      void const* A,
      void const* B,
      void* D,
      float const* scales_a,
      float const* scales_b) {
    auto ptr_a = reinterpret_cast<ElementInput const*>(A);
    auto ptr_b = reinterpret_cast<ElementInput const*>(B);
    auto ptr_scale_a = reinterpret_cast<ElementBlockScale const*>(scales_a);
    auto ptr_scale_b = reinterpret_cast<ElementBlockScale const*>(scales_b);
    auto ptr_output = reinterpret_cast<ElementOutput*>(D);

    (*this)(ptr_a, ptr_b, ptr_scale_a, ptr_scale_b, ptr_output, shape_m, shape_n, shape_k);
  }

  CUTE_DEVICE auto gmem_tensor_init(
      typename KT::ElementInput const* ptr_a,
      typename KT::ElementInput const* ptr_b,
      typename KT::ElementBlockScale const* ptr_scale_a,
      typename KT::ElementBlockScale const* ptr_scale_b,
      uint32_t M,
      uint32_t N,
      uint32_t K,
      int* SharedStorageBase) {
    using X = cute::Underscore;

    uint32_t const ScaleM = (((M + 3) >> 2) << 2);  // align 4
    uint32_t const ScaleN = (N + KT::ScaleGranularityN - 1) / KT::ScaleGranularityN;
    uint32_t const ScaleK = (K + KT::ScaleGranularityK - 1) / KT::ScaleGranularityK;

    auto mA_mk =
        cute::make_tensor(cute::make_gmem_ptr(ptr_a), cute::make_shape(M, K), cute::make_stride(K, cute::_1{}));

    auto mB_nk =
        cute::make_tensor(cute::make_gmem_ptr(ptr_b), cute::make_shape(N, K), cute::make_stride(K, cute::_1{}));

    auto mSFA_mk = cute::make_tensor(
        cute::make_gmem_ptr(ptr_scale_a), cute::make_shape(ScaleM, ScaleK), cute::make_stride(cute::_1{}, ScaleM));

    auto mSFB_nk = cute::make_tensor(
        cute::make_gmem_ptr(ptr_scale_b), cute::make_shape(ScaleN, ScaleK), cute::make_stride(ScaleK, cute::_1{}));

    auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);                               // (m,n,k)
    auto gA = cute::local_tile(mA_mk, typename KT::TileShape{}, cta_coord, cute::Step<_1, X, _1>{});  // (BLK_M,BLK_K,k)
    auto gB = cute::local_tile(mB_nk, typename KT::TileShape{}, cta_coord, cute::Step<X, _1, _1>{});  // (BLK_N,BLK_K,k)
    auto gSFA = cute::local_tile(
        mSFA_mk, typename KT::ScalePerTileShape{}, cta_coord, cute::Step<_1, X, _1>{});  // (BLK_M,BLK_K)
    auto gSFB = cute::local_tile(
        mSFB_nk, typename KT::ScalePerTileShape{}, cta_coord, cute::Step<X, _1, _1>{});  // (BLK_N,BLK_K)

    typename KT::SharedStorageLoad* load_storage = reinterpret_cast<typename KT::SharedStorageLoad*>(SharedStorageBase);
    auto sA = cute::make_tensor(cute::make_smem_ptr(load_storage->smem_a.data()), typename KT::SmemLayoutA{});
    auto sB = cute::make_tensor(cute::make_smem_ptr(load_storage->smem_b.data()), typename KT::SmemLayoutB{});
    auto sSFA = cute::make_tensor(cute::make_smem_ptr(load_storage->smem_sfa.data()), typename KT::SmemLayoutSFA{});
    auto sSFB = cute::make_tensor(cute::make_smem_ptr(load_storage->smem_sfb.data()), typename KT::SmemLayoutSFB{});

    return cute::make_tuple(gA, gB, gSFA, gSFB, sA, sB, sSFA, sSFB);
  }

  template <class Accumulator, class SharedStorage, class ElementOutput>
  CUTE_DEVICE void
  epilogue_with_smem(Accumulator& accum, SharedStorage& shared_storage, ElementOutput* o, int M, int N) {
    // convert type
    auto epi = cute::make_fragment_like<ElementOutput>(accum);
    cute::for_each(cute::make_int_sequence<cute::size(epi)>{}, [&](auto i) { epi(i) = ElementOutput(accum(i)); });

    auto sO = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_o.data()), typename KT::SmemLayoutO{});
    // copy rf -> smem
    typename KT::TiledMma mma;
    auto tiled_copy_R2S = cute::make_tiled_copy_C(typename KT::SmemCopyAtomR2S{}, mma);
    auto thr_copy_R2S = tiled_copy_R2S.get_slice(threadIdx.x);
    auto tRS_rO = thr_copy_R2S.retile_S(epi);
    auto tRS_sO = thr_copy_R2S.partition_D(sO);

    cute::copy(tiled_copy_R2S, tRS_rO, tRS_sO);
    __syncthreads();

    // copy smem -> rf
    typename KT::TiledCopyS2G tiled_copy_S2G;
    auto thr_copy_S2G = tiled_copy_S2G.get_slice(threadIdx.x);
    auto tSR_sO = thr_copy_S2G.partition_S(sO);
    auto tSR_rO = cute::make_tensor<KT::ElementOutput>(cute::shape(tSR_sO));

    cute::copy(tiled_copy_S2G, tSR_sO, tSR_rO);
    __syncthreads();

    // copy rf -> gmem
    auto mO = cute::make_tensor(cute::make_gmem_ptr(o), cute::make_shape(M, N), cute::make_stride(N, cute::_1{}));
    auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);
    auto gO = cute::local_tile(mO, typename KT::TileShape{}, cta_coord, cute::Step<cute::_1, cute::_1, X>{});
    auto cO = cute::make_identity_tensor(cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));
    auto tRG_rO = thr_copy_S2G.retile_S(tSR_rO);
    auto tRG_gO = thr_copy_S2G.partition_D(gO);
    auto tRG_cO = thr_copy_S2G.partition_D(cO);

    int residue_m = M - KT::kTileM * blockIdx.x;
    int residue_n = N - KT::kTileN * blockIdx.y;
    CUTE_UNROLL
    for (int m = 0; m < cute::size<1>(tRG_gO); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < cute::size<2>(tRG_gO); ++n) {
        if (cute::get<0>(tRG_cO(0, m, n)) < residue_m && cute::get<1>(tRG_cO(0, m, n)) < residue_n) {
          cute::copy(typename KT::GmemCopyAtomR2G{}, tRG_rO(cute::_, m, n), tRG_gO(cute::_, m, n));
        }
      }
    }
  }

  template <class TensorD, class TensorC, class TensorScale, class Index>
  CUTE_DEVICE void promote(TensorD& accum, TensorC const& temp_accum, TensorScale const& scale, Index n_block) {
    using AccumType = typename TensorD::value_type;
    for (int mma_m = 0; mma_m < cute::get<1>(cute::shape<0>(accum)); ++mma_m) {
      CUTE_UNROLL
      for (int mma_n = 0; mma_n < cute::get<0>(cute::shape<0>(accum)); ++mma_n) {
        CUTE_UNROLL
        for (int mma_iter_m = 0; mma_iter_m < cute::size<1>(accum); ++mma_iter_m) {
          CUTE_UNROLL
          for (int mma_iter_n = 0; mma_iter_n < cute::size<2>(accum); ++mma_iter_n) {
            auto coord_d = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n, n_block);
            auto coord_c = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n);
            accum(coord_d) += temp_accum(coord_c) * scale(mma_m, mma_iter_m, cute::_0{});
          }
        }
      }
    }
  }

  /// Executes one GEMM
  CUTE_DEVICE
  void operator()(
      typename KT::ElementInput const* ptr_a,
      typename KT::ElementInput const* ptr_b,
      typename KT::ElementBlockScale const* ptr_scale_a,
      typename KT::ElementBlockScale const* ptr_scale_b,
      typename KT::ElementOutput* ptr_output,
      uint32_t M,
      uint32_t N,
      uint32_t K) {
    // Dynamic shared memory base pointer
    extern __shared__ int SharedStorageBase[];

    auto [gA, gB, gSFA, gSFB, sA, sB, sSFA, sSFB] =
        gmem_tensor_init(ptr_a, ptr_b, ptr_scale_a, ptr_scale_b, M, N, K, SharedStorageBase);
    typename KT::GmemTiledCopyA g2s_copy_A;
    typename KT::GmemTiledCopyB g2s_copy_B;
    auto g2s_thr_copy_A = g2s_copy_A.get_slice(threadIdx.x);
    auto g2s_thr_copy_B = g2s_copy_B.get_slice(threadIdx.x);

    auto tAgA = g2s_thr_copy_A.partition_S(gA);  // (ACPY,ACPY_M,ACPY_K,k)
    auto tAsA = g2s_thr_copy_A.partition_D(sA);  // (ACPY,ACPY_M,ACPY_K,Stage)
    auto tBgB = g2s_thr_copy_B.partition_S(gB);  // (BCPY,BCPY_N,BCPY_K,k)
    auto tBsB = g2s_thr_copy_B.partition_D(sB);  // (BCPY,BCPY_N,BCPY_K,Stage)

    typename KT::GmemTiledCopySFA g2s_copy_SFA;
    typename KT::GmemTiledCopySFB g2s_copy_SFB;
    auto g2s_thr_copy_SFA = g2s_copy_SFA.get_slice(threadIdx.x);
    auto g2s_thr_copy_SFB = g2s_copy_SFB.get_slice(threadIdx.x);

    auto tAgSFA = g2s_thr_copy_SFA.partition_S(gSFA);  // (ACPY,ACPY_M,ACPY_K,Stage)
    auto tAsSFA = g2s_thr_copy_SFA.partition_D(sSFA);  // (ACPY,ACPY_M,ACPY_K,Stage)
    auto tBgSFB = g2s_thr_copy_SFB.partition_S(gSFB);  // (BCPY,BCPY_N,BCPY_K,Stage)
    auto tBsSFB = g2s_thr_copy_SFB.partition_D(sSFB);  // (BCPY,BCPY_N,BCPY_K,Stage)

    auto cA = make_identity_tensor(cute::make_shape(cute::size<0>(sA), cute::size<1>(sA)));
    auto tAcA = g2s_thr_copy_A.partition_S(cA);

    auto cB = make_identity_tensor(cute::make_shape(cute::size<0>(sB), cute::size<1>(sB)));
    auto tBcB = g2s_thr_copy_B.partition_S(cB);

    auto cSFA = cute::make_identity_tensor(typename KT::GmemTiledCopySFA::Tiler_MN{});
    auto tAcSFA = g2s_thr_copy_SFA.partition_S(cSFA);

    int residue_m = M - KT::kTileM * blockIdx.x;
    int residue_n = N - KT::kTileN * blockIdx.y;
    residue_m = residue_m > KT::kTileM ? KT::kTileM : residue_m;
    residue_n = residue_n > KT::kTileN ? KT::kTileN : residue_n;

    auto tApA = cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(tAsA), cute::size<2>(tAsA)), cute::Stride<cute::_1, cute::_0>{});
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<0>(tApA); ++m) {
      tApA(m, 0) = cute::get<0>(tAcA(0, m, 0)) < residue_m;  // blk_m coord < residue_m
    }

    auto tBpB = cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(tBsB), cute::size<2>(tBsB)), cute::Stride<cute::_1, cute::_0>{});
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < cute::size<0>(tBpB); ++n) {
      tBpB(n, 0) = cute::get<0>(tBcB(0, n, 0)) < residue_n;  // blk_n coord < residue_n
    }

    auto tApSFA = cute::make_tensor<bool>(
        cute::make_shape(cute::size<1>(tAsSFA), cute::size<2>(tAsSFA)), cute::Stride<cute::_1, cute::_0>{});
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<0>(tApSFA); ++m) {
      tApSFA(m, 0) = cute::get<0>(tAcSFA(0, m, 0)) < residue_m;  // blk_m coord < residue_m
    }

    // prefetch gmem A/B
    cute::clear(tAsA);
    cute::clear(tBsB);
    cute::clear(tAsSFA);
    cute::clear(tBsSFB);

    int k_tile_count = cute::size<2>(gA);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe) {
      if (k_pipe >= k_tile_count) {
        cute::clear(tApA);
        cute::clear(tBpB);
        cute::clear(tApSFA);
      }
      auto k_tile_iter = std::min(k_pipe, k_tile_count - 1);
      cute::copy_if(
          g2s_copy_A, tApA, tAgA(cute::_, cute::_, cute::_, k_tile_iter), tAsA(cute::_, cute::_, cute::_, k_pipe));
      cute::copy_if(
          g2s_copy_B, tBpB, tBgB(cute::_, cute::_, cute::_, k_tile_iter), tBsB(cute::_, cute::_, cute::_, k_pipe));
      cute::copy_if(
          g2s_copy_SFA,
          tApSFA,
          tAgSFA(cute::_, cute::_, cute::_, k_tile_iter),
          tAsSFA(cute::_, cute::_, cute::_, k_pipe));
      cute::copy(
          g2s_copy_SFB, tBgSFB(cute::_, cute::_, cute::_, k_tile_iter), tBsSFB(cute::_, cute::_, cute::_, k_pipe));

      cute::cp_async_fence();
    }

    typename KT::TiledMma mma;
    auto thr_mma = mma.get_slice(threadIdx.x);
    auto accum = cute::partition_fragment_C(
        mma,
        cute::make_shape(
            cute::Int<KT::kTileM>{}, cute::Int<KT::kMmaPermN>{}, cute::Int<KT::NUM_GROUP_N>{}));  // (MMA,MMA_M,MMA_N)
    auto temp = cute::partition_fragment_C(
        mma, cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kMmaPermN>{}));  // (MMA,MMA_M,MMA_N)

    auto mma_shape_A = cute::partition_shape_A(mma, cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileK>{}));
    auto tCrA = cute::make_tensor<typename KT::ElementInput>(mma_shape_A);

    auto mma_shape_B = cute::partition_shape_B(
        mma, cute::make_shape(cute::Int<KT::kMmaPermN>{}, cute::Int<KT::kTileK>{}, cute::Int<KT::NUM_GROUP_N>{}));
    auto tCrB = cute::make_tensor<typename KT::ElementInput>(mma_shape_B);

    auto s2r_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_slice(threadIdx.x);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);  // (CPY,CPY_M,CPY_K,Stage)
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);   // (CPY,CPY_M,CPY_K)
    static_assert(is_static<decltype(tXrA.layout())>::value, "tXrA layout must be static");

    auto s2r_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_slice(threadIdx.x);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);  // (CPY,CPY_N,CPY_K,Stage)
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB)(cute::_, cute::Int<0>{}, cute::_, cute::_);

    typename KT::SmemTiledCopySFA s2r_copy_SFA;
    typename KT::SmemTiledCopySFB s2r_copy_SFB;
    auto s2r_thr_copy_SFA = s2r_copy_SFA.get_slice(threadIdx.x);
    auto s2r_thr_copy_SFB = s2r_copy_SFB.get_slice(threadIdx.x);

    auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
    auto tXrSFA = cute::make_fragment_like(tXsSFA(cute::_, cute::_, cute::_, 0));
    auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
    auto tXrSFB = cute::make_fragment_like(tXsSFB(cute::_, cute::_, cute::_, 0));
    auto scale = cute::make_fragment_like(tXrSFA);

    int smem_pipe_write = KT::Stages - 1;
    int smem_pipe_read = 0;

    auto tXsA_read = tXsA(cute::_, cute::_, cute::_, smem_pipe_read);
    auto tXsB_read = tXsB(cute::_, cute::_, cute::_, smem_pipe_read);
    auto tXsSFA_read = tXsSFA(cute::_, cute::_, cute::_, smem_pipe_read);
    auto tXsSFB_read = tXsSFB(cute::_, cute::_, cute::_, smem_pipe_read);
    cute::cp_async_wait<KT::Stages - 2>();
    __syncthreads();
    // prefetch smem -> rf
    cute::copy(s2r_copy_SFA, tXsSFA_read, tXrSFA);
    cute::copy(s2r_copy_SFB, tXsSFB_read, tXrSFB);
    cute::copy(s2r_copy_A, tXsA_read, tXrA);
    cute::copy(s2r_copy_B, tXsB_read(cute::_, cute::Int<0>{}, cute::_), tXrB(cute::_, cute::_, cute::Int<0>{}));

    cute::clear(accum);
    int k_tile_iter = KT::Stages - 1;
    while (k_tile_iter < k_tile_count) {
      cute::for_each(cute::make_int_sequence<KT::NUM_GROUP_N>{}, [&](auto n_block) {
        if constexpr (n_block == KT::NUM_GROUP_N - 1) {
          tXsA_read = tXsA(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsB_read = tXsB(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsSFA_read = tXsSFA(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsSFB_read = tXsSFB(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 2>();
          __syncthreads();
          cute::copy(s2r_copy_SFA, tXsSFA_read, tXrSFA);
          cute::copy(s2r_copy_SFB, tXsSFB_read, tXrSFB);
        }
        auto n_block_next = (n_block + cute::_1{}) % KT::NUM_GROUP_N;
        cute::copy(s2r_copy_B, tXsB_read(cute::_, n_block_next, cute::_), tXrB(cute::_, cute::_, n_block_next));
        if constexpr (n_block == 0) {
          // gmem -> smem
          cute::copy_if(
              g2s_copy_A,
              tApA,
              tAgA(cute::_, cute::_, cute::_, k_tile_iter),
              tAsA(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy_if(
              g2s_copy_B,
              tBpB,
              tBgB(cute::_, cute::_, cute::_, k_tile_iter),
              tBsB(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy_if(
              g2s_copy_SFA,
              tApSFA,
              tAgSFA(cute::_, cute::_, cute::_, k_tile_iter),
              tAsSFA(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::copy(
              g2s_copy_SFB,
              tBgSFB(cute::_, cute::_, cute::_, k_tile_iter),
              tBsSFB(cute::_, cute::_, cute::_, smem_pipe_write));
          cute::cp_async_fence();
          k_tile_iter++;
          smem_pipe_write = smem_pipe_read;
          ++smem_pipe_read;
          smem_pipe_read = smem_pipe_read == KT::Stages ? 0 : smem_pipe_read;
          cute::for_each(
              cute::make_int_sequence<cute::size(scale)>{}, [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });
        }
        cute::clear(temp);
        cute::gemm(mma, tCrA, tCrB(cute::_, cute::_, cute::_, n_block), temp);
        if constexpr (n_block == KT::NUM_GROUP_N - 1) {
          cute::copy(s2r_copy_A, tXsA_read, tXrA);
        }
        promote(accum, temp, scale, n_block);
      });
    }
    // load tail
    cute::for_each(cute::make_int_sequence<KT::Stages - 2>{}, [&](auto WaitIndex) {
      using WaitIndex_t = decltype(WaitIndex);
      cute::for_each(cute::make_int_sequence<KT::NUM_GROUP_N>{}, [&](auto n_block) {
        if constexpr (n_block == KT::NUM_GROUP_N - 1) {
          tXsA_read = tXsA(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsB_read = tXsB(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsSFA_read = tXsSFA(cute::_, cute::_, cute::_, smem_pipe_read);
          tXsSFB_read = tXsSFB(cute::_, cute::_, cute::_, smem_pipe_read);
          cute::cp_async_wait<KT::Stages - 3 - WaitIndex_t::value>();
          __syncthreads();
          cute::copy(s2r_copy_SFA, tXsSFA_read, tXrSFA);
          cute::copy(s2r_copy_SFB, tXsSFB_read, tXrSFB);
        }
        auto n_block_next = (n_block + cute::_1{}) % KT::NUM_GROUP_N;
        cute::copy(s2r_copy_B, tXsB_read(cute::_, n_block_next, cute::_), tXrB(cute::_, cute::_, n_block_next));
        if constexpr (n_block == 0) {
          ++smem_pipe_read;
          smem_pipe_read = smem_pipe_read == KT::Stages ? 0 : smem_pipe_read;
          cute::for_each(
              cute::make_int_sequence<cute::size(scale)>{}, [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });
        }
        cute::clear(temp);
        cute::gemm(mma, tCrA, tCrB(cute::_, cute::_, cute::_, n_block), temp);
        if constexpr (n_block == KT::NUM_GROUP_N - 1) {
          cute::copy(s2r_copy_A, tXsA_read, tXrA);
        }
        promote(accum, temp, scale, n_block);
      });
    });
    // mma tail
    cute::for_each(cute::make_int_sequence<KT::NUM_GROUP_N>{}, [&](auto n_block) {
      auto n_block_next = (n_block + cute::_1{}) % KT::NUM_GROUP_N;
      cute::copy(s2r_copy_B, tXsB_read(cute::_, n_block_next, cute::_), tXrB(cute::_, cute::_, n_block_next));
      cute::clear(temp);
      if constexpr (n_block == 0) {
        cute::for_each(cute::make_int_sequence<cute::size(scale)>{}, [&](auto i) { scale(i) = tXrSFA(i) * tXrSFB(0); });
      }
      cute::gemm(mma, tCrA, tCrB(cute::_, cute::_, cute::_, n_block), temp);
      promote(accum, temp, scale, n_block);
    });
    // epilogue
    __syncthreads();  // sync before using store smem
    typename KT::SharedStorageStore* store_storage =
        reinterpret_cast<typename KT::SharedStorageStore*>(SharedStorageBase);
    epilogue_with_smem(accum, *store_storage, ptr_output, M, N);
  }
};

}  // namespace ada_blockwise_gemm