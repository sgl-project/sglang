/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
#include <random>

#include "cutlass/util/print_error.hpp"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"

using namespace cute;

struct AmpereUnpredicatedFprop {
  //
  // Static config for conv problem shape
  //
  using D = _6;
  using H = _4;
  using W = _4;

  using T = _3;
  using R = _3;
  using S = _3;

  using Z = _4;
  using P = _2;
  using Q = _2;

  using C = _64;
  using K = _128;

  // Tiler config
  using Tiler_K = decltype(cute::min(K{}, _128{}));
  using Tiler_C = decltype(cute::min(C{}, _32{}));
  using Tiler_N = _4;
  using TileM = Tiler_K;
  using TileN = Shape<Tiler_N, Z, P, Q>;
  using TileK = Shape<Tiler_C,_1,_1,_1>;
  using PIPE  = _3;
  using TilerFlt = Shape<TileM, TileK>;
  using TilerAct = Shape<TileN, TileK>;
  using TilerOut = Shape<TileM, TileN>;

  using TileSizeM = Int<size(TileM{})>;
  using TileSizeN = Int<size(TileN{})>;
  using TileSizeK = Int<size(TileK{})>;
  static constexpr int Stages = PIPE::value;

  using ElementFlt = tfloat32_t;
  using ElementAct = tfloat32_t;
  using ElementOut = float;

  using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
    Layout<Shape<_2,_2,_1>>,
    Tile<_32,_32,Underscore>>;

  static constexpr int MaxThreadsPerBlock = size(TiledMma{});
  static constexpr int MinBlocksPerMultiprocessor = 1;

  union SharedStorage {
    struct {
      ElementFlt sAMatrix[size(TileM{}) * size(TileK{}) * size(PIPE{})];
      ElementAct sBMatrix[size(TileN{}) * size(TileK{}) * size(PIPE{})];
    } mainloop;

    struct {
      ElementOut sCMatrix[size(TileM{}) * size(TileN{})];
    } epilogue;
  };

  //
  // Stencil tensor
  //

  using GmemLayoutFlt = decltype(make_ordered_layout(
    Shape< K, Shape< C, T, R, S>>{},
    tuple<_4, tuple<_0,_3,_2,_1>>{}));

  // We have 64 elements * 32b each in the major mode that we can vectorize
  // Max vector size is 128b, so lay 16 threads along the major mode with a vector size of 4
  // Rest along the minor mode
  using GmemTiledCopyFlt = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementFlt>{},
    Layout<Shape <_16, _8>,
           Stride< _8, _1>>{},
    Layout<Shape < _1, _4>>{}));

  // Following layout is also correct, but trades off dynamic strides in the slice for bank conflict free accesses
  // using SmemLayoutFlt = decltype(
  //     composition(Swizzle<3,2,3>{},
  //                 make_ordered_layout(
  //                     Shape<TileSizeM,TileSizeK,PIPE>{},
  //                     tuple<       _1,       _0,  _2>{})));

  using SmemLayoutAtomFlt = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape <_8,Shape <_4, _2>>,
                       Stride<_4,Stride<_1,_32>>>{}));

  using SmemCopyAtomFlt = Copy_Atom<SM75_U32x4_LDSM_N, ElementFlt>;

  //
  // Activation tensor
  //

  // Activation tensor is major in the contraction mode, so vectorize that mode first
  // Then lay out the rest of the threads along the other mode
  using GmemTiledCopyAct = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementAct>{},
    Layout<Shape <_16, _8>,
           Stride< _8, _1>>{},
    Layout<Shape < _1, _4>>{}));

  // Following layout is also correct, but trades off dynamic strides in the slice for bank conflict free accesses
  // using SmemLayoutAct = decltype(
  //     composition(Swizzle<3,2,3>{},
  //                 make_ordered_layout(
  //                     Shape<TileSizeN,TileSizeK,PIPE>{},
  //                     tuple<       _1,       _0,  _2>{})));

  using SmemLayoutAtomAct = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape <_8,Shape <_4, _2>>,
                       Stride<_4,Stride<_1,_32>>>{}));

  using SmemCopyAtomAct = Copy_Atom<SM75_U32x4_LDSM_N, ElementAct>;

  //
  // Output tensor
  //

  using GmemTiledCopyOut = decltype(make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, ElementAct>{},
    Layout<Shape <_8, _16>,
           Stride<_1,  _8>>{},
    Layout<Shape <_4,  _1>>{}));

  using SmemCopyAtomOut = Copy_Atom<UniversalCopy<uint32_t>, ElementOut>;

  // This can be optimized to make accesses BCF, but we use a col-major layout here to show off composability
  using SmemLayoutOut = Layout<Shape<TileSizeM, TileSizeN>>;

  //
  // Conv functor
  //
  template <class EngineFlt, class TensorActivation, class TensorOutput>
  void __device__
  operator()(cute::Tensor<EngineFlt, GmemLayoutFlt> mFlt, // ( K,        (C,T,R,S))
             TensorActivation                       mAct, // ((N,Z,P,Q), (C,T,R,S))
             TensorOutput                           mOut, // ( K,        (N,Z,P,Q))
             char* smem_buf) const {
    using namespace cute;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm80CpAsyncUnpredicated<PIPE::value>,
        Shape<TileM,TileN,TileK>,
        ElementFlt,
        Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
        ElementAct,
        Underscore, // Ignore the stride, we are passing full cute::Tensor to operator()
        TiledMma,
        GmemTiledCopyFlt,
        SmemLayoutAtomFlt,
        SmemCopyAtomFlt,
        cute::identity,
        GmemTiledCopyAct,
        SmemLayoutAtomAct,
        SmemCopyAtomAct,
        cute::identity>;

    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, TilerOut{});
    clear(accum);

    // Set up tensors
    // NOTE: blockIdx.x projects onto act-NDHW mode, y along the flt-K mode for the sake of higher dynamic range in NDHW
    Tensor gA_mk = local_tile(mFlt, TilerFlt{}, make_coord(_,_));                              // (BLK_M,BLK_K,m',k')
    Tensor gB_nk = local_tile(mAct, TilerAct{}, make_coord(_,_));                              // (BLK_N,BLK_K,n',_1)
    Tensor gC_mn = local_tile(mOut, TilerOut{}, make_coord(_,_));                              // (BLK_M,BLK_N,m',n')

    // Compute m_coord and n_coord with their post-tiled shapes
    auto m_coord = idx2crd(int(blockIdx.y), shape<2>(gA_mk));
    auto n_coord = idx2crd(int(blockIdx.x), shape<2>(gB_nk));
    Tensor gA = gA_mk(_,_,m_coord,_);                                                          // (BLK_M,BLK_K,k')
    Tensor gB = gB_nk(_,_,n_coord,_);                                                          // (BLK_N,BLK_K,_1)
    Tensor gC = gC_mn(_,_,m_coord,n_coord);                                                    // (BLK_M,BLK_N)

    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);

    CollectiveMainloop collective_mma;
    collective_mma(
      accum,
      gA,
      gB,
      accum,
      k_tile_iter, k_tile_count,
      Underscore{}, // no residue since we do not support predication
      threadIdx.x,
      smem_buf);

    //
    // Epilogue
    //
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sC = make_tensor(make_smem_ptr(&storage.epilogue.sCMatrix[0]), SmemLayoutOut{});

    auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomOut{}, tiled_mma);
    auto smem_thr_copy_C = smem_tiled_copy_C.get_slice(threadIdx.x);
    auto tCrC = smem_thr_copy_C.retile_S(accum);
    auto tCsC = smem_thr_copy_C.partition_D(sC);
    copy(smem_tiled_copy_C, tCrC, tCsC);

    __syncthreads();

    GmemTiledCopyOut gmem_tiled_copy_C;
    auto gmem_thr_copy_C = gmem_tiled_copy_C.get_slice(threadIdx.x);
    auto tDsC = gmem_thr_copy_C.partition_S(sC);
    auto tDgC = gmem_thr_copy_C.partition_D(gC);
    copy(gmem_tiled_copy_C, tDsC, tDgC);

    #if 0
      if (thread0()) {
        print("mAct = "); print(mAct);          print('\n');
        print("mFlt = "); print(mFlt);          print('\n');
        print("mOut = "); print(mOut);          print('\n');
        print("gA   = "); print(gA);            print('\n');
        print("gB   = "); print(gB);            print('\n');
        print("gC   = "); print(gC);            print('\n');
        print("sA   = "); print(sA.layout());   print('\n');
        print("sB   = "); print(sB.layout());   print('\n');
        print("sC   = "); print(sC.layout());   print('\n');
        print("tAgA = "); print(tAgA.layout()); print('\n');
        print("tBgB = "); print(tBgB.layout()); print('\n');
        print("tAsA = "); print(tAsA.layout()); print('\n');
        print("tBsB = "); print(tBsB.layout()); print('\n');
        print("tCsA = "); print(tCsA.layout()); print('\n');
        print("tCsB = "); print(tCsB.layout()); print('\n');
        print("tCrC = "); print(tCrC.layout()); print('\n');
        print("tCsC = "); print(tCsC.layout()); print('\n');
        print("tDsC = "); print(tDsC.layout()); print('\n');
        print("tDgC = "); print(tDgC.layout()); print('\n');
        print("gmem tiled copy A = "); print(gmem_tiled_copy_A); print('\n');
        print("gmem tiled copy B = "); print(gmem_tiled_copy_B); print('\n');
        print("gmem tiled copy C = "); print(gmem_tiled_copy_C); print('\n');
        print("k_tile_count = "); print(size<2>(gA)); print('\n');
        print("k_tile_iter  = "); print(*k_tile_iter); print('\n');
        print("K_BLOCK_MAX  = "); print(K_BLOCK_MAX); print('\n');
    }
    #endif
  }
};

template <class TensorFlt, class TensorAct, class TensorOut>
inline int
fprop_reference(
    TensorFlt mStencil,    // Logical MK: ( K,        (C,T,R,S))
    TensorAct mActivation, // Logical NK: ((N,Z,P,Q), (C,T,R,S))
    TensorOut mOutput,     // Logical MN: ( K,        (N,Z,P,Q))
    TensorOut mOutputRef) {
  int32_t N = size<1,0>(mOutputRef); 
  int32_t Z = size<1,1>(mOutputRef);
  int32_t P = size<1,2>(mOutputRef);
  int32_t Q = size<1,3>(mOutputRef);
  int32_t T = size<1,3>(mStencil);
  int32_t R = size<1,2>(mStencil);
  int32_t S = size<1,1>(mStencil);
  int32_t C = size<1,0>(mStencil);

  size_t K    = static_cast<size_t>(size<0>(mOutputRef));
  size_t NZPQ = static_cast<size_t>(size<1>(mOutputRef));
  size_t CTRS = static_cast<size_t>(size<1>(mStencil));

#if defined(_OPENMP)
  #pragma omp parallel for
#endif
  for (size_t logical_m = 0; logical_m < K; ++logical_m) {
    for (size_t logical_n = 0; logical_n < NZPQ; ++logical_n) {
      auto accumulator = float(0);
      for (size_t logical_k = 0; logical_k < CTRS; ++logical_k) {
        accumulator += mStencil(logical_m, logical_k) * mActivation(logical_n, logical_k);
      }
      mOutputRef(logical_m, logical_n) = accumulator;
    }
  }

  return print_relative_error(mOutput, mOutputRef,  /*print_verbose*/ false,  /*print_error*/ true, /*error_margin*/ 0.01);
}
