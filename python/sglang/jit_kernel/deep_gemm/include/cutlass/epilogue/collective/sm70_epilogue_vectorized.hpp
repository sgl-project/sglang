/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class StrideC,
  class StrideD,
  class ThreadEpilogueOp,
  class SmemLayout,
  class CopyAtomR2S,
  class TiledCopyS2R,
  class CopyAtomR2G,
  class EpilogueScheduleType = EpilogueSimtVectorized,
  class Enable = void
>
class Epilogue {
  static_assert(cute::is_same_v<EpilogueScheduleType, EpilogueSimtVectorized> ||
                cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized>,
                "Could not find an epilogue specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Epilogue Vectorized
/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class SmemLayout_,
  class CopyAtomR2S_,
  class TiledCopyS2R_,
  class CopyAtomR2G_,
  class EpilogueScheduleType_
>
class Epilogue<
        StrideC_,
        StrideD_,
        ThreadEpilogueOp_,
        SmemLayout_,
        CopyAtomR2S_,
        TiledCopyS2R_,
        CopyAtomR2G_,
        EpilogueScheduleType_,
        cute::enable_if_t<
          cute::is_same_v<EpilogueScheduleType_, EpilogueSimtVectorized>
        >
      > {
public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using ElementBias = typename detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::type;
  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = CopyAtomR2G;

  static constexpr bool IsEpilogueBiasSupported = detail::IsThreadEpilogueOpWithBias<ThreadEpilogueOp>::value;
  using StrideBias = cute::conditional_t<detail::is_m_major<StrideD>(), Stride<_1,_0,int64_t>, Stride<_0,_1,int64_t>>;

  static_assert(cute::rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  static constexpr bool IsActHasArgs = detail::IsThreadEpilogueOpWithElementwiseArguments<ThreadEpilogueOp>::value;

  // Host side epilogue arguments
  template<class ThreadEpiOp, class = void>
  struct ThreadEpilogueOpArguments {
    ElementScalar alpha{0};
    ElementScalar beta{0};
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias{};
  };

  template<class ThreadEpiOp>
  struct ThreadEpilogueOpArguments<
          ThreadEpiOp,
          cute::enable_if_t<detail::IsThreadEpilogueOpWithElementwiseArguments<ThreadEpiOp>::value>> {
    ElementScalar alpha{0};
    ElementScalar beta{0};
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias{};
    typename ThreadEpiOp::ElementwiseArguments activation{};
  };

  struct Arguments {
    ThreadEpilogueOpArguments<ThreadEpilogueOp> thread{};
    using StrideBias = decltype(thread.dBias);
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  template<class ThreadEpiOp, class = void>
  struct ParamsType {
    typename ThreadEpiOp::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
    StrideBias dBias{};
  };

  template<class ThreadEpiOp>
  struct ParamsType<
          ThreadEpiOp,
          cute::enable_if_t<detail::IsThreadEpilogueOpWithElementwiseArguments<ThreadEpiOp>::value>> {
    typename ThreadEpiOp::Params thread{};
    typename ThreadEpiOp::ElementwiseArguments activation{};
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    ElementBias const* ptr_Bias = nullptr;
    StrideBias dBias{};
  };

  using Params = ParamsType<ThreadEpilogueOp>;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    typename ThreadEpilogueOp::Params thread_op_args;
    thread_op_args.alpha = args.thread.alpha;
    thread_op_args.beta = args.thread.beta;
    thread_op_args.alpha_ptr = args.thread.alpha_ptr;
    thread_op_args.beta_ptr = args.thread.beta_ptr;

    if constexpr (IsActHasArgs) {
      return {
        thread_op_args,
        args.thread.activation,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr,
        args.thread.dBias
      };
    }
    else {
      return {
        thread_op_args,
        args.ptr_C,
        args.dC,
        args.ptr_D,
        args.dD,
        args.thread.bias_ptr,
        args.thread.dBias
      };
    }
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  Epilogue(Params const& params_)
      : params(params_), epilogue_op(params_.thread) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine,FrgLayout> const& accumulators,                   // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if CUDA_BARRIER_ENABLED
    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
#else
    auto synchronize = [] () { __syncthreads(); };
#endif

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M,N,L), params.dC);             //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);             //             (m,n,l)
    Tensor mBias_mnl = make_tensor(make_gmem_ptr(params.ptr_Bias), make_shape(M,N,L), params.dBias);    //             (m,n,l)

    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});             // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});             // (BLK_M,BLK_N,m,n,l)
    Tensor gBias_mnl = local_tile(mBias_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});       // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gBias = gBias_mnl(_,_,m_coord,n_coord,l_coord);                                             // (BLK_M,BLK_N)

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sAcc = make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});            // (SMEM_M,SMEM_N)

    // Partition sAcc to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto thread_r2s     = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);                              // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tRS_sAcc = thread_r2s.partition_D(sAcc);                                   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile  = make_shape(size<0>(sAcc), size<1>(sAcc));
    Tensor gCt = flat_divide(gC, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);                                                // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gBiast = flat_divide(gBias, tile);                                          // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sAcc, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto thread_s2r     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tSR_sAcc = thread_s2r.partition_S(sAcc);                      //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_gC = thread_s2r.partition_D(gCt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tSR_gD = thread_s2r.partition_D(gDt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tSR_gBias = thread_s2r.partition_D(gBiast);                   // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tSR_rAcc = make_tensor<ElementAccumulator>(take<0,3>(shape(tSR_gC)));       // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_rC = make_tensor<ElementC>(shape(tSR_rAcc));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_rD = make_tensor<ElementD>(shape(tSR_rAcc));                            // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_rBias = make_tensor_like(tSR_gBias);                      // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD   = make_identity_tensor(make_shape(size<0>(gD),size<1>(gD)));           // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt  = flat_divide(cD, tile);                                 //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tSR_cD = thread_s2r.partition_D(cDt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tRS_rAcc) % size<3>(tSR_gC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tRS_rAcc) % size<4>(tSR_gC) == 0);  // TILE_N divides MMA_N

#if 0
    if (thread_idx == 0 && m_coord == 0 && n_coord == 0) {
      print("aC   : "); print(accumulators.layout()); print("\n");
      print("gC   : "); print(gC.layout()); print("\n");
      print("gD   : "); print(gD.layout()); print("\n");
      print("gBias   : "); print(gBias.layout()); print("\n");
      print("sAcc   : "); print(sAcc.layout()); print("\n");
      print("\n");
      print("tRS_sAcc : "); print(tRS_sAcc.layout()); print("\n");
      print("tRS_rAcc : "); print(tRS_rAcc.layout()); print("\n");
      print("\n");
      print("gDt  : "); print(gDt.layout()); print("\n");
      print("tSR_sAcc : "); print(tSR_sAcc.layout()); print("\n");
      print("tSR_rAcc : "); print(tSR_rAcc.layout()); print("\n");
      print("\n");
      print("tSR_rC : "); print(tSR_rC.layout()); print("\n");
      print("tSR_rD : "); print(tSR_rD.layout()); print("\n");
      print("tSR_gC : "); print(tSR_gC.layout()); print("\n");
      print("tSR_gD : "); print(tSR_gD.layout()); print("\n");
      print("\n");
      print("gBiast  : "); print(gBiast.layout()); print("\n");
      print("tSR_gBias  : "); print(tSR_gBias.layout()); print("\n");
      print("tSR_rBias  : "); print(tSR_rBias.layout()); print("\n");
    }
#endif

    if constexpr (IsEpilogueBiasSupported) {
      if (params.ptr_Bias) {
        // Filter so we don't issue redundant copies over stride-0 modes
        // (only works if 0-strides are in same location, which is by construction)
        Tensor tSR_gBias_flt = filter_zeros(tSR_gBias);
        Tensor tSR_rBias_flt = filter_zeros(tSR_rBias);
        Tensor tSR_cD_flt = filter_zeros(tSR_cD, tSR_gBias.stride());
        Tensor tSR_pD_flt = cute::lazy::transform(tSR_cD_flt, [&](auto const& c){ return elem_less(c, take<0,2>(residue_mnk)); });

        // Step 0. Copy Bias from GMEM to fragment
        copy_if(tSR_pD_flt, tSR_gBias_flt, tSR_rBias_flt);
      }
    }

    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n) {
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tRS_sAcc); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tRS_sAcc); ++pipe_n) {
            int mma_m = step_m * size<1>(tRS_sAcc) + pipe_m;
            int mma_n = step_n * size<2>(tRS_sAcc) + pipe_n;

            copy(tiled_r2s, tRS_rAcc(_,mma_m,mma_n), tRS_sAcc(_,pipe_m,pipe_n));
          }
        }

        // Step 2. Wait for SMEM writes to complete
        synchronize();

        // Step 3. Copy from SMEM into a fragment
        copy(tiled_s2r, tSR_sAcc, tSR_rAcc);

        // Step 4. Wait for SMEM reads to complete
        synchronize();

        Tensor tSR_gDmn = tSR_gD(_,_,_,step_m,step_n);
        Tensor tSR_cDmn = tSR_cD(_,_,_,step_m,step_n);

        if constexpr (IsEpilogueBiasSupported) {
          Tensor tSR_rBiasmn = tSR_rBias(_,_,_,step_m,step_n);

          if (epilogue_op.is_source_needed()) {
            // source is needed
            Tensor tSR_gCmn = tSR_gC(_,_,_,step_m,step_n);

            // Step 5. Copy C from GMEM to a fragment
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
                // Predication
                if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int i = 0; i < size<0>(tSR_rAcc); ++i) {
                    tSR_rC(i,m,n) = tSR_gCmn(i,m,n);
                  }
                }
              }
            }

            // Step 6. Elementwise operation with conversion
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSR_rAcc); ++i) {
              if constexpr (IsActHasArgs) {
                epilogue_op(tSR_rD(i), tSR_rD(i), tSR_rAcc(i), tSR_rC(i), tSR_rBiasmn(i), params.activation);
              } else {
                epilogue_op(tSR_rD(i), tSR_rD(i), tSR_rAcc(i), tSR_rC(i), tSR_rBiasmn(i));
              }
            }
          }
          else {
            // source is not needed, avoid load and lift compute

            // Step 5. Elementwise operation with conversion
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSR_rAcc); ++i) {
              if constexpr (IsActHasArgs) {
                epilogue_op(tSR_rD(i), tSR_rD(i), tSR_rAcc(i), tSR_rBiasmn(i), params.activation);
              } else {
                epilogue_op(tSR_rD(i), tSR_rD(i), tSR_rAcc(i), tSR_rBiasmn(i));
              }
            }
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
              // Predication
              if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                // The Last Step. Copy to GMEM
                copy(CopyAtomR2G{}, tSR_rD(_,m,n), tSR_gDmn(_,m,n));
              }
            }
          }
        } else {
          if (epilogue_op.is_source_needed()) {
            // source is needed
            Tensor tSR_gCmn = tSR_gC(_,_,_,step_m,step_n);

            // Step 5. Copy C from GMEM to a fragment
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
              CUTLASS_PRAGMA_UNROLL
              for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
                // Predication
                if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int i = 0; i < size<0>(tSR_rAcc); ++i) {
                    tSR_rC(i,m,n) = tSR_gCmn(i,m,n);
                  }
                }
              }
            }

            // Step 6. Elementwise operation with conversion
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSR_rAcc); ++i) {
              tSR_rD(i) = epilogue_op(tSR_rAcc(i), tSR_rC(i));
            }
          }
          else {
            // source is not needed, avoid load and lift compute

            // Step 5. Elementwise operation with conversion
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSR_rAcc); ++i) {
              tSR_rD(i) = epilogue_op(tSR_rAcc(i));
            }
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
              // Predication
              if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                // The Last Step. Copy to GMEM
                copy(CopyAtomR2G{}, tSR_rD(_,m,n), tSR_gDmn(_,m,n));
              }
            }
          }
        }
      }
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
