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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Ptr Array Epilogue Vectorized
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
          cute::is_same_v<EpilogueScheduleType_, EpiloguePtrArraySimtVectorized>
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
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;

  using SmemLayout   = SmemLayout_;
  using CopyAtomR2S  = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G  = CopyAtomR2G_;

  using GmemTiledCopyC = TiledCopyS2R;
  using GmemTiledCopyD = TiledCopyS2R;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;

  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  using TensorMapStorage = SharedStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const&,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
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
      : params(params_) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    // For Ptr-Array or Grouped Gemm we cannot determine if source is needed based on first beta.
    return true;
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
    // Batches are managed by using appropriate pointers to C and D matrices
    const int32_t mock_L = 1;
    const int32_t mock_l_coord = 0;
    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    // If scalar alpha/beta are provided, i.e., same alpha/beta applies to all batches/groups.
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups,
    // we get the correct alpha/beta values for the current batch/group using group index.
    ThreadEpilogueOp epilogue_op = ThreadEpilogueOp(params.thread, l_coord);

    if (epilogue_op.is_source_needed() && params.dC == nullptr) {
      // Beta value is non-zero while pointer to C is a nullptr
      assert(0);
    }

    InternalStrideC stride_c;
    InternalStrideD stride_d;
    if constexpr (!cute::is_same_v<InternalStrideC, StrideC>) {
      // If grouped gemm
      if (epilogue_op.is_source_needed()) {
        stride_c = params.dC[l_coord];
      }
      stride_d = params.dD[l_coord];
    }
    else {
      stride_c = params.dC;
      stride_d = params.dD;
    }

    // Represent the full output tensor
    ElementC const* ptr_C_l = nullptr;
    if (epilogue_op.is_source_needed()) {
      ptr_C_l = params.ptr_C[l_coord];
    }
    Tensor mC_mnl = make_tensor(make_gmem_ptr(ptr_C_l), make_shape(M,N,mock_L), stride_c);      //             (m,n,l)
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D[l_coord]), make_shape(M,N,mock_L), stride_d);      //             (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});      // (BLK_M,BLK_N,m,n,l)

    Tensor gC = gC_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                   // (BLK_M,BLK_N)

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

    // Partition sAcc, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto thread_s2r     = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tSR_sAcc = thread_s2r.partition_S(sAcc);                      //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_gC = thread_s2r.partition_D(gCt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tSR_gD = thread_s2r.partition_D(gDt);                         // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tSR_rAcc = make_tensor<ElementAccumulator>(take<0,3>(shape(tSR_gC)));       // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tSR_rD = make_tensor<ElementOutput>(shape(tSR_rAcc));                       // ((Atom,AtomNum),ATOM_M,ATOM_N)

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
      print("sAcc   : "); print(sAcc.layout()); print("\n");
      print("\n");
      print("tRS_sAcc : "); print(tRS_sAcc.layout()); print("\n");
      print("tRS_rAcc : "); print(tRS_rAcc.layout()); print("\n");
      print("\n");
      print("gDt  : "); print(gDt.layout()); print("\n");
      print("tSR_sAcc : "); print(tSR_sAcc.layout()); print("\n");
      print("tSR_rAcc : "); print(tSR_rAcc.layout()); print("\n");
      print("\n");
      print("tSR_rD : "); print(tSR_rD.layout()); print("\n");
      print("tSR_gC : "); print(tSR_gC.layout()); print("\n");
      print("tSR_gD : "); print(tSR_gD.layout()); print("\n");
      print("\n");
    }
#endif

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

        if (epilogue_op.is_source_needed()) {
          // source is needed
          Tensor tSR_gCmn = tSR_gC(_,_,_,step_m,step_n);

          Tensor tSR_rCmn = make_tensor<ElementC>(shape(tSR_gCmn));                     // ((Atom,AtomNum),ATOM_M,ATOM_N)

          // Step 5. Copy C from GMEM to a fragment
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
              // Predication
              if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size<0>(tSR_rAcc); ++i) {
                  tSR_rCmn(i,m,n) = tSR_gCmn(i,m,n);
                }
              }
            }
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
              // Predication
              if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                // Step 6. Elementwise operation with conversion
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size<0>(tSR_rAcc); ++i) {
                  tSR_rD(i,m,n) = epilogue_op(tSR_rAcc(i,m,n), tSR_rCmn(i,m,n));
                }
                // Step 7. Copy to GMEM
                copy(CopyAtomR2G{}, tSR_rD(_,m,n), tSR_gDmn(_,m,n));
              }
            }
          }
        }
        else {
          // source is not needed, avoid load and lift compute

          // Step 5. Elementwise operation with conversion
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSR_rAcc); ++i) {
            tSR_rD(i) = epilogue_op(tSR_rAcc(i));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tSR_gDmn); ++m) {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < size<2>(tSR_gDmn); ++n) {
              // Predication
              if (elem_less(tSR_cDmn(0,m,n), take<0,2>(residue_mnk))) {
                // Step 6. Copy to GMEM
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
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
