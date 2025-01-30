/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"

#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"

#include "cutlass_extensions/arch/copy_red_global.hpp"
#include "cutlass_extensions/util/gather_tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace epilogue
{
namespace collective
{

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class StrideC_, class ElementD_, class StrideD_, class ThreadEpilogueOp_, class ElementBias, class StrideBias,
    class ElementScale, class StrideScale, class EpilogueTile, class SmemLayoutAtomD, class CopyOpR2S, class CopyOpS2R,
    class CopyOpR2G>
class EpilogueMoeFusedFinalize
{
public:
    using EpilogueSchedule = PtrArrayNoSmemWarpSpecialized;
    using DispatchPolicy = PtrArrayNoSmemWarpSpecialized;

    using ThreadEpilogueOp = ThreadEpilogueOp_;
    using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
    using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
    using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
    using ElementIntermediate = typename ThreadEpilogueOp::ElementD;

    using ElementC = typename ThreadEpilogueOp::ElementC;
    using StrideC = StrideC_;
    using InternalStrideC = cute::remove_pointer_t<StrideC>;
    using ElementD = ElementD_;
    using StrideD = StrideD_;
    using InternalStrideD = cute::remove_pointer_t<StrideD>;

    static_assert(!is_same_v<InternalStrideC, StrideC>, "Stride C must be a pointer");
    static_assert(is_same_v<InternalStrideD, StrideD>, "Stride D must not be a pointer");

    using CopyAtomR2S = Copy_Atom<CopyOpR2S, ElementAccumulator>;
    using CopyAtomS2R = Copy_Atom<CopyOpS2R, ElementAccumulator>;
    using CopyAtomR2G = Copy_Atom<CopyOpR2G, ElementD>;
    static constexpr int AlignmentD = CopyAtomR2G::NumValSrc;

    using SmemLayoutD = decltype(tile_to_shape(SmemLayoutAtomD{}, EpilogueTile{}));

    constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});

    struct SharedStorage
    {
        alignas(SmemAlignmentD) cute::ArrayEngine<ElementAccumulator, cosize_v<SmemLayoutD>> smem_D;
    };

    struct TensorMapStorage
    {
    };

    struct Arguments
    {
        typename ThreadEpilogueOp::Params thread{};
        ElementC const** ptr_C{};
        StrideC dC{};
        ElementD* ptr_D{};
        StrideD dD{};
        ElementBias const* ptr_bias;
        StrideBias dBias{};
        ElementScale const* ptr_scale;
        StrideScale dScale{};
        int64_t const* group_offset{};
        int32_t const* scatter_index{};
        cutlass::FastDivmod num_rows_in_final_output;
    };

    using Params = Arguments;

    //
    // Methods
    //

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const&, Arguments const& args, [[maybe_unused]] void* workspace)
    {
        return args;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count = 0)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
        void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    template <class ProblemShape>
    CUTLASS_HOST_DEVICE static bool can_implement(
        [[maybe_unused]] ProblemShape problem_shape, [[maybe_unused]] Arguments const& args)
    {
        bool implementable = true;
        if (problem_shape.is_host_problem_shape_available())
        {
            // Check alignment for all problem sizes
            for (int i = 0; i < problem_shape.groups(); i++)
            {
                auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(i), 1);
                auto [M, N, K, L] = problem_shape_MNKL;
                implementable = implementable
                    && cutlass::detail::check_alignment<AlignmentD>(cute::make_shape(M, N, L), InternalStrideD{});
            }
        }

        if (!implementable)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for selected global "
                "reduction instruction.\n");
        }
        return implementable;
    }

    CUTLASS_HOST_DEVICE
    EpilogueMoeFusedFinalize(Params const& params_)
        : params(params_)
    {
    }

    CUTLASS_DEVICE
    bool is_source_needed()
    {
        // For Ptr-Array or Grouped Gemm we cannot determine if source is needed based on first beta.
        return params.ptr_C != nullptr
            && (params.thread.beta_ptr_array || params.thread.beta_ptr || params.thread.beta != 0);
    }

    template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout,
        class TiledMma, class ResidueMNK>
    CUTLASS_HOST_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK blk_shape_MNK,
        BlockCoordMNKL blk_coord_mnkl, cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TiledMma tiled_mma,
        ResidueMNK residue_mnk, int thread_idx, [[maybe_unused]] char* smem_buf)
    {
        using namespace cute;
        using X = Underscore;

        static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
        static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
        static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
        static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

        auto synchronize = [&]()
        { cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

        // Separate out problem shape for convenience
        auto M = get<0>(problem_shape_mnkl);
        auto N = get<1>(problem_shape_mnkl);
        auto L = get<3>(problem_shape_mnkl);

        auto mma_tile_m = tile_size<0>(tiled_mma);
        auto mma_tile_n = tile_size<1>(tiled_mma);
        auto epi_tile_m = size<0>(EpilogueTile{});
        auto epi_tile_n = size<1>(EpilogueTile{});

        CUTE_STATIC_ASSERT(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");
        CUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "EPI_TILE_N must divide MMA_TILE_N");

        // Batches are managed by using appropriate pointers to C and D matrices
        int32_t const mock_L = 1;
        int32_t const mock_l_coord = 0;

        // Slice to get the tile this CTA is responsible for
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

        // If scalar alpha/beta are provided, i.e., same alpha/beta applies to all batches/groups.
        // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups,
        // we get the correct alpha/beta values for the current batch/group using group index.
        ThreadEpilogueOp epilogue_op(params.thread, l_coord);

        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        Tensor sD_ = make_tensor(make_smem_ptr(storage.smem_D.begin()), SmemLayoutD{});
        Tensor sD = as_position_independent_swizzle_tensor(sD_);

        // Function to scatter output rows
        auto& num_rows = params.num_rows_in_final_output;
        auto read_scatter_map = IndexedGather(make_gmem_ptr(params.scatter_index + params.group_offset[l_coord]));
        auto get_scatter_idx = [&](auto i)
        {
            auto scatter = read_scatter_map(i);
            int quot, rem;
            num_rows(quot, rem, scatter);
            return rem;
        };

        // Represent the full output tensor
        ElementC const* ptr_C = epilogue_op.is_source_needed() ? params.ptr_C[l_coord] : nullptr;
        auto dC = epilogue_op.is_source_needed() ? params.dC[l_coord] : InternalStrideC{};
        Tensor mC_mnl = make_tensor(make_gmem_ptr(ptr_C), make_shape(M, N, mock_L), dC);        // (m,n,l)
        Tensor mD_mnl = make_gather_tensor(
            make_gmem_ptr(params.ptr_D), make_shape(M, N, mock_L), params.dD, get_scatter_idx); // (m,n,l)

        // Use fake shape for bias, it doesn't matter
        bool const is_bias_needed = params.ptr_bias != nullptr;
        Tensor mBias_mnl = make_tensor(make_gmem_ptr(params.ptr_bias), make_shape(M, N, 1), params.dBias);
        Tensor mScale_mnl = make_tensor(
            make_gmem_ptr(params.ptr_scale + params.group_offset[l_coord]), make_shape(M, N), params.dScale);

        Tensor gC_mnl
            = local_tile(mC_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)
        Tensor gD_mnl
            = local_tile(mD_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)

        Tensor gC = gC_mnl(_, _, m_coord, n_coord, mock_l_coord);                        // (BLK_M,BLK_N)
        Tensor gD = gD_mnl(_, _, m_coord, n_coord, mock_l_coord);                        // (BLK_M,BLK_N)

        Tensor gC_epi = flat_divide(gC, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
        Tensor gD_epi = flat_divide(gD, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        Tensor gBias_mnl
            = local_tile(mBias_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)
        Tensor gScale_mnl
            = local_tile(mScale_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)

        Tensor gBias = gBias_mnl(_, _, m_coord, n_coord, l_coord);                           // (BLK_M,BLK_N)
        Tensor gScale = gScale_mnl(_, _, m_coord, n_coord);                                  // (BLK_M,BLK_N)

        Tensor gBias_epi = flat_divide(gBias, EpilogueTile{});   // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
        Tensor gScale_epi = flat_divide(gScale, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        // Get the smallest tiled copy we can use to retile the accumulators
        TiledCopy tiled_copy_C_atom
            = make_tiled_copy_C_atom(Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>{}, tiled_mma);
        TiledCopy tiled_r2s = make_tiled_copy_S(CopyAtomR2S{}, tiled_copy_C_atom);

        auto thread_r2s = tiled_r2s.get_thread_slice(thread_idx);
        Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);            // ((R2S,R2S_V),MMA_M,MMA_N)
        Tensor tRS_sD = thread_r2s.partition_D(sD);                     // ((R2S,R2S_V),R2S_M,R2S_N)
        Tensor tRS_rD = make_tensor<ElementAccumulator>(shape(tRS_sD)); // ((R2S,R2S_V),R2S_M,R2S_N)

        // Make a tiled copy vectorized along major direction of D
        auto tiled_s2r = [&]()
        {
            if constexpr (cutlass::gemm::detail::is_k_major<StrideD>())
            {
                constexpr int NumThreadsMajor = epi_tile_n / AlignmentD;
                constexpr int NumThreadsMinor = cute::size(tiled_mma) / NumThreadsMajor;
                return make_tiled_copy(CopyAtomS2R{},
                    Layout<Shape<Int<NumThreadsMinor>, Int<NumThreadsMajor>>, Stride<Int<NumThreadsMajor>, _1>>{},
                    Layout<Shape<_1, Int<AlignmentD>>>{});
            }
            else if constexpr (cutlass::gemm::detail::is_mn_major<StrideD>())
            {
                constexpr int NumThreadsMajor = epi_tile_m / AlignmentD;
                constexpr int NumThreadsMinor = cute::size(tiled_mma) / NumThreadsMajor;
                return make_tiled_copy(CopyAtomS2R{},
                    Layout<Shape<Int<NumThreadsMajor>, Int<NumThreadsMinor>>, Stride<_1, Int<NumThreadsMajor>>>{},
                    Layout<Shape<Int<AlignmentD>, _1>>{});
            }
            else
            {
                static_assert(cute::is_void_v<StrideD>, "Unsupported D gmem layout.");
            }
        }();

        auto thread_s2r = tiled_s2r.get_thread_slice(thread_idx);
        Tensor tSR_sD = thread_s2r.partition_S(sD);             // ((S2R,S2R_V),S2R_M,S2R_N)
        Tensor tSR_gD = thread_s2r.partition_D(gD_epi);         // ((S2R,S2R_V),S2R_M,S2R_N,EPI_M,EPI_N)
        Tensor tSR_gC = thread_s2r.partition_D(gC_epi);         // ((S2R,S2R_V),S2R_M,S2R_N,EPI_M,EPI_N)
        Tensor tSR_gBias = thread_s2r.partition_D(gBias_epi);   // ((S2R,S2R_V),S2R_M,S2R_N,EPI_M,EPI_N)
        Tensor tSR_gScale = thread_s2r.partition_D(gScale_epi); // ((S2R,S2R_V),S2R_M,S2R_N,EPI_M,EPI_N)

        // Allocate intermediate registers for a single subtile
        Tensor tSR_rD = make_tensor<ElementAccumulator>(take<0, 3>(shape(tSR_gD)));        // ((S2R,S2R_V),S2R_M,S2R_N)
        Tensor tSR_rD_final = make_tensor<ElementD>(shape(tSR_rD));                        // ((S2R,S2R_V),S2R_M,S2R_N)
        Tensor tSR_rC = make_tensor<ElementC>(shape(tSR_rD));                              // ((S2R,S2R_V),S2R_M,S2R_N)
        Tensor tSR_rBias = make_tensor<ElementBias>(tSR_gBias(_, _, _, 0, 0).layout());    // ((S2R,S2R_V),S2R_M,S2R_N)
        Tensor tSR_rScale = make_tensor<ElementScale>(tSR_gScale(_, _, _, 0, 0).layout()); // ((S2R,S2R_V),S2R_M,S2R_N)

        // Make an identity coordinate tensor for predicating our output MN tile
        Tensor cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
        Tensor cD_epi = flat_divide(cD, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
        Tensor tSR_cD = thread_s2r.partition_D(cD_epi);  // ((S2R,S2R_V),S2R_M,S2R_N,EPI_M,EPI_N)

        // epilogue subtile loop
        CUTLASS_PRAGMA_UNROLL
        for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n)
            {
                int mma_m = (epi_m * epi_tile_m) / mma_tile_m;
                int mma_n = (epi_n * epi_tile_n) / mma_tile_n;
                Tensor tRS_rAcc_mn = tRS_rAcc(_, mma_m, mma_n);

                int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
                int r2s_v = epi_n_in_mma * size(tRS_rD);
                CUTLASS_PRAGMA_UNROLL
                for (int epi_v = 0; epi_v < size(tRS_rD); ++epi_v)
                {
                    tRS_rD(epi_v) = tRS_rAcc_mn(r2s_v + epi_v);
                }

                copy(tiled_r2s, tRS_rD, tRS_sD);
                synchronize();

                copy(tiled_s2r, tSR_sD, tSR_rD);
                synchronize();

                Tensor tSR_gC_mn = tSR_gC(_, _, _, epi_m, epi_n);
                Tensor tSR_gBias_mn = tSR_gBias(_, _, _, epi_m, epi_n);
                Tensor tSR_gScale_mn = tSR_gScale(_, _, _, epi_m, epi_n);
                Tensor tSR_cD_mn = tSR_cD(_, _, _, epi_m, epi_n);
                Tensor tSR_gD_mn = tSR_gD(_, _, _, epi_m, epi_n);

                if (epilogue_op.is_source_needed())
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < size<1>(tSR_rD); ++m)
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int n = 0; n < size<2>(tSR_rD); ++n)
                        {
                            if (elem_less(tSR_cD_mn(0, m, n), make_coord(get<0>(residue_mnk), get<1>(residue_mnk))))
                            {
                                copy(tSR_gC_mn(_, m, n), tSR_rC(_, m, n));
                                if (is_bias_needed)
                                {
                                    copy(tSR_gBias_mn(_, m, n), tSR_rBias(_, m, n));
                                }
                                copy(tSR_gScale_mn(_, m, n), tSR_rScale(_, m, n));
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < size<0>(tSR_rD); ++i)
                                {
                                    auto epi_value = epilogue_op(tSR_rD(i, m, n), tSR_rC(i, m, n));
                                    if (is_bias_needed)
                                    {
                                        epi_value += static_cast<ElementCompute>(tSR_rBias(i, m, n));
                                    }
                                    tSR_rD_final(i, m, n) = static_cast<ElementD>(tSR_rScale(i, m, n) * epi_value);
                                }
                                copy(CopyAtomR2G{}, tSR_rD_final(_, m, n), tSR_gD_mn(_, m, n));
                            }
                        }
                    }
                }
                else
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int m = 0; m < size<1>(tSR_rD); ++m)
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int n = 0; n < size<2>(tSR_rD); ++n)
                        {
                            if (elem_less(tSR_cD_mn(0, m, n), make_coord(get<0>(residue_mnk), get<1>(residue_mnk))))
                            {
                                if (is_bias_needed)
                                {
                                    copy(tSR_gBias_mn(_, m, n), tSR_rBias(_, m, n));
                                }
                                copy(tSR_gScale_mn(_, m, n), tSR_rScale(_, m, n));
                                CUTLASS_PRAGMA_UNROLL
                                for (int i = 0; i < size<0>(tSR_rD); ++i)
                                {
                                    auto epi_value = epilogue_op(tSR_rD(i, m, n));
                                    if (is_bias_needed)
                                    {
                                        epi_value += static_cast<ElementCompute>(tSR_rBias(i, m, n));
                                    }
                                    tSR_rD_final(i, m, n) = static_cast<ElementD>(tSR_rScale(i, m, n) * epi_value);
                                }
                                copy(CopyAtomR2G{}, tSR_rD_final(_, m, n), tSR_gD_mn(_, m, n));
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

namespace detail
{

template <class Element, class MaxVec>
constexpr auto get_vectorized_atomic_add_op()
{
    using namespace cute;

    auto constexpr MaxVecSize = size(MaxVec{});

    if constexpr (is_same_v<Element, cutlass::half_t>)
    {
        if constexpr (MaxVecSize >= 8)
        {
            return SM90_RED_ADD_NOFTZ_F16x2_V4{};
        }
        else if constexpr (MaxVecSize >= 4)
        {
            return SM90_RED_ADD_NOFTZ_F16x2_V2{};
        }
        else if constexpr (MaxVecSize >= 2)
        {
            return SM70_RED_ADD_NOFTZ_F16x2{};
        }
        else
        {
            return SM70_RED_ADD_NOFTZ_F16{};
        }
    }
    else if constexpr (is_same_v<Element, cutlass::bfloat16_t>)
    {
        if constexpr (MaxVecSize >= 8)
        {
            return SM90_RED_ADD_NOFTZ_BF16x2_V4{};
        }
        else if constexpr (MaxVecSize >= 4)
        {
            return SM90_RED_ADD_NOFTZ_BF16x2_V2{};
        }
        else if constexpr (MaxVecSize >= 2)
        {
            return SM90_RED_ADD_NOFTZ_BF16x2{};
        }
        else
        {
            return SM90_RED_ADD_NOFTZ_BF16{};
        }
    }
    else
    {
        // non-vectorized atomic add for all other types until supported
        return TypedAtomicAdd<Element>{};
    }
}

} // namespace detail

template <class TileShape, class ElementC, class StrideC, class ElementD, class StrideD, class ElementAccumulator,
    class ElementCompute, class ElementBias, class StrideBias, class ElementScale, class StrideScale>
struct EpilogueMoeFusedFinalizeBuilder
{

    // assuming cooperative kernel schedule
    using EpiTileN = decltype(cute::min(size<1>(TileShape{}), _32{}));
    using EpilogueTile = Shape<_128, EpiTileN>;

    // Output of linear combination is ElementCompute instead of ElementD
    // since we will be doing more computate on it, no need to cast yet.
    using ThreadEpilogueOp
        = cutlass::epilogue::thread::LinearCombination<ElementCompute, 1, ElementAccumulator, ElementCompute,
            cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, ElementC>;

    using SmemLayoutAtomD
        = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<StrideD, ElementAccumulator, EpilogueTile>());
    using CopyAtomR2S = decltype(detail::sm90_get_smem_store_op_for_accumulator<StrideD, ElementAccumulator>());
    using CopyAtomS2R = DefaultCopy;
    using CopyAtomR2G = decltype(detail::get_vectorized_atomic_add_op<ElementD, EpiTileN>());

    template <class EpilogueOp>
    struct Sm90TmaWarpSpecializedAdapterWithSmemStorage : detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp>
    {
        // We need to override this one using declaration because otherwise we double up on the smem
        using TensorMapStorage = typename EpilogueOp::TensorMapStorage;

        using Base = detail::Sm90TmaWarpSpecializedAdapter<EpilogueOp>;

        CUTLASS_HOST_DEVICE
        Sm90TmaWarpSpecializedAdapterWithSmemStorage(
            typename EpilogueOp::Params const& params, [[maybe_unused]] typename Base::TensorStorage& shared_tensors)
            : Base(params)
        {
        }

        // These functions depend on the type of TensorMapStorage
        template <bool IsLoad>
        CUTLASS_DEVICE void tensormaps_perform_update([[maybe_unused]] TensorMapStorage& shared_tensormap,
            [[maybe_unused]] typename EpilogueOp::Params const& params,
            [[maybe_unused]] cute::TmaDescriptor const* tensormap, [[maybe_unused]] int32_t next_batch)
        {
        }

        template <bool IsLoad>
        CUTLASS_DEVICE void tensormaps_cp_fence_release([[maybe_unused]] TensorMapStorage& shared_tensormap,
            [[maybe_unused]] cute::TmaDescriptor const* tensormap, [[maybe_unused]] uint32_t lane_predicate)
        {
        }
    };

    using CollectiveOp = Sm90TmaWarpSpecializedAdapterWithSmemStorage<
        EpilogueMoeFusedFinalize<StrideC, ElementD, StrideD, ThreadEpilogueOp, ElementBias, StrideBias, ElementScale,
            StrideScale, EpilogueTile, SmemLayoutAtomD, CopyAtomR2S, CopyAtomS2R, CopyAtomR2G>>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
