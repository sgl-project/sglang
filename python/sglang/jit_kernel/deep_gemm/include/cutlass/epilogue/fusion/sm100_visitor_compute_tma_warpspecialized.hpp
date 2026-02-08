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
  \brief Visitor tree compute operations for the sm100 TMA warp-specialized (ws) epilogue
*/



#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp" 
#include "cutlass/epilogue/thread/activation.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                   BatchNormApply
//
// This node aims to do the batch norm apply. The procedure is described as follows:
//
//                    output = (input - mean) * inv_stddev * alpha + bias
//
// while: (1) input & output are 2 matrices with shape (M, N),
//            which are frg_input & return value of the visit function
//
//        (2) mean, inv_stddev, alpha & bias are 4 vectors with shape (N).
//            which are loaded by ProducerLoadCallbacks
//
// To avoid redundant calculations in EVT, this node simplify the procedure as follows:
//
//                              output = input * alpha' + bias'
//
// while alpha' & bias' are 2 vectors with shape (N) calculated by mean, inv_stddev, alpha & bias
//
// The calculation among vectors is described as follows:
//
//                               alpha' = alpha * inv_stddev
//                               bias' = bias - mean * alpha'
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  // reuses the mbarriers from the epilogue subtile load pipeline, so this must be at least
  // this should just match CLC stage count
  int Stages,
  class CtaTileShapeMNK,
  class ElementScalar,
  class ElementCompute,
  class ElementOutput,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<ElementScalar>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm100BatchNormApply {
  static_assert(Alignment * sizeof_bits_v<ElementScalar> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(cute::is_same_v<StrideMNL, Stride<_0,_1,_0>>); // row vector broadcast for alpha, bias, mean & inv_stddev

  using SmemLayout = decltype(make_layout(make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                              make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{}))));

  using ElementCol = cute::conditional_t<(sizeof(ElementCompute) > sizeof(ElementScalar)), ElementCompute, ElementScalar>;

  struct SharedStorage {
    alignas(16) array_aligned<ElementCol, size<1>(CtaTileShapeMNK{}) * Stages> smem_alpha;
    alignas(16) array_aligned<ElementCol, size<1>(CtaTileShapeMNK{}) * Stages> smem_bias;
    alignas(16) array_aligned<ElementScalar, size<1>(CtaTileShapeMNK{}) * Stages> smem_mean;
    alignas(16) array_aligned<ElementScalar, size<1>(CtaTileShapeMNK{}) * Stages> smem_inv_stddev;
  };

  struct Arguments {
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* bias_ptr = nullptr;
    ElementScalar const* mean_ptr = nullptr;
    ElementScalar const* inv_stddev_ptr = nullptr;
    StrideMNL dVec = {};
  };

  struct Params {
    using TMA_Vec = decltype(make_tma_atom(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr<ElementScalar const>(nullptr), repeat_like(StrideMNL{}, int32_t(0)), append<3>(StrideMNL{}, _0{})),
        take<0,2>(SmemLayout{}),
        take<0,2>(CtaTileShapeMNK{})));

    TMA_Vec tma_load_alpha;
    TMA_Vec tma_load_bias;
    TMA_Vec tma_load_mean;
    TMA_Vec tma_load_inv_stddev;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;

    Tensor tensor_alpha = make_tensor(make_gmem_ptr(args.alpha_ptr), make_layout(make_shape(size(M),N,size(L)), append<3>(args.dVec, _0{})));
    Tensor tensor_bias = make_tensor(make_gmem_ptr(args.bias_ptr), make_layout(make_shape(size(M),N,size(L)), append<3>(args.dVec, _0{})));
    Tensor tensor_mean = make_tensor(make_gmem_ptr(args.mean_ptr), make_layout(make_shape(size(M),N,size(L)), append<3>(args.dVec, _0{})));
    Tensor tensor_inv_stddev = make_tensor(make_gmem_ptr(args.inv_stddev_ptr), make_layout(make_shape(size(M),N,size(L)), append<3>(args.dVec, _0{})));

    typename Params::TMA_Vec tma_load_alpha = make_tma_atom(SM90_TMA_LOAD{}, tensor_alpha, take<0,2>(SmemLayout{}), take<0,2>(CtaTileShapeMNK{}));
    typename Params::TMA_Vec tma_load_bias = make_tma_atom(SM90_TMA_LOAD{}, tensor_bias, take<0,2>(SmemLayout{}), take<0,2>(CtaTileShapeMNK{}));
    typename Params::TMA_Vec tma_load_mean = make_tma_atom(SM90_TMA_LOAD{}, tensor_mean, take<0,2>(SmemLayout{}), take<0,2>(CtaTileShapeMNK{}));
    typename Params::TMA_Vec tma_load_inv_stddev = make_tma_atom(SM90_TMA_LOAD{}, tensor_inv_stddev, take<0,2>(SmemLayout{}), take<0,2>(CtaTileShapeMNK{}));

    return Params{tma_load_alpha, tma_load_bias, tma_load_mean, tma_load_inv_stddev};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
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

  CUTLASS_HOST_DEVICE
  Sm100BatchNormApply() { }

  CUTLASS_HOST_DEVICE
  Sm100BatchNormApply(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params),
        smem_alpha(const_cast<ElementScalar*>(shared_storage.smem_alpha.data())),
        smem_bias(const_cast<ElementScalar*>(shared_storage.smem_bias.data())),
        smem_mean(const_cast<ElementScalar*>(shared_storage.smem_mean.data())),
        smem_inv_stddev(const_cast<ElementScalar*>(shared_storage.smem_inv_stddev.data())),
        smem_col_alpha(const_cast<ElementCompute*>(shared_storage.smem_alpha.data())),
        smem_col_bias(const_cast<ElementCompute*>(shared_storage.smem_bias.data())) { }

  Params const* params_ptr;
  ElementScalar* smem_alpha;
  ElementScalar* smem_bias;
  ElementScalar* smem_mean;
  ElementScalar* smem_inv_stddev;
  ElementCompute* smem_col_alpha;
  ElementCompute* smem_col_bias;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return true;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <int EpiTiles, class GTensor, class STensor>
  struct ProducerLoadCallbacks : EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    ProducerLoadCallbacks(GTensor&& gAlpha, GTensor&& gBias, GTensor&& gMean, GTensor&& gInvStddev,
      STensor&& sAlpha, STensor&& sBias, STensor&& sMean, STensor&& sInvStddev, Params const* params_ptr)
      : gAlpha(cute::forward<GTensor>(gAlpha)),
        gBias(cute::forward<GTensor>(gBias)),
        gMean(cute::forward<GTensor>(gMean)),
        gInvStddev(cute::forward<GTensor>(gInvStddev)),
        sAlpha(cute::forward<STensor>(sAlpha)),
        sBias(cute::forward<STensor>(sBias)),
        sMean(cute::forward<STensor>(sMean)),
        sInvStddev(cute::forward<STensor>(sInvStddev)),
        params_ptr(params_ptr) {}

    GTensor gAlpha;
    GTensor gBias;
    GTensor gMean;
    GTensor gInvStddev;

    STensor sAlpha;
    STensor sBias;
    STensor sMean;
    STensor sInvStddev;

    Params const* params_ptr;

    CUTLASS_DEVICE void
    step(uint64_t* full_mbarrier_ptr, int epi_m, int epi_n, int load_iteration, bool issue_tma_load) {
      if (epi_m == 0 && epi_n == 0 && issue_tma_load) {
        // Increment the expect-tx count of the first subtile's mbarrier by the row vector's byte-size
        constexpr uint32_t copy_bytes = size<1>(CtaTileShapeMNK{}) * bits_to_bytes(sizeof_bits_v<ElementScalar>) * 4;
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full_mbarrier_ptr, copy_bytes);
        // Issue the TMA bulk copy
        int pipe_index = (load_iteration / EpiTiles) % Stages;
        copy(params_ptr->tma_load_alpha.with(*full_mbarrier_ptr), gAlpha, sAlpha(_,pipe_index));
        copy(params_ptr->tma_load_bias.with(*full_mbarrier_ptr), gBias, sBias(_,pipe_index));
        copy(params_ptr->tma_load_mean.with(*full_mbarrier_ptr), gMean, sMean(_,pipe_index));
        copy(params_ptr->tma_load_inv_stddev.with(*full_mbarrier_ptr), gInvStddev, sInvStddev(_,pipe_index));
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    Tensor mAlpha = params_ptr->tma_load_alpha.get_tma_tensor(make_shape(size(M),N,size(L)));
    Tensor mBias  = params_ptr->tma_load_bias.get_tma_tensor(make_shape(size(M),N,size(L)));
    Tensor mMean  = params_ptr->tma_load_mean.get_tma_tensor(make_shape(size(M),N,size(L)));
    Tensor mInvStddev = params_ptr->tma_load_inv_stddev.get_tma_tensor(make_shape(size(M),N,size(L)));

    Tensor gAlpha = local_tile(mAlpha, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));             // (CTA_M,CTA_N)
    Tensor gBias  = local_tile(mBias,  take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));             // (CTA_M,CTA_N)
    Tensor gMean  = local_tile(mMean,  take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));             // (CTA_M,CTA_N)
    Tensor gInvStddev = local_tile(mInvStddev, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));     // (CTA_M,CTA_N)

    Tensor sAlpha = make_tensor(make_smem_ptr(smem_alpha), SmemLayout{});                         // (CTA_M,CTA_N,PIPE)
    Tensor sBias  = make_tensor(make_smem_ptr(smem_bias), SmemLayout{});                          // (CTA_M,CTA_N,PIPE)
    Tensor sMean  = make_tensor(make_smem_ptr(smem_mean), SmemLayout{});                          // (CTA_M,CTA_N,PIPE)
    Tensor sInvStddev = make_tensor(make_smem_ptr(smem_inv_stddev), SmemLayout{});                // (CTA_M,CTA_N,PIPE)

    auto [tCgAlpha,     tCsAlpha]     = tma_partition(params_ptr->tma_load_alpha, group_modes<0,2>(sAlpha), group_modes<0,2>(gAlpha));
    auto [tCgBias,      tCsBias]      = tma_partition(params_ptr->tma_load_bias,  group_modes<0,2>(sBias),  group_modes<0,2>(gBias));
    auto [tCgMean,      tCsMean]      = tma_partition(params_ptr->tma_load_mean,  group_modes<0,2>(sMean),  group_modes<0,2>(gMean));
    auto [tCgInvStddev, tCsInvStddev] = tma_partition(params_ptr->tma_load_inv_stddev, group_modes<0,2>(sInvStddev), group_modes<0,2>(gInvStddev));

    constexpr int EpiTiles = decltype(size(ceil_div(shape(take<0,2>(args.tile_shape_mnk)), args.epi_tile)))::value;
    return ProducerLoadCallbacks<EpiTiles, decltype(tCgAlpha), decltype(tCsAlpha)>(
      cute::move(tCgAlpha), cute::move(tCgBias), cute::move(tCgMean), cute::move(tCgInvStddev),
      cute::move(tCsAlpha), cute::move(tCsBias), cute::move(tCsMean), cute::move(tCsInvStddev), params_ptr);
  }

  template <int EpiTiles, class SR_RTensor, class SR_STensor, class SR_CTensor, class SR_SCTensor, class RTensor, class STensor, class ThrNum>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
      SR_RTensor&& tSR_rAlpha, SR_RTensor&& tSR_rBias,
      SR_RTensor&& tSR_rMean, SR_RTensor&& tSR_rInvStddev,
      SR_STensor&& tSR_sAlpha, SR_STensor&& tSR_sBias,
      SR_STensor&& tSR_sMean, SR_STensor&& tSR_sInvStddev,
      SR_CTensor&& tSR_cAlpha,
      SR_SCTensor&& tSR_sColAlpha, SR_SCTensor&& tSR_sColBias,
      RTensor&& tCrAlpha, RTensor&& tCrBias,
      STensor&& tCsAlpha, STensor&& tCsBias,
      ThrNum thr_num,
      Params const* params_ptr)
      :
        tSR_rAlpha(cute::forward<SR_RTensor>(tSR_rAlpha)), tSR_rBias(cute::forward<SR_RTensor>(tSR_rBias)),
        tSR_rMean(cute::forward<SR_RTensor>(tSR_rMean)), tSR_rInvStddev(cute::forward<SR_RTensor>(tSR_rInvStddev)),
        tSR_sAlpha(cute::forward<SR_STensor>(tSR_sAlpha)), tSR_sBias(cute::forward<SR_STensor>(tSR_sBias)),
        tSR_sMean(cute::forward<SR_STensor>(tSR_sMean)), tSR_sInvStddev(cute::forward<SR_STensor>(tSR_sInvStddev)),
        tSR_cAlpha(cute::forward<SR_CTensor>(tSR_cAlpha)),
        tSR_sColAlpha(cute::forward<SR_SCTensor>(tSR_sColAlpha)), tSR_sColBias(cute::forward<SR_SCTensor>(tSR_sColBias)),
        tCrAlpha(cute::forward<RTensor>(tCrAlpha)), tCrBias(cute::forward<RTensor>(tCrBias)),
        tCsAlpha(cute::forward<STensor>(tCsAlpha)), tCsBias(cute::forward<STensor>(tCsBias)),
        thr_num(thr_num),
        params_ptr(params_ptr) {}

    SR_RTensor tSR_rAlpha;
    SR_RTensor tSR_rBias;
    SR_RTensor tSR_rMean;
    SR_RTensor tSR_rInvStddev;
    SR_STensor tSR_sAlpha;
    SR_STensor tSR_sBias;
    SR_STensor tSR_sMean;
    SR_STensor tSR_sInvStddev;
    SR_CTensor tSR_cAlpha;
    SR_SCTensor tSR_sColAlpha;
    SR_SCTensor tSR_sColBias;

    ThrNum thr_num;

    RTensor tCrAlpha;                                                                              // (CPY,CPY_M,CPY_N)
    RTensor tCrBias;                                                                               // (CPY,CPY_M,CPY_N)

    STensor tCsAlpha;                                                             // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)
    STensor tCsBias;                                                              // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)

    Params const* params_ptr;

    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
      if (epi_m == 0 && epi_n == 0) { // Assumes M-major subtile loop
        // Filter so we don't issue redundant copies over stride-0 modes
        // (only works if 0-strides are in same location, which is by construction)
        auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(thr_num, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
        int pipe_index = (load_iteration / EpiTiles) % Stages;

        Tensor tSR_rAlpha_flt = filter_zeros(tSR_rAlpha);
        Tensor tSR_rBias_flt = filter_zeros(tSR_rBias);
        Tensor tSR_rMean_flt = filter_zeros(tSR_rMean);
        Tensor tSR_rInvStddev_flt = filter_zeros(tSR_rInvStddev);
        Tensor tSR_sAlpha_flt = filter_zeros(tSR_sAlpha(_,_,_,pipe_index));
        Tensor tSR_sBias_flt = filter_zeros(tSR_sBias(_,_,_,pipe_index));
        Tensor tSR_sMean_flt = filter_zeros(tSR_sMean(_,_,_,pipe_index));
        Tensor tSR_sInvStddev_flt = filter_zeros(tSR_sInvStddev(_,_,_,pipe_index));
        Tensor tSR_cAlpha_flt = filter_zeros(tSR_cAlpha, tSR_rAlpha.stride());

        for (int i = 0; i < size(tSR_rAlpha_flt); ++i) {
          if (get<1>(tSR_cAlpha_flt(i)) >= size<1>(CtaTileShapeMNK{})) {
            // OOB of SMEM
            continue;
          }
          tSR_rAlpha_flt(i) = tSR_sAlpha_flt(i);
          tSR_rBias_flt(i) = tSR_sBias_flt(i);
          tSR_rMean_flt(i) = tSR_sMean_flt(i);
          tSR_rInvStddev_flt(i) = tSR_sInvStddev_flt(i);
        }

        constexpr int RegFragSize = cute::min(size(tSR_rAlpha_flt), cute::max(1, static_cast<int>(sizeof(uint32_t) / sizeof(ElementCompute))));
        Tensor tSR_rAlpha_frg = recast<Array<ElementCompute, RegFragSize>>(tSR_rAlpha_flt);            // (FRG_V)
        Tensor tSR_rBias_frg = recast<Array<ElementCompute, RegFragSize>>(tSR_rBias_flt);              // (FRG_V)
        Tensor tSR_rMean_frg = recast<Array<ElementCompute, RegFragSize>>(tSR_rMean_flt);              // (FRG_V)
        Tensor tSR_rInvStddev_frg = recast<Array<ElementCompute, RegFragSize>>(tSR_rInvStddev_flt);    // (FRG_V)

        cutlass::multiplies<Array<ElementCompute, RegFragSize>> mul;
        cutlass::negate<Array<ElementCompute, RegFragSize>> negate;
        cutlass::multiply_add<Array<ElementCompute, RegFragSize>> mul_add;

        // We do computation among vectors before computation among matrices
        //                alpha' = alpha * inv_stddev
        //                bias' = bias - alpha' * mean
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSR_rAlpha_frg); ++i) {
          tSR_rAlpha_frg(i) = mul(tSR_rAlpha_frg(i), tSR_rInvStddev_frg(i));
          tSR_rBias_frg(i) = mul_add(tSR_rAlpha_frg(i), negate(tSR_rMean_frg(i)), tSR_rBias_frg(i));
        }

        Tensor tSR_sColAlpha_flt = filter_zeros(tSR_sColAlpha(_,_,_,pipe_index));
        Tensor tSR_sColBias_flt = filter_zeros(tSR_sColBias(_,_,_,pipe_index));
        // After computation, 4 vectors -> 2 vectors
        for (int i = 0; i < size(tSR_rAlpha_flt); ++i) {
          if (get<1>(tSR_cAlpha_flt(i)) >= size<1>(CtaTileShapeMNK{})) {
            // OOB of SMEM
            continue;
          }
          tSR_sColAlpha_flt(i) = tSR_rAlpha_flt(i);
          tSR_sColBias_flt(i) = tSR_rBias_flt(i);
        }

        synchronize();

        // To do bn_apply with Acc, reload these 2 vectors with the consistent shape
        copy_aligned(tCsAlpha(_,_,_,_,_,pipe_index), tCrAlpha);
        copy_aligned(tCsBias(_,_,_,_,_,pipe_index), tCrBias);
      }
    }

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_inputs) {
        constexpr int RegFragSize = cute::max(1, static_cast<int>(sizeof(uint32_t) / sizeof(ElementCompute)));
      cutlass::multiply_add<Array<ElementCompute, RegFragSize>> mul_add;

      Array<ElementCompute, FragmentSize> frg_apply;

      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;

      ConvertInput convert_input{};
      ConvertOutput convert_output{};

      Array frg_I = convert_input(frg_inputs);

      Tensor tCrAlpha_frg = recast<Array<ElementCompute, RegFragSize>>(tCrAlpha(_,_,_,epi_m,epi_n));
      Tensor tCrBias_frg = recast<Array<ElementCompute, RegFragSize>>(tCrBias(_,_,_,epi_m,epi_n));

      constexpr int RegFragArraySize = FragmentSize / RegFragSize;
      using RegFragArr = Array<Array<ElementCompute, RegFragSize>, RegFragArraySize>;
      RegFragArr& frg_I_ = reinterpret_cast<RegFragArr&>(frg_I);
      RegFragArr& frg_apply_ = reinterpret_cast<RegFragArr&>(frg_apply);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < RegFragArraySize; ++i) {
        frg_apply_[i] = mul_add(tCrAlpha_frg(epi_v * RegFragArraySize + i), frg_I_[i], tCrBias_frg(epi_v * RegFragArraySize + i));
      }

      return convert_output(frg_apply);
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    using ThreadCount = decltype(size(args.tiled_copy));

    Tensor sAlpha = make_tensor(make_smem_ptr(smem_alpha),                                        // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor sBias = make_tensor(make_smem_ptr(smem_bias),                                          // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor sColAlpha = make_tensor(make_smem_ptr(smem_col_alpha),                                 // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor sColBias = make_tensor(make_smem_ptr(smem_col_bias),                                   // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor sMean = make_tensor(make_smem_ptr(smem_mean),                                          // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor sInvStddev = make_tensor(make_smem_ptr(smem_inv_stddev),                               // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));

    // S2R: Smem to Reg
    auto tiled_s2r = make_tiled_copy(Copy_Atom<DefaultCopy, ElementScalar>{},
                                     Layout< Shape<_1, ThreadCount>,
                                            Stride<_0,          _1>>{},
                                     Layout<_1>{});
    auto thr_s2r = tiled_s2r.get_slice(args.thread_idx);
    Tensor tSR_sAlpha = thr_s2r.partition_S(sAlpha);
    Tensor tSR_sBias = thr_s2r.partition_S(sBias);
    Tensor tSR_sMean = thr_s2r.partition_S(sMean);
    Tensor tSR_sInvStddev = thr_s2r.partition_S(sInvStddev);
    Tensor tSR_sColAlpha = thr_s2r.partition_S(sColAlpha);
    Tensor tSR_sColBias = thr_s2r.partition_S(sColBias);
    Tensor tSR_cAlpha = thr_s2r.partition_S(args.cD);

    Tensor tSR_rAlpha = make_tensor_like<ElementCompute>(take<0,3>(tSR_sAlpha)); // need to check
    Tensor tSR_rBias = make_tensor_like<ElementCompute>(take<0,3>(tSR_sBias));
    Tensor tSR_rMean = make_tensor_like<ElementCompute>(take<0,3>(tSR_sMean));
    Tensor tSR_rInvStddev = make_tensor_like<ElementCompute>(take<0,3>(tSR_sInvStddev));

    Tensor tCsAlpha = sm90_partition_for_epilogue<ReferenceSrc>(                  // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)
                      sColAlpha, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCsBias = sm90_partition_for_epilogue<ReferenceSrc>(                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)
                      sColBias, args.epi_tile, args.tiled_copy, args.thread_idx);

    Tensor tCrAlpha = make_tensor_like<ElementCompute>(take<0,5>(tCsAlpha));                       // (CPY,CPY_M,CPY_N)
    Tensor tCrBias = make_tensor_like<ElementCompute>(take<0,5>(tCsBias));                         // (CPY,CPY_M,CPY_N)

    constexpr int EpiTiles = decltype(size<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile)))::value;
    return ConsumerStoreCallbacks<EpiTiles
    , decltype(tSR_rAlpha), decltype(tSR_sAlpha), decltype(tSR_cAlpha), decltype(tSR_sColAlpha), decltype(tCrAlpha), decltype(tCsAlpha), ThreadCount
    >(
      cute::move(tSR_rAlpha), cute::move(tSR_rBias),
      cute::move(tSR_rMean), cute::move(tSR_rInvStddev),
      cute::move(tSR_sAlpha), cute::move(tSR_sBias),
      cute::move(tSR_sMean), cute::move(tSR_sInvStddev),
      cute::move(tSR_cAlpha),
      cute::move(tSR_sColAlpha), cute::move(tSR_sColBias),
      cute::move(tCrAlpha), cute::move(tCrBias),
      cute::move(tCsAlpha), cute::move(tCsBias),
      ThreadCount{},
      params_ptr);
  }
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
