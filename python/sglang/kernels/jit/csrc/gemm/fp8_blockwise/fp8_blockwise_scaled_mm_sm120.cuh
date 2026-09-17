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

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

namespace sglang {

using namespace host;

#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace detail {

/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using namespace cute;

/// \brief CUTLASS SM120 producer with streaming weight loads for small decode batches.
/// \tparam Base The blockwise collective returned by the CUTLASS builder.
template <class Base>
struct Sm120Fp8DecodeMainloop : Base {
  using typename Base::DispatchPolicy;
  using typename Base::MainloopPipeline;
  using typename Base::Params;
  using typename Base::PipelineState;
  using typename Base::SmemLayoutA;
  using typename Base::SmemLayoutB;
  using typename Base::SmemLayoutScaleA;
  using typename Base::SmemLayoutScaleB;
  using typename Base::TensorStorage;
  using ClusterShape = typename Base::DispatchPolicy::ClusterShape;
  using Base::ScaleMsPerTile;
  using Base::ScaleNsPerTile;
  using typename Base::SmemBlockScalingCopyAtomA;
  using typename Base::SmemBlockScalingCopyAtomB;
  template <class TensorA, class TensorB, class TensorSFA, class TensorSFB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter,
      int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_scale_A.data()), SmemLayoutScaleA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_scale_B.data()), SmemLayoutScaleB{});

    // Prepare the TMA loads for A and B.

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);  // (BLK_N,BLK_K,k)

    // Block scaling: load_scale has scaling tensors in global memory which are not tiled
    Tensor mSFA_mkl = get<2>(load_inputs);
    Tensor mSFB_nkl = get<3>(load_inputs);
    auto scales_m = get<0>(mSFA_mkl.shape());
    auto scales_n = get<0>(mSFB_nkl.shape());

    Tensor cSFA_mkl = make_identity_tensor(mSFA_mkl.shape());
    Tensor cSFB_nkl = make_identity_tensor(mSFB_nkl.shape());
    Tensor gSFA = local_tile(
        mSFA_mkl, make_tile(Int<ScaleMsPerTile>{}), make_coord(m_coord, _, l_coord));  // (ScaleMsPerTile,k,1)
    Tensor cSFA = local_tile(cSFA_mkl, make_tile(Int<ScaleMsPerTile>{}), make_coord(m_coord, _, l_coord));
    Tensor gSFB = local_tile(
        mSFB_nkl, make_tile(Int<ScaleNsPerTile>{}), make_coord(n_coord, _, l_coord));  // (ScaleNsPerTile,k,1)
    Tensor cSFB = local_tile(cSFB_nkl, make_tile(Int<ScaleNsPerTile>{}), make_coord(n_coord, _, l_coord));

    TiledCopy scale_copy_a = make_tiled_copy(SmemBlockScalingCopyAtomA{}, Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
    TiledCopy scale_copy_b = make_tiled_copy(SmemBlockScalingCopyAtomB{}, Layout<Shape<_32>>{}, Layout<Shape<_1>>{});

    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(thread_idx);
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(thread_idx);

    Tensor tAgA_SFA = thr_scale_copy_a.partition_S(gSFA);
    Tensor tAcA_SFA = thr_scale_copy_a.partition_S(cSFA);
    Tensor tAsA_SFA = thr_scale_copy_a.partition_D(sSFA);

    Tensor tBgB_SFB = thr_scale_copy_b.partition_S(gSFB);
    Tensor tBcB_SFB = thr_scale_copy_b.partition_S(cSFB);
    Tensor tBsB_SFB = thr_scale_copy_b.partition_D(sSFB);

    Tensor tApA_SFA = make_tensor<bool>(shape(tAsA_SFA(_, _, 0)));
    Tensor tBpB_SFB = make_tensor<bool>(shape(tBsB_SFB(_, _, 0)));

    auto scale_m_lim = std::min(scales_m, (m_coord + 1) * ScaleMsPerTile);
    auto scale_n_lim = std::min(scales_n, (n_coord + 1) * ScaleNsPerTile);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tApA_SFA); ++i)
      tApA_SFA(i) = get<0>(tAcA_SFA(i)) < scale_m_lim;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tBpB_SFB); ++i)
      tBpB_SFB(i) = get<0>(tBcB_SFB(i)) < scale_n_lim;

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA,TMA_N,TMA_K,PIPE)

    // TMA Multicast Masks
    Layout cta_layout_mnk = make_layout(ClusterShape{});
    auto cta_coord_mnk = cta_layout_mnk.get_flat_coord(block_rank_in_cluster);

    uint16_t mcast_mask_a = create_tma_multicast_mask<1>(cta_layout_mnk, cta_coord_mnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<0>(cta_layout_mnk, cta_coord_mnk);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (lane_predicate) {
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        // A is the streaming weight operand; retain the normal policy for activations.
        copy(
            mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a, cute::TMA::CacheHintSm90::EVICT_FIRST),
            tAgA(_, _, _, *k_tile_iter),
            tAsA(_, _, _, write_stage));
        copy(
            mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b, cute::TMA::CacheHintSm90::EVICT_NORMAL),
            tBgB(_, _, _, *k_tile_iter),
            tBsB(_, _, _, write_stage));
      }

      // Copy scale tensors
      copy_if(scale_copy_a, tApA_SFA, tAgA_SFA(_, _, *k_tile_iter), tAsA_SFA(_, _, write_stage));
      copy_if(scale_copy_b, tBpB_SFB, tBgB_SFB(_, _, *k_tile_iter), tBsB_SFB(_, _, write_stage));
      pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }
};

template <typename OutType, int TM, int WN, int STAGES>
__global__ void __launch_bounds__(32) sm120_fp8_decode_warp(
    float* __restrict__ tmp,
    OutType* __restrict__ out,
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    const float* __restrict__ sa,
    const float* __restrict__ sb,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t splits) {
  static_assert(TM == 8 || TM == 16);
  static_assert(WN == 16 || WN == 32);
  extern __shared__ __align__(128) uint8_t storage[];
  const int32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, row = lane / 4, col = lane % 4;
  auto* mem = reinterpret_cast<uint8_t (*)[(TM + WN) * 128]>(storage + warp * STAGES * (TM + WN) * 128);
  int32_t n0 = (blockIdx.x + warp) * WN, m0 = blockIdx.y * TM;
  int32_t groups = k / 128, begin = groups * blockIdx.z / splits, end = groups * (blockIdx.z + 1) / splits;
  // Apply both scales after each 128-element FP32 dot product.
  float acc[WN / 16][TM / 8][4] = {};
  // Each stage is warp-local; XOR swizzling makes ldmatrix loads bank-conflict free.
  auto load = [&](int32_t g) {
    int32_t stage = (g - begin) % STAGES;
#pragma unroll
    for (int v = lane; v < (TM + WN) * 8; v += 32) {
      int32_t r = v / 8, c = (v % 8) * 16;
      bool valid = r < WN || m0 + r - WN < m;
      const uint8_t* src =
          r < WN ? b + int64_t(n0 + r) * k + g * 128 + c : a + int64_t(valid ? m0 + r - WN : 0) * k + g * 128 + c;
      cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
          *reinterpret_cast<const cute::uint128_t*>(src),
          *reinterpret_cast<cute::uint128_t*>(mem[stage] + r * 128 + (c ^ ((r % 8) * 16))),
          valid);
    }
    cute::cp_async_fence();
  };
  for (int s = 0; s < STAGES - 1; ++s) {
    if (begin + s < end) {
      load(begin + s);
    }
  }
  for (int g = begin; g < end; ++g) {
    if (g + STAGES - 1 < end) {
      load(g + STAGES - 1);
      cute::cp_async_wait<STAGES - 1>();
    } else {
      cute::cp_async_wait<0>();
    }
    __syncwarp();
    auto* w = mem[(g - begin) % STAGES];
    auto* x = w + WN * 128;
    float part[WN / 16][TM / 8][4] = {};
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t br[TM / 8][2];
#pragma unroll
      for (int mm = 0; mm < TM / 8; ++mm) {
        int32_t ar = mm * 8 + lane % 8, ac = kk * 32 + (lane / 8 % 2) * 16;
        cute::SM75_U32x2_LDSM_N::copy(
            *reinterpret_cast<const cute::uint128_t*>(x + ar * 128 + (ac ^ ((ar % 8) * 16))), br[mm][0], br[mm][1]);
      }
#pragma unroll
      for (int wn = 0; wn < WN / 16; ++wn) {
        uint32_t ar[4];
        int32_t wr = wn * 16 + lane % 16, wc = kk * 32 + (lane / 16) * 16;
        cute::SM75_U32x4_LDSM_N::copy(
            *reinterpret_cast<const cute::uint128_t*>(w + wr * 128 + (wc ^ ((wr % 8) * 16))),
            ar[0],
            ar[1],
            ar[2],
            ar[3]);
#pragma unroll
        for (int mm = 0; mm < TM / 8; ++mm) {
          auto& p = part[wn][mm];
          cute::SM120_16x8x32_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float>::fma(
              p[0], p[1], p[2], p[3], ar[0], ar[1], ar[2], ar[3], br[mm][0], br[mm][1], p[0], p[1], p[2], p[3]);
        }
      }
    }
#pragma unroll
    for (int wn = 0; wn < WN / 16; ++wn) {
      float ws = sb[((n0 + wn * 16) / 128) * groups + g];
#pragma unroll
      for (int mm = 0; mm < TM / 8; ++mm) {
        int32_t token = m0 + mm * 8 + col * 2;
        float s0 = token < m ? ws * sa[g * m + token] : 0, s1 = token + 1 < m ? ws * sa[g * m + token + 1] : 0;
        acc[wn][mm][0] += part[wn][mm][0] * s0;
        acc[wn][mm][1] += part[wn][mm][1] * s1;
        acc[wn][mm][2] += part[wn][mm][2] * s0;
        acc[wn][mm][3] += part[wn][mm][3] * s1;
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int wn = 0; wn < WN / 16; ++wn) {
#pragma unroll
    for (int mm = 0; mm < TM / 8; ++mm) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        int32_t token = m0 + mm * 8 + col * 2 + v % 2, output_col = n0 + wn * 16 + row + (v / 2) * 8;
        if (token < m) {
          if (splits == 1) {
            out[int64_t(token) * n + output_col] = OutType(acc[wn][mm][v]);
          } else {
            tmp[(int64_t(blockIdx.z) * m + token) * n + output_col] = acc[wn][mm][v];
          }
        }
      }
    }
  }
}

template <typename OutType>
__global__ void
sm120_fp8_decode_reduce(OutType* __restrict__ out, const float* __restrict__ tmp, int64_t size, int32_t splits) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    float v = 0;
    for (int32_t s = 0; s < splits; ++s) {
      v += tmp[s * size + i];
    }
    out[i] = OutType(v);
  }
}
}  // namespace detail

template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    int TileSizeM_ = 128,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  // A matrix configuration
  using ElementA = cutlass::float_e4m3_t;        // Element type for A matrix operand
  using LayoutATag = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A matrix in units of
                                                    // elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::float_e4m3_t;           // Element type for B matrix operand
  using LayoutBTag = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B matrix in units of
                                                    // elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = OutType;                      // Element type for D matrix operand
  using ElementC = void;                         // Element type for C matrix operand
  using LayoutCTag = cutlass::layout::RowMajor;  // Layout type for C matrix operand
  using LayoutDTag = cutlass::layout::RowMajor;  // Layout type for D matrix operand
  constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;  // Memory access granularity/alignment of C matrix in units of
                                                    // elements (up to 16 bytes)
  constexpr int AlignmentC =
      AlignmentD;  // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Kernel functional config
  using ElementAccumulator = float;      // Element type for internal accumulation
  using ArchTag = cutlass::arch::Sm120;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag - changed from OpClassBlockScaledTensorOp

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::MN,
      cute::UMMA::Major::K>;
  // FP8 Block-wise scaling configuration
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand

  constexpr bool kCanUsePingpong = (64 % ScaleGranularityM == 0);

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  auto scales_a_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());
  auto scales_b_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());

  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptr, stride_a, b_ptr, stride_b, scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CooperativeCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CooperativeStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CooperativeCollectiveEpilogue::SharedStorage))>;

  using CooperativeCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      CooperativeStageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using CooperativeGemmKernelStreamK = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CooperativeCollectiveMainloop,
      CooperativeCollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using CooperativeGemmKernelVoid = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, CooperativeCollectiveMainloop, CooperativeCollectiveEpilogue, void>;

  auto run_cooperative = [&]() -> cutlass::Status {
    static const uint32_t kNumSM = host::runtime::get_sm_count(a.device().device_id);
    constexpr int kTileM = size<0>(MmaTileShape{});
    constexpr int kTileN = size<1>(MmaTileShape{});
    uint64_t tiles = static_cast<uint64_t>((m + kTileM - 1) / kTileM) * ((n + kTileN - 1) / kTileN);
    uint32_t last_wave = static_cast<uint32_t>(tiles % kNumSM);
    if (last_wave == 0) last_wave = kNumSM;
    float waste = 1.0f - static_cast<float>(last_wave) / static_cast<float>(kNumSM);
    return (waste > 0.5f) ? run_gemm(CooperativeGemmKernelStreamK{}) : run_gemm(CooperativeGemmKernelVoid{});
  };

  cutlass::Status status = cutlass::Status::kSuccess;
  if constexpr (kCanUsePingpong) {
    using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;
    using PingpongCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        PerSmTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutCTag,
        AlignmentC,
        ElementD,
        LayoutDTag,
        AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using PingpongStageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename PingpongCollectiveEpilogue::SharedStorage))>;

    using PingpongCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        cute::tuple<LayoutATag, LayoutSFA>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutBTag, LayoutSFB>,
        AlignmentB,
        ElementAccumulator,
        PingpongMmaTileShape_MNK,
        ClusterShape,
        PingpongStageCount,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120>::CollectiveOp;

    using PingpongGemmKernel = cutlass::gemm::kernel::
        GemmUniversal<Shape<int, int, int, int>, PingpongCollectiveMainloop, PingpongCollectiveEpilogue, void>;

    if (m <= 64) {
      status = run_gemm(PingpongGemmKernel{});
      if (status != cutlass::Status::kSuccess) {
        status = run_cooperative();
      }
    } else {
      status = run_cooperative();
    }
  } else {
    status = run_cooperative();
  }

  CUTLASS_CHECK(status);
}

// Transposed GEMM D^T = Wgemm(weight, activation): puts tokens on the N axis.
// EpilogueTileShape selects the epilogue subtile; EpilogueTileAuto resolves to
// (64, min(CTA_N,32)) here -- see sm120_builder.inl.
template <
    typename OutType,
    typename MmaTileShape,
    typename PerSmTileShape,
    typename EpilogueTileShape,
    typename ScalesPerTile,
    bool OptimizeDecode = false,
    class ClusterShape = Shape<_1, _1, _1>>
void launch_sm120_fp8_blockwise_scaled_mm_swapab(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;        // A' = weight
  using LayoutATag = cutlass::layout::RowMajor;  // weight [N, K] is row-major
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;           // B' = activation
  using LayoutBTag = cutlass::layout::ColumnMajor;  // activation as [K, M] column-major
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::ColumnMajor;  // D' = out^T is column-major
  using LayoutDTag = cutlass::layout::ColumnMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM = size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN = size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK = size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  // Operands are swapped, so the scale majors swap relative to the non-swap path:
  // SFA (weight) is K-major; SFB (per-token activation) is MN-major.
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      ScaleGranularityM,
      ScaleGranularityN,
      ScaleGranularityK,
      cute::UMMA::Major::K,
      cute::UMMA::Major::MN>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  int m = a.size(0);  // original tokens  -> swapped N'
  int k = a.size(1);
  int n = b.size(1);  // original weight cols -> swapped M'

  auto weight_ptr = static_cast<ElementA*>(b.data_ptr());
  auto act_ptr = static_cast<ElementB*>(a.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  auto weight_scale_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());
  auto act_scale_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());

  // Swapped problem shape (M', N', K) = (n, m, k).
  LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1));
  LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1));

  auto run_gemm = [&](auto tag) -> cutlass::Status {
    using GemmKernel = decltype(tag);
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    Gemm gemm_op;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(n, m, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        weight_ptr, stride_a, act_ptr, stride_b, weight_scale_ptr, layout_SFA, act_scale_ptr, layout_SFB};

    typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, stride_c, c_ptr, stride_c};
    epilogue_args.thread.alpha = 1.0f;

    typename Gemm::Arguments args = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {n, m, k, 1},
        mainloop_args,
        epilogue_args,
    };

    auto can_implement = gemm_op.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
      return can_implement;
    }

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace_tensor = host::ffi::alloc_workspace_tensor(workspace_size, a.device());
    void* workspace = (workspace_size == 0) ? nullptr : workspace_tensor.data_ptr();

    auto init_status = gemm_op.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
      return init_status;
    }

    return gemm_op.run(stream);
  };

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      EpilogueTileShape,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutATag, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutBTag, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using CollectiveMainloop =
      cute::conditional_t<OptimizeDecode, detail::Sm120Fp8DecodeMainloop<BaseMainloop>, BaseMainloop>;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  CUTLASS_CHECK(run_gemm(GemmKernel{}));
}

/// \brief Run a warp-local FP8 GEMM with FP32 split-K reduction for at most 16 tokens.
/// \tparam OutType FP16 or BF16 output element type.
template <typename OutType>
void launch_sm120_fp8_decode_warp(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  const int32_t m = static_cast<int32_t>(a.size(0));
  const int32_t k = static_cast<int32_t>(a.size(1));
  const int32_t n = static_cast<int32_t>(b.size(1));
  constexpr int32_t splits = 8;
  const int64_t output_size = int64_t(m) * n;
  auto workspace = host::ffi::alloc_workspace_tensor(splits * output_size * sizeof(float), a.device());
  auto* partials = static_cast<float*>(workspace.data_ptr());
  auto launch = [&]<int TM, int WN>() {
    constexpr int stages = 2;
    constexpr int smem = stages * (TM + WN) * 128;
    host::LaunchKernel(dim3(n / WN, 1, splits), 32, stream, smem)(
        detail::sm120_fp8_decode_warp<OutType, TM, WN, stages>,
        partials,
        static_cast<OutType*>(out.data_ptr()),
        static_cast<const uint8_t*>(a.data_ptr()),
        static_cast<const uint8_t*>(b.data_ptr()),
        static_cast<const float*>(scales_a.data_ptr()),
        static_cast<const float*>(scales_b.data_ptr()),
        m,
        n,
        k,
        splits);
  };
  if (m <= 8) {
    launch.template operator()<8, 16>();
  } else {
    launch.template operator()<16, 32>();
  }
  host::LaunchKernel(host::div_ceil(output_size, int64_t(256)), 256, stream)(
      detail::sm120_fp8_decode_reduce<OutType>, static_cast<OutType*>(out.data_ptr()), partials, output_size, splits);
}

// swapAB (tile N=32) beats the non-swap 128x128 path for M<=64 or M%4!=0
// (cold-L2 CUPTI benchmarks, up to ~1.2x); tile N=16 is unsupported by the
// SM120 blockwise collective (needs EPI_TILE_N=32 | CTA_N and B LDSM N>=32).
template <typename OutType>
void sm120_fp8_blockwise_dispatch_shape(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    cudaStream_t stream) {
  const int m = a.size(0);
  const int n = b.size(1);
  const int k = a.size(1);
  using EpilogueTileShape = Shape<_128, _64>;

  if (m > 0 && m <= 64 && m % 4 == 0 && n >= 4096 && k >= 4096) {
    const int num_sms = host::runtime::get_sm_count(a.device().device_id);
    const int weight_tiles = (n + 127) / 128;
    // Warp-local tiles expose enough parallelism for narrow, short reductions.
    if (m <= 16 && n <= 32 * num_sms && n % 128 == 0 && k % 128 == 0 && k <= 8192 && (m <= 8 || k >= 6144)) {
      launch_sm120_fp8_decode_warp<OutType>(out, a, b, scales_a, scales_b, stream);
      return;
    }
    // A wider token tile removes a partial second wave for medium-width weights.
    if (m > 32 && weight_tiles < num_sms && weight_tiles * 2 > num_sms) {
      launch_sm120_fp8_blockwise_scaled_mm_swapab<
          OutType,
          Shape<_128, _64, _128>,
          Shape<_128, _64, _128>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          Shape<_1, _64, _1>,
          true>(out, a, b, scales_a, scales_b, stream);
      return;
    }
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _32, _128>,
        Shape<_128, _32, _128>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Shape<_1, _32, _1>,
        true>(out, a, b, scales_a, scales_b, stream);
    return;
  }

  // swapAB keeps the weight on the gemm-M axis and the tokens on gemm-N, so it reads the
  // weight once per token tile and needs only (128+TileN)*128 bytes of smem per stage.
  // It stays ahead of the non-swapAB path well past the old m<=64 crossover: autotuned
  // over 12 tactics x M in {4..1024} x all five Qwen3.x-27B-FP8 TP1 decode shapes
  // (cold-L2 CUPTI + CUDA graph, one GPU per shape), the old crossover cost 4.5-7.3% of a
  // full pass for m in [96, 256] -- e.g. out_proj at m=96: 51.74us non-swapAB (StreamK,
  // 188 CTAs) vs 38.23us swapAB (120 CTAs).
  if (m <= 128 || (m % 4 != 0)) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _32, _128>,
        Shape<_128, _32, _128>,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Shape<_1, _32, _1>>(out, a, b, scales_a, scales_b, stream);
    return;
  }

  // 128 < m <= 256: a 64-wide token tile amortizes the weight read over twice as many
  // tokens per pass. 3 stages instead of 4 ((128+64)*128 = 24576 B per stage), which the
  // sweep shows costs nothing.
  if (m <= 256) {
    launch_sm120_fp8_blockwise_scaled_mm_swapab<
        OutType,
        Shape<_128, _64, _128>,
        Shape<_128, _64, _128>,
        Shape<_128, _32>,
        Shape<_1, _64, _1>>(out, a, b, scales_a, scales_b, stream);
    return;
  }

  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape = Shape<_128, _128, _128>;
  using ScalesPerTile = Shape<_128, _1, _1>;
  launch_sm120_fp8_blockwise_scaled_mm<OutType, MmaTileShape, PerSmTileShape, EpilogueTileShape, ScalesPerTile>(
      out, a, b, scales_a, scales_b, stream);
}

inline void fp8_blockwise_scaled_mm_sm120(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b) {
  RuntimeCheck(mat_a.device().device_type == kDLCUDA, "mat_a must be a CUDA tensor");
  RuntimeCheck(mat_b.device().device_type == kDLCUDA, "mat_b must be a CUDA tensor");

  RuntimeCheck(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  RuntimeCheck(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  RuntimeCheck(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  RuntimeCheck(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  RuntimeCheck(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  RuntimeCheck(
      (mat_a.size(1) * (mat_a.dtype().bits / 8)) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(
      (mat_b.size(0) * (mat_b.dtype().bits / 8)) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_a.dtype()), "mat_a must be Float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(mat_b.dtype()), "mat_b must be Float8_e4m3fn");

  RuntimeCheck(mat_a.size(0) == scales_a.size(0), "size of scales_a is not matched");
  RuntimeCheck(mat_a.size(1) / 128 == scales_a.size(1), "size of scales_a is not matched");
  RuntimeCheck(mat_b.size(0) / 128 == scales_b.size(0), "size of scales_b is not matched");
  RuntimeCheck(mat_b.size(1) / 128 == scales_b.size(1), "size of scales_b is not matched");
  RuntimeCheck(host::is_type<float>(scales_a.dtype()), "scales_a must be Float32");
  RuntimeCheck(host::is_type<float>(scales_b.dtype()), "scales_b must be Float32");

  RuntimeCheck(
      (out.size(1) * (out.dtype().bits / 8)) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  const cudaStream_t stream = LaunchKernel::resolve_device(mat_a.device());

  if (host::is_type<bf16_t>(out.dtype())) {
    sm120_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, stream);
  } else if (host::is_type<fp16_t>(out.dtype())) {
    sm120_fp8_blockwise_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, stream);
  } else {
    Panic("out_dtype must be Half or BFloat16");
  }
}

#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

}  // namespace sglang
