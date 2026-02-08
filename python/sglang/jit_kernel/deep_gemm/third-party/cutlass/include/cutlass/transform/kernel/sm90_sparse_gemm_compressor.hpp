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
  \brief Compress utils specific for SM90 structure sparse kernels
*/

#pragma once

#include "cute/container/bit_field.hpp"    // cute::bit_field
#include "cute/numeric/numeric_types.hpp"  // cute::sizeof_bits_v, cute::uint_bit_t
#include "cute/tensor.hpp"                 // cute::Tensor, cute::make_tensor
#include "cute/algorithm/cooperative_copy.hpp" // cute::cooperative_copy
#include "cutlass/arch/arch.h"             // cutlass::arch::Sm90
#include "cutlass/cuda_host_adapter.hpp"   // cutlass::CudaHostAdapter
#include "cutlass/cutlass.h"               // cutlass::Status
#include "cutlass/gemm/gemm.h"             // cutlass::TagToStrideA_t
#include "cutlass/fast_math.h"             // cutlass::ceil_div, cutlass::round_up
#include "cutlass/kernel_hardware_info.h"  // cutlass::KernelHardwareInfo
#include "cutlass/numeric_size.h"          // cutlass::bits_to_bytes
#include "cutlass/numeric_types.h"         // cutlass::has_negative_zero_v
#include "cutlass/cuda_host_adapter.hpp"   // cutlass::CudaHostAdapter

namespace cutlass::transform::kernel {

using namespace cute;

template<
  class ProblemShape_,
  class ElementA_,
  class LayoutATag_,
  class SparseConfig_
>
class SM90StructuredSparseCompressor {
public:
  using SparseConfig = SparseConfig_;
  using ProblemShape = ProblemShape_;

  // * EltA
  using ElementA = ElementA_;
  using ElementAUint = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  using ElementAMma = typename SparseConfig::ElementAMma;
  using ElementAMmaRaw = typename SparseConfig::ElementAMmaRaw;
  using ElementAMmaRawUnit = cute::uint_bit_t<cute::sizeof_bits_v<ElementAMmaRaw>>;
  using ElementASparsity = typename SparseConfig::ElementASparsity;
  using ElementAMmaSparsity = typename SparseConfig::ElementAMmaSparsity;
  using ElementAUintCompressed = cute::sparse_elem<ElementASparsity{}, ElementAUint>;
  using LayoutATag = LayoutATag_;
  using LayoutA = LayoutATag;
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutATag>;

  // * EltE
  using ElementEMma = typename SparseConfig::ElementEMma;
  using ElementEMmaRaw = typename SparseConfig::ElementEMmaRaw;
  using ElementEMmaSparsity = typename SparseConfig::ElementEMmaSparsity;
  // Data Type for storing one chunk's metadata
  static constexpr int ElementEBitsPerChunk = typename SparseConfig::ElementEBitsPerChunk{};
  CUTE_STATIC_ASSERT(ElementEBitsPerChunk == 4, "ElementEBitsPerChunk is 4 for SM90");
  using ElementEChunk = cute::uint_bit_t<ElementEBitsPerChunk>;
  CUTE_STATIC_ASSERT(cute::is_same_v<ElementEChunk, cute::uint4_t>, "ElementEChunk is uint4_t for SM90");
  using ElementESparsityPerChunk = Int<ElementEMmaSparsity{} / (cute::sizeof_bits_v<ElementEMmaRaw> / ElementEBitsPerChunk)>;

  // AtomE
  using TensorEAtom = typename SparseConfig::TensorEAtom;
  using TensorEAtomK = typename SparseConfig::TensorEAtomK;
  using TensorEAtomM = typename SparseConfig::TensorEAtomM;

  static constexpr int ElemsARawPerElementAMmaRaw = typename SparseConfig::ElemsARawPerElementAMmaRaw{};
  static constexpr int LogicalElemsAPerChunk = typename SparseConfig::LogicalElemsAPerChunk{};
  static constexpr int PhysicalElemsAPerChunk = typename SparseConfig::PhysicalElemsAPerChunk{};
  static constexpr int LogicalElemsAMmaRawPerChunk = cutlass::ceil_div(LogicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);
  static constexpr int PhysicalElemsAMmaRawPerChunk = cutlass::ceil_div(PhysicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);

  // * Alignment
  static constexpr int TensorEAlignmentM = typename SparseConfig::TensorEAlignmentM{};
  static constexpr int TensorEAlignmentK = typename SparseConfig::TensorEAlignmentK{};
  static constexpr int TensorAAlignmentK = typename SparseConfig::TensorAAlignmentK{};
  static constexpr int TensorAAlignmentM = typename SparseConfig::TensorAAlignmentM{};

  // Required by `device_kernel`
  static constexpr int MaxThreadsPerBlock = TensorEAtomM{};
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using ArchTag = arch::Sm90;

  struct SharedStorage {
    ElementEMma cEsE[cute::size(TensorEAtom{})];
    ElementAUintCompressed cACsAC[cute::size(TensorEAtom{})];
    ElementAUint cAsA[cute::size(TensorEAtom{})];
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct TransformArguments {
    void const* ptr_A{nullptr};
    StrideA dA{};
    void* ptr_ACompress{nullptr};
    void* ptr_E{nullptr};
  };

  using TransformParams = TransformArguments;

  struct Arguments {
    ProblemShape problem_shape{};
    TransformArguments transform{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    ProblemShape problem_shape{};
    TransformParams transform{};
    KernelHardwareInfo hw_info{};
    void* workspace = nullptr;
  };

public:
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace = nullptr) {
    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::to_underlying_arguments()");
    return Params{{args.problem_shape},
                  {args.transform.ptr_A, args.transform.dA, args.transform.ptr_ACompress, args.transform.ptr_E},
                  {args.hw_info},
                  workspace};
  }

  static Status
  can_implement(Arguments const& args) {
    auto [M, N, K, L] = args.problem_shape;
    if (K % LogicalElemsAPerChunk != 0) {
      CUTLASS_TRACE_HOST("SM90 Sparse Compressor CAN NOT IMPLEMENT: GemmK not multiplier of logical chunk size");
      return Status::kErrorInvalidProblem;
    }
    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::can_implement() (True)");
    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    CUTLASS_UNUSED(args);
    // Backward compatible with host compressor
    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::get_workspace_size() (" << SharedStorageSize << ")");
    return SharedStorageSize;
  }

  static Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    CUTLASS_UNUSED(args);
    CUTLASS_UNUSED(workspace);
    CUTLASS_UNUSED(stream);
    CUTLASS_UNUSED(cuda_adapter);
    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::initialize_workspace()");
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    constexpr int MaxAlignmentM = cutlass::const_max(TensorEAlignmentM, TensorAAlignmentM);
    constexpr int MaxAlignmentK = cutlass::const_max(TensorEAlignmentK, TensorAAlignmentK);
    const auto [GemmM, GemmN, GemmK, GemmL] = params.problem_shape;

    const int GemmMAlignedMax = cutlass::round_up(GemmM, MaxAlignmentM);
    const int GemmKAlignedMax = cutlass::round_up(GemmK, MaxAlignmentK);

    const int gridDim_X = cutlass::ceil_div(GemmMAlignedMax, TensorEAtomM{});
    const int gridDim_Y = cutlass::ceil_div(GemmKAlignedMax, TensorEAtomK{});
    const int gridDim_Z = GemmL;

    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::get_grid_shape() ("
      << gridDim_X << ", "
      << gridDim_Y << ", "
      << gridDim_Z << ")");
    return dim3(gridDim_X, gridDim_Y, gridDim_Z);
  }

  static dim3
  get_block_shape() {
    CUTLASS_TRACE_HOST("SM90StructuredSparseCompressor::get_block_shape() ("
      << MaxThreadsPerBlock << ", "
      << 1 << ", "
      << 1 << ")");
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTE_DEVICE
  void
  operator()(Params params, void* smem_buf = nullptr) {
    run(params, smem_buf);
  }

  CUTE_DEVICE
  static void
  run(Params params, void* smem_buf = nullptr) {
    structure_sparse_compress(params, smem_buf);
  }

private:

  struct MetadataOneChunk1to2 {

    CUTE_DEVICE
    void set_metadata_bits(int elt_log_idx, int elt_phy_idx) {
      auto metadata_bits = [&]() -> uint8_t {
        CUTLASS_ASSERT(elt_log_idx >= 0 && elt_log_idx < 2);
        switch (elt_log_idx) {
          case 0:
            return 0b0100;
          case 1:
            return 0b1110;
          default:
            CUTE_GCC_UNREACHABLE;
        }
      };

      storage_ |= (metadata_bits() << (4 * elt_phy_idx));
    }


    CUTE_DEVICE
    ElementEChunk storage() const {
      return ElementEChunk{storage_};
    }

  private:
    uint8_t storage_ = 0b0000;
  };

  struct MetadataOneChunk2to4{

    CUTE_DEVICE
    void set_metadata_bits(int elt_log_idx, int elt_phy_idx) {
      auto metadata_bits = [&]() -> uint8_t {
        CUTLASS_ASSERT(elt_log_idx >= 0 && elt_log_idx < 4);
        switch (elt_log_idx) {
          case 0:
            return 0b00;
          case 1:
            return 0b01;
          case 2:
            return 0b10;
          case 3:
            return 0b11;
          default:
            CUTLASS_ASSERT(false);
            CUTE_GCC_UNREACHABLE;
            return 0b00;
        }
      };

      storage_ |= (metadata_bits() << (2 * elt_phy_idx));
    }

    CUTE_DEVICE
    ElementEChunk storage() const {
      return ElementEChunk{storage_};
    }

  private:
    uint8_t storage_ = 0b0000;
  };

  using MetadataOneChunk = cute::conditional_t<SparseConfig::IsTF32,
                                               MetadataOneChunk1to2,
                                               MetadataOneChunk2to4>;

private:

  CUTE_DEVICE
  static void
  structure_sparse_compress(Params params, void* smem_buf) {
    // * Input Params
    auto [GemmM, GemmN, GemmK, GemmL] = params.problem_shape;
    auto [ptr_A, dA, ptr_ACompress, ptr_E] = params.transform;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    [[maybe_unused]] const int gridDim_X = gridDim.x;
    [[maybe_unused]] const int gridDim_Y = gridDim.y;
    [[maybe_unused]] const int gridDim_Z = gridDim.z;
    [[maybe_unused]] const int blockDim_X = blockDim.x;

    // * Global Tensor Layout
    const cute::Layout layout_gA = make_layout(make_shape(GemmM, GemmK, GemmL), dA);
    const cute::Layout layout_gAC = SparseConfig::fill_layoutA(params.problem_shape);
    const cute::Layout layout_gE = SparseConfig::fill_layoutE(params.problem_shape);

    // * Construct Global Tensor
    const cute::Tensor gA   = make_tensor(make_gmem_ptr(cute::recast_ptr<ElementAUint>(ptr_A)), layout_gA);
    cute::Tensor gAC_sparse = make_tensor(make_gmem_ptr(cute::recast_ptr<ElementAUintCompressed>(ptr_ACompress)), layout_gAC );
    cute::Tensor gAC        = cute::recast<ElementAUint>(gAC_sparse);
    cute::Tensor gE_sparse  = make_tensor(make_gmem_ptr(cute::recast_ptr<ElementEMma>(ptr_E)), layout_gE);
    cute::Tensor gE         = cute::recast<ElementEMmaRaw>(gE_sparse);

    // * CTA Tensor Layout
    using cAsA_layout_row = decltype(make_layout(make_shape(TensorEAtomM{}, TensorEAtomK{}), LayoutRight{}));
    using cAsA_layout_col = decltype(make_layout(make_shape(TensorEAtomM{}, TensorEAtomK{}), LayoutLeft{}));
    using cAsA_layout     = cute::conditional_t<cute::is_same_v<LayoutATag, layout::RowMajor>, cAsA_layout_row, cAsA_layout_col>;
    using cACsAC_layout   = decltype(make_layout(make_shape(TensorEAtomM{}, TensorEAtomK{} / ElementASparsity{}), LayoutRight{}));
    using cEsE_layout     = decltype(make_layout(make_shape(TensorEAtomM{}, TensorEAtomK{} / ElementEMmaSparsity{}), LayoutRight{}));

    CUTE_STATIC_ASSERT(cute::is_static_v<TensorEAtom>, "TensorEAtom needs to be static");
    CUTE_STATIC_ASSERT(cute::is_static_v<cAsA_layout>, "cAsA_layout needs to be static");
    CUTE_STATIC_ASSERT(cute::is_static_v<cACsAC_layout>, "cACsAC_layout needs to be static");
    CUTE_STATIC_ASSERT(cute::is_static_v<cEsE_layout>, "cEsE_layout needs to be static");

    const int blockIdx_X = blockIdx.x;
    const int blockIdx_Y = blockIdx.y;
    const int blockIdx_Z = blockIdx.z;
    const int threadIdx_X = threadIdx.x;

    // * Construct CTA Tensor
    const auto cta_coord = make_coord(blockIdx_X, blockIdx_Y, blockIdx_Z);
    cute::Tensor cAgA   = cute::recast<ElementAMmaRawUnit>(local_tile(gA, shape(cAsA_layout{}), cta_coord));
    cute::Tensor cACgAC = cute::recast<ElementAMmaRawUnit>(local_tile(gAC, shape(cACsAC_layout{}), cta_coord));
    cute::Tensor cEgE   = local_tile(gE, shape(cEsE_layout{}), cta_coord);

    cute::Tensor cAsA   = cute::recast<ElementAMmaRawUnit>(make_tensor(make_smem_ptr(cute::recast_ptr<ElementAUint>(shared_storage.cAsA)), cAsA_layout{}));
    cute::Tensor cACsAC = cute::recast<ElementAMmaRawUnit>(make_tensor(make_smem_ptr(cute::recast_ptr<ElementAUint>(shared_storage.cACsAC)), cACsAC_layout{}));
    cute::Tensor cEsE   = make_tensor(make_smem_ptr(cute::recast_ptr<ElementEMmaRaw>(shared_storage.cEsE)), cEsE_layout{});
    cute::Tensor cEsE_chunk = cute::recast<ElementEChunk>(cEsE);

    // * Handle in unit of Chunk when compress
    using OneChunkSizeA  = Int<LogicalElemsAMmaRawPerChunk>;
    using OneChunkSizeAC = Int<PhysicalElemsAMmaRawPerChunk>;
    using OneChunkSizeE  = Int<LogicalElemsAPerChunk / ElementESparsityPerChunk{}>;
    using NumOneChunkK   = Int<cutlass::ceil_div(TensorEAtomK{}, LogicalElemsAPerChunk)>;

    cute::Tensor cAsA_log_chunk   = logical_divide(cAsA, make_shape(_, OneChunkSizeA{}));
    cute::Tensor cACsAC_log_chunk = logical_divide(cACsAC, make_shape(_, OneChunkSizeAC{}));
    cute::Tensor cEsE_log_chunk   = logical_divide(cEsE_chunk, make_shape(_, OneChunkSizeE{}));

    // * Corner Case Handle
    const auto GemmM_within_Cta = (GemmM - blockIdx_X * TensorEAtomM{} > TensorEAtomM{}) ? TensorEAtomM{} : GemmM - blockIdx_X * TensorEAtomM{};
    const auto GemmK_within_Cta = ( (GemmK - blockIdx_Y * TensorEAtomK{} > TensorEAtomK{}) ? TensorEAtomK{} : GemmK - blockIdx_Y * TensorEAtomK{} ) / ElemsARawPerElementAMmaRaw;
    const auto GemmK_NumOneChunk_within_Cta = GemmK_within_Cta / LogicalElemsAMmaRawPerChunk;

    const auto GemmMAlignedAC = cutlass::round_up(GemmM, TensorAAlignmentM);
    const auto GemmKAlignedAC = cutlass::round_up(GemmK, TensorAAlignmentK);
    const auto GemmMAlignedAC_within_Cta = (GemmMAlignedAC - blockIdx_X * TensorEAtomM{} > TensorEAtomM{}) ? TensorEAtomM{} : GemmMAlignedAC - blockIdx_X * TensorEAtomM{};
    const auto GemmKAlignedAC_within_Cta = ( (GemmKAlignedAC - blockIdx_Y * TensorEAtomK{} > TensorEAtomK{}) ? TensorEAtomK{} : GemmKAlignedAC - blockIdx_Y * TensorEAtomK{} ) / ElemsARawPerElementAMmaRaw;

    // * Clear CTA Smem Tensor
    cooperative_clear<MaxThreadsPerBlock>(threadIdx_X, cACsAC);
    cooperative_clear<MaxThreadsPerBlock>(threadIdx_X, cEsE);

    // * Input CTA Tensor G to S
    if (GemmM_within_Cta == TensorEAtomM{} && GemmK_within_Cta == TensorEAtomK{}) {
      copy_vec_pred<false, LayoutATag>(cAgA, cAsA, threadIdx_X, GemmM_within_Cta, GemmK_within_Cta);
    }
    else {
      copy_vec_pred<true, LayoutATag>(cAgA, cAsA, threadIdx_X, GemmM_within_Cta, GemmK_within_Cta);
    }

    // Construct a sign bit mask for handling negative zeros 
    ElementAMmaRawUnit sign_mask = ElementAMmaRawUnit{ 0 };
    if constexpr (has_negative_zero_v<ElementA>) {
      ElementAMmaRawUnit one_sign_mask = static_cast<ElementAMmaRawUnit>(~(ElementAMmaRawUnit{ 1 } << (cute::sizeof_bits_v<ElementA> - 1)));
      for (int i = 0; i < sizeof(ElementAMmaRawUnit) / sizeof(ElementAUint); ++i) {
        sign_mask = static_cast<ElementAMmaRawUnit>((int32_t)sign_mask | (int32_t)one_sign_mask << (i * cute::sizeof_bits_v<ElementA>));
      }
    }

    // * Compress
    // cACsAC is always row major order
    // TensorEAtomM threads perform the compression, each thread compress one row
    const int row_i = threadIdx_X;
    if (row_i < GemmM_within_Cta) {

      CUTE_UNROLL
      for (int col_chunk_i = 0; col_chunk_i < NumOneChunkK{}; ++col_chunk_i) {
        if (col_chunk_i < GemmK_NumOneChunk_within_Cta) {
          // Compress is handled in unit of ElementAMmaRawUnit
          cute::Tensor tAsA   = cAsA_log_chunk(row_i, make_coord(_, col_chunk_i));
          cute::Tensor tACsAC = cACsAC_log_chunk(row_i, make_coord(_, col_chunk_i));
          cute::Tensor tEsE   = cEsE_log_chunk(row_i, make_coord(_, col_chunk_i));

          int non_zero_cnt = 0;
          // None zero element indx
          // e.g.
          //  2:4 sparsity [x 0 0 x]
          //  non_zero_elt_log_idx = [0, 3]
          int non_zero_elt_log_idx[OneChunkSizeAC{}] = { 0 };

          // * Find None Zero Element Idx within Chunk
          CUTE_UNROLL
          for (int elt_log_idx = 0; elt_log_idx < OneChunkSizeA{}; ++elt_log_idx) {
            ElementAMmaRawUnit elem_A = tAsA[elt_log_idx];
            
            // Handle negative 0
            ElementAMmaRawUnit masked_elem_A = elem_A;
            if constexpr (has_negative_zero_v<ElementA>) {
              masked_elem_A = elem_A & sign_mask;
            }

            if (masked_elem_A != ElementAMmaRawUnit{0}) {
              non_zero_elt_log_idx[non_zero_cnt] = elt_log_idx;
              tACsAC[non_zero_cnt] = elem_A;
              non_zero_cnt++;
            }
          }

          // * Corner Case for 2:4 sparsity
          if constexpr (cute::sizeof_bits_v<ElementAMmaRawUnit> < 32) {
            // i.e. [0 0 0 x] -> [(0) 0 0 x]
            if (non_zero_cnt == 1 && non_zero_elt_log_idx[0] == 3) {
              tACsAC[1] = tACsAC[0];
              tACsAC[0] = ElementAMmaRawUnit{0};
              non_zero_elt_log_idx[0] = 0;
              non_zero_elt_log_idx[1] = 3;
            }
            // i.e. [0 0 x 0] -> [0 0 x (0)]
            // i.e. [0 x 0 0] -> [0 x 0 (0)]
            // i.e. [x 0 0 0] -> [x 0 0 (0)]
            else if (non_zero_cnt == 1) {
              tACsAC[1] = ElementAMmaRawUnit{0};
              non_zero_elt_log_idx[1] = 3;
            }
          }

          // * Set Metadata Bits
          MetadataOneChunk metadata_one_chunk;
          CUTE_UNROLL
          for (int elt_phy_idx = 0; elt_phy_idx < OneChunkSizeAC{}; elt_phy_idx++) {
            metadata_one_chunk.set_metadata_bits(non_zero_elt_log_idx[elt_phy_idx], elt_phy_idx);
          }
          tEsE[0] = metadata_one_chunk.storage();

        }
        else {
          break;
        }
      }
    }

    // * Sync after Compress
    __syncthreads();

    // * Output Cta Tensor S to G
    if (GemmM_within_Cta > 0 && GemmK_within_Cta > 0) {
      constexpr int MaxVecBits = 128; // STG.128
      cute::cooperative_copy<MaxThreadsPerBlock, MaxVecBits>(threadIdx_X, cEsE, cEgE);
    }

    if (GemmMAlignedAC_within_Cta == TensorEAtomM{} && GemmKAlignedAC_within_Cta == TensorEAtomK{}) {
      copy_vec_pred<false, LayoutATag>(cACsAC, cACgAC, threadIdx_X, GemmMAlignedAC_within_Cta, (GemmKAlignedAC_within_Cta / ElementASparsity::value));
    }
    else {
      copy_vec_pred<true, LayoutATag>(cACsAC, cACgAC, threadIdx_X, GemmMAlignedAC_within_Cta, (GemmKAlignedAC_within_Cta / ElementASparsity::value));
    }

  } // end of structure_sparse_compress()

  template<uint32_t NumThreads,
           typename TensorSrc>
  CUTE_DEVICE
  static void
  cooperative_clear(
    uint32_t const& tid,
    TensorSrc dSrc) {
    
    auto dSrctSrc = local_partition(dSrc, make_layout(make_shape(NumThreads, _1{})), tid);
    cute::clear(dSrctSrc);

    // Sync all thread data access
    __syncthreads();
  }

  template <bool pred,
            typename LayoutTag,
            typename TensorSrc,
            typename TensorDst>
  CUTE_DEVICE
  static void
  copy_vec_pred(
      TensorSrc dSrc,
      TensorDst dDst,
      int threadIdx_X,
      int valid_rows,
      int valid_cols) {

    constexpr bool IsRowMajor = cute::is_same_v<LayoutTag, cutlass::layout::RowMajor>;
    using Element = typename TensorSrc::element_type;
    constexpr bool IsQmmaF6 = cute::sizeof_bits_v<Element> == 6;

    CUTE_STATIC_ASSERT(cute::is_static_v<decltype(shape(dSrc))>, "shape(dSrc) needs to be static");
    CUTE_STATIC_ASSERT(cute::is_static_v<decltype(shape(dDst))>, "shape(dDst) needs to be static");
    CUTE_STATIC_ASSERT(cute::sizeof_bits_v<typename TensorSrc::element_type> == cute::sizeof_bits_v<typename TensorDst::element_type>,
      "dSrc and dDst need to have same element bit width");
    CUTE_STATIC_ASSERT(cute::size(dSrc) == cute::size(dDst), "dSrc and dDst need to have same size");

    // ValueShape
    using ValueShape = 
      cute::conditional_t<IsQmmaF6,
                          Shape<Int<1>, Int<1>>,
      cute::conditional_t<IsRowMajor,
                          Shape<Int<1>, Int<128 / sizeof_bits_v<Element>>>,
                          Shape<Int<128 / sizeof_bits_v<Element>>, Int<1>>>
      >;

    constexpr int ValueShapeRows = shape<0>(ValueShape{});
    constexpr int ValueShapeCols = shape<1>(ValueShape{});

    // ThreadShape
    using ThreadShape = 
      cute::conditional_t<IsQmmaF6,
                          cute::conditional_t<IsRowMajor,
                                              Shape<Int<MaxThreadsPerBlock>, Int<1>>,
                                              Shape<Int<1>, Int<MaxThreadsPerBlock>>>,
      cute::conditional_t<IsRowMajor,
                          Shape<Int<MaxThreadsPerBlock / (shape<1>(dSrc) / ValueShapeCols)>, Int<                     (shape<1>(dSrc) / ValueShapeCols)>>,
                          Shape<Int<                     (shape<0>(dSrc) / ValueShapeRows)>, Int<MaxThreadsPerBlock / (shape<0>(dSrc) / ValueShapeRows)>>>
      >;

    constexpr int ThreadShapeRows = shape<0>(ThreadShape{});
    constexpr int ThreadShapeCols = shape<1>(ThreadShape{});

    const int threadIdx_X_row = threadIdx_X / ThreadShapeCols;
    const int threadIdx_X_col = threadIdx_X % ThreadShapeCols;

    // Row Major
    if constexpr (IsRowMajor) {
      CUTE_UNROLL
      for (int iter_row_blk = 0; iter_row_blk < cutlass::ceil_div(shape<0>(dSrc), ThreadShapeRows * ValueShapeRows); ++iter_row_blk) {
        CUTE_UNROLL
        for (int col_chunk_i = 0; col_chunk_i < cutlass::ceil_div(shape<1>(dSrc) , ThreadShapeCols * ValueShapeCols); ++col_chunk_i) {
          CUTE_UNROLL
          for (int iter_row_thr = 0; iter_row_thr < ValueShapeRows; ++iter_row_thr) {
            CUTE_UNROLL
            for (int iter_col_thr = 0; iter_col_thr < ValueShapeCols; ++iter_col_thr) {
              const int row_i = (iter_row_blk * ThreadShapeRows + threadIdx_X_row) * ValueShapeRows + iter_row_thr;
              const int col_i = (col_chunk_i * ThreadShapeCols + threadIdx_X_col) * ValueShapeCols + iter_col_thr;
              if constexpr ( (not pred) and (not IsQmmaF6) ) {
                dDst(row_i, col_i) = dSrc(row_i, col_i);
              }
              else {
                if (row_i < valid_rows && col_i < valid_cols) {
                  dDst(row_i, col_i) = dSrc(row_i, col_i);
                }
              }
            }
          }
        }
      }
    }
    // Col Major
    else {
      CUTE_UNROLL
      for (int col_chunk_i = 0; col_chunk_i < cutlass::ceil_div(shape<1>(dSrc) , ThreadShapeCols * ValueShapeCols); ++col_chunk_i) {
        CUTE_UNROLL
        for (int iter_row_blk = 0; iter_row_blk < cutlass::ceil_div(shape<0>(dSrc), ThreadShapeRows * ValueShapeRows); ++iter_row_blk) {
          CUTE_UNROLL
          for (int iter_col_thr = 0; iter_col_thr < ValueShapeCols; ++iter_col_thr) {
            CUTE_UNROLL
            for (int iter_row_thr = 0; iter_row_thr < ValueShapeRows; ++iter_row_thr) {
              const int row_i = (iter_row_blk * ThreadShapeRows + threadIdx_X_row) * ValueShapeRows + iter_row_thr;
              const int col_i = (col_chunk_i * ThreadShapeCols + threadIdx_X_col) * ValueShapeCols + iter_col_thr;
              if constexpr ( (not pred) and (not IsQmmaF6) ) {
                dDst(row_i, col_i) = dSrc(row_i, col_i);
              }
              else {
                if (row_i < valid_rows && col_i < valid_cols) {
                  dDst(row_i, col_i) = dSrc(row_i, col_i);
                }
              }
            }
          }
        }
      }
    }
  
    // Sync all thread data access
    __syncthreads();
  } // end of copy_vec_pred()
  
};

}  // namespace cutlass::transform::kernel
