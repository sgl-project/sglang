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

#include <algorithm>                       // std::fill
#include <array>                           // std::array
#include <cstdio>
#include <random>                          // std::mt19937

#include "cute/container/bit_field.hpp"    // cute::bit_field
#include "cute/numeric/numeric_types.hpp"  // cute::sizeof_bits_v
#include "cute/tensor.hpp"                 // cute::Tensor, cute::make_tensor, cute::print_tensor
#include "cutlass/arch/arch.h"             // cutlass::arch::Sm90
#include "cutlass/cutlass.h"               // cutlass::Status
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/layout.hpp"       // cutlass::TagToStrideA_t
#include "cutlass/fast_math.h"             // cutlass::ceil_div, cutlass::round_up
#include "cutlass/kernel_hardware_info.h"  // cutlass::KernelHardwareInfo
#include "cutlass/util/packed_stride.hpp"  // cutlass::make_cute_packed_stride
#include "cutlass/numeric_size.h"          // cutlass::bits_to_bytes
#include "cutlass/cuda_host_adapter.hpp"   // cutlass::CudaHostAdapter

namespace cutlass
{
namespace transform
{
namespace kernel
{

using namespace cute;

namespace detail {

  template<typename T>
  CUTLASS_HOST_DEVICE
  static uint8_t
  encode_in_chunk_idx_legacy(int in_chunk_idx){
    if (sizeof(T) == 4) {
      return in_chunk_idx == 0 ? 0b0100 : 0b1110;
    }
    else {
      uint8_t res = 0;
      if (in_chunk_idx == 0) {
        res = 0b00;
      }
      else if (in_chunk_idx == 1) {
        res = 0b01;
      }
      else if (in_chunk_idx == 2) {
        res = 0b10;
      }
      else {
        res = 0b11;
      }
      return res;
    }
  }

  template <
    class SparseConfig,
    class EngineA,
    class LayoutA,
    class EngineAc,
    class LayoutAc
  >
  CUTLASS_HOST_DEVICE
  static void
  compress_two_chunks_legacy(
    Tensor<EngineA, LayoutA> tensorA,
    Tensor<EngineAc, LayoutAc> tensorAc,
    uint8_t& meta_two_chunk,
    int effective_elems) {

    using ElementA = typename EngineAc::value_type;

    static constexpr int LogicalElemsAPerChunk  = typename SparseConfig::LogicalElemsAPerChunk{};
    static constexpr int PhysicalElemsAPerChunk  = typename SparseConfig::PhysicalElemsAPerChunk{};
    static constexpr int ElemsARawPerElementAMmaRaw    = typename SparseConfig::ElemsARawPerElementAMmaRaw{};
    static constexpr int ElementEBitsPerElementAMma = typename SparseConfig::ElementEBitsPerElementAMma{};
    static constexpr int LogicalSubChunk     = ceil_div(LogicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);
    static constexpr int PhysicalSubChunk    = ceil_div(PhysicalElemsAPerChunk, ElemsARawPerElementAMmaRaw);

    /*
    Legal metadata chunk in SM90
    Index   Bin   HEX
    0, 1  0b0100   4
    1, 2  0b1001   9
    2, 3  0b1110   E
    0, 2  0b1000   8
    1, 3  0b1101   D
    0, 3  0b1100   C
    2, 1  0b0110   6  (Not used)
    -----------------------------------
    TF32
    0     0b0100   4
    1     0b1110   E
    */

    if (effective_elems <= 0) {
      return;
    }

    // initialize
    // 0 is the initial value for this function while 0x44 is the initial value for hardware.
    meta_two_chunk = 0;

    for (int chunk_idx = 0; chunk_idx < 2; ++chunk_idx) {
      // If Only One Chunk within this Two Chunk
      if ( effective_elems <= chunk_idx * ElemsARawPerElementAMmaRaw * LogicalSubChunk ) {
        break;
      }
      /// init result;
      int non_zero_cnt = 0;
      int32_t nnz_chunk_idx[PhysicalSubChunk] = { 0 };
      ElementA Ac_chunk[PhysicalSubChunk][ElemsARawPerElementAMmaRaw] = { ElementA{0} };

      for (int subchunk_idx = 0; subchunk_idx < LogicalSubChunk; ++subchunk_idx) {
        bool is_nz = true;
        ElementA subchunk_elems[ElemsARawPerElementAMmaRaw] = { ElementA{0} };
        /// Check if subchunk is non-zero
        for(int elem_idx = 0; elem_idx < ElemsARawPerElementAMmaRaw; elem_idx++) {
          int offset = chunk_idx * LogicalElemsAPerChunk + subchunk_idx * ElemsARawPerElementAMmaRaw + elem_idx;
          subchunk_elems[elem_idx] = offset < effective_elems ? tensorA(offset) : ElementA(0);
          
          ElementA zero = static_cast<ElementA>(0);
          ElementA minus_zero = static_cast<ElementA>(ElementA(1) << cutlass::sizeof_bits_v<ElementA> - 1);
          if (subchunk_elems[elem_idx] != zero && subchunk_elems[elem_idx] != minus_zero) {
            if (non_zero_cnt >= PhysicalSubChunk) {
              #ifdef  __CUDA_ARCH__
                asm volatile ("brkpt;\n" ::);
              #else
                throw std::runtime_error("Found extra non-zero elements in a chunk!\n");
              #endif
            }
            is_nz = false;
          }
        }

        /// There is non-zero element in the subchunk
        if(!is_nz) {
          nnz_chunk_idx[non_zero_cnt] = subchunk_idx;
          memcpy(Ac_chunk[non_zero_cnt], subchunk_elems, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
          non_zero_cnt++;
        }
      }

      /*
      Special cases
      nnz == 1 and non-tf32 and nnz_idx = 3
      */
      ElementA elementA_zeros[ElemsARawPerElementAMmaRaw] = { ElementA{0} };
      if constexpr (sizeof_bits_v<ElementA> < 32) {
        if (non_zero_cnt == 1 && nnz_chunk_idx[0] == 3) {
          memcpy(Ac_chunk[1], Ac_chunk[0], sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
          memcpy(Ac_chunk[0], elementA_zeros, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
          nnz_chunk_idx[1] = 3;
          nnz_chunk_idx[0] = 0;
        }
        else if (non_zero_cnt == 1) {
          memcpy(Ac_chunk[1], elementA_zeros, sizeof(ElementA) * ElemsARawPerElementAMmaRaw);
          nnz_chunk_idx[1] = 3;
        }
      }

      /// Setup metadata
      uint8_t meta_chunk = 0;
      for (int i = 0; i < PhysicalSubChunk; i++) {
        meta_chunk = static_cast<uint8_t>(meta_chunk | (encode_in_chunk_idx_legacy<ElementA>(nnz_chunk_idx[i]) << (i * ElementEBitsPerElementAMma)));
        for(int j = 0; j < ElemsARawPerElementAMmaRaw; j++) {
          tensorAc(chunk_idx * PhysicalElemsAPerChunk + i * ElemsARawPerElementAMmaRaw + j) = Ac_chunk[i][j];
        }
      }
      meta_two_chunk = uint8_t(meta_two_chunk | (meta_chunk << (chunk_idx * _4{})));
    }
  }
}

template<
  class ProblemShape_,
  class ElementA_,
  class LayoutATag_,
  class SparseConfig_
>
class SM90StructuredSparseCompressorLegacy {
public:
  using SparseConfig = SparseConfig_;
  using ProblemShape = ProblemShape_;

  // * EltA
  using ElementA = ElementA_;
  using ElementAUint = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
  static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();
  using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA,
                                            cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>,
                                            ElementA>;
  using ElementAMma = typename SparseConfig::ElementAMma;
  using ElementAMmaRaw = typename SparseConfig::ElementAMmaRaw;
  using ElementASparsity = typename SparseConfig::ElementASparsity;
  using ElementAMmaSparsity = typename SparseConfig::ElementAMmaSparsity;
  using LayoutATag = LayoutATag_;
  using LayoutA = LayoutATag;
  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutATag>;

  // * EltE
  using ElementEMma = typename SparseConfig::ElementEMma;
  using ElementEMmaRaw = typename SparseConfig::ElementEMmaRaw;
  using ElementEMmaSparsity = typename SparseConfig::ElementEMmaSparsity;

  // * AtomE
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
  static constexpr int MaxThreadsPerBlock = 1;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  using ArchTag = arch::Sm90;

  struct SharedStorage {
    /* empty, no smem needed */
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct TransformArguments {
    ArrayElementA const* ptr_A{nullptr};
    StrideA dA{};
    ArrayElementA* ptr_ACompress{nullptr};
    ElementEMmaRaw* ptr_E{nullptr};
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

  static Params
  to_underlying_arguments(Arguments & args, void* workspace) {
    return Params{{args.problem_shape},
                  {args.transform.ptr_A, args.transform.dA, args.transform.ptr_ACompress, args.transform.ptr_E},
                  {args.hw_info},
                  workspace};
  }

  static Status
  can_implement(Arguments const& args) {
    auto [M, N, K, L] = args.problem_shape;
    if (K % LogicalElemsAPerChunk != 0) {
      CUTLASS_TRACE_HOST("SM90 Sparse Compressor CAN NOT IMPLEMENT: GemmK not multiplier of logical chunk size\n");
      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    auto problem = args.problem_shape;
    const int m = cute::size<0>(problem);
    const int k = cute::size<2>(problem);
    const int l = cute::size<3>(problem);
    const int metadata_k = round_up(k, TensorEAlignmentK);
    const int metadata_m = round_up(m, TensorEAlignmentM);
    const int metadata_bytes = metadata_m * metadata_k / ElementEMmaSparsity{} * l;
    return metadata_bytes;
  }

  static Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
    cudaError_t cuda_error;

    auto workspace_size = get_workspace_size(args);
    if (workspace_size == 0) {
      return Status::kSuccess;
    } else if (workspace == nullptr) {
      return Status::kErrorInternal;
    }

    cudaPointerAttributes attri;
    cuda_error = cudaPointerGetAttributes(&attri, workspace);
    if (cuda_error != cudaSuccess) {
      return Status::kErrorInternal;
    }

    if ( attri.type == cudaMemoryTypeDevice ) {
#if defined(CUTLASS_ENABLE_CUDA_HOST_ADAPTER) && CUTLASS_ENABLE_CUDA_HOST_ADAPTER
      CUTLASS_ASSERT(cuda_adapter);
      if (Status::kSuccess != cuda_adapter->memsetDevice(workspace, static_cast<uint8_t>(0), workspace_size, stream)) {
        return Status::kErrorInternal;
      }
#else
      cudaMemsetAsync(workspace, 0, workspace_size, stream);
      cuda_error = cudaGetLastError();
      if (cuda_error != cudaSuccess) {
        return Status::kErrorInternal;
      }
#endif
    } else {
      memset(workspace, 0, workspace_size);
    }

    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    return dim3(1, 1, 1);
  }

  static dim3
  get_block_shape() {
    return dim3(1, 1, 1);
  }

  CUTE_HOST_DEVICE
  void
  operator()(Params params, char* smem_buf = nullptr) {
    run(params, smem_buf);
  }

  CUTE_HOST_DEVICE
  static void
  run(Params params, char* smem_buf = nullptr) {
    do_compress_device_host(params);
  }

private:

  CUTE_HOST_DEVICE
  static void
  do_compress_device_host(Params params) {
    auto [m, n, k, l] = params.problem_shape;
    auto [ptr_A, dA, ptr_ACompress, ptr_E] = params.transform;
    auto workspace = params.workspace;

    const int aligned_k = (k + TensorAAlignmentK - 1) / TensorAAlignmentK * TensorAAlignmentK;
    const int aligned_m = (m + TensorAAlignmentM - 1) / TensorAAlignmentM * TensorAAlignmentM;
    const int metadata_k = (k + TensorEAlignmentK - 1) / TensorEAlignmentK * TensorEAlignmentK;
    const int metadata_m = (m + TensorEAlignmentM - 1) / TensorEAlignmentM * TensorEAlignmentM;
    const int k_compressed = aligned_k / ElementASparsity{};

    // Convert to CuTe tensors. But don't want to use sparse_ptr, which is making everything complicated here.
    cute::Tensor tensorA = make_tensor(recast_ptr<ElementAUint>(ptr_A), make_layout(make_shape(m, k, l), dA));

    cute::Tensor tensorAc = make_tensor(recast_ptr<ElementAUint>(ptr_ACompress),
                      make_shape(aligned_m, k_compressed, l),
                      make_cute_packed_stride(StrideA{}, cute::make_shape(aligned_m, k_compressed, l)));

    cute::Tensor tensorE_raw_compress_logical = make_tensor(recast_ptr<sparse_elem<ElementEMmaSparsity{},ElementEMmaRaw>>(workspace),
                                make_shape(metadata_m, make_shape(TensorEAtomK{}, metadata_k / TensorEAtomK{}), l),
                                make_stride(TensorEAtomK{}, make_stride(_1{}, metadata_m*TensorEAtomK{}), metadata_m*metadata_k));

    cute::Tensor tensorE_raw_compress = recast<uint8_t>(tensorE_raw_compress_logical);

    // The following vars are all logical.
    int atom_m = size<0>(TensorEAtom{});
    int atom_k = size<1>(TensorEAtom{});
    int tiled_m = metadata_m / atom_m;
    int tiled_ke = metadata_k / atom_k;
    // Col major when viewing atoms
    int stride_tile_m = cosize(TensorEAtom{});
    int stride_tile_ke = atom_k * metadata_m;

    // Logical metadata tensor
    cute::Tensor tensorE_logical = make_tensor(recast_ptr<sparse_elem<ElementEMmaSparsity{},ElementEMmaRaw>>(ptr_E),
                           make_layout(make_shape(append(shape<0>(TensorEAtom{}), tiled_m),
                                       append(shape<1>(TensorEAtom{}), tiled_ke),
                                       shape<2>(tensorE_raw_compress_logical)),
                                 make_stride(append(stride<0>(TensorEAtom{}), stride_tile_m),
                                       append(stride<1>(TensorEAtom{}), stride_tile_ke),
                                       stride<2>(tensorE_raw_compress_logical))));
    // Physical metadata tensor
    cute::Tensor tensorE = recast<uint8_t>(tensorE_logical);

    // void do_init()
    cute::clear(tensorAc);
    cute::clear(tensorE_raw_compress);

    // void do_raw_compress()
    using TileStepA = Int<LogicalElemsAPerChunk * 2>;
    using TileStepAc = Int<TileStepA{} / 2>;

    cute::Tensor tensorATiled = logical_divide(tensorA, make_shape(_, TileStepA{}, _));
    cute::Tensor tensorAcTiled = logical_divide(tensorAc, make_shape(_, TileStepAc{}, _));

    for (int batch_idx = 0; batch_idx < l; batch_idx++) {
      for (int m_idx = 0; m_idx < m; m_idx++) {
        for (int tiler_k_idx = 0; tiler_k_idx < size<1,1>(tensorATiled); tiler_k_idx++) {
          int effective_elems = cute::min(TileStepA{}, k - (tiler_k_idx * TileStepA{}));
          detail::compress_two_chunks_legacy<SparseConfig>(tensorATiled(m_idx, make_coord(_, tiler_k_idx), batch_idx),
                                                     tensorAcTiled(m_idx, make_coord(_, tiler_k_idx), batch_idx),
                                                     tensorE_raw_compress(m_idx, tiler_k_idx, batch_idx),
                                                     effective_elems);
        }
      }
    }

    // void do_reorder()
    // Fast path when we don't permute.
    if constexpr (sizeof_bits_v<ElementAUint> <= 8) {
      memcpy(tensorE.data(), tensorE_raw_compress.data(), tensorE.size());
    }
    else {
      cute::copy(tensorE_raw_compress, tensorE);
    }

    #if 0
    print("--> TensorA\n");
    auto tensorA_eltA = cute::recast<ElementA>(tensorA);
    cute::print_tensor(tensorA_eltA); printf("\n\n");

    print("--> REF TensorAC\n");
    auto tensorAc_eltA = cute::recast<ElementA>(tensorAc);
    cute::print_tensor(tensorAc_eltA); printf("\n\n");

    print("--> REF TensorE\n");
    cute::print_tensor(tensorE); printf("\n\n");
    #endif

  }
};

}  // namespace kernel
}  // namespace transform
}  // namespace cutlass
