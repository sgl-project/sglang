/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief 
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/epilogue/warp/tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// This is an optimization available on CUDA 11.2 and beyond that eliminates branches in the epilogue.
#define CUTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED ((__CUDACC_VER_MAJOR__ * 10 + __CUDACC_VER_MINOR__) >= 112)

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory. This is optimized
/// for mixed-precision epilogues in which the accumulators are 32b in width, but the output
/// data type is smaller. 
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,              ///< data type of accumulator element
  int ElementSizeBits,            ///< Size of accumulator element in bits
  int OutputSizeBits,             ///< Size of output element in bits
  int OutputElementCount,         ///< number of elements in output vector
  int ContiguousLanes,            ///< Number of consecutive lanes writing to contiguous memory
  bool EightBitsOutputOrLess = (OutputSizeBits <= 8)
>
class TileIteratorTensorOpMixed {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = OutputElementCount;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 
      (OutputElementCount * sizeof_bits<Element>::value) / (const_min(128, OutputElementCount * sizeof_bits<Element>::value));

    // Currently support max 4 ptr
    static constexpr int kMaxPointerCount{4};

    static_assert(kPointerCount <= kMaxPointerCount, "Can only accommodate four pointers at present.");
    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerAccess>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount] = {nullptr};

  /// Stride in units of AccessType
  int stride_{0};

  /// Logical column in which warp tile is aligned
  int warp_column_{0};

public:

  /// Default constructor
  TileIteratorTensorOpMixed() = default;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / Policy::kElementsPerAccess),
    warp_column_(0) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = (lane_in_quad % 2) + (((lane_in_quad / 2) + i) % Detail::kPointerCount) * 2;

      ptr += column_idx;

      pointers_[i % Detail::kPointerCount] = ptr;
    }
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / Policy::kElementsPerAccess;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += tile_offset.row() * Shape::kRow * stride_ + 
        tile_offset.column() * Shape::kColumn / Policy::kElementsPerAccess;
    }

    warp_column_ += tile_offset.column() * Shape::kColumn;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    AccessType *ptr = pointers_[0];

#if CUTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

    // When the optimization is enabled, small tiles require separate logic.
    bool kN32_optimization = (WarpShape::kN * Detail::kLanesInQuad * Policy::kElementsPerAccess * sizeof_bits<Element>::value) % 1024 == 0;
    if (kN32_optimization) {
      
      int ptr_idx = ((warp_column_ * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;
      
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      } else if (ptr_idx == 1) {
	if constexpr (AccessType::kElements >= 2) {
          ptr = pointers_[1];
	}
      } else if (ptr_idx == 2) {
	if constexpr (AccessType::kElements >= 3) {
          ptr = pointers_[2];
	}
      } else if (ptr_idx == 3) {
	if constexpr (AccessType::kElements >= 4) {
          ptr = pointers_[3];
	}
      }
    }

#endif

    CUTLASS_PRAGMA_UNROLL
    for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      
#if CUTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

      //
      // When the optimization is enabled, this expression suffices to obtain the SMEM pointer.
      //
      if (WarpShape::kN == 64) {
        ptr = pointers_[n / 4];
      }
      else if (!kN32_optimization)
#endif
      {
        // This is the reference implementation
        int column_idx = warp_column_ + n * Detail::kLanesInQuad * Policy::kElementsPerAccess;
        int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;
  
        if (ptr_idx == 0) {
          ptr = pointers_[0 % Detail::kPointerCount];
        }
        else if (ptr_idx == 1) {
          ptr = pointers_[1 % Detail::kPointerCount];
        }
        else if (ptr_idx == 2) {
          ptr = pointers_[2 % Detail::kPointerCount];
        }
        else if (ptr_idx == 3) {
          ptr = pointers_[3 % Detail::kPointerCount];
        }
      }

      int offset = n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess;
      ptr[offset] = frag_ptr[n];
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int64_t n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int column_idx = warp_column_ + n * Detail::kLanesInQuad * Policy::kElementsPerAccess;
      int ptr_idx = ((column_idx * sizeof_bits<Element>::value) / 1024) % Detail::kPointerCount;

      AccessType const *smem_ptr = pointers_[ptr_idx];
      frag_ptr[n] = smem_ptr[n * Detail::kLanesInQuad + pointer_offset / Policy::kElementsPerAccess];
    }
  }

  /// Load
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }
  
  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int32_t x 16 => int8_t/int4b_t x 16
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape),
  int OutputSizeBits              ///< Size of output element in bits
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, OutputSizeBits, 16, 8, true> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = int32_t;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 16;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    /// Offsets added 
    static int const kOffsetCount = 4;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount] = {nullptr};

  /// Stride in units of AccessType
  int stride_{0};

  /// Uniform offset in bytes added to warp tile iterator
  int uniform_offset_[Detail::kOffsetCount] = {0};

public:

  /// Default constructor
  TileIteratorTensorOpMixed() = default;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;
    
      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ 0) * 4 * sizeof(AccessType);
    }
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ + 
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ tile_offset.column()) * 4 * sizeof(AccessType);
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);
      int offset_idx = (n % 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
      uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType) + uniform_offset_[offset_idx];

      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for int32_t x 8 => int8_t/int4b_t x 8
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  int OutputSizeBits              ///< Size of output element in bits
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, int32_t, 32, OutputSizeBits, 8, 8, true> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = int32_t;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 8;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount] = {nullptr};

  /// Stride in units of AccessType
  int stride_{0};

public:

  /// Default constructor
  TileIteratorTensorOpMixed() = default;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) { 

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;
    
      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ + 
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;
   
    if (tile_offset.column() % 2) {
      auto tmp = pointers_[0];
      pointers_[0] = pointers_[1];
      pointers_[1] = tmp;
    }
 
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements + (n % 4) * 4;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
      uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType);

      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Set smem base address
  CUTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float x 16 => float_e4m3_t/float_e5m2_t x 16
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_         ///< matrix multiply operation shape (concept: gemm::GemmShape),
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, float, 32, 8, 16, 8> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = float;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 16;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element,
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    /// Offsets added
    static int const kOffsetCount = 4;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount] = {nullptr};

  /// Stride in units of AccessType
  int stride_{0};

  /// Uniform offset in bytes added to warp tile iterator
  int uniform_offset_[Detail::kOffsetCount] = {0};

public:

  /// Default constructor
  TileIteratorTensorOpMixed() = default;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;

      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ 0) * 4 * sizeof(AccessType);
    }
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {

    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ +
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kOffsetCount; ++i) {
      uniform_offset_[i] = (i ^ tile_offset.column()) * 4 * sizeof(AccessType);
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);
      int offset_idx = (n % 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
      uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType) + uniform_offset_[offset_idx];

      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float x 8 => float_e4m3_t/float_e5m2_t x 8
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_         ///< matrix multiply operation shape (concept: gemm::GemmShape)
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, float, 32, 8, 8, 8> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = float;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = 8;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element,
    Policy::OperatorCount::kColumn * Policy::kElementsPerAccess>;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;

    /// Number of pointers needed to write accumulators
    static int const kPointerCount = 2;

    static_assert(sizeof(Element) == 4, "This can only be used with 32b accumulator data types (f32, s32).");
  };

  /// Padding quantity
  using Padding = MatrixShape<0, Detail::kLanesInQuad * 2>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, 2>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointers_[Detail::kPointerCount] = {nullptr};

  /// Stride in units of AccessType
  int stride_{0};

public:

  /// Default constructor
  TileIteratorTensorOpMixed() = default;

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ):
    stride_(ref.stride()[0] / AccessType::kElements) {

    int quad_id = (lane_id / Detail::kLanesInQuad);
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      AccessType *ptr = reinterpret_cast<AccessType *>(ref.data()) + quad_id * stride_;
      int column_idx = lane_in_quad ^ (i * 2);

      ptr += column_idx;

      if (i == 0) {
        pointers_[0] = ptr;
      }
      else if (i == 1) {
        pointers_[1] = ptr;
      }
    }
  }

  /// Adds a pointer offset
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    CUTLASS_PRAGMA_UNROLL
    for (int64_t i = 0; i < Detail::kPointerCount; ++i) {
      pointers_[i] += pointer_offset / AccessType::kElements;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {

    int ptr_offset = tile_offset.row() * Shape::kRow * stride_ +
      tile_offset.column() * Shape::kColumn / AccessType::kElements;

    pointers_[0] += ptr_offset;
    pointers_[1] += ptr_offset;

    if (tile_offset.column() % 2) {
      auto tmp = pointers_[0];
      pointers_[0] = pointers_[1];
      pointers_[1] = tmp;
    }

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {

    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {

      int ptr_idx = (n / 4);

      AccessType *ptr;
      if (ptr_idx == 0) {
        ptr = pointers_[0];
      }
      else if (ptr_idx == 1) {
        ptr = pointers_[1];
      }

      int offset = (n / 4) * 16 + pointer_offset / AccessType::kElements + (n % 4) * 4;

#if 0
      //
      // Using inline PTX to avoid generic memory
      //
      AccessType *smem_ptr = pointers_[ptr_idx];
      smem_ptr[offset] = frag_ptr[n];
#else
      uint32_t smem_addr = arch::cutlass_get_smem_pointer(ptr);
      uint32_t const *data = reinterpret_cast<uint32_t const *>(frag_ptr + n);
      uint32_t offset_in_bytes = offset * sizeof(AccessType);

      asm volatile(
        "{ .reg .u32 smem_ptr; add.u32 smem_ptr, %0, %1; st.shared.v2.u32 [smem_ptr], {%2, %3}; }\n"
        : : "r"(smem_addr), "r"(offset_in_bytes), "r"(data[0]), "r"(data[1])
      );
#endif
    }
  }

  /// Store
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#undef CUTLASS_EPILOGUE_WARP_TILE_ITERATOR_TENSOR_OP_MIXED_OPTIMIZATION_ENABLED

/////////////////////////////////////////////////////////////////////////////////////////////////
