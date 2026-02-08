![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Layouts and Tensors")

Note: This document talks about CUTLASS 2.x layout tag types.
CUTLASS 3.0 deprecates all legacy 2.x layout tags in favour of a single `cute::Layout<Shape, Stride>`
vocabulary type for all thread and data tensors. Please refer to the
[documentation for cute layouts](cute/01_layout.md) for more details about CUTLASS 3.0's definition of "layout".

# Layouts and Tensors

_Tensors_ are mathematical objects represented by a multidimensional array of numeric elements in memory.
These may define two dimensional matrices upon which classical linear algebra computations may be defined or
higher dimensional objects frequently used to structure data used by Deep Learning applications and frameworks.

This document describes design patterns used in CUTLASS to map logical index spaces onto memory (Layouts) and to
indirectly reference tensors in memory (TensorRef and TensorView objects).

As described, CUTLASS adheres to the following terminology which is consistent with the C++ Standard Library.

* *size* (scalar): number of elements in a tensor
* *capacity* (scalar): number of elements needed to represent tensor in memory (may be larger than _size_)
* *rank* (scalar): number of logical dimensions describing tensor
* *extent* (vector): size of each logical dimension in a tensor

## CUTLASS Layout Concept

CUTLASS Layouts are a systematic design pattern for the following:
* Mapping _logical_ index space to _physical_ offsets in memory
* Storing the dynamic state needed in the above computation
* Defining a type system for partial specialization of other CUTLASS components

_Concept:_ layouts satisfy the following concept.
```c++
/// CUTLASS Layout concept example
struct LayoutConcept {

  /// Logical rank of tensor
  static int const kRank;

  /// Rank of stride vector
  static int const kStrideRank;

  /// Index type used for coordinates
  struct Index;

  /// Long index type used for offsets
  struct LongIndex;

  /// Logical coordinate - satisfies Coord<kRank, ..>
  struct TensorCoord;

  /// Stride object - satisfies Coord<kStrideRank, ..>
  struct Stride

  //
  // Methods
  //

  /// Constructor
  CUTLASS_HOST_DEVICE
  LayoutConcept();

  /// Ctor
  CUTLASS_HOST_DEVICE
  LayoutConcept(Stride stride);

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static LayoutConcept packed(TensorCoord const &extent);

  /// Function call operator returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (row, column)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const;

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const;

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const;

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride();

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const;
};
```

_Layout_ objects generalize leading dimensions of matrices typical in _BLAS_ implementations. For example, cuBLAS assumes
Fortran-style _column-major_ layouts of matrices and refers to this as the matrix's "leading dimension."

```c++
cublasGemmEx(
  ...
  ptr_A,      // pointer to first element of matrix A
  lda,        // leading dimension
  ...
);
```
This implies an element at coordinate (_row_, _column_) has offset `row + lda * column`.

This is equivalently represented by CUTLASS's `layout::ColumnMajor` type as follows.
```c++

layout::ColumnMajor layout(lda); 

int offset = layout({row, column});     // returns row  + lda * column
```

Other layout functions are possible such as row-major:
```c++

layout::RowMajor layout(lda); 

int offset = layout({row, column});     // returns lda * row + column
```

In both cases, the _logical_ coordinate (_row_, _column_) is represented by the same object. This enables an algorithm to be
implemented as generic template, with locations within tensors always specified in logical space. _Layout_ objects map this to 
physical offsets in memory.

The layout's `::packed()` static method may be used to construct a layout object given the extent of a densely packed tensor.
This method is needed when an algorithm must define a buffer of arbitrary layout.

Example:
```c++

typename ArbitraryLayout::TensorCoord extent = make_Coord(...);
typename ArbitraryLayout::TensorCoord coord;

ArbitraryLayout layout = ArbitraryLayout::packed(extent);

int offset = layout({coord});
```

The layout's `::capacity()` method computes the number of locations in memory needed to represent a tensor. This is
useful when allocating memory, as more storage may be needed than what is strictly necessary for a fully packed
tensor.

Example:
```c++

int lda = columns + padding;
MatrixCoord extent{rows, columns};

layout::RowMajor layout(lda);

auto capacity = layout.capacity(extent);    // returns rows * (columns + padding) 
```

## Accessing elements within a tensor

### TensorRef

`TensorRef<class T, class Layout>` is a structure containing both a pointer to the start of a 
tensor and a layout object to access its elements. This is a convenient object which may be
passed to functions to limit an explosion of arguments when the number of stride elements is
numerous. 

Example:
```c++
int4_t *ptr = ...;
int ldm = ...;

int row = ...;
int column = ...;

layout::ColumnMajor layout(ldm);
TensorRef<int4_t, layout::ColumnMajor> ref(ptr, layout);

int4_t x = ref.at({row, column});     // loads a 4-bit signed integer from the tensor

ref.at({row, column}) = x * 2_s4;     // transforms this quantity and stores it back
```

### TensorView

Matrices and tensors used in linear algebra computations are invariably finite. `TensorView<class T, class Layout>` extends `TensorRef<>` by
adding an `extent` vector to describe the logical extent of the tensor or matrix.

Example:
```c++
int4_t *ptr = ...;
int ldm = ...;
MatrixCoord extent = ...;

int row = ...;
int column = ...;

layout::ColumnMajor layout(ldm);
TensorView<int4_t, layout::ColumnMajor> view(ptr, layout, extent);

MatrixCoord coord = {row, column};

if (view.contains(coord)) {     // verify coordinate is in bounds before performing access
  
  int4_t x = ref.at(coord);  
  ref.at({row, column}) = x * 2_s4;
}

```

A `TensorView<>` may be constructed from a `TensorRef<>` succinctly as follows:
```c++
layout::ColumnMajor layout(ldm);
TensorRef<int4_t, layout::ColumnMajor> ref(ptr, layout);

TensorView<int4_t, layout::ColumnMajor> view(ref, extent);    // construct TensorView from TensorRef and extent
```

Note, computations avoid becoming overdetermined by accepting a single problem size component
and `TensorRef` objects for each of the operands whose extents are implied as a precondition of the operation. By avoiding
redundant storage of extent quantities, CUTLASS minimizes capacity utilization of precious resources such as constant memory.
This is consistent with BLAS conventions.

## Summary:

The design patterns described in this document form a hierarchy:
* `T *ptr;` is a pointer to a contiguous sequence of elements of type `T`
* `Layout layout;` is an object mapping an index space to a linear offset
* `TensorRef<T, Layout> ref(ptr, layout);` is an object pointing to an _unbounded_ tensor containing elements of type `T` and a layout of type `Layout`
* `TensorView<T, Layout> view(ref, extent);` is an object pointing to a _bounded_ tensor containing elements of type `T` and a layout of type `Layout`

### Appendix: Existing Layouts

This section enumerates several existing Layout types defined in CUTLASS.

Matrix layouts:
- `PitchLinear`: data layout defined by _contiguous_ and _strided_ dimensions. _contiguous_ refers to consecutive elements in memory, where as _strided_ refers to data separated by a uniform stride
-- Rank: 2
-- TensorCoord type: `PitchLinearCoord`
-- Shape type: `PitchLinearShape`
-- Stride rank: 1

- `ColumnMajor`: data layout defined by _rows_ and _columns_ dimensions. Can be mapped to `PitchLinear` by: (_contiguous_ = _rows_, _strided_ = _columns_)
-- Rank: 2
-- TensorCoord type: `MatrixCoord`
-- Shape type: `MatrixShape`
-- Stride rank: 1

- `RowMajor`: data layout defined by _rows_ and _columns_ dimensions. Can be mapped to `PitchLinear` by: (_contiguous_ = _columns_, _strided_ = _rows_)
-- Rank: 2
-- TensorCoord type: `MatrixCoord`
-- Shape type: `MatrixShape`
-- Stride rank: 1

- `ColumnMajorInterleaved<k>`: data layout defined by _rows_ and _columns_ dimensions. Data is packed into a 'column-major' arrangement of row vectors of fixed length.
-- Rank: 2
-- TensorCoord type: `MatrixCoord`
-- Shape type: `MatrixShape`
-- Stride rank: 1

- `RowMajorInterleaved<k>`: data layout defined by _rows_ and _columns_ dimensions. Data is packed into a 'row-major' arrangement of column vectors of fixed length.
-- Rank: 2
-- TensorCoord type: `MatrixCoord`
-- Shape type: `MatrixShape`
-- Stride rank: 1

Tensor layouts:
- `TensorNHWC`:

Permuted Shared Memory Layouts:
- `TensorOpCongruous<ElementSize>`
- `TensorOpCrosswise<ElementSize>`


### Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
