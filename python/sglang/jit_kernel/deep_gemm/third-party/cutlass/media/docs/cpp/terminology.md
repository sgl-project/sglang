![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Terminology")

# CUTLASS Terminology

**cute::Layout**: A `cute::Layout` vocabulary type composed of the hierarchical `cute::Shape` and `cute::Stride`
tuples that is used throughout CUTLASS 3.0 to represent and manipulate thread and data layouts. More details are included in the [CuTe specific tensor type documentation](cute/03_tensor.md).

**cute::Tensor**: A pointer backed by a `cute::Layout` used to represent a tensor. More details are included in the [CuTe specific tensor type documentation](cute/03_tensor.md).

**Capacity**: (scalar) physical number of elements in memory required to store a multidimensional object; expressed as the type's LongIndex type
  - example: the capacity of a column-major matrix is `lda * N`

**Element**: data type describing one item in a multidimensional tensor, array, or matrix

**Extent**: (vector-valued quantity) the logical size of each dimension of a multidimensional index space. Consistent with the [C++ Standard Library](https://en.cppreference.com/w/cpp/types/extent).
  - `Coord<N> extent()`
  - `Index extent(int dim)`

**Fragment**: a register-backed array of elements used to store a thread's part of a tile

**Index**: signed integer representing quantities aligned with a logical dimension

**Layout**: functor mapping logical coordinates of a tensor to linear offset (as LongIndex); owns stride vectors, if any. 

**LongIndex**: signed integer representing offsets in memory; typically wider than Index type

**Numeric Type**: a CUTLASS data type used to represent real-valued quantities; is trivially copyable.

**Pitch Linear**: linear memory allocation obtained from a user-defined 2-D size, which specifies the 
contiguous and strided dimensions of a tile. 

**Planar Complex**: representation of complex tensors as two real-valued tensors, with real elements in one part and imaginary elements in another part of identical layout, separated by an offset

**Policy**: additional details extending the interface of a template guiding internal implementation; 
  typically used to target specific design points known to be efficient

**Rank**: number of dimensions in a multidimensional index space, array, tensor, or matrix. Consistent with 
  [C++ Standard Library](https://en.cppreference.com/w/cpp/types/rank)

**Register**: in device code, registers are the most efficient storage for statically sized arrays of elements.
  Arrays may be expected to be stored in registers if all accesses are made via constexpr indices or within
  fully unrolled loops.

**Residue**: partial tile or matrix computation which may require special accommodation for functional correctness or performance

**Size**: (scalar) number of logical elements in a tensor; equal to the product of each member of `extent()`
  - `LongIndex size()`

`sizeof_bits<T>::value` - template pattern returning the size of a numeric type or array in units of bits

**Storage**: when appropriate, refers to some alternative type used to store a packed collection of elements; 
  may be used to handle bit-level packing or make types safe for use in unions

**TensorRef**: contains base pointer and _Layout_ object for referencing infinitely-sized tensor object

**TensorView**: contains _TensorRef_ and extent of a finite mathematical object

**Tile**: partitions of a tensor that have constant extents and layout known at compile time

**Trait**: characteristics of a fully-specialized type, typically used in metaprogramming reflection

**View**: an object containing references to a data structure that it does not own; typically, construction of views is lightweight

**Warp**: a collection of hardware threads executing in lock-step; warp-level operations typically rely on cooperation among the threads within the warp

`AlignedBuffer<T, N>`: statically sized array type; union-safe, no construction guarantee for elements

`Array<T, N>`: container for holding numeric types - handles bit packing for small numeric types (e.g. int4_t, uint4_t, bin1_t)
  `sizeof(Array<T, N>)` - gives expected value in units of bytes with minimum storage of `1 B`: (sizeof_bits<T>::value * N) / 8

**Operator**: an object performing a computation on matrix or tensor objects. May be further refined by scope within the execution model hierarchy. Deprecated starting CUTLASS 3.0,
replaced by [MMA and Copy atoms from CuTe](cute/0t_mma_atom.md).

**Tile Iterator**: abstraction for accessing and traversing a sequence of tiles in a tensor; CUTLASS specifies 
  [formal concepts for tile iterators](tile_iterator_concept.md). Deprecated starting CUTLASS 3.0.
  Replaced by `cute::Layout` in equivalent usage scenarios to represent data tensors.

**Thread Map**: abstraction for defining how threads are mapped to a given tile. Deprecated starting CUTLASS 3.0.
  Replaced by `cute::Layout` in equivalent usage scenarios to represent thread tensors.

[comment]: <> (Don't remove this. This "##" is to prevent Sphinx from throwing build WARNING.)
## 

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
