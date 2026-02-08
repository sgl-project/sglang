![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Tile Iterator Concepts")

# Tile Iterator Concepts

Note: CUTLASS 3.0 deprecates all tile access iterators in favour of CuTe's single
vocabulary type `cute::Tensor`, which is parameterized on `cute::Layout`.
`cute::Tensor`s can therefore be manipulated with the same layout algebra as all CuTe layouts.
This removes the need for bespoke types that encapsulate iterator properties.
The following text thus only applies to legacy CUTLASS 2.x API and related types.

CUTLASS 2.x implements generic algorithms on tiles of matrix or tensors of constant size. These may
be considered as partitions of tensors of infinite size, with a range of partitions accessible
by _tile iterators_.

Various data structures may make operations such as random access to tiles inexpensive,
while data structures may not offer random access at all. For example, iterating over a linked
list of matrices requires sequential traversal. Algorithms implemented in terms of sequences of tiles 
should require only the minimum set of operators be defined for tile iterators.

This document describes a set of C++ concepts which may be used to define tile iterators used
by CUTLASS algorithms.  ("Concept" here does not refer to a C++20 concept that uses the `concept` keyword.
Rather, it refers to a set of requirements on a type.)
Each concept specifies members and type definitions that a tile iterator
must implement. Frequently, a tile iterator implements several concepts, and its members are
the union of the members from each individual concept. These definitions were inspired by
[Boost "New style" iterator concepts](https://www.boost.org/doc/libs/1_40_0/libs/iterator/doc/new-iter-concepts.html).

The set of all possible combinations of these concepts is quite large, however most tile iterator
templates can be described by one of several combinations. The section 
Frequently Used Tile Iterator Concepts describes several common interfaces used throughout CUTLASS.

## Definitions

**_Base Tile Iterator Concept_.** All tile iterators must describe an _Element_ type as well as a _Shape_.
```c++
/// Base concept for all tile iterators
struct TileIteratorConcept {
  using Element;           ///< Element type composing tile (concept: numeric type or Array<>)
  using Shape;             ///< Shape type describing extent of tile. The shape concept depends 
                           ///  on iterator implementation.
};
```

**_Contiguous Memory Tile Iterator Concept_.** Iterators over tiles stored arbitrarily within
a continuous block of data in memory. Linear offset in units of _Element_ may be added to 
internally held pointers to 'move' the iterator in memory.

```c++
/// Tile iterator over partitions of a tensor in contiguous memory which may be referenced via a
/// TensorRef object.
struct ContiguousMemoryTileIterator : public TileIteratorConcept {

  using Index;            ///< index type used to add pointer offsets

  /// Adds a linear offset in units of Element to internal pointer(s) into tensor
  CUTLASS_DEVICE 
  void add_pointer_offset(Index pointer_offset);
};
```

**_Readable Tile Iterator Concept_.** Iterators that may be read from define a `Fragment` type holding
each thread's part of the data to be loaded. An explicit `load()` method reads the tile from memory,
and places each thread's part in its `Fragment` object. 

```c++
/// Tile iterator capable of loading tiles from memory into fragments
struct ReadableTileIteratorConcept {
  
  using Fragment;              ///< fragment object derived from cutlass::Array<Element, N>

  CUTLASS_DEVICE
  void load(Fragment &frag);   ///< loads a fragment from memory
};
```

**_Readable Contiguous Tile Iterator Concept_.** Iterators reading from contiguous memory
support an optional pointer offset that is added to any internally managed pointers before
performing the load. This provides a convenient method to fold an offset in with load
operations.

```c++
/// Union of the following tile iterator concepts:
///
///   - ReadableTileIteratorConcept
///   - ContiguousMemoryTileIterator
///
struct ReadableContiguousTileIteratorConcept : 
  public ReadableTileIteratorConcept, 
  public ContiguousMemoryTileIterator {

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset);                      ///< loads a tile with a linear offset
};
```

**_Writeable Tile Iterator Concept_.** Iterators that may write to memory define a `Fragment` type holding
each thread's part of the data to be written. An explicit `store()` method writes the tile to memory. 

```c++
/// Tile iterator capable of storing tiles from memory
struct WriteableTileIteratorConcept {

  using Fragment;                     ///< fragment object derived from cutlass::Array<Element, N>

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag);             ///< stores a fragment to memory
};
```

**_Writeable Contiguous Tile Iterator Concept_.** Iterators writing to contiguous memory
support an optional pointer offset that is added to any internally managed pointers before
performing the store operation. This provides a convenient method to fold an offset into the
store.
```c++
/// Union of the following tile iterator concepts:
///
///   - WriteableTileIteratorConcept
///   - ContiguousMemoryTileIterator
///
struct WriteableContiguousTileIteratorConcept : 
  public WriteableTileIteratorConcept, 
  public ContiguousMemoryTileIterator {

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store to the tensor
    Index pointer_offset);                      ///< stores a tile with a linear offset
};
```

**_Forward Tile Iterator Concept_.** This concept offers traversal "forward" by one tile in
a pre-defined sequence. Often, this sequence is relevant to the context in which the iterator
was defined, such as along the _K_ dimension of a GEMM operation. Equality operators are defined
to determine whether two iterators point to the same tile.
```c++
/// Tile iterator that may be incremented along a traversal sequence.
struct ForwardTileIteratorConcept {

  CUTLASS_DEVICE bool operator==(TileIterator const &it);        ///< true if iterators point to same tile, false if otherwise
  CUTLASS_DEVICE bool operator!=(TileIterator const &it);        ///< false if iterators point to same tile, true if otherwise

  CUTLASS_DEVICE ForwardTileIteratorConcept & operator++();      ///< pre-increment - advance to next tile in sequence
  CUTLASS_DEVICE ForwardTileIteratorConcept operator++(int);     ///< post-increment - advance to next tile in sequence
};
```

**_Bidirectional Tile Iterator Concept_.** This concept permits traversal both forward and backward.
```c++
/// Tile iterator which may be traverse in both directions along a defined sequence.
struct BidirectionalTileIteratorConcept : public ForwardTileIteratorConcept {

  CUTLASS_DEVICE
  BidirectionalTileIteratorConcept & operator--();      ///< pre-decrement - traverse to previous tile in sequence

  CUTLASS_DEVICE
  BidirectionalTileIteratorConcept operator--(int);     ///< post-decrement - traverse to previous tile in sequence
};
```

**_Random Access Tile Iterator Concept_.** This iterator defines random access operations in the logical
coordinate system of the underlying tensor. Thus, tensors must have a defined _Layout_ with associated
_TensorCoord_ coordinate describing logical position within the tensor and _TensorRef_ reference type. 
It may be advanced forward or backwards by an offset specified as units of whole tiles along each dimension.
```c++
/// Tile iterator offering random access to tiles in contiguous memory.
struct RandomAccessTileIteratorConcept : 
  public BidirectionalTileIteratorConcept, 
  public ContiguousMemoryTileIterator {

  using Layout;           ///< Layout object mapping 
  using TensorRef;        ///< Tensor Reference object
  using TensorCoord;      ///< Logical coordinate in referenced tensor

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  RandomAccessTileIteratorConcept & add_tile_offset(TensorCoord const &tile_offset);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  RandomAccessTileIteratorConcept & operator+=(TensorCoord const &tile_offset);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  RandomAccessTileIteratorConcept & operator-=(TensorCoord const &tile_offset);
};
```

**_Readable Random Access Tile Iterator Concept_.** Readable random access iterators
accept an additional tile offset in logical coordinate space when loading fragments.
```c++
/// Loads a fragment with a logical coordinate offset in units of whole tiles.
struct ReadableRandomAccessTileIteratorConcept : 
  public RandomAccessTileIteratorConcept, 
  public ReadableTileIteratorConcept {

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset);            ///< loads a tile with a logical offset in units of whole tiles
};
```

**_Readable Random Access Contiguous Tile Iterator Concept_.** Readable random access iterators
accept an additional tile offset in logical coordinate space when loading fragments.
```c++
/// Loads a fragment with a logical coordinate offset in units of whole tiles.
struct ReadableRandomAccessContiguousTileIteratorConcept : 
  public ReadableRandomAccessTileIteratorConcept, 
  ReadableContiguousTileIteratorConcept {

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset);                      ///< loads a tile with a logical offset AND a pointer offset
};
```
**_Writeable Random Access Tile Iterator Concept_.** Writeable random access iterators
accept an additional tile offset in logical coordinate space when storing fragments.
```c++
/// Stores a fragment with a logical coordinate offset in units of whole tiles.
struct WriteableRandomAccessTileIteratorConcept : 
  public RandomAccessTileIteratorConcept, 
  public WriteableContiguousTileIteratorConcept {
  
  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE 
  void store(
    Fragment const &frag,                       ///< fragment to store to the location pointed to by the tensor
    TensorCoord const &tile_offset);            ///< stores a tile with a given offset from the current iterator
};
```

**_Writeable Random Access Contiguous Tile Iterator Concept_.** Writeable random access iterators
accept an additional tile offset in logical coordinate space when storing fragments.
```c++
/// Stores a fragment with a logical coordinate offset in units of whole tiles.
struct WriteableRandomAccessContiguousTileIteratorConcept : 
  public WriteableRandomAccessTileIteratorConcept, 
  public WriteableContiguousTileIteratorConcept {

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE 
  void store(
    Fragment const &frag,                       ///< fragment to store to the location pointed to by the tensor
    TensorCoord const &tile_offset,             ///< stores a tile with a logical offset in units of whole tiles
    Index pointer_offset);                      ///< stores a tile witha logical offset AND a pointer offset
};
```

**_Masked Tile Iterator Concept_.** Matrix and tensors may not always be multiples of whole tiles. 
Masked tile iterators define a `Mask` type which may be used to guard accesses to memory. The
semantics and interface of this `Mask` are implementation-defined details of each tile iterator,
but several convenience methods are defined for interacting with the mask such as efficiently
clearing or enabling all guarded memory accesses.
```c++
/// Supports iterating over tiles that are not 'whole' in memory. Iterator maintains a mask object
/// which guards against out-of-bounds access.
///
/// Note, this concept definition does not formally define operations on the mask or methods it
/// supports. These remain implementation-dependent details of iterators implementing this concept.
struct MaskedTileIteratorConcept {

  using Mask;                                        ///< mask object used to guard against acceses.

  CUTLASS_DEVICE void clear_mask();                  ///< efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask();                 ///< efficiently enables all accesses guarded by mask

  CUTLASS_DEVICE void get_mask(Mask &mask);          ///< gets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask);    ///< sets the mask
};
```

## Frequently Used Tile Iterator Concepts

This section describes several frequently used compositions of the basic tile iterator concepts. They are
listed here as complete type declarations for convenience of the reader.

**_Writeable, Readable, Forward, Contiguous Memory Tile Iterator Concept_.** 
This combines several of the basic iterator concepts to
yield a tile iterator capable of loading and storing tiles as well as advancing forward along a traversal sequence.
```c++
/// This tile iterator embodies several of the above:
///
///   - ForwardTileIteratorConcept
///   - ReadableContiguousTileIteratorConcept
///   - WriteableContiguousTileIteratorConcept
/// 
/// It is restated explicitly for convenience of the reader.
/// 
struct WriteableReadableForwardContiguousTileIteratorConcept {

  //
  // Data types
  //
  
  using Element;           ///< Element type composing tile.
  using Shape;             ///< Shape type describing extent of tile. The shape concept depends 
                           ///  on iterator implementation
  using Index;             ///< index type used as base for TensorCoord
  using Fragment;          ///< fragment object derived from cutlass::Array<Element, N>

  //
  // Methods
  //

  /// Adds a linear offset in units of Element to internal pointer(s) into tensor
  CUTLASS_DEVICE 
  void add_pointer_offset(Index offset);

  /// true if iterators point to same tile, false if otherwise
  CUTLASS_DEVICE bool operator==(WriteableReadableForwardContiguousTileIteratorConcept const &it);

  ///< false if iterators point to same tile, true if otherwise
  CUTLASS_DEVICE bool operator!=(WriteableReadableForwardContiguousTileIteratorConcept const &it);

  /// pre-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableForwardContiguousTileIteratorConcept & 
  operator++();

  ///< post-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableForwardContiguousTileIteratorConcept 
  operator++(int);

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag);                    ///< fragment to be loaded from memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag);             ///< fragment to store to memory

  /// Stores a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store to memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when storing
};
```

**_Writeable, Readable, Random Access, Contiguous Memory Tile Iterator Concept_.** 
This combines several of the basic iterator concepts to
yield a tile iterator with random access suitable for loading matrix operands for GEMM.
```c++
/// This tile iterator embodies several of the above:
///
///   - ReadableRandomAccessContiguousTileIteratorConcept
///   - WriteableRandomAccessContiguousTileIteratorConcept
/// 
/// It is restated explicitly for convenience of the reader.
/// 
struct WriteableReadableRandomAccessContiguousTileIteratorConcept {

  //
  // Data types
  //
  
  using Element;           ///< Element type composing tile.
  using Shape;             ///< Shape type describing extent of tile. The shape concept depends 
                           ///  on iterator implementation
  using Layout;            ///< Layout object mapping 
  using TensorRef;         ///< Tensor Reference object
  using TensorCoord;       ///< Logical coordinate in referenced tensor
  using Index;             ///< index type used as base for TensorCoord
  using Fragment;          ///< fragment object derived from cutlass::Array<Element, N>

  //
  // Methods
  //

  /// Adds a linear offset in units of Element to internal pointer(s) into tensor
  CUTLASS_DEVICE 
  void add_pointer_offset(Index pointer_offset);

  /// true if iterators point to same tile, false if otherwise
  CUTLASS_DEVICE bool operator==(WriteableReadableRandomAccessContiguousTileIteratorConcept const &it);

  ///< false if iterators point to same tile, true if otherwise
  CUTLASS_DEVICE bool operator!=(WriteableReadableRandomAccessContiguousTileIteratorConcept const &it);

  /// pre-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & 
  operator++();

  ///< post-increment - traverse to next tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept 
  operator++(int);

  /// pre-decrement - traverse to previous tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & 
  operator--();

  ///< post-decrement - traverse to previous tile in sequence
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept 
  operator--(int);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & operator+=(TensorCoord const &tile_offset);

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  WriteableReadableRandomAccessContiguousTileIteratorConcept & operator-=(TensorCoord const &tile_offset);

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag);                    ///< fragment to be loaded from memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to be loaded from memory
    TensorCoord const &tile_offset);            ///< loads a tile with a logical offset in units of whole tiles

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to be loaded from memory
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset);                      ///< loads a tile with a logical offset AND a pointer offset

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag);             ///< fragment to store to memory

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store to memory
    Index pointer_offset);                      ///< linear offset (in units of Element) when loading

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE 
  void store(
    Fragment const &frag,                       ///< fragment to store to memory
    TensorCoord const &tile_offset);            ///< stores with logical offset in units of whole tiles

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE 
  void store(
    Fragment const &frag,                       ///< fragment to store to memory
    TensorCoord const &tile_offset,             ///< stores with logical offset in units of whole tiles
    Index pointer_offset);
};
```

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
