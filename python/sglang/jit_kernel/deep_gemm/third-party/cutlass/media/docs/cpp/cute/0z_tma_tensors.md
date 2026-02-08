# CuTe TMA Tensors

Along your travels, you may find strange looking CuTe Tensors that are printed as something like
```
ArithTuple(0,_0,_0,_0) o ((_128,_64),2,3,1):((_1@0,_1@1),_64@1,_1@2,_1@3)
```
What is an `ArithTuple`? Are those tensor strides? What do those mean? What is this for?

This documentation intends to answer those questions and introduce some of the more advanced features of CuTe.

## Introduction to TMA instructions

The Tensor Memory Accelerator (TMA) is a set of instructions for copying possibly multidimensional arrays between global and shared memory.  TMA was introduced in the Hopper architecture.  A single TMA instruction can copy an entire tile of data all at once.  As a result, the hardware no longer needs to compute individual memory addresses and issue a separate copy instruction for each element of the tile.

To accomplish this, the TMA instruction is given a *TMA descriptor*, which is a packed representation of a multidimensional tensor in global memory with 1, 2, 3, 4, or 5 dimensions. The TMA descriptor holds

* the base pointer of the tensor;

* the data type of the tensor's elements (e.g., `int`, `float`, `double`, or `half`);

* the size of each dimension;

* the stride within each dimension; and

* other flags representing the smem box size, smem swizzling patterns, and out-of-bounds access behavior.

This descriptor must be created on the host before kernel execution.
It is shared between all thread blocks that will be issuing TMA instructions.
Once inside the kernel, the TMA is executed with the following parameters:

* pointer to the TMA descriptor;

* pointer to the SMEM; and

* coordinates into the GMEM tensor represented within the TMA descriptor.

For example, the interface for TMA-store with 3-D coordinates looks like this.

```cpp
struct SM90_TMA_STORE_3D {
  CUTE_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2) {
    // ... invoke CUDA PTX instruction ...
  }
};
```

We observe that the TMA instruction does not directly consume pointers to global memory. Indeed, the global memory pointer is contained in the descriptor, is considered constant, and is NOT a separate parameter to the TMA instruction. Instead, the TMA consumes TMA coordinates into the TMA's view of global memory that is defined in the TMA descriptor.

That means that an ordinary CuTe Tensor that stores a GMEM pointer and computes offsets and new GMEM pointers is useless to the TMA.

What do we do?

## Building a TMA Tensor

### Implicit CuTe Tensors

All CuTe Tensors are compositions of Layouts and Iterators. An ordinary global memory tensor's iterator is its global memory pointer. However, a CuTe Tensor's iterator doesn't have to be a pointer; it can be any random-access iterator.

One example of such an iterator is a *counting iterator*.
This represents a possibly infinite sequence of integers that starts at some value.
We call the members of this sequence *implicit integers*,
because the sequence is not explicitly stored in memory.
The iterator just stores its current value.

We can use a counting iterator to create a tensor of implicit integers,
```cpp
Tensor A = make_tensor(counting_iterator<int>(42), make_shape(4,5));
print_tensor(A);
```
which outputs
```
counting_iter(42) o (4,5):(_1,4):
   42   46   50   54   58
   43   47   51   55   59
   44   48   52   56   60
   45   49   53   57   61
```
This tensor maps logical coordinates to on-the-fly computed integers. Because it's still a CuTe Tensor, it can still be tiled and partitioned and sliced just like a normal tensor by accumulating integer offsets into the iterator.

But the TMA doesn't consume pointers or integers, it consumes coordinates. Can we make a tensor of implicit TMA
coordinates for the TMA instruction to consume? If so, then we could presumably also tile and partition and slice that tensor of coordinates so that we would always have the right TMA coordinate to give to the instruction.

### ArithTupleIterators and ArithTuples

First, we build a `counting_iterator` equivalent for TMA coordinates. It should support

* dereference to a TMA coordinate, and

* offset by another TMA coordinate.

We'll call this an `ArithmeticTupleIterator`. It stores a coordinate (a tuple of integers) that is represented as an `ArithmeticTuple`. The `ArithmeticTuple` is simply a (public subclass of) `cute::tuple` that has an overloaded `operator+` so that it can be offset by another tuple. The sum of two tuples is the tuple of the sum of the elements.

Now similar to `counting_iterator<int>(42)` we can create an implicit "iterator" (but without increment or other common iterator operations) over tuples that can be dereferenced and offset by other tuples
```cpp
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
print(*citer_2);
```
which outputs
```
(42,7,_9)
```

A TMA Tensor can use an iterator like this to store the current TMA coordinate "offset". The "offset" here is in quotes because it's clearly not a normal 1-D array offset or pointer.

In summary, one creates a TMA descriptor for the *whole global memory tensor*. The TMA descriptor defines a view into that tensor and the instruction takes TMA coordinates into that view. In order to generate and track those TMA coordinates, we define an implicit CuTe Tensor of TMA coordinates that can be tiled, sliced, and partitioned the exact same way as an ordinary CuTe Tensor.

We can now track and offset TMA coordinates with this iterator, but how do we get CuTe Layouts to generate non-integer offsets?

### Strides aren't just integers

Ordinary tensors have a layout that maps
a logical coordinate `(i,j)` into a 1-D linear index `k`.
This mapping is the inner-product of the coordinate with the strides.

TMA Tensors hold iterators of TMA coordinates.
Thus, a TMA Tensor's Layout must map a logical coordinate
to a TMA coordinate, rather than to a 1-D linear index.

To do this, we can abstract what a stride is. Strides need not be integers, but rather any algebraic object that supports inner-product with the integers (the logical coordinate). The obvious choice is the `ArithmeticTuple` we used earlier since they can be added to each other, but this time additionally equipped with an `operator*` so it can also be scaled by an integer.

#### Aside: Integer-module strides

A group of objects that support addition between elements and product between elements and integers is called an integer-module.

Formally, an integer-module is an abelian group `(M,+)` equipped with `Z*M -> M`, where `Z` are the integers. That is, an integer-module `M` is
a group that supports inner products with the integers.
The integers are an integer-module.
Rank-R tuples of integers are an integer-module.

In principle, layout strides may be any integer-module.

#### Basis elements

CuTe's basis elements live in the header file `cute/numeric/arithmetic_tuple.hpp`.
To make it easy to create `ArithmeticTuple`s that can be used as strides, CuTe defines normalized basis elements using the `E` type alias. "Normalized" means that the scaling factor of the basis element is the compile-time integer 1.

| C++ object | Description             | String representation |
| ---        | ---                     | ---                   |
| `E<>{}`    | `1`                     | `1`                   |
| `E<0>{}`   | `(1,0,...)`             | `1@0`                 |
| `E<1>{}`   | `(0,1,0,...)`           | `1@1`                 |
| `E<0,0>{}` | `((1,0,...),0,...)`     | `1@0@0`               |
| `E<0,1>{}` | `((0,1,0,...),0,...)`   | `1@1@0`               |
| `E<1,0>{}` | `(0,(1,0,...),0,...)`   | `1@0@1`               |
| `E<1,1>{}` | `(0,(0,1,0,...),0,...)` | `1@1@1`               |

The "description" column in the above table
interprets each basis element as an infinite tuple of integers,
where all the tuple's entries not specified by the element's type are zero.
We count tuple entries from left to right, starting with zero.
For example, `E<1>{}` has a 1 in position 1: `(0,1,0,...)`.
`E<3>{}` has a 1 in position 3: `(0,0,0,1,0,...)`.

Basis elements can be *nested*.
For instance, in the above table, `E<0,1>{}` means that
in position 0 there is a `E<1>{}`: `((0,1,0,...),0,...)`. Similarly,
`1@1@0` means that `1` is lifted to position 1 to create `1@1`: `(0,1,0,...)`
which is then lifted again to position 0.

Basis elements can be *scaled*.
That is, they can be multiplied by an integer *scaling factor*.
For example, in `5*E<1>{}`, the scaling factor is `5`.
`5*E<1>{}` prints as `5@1` and means `(0,5,0,...)`.
The scaling factor commutes through any nesting.
For instance, `5*E<0,1>{}` prints as `5@1@0`
and means `((0,5,0,...),0,...)`.

Basis elements can also be added together,
as long as their hierarchical structures are compatible.
For example, `3*E<0>{} + 4*E<1>{}` results in `(3,4,0,...)`.
Intuitively, "compatible" means that
the nested structure of the two basis elements
matches well enough to add the two elements together.

#### Linear combinations of strides

Layouts work by taking the inner product
of the natural coordinate with their strides.
For strides made of integer elements, e.g., `(1,100)`,
the inner product of the input coordinate `(i,j)`
and the stride is `i + 100j`.
Offsetting an "ordinary" tensor's pointer and this index
gives the pointer to the tensor element at `(i,j)`.

For strides of basis elements, we still compute the inner product of the natural coordinate with the strides.
For example, if the stride is `(1@0,1@1)`,
then the inner product of the input coordinate `(i,j)`
with the strides is `i@0 + j@1 = (i,j)`.
That translates into the (TMA) coordinate `(i,j)`.
If we wanted to reverse the coordinates,
then we could use `(1@1,1@0)` as the stride.
Evaluating the layout would give `i@1 + j@0 = (j,i)`.

A linear combination of basis elements
can be interpreted as a possibly multidimensional and hierarchical coordinate.
For instance, `2*2@1@0 + 3*1@1 + 4*5@1 + 7*1@0@0`
means `((0,4,...),0,...) + (0,3,0,...) + (0,20,0,...) + ((7,...),...) = ((7,4,...),23,...)`
and can be interpreted as the coordinate `((7,4),23)`.

Thus, linear combinations of these strides can be used to generate TMA coordinates.
These coordinates, in turn, can be used to offset TMA coordinate iterators.

### Application to TMA Tensors

Now we can build CuTe Tensors like the one seen in the introduction.

```cpp
Tensor a = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<0>{}, E<1>{}));
print_tensor(a);

Tensor b = make_tensor(make_inttuple_iter(0,0),
                       make_shape (     4,      5),
                       make_stride(E<1>{}, E<0>{}));
print_tensor(b);
```
prints
```
ArithTuple(0,0) o (4,5):(_1@0,_1@1):
  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)
  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)
  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)
  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)

ArithTuple(0,0) o (4,5):(_1@1,_1@0):
  (0,0)  (1,0)  (2,0)  (3,0)  (4,0)
  (0,1)  (1,1)  (2,1)  (3,1)  (4,1)
  (0,2)  (1,2)  (2,2)  (3,2)  (4,2)
  (0,3)  (1,3)  (2,3)  (3,3)  (4,3)
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
