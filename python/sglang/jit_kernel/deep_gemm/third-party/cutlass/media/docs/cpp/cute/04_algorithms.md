# CuTe Tensor algorithms

This section summarizes the interfaces and implementations
of common numerical algorithms performed on `Tensor`s.

The implementation of these algorithms may be found in the
[include/cute/algorithm/](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/)
directory.

## `copy`

CuTe's `copy` algorithm copies the elements of a source `Tensor`
into the elements of a destination `Tensor`.
The various overloads of `copy` can be found in
[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp).

### Interface and specialization opportunities

A `Tensor` encapsulates the data type, data location,
and possibly also the shape and stride of the tensor at compile time.
As a result, `copy` can and does dispatch,
based on the types of its arguments,
to use any of various synchronous or asynchronous hardware copy instructions.

The `copy` algorithm has two main overloads.
The first just takes the source `Tensor` and the destination `Tensor`.

```c++
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

The second takes those two parameters, plus a `Copy_Atom`.

```c++
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

The two-parameter `copy` overload picks a default implementation
based only on the types of the two `Tensor` parameters.
The `Copy_Atom` overload lets callers override that default
by specifying a nondefault `copy` implementation.

### Parallelism and synchronization depend on parameter types

Either the default implementation or
the implementation selected by a `Copy_Atom` overload
may use none or all available parallelism,
and may have a variety of synchronization semantics.
The behavior depends on `copy`'s parameter types.
Users are expected to figure this out based on their knowledge
of the architecture on which they are running.
(Developers often write a custom optimized kernel
for each GPU architecture.)

The `copy` algorithm may be sequential per thread,
or it may be parallel across some collection of threads
(e.g., a block or cluster).

If `copy` is parallel,
then the collection of participating threads
may need synchronization before any thread in the collection
may assume that the copy operation has completed.
For example, if the participating threads form a thread block,
then users must invoke `__syncthreads()`
or the Cooperative Groups equivalent
before they may use the results of `copy`.

The `copy` algorithm may use asynchronous copy instructions,
such as `cp.async`, or its C++ interface `memcpy_async`.
In that case, users will need to perform
the additional synchronization appropriate to that underlying implementation
before they may use the results of the `copy` algorithm.
[The CuTe GEMM tutorial example](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)
shows one such synchronization method.
More optimized GEMM implementations use pipelining techniques
to overlap asynchronous `copy` operations with other useful work.

### A generic copy implementation

A simple example of a generic `copy` implementation
for any two `Tensor`s looks like this.

```c++
template <class TA, class ALayout,
          class TB, class BLayout>
CUTE_HOST_DEVICE
void
copy(Tensor<TA, ALayout> const& src,  // Any logical shape
     Tensor<TB, BLayout>      & dst)  // Any logical shape
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

This generic `copy` algorithm addresses both `Tensor`s
with 1-D logical coordinates, thus traversing both `Tensor`s
in a logical column-major order.
Some reasonable architecture-independent optimizations
would include the following.

1. If the two `Tensor`s have known memory spaces with optimized
   access instructions (like `cp.async`), then dispatch to the
   custom instruction.

2. The two `Tensor`s have static layouts and it can be proven
that element vectorization is valid -- for example, four `ld.global.b32`s
can be combined into a single `ld.global.b128` -- then vectorize the source
   and destinations tensors.

3. If possible, validate that the copy instruction to be used is
   appropriate for the source and destination tensors.

CuTe's optimized copy implementations can do all of these.

## `copy_if`

CuTe's `copy_if` algorithm lives in the same header as `copy`,
[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp).
The algorithm takes source and destination `Tensor` parameters like `copy`,
but it also takes a "predication `Tensor`"
with the same shape as the input and output.
Elements of the source `Tensor` are only copied
if the corresponding predication `Tensor` element is nonzero.

For details on why and how to use `copy_if`,
please refer to the
["predication" section of the tutorial](./0y_predication.md).

## `gemm`

### What `gemm` computes

The `gemm` algorithm takes three `Tensor`s, A, B, and C.
What it does depends on the number of modes
that its `Tensor` parameters have.
We express these modes using letters.

* V indicates a "vector," a mode of independent elements.

* M and N indicate the number of rows resp. columns
  of the matrix result C of the BLAS's GEMM routine.

* K indicates the "reduction mode" of GEMM,
  that is, the mode along which GEMM sums.
  Please see the [GEMM tutorial](./0x_gemm_tutorial.md) for details.

We list the modes of the input `Tensor`s A and B,
and the output `Tensor` C,
using a notation `(...) x (...) => (...)`.
The two leftmost `(...)` describe A and B (in that order),
and the `(...)` to the right of the `=>` describes C.

1. `(V) x (V) => (V)`. The element-wise product of vectors: C<sub>v</sub> += A<sub>v</sub> B<sub>v</sub>. Dispatches to FMA or MMA.

2. `(M) x (N) => (M,N)`. The outer product of vectors: C<sub>mn</sub> += A<sub>m</sub> B<sub>n</sub>. Dispatches to (4) with V=1.

3. `(M,K) x (N,K) => (M,N)`. The product of matrices: C<sub>mn</sub> += A<sub>mk</sub> B<sub>nk</sub>. Dispatches to (2) for each K.

4. `(V,M) x (V,N) => (V,M,N)`. The batched outer product of vectors: C<sub>vmn</sub> += A<sub>vm</sub> B<sub>vn</sub>. Optimizes for register reuse and dispatches to (1) for each M, N.

5. `(V,M,K) x (V,N,K) => (V,M,N)`. The batched product of matrices: C<sub>vmn</sub> += A<sub>vmk</sub> B<sub>vnk</sub>. Dispatches to (4) for each K.

Please refer to the [GEMM tutorial](./0x_gemm_tutorial.md)
for an overview of CuTe's convention for ordering the modes.
For example, if K appears, it always appears rightmost ("outermost").
If V appears, it always appears leftmost ("innermost").

### Dispatch to optimized implementations

Just like with `copy`, CuTe's implementations of `gemm`
uses its `Tensor` arguments' types to dispatch
to an appropriately optimized implementation.
Also like `copy`, `gemm` takes an optional `MMA_Atom` parameter
that lets callers override the default `FMA` instruction
that CuTe would select based on the `Tensor` arguments' types.

For more information on `MMA_Atom` and on specialization of `gemm`
for different architectures, please refer to the
[MMA section of the tutorial](./0t_mma_atom.md).

## `axpby`

The `axpby` algorithm lives in the header file
[`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp).
It assigns to $y$ the result of $\alpha x + \beta y$,
where $\alpha$ and $\beta$ are scalars and $x$ and $y$ are `Tensor`s.
The name stands for "Alpha times X Plus Beta times Y,"
and is a generalization of the original BLAS "AXPY" routine
("Alpha times X Plus Y").

## `fill`

The `fill` algorithm lives in the header file
[`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp).
It overwrites the elements of its `Tensor` output argument
with a given scalar value.

## `clear`

The `clear` algorithm lives in the header file
[`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp).
It overwrites the elements of its `Tensor` output argument with zeros.

## Other algorithms

CuTe provides other algorithms.
Their header files can be found in the
[`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm)
directory.

## Copyright

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
