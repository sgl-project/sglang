# Getting Started With CuTe

CuTe is a collection of C++ CUDA template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. CuTe provides `Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.

The core abstraction of CuTe are the hierarchically multidimensional layouts which can be composed with data arrays to represent tensors. The representation of layouts is powerful enough to represent nearly everything we need to implement efficient dense linear algebra. Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

## System Requirements

CuTe shares CUTLASS 3.x's software requirements,
including NVCC with a C++17 host compiler.

## Knowledge prerequisites

CuTe is a CUDA C++ header-only library.  It requires C++17
(the revision of the C++ Standard that was released in 2017).

Throughout this tutorial, we assume intermediate C++ experience.
For example, we assume that readers know
how to read and write templated functions and classes, and
how to use the `auto` keyword to deduce a function's return type.
We will be gentle with C++ and explain some things
that you might already know.

We also assume intermediate CUDA experience.
For example, readers must know
the difference between device and host code,
and how to launch kernels.

## Building Tests and Examples

CuTe's tests and examples build and run as part of CUTLASS's normal build process.

CuTe's unit tests live in the [`test/unit/cute`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute) subdirectory.

CuTe's examples live in the [`examples/cute`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) subdirectory.

## Library Organization

CuTe is a header-only C++ library, so there is no source code that needs building. Library headers are contained within the top level [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) directory, with components of the library grouped by directories that represent their semantics.

|        Directory       |        Contents        |
|------------------------|------------------------|
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | Each header in the top level corresponds to one of the fundamental building blocks of CuTe, such as [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) and [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp). |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | Implementations of STL-like objects, such as tuple, array, and aligned array.  |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | Fundamental numeric data types that include nonstandard floating-point types, nonstandard integer types, complex numbers, and integer sequence.  |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | Implementations of utility algorithms such as copy, fill, and clear that automatically leverage architecture-specific features if available. |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | Wrappers for architecture-specific matrix-matrix multiply and copy instructions. |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | Meta-information for instructions in `arch` and utilities like partitioning and tiling.

## Tutorial

This directory contains a CuTe tutorial in Markdown format.
The file
[`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md)
explains how to implement dense matrix-matrix multiply using CuTe components.
It gives a broad overview of CuTe and thus would be a good place to start.

Other files in this directory discuss specific parts of CuTe.

* [`01_layout.md`](./01_layout.md) describes `Layout`, CuTe's core abstraction.

* [`02_layout_algebra.md`](./02_layout_algebra.md) describes more advanced `Layout` operations and the CuTe layout algebra.

* [`03_tensor.md`](./03_tensor.md) describes `Tensor`,
  a multidimensional array abstraction which composes `Layout`
  with an array of data.

* [`04_algorithms.md`](./04_algorithms.md) summarizes CuTe's
  generic algorithms that operate on `Tensor`s.

* [`0t_mma_atom.md`](./0t_mma_atom.md) demonstrates CuTe's meta-information and interface to our GPUs'
  architecture-specific Matrix Multiply-Accumulate (MMA) instructions.

* [`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md) walks through building a GEMM from scratch using CuTe.

* [`0y_predication.md`](./0y_predication.md) explains what to do
  if a tiling doesn't fit evenly into a matrix.

* [`0z_tma_tensors.md`](./0z_tma_tensors.md) explains an advanced `Tensor` type that CuTe uses to support TMA loads and stores.

## Quick Tips

### How do I print CuTe objects on host or device?

The `cute::print` function has overloads for almost all CuTe types, including Pointers, Integers, Strides, Shapes, Layouts, and Tensors.  When in doubt, try calling `print` on it.

CuTe's print functions work on either host or device.
Note that on device, printing is expensive.
Even just leaving print code in place on device,
even if it is never called
(e.g., printing in an `if` branch that is not taken at run time),
may generate slower code.
Thus, be sure to remove code that prints on device after debugging.

You might also only want to print on thread 0 of each threadblock, or threadblock 0 of the grid.  The `thread0()` function returns true only for global thread 0 of the kernel, that is, for thread 0 of threadblock 0.  A common idiom for printing CuTe objects to print only on global thread 0.

```c++
if (thread0()) {
  print(some_cute_object);
}
```

Some algorithms depend on some thread or threadblock,
so you may need to print on threads or threadblocks other than zero.
The header file
[`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp),
among other utilities,
includes the function `bool thread(int tid, int bid)`
that returns `true` if running on thread `tid` and threadblock `bid`.

#### Other output formats

Some CuTe types have special printing functions that use a different output format.

The `cute::print_layout` function will display any rank-2 layout in a plain text table. This is excellent for visualizing the map from coordinates to indices.

The `cute::print_tensor` function will display any rank-1, rank-2, rank-3, or rank-4 tensor in a plain text multidimensional table. The values of the tensor are printed so you can verify the tile of data is what you expect after a copy, for example.

The `cute::print_latex` function will print LaTeX commands that you can use to build a nicely formatted and colored tables via `pdflatex`. This work for `Layout`, `TiledCopy`, and `TiledMMA`, which can be very useful to get a sense of layout patterns and partitioning patterns within CuTe.

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
