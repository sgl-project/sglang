# CUTLASS 3.0 GEMM Backwards Compatibility

Although CUTLASS 3.0 restructures the GEMM hierarchy and introduces new types for the
threadblock layer and below, we intend the entire source code to be usable in user applications.
We expect users to be able to `#include` any source file from CUTLASS 3.0, whether
they implement the 2.x or the 3.x API, without breaking user builds. This means that a single
translation unit should be able to contain any valid kernel regardless of its API version. The
sections below discuss how `device` and `kernel` layer type names are made compatible across the
two API versions, and what the users can expect out of the `threadblock` layer API going forward.

## Compatible Device API

The entry point for CUTLASS's Device GEMM API
is the class
`cutlass::gemm::device::GemmUniversalAdapter`.
This class lives in the header file
[include/cutlass/gemm/device/gemm_universal_adapter.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).

`GemmUniversalAdapter` is a "universal adapter"
and serves as a common device interface
for both CUTLASS 3.x and CUTLASS 2.x kernels.
Its template parameter `GemmKernel`,
the GEMM kernel type, can be any of the following:

* `cutlass::gemm::kernel::GemmUniversal`,
  implementing CUTLASS 3.x API kernels;
* `cutlass::gemm::kernel::GemmUniversal`,
  implementing CUTLASS 2.x API kernels;
* Any valid CUTLASS 2.x `kernel` layer GEMM that
  was previously composable with `device::GemmUniversalAdapter`

Users implementing new kernels in either API should prefer
using `kernel::GemmUniversal` as the kernel type
and compose it with `device::GemmUniversalAdapter`.
Users with existing `kernel::Gemm` kernels
can continue to use them as template arguments
of `device::GemmUniversalAdapter`. They can adopt
`GemmUniversal` as a gradual migration path,
since `GemmUniversal` accepts either 3.0 or 2.x collectives.
Please see the [next section for `kernel::GemmUniversal`](#compatible-kernel-api) for details.

`GemmUniversalAdapter` presents a single
host-side interface to both 3.0 and 2.x kernels.
CUTLASS accomplishes this by
specializing `GemmUniversalAdapter`'s implementation
on either 2.x API implementing kernel layer GEMMs, or 3.x API
implementing kernel layer GEMMs (as detected by `gemm::detail::IsCutlass3GemmKernel`
discussed below). As a result, `GemmUniversalAdapter`'s behavior
might differ between the two specializations.

### Device API design differences

In CUTLASS 2.x, the Device API was more closely tied
to the Kernel API.  In CUTLASS 3.0, the Device API
accepts any kernel type that meets the Kernel API
interface requirements.  CUTLASS 3.0's Device API code is
parameterized by the kernel type, but this code
is *generic*; the same code works for any kernel type.

The device layer compatibility interface, `device::GemmUniversalAdapter`,
also provides reflective mappings from 3.0-specific types
back to the closest possible 2.x equivalent types. This is [discussed further in the section below](#conversions-between-2x-tags-and-30-types).

CUTLASS 3.0's `device::GemmUniversalAdapter` also exposes some new APIs that the 2.x `device::GemmUniversalAdapter` implementation does not. Most notably, this includes the ability to bypass the `GemmKernel::Arguments` to `GemmKernel::Params` lowering.

```c++
// Primary run() entry point API that is static allowing users to create and manage their own params.
static Status
run(Params& params, cudaStream_t stream = nullptr);
```

This new API is useful for the following scenarios.

* Running again does not require reinvoking `GemmKernel::to_underlying_arguments()`
* Manual control over construction of `GemmKernel::Params` for custom kernels with custom stride types
* Fully static problem shapes and strides for bespoke kernels where no argument mapping needs to take place

## Compatible Kernel API

CUTLASS 3.x API shares the kernel layer API with CUTLASS 2.x
through the single entry point type `cutlass::gemm::kernel::GemmUniversal`.
All kernel layer GEMMs are viewed as a composition of a collective mainloop
and a collective epilogue.

**`kernel::GemmUniversal` implements both 2.x and 3.x APIs**

The entry point for CUTLASS's kernel API is the class
`cutlass::gemm::kernel::GemmUniversal`.
This class' declaration lives in the header file
[include/cutlass/gemm/kernel/gemm_universal.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.hpp).

```c++
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 * SFIANE shims both 2.x and 3.0 API kernels based on ProblemShapeOrThreadblockMma_.
**/
template <
  class ProblemShapeOrThreadblockMma_,
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
```

We call this class "universal" because it can be built
using either the CUTLASS 3.0 or the 2.x mainloops and epilogues.
If `GemmUniversal`'s first template argument
(`ProblemShapeOrThreadblockMma_`) is a `cute::tuple`,
then `GemmUniversal` assumes that
the remaining three template arguments
(the mainloop, epilogue, and grid swizzle)
implement the 3.0 APIs.
Otherwise, `GemmUniversal` assumes that
the remaining three template arguments
implement the 2.x APIs.
All the template arguments must be either
CUTLASS 3.0 or CUTLASS 2.x types. For example,
`GemmUniversal` does not permit using
a 2.x mainloop with a 3.0 collective epilogue.

CUTLASS 3.x implements various embodiments of `kernel::GemmUniversal`.
Each kernel layer schedule is specialized
for a GEMM scheduling algorithm and GPU architecture.
Specializations of `kernel::GemmUniversal` for 3.0 APIs live in
any of various `gemm_*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
The specialization to which to dispatch is decided through the dispatch policy's `Schedule` type.

Specializations for 2.x APIs live in the header file
[include/cutlass/gemm/kernel/gemm_universal.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.h).

### Kernel API design differences

The CUTLASS 2.x Kernel API was more closely tied
to the Device API, as we mentioned above.
In particular, the 2.x Device API specified the grid shape
used to launch the Kernel API.
In CUTLASS 3.0, the Kernel API controls its own grid shape,
while the device adapter simply queries the kernel with which it needs to be launched.

This change is required to support various kernel schedules
that may need their own schedule specific grid planning logic.
For example, persistent kernel schedules generally only launch with
as many threadblocks as the number of multiprocessors on the GPU.

All CUTLASS 3 `kernel::GemmUniversal` specializations expose the following (static) API:

```c++
// Returns true if the kernel can execute the provided GEMM arguments.
static bool
can_implement(Arguments const& args);

// Returns a dim3 representing the threadblock shape.
static dim3
get_block_shape();

// Returns a dim3 representing the grid shape in terms of threadblocks.
static dim3
get_grid_shape(Params const& params);
```

The device adapter simply queries the kernel for these three before launching it on the device.
CUTLASS 3.0 provides a meta-function to detect whether a `cutlass::gemm::kernel::*` implements
the 3.x API or 2.x API:

```c++
// include/cutlass/gemm/gemm.h

namespace cutlass:gemm::detail {

// The following metafunction is used to detect whether a
// `kernel::Gemm` or `kernel::GemmUniversal` implements the CUTLASS 3.x API,
// by checking whether the problem shape type is aliased within.
template <class GemmKernel, class = void>
struct IsCutlass3GemmKernel;

} // namespace cutlass:gemm::detail
```

Users can dispatch their generic code against 2.x and 3.x specializations with
this as a type trait for the kernel API version.

## Threadblock API and Inner Loops

Much of the CUTLASS 3 GEMM hierarchy for mainloops and inner loops diverges
from that of CUTLASS 2.x.  With that also comes the introduction of the
`cutlass::gemm::collective` layer as a direct replacement and a superset
of the 2.x `cutlass::gemm::threadblock` layer. Going forward,
CUTLASS 3.x will discontinue new developments in the following namespaces.

* `cutlass::*::threadblock::*`
* `cutlass::*::warp::*`
* `cutlass::gemm::thread::*`
* `cutlass::arch::*` (except `barrier.h`)

`cutlass::gemm::collective`s are a superset of the threadblock layer where
all new mainloops will be developed. Users should look to the `CollectiveMma` type
if they wish to author custom mainloop code in the 3.x API.

Similarly, for the GEMM inner loops, `cute::MMA_Atom`s replace the
`gemm::warp` and `gemm::thread` layer code. Going forward, all new PTX instructions
and associated metadata development will occur directly inside [`cute/arch/*.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch/) and [`cute/atom/*.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom/).

The desired inner loop MMA iteration order and tiling can be achieved through careful
selection of the atom layout, value layout, and permutations of the `cute::TiledMma`.

For epilogues, the `cutlass::epilogue::collective` layer replaces `cutlass::threadblock::collective`.  However, the thread-level epilogue elementwise operations
in `cutlass::epilogue::thread` will continue to be used in 3.x kernels as well, albeit, with
a more idiomatic epilogue vectorization strategy.
[Example 50](https://github.com/NVIDIA/cutlass/tree/main/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu)
shows how to use 2.x epilogue thread operators with 3.0 API kernels.

## Porting from 2.x to 3.0 API

### CUTLASS 2.x layout tags and CUTLASS 3.0 major modes

CUTLASS 2.x and CUTLASS 3.0 use both
different wording and different types
to describe the permitted layouts
of GEMM's input matrices A and B.

CUTLASS 3.0 does not use the terms "column major"
or "row major" to describe matrix layouts.
Starting with CUTLASS 3.0, adoption of CuTe allows us to decouple

* the coordinate mode order (logical shape) of layouts from

* the index space stride order of the backing storage.

In line with our switch to a conceptual GEMM hierarchy, we view the major modes not from a BLAS-3 perspective.
Rather, we divide the modes into two categories.

* "Inner modes" or "K-modes" are contracted over during the GEMM.
  Therefore, they are not present in the output tensor.

* "Outer modes" or "MN-modes" are preserved in the output.

Now, instead of `RowMajor` or `ColumnMajor`, whose major stride depends on whether we are referring to the
A or the B matrix, we uniformly employ the "K major" or "MN major" terminology and enforce the convention of all tensors having the shape `[M/N, K, L]` regardless of which mode is major.  That is,

* the input matrix A has shape M x K,
* the input matrix B has shape N x K, and
* the input/output matrices C/D have shape M x N.

Note that this convention for B
differs from the BLAS's GEMM interface,
which specifies that B has shape K x N.

CUTLASS 3.0 uses these names of the modes
to specify which mode of a matrix has stride 1.
For the matrix A,

* "M major" means that the matrix is stride 1
  in the M mode, and
* "K major" means that the matrix is stride 1
  in the K mode.

For the matrix B,

* "N major" means that the matrix is stride 1
  in the N mode (which for B is mode 0,
  because the convention is that B is N x K); and
* "K major" means that the matrix is stride 1
  in the K mode (which for B is mode 1).

CUTLASS 2.x defines "layout tag" classes
`cutlass::layout::ColumnMajor` and `cutlass::layout::RowMajor`,
that live in the header file
[`cutlass/layout/matrix.h`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/matrix.h).
The interpretation of these layouts in GEMM
depends on whether they are applied
to the input matrix A or B. For the matrix A, "column major" means
that mode corresponding to M extent has stride 1,
and "row major" means that mode corresponding to K extent has stride 1.
This is the usual computer science definition
of column major and row major for a rank-2 array.
For the matrix B, the opposite holds:
"column major" means that mode corresponding to N extent has stride 1,
and "row major" means that mode corresponding to K extent has stride 1.

Using the convention of `[outer, inner, batch]` mode order for tensor logical shapes
avoids potential confusion with the meaning of column major and row major
changing depending on whether they are applied to A or B.

The table below summarizes our mode order convention and
mapping of 2.x layout tags to corresponding M-major, N-major, or K-major strides.

| Matrix | CUTLASS 2.x layout | 2.x Shape  | Logical major mode| 3.x Shape/Stride  | Major ordinal |
| ---    | ---                | ---        | ---               | ---               | ---           |
| A      | `ColumnMajor`      | M x K      | M major           | M x K x L         | 0 (outer)     |
| A      | `RowMajor`         | M x K      | K major           | M x K x L         | 1 (inner)     |
| B      | `RowMajor`         | K x N      | N major           | N x K x L         | 0 (outer)     |
| B      | `ColumnMajor`      | K x N      | K major           | N x K x L         | 1 (inner)     |
| C      | `ColumnMajor`      | M x N      | M major           | M x N x L         | 0 (outer)     |
| C      | `RowMajor`         | M x N      | N major           | M x N x L         | 1 (inner)     |

Notice that in CUTLASS 3.0, interpretation of layouts no longer changes based on
whether we are talking about the A or B matrix. M and N major inputs always have a
static size-1 stride in their 0th (outer) mode. Similarly, K major inputs
always contain the static size-1 stride in their 1st mode. This uniformity in stride order
allows us to represent tensor layouts much more cleanly and treat both A and B equally in our interfaces.
See for example the following snippet from our [`kernel/sm70_gemm.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm70_gemm.hpp)
for Ampere kernel schedules.

```c++
// Represent the full tensors
Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA); // (m,k,l)
Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); // (n,k,l)

// Get batch slice
Tensor mA_mk = mA_mkl(_,_,get<3>(blk_coord_mnkl)); // (m,k)
Tensor mB_nk = mB_nkl(_,_,get<3>(blk_coord_mnkl)); // (n,k)

// Slice to get the tiles for which this thread block is responsible
Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{}); // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{}); // (BLK_N,BLK_K,k)
```

As seem in this snippet, all input tensors have the logical shape `[outer, inner, batch]`,
and the strides could represent either outer or inner
(or any other complex hierarchical stride) major storage.
CuTe layouts always maintain the logical consistency of the coordinate spaces regardless of the strides.

By convention, in CUTLASS 3.0, we treat the M and N mode as the 0th mode,
and K mode as the 1st mode of the stride.

### Conversions between 2.x tags and 3.0 types

Starting with CUTLASS 3.0, all layouts are described using
`cute::Shape` and `cute::Stride` which compose into a `cute::Layout<Shape, Stride>`.
In CUTLASS 2.x, various layout tags such as `cutlass::layout::RowMajor` are used to specialize
template implementations. These tag types only encode information about the tensor strides,
as 2.x layouts did not incorporate any concept of tensor shape in the layout tags themselves.
Users may find a need to convert between CUTLASS 2.x layout tags, and 3.0
CuTe stride types. CUTLASS 3.0 `gemm::collective::CollectiveBuilder` interfaces
also accept these 2.x layout tags as input parameters in their template API as a convenience for users.
At every entry point into CUTLASS 3.0, these tags get converted to their corresponding CuTe Stride type with
metafunctions that best approximate their corresponding `cute::Stride`.

* `cutlass::gemm::detail::TagToStrideA_t<LayoutTag>`
* `cutlass::gemm::detail::TagToStrideB_t<LayoutTag>`
* `cutlass::gemm::detail::TagToStrideC_t<LayoutTag>`

By convention, and to match user expectations, the `cute::Stride` types that these
map onto always contain one static mode corresponding to the layout tag, and two 64-bit
dynamic stride modes corresponding to the minor mode and the batch mode. Batch
mode is included by default as all CUTLASS 3.0 kernels support packed batch-mode GEMMs
out of the box.

The [`cutlass/gemm/gemm.h#440`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/gemm.h#440)
header file includes functions
that can be useful for converting
from CUTLASS 3.0 `cute::Stride`s back to CUTLASS 2.x layout tags.

* `cutlass::gemm::detail::StrideToLayoutTagA_t<CuteStride>`
* `cutlass::gemm::detail::StrideToLayoutTagB_t<CuteStride>`
* `cutlass::gemm::detail::StrideToLayoutTagC_t<CuteStride>`

These metafunctions take the CuTe Stride as a template parameter and
attempt to find the size-1 stride in the idiomatic M, N, or K modes
to best approximate a corresponding 2.x layout tag type.
Note that this may not work in general for any `cute::Stride`
as the mapping between the stride and tag type is not bijective.

These mapping utilities are kept in a `detail` namespace
as we do not guarantee stability of their implementation.
Their behavior may change in future releases as we add new features.
However, we do expect these type names to remain stable. For users who want
these 2.x reflective types from an assembled kernel with a more stable API,
the specialization of `cutlass::gemm::device::GemmUniversalAdapter`
for CUTLASS 3.0 kernel provides all aliases for all 2.x type aliases
in addition to the layout tags. You can see how they are used in the header file
[`cutlass/gemm/device/gemm_universal_adapter.h`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).
Here is an excerpt.

```c++
  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = cutlass::arch::OpMultiplyAdd;

  // If our TiledMMA's instruction thread layout size is larger than 1,
  // we know it's a tensorop
  using OperatorClass = std::conditional_t<
      (cute::size(typename GemmKernel::TiledMma::AtomThrID{}) > 1),
      cutlass::arch::OpClassTensorOp, cutlass::arch::OpClassSimt>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // We get the instruction shape directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  static int constexpr kStages = CollectiveMainloop::DispatchPolicy::Stages;
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x,
  // but we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M,
  // and the rest along N, with none along K.  We also always round up
  // the warp count to 4 if the tiled mma is smaller than 128 threads.
  static constexpr int WarpsInMma = std::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = gemm::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA>();
  static int constexpr kAlignmentB = gemm::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB>();
```

CUTLASS's library and profiler use these reflective interfaces to
obtain the kernel's configuration parameters. Users can use these to approximate the CUTLASS 2.x types
for 3.0 API kernels.  However, the reflective interfaces cannot always match the types exactly,
as the mappings are not always bijective.

### Copyright

Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
