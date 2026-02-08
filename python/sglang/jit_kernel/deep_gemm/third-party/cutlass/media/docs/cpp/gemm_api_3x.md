![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS GEMM API")

# CUTLASS 3.0 GEMM API

CUTLASS presents a uniform programming model
for matrix multiply-accumulate (MMA) operations
at different levels of the GPU system hierarchy.
CUTLASS 3.0 has GEMM APIs corresponding to the following levels
in order of highest to the lowest level.

1. Device
2. Kernel
3. Collective
4. Tiled MMA and Copy
5. Atom

This document will cover the first three levels in detail:
Device, Kernel, and Collective.
It also briefly discusses the Tiled MMA/Copy and Atom level,
and then refers readers to CuTe's tutorial for more information.

## CUTLASS GEMM Model

CUTLASS implements algorithms that express
the classical "triply nested loop" GEMM algorithm
with a tiled structure mirroring the above hierarchy.

The following pseudocode describes the model for a GEMM kernel
targeting a warp-synchronous matrix multiply instruction like `mma.sync.`
The entire operation is referred to as "Gemm,"
as it is assumed that an epilogue operation
performs the general matrix update similar to BLAS.
This is pseudocode and is only meant to illustrate which parts of the layers
correspond to the inner or outer loops of the GEMM.

```c++
// cutlass::gemm::kernel::GemmUniversal: ClusterTileM and ClusterTileN loops
//   are either rasterized by the hardware or scheduled by the kernel in persistent kernels.
// Parallelism over thread block clusters
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // cutlass::gemm::collective::CollectiveMma: mainloop that iterates over all k-tiles
    // No loop unrolling is performed at this stage
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // loops inside cute::gemm(tiled_mma, a, b, c); Dispatch 5: (V,M,K) x (V,N,K) => (V,M,N)
      // TiledMma uses the hardware instruction provided through its Mma_Atom
      // TiledMma's atom layout, value layout, and permutations define the iteration order
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {

            // TiledMma's vector mode dispatches to the underlying instruction.
            mma.call(d, a, b, c);
          } // tiled_mma_n
        } // tiled_mma_m
      } // tiled_mma_k
    } // k_tile mainloop
  } // cluster_m
} // cluster_n
```

The first three nested `for` loops
correspond to parallelism over thread block clusters.
The code does not actually express them as explicit `for` loops.
Instead, the parallelization scheme over tiles
is implied by CUDA grid launch semantics.
However, for persistent kernels,
these three loops are expressed in the source code 
as a single `while` loop that queries the
[work tile scheduler](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp)
for problem tiles on which to compute.

Inside the three nested `for` loops,
one finds code that pulls matrix tiles
from global memory into more "local" memory
(like shared memory or registers)
and computes MMAs.
These tiled copy and tiled mma iterations are generally
fully static and get fully unrolled.

## CUTLASS GEMM Components

CUTLASS expresses the above loop nest
with the following components which are specialized for
data type, layout, and math instruction.

| API level            | API Class and/or function names                   |
| ---                  | ---                                               |
| Device               | `cutlass::gemm::device::GemmUniversalAdapter`     |
| Kernel               | `cutlass::gemm::kernel::GemmUniversal`            |
| Collective           | `cutlass::gemm::collective::CollectiveMma` <br /> `cutlass::epilogue::collective::DefaultEpilogue` <br /> `cutlass::epilogue::collective::Epilogue`        <br /> |
| Tiled (MMA and Copy) | `cute::TiledMma` and `cute::TiledCopy` <br /> `cute::gemm()` and `cute::copy()` |
| Atom                 | `cute::Mma_Atom` and `cute::Copy_Atom` |

In CUTLASS 3.0, we assemble kernels
by first composing a collective mainloop and collective epilogue
together at the kernel layer,
and then wrapping them with a host-side adapter
to form a GEMM handle to that kernel.

The following sections describe these components
in the order a user should instantiate them
in order to assemble a kernel.  This order is

1. assemble the required collective mainloop and epilogues,

2. compose them together to build a kernel type, and

3. wrap up the kernel with a device layer adapter.

This order is also reflected in the [CUTLASS 3.0 Hopper kernel examples](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) as seen in the excerpt below.

```c++
// Step 1: Generate the required collective layer mainloop specialization
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TilesShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Step 2: Specify the collective layer epilogue type
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    ElementC,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;

// Step 3: Compose the mainloop and epilogue together at the kernel layer
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 4: Wrap up the kernel::GemmUniversal kernel class
// with the device adapter to obtain a host-side handle to the kernel
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Towards the end, we also briefly cover CuTe's tiled mma and copy as well as the atom layer APIs,
before redirecting users to CuTe-specific documentation for further details.

### Collective API

A Collective is "the largest collection of threads
onto which mma atoms and copy atoms are tiled."
That is, it is the largest number of threads in a grid
that can cooperate by leveraging hardware features
for accelerated communication and synchronization.
These hardware features include

* asynchronous array copy
  (e.g., from global memory to shared memory);

* MMA instructions
  for small tiles that live in shared memory;

* synchronization operations for clusters,
  thread blocks, and/or warps; and/or

* hardware acceleration (such as barriers)
  for ensuring that data dependencies
  between asynchronous operations are met.

A Collective uses the `TiledMma` and `TiledCopy` API (see below)
to access operations that copy and perform MMA on tiles.

Different units of parallelism
(e.g., threads, warps, or thread blocks)
in a Collective might have different roles.
For example, in "warp-specialized" algorithms,
some warps may be responsible for copying data,
while others may be responsible for computation.
Nevertheless, the different units of parallelism
still need to share data and coordinate access
to the shared data. For example,
the producer warps in a warp-specialized algorithm
that copy input matrix tiles into shared memory
need to let the consumer MMA warp(s) know
that their MMA inputs are ready.
We contrast this with the `kernel::` layer API,
which schedules the collectives over *independent* tiles in the grid.

The Collective API includes both the "mainloop"
of matrix multiply-accumulate, and the epilogue.
This API is the composition point for optimizations
such as mainloop fusions and epilogue fusions.
It is responsible for implementing
the `k_tile` loop in the above triply nested loop pseudocode.

### Collective Mainloops

The `cutlass::gemm::collective::CollectiveMma` class
is the primary interface to the collective
matrix multiply-accumulate (MMA) mainloops.
"Mainloop" refers to the "main loop" over tiles --
the "cluster tile k" loop in the pseudocode
near the top of this document.
Any looping over multiple tiles that
the algorithm might need to do would happen here.

The `CollectiveMma` class is declared in the header
[cutlass/gemm/collective/collective_mma.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_mma.hpp).

```c++
namespace cutlass::gemm::collective {

template <
  class DispatchPolicy,
  class TileShape,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class TiledMma,
  class GmemTiledCopyA,
  class SmemLayoutAtomA,
  class SmemCopyAtomA,
  class TransformA,
  class GmemTiledCopyB,
  class SmemLayoutAtomB,
  class SmemCopyAtomB,
  class TransformB
>
struct CollectiveMma {
  static_assert(sizeof(ElementA) == 0, "Could not find a mainloop specialization.");
};

} // namespace cutlass::gemm::collective
```

- `DispatchPolicy` is the most important type for a collective, and is
[covered in more detail below](#collective-dispatch-policies).

- `StrideA` and `StrideB` are instances of type `cute::Stride` that represent the global memory layout of A and B tensors. These strides are required to be rank-3, representing the modes `[outer, inner, batch]`. Each of the 3 ranks can be a multi-modal hierarchical stride; this would apply if implementing a tensor contraction.

- `TiledMma` is an instance of `cute::TiledMma`.

- `GmemTiledCopyA` and `GmemTiledCopyB` are instances of `cute::TiledCopy` types. Both tiled operation types are [covered in more detail below](#tiled-mma-and-copy).

- `SmemLayoutAtomA` and `SmemLayoutAtomB` are instances of type `cute::Layout` and represent the smallest
layout that will get tiled over the entire collective's shared memory. This layout does _not_ include the
pipeline mode, and therefore, both are expected to be rank 2 layouts of shape [`outer`, `inner`].

- `SmemCopyAtomA` and `SmemCopyAtomB` are `Copy_Atom`s to be used for moving data from shared memory
into register memory.

Notice that CUTLASS 3.0 mainloops do not accept a dedicated accumulator element type.
We obtain the accumulator type from the `typename TiledMma::ValTypeC`. Note also that
top level API's `ElementA` and `ElementB` can differ from those of the MMA facing
`typename TiledMma::ValTypeA` and `typename TiledMma::ValTypeB`, allowing TMA or user
supplied transform operations to perform type conversions.

### Collective Dispatch Policies

`CollectiveMma` implementations are not generic.
Instead, they must be specialized for each algorithm and GPU architecture.
Users can dispatch to a `CollectiveMma` specialization
by picking template arguments matching that specialization.
CUTLASS 3.0 adopts a tag-based dispatch policy type to specialize
mainloop implementations and add tuning knobs to them.

Below is an example of one of the dispatch policies that is used to dispatch to a Hopper TMA
warp-specialized mainloop implementation:

```c++
// n-buffer in smem (Hopper TMA),
// pipelined with Hopper GMMA and TMA,
// warp-specialized dynamic schedule
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

The `Stages_` template parameter lets the user freely vary the number of pipeline stages,
while the `ClusterShape_` type allows for parameterization over the shape of the threadblock
cluster over which TMA multicast will take place.

The collective dispatch policy is also the primary point of composing various kernel schedules
freely with any mainloop. Each mainloop policy either prescribes a `Schedule` with which
it needs to be run, or exposes a template API that lets the user pick a subset of the following schedules:

```c++
struct KernelCpAsyncWarpSpecialized { };
struct KernelCpAsyncWarpSpecializedPingpong { };
struct KernelCpAsyncWarpSpecializedCooperative { };
struct KernelTma { };
struct KernelTmaWarpSpecialized { };
struct KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedCooperative { };
```

- A single kernel schedule can support multiple mainloop implementations. For example,
`KernelMultistage` can be composed with many different mainloop implementations across GPU
architectures such as `MainloopSm70TwoStage`, `MainloopSm80CpAsyncUnpredicated`, and many more.

- A single mainloop can be composed with multiple
possible kernel schedules. For example, the `MainloopSm90TmaGmmaWarpSpecialized` can be
composed with any of the `KernelTmaWarpSpecialized`, `KernelTmaWarpSpecializedPingpong` or `KernelTmaWarpSpecializedCooperative`
kernel schedules.

As [discussed in the CUTLASS 3.0 design documentation](cutlass_3x_design.md), adopting tag
dispatch policies for our core vocabulary types allows us to maintain a single type name for
all operations that conceptually belong to the same class. This design has the following benefits.

- It *avoids code duplication* in cases where mainloops can be composed with multiple kernels or vice versa.
- It *makes writing generic code easier*, as the primary type name `CollectiveMma` does not change across any implementation.
- It *provides a clear, singular extension point* for users to plug in new, custom mainloops implementations specialized on their own dispatch policies.

### Collective Builder for `CollectiveMma`s

The primary `CollectiveMma` is intended to be an expert user interface that allows full control over
all the properties of the collective's GPU micro-kernel. However, often a user just wants an
off-the-shelf GEMM mainloop implementation parameterized on simple configuration parameters. CUTLASS 3.0
provides [`cutlass::gemm::collective::CollectiveBuilder`](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/collective/collective_builder.hpp) for such scenarios.

```c++
namespace cutlass::gemm::collective {
template <
  class ArchTag,
  class OpClass,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType,
  class Enable = void
>
struct CollectiveBuilder {
  static_assert(sizeof(ElementA) == 0, "Could not build a collective for given parameters.");
};
} // namespace cutlass::gemm::collective
```

`CollectiveBuilder` accepts CUTLASS 2.x equivalent input template arguments, and attempts to build
the best performing `CollectiveMma` from the given parameters.

- `ArchTag` is one of the SM architectures tags from `cutlass::arch::Sm*`.
- `OpClass` is one of the operator class tags from `cutlass::arch::OpClass*`.
- `ElementA` and `ElementB` are the logical value types of the A resp. B tensors.
- `ElementAccumulator` is the accumulator type to be used in the instruction.
- `GmemLayoutA` and `GmemLayoutB` are CUTLASS 2.x layout tags, `layout::RowMajor` or `layout::ColumnMajor`.
- `AlignmentA` and `AlignmentB` are global memory alignments of A and B tensors in terms of element count.
- `TileShape_MNK` is an instance of `cute::Shape` that is rank-3, representing the MxNxK collective tile shape.
- `ClusterShape_MNK` is an instance of `cute::Shape` that is rank-3, representing the MxNxK threadblock cluster tile shape.
- `StageCountType` is either `collective::StageCountAuto` or an instance of `collective::StageCount<N>`.
- `KernelScheduleType` is either `collective::KernelScheduleAuto` or one of the specific kernel schedule tags discussed in the [dispatch policy section](#collective-dispatch-policies) above.

`StageCountAuto` allows the collective builder to compute the size of a single stage's size in shared memory
and maximize the shared memory usage assuming 1 threadblock / multiprocessor occupancy.

`KernelScheduleAuto` allows the collective builder to pick the best kernel schedule available for the
given set of parameters, or let's the user override this with a specific kernel schedule type.

Note that collective builders are still in beta, and their functionality
does not map onto the full design space that the primary expert `CollectiveMma` API
allows for. We expect their supported mainloop types to expand in future releases, but 
with 3.0, only SM90 tensorop kernels are supported through the builder API. The builder API
may also change in the future as we adopt user feedback.

If the builder is able to provide a collective mainloop type for the given set of parameters,
it will be aliased within as `CollectiveOp`. For more information on how to
parameterize kernels conveniently with the collective builder, please see example [49_hopper_gemm_with_collective_builder](https://github.com/NVIDIA/cutlass/tree/main/examples/49_hopper_gemm_with_collective_builder).

### Epilogue

The collective epilogue implements element-wise operations
involving the output matrix.  Users can provide a custom
epilogue, or use one of the standard epilogues.
These live in the directory
[include/cutlass/epilogue/collective/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/epilogue/collective/),
and include classes like
`cutlass::epilogue::collective::DefaultEpilogue`
and
`cutlass::epilogue::collective::Epilogue`.
CUTLASS's provided collective epilogues
do not live under `include/cutlass/gemm`
or in the `cutlass::gemm` namespace,
because they can be used for computations
other than GEMM.

## Kernel API

The kernel is "a collection of all clusters in the grid."
The kernel layer schedules have four main responsibilities.

- Ordering the execution of collectives within the kernel, performing any synchronization between that may be necessary
- Marshalling the threads of a warp specialized schedules into their respective roles
- Performing any necessary grid swizzling logic
- Tiling the input tensors with the threadblock cluster value tile before invoking the collectives on them

The Kernel API is the entry point for a grid of thread blocks
that may or may not be organized in a cluster.
It is the composition point for fusing back-to-back GEMMs,
epilogues, and/or other operations.

The entry point API for CUTLASS 3.0 kernel is the class
`cutlass::gemm::kernel::GemmUniversal`, found in the header file
[include/cutlass/gemm/kernel/gemm_universal.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.hpp).
`GemmUniversal` is a stateless universal device kernel
that implements GEMM as the composition of two parts:

* a collective mainloop, and
* a collective epilogue

```cpp
namespace cutlass::gemm::kernel {
/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
**/
template <
  class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
  class CollectiveMainloopOrEpilogue_,
  class CollectiveEpilogueOrThreadblockSwizzle_,
  class TileScheduler_ = void,
  class Enable = void
>
class GemmUniversal;
} // namespace cutlass::gemm::kernel
```

*Stateless* means that the caller --
for example, the Device API described above --
manages the kernel's state.
The kernel just takes input and output parameters (`Params`).

*Universal* means that `GemmUniversal` works
for both CUTLASS 3.0 and 2.x interfaces
and across a broad range of kernel schedules.
If `GemmUniversal`'s first template argument is a `cute::Shape`,
then `GemmUniversal` assumes that the remaining template arguments
implement the 3.0 APIs.  Otherwise, `GemmUniversal` assumes that
the remaining template arguments implement the 2.x APIs.
Starting with CUTLASS 3.0, the problem shape has been promoted
to a top-level template API for the GEMM kernel.
This supports fully static GEMM instantiations
where the user expects to know some or all
of the problem shapes at compile time
in order to extract even more performance.

The *collective mainloop* implements MMA on local tiles.
The *collective epilogue* addresses any operations after the MMA,
such as applying the `beta * C` part of `C := beta * C + alpha * A * B`.
We will explain *collective* in more detail below.

Specializations of `kernel::GemmUniversal` for 3.0 APIs live in 
any of various `gemm_*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
Specializations for 2.x APIs can be found in the header file
[include/cutlass/gemm/kernel/gemm_universal.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/gemm_universal.h).

CUTLASS 3.x implements various embodiments of `kernel::GemmUniversal`.
Each kernel layer schedule is specialized
for a GEMM scheduling algorithm and GPU architecture.
Specializations of `kernel::GemmUniversal` for 3.0 APIs live in 
any of various `include/cutlass/gemm/kernel/{arch_tag}*.hpp` files in the directory
[include/cutlass/gemm/kernel/](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/).
Which specialization to dispatch to is decided through the dispatch policy's `Schedule` type.

For example, the header file
[include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)
has a specialization of `kernel::GemmUniversal` for Hopper
that uses a warp-specialized mainloop with a persistent scheduling algorithm,
while the header file
[include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp)
has a specialization of `GemmUniversal` for Hopper
that uses a warp-specialized but non-persistent algorithm.

To support composition between supported kernel schedules and mainloop dispatch policies without having to
duplicate collective mainloop implementations, GEMM kernel layer schedules can be composed with
any mainloop that specifies their corresponding kernel schedule as their `Schedule` type in the policy.
This is discussed in detail in the [collective dispatch policy section](#collective-dispatch-policies) above.

```c++
// An example of the SM90 KernelMultistage kernel's
// specialization logic that allows it to be composed
// with many mainloops such as `MainloopSm80CpAsync`
// and `MainloopSm70TwoStage`.
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  std::enable_if_t<std::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```

## Device API

The Device API is a universal, kernel-agnostic host interface
for kernel launch and managing the lifetime of 
reusable host-side parameters.

This API is how users' host-side .cu code
invokes CUTLASS's single-GPU GEMM kernels.
It serves the same purpose as cuBLAS and behaves similarly.

The entry point for the Device GEMM API is the class
`cutlass::gemm::device::GemmUniversalAdapter`.
This class lives in the header file
[include/cutlass/gemm/device/gemm_universal_adapter.h](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/device/gemm_universal_adapter.h).
`GemmUniversalAdapter` is a stateful, reusable handle,
which is parameterized on the `cutlass::gemm::kernel` type.

```c++
/*! 
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::*

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behavior might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;
```

*Stateful* means that the handle instance contains state
that the kernel needs to run.
This means that the user must initialize the handle first,
then use the initialized handle instance to run the kernel.
Statefulness also means that the handle can manage the lifetime
of the kernel's `Params` -- the parameters of the kernel itself.
An important duty of `GemmUniversalAdapter`
is to map from the user's `Arguments` --
what the user sees as the kernel's parameters --
to the `Params` that the kernel actually sees.
For power users, the class exposes new static methods
in 3.0 APIs that can bypass stateful methods
or go directly to `Params` without intermediate `Arguments`.

*Reusable* means that the handle instance can be used
to call the kernel multiple times with different arguments
(e.g., different matrices).
Reusing the handle may be more efficient than just
creating a new handle for each kernel invocation.

*Parameterized on the kernel type* means that
the `GemmUniversalAdapter` class' behavior
depends on the GEMM kernel type (see the next section).
Specifically, `GemmUniversalAdapter` has a template parameter
`GemmKernel`, which is the GEMM kernel type.
Valid template arguments for `GemmKernel` are

* `cutlass::gemm::kernel::GemmUniversal`,
  implementing CUTLASS 3.x API kernels;
* `cutlass::gemm::kernel::GemmUniversal`,
  implementing CUTLASS 2.x API kernels; or
* Any valid CUTLASS 2.x `kernel` layer GEMM that
  was previously composable with the `device::GemmUniversalAdapter`.

`GemmUniversalAdapter` presents a single
host-side interface to both 3.0 and 2.x kernels.
CUTLASS accomplishes this by
specializing `GemmUniversalAdapter`'s implementation
on either the 2.x API implementing kernel layer GEMMs, or on the 3.x API
implementing kernel layer GEMMs. The metafunction [`cutlass::gemm::detail::IsCutlass3GemmKernel`](cutlass_3x_backwards_compatibility.md#kernel-api-design-differences)
is what `GemmUniversalAdapter` uses to distinguish between 2.x and 3.x kernels.

`GemmUniversalAdapter` sets up and launches the kernel, using the 
CUDA extended launch API for threadblock cluster support if required.
Note, `GemmUniversalAdapter` does *not* specify the grid shape.
The kernel controls the grid shape
and other kernel-specific launch parameters.
This makes it possible for all 3.0 kernels
to use the same kernel launch code,
thus factoring out kernel launch from the actual kernel.

## Tiled MMA and Copy

The Tiled MMA or Copy are tilings of MMA atoms resp. Copy atoms
across threads and data, with possible permutations applied to the 
resulting tiling. This layer is most analogous to the warp level
tiling of MMA instructions in CUTLASS 2.x. However, it views the tiling
from the perspective of all threads participating in the operation
and generalizes the concept to copy operations as well. The purpose
of this layer is to build composable GPU micro-kernels out of a plethora
of hardware accelerated math and data movement operations, each with their
unit layouts in threads and data. The tiled MMA and Copy types present
all these various hardware accelerated CuTe Atoms with a single, consistent
API.

The resulting tiled operation acts as a single MMA or copy operation
that users can invoke in the "inner" loop
of the three-nested-loops pseudocode
at the top of this document using `cute::gemm()` or `cute::copy()`.

We call this API "tiled" because it constructs
larger operations out of the Atoms provided by CuTe,
as if fitting together individual tiles
to build a reusable component of a mosaic.
For example, CuTe might provide an MMA Atom
that users can call on a single warp,
for fixed M, N, and K dimensions.
CUTLASS can then use CuTe operations like `make_tiled_mma`
to turn this Atom into an operation
that works on an entire thread block,
for larger M, N, and K dimensions.

## Atom API

An "Atom" is the smallest collection of threads and data
that must participate in the execution of a hardware-accelerated
math or copy operation.

An Atom is "atomic" (indivisible) not in the sense of
concurrent memory operations like `atomicAdd`
(which are "indivisible in time (causality)"),
but in the sense of indivisibility in "space" --
the number of values and the groups of parallel workers
that must participate in the operation together.

An Atom uses CuTe Layouts to express the required
dimensions and strides of its input and output arrays.
Generally these are fixed at compile time.

The Atom API wraps calls to actual hardware instructions
that accelerate MMA or copy operations.
Users can ask for GPU architecture-specific implementations,
or just pick generic implementations and rely on
whatever GPU architectures were enabled.

For more information about Atoms,
please refer to CuTe's tutorial, e.g., the sections on

* [algorithms](./cute/04_algorithms.md) like `gemm` and `copy`,

* [MMA Atoms](./cute/0t_mma_atom.md#cute-mma-atoms), and

* [a GEMM example](./cute/0x_gemm_tutorial.md).

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
