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
  \brief Visitor tree operation base implementation to enable composable fusions
         for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/detail/helper_macros.hpp"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using cute::tuple;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partitioning Helpers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
  class CtaTileMN,
  class EpilogueTile,
  class TiledCopy
>
CUTLASS_HOST_DEVICE
constexpr auto
sm90_partition_for_epilogue(
    CtaTileMN cT,          // (CTA_M,CTA_N,...)
    EpilogueTile epi_tile, // (EPI_TILE_M,EPI_TILE_N)
    TiledCopy tiled_copy,
    int thread_idx) {
  ThrCopy thread_copy = tiled_copy.get_thread_slice(thread_idx);
  Tensor cT_epi = flat_divide(cT, epi_tile);                                 // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N,...)
  if constexpr (ReferenceSrc) {
    return thread_copy.partition_S(cT_epi);                                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,...)
  }
  else {
    return thread_copy.partition_D(cT_epi);                                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,...)
  }
}

template <
  bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
  class Engine, class LayoutMNL,
  class TileShapeMNK,
  class TileCoordMNKL,
  class EpilogueTile,
  class TiledCopy
>
CUTLASS_HOST_DEVICE
constexpr auto
sm90_partition_for_epilogue(
    Tensor<Engine, LayoutMNL> mT,  // (M,N,L)
    TileShapeMNK tile_shape_mnk,   // (CTA_M,CTA_N,CTA_K)
    TileCoordMNKL tile_coord_mnkl, // (m,n,k,l)
    EpilogueTile epi_tile,         // (EPI_TILE_M,EPI_TILE_N)
    TiledCopy tiled_copy,
    int thread_idx) {
  auto [m, n, k, l] = tile_coord_mnkl;
  auto coord_shape =
      make_coord(m, n, l)
    ;
  Tensor cT = local_tile(mT, take<0,2>(tile_shape_mnk), coord_shape);                                  // (CTA_M,CTA_N)
  Tensor tCcT =
    sm90_partition_for_epilogue<ReferenceSrc>(cT, epi_tile, tiled_copy, thread_idx);   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

  return tCcT;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Visitor Implementation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Producer load callbacks, called by the epilogue load warp.
// Operations usually only define this if TMA load is needed. Most operations will reuse this empy implementation
// Load callbacks are responsible for issuing corresponding mbarrier expect-tx ops for any TMA loads issued, but
// are not responsible for issuing the producer_commit barrier arrival, which is issued by the collective instead
// If this is non-empty, is_producer_load_needed must be true.
//
template <class CallbacksTuple>
struct ProducerLoadCallbacksImpl {
  // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
  CallbacksTuple callbacks_tuple;

  // Before entry of the subtile load loop
  CUTLASS_DEVICE void
  begin() {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.begin();
      }
    );
  }

  // Entry of the subtile load loop. Aux loads usually performed here
  // Upon entry the producer acquire of the current subtile lock has completed.
  // Upon exit all TMA loads for this subtile must have been issued, with corresponding expect-tx operations
  CUTLASS_DEVICE void
  step(uint64_t* full_mbarrier_ptr, int epi_m, int epi_n, int load_iteration, bool issue_tma_load) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.step(full_mbarrier_ptr, epi_m, epi_n, load_iteration, issue_tma_load);
      }
    );
  }

  // Exit of the subtile load loop.
  CUTLASS_DEVICE void
  end() {
    for_each(callbacks_tuple,
      [] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.end();
      }
    );
  }
};


//
// Consumer store callbacks, called by the epilogue store warps.
// All operations must redefine this, with optional inheritance from this empty implementation.
//
template <class CallbacksTuple>
struct ConsumerStoreCallbacksImpl {
  // Callbacks can store non-persistent variables (e.g. tensors) or copies of persistent variables
  CallbacksTuple callbacks_tuple;

  // Before entry of subtile store loop. Gmem broadcasts usually performed here.
  CUTLASS_DEVICE void
  begin() {
    for_each(callbacks_tuple,
      [] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.begin();
      }
    );
  }

  // Is a thread sync needed after begin(). Allows chaining async copies across multiple nodes
  CUTLASS_DEVICE bool
  begin_sync_needed() const {
    return cute::apply(callbacks_tuple,
      [] (auto const&... callbacks) {
        return (false || ... || callbacks.begin_sync_needed());
      }
    );
  }

  // Start of subtile store iteration
  CUTLASS_DEVICE void
  begin_loop(int epi_m, int epi_n) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.begin_loop(epi_m, epi_n);
      }
    );
  }

  // Before visit callback. Smem broadcasts usually performed here.
  // Upon entry, all producer loads for this subtile are completed and visible.
  CUTLASS_DEVICE void
  previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.previsit(epi_m, epi_n, load_iteration, is_producer_load_needed);
      }
    );
  }

  // Perform the fused elementwise computation
  template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
  CUTLASS_DEVICE auto // returns an Array
  visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
        Array<ElementInputs, FragmentSize> const&... frg_inputs) // depends on the N-naryness of the op
    = delete; // Must be implemented for each operation

  // After visit call. Smem reductions usually performed here
  // reduction_buffer is an arbitrary smem tensor that can be used for workspace
  // It is each nodes reponsibility to assert that this buffer is sufficiently sized
  // and to ensure that this buffer is no longer needed upon callback exit
  // i.e. results are synchronized and no longer in the reduction buffer
  //
  // visit_results is a rmem tensor that contains the results of visit() for an entire
  // on the current epilogue subtile
  template <class STensor, class SyncFn, class VTensor>
  CUTLASS_DEVICE void
  reduce(STensor&& reduction_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.reduce(reduction_buffer, sync_fn, epi_m, epi_n, is_last_iteration, visit_results);
      }
    );
  }

  // After reduce call, before smem async fence. Smem stores usually performed here.
  // Upon exit, all smem stores for TMA must have been issued
  CUTLASS_DEVICE void
  postreduce(int epi_m, int epi_n, int store_iteration, bool issue_smem_store) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.postreduce(epi_m, epi_n, store_iteration, issue_smem_store);
      }
    );
  }

  // After smem async fence, before TMA store commit. Aux stores usually performed here
  // Upon exit, all TMA stores for this subtile must have been issued
  // Because of the TMA store delay optimization, this entry point must ONLY be used for TMA stores
  // other gmem stores can be placed in the reduce or postreduce entry points
  CUTLASS_DEVICE void
  tma_store(int epi_m, int epi_n, int store_iteration, bool issue_tma_store) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.tma_store(epi_m, epi_n, store_iteration, issue_tma_store);
      }
    );
  }

  // End of subtile store iteration
  CUTLASS_DEVICE void
  end_loop(int epi_m, int epi_n) {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.end_loop(epi_m, epi_n);
      }
    );
  }

  // Exit of subtile store loop. Gmem reductions usually performed here.
  CUTLASS_DEVICE void
  end() {
    for_each(callbacks_tuple,
      [&] (auto& callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        callbacks.end();
      }
    );
  }
};

template<
  class ProblemShapeMNKL,
  class TileShapeMNK,
  class TileCoordMNKL,
  class TiledMma,
  class EpilogueTile
>
struct ProducerLoadArgs {
  ProblemShapeMNKL problem_shape_mnkl;
  TileShapeMNK tile_shape_mnk;
  TileCoordMNKL tile_coord_mnkl;
  TiledMma tiled_mma;
  EpilogueTile epi_tile;
  int thread_idx;

  CUTLASS_DEVICE
  ProducerLoadArgs(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_mnk,
      TileCoordMNKL tile_coord_mnkl,
      TiledMma tiled_mma,
      EpilogueTile epi_tile,
      int thread_idx)
  : problem_shape_mnkl(problem_shape_mnkl),
    tile_shape_mnk(tile_shape_mnk),
    tile_coord_mnkl(tile_coord_mnkl),
    tiled_mma(tiled_mma),
    epi_tile(epi_tile),
    thread_idx(thread_idx) {}
};

template<
  class ProblemShapeMNKL,
  class TileShapeMNK,
  class TileCoordMNKL,
  class TiledMma,
  class EpilogueTile,
  class TiledCopy,
  class CoordTensor,
  class Residue,
  class ThrCoordTensor,
  class ThrResidue,
  class ThrSrcTensor
>
struct ConsumerStoreArgs {
  ProblemShapeMNKL problem_shape_mnkl;
  TileShapeMNK tile_shape_mnk;
  TileCoordMNKL tile_coord_mnkl;
  TiledMma tiled_mma;
  EpilogueTile epi_tile;
  TiledCopy tiled_copy;
  CoordTensor cD;
  Residue residue_cD;
  ThrCoordTensor tCcD;
  ThrResidue residue_tCcD;
  ThrSrcTensor & tCrC;
  int thread_idx;

  CUTLASS_DEVICE
  ConsumerStoreArgs(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_mnk,
      TileCoordMNKL tile_coord_mnkl,
      TiledMma tiled_mma,
      EpilogueTile epi_tile,
      TiledCopy tiled_copy,
      CoordTensor cD,
      Residue residue_cD,
      ThrCoordTensor tCcD,
      ThrResidue residue_tCcD,
      ThrSrcTensor & tCrC,
      int thread_idx)
  : problem_shape_mnkl(problem_shape_mnkl),
    tile_shape_mnk(tile_shape_mnk),
    tile_coord_mnkl(tile_coord_mnkl),
    tiled_mma(tiled_mma),
    epi_tile(epi_tile),
    tiled_copy(tiled_copy),
    cD(cD),
    residue_cD(residue_cD),
    tCcD(tCcD),
    residue_tCcD(residue_tCcD),
    tCrC(tCrC),
    thread_idx(thread_idx) {}
};

template <class... Ops>
struct Sm90VisitorImplBase {
  // Shared memory allocation
  using SharedStorage = tuple<typename Ops::SharedStorage...>;
  // Host side fusion arguments
  using Arguments = tuple<typename Ops::Arguments...>;
  // Device side fusion params (Kernel-entry API)
  using Params = tuple<typename Ops::Params...>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
    return transform_apply(tuple<Ops...>{}, args,
      [&] (auto&& op, auto const& op_args) CUTLASS_LAMBDA_FUNC_INLINE {
        using Op = cute::remove_cvref_t<decltype(op)>;
        auto ret = Op::to_underlying_arguments(problem_shape, op_args, op_workspace);
        if (op_workspace != nullptr) {
          size_t op_workspace_size = Op::get_workspace_size(problem_shape, op_args);
          op_workspace += round_nearest(op_workspace_size, MinWorkspaceAlignment);
        }
        return ret;
      },
      [] (auto&&... op_params) CUTLASS_LAMBDA_FUNC_INLINE { return cute::make_tuple(op_params...); }
    );
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return transform_apply(tuple<Ops...>{}, args,
      [&] (auto&& op, auto const& op_args) CUTLASS_LAMBDA_FUNC_INLINE {
        using Op = cute::remove_cvref_t<decltype(op)>;
        return Op::can_implement(problem_shape, op_args);
      },
      [&] (auto&&... implementable) CUTLASS_LAMBDA_FUNC_INLINE {
        return (true && ... && implementable);
      }
    );
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return transform_apply(tuple<Ops...>{}, args,
      [&] (auto&& op, auto const& op_args) CUTLASS_LAMBDA_FUNC_INLINE {
        using Op = cute::remove_cvref_t<decltype(op)>;
        size_t op_workspace_size = Op::get_workspace_size(problem_shape, op_args);
        return round_nearest(op_workspace_size, MinWorkspaceAlignment);
      },
      [&] (auto&&... op_workspace_size) CUTLASS_LAMBDA_FUNC_INLINE {
        return (0 + ... + op_workspace_size);
      }
    );
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* op_workspace = reinterpret_cast<uint8_t*>(workspace);
    return transform_apply(tuple<Ops...>{}, args,
      // Initialize each operation's workspace, stopping at the first error
      [&] (auto&& op, auto const& op_args) CUTLASS_LAMBDA_FUNC_INLINE {
        if (status != Status::kSuccess) {
          return status;
        }

        using Op = cute::remove_cvref_t<decltype(op)>;
        status = Op::initialize_workspace(problem_shape, op_args, op_workspace, stream, cuda_adapter);
        if (op_workspace != nullptr) {
          size_t op_workspace_size = Op::get_workspace_size(problem_shape, op_args);
          op_workspace += round_nearest(op_workspace_size, MinWorkspaceAlignment);
        }
        return status;
      },
      // Return the final status
      [&] (auto const&...ops) CUTLASS_LAMBDA_FUNC_INLINE { return status; }
    );
  }

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops(transform_apply(tuple<Ops...>{}, params, shared_storage,
        [] (auto&& op, auto const& op_params, auto&& op_storage) CUTLASS_LAMBDA_FUNC_INLINE {
          using Op = cute::remove_cvref_t<decltype(op)>;
          return Op(op_params, op_storage);
        },
        [] (auto&&... ops) CUTLASS_LAMBDA_FUNC_INLINE { return cute::make_tuple(ops...); }
      )) {}

  // Ops can store kernel persistent variables (e.g. descriptors, scalars, wave counters)
  tuple<Ops...> ops;
};

template <class... Ops>
struct Sm90VisitorImpl : Sm90VisitorImplBase<Ops...> {

  using Impl = Sm90VisitorImplBase<Ops...>;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  CUTLASS_HOST_DEVICE
  Sm90VisitorImpl() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImpl(Params const& params, SharedStorage const& shared_storage)
    : Impl(params, shared_storage) {}

  using Impl::ops;

  //
  // Queries for kernel runtime
  //

  // Is a specialized warp for producer TMA loads needed
  // e.g. Aux tensor loads, broadcasts using TMA bulk copy
  // This condition cannot change between work tiles because it is used
  // to determine whether the load warp should exit early or not
  // e.g. for batched beta this must always be true regardless of current batch idx
  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return cute::apply(ops,
      [] (auto const&... op) CUTLASS_LAMBDA_FUNC_INLINE {
        return (false || ... || op.is_producer_load_needed());
      }
    );
  }

  // Is a producer TMA load specifically for C needed
  // If this is true then is_producer_load_needed must also be true
  // This condition can change between work tiles because it is only used
  // to determine whether the TMA and smem loads for C of a given tile should happen
  // e.g. for batched beta this can be false depending on current batch idx
  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return cute::apply(ops,
      [] (auto const&... op) CUTLASS_LAMBDA_FUNC_INLINE {
        return (false || ... || op.is_C_load_needed());
      }
    );
  }

  // Producer load callbacks factory
  // All operations must redefine this, but most can just dispatch to the base impl
  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return transform_apply(ops,
      [&] (auto& op) CUTLASS_LAMBDA_FUNC_INLINE {
        return op.get_producer_load_callbacks(args);
      },
      [] (auto&&... callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return ProducerLoadCallbacksImpl<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }

  // Consumer store callbacks factory
  // All operations must redefine this
  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    return transform_apply(ops,
      [&] (auto& op) CUTLASS_LAMBDA_FUNC_INLINE {
        return op.template get_consumer_store_callbacks<ReferenceSrc>(args);
      },
      [] (auto&&... callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
        auto callbacks_tuple = cute::make_tuple(callbacks...);
        return ConsumerStoreCallbacksImpl<decltype(callbacks_tuple)>{callbacks_tuple};
      }
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Convenience aliases
using EmptyProducerLoadCallbacks = ProducerLoadCallbacksImpl<cute::tuple<>>;
using EmptyConsumerStoreCallbacks = ConsumerStoreCallbacksImpl<cute::tuple<>>;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tree visitor
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class NodeOp, class... ChildOps>
struct Sm90TreeVisitor : Sm90VisitorImpl<ChildOps..., NodeOp> {

  using Impl = Sm90VisitorImpl<ChildOps..., NodeOp>;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor() {}

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor(
      Params const& params,
      SharedStorage const& shared_storage)
    : Impl(params, shared_storage) {}

  template<class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      constexpr int Rm1 = sizeof...(ChildOps);
      return cute::detail::tapply(callbacks_tuple,
        [&] (auto& child_callbacks) CUTLASS_LAMBDA_FUNC_INLINE {
          return child_callbacks.visit(frg_acc, epi_v, epi_m, epi_n); // child ops must be nullary (e.g. loads, trees)
        },
        [&] (auto&&... frg_inputs) CUTLASS_LAMBDA_FUNC_INLINE {
          return get<Rm1>(callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
        },
        make_seq<Rm1>{} // restrict the transform to R-1 child ops, apply is for node op
      );
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto callbacks_impl = Sm90VisitorImpl<ChildOps..., NodeOp>::
      template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(callbacks_impl)>(cute::move(callbacks_impl));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// DAG visitors
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Most DAG fusions can be represented as a set of output trees with a common input tree
// The common input is first evaluated, then the result is passed as the acc fragment to the output trees
template <class InputTree, class OutputTree, class... AuxOutTrees>
struct Sm90SplitTreeVisitor : Sm90VisitorImpl<InputTree, AuxOutTrees..., OutputTree> {

  using Sm90VisitorImpl<InputTree, AuxOutTrees..., OutputTree>::Sm90VisitorImpl;

  template<class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array frg_input = get<0>(callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n);

      constexpr int Rm2 = sizeof...(AuxOutTrees);
      cute::for_each(make_seq<Rm2>{}, // restrict the sequence to aux out trees
        [&] (auto I) CUTLASS_LAMBDA_FUNC_INLINE {
          get<I+1>(callbacks_tuple).visit(frg_input, epi_v, epi_m, epi_n);
        }
      );

      return get<Rm2+1>(callbacks_tuple).visit(frg_input, epi_v, epi_m, epi_n);
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto callbacks_impl = Sm90VisitorImpl<InputTree, AuxOutTrees..., OutputTree>::
      template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(callbacks_impl)>(cute::move(callbacks_impl));
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  // deducing the output type for all the nodes is tricky so we just convert them all to a common type
  // if multiple compute types are needed then split into multiple subgraphs grouped by type
  class ElementCompute,
  class EdgeTuple, // tuple of int_sequence, each sequence is the children indices (indexed by topological order) for each node
  class... Ops     // in topological order, last op is the output. EdgeTuple must match this order
>
struct Sm90TopologicalVisitor : Sm90VisitorImpl<Ops...> {
  static_assert(is_static_v<EdgeTuple>);
  static_assert(cute::rank(EdgeTuple{}) == sizeof...(Ops));
  static_assert(sizeof...(Ops) > 1);

  using Sm90VisitorImpl<Ops...>::Sm90VisitorImpl;

  template<class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(CallbacksImpl&& impl)
      : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      constexpr int Rm1 = sizeof...(Ops) - 1;
      auto frg_compute_tuple = cute::repeat<Rm1>(Array<ElementCompute, FragmentSize>{});

      return cute::detail::tapply(EdgeTuple{}, callbacks_tuple, frg_compute_tuple,
        // Visit the first R-1 ops in topological order
        [&] (auto&& edge_seq, auto& callbacks, auto& frg_compute) CUTLASS_LAMBDA_FUNC_INLINE {
          frg_compute = cute::detail::apply(frg_compute_tuple,
            // Compute the current op with children inputs
            [&] (auto const&... frg_inputs) CUTLASS_LAMBDA_FUNC_INLINE {
              auto frg_output = callbacks.visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
              using ElementOutput = typename decltype(frg_output)::Element;
              using ConvertOutput = NumericArrayConverter<ElementCompute, ElementOutput, FragmentSize>;
              ConvertOutput convert_output{};

              return convert_output(frg_output);
            },
            // Get inputs in the sequence given by the children indices of the current op
            edge_seq
          );
          return frg_compute; // unused
        },
        // Visit the last op
        [&] (auto const&...ops) CUTLASS_LAMBDA_FUNC_INLINE {
          return cute::detail::apply(frg_compute_tuple,
            // Compute the last op with children inputs
            [&] (auto const&... frg_inputs) CUTLASS_LAMBDA_FUNC_INLINE {
              return get<Rm1>(callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n, frg_inputs...);
            },
            // Get inputs in the sequence given by the children indices of the last op
            get<Rm1>(EdgeTuple{})
          );
        },
        // Transform to visit R-1 ops, apply to visit last op
        make_seq<Rm1>{}
      );
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto callbacks_impl = Sm90VisitorImpl<Ops...>::
      template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(callbacks_impl)>(cute::move(callbacks_impl));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Base specializations so we can have standard layout params and simple aggregate initializers
namespace detail {

template <class Op0>
struct Sm90VisitorImplBase<Op0> {

  // Retain tuple for SharedStorage because empty structs have 1B alignment
  // tuples use multiple inheritance, avoids this problem
  using SharedStorage = tuple<
    typename Op0::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
  };

  struct Params {
    typename Op0::Params op_0;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, workspace)
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Op0::can_implement(problem_shape, args.op_0);
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage))
      }) {}

  tuple<Op0> ops;
};

template <class Op0, class Op1>
struct Sm90VisitorImplBase<Op0, Op1> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    size_t op_0_workspace_size = Op0::get_workspace_size(problem_shape, args.op_0);
    uint8_t* op_0_workspace = reinterpret_cast<uint8_t*>(workspace);
    uint8_t* op_1_workspace = op_0_workspace + op_0_workspace_size;
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, op_0_workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, op_1_workspace)
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Op0::can_implement(problem_shape, args.op_0) && 
           Op1::can_implement(problem_shape, args.op_1);
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage))
      }) {}

  tuple<Op0, Op1> ops;
};

template <class Op0, class Op1, class Op2>
struct Sm90VisitorImplBase<Op0, Op1, Op2> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage,
    typename Op2::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
    typename Op2::Arguments op_2;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
    typename Op2::Params op_2;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    size_t op_0_workspace_size = Op0::get_workspace_size(problem_shape, args.op_0);
    size_t op_1_workspace_size = Op1::get_workspace_size(problem_shape, args.op_1);
    uint8_t* op_0_workspace = reinterpret_cast<uint8_t*>(workspace);
    uint8_t* op_1_workspace = op_0_workspace + op_0_workspace_size;
    uint8_t* op_2_workspace = op_1_workspace + op_1_workspace_size;
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, op_0_workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, op_1_workspace),
      Op2::to_underlying_arguments(problem_shape, args.op_2, op_2_workspace)
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Op0::can_implement(problem_shape, args.op_0) && 
           Op1::can_implement(problem_shape, args.op_1) &&
           Op2::can_implement(problem_shape, args.op_2);          
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op2::initialize_workspace(problem_shape, args.op_2, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage)),
        Op2(params.op_2, get<2>(shared_storage))
      }) {}

  tuple<Op0, Op1, Op2> ops;
};

template <class Op0, class Op1, class Op2, class Op3>
struct Sm90VisitorImplBase<Op0, Op1, Op2, Op3> {

  using SharedStorage = tuple<
    typename Op0::SharedStorage,
    typename Op1::SharedStorage,
    typename Op2::SharedStorage,
    typename Op3::SharedStorage
  >;

  struct Arguments {
    typename Op0::Arguments op_0;
    typename Op1::Arguments op_1;
    typename Op2::Arguments op_2;
    typename Op3::Arguments op_3;
  };

  struct Params {
    typename Op0::Params op_0;
    typename Op1::Params op_1;
    typename Op2::Params op_2;
    typename Op3::Params op_3;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    size_t op_0_workspace_size = Op0::get_workspace_size(problem_shape, args.op_0);
    size_t op_1_workspace_size = Op1::get_workspace_size(problem_shape, args.op_1);
    size_t op_2_workspace_size = Op2::get_workspace_size(problem_shape, args.op_2);
    uint8_t* op_0_workspace = reinterpret_cast<uint8_t*>(workspace);
    uint8_t* op_1_workspace = op_0_workspace + op_0_workspace_size;
    uint8_t* op_2_workspace = op_1_workspace + op_1_workspace_size;
    uint8_t* op_3_workspace = op_2_workspace + op_2_workspace_size;
    return Params{
      Op0::to_underlying_arguments(problem_shape, args.op_0, op_0_workspace),
      Op1::to_underlying_arguments(problem_shape, args.op_1, op_1_workspace),
      Op2::to_underlying_arguments(problem_shape, args.op_2, op_2_workspace),
      Op3::to_underlying_arguments(problem_shape, args.op_3, op_3_workspace)
    };
  }
  
  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Op0::can_implement(problem_shape, args.op_0) && 
           Op1::can_implement(problem_shape, args.op_1) &&
           Op2::can_implement(problem_shape, args.op_2) &&
           Op3::can_implement(problem_shape, args.op_3); 
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += Op3::get_workspace_size(problem_shape, args.op_3);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    status = Op0::initialize_workspace(problem_shape, args.op_0, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op0::get_workspace_size(problem_shape, args.op_0);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op1::initialize_workspace(problem_shape, args.op_1, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op1::get_workspace_size(problem_shape, args.op_1);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op2::initialize_workspace(problem_shape, args.op_2, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op2::get_workspace_size(problem_shape, args.op_2);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = Op3::initialize_workspace(problem_shape, args.op_3, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += Op3::get_workspace_size(problem_shape, args.op_3);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase() {}

  CUTLASS_HOST_DEVICE
  Sm90VisitorImplBase(Params const& params, SharedStorage const& shared_storage)
    : ops({
        Op0(params.op_0, get<0>(shared_storage)),
        Op1(params.op_1, get<1>(shared_storage)),
        Op2(params.op_2, get<2>(shared_storage)),
        Op3(params.op_3, get<3>(shared_storage))
      }) {}

  tuple<Op0, Op1, Op2, Op3> ops;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
