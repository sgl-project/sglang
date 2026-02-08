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
  \brief Visitor tree compute operations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/detail/helper_macros.hpp"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// N-nary Elementwise Compute Operation
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// The template argument provided for ComputeFn must be able to accept
// exactly one template parameter.  In Standard C++, it's OK for
// ComputeFn to have other template parameters, as long as those have
// defaults.  For example, the following struct Foo would work.
//
// template<class A, class B = A>
// struct Foo {
//   CUTLASS_HOST_DEVICE auto operator() (A a, B b);
// };
//
// However, some compilers, such as Clang, require that the argument
// take _exactly_ one template parameter.  This is nonstandard C++
// behavior.  One work-around for this case is to create a subclass
// with exactly one template parameter, and then use that subclass as
// the template argument.
//
// template<class A>
// struct FooHomogeneous : public Foo<A, A> {};
//
template<
  template <class> class ComputeFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class = void
>
struct Sm90Compute {
private:
  using EmptyArguments = typename Sm90VisitorImpl<>::Arguments;

  template <class Fn, class = void>
  struct ComputeArguments {
    using type = EmptyArguments;
  };

  // partial specialization for compute fns that define an Arguments member, e.g. activation hyperparameters
  template <class Fn>
  struct ComputeArguments<Fn, platform::void_t<typename Fn::Arguments>> {
    using type = typename Fn::Arguments;
  };

public:
  struct SharedStorage { };

  using Arguments = typename ComputeArguments<ComputeFn<ElementCompute>>::type;

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE
  Sm90Compute()
      : params() {}

  CUTLASS_HOST_DEVICE
  Sm90Compute(Params const& params, SharedStorage const& shared_storage)
      : params(params) {}

  Params const params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(Params const& params)
      : params(params) {}

    Params const& params;

    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      return transform_apply(cute::make_tuple(frg_inputs...),
        [&] (auto&& frg_input) CUTLASS_LAMBDA_FUNC_INLINE {
          using ElementInput = typename cute::remove_cvref_t<decltype(frg_input)>::Element;
          using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
          ConvertInput convert_input{};

          return convert_input(frg_input);
        },
        [&] (auto&&... cvt_frg_inputs) CUTLASS_LAMBDA_FUNC_INLINE {
          using ComputeOutput = ComputeFn<Array<ElementCompute, FragmentSize>>;
          ComputeOutput compute_output{};

          if constexpr (cute::is_same_v<Arguments, EmptyArguments>) {
            using ElementComputeOutput =
                typename cute::remove_cvref_t<decltype(compute_output(cvt_frg_inputs...))>::Element;
            using ConvertOutput = NumericArrayConverter<ElementOutput, ElementComputeOutput, FragmentSize, RoundStyle>;
            ConvertOutput convert_output{};
            return convert_output(compute_output(cvt_frg_inputs...));
          }
          else {
            using ElementComputeOutput =
                typename cute::remove_cvref_t<decltype(compute_output(cvt_frg_inputs..., params))>::Element;
            using ConvertOutput = NumericArrayConverter<ElementOutput, ElementComputeOutput, FragmentSize, RoundStyle>;
            ConvertOutput convert_output{};
            return convert_output(compute_output(cvt_frg_inputs..., params));
          }
        }
      );
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks(params);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Performance Optimized Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// beta * C + Z
template <
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class InputScaleOp,  // beta
  class ElementSource, // C
  class InputAddOp     // Z
>
struct Sm90TreeVisitor<
  Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle,
              cute::void_t<decltype(declval<InputScaleOp>().is_zero())>>,
  InputScaleOp,
  Sm90SrcFetch<ElementSource>,
  InputAddOp
> : Sm90VisitorImpl<
      InputScaleOp,
      Sm90SrcFetch<ElementSource>,
      InputAddOp,
      Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>
    >
{
  using Impl =
    Sm90VisitorImpl<
      InputScaleOp,
      Sm90SrcFetch<ElementSource>,
      InputAddOp,
      Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>
    >;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor() {}

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor(
      Params const& params,
      SharedStorage const& shared_storage)
    : Impl(params, shared_storage) {}

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    auto const& scale_op = get<0>(Impl::ops);
    auto const& added_op = get<2>(Impl::ops);
    if constexpr (detail::IsScalarBroadcast<InputScaleOp>::value && not is_void_v<ElementSource>) {
      return (get<2>(scale_op.params_ptr->dScalar[0]) != 0 && scale_op.params_ptr->scalar_ptrs[0] != nullptr) ||
              is_C_load_needed() ||
              added_op.is_producer_load_needed();
    }
    else {
      return is_C_load_needed() || added_op.is_producer_load_needed();
    }
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    auto const& scale_op = get<0>(Impl::ops);
    auto const& src_op = get<1>(Impl::ops);
    auto const& added_op = get<2>(Impl::ops);
    return (not scale_op.is_zero() && src_op.is_C_load_needed()) || added_op.is_C_load_needed();
  }

  template <class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(bool is_C_load_needed, CallbacksImpl&& impl)
      : is_C_load_needed(is_C_load_needed), CallbacksImpl(cute::forward<CallbacksImpl>(impl)) { }

    bool is_C_load_needed;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array frg_added = get<2>(CallbacksImpl::callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n);

      using ElementZ = typename decltype(frg_added)::Element;
      using ConvertZ = NumericArrayConverter<ElementCompute, ElementZ, FragmentSize, RoundStyle>;
      using ConvertI = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;
      ConvertZ convert_Z{};
      ConvertI convert_I{};

      Array frg_I = convert_Z(frg_added);

      if constexpr (!is_void_v<ElementSource>) {
        Array frg_scalar = get<0>(CallbacksImpl::callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n);
        Array frg_source = get<1>(CallbacksImpl::callbacks_tuple).visit(frg_acc, epi_v, epi_m, epi_n);

        using ElementX = typename decltype(frg_scalar)::Element;
        using ElementY = typename decltype(frg_source)::Element;
        using ConvertX = NumericArrayConverter<ElementCompute, ElementX, FragmentSize, RoundStyle>;
        using ConvertY = NumericArrayConverter<ElementCompute, ElementY, FragmentSize, RoundStyle>;
        using ComputeI = multiply_add<Array<ElementCompute, FragmentSize>>;
        ConvertX convert_X{};
        ConvertY convert_Y{};
        ComputeI compute_I{};

        frg_I = compute_I(convert_X(frg_scalar), convert_Y(frg_source), frg_I);
      }

      return convert_I(frg_I);
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto callbacks_tuple = Impl::template get_consumer_store_callbacks<ReferenceSrc>(args);
    bool is_C_load_needed = this->is_C_load_needed();
    if (not is_C_load_needed) {
      cute::clear(args.tCrC);
    }
    return ConsumerStoreCallbacks<decltype(callbacks_tuple)>(
        is_C_load_needed, std::move(callbacks_tuple));
  }
};

// ReLU with aux bit tensor dReLU/dZ
// Aux(i) = Z(i) >= 0 ? 1 : 0
namespace detail {
// Placeholder node so we can retain standard EVT structure
template <class StrideMNL>
struct Sm90ReLUAuxStore : Sm90VisitorImpl<> {
  struct SharedStorage {};

  struct Arguments {
    cutlass::uint1b_t* ptr_aux = nullptr;
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90ReLUAuxStore() { }

  CUTLASS_HOST_DEVICE
  Sm90ReLUAuxStore(Params const& params, SharedStorage const& shared_storage) { }
};
} // namespace detail

// Specialization on the generic compute+aux EVT
template <
  // Compute node
  template <class> class Activation,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  // Aux node
  int Stages,
  class EpilogueTile,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpR2S,
  int Alignment,
  bool EnableNullptr,
  // Input node
  class InputOp
>
struct Sm90TreeVisitor<
  Sm90Compute<Activation, ElementOutput, ElementCompute, RoundStyle,
              cute::enable_if_t<cute::is_same_v<Activation<ElementCompute>, cutlass::epilogue::thread::ReLu<ElementCompute>>  ||
                                cute::is_same_v<Activation<ElementCompute>, cutlass::epilogue::thread::Clamp<ElementCompute>> ||
                                cute::is_same_v<Activation<ElementCompute>, cutlass::epilogue::thread::ThresholdReLU<ElementCompute>> >>,
  Sm90TreeVisitor<
    Sm90AuxStore<
      Stages,
      EpilogueTile,
      cutlass::uint1b_t,
      RoundStyle,
      StrideMNL,
      SmemLayoutAtom,
      CopyOpR2S,
      Alignment,
      EnableNullptr
    >,
    InputOp
  >
> : Sm90VisitorImpl<
      Sm90VisitorImpl<
        InputOp,
        detail::Sm90ReLUAuxStore<StrideMNL>
      >,
      Sm90Compute<Activation, ElementOutput, ElementCompute, RoundStyle>
    >
{
  using Impl =
    Sm90VisitorImpl<
      Sm90VisitorImpl<
        InputOp,
        detail::Sm90ReLUAuxStore<StrideMNL>
      >,
      Sm90Compute<Activation, ElementOutput, ElementCompute, RoundStyle>
    >;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor() {}

  CUTLASS_HOST_DEVICE
  Sm90TreeVisitor(Params const& params_, SharedStorage const& shared_storage)
    : params(params_), Impl(params_, shared_storage) {}

  Params const& params;

  template <class RTensor, class GTensor, class CTensor, class ThrResidue, class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        RTensor&& tC_rAux,
        GTensor&& tC_gAux,
        CTensor tC_cAux,
        ThrResidue residue_tC_cAux,
        Params const& params,
        CallbacksImpl&& impl)
      : tC_rAux(cute::forward<RTensor>(tC_rAux)),
        tC_gAux(cute::forward<GTensor>(tC_gAux)),
        tC_cAux(tC_cAux),
        residue_tC_cAux(residue_tC_cAux),
        params(params),
        CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    RTensor tC_rAux;                                                                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    GTensor tC_gAux;                                                                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    CTensor tC_cAux;                                                                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tC_cAux;
    Params const& params;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      // Unpack callbacks + params
      auto& [callbacks_input_aux, callbacks_compute] = CallbacksImpl::callbacks_tuple;
      auto& [callbacks_input, callbacks_aux] = callbacks_input_aux.callbacks_tuple;
      auto const& [params_input_aux, params_compute] = params;
      auto const& [params_input, params_aux] = params_input_aux;

      // Visit the input node
      Array frg_input = callbacks_input.visit(frg_acc, epi_v, epi_m, epi_n);

      // Compute activation + aux
      using ElementInput = typename decltype(frg_input)::Element;
      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ConvertAux = PackPredicates<FragmentSize>;
      using ComputeOutput = Activation<ElementCompute>;
      using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};
      ComputeOutput relu{};
      ConvertAux convert_aux{};
      ConvertOutput convert_output{};

      Array frg_compute = convert_input(frg_input);
      bool frg_aux[FragmentSize];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        ElementCompute pre_relu = frg_compute[i];
        if constexpr (cute::is_same_v<Activation<ElementCompute>, cutlass::epilogue::thread::Clamp<ElementCompute>> ||
                      cute::is_same_v<Activation<ElementCompute>, cutlass::epilogue::thread::ThresholdReLU<ElementCompute>>) {
          frg_compute[i] = relu(frg_compute[i], params_compute);
        }
        else {
          frg_compute[i] = relu(frg_compute[i]);
        }
        if constexpr (cute::is_same_v<ElementCompute, float>) {
          uint32_t aux;
          asm volatile("set.equ.u32.f32 %0, %1, %2;\n" : "=r"(aux) : "f"(frg_compute[i]), "f"(pre_relu)); // NaN outputs 1 in Aux
          frg_aux[i] = static_cast<bool>(aux);
        } else if constexpr (cute::is_same_v<ElementCompute, cutlass::half_t>) {
          uint32_t aux;
          cutlass::half_t compute = frg_compute[i];
          asm volatile("set.equ.u32.f16 %0, %1, %2;\n" : "=r"(aux) : "h"(compute.raw()), "h"(pre_relu.raw())); // NaN outputs 1 in Aux
          frg_aux[i] = static_cast<bool>(aux);
        } else {
          frg_aux[i] = frg_compute[i] == pre_relu;
        }
      }

      static_assert(FragmentSize % 8 == 0, "Predicate vector must be byte-aligned");
      Tensor tC_rAux_frg = recast<typename ConvertAux::result_type>(coalesce(tC_rAux(_,_,_,epi_m,epi_n)));   // (EPI_V)
      tC_rAux_frg(epi_v) = convert_aux(frg_aux);

      return convert_output(frg_compute);
    }

    CUTLASS_DEVICE void
    end() {
      // Unpack callbacks + params
      auto& [callbacks_input_aux, callbacks_compute] = CallbacksImpl::callbacks_tuple;
      auto& [callbacks_input, callbacks_aux] = callbacks_input_aux.callbacks_tuple;
      auto const& [params_input_aux, params_compute] = params;
      auto const& [params_input, params_aux] = params_input_aux;

      // Visit the input node
      callbacks_input.end();

      // Nullptr is no-op
      if constexpr (EnableNullptr) {
        if (params_aux.ptr_aux == nullptr) {
          return;
        }
      }

      // Compute vectorization
      constexpr auto MCL = decltype(max_common_layout(tC_rAux, tC_gAux)){};
      constexpr int V = cute::min(Alignment, size(MCL));
      // Copy vectorizes into byte-aligned stores
      if constexpr (V > 1 && V % 8 == 0) {
        using VecType = uint_bit_t<V>;
        Tensor tC_rAux_vec = recast<VecType>(tC_rAux);
        Tensor tC_gAux_vec = recast<VecType>(tC_gAux);
        Tensor tC_cAux_vec = tensor<1>(zipped_divide(tC_cAux, MCL.compose(Int<V>{})));
        Tensor tC_pAux_vec = cute::lazy::transform(tC_cAux_vec, [&](auto const& c){ return elem_less(c, residue_tC_cAux); });
        copy_if(tC_pAux_vec, tC_rAux_vec, tC_gAux_vec);
      }
      // sub-byte vectorization, must serialize threads
      else {
        // Assumes no inter-warp sharing of bytes (most copy layouts should satisfy this)
        int lane_idx = canonical_lane_idx();
        Tensor tC_pAux = cute::lazy::transform(tC_cAux, [&](auto const& c){ return elem_less(c, residue_tC_cAux); });
        CUTLASS_PRAGMA_NO_UNROLL
        for (int i = 0; i < NumThreadsPerWarp; ++i) {
          if (lane_idx == i) {
            copy_if(tC_pAux, tC_rAux, tC_gAux);
          }
          __syncwarp();
        }
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    // Unpack params
    auto const& [params_input_aux, params_compute] = params;
    auto const& [params_input, params_aux] = params_input_aux;

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    gmem_ptr ptr_aux = make_gmem_ptr<cutlass::uint1b_t>(params_aux.ptr_aux);
    Tensor mAux = make_tensor(ptr_aux, make_layout(make_shape(M,N,L), params_aux.dAux));                     // (M,N,L)
    Tensor gAux = local_tile(mAux, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));                 // (CTA_M,CTA_N)

    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gAux, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tC_rAux = make_tensor<cutlass::uint1b_t>(shape(tC_gAux));                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    auto callbacks_impl = Impl::template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tC_gAux), decltype(args.tCcD), decltype(args.residue_tCcD), decltype(callbacks_impl)>(
        cute::move(tC_rAux), cute::move(tC_gAux), args.tCcD, args.residue_tCcD, params, cute::move(callbacks_impl));
  }
};

// Aux load for uint1b_t
template <
  int Stages,
  class EpilogueTile,
  class StrideMNL,
  class SmemLayoutAtom,
  class CopyOpS2R,
  int Alignment,
  bool EnableNullptr
>
struct Sm90AuxLoad<
  Stages,
  EpilogueTile,
  cutlass::uint1b_t,
  StrideMNL,
  SmemLayoutAtom,
  CopyOpS2R,
  Alignment,
  EnableNullptr
> {
  static_assert(Alignment % 128 == 0, "sub-16B alignment not supported yet");

  struct SharedStorage {};

  struct Arguments {
    cutlass::uint1b_t const* ptr_aux = nullptr;
    cutlass::uint1b_t null_default = cutlass::uint1b_t(0);
    StrideMNL dAux = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90AuxLoad() { }

  CUTLASS_HOST_DEVICE
  Sm90AuxLoad(Params const& params, SharedStorage const&)
      : params(params) { }

  Params const params;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class RTensor, class GTensor, class CTensor, class ThrResidue>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& tC_rAux_, GTensor&& tC_gAux_, CTensor tC_cAux_, ThrResidue residue_tC_cAux_, Params const& params_)
      : tC_rAux(cute::forward<RTensor>(tC_rAux_)),
        tC_gAux(cute::forward<GTensor>(tC_gAux_)),
        tC_cAux(tC_cAux_),
        residue_tC_cAux(residue_tC_cAux_),
        params(params_) {}

    RTensor tC_rAux;                                                                   // (CPY,CPY_M,CPY_N,{EPI_M,EPI_N})
    GTensor tC_gAux;                                                                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    CTensor tC_cAux;                                                                   // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tC_cAux;
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if constexpr (decltype(cute::rank(tC_rAux))::value == 5) {
        if constexpr (EnableNullptr) {
          if (params.ptr_aux == nullptr) {
            return;
          }
        }

        constexpr auto MCL = decltype(max_common_layout(tC_rAux, tC_gAux)){};
        constexpr int V = cute::min(Alignment, size(MCL));
        if constexpr (V > 1) {
          using VecType = uint_bit_t<V>;
          Tensor tC_gAux_vec = recast<VecType>(tC_gAux);
          Tensor tC_rAux_vec = recast<VecType>(tC_rAux);
          Tensor tC_cAux_vec = tensor<1>(zipped_divide(tC_cAux, MCL.compose(Int<V>{})));
          Tensor tC_pAux_vec = cute::lazy::transform(tC_cAux_vec, [&](auto const& c){ return elem_less(c, residue_tC_cAux); });
          copy_if(tC_pAux_vec, tC_gAux_vec, tC_rAux_vec);
        }
        else {
          Tensor tC_pAux = cute::lazy::transform(tC_cAux, [&](auto const& c){ return elem_less(c, residue_tC_cAux); });
          copy_if(tC_pAux, tC_gAux, tC_rAux);
        }
      }
    }

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      if constexpr (decltype(cute::rank(tC_rAux))::value == 3) {
        if constexpr (EnableNullptr) {
          if (params.ptr_aux == nullptr) {
            return;
          }
        }

        Tensor tC_pAux = cute::lazy::transform(tC_cAux(_,_,_,epi_m,epi_n), [&](auto const& c){ return elem_less(c, residue_tC_cAux); });
        copy_if(tC_pAux, tC_gAux(_,_,_,epi_m,epi_n), tC_rAux);
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      using ElementRegister = typename remove_cvref_t<RTensor>::value_type;
      if constexpr (decltype(cute::rank(tC_rAux))::value == 3) {
        return recast<Array<ElementRegister, FragmentSize>>(coalesce(tC_rAux))(epi_v);
      }
      else {
        return recast<Array<ElementRegister, FragmentSize>>(coalesce(tC_rAux(_,_,_,epi_m,epi_n)))(epi_v);
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    gmem_ptr ptr_aux = make_gmem_ptr<cutlass::uint1b_t const>(params.ptr_aux);
    Tensor mAux = make_tensor(ptr_aux, make_layout(make_shape(M,N,L), params.dAux));                         // (M,N,L)
    Tensor gAux = local_tile(mAux, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));                 // (CTA_M,CTA_N)

    Tensor tC_gAux = sm90_partition_for_epilogue<ReferenceSrc>(                        // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
                      gAux, args.epi_tile, args.tiled_copy, args.thread_idx);

    // If byte-unaligned vectorization, store in registers as uint32_t to reduce redundant pack+unpack instruction sequences
    constexpr int V = decltype(max_common_vector(tC_gAux.layout(), make_layout(tC_gAux.shape())))::value;
    Tensor tC_rAux = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (V % 8 != 0) {
        return make_tensor<uint32_t>(take<0,3>(shape(tC_gAux)));                       // (CPY,CPY_M,CPY_N)
      } else {
        return make_tensor<cutlass::uint1b_t>(shape(tC_gAux));                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      }
    }();

    if constexpr (EnableNullptr) {
      if (params.ptr_aux == nullptr) {
        fill(tC_rAux, params.null_default);
      }
    }

    return ConsumerStoreCallbacks<decltype(tC_rAux), decltype(tC_gAux), decltype(args.tCcD), decltype(args.residue_tCcD)>(
        cute::move(tC_rAux), cute::move(tC_gAux), args.tCcD, args.residue_tCcD, params);
  }
};

// dReLU specialization
template<
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle
>
struct Sm90Compute<
  cutlass::epilogue::thread::dReLU,
  ElementOutput,
  ElementCompute,
  RoundStyle
> : Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    template <typename ElementAccumulator, typename ElementInput, typename ElementAux, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput      , FragmentSize> const& frg_input,
          Array<ElementAux        , FragmentSize> const& frg_aux) {
      using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
      using ComputeOutput = cutlass::epilogue::thread::dReLU<Array<ElementCompute, FragmentSize>>;
      using ConvertOutput = NumericArrayConverter<ElementOutput, ElementCompute, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};
      ComputeOutput compute_output{};
      ConvertOutput convert_output{};

      return convert_output(compute_output(convert_input(frg_input), frg_aux)); // don't convert frg_aux for dReLU
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks();
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
