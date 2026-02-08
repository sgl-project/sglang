/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/* \file
  \brief Defines operations for all CONV operation kinds in CUTLASS Library.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/trace.h"
#include <utility>
#include <variant>
#if defined(CUTLASS_DEBUG_TRACE_LEVEL)
#include <sstream>
#endif

namespace cutlass::library {

namespace detail {

template<class ValueType, size_t ... Indices>
constexpr cute::array<ValueType, 1u + sizeof...(Indices)>
vector_to_array_strides_helper(const std::vector<ValueType>& v,
                               std::index_sequence<Indices...>)
{
  return {v[(sizeof...(Indices) - 1u) - Indices]..., ValueType(1)};
}

template<class ValueType, size_t Size>
cute::array<ValueType, Size>
vector_to_array_strides(const std::vector<ValueType>& v, std::integral_constant<size_t, Size>)
{
  static_assert(Size != 0);
  CUTLASS_ASSERT(v.size() + 1u == Size);
  return vector_to_array_strides_helper(v, std::make_index_sequence<Size - 1u>{});
}

template<class Index, class LongIndex, size_t ... Indices>
constexpr cute::array<int64_t, 1u + sizeof...(Indices)>
coord_to_array_strides_helper(
  const ::cutlass::Coord<int(sizeof...(Indices)), Index, LongIndex> coord,
  std::index_sequence<Indices...>)
{
  return {int64_t(coord[(sizeof...(Indices) - 1u) - Indices])..., int64_t(1)};
}

template<int Rank, class Index, class LongIndex>
cute::array<int64_t, 1u + size_t(Rank)>
coord_to_array_strides(const ::cutlass::Coord<Rank, Index, LongIndex>& coord)
{
  static_assert(Rank >= 0);
  return coord_to_array_strides_helper(coord, std::make_index_sequence<Rank>{});
}

} // namespace detail

// Tells the profiler about CUTLASS 3's 2-D and 3-D convolutions.
// For CUTLASS 2's 2-D convolutions, see Conv2dOperation.
// For CUTLASS 2's 3-D convolutions, see Conv3dOperation.
template<class Operator_>
class ConvOperation3x : public Operation {
public:
  using Operator = Operator_;

  static_assert(Operator::NumSpatialDimensions == 2 ||
    Operator::NumSpatialDimensions == 3,
    "The profiler currently only supports convolutions with 2 or 3 spatial dimensions.");
  using LayoutA = cute::conditional_t<Operator::NumSpatialDimensions == 3,
    cutlass::layout::TensorNDHWC,
    cute::conditional_t<Operator::NumSpatialDimensions == 2,
      cutlass::layout::TensorNHWC,
      cutlass::layout::TensorNWC>
    >;
  using LayoutB = LayoutA;
  using LayoutC = LayoutA;

  using ElementA = typename Operator::ElementA;
  using ElementB = typename Operator::ElementB;
  using ElementC = typename Operator::ElementC;
  using ElementD = typename Operator::ElementD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;
  static cutlass::conv::Operator const kConvolutionalOperator = Operator::kConvolutionalOperator;

  ConvOperation3x(const char* name = "unknown_cutlass_3_conv") {
    // Initialize OperationDescription (the base class)
    description_.name = name;
    description_.provider = Provider::kCUTLASS;

    if constexpr (Operator::NumSpatialDimensions == 2) {
      description_.kind = OperationKind::kConv2d;
    }
    else if constexpr (Operator::NumSpatialDimensions == 3) {
      description_.kind = OperationKind::kConv3d;
    }
    else {
      static_assert(::cutlass::detail::dependent_false<Operator>,
        "This class currently only supports 2-D and 3-D convolutions.");
    }

    description_.tile_description.threadblock_shape = make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = make_Coord(
      Operator::WarpCount::kM,
      Operator::WarpCount::kN,
      Operator::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape = make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator =
      NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class =
      OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      MathOperationID::kMultiplyAdd;

    description_.tile_description.minimum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability =
      ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;

    // Initialize ConvDescription (the subclass)

    // kConvDim does not exist in Operator for CUTLASS 3 convolutions.
    // For CUTLASS 2 convolutions, it is the number of spatial dimensions.
    description_.conv_dim = Operator::NumSpatialDimensions;
    description_.conv_kind = ConvKindMap<kConvolutionalOperator>::kId;

    description_.iterator_algorithm = {};

    description_.A = make_TensorDescription<ElementA, LayoutA>();
    description_.B = make_TensorDescription<ElementB, LayoutB>();
    description_.C = make_TensorDescription<ElementC, LayoutC>();
    description_.element_epilogue = NumericTypeMap<ElementCompute>::kId;
  }

  ~ConvOperation3x() override = default;

  OperationDescription const& description() const override {
    return static_cast<OperationDescription const&>(description_);
  }

private:
  Status update_operator_arguments_from_configuration_2d_or_3d(
    typename Operator::Arguments& out_args,
    void const* configuration) const {
    Status status = Status::kInvalid;

    CUTLASS_ASSERT(configuration != nullptr);

    if constexpr (Operator::NumSpatialDimensions == 2) {
      CUTLASS_ASSERT(description_.kind == OperationKind::kConv2d);
      // tools/library/include/cutlass/library/library.h
      // defines Conv2dConfiguration.
      // tools/profiler/include/cutlass/profiler/conv2d_operation_profiler.h
      // uses Conv2dConfiguration.
      auto* conf_ptr = reinterpret_cast<Conv2dConfiguration const*>(configuration);
      status = update_operator_arguments_from_configuration(out_args, *conf_ptr);
    }
    else if constexpr (Operator::NumSpatialDimensions == 3) {
      CUTLASS_ASSERT(description_.kind == OperationKind::kConv3d);
      auto* conf_ptr = reinterpret_cast<Conv3dConfiguration const*>(configuration);
      status = update_operator_arguments_from_configuration(out_args, *conf_ptr);
    }
    else {
      static_assert(::cutlass::detail::dependent_false<Operator>,
        "This class currently only supports 2-D and 3-D convolutions.");
    }

    return status;
  }

public:
  Status can_implement(
    void const* configuration,
    void const* arguments) const override {
    Status status = Status::kInvalid;

    // gemm_operation_3x.hpp accesses "configuration" as
    // GemmUniversalConfiguration (which lives in
    // tools/library/include/cutlass/library/library.h) and
    // "arguments" as GemmUniversalArguments (which lives in
    // tools/library/include/cutlass/library/library.h).
    // Those things don't apply to convolutions.
    // Despite the existence of ConvUniversal, there's no
    // corresponding "ConvUniversalConfiguration" or
    // "ConvUniversalArguments."

    CUTLASS_ASSERT(configuration != nullptr);
    CUTLASS_ASSERT(arguments != nullptr);

    typename Operator::Arguments out_args{};
    status = update_operator_arguments_from_configuration_2d_or_3d(out_args, configuration);
    if (status != Status::kSuccess) {
      CUTLASS_TRACE_HOST("*** can_implement: update_operator_arguments_from_configuration_2d_or_3d failed");
      return status;
    }

    auto* in_args_ptr = reinterpret_cast<ConvArguments const*>(arguments);
    status = update_operator_arguments_from_arguments(out_args, *in_args_ptr);
    if (status != Status::kSuccess) {
      CUTLASS_TRACE_HOST("*** can_implement: update_operator_arguments_from_arguments failed");
      return status;
    }

    return Operator::can_implement(out_args);
  }

  uint64_t get_host_workspace_size(void const* /* configuration */) const override {
    return sizeof(Operator);
  }

  uint64_t get_device_workspace_size(
    void const* configuration,
    void const* arguments = nullptr) const override
  {
    // This presumes that at least one of configuration or arguments is nonnull.
    Status status = Status::kInvalid;

    // gemm_operation_3x.hpp has get_device_workspace_size return 0 on
    // error.  It's not clear that this is what we want -- perhaps we
    // should return something like expected<uint64_t, Status>? -- but
    // it's the only option that preserves the current interface.
    constexpr uint64_t error_indication = 0;

    typename Operator::Arguments out_args{};
    if (configuration != nullptr) {
      status = update_operator_arguments_from_configuration_2d_or_3d(out_args, configuration);
      if (status != Status::kSuccess) {
        return error_indication;
      }
    }
    if (arguments != nullptr) {
      auto* in_args_ptr = reinterpret_cast<ConvArguments const*>(arguments);
      status = update_operator_arguments_from_arguments(out_args, *in_args_ptr);
      if (status != Status::kSuccess) {
        return error_indication;
      }
    }

    if (status == Status::kSuccess) {
      return static_cast<uint64_t>(Operator::get_workspace_size(out_args));
    }
    else {
      return error_indication;
    }
  }

  Status initialize(
    void const* configuration,
    void* host_workspace,
    void* /* device_workspace */ = nullptr,
    cudaStream_t stream = nullptr) const override
  {
    Status status = Status::kInvalid;

    if (configuration == nullptr) {
      CUTLASS_TRACE_HOST("Input configuration is null.");
      return Status::kInvalid;
    }

    typename Operator::Arguments out_args{};
    status = update_operator_arguments_from_configuration_2d_or_3d(out_args, configuration);
    if (status != Status::kSuccess) {
      // Any kind of failure invalidates the last successful configuration.
      clear_last_successful_config();
      return status;
    }
    else {
      set_last_successful_config(configuration);
    }

    if (host_workspace == nullptr) {
      CUTLASS_TRACE_HOST("host_workspace is null.");
      return Status::kInvalid;
    }
    (void) new (host_workspace) Operator;
    return status;

    // CUTLASS 2 convolutions call the Operator's initialize function
    // here, like this.
    //
    //return op->initialize(args, device_workspace, stream);
    //
    // CUTLASS 3 convolutions (ConvUniversal), like CUTLASS 3 Gemms
    // (GemmUniversal), lack an "initialize" member function.
  }

  Status run(
    void const* arguments,
    void* host_workspace,
    void* device_workspace = nullptr,
    cudaStream_t stream = nullptr) const override
  {
    auto status = Status::kInvalid;

    // The Operator doesn't appear to save the last configuration (it
    // doesn't have a way to do that, since it lacks an initialize()
    // member function), so we have to use the stored configuration
    // from the last successful initialize() call (if any).
    typename Operator::Arguments out_args{};
    status = update_operator_arguments_from_stored_configuration(out_args);
    if (status != Status::kSuccess) {
      CUTLASS_TRACE_HOST("Updating from previous successful configuration failed.");
      return status;
    }

    if (arguments == nullptr) {
      CUTLASS_TRACE_HOST("Input argument 'arguments' is null.");
      return Status::kInvalid;
    }
    auto* in_args_ptr = reinterpret_cast<ConvArguments const*>(arguments);
    status = update_operator_arguments_from_arguments(out_args, *in_args_ptr);
    if (status != Status::kSuccess) {
      return status;
    }

    auto* op = reinterpret_cast<Operator*>(host_workspace);
    return op->run(out_args, device_workspace, stream, nullptr, in_args_ptr->use_pdl);
  }

private:
  ConvDescription description_;
  // Result of initialize() calling
  // update_operator_arguments_from_configuration() successfully.
  // This is needed because run() doesn't take a configuration, just
  // arguments, and the kernel doesn't appear to save the
  // configuration from the last initialize() call.
  //
  // Unfortunately, this must be declared mutable, because it must be
  // set in initialize(), and initialize() is inherited as const.
  mutable std::variant<
    std::monostate,
    Conv2dConfiguration,
    Conv3dConfiguration> last_successful_config_{std::monostate{}};

  // Clear the last configuration resulting from a successful initialize() call.
  //
  // Unfortunately, this must be declared const, because initialize() is.
  void clear_last_successful_config() const {
    last_successful_config_ = std::monostate{};
  }

  // Set the last configuration resulting from a successful initialize() call.
  //
  // Unfortunately, this must be declared const, because initialize() is.
  void set_last_successful_config(void const* configuration) const {
    CUTLASS_ASSERT(configuration != nullptr);

    if constexpr (Operator::NumSpatialDimensions == 2) {
      CUTLASS_ASSERT(description_.kind == OperationKind::kConv2d);
      auto* conf_ptr = reinterpret_cast<Conv2dConfiguration const*>(configuration);
      last_successful_config_ = *conf_ptr;
    } else if constexpr (Operator::NumSpatialDimensions == 3) {
      CUTLASS_ASSERT(description_.kind == OperationKind::kConv3d);
      auto* conf_ptr = reinterpret_cast<Conv3dConfiguration const*>(configuration);
      last_successful_config_ = *conf_ptr;
    }
    else {
      static_assert(::cutlass::detail::dependent_false<Operator>,
        "This class currently only supports 2-D and 3-D convolutions.");
    }
  }

  // Whether a configuration from a successful initialize() call exists.
  bool last_successful_config_exists() const {
    return not std::holds_alternative<std::monostate>(last_successful_config_);
  }

  // Visitor for update_operator_arguments_from_stored_configuration.
  struct ConfigurationVisitor {
    typename Operator::Arguments& out_args;

    Status operator() (std::monostate const&) const {
      CUTLASS_TRACE_HOST("No successful previous configuration exists.  "
        "One cause is calling run() before a successful initialize() call.");
      return Status::kInvalid;
    }
    Status operator() (Conv2dConfiguration const& conf2d) const {
      return update_operator_arguments_from_configuration(out_args, conf2d);
    }
    Status operator() (Conv3dConfiguration const& conf3d) const {
      return update_operator_arguments_from_configuration(out_args, conf3d);
    }
  };

  // Like update_operator_arguments_from_configuration, but on the
  // stored configuration from the last successful initialize() call,
  // if any.  If there was no last successful initialize() call,
  // then return Status::kInvalid.
  //
  // Unfortunately, this must be declared const, because run() is.
  Status update_operator_arguments_from_stored_configuration(
    typename Operator::Arguments& out_args) const
  {
    return std::visit(ConfigurationVisitor{out_args}, last_successful_config_);
  }

  template<class FusionArgs, class = void>
  struct UpdateFusionArgs {
    static Status update_(
      FusionArgs const&,
      ConvArguments const&)
    {
      // For custom EVT, it is the user's responsibility to ensure
      // that alpha and beta are updated appropriately.
      return Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static Status update_(
      FusionArgs& fusion_args,
      ConvArguments const& arguments)
    {
      if (arguments.pointer_mode == ScalarPointerMode::kHost) {
        fusion_args.alpha = *static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta = *static_cast<ElementCompute const *>(arguments.beta);
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;

        return Status::kSuccess;
      }
      else if (arguments.pointer_mode == ScalarPointerMode::kDevice) {
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta_ptr = static_cast<ElementCompute const *>(arguments.beta);

        return Status::kSuccess;
      }
      else {
        return Status::kErrorInvalidProblem;
      }
    }
  };

  static Status update_operator_arguments_from_configuration(
    typename Operator::Arguments& out_args,
    Conv2dConfiguration const& config)
  {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("ConvOperator3x::"
      "update_operator_arguments_from_configuration"
      "(Conv2dConfiguration)\n");
#endif    
    using detail::vector_to_array_strides;

    constexpr int num_spatial_dims = Operator::NumSpatialDimensions;
    if constexpr (num_spatial_dims != 2) {
      CUTLASS_TRACE_HOST("You can only use Conv2dConfiguration "
        "with an Operator whose NumSpatialDimensions is exactly 2.");
      return Status::kInvalid;
    }
    else {
      // Convolutions split the metadata (in Conv2dConfiguration) from
      // the data (ConvArguments, which only has pointers and a single
      // enum value).  Thus, this class will need both the
      // configuration and the (user's input) arguments to set up the
      // kernel's arguments.  This function can fill in what the
      // configuration has now, but the class will need the user's
      // input arguments later.
      if (config.split_k_mode != conv::SplitKMode::kSerial) {
        CUTLASS_TRACE_HOST("CUTLASS 3 convolutions currently only support split_k_mode = kSerial.");
        return Status::kInvalid;
      }
      // config.problem_size.split_k_slices is only meaningful if
      // split_k_mode != kSerial.  If this code later supports other
      // split_k_mode values, then it will also need to read
      // split_k_slices.

      const int N = config.problem_size.N;
      const int H = config.problem_size.H;
      const int W = config.problem_size.W;
      const int C = config.problem_size.C;
      const int K = config.problem_size.K;
      const int R = config.problem_size.R;
      const int S = config.problem_size.S;
      const int pad_h = config.problem_size.pad_h;
      const int pad_w = config.problem_size.pad_w;
      const int traversal_stride_h = config.problem_size.stride_h;
      const int traversal_stride_w = config.problem_size.stride_w;
      const int dilation_h = config.problem_size.dilation_h;
      const int dilation_w = config.problem_size.dilation_w;

      // CUTLASS 3's implicit GEMM convolution kernels currently only
      // support cross correlation (passing over the activation and
      // filter tensors in the same order).  The convolution mode is
      // future work.
      const auto mode = config.problem_size.mode;
      if (mode != cutlass::conv::Mode::kCrossCorrelation) {
        CUTLASS_TRACE_HOST("Convolution modes other than kCrossCorrelation "
          "are not currently supported.");
        return Status::kInvalid;
      }

      constexpr int num_spatial_dims = Operator::NumSpatialDimensions;
      constexpr size_t stride_size = size_t(num_spatial_dims) + 2u;
      constexpr auto the_stride_size = std::integral_constant<size_t, stride_size>{};

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      std::cerr << "  num_spatial_dims = " << num_spatial_dims << "\n"
                << "  stride_size = " << stride_size << "\n";
      auto print_stride = [] (auto const& stride, char const variable_name[]) {
        std::cerr << "  " << variable_name << ": [";
        for (size_t k = 0; k < stride.size(); ++k) {
          std::cerr << stride[k];
          if (k + 1u < stride.size()) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      };
      print_stride(config.stride_a, "config.stride_a");
      print_stride(config.stride_b, "config.stride_b");
      print_stride(config.stride_c, "config.stride_c");
#endif

      // Conv2dConfiguration stores the strides as std::vector,
      // so the code needs to check the run-time vector lengths.
      if (config.stride_a.size() + 1u != stride_size) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL)
        std::ostringstream os;
        os << "config.stride_a.size() + 1u = "
           << (config.stride_a.size() + 1u)
           << " != num_spatial_dims + 2u = " << stride_size;
        CUTLASS_TRACE_HOST( os.str() );
#endif
        return Status::kInvalid;
      }
      if (config.stride_b.size() + 1u != stride_size) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL)
        std::ostringstream os;
        os << "config.stride_b.size() + 1u = "
           << (config.stride_b.size() + 1u)
           << " != num_spatial_dims + 2u = " << stride_size;
        CUTLASS_TRACE_HOST( os.str() );
#endif
        return Status::kInvalid;
      }
      if (config.stride_c.size() + 1u != stride_size) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL)
        std::ostringstream os;
        os << "config.stride_c.size() + 1u = "
           << (config.stride_c.size() + 1u)
           << " != num_spatial_dims + 2u = " << stride_size;
        CUTLASS_TRACE_HOST( os.str() );
#endif
        return Status::kInvalid;
      }

      constexpr cutlass::conv::Operator conv_op = Operator::DispatchPolicy::ConvOp;
      using problem_shape_type =
        cutlass::conv::ConvProblemShape<conv_op, num_spatial_dims>;
      // cute::array<int64_t, RankT>; must convert to the kernel's native strides
      using TensorStride = typename problem_shape_type::TensorStride;

      const TensorStride stride_A = vector_to_array_strides(config.stride_a, the_stride_size);
      const TensorStride stride_B = vector_to_array_strides(config.stride_b, the_stride_size);
      const TensorStride stride_C = vector_to_array_strides(config.stride_c, the_stride_size);

      // cutlass::library::Conv2dConfiguration has no member stride_d.
      // The code below imitates the testbed,
      // which just sets D's strides to C's strides.

      const int num_groups = config.problem_size.groups;
      if (num_groups != 1) {
        CUTLASS_TRACE_HOST("CUTLASS 3 kernels currently only support groups = 1.");
        return Status::kInvalid;
      }
      // ConvProblemShape is how CUTLASS 3 kernels represent
      // convolution problems.  ConvProblemShape's constructors take
      // shape_act, stride_act, shape_flt, and stride_flt, and set
      // shape_A, stride_A, shape_B, stride_B, shape_C, and stride_C
      // according to Fprop / Dgrad / Wgrad.
      //
      // This means that stride_act isn't always config.stride_A,
      // depending on Fprop / Dgrad / Wgrad.  The code here "undoes"
      // the logic in Conv2dWorkspace::set_stride_vector so that we
      // can recover the strides of the activation and filter tensors.
      // It doesn't need to worry about the so-called "output" tensor
      // (which might not be C), as ConvProblemShape's constructor
      // figures out its shapes and strides.
      using TensorExtent = typename problem_shape_type::TensorExtent;
      TensorExtent shape_act{N, H, W, C};
      auto stride_act = [&] () {
        // Some compilers consider conv_op (defined above), as
        // captured by this lambda, as "not a constant expression."
        constexpr auto conv_kind = Operator::DispatchPolicy::ConvOp;
        if constexpr (conv_kind == cutlass::conv::Operator::kFprop) {
          return stride_A;
        }
        else if constexpr (conv_kind == cutlass::conv::Operator::kDgrad) {
          return stride_C;
        }
        else { // conv_kind == cutlass::conv::Operator::kWgrad
          return stride_B;
        }
      } ();
      TensorExtent shape_flt{K, R, S, C};
      auto stride_flt = [&] () {
        // Some compilers consider conv_op (defined above), as
        // captured by this lambda, as "not a constant expression."
        constexpr auto conv_kind = Operator::DispatchPolicy::ConvOp;
        if constexpr (conv_kind == cutlass::conv::Operator::kFprop) {
          return stride_B;
        }
        else if constexpr (conv_kind == cutlass::conv::Operator::kDgrad) {
          return stride_B;
        }
        else { // conv_kind == cutlass::conv::Operator::kWgrad
          return stride_C;
        }
      } ();
      
      problem_shape_type problem_shape(
        /* mode             = */ mode,
        /* shape_act        = */ shape_act,
        /* stride_act       = */ stride_act,
        /* shape_flt        = */ shape_flt,
        /* stride_flt       = */ stride_flt,
        /* lower_padding    = */ {pad_h, pad_w},
        /* upper_padding    = */ {pad_h, pad_w},
        /* traversal_stride = */ {traversal_stride_h, traversal_stride_w},
        /* dilation         = */ {dilation_h, dilation_w},
                                 num_groups);
      out_args.problem_shape = problem_shape;

      // ConvProblemShape's constructor sets its shape_C member.
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("\n  problem_shape.shape_C: ");
      print(problem_shape.shape_C);
      printf("\n  problem_shape.stride_C: ");
      print(problem_shape.stride_C);
      printf("\n");
#endif
      // Initialization of C's and D's strides follows the CUTLASS 3
      // convolutions testbed (test/unit/conv/device_3x/testbed_conv.hpp).
      {
        using StrideC = typename Operator::ConvKernel::StrideC;
        using StrideD = typename Operator::ConvKernel::StrideD;
        auto stride_C = StrideC{};
        auto stride_D = StrideD{};

        if constexpr (conv_op == cutlass::conv::Operator::kWgrad) {
          stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, problem_shape.shape_C, problem_shape.stride_C, conv_op);
          stride_D = cutlass::make_cute_packed_stride(
            StrideD{}, problem_shape.shape_C, problem_shape.stride_C, conv_op);
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          std::cerr << "  Wgrad: stride_C: " << stride_C << "\n";
#endif
        }
        else {
          cute::for_each(cute::make_seq<cute::rank<0>(StrideC{})>{}, [&](auto i) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            const auto stride_C_i = problem_shape.stride_C[problem_shape_type::RankT-2-i];
            std::cerr << "  Fprop or Dgrad: get<0, " << i << ">(stride_C): "
                      << stride_C_i << "\n";
#endif
            cute::get<0, i>(stride_C) = problem_shape.stride_C[problem_shape_type::RankT-2-i];
          });
          cute::for_each(cute::make_seq<cute::rank<0>(StrideD{})>{}, [&](auto i) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            const auto stride_D_i = problem_shape.stride_C[problem_shape_type::RankT-2-i];
            std::cerr << "  Fprop or Dgrad: get<0, " << i << ">(stride_D): "
                      << stride_D_i << "\n";
#endif
            cute::get<0, i>(stride_D) = problem_shape.stride_C[problem_shape_type::RankT-2-i];
          });
        }
        out_args.epilogue.dC = stride_C;
        out_args.epilogue.dD = stride_D;
      }
      return Status::kSuccess;
    }
  }

  static Status update_operator_arguments_from_configuration(
    typename Operator::Arguments& out_args,
    Conv3dConfiguration const& config)
  {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("ConvOperator3x::"
      "update_operator_arguments_from_configuration"
      "(Conv3dConfiguration)\n");
#endif    
    using detail::coord_to_array_strides;

    constexpr int num_spatial_dims = Operator::NumSpatialDimensions;
    if constexpr (num_spatial_dims != 3) {
      CUTLASS_TRACE_HOST("You can only use Conv3dConfiguration "
        "with an Operator whose NumSpatialDimensions is exactly 3.");
      return Status::kInvalid;
    }
    else {
      // Convolutions split the metadata (in Conv3dConfiguration) from
      // the data (ConvArguments, which only has pointers and a single
      // enum value).  Thus, this class will need both the
      // configuration and the (user's input) arguments to set up the
      // kernel's arguments.  This function can fill in what the
      // configuration has now, but the class will need the user's
      // input arguments later.
      if (config.split_k_mode != conv::SplitKMode::kSerial) {
        CUTLASS_TRACE_HOST("CUTLASS 3 convolutions currently only support split_k_mode = kSerial.");
        return Status::kInvalid;
      }
      // config.problem_size.split_k_slices is only meaningful if
      // split_k_mode != kSerial.  If this code later supports other
      // split_k_mode values, then it will also need to read
      // split_k_slices.

      const int N = config.problem_size.N;
      const int D = config.problem_size.D;
      const int H = config.problem_size.H;
      const int W = config.problem_size.W;
      const int C = config.problem_size.C;
      const int K = config.problem_size.K;
      const int T = config.problem_size.T;
      const int R = config.problem_size.R;
      const int S = config.problem_size.S;
      const int pad_d = config.problem_size.pad_d;
      const int pad_h = config.problem_size.pad_h;
      const int pad_w = config.problem_size.pad_w;
      const int traversal_stride_d = config.problem_size.stride_d;
      const int traversal_stride_h = config.problem_size.stride_h;
      const int traversal_stride_w = config.problem_size.stride_w;
      const int dilation_d = config.problem_size.dilation_d;
      const int dilation_h = config.problem_size.dilation_h;
      const int dilation_w = config.problem_size.dilation_w;

      // CUTLASS 3's implicit GEMM convolution kernels currently only
      // support cross correlation (passing over the activation and
      // filter tensors in the same order).  The convolution mode is
      // future work.
      const auto mode = config.problem_size.mode;
      if (mode != cutlass::conv::Mode::kCrossCorrelation) {
        CUTLASS_TRACE_HOST("Convolution modes other than kCrossCorrelation "
          "are not currently supported.");
        return Status::kInvalid;
      }

      using Stride = cutlass::layout::TensorNDHWC::Stride;
      static_assert(std::is_same_v<Stride, cutlass::Coord<4>>);

      const cutlass::library::ConvKind conv_kind = [] () {
        constexpr cutlass::conv::Operator op = Operator::DispatchPolicy::ConvOp;
        if constexpr (op == cutlass::conv::Operator::kFprop) {
          return library::ConvKind::kFprop;
        }
        else if constexpr (op == cutlass::conv::Operator::kDgrad) {
          return library::ConvKind::kDgrad;
        }
        else /* if constexpr (op == cutlass::conv::Operator::kWgrad) */ {
          return library::ConvKind::kWgrad;
        }
      } ();
      const Stride input_stride_a = config.layout_a(conv_kind).stride();
      const Stride input_stride_b = config.layout_b(conv_kind).stride();
      const Stride input_stride_c = config.layout_c(conv_kind).stride();

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      constexpr size_t stride_size = size_t(num_spatial_dims) + 2u;
      std::cerr << "  num_spatial_dims = " << num_spatial_dims << "\n"
                << "  stride_size = " << stride_size << "\n";
      auto print_stride = [] (Stride const& stride, char const variable_name[]) {
        std::cerr << "  " << variable_name << ": [";
        for (size_t k = 0; k < Stride::kRank; ++k) {
          std::cerr << stride[static_cast<int>(k)];
          if (k + 1u < Stride::kRank) {
            std::cerr << ", ";
          }
        }
        std::cerr << "]\n";
      };
      print_stride(input_stride_a, "input_stride_a");
      print_stride(input_stride_b, "input_stride_b");
      print_stride(input_stride_c, "input_stride_c");
#endif
      // Conv3dConfiguration stores the strides as Coord (with
      // compile-time size), so there's no need to check sizes here
      // (unlike Conv2dConfiguration, which stores strides as
      // std::vector).

      constexpr cutlass::conv::Operator conv_op = Operator::DispatchPolicy::ConvOp;
      using problem_shape_type =
        cutlass::conv::ConvProblemShape<conv_op, num_spatial_dims>;
      // cute::array<int64_t, RankT>; must convert to the kernel's native strides
      using TensorStride = typename problem_shape_type::TensorStride;

      const TensorStride stride_A = coord_to_array_strides(input_stride_a);
      const TensorStride stride_B = coord_to_array_strides(input_stride_b);
      const TensorStride stride_C = coord_to_array_strides(input_stride_c);

      const int num_groups = config.problem_size.groups;
      if (num_groups != 1) {
        CUTLASS_TRACE_HOST("CUTLASS 3 kernels currently only support groups = 1.");
        return Status::kInvalid;
      }
      // ConvProblemShape is how CUTLASS 3 kernels represent
      // convolution problems.  ConvProblemShape's constructors take
      // shape_act, stride_act, shape_flt, and stride_flt, and set
      // shape_A, stride_A, shape_B, stride_B, shape_C, and stride_C
      // according to Fprop / Dgrad / Wgrad.
      //
      // Conv3dConfiguration differs a bit from Conv2dConfiguration,
      // but the idea is the same: the "input_stride_a" from config
      // depends on conv_kind (Fprop, Dgrad, or Wgrad), so stride_act
      // isn't always input_stride_a.  Analogously, stride_flt isn't
      // always input_stride_b.  The code here "undoes" the logic in
      // config.layout_a(conv_kind) and config.layout_b(conv_kind)
      // (analogous to Conv2dWorkspace::set_stride_vector) so that we
      // can recover the strides of the activation and filter tensors.
      // It doesn't need to worry about the so-called "output" tensor
      // (which might not be C), as ConvProblemShape's constructor
      // figures out its shapes and strides.
      using TensorExtent = typename problem_shape_type::TensorExtent;
      TensorExtent shape_act{N, D, H, W, C};
      auto stride_act = [&] () {
        // Some compilers consider conv_op (defined above), as
        // captured by this lambda, as "not a constant expression."
        constexpr auto conv_kind = Operator::DispatchPolicy::ConvOp;
        if constexpr (conv_kind == cutlass::conv::Operator::kFprop) {
          return stride_A;
        }
        else if constexpr (conv_kind == cutlass::conv::Operator::kDgrad) {
          return stride_C;
        }
        else { // conv_kind == cutlass::conv::Operator::kWgrad
          return stride_B;
        }
      } ();
      TensorExtent shape_flt{K, T, R, S, C};
      auto stride_flt = [&] () {
        // Some compilers consider conv_op (defined above), as
        // captured by this lambda, as "not a constant expression."
        constexpr auto conv_kind = Operator::DispatchPolicy::ConvOp;
        if constexpr (conv_kind == cutlass::conv::Operator::kFprop) {
          return stride_B;
        }
        else if constexpr (conv_kind == cutlass::conv::Operator::kDgrad) {
          return stride_B;
        }
        else { // conv_kind == cutlass::conv::Operator::kWgrad
          return stride_C;
        }
      } ();

      problem_shape_type problem_shape(
        /* mode             = */ mode,
        /* shape_act        = */ shape_act,
        /* stride_act       = */ stride_act,
        /* shape_flt        = */ shape_flt,
        /* stride_flt       = */ stride_flt,
        /* lower_padding    = */ {pad_d, pad_h, pad_w},
        /* upper_padding    = */ {pad_d, pad_h, pad_w},
        /* traversal_stride = */ {traversal_stride_d, traversal_stride_h, traversal_stride_w},
        /* dilation         = */ {dilation_d, dilation_h, dilation_w},
                                 num_groups);
      out_args.problem_shape = problem_shape;

      // ConvProblemShape's constructor sets its shape_C member.
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      printf("\n  problem_shape.shape_C: ");
      print(problem_shape.shape_C);
      printf("\n  problem_shape.stride_C: ");
      print(problem_shape.stride_C);
      printf("\n");
#endif
      // Initialization of C's and D's strides follows the CUTLASS 3
      // convolutions testbed (test/unit/conv/device_3x/testbed_conv.hpp).
      {
        using StrideC = typename Operator::ConvKernel::StrideC;
        using StrideD = typename Operator::ConvKernel::StrideD;
        auto stride_C = StrideC{};
        auto stride_D = StrideD{};

        if constexpr (conv_op == cutlass::conv::Operator::kWgrad) {
          stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, problem_shape.shape_C, problem_shape.stride_C, conv_op);
          stride_D = cutlass::make_cute_packed_stride(
            StrideD{}, problem_shape.shape_C, problem_shape.stride_C, conv_op);
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          std::cerr << "  Wgrad: stride_C: " << stride_C << "\n";
#endif
        }
        else {
          cute::for_each(cute::make_seq<cute::rank<0>(StrideC{})>{}, [&](auto i) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            const auto stride_C_i = problem_shape.stride_C[problem_shape_type::RankT-2-i];
            std::cerr << "  Fprop or Dgrad: get<0, " << i << ">(stride_C): "
                      << stride_C_i << "\n";
#endif
            cute::get<0, i>(stride_C) = problem_shape.stride_C[problem_shape_type::RankT-2-i];
          });
          cute::for_each(cute::make_seq<cute::rank<0>(StrideD{})>{}, [&](auto i) {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            const auto stride_D_i = problem_shape.stride_C[problem_shape_type::RankT-2-i];
            std::cerr << "  Fprop or Dgrad: get<0, " << i << ">(stride_D): "
                      << stride_D_i << "\n";
#endif
            cute::get<0, i>(stride_D) = problem_shape.stride_C[problem_shape_type::RankT-2-i];
          });
        }
        out_args.epilogue.dC = stride_C;
        out_args.epilogue.dD = stride_D;
      }
      return Status::kSuccess;
    }
  }

  Status update_operator_arguments_from_arguments(
    typename Operator::Arguments& out_args,
    ConvArguments const& in_args) const
  {
#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("ConvOperation3x::update_operator_arguments_from_arguments\n");
#endif
    auto status = UpdateFusionArgs<decltype(out_args.epilogue.thread)>::update_(
      out_args.epilogue.thread, in_args);
    if (status != Status::kSuccess) {
      return status;
    }

    out_args.mainloop.ptr_A = reinterpret_cast<ElementA const*>(in_args.A);
    out_args.mainloop.ptr_B = reinterpret_cast<ElementB const*>(in_args.B);

    out_args.epilogue.ptr_C = reinterpret_cast<ElementC const*>(in_args.C);
    out_args.epilogue.ptr_D = reinterpret_cast<ElementD*>(in_args.D);

    return Status::kSuccess;
  }
};

} // namespace cutlass::library
