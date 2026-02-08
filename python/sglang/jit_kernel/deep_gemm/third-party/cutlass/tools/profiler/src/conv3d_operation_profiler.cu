/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
   \brief Convolution 3D profiling

*/

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include "cutlass/core_io.h"

#include "cutlass/profiler/conv3d_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cutlass::library;

namespace cutlass {
namespace profiler {


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Ctor
Conv3dOperationProfiler::Conv3dOperationProfiler(Options const &options):
  OperationProfiler(
    options,
    library::OperationKind::kConv3d,
    {
      {ArgumentTypeID::kEnumerated, {"conv_kind"}, "Convolutional operator (fprop, dgrad, wgrad)"},
      {ArgumentTypeID::kInteger, {"n", "input_n"}, "Input N dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"d", "input_d"}, "Input D dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"h", "input_h"}, "Input H dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"w", "input_w"}, "Input W dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"c", "input_c"}, "Input C dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"k", "filter_k"}, "Filter K dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"t", "filter_t"}, "Filter T dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"r", "filter_r"}, "Filter R dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"s", "filter_s"}, "Filter S dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"z", "output_z"}, "Output Z dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"p", "output_p"}, "Output P dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"q", "output_q"}, "Output Q dimension of the Conv3d problem space"},
      {ArgumentTypeID::kInteger, {"pad_d"}, "Padding in D direction"},
      {ArgumentTypeID::kInteger, {"pad_h"}, "Padding in H direction"},
      {ArgumentTypeID::kInteger, {"pad_w"}, "Padding in W direction"},
      {ArgumentTypeID::kInteger, {"stride_d"}, "Stride in D direction"},
      {ArgumentTypeID::kInteger, {"stride_h"}, "Stride in H direction"},
      {ArgumentTypeID::kInteger, {"stride_w"}, "Stride in W direction"},
      {ArgumentTypeID::kInteger, {"dilation_d"}, "Dilation in D direction"},
      {ArgumentTypeID::kInteger, {"dilation_h"}, "Dilation in H direction"},
      {ArgumentTypeID::kInteger, {"dilation_w"}, "Dilation in W direction"},
      {ArgumentTypeID::kTensor, {"Activation"}, "Tensor storing the Activation operand"},
      {ArgumentTypeID::kTensor, {"Filter"}, "Tensor storing the Filter operand"},
      {ArgumentTypeID::kTensor, {"Output"}, "Tensor storing the Output operand"},
      {ArgumentTypeID::kEnumerated, {"conv_mode"}, "Convolution filter mode (conv, cross)"},
      {ArgumentTypeID::kEnumerated, {"iterator_algorithm", "iterator_algo"}, "Convolution iterator algorithm (analytic, optimized)"},
      {ArgumentTypeID::kScalar, {"alpha", "epilogue::alpha"}, "Epilogue scalar alpha"},
      {ArgumentTypeID::kScalar, {"beta", "epilogue::beta"}, "Epilogue scalar beta"},
      {ArgumentTypeID::kEnumerated, {"split_k_mode", "split-k-mode"}, "SplitK mode for serial or parallel reduction (serial, parallel)"},
      {ArgumentTypeID::kInteger, {"split_k_slices", "split-k-slices"}, "Number of partitions of K dimension"},
      {ArgumentTypeID::kEnumerated, {"eq_gemm_provider", "eq-gemm-provider"}, "Enable profiling equivalent gemm by the following providers (cutlass)"},
    },
    { library::Provider::kReferenceDevice, library::Provider::kReferenceHost, library::Provider::kCUDNN }
  ) {

  description_ = "      Conv3d operation. Output(Tensor5D) = alpha * Input(Tensor5D) * Filter(Tensor5D) + beta * Input(Tensor5D)";

}

/// Destructor
Conv3dOperationProfiler::~Conv3dOperationProfiler() {

}


/// Prints usage statement for the math function
void Conv3dOperationProfiler::print_usage(std::ostream &out) const {
  out << "Conv3d" << "\n\n";

  OperationProfiler::print_usage(out);
}

/// Prints examples
void Conv3dOperationProfiler::print_examples(std::ostream &out) const {

  out << "\nExamples:\n\n"
      << "Profile a particular convolution (specify all the convolution parameters):\n"
      << " $ cutlass_profiler --operation=Conv3d"
            " --Activation=f16:ndhwc --Filter=f16:ndhwc --Output=f16 --accumulator-type=f32"
            " --n=32 --d=16 --h=14 --w=14 --c=8 --k=64 --t=3 --r=3 --s=3"
            " --pad_d=1 --pad_h=1 --pad_w=1"
            " --stride_d=1 --stride::h=1 --stride::w=1"
            " --dilation_d=1 --dilation::h=1 --dilation::w=1\n\n";
}

#if 0
// used this for debugging
static std::string byte_string(std::vector<uint8_t> const &bytes) {
  std::stringstream ss;

  ss << "0x";

  for (size_t idx = bytes.size(); idx > 0; --idx) {
    ss << std::hex << std::setw(2) << std::setfill('0') << uint32_t(bytes.at(idx - 1));
  }

  return ss.str();
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////


/// Total number of bytes loaded
int64_t Conv3dOperationProfiler::Conv3dProblem::bytes(library::ConvDescription const &operation_desc) const {
  cutlass::gemm::GemmCoord mnk = eq_gemm_size(operation_desc.conv_kind);

 // Input bytes read and Output bytes written for the gemm problem
  int64_t bytes_ =
    int64_t(library::sizeof_bits(operation_desc.A.element) * mnk.m() / 8) * mnk.k() +
    int64_t(library::sizeof_bits(operation_desc.B.element) * mnk.n() / 8) * mnk.k() +
    int64_t(library::sizeof_bits(operation_desc.C.element) * mnk.m() / 8) * mnk.n();

  // Set is_beta_zero true if beta is zero
  bool is_beta_zero = std::all_of(beta.begin(), beta.end(), [](uint8_t i) { return i==0; });

  // Output bytes read for the gemm problem for non-zero beta values
  if (!is_beta_zero) {
    bytes_ += int64_t(library::sizeof_bits(operation_desc.C.element) * mnk.m() / 8) * mnk.n();
  }

  return bytes_;
}

/// Total number of flops computed
int64_t Conv3dOperationProfiler::Conv3dProblem::flops(
  library::ConvDescription const &operation_desc) const {

  cutlass::gemm::GemmCoord mnk = eq_gemm_size(operation_desc.conv_kind);

  int64_t flops_mainloop_ = int64_t(mnk.m()) * mnk.n() * mnk.k() * 2;
  int64_t flops_epilogue_ = int64_t(mnk.m()) * int64_t(mnk.n()) * 2;

  // Adjust mainloop flop for dgrad strided
  if (operation_desc.conv_kind == library::ConvKind::kDgrad) {
    flops_mainloop_ = flops_mainloop_ / ( stride_d * stride_h * stride_w);
  }

  return (flops_mainloop_ + flops_epilogue_);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Extracts the problem dimensions
Status Conv3dOperationProfiler::initialize_configuration(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::ConvDescription const &operation_desc =
    static_cast<library::ConvDescription const &>(operation->description());

  if (!arg_as_int(problem_.n, "n", problem_space, problem)) {
    // default value
    problem_.n = 1;
  }

  if (!arg_as_int(problem_.d, "d", problem_space, problem)) {
    // default value
    problem_.d = 8;
  }

  if (!arg_as_int(problem_.h, "h", problem_space, problem)) {
    // default value
    problem_.h = 14;
  }

  if (!arg_as_int(problem_.w, "w", problem_space, problem)) {
    // default value
    problem_.w = 14;
  }

  if (!arg_as_int(problem_.c, "c", problem_space, problem)) {
    // default value
    problem_.c = 32;
  }

  if (!arg_as_int(problem_.k, "k", problem_space, problem)) {
    // default value
    problem_.k = 32;
  }

  if (!arg_as_int(problem_.t, "t", problem_space, problem)) {
    // default value
    problem_.t = 3;
  }

  if (!arg_as_int(problem_.r, "r", problem_space, problem)) {
    // default value
    problem_.r = 3;
  }

  if (!arg_as_int(problem_.s, "s", problem_space, problem)) {
    // default value
    problem_.s = 3;
  }

  if (!arg_as_int(problem_.pad_d, "pad_d", problem_space, problem)) {
    // default value
    problem_.pad_d = 1;
  }

  if (!arg_as_int(problem_.pad_w, "pad_w", problem_space, problem)) {
    // default value
    problem_.pad_w = 1;
  }
  if (!arg_as_int(problem_.pad_h, "pad_h", problem_space, problem)) {
    // default value
    problem_.pad_h = 1;
  }

  if (!arg_as_int(problem_.stride_d, "stride_d", problem_space, problem)) {
    // default value
    problem_.stride_d = 1;
  }

  if (!arg_as_int(problem_.stride_h, "stride_h", problem_space, problem)) {
    // default value
    problem_.stride_h = 1;
  }

  if (!arg_as_int(problem_.stride_w, "stride_w", problem_space, problem)) {
    // default value
    problem_.stride_w = 1;
  }

  if (!arg_as_int(problem_.dilation_d, "dilation_d", problem_space, problem)) {
    // default value
    problem_.dilation_d = 1;
  }

  if (!arg_as_int(problem_.dilation_h, "dilation_h", problem_space, problem)) {
    // default value
    problem_.dilation_h = 1;
  }

  if (!arg_as_int(problem_.dilation_w, "dilation_w", problem_space, problem)) {
    // default value
    problem_.dilation_w = 1;
  }

  ////////////////////////  Convolution output dimensions p and q ////////////////////////
  // Cutlass convolutions support arbitrary output sizes and not constrained by         //
  // input, filter, padding, striding, dilation sizes.                                  //
  // cuDNN sets the output dimensions (p, q)  using following equations:                //
  //                                                                                    //
  // output = div_up(input + 2 * pad - ((filter - 1) * dilation + 1) + 1, stride)       //
  // where; div_up(a, b) : (a - 1)/b + 1                                                //
  //                                                                                    //
  // Thus, when output p and q dimensions are unspecified by the user                   //
  // cutlass profiler sets p and q which are cuDNN compliant.                           //
  //                                                                                    //
  ////////////////////////////////////////////////////////////////////////////////////////
  // set convolution output z
  if (!arg_as_int(problem_.z, "z", problem_space, problem)) {
    // default value (set using cudnn formula for output height, when p is not provided)
    problem_.z = (
                    problem_.d +
                    2 * problem_.pad_d -
                    ((problem_.t - 1) * problem_.dilation_d + 1)
                 ) / (problem_.stride_d)
                + 1;
  }

  // set convolution output p
  if (!arg_as_int(problem_.p, "p", problem_space, problem)) {
    // default value (set using cudnn formula for output height, when p is not provided)
    problem_.p = (
                    problem_.h +
                    2 * problem_.pad_h -
                    ((problem_.r - 1) * problem_.dilation_h + 1)
                 ) / (problem_.stride_h)
                + 1;
  }

  // set convolution output q
  if (!arg_as_int(problem_.q, "q", problem_space, problem)) {
    // default value (set using cudnn formula for output width, when q is not provided)
    problem_.q = (
                    problem_.w +
                    2 * problem_.pad_w -
                    ((problem_.s - 1) * problem_.dilation_w + 1)
                 ) / (problem_.stride_w)
                + 1;
  }
  /////////////////////////////////////////////////////////////////////////////////////////


  if (!arg_as_SplitKModeID(problem_.split_k_mode, "split_k_mode", problem_space, problem)) {
    // default value
    problem_.split_k_mode = library::SplitKMode::kSerial;
  }

  if (!arg_as_int(problem_.split_k_slices, "split_k_slices", problem_space, problem)) {
    // default value
    problem_.split_k_slices = 1;
  }

  if (!arg_as_ConvModeID(problem_.conv_mode, "conv_mode", problem_space, problem)) {
    // default value
    problem_.conv_mode = library::ConvModeID::kCrossCorrelation;
  }

  if (!arg_as_ProviderID(problem_.eq_gemm_provider, "eq_gemm_provider", problem_space, problem)) {
    // default value
    problem_.eq_gemm_provider = library::Provider::kNone;
  }

  if (!conv_kind_satisfies(operation_desc.conv_kind, "conv_kind", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!iterator_algorithm_satisfies(operation_desc.iterator_algorithm, "iterator_algorithm", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.activation(), "Activation", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.filter(), "Filter", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!tensor_description_satisfies(operation_desc.output(), "Output", problem_space, problem)) {
    return Status::kErrorInvalidProblem;
  }

  if (!arg_as_scalar(
    problem_.alpha,
    operation_desc.element_epilogue,
    "alpha",
    problem_space,
    problem)) {

    if (!cast_from_double(problem_.alpha, operation_desc.element_epilogue, 1)) {
      return Status::kErrorInternal;
    }
  }

  if (!arg_as_scalar(
    problem_.beta,
    operation_desc.element_epilogue,
    "beta",
    problem_space,
    problem)) {

    if (!cast_from_double(problem_.beta, operation_desc.element_epilogue, 0)) {
      return Status::kErrorInternal;
    }
  }

  // initialize library::ConvConfiguration
  conv_workspace_.configuration.problem_size = conv::Conv3dProblemSize(
                                                int(problem_.n),
                                                int(problem_.d),
                                                int(problem_.h),
                                                int(problem_.w),
                                                int(problem_.c),
                                                int(problem_.k),
                                                int(problem_.t),
                                                int(problem_.r),
                                                int(problem_.s),
                                                int(problem_.z),
                                                int(problem_.p),
                                                int(problem_.q),
                                                int(problem_.pad_d),
                                                int(problem_.pad_h),
                                                int(problem_.pad_w),
                                                int(problem_.stride_d),
                                                int(problem_.stride_h),
                                                int(problem_.stride_w),
                                                int(problem_.dilation_d),
                                                int(problem_.dilation_h),
                                                int(problem_.dilation_w),
                                                static_cast<conv::Mode>(static_cast<int>(problem_.conv_mode)),
                                                int(problem_.split_k_slices),
                                                1 // groups
                                              );

  conv_workspace_.configuration.split_k_mode = static_cast<conv::SplitKMode>(static_cast<int>(problem_.split_k_mode));

  conv_workspace_.configuration.layout_activations.stride() = make_Coord(
    int(problem_.c),
    int(problem_.w) * int(problem_.c),
    int(problem_.h) * int(problem_.w) * int(problem_.c),
    int(problem_.d) * int(problem_.h) * int(problem_.w) * int(problem_.c)
  );

  conv_workspace_.configuration.layout_filters.stride() = make_Coord(
    int(problem_.c),
    int(problem_.s) * int(problem_.c),
    int(problem_.r) * int(problem_.s) * int(problem_.c),
    int(problem_.t) * int(problem_.r) * int(problem_.s) * int(problem_.c)
  );

  conv_workspace_.configuration.layout_output.stride() = make_Coord(
    int(problem_.k),
    int(problem_.q) * int(problem_.k),
    int(problem_.q) * int(problem_.p) * int(problem_.k),
    int(problem_.z) * int(problem_.q) * int(problem_.p) * int(problem_.k)
  );


  // initialize library::ConvArguments
  conv_workspace_.arguments.A            = nullptr;
  conv_workspace_.arguments.B            = nullptr;
  conv_workspace_.arguments.C            = nullptr;
  conv_workspace_.arguments.D            = nullptr;
  conv_workspace_.arguments.alpha        = problem_.alpha.data();
  conv_workspace_.arguments.beta         = problem_.beta.data();
  conv_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  // initialize reduction operation for parallel splitKMode not supported for conv3d
  if(conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
    if(!initialize_reduction_configuration_(options, report, device_context, operation, problem_space, problem)) {
      return Status::kErrorInternal;
    }
  }

  initialize_result_(this->model_result_, options, operation_desc, problem_space);

  return operation->can_implement(&conv_workspace_.configuration, &conv_workspace_.arguments);
}

/// Initializes the performance result
void Conv3dOperationProfiler::initialize_result_(
  PerformanceResult &result,
  Options const &options,
  library::ConvDescription const &operation_desc,
  ProblemSpace const &problem_space) {

  result.provider = library::Provider::kCUTLASS;
  result.disposition = Disposition::kNotRun;
  result.status = Status::kSuccess;
  result.operation_name = operation_desc.name;

  result.arguments.resize(problem_space.rank());

  set_argument(result, "Activation", problem_space,
    std::string(library::to_string(operation_desc.activation().element))
    + ":" + library::to_string(operation_desc.activation().layout));

  set_argument(result, "Filter", problem_space,
    std::string(library::to_string(operation_desc.filter().element))
    + ":" + library::to_string(operation_desc.filter().layout));

  set_argument(result, "Output", problem_space,
    std::string(library::to_string(operation_desc.output().element))
    + ":" + library::to_string(operation_desc.output().layout));

  set_argument(result, "conv_kind", problem_space, library::to_string(operation_desc.conv_kind));

  set_argument(result, "iterator_algorithm", problem_space, std::string(library::to_string(operation_desc.iterator_algorithm)));

  set_argument(result, "n", problem_space, problem_.n);
  set_argument(result, "d", problem_space, problem_.d);
  set_argument(result, "h", problem_space, problem_.h);
  set_argument(result, "w", problem_space, problem_.w);
  set_argument(result, "c", problem_space, problem_.c);

  set_argument(result, "k", problem_space, problem_.k);
  set_argument(result, "t", problem_space, problem_.t);
  set_argument(result, "r", problem_space, problem_.r);
  set_argument(result, "s", problem_space, problem_.s);

  set_argument(result, "z", problem_space, problem_.z);
  set_argument(result, "p", problem_space, problem_.p);
  set_argument(result, "q", problem_space, problem_.q);

  set_argument(result, "pad_d", problem_space, problem_.pad_d);
  set_argument(result, "pad_h", problem_space, problem_.pad_h);
  set_argument(result, "pad_w", problem_space, problem_.pad_w);

  set_argument(result, "stride_d", problem_space, problem_.stride_d);
  set_argument(result, "stride_h", problem_space, problem_.stride_h);
  set_argument(result, "stride_w", problem_space, problem_.stride_w);

  set_argument(result, "dilation_d", problem_space, problem_.dilation_d);
  set_argument(result, "dilation_h", problem_space, problem_.dilation_h);
  set_argument(result, "dilation_w", problem_space, problem_.dilation_w);

  set_argument(result, "split_k_mode", problem_space,
    std::string(library::to_string(problem_.split_k_mode)));
  set_argument(result, "split_k_slices", problem_space, problem_.split_k_slices);

  set_argument(result, "conv_mode", problem_space,
    std::string(library::to_string(problem_.conv_mode)));

  set_argument(result, "alpha", problem_space,
    library::lexical_cast(problem_.alpha, operation_desc.element_epilogue));

  set_argument(result, "beta", problem_space,
    library::lexical_cast(problem_.beta, operation_desc.element_epilogue));

  set_argument(result, "eq_gemm_provider", problem_space,
    std::string(library::to_string(problem_.eq_gemm_provider)));

  OperationProfiler::initialize_result_(result, operation_desc, problem_space);

  // Bytes of activation, filter, and output tensors
  result.bytes = problem_.bytes(operation_desc);

  // Theoretical flops required for the computation
  result.flops = problem_.flops(operation_desc);

  // Measured runtime
  result.runtime = 0;

}

/// Initialize reduction problem dimensions and library::Operation
bool Conv3dOperationProfiler::initialize_reduction_configuration_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  library::ConvDescription const &conv_desc =
    static_cast<library::ConvDescription const &>(operation->description());

  library::ConvKind const &conv_kind = conv_desc.conv_kind;

  if (!cast_from_double(problem_.alpha_one, conv_desc.element_epilogue, 1)) {
   return false;
  }

  if (!cast_from_double(problem_.beta_zero, conv_desc.element_epilogue, 0)) {
   return false;
  }

  /// This chooses the appropriate stride element of the row-major C tensor.
  int const & tensor_c_stride_idx = (conv_kind == library::ConvKind::kWgrad ? 3 : 0);

  /// initialize library::ReductionConfiguration
  conv_workspace_.reduction_configuration.problem_size     = problem_.eq_gemm_size(conv_kind).mn();
  conv_workspace_.reduction_configuration.partitions       = int(problem_.split_k_slices);
  conv_workspace_.reduction_configuration.partition_stride = problem_.eq_gemm_size(conv_kind).mn().product();
  conv_workspace_.reduction_configuration.ldw              = conv_workspace_.configuration.layout_c(conv_kind).stride()[tensor_c_stride_idx];
  conv_workspace_.reduction_configuration.lds              = conv_workspace_.configuration.layout_c(conv_kind).stride()[tensor_c_stride_idx];
  conv_workspace_.reduction_configuration.ldd              = conv_workspace_.configuration.layout_c(conv_kind).stride()[tensor_c_stride_idx];

  // find reduction operation
  library::ReductionFunctionalKey reduction_key(
    library::Provider::kCUTLASS,
    conv_desc.tile_description.math_instruction.element_accumulator,  // element workspace
    conv_desc.tile_description.math_instruction.element_accumulator,  // element accumulator
    conv_desc.C.element,                                              // element output
    conv_desc.element_epilogue                                        // element compute
  );

#if 0// debug print to check which reduction instance is selected
    std::cout << reduction_key << "\n";
#endif
  auto reduction_it = Singleton::get().operation_table.reduction_operations.find(reduction_key);

  if(reduction_it == Singleton::get().operation_table.reduction_operations.end()) {

    return false;
  }

  // initialize reduction operation required for parallel split-k conv2d operator
  reduction_op_ = reduction_it->second;

  // reduction operation found and initialized
  return true;
}


/// Initializes workspace
Status Conv3dOperationProfiler::initialize_workspace(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (options.device.devices.size() != 1) {
    throw std::runtime_error("This operation profiler only supports a single "
                             "device.");
  }

  cudaError_t result;
  result = cudaSetDevice(options.device.device_id(0));
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice() failed.");
  }

  // initialize conv2d underlying operation to handle parallel reduction
  library::Operation const* underlying_operation = operation;

  if(conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_conv_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  library::ConvDescription const &operation_desc =
    static_cast<library::ConvDescription const &>(underlying_operation->description());

  // Compute the number of copies of the problem to avoid L2 camping.
  if (!options.profiling.workspace_count) {
    int64_t bytes = problem_.bytes(operation_desc);
    if (bytes < 3 * int64_t(options.device.properties[0].l2CacheSize)) {
      conv_workspace_.problem_count =
        1 + int((3 * int64_t(options.device.properties[0].l2CacheSize)) / bytes);
    }
    else {
      conv_workspace_.problem_count = 1;
    }
  }
  else {
    conv_workspace_.problem_count = options.profiling.workspace_count;
  }


  if (options.execution_mode != ExecutionMode::kDryRun) {
    int seed_shift = 0;
    conv_workspace_.A = device_context.allocate_and_initialize_tensor(
      options,
      "A",
      operation_desc.A.element,
      operation_desc.A.layout,
      problem_.extent_a(operation_desc.conv_kind),
      conv_workspace_.stride_a(operation_desc.conv_kind),
      conv_workspace_.problem_count,
      seed_shift++,
      0 // device_index
    );

    conv_workspace_.B = device_context.allocate_and_initialize_tensor(
      options,
      "B",
      operation_desc.B.element,
      operation_desc.B.layout,
      problem_.extent_b(operation_desc.conv_kind),
      conv_workspace_.stride_b(operation_desc.conv_kind),
      conv_workspace_.problem_count,
      seed_shift++,
      0 // device_index
    );

    conv_workspace_.C = device_context.allocate_and_initialize_tensor(
      options,
      "C",
      operation_desc.C.element,
      operation_desc.C.layout,
      problem_.extent_c(operation_desc.conv_kind),
      conv_workspace_.stride_c(operation_desc.conv_kind),
      conv_workspace_.problem_count,
      seed_shift++,
      0 // device_index
    );

    conv_workspace_.Computed = device_context.allocate_tensor(
      options,
      "D",
      operation_desc.C.element,
      operation_desc.C.layout,
      problem_.extent_c(operation_desc.conv_kind),
      conv_workspace_.stride_c(operation_desc.conv_kind),
      conv_workspace_.problem_count,
      0 // device_index
    );

    conv_workspace_.Reference = device_context.allocate_tensor(
      options,
      "Reference",
      operation_desc.C.element,
      operation_desc.C.layout,
      problem_.extent_c(operation_desc.conv_kind),
      conv_workspace_.stride_c(operation_desc.conv_kind),
      conv_workspace_.problem_count,
      0 // device_index
    );

  }

  //
  // Initialize the CUTLASS operation
  //
  Status status = Status::kSuccess;

  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    if (options.execution_mode != ExecutionMode::kDryRun) {

      uint64_t workspace_size = underlying_operation->get_host_workspace_size(&conv_workspace_.configuration);
      conv_workspace_.host_workspace.resize(workspace_size, 0);

      workspace_size = underlying_operation->get_device_workspace_size(&conv_workspace_.configuration);
      conv_workspace_.device_workspace.reset(library::NumericTypeID::kU8, workspace_size);

      status = underlying_operation->initialize(
        &conv_workspace_.configuration,
        conv_workspace_.host_workspace.data(),
        conv_workspace_.device_workspace.data());

      if (status != Status::kSuccess) {
        return status;
      }

      if (conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
        workspace_size = reduction_op_->get_host_workspace_size(&conv_workspace_.reduction_configuration);
        conv_workspace_.reduction_host_workspace.resize(workspace_size, 0);

        status = reduction_op_->initialize(
          &conv_workspace_.reduction_configuration,
          conv_workspace_.reduction_host_workspace.data(),
          nullptr);

        if (status != Status::kSuccess) {
          return status;
        }
      }
    }

    //
    // If CUTLASS is enabled, generate a result for it
    //
    results_.push_back(model_result_);
    results_.back().provider = library::Provider::kCUTLASS;
    results_.back().op_kind = library::OperationKind::kConv3d;
    results_.back().disposition = Disposition::kNotRun;

    for(auto provider : verification_providers_) {
      results_.back().verification_map[provider] = Disposition::kNotRun;
    }
  }

  return status;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Verifies CUTLASS against references
bool Conv3dOperationProfiler::verify_cutlass(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  if (!options.profiling.provider_enabled(library::Provider::kCUTLASS)) {
    return true;
  }

  if (options.execution_mode == ExecutionMode::kDryRun) {
    return true;
  }

  cudaError_t result;

  // Initialize structure containing Conv arguments
  set_cutlass_operator_arguments_();

  conv_workspace_.Computed->copy_from_device(conv_workspace_.C->data());

  //
  // Run the CUTLASS operation
  //
  // initialize conv2d underlying operation to handle parallel reduction
  library::Operation const* underlying_operation = operation;

  if(conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_conv_operation_for_parallel_reduction(operation))) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }
  }

#if 0
  std::cout << "profiling         : " << std::endl
            << "conv2d            : " << operation->description().name << std::endl
            << "underlying conv2d : " << underlying_operation->description().name << std::endl
            << "reduction         : " << reduction_op_->description().name << std::endl;
#endif

  // run cutlass conv2d operation
  results_.back().status = underlying_operation->run(
    &conv_workspace_.arguments,
    conv_workspace_.host_workspace.data(),
    conv_workspace_.device_workspace.data());

  if (results_.back().status != Status::kSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  // Run parallel reduction kernel for parallel split_k_mode
  if (conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {

    results_.back().status = reduction_op_->run(
      &conv_workspace_.reduction_arguments,
      conv_workspace_.reduction_host_workspace.data(),
      nullptr);

    if (results_.back().status != Status::kSuccess) {
      results_.back().disposition = Disposition::kFailed;
      return false;
    }

  }

  // Synchronize before running device reference
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    results_.back().disposition = Disposition::kFailed;
    return false;
  }

  // CUTLASS op ran the but not yet verified against any verification provider
  results_.back().disposition = Disposition::kNotVerified;

  //
  // Run verification providers
  //

  if (options.verification.enabled) {

#if CUTLASS_ENABLE_CUDNN
    // Run verification cudnn reference
    if (options.verification.provider_enabled(library::Provider::kCUDNN)) {

      // Guard against unsupported cases
      auto const & conv_desc = static_cast<library::ConvDescription const &>(operation->description());

      Status status = cudnn_satisfies(conv_desc, conv_workspace_.configuration);

      // Initialize reference data to the source data
      conv_workspace_.Reference->copy_from_device(conv_workspace_.C->data());

      if (status == Status::kSuccess) {
        // call cudnn verification if supported
        verify_with_cudnn_(
          options,
          report,
          device_context,
          operation,
          problem_space,
          problem);
      }

      else if (status == Status::kErrorInvalidProblem) {
        results_.back().verification_map[library::Provider::kCUDNN] = Disposition::kInvalidProblem;
      }

      else {
        // set verification map for cudnn to not supported
        results_.back().verification_map[library::Provider::kCUDNN] = Disposition::kNotSupported;
      }
    }
#endif // #if CUTLASS_ENABLE_CUDNN

    // Run verification host reference
    if (options.verification.provider_enabled(library::Provider::kReferenceHost)) {

      // Restore reference data back to initial source data
      conv_workspace_.Reference->copy_from_device(conv_workspace_.C->data());

      verify_with_host_reference_(
        options,
        report,
        device_context,
        operation,
        problem_space,
        problem);
    }

    // Update disposition to worst case verification outcome among all
    // verification providers which are supported
    bool is_any_verification_run_passed = false;
    for(auto &m : results_.back().verification_map) {
      if(m.second == Disposition::kFailed || m.second == Disposition::kIncorrect) {
        results_.back().disposition = m.second;
        return true;
      }
      if(!is_any_verification_run_passed && m.second == Disposition::kPassed) {
        is_any_verification_run_passed = true;
      }
    }

    if(is_any_verification_run_passed) {
      results_.back().disposition = Disposition::kPassed;
    }
  }

  // Return true means continue profiling
  return true;
}


/// Verifies CUTLASS against host reference
bool Conv3dOperationProfiler::verify_with_host_reference_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  Status status;

  //
  // Find host reference operation using conv functional description key
  //
  library::OperationDescription const &desc = operation->description();

  auto &conv_desc = static_cast<library::ConvDescription const &>(desc);

  library::ConvFunctionalKey conv_key(
    library::Provider::kReferenceHost,
    conv_desc.conv_kind,
    conv_desc.A.element,
    conv_desc.A.layout,
    conv_desc.B.element,
    conv_desc.B.layout,
    conv_desc.C.element,
    conv_desc.C.layout,
    conv_desc.tile_description.math_instruction.element_accumulator,
    conv_desc.element_epilogue);

#if 0 // debug print to check which host reference instance is selected
    std::cout << conv_key << "\n";
#endif

  auto operators_it = Singleton::get().operation_table.conv3d_operations.find(conv_key);

  if(operators_it == Singleton::get().operation_table.conv3d_operations.end()) {

    results_.back().verification_map[library::Provider::kReferenceHost] = Disposition::kNotRun;
    return true;
  }

  // conv3d host reference minimum cc is 0 (CPU) and no iterator algorithm
  library::ConvPreferenceKey preference_key(0, library::IteratorAlgorithmID::kNone);
  auto cc_it = operators_it->second.find(preference_key);

  if(cc_it == operators_it->second.end()) {
    results_.back().verification_map[library::Provider::kReferenceHost] = Disposition::kNotRun;
    return true;
  }

  // host reference has only one instances in ConvOperationVectorMap
  library::Operation const *reference_op = cc_it->second[0];

  //
  // Copy input tensors A, B, and C from device to host buffers
  //
  conv_workspace_.host_tensor_a.resize(conv_workspace_.A->bytes());
  conv_workspace_.host_tensor_b.resize(conv_workspace_.B->bytes());
  conv_workspace_.host_tensor_c.resize(conv_workspace_.C->bytes());
  conv_workspace_.A->copy_to_host(conv_workspace_.host_tensor_a.data());
  conv_workspace_.B->copy_to_host(conv_workspace_.host_tensor_b.data());
  conv_workspace_.C->copy_to_host(conv_workspace_.host_tensor_c.data());

  //
  // Initialize structure containing Conv3d arguments
  //
  conv_workspace_.arguments.A = conv_workspace_.host_tensor_a.data();
  conv_workspace_.arguments.B = conv_workspace_.host_tensor_b.data();
  conv_workspace_.arguments.C = conv_workspace_.host_tensor_c.data();
  conv_workspace_.arguments.D = conv_workspace_.host_tensor_c.data();
  conv_workspace_.arguments.alpha = problem_.alpha.data();
  conv_workspace_.arguments.beta = problem_.beta.data();
  conv_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  //
  // Initialize host reference operation
  //
  std::vector<uint8_t> host_workspace_reference_op;

  uint64_t workspace_size = reference_op->get_host_workspace_size(&conv_workspace_.configuration);
  host_workspace_reference_op.resize(workspace_size, 0);

  reference_op->initialize(
    &conv_workspace_.configuration,
    host_workspace_reference_op.data());

  //
  // Run host reference operation
  //
  status = reference_op->run(
    &conv_workspace_.arguments,
    host_workspace_reference_op.data());

  // Handle errors
  if (status != Status::kSuccess) {
    results_.back().verification_map[library::Provider::kReferenceHost] = Disposition::kNotVerified;
    return true;
  }

  //
  // Copy host reference output to device memory for equality check on device
  //
  conv_workspace_.Reference->copy_from_host(conv_workspace_.arguments.D);

  //
  // Verify results
  //
  results_.back().verification_map[library::Provider::kReferenceHost] = compare_tensors(
    options,
    *conv_workspace_.Computed,
    *conv_workspace_.Reference,
    conv_workspace_.Computed->batch_stride()
  );

  // Save workspace if incorrect
  if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
    results_.back().verification_map[library::Provider::kReferenceHost] == Disposition::kIncorrect) {

    save_workspace(
      device_context,
      options,
      static_cast<library::ConvDescription const &>(operation->description()),
      library::Provider::kCUTLASS,
      library::Provider::kReferenceHost);
  }

  // Return true means continue profiling
  return true;
}


/// Verifies CUTLASS against host reference
bool Conv3dOperationProfiler::verify_with_device_reference_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  // TODO: verify cutlass conv3d against device reference

  // Return true means continue profiling
  return true;
}

/// Measures performance results
bool Conv3dOperationProfiler::profile(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {


  if (options.profiling.provider_enabled(library::Provider::kCUTLASS)) {

    set_cutlass_operator_arguments_();

    results_.back().status = profile_cutlass_(
      results_.back(),
      options,
      operation,
      &conv_workspace_.arguments,
      conv_workspace_.host_workspace.data(),
      conv_workspace_.device_workspace.data()
    );
  }
  return true;

}

/// Updates the arguments structure for the CUTLASS operator based on
/// the problem index.
void Conv3dOperationProfiler::set_cutlass_operator_arguments_(int problem_idx) {
  // Initialize structure containing Conv3d arguments
  conv_workspace_.arguments.A = conv_workspace_.A->batch_data(problem_idx);
  conv_workspace_.arguments.B = conv_workspace_.B->batch_data(problem_idx);
  conv_workspace_.arguments.C = conv_workspace_.C->batch_data(problem_idx);
  conv_workspace_.arguments.D = conv_workspace_.Computed->batch_data(problem_idx);
  conv_workspace_.arguments.alpha = problem_.alpha.data();
  conv_workspace_.arguments.beta = problem_.beta.data();
  conv_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  if (conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
    // update library::ConvArguments for parallel split-k reduction
    conv_workspace_.arguments.D = conv_workspace_.device_workspace.data();
    conv_workspace_.arguments.alpha = problem_.alpha_one.data();
    conv_workspace_.arguments.beta = problem_.beta_zero.data();

    /// initialize library::ReductionArguments
    conv_workspace_.reduction_arguments.workspace           = conv_workspace_.device_workspace.data();
    conv_workspace_.reduction_arguments.source              = conv_workspace_.C->batch_data(problem_idx);
    conv_workspace_.reduction_arguments.destination         = conv_workspace_.Computed->batch_data(problem_idx);
    conv_workspace_.reduction_arguments.alpha               = problem_.alpha.data();
    conv_workspace_.reduction_arguments.beta                = problem_.beta.data();
    conv_workspace_.reduction_arguments.pointer_mode        = library::ScalarPointerMode::kHost;
  }
}

/// Method to profile a CUTLASS Operation
Status Conv3dOperationProfiler::profile_cutlass_(
  PerformanceResult &result,
  Options const &options,
  library::Operation const *operation,
  void *arguments,
  void *host_workspace,
  void *device_workspace) {

  // initialize conv2d underlying operation to handle parallel reduction
  library::Operation const* underlying_operation = operation;

  if(conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
    if (!(underlying_operation = library::find_conv_operation_for_parallel_reduction(operation))) {
      return Status::kErrorNotSupported;
    }
  }

  auto func = [&](cudaStream_t, int iteration) {
    // Setup rotating workspace
    int problem_idx = iteration % conv_workspace_.problem_count;

    set_cutlass_operator_arguments_(problem_idx);

    // Run underlying conv2d operation
    Status status = underlying_operation->run(
      arguments,
      host_workspace,
      device_workspace);

    // Run parallel reduction kernel for parallel split_k_mode
    if (conv_workspace_.configuration.split_k_mode == conv::SplitKMode::kParallel) {
      status = reduction_op_->run(
        &conv_workspace_.reduction_arguments,
        conv_workspace_.reduction_host_workspace.data(),
        nullptr);
    }

    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  };

  return profile_kernel_(result, options, func);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
#if CUTLASS_ENABLE_CUDNN

/// Verifies CUTLASS against cudnn reference
bool Conv3dOperationProfiler::verify_with_cudnn_(
  Options const &options,
  PerformanceReport &report,
  DeviceContext &device_context,
  library::Operation const *operation,
  ProblemSpace const &problem_space,
  ProblemSpace::Problem const &problem) {

  auto &conv_desc = static_cast<library::ConvDescription const &>(operation->description());

  //
  // Construct cudnn operators
  //

  CudnnCreate handle;
  cudnnStatus_t status = handle.get_cudnn_create_status();

  if (status != CUDNN_STATUS_SUCCESS) {

    results_.back().verification_map[library::Provider::kCUDNN] = get_cutlass_disposition(status);
    return true;
  }

  //
  // Initialize state
  //

  // Initialize structure containing Conv2d arguments
  conv_workspace_.arguments.A = conv_workspace_.A->data();
  conv_workspace_.arguments.B = conv_workspace_.B->data();
  conv_workspace_.arguments.D = conv_workspace_.Reference->data();
  conv_workspace_.arguments.alpha = problem_.alpha.data();
  conv_workspace_.arguments.beta = problem_.beta.data();
  conv_workspace_.arguments.pointer_mode = library::ScalarPointerMode::kHost;

  // cuDNN does not support four tensor arguments, so we copy the tensor C data into
  // tensor D.
  conv_workspace_.Reference->copy_from_device(conv_workspace_.C->data());
  conv_workspace_.arguments.C = conv_workspace_.arguments.D;

  try {

    //
    // Construct dispatcher to cudnn operator
    //

    detail::cudnnConvDispatcher conv_op(
      conv_desc,
      conv_workspace_.configuration,
      conv_workspace_.arguments,
      handle
    );

    if (conv_op.status != Status::kSuccess) {
      if (conv_op.status == Status::kErrorNotSupported) {
        results_.back().verification_map[library::Provider::kCUDNN] = Disposition::kNotSupported;

      } else {
        results_.back().verification_map[library::Provider::kCUDNN] = Disposition::kFailed;
      }
      return true;
    }


    status = conv_op(handle);

    // Handle errors
    if (status != CUDNN_STATUS_SUCCESS) {

      results_.back().verification_map[library::Provider::kCUDNN] = get_cutlass_disposition(status);
      return true;
    }

    //
    // Verify results
    //

    results_.back().verification_map[library::Provider::kCUDNN] = compare_tensors(
      options,
      *conv_workspace_.Computed,
      *conv_workspace_.Reference
    );

    // Save workspace if incorrect
    if (options.verification.save_workspace == SaveWorkspace::kIncorrect &&
      results_.back().verification_map[library::Provider::kCUDNN] == Disposition::kIncorrect) {

      save_workspace(
        device_context,
        options,
        conv_desc,
        library::Provider::kCUTLASS,
        library::Provider::kCUDNN);
    }
  }
  catch (...) {
    results_.back().verification_map[library::Provider::kCUDNN] = Disposition::kFailed;
  }

  // Return true means continue profiling
  return true;

}

#endif // #if CUTLASS_ENABLE_CUDNN

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
