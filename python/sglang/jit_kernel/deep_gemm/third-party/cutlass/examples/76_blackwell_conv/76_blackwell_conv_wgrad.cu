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

/*! \file
    \brief Simple wgrad convolution example targeting NVIDIA Blackwell SM100 Tensor Core MMA using CUTLASS 3.x APIs.

    This example demonstrate a simple way to instantiate and run a wgrad convolution kernel using the new CUTLASS 3.0
    APIs on NVIDIA Blackwell SM100 architecture.

    The basic computation logic of wgrad convolution kernel is, take 3D convolution as an example:
        Xformed Activation (NZPQK) * Activation (NDHWC) = Weight/Filter (KTRSC)

    where in terms of GEMM perspective,
        Matrix A = Xformed Activation, Matrix B = Activation, Matrix C = Weight/Filter

    This example instantiates a simple wgrad kernel using TMA + UMMA + Warp Specialized design with input and output types are fp16.
    Alpha/beta scaling is supported while fusions like relu/bias/per-channel scaling are not supported in this example.

    Usage:

      $ ./examples/76_blackwell_conv/76_blackwell_conv_wgrad --n=4 --d=1 --h=8 --w=8 --c=64 --k=64 --t=1 --r=3 --s=3 --pad_d=0
        --pad_h=1 --pad_w=1 --stride_d=1 --stride_h=1 --stride_w=1 --dilation_d=1 --dilation_h=1 --dilation_w=1
*/



#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Conv kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// Activation matrix configuration
using         ElementAct  = half_t;                                          // Element type for activation matrix
constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;  // Memory access granularity/alignment of activation matrix in units of elements (up to 16 bytes)

// Weight/Filter matrix configuration
using         ElementFlt  = half_t;                                          // Element type for weight/filter matrix operand
constexpr int AlignmentFlt = 128 / cutlass::sizeof_bits<ElementFlt>::value;  // Memory access granularity/alignment of weight/filter matrix in units of elements (up to 16 bytes)

// Xformed activation matrix configuration
using         ElementXformedAct = half_t;                                    // Element type for xformed activation matrix operand
constexpr int AlignmentXformedAct = 128 / cutlass::sizeof_bits<ElementXformedAct>::value; // Memory access granularity/alignment of xformed activation matrix in units of elements (up to 16 bytes)

// Layout of matrix A/B/C in gemm's perspecitive.
using LayoutA = cutlass::layout::TensorNDHWC;
using LayoutB = cutlass::layout::TensorNDHWC;
using LayoutC = cutlass::layout::TensorKCSRT;

// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ElementCompute      = float;                                           // Element type for internal computation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag
constexpr cutlass::conv::Operator ConvOp = cutlass::conv::Operator::kWgrad;  // Convolution operation

// Kernel Perf config
using TileShape           = Shape<_128,Shape<_128>,Shape<_64>>;              // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>;                                 // Shape of the threadblocks in a cluster

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementFlt, LayoutC, AlignmentFlt,
    ElementFlt, LayoutC, AlignmentFlt,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ConvOp,
    ElementXformedAct, LayoutA, AlignmentXformedAct,
    ElementAct, LayoutB, AlignmentAct,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::conv::collective::KernelScheduleAuto
  >::CollectiveOp;

// Compose into a kernel
using ProblemShape=cutlass::conv::ConvProblemShape<ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
using ConvKernel = cutlass::conv::kernel::ConvUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
  >; 

using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

using StrideC = typename Conv::ConvKernel::StrideC;
using StrideD = typename Conv::ConvKernel::StrideD;

//
// Data members
//

/// Initialization
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<ElementXformedAct> block_A;
cutlass::DeviceAllocation<ElementAct> block_B;
cutlass::DeviceAllocation<ElementFlt> block_C;
cutlass::DeviceAllocation<ElementFlt> block_D;
cutlass::DeviceAllocation<ElementFlt> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int n, d, h, w, c, k, t, r, s, z, p, q;
  int pad_d, pad_h, pad_w;
  int stride_d, stride_h, stride_w;
  int dilation_d, dilation_h, dilation_w;

  Options():
    help(false),
    n(4), d(1), h(8), w(8), c(64), k(64), t(1), r(3), s(3),
    pad_d(0), pad_h(1), pad_w(1),
    stride_d(1), stride_h(1), stride_w(1),
    dilation_d(1), dilation_h(1), dilation_w(1),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("d", d);
    cmd.get_cmd_line_argument("h", h);
    cmd.get_cmd_line_argument("w", w);
    cmd.get_cmd_line_argument("c", c);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("t", t);
    cmd.get_cmd_line_argument("r", r);
    cmd.get_cmd_line_argument("s", s);
    cmd.get_cmd_line_argument("pad_d", pad_d);
    cmd.get_cmd_line_argument("pad_h", pad_h);
    cmd.get_cmd_line_argument("pad_w", pad_w);
    cmd.get_cmd_line_argument("stride_d", stride_d);
    cmd.get_cmd_line_argument("stride_h", stride_h);
    cmd.get_cmd_line_argument("stride_w", stride_w);
    cmd.get_cmd_line_argument("dilation_d", dilation_d);
    cmd.get_cmd_line_argument("dilation_h", dilation_h);
    cmd.get_cmd_line_argument("dilation_w", dilation_w);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    // Calculate z,p,q based on inputs.
    z = 1 + (d + 2 * pad_d - ((t - 1) * dilation_d + 1)) / stride_d;
    p = 1 + (h + 2 * pad_h - ((r - 1) * dilation_h + 1)) / stride_h;
    q = 1 + (w + 2 * pad_w - ((s - 1) * dilation_w + 1)) / stride_w;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "76_blackwell_conv_wgrad\n\n"
      << "  Blackwell FP16 wgrad convolution using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --n=<int>                   Sets the batch size of the Activation\n"
      << "  --d=<int>                   Sets the depth size of the Activation\n"
      << "  --h=<int>                   Sets the height of the Activation\n"
      << "  --w=<int>                   Sets the width of the Activation\n"
      << "  --c=<int>                   Sets the channel size of the Activation\n"
      << "  --k=<int>                   Sets the image numbers of the Filter\n"
      << "  --t=<int>                   Sets the depth size of the Filter\n"
      << "  --r=<int>                   Sets the height of the Filter\n"
      << "  --s=<int>                   Sets the width of the Filter\n"
      << "  --pad_d=<int>               Sets the padding size in depth\n"
      << "  --pad_h=<int>               Sets the padding size in height\n"
      << "  --pad_w=<int>               Sets the padding size in width\n"
      << "  --stride_d=<int>            Sets the traversal stride size in depth\n"
      << "  --stride_h=<int>            Sets the traversal stride size in height\n"
      << "  --stride_w=<int>            Sets the traversal stride size in width\n"
      << "  --dialtion_d=<int>          Sets the filter dilation size in depth\n"
      << "  --dialtion_h=<int>          Sets the filter dilation size in height\n"
      << "  --dialtion_w=<int>          Sets the filter dilation size in width\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "76_blackwell_conv_wgrad" << " --n=4 --d=1 --h=8 --w=8 --c=64 --k=64 --t=1 --r=3 --s=3 --pad_d=0"
      << "  --pad_h=1 --pad_w=1 --stride_d=1 --stride_h=1 --stride_w=1 --dilation_d=1 --dilation_h=1 --dilation_w=1 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * k * (t * r * s * c) * (n * z * p * q);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Conv setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the Conv and reference Conv
void initialize(const Options &options) {

  // Construct ConvProblemShape
  ProblemShape problem_shape(
    cutlass::conv::Mode::kCrossCorrelation,
    {options.n, options.d, options.h, options.w, options.c},      // ndhwc
    {options.k, options.t, options.r, options.s, options.c},      // ktrsc
    {options.pad_d, options.pad_h, options.pad_w},                // padding lower (pad_d, pad_h, pad_w)
    {options.pad_d, options.pad_h, options.pad_w},                // padding upper (pad_d, pad_h, pad_w)
    {options.stride_d, options.stride_h, options.stride_w},       // stride (stride_d, stride_h, stride_w)
    {options.dilation_d, options.dilation_h, options.dilation_w}, // dilation (dilation_d, dilation_h, dilation_w)
    1                                                             // group
  );

  // Setup stride_C/D
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, problem_shape.shape_C, problem_shape.stride_C, ConvOp);
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, problem_shape.shape_C, problem_shape.stride_C, ConvOp);

  block_A.reset(problem_shape.size_A());
  block_B.reset(problem_shape.size_B());
  block_C.reset(problem_shape.size_C());
  block_D.reset(problem_shape.size_C());
  block_ref_D.reset(problem_shape.size_C());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Conv::Arguments args_from_options(const Options &options)
{
  // Construct ConvProblemShape
  ProblemShape problem_shape(
    cutlass::conv::Mode::kCrossCorrelation,
    {options.n, options.d, options.h, options.w, options.c},      // ndhwc
    {options.k, options.t, options.r, options.s, options.c},      // ktrsc
    {options.pad_d, options.pad_h, options.pad_w},                // padding lower (pad_d, pad_h, pad_w)
    {options.pad_d, options.pad_h, options.pad_w},                // padding upper (pad_d, pad_h, pad_w)
    {options.stride_d, options.stride_h, options.stride_w},       // stride (stride_d, stride_h, stride_w)
    {options.dilation_d, options.dilation_h, options.dilation_w}, // dilation (dilation_d, dilation_h, dilation_w)
    1                                                             // group
  );

  typename Conv::Arguments arguments{
    problem_shape,
    {block_A.get(), block_B.get()},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({options.n, options.z, options.p, options.q, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({options.n, options.d, options.h, options.w, options.c}));
  cutlass::TensorRef ref_C(block_C.get(), LayoutA::packed({options.k, options.t, options.r, options.s, options.c}));
  cutlass::TensorRef ref_D(block_ref_D.get(), LayoutA::packed({options.k, options.t, options.r, options.s, options.c}));

  //
  // Compute reference output
  //

  // Construct Conv3dProblemSize with user defined inputs.
  cutlass::conv::Conv3dProblemSize problem_size(      
    cutlass::Tensor5DCoord(options.n, options.d, options.h, options.w, options.c),      // ndhwc
    cutlass::Tensor5DCoord(options.k, options.t, options.r, options.s, options.c),      // ktrsc
    cutlass::make_Coord(options.pad_d, options.pad_h, options.pad_w),                   // padding
    cutlass::make_Coord(options.stride_d, options.stride_h, options.stride_w),          // stride (stride_d, stride_h, stride_w)
    cutlass::make_Coord(options.dilation_d, options.dilation_h, options.dilation_w),    // dilation (dilation_d, dilation_h, dilation_w)
    cutlass::Tensor5DCoord(options.n, options.z, options.p, options.q, options.k)       // nzpqk
  );

  // Launch device reference conv kernel
  cutlass::reference::device::Conv3dWgrad(problem_size, ref_A, ref_B, ref_C, ref_D, options.alpha, options.beta);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Conv conv;

  // Create a structure of conv kernel arguments suitable for invoking an instance of Conv
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Conv::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(conv.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(conv.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(conv.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(conv.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(conv.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size:" << std::endl;
    std::cout << "      Activation(n,d,h,w,c) = (" << options.n << ',' << options.d << ',' << options.h << ',' << options.w << ',' << options.c << "), ";
    std::cout << "  Filter(k,t,r,s,c) = (" << options.k << ',' << options.t << ',' << options.r << ',' << options.s << ',' << options.c << "), ";
    std::cout << "  Xformed Activation(n,z,p,q,k) = (" << options.n << ',' << options.z << ',' << options.p << ',' << options.q << ',' << options.k << ")" << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  
  if (__CUDACC_VER_MAJOR__ < 13) {
    if (props.major != 10 && (props.minor != 0 || props.minor != 1)) {
      std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100 or 101)." << std::endl;
      return 0;
    } 
  }
  else {
    if ((props.major != 10 || props.major != 11) && props.minor != 0) {
      std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100 or 110)." << std::endl;
      return 0;
    }
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Conv>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
