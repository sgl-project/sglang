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
/*! \file
    \brief Unit tests for epilogues
*/
#pragma once

#include <fstream>

#include "../../common/cutlass_unit_test.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/half.h"
#include "cutlass/complex.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/util/host_tensor_planar_complex.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace kernel {

template <typename Epilogue>
__global__ void epilogue_planar_complex_threadblock(
  typename Epilogue::OutputTileIterator::Params params_D,
  typename Epilogue::OutputTileIterator::Element *ptr_D,
  int64_t imaginary_stride_D,
  typename Epilogue::OutputTileIterator::Params params_C,
  typename Epilogue::OutputTileIterator::Element *ptr_C,
  int64_t imaginary_stride_C,
  typename Epilogue::OutputOp::Params params_output_op,
  cutlass::MatrixCoord problem_size,
  cutlass::TensorRef<
    typename Epilogue::WarpMmaOperator::ElementC, 
    typename Epilogue::WarpMmaOperator::LayoutC> accumulator_ref,
  int64_t imaginary_stride_accum,
  int epilogue_count = 1) {

  __shared__ typename Epilogue::SharedStorage shared_storage;

  int thread_idx = threadIdx.x;
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  //
  // Construct the epilogue
  //

  // Tile iterator writing to output tile
  typename Epilogue::OutputTileIterator iterator_D_real(
    params_D,
    ptr_D,
    problem_size,
    thread_idx
  );

  typename Epilogue::OutputTileIterator iterator_D_imag(
    params_D,
    ptr_D + imaginary_stride_D,
    problem_size,
    thread_idx
  );

  // Tile iterator writing to output tile
  typename Epilogue::OutputTileIterator iterator_C_real(
    params_C,
    ptr_C,
    problem_size,
    thread_idx
  );

  typename Epilogue::OutputTileIterator iterator_C_imag(
    params_C,
    ptr_C + imaginary_stride_C,
    problem_size,
    thread_idx
  );

  // Epilogue operator
  Epilogue epilogue(
    shared_storage, 
    thread_idx, 
    warp_idx, 
    lane_idx);

  //
  // Initialize the accumulators
  //

  int warp_mn = warp_idx % (Epilogue::WarpCount::kM * Epilogue::WarpCount::kN);
  int warp_m = warp_mn % Epilogue::WarpCount::kM;
  int warp_n = warp_mn / Epilogue::WarpCount::kM;

  accumulator_ref.add_coord_offset({
    warp_m * Epilogue::WarpMmaOperator::Shape::kM, 
    warp_n * Epilogue::WarpMmaOperator::Shape::kN});

  //
  // Load accumulators
  //

  typename Epilogue::WarpMmaOperator::IteratorC accumulator_iterator(accumulator_ref, lane_idx);
  
  typename Epilogue::AccumulatorTile accumulators;

  accumulators.clear();

  accumulator_iterator.load(accumulators.real);
  accumulator_iterator.load_with_pointer_offset(accumulators.imag, imaginary_stride_accum);

  //
  // Perform the epilogue operation
  //

  typename Epilogue::OutputOp output_op(params_output_op);

  // Place the epilogue in a loop so assembly is clearly visible
  for (int iter = 0; iter < epilogue_count; ++iter) {
    epilogue(
      output_op, 
      iterator_D_real, 
      iterator_D_imag, 
      accumulators, 
      iterator_C_real, 
      iterator_C_imag); 
  }
}

} // namespace kernel
} // namespace test


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Epilogue_
>
class EpiloguePlanarComplexTestbed {
public:

  using Epilogue = Epilogue_;
  using ElementAccumulator = typename Epilogue::ElementAccumulator;
  using ElementCompute = typename Epilogue::OutputOp::ElementCompute;
  using ElementOutput = typename Epilogue::ElementOutput;
  using OutputOpParams = typename Epilogue::OutputOp::Params;

  using ComplexElementOutput = cutlass::complex<ElementOutput>;
  using ComplexElementAccumulator = cutlass::complex<ElementAccumulator>;
  using ComplexElementCompute = cutlass::complex<ElementCompute>;

public:

  //
  // Data members
  //

  cutlass::MatrixCoord quantized_size;
  cutlass::HostTensorPlanarComplex<ElementAccumulator, cutlass::layout::RowMajor> accumulator_tensor;
  cutlass::HostTensorPlanarComplex<ElementOutput, cutlass::layout::RowMajor> source_tensor;
  cutlass::HostTensorPlanarComplex<ElementOutput, cutlass::layout::RowMajor> output_tensor;

public:

  //
  // Methods
  //

  EpiloguePlanarComplexTestbed(): 
    quantized_size(Epilogue::Shape::kM, Epilogue::Shape::kN),
    accumulator_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}),
    source_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}),
    output_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}) {

    //
    // Initialize problem space
    //

    #if 1
    uint64_t seed = 2019;

    cutlass::reference::host::TensorFillRandomUniform(
      accumulator_tensor.host_view(), 
      seed, 
      20, 
      -20, 
      0);

    cutlass::reference::host::TensorFillRandomUniform(
      source_tensor.host_view(),
      seed + 2018, 
      20, 
      -20, 
      0);
    #else

    cutlass::reference::host::BlockFillSequential(accumulator_tensor.host_data(), accumulator_tensor.capacity());

    #endif
  }

  bool run_all() {
   
    cutlass::complex<float> alpha_values[3];

    alpha_values[0] = cutlass::complex<float>(1, 0);
    alpha_values[1] = cutlass::complex<float>(0, 0);
    alpha_values[2] = cutlass::complex<float>(2.25f, -0.5f);

    cutlass::complex<float> beta_values[3];

    beta_values[0] = cutlass::complex<float>(0, 0);
    beta_values[1] = cutlass::complex<float>(1, 0);
    beta_values[2] = cutlass::complex<float>(0.5f, -2.25f);

    // Test runtime explodes if we tried to test every case exhaustively. This tests the full
    // output tile and several smaller sizes to stress predication.
    for (int m_idx = 0; m_idx < 3; ++m_idx) {
      for (int n_idx = 0; n_idx < 3; ++n_idx) {

        cutlass::MatrixCoord problem_size(
          quantized_size.row() - m_idx * 3,
          quantized_size.column() - n_idx * Epilogue::kElementsPerAccess
        );

        for (auto const &alpha : alpha_values) {
          for (auto const &beta : beta_values) {

            bool passed = run(problem_size, {alpha, beta});

            if (!passed) {
              return false;
            }
          }
        }
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
    cutlass::MatrixCoord problem_size,
    OutputOpParams output_params) { 

    //
    // Initialize problem space
    //

    ComplexElementOutput default_output = ComplexElementOutput(ElementOutput(-127), ElementOutput(-101));

    cutlass::reference::host::TensorFill(output_tensor.host_view(), default_output);

    accumulator_tensor.sync_device();
    output_tensor.sync_device();
    source_tensor.sync_device();

    //
    // Initialize epilogue parameters
    //

    typename Epilogue::OutputTileIterator::Params params_D(output_tensor.layout());
    typename Epilogue::OutputTileIterator::Params params_C(source_tensor.layout());

    //
    // Launch kernel
    //

    dim3 grid(1, 1);
    dim3 block(Epilogue::WarpCount::kCount * 32, 1);

    test::kernel::epilogue_planar_complex_threadblock<Epilogue><<< grid, block >>>(
      params_D,
      output_tensor.device_data(),
      output_tensor.imaginary_stride(),
      params_C,
      source_tensor.device_data(),
      source_tensor.imaginary_stride(),
      output_params,
      problem_size, 
      accumulator_tensor.device_view_real(),
      accumulator_tensor.imaginary_stride()
    );

    cudaError_t result = cudaDeviceSynchronize();

    if (result != cudaSuccess) {
      std::cerr << "Kernel error: " << cudaGetErrorString(result) << std::endl;
      return false;
    }

    //
    // Verify results
    //
    output_tensor.sync_host();

    int errors = 0;
    int const kMaxErrors = 5;

    for (int r = 0; errors < kMaxErrors && r < quantized_size.row(); ++r) {
      for (int c = 0; errors < kMaxErrors && c < quantized_size.column(); ++c) {

        cutlass::MatrixCoord coord{r, c};
        ComplexElementOutput got = output_tensor.at(coord);
        
        ComplexElementOutput expected = default_output;

        if (coord.row() < problem_size.row() && coord.column() < problem_size.column()) {

          ComplexElementOutput src = source_tensor.at(coord);

          ComplexElementCompute tmp = 
            output_params.alpha * ComplexElementCompute(accumulator_tensor.at(coord)) + 
            output_params.beta * ComplexElementCompute(src.real(), src.imag());

          expected = ComplexElementOutput(ElementOutput(tmp.real()), ElementOutput(tmp.imag()));
        }

        if (expected != got) {

          using OutputIO = cutlass::ScalarIO<ComplexElementOutput>;

          EXPECT_TRUE(false)
            << "-------\n"
            << "Error - output element (" << coord << ") - expected: " 
            << OutputIO(expected) 
            << ",  got: " << OutputIO(got) << std::endl;

          ++errors;
        }
      }
    }

    //
    // Report results on error
    //

    if (errors) {


      std::cout << "Incorrect result for problem(" 
      << problem_size.row() << ", " 
      << problem_size.column() << ") for alpha: " << output_params.alpha << ", beta: " << output_params.beta << std::endl;

      std::stringstream ss;
      ss 
        << "output_tensor_op_" << Epilogue::Shape::kM << "x" << Epilogue::Shape::kN << "_" 
        << Epilogue::WarpTileIterator::WarpShape::kM << "x" 
        << Epilogue::WarpTileIterator::WarpShape::kN 
        << "_slice_" << Epilogue::WarpCount::kK << ".csv"; 

      std::ofstream output_file(ss.str()); 
      output_file << output_tensor.host_view(); 

      std::cout << "Wrote workspace to '" << ss.str() << "'" << std::endl;
    }

    return !errors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
