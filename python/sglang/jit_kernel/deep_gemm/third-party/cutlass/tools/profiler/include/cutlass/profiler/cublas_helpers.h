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
   \brief Helper functions for mapping CUTLASS concepts to cuBLAS.
*/

#pragma once

#if CUTLASS_ENABLE_CUBLAS
#include <cublas_v2.h>
#include <cublasLt.h>

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass/library/util.h"
#include "cutlass/blas3.h"

#include "options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a cuBLAS status to cutlass::Status
Status get_cutlass_status(cublasStatus_t cublas);

/// Converts a cuBLAS status to cutlass::profiler::Disposition
Disposition get_cutlass_disposition(cublasStatus_t cublas_status);

/// Maps a CUTLASS tensor layout to a cuBLAS transpose operation
bool get_cublas_transpose_operation(
  cublasOperation_t &operation,
  library::LayoutTypeID layout,
  library::ComplexTransform transform = library::ComplexTransform::kNone);

/// Maps a CUTLASS numeric type to a cuBLAS data type enumeration
bool get_cublas_datatype(cublasDataType_t &data_type, library::NumericTypeID element_type);

/// Gets the cublas algorithm given threadblock tile dimensions and math opcode class
cublasGemmAlgo_t get_cublas_gemm_algo(
  int cta_m, 
  int cta_n, 
  int cta_k, 
  library::OpcodeClassID opcode_class);

/// Returns a status if cuBLAS can satisfy a particular GEMM description
Status cublas_satisfies(library::GemmDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular RankK description
Status cublas_satisfies(library::RankKDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular TRMM description
Status cublas_satisfies(library::TrmmDescription const &desc);

/// Returns a status if cuBLAS can satisfy a particular SYMM/HEMM description
Status cublas_satisfies(library::SymmDescription const &desc);

/// This is a helper class to create cublasHandle_t automatically on CublasCreate object creation and 
/// to destroy cublasHandle_t on CublasCreate object destruction. 
/// Additionally, it provides implicit cast from CublasCreate's object to cublasHandle_t's object
class CublasCreate {
private:
  cublasHandle_t handle;
  cublasStatus_t status;

public:
  CublasCreate() {
    status = cublasCreate(&handle);
  }

  ~CublasCreate() {
    cublasDestroy(handle);
  }

  /// Implicit cast CublasCreate object to cublasHandle_t
  operator cublasHandle_t() const { return handle; }

  /// returns cublasStatus_t for handle creation
  cublasStatus_t get_cublas_create_status() { return status; }
};

/// This is a helper class to create cublasLtHandle_t automatically on CublasLtCreate object creation and 
/// to destroy cublasLtHandle_t on CublasLtCreate object destruction. 
/// Additionally, it provides implicit cast from CublasLtCreate's object to cublasLtHandle_t's object
class CublasLtCreate {
private:
  cublasLtHandle_t handle;
  cublasStatus_t status;

public:
  CublasLtCreate() {
    status = cublasLtCreate(&handle);
  }

  ~CublasLtCreate() {
    cublasLtDestroy(handle);
  }

  /// Implicit cast CublasLtCreate object to cublasLtHandle_t
  operator cublasLtHandle_t() const { return handle; }

  /// returns cublasLtStatus_t for handle creation
  cublasStatus_t get_cublaslt_create_status() { return status; }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Selects one or more cuBLAS algorithms.
static void select_cublas_algorithms(
  std::vector<cublasGemmAlgo_t> &algorithms,
  Options const &options, 
  library::GemmDescription const &op_desc) {

  library::OpcodeClassID const & opcode_class = 
    op_desc.tile_description.math_instruction.opcode_class;

  switch (options.library.algorithm_mode) {
    case AlgorithmMode::kMatching:
    {
      algorithms.push_back(get_cublas_gemm_algo(
        op_desc.tile_description.threadblock_shape.m(), 
        op_desc.tile_description.threadblock_shape.n(), 
        op_desc.tile_description.threadblock_shape.k(), 
        opcode_class));
      break;
    }

    case AlgorithmMode::kBest:
    {
      // Choose first enumerated mode. If none are enumerated, choose based on opcode class
      // and evaluate all of them.

      if (options.library.algorithms.empty()) {
        // Enumerate all algorithms
        if (opcode_class == library::OpcodeClassID::kSimt) {
          
          for (int algo = CUBLAS_GEMM_DEFAULT; 
            algo <= CUBLAS_GEMM_ALGO23; 
            ++algo) {

            algorithms.push_back(cublasGemmAlgo_t(algo));
          }
        }
        else {
          
          for (int algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; 
            algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP; 
            ++algo) {

            algorithms.push_back(cublasGemmAlgo_t(algo));
          }
        }
      }
      else {
        // Use the listed algorithms
        algorithms.reserve(options.library.algorithms.size());

        for (int algo : options.library.algorithms) {
          algorithms.push_back(reinterpret_cast<cublasGemmAlgo_t const &>(algo));
        }
      }

      break;
    }

    case AlgorithmMode::kDefault:
    {

      // Use the library's default algorithm
      algorithms.push_back((opcode_class == library::OpcodeClassID::kSimt ? 
        CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP)); 

      break;
    }
    default:
    {
      break;
    }
  }
}

/// Dispatcher to cublasGemmEx() 
struct cublasGemmExDispatcher {

  //
  // Data members
  //
  library::GemmUniversalConfiguration configuration;
  library::GemmUniversalArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  cublasOperation_t trans_A;
  cublasOperation_t trans_B;
  cudaDataType_t data_type_A;
  cudaDataType_t data_type_B;
  cudaDataType_t data_type_C;
  cudaDataType_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  cublasComputeType_t compute_type;
#endif

  cublasGemmAlgo_t algo;
  Status status;
  
  //
  // Methods
  //

  cublasGemmExDispatcher( 
    library::GemmDescription const &op_desc,
    library::GemmUniversalConfiguration configuration_,
    library::GemmUniversalArguments arguments_,
    cublasGemmAlgo_t algorithm = CUBLAS_GEMM_DFALT
  );

  /// Executes GEMM using these arguments
  cublasStatus_t operator()(cublasHandle_t handle);
};

/// Dispatcher to cublaslt kernels 
//
struct cublasLtGemmExDispatcher {

  //
  // Data members
  //
  library::GemmDescription const &op_desc;
  library::GemmUniversalConfiguration configuration;
  library::GemmUniversalArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  cublasOperation_t trans_A;
  cublasOperation_t trans_B;
  cudaDataType_t data_type_A;
  cudaDataType_t data_type_B;
  cudaDataType_t data_type_C;
  cudaDataType_t compute_data_type = CUDA_R_32F;

  //cublasLt-specific data structures
  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;
  
  //is set by call to get_cublaslt_algo()
  cublasLtMatmulHeuristicResult_t heuristicResult_;
  void *workspace = nullptr;

  Status status;

#if (__CUDACC_VER_MAJOR__ >= 11)
  cublasComputeType_t compute_type;
#endif

  //
  // Methods
  //

  cublasLtGemmExDispatcher( 
    library::GemmDescription const &op_desc,
    library::GemmUniversalConfiguration configuration_,
    library::GemmUniversalArguments arguments_
  );

  /// Initialize the cublasLt variables
  void initialize_cublaslt();
  

  /// Runs auto-tuning for the cublas heuristics
  bool get_cublaslt_algo(cublasLtHandle_t handle,
    AlgorithmMode algorithm_mode 
    ); 

  /// Executes GEMM using these arguments
  cublasStatus_t operator()(cublasLtHandle_t handle, cudaStream_t stream = nullptr);

  ~cublasLtGemmExDispatcher(){

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);

    if (workspace) {
      cudaFree(workspace);
    }

  } 

};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublas rank k update kernels 
struct cublasRankKDispatcher {

  //
  // Data members
  //
  library::RankKConfiguration configuration;
  library::RankKArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  cublasOperation_t trans_A;
  cublasFillMode_t uplo;
  cudaDataType_t data_type_A;
  cudaDataType_t data_type_C;
  cudaDataType_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  cublasComputeType_t compute_type;
#endif

  int num_ranks;       //(rank-k or rank-2k)
  BlasMode blas_mode; //(symmetric or hermitian)
  Status status;
  
  //
  // Methods
  //

  cublasRankKDispatcher( 
    library::RankKDescription const &op_desc,
    library::RankKConfiguration configuration_,
    library::RankKArguments arguments_
  );

  /// Executes RankK using these arguments
  cublasStatus_t operator()(cublasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublasTrmm() 
struct cublasTrmmDispatcher {

  //
  // Data members
  //
  library::TrmmConfiguration configuration;
  library::TrmmArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  cublasOperation_t trans_A;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasDiagType_t diag;
  cudaDataType_t data_type_A;
  cudaDataType_t data_type_B;
  cudaDataType_t data_type_D;
  cudaDataType_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  cublasComputeType_t compute_type;
#endif

  Status status;
  
  //
  // Methods
  //

  cublasTrmmDispatcher( 
    library::TrmmDescription const &op_desc,
    library::TrmmConfiguration configuration_,
    library::TrmmArguments arguments_
  );

  /// Executes TRMM using these arguments
  cublasStatus_t operator()(cublasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Dispatcher to cublas symm/hemm update kernels 
struct cublasSymmDispatcher {

  //
  // Data members
  //
  library::SymmConfiguration configuration;
  library::SymmArguments arguments;

  // cublas-specific data structures to fill cublas API call arguments
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cudaDataType_t data_type_A;
  cudaDataType_t data_type_B;
  cudaDataType_t data_type_C;
  cudaDataType_t compute_data_type;

#if (__CUDACC_VER_MAJOR__ >= 11)
  cublasComputeType_t compute_type;
#endif
  
  BlasMode blas_mode; //(symmetric or hermitian)
  Status status;
  
  //
  // Methods
  //

  cublasSymmDispatcher( 
    library::SymmDescription const &op_desc,
    library::SymmConfiguration configuration_,
    library::SymmArguments arguments_
  );

  /// Executes Symm using these arguments
  cublasStatus_t operator()(cublasHandle_t handle);
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

} // namespace profiler
} // namespace cutlass


#endif // #if CUTLASS_ENABLE_CUBLAS
