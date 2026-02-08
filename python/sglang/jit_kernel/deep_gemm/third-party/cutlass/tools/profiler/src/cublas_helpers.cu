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

#include <stdexcept>

#if CUTLASS_ENABLE_CUBLAS
#include "cutlass/profiler/cublas_helpers.h"

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a cuBLAS status to cutlass::Status
Status get_cutlass_status(cublasStatus_t cublas) {

  switch (cublas) {
    case CUBLAS_STATUS_SUCCESS: 
      return Status::kSuccess;
    case CUBLAS_STATUS_INVALID_VALUE:
      return Status::kErrorInvalidProblem;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return Status::kErrorNotSupported;
    default: break;
  }
  return Status::kErrorInternal;
}

/// Converts a cuBLAS status to cutlass::profiler::Disposition
Disposition get_cutlass_disposition(cublasStatus_t cublas_status) {

  if (cublas_status == CUBLAS_STATUS_INVALID_VALUE) {
    return Disposition::kInvalidProblem;
  }
  else if (cublas_status == CUBLAS_STATUS_NOT_SUPPORTED) {
    return Disposition::kNotSupported;
  }
  return Disposition::kFailed;
}

/// Maps a CUTLASS tensor layout to a cuBLAS transpose operation
bool get_cublas_transpose_operation(
  cublasOperation_t &operation,
  library::LayoutTypeID layout, 
  library::ComplexTransform transform) {

  switch (layout) {
    case library::LayoutTypeID::kColumnMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = CUBLAS_OP_N;
        return true;
      }
      else {
        return false;
      }
      break;
    case library::LayoutTypeID::kRowMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = CUBLAS_OP_T;
        return true;
      }
      else if (transform == library::ComplexTransform::kConjugate) {
        operation = CUBLAS_OP_C;
        return true;
      }
      break;
    default: break;
  }

  return false;
}

/// Maps a CUTLASS numeric type to a cuBLAS data type enumeration
bool get_cublas_datatype(cublasDataType_t &data_type, library::NumericTypeID element_type) {
  switch (element_type) {
  case library::NumericTypeID::kFE4M3:
#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
    data_type = CUDA_R_8F_E4M3;
    return true;
#endif
    break;
  
  case library::NumericTypeID::kFE5M2:
#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
    data_type = CUDA_R_8F_E5M2;
    return true;
#endif
    break;

  case library::NumericTypeID::kF16:
    data_type = CUDA_R_16F;
    return true;
    
  case library::NumericTypeID::kBF16:
    data_type = CUDA_R_16BF;
    return true;
  
  case library::NumericTypeID::kTF32: 
    break;
  
  case library::NumericTypeID::kF32:
    data_type = CUDA_R_32F;
    return true;
    
  case library::NumericTypeID::kF64: 
    data_type = CUDA_R_64F;
    return true;
  
  case library::NumericTypeID::kS4: 
    break;
  
  case library::NumericTypeID::kS8: 
    data_type = CUDA_R_8I;
    return true;
    
  case library::NumericTypeID::kS16: 
    break;
 
  case library::NumericTypeID::kS32: 
    data_type = CUDA_R_32I;
    return true;
    
  case library::NumericTypeID::kS64: 
    break;
  
  case library::NumericTypeID::kU4: 
    break;
  
  case library::NumericTypeID::kU8: 
    data_type = CUDA_R_8U;
    return true;
    
  case library::NumericTypeID::kU16: 
    break;
    
  case library::NumericTypeID::kU32: 
    data_type = CUDA_R_32U;
    return true;
    
  case library::NumericTypeID::kU64: 
    break;

  case library::NumericTypeID::kB1: 
    break;

  case library::NumericTypeID::kCF32:
    data_type = CUDA_C_32F;
    return true;

  case library::NumericTypeID::kCF64:
    data_type = CUDA_C_64F;
    return true;
  
  case library::NumericTypeID::kInvalid:
  
  default: 
    break;
  }

  return false;
}

/// Maps a cutlass::SideMode to cuBLAS side mode
bool get_cublas_side_mode(cublasSideMode_t& side, SideMode side_mode) {

  switch (side_mode) {
    case SideMode::kLeft: 
      side = CUBLAS_SIDE_LEFT;
      return true;
    case SideMode::kRight: 
      side = CUBLAS_SIDE_RIGHT;
      return true;
    default: break;
  }

  return false;
}

/// Maps a cutlass::FillMode to cuBLAS fill mode
bool get_cublas_fill_mode(cublasFillMode_t& uplo, FillMode fill_mode) {

  switch (fill_mode) {
    case FillMode::kLower: 
      uplo = CUBLAS_FILL_MODE_LOWER;
      return true;
    case FillMode::kUpper: 
      uplo = CUBLAS_FILL_MODE_UPPER;
      return true;
    default: break;
  }

  return false;
}

/// Maps a cutlass::DiagType to cuBLAS diag type
bool get_cublas_diag_type(cublasDiagType_t& diag, DiagType diag_type) {

  switch (diag_type) {
    case DiagType::kNonUnit: 
      diag = CUBLAS_DIAG_NON_UNIT;
      return true;
    case DiagType::kUnit: 
      diag = CUBLAS_DIAG_UNIT;
      return true;
    default: break;
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets the cublas algorithm given threadblock tile dimensions and math opcode class
cublasGemmAlgo_t get_cublas_gemm_algo(int cta_m, int cta_n, int cta_k, library::OpcodeClassID opcode_class) {
  return (opcode_class == library::OpcodeClassID::kSimt ? 
    CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular GEMM description
Status cublas_satisfies(library::GemmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

 // Refer to https://docs.nvidia.com/cuda/cublas/#id105
 // input type A and B FE5M2 not supported in cuBLASLt
  if(desc.A.element == library::NumericTypeID::kFE5M2 &&
    desc.B.element == library::NumericTypeID::kFE5M2){

    return Status::kErrorNotSupported;
  }

 // Refer to https://docs.nvidia.com/cuda/cublas/#id105
 // input type A and B are FE5M2 and FE4M3 then D type should be F32
  if (desc.A.element == library::NumericTypeID::kFE5M2 &&
    desc.B.element == library::NumericTypeID::kFE4M3 &&
    desc.C.element == library::NumericTypeID::kF32 &&
    desc.D.element != library::NumericTypeID::kF32 ){

    return Status::kErrorNotSupported;
  }


  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasGemmExDispatcher::cublasGemmExDispatcher(
  library::GemmDescription const &op_desc,
  library::GemmUniversalConfiguration configuration_,
  library::GemmUniversalArguments arguments_,
  cublasGemmAlgo_t algorithm
):
  configuration(configuration_), arguments(arguments_), algo(algorithm), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_transpose_operation(trans_B, op_desc.B.layout, op_desc.transform_B));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    op_desc.B.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case CUDA_R_32F:
      case CUDA_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_64F:
      case CUDA_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case CUDA_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes GEMM using these arguments
cublasStatus_t cublasGemmExDispatcher::operator()(cublasHandle_t handle) {

  if (configuration.mode == library::GemmUniversalMode::kBatched) {
    return cublasGemmStridedBatchedEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.batch_stride_A,
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.batch_stride_B,
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
      arguments.batch_stride_C,
      configuration.batch_count,
  #if (__CUDACC_VER_MAJOR__ >= 11)
      compute_type,
  #else
      compute_data_type,
  #endif
      algo
    );
  }
  else {
    return cublasGemmEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
  #if (__CUDACC_VER_MAJOR__ >= 11)
      compute_type,
  #else
      compute_data_type,
  #endif
      algo
    );
  }
}


cublasLtGemmExDispatcher::cublasLtGemmExDispatcher(
  library::GemmDescription const &op_desc,
  library::GemmUniversalConfiguration configuration_,
  library::GemmUniversalArguments arguments_
):
  op_desc(op_desc), configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_transpose_operation(trans_B, op_desc.B.layout, op_desc.transform_B));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    op_desc.B.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case CUDA_R_32F:
      case CUDA_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_64F:
      case CUDA_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case CUDA_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

void cublasLtGemmExDispatcher::initialize_cublaslt(){

  // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
  // set the transforms for A and B
  cublasLtMatmulDescCreate(&operationDesc, compute_type, compute_data_type);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_A, sizeof(trans_A));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_B, sizeof(trans_B));

  uint64_t contiguous_A = (trans_A == CUBLAS_OP_N ? configuration.problem_size.m() : configuration.problem_size.k());
  uint64_t strided_A = (trans_A == CUBLAS_OP_N ? configuration.problem_size.k() :  configuration.problem_size.m());
  uint64_t contiguous_B = (trans_B == CUBLAS_OP_N ? configuration.problem_size.k() :  configuration.problem_size.n());
  uint64_t strided_B = (trans_B == CUBLAS_OP_N ? configuration.problem_size.n() :  configuration.problem_size.k());

  // create matrix descriptors, we are good with the details here so no need to set any extra attributes
  // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
  cublasLtMatrixLayoutCreate(&Adesc, data_type_A, contiguous_A, strided_A,  configuration.lda);
  cublasLtMatrixLayoutCreate(&Bdesc, data_type_B, contiguous_B, strided_B,  configuration.ldb);
  cublasLtMatrixLayoutCreate(&Cdesc, data_type_C, configuration.problem_size.m(), configuration.problem_size.n(), configuration.ldc);
  cublasLtMatrixLayoutCreate(&Ddesc, data_type_C, configuration.problem_size.m(), configuration.problem_size.n(), configuration.ldd);

}

bool cublasLtGemmExDispatcher::get_cublaslt_algo(cublasLtHandle_t handle,
                                 AlgorithmMode algorithm_mode
                                 ){
  const int requestedAlgoCount = 8; //By default gets 8 algorithms from GetHeuristic Call. CublasLt heuristics provide at max 8 algorithms. 
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = {};

#if (__CUDACC_VER_MAJOR__ >= 12)
  //Decide based upon the unique operation identifier whether to turn on fast accum for cublas kernel or not.
  std::string operation_name(op_desc.name);
  if(operation_name.find("fastaccum") != std::string::npos){
    const int8_t fastAccuMode = 1;
    cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &fastAccuMode,
        sizeof(fastAccuMode));
  }
#endif // __CUDACC_VER_MAJOR__ >= 12

  //Using 32MB for hopper kernel. This is the max workspace size for the call to cublasLtMatmulAlgoGetHeuristic()
  size_t workspaceSizeForHeuristics = 32ULL * 1024 * 1024;
  void* workspaceHeuristic = nullptr;

  cudaError_t result = cudaMalloc((void **)&workspaceHeuristic, workspaceSizeForHeuristics);
  if (result != cudaSuccess) {
    throw std::bad_alloc();
  }

  // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
  // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSizeForHeuristics, sizeof(workspaceSizeForHeuristics));

  cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount, heuristicResult, &returnedResults);

  if (returnedResults == 0) {
    cudaFree(workspaceHeuristic);
    return false;
  }

  int bestAlgoIdx = 0;
  //
  //Auto Tuning to get the best kernel for the given problem
  //
  if (algorithm_mode == AlgorithmMode::kBest) {
    float time = 0;
    float bestAlgoTime = 0;
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    
    cudaStreamCreate(&stream);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
      
    constexpr int repeatAlgoCheck = 5;
    std::vector<float> algoTimes(repeatAlgoCheck);
    
    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
      for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
        cudaEventRecord(startEvent, stream);
  
        cublasStatus_t status = cublasLtMatmul(handle,
                 operationDesc,
                 arguments.alpha,
                 arguments.A,
                 Adesc,
                 arguments.B,
                 Bdesc,
                 arguments.beta,
                 arguments.C,
                 Cdesc,
                 arguments.D,
                 Ddesc,
                 &heuristicResult[algoIdx].algo,
                 workspaceHeuristic,
                 heuristicResult[algoIdx].workspaceSize,
                 stream);
  
        // Handle errors
        if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << "cublasLtMatmul AutoTuning failed with status: " << cublasLtGetStatusName(status) << std::endl;
          cudaFree(workspaceHeuristic);
          return false;
        }
  
        cudaEventRecord(stopEvent, stream);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        algoTimes[checkIdx] = time;
  
      }
  
      const size_t size = algoTimes.size();
      if (size == 0) {
        time = 0;
      }
    
      std::sort(algoTimes.begin(), algoTimes.end());
    
      const size_t mid = size / 2;
      if (size % 2 == 0) {
        time = (algoTimes[mid] + algoTimes[mid - 1]) / 2;
      }
      else {
        time = algoTimes[mid];
      }
    
      if (algoIdx == 0 || time < bestAlgoTime) {
        bestAlgoTime = time;
        bestAlgoIdx = algoIdx;
      }
    }
  

#if defined(CUTLASS_DEBUG_TRACE_LEVEL) && (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    std::cout << "\n";
    std::cout << "# Algorithms checked: " << returnedResults << "\n";
    std::cout << "WorkspaceSize Allocated: " << heuristicResult[bestAlgoIdx].workspaceSize << "\n";
    std::cout << "Algorithm selected after auto-tuning is:" << "\n";
    
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;
  
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
    cublasLtMatmulAlgoConfigGetAttribute(&heuristicResult[bestAlgoIdx].algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
  
    printf("algo={ Id=%d, tileIdx=%d splitK=%d reduc=%d swizzle=%d custom=%d }\n",
        algoId, tile, numSplitsK, reductionScheme, swizzle, customOption);
#endif

    if (stream) cudaStreamDestroy(stream);
    if (startEvent) cudaEventDestroy(startEvent);
    if (stopEvent) cudaEventDestroy(stopEvent);

  }

  //setting algorithm for the dispatcher
  heuristicResult_ = heuristicResult[bestAlgoIdx];
  result = cudaMalloc((void **)&workspace, heuristicResult_.workspaceSize);
  if (result != cudaSuccess) {
    throw std::bad_alloc();
  }
  
  cudaFree(workspaceHeuristic);
  return true;
}

cublasStatus_t cublasLtGemmExDispatcher::operator()(cublasLtHandle_t handle, cudaStream_t stream)
{
  return cublasLtMatmul(handle,
    operationDesc,
    arguments.alpha,
    arguments.A,
    Adesc,
    arguments.B,
    Bdesc,
    arguments.beta,
    arguments.C,
    Cdesc,
    arguments.D,
    Ddesc,
    &heuristicResult_.algo,
    workspace,
    heuristicResult_.workspaceSize,
    stream); //number of streams is set to 0
  
}

}
// namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular RankK description
Status cublas_satisfies(library::RankKDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasRankKDispatcher::cublasRankKDispatcher(
  library::RankKDescription const &op_desc,
  library::RankKConfiguration configuration_,
  library::RankKArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  blas_mode = op_desc.blas_mode;
  num_ranks = op_desc.num_ranks;

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case CUDA_R_32F:
      case CUDA_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_64F:
      case CUDA_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case CUDA_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes RankK using these arguments
cublasStatus_t cublasRankKDispatcher::operator()(cublasHandle_t handle) {
 
  // SYRK and HERK
  if (num_ranks == 1) {
    if (data_type_A == data_type_C && data_type_A == CUDA_R_64F) {
      return cublasDsyrk(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const double*>(arguments.alpha),
        static_cast<const double*>(arguments.A),
        int(configuration.lda),
        static_cast<const double*>(arguments.beta),
        static_cast<double*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == CUDA_R_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      return cublasSsyrk(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const float*>(arguments.alpha),
        static_cast<const float*>(arguments.A),
        int(configuration.lda),
        static_cast<const float*>(arguments.beta),
        static_cast<float*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == CUDA_C_64F) {
      
        if (blas_mode == BlasMode::kHermitian) {
          return cublasZherk(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            static_cast<const double*>(arguments.alpha),
            static_cast<const cuDoubleComplex*>(arguments.A),
            int(configuration.lda),
            static_cast<const double*>(arguments.beta),
            static_cast<cuDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }    
        else {
          return cublasZsyrk(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            static_cast<const cuDoubleComplex*>(arguments.alpha),
            static_cast<const cuDoubleComplex*>(arguments.A),
            int(configuration.lda),
            static_cast<const cuDoubleComplex*>(arguments.beta),
            static_cast<cuDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }
  
    } else if (data_type_A == data_type_C && data_type_A == CUDA_C_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      if (blas_mode == BlasMode::kHermitian) {
        return cublasCherk(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          static_cast<const float*>(arguments.alpha),
          static_cast<const cuComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const float*>(arguments.beta),
          static_cast<cuComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
      else {
        return cublasCsyrk(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          static_cast<const cuComplex*>(arguments.alpha),
          static_cast<const cuComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const cuComplex*>(arguments.beta),
          static_cast<cuComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
    } else {
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }
  } 

  // SYR2K and HER2K
  else if (num_ranks == 2) {
    if (data_type_A == data_type_C && data_type_A == CUDA_R_64F) {
      return cublasDsyr2k(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const double*>(arguments.alpha),
        static_cast<const double*>(arguments.A),
        int(configuration.lda),
        static_cast<const double*>(arguments.B),
        int(configuration.ldb),
        static_cast<const double*>(arguments.beta),
        static_cast<double*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == CUDA_R_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      return cublasSsyr2k(
        handle,
        uplo,
        trans_A,
        configuration.problem_size.n(),
        configuration.problem_size.k(),
        static_cast<const float*>(arguments.alpha),
        static_cast<const float*>(arguments.A),
        int(configuration.lda),
        static_cast<const float*>(arguments.B),
        int(configuration.ldb),
        static_cast<const float*>(arguments.beta),
        static_cast<float*>(arguments.D),
        int(configuration.ldc)
      );
    } else if (data_type_A == data_type_C && data_type_A == CUDA_C_64F) {
      
        if (blas_mode == BlasMode::kHermitian) {
          return cublasZher2k(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            static_cast<const cuDoubleComplex*>(arguments.alpha),
            static_cast<const cuDoubleComplex*>(arguments.A),
            int(configuration.lda),
            static_cast<const cuDoubleComplex*>(arguments.B),
            int(configuration.ldb),
            static_cast<const double*>(arguments.beta),
            static_cast<cuDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }    
        else {
          return cublasZsyr2k(
            handle,
            uplo,
            trans_A,
            configuration.problem_size.n(),
            configuration.problem_size.k(),
            static_cast<const cuDoubleComplex*>(arguments.alpha),
            static_cast<const cuDoubleComplex*>(arguments.A),
            int(configuration.lda),
            static_cast<const cuDoubleComplex*>(arguments.B),
            int(configuration.ldb),
            static_cast<const cuDoubleComplex*>(arguments.beta),
            static_cast<cuDoubleComplex*>(arguments.D),
            int(configuration.ldc)
          );
        }
  
    } else if (data_type_A == data_type_C && data_type_A == CUDA_C_32F) {
  
  #if (__CUDACC_VER_MAJOR__ >= 11)
      if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
        return CUBLAS_STATUS_NOT_SUPPORTED; 
  #endif
  
      if (blas_mode == BlasMode::kHermitian) {
        return cublasCher2k(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          static_cast<const cuComplex*>(arguments.alpha),
          static_cast<const cuComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const cuComplex*>(arguments.B),
          int(configuration.ldb),
          static_cast<const float*>(arguments.beta),
          static_cast<cuComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
      else {
        return cublasCsyr2k(
          handle,
          uplo,
          trans_A,
          configuration.problem_size.n(),
          configuration.problem_size.k(),
          static_cast<const cuComplex*>(arguments.alpha),
          static_cast<const cuComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const cuComplex*>(arguments.B),
          int(configuration.ldb),
          static_cast<const cuComplex*>(arguments.beta),
          static_cast<cuComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }
    } else {
      return CUBLAS_STATUS_NOT_SUPPORTED;
    }
  }
  else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular TRMM description
Status cublas_satisfies(library::TrmmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.D.element == library::NumericTypeID::kS4 || 
    desc.D.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasTrmmDispatcher::cublasTrmmDispatcher(
  library::TrmmDescription const &op_desc,
  library::TrmmConfiguration configuration_,
  library::TrmmArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_cublas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_cublas_side_mode(side, op_desc.side_mode));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_diag_type(diag, op_desc.diag_type));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_cublas_datatype(data_type_D, op_desc.D.element));

  // if A is Transposed, then for cuBLAS that is inverted Fill Mode. 
  if (trans_A == CUBLAS_OP_T || trans_A == CUBLAS_OP_C) {
    if (uplo == CUBLAS_FILL_MODE_LOWER)
      uplo = CUBLAS_FILL_MODE_UPPER;
    else
      uplo = CUBLAS_FILL_MODE_LOWER;
  }

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case CUDA_R_32F:
      case CUDA_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_64F:
      case CUDA_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case CUDA_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes TRMM using these arguments
cublasStatus_t cublasTrmmDispatcher::operator()(cublasHandle_t handle) {
 
  if (data_type_A == data_type_D && data_type_A == CUDA_R_64F) {
    return cublasDtrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const double*>(arguments.alpha),
      static_cast<const double*>(arguments.A),
      int(configuration.lda),
      static_cast<const double*>(arguments.B),
      int(configuration.ldb),
      static_cast<double*>(arguments.D),
      int(configuration.ldd)
    );
  } else if (data_type_A == data_type_D && data_type_A == CUDA_R_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
      return CUBLAS_STATUS_NOT_SUPPORTED; 
#endif

    return cublasStrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const float*>(arguments.alpha),
      static_cast<const float*>(arguments.A),
      int(configuration.lda),
      static_cast<const float*>(arguments.B),
      int(configuration.ldb),
      static_cast<float*>(arguments.D),
      int(configuration.ldd)
    );
  } else if (data_type_A == data_type_D && data_type_A == CUDA_C_64F) {
    return cublasZtrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const cuDoubleComplex*>(arguments.alpha),
      static_cast<const cuDoubleComplex*>(arguments.A),
      int(configuration.lda),
      static_cast<const cuDoubleComplex*>(arguments.B),
      int(configuration.ldb),
      static_cast<cuDoubleComplex*>(arguments.D),
      int(configuration.ldd)
    );
  } else if (data_type_A == data_type_D && data_type_A == CUDA_C_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
      return CUBLAS_STATUS_NOT_SUPPORTED; 
#endif

    return cublasCtrmm(
      handle,
      side,
      uplo,
      trans_A,
      diag,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const cuComplex*>(arguments.alpha),
      static_cast<const cuComplex*>(arguments.A),
      int(configuration.lda),
      static_cast<const cuComplex*>(arguments.B),
      int(configuration.ldb),
      static_cast<cuComplex*>(arguments.D),
      int(configuration.ldd)
    );
  } else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if cuBLAS can satisfy a particular Symm description
Status cublas_satisfies(library::SymmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in cuBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in cuBLAS
  if (desc.B.element == library::NumericTypeID::kBF16 || 
    desc.B.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  // only column major layout is supported in cuBLAS
  if (desc.A.layout != library::LayoutTypeID::kColumnMajor || 
      desc.transform_A != library::ComplexTransform::kNone) {
  
    return Status::kErrorNotSupported;
}

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

cublasSymmDispatcher::cublasSymmDispatcher(
  library::SymmDescription const &op_desc,
  library::SymmConfiguration configuration_,
  library::SymmArguments arguments_
):
  configuration(configuration_), arguments(arguments_), status(Status::kSuccess) {

  blas_mode = op_desc.blas_mode;

  bool good = true;

  good = (good && get_cublas_side_mode(side, op_desc.side_mode));
  good = (good && get_cublas_fill_mode(uplo, op_desc.fill_mode));
  good = (good && get_cublas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_cublas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_cublas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  // cuBLAS introduces a separate cublasComputeType enumerant to more precisely describe
  // internal numerical data types used in the computation.
#if (__CUDACC_VER_MAJOR__ >= 11)
  library::OpcodeClassID const & opcode_class =
    op_desc.tile_description.math_instruction.opcode_class;

  if (good &&
    op_desc.A.element == library::NumericTypeID::kF32 &&
    opcode_class == library::OpcodeClassID::kTensorOp) {

    compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  else if (good) {
    bool const isPedantic = false;
    switch (compute_data_type) {
      case CUDA_R_32F:
      case CUDA_C_32F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
        break;
      case CUDA_R_64F:
      case CUDA_C_64F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
        break;
      case CUDA_R_16F:
        compute_type = isPedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
        break;
      case CUDA_R_32I:
        compute_type = isPedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
        break;
      default:
        good = false;
        break;
    }
  }
#endif // __CUDACC_VER_MAJOR__ >= 11

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes Symm using these arguments
cublasStatus_t cublasSymmDispatcher::operator()(cublasHandle_t handle) {
 
  // SYMM and HEMM
  if (data_type_A == data_type_C && data_type_A == CUDA_R_64F) {
    return cublasDsymm(
      handle,
      side,
      uplo,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const double*>(arguments.alpha),
      static_cast<const double*>(arguments.A),
      int(configuration.lda),
      static_cast<const double*>(arguments.B),
      int(configuration.ldb),
      static_cast<const double*>(arguments.beta),
      static_cast<double*>(arguments.D),
      int(configuration.ldc)
    );
  } else if (data_type_A == data_type_C && data_type_A == CUDA_R_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
      return CUBLAS_STATUS_NOT_SUPPORTED; 
#endif

    return cublasSsymm(
      handle,
      side,
      uplo,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      static_cast<const float*>(arguments.alpha),
      static_cast<const float*>(arguments.A),
      int(configuration.lda),
      static_cast<const float*>(arguments.B),
      int(configuration.ldb),
      static_cast<const float*>(arguments.beta),
      static_cast<float*>(arguments.D),
      int(configuration.ldc)
    );
  } else if (data_type_A == data_type_C && data_type_A == CUDA_C_64F) {
    
      if (blas_mode == BlasMode::kHermitian) {
        return cublasZhemm(
          handle,
          side,
          uplo,
          configuration.problem_size.m(),
          configuration.problem_size.n(),
          static_cast<const cuDoubleComplex*>(arguments.alpha),
          static_cast<const cuDoubleComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const cuDoubleComplex*>(arguments.B),
          int(configuration.ldb),
          static_cast<const cuDoubleComplex*>(arguments.beta),
          static_cast<cuDoubleComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }    
      else {
        return cublasZsymm(
          handle,
          side,
          uplo,
          configuration.problem_size.m(),
          configuration.problem_size.n(),
          static_cast<const cuDoubleComplex*>(arguments.alpha),
          static_cast<const cuDoubleComplex*>(arguments.A),
          int(configuration.lda),
          static_cast<const cuDoubleComplex*>(arguments.B),
          int(configuration.ldb),
          static_cast<const cuDoubleComplex*>(arguments.beta),
          static_cast<cuDoubleComplex*>(arguments.D),
          int(configuration.ldc)
        );
      }

  } else if (data_type_A == data_type_C && data_type_A == CUDA_C_32F) {

#if (__CUDACC_VER_MAJOR__ >= 11)
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS)
      return CUBLAS_STATUS_NOT_SUPPORTED; 
#endif

    if (blas_mode == BlasMode::kHermitian) {
      return cublasChemm(
        handle,
        side,
        uplo,
        configuration.problem_size.m(),
        configuration.problem_size.n(),
        static_cast<const cuComplex*>(arguments.alpha),
        static_cast<const cuComplex*>(arguments.A),
        int(configuration.lda),
        static_cast<const cuComplex*>(arguments.B),
        int(configuration.ldb),
        static_cast<const cuComplex*>(arguments.beta),
        static_cast<cuComplex*>(arguments.D),
        int(configuration.ldc)
      );
    }
    else {
      return cublasCsymm(
        handle,
        side,
        uplo,
        configuration.problem_size.m(),
        configuration.problem_size.n(),
        static_cast<const cuComplex*>(arguments.alpha),
        static_cast<const cuComplex*>(arguments.A),
        int(configuration.lda),
        static_cast<const cuComplex*>(arguments.B),
        int(configuration.ldb),
        static_cast<const cuComplex*>(arguments.beta),
        static_cast<cuComplex*>(arguments.D),
        int(configuration.ldc)
      );
    }
  } else {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

#endif // #if CUTLASS_ENABLE_CUBLAS
