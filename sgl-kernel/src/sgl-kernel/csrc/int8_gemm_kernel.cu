#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include "utils.hpp"

using ElementAccumulator = int32_t;     // <- data type of accumulator
using ElementCompute = float;           // <- data type of epilogue operations
using ElementInputA = int8_t;           // <- data type of elements in input matrix A
using ElementInputB = int8_t;           // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;  // <- data type of elements in output matrix D

void cutlass_int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b, int m, int n,
                            int k, int64_t lda, int64_t ldb, int64_t ldc) {
  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombination<ElementOutput,  // Output data type
                                                   128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                   ElementAccumulator,  // Accumulator type
                                                   ElementCompute       // Compute type
                                                   >;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  constexpr int NumStages = 5;

  using Gemm =
      cutlass::gemm::device::Gemm<ElementInputA, cutlass::layout::RowMajor, ElementInputB, cutlass::layout::ColumnMajor,
                                  ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
                                  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape,
                                  InstructionShape, EpilogueOutputOp, SwizzleThreadBlock, NumStages>;

  Gemm gemm_op;
  cutlass::Status status;

  auto a_ptr = static_cast<const int8_t*>(mat_a.data_ptr());
  auto b_ptr = static_cast<const int8_t*>(mat_b.data_ptr());
  auto o_ptr = static_cast<cutlass::half_t*>(out.data_ptr());

  float alpha = 1.0f;
  float beta = 0.0f;

  status = gemm_op({{m, n, k}, {a_ptr, lda}, {b_ptr, ldb}, {o_ptr, ldc}, {o_ptr, ldc}, {alpha, beta}});

  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

void int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                    const torch::Tensor& scales_a, const torch::Tensor& scales_b, std::optional<torch::Tensor> bias) {
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(0) == out.size(0) && mat_b.size(1) == out.size(1), "out has incorrect shape");
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16,
              "out must be Half or BFloat16");

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  cutlass_int8_scaled_mm(out, mat_a, mat_b, m, n, k, mat_a.stride(0), mat_b.stride(1), out.stride(0));
}
