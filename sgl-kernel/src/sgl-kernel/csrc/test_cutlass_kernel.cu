#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include "utils.hpp"

void cutlass_fp16_gemm(torch::Tensor& out, torch::Tensor& mat_a, torch::Tensor& mat_b, int m, int n, int k, int64_t lda, int64_t ldb,
                       int64_t ldc) {
  using Gemm = cutlass::gemm::device::Gemm<cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
                                           cutlass::layout::ColumnMajor, cutlass::half_t, cutlass::layout::RowMajor,
                                           float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;

  Gemm gemm_op;
  cutlass::Status status;

  auto a_ptr = static_cast<cutlass::half_t const*>(mat_a.data_ptr());
  auto b_ptr = static_cast<cutlass::half_t const*>(mat_b.data_ptr());
  auto o_ptr = static_cast<cutlass::half_t*>(out.data_ptr());

  float alpha = 1.0f;
  float beta = 0.0f;

  status = gemm_op({{m, n, k}, {a_ptr, lda}, {b_ptr, ldb}, {o_ptr, ldc}, {o_ptr, ldc}, {alpha, beta}});

  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

void test_fp16_mm(torch::Tensor& out, torch::Tensor& mat_a, torch::Tensor& mat_b) {
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(0) == out.size(0) && mat_b.size(1) == out.size(1), "out has incorrect shape");
  TORCH_CHECK(mat_a.scalar_type() == torch::kHalf, "mat_a must be Half");
  TORCH_CHECK(mat_b.scalar_type() == torch::kHalf, "mat_b must be Half");
  TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be Half");

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  cutlass_fp16_gemm(out, mat_a, mat_b, m, n, k, mat_a.stride(0), mat_b.stride(1),
                    out.stride(0));
}
