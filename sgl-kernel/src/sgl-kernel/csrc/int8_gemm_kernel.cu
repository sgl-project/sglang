#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "iostream"
#include "utils.hpp"

template <typename OutputType>
void cutlass_int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                            const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                            const c10::optional<torch::Tensor>& bias) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementOutput = OutputType;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  constexpr int NumStages = 5;

  using DefaultGemmConf = cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, ArchTag, ElementInputA,
                                                                          ElementInputB, ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA, cutlass::layout::RowMajor, DefaultGemmConf::kAlignmentA, ElementInputB,
      cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB, ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
      ThreadblockSwizzle, NumStages, true, typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape, GemmKernel_::kThreadCount, AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator, ElementAccumulator, ElementCompute, EpilogueOutputOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  int64_t lda = mat_a.stride(0);
  int64_t ldb = mat_b.stride(1);
  int64_t ldd = out.stride(0);

  ElementOutput* bias_ptr = nullptr;
  int64_t ldc = 0;
  if (bias) {
    bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
    ldc = bias->stride(0);
  }

  typename EpilogueOutputOp::Params linearScalingParams;
  typename EpilogueVisitor::Arguments visitor_args{linearScalingParams};

  typename Gemm::Arguments args{{m, n, k},    {a_ptr, lda},    {b_ptr, ldb}, {b_s_ptr, 0},
                                {a_s_ptr, 0}, {bias_ptr, ldc}, {o_ptr, ldd}, visitor_args};

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

void int8_scaled_mm(torch::Tensor& out, const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                    const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(out.stride(1) == 1, "out must be a row major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(0) == out.size(0) && mat_b.size(1) == out.size(1), "out has incorrect shape");
  TORCH_CHECK(mat_a.stride(0) % 16 == 0, "mat_a.stride(0) must be a multiple of 16 for memory alignment")
  TORCH_CHECK(mat_b.stride(1) % 16 == 0, "mat_b.stride(1) must be a multiple of 16 for memory alignment")
  TORCH_CHECK(out.stride(0) % 8 == 0, "out.stride(0) should be a multiple of 8 for memory alignment")
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(out.scalar_type() == torch::kHalf || out.scalar_type() == torch::kBFloat16,
              "out must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  // if (bias) {
  //   TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched")
  //   TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous")
  // }

  if (out.scalar_type() == torch::kBFloat16) {
    cutlass_int8_scaled_mm<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}
