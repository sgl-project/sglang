#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/gemm/collective/collective_builder.hpp"
#include "cutlass_extensions/gemm/dispatch_policy.hpp"
#include "utils.h"

using namespace cute;

template <typename OutType, typename TileShape, typename ClusterShape, int ScaleGranularityM = 1>
void launch_sm90_fp8_blockwise_scaled_mm(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
                                         const torch::Tensor& scales_a, const torch::Tensor& scales_b) {
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutType>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = AlignmentC;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<ScaleGranularityM>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueSchedule, StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                           CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;

  int m = a.size(0);
  int k = a.size(1);
  int n = b.size(1);

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB*>(b.data_ptr());
  auto o_ptr = static_cast<ElementD*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementBlockScale*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementBlockScale*>(scales_b.data_ptr());

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, stride_a, b_ptr, stride_b, 4, a_s_ptr, b_s_ptr};
  typename GemmKernel::EpilogueArguments epilogue_args{{}, nullptr, stride_d, o_ptr, stride_d};

  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      mainloop_args,
      epilogue_args,
  };

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess, cutlassGetStatusString(can_implement))

  auto status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status))
}

template <typename OutType>
void sm90_fp8_blockwise_dispatch_shape(torch::Tensor& out, const torch::Tensor& a, const torch::Tensor& b,
                                       const torch::Tensor& scales_a, const torch::Tensor& scales_b) {
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  launch_sm90_fp8_blockwise_scaled_mm<OutType, TileShape, ClusterShape>(out, a, b, scales_a, scales_b);
}

torch::Tensor fp8_blockwise_scaled_mm(const torch::Tensor& mat_a, const torch::Tensor& mat_b,
                                      const torch::Tensor& scales_a, const torch::Tensor& scales_b,
                                      const torch::Dtype& out_dtype) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK((mat_a.size(1) * mat_a.element_size()) % 16 == 0,
              "mat_a must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK((mat_b.size(0) * mat_b.element_size()) % 16 == 0,
              "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  auto is_contiguous_vector = [](const torch::Tensor& t) {
    auto t_sizes = t.sizes();
    return t.is_contiguous() &&
           (t.dim() == 1 || (t.dim() == 2 && *std::min_element(t_sizes.begin(), t_sizes.end()) == 1));
  };

  TORCH_CHECK(mat_a.size(0) == scales_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(mat_a.size(1) / 128 == scales_a.size(1), "size of scales_a is not matched");
  TORCH_CHECK(scales_a.stride(0) == 1 || is_contiguous_vector(scales_a), "scales_a must be M major");
  TORCH_CHECK(mat_b.size(0) / 128 == scales_b.size(0), "size of scales_b is not matched");
  TORCH_CHECK(mat_b.size(1) / 128 == scales_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_b.stride(0) == 1 || is_contiguous_vector(scales_b), "scales_b must be K major");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  auto sm_version = getSMVersion();

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version >= 90) {
    if (out_dtype == torch::kBFloat16) {
      sm90_fp8_blockwise_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b);
    } else {
      sm90_fp8_blockwise_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b);
    }
    return out;
  }
#endif
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(false,
                              "No implemented fp8_blockwise_scaled_mm for current compute capability: ", sm_version);
}
