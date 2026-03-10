#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>

using namespace host;
using namespace cute;

struct WorkspaceKey {
  int device_id;
  uintptr_t stream;
  auto operator==(const WorkspaceKey&) const -> bool = default;
};

struct WorkspaceKeyHash {
  auto operator()(const WorkspaceKey& key) const -> size_t {
    size_t h1 = std::hash<int>{}(key.device_id);
    size_t h2 = std::hash<uintptr_t>{}(key.stream);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct WorkspaceState {
  void* ptr = nullptr;
  size_t bytes = 0;
};

inline auto get_cached_workspace(size_t required_bytes, int device_id, cudaStream_t stream) -> void* {
  if (required_bytes == 0) {
    return nullptr;
  }

  thread_local std::unordered_map<WorkspaceKey, WorkspaceState, WorkspaceKeyHash> cache;
  WorkspaceKey key{device_id, reinterpret_cast<uintptr_t>(stream)};
  auto& ws = cache[key];

  if (ws.ptr != nullptr && ws.bytes >= required_bytes) {
    return ws.ptr;
  }

  RuntimeDeviceCheck(cudaSetDevice(device_id));
  if (ws.ptr != nullptr) {
    RuntimeDeviceCheck(cudaFreeAsync(ws.ptr, stream));
    ws.ptr = nullptr;
    ws.bytes = 0;
  }
  RuntimeDeviceCheck(cudaMallocAsync(&ws.ptr, required_bytes, stream));
  ws.bytes = required_bytes;
  return ws.ptr;
}

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}

template <
    typename ElementAB,
    typename ElementC,
    typename ElementSF,
    typename ElementAccumulator,
    typename LayoutSFA,
    typename LayoutSFB,
    typename ScaleConfig>
__global__ void __get_group_gemm_starts(
    ElementAB** a_offsets,
    ElementAB** b_offsets,
    ElementC** out_offsets,
    ElementSF** a_scales_offsets,
    ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets,
    LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int,
    ElementAB* a_base_as_int,
    ElementAB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int,
    ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int,
    const int32_t* expert_offsets,
    const int32_t* sf_offsets,
    const int32_t* problem_sizes_as_shapes,
    const int K,
    const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  // Originally int32_t but upcasting to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  // size for block in block scale.
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert((m >= 0 && n == N && k == K && k % 2 == 0) && "unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / group_size);
  // Shape of A as uint8/byte = [M, K // 2]
  // Shape of B as uint8/byte = [E, N, K // 2]
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;

  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  // Shape of a_scale = [sum(sf_sizes), K // group_size]
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) == 0 && "TMA requires 128-byte alignment");

  // Shape of B scale = [E, N, K // group_size]
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) == 0 && "TMA requires 128-byte alignment");
  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

#define __CALL_GET_STARTS_KERNEL_BLOCKSCALE(                                                            \
    ELEMENT_AB_TYPE, SF_TYPE, TYPE_CHECK, C_TYPE, LayoutSFA, LayoutSFB, ScaleConfig)                    \
  else if (TYPE_CHECK) {                                                                                \
    __get_group_gemm_starts<ELEMENT_AB_TYPE, C_TYPE, SF_TYPE, float, LayoutSFA, LayoutSFB, ScaleConfig> \
        <<<1, num_experts, 0, stream>>>(                                                                \
            static_cast<ELEMENT_AB_TYPE**>(a_starts.data_ptr()),                                        \
            static_cast<ELEMENT_AB_TYPE**>(b_starts.data_ptr()),                                        \
            static_cast<C_TYPE**>(out_starts.data_ptr()),                                               \
            static_cast<SF_TYPE**>(a_scales_starts.data_ptr()),                                         \
            static_cast<SF_TYPE**>(b_scales_starts.data_ptr()),                                         \
            static_cast<float**>(alpha_starts.data_ptr()),                                              \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),                                        \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),                                        \
            static_cast<ELEMENT_AB_TYPE*>(a_tensors.data_ptr()),                                        \
            static_cast<ELEMENT_AB_TYPE*>(b_tensors.data_ptr()),                                        \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                                               \
            static_cast<SF_TYPE*>(a_scales.data_ptr()),                                                 \
            static_cast<SF_TYPE*>(b_scales.data_ptr()),                                                 \
            static_cast<float*>(alphas.data_ptr()),                                                     \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                                           \
            static_cast<int32_t*>(sf_offsets.data_ptr()),                                               \
            static_cast<int32_t*>(problem_sizes.data_ptr()),                                            \
            K,                                                                                          \
            N);                                                                                         \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts(
    const tvm::ffi::TensorView a_starts,
    const tvm::ffi::TensorView b_starts,
    const tvm::ffi::TensorView out_starts,
    const tvm::ffi::TensorView a_scales_starts,
    const tvm::ffi::TensorView b_scales_starts,
    const tvm::ffi::TensorView alpha_starts,
    const tvm::ffi::TensorView layout_sfa,
    const tvm::ffi::TensorView layout_sfb,
    /*these are used for their base addresses*/
    tvm::ffi::TensorView const& a_tensors,
    tvm::ffi::TensorView const& b_tensors,
    tvm::ffi::TensorView const& out_tensors,
    tvm::ffi::TensorView const& a_scales,
    tvm::ffi::TensorView const& b_scales,
    tvm::ffi::TensorView const& alphas,
    tvm::ffi::TensorView const& expert_offsets,
    tvm::ffi::TensorView const& sf_offsets,
    tvm::ffi::TensorView const& problem_sizes,
    int M,
    int N,
    int K) {
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto stream = LaunchKernel::resolve_device(a_tensors.device());

  RuntimeCheck(out_tensors.size(1) == N, "Output tensor shape doesn't match expected shape");
  RuntimeCheck(
      K / 2 == b_tensors.size(2),
      "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
      " dimension must match");
  if (false) {
  }
  //(ELEMENT_AB_TYPE, BS_TYPE, TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB,
  // ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(
      cutlass::float_e2m1_t,
      cutlass::float_ue4m3_t,
      host::is_type<bf16_t>(out_tensors.dtype()),
      cutlass::bfloat16_t,
      LayoutSFA,
      LayoutSFB,
      ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE(
      cutlass::float_e2m1_t,
      cutlass::float_ue4m3_t,
      host::is_type<fp16_t>(out_tensors.dtype()),
      cutlass::half_t,
      LayoutSFA,
      LayoutSFB,
      ScaleConfig)
  else {
    Panic("Invalid output type (must be float16 or bfloat16)");
  }
}

void run_fp4_blockwise_scaled_group_mm_sm120(
    tvm::ffi::TensorView output,
    const tvm::ffi::TensorView a,
    const tvm::ffi::TensorView b,
    const tvm::ffi::TensorView a_blockscale,
    const tvm::ffi::TensorView b_blockscales,
    const tvm::ffi::TensorView alphas,
    const tvm::ffi::TensorView ab_strides,
    const tvm::ffi::TensorView c_strides,
    const tvm::ffi::TensorView problem_sizes,
    const tvm::ffi::TensorView expert_offsets,
    const tvm::ffi::TensorView sf_offsets,
    const tvm::ffi::TensorView a_ptrs,
    const tvm::ffi::TensorView b_ptrs,
    const tvm::ffi::TensorView out_ptrs,
    const tvm::ffi::TensorView a_scales_ptrs,
    const tvm::ffi::TensorView b_scales_ptrs,
    const tvm::ffi::TensorView alpha_ptrs,
    const tvm::ffi::TensorView layout_sfa,
    const tvm::ffi::TensorView layout_sfb,
    int M,
    int N,
    int K) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using ThreadBlockShape = Shape<_128, _128, _128>;
  // on the tile size

  using ClusterShape = Shape<_1, _1, _1>;

  using FusionOperation =
      cutlass::epilogue::fusion::LinearCombination<ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ThreadBlockShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA*,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      ElementAccumulator,
      ThreadBlockShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));

  run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      alpha_ptrs,
      layout_sfa,
      layout_sfb,
      a,
      b,
      output,
      a_blockscale,
      b_blockscales,
      alphas,
      expert_offsets,
      sf_offsets,
      problem_sizes,
      M,
      N,
      K);

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Set the Scheduler info
  cutlass::KernelHardwareInfo hw_info;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.device().device_id;
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  hw_info.sm_count = std::min(cached_sm_counts[hw_info.device_id], std::numeric_limits<int>::max());

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(ab_strides.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(ab_strides.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array = reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = 0.0f;

  // Gemm Arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());
  void* workspace = get_cached_workspace(workspace_size, hw_info.device_id, stream);

  auto can_implement_status = gemm_op.can_implement(args);
  RuntimeCheck(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM: ",
      cutlassGetStatusString(can_implement_status));

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to initialize GEMM: ", cutlassGetStatusString(status));

  status = gemm_op.run(args, workspace, stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to run GEMM: ", cutlassGetStatusString(status));
}

template <typename OutType>
void run_fp4_blockwise_scaled_group_mm_sm100(
    tvm::ffi::TensorView output,
    const tvm::ffi::TensorView a,
    const tvm::ffi::TensorView b,
    const tvm::ffi::TensorView a_blockscale,
    const tvm::ffi::TensorView b_blockscales,
    const tvm::ffi::TensorView alphas,
    const tvm::ffi::TensorView ab_strides,
    const tvm::ffi::TensorView c_strides,
    const tvm::ffi::TensorView problem_sizes,
    const tvm::ffi::TensorView expert_offsets,
    const tvm::ffi::TensorView sf_offsets,
    const tvm::ffi::TensorView a_ptrs,
    const tvm::ffi::TensorView b_ptrs,
    const tvm::ffi::TensorView out_ptrs,
    const tvm::ffi::TensorView a_scales_ptrs,
    const tvm::ffi::TensorView b_scales_ptrs,
    const tvm::ffi::TensorView alpha_ptrs,
    const tvm::ffi::TensorView layout_sfa,
    const tvm::ffi::TensorView layout_sfb,
    int M,
    int N,
    int K) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassTensorOp;             // Epilogue Operator class tag
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;  // Mainloop Operator class tag
  using StageCountType = cutlass::gemm::collective::StageCountAuto;         // Stage count maximized based
                                                                            // on the tile size

  using ClusterShape = Shape<_1, _1, _1>;
  struct MMA1SMConfig {
    using MmaTileShape = Shape<_128, _128, _128>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100;  // Kernel to launch
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;           // Epilogue to launch
  };

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      EpilogueOperatorClass,
      typename MMA1SMConfig::MmaTileShape,
      ClusterShape,
      Shape<_128, _64>,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC*,
      AlignmentC,
      ElementD,
      LayoutC*,
      AlignmentD,
      typename MMA1SMConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      MainloopOperatorClass,
      ElementA,
      LayoutA*,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      ElementAccumulator,
      typename MMA1SMConfig::MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename MMA1SMConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Gemm = Gemm1SM;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));

  run_get_group_gemm_starts<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      alpha_ptrs,
      layout_sfa,
      layout_sfb,
      a,
      b,
      output,
      a_blockscale,
      b_blockscales,
      alphas,
      expert_offsets,
      sf_offsets,
      problem_sizes,
      M,
      N,
      K);

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes = static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Set the Scheduler info
  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
      typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.device().device_id;
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }
  hw_info.sm_count = std::min(cached_sm_counts[hw_info.device_id], std::numeric_limits<int>::max());

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(ab_strides.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(ab_strides.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array = reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};

  // Gemm Arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  const cudaStream_t stream = LaunchKernel::resolve_device(a.device());
  void* workspace = get_cached_workspace(workspace_size, hw_info.device_id, stream);

  auto can_implement_status = gemm_op.can_implement(args);
  RuntimeCheck(
      can_implement_status == cutlass::Status::kSuccess,
      "Failed to implement GEMM: ",
      cutlassGetStatusString(can_implement_status));

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to initialize GEMM: ", cutlassGetStatusString(status));

  status = gemm_op.run(args, workspace, stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Failed to run GEMM: ", cutlassGetStatusString(status));
}

void cutlass_fp4_group_mm_sm100a_sm120a(
    tvm::ffi::TensorView output,
    const tvm::ffi::TensorView a,
    const tvm::ffi::TensorView b,
    const tvm::ffi::TensorView a_blockscale,
    const tvm::ffi::TensorView b_blockscales,
    const tvm::ffi::TensorView alphas,
    const tvm::ffi::TensorView ab_strides,
    const tvm::ffi::TensorView c_strides,
    const tvm::ffi::TensorView problem_sizes,
    const tvm::ffi::TensorView expert_offsets,
    const tvm::ffi::TensorView sf_offsets,
    const tvm::ffi::TensorView a_ptrs,
    const tvm::ffi::TensorView b_ptrs,
    const tvm::ffi::TensorView out_ptrs,
    const tvm::ffi::TensorView a_scales_ptrs,
    const tvm::ffi::TensorView b_scales_ptrs,
    const tvm::ffi::TensorView alpha_ptrs,
    const tvm::ffi::TensorView layout_sfa,
    const tvm::ffi::TensorView layout_sfb) {
  auto check_cuda_contig = [](const tvm::ffi::TensorView t, const char* name) {
    RuntimeCheck(t.device().device_type == kDLCUDA, name, " must be a CUDA tensor");
    RuntimeCheck(t.is_contiguous(), name, " must be contiguous");
  };

  check_cuda_contig(output, "output");
  check_cuda_contig(a, "a");
  check_cuda_contig(b, "b");
  check_cuda_contig(a_blockscale, "a_blockscale");
  check_cuda_contig(b_blockscales, "b_blockscales");
  check_cuda_contig(alphas, "alphas");
  check_cuda_contig(ab_strides, "ab_strides");
  check_cuda_contig(c_strides, "c_strides");
  check_cuda_contig(problem_sizes, "problem_sizes");
  check_cuda_contig(expert_offsets, "expert_offsets");
  check_cuda_contig(sf_offsets, "sf_offsets");
  check_cuda_contig(a_ptrs, "a_ptrs");
  check_cuda_contig(b_ptrs, "b_ptrs");
  check_cuda_contig(out_ptrs, "out_ptrs");
  check_cuda_contig(a_scales_ptrs, "a_scales_ptrs");
  check_cuda_contig(b_scales_ptrs, "b_scales_ptrs");
  check_cuda_contig(alpha_ptrs, "alpha_ptrs");
  check_cuda_contig(layout_sfa, "layout_sfa");
  check_cuda_contig(layout_sfb, "layout_sfb");

  RuntimeCheck(
      output.device() == a.device() && a.device() == b.device() && a.device() == a_blockscale.device() &&
          a.device() == b_blockscales.device() && a.device() == alphas.device() && a.device() == ab_strides.device() &&
          a.device() == c_strides.device() && a.device() == problem_sizes.device() &&
          a.device() == expert_offsets.device() && a.device() == sf_offsets.device() && a.device() == a_ptrs.device() &&
          a.device() == b_ptrs.device() && a.device() == out_ptrs.device() && a.device() == a_scales_ptrs.device() &&
          a.device() == b_scales_ptrs.device() && a.device() == alpha_ptrs.device() &&
          a.device() == layout_sfa.device() && a.device() == layout_sfb.device(),
      "all tensors must be on the same device");

  RuntimeCheck(host::is_type<uint8_t>(a.dtype()), "a must be uint8");
  RuntimeCheck(host::is_type<uint8_t>(b.dtype()), "b must be uint8");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(a_blockscale.dtype()), "a_blockscale must be float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(b_blockscales.dtype()), "b_blockscales must be float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(alphas.dtype()), "alphas must be float32");
  RuntimeCheck(host::is_type<int64_t>(ab_strides.dtype()), "ab_strides must be int64");
  RuntimeCheck(host::is_type<int64_t>(c_strides.dtype()), "c_strides must be int64");
  RuntimeCheck(host::is_type<int32_t>(problem_sizes.dtype()), "problem_sizes must be int32");
  RuntimeCheck(host::is_type<int32_t>(expert_offsets.dtype()), "expert_offsets must be int32");
  RuntimeCheck(host::is_type<int32_t>(sf_offsets.dtype()), "sf_offsets must be int32");
  RuntimeCheck(host::is_type<int64_t>(a_ptrs.dtype()), "a_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(b_ptrs.dtype()), "b_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(out_ptrs.dtype()), "out_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(a_scales_ptrs.dtype()), "a_scales_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(b_scales_ptrs.dtype()), "b_scales_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(alpha_ptrs.dtype()), "alpha_ptrs must be int64");
  RuntimeCheck(host::is_type<int64_t>(layout_sfa.dtype()), "layout_sfa must be int64");
  RuntimeCheck(host::is_type<int64_t>(layout_sfb.dtype()), "layout_sfb must be int64");
  RuntimeCheck(
      host::is_type<bf16_t>(output.dtype()) || host::is_type<fp16_t>(output.dtype()),
      "output must be bfloat16 or float16");

  RuntimeCheck(a.dim() == 2, "a must be 2D");
  RuntimeCheck(b.dim() == 3, "b must be 3D");
  RuntimeCheck(a_blockscale.dim() == 2, "a_blockscale must be 2D");
  RuntimeCheck(b_blockscales.dim() == 3, "b_blockscales must be 3D");
  RuntimeCheck(alphas.dim() == 1, "alphas must be 1D");
  RuntimeCheck(ab_strides.dim() == 1, "ab_strides must be 1D");
  RuntimeCheck(c_strides.dim() == 1, "c_strides must be 1D");
  RuntimeCheck(problem_sizes.dim() == 2, "problem_sizes must be 2D");
  RuntimeCheck(expert_offsets.dim() == 1, "expert_offsets must be 1D");
  RuntimeCheck(sf_offsets.dim() == 1, "sf_offsets must be 1D");
  RuntimeCheck(a_ptrs.dim() == 1, "a_ptrs must be 1D");
  RuntimeCheck(b_ptrs.dim() == 1, "b_ptrs must be 1D");
  RuntimeCheck(out_ptrs.dim() == 1, "out_ptrs must be 1D");
  RuntimeCheck(a_scales_ptrs.dim() == 1, "a_scales_ptrs must be 1D");
  RuntimeCheck(b_scales_ptrs.dim() == 1, "b_scales_ptrs must be 1D");
  RuntimeCheck(alpha_ptrs.dim() == 1, "alpha_ptrs must be 1D");
  RuntimeCheck(layout_sfa.dim() == 2, "layout_sfa must be 2D");
  RuntimeCheck(layout_sfb.dim() == 2, "layout_sfb must be 2D");
  RuntimeCheck(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");

  const int num_experts = static_cast<int>(expert_offsets.size(0));
  RuntimeCheck(problem_sizes.size(0) == num_experts, "problem_sizes size mismatch with expert_offsets");
  RuntimeCheck(sf_offsets.size(0) == num_experts, "sf_offsets size mismatch with expert_offsets");
  RuntimeCheck(alphas.size(0) == num_experts, "alphas size mismatch with expert_offsets");
  RuntimeCheck(ab_strides.size(0) == num_experts, "ab_strides size mismatch with expert_offsets");
  RuntimeCheck(c_strides.size(0) == num_experts, "c_strides size mismatch with expert_offsets");
  RuntimeCheck(a_ptrs.size(0) == num_experts, "a_ptrs size mismatch with expert_offsets");
  RuntimeCheck(b_ptrs.size(0) == num_experts, "b_ptrs size mismatch with expert_offsets");
  RuntimeCheck(out_ptrs.size(0) == num_experts, "out_ptrs size mismatch with expert_offsets");
  RuntimeCheck(a_scales_ptrs.size(0) == num_experts, "a_scales_ptrs size mismatch with expert_offsets");
  RuntimeCheck(b_scales_ptrs.size(0) == num_experts, "b_scales_ptrs size mismatch with expert_offsets");
  RuntimeCheck(alpha_ptrs.size(0) == num_experts, "alpha_ptrs size mismatch with expert_offsets");
  RuntimeCheck(layout_sfa.size(0) == num_experts && layout_sfa.size(1) == 5, "layout_sfa must be [num_experts, 5]");
  RuntimeCheck(layout_sfb.size(0) == num_experts && layout_sfb.size(1) == 5, "layout_sfb must be [num_experts, 5]");

  int M = static_cast<int>(a.size(0));
  int N = static_cast<int>(b.size(1));
  int K = static_cast<int>(2 * b.size(2));
  RuntimeCheck(output.dim() == 2, "output must be 2D");
  RuntimeCheck(output.size(0) == M && output.size(1) == N, "output shape mismatch");

  auto sm_version = getSMVersion(a.device().device_id);
  if (sm_version == 100 || sm_version == 103) {
    if (host::is_type<bf16_t>(output.dtype())) {
      run_fp4_blockwise_scaled_group_mm_sm100<cutlass::bfloat16_t>(
          output,
          a,
          b,
          a_blockscale,
          b_blockscales,
          alphas,
          ab_strides,
          c_strides,
          problem_sizes,
          expert_offsets,
          sf_offsets,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          alpha_ptrs,
          layout_sfa,
          layout_sfb,
          M,
          N,
          K);
    } else {
      run_fp4_blockwise_scaled_group_mm_sm100<cutlass::half_t>(
          output,
          a,
          b,
          a_blockscale,
          b_blockscales,
          alphas,
          ab_strides,
          c_strides,
          problem_sizes,
          expert_offsets,
          sf_offsets,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          alpha_ptrs,
          layout_sfa,
          layout_sfb,
          M,
          N,
          K);
    }
  } else if (sm_version >= 120) {
    if (host::is_type<bf16_t>(output.dtype())) {
      run_fp4_blockwise_scaled_group_mm_sm120(
          output,
          a,
          b,
          a_blockscale,
          b_blockscales,
          alphas,
          ab_strides,
          c_strides,
          problem_sizes,
          expert_offsets,
          sf_offsets,
          a_ptrs,
          b_ptrs,
          out_ptrs,
          a_scales_ptrs,
          b_scales_ptrs,
          alpha_ptrs,
          layout_sfa,
          layout_sfb,
          M,
          N,
          K);
    } else {
      Panic("SM120 path currently supports only bfloat16 output");
    }
  } else {
    RuntimeCheck(false, "Unsupported SM version: ", sm_version);
  }
}

void cutlass_fp4_group_mm(
    tvm::ffi::TensorView output,
    const tvm::ffi::TensorView a,
    const tvm::ffi::TensorView b,
    const tvm::ffi::TensorView a_blockscale,
    const tvm::ffi::TensorView b_blockscales,
    const tvm::ffi::TensorView alphas,
    const tvm::ffi::TensorView ab_strides,
    const tvm::ffi::TensorView c_strides,
    const tvm::ffi::TensorView problem_sizes,
    const tvm::ffi::TensorView expert_offsets,
    const tvm::ffi::TensorView sf_offsets,
    const tvm::ffi::TensorView a_ptrs,
    const tvm::ffi::TensorView b_ptrs,
    const tvm::ffi::TensorView out_ptrs,
    const tvm::ffi::TensorView a_scales_ptrs,
    const tvm::ffi::TensorView b_scales_ptrs,
    const tvm::ffi::TensorView alpha_ptrs,
    const tvm::ffi::TensorView layout_sfa,
    const tvm::ffi::TensorView layout_sfb) {
  cutlass_fp4_group_mm_sm100a_sm120a(
      output,
      a,
      b,
      a_blockscale,
      b_blockscales,
      alphas,
      ab_strides,
      c_strides,
      problem_sizes,
      expert_offsets,
      sf_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      alpha_ptrs,
      layout_sfa,
      layout_sfb);
}
