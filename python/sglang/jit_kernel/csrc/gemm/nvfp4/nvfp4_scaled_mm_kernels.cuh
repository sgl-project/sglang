/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_map>

using namespace host;

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                        \
  {                                                                                  \
    cutlass::Status error = status;                                                  \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;

// Helper function for next power of 2
inline uint32_t next_pow_2(uint32_t x) {
  if (x == 0) return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
// Config(half_t/bfloat16_t) for M <= 128
template <typename T>
struct KernelConfigM128 {
  using OutputType = T;
  using MmaTileShape = Shape<_128, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM128<T>::preferred_cluster(1, 4, 1);
template <typename T>
const dim3 KernelConfigM128<T>::fallback_cluster(1, 2, 1);

// Config(half_t/bfloat16_t) for M <= 256
template <typename T>
struct KernelConfigM256 {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigM256<T>::preferred_cluster(2, 4, 1);
template <typename T>
const dim3 KernelConfigM256<T>::fallback_cluster(2, 1, 1);

// Default config(half_t/bfloat16_t) for M > 256
template <typename T>
struct KernelConfigDefault {
  using OutputType = T;
  using MmaTileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = Shape<_128, _64>;  // Avoid register spilling
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
template <typename T>
const dim3 KernelConfigDefault<T>::preferred_cluster(4, 4, 1);
template <typename T>
const dim3 KernelConfigDefault<T>::fallback_cluster(2, 1, 1);

struct KernelConfigFp32 {
  using OutputType = float;
  using MmaTileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<int, int, _1>;
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;
  const static dim3 preferred_cluster;
  const static dim3 fallback_cluster;
};
const dim3 KernelConfigFp32::preferred_cluster = dim3(1, 4, 1);
const dim3 KernelConfigFp32::fallback_cluster = dim3(1, 2, 1);

// SM120 specific configurations
struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename KernelConfig>
struct Fp4GemmSm100 {
  using Config = KernelConfig;  // For generating args
  using OutputType = typename KernelConfig::OutputType;
  // A matrix configuration
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  // B matrix configuration
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // C/D matrix configuration
  using ElementD = OutputType;
  using ElementC = OutputType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Kernel functional config
  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Kernel Perf config
  using MmaTileShape = typename KernelConfig::MmaTileShape;
  using ClusterShape = typename KernelConfig::ClusterShape;
  using EpilogueTile = typename KernelConfig::EpilogueTile;
  using EpilogueSchedule = typename KernelConfig::EpilogueSchedule;
  using MainloopSchedule = typename KernelConfig::MainloopSchedule;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      void,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA = decltype(cute::make_layout(make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB = decltype(cute::make_layout(make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutC = decltype(cute::make_layout(make_shape(0, 0, 0), StrideC{}));
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using LayoutD = decltype(cute::make_layout(make_shape(0, 0, 0), StrideD{}));
};

// SM120 specific GEMM template
template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape_MNK,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename T>
typename T::Gemm::Arguments args_from_options(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t M,
    int64_t N,
    int64_t K) {
  using ElementA = typename T::Gemm::ElementA;
  using ElementB = typename T::Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = typename T::Gemm::ElementD;
  using ElementCompute = float;
  using StrideA = typename T::StrideA;
  using StrideB = typename T::StrideB;
  using StrideD = typename T::StrideD;
  using Sm1xxBlkScaledConfig = typename T::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename T::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       nullptr,
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());
  using KernelConfig = typename T::Config;
  arguments.hw_info.cluster_shape = KernelConfig::preferred_cluster;
  arguments.hw_info.cluster_shape_fallback = KernelConfig::fallback_cluster;
  return arguments;
}

template <typename T>
void runGemm(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename T::Gemm gemm;
  auto arguments = args_from_options<T>(D, A, B, A_sf, B_sf, alpha, m, n, k);

  size_t workspace_size = T::Gemm::get_workspace_size(arguments);
  int device_id = A.device().device_id;
  void* workspace = get_cached_workspace(workspace_size, device_id, stream);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

// SM120 specific args_from_options function
template <typename Gemm>
typename Gemm::Arguments args_from_options_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int M,
    int N,
    int K) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()),
       layout_SFB},
      {{}, static_cast<ElementD const*>(D.data_ptr()), stride_D, static_cast<ElementD*>(D.data_ptr()), stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  return arguments;
}

// SM120 specific runGemm function
template <typename Gemm>
void runGemmSm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options_sm120<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  int device_id = A.device().device_id;
  void* workspace = get_cached_workspace(workspace_size, device_id, stream);

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace, stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace, stream));
}

// Dispatch function to select appropriate config based on M
template <typename OutType>
void cutlassFp4GemmDispatch(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  if (m <= 128) {
    // m in [1, 128]
    runGemm<Fp4GemmSm100<KernelConfigM128<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else if (m <= 256) {
    // m in (128, 256]
    runGemm<Fp4GemmSm100<KernelConfigM256<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    // m in (256, inf)
    runGemm<Fp4GemmSm100<KernelConfigDefault<OutType>>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

// Dispatch function to select appropriate config based on M
template <>
void cutlassFp4GemmDispatch<float>(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  runGemm<Fp4GemmSm100<KernelConfigFp32>>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
}

// SM120 specific dispatch functions
void cutlass_fp4_bf16_gemm_dispatch_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

void cutlass_fp4_f16_gemm_dispatch_sm120(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemmSm120<Fp4GemmSm120<sm120_fp4_config_default, cutlass::half_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

#else
template <typename T>
void cutlassFp4GemmDispatch(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  RuntimeCheck(
      false,
      "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
      "a CUTLASS 3.8 source directory to enable support.");
}
#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||
        // defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

inline int getSMVersion(int device_id) {
  int sm_major = 0;
  int sm_minor = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device_id));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device_id));
  return sm_major * 10 + sm_minor;
}

void cutlass_scaled_fp4_mm_sm100a_sm120a(
    tvm::ffi::TensorView D,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView B,
    tvm::ffi::TensorView A_sf,
    tvm::ffi::TensorView B_sf,
    tvm::ffi::TensorView alpha) {
  RuntimeCheck(A.device().device_type == kDLCUDA, "a must be a CUDA tensor");
  RuntimeCheck(B.device().device_type == kDLCUDA, "b must be a CUDA tensor");
  RuntimeCheck(A_sf.device().device_type == kDLCUDA, "scale_a must be a CUDA tensor");
  RuntimeCheck(B_sf.device().device_type == kDLCUDA, "scale_b must be a CUDA tensor");
  RuntimeCheck(alpha.device().device_type == kDLCUDA, "alpha must be a CUDA tensor");
  RuntimeCheck(D.device().device_type == kDLCUDA, "out must be a CUDA tensor");

  RuntimeCheck(A.device() == B.device(), "a and b must be on same device");
  RuntimeCheck(A.device() == A_sf.device(), "a and scale_a must be on same device");
  RuntimeCheck(A.device() == B_sf.device(), "a and scale_b must be on same device");
  RuntimeCheck(A.device() == alpha.device(), "a and alpha must be on same device");
  RuntimeCheck(A.device() == D.device(), "a and out must be on same device");

  RuntimeCheck(A.is_contiguous(), "a must be contiguous");
  RuntimeCheck(B.is_contiguous(), "b must be contiguous");
  RuntimeCheck(A_sf.is_contiguous(), "scale_a must be contiguous");
  RuntimeCheck(B_sf.is_contiguous(), "scale_b must be contiguous");
  RuntimeCheck(alpha.is_contiguous(), "alpha must be contiguous");
  RuntimeCheck(D.is_contiguous(), "out must be contiguous");

  RuntimeCheck(host::is_type<uint8_t>(A.dtype()), "a must be uint8");
  RuntimeCheck(host::is_type<uint8_t>(B.dtype()), "b must be uint8");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(A_sf.dtype()), "scale_a must be float8_e4m3fn");
  RuntimeCheck(host::is_type<fp8_e4m3_t>(B_sf.dtype()), "scale_b must be float8_e4m3fn");
  RuntimeCheck(host::is_type<float>(alpha.dtype()), "alpha must be float32");

  RuntimeCheck(A.dim() == 2, "a must be a matrix");
  RuntimeCheck(B.dim() == 2, "b must be a matrix");
  RuntimeCheck(A_sf.dim() == 2, "scale_a must be a matrix");
  RuntimeCheck(B_sf.dim() == 2, "scale_b must be a matrix");
  RuntimeCheck(alpha.numel() == 1, "alpha must have exactly one element");

  RuntimeCheck(
      A.size(1) == B.size(1),
      "a and b shapes cannot be multiplied (",
      A.size(0),
      "x",
      A.size(1),
      " and ",
      B.size(0),
      "x",
      B.size(1),
      ")");

  const auto m = static_cast<int64_t>(A.size(0));
  const auto n = static_cast<int64_t>(B.size(0));
  const auto k = static_cast<int64_t>(A.size(1) * 2);

  RuntimeCheck(D.dim() == 2, "out must be 2D");
  RuntimeCheck(D.size(0) == m, "out first dim must equal m");
  RuntimeCheck(D.size(1) == n, "out second dim must equal n");

  constexpr int alignment = 32;
  RuntimeCheck(k % alignment == 0, "Expected k to be divisible by ", alignment, ", but got k: ", k);
  RuntimeCheck(n % alignment == 0, "Expected n to be divisible by ", alignment, ", but got n: ", n);

  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  const int64_t rounded_m = round_up(m, 128);
  const int64_t rounded_n = round_up(n, 128);
  const int64_t rounded_k = round_up(k / 16, 4);

  RuntimeCheck(
      A_sf.size(1) == B_sf.size(1),
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      " and ",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");
  RuntimeCheck(
      A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
      "scale_a must be padded/swizzled to shape (",
      rounded_m,
      "x",
      rounded_k,
      "), got (",
      A_sf.size(0),
      "x",
      A_sf.size(1),
      ")");
  RuntimeCheck(
      B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
      "scale_b must be padded/swizzled to shape (",
      rounded_n,
      "x",
      rounded_k,
      "), got (",
      B_sf.size(0),
      "x",
      B_sf.size(1),
      ")");

  const cudaStream_t stream = LaunchKernel::resolve_device(A.device());
  const int sm_version = getSMVersion(A.device().device_id);

  if (sm_version >= 120) {
    if (host::is_type<fp16_t>(D.dtype())) {
      cutlass_fp4_f16_gemm_dispatch_sm120(
          D, A, B, A_sf, B_sf, alpha, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    } else if (host::is_type<bf16_t>(D.dtype())) {
      cutlass_fp4_bf16_gemm_dispatch_sm120(
          D, A, B, A_sf, B_sf, alpha, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    } else {
      Panic("Unsupported output data type of nvfp4 mm sm120");
    }
  } else {
    if (host::is_type<fp16_t>(D.dtype())) {
      cutlassFp4GemmDispatch<cutlass::half_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (host::is_type<bf16_t>(D.dtype())) {
      cutlassFp4GemmDispatch<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else if (host::is_type<float>(D.dtype())) {
      cutlassFp4GemmDispatch<float>(D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
    } else {
      Panic("Unsupported output data type of nvfp4 mm");
    }
  }
}
