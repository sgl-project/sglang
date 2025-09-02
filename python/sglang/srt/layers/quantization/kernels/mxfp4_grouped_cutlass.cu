// ===============================================
// File: mxfp4_grouped_cutlass.cu
// Desc: CUTLASS 3.x SM120 FP4 weight-only GEMM implementation
//       Tuned for RTX 5090 (Blackwell) with large-N MoE workloads
// ===============================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "mxfp4_grouped_common.h"
#include <vector>
#include <stdint.h>
#include <assert.h>
#include <cstdlib>
#include <string>
#include <stdexcept>

// Only compile CUTLASS code if flag is set
#ifdef USE_CUTLASS_FP4

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cute/tensor.hpp>  // needed for `using namespace cute;`

// CUTLASS 3.x namespaces
using namespace cute;

#endif // USE_CUTLASS_FP4

#ifdef USE_CUTLASS_FP4

// ===============================================
// CUTLASS 3.x Kernel Configuration for SM120 FP4
// ===============================================

namespace cutlass_mxfp4 {

// Tile configuration tuned for large-N MoE
// 128x256x64 with TMA warp-specialized ping-pong
using TileShape_MNK = Shape<_128, _256, _64>;
using ClusterShape_MNK = Shape<_1, _2, _1>;

// SM120 architecture tag
using ArchTag = cutlass::arch::Sm120;

// Operand types
using ElementA = cutlass::bfloat16_t;       // Input X
using ElementB = cutlass::float_e2m1_t;     // Packed FP4 weights
using ElementC = cutlass::bfloat16_t;       // Output Y
using ElementAccumulator = float;           // Accumulator precision

// Layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// Special layout for packed weights - RowMajorInterleaved<64>
// This improves TMA efficiency for FP4 packed data
template<int Interleave>
struct RowMajorInterleaved {
  static constexpr int kInterleave = Interleave;
  using UnderlyingLayout = cutlass::layout::RowMajor;
  
  template<class Shape>
  using TVLayout = decltype(
    make_layout(
      make_shape(get<0>(Shape{}), get<1>(Shape{}) / kInterleave, Int<kInterleave>{}),
      make_stride(get<1>(Shape{}) * Int<kInterleave>{}, Int<kInterleave>{}, _1{})
    )
  );
};

// Select layout based on pack_layout flag
template<int PackLayout>
using LayoutB = typename std::conditional<
  PackLayout == 0,
  cutlass::layout::RowMajor,
  RowMajorInterleaved<64>
>::type;

// Alignment settings for TMA
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

// Build the GEMM kernel using CollectiveBuilder
template<int PackLayout>
struct GemmKernel {
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB<PackLayout>, AlignmentB,
    ElementAccumulator,
    TileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename cutlass::gemm::collective::SharedStorage<
        ElementA, ElementB, TileShape_MNK, ClusterShape_MNK
      >))
    >,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong  // TMA warp-specialized for SM120
  >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentA,
    ElementC, LayoutC, AlignmentA,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Weight-only dequantization functor
template<typename Element, typename Scale>
struct MxFp4Dequantize {
  const Scale* scales;
  int group_size;
  
  CUTLASS_DEVICE
  MxFp4Dequantize(const Scale* scales_, int group_size_) 
    : scales(scales_), group_size(group_size_) {}
  
  CUTLASS_DEVICE
  Element operator()(const cutlass::float_e2m1_t& packed, int row, int col) const {
    // Dequantize FP4 to BF16 using per-group scale
    int scale_idx = col / group_size;
    Scale scale = scales[row * (col / group_size) + scale_idx];
    
    // Convert E2M1 to float and apply scale
    float val = static_cast<float>(packed) * static_cast<float>(scale);
    return static_cast<Element>(val);
  }
};

// CUTLASS runner for a single GEMM
template<int PackLayout>
void run_cutlass_gemm(const GroupedDesc& desc, cudaStream_t stream) {
  using Gemm = typename GemmKernel<PackLayout>::Gemm;
  
  // Problem size
  cutlass::gemm::GemmCoord problem_size(desc.M, desc.N, desc.K);
  
  // Initialize GEMM arguments
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    1,  // batch_count
    {1.0f, 0.0f},  // alpha, beta
    desc.X,
    desc.Wq, 
    nullptr,  // bias (not used)
    desc.Y,
    desc.Y,   // D = C
    desc.lda,
    desc.ldwq,
    0,  // ldc (unused)
    desc.ldy,
    desc.ldy
  };
  
  // Create GEMM instance
  Gemm gemm_op;
  
  // Get workspace size
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  
  // Allocate workspace if needed
  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&workspace, workspace_size);
  }
  
  // Initialize and run
  cutlass::Status status = gemm_op.initialize(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to initialize CUTLASS GEMM");
  }
  
  status = gemm_op.run(stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to run CUTLASS GEMM");
  }
  
  // Free workspace
  if (workspace) {
    cudaFree(workspace);
  }
}

} // namespace cutlass_mxfp4

#endif // USE_CUTLASS_FP4

// ===============================================
// FlashInfer Backend (Alternative)
// ===============================================

#ifdef USE_FLASHINFER_BACKEND
extern "C" {
  // FlashInfer's FP4 weight-only routine
  void flashinfer_fp4_gemm(
    const void* x, const void* w, const void* scales, void* y,
    int M, int N, int K, int group_size,
    cudaStream_t stream);
}
#endif

// ===============================================
// Main Launcher
// ===============================================

void launch_grouped_mxfp4_weightonly(
    const std::vector<GroupedDesc>& descs,
    int sm_arch, cudaStream_t stream) {
  
  // Validate SM architecture
  if (sm_arch < 120) {
    // Warning: Sub-optimal on older GPUs
    // Could fall back to different kernel here
  }
  
  for (const auto& d : descs) {
#ifdef USE_CUTLASS_FP4
    // Route to CUTLASS based on pack layout
    if (d.pack_layout == 0) {
      cutlass_mxfp4::run_cutlass_gemm<0>(d, stream);
    } else {
      cutlass_mxfp4::run_cutlass_gemm<1>(d, stream);
    }
#elif defined(USE_FLASHINFER_BACKEND)
    // Route to FlashInfer
    flashinfer_fp4_gemm(
      d.X, d.Wq, d.Scales, d.Y,
      d.M, d.N, d.K, d.group_size,
      stream
    );
#else
    // Hard guard unless explicitly allowed
    const char* ok = std::getenv("SGLANG_ALLOW_STUB_KERNEL");
    if (!ok || std::string(ok) != "1") {
      throw std::runtime_error("mxfp4_grouped: stub backend linked. "
                               "Build with -DUSE_CUTLASS_FP4 or -DUSE_FLASHINFER_BACKEND");
    }
    // Fallback stub - just zero the output for build sanity
    const int64_t size = d.M * d.N;
    cudaMemsetAsync(d.Y, 0, size * sizeof(__nv_bfloat16), stream);
#endif
  }
  
  // Optional: For better performance, batch all descs into single kernel
  // This requires CUTLASS grouped GEMM API or custom batching logic
}