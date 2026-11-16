// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/utils.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <tvm/ffi/container/tensor.h>

#include <type_traits>

namespace sglang::inkling2_nvfp4 {

using namespace cute;
using Problems = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using FP4 = cutlass::float_e2m1_t;
using FP8 = cutlass::float_ue4m3_t;

// kSwap: weights are A (M = n) and token rows are B (N = rows), so D is the
// column-major (n, rows) view of the row-major (rows, n) output.
template <
    class Arch,
    class Tile,
    class Schedule,
    bool kSwap,
    class Cluster = Shape<_1, _1, _1>,
    class EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>
struct Config {
  using LayoutD = std::conditional_t<kSwap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      void,
      LayoutD*,
      0,
      float,
      LayoutD*,
      4,
      EpilogueSchedule>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassBlockScaledTensorOp,
      tuple<FP4, FP8>,
      cutlass::layout::RowMajor*,
      32,
      tuple<FP4, FP8>,
      cutlass::layout::ColumnMajor*,
      32,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Problems, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using LayoutSFA = typename Mainloop::InternalLayoutSFA;
  using LayoutSFB = typename Mainloop::InternalLayoutSFB;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  static constexpr bool swap = kSwap;
};

// SM103 runs the Ultra (K = 768) kernels; other Blackwell parts (SM100) the
// 1-SM NVFP4 kernels with the same scale layout. 128x192 Wide tiles serve
// prefill-sized row counts.
#if SGL_CUDA_ARCH == 1030
using Default = Config<
    cutlass::arch::Sm103,
    Shape<_128, _128, Int<768>>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103,
    false>;
using Wide = Config<
    cutlass::arch::Sm103,
    Shape<_128, Int<192>, Int<768>>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103,
    false>;
#else
using Default = Config<
    cutlass::arch::Sm100,
    Shape<_128, _128, _256>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100,
    false>;
using Wide = Config<
    cutlass::arch::Sm100,
    Shape<_128, Int<192>, _256>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100,
    false>;
#endif
using Swap = Config<
    cutlass::arch::Sm100,
    Shape<_128, _64, _256>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100,
    true>;
using ScaleConfig = Default::ScaleConfig;

template <class C>
struct Buffers {
  Shape<int, int, int>* problems;
  FP4 const** a;
  FP4 const** b;
  FP8 const** sa;
  FP8 const** sb;
  float** d;
  typename C::StrideA* stride_a;
  typename C::StrideB* stride_b;
  typename C::StrideD* stride_d;
  typename C::LayoutSFA* layout_a;
  typename C::LayoutSFB* layout_b;
  FP8* x_scales;
};

template <typename T>
void reserve(T*& pointer, size_t count, char* base, size_t& offset) {
  offset = (offset + 127) / 128 * 128;
  pointer = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
  offset += count * sizeof(T);
}

template <class C>
size_t buffers(Buffers<C>& b, char* base, int experts, int rows, int n, int k) {
  size_t offset = 0;
  reserve(b.problems, experts, base, offset);
  reserve(b.a, experts, base, offset);
  reserve(b.b, experts, base, offset);
  reserve(b.sa, experts, base, offset);
  reserve(b.sb, experts, base, offset);
  reserve(b.d, experts, base, offset);
  reserve(b.stride_a, experts, base, offset);
  reserve(b.stride_b, experts, base, offset);
  reserve(b.stride_d, experts, base, offset);
  reserve(b.layout_a, experts, base, offset);
  reserve(b.layout_b, experts, base, offset);
  int scale_k = (k + 63) / 64 * 4;
  reserve(b.x_scales, int64_t((rows + 127 * experts + 127) / 128 * 128) * scale_k, base, offset);
  return (offset + 127) / 128 * 128;
}

__host__ __device__ inline int64_t weight_scale_size(int64_t n, int64_t k) {
  return (n + 127) / 128 * 128 * ((k + 63) / 64 * 4);
}

template <class C>
typename C::Gemm::Arguments arguments(Buffers<C> b, int experts, int sm_count) {
  cutlass::KernelHardwareInfo hardware;
  hardware.sm_count = sm_count;
  typename C::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {experts, b.problems, nullptr},
      {b.a, b.stride_a, b.b, b.stride_b, b.sa, b.layout_a, b.sb, b.layout_b},
      {{}, nullptr, nullptr, b.d, b.stride_d},
      hardware};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  return args;
}

template <class C>
int64_t workspace_size_impl(int64_t experts, int64_t rows, int64_t n, int64_t k, int64_t sm_count) {
  Buffers<C> b;
  auto bytes = buffers(b, nullptr, experts, rows, n, k);
  return bytes + C::Gemm::get_workspace_size(arguments(b, experts, sm_count));
}

int64_t workspace_size(int64_t experts, int64_t rows, int64_t n, int64_t k, int64_t sm_count) {
  return workspace_size_impl<Default>(experts, rows, n, k, sm_count);
}

int64_t wide_workspace_size(int64_t experts, int64_t rows, int64_t n, int64_t k, int64_t sm_count) {
  return workspace_size_impl<Wide>(experts, rows, n, k, sm_count);
}

int64_t swap_workspace_size(int64_t experts, int64_t rows, int64_t n, int64_t k, int64_t sm_count) {
  return workspace_size_impl<Swap>(experts, rows, n, k, sm_count);
}

template <class C>
__device__ void prepare_expert(
    Buffers<C> b,
    FP4 const* x,
    FP4 const* w,
    FP8 const* w_scales,
    float* y,
    int const* counts,
    int const* offsets,
    int e,
    int n,
    int k) {
  int m = counts[e];
  int start = offsets[e];
  int scale_k = (k + 63) / 64 * 4;
  int scale_start = (start - offsets[0] + e * 127) / 128 * 128;
  FP4 const* xe = x + int64_t(start) * k / 2;
  FP4 const* we = w + int64_t(e) * n * k / 2;
  FP8 const* xse = b.x_scales + int64_t(scale_start) * scale_k;
  FP8 const* wse = w_scales + int64_t(e) * weight_scale_size(n, k);
  b.d[e] = y + int64_t(start) * n;
  if constexpr (C::swap) {
    b.problems[e] = make_shape(n, m, k);
    b.a[e] = we;
    b.b[e] = xe;
    b.sa[e] = wse;
    b.sb[e] = xse;
    b.stride_a[e] = cutlass::make_cute_packed_stride(typename C::StrideA{}, make_shape(n, k, 1));
    b.stride_b[e] = cutlass::make_cute_packed_stride(typename C::StrideB{}, make_shape(m, k, 1));
    b.stride_d[e] = cutlass::make_cute_packed_stride(typename C::StrideD{}, make_shape(n, m, 1));
    b.layout_a[e] = C::ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1));
    b.layout_b[e] = C::ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1));
  } else {
    b.problems[e] = make_shape(m, n, k);
    b.a[e] = xe;
    b.b[e] = we;
    b.sa[e] = xse;
    b.sb[e] = wse;
    b.stride_a[e] = cutlass::make_cute_packed_stride(typename C::StrideA{}, make_shape(m, k, 1));
    b.stride_b[e] = cutlass::make_cute_packed_stride(typename C::StrideB{}, make_shape(n, k, 1));
    b.stride_d[e] = cutlass::make_cute_packed_stride(typename C::StrideD{}, make_shape(m, n, 1));
    b.layout_a[e] = C::ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    b.layout_b[e] = C::ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
  }
}

// Launched with PDL: it reads only routing metadata that is complete before the
// previous kernel lets it launch, and waits for that kernel before exiting.
template <class C>
__global__ void prepare(
    Buffers<C> b,
    FP4 const* x,
    FP4 const* w,
    FP8 const* w_scales,
    float* y,
    int const* counts,
    int const* offsets,
    int experts,
    int n,
    int k) {
  asm volatile("griddepcontrol.launch_dependents;");
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e < experts) prepare_expert(b, x, w, w_scales, y, counts, offsets, e, n, k);
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

template <class C>
__global__ void swizzle_x(Buffers<C> b, FP8 const* scales, int const* counts, int const* offsets, int experts, int k) {
  int row = blockIdx.x;
  if (row < offsets[0] || row >= offsets[experts - 1] + counts[experts - 1]) return;
  int lo = 0, hi = experts;
  while (lo + 1 < hi) {
    int mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  int local_row = row - offsets[lo];
  for (int col = threadIdx.x; col < k / 16; col += blockDim.x) {
    auto* dst = const_cast<FP8*>(C::swap ? b.sb[lo] : b.sa[lo]);
    auto offset = C::swap ? b.layout_b[lo](make_coord(local_row, col * 16, 0))
                          : b.layout_a[lo](make_coord(local_row, col * 16, 0));
    dst[offset] = scales[int64_t(row) * (k / 16) + col];
  }
}

// The weight scale layout depends only on (n, k), so it is built once at load.
__global__ void swizzle_w(FP8* dst, FP8 const* scales, int n, int k) {
  int row = blockIdx.x;
  int e = blockIdx.y;
  auto layout = ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, n, k, 1));
  dst += int64_t(e) * weight_scale_size(n, k);
  for (int col = threadIdx.x; col < k / 16; col += blockDim.x) {
    dst[layout(make_coord(row, col * 16, 0))] = scales[(int64_t(e) * n + row) * (k / 16) + col];
  }
}

void swizzle_weight_scales(tvm::ffi::TensorView scales, tvm::ffi::TensorView out) {
  using host::RuntimeCheck;
  int experts = scales.size(0), n = scales.size(1), k = scales.size(2) * 16;
  RuntimeCheck(
      out.numel() == experts * weight_scale_size(n, k), "Swizzled NVFP4 weight scale buffer has the wrong size");
  auto stream = host::LaunchKernel::resolve_device(scales.device());
  swizzle_w<<<dim3(n, experts), 128, 0, stream>>>(
      static_cast<FP8*>(out.data_ptr()), static_cast<FP8 const*>(scales.data_ptr()), n, k);
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "Cannot swizzle NVFP4 weight scales");
}

template <class C>
void run_impl(
    tvm::ffi::TensorView x,
    FP8 const* xs,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView workspace,
    int64_t sm_count);

void run(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView workspace,
    int64_t sm_count) {
  run_impl<Default>(x, static_cast<FP8 const*>(xs.data_ptr()), w, ws, counts, offsets, y, workspace, sm_count);
}

void run_swap(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView workspace,
    int64_t sm_count) {
  run_impl<Swap>(x, static_cast<FP8 const*>(xs.data_ptr()), w, ws, counts, offsets, y, workspace, sm_count);
}

// Activation scales already written in the GEMM layout at wide_x_scales_offset.
void run_wide_preswizzled(
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView workspace,
    int64_t sm_count) {
  run_impl<Wide>(x, nullptr, w, ws, counts, offsets, y, workspace, sm_count);
}

int64_t wide_x_scales_offset(int64_t experts, int64_t rows, int64_t n, int64_t k) {
  Buffers<Wide> b;
  buffers(b, nullptr, experts, rows, n, k);
  return reinterpret_cast<int64_t>(b.x_scales);
}

template <class C>
void run_impl(
    tvm::ffi::TensorView x,
    FP8 const* xs,
    tvm::ffi::TensorView w,
    tvm::ffi::TensorView ws,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView offsets,
    tvm::ffi::TensorView y,
    tvm::ffi::TensorView workspace,
    int64_t sm_count) {
  using host::RuntimeCheck;
  int experts = w.size(0), n = w.size(1), k = w.size(2) * 2;
  int rows = x.size(0);
  auto stream = host::LaunchKernel::resolve_device(x.device());
  Buffers<C> b;
  size_t bytes = buffers(b, static_cast<char*>(workspace.data_ptr()), experts, rows, n, k);
  auto args = arguments(b, experts, sm_count);
  RuntimeCheck(workspace.numel() >= bytes + C::Gemm::get_workspace_size(args), "NVFP4 workspace is too small");
  RuntimeCheck(ws.numel() == experts * weight_scale_size(n, k), "NVFP4 weight scales must be pre-swizzled");
  auto* count_ptr = static_cast<int const*>(counts.data_ptr());
  auto* offset_ptr = static_cast<int const*>(offsets.data_ptr());
  cudaLaunchAttribute pdl[1];
  pdl[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = (experts + 127) / 128;
  config.blockDim = 128;
  config.stream = stream;
  config.attrs = pdl;
  config.numAttrs = 1;
  RuntimeCheck(
      cudaLaunchKernelEx(
          &config,
          prepare<C>,
          b,
          static_cast<FP4 const*>(x.data_ptr()),
          static_cast<FP4 const*>(w.data_ptr()),
          static_cast<FP8 const*>(ws.data_ptr()),
          static_cast<float*>(y.data_ptr()),
          count_ptr,
          offset_ptr,
          experts,
          n,
          k) == cudaSuccess,
      "Cannot launch NVFP4 GEMM prepare");
  if (xs != nullptr) {
    auto scale_bytes = static_cast<char*>(workspace.data_ptr()) + bytes - reinterpret_cast<char*>(b.x_scales);
    RuntimeCheck(cudaMemsetAsync(b.x_scales, 0, scale_bytes, stream) == cudaSuccess, "Cannot initialize NVFP4 scales");
    swizzle_x<C><<<rows, 128, 0, stream>>>(b, xs, count_ptr, offset_ptr, experts, k);
  }
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "Cannot prepare NVFP4 GEMM");
  typename C::Gemm gemm;
  RuntimeCheck(gemm.can_implement(args) == cutlass::Status::kSuccess, "Unsupported NVFP4 GEMM shape");
  auto status = gemm.initialize(args, static_cast<char*>(workspace.data_ptr()) + bytes, stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Cannot initialize NVFP4 GEMM: ", int(status));
  status = gemm.run(stream, nullptr, /*launch_with_pdl=*/true);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Cannot run NVFP4 GEMM: ", int(status));
}

}  // namespace sglang::inkling2_nvfp4
