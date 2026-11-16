// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/utils.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <tvm/ffi/container/tensor.h>

namespace sglang::inkling2_mxfp8 {

using namespace cute;
using Problems = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using FP8 = cutlass::float_e4m3_t;
using SF = cutlass::float_ue8m0_t;
using BF16 = cutlass::bfloat16_t;

struct Config {
  using Tile = Shape<_128, _128, _128>;
  using Cluster = Shape<_1, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      void,
      cutlass::layout::RowMajor*,
      8,
      BF16,
      cutlass::layout::RowMajor*,
      8,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassBlockScaledTensorOp,
      tuple<FP8, SF>,
      cutlass::layout::RowMajor*,
      16,
      tuple<FP8, SF>,
      cutlass::layout::ColumnMajor*,
      16,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Problems, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using LayoutSFA = typename Mainloop::InternalLayoutSFA;
  using LayoutSFB = typename Mainloop::InternalLayoutSFB;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
};

struct Buffers {
  Shape<int, int, int>* problems;
  FP8 const** a;
  FP8 const** b;
  SF const** sa;
  SF const** sb;
  BF16** d;
  typename Config::StrideA* stride_a;
  typename Config::StrideB* stride_b;
  typename Config::StrideD* stride_d;
  typename Config::LayoutSFA* layout_a;
  typename Config::LayoutSFB* layout_b;
  SF* x_scales;
};

template <typename T>
void reserve(T*& pointer, size_t count, char* base, size_t& offset) {
  offset = (offset + 127) / 128 * 128;
  pointer = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
  offset += count * sizeof(T);
}

// Scale bytes of an (rows, k) operand in the 128x4 atom layout.
__host__ __device__ inline int64_t scale_size(int64_t rows, int64_t k) {
  return (rows + 127) / 128 * 128 * ((k + 127) / 128 * 4);
}

size_t buffers(Buffers& b, char* base, int experts, int rows, int k) {
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
  reserve(b.x_scales, scale_size(rows + 127 * int64_t(experts), k), base, offset);
  return (offset + 127) / 128 * 128;
}

Config::Gemm::Arguments arguments(Buffers b, int experts, int sm_count) {
  cutlass::KernelHardwareInfo hardware;
  hardware.sm_count = sm_count;
  Config::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {experts, b.problems, nullptr},
      {b.a, b.stride_a, b.b, b.stride_b, b.sa, b.layout_a, b.sb, b.layout_b},
      {{}, nullptr, nullptr, b.d, b.stride_d},
      hardware};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  return args;
}

int64_t workspace_size(int64_t experts, int64_t rows, int64_t n, int64_t k, int64_t sm_count) {
  Buffers b;
  auto bytes = buffers(b, nullptr, experts, rows, k);
  return bytes + Config::Gemm::get_workspace_size(arguments(b, experts, sm_count));
}

int64_t x_scales_offset(int64_t experts, int64_t rows, int64_t k) {
  Buffers b;
  buffers(b, nullptr, experts, rows, k);
  return reinterpret_cast<int64_t>(b.x_scales);
}

__global__ void prepare(
    Buffers b,
    FP8 const* x,
    FP8 const* w,
    SF const* w_scales,
    BF16* y,
    int const* counts,
    int const* offsets,
    int experts,
    int n,
    int k) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  int m = counts[e];
  int start = offsets[e];
  int scale_start = (start - offsets[0] + e * 127) / 128 * 128;
  b.problems[e] = make_shape(m, n, k);
  b.a[e] = x + int64_t(start) * k;
  b.b[e] = w + int64_t(e) * n * k;
  b.sa[e] = b.x_scales + int64_t(scale_start) * ((k + 127) / 128 * 4);
  b.sb[e] = w_scales + int64_t(e) * scale_size(n, k);
  b.d[e] = y + int64_t(start) * n;
  b.stride_a[e] = cutlass::make_cute_packed_stride(Config::StrideA{}, make_shape(m, k, 1));
  b.stride_b[e] = cutlass::make_cute_packed_stride(Config::StrideB{}, make_shape(n, k, 1));
  b.stride_d[e] = cutlass::make_cute_packed_stride(Config::StrideD{}, make_shape(m, n, 1));
  b.layout_a[e] = Config::ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  b.layout_b[e] = Config::ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
}

// Row-major (rows, k / 32) activation scales -> per-expert SFA layout.
__global__ void swizzle_x(Buffers b, SF const* scales, int const* counts, int const* offsets, int experts, int k) {
  for (int row = offsets[0] + blockIdx.x; row < offsets[experts]; row += gridDim.x) {
    int lo = 0, hi = experts;
    while (lo + 1 < hi) {
      int mid = (lo + hi) / 2;
      if (offsets[mid] <= row)
        lo = mid;
      else
        hi = mid;
    }
    int local_row = row - offsets[lo];
    auto layout = Config::ScaleConfig::tile_atom_to_shape_SFA(make_shape(counts[lo], 1, k, 1));
    auto* dst = const_cast<SF*>(b.sa[lo]);
    for (int col = threadIdx.x; col < k / 32; col += blockDim.x)
      dst[layout(make_coord(local_row, col * 32, 0))] = scales[int64_t(row) * (k / 32) + col];
  }
}

// The weight scale layout depends only on (n, k), so it is built once at load.
__global__ void swizzle_w(SF* dst, SF const* scales, int n, int k) {
  int row = blockIdx.x;
  int e = blockIdx.y;
  auto layout = Config::ScaleConfig::tile_atom_to_shape_SFB(make_shape(1, n, k, 1));
  dst += int64_t(e) * scale_size(n, k);
  for (int col = threadIdx.x; col < k / 32; col += blockDim.x)
    dst[layout(make_coord(row, col * 32, 0))] = scales[(int64_t(e) * n + row) * (k / 32) + col];
}

void swizzle_weight_scales(tvm::ffi::TensorView scales, tvm::ffi::TensorView out) {
  using host::RuntimeCheck;
  int experts = scales.size(0), n = scales.size(1), k = scales.size(2) * 32;
  RuntimeCheck(out.numel() == experts * scale_size(n, k), "Swizzled MXFP8 weight scale buffer has the wrong size");
  auto stream = host::LaunchKernel::resolve_device(scales.device());
  swizzle_w<<<dim3(n, experts), 128, 0, stream>>>(
      static_cast<SF*>(out.data_ptr()), static_cast<SF const*>(scales.data_ptr()), n, k);
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "Cannot swizzle MXFP8 weight scales");
}

// x: (rows, k) FP8 rows in expert order; xs: row-major (rows, k / 32) scales,
// or empty when already written in the GEMM layout at x_scales_offset.
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
  using host::RuntimeCheck;
  int experts = w.size(0), n = w.size(1), k = w.size(2);
  int rows = x.size(0);
  auto stream = host::LaunchKernel::resolve_device(x.device());
  Buffers b;
  size_t bytes = buffers(b, static_cast<char*>(workspace.data_ptr()), experts, rows, k);
  auto args = arguments(b, experts, sm_count);
  RuntimeCheck(workspace.numel() >= bytes + Config::Gemm::get_workspace_size(args), "MXFP8 workspace is too small");
  RuntimeCheck(ws.numel() == experts * scale_size(n, k), "MXFP8 weight scales must be pre-swizzled");
  auto* count_ptr = static_cast<int const*>(counts.data_ptr());
  auto* offset_ptr = static_cast<int const*>(offsets.data_ptr());
  prepare<<<(experts + 127) / 128, 128, 0, stream>>>(
      b,
      static_cast<FP8 const*>(x.data_ptr()),
      static_cast<FP8 const*>(w.data_ptr()),
      static_cast<SF const*>(ws.data_ptr()),
      static_cast<BF16*>(y.data_ptr()),
      count_ptr,
      offset_ptr,
      experts,
      n,
      k);
  if (xs.numel()) {
    auto scale_bytes = static_cast<char*>(workspace.data_ptr()) + bytes - reinterpret_cast<char*>(b.x_scales);
    RuntimeCheck(cudaMemsetAsync(b.x_scales, 0, scale_bytes, stream) == cudaSuccess, "Cannot initialize MXFP8 scales");
    swizzle_x<<<std::max(1, std::min<int>(rows, 4 * sm_count)), 128, 0, stream>>>(
        b, static_cast<SF const*>(xs.data_ptr()), count_ptr, offset_ptr, experts, k);
  }
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "Cannot prepare MXFP8 GEMM");
  Config::Gemm gemm;
  RuntimeCheck(gemm.can_implement(args) == cutlass::Status::kSuccess, "Unsupported MXFP8 GEMM shape");
  auto status = gemm.initialize(args, static_cast<char*>(workspace.data_ptr()) + bytes, stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Cannot initialize MXFP8 GEMM: ", int(status));
  status = gemm.run(stream);
  RuntimeCheck(status == cutlass::Status::kSuccess, "Cannot run MXFP8 GEMM: ", int(status));
}

}  // namespace sglang::inkling2_mxfp8
