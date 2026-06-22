// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// cosmos_gemm_bf16.cu — bf16 CUTLASS GEMMs for the Cosmos DiT block.
//
// The existing `cutlass_linear_layer_rrr*` kernels in `ops.cu` are fp16-only
// and accept WAN's pre-transposed weight layout `[in_features, out_features]`
// row-major. Cosmos consumes raw PyTorch state_dict weights directly (no
// offline weight extraction step like WAN's `extract_weights.py`), so its
// weights are stored as `[out_features, in_features]` row-major -- the
// PyTorch `nn.Linear.weight` convention.
//
// To match `output = input @ weight.T` (PyTorch's `nn.Linear` math) without
// a pre-transpose pass, these kernels use the **RCR** (Row-Col-Row) CUTLASS
// layout: A row-major × B column-major → C row-major. PyTorch's
// `[out, in]` row-major weight, when reinterpreted as `[in, out]`
// column-major (same byte layout, different stride interpretation), is
// exactly the B operand CUTLASS expects for an RCR GEMM. The result is
// `output = input @ weight^T` -- matching PyTorch's `nn.Linear`.
//
// Tile shapes mirror the fp16 RRR kernels (256×128×32 SM80 tensor cores
// for large M, 128×128×32 for small M) so the bf16 path inherits the same
// large-M / small-M tuning the WAN team converged on.

#include "cosmos_block.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/bfloat16.h"

namespace omnidreams_singleview {

namespace {
// Same threshold heuristic as `cutlass_linear_layer_rrr` in `ops.cu`:
// large M -> 256x128x32, small M -> 128x128x32.
constexpr int k_cosmos_gemm_tile_threshold = 1024;
} // namespace

template <typename ElementOutput_, int ElementsPerAccess, typename ElementAccumulator_, typename ElementCompute_>
class LinearCombinationGateTimesAccumPlusResidual {
public:
  using ElementOutput = ElementOutput_;
  using ElementD = ElementOutput;
  using ElementC = ElementOutput;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementVector = ElementOutput;
  using ElementZ = ElementOutput;
  using ElementT = ElementOutput;

  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  static bool const kIsSingleSource = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = false;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentVector = cutlass::Array<ElementVector, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

  struct Params {
    ElementCompute alpha;
    CUTLASS_HOST_DEVICE
    Params() : alpha(ElementCompute(1)) {}
    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha_) : alpha(alpha_) {}
  };

private:
  ElementCompute alpha_;

public:
  CUTLASS_HOST_DEVICE
  LinearCombinationGateTimesAccumPlusResidual(Params const& params) : alpha_(params.alpha) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_Z,
      FragmentT&,
      FragmentAccumulator const& AB,
      FragmentC const& frag_C,
      FragmentCompute const& V) const {
    FragmentCompute acc =
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute residual =
        cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      result[i] = residual[i] + alpha_ * V[i] * acc[i];
    }
    frag_Z = cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess>()(result);
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_Z,
      FragmentT& frag_T,
      FragmentAccumulator const& AB,
      FragmentCompute const& V) const {
    FragmentC zeros;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) zeros[i] = ElementC(0);
    operator()(frag_Z, frag_T, AB, zeros, V);
  }
};

// ---------------------------------------------------------------------------
// D = A @ B^T + bias  (PyTorch nn.Linear math)
//   A: [N, in_features]    row-major
//   B: [out_features, in_features]  row-major  (PyTorch weight)
//      reinterpreted as [in_features, out_features] column-major for CUTLASS
//   D: [N, out_features]   row-major
// bias: [out_features] or nullptr (broadcast via stride-0 trick)
// ---------------------------------------------------------------------------
cudaError_t cutlass_linear_layer_rrr_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,    // [out_features, in_features] PyTorch layout
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;       // ← key change: B is column-major
  using LayoutC = cutlass::layout::RowMajor;

  const ElementInputA* a_ptr = input_row;
  const ElementInputB* b_ptr = weight_row;
  const ElementOutput* c_ptr = bias ? bias : output_row;
  int ldc = bias ? 0 : out_features;
  ElementOutput beta = bias ? ElementOutput(1.0f) : ElementOutput(0.0f);

  cutlass::Status status;

  // For ColumnMajor B with logical shape [in_features, out_features],
  // the leading dimension (stride between columns) equals the number of
  // ROWS, which is in_features. PyTorch's [out_features, in_features]
  // row-major weight stored contiguously has byte layout
  // weight[i, j] = data[i*in_features + j]. Reinterpreting that same
  // memory as ColumnMajor [in_features, out_features] with ldb=in_features
  // gives B[k, n] = data[n*in_features + k] = weight[n, k] -- exactly the
  // transpose we want for `output = input @ weight^T`.
  if (N < k_cosmos_gemm_tile_threshold) {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>>;
    Gemm gemm_op;
    Gemm::Arguments args(
        {N, out_features, in_features},   // (M, N, K)  problem size
        {a_ptr, in_features},             // A: row-major, lda = in_features
        {b_ptr, in_features},             // B: col-major, ldb = in_features
        {c_ptr, ldc},
        {output_row, out_features},
        {ElementOutput(1.0f), beta});
    status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
  } else {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>>;
    Gemm gemm_op;
    Gemm::Arguments args(
        {N, out_features, in_features},
        {a_ptr, in_features},
        {b_ptr, in_features},
        {c_ptr, ldc},
        {output_row, out_features},
        {ElementOutput(1.0f), beta});
    status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
  }

  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ---------------------------------------------------------------------------
// D = A @ B + bias, where B is already prepared as [in_features, out_features]
// row-major. This is the native row/row/row layout used by the fast WAN path.
// ---------------------------------------------------------------------------
cudaError_t cutlass_linear_layer_rrr_bf16_prepared(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  const ElementOutput* c_ptr = bias ? bias : output_row;
  int ldc = bias ? 0 : out_features;
  ElementOutput beta = bias ? ElementOutput(1.0f) : ElementOutput(0.0f);
  cutlass::Status status;

  if (N < k_cosmos_gemm_tile_threshold) {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>>;
    Gemm gemm_op;
    typename Gemm::Arguments args(
        {N, out_features, in_features},
        {input_row, in_features},
        {weight_row_prepared, out_features},
        {c_ptr, ldc},
        {output_row, out_features},
        {ElementOutput(1.0f), beta});
    status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
  } else {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>>;
    Gemm gemm_op;
    typename Gemm::Arguments args(
        {N, out_features, in_features},
        {input_row, in_features},
        {weight_row_prepared, out_features},
        {c_ptr, ldc},
        {output_row, out_features},
        {ElementOutput(1.0f), beta});
    status = gemm_op.initialize(args, nullptr);
    if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
    status = gemm_op(stream);
  }

  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ---------------------------------------------------------------------------
// D = GELU(A @ B^T + bias)
// ---------------------------------------------------------------------------
cudaError_t cutlass_linear_layer_rrr_gelu_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      float>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOp>;

  Gemm gemm_op;
  typename EpilogueOp::Params epilogue_params(1.0f, 1.0f);
  Gemm::Arguments args(
      {N, out_features, in_features},
      {input_row, in_features},
      {weight_row, in_features},
      {bias, 0},
      {output_row, out_features},
      epilogue_params);

  cutlass::Status status = gemm_op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ---------------------------------------------------------------------------
// D = GELU(A @ B + bias), B prepared as [in_features, out_features] row-major.
// ---------------------------------------------------------------------------
cudaError_t cutlass_linear_layer_rrr_gelu_bf16_prepared(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      float>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOp>;

  Gemm gemm_op;
  typename EpilogueOp::Params epilogue_params(1.0f, 1.0f);
  typename Gemm::Arguments args(
      {N, out_features, in_features},
      {input_row, in_features},
      {weight_row_prepared, out_features},
      {bias, 0},
      {output_row, out_features},
      epilogue_params);

  cutlass::Status status = gemm_op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t cutlass_linear_layer_rrr_bf16_prepared_gated_residual(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row_prepared,
    const cutlass::bfloat16_t* gate,
    cutlass::bfloat16_t* residual_inout,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  using EpilogueOp = LinearCombinationGateTimesAccumPlusResidual<
      ElementOutput, kElementsPerAccess, ElementAccumulator, ElementCompute>;

  using DefaultConfig = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;

  using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      DefaultConfig::kStages,
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB>;

  Gemm gemm_op;
  typename EpilogueOp::Params epilogue_params(ElementCompute(1.0f));
  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, out_features, in_features},
      1,
      epilogue_params,
      input_row,
      weight_row_prepared,
      residual_inout,
      residual_inout,
      const_cast<cutlass::bfloat16_t*>(gate),
      nullptr,
      int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0), int64_t(0),
      int64_t(in_features), int64_t(out_features),
      int64_t(out_features),
      int64_t(out_features),
      int64_t(0),
      int64_t(0));

  cutlass::Status status = gemm_op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

// ---------------------------------------------------------------------------
// D = SiLU(A @ B^T + bias)
// ---------------------------------------------------------------------------
cudaError_t cutlass_linear_layer_rrr_silu_bf16(
    const cutlass::bfloat16_t* input_row,
    const cutlass::bfloat16_t* weight_row,
    const cutlass::bfloat16_t* bias,
    cutlass::bfloat16_t* output_row,
    int N, int in_features, int out_features,
    cudaStream_t stream)
{
  if (N <= 0 || in_features <= 0 || out_features <= 0) return cudaSuccess;

  using ElementInputA = cutlass::bfloat16_t;
  using ElementInputB = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
      ElementOutput,
      kElementsPerAccess,
      ElementAccumulator,
      float>;

  // adaln-LoRA's SiLU GEMM has small-N (just one row per batch); use the
  // 128x128x32 tile to get reasonable coverage for small matrices.
  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutA,
      ElementInputB, LayoutB,
      ElementOutput, LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EpilogueOp>;

  Gemm gemm_op;
  typename Gemm::Arguments args(
      {N, out_features, in_features},
      {input_row, in_features},
      {weight_row, in_features},
      {bias, 0},
      {output_row, out_features},
      {1.0f, 1.0f});

  cutlass::Status status = gemm_op.initialize(args, nullptr);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;
  status = gemm_op(stream);
  return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

} // namespace omnidreams_singleview
